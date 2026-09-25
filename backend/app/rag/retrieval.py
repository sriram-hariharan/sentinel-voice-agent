import logging
import re
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import PolicyChunkRecord, PolicyDocumentRecord
from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.rag.models import (
    PolicyChunk,
    PolicySearchHit,
    RetrievalMode,
    RetrievedPolicyChunk,
)

logger = logging.getLogger(__name__)

_GENERIC_POLICY_QUERY_LANGUAGE = re.compile(
    r"\b(?:sentinelvoice(?:\s+bank)?|synthetic\s+bank|bank\s+policy|"
    r"policy|policies)\b",
    re.IGNORECASE,
)


class PolicySearchError(RuntimeError):
    """Raised when one PostgreSQL policy-search channel fails."""


def _semantic_query(query: str) -> str:
    focused = " ".join(
        _GENERIC_POLICY_QUERY_LANGUAGE.sub(" ", query).split()
    )
    return focused or query


class PolicySearchBackend(Protocol):
    async def keyword_search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[PolicySearchHit]: ...

    async def vector_search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
    ) -> list[PolicySearchHit]: ...


def _chunk_from_records(
    chunk: PolicyChunkRecord,
    document: PolicyDocumentRecord,
) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk.chunk_id,
        policy_id=document.document_id,
        title=document.title,
        version=document.version,
        effective_date=document.effective_date,
        section=chunk.section,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
    )


class PostgresPolicySearch:
    """Searches only the two policy tables in the current DB session."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def keyword_search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[PolicySearchHit]:
        ts_query = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(
            PolicyChunkRecord.search_vector,
            ts_query,
        ).label("keyword_score")
        statement = (
            select(PolicyChunkRecord, PolicyDocumentRecord, rank)
            .join(
                PolicyDocumentRecord,
                PolicyDocumentRecord.document_id
                == PolicyChunkRecord.document_id,
            )
            .where(PolicyChunkRecord.search_vector.op("@@")(ts_query))
            .order_by(rank.desc(), PolicyChunkRecord.chunk_id)
            .limit(limit)
        )
        try:
            rows = (await self._db.execute(statement)).all()
        except Exception as exc:
            raise PolicySearchError("Keyword policy search failed") from exc

        return [
            PolicySearchHit(
                chunk=_chunk_from_records(chunk, document),
                keyword_rank=index,
                keyword_score=float(score),
            )
            for index, (chunk, document, score) in enumerate(rows, start=1)
        ]

    async def vector_search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
    ) -> list[PolicySearchHit]:
        distance = PolicyChunkRecord.embedding.cosine_distance(
            list(vector)
        ).label("distance")
        statement = (
            select(PolicyChunkRecord, PolicyDocumentRecord, distance)
            .join(
                PolicyDocumentRecord,
                PolicyDocumentRecord.document_id
                == PolicyChunkRecord.document_id,
            )
            .order_by(distance, PolicyChunkRecord.chunk_id)
            .limit(limit)
        )
        try:
            rows = (await self._db.execute(statement)).all()
        except Exception as exc:
            raise PolicySearchError("Vector policy search failed") from exc

        return [
            PolicySearchHit(
                chunk=_chunk_from_records(chunk, document),
                vector_rank=index,
                vector_similarity=1.0 - float(distance_value),
            )
            for index, (chunk, document, distance_value) in enumerate(
                rows,
                start=1,
            )
        ]


def reciprocal_rank_fusion(
    keyword_hits: Sequence[PolicySearchHit],
    vector_hits: Sequence[PolicySearchHit],
    *,
    rrf_k: int = 60,
) -> list[RetrievedPolicyChunk]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    combined: dict[str, dict] = {}
    for hit in keyword_hits:
        entry = combined.setdefault(
            hit.chunk.chunk_id,
            {"chunk": hit.chunk, "score": 0.0},
        )
        entry["keyword_rank"] = hit.keyword_rank
        entry["keyword_score"] = hit.keyword_score
        if hit.keyword_rank is not None:
            entry["score"] += 1.0 / (rrf_k + hit.keyword_rank)

    for hit in vector_hits:
        entry = combined.setdefault(
            hit.chunk.chunk_id,
            {"chunk": hit.chunk, "score": 0.0},
        )
        entry["vector_rank"] = hit.vector_rank
        entry["vector_similarity"] = hit.vector_similarity
        if hit.vector_rank is not None:
            entry["score"] += 1.0 / (rrf_k + hit.vector_rank)

    results = [
        RetrievedPolicyChunk(
            chunk=entry["chunk"],
            keyword_rank=entry.get("keyword_rank"),
            vector_rank=entry.get("vector_rank"),
            keyword_score=entry.get("keyword_score"),
            vector_similarity=entry.get("vector_similarity"),
            rrf_score=entry["score"],
        )
        for entry in combined.values()
    ]
    return sorted(
        results,
        key=lambda result: (-result.rrf_score, result.chunk.chunk_id),
    )


class PolicyRetriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        min_vector_similarity: float = 0.45,
        rrf_k: int = 60,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._min_vector_similarity = min_vector_similarity
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        *,
        query: str,
        search: PolicySearchBackend,
        top_k: int = 4,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> list[RetrievedPolicyChunk]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        if not query.strip():
            return []

        candidate_limit = min(30, max(top_k * 3, top_k))
        focused_query = _semantic_query(query)
        keyword_hits: list[PolicySearchHit] = []
        vector_hits: list[PolicySearchHit] = []

        if mode in {RetrievalMode.KEYWORD, RetrievalMode.HYBRID}:
            try:
                keyword_hits = await search.keyword_search(
                    focused_query,
                    limit=candidate_limit,
                )
            except Exception:
                logger.exception(
                    "policy keyword retrieval failed",
                    extra={"retrieval_mode": mode.value},
                )

        if mode in {RetrievalMode.VECTOR, RetrievalMode.HYBRID}:
            try:
                vector = await self._embedding_provider.embed_query(
                    focused_query
                )
                raw_vector_hits = await search.vector_search(
                    vector,
                    limit=candidate_limit,
                )
                vector_hits = [
                    hit
                    for hit in raw_vector_hits
                    if hit.vector_similarity is not None
                    and hit.vector_similarity >= self._min_vector_similarity
                ]
            except Exception:
                logger.exception(
                    "policy vector retrieval failed",
                    extra={"retrieval_mode": mode.value},
                )

        fused = reciprocal_rank_fusion(
            keyword_hits,
            vector_hits,
            rrf_k=self._rrf_k,
        )
        results = fused[:top_k]
        logger.info(
            "policy retrieval completed",
            extra={
                "retrieval_mode": mode.value,
                "keyword_hit_count": len(keyword_hits),
                "vector_hit_count": len(vector_hits),
                "result_count": len(results),
            },
        )
        return results
