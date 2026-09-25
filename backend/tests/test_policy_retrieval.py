from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.rag.models import (
    PolicyChunk,
    PolicySearchHit,
    RetrievalMode,
)
from backend.app.rag.retrieval import (
    PolicyRetriever,
    PostgresPolicySearch,
    reciprocal_rank_fusion,
)


def _chunk(identifier: str, policy_id: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=identifier,
        policy_id=policy_id,
        title=policy_id.replace("-", " ").title(),
        version="1.0",
        effective_date=date(2026, 9, 1),
        section="Rules",
        chunk_index=0,
        content=f"Policy evidence for {policy_id}.",
    )


class FakeEmbeddingProvider:
    dimensions = 3

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.fail:
            raise RuntimeError("embedding failed")
        return [1.0, 0.0, 0.0]

    async def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeSearch:
    def __init__(
        self,
        *,
        keyword=(),
        vector=(),
        fail_keyword: bool = False,
        fail_vector: bool = False,
    ) -> None:
        self.keyword = list(keyword)
        self.vector = list(vector)
        self.fail_keyword = fail_keyword
        self.fail_vector = fail_vector
        self.keyword_queries: list[str] = []
        self.keyword_limits: list[int] = []
        self.vector_limits: list[int] = []

    async def keyword_search(self, query: str, *, limit: int):
        self.keyword_queries.append(query)
        self.keyword_limits.append(limit)
        if self.fail_keyword:
            raise RuntimeError("keyword query failed")
        return self.keyword[:limit]

    async def vector_search(self, vector, *, limit: int):
        self.vector_limits.append(limit)
        if self.fail_vector:
            raise RuntimeError("vector query failed")
        return self.vector[:limit]


def _keyword_hit(chunk: PolicyChunk, rank: int) -> PolicySearchHit:
    return PolicySearchHit(
        chunk=chunk,
        keyword_rank=rank,
        keyword_score=1.0 / rank,
    )


def _vector_hit(
    chunk: PolicyChunk,
    rank: int,
    similarity: float = 0.8,
) -> PolicySearchHit:
    return PolicySearchHit(
        chunk=chunk,
        vector_rank=rank,
        vector_similarity=similarity,
    )


@pytest.mark.asyncio
async def test_keyword_retrieval_is_independently_testable() -> None:
    chunk = _chunk("disputes:window:00", "transaction-disputes")
    embeddings = FakeEmbeddingProvider()
    retriever = PolicyRetriever(embedding_provider=embeddings)

    results = await retriever.retrieve(
        query="dispute window",
        search=FakeSearch(keyword=[_keyword_hit(chunk, 1)]),
        mode=RetrievalMode.KEYWORD,
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [chunk.chunk_id]
    assert embeddings.queries == []


@pytest.mark.asyncio
async def test_vector_retrieval_uses_fake_deterministic_embedding() -> None:
    chunk = _chunk("status:pending:00", "transaction-status")
    embeddings = FakeEmbeddingProvider()
    retriever = PolicyRetriever(embedding_provider=embeddings)

    results = await retriever.retrieve(
        query="purchase waiting to complete",
        search=FakeSearch(vector=[_vector_hit(chunk, 1)]),
        mode=RetrievalMode.VECTOR,
        top_k=2,
    )

    assert embeddings.queries == ["purchase waiting to complete"]
    assert results[0].chunk.policy_id == "transaction-status"


def test_rrf_rewards_evidence_found_by_both_channels() -> None:
    shared = _chunk("shared:00", "transaction-disputes")
    lexical = _chunk("lexical:00", "transaction-status")
    semantic = _chunk("semantic:00", "unauthorized-card-transactions")

    results = reciprocal_rank_fusion(
        [_keyword_hit(lexical, 1), _keyword_hit(shared, 2)],
        [_vector_hit(semantic, 1), _vector_hit(shared, 2)],
    )

    assert results[0].chunk == shared
    assert results[0].keyword_rank == 2
    assert results[0].vector_rank == 2


@pytest.mark.asyncio
async def test_hybrid_recovers_semantic_hit_missed_by_keyword() -> None:
    lexical = _chunk("lexical:00", "support-escalation")
    semantic = _chunk("semantic:00", "transaction-status")
    retriever = PolicyRetriever(
        embedding_provider=FakeEmbeddingProvider()
    )

    results = await retriever.retrieve(
        query="purchase waiting to complete",
        search=FakeSearch(
            keyword=[_keyword_hit(lexical, 1)],
            vector=[_vector_hit(semantic, 1)],
        ),
        top_k=2,
    )

    assert {result.chunk for result in results} == {lexical, semantic}


@pytest.mark.asyncio
async def test_top_k_is_enforced_and_bounded() -> None:
    hits = [
        _keyword_hit(_chunk(f"chunk:{index}", f"policy-{index}"), index)
        for index in range(1, 8)
    ]
    retriever = PolicyRetriever(
        embedding_provider=FakeEmbeddingProvider()
    )

    results = await retriever.retrieve(
        query="policy",
        search=FakeSearch(keyword=hits),
        mode=RetrievalMode.KEYWORD,
        top_k=3,
    )

    assert len(results) == 3
    with pytest.raises(ValueError, match="top_k"):
        await retriever.retrieve(
            query="policy",
            search=FakeSearch(),
            top_k=11,
        )


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_to_keyword() -> None:
    chunk = _chunk("disputes:window:00", "transaction-disputes")
    retriever = PolicyRetriever(
        embedding_provider=FakeEmbeddingProvider(fail=True)
    )

    results = await retriever.retrieve(
        query="dispute window",
        search=FakeSearch(keyword=[_keyword_hit(chunk, 1)]),
    )

    assert [result.chunk for result in results] == [chunk]


@pytest.mark.asyncio
async def test_vector_query_failure_falls_back_to_keyword() -> None:
    chunk = _chunk("disputes:window:00", "transaction-disputes")
    retriever = PolicyRetriever(
        embedding_provider=FakeEmbeddingProvider()
    )

    results = await retriever.retrieve(
        query="dispute window",
        search=FakeSearch(
            keyword=[_keyword_hit(chunk, 1)],
            fail_vector=True,
        ),
    )

    assert [result.chunk for result in results] == [chunk]


@pytest.mark.asyncio
async def test_keyword_query_failure_falls_back_to_vector() -> None:
    chunk = _chunk("status:pending:00", "transaction-status")
    retriever = PolicyRetriever(
        embedding_provider=FakeEmbeddingProvider()
    )

    results = await retriever.retrieve(
        query="purchase waiting to complete",
        search=FakeSearch(
            vector=[_vector_hit(chunk, 1)],
            fail_keyword=True,
        ),
    )

    assert [result.chunk for result in results] == [chunk]


@pytest.mark.asyncio
async def test_dense_query_ignores_generic_policy_product_language() -> None:
    embeddings = FakeEmbeddingProvider()
    retriever = PolicyRetriever(embedding_provider=embeddings)
    search = FakeSearch()

    await retriever.retrieve(
        query="What is SentinelVoice policy for mortgage refinancing?",
        search=search,
    )

    assert embeddings.queries == ["What is for mortgage refinancing?"]
    assert search.keyword_queries == ["What is for mortgage refinancing?"]


@pytest.mark.asyncio
async def test_postgres_search_statements_are_scoped_to_policy_tables() -> None:
    db = AsyncMock()
    query_result = MagicMock()
    query_result.all.return_value = []
    db.execute.return_value = query_result
    search = PostgresPolicySearch(db)

    await search.keyword_search("pending transaction", limit=3)
    await search.vector_search([0.0] * 384, limit=3)

    statements = [str(call.args[0]) for call in db.execute.await_args_list]
    assert all("policy_chunks" in statement for statement in statements)
    assert all("policy_documents" in statement for statement in statements)
    assert all("customers" not in statement for statement in statements)
    assert all("transactions " not in statement for statement in statements)
