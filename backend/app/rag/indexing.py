from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import PolicyChunkRecord, PolicyDocumentRecord
from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.rag.chunking import (
    PolicyDocumentError,
    chunk_policy_document,
    content_hash,
)
from backend.app.rag.models import PolicyDocument, PolicyIndexReport


@dataclass(frozen=True)
class PolicyIndexPlan:
    changed_policy_ids: tuple[str, ...]
    unchanged_policy_ids: tuple[str, ...]
    removed_policy_ids: tuple[str, ...]


def ensure_document_unchanged(document: PolicyDocument) -> None:
    path = Path(document.source_path)
    try:
        current_hash = content_hash(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolicyDocumentError(
            f"Policy source changed or disappeared during indexing: {path}"
        ) from exc
    if current_hash != document.content_hash:
        raise PolicyDocumentError(
            f"Policy source changed during indexing: {path}"
        )


def plan_policy_index(
    documents: Sequence[PolicyDocument],
    existing_hashes: dict[str, str],
    *,
    reset: bool = False,
) -> PolicyIndexPlan:
    incoming = {document.policy_id: document for document in documents}
    changed = tuple(
        sorted(
            policy_id
            for policy_id, document in incoming.items()
            if reset or existing_hashes.get(policy_id) != document.content_hash
        )
    )
    unchanged = tuple(sorted(set(incoming) - set(changed)))
    removed = tuple(sorted(set(existing_hashes) - set(incoming)))
    return PolicyIndexPlan(
        changed_policy_ids=changed,
        unchanged_policy_ids=unchanged,
        removed_policy_ids=removed,
    )


async def index_policy_documents(
    *,
    db: AsyncSession,
    documents: Sequence[PolicyDocument],
    embedding_provider: EmbeddingProvider,
    reset: bool = False,
) -> PolicyIndexReport:
    if not documents:
        raise ValueError("At least one policy document is required")

    existing_rows = (
        await db.execute(
            select(
                PolicyDocumentRecord.document_id,
                PolicyDocumentRecord.content_hash,
            )
        )
    ).all()
    existing_hashes = {
        str(document_id): str(document_hash)
        for document_id, document_hash in existing_rows
    }
    plan = plan_policy_index(
        documents,
        existing_hashes,
        reset=reset,
    )
    documents_by_id = {
        document.policy_id: document for document in documents
    }

    if reset:
        await db.execute(delete(PolicyChunkRecord))
        await db.execute(delete(PolicyDocumentRecord))
    elif plan.removed_policy_ids:
        await db.execute(
            delete(PolicyDocumentRecord).where(
                PolicyDocumentRecord.document_id.in_(
                    plan.removed_policy_ids
                )
            )
        )

    chunks_indexed = 0
    for policy_id in plan.changed_policy_ids:
        document = documents_by_id[policy_id]
        ensure_document_unchanged(document)
        chunks = chunk_policy_document(document)
        embedding_texts = [
            f"{chunk.title}\n{chunk.section}\n{chunk.content}"
            for chunk in chunks
        ]
        embeddings = await embedding_provider.embed_documents(
            embedding_texts
        )
        ensure_document_unchanged(document)
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding provider returned the wrong vector count"
            )

        if not reset:
            await db.execute(
                delete(PolicyChunkRecord).where(
                    PolicyChunkRecord.document_id == policy_id
                )
            )

        record = await db.get(PolicyDocumentRecord, policy_id)
        if record is None:
            record = PolicyDocumentRecord(document_id=policy_id)
            db.add(record)

        record.slug = document.policy_id
        record.title = document.title
        record.version = document.version
        record.effective_date = document.effective_date
        record.source_path = document.source_path
        record.content_hash = document.content_hash
        await db.flush()

        db.add_all(
            PolicyChunkRecord(
                chunk_id=chunk.chunk_id,
                document_id=document.policy_id,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        )
        chunks_indexed += len(chunks)

    await db.commit()
    return PolicyIndexReport(
        documents_seen=len(documents),
        documents_indexed=len(plan.changed_policy_ids),
        documents_unchanged=len(plan.unchanged_policy_ids),
        documents_removed=len(plan.removed_policy_ids),
        chunks_indexed=chunks_indexed,
    )
