from pathlib import Path

import pytest

from backend.app.rag.chunking import (
    PolicyDocumentError,
    chunk_policy_document,
    content_hash,
    load_policy_documents,
    parse_policy_document,
)
from backend.app.rag.indexing import (
    ensure_document_unchanged,
    plan_policy_index,
)

POLICY_DIRECTORY = Path("data/policies")


def test_version_controlled_policy_documents_parse_with_metadata() -> None:
    documents = load_policy_documents(POLICY_DIRECTORY)

    assert len(documents) == 7
    assert {document.policy_id for document in documents} == {
        "debit-card-freeze",
        "information-access",
        "retrieved-content-safety",
        "support-escalation",
        "transaction-disputes",
        "transaction-status",
        "unauthorized-card-transactions",
    }
    assert all(document.version == "1.0" for document in documents)
    assert all(document.sections for document in documents)


def test_heading_aware_chunking_is_stable_and_compact() -> None:
    document = parse_policy_document(
        POLICY_DIRECTORY / "transaction-disputes.md"
    )

    first = chunk_policy_document(document, max_chars=500)
    second = chunk_policy_document(document, max_chars=500)

    assert first == second
    assert [chunk.section for chunk in first] == [
        "Filing window",
        "Transaction eligibility",
        "Review process",
    ]
    assert all(len(chunk.content) <= 500 for chunk in first)
    assert first[0].chunk_id == "transaction-disputes:filing-window:00"


def test_malformed_policy_document_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("# No frontmatter\n\nPolicy text", encoding="utf-8")

    with pytest.raises(PolicyDocumentError, match="frontmatter"):
        parse_policy_document(path)


def test_repeated_headings_still_produce_unique_stable_chunk_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repeated.md"
    path.write_text(
        """---
policy_id: repeated-policy
title: Repeated Policy
version: 1.0
effective_date: 2026-09-01
---
## Rules

First rule.

## Rules

Second rule.
""",
        encoding="utf-8",
    )

    chunks = chunk_policy_document(parse_policy_document(path))

    assert [chunk.chunk_id for chunk in chunks] == [
        "repeated-policy:rules:00",
        "repeated-policy:rules-01:00",
    ]


def test_content_hash_and_index_plan_are_idempotent() -> None:
    documents = load_policy_documents(POLICY_DIRECTORY)
    existing = {
        document.policy_id: document.content_hash for document in documents
    }

    plan = plan_policy_index(documents, existing)

    assert plan.changed_policy_ids == ()
    assert len(plan.unchanged_policy_ids) == 7
    assert plan.removed_policy_ids == ()
    assert content_hash("same\r\n") == content_hash("same\n")


def test_index_plan_replaces_changed_and_removes_deleted_policy() -> None:
    documents = load_policy_documents(POLICY_DIRECTORY)
    existing = {
        documents[0].policy_id: "0" * 64,
        "removed-policy": "1" * 64,
    }

    plan = plan_policy_index(documents, existing)

    assert documents[0].policy_id in plan.changed_policy_ids
    assert "removed-policy" in plan.removed_policy_ids


def test_changed_source_during_indexing_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        """---
policy_id: test-policy
title: Test Policy
version: 1.0
effective_date: 2026-09-01
---
## Rules

Original content.
""",
        encoding="utf-8",
    )
    document = parse_policy_document(path)
    path.write_text(path.read_text() + "Changed.\n", encoding="utf-8")

    with pytest.raises(PolicyDocumentError, match="changed"):
        ensure_document_unchanged(document)
