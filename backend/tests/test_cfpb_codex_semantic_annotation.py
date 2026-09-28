import json
from pathlib import Path
from typing import Any

import pytest

from scripts import annotate_cfpb_semantic_with_codex as codex_workflow
from scripts import build_cfpb_annotation_workfile as contract
from scripts import review_cfpb_semantic_annotations as reviewer


def source_row(
    complaint_id: str,
    narrative: str,
    *,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "complaint_id": complaint_id,
        "narrative": narrative,
        "narrative_sha256": contract.sha256_bytes(narrative.encode("utf-8")),
        "mapping_status": "NEAR_MATCH",
        "candidate_sentinelvoice_intents": candidates or ["account_balance"],
        "product": "Synthetic product",
        "sub_product": "Synthetic sub-product",
        "issue": "Synthetic issue",
        "sub_issue": "Synthetic sub-issue",
        **contract.initialized_annotation_fields(),
    }


def semantic_fields(
    *,
    category: str = "SINGLE_SUPPORTED_INTENT",
    intents: list[str] | None = None,
    confidence: str = "HIGH",
    secondary: bool = False,
) -> dict[str, Any]:
    return {
        "review_category": category,
        "supported_intents": ["account_balance"] if intents is None else intents,
        "annotation_confidence": confidence,
        "annotation_note": "Concise synthetic rationale.",
        "secondary_review_required": secondary,
    }


def result_for(
    batch_row: dict[str, Any],
    **semantic_overrides: Any,
) -> dict[str, Any]:
    fields = semantic_fields(**semantic_overrides)
    return {
        "annotation_source": codex_workflow.ANNOTATION_SOURCE,
        "annotator_model": batch_row["annotator_model"],
        "prompt_version": batch_row["prompt_version"],
        "batch_id": batch_row["batch_id"],
        "complaint_id": batch_row["complaint_id"],
        "narrative_sha256": batch_row["narrative_sha256"],
        **fields,
    }


def prepared_batch(
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    _, batch = codex_workflow.prepare_batch(
        rows,
        [],
        limit=limit or len(rows),
        batch_id="synthetic-batch",
    )
    return batch


def successful_record(
    row: dict[str, Any],
    **semantic_overrides: Any,
) -> dict[str, Any]:
    batch_row = prepared_batch([row])[0]
    return codex_workflow.successful_first_pass_record(
        result_for(batch_row, **semantic_overrides),
        batch_row,
    )


def non_qc_source(prefix: str = "Synthetic non-QC narrative") -> dict[str, Any]:
    candidates = (
        source_row("1", f"{prefix} {index}.") for index in range(100)
    )
    return next(
        row
        for row in candidates
        if not codex_workflow.deterministic_qc_selected(
            row["narrative_sha256"]
        )
    )


def test_codex_first_pass_row_validation_and_provenance() -> None:
    row = non_qc_source()
    batch_row = prepared_batch([row])[0]
    result = result_for(batch_row)

    validated = codex_workflow.validate_codex_result(result, batch_row)
    first_pass = codex_workflow.successful_first_pass_record(
        validated,
        batch_row,
    )
    codex_workflow.validate_first_pass_record(first_pass)

    assert first_pass["annotation_source"] == "CODEX_FIRST_PASS"
    assert first_pass["annotator_model"] == "codex"
    assert first_pass["prompt_version"] == codex_workflow.PROMPT_VERSION
    assert first_pass["batch_id"] == "synthetic-batch"
    assert "narrative" not in first_pass


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"category": "SINGLE_SUPPORTED_INTENT", "intents": []}, "exactly one"),
        (
            {
                "category": "SINGLE_SUPPORTED_INTENT",
                "intents": ["account_balance", "card_status"],
            },
            "exactly one",
        ),
        (
            {"category": "MULTI_SUPPORTED_INTENT", "intents": ["card_status"]},
            "at least two",
        ),
        (
            {"category": "UNSUPPORTED", "intents": ["account_balance"]},
            "empty supported_intents",
        ),
        ({"confidence": "CERTAIN"}, "confidence"),
    ],
)
def test_invalid_codex_annotations_are_not_accepted(
    overrides: dict[str, Any],
    message: str,
) -> None:
    row = source_row("1", "Synthetic invalid annotation narrative.")
    batch_row = prepared_batch([row])[0]

    with pytest.raises(ValueError, match=message):
        codex_workflow.validate_codex_result(
            result_for(batch_row, **overrides),
            batch_row,
        )


def test_import_appends_validated_results_in_source_order() -> None:
    rows = [
        source_row("1", "Synthetic first import narrative."),
        source_row("2", "Synthetic second import narrative."),
    ]
    batch = prepared_batch(rows)
    results = [result_for(batch[1]), result_for(batch[0])]

    merged, summary = codex_workflow.import_codex_results(
        rows,
        [],
        batch,
        results,
    )

    assert [record["narrative_sha256"] for record in merged] == [
        row["narrative_sha256"] for row in rows
    ]
    assert summary == {"succeeded": 2, "invalid": 0, "missing": 0}


def test_import_marks_invalid_and_missing_rows_for_human_review() -> None:
    rows = [
        source_row("1", "Synthetic invalid import narrative."),
        source_row("2", "Synthetic missing import narrative."),
    ]
    batch = prepared_batch(rows)
    invalid = result_for(
        batch[0],
        category="SINGLE_SUPPORTED_INTENT",
        intents=[],
    )

    merged, summary = codex_workflow.import_codex_results(
        rows,
        [],
        batch,
        [invalid],
    )

    assert summary == {"succeeded": 0, "invalid": 1, "missing": 1}
    assert [record["status"] for record in merged] == ["INVALID", "MISSING"]
    assert all(record["human_review_required"] for record in merged)
    assert all(record["review_category"] is None for record in merged)


def test_duplicate_results_and_successful_overwrite_are_rejected() -> None:
    row = source_row("1", "Synthetic duplicate-prevention narrative.")
    batch = prepared_batch([row])
    result = result_for(batch[0])

    with pytest.raises(ValueError, match="duplicate hash"):
        codex_workflow.import_codex_results(
            [row],
            [],
            batch,
            [result, result],
        )

    existing = [successful_record(row)]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        codex_workflow.import_codex_results(
            [row],
            existing,
            batch,
            [result],
        )


def test_explicit_successful_replacement_is_supported() -> None:
    row = source_row("1", "Synthetic explicit replacement narrative.")
    batch = prepared_batch([row])
    existing = [successful_record(row)]
    replacement = result_for(
        batch[0],
        category="UNSUPPORTED",
        intents=[],
        confidence="MEDIUM",
    )

    merged, summary = codex_workflow.import_codex_results(
        [row],
        existing,
        batch,
        [replacement],
        replace_successful=True,
    )

    assert summary["succeeded"] == 1
    assert merged[0]["review_category"] == "UNSUPPORTED"


def test_resume_skips_successes_and_requeues_unresolved_rows() -> None:
    rows = [
        source_row("1", "Synthetic completed resume narrative."),
        source_row("2", "Synthetic invalid resume narrative."),
        source_row("3", "Synthetic pending resume narrative."),
    ]
    first_success = successful_record(rows[0])
    invalid_batch = prepared_batch([rows[1]])[0]
    invalid = codex_workflow.unresolved_first_pass_record(
        invalid_batch,
        status="INVALID",
        reason="CODEX_ANNOTATION_INVALID",
        validation_error=ValueError("synthetic"),
    )

    _, batch = codex_workflow.prepare_batch(
        rows,
        [first_success, invalid],
        limit=2,
        batch_id="resume-batch",
    )
    progress = codex_workflow.first_pass_progress(
        rows,
        [first_success, invalid],
    )

    assert [record["narrative_sha256"] for record in batch] == [
        rows[1]["narrative_sha256"],
        rows[2]["narrative_sha256"],
    ]
    assert progress == {
        "total": 3,
        "succeeded": 1,
        "invalid": 1,
        "missing": 0,
        "pending": 1,
        "human_review_required": 1,
    }


def test_atomic_first_pass_persistence_round_trip(tmp_path: Path) -> None:
    rows = [
        source_row("1", "Synthetic atomic first row."),
        source_row("2", "Synthetic atomic second row."),
    ]
    batch = prepared_batch(rows)
    merged, _ = codex_workflow.import_codex_results(
        rows,
        [],
        batch,
        [result_for(record) for record in batch],
    )
    output_path = tmp_path / "first-pass.jsonl"

    reviewer.atomic_write_records(output_path, merged)

    assert codex_workflow.load_first_pass_records(output_path) == merged
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_reviewer_targets_only_flagged_codex_rows(tmp_path: Path) -> None:
    unflagged = non_qc_source("Synthetic reviewer unflagged narrative")
    flagged = source_row("2", "Synthetic reviewer protected narrative.")
    first_pass = [
        successful_record(unflagged),
        successful_record(flagged, intents=["freeze_card"]),
    ]
    path = tmp_path / "first-pass.jsonl"
    reviewer.atomic_write_records(path, first_pass)

    context = reviewer.load_first_pass_context(path, [unflagged, flagged])

    assert reviewer.flagged_review_indices(
        [unflagged, flagged],
        context,
    ) == [1]
    assert unflagged["adjudication_status"] == "UNREVIEWED"
    assert flagged["adjudication_status"] == "UNREVIEWED"


def test_batch_allowlist_excludes_classifier_and_existing_human_fields() -> None:
    row = {
        **source_row("1", "Synthetic leakage-check narrative."),
        "classifier_prediction": "SECRET_CLASSIFIER",
        "svm_margin": "SECRET_SVM",
        "logistic_probability": "SECRET_LOGISTIC",
        "abstention_result": "SECRET_ABSTENTION",
        "review_category": "UNSUPPORTED",
        "annotation_confidence": "HIGH",
        "annotation_note": "SECRET_HUMAN_NOTE",
        "reviewer_id": "SECRET_REVIEWER",
        "adjudication_status": "REVIEWED",
    }
    before = json.loads(json.dumps(row))

    batch = prepared_batch([row])
    serialized = json.dumps(batch)

    assert row == before
    assert "SECRET_" not in serialized
    assert "taxonomy_candidate_intents_hints_only" in serialized


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        (semantic_fields(confidence="LOW"), "LOW_CONFIDENCE"),
        (
            semantic_fields(category="UNCLEAR_OR_INSUFFICIENT", intents=[]),
            "UNCLEAR_OR_INSUFFICIENT",
        ),
        (
            semantic_fields(
                category="MULTI_SUPPORTED_INTENT",
                intents=["account_balance", "recent_transactions"],
            ),
            "MULTI_SUPPORTED_INTENT",
        ),
        (semantic_fields(intents=["freeze_card"]), "PROTECTED_WRITE_FREEZE_CARD"),
        (
            semantic_fields(intents=["create_dispute"]),
            "PROTECTED_WRITE_CREATE_DISPUTE",
        ),
        (
            semantic_fields(secondary=True),
            "CODEX_REQUESTED_SECONDARY_REVIEW",
        ),
    ],
)
def test_mandatory_human_review_flags(
    fields: dict[str, Any],
    reason: str,
) -> None:
    reasons = codex_workflow.human_review_reasons(
        fields,
        narrative_sha256="a" * 64,
    )

    assert reason in reasons


def test_deterministic_qc_is_stable_and_only_for_uncomplicated_rows() -> None:
    selected_hash = next(
        f"{value:064x}"
        for value in range(10_000)
        if codex_workflow.deterministic_qc_selected(f"{value:064x}")
    )

    first = codex_workflow.human_review_reasons(
        semantic_fields(),
        narrative_sha256=selected_hash,
    )
    second = codex_workflow.human_review_reasons(
        semantic_fields(),
        narrative_sha256=selected_hash,
    )
    already_flagged = codex_workflow.human_review_reasons(
        semantic_fields(confidence="LOW"),
        narrative_sha256=selected_hash,
    )

    assert first == second == ["DETERMINISTIC_QC_SAMPLE"]
    assert "DETERMINISTIC_QC_SAMPLE" not in already_flagged


def test_protected_writes_are_not_inferred_from_text_or_hints() -> None:
    row = source_row(
        "1",
        "A synthetic history mentions theft, fraud, and a prior dispute.",
        candidates=["freeze_card", "create_dispute"],
    )
    batch_row = prepared_batch([row])[0]
    result = result_for(
        batch_row,
        category="NO_CURRENT_REQUEST",
        intents=[],
    )

    first_pass = codex_workflow.successful_first_pass_record(result, batch_row)

    assert first_pass["supported_intents"] == []
    assert "PROTECTED_WRITE_FREEZE_CARD" not in first_pass["human_review_reasons"]
    assert "PROTECTED_WRITE_CREATE_DISPUTE" not in first_pass["human_review_reasons"]


def test_safe_reset_requires_confirmation_and_preserves_backup(tmp_path: Path) -> None:
    path = tmp_path / "semantic-review.jsonl"
    backup_path = tmp_path / "semantic-review.backup.jsonl"
    reviewed = {
        **source_row("1", "Synthetic manually reviewed narrative."),
        "review_category": "SINGLE_SUPPORTED_INTENT",
        "supported_intents": ["account_balance"],
        "annotation_confidence": "HIGH",
        "annotation_note": "Synthetic manual label.",
        "reviewer_id": "synthetic-reviewer",
        "adjudication_status": "REVIEWED",
    }
    untouched = source_row("2", "Synthetic untouched narrative.")
    reviewer.atomic_write_records(path, [reviewed, untouched])
    original = path.read_bytes()

    with pytest.raises(ValueError, match="requires confirmation"):
        reviewer.reset_reviewed_annotations(
            path,
            [reviewed, untouched],
            confirmation="wrong",
            backup_path=backup_path,
        )
    assert path.read_bytes() == original
    assert not backup_path.exists()

    reset, count, created_backup = reviewer.reset_reviewed_annotations(
        path,
        [reviewed, untouched],
        confirmation=reviewer.RESET_CONFIRMATION,
        backup_path=backup_path,
    )

    assert count == 1
    assert created_backup == backup_path
    assert backup_path.read_bytes() == original
    assert reset[0]["adjudication_status"] == "UNREVIEWED"
    assert reset[1] == untouched


def test_no_external_provider_dependency() -> None:
    source = Path(codex_workflow.__file__).read_text(encoding="utf-8")

    assert "GroqLLMProvider" not in source
    assert "GROQ_API_KEY" not in source
    assert "backend.app.providers" not in source
    assert "AsyncGroq" not in source


def test_codex_hybrid_export_is_text_free(tmp_path: Path) -> None:
    row = non_qc_source("Synthetic export narrative")
    row["company"] = "Synthetic company"
    row["state"] = "ZZ"
    first_pass = successful_record(row)
    manifest_record = {
        "candidate_sentinelvoice_intents": row[
            "candidate_sentinelvoice_intents"
        ],
        "complaint_id": row["complaint_id"],
        "mapping_status": row["mapping_status"],
        "narrative_sha256": row["narrative_sha256"],
    }
    manifest = {
        "schema_version": contract.REVIEW_POOL_SCHEMA_VERSION,
        "selected_records": [manifest_record],
        "summary": {
            "selected_by_mapping_status": {
                "NEAR_MATCH": 1,
                "AMBIGUOUS": 0,
                "UNSUPPORTED": 0,
            },
            "total_selected": 1,
        },
    }
    output_path = tmp_path / "labels.json"

    artifact = contract.export_semantic_labels(
        [row],
        manifest,
        workfile_sha256="synthetic-workfile-sha256",
        review_pool_manifest_sha256="synthetic-manifest-sha256",
        output_path=output_path,
        codex_first_pass_rows=[first_pass],
        codex_first_pass_sha256="synthetic-first-pass-sha256",
        expected_lane_counts={
            "NEAR_MATCH": 1,
            "AMBIGUOUS": 0,
            "UNSUPPORTED": 0,
        },
    )

    exported = artifact["records"][0]
    assert artifact["partial"] is False
    assert exported["annotation_provenance"] == "CODEX_FIRST_PASS_ACCEPTED"
    assert exported["adjudication_status"] == "CODEX_ACCEPTED"
    assert "reviewer_id" not in exported
    assert "company" not in exported
    assert "state" not in exported
    assert row["narrative"] not in output_path.read_text(encoding="utf-8")
