import json
from pathlib import Path
from typing import Any

import pytest

from scripts import annotate_cfpb_semantic_with_codex as codex_workflow
from scripts import build_cfpb_annotation_workfile as contract
from scripts import export_cfpb_semantic_final_labels as final_export
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


def second_pass_result_for(
    batch_row: dict[str, Any],
    **semantic_overrides: Any,
) -> dict[str, Any]:
    result = result_for(batch_row, **semantic_overrides)
    result["annotation_source"] = codex_workflow.SECOND_PASS_ANNOTATION_SOURCE
    return result


def adjudication_result_for(
    batch_row: dict[str, Any],
    *,
    resolution_status: str = "RESOLVED",
    category: str | None = "SINGLE_SUPPORTED_INTENT",
    intents: list[str] | None = None,
    confidence: str | None = "HIGH",
    note: str = "Concise synthetic adjudication rationale.",
    unresolved_reason: str = "",
) -> dict[str, Any]:
    return {
        "annotation_source": codex_workflow.ADJUDICATION_ANNOTATION_SOURCE,
        "annotator_model": batch_row["annotator_model"],
        "prompt_version": batch_row["prompt_version"],
        "batch_id": batch_row["batch_id"],
        "complaint_id": batch_row["complaint_id"],
        "narrative_sha256": batch_row["narrative_sha256"],
        "resolution_status": resolution_status,
        "final_review_category": category,
        "final_supported_intents": (
            ["account_balance"] if intents is None else intents
        ),
        "final_annotation_confidence": confidence,
        "final_annotation_note": note,
        "unresolved_reason": unresolved_reason,
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


def qc_source(prefix: str = "Synthetic QC narrative") -> dict[str, Any]:
    candidates = (
        source_row("1", f"{prefix} {index}.") for index in range(100)
    )
    return next(
        row
        for row in candidates
        if codex_workflow.deterministic_qc_selected(
            row["narrative_sha256"]
        )
    )


def successful_second_pass_record(
    row: dict[str, Any],
    first_pass: dict[str, Any],
    **semantic_overrides: Any,
) -> dict[str, Any]:
    _, batch = codex_workflow.prepare_second_pass_batch(
        [row],
        [first_pass],
        [],
        limit=1,
        batch_id="synthetic-second-pass-batch",
    )
    return codex_workflow.successful_second_pass_record(
        second_pass_result_for(batch[0], **semantic_overrides),
        batch[0],
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


def test_prepare_accepts_limit_200_and_rejects_above_cap() -> None:
    rows = [
        source_row(str(index), f"Synthetic batch-limit narrative {index}.")
        for index in range(201)
    ]

    _, batch = codex_workflow.prepare_batch(rows, [], limit=200)

    assert len(batch) == 200
    with pytest.raises(ValueError, match="between 1 and 200"):
        codex_workflow.prepare_batch(rows, [], limit=201)


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


def test_pass_b_selection_is_derived_from_pass_a_review_logic() -> None:
    rows = [
        non_qc_source("Synthetic unflagged selection narrative"),
        non_qc_source("Synthetic low-confidence selection narrative"),
        non_qc_source("Synthetic protected selection narrative"),
    ]
    for index, row in enumerate(rows, start=1):
        row["complaint_id"] = str(index)
    first_pass = [
        successful_record(rows[0]),
        successful_record(rows[1], confidence="LOW"),
        successful_record(rows[2], intents=["freeze_card"]),
    ]

    required = codex_workflow.second_pass_required_hashes(rows, first_pass)

    assert required == [
        rows[1]["narrative_sha256"],
        rows[2]["narrative_sha256"],
    ]


def test_pass_b_batch_is_blind_to_pass_a_and_classifier_fields() -> None:
    row = {
        **non_qc_source("Synthetic blind Pass-B narrative"),
        "classifier_prediction": "SECRET_CLASSIFIER",
        "svm_score": "SECRET_SVM",
        "logistic_probability": "SECRET_LOGISTIC",
    }
    first_pass = successful_record(
        row,
        category="MULTI_SUPPORTED_INTENT",
        intents=["account_balance", "card_status"],
        confidence="LOW",
        secondary=True,
    )

    _, batch = codex_workflow.prepare_second_pass_batch(
        [row],
        [first_pass],
        [],
        limit=1,
    )
    serialized = json.dumps(batch)

    assert not set(codex_workflow.SEMANTIC_FIELDS).intersection(batch[0])
    assert "SECRET_" not in serialized
    assert batch[0]["annotation_source"] == "CODEX_SECOND_PASS"


def test_pass_b_resume_skips_successful_hashes() -> None:
    rows = [
        non_qc_source("Synthetic completed Pass-B narrative"),
        non_qc_source("Synthetic pending Pass-B narrative"),
    ]
    rows[0]["complaint_id"] = "1"
    rows[1]["complaint_id"] = "2"
    first_pass = [
        successful_record(row, secondary=True) for row in rows
    ]
    completed = successful_second_pass_record(rows[0], first_pass[0])

    _, batch = codex_workflow.prepare_second_pass_batch(
        rows,
        first_pass,
        [completed],
        limit=2,
    )

    assert [row["narrative_sha256"] for row in batch] == [
        rows[1]["narrative_sha256"]
    ]


def test_pass_b_invalid_and_missing_results_are_requeued() -> None:
    rows = [
        non_qc_source("Synthetic invalid Pass-B narrative"),
        non_qc_source("Synthetic missing Pass-B narrative"),
    ]
    rows[0]["complaint_id"] = "1"
    rows[1]["complaint_id"] = "2"
    first_pass = [
        successful_record(row, secondary=True) for row in rows
    ]
    _, batch = codex_workflow.prepare_second_pass_batch(
        rows,
        first_pass,
        [],
        limit=2,
        batch_id="synthetic-pass-b-invalid",
    )
    invalid = second_pass_result_for(
        batch[0],
        category="SINGLE_SUPPORTED_INTENT",
        intents=[],
    )

    imported, summary = codex_workflow.import_second_pass_results(
        rows,
        first_pass,
        [],
        batch,
        [invalid],
    )
    _, retry = codex_workflow.prepare_second_pass_batch(
        rows,
        first_pass,
        imported,
        limit=2,
    )

    assert summary == {"succeeded": 0, "invalid": 1, "missing": 1}
    assert [row["status"] for row in imported] == ["INVALID", "MISSING"]
    assert [row["narrative_sha256"] for row in retry] == [
        row["narrative_sha256"] for row in rows
    ]


def test_exact_safe_ab_agreement_is_provisionally_resolved() -> None:
    row = qc_source("Synthetic safe agreement narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(row, first_pass)

    comparisons, summary = codex_workflow.compare_annotation_passes(
        [row],
        [first_pass],
        [second_pass],
    )
    preview = codex_workflow.build_provisional_final_labels(
        [row],
        [first_pass],
        [second_pass],
        [],
    )

    assert comparisons[0]["safe_agreement"] is True
    assert comparisons[0]["pass_c_required"] is False
    assert summary["exact_strong_agreements"] == 1
    assert preview[0]["provisional_label_source"] == "CODEX_DUAL_PASS_AGREEMENT"


def test_ab_disagreement_routes_to_pass_c() -> None:
    row = qc_source("Synthetic disagreement narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        category="UNSUPPORTED",
        intents=[],
    )

    comparisons, summary = codex_workflow.compare_annotation_passes(
        [row],
        [first_pass],
        [second_pass],
    )

    assert comparisons[0]["adjudication_reasons"] == [
        "SEMANTIC_DISAGREEMENT"
    ]
    assert comparisons[0]["pass_c_required"] is True
    assert summary["ab_disagreements"] == summary["pass_c_required"] == 1


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"confidence": "LOW"}, "LOW_CONFIDENCE"),
        (
            {"category": "UNCLEAR_OR_INSUFFICIENT", "intents": []},
            "UNCLEAR_OR_INSUFFICIENT",
        ),
        (
            {
                "category": "MULTI_SUPPORTED_INTENT",
                "intents": ["account_balance", "recent_transactions"],
            },
            "MULTI_SUPPORTED_INTENT",
        ),
        ({"intents": ["freeze_card"]}, "PROTECTED_WRITE_FREEZE_CARD"),
        ({"intents": ["create_dispute"]}, "PROTECTED_WRITE_CREATE_DISPUTE"),
        ({"secondary": True}, "SECONDARY_REVIEW_REQUIRED"),
    ],
    ids=("low", "unclear", "multi", "freeze", "dispute", "secondary"),
)
def test_semantic_risks_route_to_pass_c_even_on_exact_agreement(
    overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    row = non_qc_source(f"Synthetic Pass-C risk {expected_reason}")
    first_pass = successful_record(row, **overrides)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        **overrides,
    )

    comparisons, _ = codex_workflow.compare_annotation_passes(
        [row],
        [first_pass],
        [second_pass],
    )

    assert comparisons[0]["exact_semantic_agreement"] is True
    assert comparisons[0]["pass_c_required"] is True
    assert expected_reason in comparisons[0]["adjudication_reasons"]


def test_pass_c_can_resolve_with_a_corrected_third_label() -> None:
    row = qc_source("Synthetic corrected third-label narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        category="UNSUPPORTED",
        intents=[],
    )
    _, batch = codex_workflow.prepare_adjudication_batch(
        [row],
        [first_pass],
        [second_pass],
        [],
        limit=1,
    )
    corrected = adjudication_result_for(
        batch[0],
        category="SINGLE_SUPPORTED_INTENT",
        intents=["card_status"],
    )

    adjudications, _ = codex_workflow.import_adjudication_results(
        [row],
        [first_pass],
        [second_pass],
        [],
        batch,
        [corrected],
    )
    preview = codex_workflow.build_provisional_final_labels(
        [row],
        [first_pass],
        [second_pass],
        adjudications,
    )

    assert preview[0]["supported_intents"] == ["card_status"]
    assert preview[0]["provisional_label_source"] == "CODEX_ADJUDICATOR"


def test_pass_c_unresolved_has_no_final_label() -> None:
    row = qc_source("Synthetic unresolved adjudication narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        category="UNSUPPORTED",
        intents=[],
    )
    _, batch = codex_workflow.prepare_adjudication_batch(
        [row],
        [first_pass],
        [second_pass],
        [],
        limit=1,
    )
    unresolved = adjudication_result_for(
        batch[0],
        resolution_status="UNRESOLVED",
        category=None,
        intents=[],
        confidence=None,
        note="",
        unresolved_reason="Synthetic evidence remains materially ambiguous.",
    )

    adjudications, _ = codex_workflow.import_adjudication_results(
        [row],
        [first_pass],
        [second_pass],
        [],
        batch,
        [unresolved],
    )
    preview = codex_workflow.build_provisional_final_labels(
        [row],
        [first_pass],
        [second_pass],
        adjudications,
    )
    status = codex_workflow.workflow_status(
        [row],
        [first_pass],
        [second_pass],
        adjudications,
    )

    assert preview == []
    assert status["pass_c_unresolved"] == 1
    assert status["human_review_remaining"] == 1


def test_pass_c_resume_is_hash_based_and_skips_successes() -> None:
    rows = [
        qc_source("Synthetic first Pass-C resume narrative"),
        qc_source("Synthetic second Pass-C resume narrative"),
    ]
    rows[0]["complaint_id"] = "1"
    rows[1]["complaint_id"] = "2"
    first_pass = [successful_record(row) for row in rows]
    second_pass = [
        successful_second_pass_record(
            row,
            first,
            category="UNSUPPORTED",
            intents=[],
        )
        for row, first in zip(rows, first_pass, strict=True)
    ]
    _, first_batch = codex_workflow.prepare_adjudication_batch(
        rows,
        first_pass,
        second_pass,
        [],
        limit=1,
    )
    adjudications, _ = codex_workflow.import_adjudication_results(
        rows,
        first_pass,
        second_pass,
        [],
        first_batch,
        [adjudication_result_for(first_batch[0])],
    )

    _, resumed = codex_workflow.prepare_adjudication_batch(
        rows,
        first_pass,
        second_pass,
        adjudications,
        limit=2,
    )

    assert [row["narrative_sha256"] for row in resumed] == [
        rows[1]["narrative_sha256"]
    ]


def test_successful_pass_c_result_cannot_be_silently_overwritten() -> None:
    row = qc_source("Synthetic Pass-C overwrite narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        category="UNSUPPORTED",
        intents=[],
    )
    _, batch = codex_workflow.prepare_adjudication_batch(
        [row], [first_pass], [second_pass], [], limit=1
    )
    result = adjudication_result_for(batch[0])
    existing, _ = codex_workflow.import_adjudication_results(
        [row], [first_pass], [second_pass], [], batch, [result]
    )

    with pytest.raises(ValueError, match="refusing to overwrite"):
        codex_workflow.import_adjudication_results(
            [row],
            [first_pass],
            [second_pass],
            existing,
            batch,
            [result],
        )


def test_pass_b_and_pass_c_atomic_persistence_round_trips(
    tmp_path: Path,
) -> None:
    row = qc_source("Synthetic later-pass atomic narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        category="UNSUPPORTED",
        intents=[],
    )
    second_path = tmp_path / "second-pass.jsonl"
    reviewer.atomic_write_records(second_path, [second_pass])
    _, batch = codex_workflow.prepare_adjudication_batch(
        [row], [first_pass], [second_pass], [], limit=1
    )
    adjudication = codex_workflow.successful_adjudication_record(
        adjudication_result_for(batch[0]),
        batch[0],
    )
    adjudication_path = tmp_path / "adjudication.jsonl"
    reviewer.atomic_write_records(adjudication_path, [adjudication])

    assert codex_workflow.load_second_pass_records(second_path) == [second_pass]
    assert codex_workflow.load_adjudication_records(adjudication_path) == [
        adjudication
    ]


def test_pass_c_batch_excludes_classifier_output_fields() -> None:
    row = {
        **qc_source("Synthetic Pass-C leakage narrative"),
        "classifier_prediction": "SECRET_CLASSIFIER",
        "svm_margin": "SECRET_SVM",
        "logistic_probability": "SECRET_LOGISTIC",
    }
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        category="UNSUPPORTED",
        intents=[],
    )

    _, batch = codex_workflow.prepare_adjudication_batch(
        [row], [first_pass], [second_pass], [], limit=1
    )

    assert "SECRET_" not in json.dumps(batch)
    assert batch[0]["pass_a"]["annotation_note"]
    assert batch[0]["pass_b"]["annotation_note"]


def test_existing_pass_a_cli_commands_remain_available() -> None:
    parser = codex_workflow.build_parser()

    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["status"]).command == "status"
    assert parser.parse_args(
        ["import", "--batch", "batch.jsonl", "--annotations", "result.jsonl"]
    ).command == "import"


def test_workflow_status_is_deterministic_and_internally_consistent() -> None:
    row = qc_source("Synthetic workflow status narrative")
    first_pass = successful_record(row)
    second_pass = successful_second_pass_record(row, first_pass)

    first = codex_workflow.workflow_status(
        [row], [first_pass], [second_pass], []
    )
    second = codex_workflow.workflow_status(
        [row], [first_pass], [second_pass], []
    )

    assert first == second
    assert first["total_holdout"] == 1
    assert first["pass_b_required"] == first["pass_b_complete"] == 1
    assert first["pass_b_pending"] == first["pass_c_required"] == 0
    assert first["final_labels_currently_available"] == 1
    assert first["human_review_remaining"] == 0


def final_adjudication_record(
    row: dict[str, Any],
    first_pass: dict[str, Any],
    second_pass: dict[str, Any],
    *,
    resolution_status: str = "RESOLVED",
    category: str | None = "SINGLE_SUPPORTED_INTENT",
    intents: list[str] | None = None,
    confidence: str | None = "HIGH",
) -> dict[str, Any]:
    _, batch = codex_workflow.prepare_adjudication_batch(
        [row],
        [first_pass],
        [second_pass],
        [],
        limit=1,
        batch_id="synthetic-final-export-adjudication",
    )
    unresolved = resolution_status == "UNRESOLVED"
    result = adjudication_result_for(
        batch[0],
        resolution_status=resolution_status,
        category=None if unresolved else category,
        intents=[] if unresolved else intents,
        confidence=None if unresolved else confidence,
        note="" if unresolved else "Synthetic final adjudication rationale.",
        unresolved_reason=(
            "Synthetic record cannot be resolved." if unresolved else ""
        ),
    )
    return codex_workflow.successful_adjudication_record(result, batch[0])


def test_final_export_pass_a_only_record_uses_pass_a() -> None:
    row = non_qc_source("Synthetic final Pass-A-only narrative")
    first_pass = successful_record(row, intents=["transaction_details"])

    records = final_export.build_final_labels(
        [row], [first_pass], [], [], expected_count=1
    )

    assert records[0]["final_annotation_source"] == "CODEX_FIRST_PASS"
    assert records[0]["final_supported_intents"] == ["transaction_details"]
    assert records[0]["pass_b_required"] is False
    assert records[0]["pass_c_required"] is False


def test_final_export_safe_ab_agreement_uses_agreed_semantics() -> None:
    row = qc_source("Synthetic final safe agreement narrative")
    first_pass = successful_record(row, intents=["recent_transactions"])
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        intents=["recent_transactions"],
    )

    records = final_export.build_final_labels(
        [row], [first_pass], [second_pass], [], expected_count=1
    )

    assert records[0]["final_annotation_source"] == "CODEX_DUAL_PASS_AGREEMENT"
    assert records[0]["final_supported_intents"] == ["recent_transactions"]
    assert records[0]["pass_b_required"] is True
    assert records[0]["pass_c_required"] is False


def test_final_export_resolved_pass_c_overrides_ab() -> None:
    row = qc_source("Synthetic final Pass-C override narrative")
    first_pass = successful_record(row, intents=["account_balance"])
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        intents=["card_status"],
    )
    adjudication = final_adjudication_record(
        row,
        first_pass,
        second_pass,
        intents=["transaction_details"],
    )

    records = final_export.build_final_labels(
        [row],
        [first_pass],
        [second_pass],
        [adjudication],
        expected_count=1,
    )

    assert records[0]["final_annotation_source"] == "CODEX_ADJUDICATOR"
    assert records[0]["final_supported_intents"] == ["transaction_details"]
    assert records[0]["pass_c_required"] is True


def test_final_export_rejects_unresolved_pass_c() -> None:
    row = qc_source("Synthetic final unresolved Pass-C narrative")
    first_pass = successful_record(row, intents=["account_balance"])
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        intents=["card_status"],
    )
    adjudication = final_adjudication_record(
        row,
        first_pass,
        second_pass,
        resolution_status="UNRESOLVED",
    )

    with pytest.raises(ValueError, match="UNRESOLVED Pass-C"):
        final_export.build_final_labels(
            [row],
            [first_pass],
            [second_pass],
            [adjudication],
            expected_count=1,
        )


@pytest.mark.parametrize(
    ("category", "intents", "expected"),
    [
        ("SINGLE_SUPPORTED_INTENT", ["account_balance"], "account_balance"),
        ("UNSUPPORTED", [], "unsupported_or_uncertain"),
        ("UNCLEAR_OR_INSUFFICIENT", [], "unsupported_or_uncertain"),
        ("NO_CURRENT_REQUEST", [], "unsupported_or_uncertain"),
    ],
)
def test_frozen_primary_single_label_mapping(
    category: str,
    intents: list[str],
    expected: str,
) -> None:
    assert final_export.primary_single_label_target(category, intents) == expected


def test_multi_intent_has_no_forced_single_label_target() -> None:
    assert (
        final_export.primary_single_label_target(
            "MULTI_SUPPORTED_INTENT",
            ["account_balance", "recent_transactions"],
        )
        is None
    )


def test_multi_intent_set_membership_metric_semantics() -> None:
    intents = ["account_balance", "recent_transactions"]

    assert final_export.multi_intent_prediction_is_hit(
        "recent_transactions", intents
    )
    assert not final_export.multi_intent_prediction_is_hit("card_status", intents)
    assert not final_export.multi_intent_prediction_is_hit(
        "unsupported_or_uncertain", intents
    )


def test_final_export_is_text_free_and_rejects_extra_narrative_field() -> None:
    row = non_qc_source("SECRET_SYNTHETIC_FINAL_NARRATIVE")
    first_pass = successful_record(row)
    records = final_export.build_final_labels(
        [row], [first_pass], [], [], expected_count=1
    )
    payload = contract.stable_jsonl_bytes(records).decode("utf-8")

    assert "SECRET_SYNTHETIC_FINAL_NARRATIVE" not in payload
    assert "narrative" not in records[0]
    invalid = {**records[0], "narrative": row["narrative"]}
    with pytest.raises(ValueError, match="fields differ"):
        final_export.validate_final_export_records([invalid], expected_count=1)


def test_final_export_omits_annotation_and_adjudication_notes() -> None:
    row = qc_source("Synthetic final omitted-note narrative")
    first_pass = successful_record(row, intents=["account_balance"])
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        intents=["card_status"],
    )
    adjudication = final_adjudication_record(
        row,
        first_pass,
        second_pass,
        intents=["recent_transactions"],
    )

    record = final_export.build_final_labels(
        [row],
        [first_pass],
        [second_pass],
        [adjudication],
        expected_count=1,
    )[0]

    assert "annotation_note" not in record
    assert "final_annotation_note" not in record
    assert "unresolved_reason" not in record


def test_final_export_omits_classifier_derived_source_fields() -> None:
    row = {
        **non_qc_source("Synthetic final classifier-blind narrative"),
        "classifier_prediction": "SECRET_CLASSIFIER",
        "svm_score": "SECRET_SVM",
        "logistic_probability": "SECRET_LOGISTIC",
    }
    first_pass = successful_record(row)

    records = final_export.build_final_labels(
        [row], [first_pass], [], [], expected_count=1
    )
    serialized = json.dumps(records)

    assert "SECRET_" not in serialized
    assert not contract.FORBIDDEN_MODEL_FIELDS.intersection(records[0])


def test_final_export_rejects_duplicate_hashes() -> None:
    first = non_qc_source("Synthetic final duplicate-hash narrative")
    second = {**first, "complaint_id": "2"}
    first_pass = successful_record(first)

    with pytest.raises(ValueError, match="hashes must be unique"):
        final_export.build_final_labels(
            [first, second],
            [first_pass, {**first_pass, "complaint_id": "2"}],
            [],
            [],
            expected_count=2,
        )


def test_final_export_rejects_invalid_category_intent_contract() -> None:
    row = non_qc_source("Synthetic final invalid-category narrative")
    first_pass = {
        **successful_record(row),
        "supported_intents": [],
    }

    with pytest.raises(ValueError, match="exactly one intent"):
        final_export.build_final_labels(
            [row], [first_pass], [], [], expected_count=1
        )


def test_final_export_order_and_bytes_are_deterministic() -> None:
    rows = [
        non_qc_source("Synthetic final deterministic second narrative"),
        non_qc_source("Synthetic final deterministic first narrative"),
    ]
    rows[0]["complaint_id"] = "2"
    rows[1]["complaint_id"] = "1"
    first_pass = [successful_record(row) for row in rows]

    first = final_export.build_final_labels(
        rows, first_pass, [], [], expected_count=2
    )
    second = final_export.build_final_labels(
        rows, first_pass, [], [], expected_count=2
    )

    assert [record["complaint_id"] for record in first] == ["2", "1"]
    assert contract.stable_jsonl_bytes(first) == contract.stable_jsonl_bytes(second)


def test_final_export_write_is_atomic_and_reproducible(tmp_path: Path) -> None:
    row = non_qc_source("Synthetic final atomic-write narrative")
    first_pass = successful_record(row)
    records = final_export.build_final_labels(
        [row], [first_pass], [], [], expected_count=1
    )
    output_path = tmp_path / "final-labels.jsonl"

    first_payload = final_export.write_final_labels(
        output_path, records, expected_count=1
    )
    second_payload = final_export.write_final_labels(
        output_path, records, expected_count=1
    )

    assert first_payload == second_payload == output_path.read_bytes()
    assert not list(tmp_path.glob(".final-labels.jsonl.*.tmp"))


def test_final_export_requires_exactly_1800_by_default() -> None:
    row = non_qc_source("Synthetic final completeness narrative")
    first_pass = successful_record(row)

    with pytest.raises(ValueError, match="expected 1800 source"):
        final_export.build_final_labels([row], [first_pass], [], [])


def test_final_export_preserves_protected_write_label() -> None:
    row = non_qc_source("Synthetic explicit protected-write request")
    first_pass = successful_record(row, intents=["freeze_card"])
    second_pass = successful_second_pass_record(
        row,
        first_pass,
        intents=["freeze_card"],
    )
    adjudication = final_adjudication_record(
        row,
        first_pass,
        second_pass,
        intents=["freeze_card"],
    )

    record = final_export.build_final_labels(
        [row],
        [first_pass],
        [second_pass],
        [adjudication],
        expected_count=1,
    )[0]

    assert record["final_supported_intents"] == ["freeze_card"]
    assert record["primary_single_label_target"] == "freeze_card"


def test_final_export_schema_contains_no_v2_c1_prediction_field() -> None:
    assert "prediction" not in final_export.FINAL_LABEL_FIELDS
    assert "predicted_intent" not in final_export.FINAL_LABEL_FIELDS
    assert "svm_score" not in final_export.FINAL_LABEL_FIELDS
    assert "logistic_probability" not in final_export.FINAL_LABEL_FIELDS
