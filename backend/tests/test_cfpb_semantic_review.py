from pathlib import Path

import pytest

from scripts import build_cfpb_annotation_workfile as workflow

SYNTHETIC_COUNTS = {
    "NEAR_MATCH": 1,
    "AMBIGUOUS": 1,
    "UNSUPPORTED": 1,
}


def source_row(
    complaint_id: str,
    narrative: str,
    mapping_status: str,
    candidates: list[str],
) -> dict[str, object]:
    return {
        "candidate_sentinelvoice_intents": candidates,
        "complaint_id": complaint_id,
        "date_received": "2025-01-01",
        "issue": f"Synthetic {mapping_status} issue",
        "length_bin": "1-200",
        "mapping_status": mapping_status,
        "narrative": narrative,
        "narrative_length": len(narrative),
        "narrative_sha256": workflow.sha256_bytes(narrative.encode("utf-8")),
        "product": "Synthetic product",
        "source_archive": "synthetic.zip",
        "sub_issue": "Synthetic sub-issue",
        "sub_product": "Synthetic sub-product",
    }


def manifest_record(row: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_sentinelvoice_intents": row[
            "candidate_sentinelvoice_intents"
        ],
        "complaint_id": row["complaint_id"],
        "date_received": row["date_received"],
        "issue": row["issue"],
        "length_bin": row["length_bin"],
        "mapping_status": row["mapping_status"],
        "narrative_length": row["narrative_length"],
        "narrative_sha256": row["narrative_sha256"],
        "privacy_screening_signal": False,
        "product": row["product"],
        "review_length_group": "lte_500",
        "selection_key": "f" * 64,
        "selection_rank": 1,
        "source_archive": row["source_archive"],
        "sub_issue": row["sub_issue"],
        "sub_product": row["sub_product"],
    }


@pytest.fixture()
def synthetic_inputs(tmp_path: Path) -> dict[str, object]:
    near = source_row(
        "101",
        "Please show my current balance.",
        "NEAR_MATCH",
        ["account_balance", "recent_transactions"],
    )
    ambiguous = source_row(
        "102",
        "Please explain this charge and freeze my card now.",
        "AMBIGUOUS",
        ["transaction_details", "freeze_card", "create_dispute"],
    )
    unsupported = source_row(
        "103",
        "Please help refinance my mortgage.",
        "UNSUPPORTED",
        ["unsupported_or_uncertain"],
    )
    source_rows = [ambiguous, unsupported, near]
    manifest = {
        "schema_version": workflow.REVIEW_POOL_SCHEMA_VERSION,
        "selected_records": [
            manifest_record(ambiguous),
            manifest_record(unsupported),
            manifest_record(near),
        ],
        "summary": {
            "selected_by_mapping_status": dict(SYNTHETIC_COUNTS),
            "total_selected": 3,
        },
    }
    return {
        "manifest": manifest,
        "output_path": tmp_path / "semantic-review.jsonl",
        "source_rows": source_rows,
        "tmp_path": tmp_path,
    }


def build_synthetic_workfile(inputs: dict[str, object]) -> list[dict[str, object]]:
    return workflow.build_annotation_workfile(
        inputs["source_rows"],
        inputs["manifest"],
        output_path=inputs["output_path"],
        reset_existing=True,
        expected_lane_counts=SYNTHETIC_COUNTS,
    )


def completed_row(
    row: dict[str, object],
    *,
    category: str,
    intents: list[str],
    confidence: str = "HIGH",
) -> dict[str, object]:
    return {
        **row,
        "adjudication_status": "REVIEWED",
        "annotation_confidence": confidence,
        "annotation_note": "Synthetic rationale based on the current request.",
        "review_category": category,
        "reviewer_id": "reviewer-synthetic",
        "secondary_review_required": False,
        "supported_intents": intents,
    }


def test_production_reconciliation_declares_exactly_1800_semantic_rows() -> None:
    manifest = {
        "schema_version": workflow.REVIEW_POOL_SCHEMA_VERSION,
        "summary": {
            "selected_by_mapping_status": dict(
                workflow.DEFAULT_POOL_LANE_COUNTS
            ),
            "total_selected": 3_800,
        },
    }

    reconciliation = workflow.reconcile_review_pool_manifest(manifest)

    assert reconciliation == {
        "excluded_unsupported_count": 2_000,
        "semantic_review_count": 1_800,
        "semantic_review_statuses": ["NEAR_MATCH", "AMBIGUOUS"],
    }


def test_workfile_includes_only_semantic_lanes_without_auto_labels(
    synthetic_inputs: dict[str, object],
) -> None:
    rows = build_synthetic_workfile(synthetic_inputs)

    assert [row["mapping_status"] for row in rows] == [
        "AMBIGUOUS",
        "NEAR_MATCH",
    ]
    assert all(row["review_category"] is None for row in rows)
    assert all(row["supported_intents"] == [] for row in rows)
    assert all(row["annotation_confidence"] is None for row in rows)
    assert all(row["adjudication_status"] == "UNREVIEWED" for row in rows)
    assert all(not workflow.FORBIDDEN_MODEL_FIELDS.intersection(row) for row in rows)


def test_protected_action_language_is_not_automatically_labeled(
    synthetic_inputs: dict[str, object],
) -> None:
    rows = build_synthetic_workfile(synthetic_inputs)
    protected = next(row for row in rows if "freeze my card" in row["narrative"])

    assert protected["review_category"] is None
    assert protected["supported_intents"] == []
    assert protected["adjudication_status"] == "UNREVIEWED"


@pytest.mark.parametrize(
    ("category", "intents"),
    [
        ("SINGLE_SUPPORTED_INTENT", ["account_balance"]),
        (
            "MULTI_SUPPORTED_INTENT",
            ["freeze_card", "transaction_details"],
        ),
        ("UNSUPPORTED", []),
        ("UNCLEAR_OR_INSUFFICIENT", []),
        ("NO_CURRENT_REQUEST", []),
    ],
)
def test_valid_category_and_supported_intent_cardinality(
    category: str,
    intents: list[str],
) -> None:
    row = completed_row(
        {
            **workflow.initialized_annotation_fields(),
            "narrative_sha256": "a" * 64,
        },
        category=category,
        intents=intents,
    )

    workflow.validate_annotation_fields(row)


@pytest.mark.parametrize(
    ("category", "intents", "message"),
    [
        ("SINGLE_SUPPORTED_INTENT", [], "exactly one"),
        ("SINGLE_SUPPORTED_INTENT", ["account_balance", "card_status"], "exactly one"),
        ("MULTI_SUPPORTED_INTENT", ["account_balance"], "at least two"),
        ("UNSUPPORTED", ["account_balance"], "empty supported_intents"),
    ],
)
def test_invalid_category_cardinality_is_rejected(
    category: str,
    intents: list[str],
    message: str,
) -> None:
    row = completed_row(
        workflow.initialized_annotation_fields(),
        category=category,
        intents=intents,
    )

    with pytest.raises(ValueError, match=message):
        workflow.validate_annotation_fields(row)


def test_supported_intents_must_be_allowed_unique_and_sorted() -> None:
    base = workflow.initialized_annotation_fields()

    with pytest.raises(ValueError, match="unique and sorted"):
        workflow.validate_annotation_fields(
            completed_row(
                base,
                category="MULTI_SUPPORTED_INTENT",
                intents=["transaction_details", "account_balance"],
            )
        )
    with pytest.raises(ValueError, match="unsupported supported_intents"):
        workflow.validate_annotation_fields(
            completed_row(
                base,
                category="SINGLE_SUPPORTED_INTENT",
                intents=["unsupported_or_uncertain"],
            )
        )


def test_annotation_confidence_must_use_the_frozen_values() -> None:
    row = completed_row(
        workflow.initialized_annotation_fields(),
        category="SINGLE_SUPPORTED_INTENT",
        intents=["account_balance"],
        confidence="CERTAIN",
    )

    with pytest.raises(ValueError, match="invalid annotation_confidence"):
        workflow.validate_annotation_fields(row)


def test_rerun_preserves_partial_annotations_and_deterministic_order(
    synthetic_inputs: dict[str, object],
) -> None:
    rows = build_synthetic_workfile(synthetic_inputs)
    rows[0] = completed_row(
        rows[0],
        category="MULTI_SUPPORTED_INTENT",
        intents=["freeze_card", "transaction_details"],
    )
    synthetic_inputs["output_path"].write_bytes(workflow.stable_jsonl_bytes(rows))

    refreshed = workflow.build_annotation_workfile(
        synthetic_inputs["source_rows"],
        synthetic_inputs["manifest"],
        output_path=synthetic_inputs["output_path"],
        reset_existing=False,
        expected_lane_counts=SYNTHETIC_COUNTS,
    )

    assert [row["narrative_sha256"] for row in refreshed] == [
        row["narrative_sha256"] for row in rows
    ]
    assert refreshed[0]["review_category"] == "MULTI_SUPPORTED_INTENT"
    assert refreshed[0]["supported_intents"] == [
        "freeze_card",
        "transaction_details",
    ]


def test_incomplete_reviews_block_export_unless_partial_is_explicit(
    synthetic_inputs: dict[str, object],
) -> None:
    rows = build_synthetic_workfile(synthetic_inputs)
    output_path = synthetic_inputs["tmp_path"] / "labels.json"

    with pytest.raises(ValueError, match="unresolved reviews"):
        workflow.export_semantic_labels(
            rows,
            synthetic_inputs["manifest"],
            workfile_sha256="workfile-sha256",
            review_pool_manifest_sha256="manifest-sha256",
            output_path=output_path,
            expected_lane_counts=SYNTHETIC_COUNTS,
        )

    artifact = workflow.export_semantic_labels(
        rows,
        synthetic_inputs["manifest"],
        workfile_sha256="workfile-sha256",
        review_pool_manifest_sha256="manifest-sha256",
        output_path=output_path,
        allow_partial=True,
        expected_lane_counts=SYNTHETIC_COUNTS,
    )

    assert artifact["partial"] is True
    assert artifact["summary"]["unresolved_count"] == 2


def test_completed_export_is_text_free_and_preserves_holdout_hashes(
    synthetic_inputs: dict[str, object],
) -> None:
    rows = build_synthetic_workfile(synthetic_inputs)
    rows = [
        completed_row(
            rows[0],
            category="MULTI_SUPPORTED_INTENT",
            intents=["freeze_card", "transaction_details"],
        ),
        completed_row(
            rows[1],
            category="SINGLE_SUPPORTED_INTENT",
            intents=["account_balance"],
        ),
    ]
    output_path = synthetic_inputs["tmp_path"] / "complete-labels.json"

    artifact = workflow.export_semantic_labels(
        rows,
        synthetic_inputs["manifest"],
        workfile_sha256="workfile-sha256",
        review_pool_manifest_sha256="manifest-sha256",
        output_path=output_path,
        expected_lane_counts=SYNTHETIC_COUNTS,
    )
    serialized = output_path.read_text(encoding="utf-8")

    assert artifact["partial"] is False
    assert artifact["summary"]["completed_count"] == 2
    assert {record["narrative_sha256"] for record in artifact["records"]} == {
        row["narrative_sha256"] for row in rows
    }
    assert all("narrative" not in record for record in artifact["records"])
    assert all(row["narrative"] not in serialized for row in rows)
    assert all("reviewer_id" not in record for record in artifact["records"])
    assert artifact["holdout_policy"] == {
        "final_external_evaluation_allowed_after_labels_are_frozen": True,
        "hyperparameter_selection_prohibited": True,
        "later_cfpb_training_data_must_use_disjoint_narratives": True,
        "model_selection_prohibited": True,
        "narrative_hashes_must_not_be_used_for_training": True,
    }
