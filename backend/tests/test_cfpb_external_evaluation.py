import json
from pathlib import Path
from typing import Any

import pytest
from sklearn.pipeline import Pipeline

from scripts import run_cfpb_external_evaluation as external_evaluation


@pytest.fixture(scope="module")
def metadata() -> dict[str, Any]:
    return external_evaluation.validate_metadata_and_hashes()


@pytest.fixture(scope="module")
def inputs(metadata: dict[str, Any]) -> dict[str, Any]:
    return external_evaluation.load_and_validate_cfpb_inputs(metadata)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return external_evaluation.build_report()


def test_exact_holdout_linkage_selects_1800_from_3800(
    inputs: dict[str, Any],
) -> None:
    selected = inputs["all"]

    assert inputs["source_pool_count"] == 3_800
    assert len(selected) == 1_800
    assert len(inputs["primary"]) == 1_776
    assert len(inputs["multi"]) == 24
    assert len({row["narrative_sha256"] for row in selected}) == 1_800
    assert all(row["text"] for row in selected)


def test_frozen_label_mapping_is_preserved(inputs: dict[str, Any]) -> None:
    for row in inputs["all"]:
        category = row["final_review_category"]
        target = row["primary_single_label_target"]
        if category == "SINGLE_SUPPORTED_INTENT":
            assert target == row["final_supported_intents"][0]
        elif category == "MULTI_SUPPORTED_INTENT":
            assert target is None
            assert row["primary_single_label_evaluable"] is False
        else:
            assert category in {
                "UNSUPPORTED",
                "UNCLEAR_OR_INSUFFICIENT",
                "NO_CURRENT_REQUEST",
            }
            assert target == "unsupported_or_uncertain"


def test_multi_intent_set_membership_rule() -> None:
    records = [
        {"final_supported_intents": ["account_balance", "card_status"]},
        {"final_supported_intents": ["create_dispute", "transaction_details"]},
        {"final_supported_intents": ["card_status", "freeze_card"]},
    ]

    metrics = external_evaluation.multi_intent_metrics(
        records,
        ["card_status", "account_balance", "unsupported_or_uncertain"],
    )

    assert metrics["example_count"] == 3
    assert metrics["prediction_is_supported_intent_hit_count"] == 1
    assert metrics["prediction_is_supported_intent_accuracy"] == 1 / 3


def test_protected_write_accounting_uses_final_semantic_membership() -> None:
    records = [
        {"final_supported_intents": ["freeze_card"]},
        {"final_supported_intents": ["create_dispute", "transaction_details"]},
        {"final_supported_intents": ["account_balance"]},
        {"final_supported_intents": ["create_dispute"]},
    ]
    predictions = ["freeze_card", "create_dispute", "freeze_card", "card_status"]

    metrics = external_evaluation.protected_write_metrics(records, predictions)

    assert metrics["by_intent"]["freeze_card"] == {
        "false_discovery_rate": 0.5,
        "false_positive_count": 1,
        "false_positive_rate": 1 / 3,
        "gold_negative_count": 3,
        "gold_supported_count": 1,
        "predicted_positive_count": 2,
        "true_positive_count": 1,
    }
    assert metrics["by_intent"]["create_dispute"]["gold_supported_count"] == 2
    assert metrics["by_intent"]["create_dispute"]["true_positive_count"] == 1


def test_predictions_use_verified_frozen_artifact(report: dict[str, Any]) -> None:
    frozen = report["frozen_classifier"]

    assert frozen["artifact_sha256"] == external_evaluation.EXPECTED_ARTIFACT_SHA256
    assert frozen["components"]["selected_intent_model"] == "LinearSVC C=0.5"
    assert frozen["components"]["logistic_probability_intent_model"] == (
        "LogisticRegression C=2.0"
    )
    assert frozen["components"]["independent_risk_model"] == "LinearSVC"
    assert frozen["modified_or_retrained"] is False
    assert frozen["runtime_authority"] is False


def test_artifact_and_source_hash_validation(
    tmp_path: Path,
    report: dict[str, Any],
) -> None:
    assert report["integrity"]["final_labels_sha256"] == (
        external_evaluation.EXPECTED_LABEL_SHA256
    )
    assert report["integrity"]["source_pool_sha256"] == (
        external_evaluation.sha256_path(external_evaluation.SOURCE_POOL_PATH)
    )
    assert report["frozen_classifier"]["artifact_sha256"] == (
        external_evaluation.sha256_path(external_evaluation.ARTIFACT_PATH)
    )

    changed = tmp_path / "changed.jsonl"
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        external_evaluation.verify_sha256(
            changed,
            external_evaluation.EXPECTED_LABEL_SHA256,
            "test frozen labels",
        )


def test_evaluation_never_calls_fit_or_fit_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("external evaluation must not fit a model")

    monkeypatch.setattr(Pipeline, "fit", fail_fit)
    monkeypatch.setattr(Pipeline, "fit_transform", fail_fit)
    report = external_evaluation.build_report()
    source = Path(external_evaluation.__file__).read_text(encoding="utf-8")

    assert ".fit(" not in source
    assert ".fit_transform(" not in source
    assert report["training_or_tuning_performed"] is False


def test_frozen_abstention_rule_is_not_selected_on_cfpb(
    report: dict[str, Any],
) -> None:
    abstention = report["frozen_advisory_abstention_analysis"]
    rule = abstention["frozen_rule"]

    assert rule["confidence_threshold"] == 0.0
    assert rule["margin_threshold"] == 0.2
    assert rule["require_intent_risk_agreement"] is False
    assert abstention["threshold_selection_source"] == "V2-C1 validation only"
    assert "separate from the primary SVM" in abstention["analysis_role"]


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value)) if value else set()
    return set()


def test_final_report_contains_no_narrative_or_per_record_predictions(
    report: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    serialized = json.dumps(report, sort_keys=True)
    forbidden_keys = {
        "annotation_note",
        "final_annotation_note",
        "narrative",
        "text",
        "unresolved_reason",
    }

    assert not (_all_keys(report) & forbidden_keys)
    assert inputs["all"][0]["text"] not in serialized
    assert report["consumer_narrative_text_emitted"] is False
    assert report["report_content_boundary"]["contains_per_record_predictions"] is (
        False
    )


def test_report_generation_is_byte_deterministic() -> None:
    assert external_evaluation.build_report_bytes() == (
        external_evaluation.build_report_bytes()
    )


def test_report_scoring_boundaries_and_counts(report: dict[str, Any]) -> None:
    contract = report["label_contract"]
    source = report["scored_source"]

    assert contract["total_holdout_count"] == 1_800
    assert contract["primary_single_label_evaluable_count"] == 1_776
    assert contract["excluded_multi_intent_count"] == 24
    assert contract["primary_evaluation_coverage"] == 1_776 / 1_800
    assert contract["category_counts"] == (
        external_evaluation.EXPECTED_CATEGORY_COUNTS
    )
    assert contract["protected_write_supported_counts"] == (
        external_evaluation.EXPECTED_PROTECTED_WRITE_COUNTS
    )
    assert source["source_pool_examples"] == 3_800
    assert source["selected_holdout_examples"] == 1_800
    assert source["extra_review_pool_records_not_scored"] == 2_000
    assert report["primary_svm"]["example_count"] == 1_776
    assert report["multi_intent_secondary"]["example_count"] == 24


def test_report_declares_external_advisory_no_training_boundary(
    report: dict[str, Any],
) -> None:
    boundary = report["external_metrics_boundary"]

    assert report["advisory_only"] is True
    assert report["training_or_tuning_performed"] is False
    assert boundary["external_results_used_to_select_v2c1"] is False
    assert boundary["label_and_scoring_contract_frozen_before_inference"] is True
    assert boundary[
        "directly_comparable_to_v2c1_balanced_nine_intent_test"
    ] is False
