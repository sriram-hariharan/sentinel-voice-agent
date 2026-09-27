import json
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from scripts import run_clinc_finance_external_evaluation as external_evaluation


@pytest.fixture(scope="module")
def report() -> dict:
    return external_evaluation.build_report()


def test_only_frozen_scored_lanes_are_evaluated(report: dict) -> None:
    source = report["scored_source"]

    assert source["exact_match_examples"] == 90
    assert source["unsupported_examples"] == 630
    assert source["review_pool_examples_excluded"] == 180
    assert source["review_pool_scored"] is False
    assert source["oos_examples_scored"] is False
    assert source["source_split"] == "test"
    assert source["dataset_path"].endswith("clinc_finance_external_eval.json")


def test_predictions_use_verified_frozen_artifact(report: dict) -> None:
    selection = json.loads(
        external_evaluation.MODEL_SELECTION_PATH.read_text(encoding="utf-8")
    )
    frozen = report["frozen_classifier"]

    assert frozen["artifact_sha256"] == external_evaluation.EXPECTED_ARTIFACT_SHA256
    assert frozen["artifact_sha256"] == selection["model_artifact_sha256"]
    assert frozen["components"]["selected_intent_model"] == "LinearSVC"
    assert frozen["components"]["logistic_probability_intent_model"] == (
        "LogisticRegression"
    )
    assert frozen["components"]["advisory_abstention_rule"] == selection[
        "abstention"
    ]["selected"]
    assert frozen["modified_or_retrained"] is False


def test_exact_lane_contract_and_full_prediction_space(report: dict) -> None:
    exact = report["primary_svm"]["exact_match_lane"]
    expected_labels = set(external_evaluation.EXACT_EXPECTED_LABELS)
    full_labels = set(external_evaluation.INTENT_LABELS)

    assert exact["example_count"] == 90
    assert set(exact["per_expected_class"]) == expected_labels
    assert set(exact["macro_f1_scope"]) == expected_labels
    assert set(exact["prediction_distribution_full_nine_intents"]) == full_labels
    assert set(report["frozen_classifier"]["full_intent_label_order"]) == full_labels
    assert report["external_metrics_boundary"][
        "exact_match_macro_f1_expected_class_count"
    ] == 3


def test_unsupported_lane_metrics_are_internally_consistent(report: dict) -> None:
    lane = report["primary_svm"]["unsupported_lane"]
    distribution = lane["prediction_distribution_full_nine_intents"]
    unsupported_count = distribution["unsupported_or_uncertain"]["count"]

    assert lane["example_count"] == 630
    assert sum(item["count"] for item in distribution.values()) == 630
    assert lane["unsupported_or_uncertain_recall"] == unsupported_count / 630
    assert lane["false_supported_intent_rate"] == 1 - (unsupported_count / 630)
    assert set(distribution) == set(external_evaluation.INTENT_LABELS)


def test_protected_write_false_positive_metrics() -> None:
    metrics = external_evaluation.protected_write_false_positives(
        ["freeze_card", "create_dispute", "account_balance", "freeze_card"]
    )

    assert metrics == {
        "by_intent": {
            "freeze_card": {"count": 2, "rate": 0.5},
            "create_dispute": {"count": 1, "rate": 0.25},
        },
        "combined_count": 3,
        "combined_rate": 0.75,
    }


def test_frozen_abstention_is_separate_and_unchanged(report: dict) -> None:
    abstention = report["frozen_advisory_abstention_analysis"]
    rule = abstention["frozen_rule"]

    assert rule["margin_threshold"] == 0.2
    assert rule["confidence_threshold"] == 0.0
    assert rule["require_intent_risk_agreement"] is False
    assert abstention["threshold_selection_source"] == "V2-C1 validation only"
    assert abstention["exact_match_lane"]["example_count"] == 90
    assert abstention["unsupported_lane"]["example_count"] == 630
    assert "primary SVM" in abstention["analysis_role"]


def test_evaluation_never_calls_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("external evaluation must not fit a model")

    monkeypatch.setattr(Pipeline, "fit", fail_fit)
    report = external_evaluation.build_report()

    assert report["training_or_tuning_performed"] is False


def test_report_generation_is_byte_deterministic() -> None:
    assert external_evaluation.build_report_bytes() == (
        external_evaluation.build_report_bytes()
    )


def test_source_and_artifact_hash_validation(
    tmp_path: Path, report: dict
) -> None:
    assert report["integrity"]["processed_evaluation_sha256"] == (
        external_evaluation.sha256_path(external_evaluation.INPUT_PATH)
    )
    assert report["integrity"]["raw_clinc_data_sha256"] == (
        external_evaluation.sha256_path(external_evaluation.RAW_DATA_PATH)
    )
    assert report["frozen_classifier"]["artifact_sha256"] == (
        external_evaluation.sha256_path(external_evaluation.ARTIFACT_PATH)
    )

    changed = tmp_path / "changed.json"
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        external_evaluation.verify_sha256(
            changed,
            report["integrity"]["processed_evaluation_sha256"],
            "test source",
        )


def test_review_pool_file_is_not_loaded_or_scored(report: dict) -> None:
    runner_source = Path(external_evaluation.__file__).read_text(encoding="utf-8")

    assert report["scored_source"]["review_pool_scored"] is False
    assert "clinc_finance_review_pool.json" not in runner_source


def test_svm_margin_statistics_cover_each_lane(report: dict) -> None:
    exact = report["primary_svm"]["exact_match_lane"]["decision_margin"]
    unsupported = report["primary_svm"]["unsupported_lane"]["decision_margin"]

    assert exact["example_count"] == 90
    assert unsupported["example_count"] == 630
    assert exact["top_1_minus_top_2_margin"]["minimum"] >= 0
    assert unsupported["top_1_minus_top_2_margin"]["minimum"] >= 0
    assert "not probabilities" in exact["interpretation"]
