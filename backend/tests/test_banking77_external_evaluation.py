import json
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from scripts import run_banking77_external_evaluation as external_evaluation


@pytest.fixture(scope="module")
def report() -> dict:
    return external_evaluation.build_report()


def test_only_frozen_scored_lanes_are_evaluated(report: dict) -> None:
    source = report["scored_source"]

    assert source["exact_match_examples"] == 440
    assert source["unsupported_examples"] == 1_720
    assert source["review_pool_scored"] is False
    assert source["train_csv_used"] is False
    assert source["source_split"] == "test"
    assert source["dataset_path"].endswith("banking77_external_eval.json")


def test_predictions_use_verified_frozen_artifact(report: dict) -> None:
    selection = json.loads(
        external_evaluation.MODEL_SELECTION_PATH.read_text(encoding="utf-8")
    )
    frozen = report["frozen_classifier"]

    assert frozen["artifact_sha256"] == selection["model_artifact_sha256"]
    assert frozen["components"]["selected_intent_model"] == "LinearSVC"
    assert frozen["components"]["logistic_probability_intent_model"] == (
        "LogisticRegression"
    )
    assert frozen["components"]["independent_risk_model"] == "LinearSVC"
    assert frozen["components"]["advisory_abstention_rule"] == selection[
        "abstention"
    ]["selected"]
    assert frozen["modified_or_retrained"] is False


def test_full_nine_intent_prediction_space_is_preserved(report: dict) -> None:
    expected_labels = set(external_evaluation.INTENT_LABELS)
    exact_distribution = report["primary_svm"]["exact_match_lane"][
        "prediction_distribution_full_nine_intents"
    ]
    unsupported_distribution = report["primary_svm"]["unsupported_lane"][
        "prediction_distribution_full_nine_intents"
    ]

    assert len(expected_labels) == 9
    assert set(exact_distribution) == expected_labels
    assert set(unsupported_distribution) == expected_labels
    assert set(report["frozen_classifier"]["full_intent_label_order"]) == (
        expected_labels
    )


def test_evaluation_never_calls_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("external evaluation must not fit a model")

    monkeypatch.setattr(Pipeline, "fit", fail_fit)
    report = external_evaluation.build_report()

    assert report["training_or_tuning_performed"] is False


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


def test_external_metric_boundaries_and_review_labels(report: dict) -> None:
    boundary = report["external_metrics_boundary"]

    assert boundary[
        "directly_comparable_to_v2c1_nine_intent_locked_test_macro_f1"
    ] is False
    assert boundary["exact_match_macro_f1_expected_class_count"] == 2
    assert report["scored_source"]["review_pool_scored"] is False
    assert "banking77_review_pool" not in Path(
        external_evaluation.__file__
    ).read_text(encoding="utf-8")
