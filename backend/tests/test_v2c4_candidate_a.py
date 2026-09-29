from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import evaluate_v2c4_candidate_a as evaluator
from scripts import train_v2c4_candidate_a as training

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SCRIPT_PATH = ROOT / "scripts/evaluate_v2c4_candidate_a.py"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return training.load_config()


def test_config_freezes_exact_candidate_a_recipe(
    config: dict[str, Any],
) -> None:
    candidate = config["candidate_a"]
    assert candidate["candidate_id"] == "v2c4_candidate_a_targeted_data"
    assert candidate["architecture"] == "direct_9_way_classifier"
    assert candidate["representation"] == {
        "family": "frozen_local_sentence_embedding",
        "model_identifier": "BAAI/bge-small-en-v1.5",
        "fastembed_method": "passage_embed",
        "dimensions": 384,
        "fine_tuning": False,
        "external_api": False,
    }
    assert candidate["classifier"]["class"] == "LinearSVC"
    assert candidate["classifier"]["parameters"] == (
        training.EXPECTED_CLASSIFIER_PARAMETERS
    )
    assert candidate["classifier"]["parameters"]["C"] == 4.0
    assert candidate["classifier"]["parameters"]["class_weight"] == "balanced"
    assert tuple(config["intent_order"]) == training.EXPECTED_INTENTS
    assert config["risk_by_intent"] == training.EXPECTED_RISK_BY_INTENT
    assert config["frozen_inputs"]["v2c3_baseline_artifact"]["sha256"] == (
        training.EXPECTED_BASELINE_SHA256
    )


def test_config_freezes_safety_gates_and_prohibits_search(
    config: dict[str, Any],
) -> None:
    assert config["safety_gates"] == {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "all_gates_required": True,
    }
    candidate = config["candidate_a"]
    assert candidate["hyperparameter_search"] is False
    assert candidate["threshold_tuning"] is False
    assert candidate["probability_calibration"] is False
    assert config["candidate_b_trigger"]["candidate_b_construction_allowed"] is False


def synthetic_record(
    identifier: str,
    intent: str,
    data_role: str,
    index: int,
) -> dict[str, Any]:
    return {
        "example_id": identifier,
        "intent": intent,
        "risk": training.EXPECTED_RISK_BY_INTENT[intent],
        "data_role": data_role,
        "group_id": f"group:{identifier}",
        "source_id": "synthetic-development-fixture",
        "source_split": "train",
        "original_split": "train",
        "normalized_text_sha256": f"{index:064x}",
        "text": f"Synthetic fixture {index}",
    }


def test_training_corpus_is_development_plus_augmentation_only(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    local_config = copy.deepcopy(config)
    development = [
        synthetic_record(f"dev:{index}", intent, "development", index)
        for index, intent in enumerate(training.EXPECTED_INTENTS, start=1)
    ]
    augmentation = [
        synthetic_record(
            "augmentation:1",
            "freeze_card",
            "v2c4_targeted_training_augmentation",
            20,
        )
    ]
    probe = [
        {
            **synthetic_record(
                "probe:1", "account_balance", "v2c4_model_selection_probe", 30
            ),
            "training_eligible": False,
        }
    ]
    local_config["training_corpus"].update(
        {
            "v2c3_development_expected_count": 9,
            "augmentation_expected_count": 1,
            "combined_expected_count": 10,
        }
    )
    local_config["selection_probe"]["expected_count"] = 1
    payloads = {
        "development.json": {"example_count": 9, "examples": development},
        "augmentation.json": {"example_count": 1, "examples": augmentation},
        "probe.json": {"example_count": 1, "examples": probe},
    }
    monkeypatch.setattr(
        training,
        "read_json",
        lambda path: payloads[path.name],
    )
    datasets = training.load_and_validate_datasets(
        local_config,
        {
            "v2c3_development": Path("development.json"),
            "v2c4_training_augmentation": Path("augmentation.json"),
            "v2c4_selection_probe": Path("probe.json"),
        },
    )

    assert datasets["training"] == [*development, *augmentation]
    assert {row["data_role"] for row in datasets["training"]} == {
        "development",
        "v2c4_targeted_training_augmentation",
    }
    assert all(row["example_id"] != "probe:1" for row in datasets["training"])


@pytest.mark.parametrize(
    "forbidden_role",
    [
        "v2c4_model_selection_probe",
        "consumed_v2c3_safety_challenge",
        "consumed_v2c3_external_lockbox",
        "consumed_v2c3_external_regression",
        "v2c4_final_safety_holdout",
    ],
)
def test_training_rejects_forbidden_data_roles(
    forbidden_role: str, config: dict[str, Any]
) -> None:
    row = {
        "data_role": forbidden_role,
        "source_id": "synthetic-fixture",
        "source_split": "train",
        "original_split": "train",
    }
    with pytest.raises(ValueError, match="forbidden training data role"):
        training.validate_training_role(row, config)


@pytest.mark.parametrize(
    ("source_id", "source_domain", "source_split"),
    [
        ("cfpb-complaint", "consumer-finance", "train"),
        ("banking77", "banking", "test"),
        ("clinc150", "assistant", "test"),
        ("v2c3_safety_challenge", "sentinelvoice", "challenge"),
        ("v2c3_external_lockbox", "banking", "locked"),
    ],
)
def test_training_rejects_forbidden_sources(
    source_id: str,
    source_domain: str,
    source_split: str,
    config: dict[str, Any],
) -> None:
    row = {
        "data_role": "development",
        "source_id": source_id,
        "source_domain": source_domain,
        "source_split": source_split,
        "original_split": source_split,
    }
    with pytest.raises(ValueError, match="forbidden"):
        training.validate_training_role(row, config)


def test_supporting_cv_is_group_aware_and_assigns_every_record_once(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[dict[str, Any]] = []
    embedding_rows: list[list[float]] = []
    for intent_index, intent in enumerate(training.EXPECTED_INTENTS):
        for group_index in range(10):
            records.append(
                {
                    "intent": intent,
                    "group_id": f"{intent}:group:{group_index}",
                }
            )
            embedding_rows.append([float(intent_index)])
    embeddings = np.asarray(embedding_rows, dtype=np.float32)

    class SyntheticClassifier:
        def __init__(self, **parameters: Any) -> None:
            assert parameters == training.EXPECTED_CLASSIFIER_PARAMETERS

        def fit(
            self, fold_embeddings: np.ndarray, labels: np.ndarray
        ) -> SyntheticClassifier:
            assert len(fold_embeddings) == len(labels)
            return self

        def predict(self, fold_embeddings: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    training.EXPECTED_INTENTS[int(row[0])]
                    for row in fold_embeddings
                ]
            )

    monkeypatch.setattr(training, "LinearSVC", SyntheticClassifier)
    results = training.supporting_cv(config, records, embeddings)

    assert results["method"] == "StratifiedGroupKFold"
    assert results["n_splits"] == 5
    assert results["random_state"] == 20260928
    assert results["every_record_received_one_validation_prediction"] is True
    assert results["group_leakage_detected"] is False
    assert sum(row["validation_count"] for row in results["fold_results"]) == 90
    assert all(row["group_overlap_count"] == 0 for row in results["fold_results"])


def test_both_models_receive_the_same_embedding_object() -> None:
    embeddings = np.zeros((3, 384), dtype=np.float32)

    class RecordingClassifier:
        def __init__(self, label: str) -> None:
            self.label = label
            self.received: np.ndarray | None = None

        def predict(self, values: np.ndarray) -> np.ndarray:
            self.received = values
            return np.asarray([self.label] * len(values))

    baseline = RecordingClassifier("account_balance")
    candidate = RecordingClassifier("unsupported_or_uncertain")
    evaluator.predict_both_on_shared_embeddings(baseline, candidate, embeddings)

    assert baseline.received is embeddings
    assert candidate.received is embeddings


def test_safety_metric_definitions() -> None:
    gold = [
        "freeze_card",
        "create_dispute",
        "unsupported_or_uncertain",
        "unsupported_or_uncertain",
        "account_balance",
    ]
    predicted = [
        "create_dispute",
        "account_balance",
        "unsupported_or_uncertain",
        "freeze_card",
        "account_balance",
    ]
    metrics = training.calculate_metrics(
        gold, predicted, training.EXPECTED_INTENTS
    )

    assert metrics["protected_total"] == 2
    assert metrics["protected_true_positive"] == 1
    assert metrics["protected_misses"] == 1
    assert metrics["protected_write_recall"] == pytest.approx(0.5)
    assert metrics["protected_exact_intent_accuracy"] == pytest.approx(0.0)
    assert metrics["non_protected_total"] == 3
    assert metrics["protected_false_positive"] == 1
    assert metrics["protected_write_false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["unsupported_total"] == 2
    assert metrics["unsupported_correctly_rejected"] == 1
    assert metrics["unsupported_false_supported"] == 1
    assert metrics["unsupported_or_uncertain_recall"] == pytest.approx(0.5)
    assert metrics["false_supported_rate"] == pytest.approx(0.5)


def test_macro_f1_regression_boundary_and_candidate_b_trigger(
    config: dict[str, Any],
) -> None:
    boundary = evaluator.macro_f1_regression_check(0.80, 0.79, config)
    regression = evaluator.macro_f1_regression_check(0.80, 0.789, config)
    passing_metrics = {
        "protected_write_false_positive_rate": 0.01,
        "protected_write_recall": 0.80,
        "unsupported_or_uncertain_recall": 0.80,
    }
    gates = training.apply_safety_gates(passing_metrics, config["safety_gates"])

    assert boundary["macro_f1_delta"] == pytest.approx(-0.01)
    assert boundary["material_regression"] is False
    assert regression["macro_f1_delta"] == pytest.approx(-0.011)
    assert regression["material_regression"] is True
    assert gates["all_required"] is True
    assert gates["all_passed"] is True
    assert evaluator.candidate_b_trigger(gates, boundary) == {
        "candidate_b_triggered": False,
        "reasons": [],
        "candidate_a_selected_by_step_9_rule": True,
        "candidate_b_built": False,
        "report_only": True,
    }
    assert evaluator.candidate_b_trigger(gates, regression)[
        "candidate_b_triggered"
    ] is True


def test_any_failed_safety_gate_triggers_candidate_b(
    config: dict[str, Any],
) -> None:
    failing_metrics = {
        "protected_write_false_positive_rate": 0.02,
        "protected_write_recall": 0.90,
        "unsupported_or_uncertain_recall": 0.90,
    }
    gates = training.apply_safety_gates(failing_metrics, config["safety_gates"])
    no_regression = evaluator.macro_f1_regression_check(0.80, 0.80, config)
    trigger = evaluator.candidate_b_trigger(gates, no_regression)

    assert gates["all_passed"] is False
    assert trigger["candidate_b_triggered"] is True
    assert trigger["candidate_b_built"] is False
    assert trigger["report_only"] is True


def test_frozen_baseline_artifact_sha_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "baseline.joblib"
    artifact_path.write_bytes(b"synthetic baseline")
    monkeypatch.setattr(training, "REPOSITORY_ROOT", tmp_path)
    local_config = {
        "frozen_inputs": {
            "v2c3_baseline_artifact": {
                "path": "baseline.joblib",
                "sha256": "0" * 64,
            }
        }
    }

    with pytest.raises(ValueError, match="v2c3_baseline_artifact SHA-256 mismatch"):
        training.validate_frozen_inputs(local_config)


def test_candidate_artifact_sha_is_enforced_from_training_report(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_config = copy.deepcopy(config)
    local_config["outputs"]["training_report"] = "training-report.json"
    local_config["outputs"]["candidate_artifact"] = "candidate.joblib"
    candidate_path = tmp_path / "candidate.joblib"
    candidate_path.write_bytes(b"synthetic candidate")
    training_report = {
        "schema_version": "v2c4-candidate-a-training-report.v1",
        "candidate_config_sha256": training.sha256_file(training.CONFIG_PATH),
        "artifact_metadata": {
            "sha256": "0" * 64,
            "size_bytes": candidate_path.stat().st_size,
        },
    }
    (tmp_path / "training-report.json").write_text(
        json.dumps(training_report), encoding="utf-8"
    )
    monkeypatch.setattr(training, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(
        ValueError,
        match="Candidate A artifact SHA-256 differs from training report",
    ):
        evaluator.load_models(
            local_config,
            {"v2c3_baseline_artifact": tmp_path / "unused-baseline.joblib"},
        )


def test_sealed_holdout_role_and_paths_are_forbidden(
    config: dict[str, Any],
) -> None:
    sealed_records = [
        {"data_role": evaluator.FINAL_HOLDOUT_ROLE}
        for _ in range(config["selection_probe"]["expected_count"])
    ]
    with pytest.raises(ValueError, match="final holdout is forbidden"):
        evaluator.validate_evaluation_records(sealed_records, config)

    source = EVALUATION_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "v2c4_safety_holdout.json" not in source
    assert "v2c4_safety_holdout_seed.json" not in source


def test_evaluation_report_structure_and_development_boundary() -> None:
    report = dict.fromkeys(evaluator.EXPECTED_REPORT_KEYS)
    report.update(
        {
            "analysis_role": "v2c4_development_model_selection",
            "final_acceptance_evidence": False,
            "final_holdout_accessed": False,
            "final_improvement_claimed": False,
        }
    )
    assert evaluator.report_structure_is_deterministic(report) is True

    report["final_acceptance_evidence"] = True
    assert evaluator.report_structure_is_deterministic(report) is False
