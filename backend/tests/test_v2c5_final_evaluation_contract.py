from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_ROOT / "v2c5_final_evaluation_contract.json"
SELECTED_MODEL_MANIFEST_PATH = ML_ROOT / "v2c5_selected_model.manifest.json"
FINAL_HOLDOUT_MANIFEST_PATH = ML_ROOT / "v2c5_final_holdout.manifest.json"
PROHIBITED_FINAL_HOLDOUT_PATH = ML_ROOT / "v2c5_final_holdout.json"

EXPECTED_INTENTS = [
    "account_balance",
    "account_blocked",
    "cancel_transfer",
    "card_status",
    "close_account",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "lost_or_stolen_phone",
    "passcode_recovery",
    "recent_transactions",
    "transaction_details",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]
EXPECTED_PROTECTED_WRITES = [
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def selected_model_manifest() -> dict[str, Any]:
    return json.loads(SELECTED_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final_holdout_manifest() -> dict[str, Any]:
    return json.loads(FINAL_HOLDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_schema_phase_and_frozen_status(
    contract: dict[str, Any],
) -> None:
    assert contract["schema_version"] == "v2c5-final-evaluation-contract.v1"
    assert contract["contract_version"] == "v2c5-final-evaluation-contract.v1"
    assert contract["phase"] == "V2-C5 Step 23A"
    assert contract["contract_status"] == {
        "contract_frozen": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "model_retraining_performed": False,
        "next_required": "v2c5_final_evaluation_runner",
        "runtime_behavior_changed": False,
        "threshold_tuning_performed": False,
    }


def test_exact_selected_candidate_and_manifest_binding(
    contract: dict[str, Any], selected_model_manifest: dict[str, Any]
) -> None:
    model = contract["model_under_test"]
    sources = contract["source_artifacts"]
    assert sources["selected_model_manifest"] == {
        "path": "data/evals/v2/ml/v2c5_selected_model.manifest.json",
        "sha256": sha256_file(SELECTED_MODEL_MANIFEST_PATH),
    }
    assert model["selected_candidate_id"] == selected_model_manifest[
        "selected_candidate_id"
    ]
    assert model["selected_candidate_id"] == (
        "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none"
    )
    assert model["selection_role"] == (
        "derived_and_validated_from_frozen_selected_model_manifest_not_a_new_selection"
    )


def test_selected_classifier_artifact_hash_is_bound_without_loading_artifact(
    contract: dict[str, Any], selected_model_manifest: dict[str, Any]
) -> None:
    binding = contract["source_artifacts"]["selected_classifier_artifact"]
    assert binding == selected_model_manifest["local_classifier_artifact"]
    assert binding["path"] == (
        "data/evals/v2/ml/local/v2c5_selected_classifier.joblib"
    )
    assert binding["sha256"] == (
        "c2c5ef7db1b0a0cfa8328da300f39669808f36ef6386c1f440c3e49f7c4fe6f7"
    )
    assert binding["trusted_local"] is True
    assert binding["runtime_eligible"] is False


def test_exact_representation_and_classifier_configuration(
    contract: dict[str, Any], selected_model_manifest: dict[str, Any]
) -> None:
    model = contract["model_under_test"]
    assert model["representation"] == selected_model_manifest[
        "representation_config"
    ]
    assert model["representation"] == {
        "dimensions": 384,
        "embedding_method": "passage_embed",
        "fine_tuning": False,
        "implementation": "FastEmbed",
        "l2_normalized": True,
        "model_identifier": "BAAI/bge-small-en-v1.5",
    }
    assert model["classifier"] == selected_model_manifest["classifier_config"]
    assert model["classifier"]["class"] == "LinearSVC"
    assert model["classifier"]["parameters"]["C"] == 4.0
    assert model["classifier"]["parameters"]["class_weight"] is None


def test_exact_frozen_16_label_taxonomy(
    contract: dict[str, Any],
    selected_model_manifest: dict[str, Any],
    final_holdout_manifest: dict[str, Any],
) -> None:
    labels = contract["model_under_test"]["class_labels"]
    assert labels == EXPECTED_INTENTS
    assert labels == selected_model_manifest["class_labels"]
    assert labels == final_holdout_manifest["taxonomy_intent_labels"]
    assert contract["metrics"]["confusion_matrix"]["label_order"] == labels
    assert contract["metrics"]["confusion_matrix"]["dimensions"] == [16, 16]


def test_required_source_hashes_are_derived_from_frozen_manifests(
    contract: dict[str, Any], selected_model_manifest: dict[str, Any]
) -> None:
    sources = contract["source_artifacts"]
    assert sources["model_selection_results"] == (
        selected_model_manifest["source_model_selection_results"]
    )
    assert sources["development_dataset"] == selected_model_manifest[
        "development_dataset"
    ]
    assert sources["taxonomy"] == selected_model_manifest["taxonomy"]


def test_holdout_binding_uses_manifest_only(
    contract: dict[str, Any], final_holdout_manifest: dict[str, Any]
) -> None:
    isolation = contract["holdout_isolation"]
    sources = contract["source_artifacts"]
    assert sources["final_holdout_manifest"] == {
        "path": "data/evals/v2/ml/v2c5_final_holdout.manifest.json",
        "sha256": sha256_file(FINAL_HOLDOUT_MANIFEST_PATH),
    }
    assert sources["final_holdout_contract"] == final_holdout_manifest[
        "source_artifacts"
    ]["contract"]
    assert sources["final_holdout_dataset_declaration"]["path"] == (
        final_holdout_manifest["dataset"]["path"]
    )
    assert sources["final_holdout_dataset_declaration"]["sha256"] == (
        final_holdout_manifest["dataset"]["sha256"]
    )
    assert sources["final_holdout_dataset_declaration"][
        "opened_or_hashed_by_step23a"
    ] is False
    assert isolation["allowed_step23a_holdout_input"] == (
        "data/evals/v2/ml/v2c5_final_holdout.manifest.json"
    )
    assert isolation["prohibited_step23a_path"] == str(
        PROHIBITED_FINAL_HOLDOUT_PATH.relative_to(ROOT)
    )
    assert isolation["final_holdout_dataset_opened_or_hashed_by_step23a"] is False


def test_exact_balanced_holdout_contract(
    contract: dict[str, Any], final_holdout_manifest: dict[str, Any]
) -> None:
    holdout = contract["holdout"]
    counts = final_holdout_manifest["counts"]
    assert holdout["total_example_count"] == counts["total_count"] == 640
    assert holdout["intent_count"] == 16
    assert holdout["examples_per_intent"] == 40
    assert holdout["intent_counts"] == counts["intent_counts"]
    assert set(holdout["intent_counts"].values()) == {40}
    assert holdout["protected_write_example_count"] == 160
    assert holdout["non_protected_example_count"] == 480
    assert holdout["authorship_method"] == "independently_authored_synthetic"
    assert holdout["independent_from_development_examples"] is True
    assert holdout["normalized_overlap_counts"] == final_holdout_manifest[
        "integrity_and_overlap_checks"
    ]["normalized_overlap_counts"]
    assert set(holdout["normalized_overlap_counts"].values()) == {0}
    assert holdout["evaluated_before_step23"] is False
    assert final_holdout_manifest["execution_status"][
        "final_holdout_evaluated"
    ] is False


def test_exact_final_metrics_without_new_aggregate_gate(
    contract: dict[str, Any],
) -> None:
    metrics = contract["metrics"]
    assert metrics["aggregate_metrics"] == [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    assert metrics["per_intent_metrics"] == [
        "precision",
        "recall",
        "f1",
        "support",
    ]
    assert set(metrics["subset_macro_f1"]) == {
        "historical_nine_label_macro_f1",
        "new_seven_intent_macro_f1",
    }
    assert len(metrics["subset_macro_f1"]["historical_nine_label_macro_f1"]) == 9
    assert len(metrics["subset_macro_f1"]["new_seven_intent_macro_f1"]) == 7
    assert metrics["aggregate_acceptance_threshold"] is None
    assert metrics["macro_f1_role"] == (
        "reported_final_quality_metric_not_an_additional_gate"
    )


def test_exact_three_step22_safety_definitions_and_thresholds(
    contract: dict[str, Any],
) -> None:
    safety = contract["safety_gates"]
    gates = safety["gates"]
    assert safety["protected_write_intents"] == EXPECTED_PROTECTED_WRITES
    assert safety["thresholds_match_step22"] is True
    assert safety["all_gates_mandatory_for_final_acceptance"] is True
    assert [gate["metric"] for gate in gates] == [
        "protected_write_false_positive_rate",
        "exact_protected_write_recall",
        "unsupported_or_uncertain_recall",
    ]
    assert [(gate["comparison"], gate["threshold"]) for gate in gates] == [
        ("less_than_or_equal", 0.01),
        ("greater_than_or_equal", 0.8),
        ("greater_than_or_equal", 0.8),
    ]
    assert [gate["expected_denominator"] for gate in gates] == [480, 160, 40]
    assert gates[0]["denominator"] == (
        "count(gold_intent not in protected_write_intents)"
    )
    assert gates[0]["numerator"] == (
        "count(gold_intent not in protected_write_intents and "
        "predicted_intent in protected_write_intents)"
    )
    assert gates[1]["denominator"] == (
        "count(gold_intent in protected_write_intents)"
    )
    assert gates[1]["numerator"] == (
        "count(gold_intent in protected_write_intents and "
        "predicted_intent == gold_intent)"
    )
    assert gates[2]["denominator"] == (
        "count(gold_intent == unsupported_or_uncertain)"
    )
    assert gates[2]["numerator"] == (
        "count(gold_intent == unsupported_or_uncertain and "
        "predicted_intent == unsupported_or_uncertain)"
    )


def test_once_only_evaluation_prohibits_learning_from_final_results(
    contract: dict[str, Any],
) -> None:
    once_only = contract["once_only_evaluation"]
    assert once_only["single_final_inference_and_evaluation"] is True
    assert once_only["evaluation_attempt_count"] == 1
    assert once_only["threshold_tuning_from_holdout_results"] is False
    assert once_only["model_retraining_from_holdout_results"] is False
    assert once_only["label_or_taxonomy_change_from_holdout_results"] is False
    assert once_only["candidate_reselection_from_holdout_results"] is False
    assert once_only["second_clean_evaluation_after_failure_inspection"] is False
    assert once_only[
        "holdout_examples_may_be_added_to_development_or_training"
    ] is False
    assert once_only["failed_gate_policy"] == {
        "honest_failure_reporting_required": True,
        "mandatory_gates_may_be_weakened": False,
        "rerun_may_be_presented_as_same_final_experiment": False,
        "step24_may_not_proceed_as_if_passed": True,
    }


def test_durable_started_before_access_and_rerun_refusal(
    contract: dict[str, Any],
) -> None:
    protocol = contract["evaluation_state_protocol"]
    assert protocol["initial_state"] == "not_started"
    assert set(protocol["states"]) == {
        "not_started",
        "started",
        "completed",
        "failed_after_access",
    }
    assert protocol["required_pre_access_transition"] == {
        "from": "not_started",
        "must_be_durable_before": "opening_or_reading_the_final_holdout_dataset",
        "to": "started",
    }
    assert protocol["marker_durability_requirement"] == (
        "atomically_write_and_fsync_before_holdout_open"
    )
    assert protocol["automatic_evaluation_allowed_by_state"] == {
        "completed": False,
        "failed_after_access": False,
        "not_started": True,
        "started": False,
    }
    assert protocol["rerun_refusal_states"] == [
        "started",
        "failed_after_access",
        "completed",
    ]
    assert "compromised_or_secondary" in protocol[
        "unexpected_started_state_recovery"
    ]


def test_decision_states_are_distinct_and_runtime_remains_unchanged(
    contract: dict[str, Any],
) -> None:
    decision = contract["decision_semantics"]
    assert set(decision) == {
        "final_evaluation_completed",
        "mandatory_safety_gates_passed",
        "final_model_acceptance_claimed",
        "runtime_eligibility",
        "step24_controlled_integration",
    }
    assert decision["final_evaluation_completed"][
        "does_not_imply_safety_gates_passed"
    ] is True
    assert decision["mandatory_safety_gates_passed"][
        "does_not_imply_runtime_eligibility"
    ] is True
    assert decision["final_model_acceptance_claimed"][
        "value_before_evaluation"
    ] is False
    assert decision["runtime_eligibility"]["value_before_evaluation"] is False
    assert decision["runtime_eligibility"]["value_after_gate_pass"] is False
    assert decision["runtime_eligibility"][
        "runtime_behavior_changed_by_step23"
    ] is False


def test_planned_outputs_are_text_free_and_not_generated_by_step23a(
    contract: dict[str, Any],
) -> None:
    outputs = contract["planned_outputs"]
    assert outputs == {
        "generated_by_step23a": False,
        "prediction_persistence": "aggregate_only",
        "raw_holdout_text_permitted": False,
        "results": "data/evals/v2/ml/v2c5_final_evaluation_results.json",
        "results_manifest": (
            "data/evals/v2/ml/v2c5_final_evaluation_results.manifest.json"
        ),
        "state": "data/evals/v2/ml/v2c5_final_evaluation_state.json",
    }


def test_next_required_phase_is_final_evaluation_runner(
    contract: dict[str, Any],
) -> None:
    assert contract["contract_status"]["next_required"] == (
        "v2c5_final_evaluation_runner"
    )
