from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_ROOT / "v2c5_model_selection_contract.json"
TAXONOMY_PATH = ML_ROOT / "v2c5_taxonomy_freeze.json"
TAXONOMY_MANIFEST_PATH = ML_ROOT / "v2c5_taxonomy_freeze.manifest.json"
DEVELOPMENT_PATH = ML_ROOT / "v2c5_expanded_development_dataset.json"
DEVELOPMENT_MANIFEST_PATH = (
    ML_ROOT / "v2c5_expanded_development_dataset.manifest.json"
)
FINAL_HOLDOUT_CONTRACT_PATH = ML_ROOT / "v2c5_final_holdout_contract.json"
FINAL_HOLDOUT_MANIFEST_PATH = ML_ROOT / "v2c5_final_holdout.manifest.json"

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
def taxonomy() -> dict[str, Any]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def development_manifest() -> dict[str, Any]:
    return json.loads(DEVELOPMENT_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final_holdout_manifest() -> dict[str, Any]:
    return json.loads(FINAL_HOLDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded_configurations(contract: dict[str, Any]) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    for family in contract["candidate_matrix"]["families"]:
        grid = family["hyperparameter_grid"]
        for c_value in grid["C"]:
            for class_weight in grid["class_weight"]:
                weight_id = "none" if class_weight is None else class_weight
                configurations.append(
                    {
                        "C": c_value,
                        "candidate_id": (
                            f"{family['family_id']}__C={c_value}__"
                            f"class_weight={weight_id}"
                        ),
                        "class_weight": class_weight,
                        "classifier_id": family["classifier_id"],
                        "family_id": family["family_id"],
                        "representation_id": family["representation_id"],
                    }
                )
    return configurations


def test_contract_schema_phase_and_frozen_status(contract: dict[str, Any]) -> None:
    assert contract["schema_version"] == "v2c5-model-selection-contract.v1"
    assert contract["contract_version"] == "v2c5-model-selection-contract.v1"
    assert contract["phase"] == "V2-C5 Step 22A"
    assert contract["contract_status"]["model_selection_contract_frozen"] is True


def test_required_step20_and_step21_hashes_are_pinned(
    contract: dict[str, Any],
) -> None:
    paths = {
        "expanded_development_dataset": DEVELOPMENT_PATH,
        "expanded_development_manifest": DEVELOPMENT_MANIFEST_PATH,
        "final_holdout_contract": FINAL_HOLDOUT_CONTRACT_PATH,
        "final_holdout_manifest": FINAL_HOLDOUT_MANIFEST_PATH,
        "taxonomy_freeze": TAXONOMY_PATH,
        "taxonomy_freeze_manifest": TAXONOMY_MANIFEST_PATH,
    }

    assert set(contract["source_artifacts"]) == set(paths)
    for name, path in paths.items():
        assert contract["source_artifacts"][name]["sha256"] == sha256_file(path)


def test_exact_frozen_16_intent_taxonomy(
    contract: dict[str, Any], taxonomy: dict[str, Any]
) -> None:
    assert contract["taxonomy"]["intent_count"] == 16
    assert contract["taxonomy"]["intent_label_order"] == EXPECTED_INTENTS
    assert contract["taxonomy"]["intent_label_order"] == taxonomy[
        "intent_label_order"
    ]


def test_exact_protected_write_set(
    contract: dict[str, Any], taxonomy: dict[str, Any]
) -> None:
    assert contract["taxonomy"]["protected_write_intents"] == (
        EXPECTED_PROTECTED_WRITES
    )
    assert contract["safety_gates"]["protected_write_intents"] == (
        EXPECTED_PROTECTED_WRITES
    )
    assert taxonomy["protected_write_intents"] == EXPECTED_PROTECTED_WRITES


def test_exact_five_candidate_families_and_27_configurations(
    contract: dict[str, Any],
) -> None:
    matrix = contract["candidate_matrix"]
    configurations = expanded_configurations(contract)

    assert matrix["candidate_family_count"] == 5
    assert matrix["candidate_count"] == 27
    assert len(matrix["families"]) == 5
    assert [family["configuration_count"] for family in matrix["families"]] == [
        6,
        6,
        6,
        6,
        3,
    ]
    assert len(configurations) == 27
    assert len({item["candidate_id"] for item in configurations}) == 27


def test_historical_v2c3_recipe_exists_exactly_once(
    contract: dict[str, Any],
) -> None:
    configurations = expanded_configurations(contract)
    matches = [
        item
        for item in configurations
        if item["representation_id"] == "BGE_SMALL"
        and item["classifier_id"] == "LINEAR_SVC"
        and item["C"] == 4.0
        and item["class_weight"] == "balanced"
    ]

    assert len(matches) == 1
    assert matches[0]["candidate_id"] == contract["candidate_matrix"][
        "historical_v2c3_recipe"
    ]["candidate_id"]


def test_exact_tfidf_and_feature_union_settings(contract: dict[str, Any]) -> None:
    representations = contract["representations"]

    assert representations["WORD_TFIDF"] == {
        "class": "TfidfVectorizer",
        "parameters": {
            "analyzer": "word",
            "lowercase": True,
            "min_df": 2,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
    }
    assert representations["CHAR_TFIDF"] == {
        "class": "TfidfVectorizer",
        "parameters": {
            "analyzer": "char_wb",
            "lowercase": True,
            "min_df": 2,
            "ngram_range": [3, 5],
            "sublinear_tf": True,
        },
    }
    assert representations["WORD_CHAR_TFIDF"] == {
        "class": "FeatureUnion",
        "exact_member_definitions_reused": True,
        "members": ["WORD_TFIDF", "CHAR_TFIDF"],
    }


def test_exact_bge_settings(contract: dict[str, Any]) -> None:
    assert contract["representations"]["BGE_SMALL"] == {
        "dimensions": 384,
        "embedding_method": "passage_embed",
        "fine_tuning": False,
        "implementation": "FastEmbed",
        "l2_normalized": True,
        "model_identifier": "BAAI/bge-small-en-v1.5",
    }


def test_deterministic_classifier_settings(contract: dict[str, Any]) -> None:
    classifiers = contract["classifiers"]

    assert classifiers["LINEAR_SVC"]["class"] == "LinearSVC"
    assert classifiers["LINEAR_SVC"]["fixed_parameters"] == {
        "dual": "auto",
        "fit_intercept": True,
        "intercept_scaling": 1.0,
        "loss": "squared_hinge",
        "max_iter": 2000,
        "multi_class": "ovr",
        "penalty": "l2",
        "random_state": 20260930,
        "tol": 0.0001,
        "verbose": 0,
    }
    assert classifiers["LOGISTIC_REGRESSION"]["class"] == "LogisticRegression"
    assert classifiers["LOGISTIC_REGRESSION"]["fixed_parameters"]["solver"] == (
        "lbfgs"
    )
    assert classifiers["LOGISTIC_REGRESSION"]["fixed_parameters"][
        "max_iter"
    ] == 2000


def test_exact_group_aware_cross_validation(contract: dict[str, Any]) -> None:
    cross_validation = contract["cross_validation"]

    assert cross_validation["method"] == "StratifiedGroupKFold"
    assert cross_validation["n_splits"] == 5
    assert cross_validation["shuffle"] is True
    assert cross_validation["random_state"] == 20260930
    assert cross_validation["group_field"] == "group_id"
    assert cross_validation["non_group_aware_fallback_allowed"] is False
    assert cross_validation["fail_if_group_aware_cv_unavailable"] is True
    assert cross_validation["fold_assignment"] == {
        "each_record_appears_in_exactly_one_validation_fold": True,
        "freeze_and_record_before_candidate_scoring": True,
        "group_must_not_cross_train_and_validation": True,
    }


def test_only_expanded_development_data_is_eligible(
    contract: dict[str, Any], development_manifest: dict[str, Any]
) -> None:
    dataset = contract["dataset_contract"]

    assert dataset["only_eligible_dataset"] == (
        "data/evals/v2/ml/v2c5_expanded_development_dataset.json"
    )
    assert dataset["example_count"] == 8198
    assert dataset["intent_count"] == 16
    assert dataset["classifier_input_fields"] == ["text"]
    assert dataset["eligible_data_roles"] == ["development"]
    assert dataset["group_field"] == "group_id"
    assert dataset["model_selection_probe_created"] is False
    assert development_manifest["counts"]["output_occurrences"] == 8198


def test_exact_three_mandatory_safety_gates(contract: dict[str, Any]) -> None:
    safety = contract["safety_gates"]
    gates = safety["gates"]

    assert safety["all_gates_mandatory_for_selection"] is True
    assert safety["evaluation_population"] == (
        "pooled_out_of_fold_development_predictions"
    )
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
    assert gates[0]["denominator"] == (
        "count(gold_intent not in protected_write_intents)"
    )
    assert gates[1]["numerator"] == (
        "count(gold_intent in protected_write_intents and "
        "predicted_intent == gold_intent)"
    )
    assert gates[2]["denominator"] == (
        "count(gold_intent == unsupported_or_uncertain)"
    )


def test_metrics_include_all_required_reports(contract: dict[str, Any]) -> None:
    metrics = contract["metrics"]

    assert metrics["primary_metric"]["name"] == "mean_fold_macro_f1"
    assert set(metrics["additional_metrics"]) == {
        "pooled_oof_macro_f1",
        "accuracy",
        "balanced_accuracy",
        "per_intent_precision",
        "per_intent_recall",
        "per_intent_f1",
        "confusion_matrix",
        "worst_fold_macro_f1",
        "new_seven_intent_macro_f1",
        "historical_nine_label_macro_f1",
        "local_fit_time",
        "local_prediction_latency",
        "resulting_artifact_size",
    }
    assert len(metrics["new_seven_intents"]) == 7
    assert len(metrics["historical_nine_labels"]) == 9


def test_exact_selection_order_and_numerical_tolerance(
    contract: dict[str, Any],
) -> None:
    selection = contract["selection_rule"]

    assert selection["candidate_eligibility"] == (
        "all_three_mandatory_safety_gates_pass"
    )
    assert selection["ordered_tie_breaks"] == [
        {"direction": "highest", "metric": "mean_fold_macro_f1"},
        {"direction": "highest", "metric": "worst_fold_macro_f1"},
        {
            "direction": "lowest",
            "metric": "protected_write_false_positive_rate",
        },
        {
            "direction": "highest",
            "metric": "unsupported_or_uncertain_recall",
        },
        {"direction": "lowest", "metric": "local_prediction_latency"},
        {"direction": "ascending_lexical", "metric": "candidate_id"},
    ]
    assert selection["numerical_tie_tolerance"]["absolute"] == 1e-12
    assert selection["numerical_tie_tolerance"]["relative"] == 0.0


def test_zero_gate_passing_candidates_prohibits_step23(
    contract: dict[str, Any],
) -> None:
    failure = contract["selection_rule"]["if_no_candidate_passes"]

    assert failure["selected_candidate"] is None
    assert failure["gates_weakened"] is False
    assert failure["final_holdout_accessed"] is False
    assert failure["development_failure_reported"] is True
    assert failure["step23_permitted"] is False


def test_threshold_tuning_is_prohibited(contract: dict[str, Any]) -> None:
    assert contract["safety_gates"]["threshold_tuning_permitted"] is False
    assert contract["safety_gates"]["threshold_tuning_performed"] is False
    assert "threshold_tuned_classifier" in contract["candidate_matrix"][
        "prohibited_families"
    ]


def test_no_additional_or_prohibited_datasets(contract: dict[str, Any]) -> None:
    prohibited = set(contract["dataset_contract"]["prohibited_data"])

    assert {
        "cfpb",
        "banking77_test",
        "clinc_test",
        "consumed_v2c3_challenge",
        "consumed_v2c3_external_lockbox",
        "v2c4_selection_probe",
        "v2c4_training_augmentation",
        "v2c4_postmortem",
        "v2c4_final_holdout",
        "v2c5_final_holdout",
        "newly_authored_model_selection_data",
    } == prohibited


def test_final_holdout_manifest_only_governance(
    contract: dict[str, Any], final_holdout_manifest: dict[str, Any]
) -> None:
    isolation = contract["holdout_isolation"]
    status = final_holdout_manifest["execution_status"]

    assert status["final_holdout_frozen"] is True
    assert status["final_holdout_evaluated"] is False
    assert status["step22_permitted"] is True
    assert isolation["allowed_manifest_checks"] == {
        "final_holdout_evaluated": False,
        "final_holdout_frozen": True,
        "step22_permitted": True,
    }
    assert isolation["final_holdout_dataset"]["accessed"] is False
    assert isolation["final_holdout_dataset"]["declared_sha256_from_manifest"] == (
        final_holdout_manifest["dataset"]["sha256"]
    )
    assert isolation["step22_may_open_holdout_dataset"] is False
    assert isolation["step22_may_inspect_holdout_records"] is False
    assert isolation["step22_may_perform_holdout_inference"] is False
    assert isolation["step23_owns_once_only_final_evaluation"] is True


def test_contract_step_executes_no_model_work(contract: dict[str, Any]) -> None:
    status = contract["contract_status"]

    assert status == {
        "candidate_evaluation_performed": False,
        "cross_validation_performed": False,
        "embeddings_generated": False,
        "model_selection_contract_frozen": True,
        "model_selection_performed": False,
        "model_training_performed": False,
        "next_required": "run_v2c5_development_model_selection",
        "runtime_behavior_changed": False,
        "sealed_final_holdout_accessed": False,
        "step22_execution_required": True,
        "step23_permitted": False,
    }
