from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "data/evals/v2/ml/v2c6_source_aware_model_selection_contract.json"
)

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
EXPECTED_PRIMARY_8 = [
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]
EXPECTED_PROTECTED = [
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
]
EXPECTED_SOURCE_FAMILIES = [
    "v2c6_sf1_definition_direct",
    "v2c6_sf2_scenario_narrative",
    "v2c6_sf3_boundary_conversational",
]
EXPECTED_HARD_NEGATIVE_PAIRS = [
    ["close_account", "unsupported_or_uncertain"],
    ["transfer_failed_or_declined", "unsupported_or_uncertain"],
    ["account_blocked", "unsupported_or_uncertain"],
    ["cancel_transfer", "unsupported_or_uncertain"],
    ["create_dispute", "unsupported_or_uncertain"],
    ["freeze_card", "unsupported_or_uncertain"],
    ["transfer_pending", "unsupported_or_uncertain"],
    ["cancel_transfer", "transfer_pending"],
    ["transfer_failed_or_declined", "transfer_pending"],
    ["account_blocked", "transfer_failed_or_declined"],
]
EXPECTED_CANDIDATE_IDS = [
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=balanced",
    "BGE_SMALL_LINEAR_SVC__C=1.0__class_weight=none",
    "BGE_SMALL_LINEAR_SVC__C=1.0__class_weight=balanced",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=balanced",
]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_identity_and_frozen_contract_status(contract: dict[str, Any]) -> None:
    assert contract["schema_version"] == (
        "v2c6-source-aware-model-selection-contract.v1"
    )
    assert contract["contract_version"] == contract["schema_version"]
    assert contract["phase"] == "V2-C6 Step 29G"
    assert contract["contract_status"]["contract_frozen"] is True


def test_exact_step29f_dataset_identity_and_lineage(
    contract: dict[str, Any],
) -> None:
    lineage = contract["frozen_input_lineage"]
    dataset = lineage["dataset"]

    assert dataset["path"] == (
        "data/evals/v2/ml/v2c6_remediated_development_dataset.json"
    )
    assert dataset["sha256"] == (
        "d7f78d7a76799f47bfdc3c1291505d1b964b9d143245d2d353931e8cb4f4a493"
    )
    assert dataset["record_count"] == 9008
    assert dataset["inherited_v2c5_record_count"] == 8198
    assert dataset["remediation_record_count"] == 810
    assert 8198 + 810 == 9008
    assert dataset["only_eligible_development_dataset"] is True


def test_step29f_freeze_identity_and_required_state(
    contract: dict[str, Any],
) -> None:
    freeze = contract["frozen_input_lineage"]["step29f_freeze"]

    assert freeze["path"].endswith(
        "v2c6_remediated_development_dataset.freeze.json"
    )
    assert freeze["sha256"] == (
        "7f71dda024e5a947c997977f09ed675fa1f924c38d30f332422aa2032ecbc544"
    )
    assert freeze["expected_freeze_status"] == "FROZEN"
    assert freeze["expected_next_required"] == (
        "v2c6_source_aware_model_selection_contract"
    )
    assert contract["frozen_input_lineage"][
        "step29f_freeze_must_validate_before_execution"
    ] is True


def test_exact_frozen_taxonomy_identity(contract: dict[str, Any]) -> None:
    taxonomy = contract["taxonomy"]

    assert taxonomy["intent_count"] == 16
    assert taxonomy["intent_label_order"] == EXPECTED_INTENTS
    assert taxonomy["version"] == "v2c5-taxonomy.v1"
    assert taxonomy["sha256"] == (
        "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
    )


def test_exact_intent_slices_and_protected_set(contract: dict[str, Any]) -> None:
    taxonomy = contract["taxonomy"]

    assert taxonomy["primary_remediation_8_intents"] == EXPECTED_PRIMARY_8
    assert "lost_or_stolen_phone" not in EXPECTED_PRIMARY_8
    assert "passcode_recovery" not in EXPECTED_PRIMARY_8
    assert len(taxonomy["historical_9_intents"]) == 9
    assert len(taxonomy["expanded_new_7_intents"]) == 7
    assert contract["safety_gates"]["protected_intents"] == EXPECTED_PROTECTED


def test_group_id_is_indivisible_without_label_purity_requirement(
    contract: dict[str, Any],
) -> None:
    protocol = contract["group_aware_cross_validation"]

    assert protocol["group_field"] == "group_id"
    assert protocol["group_ids_are_indivisible_split_units"] is True
    assert protocol["inherited_group_label_purity_required"] is False
    assert protocol["mixed_intent_inherited_group_count"] == 1
    assert protocol["mixed_intent_inherited_groups_remain_atomic"] is True
    assert protocol["rewrite_groups_for_stratification_permitted"] is False


def test_five_fold_group_aware_cv_is_exact(contract: dict[str, Any]) -> None:
    protocol = contract["group_aware_cross_validation"]

    assert protocol["method"] == "StratifiedGroupKFold"
    assert protocol["n_splits"] == 5
    assert protocol["shuffle"] is True
    assert protocol["random_state"] == 20260930
    assert protocol["non_group_aware_fallback_allowed"] is False
    assert protocol["complete_dataset_record_count"] == 9008
    assert protocol["expected_pooled_prediction_count"] == 9008
    assert protocol["fold_assignment"] == {
        "each_record_appears_in_exactly_one_validation_fold": True,
        "exact_membership_persisted_in_step29h": True,
        "fold_membership_hash_required": True,
        "generated_once_and_reused_for_all_candidates": True,
        "group_must_not_cross_train_and_validation": True,
    }


def test_exact_three_leave_one_family_out_rounds(
    contract: dict[str, Any],
) -> None:
    protocol = contract["source_family_holdout"]
    rounds = protocol["rounds"]

    assert protocol["source_families"] == EXPECTED_SOURCE_FAMILIES
    assert protocol["expected_round_count"] == 3
    assert len(rounds) == 3
    assert [row["held_out_source_family_id"] for row in rounds] == (
        EXPECTED_SOURCE_FAMILIES
    )
    assert {row["test_count"] for row in rounds} == {270}
    assert {row["training_count"] for row in rounds} == {8738}
    assert {row["training_inherited_count"] for row in rounds} == {8198}
    assert {row["training_other_remediation_family_count"] for row in rounds} == {
        540
    }
    assert protocol["per_family_composition"] == {
        "supported_primary_intent_count": 7,
        "supported_records_per_primary_intent": 30,
        "total_record_count": 270,
        "unsupported_or_uncertain_record_count": 60,
    }
    assert protocol["combined_out_of_family_prediction_count"] == 810
    assert protocol["held_out_family_completely_unseen_in_round_training"] is True


def test_learned_preprocessing_is_training_partition_only(
    contract: dict[str, Any],
) -> None:
    policy = contract["fitting_protocol"]

    assert policy[
        "full_dataset_supervised_fitting_before_evaluation_permitted"
    ] is False
    assert policy["tfidf_vocabulary_and_idf_fit_scope"] == "training_partition_only"
    assert policy["classifier_fit_scope"] == "training_partition_only"
    assert policy[
        "supervised_validation_labels_may_influence_features_or_training"
    ] is False
    embedding = policy["fixed_pretrained_embedding_policy"]
    assert embedding[
        "train_and_validation_or_test_partitions_transformed_separately"
    ] is True
    assert embedding["labels_may_influence_embeddings"] is False


def test_required_metric_sets_are_frozen(contract: dict[str, Any]) -> None:
    metrics = contract["metrics"]

    assert set(metrics["group_aware_cv_required"]) >= {
        "accuracy",
        "balanced_accuracy",
        "macro_f1_16",
        "per_intent_precision",
        "per_intent_recall",
        "per_intent_f1",
        "confusion_matrix",
        "historical_9_macro_f1",
        "expanded_new_7_macro_f1",
        "safety_metrics",
    }
    assert set(metrics["source_family_per_round_required"]) >= {
        "primary_8_macro_f1",
        "supported_to_unsupported_rate",
        "unsupported_to_supported_rate",
        "hard_negative_boundary_diagnostics",
    }
    assert set(metrics["source_family_pooled_required"]) >= {
        "pooled_primary_8_macro_f1",
        "mean_family_primary_8_macro_f1",
        "worst_family_primary_8_macro_f1",
        "pooled_confusion_matrix",
        "pooled_safety_metrics",
    }


def test_protected_and_unsupported_metric_definitions_are_exact(
    contract: dict[str, Any],
) -> None:
    gates = {row["metric"]: row for row in contract["safety_gates"]["gates"]}

    assert gates["protected_recall"]["numerator"] == (
        "count(gold_intent in protected_intents and predicted_intent in "
        "protected_intents)"
    )
    assert gates["protected_recall"]["denominator"] == (
        "count(gold_intent in protected_intents)"
    )
    assert gates["protected_false_positive_rate"]["numerator"] == (
        "count(gold_intent not in protected_intents and predicted_intent in "
        "protected_intents)"
    )
    assert gates["protected_false_positive_rate"]["denominator"] == (
        "count(gold_intent not in protected_intents)"
    )
    assert gates["unsupported_recall"]["numerator"] == (
        "count(gold_intent == unsupported_or_uncertain and predicted_intent == "
        "unsupported_or_uncertain)"
    )


def test_safety_thresholds_and_application_scopes_are_exact(
    contract: dict[str, Any],
) -> None:
    safety = contract["safety_gates"]
    gates = {row["metric"]: row for row in safety["gates"]}

    assert gates["protected_recall"] == {
        **gates["protected_recall"],
        "comparison": "greater_than_or_equal",
        "threshold": 0.8,
    }
    assert gates["protected_false_positive_rate"] == {
        **gates["protected_false_positive_rate"],
        "comparison": "less_than_or_equal",
        "threshold": 0.01,
    }
    assert gates["unsupported_recall"] == {
        **gates["unsupported_recall"],
        "comparison": "greater_than_or_equal",
        "threshold": 0.8,
    }
    assert safety["application_scopes"] == [
        "pooled_group_aware_cv_predictions",
        "pooled_source_family_holdout_predictions",
        "each_individual_source_family_holdout_round",
    ]
    assert safety["gates_mandatory_for_candidate_eligibility"] is True
    assert safety["thresholds_may_be_weakened_after_results"] is False


def test_hard_negative_pairs_and_diagnostics_are_exact(
    contract: dict[str, Any],
) -> None:
    diagnostics = contract["hard_negative_diagnostics"]

    assert diagnostics["required_pairs"] == EXPECTED_HARD_NEGATIVE_PAIRS
    assert len(diagnostics["required_pairs"]) == 10
    assert diagnostics["report_per_source_family"] is True
    assert diagnostics["report_pooled_across_all_source_family_rounds"] is True
    assert diagnostics["aggregate_metric"] == "hard_negative_accuracy"
    assert diagnostics["mandatory_safety_gate"] is False
    assert diagnostics["pairs_created_or_relabelled_in_step29g_or_step29h"] is False


def test_unsupported_boundary_diagnostics_are_required(
    contract: dict[str, Any],
) -> None:
    diagnostics = contract["unsupported_boundary_diagnostics"]

    assert diagnostics["report_per_source_family_and_pooled"] is True
    assert "supported_to_unsupported_rate" in diagnostics
    assert "unsupported_to_supported_rate" in diagnostics
    assert diagnostics["gates_created_post_hoc"] is False


def test_bge_small_linear_svc_control_exists_exactly_once(
    contract: dict[str, Any],
) -> None:
    candidates = contract["candidate_search_space"]["candidates"]
    controls = [row for row in candidates if row["is_v2c5_selected_recipe_control"]]

    assert len(controls) == 1
    assert controls[0] == {
        "candidate_id": "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
        "classifier_id": "LINEAR_SVC",
        "hyperparameters": {"C": 4.0, "class_weight": None},
        "hypothesis": "V2-C5 selected-recipe control after source-diverse remediation.",
        "is_v2c5_selected_recipe_control": True,
        "representation_id": "BGE_SMALL",
    }
    assert contract["representations"]["BGE_SMALL"] == {
        "dimensions": 384,
        "embedding_method": "passage_embed",
        "fine_tuning": False,
        "implementation": "FastEmbed",
        "l2_normalized": True,
        "model_identifier": "BAAI/bge-small-en-v1.5",
    }


def test_candidate_list_is_exact_bounded_and_existing_repository_only(
    contract: dict[str, Any],
) -> None:
    search = contract["candidate_search_space"]
    candidates = search["candidates"]

    assert search["candidate_count"] == 6
    assert search["maximum_candidate_count"] == 6
    assert [row["candidate_id"] for row in candidates] == EXPECTED_CANDIDATE_IDS
    assert len({row["candidate_id"] for row in candidates}) == 6
    assert {row["representation_id"] for row in candidates} == {
        "BGE_SMALL",
        "WORD_CHAR_TFIDF",
    }
    assert {row["classifier_id"] for row in candidates} == {"LINEAR_SVC"}


def test_prohibited_classifier_families_are_excluded(
    contract: dict[str, Any],
) -> None:
    search = contract["candidate_search_space"]
    serialized = json.dumps(search).lower()

    assert search["prohibited_candidate_families"] == [
        "llm_classifier",
        "remote_paid_inference",
        "deep_model_fine_tuning",
        "new_embedding_provider",
        "threshold_tuned_classifier",
        "post_hoc_rule_override",
    ]
    assert all("llm" not in row["candidate_id"].lower() for row in search["candidates"])
    assert all(
        "remote" not in row["candidate_id"].lower()
        for row in search["candidates"]
    )
    assert "fine_tuning\": true" not in serialized


def test_threshold_tuning_is_fully_prohibited(contract: dict[str, Any]) -> None:
    policy = contract["threshold_policy"]

    assert policy["native_multiclass_prediction_rule_required"] is True
    assert all(
        value is False
        for key, value in policy.items()
        if key != "native_multiclass_prediction_rule_required"
    )
    assert contract["source_family_holdout"][
        "threshold_tuning_from_rounds_permitted"
    ] is False


def test_candidate_eligibility_requirements_are_complete(
    contract: dict[str, Any],
) -> None:
    eligibility = contract["candidate_eligibility"]

    assert eligibility["eligible_status"] == "ELIGIBLE"
    assert eligibility["ineligible_status"] == "INELIGIBLE"
    assert eligibility["record_explicit_ineligibility_reasons"] is True
    assert set(eligibility["requirements"]) == {
        "all_required_evaluation_runs_completed",
        "no_group_leakage",
        "no_source_family_leakage",
        "all_expected_predictions_present_exactly_once_for_each_applicable_evaluation",
        "all_mandatory_safety_gates_pass",
        "all_required_metrics_finite_and_valid",
        "artifact_hashes_and_dataset_sha_match_step29f_freeze",
        "no_prohibited_holdout_access",
        "no_post_hoc_threshold_tuning",
    }


def test_selection_rule_is_exact_and_begins_with_worst_family(
    contract: dict[str, Any],
) -> None:
    rule = contract["selection_rule"]
    metrics = [row["metric"] for row in rule["ordered_lexicographic_criteria"]]

    assert metrics == [
        "worst_family_primary_8_macro_f1",
        "mean_family_primary_8_macro_f1",
        "pooled_source_family_primary_8_macro_f1",
        "pooled_group_cv_macro_f1_16",
        "pooled_source_family_supported_to_unsupported_rate",
        "pooled_source_family_protected_false_positive_rate",
        "pooled_source_family_protected_recall",
        "pooled_source_family_unsupported_recall",
        "candidate_complexity_rank",
        "candidate_id",
    ]
    assert rule["ordered_lexicographic_criteria"][0]["direction"] == "maximize"
    assert rule["single_weighted_aggregate_score_used"] is False
    assert rule["selection_population"] == "eligible_candidates_only"
    assert rule["numerical_tie_tolerance"] == {
        "absolute": 1e-12,
        "relative": 0.0,
        "rule": (
            "Treat numeric criteria as tied when absolute_difference is less "
            "than or equal to the absolute tolerance."
        ),
    }
    assert rule["winner_postconditions"] == {
        "fresh_untouched_v2c6_final_holdout_still_required": True,
        "once_only_final_evaluation_required_under_steps_29j_through_29m": True,
        "step29i_fit_required": True,
    }


def test_complexity_order_is_an_exact_candidate_permutation(
    contract: dict[str, Any],
) -> None:
    rule = contract["selection_rule"]
    candidate_ids = {
        row["candidate_id"]
        for row in contract["candidate_search_space"]["candidates"]
    }

    assert len(rule["complexity_order_low_to_high"]) == 6
    assert set(rule["complexity_order_low_to_high"]) == candidate_ids


def test_no_acceptable_candidate_behavior_is_frozen(
    contract: dict[str, Any],
) -> None:
    failure = contract["selection_rule"]["if_no_candidate_is_eligible"]

    assert failure == {
        "gates_weakened": False,
        "result": "NO_ACCEPTABLE_CANDIDATE",
        "selected_candidate": None,
        "winner_forced": False,
    }


def test_step29h_outputs_and_split_auditability_are_defined(
    contract: dict[str, Any],
) -> None:
    artifacts = contract["step29h_artifact_contract"]

    assert artifacts["tracked_outputs"] == [
        "data/evals/v2/ml/v2c6_source_aware_model_selection_results.json",
        "data/evals/v2/ml/v2c6_source_aware_model_selection_results.manifest.json",
    ]
    assert "exact_group_cv_fold_membership_and_hashes" in artifacts[
        "required_result_contents"
    ]
    assert "exact_source_family_round_membership_and_hashes" in artifacts[
        "required_result_contents"
    ]
    assert "selected_candidate_or_no_acceptable_candidate" in artifacts[
        "required_result_contents"
    ]
    assert artifacts["raw_utterance_text_in_result_summary_permitted"] is False


def test_final_holdouts_are_prohibited(contract: dict[str, Any]) -> None:
    isolation = contract["holdout_isolation"]

    assert isolation["consumed_v2c5_final_holdout_access_prohibited"] is True
    assert isolation["fresh_v2c6_final_holdout_access_prohibited"] is True
    assert isolation["fresh_v2c6_final_holdout_authoring_permitted_in_step29g"] is False
    assert isolation["final_holdout_data_in_outputs_prohibited"] is True
    assert "v2c5_final_holdout.json" not in json.dumps(contract)


def test_step29g_performs_no_model_work_or_runtime_change(
    contract: dict[str, Any],
) -> None:
    status = contract["contract_status"]
    governance = contract["governance"]

    assert status["cross_validation_performed"] is False
    assert status["source_family_evaluation_performed"] is False
    assert status["embeddings_generated"] is False
    assert status["model_fitting_performed"] is False
    assert status["model_selection_performed"] is False
    assert status["threshold_tuning_performed"] is False
    assert status["runtime_behavior_changed"] is False
    assert governance["contract_only"] is True
    assert governance["step29g_performs_model_work"] is False
    assert governance["candidate_selected"] is False
    assert governance["final_model_acceptance_claimed"] is False


def test_architecture_decision_and_limitations_are_explicit(
    contract: dict[str, Any],
) -> None:
    decision = contract["architecture_decision"]
    rationale = contract["evaluation_rationale"]

    assert decision["chosen"] == (
        "group_aware_cross_validation_plus_leave_one_source_family_out_"
        "validation_with_a_small_bounded_candidate_set"
    )
    assert rationale["ordinary_within_development_cv_alone_is_sufficient"] is False
    assert rationale["protocols_are_complementary"] is True
    assert rationale["stronger_causal_claim_made"] is False
    assert contract["known_limitations"]


def test_exact_next_required(contract: dict[str, Any]) -> None:
    assert contract["contract_status"]["next_required"] == (
        "v2c6_source_aware_model_selection_execution"
    )
