from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / (
    "data/evals/v2/ml/"
    "v2c6_protected_intent_gate_remediation_design_contract.json"
)
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"

PROTECTED_INTENTS = [
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
]
PRIMARY_EIGHT = [
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]
TRAINING_FAMILIES = [
    "v2c6_r3_train_sf1_minimal_explicitness",
    "v2c6_r3_train_sf2_contextual_boundary",
    "v2c6_r3_train_sf3_conversational_ambiguity",
]
EVALUATION_FAMILIES = [
    "v2c6_r3_eval_sf1_independent_casework",
    "v2c6_r3_eval_sf2_independent_naturalistic",
]
CANDIDATE_IDS = [
    "HYBRID_CONTROL_R3",
    "HYBRID_PROTECTED_VERIFIER_R3",
]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(contract: dict[str, Any], name: str) -> dict[str, Any]:
    specification = contract["source_artifacts"][name]
    return json.loads((ROOT / specification["path"]).read_text(encoding="utf-8"))


def test_contract_identity_and_frozen_status(contract: dict[str, Any]) -> None:
    schema = "v2c6-protected-intent-gate-remediation-design-contract.v1"

    assert contract["schema_version"] == schema
    assert contract["contract_version"] == schema
    assert contract["phase"] == (
        "V2-C6 protected-intent safety-gate remediation design"
    )
    assert contract["status"] == "FROZEN"
    assert contract["contract_status"]["contract_frozen"] is True
    assert contract["contract_status"]["remediation_design_frozen"] is True
    assert contract["contract_status"]["remediation_executed"] is False


def test_expected_repository_baseline_is_recorded(
    contract: dict[str, Any],
) -> None:
    assert contract["repository_baseline"] == {
        "branch": "v2/ml-routing-evaluation",
        "head": "9923061",
        "head_subject": "Record V2-C6 targeted remediation failure analysis",
    }


def test_exact_completed_analysis_result_lineage(
    contract: dict[str, Any],
) -> None:
    source = contract["source_artifacts"]["failure_analysis_result"]
    result = load_source(contract, "failure_analysis_result")

    assert source["sha256"] == (
        "047151b85e84cf4a8704ca95af445b40e066a93ebb1f487125cff9a08ca1760d"
    )
    assert sha256_file(ROOT / source["path"]) == source["sha256"]
    assert result["schema_version"] == source["schema_version"]
    assert result["execution_status"] == "COMPLETED"
    assert result["governance"]["failure_analysis_executed"] is True
    assert result["governance"]["candidate_selected"] is False
    assert result["governance"]["remediation_selected"] is False
    assert result["governance"]["step29i_authorized"] is False
    assert result["next_required"] is None


def test_exact_completed_analysis_manifest_lineage(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    source = sources["failure_analysis_manifest"]
    manifest = load_source(contract, "failure_analysis_manifest")

    assert source["sha256"] == (
        "e514b744690733f223332c36a34c7a12bdf988d4442770e49dc49ddc108d5d6a"
    )
    assert sha256_file(ROOT / source["path"]) == source["sha256"]
    assert manifest["schema_version"] == source["schema_version"]
    assert manifest["analysis_result"]["sha256"] == sources[
        "failure_analysis_result"
    ]["sha256"]
    assert manifest["failure_analysis_executed"] is True
    assert manifest["final_holdout_accessed"] is False
    assert manifest["step29i_authorized"] is False
    assert manifest["next_required"] is None


def test_measured_basis_is_descriptive_not_causal(
    contract: dict[str, Any],
) -> None:
    evidence = contract["measured_basis"]

    assert evidence["causal_root_cause_proven"] is False
    assert evidence[
        "protected_false_positive_weakness_observed_in_grouped_cv_and_fresh"
    ] is True
    assert evidence[
        "unsupported_recall_degraded_on_independent_fresh_evidence"
    ] is True
    assert evidence[
        "unsupported_to_protected_is_majority_fresh_protected_false_positive_origin"
    ] is True
    assert evidence["protected_recall_comparatively_strong"] is True
    assert evidence["generic_hierarchical_candidate_eliminated_weakness"] is False
    assert evidence["exact_hierarchical_stage_1_attribution_available"] is False


def test_exact_two_candidates(contract: dict[str, Any]) -> None:
    search = contract["candidate_search_space"]

    assert search["candidate_count"] == 2
    assert search["exact_candidate_ids"] == CANDIDATE_IDS
    assert [row["candidate_id"] for row in search["candidates"]] == CANDIDATE_IDS
    assert search["additional_candidates_permitted"] is False


def test_candidates_share_identical_primary_training_data(
    contract: dict[str, Any],
) -> None:
    candidates = contract["candidate_search_space"]["candidates"]

    assert all(
        row["primary_router"]
        == "HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none"
        for row in candidates
    )
    assert all(row["primary_training_dataset_record_count"] == 10088 for row in candidates)
    assert candidates[0]["protected_intent_verifiers"] is False
    assert candidates[1]["protected_intent_verifiers"] is True


def test_exact_four_protected_intents(contract: dict[str, Any]) -> None:
    assert contract["architecture_decision"]["protected_intents"] == (
        PROTECTED_INTENTS
    )
    assert contract["verifier_architecture"][
        "independent_binary_verifier_count"
    ] == 4


def test_protected_gate_routing_rule_is_exact(contract: dict[str, Any]) -> None:
    routing = contract["architecture_decision"]["routing_rule"]

    assert routing == {
        "non_protected_primary_prediction": "return_primary_prediction_unchanged",
        "protected_primary_prediction": (
            "invoke_verifier_dedicated_to_predicted_protected_intent"
        ),
        "verifier_accept": "preserve_primary_protected_prediction",
        "verifier_reject": "return_unsupported_or_uncertain",
    }


def test_verifier_is_routing_not_authorization(contract: dict[str, Any]) -> None:
    semantics = contract["architecture_decision"]["authorization_semantics"]

    assert semantics["verifier_is_authorization"] is False
    assert semantics["verifier_only_changes_routing"] is True
    assert set(semantics["application_controls"]) == {
        "authentication",
        "ownership",
        "explicit_confirmation",
        "idempotency",
        "protected_tool_execution",
    }


def test_exact_verifier_architecture(contract: dict[str, Any]) -> None:
    verifier = contract["verifier_architecture"]

    assert verifier["representation"] == "WORD_CHAR_TFIDF"
    assert verifier["representation_convention"] == (
        "existing_frozen_word_char_tfidf_convention"
    )
    assert verifier["classifier"] == "LinearSVC"
    assert verifier["C"] == 1.0
    assert verifier["class_weight"] is None
    assert verifier["positive_class"] == "dedicated_protected_intent"
    assert verifier["negative_class"] == (
        "targeted_unsupported_or_uncertain_hard_negatives_for_protected_intent"
    )
    assert verifier["training_source"] == "r3_targeted_training_addendum_only"
    assert verifier[
        "historical_development_records_permitted_in_verifier_fitting"
    ] is False


def test_no_threshold_calibration_weight_or_extra_model(
    contract: dict[str, Any],
) -> None:
    verifier = contract["verifier_architecture"]

    assert verifier["threshold_tuning"] is False
    assert verifier["calibration"] is False
    assert verifier["confidence_override"] is False
    assert verifier["class_weight"] is None
    assert verifier["additional_embeddings"] is False
    assert verifier["llm_verifier"] is False
    assert "generic_supported_vs_unsupported_gate" in contract[
        "prohibited_operations"
    ]


def test_exact_480_record_training_addendum(contract: dict[str, Any]) -> None:
    training = contract["targeted_training_addendum"]

    assert training["record_count"] == 480
    assert training["family_count"] == 3
    assert training["source_family_ids"] == TRAINING_FAMILIES
    assert training["consumed_examples_may_be_paraphrase_sources"] is False
    assert training["boundary_metadata_field"] == "target_protected_intent"
    assert training["positive_definition"] == (
        "explicit_current_request_to_execute_the_dedicated_protected_action"
    )
    assert sum(row["record_count"] for row in training["family_specifications"]) == 480


def test_exact_per_family_training_distribution(
    contract: dict[str, Any],
) -> None:
    families = contract["targeted_training_addendum"]["family_specifications"]
    expected = {intent: 20 for intent in PROTECTED_INTENTS}

    for family in families:
        assert family["record_count"] == 160
        assert family["positive_record_count"] == 80
        assert family["unsupported_record_count"] == 80
        assert family["positive_intent_counts"] == expected
        assert family["unsupported_target_counts"] == expected
        assert sum(family["positive_intent_counts"].values()) == 80
        assert sum(family["unsupported_target_counts"].values()) == 80


def test_exact_60_60_verifier_training_composition(
    contract: dict[str, Any],
) -> None:
    verifiers = contract["verifier_architecture"]["verifiers"]

    assert [row["positive_intent"] for row in verifiers] == PROTECTED_INTENTS
    assert all(row["positive_record_count"] == 60 for row in verifiers)
    assert all(row["negative_record_count"] == 60 for row in verifiers)
    assert all(row["total_record_count"] == 120 for row in verifiers)


def test_expanded_development_count_is_exact(contract: dict[str, Any]) -> None:
    dataset = contract["expanded_development_dataset"]

    assert dataset["existing_frozen_record_count"] == 9608
    assert dataset["new_r3_targeted_record_count"] == 480
    assert dataset["expected_expanded_record_count"] == 10088
    assert dataset["both_candidates_use_identical_primary_training_dataset"] is True


def test_duplicate_and_overlap_controls_are_fail_closed(
    contract: dict[str, Any],
) -> None:
    controls = contract["duplicate_and_overlap_controls"]

    assert controls["failure_behavior"] == "fail_closed"
    assert controls["required_zero_counts"] == [
        "exact_duplicate",
        "normalized_duplicate",
        "cross_intent_normalized_collision",
        "historical_development_normalized_overlap",
        "consumed_evaluation_normalized_overlap",
        "final_holdout_normalized_overlap",
    ]
    assert controls["raw_final_holdout_direct_access_permitted"] is False


def test_exact_new_fresh_evaluation_families(
    contract: dict[str, Any],
) -> None:
    evaluation = contract["fresh_evaluation_specification"]

    assert evaluation["family_count"] == 2
    assert evaluation["source_family_ids"] == EVALUATION_FAMILIES
    assert evaluation["per_family_record_count"] == 320
    assert evaluation["record_count"] == 640


def test_exact_primary_eight_distribution_per_fresh_family(
    contract: dict[str, Any],
) -> None:
    families = contract["fresh_evaluation_specification"]["family_specifications"]

    for family in families:
        assert family["record_count"] == 320
        assert family["protected_record_count"] == 160
        assert family["non_protected_record_count"] == 160
        assert family["intent_counts"] == {intent: 40 for intent in PRIMARY_EIGHT}


def test_exact_unsupported_boundary_allocation_per_fresh_family(
    contract: dict[str, Any],
) -> None:
    families = contract["fresh_evaluation_specification"]["family_specifications"]
    expected = {intent: 10 for intent in PROTECTED_INTENTS}

    for family in families:
        assert family["unsupported_boundary_target_counts"] == expected
        assert sum(family["unsupported_boundary_target_counts"].values()) == 40


def test_r2_fresh_evidence_is_consumed(contract: dict[str, Any]) -> None:
    governance = contract["consumed_evidence_governance"]

    assert governance["consumed_r2_fresh_evaluation_record_count"] == 640
    assert governance["may_be_reused_as_untouched_evaluation_evidence"] is False
    assert governance["permitted_role"] == (
        "consumed_diagnostic_development_evidence_only"
    )


def test_fresh_evaluation_is_independent_and_excluded_from_fitting(
    contract: dict[str, Any],
) -> None:
    independence = contract["fresh_evaluation_specification"][
        "authoring_independence"
    ]

    assert independence["independently_authored"] is True
    assert independence["training_use_permitted"] is False
    assert independence["representation_or_vocabulary_fitting_permitted"] is False
    assert independence["threshold_or_calibration_selection_permitted"] is False
    assert independence["candidate_modification_from_evaluation_permitted"] is False
    assert independence["copy_or_paraphrase_consumed_r2_fresh_records"] is False


def test_grouped_cv_protocol_is_exact(contract: dict[str, Any]) -> None:
    protocol = contract["evaluation_protocol"]["grouped_development_cv"]

    assert protocol["method"] == "StratifiedGroupKFold"
    assert protocol["n_splits"] == 5
    assert protocol["shuffle"] is True
    assert protocol["random_state"] == 20260930
    assert protocol["group_field"] == "group_id"
    assert protocol["group_id_is_atomic"] is True
    assert protocol["same_folds_for_both_candidates"] is True
    assert protocol["primary_router_fit_rule"] == (
        "fit_only_on_fold_development_training_partition"
    )
    assert protocol[
        "fold_test_records_in_primary_or_verifier_fitting_permitted"
    ] is False
    assert protocol["verifier_fold_test_leakage_permitted"] is False
    assert protocol["verifier_runs_after_primary_prediction"] is True


def test_fresh_evaluation_fit_protocol_is_exact(
    contract: dict[str, Any],
) -> None:
    protocol = contract["evaluation_protocol"]["fresh_evaluation"]

    assert protocol["control_candidate_fit"]["fit_count"] == 1
    assert protocol["gated_candidate_fit"]["fit_count"] == 1
    assert protocol["gated_candidate_fit"]["verifier_fit_population"] == (
        "all_120_relevant_r3_addendum_records_per_verifier"
    )
    assert protocol["evaluation_scopes"] == [
        *EVALUATION_FAMILIES,
        "pooled_r3_fresh_evaluation",
    ]
    assert protocol["same_fitted_candidate_used_for_both_families"] is True
    assert protocol["no_refit_between_source_families"] is True


def test_unchanged_safety_gates_and_scopes(contract: dict[str, Any]) -> None:
    safety = contract["safety_gates"]

    assert safety["gates"] == [
        {
            "comparison": "greater_than_or_equal",
            "metric": "protected_recall",
            "threshold": 0.8,
        },
        {
            "comparison": "less_than_or_equal",
            "metric": "protected_false_positive_rate",
            "threshold": 0.01,
        },
        {
            "comparison": "greater_than_or_equal",
            "metric": "unsupported_recall",
            "threshold": 0.8,
        },
    ]
    assert safety["thresholds_unchanged"] is True
    assert safety["thresholds_may_be_weakened"] is False
    assert safety["must_pass_every_gate_on_every_scope"] is True
    assert safety["application_scopes"] == [
        "pooled_group_aware_cv",
        *EVALUATION_FAMILIES,
        "pooled_r3_fresh_evaluation",
    ]


def test_fresh_family_false_positive_allowance_is_exact(
    contract: dict[str, Any],
) -> None:
    allowance = contract["safety_gates"][
        "fresh_family_protected_false_positive_allowance"
    ]

    assert allowance == {
        "maximum_passing_count": 1,
        "non_protected_gold_count": 160,
        "threshold": 0.01,
    }


def test_gate_specific_diagnostics_are_exact(contract: dict[str, Any]) -> None:
    diagnostics = contract["gate_specific_diagnostics"]

    assert diagnostics["required_metrics"] == [
        "primary_protected_predictions_presented_to_verifier",
        "verifier_accept_count",
        "verifier_reject_to_unsupported_count",
        "per_protected_intent_verifier_recall",
        "protected_false_positives_prevented_by_verifier",
        "true_protected_requests_rejected_by_verifier",
        "final_protected_false_positive_rate",
        "final_protected_recall",
        "final_unsupported_recall",
    ]
    assert diagnostics["alter_mandatory_safety_gates"] is False


def test_selection_rule_is_exact_and_unweighted(contract: dict[str, Any]) -> None:
    selection = contract["selection_rule"]

    assert selection["eligible_candidates_only"] is True
    assert selection["single_weighted_score_used"] is False
    assert selection["complexity_order"] == CANDIDATE_IDS
    assert selection["ordered_lexicographic_criteria"] == [
        {"direction": "maximize", "metric": "worst_fresh_family_primary_8_macro_f1"},
        {"direction": "maximize", "metric": "pooled_fresh_primary_8_macro_f1"},
        {"direction": "maximize", "metric": "pooled_group_cv_macro_f1_16"},
        {
            "direction": "minimize",
            "metric": "pooled_fresh_protected_false_positive_rate",
        },
        {"direction": "maximize", "metric": "pooled_fresh_protected_recall"},
        {"direction": "maximize", "metric": "pooled_fresh_unsupported_recall"},
        {"direction": "prefer_lower_rank", "metric": "model_complexity_rank"},
        {"direction": "ascending_lexical", "metric": "candidate_id"},
    ]


def test_no_eligible_candidate_never_forces_winner(
    contract: dict[str, Any],
) -> None:
    failure = contract["selection_rule"]["if_no_candidate_is_eligible"]

    assert failure == {
        "gates_weakened": False,
        "selected_candidate": None,
        "selection_status": "NO_ACCEPTABLE_CANDIDATE",
        "winner_forced": False,
    }


def test_failure_stop_rule_blocks_automatic_r4_cycle(
    contract: dict[str, Any],
) -> None:
    stop = contract["stop_rule"]
    failure = stop["failure_continuation"]

    assert stop["automatic_r4_classifier_or_data_cycle_authorized"] is False
    assert failure["next_required"] == "routing_architecture_fallback_decision"
    assert failure["candidate_expansion_permitted"] is False
    assert failure["data_authoring_permitted"] is False
    assert failure["representation_or_model_family_addition_permitted"] is False
    assert failure["gates_may_be_weakened"] is False
    assert stop["fallback_implemented_by_this_contract"] is False


def test_success_continuation_still_blocks_step29i_and_holdout(
    contract: dict[str, Any],
) -> None:
    success = contract["stop_rule"]["success_continuation"]

    assert success["next_required"] == (
        "v2c6_candidate_freeze_before_final_holdout"
    )
    assert success["step29i_automatically_authorized"] is False
    assert success["final_holdout_access_authorized"] is False


def test_final_holdout_is_strictly_prohibited(contract: dict[str, Any]) -> None:
    policy = contract["final_holdout_policy"]

    assert policy["prohibited_path"] == PROHIBITED_HOLDOUT_PATH
    assert policy["access_permitted"] is False
    assert policy["hashing_permitted"] is False
    assert policy["inspection_permitted"] is False
    assert policy["parsing_permitted"] is False
    assert policy["searching_permitted"] is False
    assert policy["final_model_acceptance_claim_permitted"] is False
    assert all(
        source["path"] != PROHIBITED_HOLDOUT_PATH
        for source in contract["source_artifacts"].values()
    )


def test_execution_governance_is_pre_execution_only(
    contract: dict[str, Any],
) -> None:
    status = contract["contract_status"]

    for field in (
        "candidate_selected",
        "data_authored",
        "dataset_mutated",
        "embeddings_generated",
        "final_holdout_accessed",
        "final_model_acceptance_claimed",
        "model_fitting_performed",
        "model_inference_performed",
        "model_training_performed",
        "remediation_executed",
        "runtime_behavior_changed",
        "step29i_authorized",
        "threshold_tuning_performed",
    ):
        assert status[field] is False


def test_step29i_remains_blocked(contract: dict[str, Any]) -> None:
    assert contract["contract_status"]["step29i_authorized"] is False
    assert contract["stop_rule"]["success_continuation"][
        "step29i_automatically_authorized"
    ] is False


def test_immediate_next_activity_is_r3_data_authoring_only(
    contract: dict[str, Any],
) -> None:
    assert contract["next_required"] == (
        "v2c6_r3_targeted_addendum_and_fresh_evaluation_authoring"
    )
    assert contract["contract_status"]["data_authored"] is False
    assert contract["contract_status"]["remediation_executed"] is False
