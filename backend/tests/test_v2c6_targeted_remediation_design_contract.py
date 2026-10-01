from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "data/evals/v2/ml/v2c6_targeted_remediation_design_contract.json"
)
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"

TARGETED_INTENTS = [
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]
CONSUMED_FAMILIES = [
    "v2c6_sf1_definition_direct",
    "v2c6_sf2_scenario_narrative",
    "v2c6_sf3_boundary_conversational",
]
TRAINING_FAMILIES = [
    "v2c6_r2_train_sf1_minimal_boundary",
    "v2c6_r2_train_sf2_contextual_scenario",
    "v2c6_r2_train_sf3_conversational_correction",
]
EVALUATION_FAMILIES = [
    "v2c6_r2_eval_sf1_independent_casework",
    "v2c6_r2_eval_sf2_independent_naturalistic",
]
CANDIDATES = [
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "HIERARCHICAL_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
]
HARD_NEGATIVE_PAIRS = [
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


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unordered_pair(pair: list[str]) -> frozenset[str]:
    return frozenset(pair)


def test_contract_identity_and_status(contract: dict[str, Any]) -> None:
    schema = "v2c6-targeted-remediation-design-contract.v1"

    assert contract["schema_version"] == schema
    assert contract["contract_version"] == schema
    assert contract["phase"] == "V2-C6 Step 29H-C"
    assert contract["status"] == "FROZEN"
    assert contract["contract_status"]["contract_frozen"] is True
    assert contract["contract_status"]["design_only"] is True


def test_exact_step29h_b_result_required(contract: dict[str, Any]) -> None:
    source = contract["source_artifacts"]["step29h_b_analysis"]
    result = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))

    assert sha256_file(ROOT / source["path"]) == source["sha256"]
    assert result["schema_version"] == source["schema_version"]
    assert result["execution_status"] == source["expected_execution_status"]


def test_step29h_b_manifest_required(contract: dict[str, Any]) -> None:
    sources = contract["source_artifacts"]
    source = sources["step29h_b_manifest"]
    manifest = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))

    assert sha256_file(ROOT / source["path"]) == source["sha256"]
    assert manifest["schema_version"] == source["schema_version"]
    assert manifest["result"]["sha256"] == sources["step29h_b_analysis"][
        "sha256"
    ]


def test_all_source_hashes_are_exact(contract: dict[str, Any]) -> None:
    for source in contract["source_artifacts"].values():
        assert source["path"] != PROHIBITED_HOLDOUT_PATH
        assert sha256_file(ROOT / source["path"]) == source["sha256"]


def test_step29h_b_classification_is_both(contract: dict[str, Any]) -> None:
    source = contract["source_artifacts"]["step29h_b_analysis"]
    result = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))
    classification = result["remediation_evidence_summary"][
        "evidence_classification"
    ]

    assert source["expected_remediation_evidence_classification"] == "both"
    assert classification["classification"] == "both"
    assert contract["measured_basis"]["remediation_evidence_classification"][
        "classification"
    ] == "both"


def test_no_remediation_already_selected(contract: dict[str, Any]) -> None:
    source = contract["source_artifacts"]["step29h_b_analysis"]
    result = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))

    assert source["expected_remediation_selected"] is False
    assert result["governance"]["remediation_selected"] is False
    assert contract["contract_status"]["remediation_executed"] is False


def test_step29i_is_blocked(contract: dict[str, Any]) -> None:
    policy = contract["step29i_policy"]

    assert policy["blocked"] is True
    assert policy["eligible_now"] is False
    assert policy["this_contract_authorizes_step29i"] is False
    assert contract["contract_status"]["step29i_authorized"] is False


def test_exact_consumed_source_families(contract: dict[str, Any]) -> None:
    assert contract["consumed_evidence_governance"]["source_families"] == (
        CONSUMED_FAMILIES
    )


def test_consumed_families_cannot_be_fresh(contract: dict[str, Any]) -> None:
    governance = contract["consumed_evidence_governance"]

    assert governance[
        "permanently_consumed_for_development_selection_and_diagnosis"
    ] is True
    assert governance["source_families_may_count_as_fresh_after_remediation"] is False
    assert set(governance["prohibited_descriptions"]) == {
        "fresh",
        "unseen",
        "independent_post_remediation_validation",
        "clean_source_generalization_evidence",
    }


def test_exact_training_remediation_total(contract: dict[str, Any]) -> None:
    assert contract["training_remediation_specification"]["record_count"] == 600


def test_exact_three_training_families(contract: dict[str, Any]) -> None:
    training = contract["training_remediation_specification"]

    assert training["family_count"] == 3
    assert training["source_family_ids"] == TRAINING_FAMILIES
    assert len(training["family_specifications"]) == 3


def test_exact_200_records_per_training_family(
    contract: dict[str, Any],
) -> None:
    families = contract["training_remediation_specification"][
        "family_specifications"
    ]

    assert all(family["record_count"] == 200 for family in families)
    assert sum(family["record_count"] for family in families) == 600
    assert all(
        set(family["intent_counts"]) == set(TARGETED_INTENTS)
        and sum(family["intent_counts"].values()) == 200
        for family in families
    )


def test_exact_supported_intent_counts_per_training_family(
    contract: dict[str, Any],
) -> None:
    families = contract["training_remediation_specification"][
        "family_specifications"
    ]

    for family in families:
        for intent in TARGETED_INTENTS[:-1]:
            assert family["intent_counts"][intent] == 20


def test_exact_unsupported_count_per_training_family(
    contract: dict[str, Any],
) -> None:
    families = contract["training_remediation_specification"][
        "family_specifications"
    ]

    assert all(
        family["intent_counts"]["unsupported_or_uncertain"] == 60
        for family in families
    )


def test_exact_eight_primary_remediation_intents(
    contract: dict[str, Any],
) -> None:
    assert contract["targeted_intents"] == TARGETED_INTENTS
    assert contract["training_remediation_specification"]["intent_count"] == 8


def test_exact_ten_hard_negative_boundaries_preserved(
    contract: dict[str, Any],
) -> None:
    boundary = contract["training_remediation_specification"][
        "hard_negative_boundaries"
    ]

    assert boundary["all_step29d_pairs_must_be_covered"] is True
    assert boundary["required_pairs"] == HARD_NEGATIVE_PAIRS


def test_tier_one_unsupported_boundaries_are_present(
    contract: dict[str, Any],
) -> None:
    tier_one = {
        unordered_pair(pair)
        for pair in contract["training_remediation_specification"][
            "hard_negative_boundaries"
        ]["tier_1"]
    }

    assert tier_one == {
        frozenset({"unsupported_or_uncertain", intent})
        for intent in [
            "cancel_transfer",
            "close_account",
            "create_dispute",
            "freeze_card",
            "transfer_pending",
            "transfer_failed_or_declined",
        ]
    }


def test_cancel_transfer_vs_transfer_pending_is_present(
    contract: dict[str, Any],
) -> None:
    tier_two = contract["training_remediation_specification"][
        "hard_negative_boundaries"
    ]["tier_2"]

    assert ["cancel_transfer", "transfer_pending"] in tier_two


def test_exact_four_candidates(contract: dict[str, Any]) -> None:
    search = contract["candidate_search_space"]

    assert search["candidate_count"] == 4
    assert search["exact_candidate_ids"] == CANDIDATES
    assert len(search["candidates"]) == 4


def test_bge_c4_unweighted_control(contract: dict[str, Any]) -> None:
    candidate = contract["candidate_search_space"]["candidates"][0]

    assert candidate["candidate_id"] == CANDIDATES[0]
    assert candidate["C"] == 4.0
    assert candidate["class_weight"] is None
    assert candidate["role"] == "semantic_control"


def test_tfidf_c1_unweighted_control(contract: dict[str, Any]) -> None:
    candidate = contract["candidate_search_space"]["candidates"][1]

    assert candidate["candidate_id"] == CANDIDATES[1]
    assert candidate["C"] == 1.0
    assert candidate["class_weight"] is None
    assert candidate["winner_claimed"] is False


def test_hybrid_candidate_is_defined(contract: dict[str, Any]) -> None:
    candidate = contract["candidate_search_space"]["candidates"][2]

    assert candidate["candidate_id"] == CANDIDATES[2]
    assert candidate["representation"] == "HYBRID_BGE_TFIDF"


def test_hierarchical_candidate_is_defined(contract: dict[str, Any]) -> None:
    candidate = contract["candidate_search_space"]["candidates"][3]

    assert candidate["candidate_id"] == CANDIDATES[3]
    assert candidate["classifier"] == "two_stage_LinearSVC"


def test_no_balanced_candidate(contract: dict[str, Any]) -> None:
    search = contract["candidate_search_space"]

    assert search["balanced_class_weighting_carried_forward"] is False
    assert all(
        candidate["class_weight"] is None for candidate in search["candidates"]
    )
    assert all("balanced" not in identifier for identifier in CANDIDATES)


def test_no_new_embedding_model(contract: dict[str, Any]) -> None:
    search = contract["candidate_search_space"]
    hybrid = contract["candidate_implementation_semantics"][CANDIDATES[2]]

    assert search["new_embedding_model_introduced"] is False
    assert hybrid["new_embedding_model_introduced"] is False


def test_no_threshold_tuning(contract: dict[str, Any]) -> None:
    semantics = contract["candidate_implementation_semantics"]

    assert semantics["common"]["threshold_tuning_permitted"] is False
    assert contract["contract_status"]["threshold_tuning_performed"] is False
    assert contract["excluded_alternatives"][
        "thresholds_chosen_from_step29h_errors"
    ] is True


def test_no_calibration(contract: dict[str, Any]) -> None:
    semantics = contract["candidate_implementation_semantics"]

    assert semantics["common"]["calibration_permitted"] is False
    assert semantics[CANDIDATES[2]]["calibration_permitted"] is False
    assert semantics[CANDIDATES[3]]["calibration_permitted"] is False


def test_hierarchical_stage_one_semantics(contract: dict[str, Any]) -> None:
    stage = contract["candidate_implementation_semantics"][CANDIDATES[3]][
        "stage_1"
    ]

    assert stage == {
        "C": 1.0,
        "class_weight": None,
        "classes": ["unsupported_or_uncertain", "supported"],
        "classifier": "LinearSVC",
        "representation": "WORD_CHAR_TFIDF",
    }


def test_hierarchical_stage_two_semantics(contract: dict[str, Any]) -> None:
    stage = contract["candidate_implementation_semantics"][CANDIDATES[3]][
        "stage_2"
    ]

    assert stage["class_count"] == 15
    assert stage["classes"] == [
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
    ]
    assert stage["excluded_intent"] == "unsupported_or_uncertain"
    assert stage["representation"] == "WORD_CHAR_TFIDF"
    assert stage["classifier"] == "LinearSVC"


def test_hierarchical_inference_is_deterministic(
    contract: dict[str, Any],
) -> None:
    semantics = contract["candidate_implementation_semantics"][CANDIDATES[3]]

    assert semantics["deterministic_inference"] == {
        "otherwise": "use_stage_2_intent_prediction",
        "when_stage_1_predicts_unsupported_or_uncertain": (
            "final_prediction_is_unsupported_or_uncertain"
        ),
    }
    assert semantics["confidence_override_permitted"] is False
    assert semantics["preprocessing_convention"] == (
        "reuse_frozen_step29g_WORD_CHAR_TFIDF_convention"
    )
    assert semantics["protected_intent_rule_special_casing_permitted"] is False


def test_hybrid_representation_semantics(contract: dict[str, Any]) -> None:
    hybrid = contract["candidate_implementation_semantics"][CANDIDATES[2]]

    assert hybrid["bge_block"]["dimensions"] == 384
    assert hybrid["tfidf_block"]["members"] == [
        "WORD_TFIDF",
        "CHAR_TFIDF",
    ]
    assert "sparse_hstack" in hybrid["concatenation"]
    assert hybrid["final_combined_row_l2_normalization"] is True
    assert hybrid["learned_fusion_layer_permitted"] is False


def test_exact_two_fresh_evaluation_families(
    contract: dict[str, Any],
) -> None:
    evaluation = contract["fresh_evaluation_specification"]

    assert evaluation["family_count"] == 2
    assert evaluation["source_family_ids"] == EVALUATION_FAMILIES


def test_exact_320_records_per_fresh_family(
    contract: dict[str, Any],
) -> None:
    families = contract["fresh_evaluation_specification"][
        "family_specifications"
    ]

    assert all(family["record_count"] == 320 for family in families)


def test_exact_40_records_per_intent_per_fresh_family(
    contract: dict[str, Any],
) -> None:
    families = contract["fresh_evaluation_specification"][
        "family_specifications"
    ]

    for family in families:
        assert family["intent_counts"] == {
            intent: 40 for intent in TARGETED_INTENTS
        }


def test_exact_640_fresh_evaluation_total(contract: dict[str, Any]) -> None:
    evaluation = contract["fresh_evaluation_specification"]

    assert evaluation["record_count"] == 640
    assert sum(
        family["record_count"] for family in evaluation["family_specifications"]
    ) == 640


def test_fresh_eval_protected_fpr_allowance_is_derived(
    contract: dict[str, Any],
) -> None:
    allowance = contract["fresh_evaluation_specification"][
        "protected_false_positive_allowance_per_family"
    ]

    assert allowance["non_protected_record_count"] == 160
    assert allowance["threshold"] == 0.01
    assert allowance["formula"] == (
        "floor(0.01 * non_protected_record_count)"
    )
    assert allowance["maximum_passing_count"] == 1


def test_fresh_evaluation_is_excluded_from_fitting(
    contract: dict[str, Any],
) -> None:
    evaluation = contract["fresh_evaluation_specification"]
    protocol = contract["training_and_evaluation_protocol"]

    assert evaluation["excluded_from_candidate_fitting"] is True
    assert protocol["fresh_evaluation_population"][
        "training_use_permitted_before_candidate_evaluation"
    ] is False
    assert protocol["fresh_source_evaluation"][
        "fresh_records_excluded_from_fitting"
    ] is True


def test_independent_fresh_authoring_is_required(
    contract: dict[str, Any],
) -> None:
    independence = contract["fresh_evaluation_specification"][
        "authoring_independence_requirements"
    ]

    assert independence["authored_separately_from_training_remediation"] is True
    assert independence["separate_prompts_templates_and_processes_required"] is True
    assert independence["must_not_paraphrase_training_examples"] is True
    assert independence[
        "must_not_use_consumed_sf1_sf2_sf3_as_paraphrase_templates"
    ] is True


def test_no_prediction_informed_authoring(contract: dict[str, Any]) -> None:
    independence = contract["fresh_evaluation_specification"][
        "authoring_independence_requirements"
    ]
    governance = contract["authoring_and_review_governance"]

    assert independence[
        "model_predictions_may_generate_or_select_examples"
    ] is False
    assert independence[
        "candidate_outputs_may_be_inspected_during_authoring"
    ] is False
    assert governance["model_in_the_loop_initial_authoring_permitted"] is False


def test_duplicate_and_leakage_controls_are_exact(
    contract: dict[str, Any],
) -> None:
    controls = contract["duplicate_and_leakage_controls"]

    assert controls["required_checks"] == [
        "exact_duplicate_text",
        "normalized_duplicate_text",
        "record_id_collision",
        "group_id_collision_where_prohibited",
        "training_evaluation_exact_overlap",
        "fresh_evaluation_historical_development_exact_overlap",
        "fresh_evaluation_consumed_sf1_sf2_sf3_exact_overlap",
        "cross_intent_exact_duplicate",
        "source_family_provenance_consistency",
    ]
    assert controls["failure_behavior"] == "fail_closed"
    assert controls["embedding_based_near_duplicate_filtering_required"] is False


def test_group_id_is_atomic(contract: dict[str, Any]) -> None:
    assert contract["group_aware_cross_validation"]["group_id_is_atomic"] is True


def test_five_fold_stratified_group_k_fold(contract: dict[str, Any]) -> None:
    protocol = contract["group_aware_cross_validation"]

    assert protocol["method"] == "StratifiedGroupKFold"
    assert protocol["n_splits"] == 5
    assert protocol["non_group_aware_fallback_allowed"] is False


def test_random_state_is_exact(contract: dict[str, Any]) -> None:
    assert contract["group_aware_cross_validation"]["random_state"] == 20260930


def test_safety_thresholds_are_unchanged(contract: dict[str, Any]) -> None:
    safety = contract["safety_gates"]

    assert safety["thresholds_unchanged_from_step29g"] is True
    assert [gate["metric"] for gate in safety["gates"]] == [
        "protected_recall",
        "protected_false_positive_rate",
        "unsupported_recall",
    ]
    assert [gate["threshold"] for gate in safety["gates"]] == [0.8, 0.01, 0.8]


@pytest.mark.parametrize(
    "scope",
    [
        "pooled_group_aware_cv",
        "v2c6_r2_eval_sf1_independent_casework",
        "v2c6_r2_eval_sf2_independent_naturalistic",
        "pooled_fresh_evaluation",
    ],
)
def test_safety_gates_are_required_on_every_frozen_scope(
    contract: dict[str, Any], scope: str
) -> None:
    safety = contract["safety_gates"]

    assert safety["gates_mandatory_for_candidate_eligibility"] is True
    assert scope in safety["application_scopes"]


def test_consumed_old_families_are_diagnostic_only(
    contract: dict[str, Any],
) -> None:
    protocol = contract["old_source_family_diagnostic_protocol"]

    assert protocol["families"] == CONSUMED_FAMILIES
    assert protocol["allowed_role"] == (
        "consumed_diagnostic_regression_evidence_only"
    )
    assert protocol["may_count_as_fresh_evidence"] is False
    assert protocol["may_replace_new_evaluation_family"] is False
    assert protocol["may_claim_remediation_generalization"] is False


def test_selection_is_deterministic_and_eligible_only(
    contract: dict[str, Any],
) -> None:
    selection = contract["selection_rule"]

    assert selection["selection_population"] == "eligible_candidates_only"
    assert selection["single_weighted_composite_score_used"] is False
    assert len(selection["ordered_lexicographic_criteria"]) == 9
    assert selection["ordered_lexicographic_criteria"][0]["metric"] == (
        "worst_fresh_family_primary_8_macro_f1"
    )
    assert selection["ordered_lexicographic_criteria"][-1]["metric"] == (
        "candidate_id"
    )


def test_no_acceptable_candidate_behavior(contract: dict[str, Any]) -> None:
    failure = contract["selection_rule"]["if_no_candidate_is_eligible"]

    assert failure["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert failure["selected_candidate"] is None


def test_winner_is_never_forced(contract: dict[str, Any]) -> None:
    failure = contract["selection_rule"]["if_no_candidate_is_eligible"]

    assert failure["winner_forced"] is False


def test_gates_are_never_weakened(contract: dict[str, Any]) -> None:
    failure = contract["selection_rule"]["if_no_candidate_is_eligible"]

    assert failure["gates_weakened"] is False
    assert contract["safety_gates"][
        "thresholds_may_be_weakened_after_results"
    ] is False


def test_complexity_ordering_is_exact(contract: dict[str, Any]) -> None:
    complexity = contract["complexity_ordering"]

    assert complexity["simplest_to_more_complex"] == [
        "WORD_CHAR_TFIDF_LINEAR_SVC",
        "BGE_SMALL_LINEAR_SVC",
        "HYBRID_BGE_TFIDF_LINEAR_SVC",
        "HIERARCHICAL_TFIDF_LINEAR_SVC",
    ]
    assert complexity["interpretation"] == (
        "final_tie_breaker_only_not_quality_ranking"
    )


def test_no_final_holdout_access(contract: dict[str, Any]) -> None:
    policy = contract["final_holdout_policy"]

    assert policy["prohibited_path"] == PROHIBITED_HOLDOUT_PATH
    assert policy["existing_final_holdout_access_permitted"] is False
    assert policy["future_v2c6_final_holdout_access_permitted"] is False
    assert contract["contract_status"]["final_holdout_accessed"] is False


def test_no_v2c6_final_holdout_creation(contract: dict[str, Any]) -> None:
    policy = contract["final_holdout_policy"]

    assert policy["final_holdout_created_by_this_step"] is False
    assert policy["future_v2c6_final_holdout_authoring_permitted"] is False
    assert contract["contract_status"][
        "future_v2c6_final_holdout_created"
    ] is False


def test_step29i_remains_blocked(contract: dict[str, Any]) -> None:
    assert contract["motivation"]["step29i_blocked"] is True
    assert contract["step29i_policy"]["blocked"] is True


def test_exact_next_required(contract: dict[str, Any]) -> None:
    expected = "v2c6_targeted_remediation_and_fresh_source_authoring"

    assert contract["contract_status"]["next_required"] == expected
    assert contract["next_required"] == (
        "v2c6_targeted_remediation_and_fresh_source_authoring"
    )


def test_no_model_execution_is_authorized(contract: dict[str, Any]) -> None:
    status = contract["contract_status"]
    prohibited = set(contract["prohibited_operations"])

    assert status["model_fitting_performed"] is False
    assert status["model_inference_performed"] is False
    assert status["embeddings_generated"] is False
    assert {"embedding_generation", "model_fitting", "model_inference"} <= (
        prohibited
    )


def test_no_dataset_authoring_was_performed(contract: dict[str, Any]) -> None:
    status = contract["contract_status"]

    assert status["new_training_records_authored"] is False
    assert status["new_evaluation_records_authored"] is False
    assert status["dataset_mutated"] is False
    assert contract["training_remediation_specification"]["record_count"] == 600
    assert contract["fresh_evaluation_specification"]["record_count"] == 640


def test_architecture_decision_is_present(contract: dict[str, Any]) -> None:
    decision = contract["architecture_decision"]

    assert decision["chosen"] == (
        "targeted_boundary_remediation_plus_bounded_candidate_expansion_plus_"
        "fresh_source_family_evaluation"
    )
    assert len(decision["alternatives_considered"]) == 6
    assert "safety_first" in decision["why_suitable_for_sentinelvoice"]


def test_architecture_tradeoffs_are_documented(contract: dict[str, Any]) -> None:
    tradeoffs = contract["architecture_decision"]["tradeoffs"]

    assert len(tradeoffs) == 4
    assert any("1240" in tradeoff for tradeoff in tradeoffs)
    assert any("no guarantee" in tradeoff for tradeoff in tradeoffs)
    assert contract["architecture_decision"]["reversibility"].startswith("High")


def test_no_causal_root_cause_claim(contract: dict[str, Any]) -> None:
    assert contract["measured_basis"]["causal_root_cause_proven"] is False
    assert "without proving a causal root cause" in contract[
        "architecture_decision"
    ]["problem_solved"]


def test_no_remediation_success_claim(contract: dict[str, Any]) -> None:
    assert contract["contract_status"]["remediation_success_claimed"] is False
    assert "remediation_success_claim" in contract["prohibited_operations"]


def test_measured_bidirectional_boundary_evidence(
    contract: dict[str, Any],
) -> None:
    measured = contract["measured_basis"]
    false_positives = measured["unsupported_to_protected_false_positives"]

    assert false_positives["observed_across_representation_families"] == [
        "BGE",
        "TFIDF",
    ]
    assert false_positives["observed_across_source_families"] == (
        CONSUMED_FAMILIES
    )
    assert measured["protected_to_unsupported_recall_misses"][
        "bidirectional_boundary_weakness_observed"
    ] is True


def test_measured_hard_negative_basis_is_exact(
    contract: dict[str, Any],
) -> None:
    assert contract["measured_basis"][
        "important_hard_negative_boundaries"
    ] == [
        ["transfer_pending", "unsupported_or_uncertain"],
        ["cancel_transfer", "unsupported_or_uncertain"],
        ["close_account", "unsupported_or_uncertain"],
        ["transfer_failed_or_declined", "unsupported_or_uncertain"],
        ["freeze_card", "unsupported_or_uncertain"],
        ["cancel_transfer", "transfer_pending"],
    ]


def test_class_weight_observation_is_experiment_specific(
    contract: dict[str, Any],
) -> None:
    observation = contract["measured_basis"][
        "balanced_class_weighting_observation"
    ]

    assert observation["generally_increased_protected_recall"] is True
    assert observation[
        "generally_increased_protected_false_positive_rate"
    ] is True
    assert observation["generally_reduced_unsupported_recall"] is True
    assert observation["descriptive_only"] is True
    assert observation["universal_claim"] is False


def test_unsupported_authoring_vocabulary_is_preserved(
    contract: dict[str, Any],
) -> None:
    unsupported = contract["training_remediation_specification"][
        "unsupported_authoring"
    ]

    assert unsupported["subtype_vocabulary"] == [
        "truly_unsupported_banking_request",
        "ambiguous_or_insufficient_information",
        "adjacent_but_unsupported_intent",
        "supported_intent_hard_negative",
        "off_domain_or_noise",
    ]
    assert len(unsupported["required_case_types"]) == 7
    assert unsupported["avoid_obvious_keyword_shortcuts"] is True


def test_review_governance_requires_human_adjudication_triggers(
    contract: dict[str, Any],
) -> None:
    governance = contract["authoring_and_review_governance"]

    assert governance["human_adjudication_mandatory_for"] == [
        "rejected",
        "needs_revision",
        "reviewer_disagreement",
        "low_confidence_review",
        "unresolved_ambiguity",
        "provenance_inconsistency",
        "protected_write_ambiguity",
    ]
    assert governance[
        "ai_assisted_review_must_not_be_described_as_independent_human_review"
    ] is True
    assert governance["provenance_may_be_inferred_from_wording"] is False
    assert "source_family_independence_basis" in governance[
        "required_record_metadata"
    ]
    assert governance["unsupported_additional_required_metadata"] == [
        "unsupported_subtype"
    ]


def test_training_and_development_population_is_exact(
    contract: dict[str, Any],
) -> None:
    population = contract["training_and_evaluation_protocol"][
        "training_and_model_development_population"
    ]

    assert population == {
        "existing_frozen_v2c6_development_record_count": 9008,
        "new_targeted_training_record_count": 600,
        "total_record_count": 9608,
    }


def test_fresh_source_evaluation_protocol_is_exact(
    contract: dict[str, Any],
) -> None:
    protocol = contract["training_and_evaluation_protocol"][
        "fresh_source_evaluation"
    ]

    assert protocol["candidate_fit_population"] == (
        "entire_9608_record_training_and_model_development_population"
    )
    assert protocol["evaluation_scopes"] == [
        *EVALUATION_FAMILIES,
        "pooled_fresh_evaluation",
    ]
    assert protocol["fresh_families_become_consumed_after_model_selection"] is True


def test_macro_f1_populations_are_not_conflated(
    contract: dict[str, Any],
) -> None:
    metrics = contract["metrics_and_comparability"]

    assert metrics["fresh_source_evaluation_primary_metric"] == (
        "primary_8_macro_f1"
    )
    assert metrics["group_cv_primary_metric"] == "macro_f1_16"
    assert metrics["macro_f1_populations_are_equivalent"] is False
    assert metrics["unqualified_direct_subtraction_permitted"] is False
