from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / (
    "data/evals/v2/ml/"
    "v2c6_targeted_remediation_failure_analysis_contract.json"
)
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"

EXPECTED_CANDIDATES = [
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "HIERARCHICAL_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
]
EXPECTED_FRESH_FAMILIES = [
    "v2c6_r2_eval_sf1_independent_casework",
    "v2c6_r2_eval_sf2_independent_naturalistic",
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


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(contract: dict[str, Any], name: str) -> dict[str, Any]:
    spec = contract["source_artifacts"][name]
    return json.loads((ROOT / spec["path"]).read_text(encoding="utf-8"))


def test_contract_identity_and_pre_execution_status(
    contract: dict[str, Any],
) -> None:
    schema = "v2c6-targeted-remediation-failure-analysis-contract.v1"

    assert contract["schema_version"] == schema
    assert contract["contract_version"] == schema
    assert contract["phase"] == (
        "V2-C6 post-targeted-remediation failure-analysis contract"
    )
    assert contract["status"] == "FROZEN"
    assert contract["contract_status"]["contract_frozen"] is True
    assert contract["contract_status"]["failure_analysis_executed"] is False


def test_every_source_artifact_path_and_hash_is_exact(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]

    assert set(sources) == {
        "fresh_evaluation_dataset",
        "fresh_evaluation_manifest",
        "targeted_development_dataset",
        "targeted_development_manifest",
        "targeted_remediation_design_contract",
        "targeted_remediation_model_selection_results",
        "targeted_remediation_model_selection_results_manifest",
    }
    for source in sources.values():
        assert source["path"] != PROHIBITED_HOLDOUT_PATH
        assert sha256_file(ROOT / source["path"]) == source["sha256"]


def test_exact_completed_result_is_the_prediction_source(
    contract: dict[str, Any],
) -> None:
    spec = contract["source_artifacts"][
        "targeted_remediation_model_selection_results"
    ]
    results = load_source(
        contract, "targeted_remediation_model_selection_results"
    )

    assert spec["path"] == contract["input_policy"]["persisted_prediction_source"]
    assert spec["sha256"] == (
        "81fc64cd3476cd4eb2c6dc1e7b555803692f4800d48a61665fa9fd7768c9f145"
    )
    assert results["schema_version"] == spec["schema_version"]
    assert results["execution_status"] == spec["expected_execution_status"]
    assert results["candidate_count"] == spec["expected_candidate_count"]


def test_no_acceptable_candidate_prerequisite_is_exact(
    contract: dict[str, Any],
) -> None:
    results = load_source(
        contract, "targeted_remediation_model_selection_results"
    )
    selection = results["selection"]
    expected = contract["motivation"]["input_state"]

    assert expected == {
        "candidate_count": 4,
        "eligible_candidate_count": 0,
        "execution_status": "COMPLETED",
        "gates_weakened": False,
        "next_required": None,
        "selected_candidate": None,
        "selection_status": "NO_ACCEPTABLE_CANDIDATE",
        "step29i_authorized": False,
        "winner_forced": False,
    }
    assert selection["eligible_candidate_count"] == 0
    assert selection["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert selection["selected_candidate"] is None
    assert selection["winner_forced"] is False
    assert selection["gates_weakened"] is False
    assert selection["step29i_authorized"] is False
    assert selection["next_required"] is None


def test_results_manifest_identity_and_governance_are_pinned(
    contract: dict[str, Any],
) -> None:
    results_spec = contract["source_artifacts"][
        "targeted_remediation_model_selection_results"
    ]
    manifest_spec = contract["source_artifacts"][
        "targeted_remediation_model_selection_results_manifest"
    ]
    manifest = load_source(
        contract, "targeted_remediation_model_selection_results_manifest"
    )

    assert manifest_spec["sha256"] == (
        "5922497a5a05e44833c525de1fc4386b9bff763db4bb2185577e8a6496f15abb"
    )
    assert manifest["schema_version"] == manifest_spec["schema_version"]
    assert manifest["results"]["sha256"] == results_spec["sha256"]
    assert manifest["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert manifest["selected_candidate_id"] is None
    assert manifest["step29i_authorized"] is False
    assert manifest["next_required"] is None


def test_all_four_candidate_ids_are_exact(contract: dict[str, Any]) -> None:
    coverage = contract["candidate_coverage"]
    results = load_source(
        contract, "targeted_remediation_model_selection_results"
    )

    assert coverage["candidate_count"] == 4
    assert coverage["candidate_ids"] == EXPECTED_CANDIDATES
    assert [row["candidate_id"] for row in results["candidate_results"]] == (
        EXPECTED_CANDIDATES
    )
    assert coverage["all_candidates_are_diagnostic_inputs"] is True
    assert coverage["candidate_selection_permitted"] is False
    assert coverage["winner_may_be_declared"] is False


def test_exact_fresh_families_are_consumed_diagnostic_evidence(
    contract: dict[str, Any],
) -> None:
    governance = contract["consumed_evidence_governance"]

    assert governance["fresh_evaluation_record_count"] == 640
    assert governance["fresh_evaluation_source_family_ids"] == (
        EXPECTED_FRESH_FAMILIES
    )
    assert governance["fresh_evaluation_already_consumed_by_model_selection"] is True
    assert governance["inspection_further_consumes_as_diagnostic_evidence"] is True
    assert governance["may_be_called_fresh_or_untouched_after_analysis"] is False


def test_protected_false_positive_dimensions_are_frozen(
    contract: dict[str, Any],
) -> None:
    analysis = contract["analysis_dimensions"]["protected_false_positives"]

    assert analysis["protected_intents"] == [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]
    assert "true_intent_by_predicted_protected_intent_matrix" in analysis[
        "required_breakdowns"
    ]
    assert analysis["required_scopes"] == [
        "pooled_group_aware_cv",
        *EXPECTED_FRESH_FAMILIES,
        "pooled_fresh_evaluation",
    ]


def test_unsupported_miss_dimensions_are_frozen(
    contract: dict[str, Any],
) -> None:
    analysis = contract["analysis_dimensions"]["unsupported_misses"]

    assert analysis["unsupported_intent"] == "unsupported_or_uncertain"
    assert "counts_by_predicted_intent" in analysis["summary_fields"]
    assert "protected_destination_count" in analysis["summary_fields"]
    assert "non_protected_destination_count" in analysis["summary_fields"]
    assert "protected_vs_non_protected_destination" in analysis[
        "required_breakdowns"
    ]


def test_protected_recall_misses_are_separate_from_false_positives(
    contract: dict[str, Any],
) -> None:
    analysis = contract["analysis_dimensions"]["protected_recall_misses"]

    assert analysis["confusion_definition"] == (
        "gold intent is protected and predicted intent is non-protected"
    )
    assert analysis[
        "must_be_reported_separately_from_protected_false_positives"
    ] is True
    assert "true_protected_intent" in analysis["required_breakdowns"]
    assert "predicted_intent" in analysis["required_breakdowns"]


def test_all_ten_directional_hard_negative_boundaries_are_frozen(
    contract: dict[str, Any],
) -> None:
    analysis = contract["analysis_dimensions"]["hard_negative_boundaries"]

    assert analysis["required_pairs"] == EXPECTED_HARD_NEGATIVE_PAIRS
    assert len(analysis["required_pairs"]) == 10
    assert analysis["combined_accuracy_alone_is_sufficient"] is False
    assert analysis["directional_fields"] == [
        "true_a_predicted_b_count",
        "true_b_predicted_a_count",
        "other_error_count",
    ]


def test_cross_candidate_overlap_is_diagnostic_only(
    contract: dict[str, Any],
) -> None:
    overlap = contract["analysis_dimensions"]["cross_candidate_overlap"]

    assert overlap["required_overlap_views"] == [
        "intersection_shared_by_all_four_candidates",
        "intersection_shared_by_bge_tfidf_and_hybrid",
        "failures_unique_to_each_candidate",
        "pairwise_intersection_count",
        "pairwise_union_count",
        "pairwise_jaccard",
    ]
    assert overlap["zero_union_jaccard_value"] is None
    assert overlap["selection_or_ranking_permitted"] is False


def test_fresh_family_comparison_is_nonjudgmental(
    contract: dict[str, Any],
) -> None:
    comparison = contract["analysis_dimensions"]["fresh_family_consistency"]

    assert comparison["source_family_ids"] == EXPECTED_FRESH_FAMILIES
    assert comparison["family_quality_ranking_permitted"] is False
    assert set(comparison["required_comparisons"]) == {
        "protected_false_positive_patterns",
        "unsupported_misses",
        "protected_recall_misses",
        "hard_negative_directional_confusion",
    }


def test_group_cv_vs_fresh_comparison_is_descriptive(
    contract: dict[str, Any],
) -> None:
    comparison = contract["analysis_dimensions"][
        "group_cv_vs_fresh_evaluation"
    ]

    assert comparison["comparisons_limited_to_persisted_prediction_evidence"] is True
    assert comparison["causal_or_source_quality_claims_permitted"] is False
    assert comparison["required_questions"] == [
        "whether_protected_false_positive_weakness_appears_in_both_grouped_cv_and_fresh_evidence",
        "whether_unsupported_recall_deterioration_is_disproportionately_fresh_source_specific",
    ]


def test_hierarchical_stage_attribution_limitation_is_explicit(
    contract: dict[str, Any],
) -> None:
    diagnosis = contract["analysis_dimensions"][
        "hierarchical_architecture_diagnosis"
    ]

    assert diagnosis["stage_level_predictions_persisted"] is False
    assert diagnosis["exact_stage_1_error_attribution_available"] is False
    assert diagnosis["model_reconstruction_permitted"] is False
    assert diagnosis["model_rerun_permitted"] is False
    assert diagnosis["required_limitation_statement"] == (
        "Stage-level predictions were not persisted, so exact Stage-1 error "
        "attribution is unavailable."
    )


def test_evidence_classification_is_noncausal_and_multi_label(
    contract: dict[str, Any],
) -> None:
    classification = contract["evidence_classification"]

    assert classification["allow_multiple_categories"] is True
    assert classification["allowed_categories"] == [
        "development_distribution_boundary_weakness",
        "fresh_source_generalization_weakness",
        "architecture_specific_weakness",
        "cross_architecture_shared_weakness",
        "insufficient_evidence",
    ]
    assert classification["causal_claims_permitted_without_deterministic_proof"] is False
    assert classification["classification_is_candidate_selection"] is False
    assert classification["classification_is_remediation_selection"] is False


def test_read_only_governance_prohibits_model_and_data_changes(
    contract: dict[str, Any],
) -> None:
    prohibited = set(contract["prohibited_operations"])

    assert {
        "model_training",
        "model_refitting",
        "embedding_generation",
        "classifier_inference",
        "threshold_tuning",
        "candidate_definition_change",
        "candidate_selection",
        "candidate_ranking",
        "safety_gate_change",
        "safety_gate_weakening",
        "training_example_addition",
        "dataset_mutation",
        "label_mutation",
        "taxonomy_mutation",
        "runtime_change",
        "remediation_selection",
        "remediation_implementation",
    }.issubset(prohibited)
    status = contract["contract_status"]
    assert status["embeddings_generated"] is False
    assert status["model_fitting_performed"] is False
    assert status["model_inference_performed"] is False
    assert status["threshold_tuning_performed"] is False
    assert status["dataset_mutated"] is False


def test_final_holdout_is_neither_input_nor_permitted(
    contract: dict[str, Any],
) -> None:
    prohibited = set(contract["prohibited_operations"])
    final_inputs = contract["input_policy"]["final_holdout_inputs"]

    assert "v2c5_final_holdout_access" in prohibited
    assert "future_final_holdout_creation" in prohibited
    assert final_inputs == [
        {
            "path": PROHIBITED_HOLDOUT_PATH,
            "status": "explicitly_prohibited_raw_input",
        },
        {
            "path": "any_future_v2c6_final_holdout",
            "status": "not_created_and_prohibited_for_this_analysis",
        },
    ]
    assert contract["contract_status"]["final_holdout_accessed"] is False
    assert contract["contract_status"]["final_holdout_created"] is False


def test_model_outputs_can_only_come_from_persisted_predictions(
    contract: dict[str, Any],
) -> None:
    policy = contract["input_policy"]

    assert policy["model_outputs_must_come_only_from_persisted_results"] is True
    assert policy["read_only_dataset_joins_permitted"] is True
    assert policy["dataset_join_fields"] == {
        "development": "example_id",
        "fresh_evaluation": "record_id",
    }
    assert policy["text_inspection"][
        "limited_to_prediction_linked_frozen_development_or_fresh_records"
    ] is True
    assert policy["text_inspection"]["tracked_raw_text_output_permitted"] is False


def test_tracked_output_paths_and_schemas_are_new_and_text_free(
    contract: dict[str, Any],
) -> None:
    artifacts = contract["execution_artifact_contract"]

    assert artifacts["tracked_outputs"] == [
        {
            "path": "data/evals/v2/ml/v2c6_targeted_remediation_failure_analysis.json",
            "schema_version": "v2c6-targeted-remediation-failure-analysis.v1",
        },
        {
            "path": (
                "data/evals/v2/ml/"
                "v2c6_targeted_remediation_failure_analysis.manifest.json"
            ),
            "schema_version": (
                "v2c6-targeted-remediation-failure-analysis-manifest.v1"
            ),
        },
    ]
    assert artifacts["raw_utterance_text_permitted_in_tracked_outputs"] is False
    assert artifacts["local_error_review_csv"]["authorized"] is False


def test_no_candidate_selection_or_step29i_authorization(
    contract: dict[str, Any],
) -> None:
    prohibited = set(contract["prohibited_operations"])

    assert "candidate_selection" in prohibited
    assert "step29i_authorization" in prohibited
    assert contract["contract_status"]["candidate_selected"] is False
    assert contract["contract_status"]["step29i_authorized"] is False
    assert contract["contract_status"]["final_model_acceptance_claimed"] is False
    assert contract["motivation"]["step29i_blocked"] is True


def test_continuation_requires_a_separate_future_contract(
    contract: dict[str, Any],
) -> None:
    continuation = contract["continuation_policy"]

    assert contract["next_required"] is None
    assert contract["contract_status"]["next_required"] is None
    assert continuation["post_analysis_next_required"] is None
    assert continuation["remediation_choice_pre_authorized"] is False
    assert continuation["separate_frozen_continuation_design_required"] is True
    assert continuation["step29i_authorized"] is False


def test_source_identity_mismatch_must_fail_closed(
    contract: dict[str, Any],
) -> None:
    validation = contract["source_artifact_validation"]

    assert validation["analysis_may_begin_before_validation"] is False
    assert validation["hash_mismatch_behavior"] == "fail_closed"
    assert {
        "path_identity",
        "sha256_identity",
        "schema_version",
        "execution_status_completed",
        "exact_four_candidate_ids",
        "selection_status_no_acceptable_candidate",
        "eligible_candidate_count_zero",
        "selected_candidate_null",
        "winner_not_forced",
        "gates_not_weakened",
        "step29i_not_authorized",
        "next_required_null",
        "development_and_fresh_dataset_identity",
        "prediction_coverage_identity",
    } == set(validation["required_checks"])


def test_contract_contains_no_preauthored_findings(contract: dict[str, Any]) -> None:
    assert contract["contract_status"]["failure_analysis_executed"] is False
    assert contract["contract_status"]["remediation_chosen"] is False
    assert contract["motivation"]["causal_root_cause_proven"] is False
    assert "findings" not in contract
    assert "selected_remediation" not in contract
