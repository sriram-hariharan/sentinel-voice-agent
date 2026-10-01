from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "data/evals/v2/ml/v2c6_post_selection_failure_analysis_contract.json"
)
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"

EXPECTED_CANDIDATES = [
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=balanced",
    "BGE_SMALL_LINEAR_SVC__C=1.0__class_weight=none",
    "BGE_SMALL_LINEAR_SVC__C=1.0__class_weight=balanced",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=balanced",
]
EXPECTED_PROTECTED_INTENTS = [
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


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_identity_phase_and_status(contract: dict[str, Any]) -> None:
    schema = "v2c6-post-selection-failure-analysis-contract.v1"

    assert contract["schema_version"] == schema
    assert contract["contract_version"] == schema
    assert contract["phase"] == "V2-C6 Step 29H-A"
    assert contract["contract_status"]["contract_frozen"] is True
    assert contract["contract_status"]["failure_analysis_executed"] is False
    assert contract["contract_status"]["next_required"] == (
        "v2c6_post_selection_failure_analysis_execution"
    )


def test_required_source_artifact_hashes_are_exact(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]

    assert set(sources) == {
        "development_dataset",
        "step29f_freeze",
        "step29g_contract",
        "step29h_results",
        "step29h_results_manifest",
    }
    for source in sources.values():
        assert source["path"] != PROHIBITED_HOLDOUT_PATH
        assert sha256_file(ROOT / source["path"]) == source["sha256"]


def test_step29h_result_and_manifest_are_required(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    results_spec = sources["step29h_results"]
    manifest_spec = sources["step29h_results_manifest"]
    results = json.loads((ROOT / results_spec["path"]).read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / manifest_spec["path"]).read_text(encoding="utf-8")
    )

    assert results_spec["schema_version"] == results["schema_version"]
    assert manifest_spec["schema_version"] == manifest["schema_version"]
    assert manifest["results"]["sha256"] == results_spec["sha256"]
    assert manifest["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"


def test_step29f_dataset_and_freeze_identity_are_pinned(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    dataset = sources["development_dataset"]
    freeze_spec = sources["step29f_freeze"]
    freeze = json.loads((ROOT / freeze_spec["path"]).read_text(encoding="utf-8"))

    assert dataset == {
        "path": "data/evals/v2/ml/v2c6_remediated_development_dataset.json",
        "record_count": 9008,
        "sha256": (
            "d7f78d7a76799f47bfdc3c1291505d1b964b9d143245d2d353931e8cb4f4a493"
        ),
    }
    assert freeze_spec["sha256"] == (
        "7f71dda024e5a947c997977f09ed675fa1f924c38d30f332422aa2032ecbc544"
    )
    assert freeze["freeze_status"] == freeze_spec["expected_freeze_status"]
    assert freeze["frozen_dataset_sha256"] == dataset["sha256"]


def test_step29g_contract_identity_is_pinned(contract: dict[str, Any]) -> None:
    source = contract["source_artifacts"]["step29g_contract"]

    assert source["path"].endswith(
        "v2c6_source_aware_model_selection_contract.json"
    )
    assert source["sha256"] == (
        "de4566b7fdb097e78fcd23b8ffa4682c4757cc7c5901c3f9a49d7da28192b1ba"
    )
    assert source["schema_version"] == (
        "v2c6-source-aware-model-selection-contract.v1"
    )


def test_source_identity_mismatch_fails_closed(contract: dict[str, Any]) -> None:
    validation = contract["source_artifact_validation"]

    assert validation["analysis_may_begin_before_validation"] is False
    assert validation["hash_mismatch_behavior"] == "fail_closed"
    assert {
        "path_identity",
        "sha256_identity",
        "schema_version",
        "step29f_freeze_status",
        "step29h_execution_status",
        "step29h_candidate_count",
        "step29h_selection_status",
        "step29h_eligible_candidate_count",
    } == set(validation["required_checks"])


def test_no_acceptable_candidate_input_state_is_required(
    contract: dict[str, Any],
) -> None:
    motivation = contract["motivation"]["step29h_input_state"]
    source = contract["source_artifacts"]["step29h_results"]
    results = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))

    assert motivation == {
        "candidate_count": 6,
        "eligible_candidate_count": 0,
        "execution_status": "COMPLETED",
        "gates_weakened": False,
        "selected_candidate": None,
        "selection_status": "NO_ACCEPTABLE_CANDIDATE",
        "winner_forced": False,
    }
    assert source["expected_execution_status"] == "COMPLETED"
    assert source["expected_candidate_count"] == 6
    assert source["expected_eligible_candidate_count"] == 0
    assert source["expected_selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert source["expected_selected_candidate"] is None
    assert results["execution_status"] == "COMPLETED"
    assert results["candidate_count"] == 6
    assert results["selection"]["eligible_candidate_count"] == 0
    assert results["selection"]["selection_status"] == (
        "NO_ACCEPTABLE_CANDIDATE"
    )
    assert results["selection"]["selected_candidate"] is None


def test_all_six_candidates_are_required(contract: dict[str, Any]) -> None:
    coverage = contract["candidate_coverage"]
    source = contract["source_artifacts"]["step29h_results"]
    results = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))

    assert coverage["all_step29h_candidates_required"] is True
    assert coverage["candidate_count"] == 6
    assert coverage["candidate_ids"] == EXPECTED_CANDIDATES
    assert coverage["winner_may_be_declared"] is False
    assert [
        result["candidate_id"] for result in results["candidate_results"]
    ] == EXPECTED_CANDIDATES


def test_diagnostic_references_do_not_create_a_winner(
    contract: dict[str, Any],
) -> None:
    references = contract["candidate_coverage"]["diagnostic_references"]

    assert references[0]["candidate_id"] == EXPECTED_CANDIDATES[0]
    assert references[1]["candidate_id"] == EXPECTED_CANDIDATES[4]
    assert "ineligible and not selected" in references[1]["interpretation"]


def test_exact_analysis_questions_are_frozen(contract: dict[str, Any]) -> None:
    questions = contract["analysis_questions"]

    assert len(questions) == 12
    assert any("non-protected intents" in question for question in questions)
    assert any("source families" in question for question in questions)
    assert any("candidate families" in question for question in questions)
    assert any("unsupported examples" in question for question in questions)
    assert any("protected examples" in question for question in questions)
    assert any("hard-negative" in question for question in questions)
    assert any("representation-specific" in question for question in questions)


def test_protected_false_positive_analysis_is_exact(
    contract: dict[str, Any],
) -> None:
    analysis = contract["protected_false_positive_analysis"]

    assert analysis["protected_intents"] == EXPECTED_PROTECTED_INTENTS
    assert analysis["confusion_definition"] == (
        "gold_intent is non-protected and predicted_intent is protected"
    )
    assert analysis["required_scopes"] == [
        "pooled_group_aware_cv",
        "pooled_source_family_holdout",
        *EXPECTED_SOURCE_FAMILIES,
    ]


def test_protected_false_positive_integer_allowance_is_derived(
    contract: dict[str, Any],
) -> None:
    calculation = contract["protected_false_positive_analysis"][
        "allowed_count_calculation"
    ]

    assert calculation["threshold"] == 0.01
    assert calculation["comparison"] == "less_than_or_equal"
    assert calculation["maximum_passing_false_positive_count_formula"] == (
        "floor(non_protected_denominator * 1 / 100)"
    )
    assert calculation["excess_false_positive_count_formula"] == (
        "max(0, observed_protected_false_positive_count - "
        "maximum_passing_false_positive_count)"
    )


def test_protected_false_positive_matrix_fields_and_sorting(
    contract: dict[str, Any],
) -> None:
    analysis = contract["protected_false_positive_analysis"]

    assert analysis["matrix_fields"] == [
        "gold_intent",
        "predicted_protected_intent",
        "count",
        "rate_among_gold_intent",
        "candidate_id",
        "evaluation_scope",
        "source_family_id",
    ]
    assert contract["sorting_rules"]["confusion_pairs"] == [
        "count_descending",
        "gold_intent_lexical_ascending",
        "predicted_intent_lexical_ascending",
    ]
    assert analysis["tracked_raw_text_permitted"] is False


def test_all_source_family_protected_fp_summaries_are_required(
    contract: dict[str, Any],
) -> None:
    fields = contract["protected_false_positive_analysis"][
        "source_family_summary_fields"
    ]

    assert fields == [
        "non_protected_denominator",
        "protected_false_positive_count",
        "protected_false_positive_rate",
        "maximum_passing_false_positive_count",
        "excess_false_positive_count",
    ]


def test_unsupported_error_types_are_separated(contract: dict[str, Any]) -> None:
    analysis = contract["unsupported_error_analysis"]

    assert analysis["unsupported_to_protected_is_separate_safety_view"] is True
    assert analysis["unsupported_to_protected_definition"] == (
        "gold_intent is unsupported_or_uncertain and predicted_intent is protected"
    )
    assert analysis["unsupported_to_other_supported_definition"] == (
        "gold_intent is unsupported_or_uncertain and predicted_intent is supported "
        "and non-protected"
    )
    assert {
        "unsupported_to_protected_count",
        "unsupported_to_protected_rate",
        "unsupported_to_other_supported_count",
        "unsupported_to_other_supported_rate",
    }.issubset(analysis["summary_fields"])


def test_unsupported_analysis_covers_pooled_and_family_scopes(
    contract: dict[str, Any],
) -> None:
    analysis = contract["unsupported_error_analysis"]

    assert analysis["required_scopes"] == [
        "pooled_group_aware_cv",
        "pooled_source_family_holdout",
        "each_individual_source_family",
    ]
    assert "candidate_id" in analysis["breakdown_dimensions"]
    assert "source_family_id" in analysis["breakdown_dimensions"]


def test_protected_recall_uses_any_protected_prediction(
    contract: dict[str, Any],
) -> None:
    analysis = contract["protected_false_negative_analysis"]

    assert analysis["protected_recall_definition"] == (
        "count(gold_intent in protected_intents and predicted_intent in "
        "protected_intents) / count(gold_intent in protected_intents)"
    )
    assert analysis["aggregate_protected_recall_miss_definition"] == (
        "gold_intent is protected and predicted_intent is non-protected"
    )


def test_protected_exact_intent_error_is_not_recall_failure(
    contract: dict[str, Any],
) -> None:
    analysis = contract["protected_false_negative_analysis"]

    assert analysis[
        "exact_intent_protected_confusion_is_protected_recall_failure"
    ] is False
    assert analysis[
        "report_aggregate_recall_misses_separately_from_exact_intent_errors"
    ] is True
    assert "cancel_transfer to freeze_card" in analysis[
        "protected_to_different_protected_example"
    ]


def test_source_family_analysis_is_complete_and_nonjudgmental(
    contract: dict[str, Any],
) -> None:
    analysis = contract["source_family_analysis"]

    assert analysis["required_family_ids"] == EXPECTED_SOURCE_FAMILIES
    assert set(analysis["required_metrics_per_candidate_and_family"]) == {
        "protected_false_positive_count",
        "protected_false_positive_rate",
        "protected_recall",
        "unsupported_recall",
        "supported_to_unsupported_rate",
        "unsupported_to_supported_rate",
        "primary_8_macro_f1",
    }
    assert analysis["source_family_quality_judgment_permitted"] is False


def test_group_cv_source_shift_safety_deltas_are_required(
    contract: dict[str, Any],
) -> None:
    analysis = contract["group_cv_vs_source_shift_delta_analysis"]

    assert analysis["required_for_every_candidate"] is True
    assert analysis["delta_direction"] == (
        "pooled_source_family_metric_minus_pooled_group_cv_metric"
    )
    assert analysis["required_safety_deltas"] == [
        "protected_false_positive_rate",
        "protected_recall",
        "unsupported_recall",
    ]


def test_macro_f1_population_difference_is_explicit(
    contract: dict[str, Any],
) -> None:
    comparison = contract["group_cv_vs_source_shift_delta_analysis"][
        "macro_f1_comparison"
    ]

    assert comparison["group_cv_label_count"] == 16
    assert comparison["group_cv_record_count"] == 9008
    assert comparison["source_family_label_count"] == 8
    assert comparison["source_family_record_count"] == 810
    assert comparison["direct_unqualified_subtraction_permitted"] is False


def test_exact_class_weight_comparisons_are_required(
    contract: dict[str, Any],
) -> None:
    comparison = contract["class_weight_comparison"]

    assert comparison["delta_direction"] == (
        "balanced_candidate_minus_matched_unweighted_candidate"
    )
    assert len(comparison["matched_pairs"]) == 3
    assert comparison["universal_effect_claim_permitted"] is False
    assert set(comparison["pooled_source_family_delta_metrics"]) == {
        "protected_recall",
        "protected_false_positive_rate",
        "unsupported_recall",
        "supported_to_unsupported_rate",
        "primary_8_macro_f1",
        "worst_family_primary_8_macro_f1",
    }


def test_exact_bge_regularization_comparisons_are_required(
    contract: dict[str, Any],
) -> None:
    comparison = contract["regularization_comparison"]

    assert comparison["delta_direction"] == (
        "c1_candidate_minus_matched_c4_candidate"
    )
    assert comparison["matched_pairs"] == [
        {
            "c1_candidate_id": EXPECTED_CANDIDATES[2],
            "c4_candidate_id": EXPECTED_CANDIDATES[0],
        },
        {
            "c1_candidate_id": EXPECTED_CANDIDATES[3],
            "c4_candidate_id": EXPECTED_CANDIDATES[1],
        },
    ]


def test_ten_hard_negative_pairs_and_aggregations_are_preserved(
    contract: dict[str, Any],
) -> None:
    analysis = contract["hard_negative_analysis"]

    assert analysis["required_pairs"] == EXPECTED_HARD_NEGATIVE_PAIRS
    assert len(analysis["required_pairs"]) == 10
    assert analysis["aggregate_dimensions"] == [
        "across_candidates",
        "across_source_families",
    ]
    assert analysis["labels_may_be_modified"] is False
    assert analysis["new_gate_threshold_may_be_created"] is False


def test_hard_negative_output_fields_are_exact(contract: dict[str, Any]) -> None:
    assert contract["hard_negative_analysis"]["record_fields"] == [
        "pair",
        "source_family_id",
        "candidate_id",
        "record_count",
        "correct_count",
        "accuracy",
        "true_a_predicted_b_count",
        "true_b_predicted_a_count",
        "other_error_count",
    ]


def test_cross_candidate_and_cross_family_recurrence_is_required(
    contract: dict[str, Any],
) -> None:
    recurrence = contract["recurrence_analysis"]

    assert recurrence["cross_candidate_and_cross_family_required"] is True
    assert recurrence["grouping_key"] == [
        "gold_non_protected_intent",
        "predicted_protected_intent",
    ]
    assert {
        "candidate_count",
        "bge_candidate_count",
        "tfidf_candidate_count",
        "source_family_count",
        "counts_by_source_family",
        "observed_in_pooled_group_cv",
        "observed_in_source_family_evaluation",
    }.issubset(recurrence["output_fields"])
    assert recurrence["systematic_pass_fail_threshold_defined"] is False


def test_tracked_error_records_prohibit_raw_text(contract: dict[str, Any]) -> None:
    schema = contract["error_record_schema"]

    assert schema["tracked_raw_utterance_text_permitted"] is False
    assert "text" not in schema["tracked_error_record_allowed_fields"]
    assert contract["execution_artifact_contract"][
        "raw_utterance_text_permitted"
    ] is False


def test_optional_local_text_review_is_ignored_and_development_only(
    contract: dict[str, Any],
) -> None:
    review = contract["error_record_schema"]["optional_local_text_review"]
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert review["path"] == (
        "data/evals/v2/ml/local/v2c6_post_selection_error_review.csv"
    )
    assert review["tracked"] is False
    assert review["ignored_by_repository"] is True
    assert review["creation_is_optional_and_separate"] is True
    assert "text" in review["allowed_columns"]
    assert "data/evals/v2/ml/local/" in gitignore
    assert "frozen V2-C6 development dataset" in review[
        "provenance_statement_required"
    ]
    assert "no final-holdout text" in review["provenance_statement_required"]


def test_source_family_evidence_is_consumed(contract: dict[str, Any]) -> None:
    governance = contract["consumed_evidence_governance"]

    assert governance[
        "source_family_evidence_consumed_for_development_model_selection"
    ] is True
    assert governance[
        "inspection_additionally_consumes_source_family_evidence"
    ] is True
    assert governance["source_family_ids"] == EXPECTED_SOURCE_FAMILIES
    assert governance["source_family_rounds_are_final_acceptance_evidence"] is False


def test_same_source_families_cannot_be_called_fresh_after_remediation(
    contract: dict[str, Any],
) -> None:
    governance = contract["consumed_evidence_governance"]

    assert governance[
        "same_source_families_may_be_called_fresh_after_remediation"
    ] is False
    assert governance["post_remediation_reuse_permitted_only_as"] == [
        "diagnostic_regression_evidence",
        "previously_observed_development_evidence",
    ]


def test_future_fresh_source_generalization_evidence_is_required(
    contract: dict[str, Any],
) -> None:
    future = contract["consumed_evidence_governance"][
        "future_clean_source_generalization_evidence"
    ]

    assert future["fresh_evidence_required"] is True
    assert future["permitted_sources"] == [
        "new_independently_authored_source_family",
        "new_independently_authored_source_family_evaluation_set",
        "fresh_untouched_final_holdout",
    ]
    assert future["source_authored_in_this_contract_step"] is False


def test_no_model_data_taxonomy_or_runtime_operations_are_permitted(
    contract: dict[str, Any],
) -> None:
    prohibited = set(contract["prohibited_operations"])

    assert {
        "embedding_generation",
        "model_fitting",
        "classifier_inference",
        "threshold_tuning",
        "candidate_search",
        "hyperparameter_search",
        "candidate_reranking",
        "model_selection",
        "dataset_mutation",
        "label_mutation",
        "taxonomy_mutation",
        "runtime_change",
        "remediation_implementation",
    }.issubset(prohibited)


def test_final_holdout_access_and_authoring_are_prohibited(
    contract: dict[str, Any],
) -> None:
    prohibited = set(contract["prohibited_operations"])
    input_policy = contract["input_policy"]

    assert "final_holdout_access" in prohibited
    assert "future_final_holdout_authoring" in prohibited
    assert input_policy["development_evidence_only"] is True
    assert input_policy["prohibited_final_holdout_inputs"] == [
        {
            "path": PROHIBITED_HOLDOUT_PATH,
            "status": "explicitly_prohibited_raw_input",
        },
        {
            "path": "any_future_v2c6_final_holdout",
            "status": "not_authored_and_prohibited_for_this_step",
        },
    ]
    assert input_policy["future_v2c6_final_holdout_access_permitted"] is False
    assert input_policy["future_v2c6_final_holdout_authoring_permitted"] is False
    assert contract["contract_status"]["final_holdout_accessed"] is False


def test_step29i_remains_blocked(contract: dict[str, Any]) -> None:
    assert contract["motivation"]["step29i_blocked"] is True
    assert contract["contract_status"]["step29i_authorized"] is False
    assert contract["contract_status"]["next_required"] != (
        "v2c6_full_development_fit"
    )


def test_no_causal_root_cause_claim_is_required(
    contract: dict[str, Any],
) -> None:
    assert contract["motivation"]["root_cause_inferred"] is False
    assert contract["remediation_evidence_summary"][
        "root_cause_language_permitted_without_causal_proof"
    ] is False
    assert any(
        "cannot prove causal root causes" in limitation
        for limitation in contract["known_limitations"]
    )


def test_remediation_summary_is_evidence_only(contract: dict[str, Any]) -> None:
    summary = contract["remediation_evidence_summary"]

    assert summary["classification_is_remediation_selection"] is False
    assert summary["remediation_implementation_permitted"] is False
    assert summary["descriptive_classifications"] == [
        "targeted_data_boundary_evidence",
        "representation_change_evidence",
        "both",
        "insufficient_evidence",
    ]
    assert set(summary["required_sections"]) == {
        "top_protected_false_positive_boundaries",
        "top_unsupported_boundaries",
        "top_protected_recall_miss_boundaries",
        "cross_family_recurring_boundaries",
        "cross_candidate_recurring_boundaries",
        "hard_negative_weaknesses",
        "class_weight_observations",
        "regularization_observations",
        "group_cv_vs_source_shift_deltas",
    }


def test_no_new_safety_threshold_is_introduced(contract: dict[str, Any]) -> None:
    prohibited = set(contract["prohibited_operations"])
    fp = contract["protected_false_positive_analysis"]["allowed_count_calculation"]

    assert fp["threshold"] == 0.01
    assert fp["threshold_source"] == (
        "Step 29G protected_false_positive_rate safety gate"
    )
    assert "safety_gate_weakening" in prohibited
    assert "new_safety_threshold_creation" in prohibited
    assert contract["hard_negative_analysis"][
        "new_gate_threshold_may_be_created"
    ] is False


def test_architecture_decision_is_evidence_first_and_reversible(
    contract: dict[str, Any],
) -> None:
    decision = contract["architecture_decision"]

    assert decision["chosen"] == (
        "evidence_first_post_selection_failure_analysis_before_further_data_or_"
        "model_changes"
    )
    alternatives = {
        row["alternative"]
        for row in decision["alternatives_rejected_for_this_contract"]
    }
    assert alternatives == {
        "immediately_add_more_examples",
        "immediately_add_new_model_families",
        "weaken_safety_gates",
        "tune_decision_thresholds",
    }
    assert decision["reversibility"].startswith("High;")
