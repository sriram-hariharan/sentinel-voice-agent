from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_DIRECTORY = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_DIRECTORY / "v2c6_generalization_diagnosis_contract.json"
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"

EXPECTED_QUESTION_MEASURES = {
    "development_source_composition": [
        "examples_per_intent",
        "unique_groups_per_intent",
        "examples_per_group_distribution_by_intent",
        "source_dataset_distribution_by_intent_where_available",
        "source_role_distribution_by_intent_where_available",
        "provenance_distribution_by_intent_where_available",
        "synthetic_vs_external_or_source_category_distribution_where_represented",
    ],
    "group_structure": [
        "group_size_distribution",
        "multi_intent_group_count_and_rate",
        "intent_count_per_group_distribution",
        "dominant_intent_share_per_group_distribution",
        "examples_in_largest_1_5_10_and_25_groups",
        "largest_group_share_by_intent",
    ],
    "duplicate_and_near_duplicate_structure": [
        "exact_duplicate_count_and_rate",
        "normalized_duplicate_count_and_rate",
        "within_intent_exact_duplicate_concentration",
        "within_intent_normalized_duplicate_concentration",
        "cross_intent_exact_duplicate_conflict_count",
        "cross_intent_normalized_duplicate_conflict_count",
        "conflicting_intent_sets_without_raw_text",
    ],
    "lexical_diversity": [
        "character_length_distribution_by_intent",
        "token_length_distribution_by_intent",
        "vocabulary_size_and_unique_token_count_by_intent",
        "type_token_ratio_by_intent",
        "hapax_token_rate_by_intent",
        "unigram_bigram_and_trigram_concentration_by_intent",
        "top_10_and_top_25_ngram_token_coverage_by_intent",
        "intent_specific_token_concentration",
    ],
    "class_imbalance": [
        "raw_examples_per_intent",
        "relative_frequency_per_intent",
        "unsupported_or_uncertain_frequency_and_share",
        "maximum_to_minimum_intent_count_ratio",
        "maximum_to_median_intent_count_ratio",
        "unsupported_or_uncertain_to_each_intent_count_ratio",
    ],
    "development_oof_behavior": [
        "selected_candidate_per_intent_precision_recall_f1_and_support",
        "selected_candidate_frozen_order_confusion_matrix",
        "unsupported_or_uncertain_prediction_count_and_rate",
        "gold_protected_to_exact_other_protected_non_protected_and_unsupported_counts",
        "gold_non_protected_predicted_as_protected_count_and_rate",
        (
            "fold_accuracy_and_macro_f1_mean_standard_deviation_"
            "minimum_maximum_and_range_where_available"
        ),
    ],
    "authoring_and_provenance_concentration": [
        "unique_authoring_template_source_and_provenance_categories_by_intent_where_available",
        "largest_category_share_by_intent_where_available",
        "category_herfindahl_hirschman_index_by_intent_where_available",
        "raw_count_to_unique_provenance_category_ratio_by_intent_where_available",
        "missing_metadata_count_and_rate_for_each_concentration_field",
    ],
    "intent_boundary_diagnostics": [
        "pairwise_development_oof_confusion_counts_and_rates",
        "pairwise_vocabulary_jaccard_overlap",
        "pairwise_top_unigram_bigram_and_trigram_overlap",
        "pairwise_normalized_text_collision_counts",
        "protected_to_unsupported_and_unsupported_to_protected_confusion_counts",
    ],
    "effective_sample_size": [
        "raw_example_count_by_intent",
        "unique_group_count_by_intent",
        "unique_normalized_text_count_by_intent",
        "unique_group_to_raw_example_ratio_by_intent",
        "unique_normalized_text_to_raw_example_ratio_by_intent",
        "unsupported_or_uncertain_raw_group_and_normalized_text_counts",
    ],
}

EXPECTED_FOCUS_INTENTS = [
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "account_blocked",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_schema_phase_and_frozen_status(contract: dict[str, Any]) -> None:
    assert contract["schema_version"] == (
        "v2c6-generalization-diagnosis-contract.v1"
    )
    assert contract["contract_version"] == contract["schema_version"]
    assert contract["phase"] == "V2-C6 Step 29A"
    assert contract["contract_status"] == {
        "contract_frozen": True,
        "diagnosis_performed": False,
        "final_holdout_accessed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "next_required": "v2c6_generalization_diagnosis",
        "remediation_implemented": False,
        "runtime_behavior_changed": False,
        "taxonomy_changed": False,
    }


def test_allowed_source_artifacts_are_hash_pinned(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    assert set(sources) == {
        "expanded_development_dataset",
        "expanded_development_manifest",
        "final_evaluation_results",
        "final_evaluation_results_manifest",
        "final_evaluation_state",
        "model_selection_results",
        "model_selection_results_manifest",
        "taxonomy_freeze",
        "taxonomy_freeze_manifest",
    }
    for specification in sources.values():
        source_path = ROOT / specification["path"]
        assert specification["path"] != PROHIBITED_HOLDOUT_PATH
        assert sha256_file(source_path) == specification["sha256"]

    state = json.loads(
        (ROOT / sources["final_evaluation_state"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert sources["final_evaluation_state"]["expected_state"] == "completed"
    assert state["state"] == "completed"


def test_consumed_v2c5_holdout_is_explicitly_prohibited(
    contract: dict[str, Any],
) -> None:
    policy = contract["input_policy"]
    prohibited = contract["prohibited_during_diagnosis"]
    assert policy["consumed_v2c5_final_holdout"] == {
        "path": PROHIBITED_HOLDOUT_PATH,
        "status": "explicitly_prohibited_raw_input",
    }
    assert policy["individual_v2c5_final_holdout_examples_permitted"] is False
    assert set(prohibited) == {
        "add_v2c5_final_holdout_examples_to_development",
        "change_taxonomy",
        "create_new_classifier",
        "inspect_individual_v2c5_final_holdout_records",
        "model_or_hyperparameter_selection_using_final_holdout_performance",
        "paraphrase_v2c5_final_holdout_examples",
        "retroactively_change_v2c5_conclusions",
        "threshold_tuning_against_final_results",
        "train_against_v2c5_final_holdout_labels_or_examples",
        "treat_v2c5_final_holdout_as_reusable_test_set",
        "weaken_safety_gates",
    }
    assert set(prohibited.values()) == {True}


def test_only_committed_aggregate_final_context_is_allowed(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    assert sources["final_evaluation_results"]["access_scope"] == "aggregate_only"
    context = contract["final_result_context"]
    assert context == {
        "allowed_role": (
            "Historical aggregate evidence motivating development-side diagnosis "
            "only."
        ),
        "development_pooled_oof_macro_f1": 0.8850476526181243,
        "final_exact_protected_write_recall": 0.4,
        "final_macro_f1": 0.5632279946795209,
        "final_new_seven_intent_macro_f1": 0.45253940739110227,
        "final_protected_write_false_positive_rate": 0.014583333333333334,
        "final_unsupported_or_uncertain_recall": 0.85,
        "individual_final_examples_permitted": False,
        "selected_candidate_id": (
            "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none"
        ),
    }


def test_exact_nine_diagnosis_questions_are_frozen(
    contract: dict[str, Any],
) -> None:
    questions = contract["diagnosis_questions"]
    assert [question["ordinal"] for question in questions] == list(range(1, 10))
    assert [question["id"] for question in questions] == list(
        EXPECTED_QUESTION_MEASURES
    )
    assert {
        question["id"]: question["required_measures"] for question in questions
    } == EXPECTED_QUESTION_MEASURES


def test_duplicate_diagnosis_does_not_invent_semantic_threshold(
    contract: dict[str, Any],
) -> None:
    question = next(
        item
        for item in contract["diagnosis_questions"]
        if item["id"] == "duplicate_and_near_duplicate_structure"
    )
    assert question["semantic_near_duplicate_threshold"] is None
    assert "No semantic-similarity threshold" in question[
        "semantic_threshold_policy"
    ]


def test_intent_boundary_focus_is_exact_and_development_only(
    contract: dict[str, Any],
) -> None:
    question = next(
        item
        for item in contract["diagnosis_questions"]
        if item["id"] == "intent_boundary_diagnostics"
    )
    assert question["focus_intents"] == EXPECTED_FOCUS_INTENTS
    assert question["scope"].startswith("Development-side evidence only.")


def test_oof_diagnosis_uses_existing_outputs_without_model_work(
    contract: dict[str, Any],
) -> None:
    question = next(
        item
        for item in contract["diagnosis_questions"]
        if item["id"] == "development_oof_behavior"
    )
    assert question["input_policy"] == (
        "Use only already-generated V2-C5 model-selection outputs; do not rerun "
        "inference, train, or select a model."
    )
    status = contract["contract_status"]
    assert status["model_training_performed"] is False
    assert status["model_selection_performed"] is False
    assert contract["prohibited_during_diagnosis"]["create_new_classifier"] is True


def test_claim_strength_levels_and_root_cause_boundary(
    contract: dict[str, Any],
) -> None:
    policy = contract["claim_policy"]
    assert [level["name"] for level in policy["levels"]] == [
        "observation",
        "hypothesis",
        "supported_diagnosis",
        "causal_claim",
    ]
    assert policy["maximum_claim_authorized_by_step29b"] == "supported_diagnosis"
    assert policy[
        "measurable_development_side_evidence_required_for_supported_diagnosis"
    ] is True
    assert policy["root_cause_claim_from_aggregate_final_metrics_only_permitted"] is False
    assert policy["insufficient_alone_for_causal_or_root_cause_claim"] == [
        "class_imbalance_exists",
        "final_performance_dropped",
        "unsupported_or_uncertain_was_overpredicted",
    ]


def test_effective_sample_size_distinguishes_three_counts(
    contract: dict[str, Any],
) -> None:
    question = next(
        item
        for item in contract["diagnosis_questions"]
        if item["id"] == "effective_sample_size"
    )
    measures = question["required_measures"]
    assert "raw_example_count_by_intent" in measures
    assert "unique_group_count_by_intent" in measures
    assert "unique_normalized_text_count_by_intent" in measures
    assert "4,769 unsupported_or_uncertain rows" in question[
        "required_interpretation"
    ]


def test_taxonomy_and_safety_gates_cannot_change(
    contract: dict[str, Any],
) -> None:
    prohibited = contract["prohibited_during_diagnosis"]
    assert prohibited["change_taxonomy"] is True
    assert prohibited["weaken_safety_gates"] is True
    assert prohibited["retroactively_change_v2c5_conclusions"] is True
    assert contract["contract_status"]["taxonomy_changed"] is False


def test_future_outputs_are_aggregate_and_text_free(
    contract: dict[str, Any],
) -> None:
    outputs = contract["future_outputs"]
    assert outputs["generated_by_step29a"] is False
    assert outputs["diagnosis"] == {
        "path": "data/evals/v2/ml/v2c6_generalization_diagnosis.json",
        "preferred_content": "aggregate_and_statistical",
        "raw_final_holdout_text_permitted": False,
        "schema_version": "v2c6-generalization-diagnosis.v1",
    }
    assert outputs["manifest"] == {
        "path": "data/evals/v2/ml/v2c6_generalization_diagnosis.manifest.json",
        "raw_final_holdout_text_permitted": False,
        "schema_version": "v2c6-generalization-diagnosis-manifest.v1",
    }


def test_remediation_is_recommendation_only(contract: dict[str, Any]) -> None:
    assert contract["allowed_remediation_recommendations"] == [
        "improve_training_data_diversity",
        "rebalance_development_data",
        "improve_group_or_source_independence",
        "improve_intent_definitions_or_boundaries",
        "hard_negative_generation",
        "unsupported_class_redesign",
        "reconsider_representation_or_classifier_family",
        "taxonomy_revision",
    ]
    assert contract["remediation_boundary"] == {
        "diagnosis_may_recommend_categories_only": True,
        "remediation_implemented_by_step29a_or_step29b": False,
        "separately_frozen_v2c6_experiment_required": True,
    }


def test_fresh_future_holdout_is_mandatory(contract: dict[str, Any]) -> None:
    assert contract["fresh_holdout_requirement"] == {
        "consumed_v2c5_final_holdout_examples_reused": False,
        "fresh_independently_authored_holdout_required": True,
        (
            "fresh_holdout_must_be_frozen_before_v2c6_model_selection_or_"
            "training_decisions_that_depend_on_evaluation"
        ): True,
        "new_v2c6_evaluation_contract_required": True,
        "runtime_integration_blocked_until_future_candidate_passes_predeclared_safety_gates": True,
        "v2c5_aggregate_results_role": "historical_evidence_only",
    }


def test_runtime_is_unchanged_and_next_phase_is_diagnosis(
    contract: dict[str, Any],
) -> None:
    status = contract["contract_status"]
    assert status["runtime_behavior_changed"] is False
    assert status["next_required"] == "v2c6_generalization_diagnosis"
