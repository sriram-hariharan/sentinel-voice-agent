from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "data/evals/v2/ml/v2c4_intervention_plan.json"

EXPECTED_INTENTS = [
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
    "unsupported_or_uncertain",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def plan() -> dict[str, Any]:
    return load_json(PLAN_PATH)


def test_plan_is_step_9_only_and_not_executed(plan: dict[str, Any]) -> None:
    assert plan["schema_version"] == "v2c4-intervention-plan.v1"
    assert plan["experiment_phase"] == "V2-C4"
    assert plan["step"] == 9
    assert plan["intervention_status"] == "planned_not_executed"
    for field in (
        "training_performed",
        "augmentation_created",
        "selection_probe_created",
        "candidate_trained",
        "candidate_selected",
        "inference_performed",
        "tuning_performed",
        "final_evaluation_performed",
        "final_holdout_accessed",
        "runtime_integration_performed",
    ):
        assert plan[field] is False
    assert all(value is False for value in plan["step_9_prohibited_outputs"].values())


def test_parent_evidence_hashes_match_declared_files(plan: dict[str, Any]) -> None:
    assert plan["parent_git_commit"] == {
        "sha": "00ca4be",
        "subject": "Record V2-C4 consumed challenge error analysis",
    }
    for artifact in plan["parent_evidence"].values():
        assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"]


def test_current_taxonomy_remains_exactly_nine_intents(
    plan: dict[str, Any],
) -> None:
    taxonomy = plan["current_taxonomy"]
    assert taxonomy["intents"] == EXPECTED_INTENTS
    assert taxonomy["intent_count"] == 9
    assert taxonomy["frozen_for_v2c4"] is True
    assert taxonomy["expansion_allowed_in_v2c4"] is False
    assert taxonomy["intent_taxonomy_expansion_deferred_to"] == "V2-C5"
    assert "invalidate" in taxonomy["reason"]


def test_candidate_a_freezes_only_targeted_data_intervention(
    plan: dict[str, Any],
) -> None:
    candidate = plan["candidate_a"]
    assert candidate["status"] == "planned_not_trained"
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
    assert candidate["classifier"]["parameters"]["C"] == 4.0
    assert candidate["classifier"]["parameters"]["class_weight"] == "balanced"
    assert candidate["only_intended_intervention"] == (
        "targeted_development_training_data"
    )
    for field in (
        "threshold_tuning",
        "probability_calibration",
        "abstention_change",
        "embedding_fine_tuning",
    ):
        assert candidate[field] is False


def test_augmentation_scope_and_lineage_are_frozen(plan: dict[str, Any]) -> None:
    augmentation = plan["targeted_training_augmentation"]
    assert augmentation["status"] == "contract_only_not_created"
    assert augmentation["target_example_count"] == 360
    assert augmentation["independently_authored"] is True
    assert augmentation["protected_positives"]["example_count"] == 120
    assert augmentation["protected_positives"]["counts_by_intent"] == {
        "create_dispute": 60,
        "freeze_card": 60,
    }
    assert augmentation["matched_protected_action_hard_negatives"][
        "example_count"
    ] == 120
    assert augmentation["unsupported_or_scope_examples"]["example_count"] == 120
    lineage = augmentation["lineage_policy"]
    assert lineage["shared_group_lineage_for_near_related_examples"] is True
    assert lineage["group_aware_fold_leakage_prohibited"] is True
    restrictions = augmentation["authorship_and_source_restrictions"]
    assert restrictions["consumed_v2c3_safety_critical_error_count"] == 26
    assert restrictions["copy_consumed_v2c3_safety_critical_errors"] is False
    assert restrictions["paraphrase_consumed_v2c3_safety_critical_errors"] is False
    assert restrictions["v2c4_final_safety_holdout_used"] is False
    assert restrictions["cfpb_used"] is False


def test_selection_probe_is_independent_and_selection_only(
    plan: dict[str, Any],
) -> None:
    probe = plan["selection_probe"]
    assert probe["status"] == "step_10_contract_only_not_created"
    assert probe["creation_step"] == 10
    assert probe["target_example_count"] == 270
    assert probe["examples_per_intent"] == 30
    assert probe["counts_by_intent"] == dict.fromkeys(EXPECTED_INTENTS, 30)
    assert probe["independently_authored"] is True
    assert probe["training_eligible"] is False
    assert probe["model_selection_eligible"] is True
    assert probe["threshold_selection_eligible"] is False
    assert probe["final_acceptance_evidence"] is False
    assert all(probe["independence_requirements"].values())
    assert probe["expected_risk_structure"] == {
        "protected_examples": 60,
        "nonprotected_examples": 210,
        "unsupported_examples": 30,
    }


def test_development_gates_and_required_metrics_are_frozen(
    plan: dict[str, Any],
) -> None:
    evaluation = plan["development_evaluation"]
    assert evaluation["baseline_evaluated_on_same_selection_probe"] is True
    assert evaluation["safety_gates"] == {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "all_gates_required": True,
    }
    assert evaluation["required_reporting"] == [
        "macro_f1",
        "accuracy",
        "per_intent_recall",
        "false_supported_rate",
        "confusion_matrix",
        "protected_exact_intent_accuracy",
        "inference_latency",
        "artifact_size",
    ]
    assert evaluation["material_macro_f1_regression"][
        "absolute_drop_threshold"
    ] == 0.01


def test_candidate_b_is_conditional_and_architecture_is_predeclared(
    plan: dict[str, Any],
) -> None:
    trigger = plan["candidate_b_trigger"]
    assert trigger["automatic_build_allowed"] is False
    assert trigger["trigger_logic"] == "any"
    assert trigger["allowed_only_if"] == [
        "candidate_a_fails_any_development_safety_gate",
        "candidate_a_macro_f1_is_more_than_0_01_below_baseline",
    ]
    assert trigger["triggered_if_candidate_a_fails_any_safety_gate"] is True
    assert trigger["triggered_if_candidate_a_material_macro_f1_regression"] is True

    candidate = plan["candidate_b"]
    assert candidate["status"] == "predeclared_not_triggered_not_built"
    assert candidate["representation_same_as_candidate_a"] is True
    assert candidate["classifier_family"] == "LinearSVC"
    assert candidate["threshold_tuning"] is False
    assert candidate["transformer_used"] is False
    assert candidate["llm_router_used"] is False
    assert candidate["stages"] == [
        {
            "stage": 1,
            "input": "all_examples",
            "classes": ["supported_current_request", "unsupported_or_uncertain"],
        },
        {
            "stage": 2,
            "input": "supported_current_request",
            "classes": ["protected_write", "nonprotected"],
        },
        {
            "stage": 3,
            "branch": "protected_write",
            "classes": ["create_dispute", "freeze_card"],
        },
        {
            "stage": 3,
            "branch": "nonprotected",
            "classes": [
                "account_balance",
                "card_status",
                "escalation",
                "informational_policy",
                "recent_transactions",
                "transaction_details",
            ],
        },
    ]
    assert candidate["uses_same_training_corpus_as_candidate_a"] is True
    assert candidate["uses_same_selection_probe_as_candidate_a"] is True
    assert candidate["candidate_b_only_augmentation_allowed"] is False


def test_selection_rule_and_group_evaluation_are_frozen(
    plan: dict[str, Any],
) -> None:
    selection = plan["model_selection_rule"]
    assert selection["primary_evidence"] == "v2c4_development_selection_probe"
    assert selection["all_three_safety_gates_mandatory"] is True
    assert selection["primary_metric_when_multiple_candidates_pass"] == "macro_f1"
    assert selection["practical_macro_f1_tie_absolute_difference_less_than"] == 0.005
    assert selection["tie_break_order"] == [
        "lower_protected_write_false_positive_rate",
        "higher_protected_write_recall",
        "higher_unsupported_or_uncertain_recall",
        "lower_false_supported_rate",
        "lower_inference_latency",
        "smaller_artifact",
        "simpler_more_reversible_architecture",
    ]
    assert selection["candidate_a_passes_without_candidate_b_trigger"] == (
        "select_candidate_a_by_design"
    )

    supporting = plan["supporting_group_aware_evaluation"]
    assert supporting == {
        "role": "supporting_development_evidence",
        "primary_selection_evidence": False,
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260928,
        "group_field": "group_id",
        "target_field": "intent",
        "contrastive_or_near_related_group_leakage_allowed": False,
    }


def test_consumed_v2c3_data_cannot_select_or_tune_models(
    plan: dict[str, Any],
) -> None:
    for policy in plan["consumed_data_rules"].values():
        assert policy["training_text_allowed"] is False
        assert policy["model_selection_allowed"] is False
        assert policy["threshold_tuning_allowed"] is False
        assert policy["final_acceptance_evidence"] is False
        assert policy["use_after_selection_frozen"] == (
            "diagnostic_regression_evidence_only"
        )


def test_final_holdout_rule_is_one_shot_after_freeze(plan: dict[str, Any]) -> None:
    final = plan["final_model_and_holdout_rule"]
    assert final["selection_probe_excluded_from_fitting"] is True
    assert final["freeze_before_final_evaluation"] == [
        "architecture",
        "hyperparameters",
        "training_data_hashes",
        "classifier_artifact",
        "embedding_metadata",
        "evaluation_code",
    ]
    assert final["final_dataset_role"] == "sealed_v2c4_final_safety_holdout"
    assert final["final_evaluation_count"] == 1
    assert final["final_safety_gates"] == plan["development_evaluation"][
        "safety_gates"
    ]
    assert final["tuning_after_final_evaluation_allowed"] is False


def test_cfpb_and_other_test_sources_are_prohibited(plan: dict[str, Any]) -> None:
    prohibited = plan["prohibited_v2c4_development_inputs"]
    assert "sealed V2-C4 final safety holdout or seed" in prohibited
    assert "CFPB narratives, annotations, labels, or metadata-derived targets" in (
        prohibited
    )
    assert "BANKING77 test" in prohibited
    assert "CLINC test" in prohibited
    assert "CLINC oos_train" in prohibited
    assert "CLINC oos_val" in prohibited
    assert "CLINC oos_test" in prohibited


def test_v2c5_is_human_adjudicated_discovery_with_a_new_holdout(
    plan: dict[str, Any],
) -> None:
    roadmap = plan["v2c5_intent_discovery_and_taxonomy_expansion"]
    assert roadmap["roadmap_order"] == ["V2-C4", "V2-C5", "V2-D"]
    assert roadmap["primary_method"] == {
        "embeddings": "frozen_sentence_embeddings",
        "candidate_cluster_discovery": "HDBSCAN",
        "taxonomy_decision": "human_in_the_loop_adjudication",
    }
    assert roadmap["clustering_role"] == "advisory_candidate_group_proposal_only"
    assert roadmap["clustering_automatically_defines_intents"] is False
    assert roadmap["fixed_target_intent_count"] is None
    assert roadmap["eligible_sources"] == (
        "development_or_training_eligible_sources_only"
    )
    assert roadmap["cfpb_training_or_taxonomy_discovery_eligible"] is False
    assert roadmap["expanded_taxonomy_requires_new_fresh_final_holdout"] is True
    assert roadmap[
        "v2c4_nine_intent_holdout_valid_for_expanded_taxonomy_final_evidence"
    ] is False
    assert roadmap[
        "more_recognized_intents_automatically_imply_more_runtime_tools"
    ] is False
    assert roadmap["implementation_status"] == "roadmap_only_not_started"
