from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "data/evals/v2/ml/v2c5_intent_discovery_contract.json"
)


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_phase_and_step_16_execution_status_are_frozen() -> None:
    contract = load_contract()

    assert contract["phase"] == "V2-C5"
    assert contract["status"] == "frozen"
    assert contract["execution_status"] == {
        "classifier_training_performed": False,
        "clustering_performed": False,
        "contract_frozen": True,
        "discovery_corpus_built": False,
        "embeddings_generated": False,
        "final_evaluation_performed": False,
        "human_adjudication_performed": False,
        "taxonomy_changed": False,
        "umap_performed": False,
        "v2c5_holdout_created": False,
    }


def test_objective_is_discovery_without_a_presumed_taxonomy_change() -> None:
    contract = load_contract()
    rules = contract["taxonomy_change_rules"]
    limits = contract["hypotheses"]["v2c4_interpretation_limits"]

    assert "coherent latent user-goal groups" in contract["objective"]
    assert rules["fixed_target_intent_count"] is None
    assert rules["cluster_automatically_defines_intent"] is False
    assert rules["current_nine_intent_taxonomy_changed_in_step_16"] is False
    assert limits["establishes_unsupported_or_uncertain_must_be_split"] is False
    assert limits["establishes_specific_new_intent_taxonomy"] is False


def test_primary_population_and_reference_anchors_are_separate() -> None:
    contract = load_contract()
    eligibility = contract["source_eligibility"]
    primary = eligibility["primary_cluster_population"]
    anchors = eligibility["reference_anchor_population"]

    assert primary["current_v2_taxonomy_label"] == "unsupported_or_uncertain"
    assert primary["supported_intents_included"] is False
    assert primary["unique_normalized_text_only"] is True
    assert primary["population_role"] == "density_driving_members"
    assert anchors["density_driving_members"] is False
    assert anchors["eligible_use"] == "later_reference_or_contrast_only"
    assert len(anchors["known_supported_intents"]) == 8


def test_eligible_source_roles_are_explicit_and_development_only() -> None:
    contract = load_contract()
    sources = contract["source_eligibility"]["eligible_source_roles"]

    assert [source["record_id"] for source in sources] == [
        "internal_train",
        "internal_validation",
        "banking77_train",
        "clinc_finance_train",
        "clinc_finance_val",
    ]
    assert all(source["development_assignment_required"] for source in sources)
    assert {
        (source["source_id"], source["source_split"])
        for source in sources
    } == {
        ("sentinelvoice_v2c1_internal", "train"),
        ("sentinelvoice_v2c1_internal", "validation"),
        ("banking77", "train"),
        ("clinc150_oos", "train"),
        ("clinc150_oos", "val"),
    }
    assert {
        source["materialized_path"] for source in sources
    } == {"data/evals/v2/ml/v2c3_development_dataset.json"}


def test_evaluation_test_lockbox_and_cfpb_sources_are_prohibited() -> None:
    contract = load_contract()
    exclusions = contract["source_exclusions"]

    required = {
        "all_test_only_or_final_evaluation_sources",
        "banking77_test",
        "cfpb",
        "clinc_test",
        "consumed_v2c3_challenge",
        "consumed_v2c3_external_lockbox",
        "v2c3_internal_locked_test",
        "v2c4_final_safety_holdout",
        "v2c4_postmortem_review_examples",
        "v2c4_selection_probe",
        "v2c4_targeted_synthetic_augmentation",
    }
    assert set(exclusions) == required
    assert exclusions["cfpb"]["discovery_eligible"] is False
    assert exclusions["banking77_test"]["discovery_eligible"] is False
    assert exclusions["clinc_test"]["discovery_eligible"] is False
    assert exclusions["consumed_v2c3_external_lockbox"]["discovery_eligible"] is False
    assert exclusions["v2c4_selection_probe"]["discovery_eligible"] is False
    assert exclusions["v2c4_final_safety_holdout"]["discovery_eligible"] is False


def test_v2c4_augmentation_is_excluded_from_primary_discovery() -> None:
    contract = load_contract()
    augmentation = contract["source_exclusions"][
        "v2c4_targeted_synthetic_augmentation"
    ]

    assert augmentation["primary_discovery_eligible"] is False
    assert (
        augmentation[
            "future_secondary_diagnostic_use_requires_separate_governance"
        ]
        is True
    )
    assert "purpose-built" in augmentation["rationale"]


def test_exact_normalized_text_deduplication_prevents_density_weighting() -> None:
    contract = load_contract()
    deduplication = contract["deduplication"]

    assert deduplication["required_before_clustering"] is True
    assert deduplication["canonical_unit"] == "normalized_text_sha256"
    assert deduplication["vectors_per_unique_normalized_text"] == 1
    assert (
        deduplication["cross_dataset_duplicates_increase_clustering_density"]
        is False
    )
    assert deduplication["duplicate_occurrences_retained_as_metadata"] is True
    assert deduplication["source_precedence"] == [
        "sentinelvoice_v2c1_internal",
        "banking77",
        "clinc150_oos",
    ]


def test_embedding_representation_is_exactly_frozen() -> None:
    contract = load_contract()
    representation = contract["representation"]

    assert representation["model"] == "BAAI/bge-small-en-v1.5"
    assert representation["provider"] == "FastEmbed"
    assert representation["method"] == "passage_embed"
    assert representation["dimensions"] == 384
    assert representation["fine_tuning"] is False
    assert representation["external_api"] is False
    assert representation["l2_normalization_required"] is True
    assert representation["input_space"] == "original_384_dimensional_embedding_space"
    assert representation["umap_reduced_vectors_used_for_clustering"] is False
    assert representation["cache_reuse_policy"]["required_exact_matches"] == [
        "text_order",
        "normalized_text_hashes",
        "model_identifier",
        "embedding_method",
        "dimensions",
    ]


def test_primary_hdbscan_configuration_is_exactly_frozen() -> None:
    contract = load_contract()
    discovery = contract["primary_discovery_method"]

    assert discovery["algorithm"] == "HDBSCAN"
    assert discovery["parameters"] == {
        "allow_single_cluster": False,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
        "metric": "euclidean",
        "min_cluster_size": 30,
        "min_samples": 10,
    }
    assert discovery["existing_labels_supplied_as_clustering_inputs"] is False
    assert discovery["supporting_algorithms_in_primary_experiment"] == []
    assert discovery["excluded_supporting_algorithms"] == [
        "KMeans",
        "AgglomerativeClustering",
        "BERTopic",
        "LLM_clustering",
        "GraphRAG",
        "other_unfrozen_clustering_methods",
    ]
    assert (
        discovery["future_robustness_comparison_requires_pre_result_contract"]
        is True
    )


def test_sensitivity_grid_is_bounded_to_nine_configurations() -> None:
    contract = load_contract()
    sensitivity = contract["sensitivity_analysis"]
    grid = sensitivity["grid"]

    assert grid["min_cluster_size"] == [15, 30, 60]
    assert grid["min_samples"] == [5, 10, 15]
    assert len(grid["min_cluster_size"]) * len(grid["min_samples"]) == 9
    assert sensitivity["configuration_count"] == 9
    assert sensitivity["all_other_parameters_identical_to_primary"] is True
    assert sensitivity["primary_30_10_result_remains_primary"] is True
    assert (
        sensitivity["post_hoc_best_looking_configuration_selection_allowed"]
        is False
    )
    assert sensitivity["semantic_inspection_may_choose_parameters"] is False


def test_umap_is_visualization_only_with_a_fixed_seed() -> None:
    contract = load_contract()
    visualization = contract["visualization"]

    assert visualization == {
        "algorithm": "UMAP",
        "clustering_input": False,
        "hdbscan_parameter_selection_evidence": False,
        "random_state": 20260928,
        "role": "two_dimensional_visualization_only",
        "taxonomy_evidence_by_itself": False,
        "visual_separation_establishes_valid_intent": False,
    }


def test_labels_and_postmortem_categories_are_metadata_or_prompts_only() -> None:
    contract = load_contract()
    labels = contract["source_eligibility"]["label_metadata_policy"]
    diagnostic = contract["hypotheses"]["v2c4_diagnostic_input"]

    assert labels["native_external_labels_may_be_interpretation_metadata"] is True
    assert labels["native_external_labels_supplied_as_clustering_features"] is False
    assert (
        labels["current_sentinelvoice_labels_supplied_as_clustering_features"]
        is False
    )
    assert diagnostic["role"] == "hypothesis_prompts_only"
    assert diagnostic["diagnostic_categories_are_clustering_features"] is False
    assert diagnostic["diagnostic_categories_are_target_classes"] is False
    assert diagnostic["diagnostic_categories_may_select_hdbscan_parameters"] is False


def test_cluster_diagnostics_do_not_automatically_create_intents() -> None:
    contract = load_contract()
    diagnostics = contract["cluster_diagnostics"]

    assert diagnostics["automatic_intent_creation_gate"] is False
    assert diagnostics["semantic_quality_pass_fail_gate"] is False
    assert "cluster_count" in diagnostics["required_outputs"]
    assert "noise_or_unassigned_count" in diagnostics["required_outputs"]
    assert "sensitivity_run_consistency_diagnostics" in diagnostics["required_outputs"]
    assert diagnostics["membership_probability_semantics"] == (
        "HDBSCAN membership probability is not a semantic correctness probability."
    )


def test_human_adjudication_and_taxonomy_rules_are_mandatory() -> None:
    contract = load_contract()
    adjudication = contract["human_adjudication"]
    rules = contract["taxonomy_change_rules"]

    assert adjudication["mandatory_before_taxonomy_change"] is True
    assert adjudication["representative_example_review_required"] is True
    assert adjudication["boundary_example_review_required"] is True
    assert adjudication["noise_is_acceptable_result"] is True
    assert adjudication["allowed_decisions"] == [
        "map_to_existing_intent",
        "candidate_new_intent",
        "remain_unsupported",
        "mixed_or_incoherent",
        "needs_split_review",
        "noise_or_insufficient_evidence",
    ]
    assert len(adjudication["new_intent_criteria"]) == 7
    assert rules["human_adjudication_controls_changes"] is True
    assert rules["recognized_intents_automatically_create_runtime_tools"] is False


def test_protected_action_and_runtime_scope_remain_explicit() -> None:
    contract = load_contract()
    guardrails = contract["protected_action_guardrails"]
    runtime = contract["runtime_scope"]

    assert guardrails["protected_intents"] == [
        "create_dispute",
        "freeze_card",
    ]
    assert (
        guardrails["topic_similarity_automatically_implies_protected_intent"]
        is False
    )
    assert guardrails["explicit_current_action_or_request_semantics_required"] is True
    assert (
        guardrails[
            "new_protected_write_intent_may_be_created_by_clustering_alone"
        ]
        is False
    )
    assert runtime["classifier_output_role"] == "advisory_only"
    assert runtime["runtime_behavior_changed"] is False
    assert runtime["v1_tool_scope_frozen"] is True
    assert runtime["new_infrastructure_allowed"] is False
    assert "new_runtime_tool_merely_because_a_cluster_exists" in (
        runtime["prohibited_additions"]
    )


def test_holdout_policy_requires_new_post_taxonomy_holdout() -> None:
    contract = load_contract()
    holdout = contract["holdout_policy"]

    assert holdout["old_v2c4_holdout_is_v2c5_final_evidence"] is False
    assert holdout["new_independently_authored_v2c5_holdout_required"] is True
    assert holdout["step_16_creates_holdout"] is False
    assert holdout["creation_timing"] == (
        "after Step 20 taxonomy freeze and before supervised model selection"
    )
    assert "taxonomy_discovery" in holdout["excluded_uses_until_final_evaluation"]
    assert "cluster_analysis" in holdout["excluded_uses_until_final_evaluation"]


def test_future_steps_17_through_23_are_recorded_in_order() -> None:
    contract = load_contract()
    steps = contract["future_supervised_evaluation"]["steps"]

    assert list(steps) == ["17", "18", "19", "20", "21", "22", "23"]
    assert steps["17"] == "build_and_freeze_eligible_discovery_corpus"
    assert steps["18"] == "run_unsupervised_discovery"
    assert steps["19"] == "perform_human_taxonomy_adjudication"
    assert steps["20"] == "freeze_expanded_taxonomy"
    assert steps["21"] == (
        "build_expanded_supervised_dataset_and_new_final_holdout"
    )
    assert steps["22"] == "train_and_select_expanded_taxonomy_classifier"
    assert steps["23"] == (
        "perform_once_only_final_evaluation_and_final_taxonomy_freeze"
    )


def test_pinned_nonsealed_inputs_match() -> None:
    contract = load_contract()
    pinned_inputs = contract["reproducibility"]["pinned_inputs"]

    assert "v2c4_final_safety_holdout" not in pinned_inputs
    for specification in pinned_inputs.values():
        assert sha256_file(ROOT / specification["path"]) == specification["sha256"]


def test_contract_serialization_is_deterministic() -> None:
    contract = load_contract()
    expected = (
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    assert CONTRACT_PATH.read_bytes() == expected
