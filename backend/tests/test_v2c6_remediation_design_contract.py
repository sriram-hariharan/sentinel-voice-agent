from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_DIRECTORY = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_DIRECTORY / "v2c6_remediation_design_contract.json"
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"

PRIMARY_REMEDIATION_INTENTS = [
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]

PROTECTED_WRITE_INTENTS = [
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_identity_and_status_are_frozen(contract: dict[str, Any]) -> None:
    assert contract["schema_version"] == "v2c6-remediation-design-contract.v1"
    assert contract["contract_version"] == contract["schema_version"]
    assert contract["phase"] == "V2-C6 Step 29C"
    assert contract["contract_status"] == {
        "contract_frozen": True,
        "development_dataset_changed": False,
        "embeddings_generated": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "new_examples_authored": False,
        "new_v2c6_final_holdout_created": False,
        "next_required": "v2c6_remediation_dataset_contract",
        "remediation_started": False,
        "runtime_behavior_changed": False,
        "taxonomy_changed": False,
        "v2c5_raw_final_holdout_accessed": False,
    }


def test_diagnosis_lineage_hashes_are_exact(contract: dict[str, Any]) -> None:
    sources = contract["source_artifacts"]
    hashed_sources = {
        name: specification
        for name, specification in sources.items()
        if name != "expanded_development_dataset_declaration"
    }
    for specification in hashed_sources.values():
        assert specification["path"] != PROHIBITED_HOLDOUT_PATH
        assert sha256_file(ROOT / specification["path"]) == specification["sha256"]

    diagnosis = json.loads(
        (ROOT / sources["generalization_diagnosis"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / sources["generalization_diagnosis_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["result"]["sha256"] == sources["generalization_diagnosis"][
        "sha256"
    ]
    assert diagnosis["source_lineage"] == manifest["source_lineage"]
    assert diagnosis["source_lineage"]["diagnosis_contract_sha256"] == sources[
        "generalization_diagnosis_contract"
    ]["sha256"]


def test_development_dataset_hash_comes_from_frozen_manifest(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    declaration = sources["expanded_development_dataset_declaration"]
    manifest = json.loads(
        (ROOT / sources["expanded_development_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )

    assert declaration["hash_source"] == sources["expanded_development_manifest"][
        "path"
    ]
    assert declaration["path"] == manifest["dataset"]["path"]
    assert declaration["sha256"] == manifest["dataset"]["sha256"]
    assert manifest["counts"]["output_occurrences"] == 8198


def test_completed_step29b_diagnosis_is_required(contract: dict[str, Any]) -> None:
    specification = contract["source_artifacts"]["generalization_diagnosis"]
    diagnosis = json.loads(
        (ROOT / specification["path"]).read_text(encoding="utf-8")
    )

    assert diagnosis["schema_version"] == specification["schema_version"]
    assert diagnosis["diagnosis_completed"] is True
    assert diagnosis["next_required"] == "v2c6_remediation_design"
    assert diagnosis["governance"]["remediation_implemented"] is False


def test_consumed_v2c5_holdout_is_prohibited(contract: dict[str, Any]) -> None:
    policy = contract["input_policy"]
    assert policy["consumed_v2c5_final_holdout"] == {
        "path": PROHIBITED_HOLDOUT_PATH,
        "status": "explicitly_prohibited_raw_input",
    }
    assert policy["individual_v2c5_final_holdout_examples_permitted"] is False
    assert all(
        source["path"] != PROHIBITED_HOLDOUT_PATH
        for source in contract["source_artifacts"].values()
    )


def test_step29b_evidence_is_recorded_without_causal_claim(
    contract: dict[str, Any],
) -> None:
    evidence = contract["evidence_base"]
    gap = evidence["development_to_final_generalization_gap"]
    imbalance = evidence["class_imbalance"]

    assert gap["development_pooled_oof_macro_f1"] == 0.8850476526181243
    assert gap["final_macro_f1"] == 0.5632279946795209
    assert gap["gap_stated_by_step29c_specification"] == 0.3218196579386034
    assert gap["step29b_serialized_gap"] == 0.32181965793860345
    assert gap["final_new_seven_intent_macro_f1"] == 0.45253940739110227
    assert gap["causality_inferred_from_final_examples"] is False
    assert imbalance["unsupported_or_uncertain_count"] == 4769
    assert imbalance["unsupported_or_uncertain_share"] == 0.5817272505489144
    assert imbalance["causality_claimed"] is False
    assert evidence["evidence_interpretation"]["exact_root_cause_proven"] is False
    concentration = evidence["source_and_authoring_concentration"]
    shares = concentration["largest_source_revision_share_by_primary_intent"]
    assert shares["cancel_transfer"] == 1.0
    assert shares["create_dispute"] == 1.0
    assert shares["freeze_card"] == 1.0
    assert shares["transfer_failed_or_declined"] == 1.0
    assert shares["transfer_pending"] == 1.0
    assert shares["close_account"] == 0.9607843137254902
    assert concentration["causality_claimed"] is False


def test_duplicate_leakage_is_not_the_primary_diagnosis(
    contract: dict[str, Any],
) -> None:
    duplicates = contract["evidence_base"]["duplicate_and_group_structure"]

    assert duplicates["raw_record_count"] == 8198
    assert duplicates["unique_group_count"] == 7863
    assert duplicates["singleton_group_share"] == 0.9905888337784561
    assert duplicates["exact_duplicate_record_count"] == 0
    assert duplicates["normalized_duplicate_record_count"] == 2
    assert duplicates["cross_intent_exact_duplicate_conflict_cluster_count"] == 0
    assert (
        duplicates["cross_intent_normalized_duplicate_conflict_cluster_count"]
        == 0
    )
    assert duplicates["ordinary_duplicate_leakage_is_main_remediation_target"] is False


def test_remediation_priority_order_is_exact(contract: dict[str, Any]) -> None:
    priorities = contract["remediation_priorities"]

    assert [item["ordinal"] for item in priorities] == [1, 2, 3, 4, 5]
    assert [item["id"] for item in priorities] == [
        "improve_source_and_authoring_diversity",
        "improve_supported_versus_unsupported_boundaries",
        "introduce_source_aware_development_evaluation",
        "improve_measured_intent_boundaries",
        "reconsider_representation_or_classifier_family_only_if_still_warranted",
    ]


def test_unsupported_boundary_evidence_is_a_priority(
    contract: dict[str, Any],
) -> None:
    boundary = contract["evidence_base"]["unsupported_boundary"]

    assert boundary["development_oof_non_unsupported_to_unsupported_count"] == 450
    assert boundary["protected_write_to_unsupported_count"] == 29
    assert boundary["target_intent_to_unsupported"]["close_account"]["count"] == 10
    assert boundary["target_intent_to_unsupported"][
        "transfer_failed_or_declined"
    ]["count"] == 30
    assert contract["remediation_priorities"][1]["id"] == (
        "improve_supported_versus_unsupported_boundaries"
    )


def test_primary_remediation_intent_list_is_exact(contract: dict[str, Any]) -> None:
    assert contract["primary_remediation_intents"] == PRIMARY_REMEDIATION_INTENTS


def test_source_diversity_policy_has_a_measurable_minimum(
    contract: dict[str, Any],
) -> None:
    policy = contract["source_diversity_policy"]

    assert policy["minimum_independent_source_families_per_primary_intent"] == 3
    assert policy["primary_intent_single_source_revision_share_must_be_less_than"] == 1.0
    assert policy["row_count_alone_is_sufficient_diversity_evidence"] is False
    assert policy["required_reported_measures"] == [
        "raw_example_count",
        "unique_group_count",
        "unique_source_family_count",
        "source_family_concentration",
    ]
    assert policy["unmet_minimum_policy"] == (
        "builder_must_fail_or_explicitly_report_unmet_coverage"
    )


def test_class_balance_policy_is_controlled_and_non_destructive(
    contract: dict[str, Any],
) -> None:
    policy = contract["class_balance_policy"]

    assert policy["all_16_classes_forced_to_equal_size"] is False
    assert policy["destructive_large_scale_removal_authorized"] is False
    assert policy["downsampling_authorized_by_step29c"] is False
    assert policy["downsampling_requires_separate_frozen_and_evaluated_design"] is True
    assert "targeted_additions_to_minority_and_expanded_intents" in policy[
        "preferred_methods"
    ]


def test_new_data_must_be_independently_authored_and_auditable(
    contract: dict[str, Any],
) -> None:
    policy = contract["new_data_policy"]

    assert policy["independent_authoring_required"] is True
    assert policy["provenance_may_be_inferred_from_wording"] is False
    assert policy["required_metadata"] == [
        "source_family_id",
        "source_revision",
        "intent",
        "risk",
        "authoring_batch",
        "group_id",
    ]
    assert policy["exact_and_normalized_duplicate_checks_required"] is True
    assert policy["cross_intent_duplicate_conflicts_permitted"] is False


def test_future_dataset_quality_gates_are_frozen(contract: dict[str, Any]) -> None:
    gates = contract["data_quality_gates"]

    assert gates["accuracy_or_f1_targets_frozen_by_step29c"] is False
    assert gates["exact_duplicate_conflict_count_maximum"] == 0
    assert gates["normalized_cross_intent_duplicate_conflict_count_maximum"] == 0
    assert gates["explicit_source_family_metadata_required"] is True
    assert gates["explicit_group_id_required"] is True
    assert gates["primary_intent_minimum_source_family_requirement_enforced"] is True
    assert gates["protected_write_counts_reported"] is True
    assert gates["source_concentration_reported"] is True
    assert gates["v2c5_final_holdout_access_or_reuse_permitted"] is False
    assert gates["runtime_changes_permitted"] is False


def test_existing_data_and_holdout_paraphrases_are_prohibited(
    contract: dict[str, Any],
) -> None:
    assert contract["new_data_policy"]["texts_that_may_not_be_paraphrased"] == [
        "v2c5_final_holdout_examples",
        "v2c5_development_examples",
        "existing_hard_negatives",
    ]
    assert contract["fresh_v2c6_final_holdout_policy"][
        "paraphrases_of_v2c5_final_holdout_permitted"
    ] is False


def test_hard_negative_pairs_and_authoring_method_are_frozen(
    contract: dict[str, Any],
) -> None:
    design = contract["hard_negative_design"]
    pairs = {tuple(pair) for pair in design["required_pairs"]}
    expected_unsupported_pairs = {
        (intent, "unsupported_or_uncertain")
        for intent in PRIMARY_REMEDIATION_INTENTS
        if intent != "unsupported_or_uncertain"
    }

    assert expected_unsupported_pairs <= pairs
    assert ("cancel_transfer", "transfer_pending") in pairs
    assert ("transfer_failed_or_declined", "transfer_pending") in pairs
    assert ("account_blocked", "transfer_failed_or_declined") in pairs
    assert design["minimal_edits_of_existing_examples_permitted"] is False
    assert design["authoring_method"].startswith("independently_authored")


def test_boundary_specifications_are_required_without_taxonomy_rewrite(
    contract: dict[str, Any],
) -> None:
    requirement = contract["boundary_specification_requirement"]

    assert requirement["required_before_hard_negative_authoring"] is True
    assert requirement["required_fields_per_primary_intent"] == [
        "positive_intent_definition",
        "explicit_exclusions",
        "nearest_competing_intents",
        "ambiguous_cases",
        "required_clarifying_cases",
        "unsupported_boundary",
        "protected_action_implications",
    ]
    assert requirement["step29c_rewrites_taxonomy_definitions"] is False


def test_unsupported_subtype_is_non_runtime_metadata(
    contract: dict[str, Any],
) -> None:
    redesign = contract["unsupported_class_redesign"]

    assert redesign["label_deleted"] is False
    assert redesign["unsupported_subtype_metadata_allowed"] is True
    assert redesign["runtime_taxonomy_changed"] is False
    assert redesign["unsupported_subtype_role"] == (
        "development_and_evaluation_metadata_only_unless_a_future_taxonomy_"
        "contract_changes_it"
    )
    assert len(redesign["conceptual_buckets"]) == 5


def test_protected_write_set_and_safety_gates_remain_exact(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_write_policy"]
    safety = contract["safety_policy"]

    assert protected["intents"] == PROTECTED_WRITE_INTENTS
    assert protected["classifier_authorizes_protected_actions"] is False
    assert protected["runtime_authorization_and_confirmation_remain_deterministic"] is True
    assert safety["minimum_gates"] == {
        "exact_protected_write_recall": {"operator": ">=", "threshold": 0.8},
        "protected_write_false_positive_rate": {
            "operator": "<=",
            "threshold": 0.01,
        },
        "unsupported_or_uncertain_recall": {
            "operator": ">=",
            "threshold": 0.8,
        },
    }
    assert safety["thresholds_weakened"] is False


def test_group_cv_and_source_holdout_are_complementary(
    contract: dict[str, Any],
) -> None:
    evaluation = contract["source_aware_development_evaluation"]

    assert evaluation["views_are_complementary"] is True
    assert evaluation["group_aware_stratified_cv"] == {
        "group_field": "group_id",
        "required": True,
    }
    assert evaluation["source_holdout"]["required"] is True
    assert evaluation["source_holdout"][
        "hold_out_entire_source_or_authoring_families_where_possible"
    ] is True
    assert evaluation["source_holdout"]["is_final_v2c6_holdout"] is False
    assert evaluation["insufficient_source_family_policy"] == (
        "remediation_builder_must_improve_source_diversity_before_source_"
        "holdout_evaluation"
    )


def test_fresh_v2c6_final_holdout_is_required(contract: dict[str, Any]) -> None:
    policy = contract["fresh_v2c6_final_holdout_policy"]

    assert policy["required"] is True
    assert policy["consumed_v2c5_final_holdout_examples_reused"] is False
    assert policy["authoring_process"] == "new_independent_authoring_session_and_process"
    assert policy["isolated_from_remediation_development_data"] is True
    assert policy["usage"] == "once_only_final_evaluation"
    assert policy["same_or_stronger_governance_than_v2c5"] is True


def test_step29c_performs_no_model_taxonomy_data_or_runtime_change(
    contract: dict[str, Any],
) -> None:
    status = contract["contract_status"]
    model_policy = contract["model_policy"]

    assert model_policy["switch_model_now"] is False
    assert model_policy["model_or_hyperparameter_selected_by_step29c"] is False
    assert model_policy["bge_small_linearsvc_family_eligible_as_baseline"] is True
    assert status["model_selection_performed"] is False
    assert status["model_training_performed"] is False
    assert status["new_examples_authored"] is False
    assert status["development_dataset_changed"] is False
    assert status["taxonomy_changed"] is False
    assert contract["taxonomy_policy"] == {
        "current_runtime_intent_count": 16,
        "revision_prioritized_by_step29c": False,
        "revision_requires_later_human_reviewed_evidence_of_taxonomy_ambiguity": True,
    }
    assert status["runtime_behavior_changed"] is False
    assert status["remediation_started"] is False


def test_future_phase_sequence_and_next_step_are_exact(
    contract: dict[str, Any],
) -> None:
    assert [phase["step"] for phase in contract["future_phases"]] == [
        "29D",
        "29E",
        "29F",
        "29G",
        "29H",
        "29I",
        "29J",
        "29K",
        "29L",
        "29M",
        "29N",
    ]
    assert contract["contract_status"]["next_required"] == (
        "v2c6_remediation_dataset_contract"
    )
