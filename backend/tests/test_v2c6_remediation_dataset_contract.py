from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_ROOT / "v2c6_remediation_dataset_contract.json"
DESIGN_CONTRACT_PATH = ML_ROOT / "v2c6_remediation_design_contract.json"
DIAGNOSIS_PATH = ML_ROOT / "v2c6_generalization_diagnosis.json"
DIAGNOSIS_MANIFEST_PATH = ML_ROOT / "v2c6_generalization_diagnosis.manifest.json"
DIAGNOSIS_CONTRACT_PATH = ML_ROOT / "v2c6_generalization_diagnosis_contract.json"
TAXONOMY_PATH = ML_ROOT / "v2c5_taxonomy_freeze.json"
TAXONOMY_MANIFEST_PATH = ML_ROOT / "v2c5_taxonomy_freeze.manifest.json"

EXPECTED_PRIMARY_INTENTS = [
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
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
EXPECTED_UNSUPPORTED_SUBTYPES = [
    "truly_unsupported_banking_request",
    "ambiguous_or_insufficient_information",
    "adjacent_but_unsupported_intent",
    "supported_intent_hard_negative",
    "off_domain_or_noise",
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


@pytest.fixture(scope="module")
def taxonomy() -> dict[str, Any]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_schema_phase_and_contract_status(contract: dict[str, Any]) -> None:
    assert contract["schema_version"] == "v2c6-remediation-dataset-contract.v1"
    assert contract["contract_version"] == (
        "v2c6-remediation-dataset-contract.v1"
    )
    assert contract["phase"] == "V2-C6 Step 29D"
    assert contract["contract_status"]["contract_frozen"] is True


def test_expected_diagnosis_and_design_lineage_hashes(
    contract: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]

    assert sources["remediation_design_contract"]["sha256"] == sha256_file(
        DESIGN_CONTRACT_PATH
    )
    assert sources["generalization_diagnosis"]["sha256"] == sha256_file(
        DIAGNOSIS_PATH
    )
    assert sources["generalization_diagnosis_manifest"][
        "sha256"
    ] == sha256_file(DIAGNOSIS_MANIFEST_PATH)
    assert sources["generalization_diagnosis_contract"][
        "sha256"
    ] == sha256_file(DIAGNOSIS_CONTRACT_PATH)
    assert sources["remediation_design_contract"]["sha256"] == (
        "ffa8ec7f20b99cc22da3473dd50d32c7aa611cf548ecb205d5c53b9e5409a442"
    )
    assert sources["generalization_diagnosis"]["sha256"] == (
        "58177bd3aacf52bbff2268fd834e883457e8241f36e35f5e84943f196f688536"
    )
    assert sources["generalization_diagnosis_manifest"]["sha256"] == (
        "bb04f4bf82704331ab4cd0c1dc5ed3a9d87761a30013315e8c52055b3c550305"
    )
    assert sources["generalization_diagnosis_contract"]["sha256"] == (
        "4276c28eff6158ee947ac26466c7b65ac8830718accba8252de606a49b0493a4"
    )


def test_exact_primary_intent_set(contract: dict[str, Any]) -> None:
    assert contract["primary_remediation_intents"] == EXPECTED_PRIMARY_INTENTS
    assert contract["boundary_authoring_specification"][
        "required_for_intents"
    ] == EXPECTED_PRIMARY_INTENTS


def test_exact_protected_write_set(contract: dict[str, Any]) -> None:
    assert contract["protected_write_policy"]["intents"] == (
        EXPECTED_PROTECTED_WRITES
    )


def test_minimum_three_source_families(contract: dict[str, Any]) -> None:
    source_policy = contract["source_family_policy"]
    volume_policy = contract["planned_volume_policy"][
        "non_unsupported_primary_intents"
    ]

    assert source_policy[
        "minimum_independent_source_families_per_primary_intent"
    ] == 3
    assert volume_policy["minimum_source_families_per_intent"] == 3
    assert source_policy["explicit_source_family_id_required"] is True
    assert source_policy["explicit_independence_metadata_required"] is True


def test_required_record_metadata(contract: dict[str, Any]) -> None:
    assert contract["record_schema"]["required_fields"] == [
        "record_id",
        "text",
        "intent",
        "risk_level",
        "group_id",
        "source_family_id",
        "source_family_independence_basis",
        "source_revision",
        "authoring_batch_id",
        "authoring_method",
        "boundary_target",
        "is_hard_negative",
        "review_status",
    ]
    assert contract["record_schema"][
        "unsupported_additional_required_fields"
    ] == ["unsupported_subtype"]
    assert contract["record_schema"][
        "wording_based_metadata_inference_permitted"
    ] is False


def test_unsupported_subtype_set_is_exact(contract: dict[str, Any]) -> None:
    subtypes = contract["unsupported_subtypes"]

    assert subtypes["categories"] == EXPECTED_UNSUPPORTED_SUBTYPES
    assert subtypes["metadata_only"] is True
    assert subtypes["runtime_intent"] == "unsupported_or_uncertain"
    assert subtypes["runtime_intents_created"] is False


def test_hard_negative_pair_coverage(contract: dict[str, Any]) -> None:
    policy = contract["hard_negative_policy"]

    assert policy["required_pairs"] == EXPECTED_HARD_NEGATIVE_PAIRS
    assert policy["both_sides_independently_authored_where_appropriate"] is True
    assert policy["keyword_swap_construction_permitted"] is False


def test_independent_authoring_is_required(contract: dict[str, Any]) -> None:
    principle = contract["authoring_principle"]
    source_policy = contract["source_family_policy"]

    assert principle["independent_authoring_required"] is True
    assert principle["author_from"] == [
        "semantic_intent_definitions",
        "human_reviewed_boundary_specifications",
    ]
    assert "different_random_seeds_with_same_prompt_or_process" in (
        source_policy["not_independent_families"]
    )


def test_paraphrase_and_minimal_edit_policy(contract: dict[str, Any]) -> None:
    prohibited = set(contract["authoring_principle"]["prohibited_derivations"])

    assert {
        "paraphrase_of_existing_development_example",
        "minimal_edit_of_existing_development_example",
        "paraphrase_of_v2c5_final_holdout_example",
        "rewrite_of_v2c5_final_holdout_example",
        "translation_of_v2c5_final_holdout_example",
        "lexical_substitution_of_v2c5_final_holdout_example",
        "stylistic_variant_of_v2c5_final_holdout_example",
    } == prohibited


def test_group_semantics_rule(contract: dict[str, Any]) -> None:
    policy = contract["group_policy"]

    assert policy["default"] == "one_semantic_scenario_one_group"
    assert policy["variants_of_same_semantic_scenario_share_group_id"] is True
    assert policy[
        "paraphrases_may_receive_new_groups_to_inflate_independence"
    ] is False
    assert policy["supports_future_stratified_group_k_fold"] is True


def test_duplicate_normalization_policy(contract: dict[str, Any]) -> None:
    policy = contract["duplication_policy"]

    assert policy["normalization"] == {
        "collapse_whitespace": True,
        "lowercase": True,
        "trim": True,
        "unicode_normalization": "NFKC",
        "version": "unicode-nfkc-lower-whitespace.v1",
    }
    assert policy[
        "embedding_or_semantic_near_duplicate_threshold_authorized"
    ] is False


def test_zero_duplicate_and_conflict_gates(contract: dict[str, Any]) -> None:
    quality = contract["data_quality_gates"]
    duplication = contract["duplication_policy"]

    assert quality["zero_exact_duplicates"] is True
    assert quality["zero_normalized_duplicates_against_frozen_development"] is True
    assert quality["zero_normalized_cross_intent_conflicts"] is True
    assert duplication["zero_normalized_cross_intent_conflicts"] is True
    assert duplication["zero_exact_duplicates_against"] == [
        "v2c5_expanded_development_dataset",
        "new_v2c6_remediation_examples",
    ]


def test_planned_volume_policy(contract: dict[str, Any]) -> None:
    policy = contract["planned_volume_policy"]
    supported = policy["non_unsupported_primary_intents"]
    unsupported = policy["unsupported_or_uncertain"]

    assert supported == {
        "intent_count": 7,
        "minimum_records_per_intent": 90,
        "minimum_records_per_source_family": 30,
        "minimum_source_families_per_intent": 3,
        "planned_total": 630,
    }
    assert unsupported["planned_total"] == 180
    assert unsupported["all_five_subtypes_must_be_represented"] is True
    assert unsupported[
        "meaningful_supported_intent_hard_negative_representation_required"
    ] is True
    assert policy["planned_new_remediation_total"] == 810
    assert supported["planned_total"] + unsupported["planned_total"] == 810


def test_blind_class_equalization_is_prohibited(
    contract: dict[str, Any],
) -> None:
    policy = contract["planned_volume_policy"]

    assert policy["blind_class_equalization_permitted"] is False
    assert policy["planning_target_only"] is True
    assert policy[
        "thousands_of_generic_unsupported_records_permitted_without_evidence"
    ] is False


def test_all_new_examples_require_human_review(
    contract: dict[str, Any],
) -> None:
    review = contract["human_review_policy"]

    assert review["all_new_examples_require_human_review"] is True
    assert review["auditable_review_decisions_required"] is True
    assert review["only_status_eligible_for_inclusion"] == "approved"


def test_review_state_allowlist(contract: dict[str, Any]) -> None:
    assert contract["human_review_policy"]["review_status_allowlist"] == [
        "unreviewed",
        "approved",
        "rejected",
        "needs_revision",
    ]


def test_protected_and_hard_negative_review_is_mandatory(
    contract: dict[str, Any],
) -> None:
    mandatory = set(
        contract["human_review_policy"]["mandatory_review_categories"]
    )

    assert {
        "protected_write_examples",
        "hard_negatives",
        "ambiguous_or_insufficient_information_unsupported_examples",
        "examples_relabelled_during_adjudication",
    }.issubset(mandatory)


def test_model_in_the_loop_initial_authoring_is_prohibited(
    contract: dict[str, Any],
) -> None:
    policy = contract["authoring_method_policy"]

    assert policy[
        "initial_authoring_may_use_current_classifier_predictions"
    ] is False
    assert policy[
        "iterative_generation_until_current_classifier_is_right_or_wrong"
    ] is False
    assert policy[
        "later_model_based_hard_example_mining_requires_separate_frozen_contract"
    ] is True


def test_source_aware_evaluation_readiness(contract: dict[str, Any]) -> None:
    readiness = contract["source_aware_evaluation_readiness"]

    assert readiness["before_step"] == "V2-C6 Step 29G"
    assert readiness["minimum_source_families_per_primary_intent"] == 3
    assert readiness[
        "source_family_holdout_feasible_without_dropping_any_primary_intent"
    ] is True
    assert readiness["group_ids_do_not_cross_prohibited_split_boundaries"] is True
    assert readiness["source_family_distribution_reported_per_intent"] is True


def test_v2c5_final_holdout_is_prohibited(contract: dict[str, Any]) -> None:
    principle = contract["authoring_principle"]
    separation = contract["final_holdout_separation"]

    assert principle["v2c5_final_holdout_may_be_used"] is False
    assert separation["v2c5_final_holdout_role"] == (
        "consumed_historical_aggregate_evidence_only"
    )
    assert contract["data_quality_gates"]["no_v2c5_final_holdout_access"] is True
    assert contract["contract_status"][
        "v2c5_raw_final_holdout_accessed"
    ] is False


def test_remediation_data_cannot_be_future_final_holdout(
    contract: dict[str, Any],
) -> None:
    separation = contract["final_holdout_separation"]

    assert separation["remediation_data_role"] == "development_data_only"
    assert separation["remediation_examples_may_be_future_final_holdout"] is False
    assert separation[
        "remediation_examples_may_be_copied_into_future_final_holdout"
    ] is False
    assert separation["future_final_holdout_authoring_steps"] == [
        "V2-C6 Step 29J",
        "V2-C6 Step 29K",
    ]


def test_frozen_taxonomy_is_unchanged(
    contract: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    frozen = contract["taxonomy"]

    assert frozen["intent_count"] == 16
    assert frozen["intent_label_order"] == taxonomy["intent_label_order"]
    assert frozen["risk_by_intent"] == taxonomy["risk_by_intent"]
    assert frozen["taxonomy_changed"] is False
    assert contract["source_artifacts"]["taxonomy_freeze"][
        "sha256"
    ] == sha256_file(TAXONOMY_PATH)
    assert contract["source_artifacts"]["taxonomy_freeze_manifest"][
        "sha256"
    ] == sha256_file(TAXONOMY_MANIFEST_PATH)


def test_safety_thresholds_are_unchanged(contract: dict[str, Any]) -> None:
    safety = contract["safety_lineage"]

    assert safety["protected_write_false_positive_rate"] == {
        "operator": "<=",
        "threshold": 0.01,
    }
    assert safety["exact_protected_write_recall"] == {
        "operator": ">=",
        "threshold": 0.8,
    }
    assert safety["unsupported_or_uncertain_recall"] == {
        "operator": ">=",
        "threshold": 0.8,
    }
    assert safety["thresholds_changed_by_step29d"] is False


def test_no_model_work_or_runtime_change(contract: dict[str, Any]) -> None:
    status = contract["contract_status"]

    assert status["embeddings_generated"] is False
    assert status["model_training_performed"] is False
    assert status["model_selection_performed"] is False
    assert status["runtime_behavior_changed"] is False
    assert status["authoring_started"] is False
    assert status["examples_authored"] is False
    assert status["examples_reviewed"] is False
    assert status["remediation_dataset_built"] is False
    assert status["development_dataset_frozen"] is False


def test_future_output_paths_are_frozen_but_not_created(
    contract: dict[str, Any],
) -> None:
    outputs = contract["future_outputs"]
    tracked = outputs["tracked_outputs"]

    assert [item["path"] for item in tracked] == [
        "data/evals/v2/ml/v2c6_remediation_examples.json",
        "data/evals/v2/ml/v2c6_remediation_examples.manifest.json",
        "data/evals/v2/ml/v2c6_remediated_development_dataset.json",
        "data/evals/v2/ml/v2c6_remediated_development_dataset.manifest.json",
    ]
    assert all(item["create_in_step29d"] is False for item in tracked)
    assert outputs["authoring_workfile"] == {
        "create_in_step29d": False,
        "path": (
            "data/evals/v2/ml/local/"
            "v2c6_remediation_authoring_workfile.json"
        ),
        "tracked": False,
    }


def test_next_required_is_exact(contract: dict[str, Any]) -> None:
    assert contract["contract_status"]["next_required"] == (
        "v2c6_remediation_authoring_and_build"
    )
