from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_ROOT / "v2c5_final_holdout_contract.json"
TAXONOMY_PATH = ML_ROOT / "v2c5_taxonomy_freeze.json"
TAXONOMY_MANIFEST_PATH = ML_ROOT / "v2c5_taxonomy_freeze.manifest.json"
DEVELOPMENT_PATH = ML_ROOT / "v2c5_expanded_development_dataset.json"
DEVELOPMENT_MANIFEST_PATH = (
    ML_ROOT / "v2c5_expanded_development_dataset.manifest.json"
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
EXPECTED_PROTECTED_WRITES = [
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def taxonomy() -> dict[str, Any]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def development_manifest() -> dict[str, Any]:
    return json.loads(DEVELOPMENT_MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(recursive_keys(child))
    return keys


def test_contract_schema_phase_and_frozen_status(
    contract: dict[str, Any],
) -> None:
    assert contract["schema_version"] == "v2c5-final-holdout-contract.v1"
    assert contract["contract_version"] == "v2c5-final-holdout-contract.v1"
    assert contract["phase"] == "V2-C5 Step 21C1"
    assert contract["contract_status"]["final_holdout_contract_frozen"] is True


def test_exact_frozen_16_label_taxonomy(
    contract: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    assert contract["taxonomy"]["intent_count"] == 16
    assert contract["taxonomy"]["intent_label_order"] == EXPECTED_INTENTS
    assert contract["taxonomy"]["intent_label_order"] == taxonomy[
        "intent_label_order"
    ]
    assert taxonomy["final_taxonomy_frozen"] is True


def test_exact_40_per_intent_and_640_total(
    contract: dict[str, Any],
) -> None:
    design = contract["holdout_design"]

    assert design["examples_per_intent"] == 40
    assert design["intent_count"] == 16
    assert design["total_example_count"] == 640
    assert design["total_example_count"] == (
        design["examples_per_intent"] * design["intent_count"]
    )
    assert design["class_balanced"] is True
    assert design["mirrors_step21b_development_distribution"] is False


def test_protected_and_non_protected_counts_are_frozen(
    contract: dict[str, Any],
) -> None:
    design = contract["holdout_design"]

    assert design["protected_write_example_count"] == 160
    assert design["non_protected_example_count"] == 480
    assert design["protected_write_example_count"] == (
        40 * len(EXPECTED_PROTECTED_WRITES)
    )
    assert design["non_protected_example_count"] == 640 - 160


def test_step20_hash_pins_are_exact(contract: dict[str, Any]) -> None:
    sources = contract["source_artifacts"]

    assert sources["taxonomy_freeze"]["sha256"] == sha256_file(TAXONOMY_PATH)
    assert sources["taxonomy_freeze_manifest"]["sha256"] == sha256_file(
        TAXONOMY_MANIFEST_PATH
    )
    assert sources["taxonomy_freeze"]["sha256"] == (
        "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
    )
    assert sources["taxonomy_freeze_manifest"]["sha256"] == (
        "359eb38e5ff63af0c9cc0f5db19951b8626f9e15cb136f23046648d655a0f7a9"
    )


def test_step21b_hash_pins_and_manifest_binding_are_exact(
    contract: dict[str, Any],
    development_manifest: dict[str, Any],
) -> None:
    sources = contract["source_artifacts"]

    assert sources["expanded_development_dataset"]["sha256"] == sha256_file(
        DEVELOPMENT_PATH
    )
    assert sources["expanded_development_manifest"]["sha256"] == sha256_file(
        DEVELOPMENT_MANIFEST_PATH
    )
    assert sources["expanded_development_dataset"]["sha256"] == (
        development_manifest["dataset"]["sha256"]
    )
    assert development_manifest["execution_status"]["step21_complete"] is False
    assert development_manifest["execution_status"]["step22_permitted"] is False


def test_risk_mapping_is_consumed_from_step20(
    contract: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    frozen = contract["taxonomy"]

    assert frozen["risk_mapping_source"] == (
        "data/evals/v2/ml/v2c5_taxonomy_freeze.json"
    )
    assert frozen["risk_by_intent"] == taxonomy["risk_by_intent"]
    assert set(frozen["risk_by_intent"]) == set(EXPECTED_INTENTS)


def test_exact_protected_write_set(
    contract: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    assert contract["taxonomy"]["protected_write_intents"] == (
        EXPECTED_PROTECTED_WRITES
    )
    assert contract["taxonomy"]["protected_write_intents"] == taxonomy[
        "protected_write_intents"
    ]


def test_old_v2c4_holdout_is_explicitly_prohibited(
    contract: dict[str, Any],
) -> None:
    assert "sealed_v2c4_final_holdout" in contract["prohibited_sources"]
    overlap = contract["overlap_policy"]
    prohibited = overlap["prohibited_overlap_inputs"]

    assert prohibited == [
        {
            "artifact": "sealed_v2c4_final_holdout",
            "path": "data/evals/v2/ml/v2c4_safety_holdout.json",
            "reason": (
                "The obsolete sealed holdout must remain unopened and cannot "
                "become V2-C5 overlap evidence."
            ),
        }
    ]
    assert overlap["sealed_v2c4_holdout_accessed"] is False
    assert overlap["v2c4_holdout_hash_or_contents_required_for_overlap_check"] is False


def test_external_test_and_consumed_evaluation_sources_are_prohibited(
    contract: dict[str, Any],
) -> None:
    prohibited = set(contract["prohibited_sources"])

    assert {
        "banking77_test",
        "clinc_test",
        "cfpb",
        "consumed_v2c3_challenge",
        "consumed_v2c3_external_lockbox",
        "v2c4_selection_probe",
        "v2c4_training_augmentation",
        "v2c4_postmortem_review_examples",
    }.issubset(prohibited)


def test_all_eligibility_flags_are_false_and_final_evaluation_is_single_use(
    contract: dict[str, Any],
) -> None:
    governance = contract["evaluation_governance"]
    false_flags = {
        "augmentation_source_eligible",
        "error_analysis_eligible_before_final_evaluation",
        "model_selection_eligible",
        "taxonomy_discovery_eligible",
        "threshold_selection_eligible",
        "training_eligible",
    }

    assert all(governance[field] is False for field in false_flags)
    assert governance["single_final_evaluation_only"] is True
    assert governance["step22_requires_created_and_frozen_final_holdout"] is True
    assert governance["final_evaluation_step"] == "V2-C5 Step 23"


def test_step22_cannot_access_or_learn_from_holdout(
    contract: dict[str, Any],
) -> None:
    governance = contract["evaluation_governance"]

    assert governance["step22_may_inspect_individual_holdout_examples"] is False
    assert governance["step22_may_run_holdout_inference"] is False
    assert governance["step22_may_tune_from_holdout"] is False
    assert governance["step22_may_use_holdout_errors"] is False


def test_step21_incomplete_and_step22_prohibited(
    contract: dict[str, Any],
) -> None:
    status = contract["contract_status"]

    assert status["step21_complete"] is False
    assert status["step22_permitted"] is False
    assert status["final_holdout_examples_created"] is False
    assert status["final_holdout_evaluated"] is False
    assert status["classifier_training_performed"] is False
    assert status["model_selection_performed"] is False
    assert status["runtime_behavior_changed"] is False
    assert status["next_required"] == (
        "independently_author_v2c5_final_holdout_examples"
    )


def test_independent_authorship_and_gold_label_rules_are_frozen(
    contract: dict[str, Any],
) -> None:
    authorship = contract["authorship_contract"]

    assert authorship["authorship_method"] == "independently_authored_synthetic"
    assert authorship["future_examples_authored_directly_for_known_gold_intent"] is True
    assert authorship["gold_label_determined_by_classifier_or_llm_prediction"] is False
    assert authorship["independent_from_development_examples"] is True
    assert authorship["raw_future_holdout_text_present"] is False


def test_required_qualitative_and_safety_boundaries_are_frozen(
    contract: dict[str, Any],
) -> None:
    coverage = contract["boundary_coverage"]

    assert coverage["each_intent_requires_mixed_formulations"] is True
    assert coverage["required_qualitative_tags"] == [
        "clear_direct",
        "natural_paraphrase_or_colloquial",
        "short_voice_style",
        "longer_contextual",
        "neighboring_intent_boundary_where_applicable",
    ]
    protected = coverage["protected_write_lane_requirements"]
    assert protected["explicit_current_action_semantics_required"] is True
    assert protected["topic_mention_alone_is_insufficient"] is True
    assert protected["required_request_variation"] == [
        "direct",
        "polite",
        "indirect",
    ]
    assert coverage["unsupported_or_uncertain_requirements"][
        "heterogeneous_out_of_scope_goals_required"
    ] is True


def test_future_overlap_policy_has_required_sources(
    contract: dict[str, Any],
) -> None:
    overlap = contract["overlap_policy"]
    artifacts = {
        value["artifact"]
        for value in overlap["required_exact_normalized_text_overlap_rejection"]
    }

    assert overlap["duplicate_normalized_text_within_holdout_rejected"] is True
    assert overlap["normalization_version"] == "unicode-nfkc-lower-whitespace.v1"
    assert overlap[
        "prefer_frozen_text_or_normalized_hashes_over_opening_source_data"
    ] is True
    assert {
        "v2c5_expanded_development_dataset",
        "consumed_v2c3_challenge",
        "consumed_v2c3_external_lockbox",
        "all_historical_development_and_training_text_used_in_v2c5",
        "future_v2c5_model_selection_probe",
    } == artifacts


def test_contract_contains_no_raw_holdout_examples(
    contract: dict[str, Any],
) -> None:
    assert contract["contract_status"]["final_holdout_examples_created"] is False
    assert contract["authorship_contract"]["raw_future_holdout_text_present"] is False
    assert {"example", "examples", "record", "records", "text", "utterance"}.isdisjoint(
        recursive_keys(contract)
    )
