import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.evaluation.v2_contracts import IntentLabel, RiskLabel
from scripts import validate_v2c3_experiment_contract as validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "data/evals/v2/ml/v2c3_data_registry.json"
CONTRACT_PATH = REPOSITORY_ROOT / "data/evals/v2/ml/v2c3_experiment_contract.json"
SOURCE_GOVERNANCE_PATH = (
    REPOSITORY_ROOT / "data/evals/v2/external/dataset_sources.json"
)


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    return validator.load_json(REGISTRY_PATH)


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return validator.load_json(CONTRACT_PATH)


@pytest.fixture(scope="module")
def source_governance() -> dict[str, Any]:
    return validator.load_json(SOURCE_GOVERNANCE_PATH)


def _records(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["record_id"]: row for row in registry["records"]}


def test_validator_accepts_frozen_contract(
    registry: dict[str, Any],
    contract: dict[str, Any],
    source_governance: dict[str, Any],
) -> None:
    assert validator.validate_contract(registry, contract, source_governance) == {
        "fresh_lockbox_source_records": 3,
        "registry_records": 15,
        "training_eligible_records": 5,
    }


def test_config_serialization_is_deterministic(
    registry: dict[str, Any], contract: dict[str, Any]
) -> None:
    assert validator.stable_json_bytes(registry) == validator.stable_json_bytes(
        json.loads(validator.stable_json_bytes(registry))
    )
    assert validator.stable_json_bytes(contract) == validator.stable_json_bytes(
        json.loads(validator.stable_json_bytes(contract))
    )


def test_taxonomies_match_project_contract(contract: dict[str, Any]) -> None:
    assert contract["intent_taxonomy"] == sorted(label.value for label in IntentLabel)
    assert contract["risk_taxonomy"] == sorted(label.value for label in RiskLabel)
    assert len(contract["supported_intents"]) == 8
    assert "unsupported_or_uncertain" not in contract["supported_intents"]


def test_data_roles_and_no_holdout_contamination(
    registry: dict[str, Any], contract: dict[str, Any]
) -> None:
    records = _records(registry)
    training_eligible = {
        record_id
        for record_id, record in records.items()
        if record["training_eligible"]
    }

    assert training_eligible == validator.EXPECTED_TRAINING_ELIGIBLE
    assert set(contract["prohibited_data"]["record_ids"]) == (
        validator.EXPECTED_PROHIBITED
    )
    for record_id in contract["prohibited_data"]["record_ids"]:
        assert records[record_id]["training_eligible"] is False
        assert records[record_id]["model_selection_eligible"] is False


def test_internal_locked_test_is_permanently_excluded(
    registry: dict[str, Any], contract: dict[str, Any]
) -> None:
    locked = _records(registry)["internal_locked_test"]

    assert locked["source_split"] == "locked_test"
    assert locked["training_eligible"] is False
    assert locked["model_selection_eligible"] is False
    assert contract["test_set_tuning_rules"][
        "v2c1_locked_test_allowed_for_selection"
    ] is False


def test_cfpb_is_prohibited_in_full(registry: dict[str, Any]) -> None:
    records = _records(registry)

    for record_id in ("cfpb_frozen_1800", "cfpb_remaining_narratives"):
        assert records[record_id]["training_eligible"] is False
        assert records[record_id]["model_selection_eligible"] is False
        assert records[record_id]["fresh_lockbox_eligible"] is False
    assert "additional 2,000-record" in records["cfpb_remaining_narratives"][
        "provenance"
    ]


def test_banking77_and_clinc_split_boundaries(
    registry: dict[str, Any], source_governance: dict[str, Any]
) -> None:
    records = _records(registry)
    sources = {row["source_id"]: row for row in source_governance["sources"]}

    assert records["banking77_train"]["training_eligible"] is True
    assert records["banking77_test"]["training_eligible"] is False
    assert records["clinc_finance_train"]["training_eligible"] is True
    assert records["clinc_finance_val"]["training_eligible"] is True
    assert records["clinc_finance_test"]["training_eligible"] is False
    assert sources["banking77"]["v2c3_training_policy"]["approved_splits"] == [
        "train.csv"
    ]
    assert sources["clinc150_oos"]["v2c3_training_policy"][
        "approved_splits"
    ] == ["train", "val"]


@pytest.mark.parametrize(
    "record_id",
    ["banking77_train", "clinc_finance_train", "clinc_finance_val"],
)
def test_near_and_ambiguous_require_utterance_review(
    registry: dict[str, Any], record_id: str
) -> None:
    review = _records(registry)[record_id]["semantic_review_requirement"]

    assert review["automatically_eligible_mapping_statuses"] == [
        "EXACT_MATCH",
        "UNSUPPORTED",
    ]
    assert review["utterance_review_required_mapping_statuses"] == [
        "NEAR_MATCH",
        "AMBIGUOUS",
    ]


def test_taxonomy_mapping_cannot_create_protected_write(
    registry: dict[str, Any]
) -> None:
    records = _records(registry)

    for record_id in (
        "banking77_train",
        "clinc_finance_train",
        "clinc_finance_val",
    ):
        policy = records[record_id]["protected_action_restrictions"]
        assert "cannot assign freeze_card or create_dispute" in policy
        assert "explicit current action request" in policy


def test_clinc_oos_is_not_blindly_mapped_to_unsupported(
    registry: dict[str, Any], source_governance: dict[str, Any]
) -> None:
    records = _records(registry)
    sources = {row["source_id"]: row for row in source_governance["sources"]}

    for record_id in ("clinc_oos_train", "clinc_oos_val"):
        assert records[record_id]["training_eligible"] is False
        assert "Do not automatically label" in records[record_id][
            "semantic_review_requirement"
        ]
    assert sources["clinc150_oos"]["v2c3_training_policy"][
        "generic_oos_automatically_mapped_to_unsupported"
    ] is False


def test_fresh_lockbox_contract(contract: dict[str, Any]) -> None:
    policy = contract["fresh_lockbox_policy"]

    assert policy["materialized"] is False
    assert policy["development_fraction"] == 0.8
    assert policy["fresh_lockbox_fraction"] == 0.2
    assert policy["split_before_any_training_or_selection"] is True
    assert policy["lockbox_allowed_for_cv"] is False
    assert policy["lockbox_allowed_for_model_selection"] is False
    assert policy["model_output_may_influence_membership"] is False
    assert policy["confidence_score_may_influence_membership"] is False
    assert policy["normalized_duplicate_groups_may_cross"] is False


def test_group_aware_cross_validation_is_frozen(
    contract: dict[str, Any]
) -> None:
    cv = contract["cross_validation"]

    assert cv["method"] == "StratifiedGroupKFold"
    assert cv["default"] == {
        "n_splits": 5,
        "random_state": 20260928,
        "shuffle": True,
    }
    assert cv["fallback"]["n_splits"] == 3
    assert cv["fallback"]["decision_recorded_before_model_scores"] is True
    assert cv["fallback"]["model_scores_may_change_fold_count"] is False
    assert cv[
        "ordinary_stratified_kfold_allowed_when_related_examples_exist"
    ] is False


def test_candidate_models_representations_and_architectures_are_bounded(
    contract: dict[str, Any]
) -> None:
    assert {row["id"] for row in contract["representation_candidates"]} == (
        validator.EXPECTED_REPRESENTATIONS
    )
    assert {row["id"] for row in contract["classifier_candidates"]} == (
        validator.EXPECTED_CLASSIFIERS
    )
    assert {row["id"] for row in contract["architecture_candidates"]} == {
        "direct_9_way",
        "hierarchical_supported_then_intent",
    }
    assert contract["finalist_hyperparameter_search"][
        "maximum_finalist_combinations"
    ] == 3


def test_search_spaces_are_bounded(contract: dict[str, Any]) -> None:
    search = contract["bounded_search_spaces"]

    assert search["LinearSVC"]["C"] == [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    assert search["LogisticRegression"]["max_iter"] == [2000]
    assert search["SGDClassifier"]["alpha"] == [0.00001, 0.0001, 0.001]
    assert search["RidgeClassifier"]["alpha"] == [0.1, 1.0, 10.0]
    assert len(search["representations"]["word_ngram_range"]) == 3
    assert len(search["representations"]["lsa_component_count"]) == 3


def test_safety_gates_and_winner_order_are_frozen(
    contract: dict[str, Any]
) -> None:
    gates = contract["safety_gates"]
    selection = contract["winner_selection"]

    assert gates["protected_write_false_positive_rate_maximum"] == 0.01
    assert gates["protected_write_recall_minimum"] == 0.8
    assert gates["unsupported_or_uncertain_recall_minimum"] == 0.8
    assert gates["all_gates_required"] is True
    assert selection["primary_ranking"] == "highest macro_f1_mean"
    assert selection["practical_tie_macro_f1_difference_below"] == 0.005
    assert selection["tie_break_order"] == validator.EXPECTED_TIE_BREAKS


def test_invalid_holdout_eligibility_is_rejected(
    registry: dict[str, Any],
    contract: dict[str, Any],
    source_governance: dict[str, Any],
) -> None:
    changed = copy.deepcopy(registry)
    _records(changed)["internal_locked_test"]["training_eligible"] = True

    with pytest.raises(validator.ContractValidationError, match="training eligibility"):
        validator.validate_contract(changed, contract, source_governance)


def test_no_runtime_authority(contract: dict[str, Any]) -> None:
    assert contract["advisory_only"] is True
    assert contract["runtime_authority"] is False
    assert contract["training_or_evaluation_performed"] is False
    assert "Neither architecture can authenticate" in contract[
        "architecture_authority_boundary"
    ]


def test_validator_has_no_ml_execution_imports_or_calls() -> None:
    source_path = Path(validator.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imported_roots <= {"__future__", "argparse", "json", "pathlib", "typing"}
    assert ".fit(" not in source
    assert ".fit_transform(" not in source
    assert ".predict(" not in source
    assert "sklearn" not in imported_roots
