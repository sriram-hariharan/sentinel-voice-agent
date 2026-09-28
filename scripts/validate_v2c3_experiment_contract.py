"""Validate the configuration-only V2-C3 governance and experiment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = (
    REPOSITORY_ROOT / "data/evals/v2/ml/v2c3_data_registry.json"
)
DEFAULT_CONTRACT_PATH = (
    REPOSITORY_ROOT / "data/evals/v2/ml/v2c3_experiment_contract.json"
)
DEFAULT_SOURCE_GOVERNANCE_PATH = (
    REPOSITORY_ROOT / "data/evals/v2/external/dataset_sources.json"
)

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
EXPECTED_RISKS = [
    "ESCALATION_OR_UNCERTAIN",
    "PRIVATE_READ",
    "PROTECTED_WRITE",
    "PUBLIC",
]
EXPECTED_RISK_BY_INTENT = {
    "account_balance": "PRIVATE_READ",
    "card_status": "PRIVATE_READ",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
EXPECTED_RECORD_IDS = {
    "banking77_test",
    "banking77_train",
    "bitext_retail_banking",
    "cfpb_frozen_1800",
    "cfpb_remaining_narratives",
    "clinc_finance_test",
    "clinc_finance_train",
    "clinc_finance_val",
    "clinc_oos_test",
    "clinc_oos_train",
    "clinc_oos_val",
    "internal_locked_test",
    "internal_train",
    "internal_validation",
    "unverified_bank_support_transcripts",
}
EXPECTED_TRAINING_ELIGIBLE = {
    "banking77_train",
    "clinc_finance_train",
    "clinc_finance_val",
    "internal_train",
    "internal_validation",
}
EXPECTED_LOCKBOX_ELIGIBLE = {
    "banking77_train",
    "clinc_finance_train",
    "clinc_finance_val",
}
EXPECTED_PROHIBITED = EXPECTED_RECORD_IDS - EXPECTED_TRAINING_ELIGIBLE
EXPECTED_REPRESENTATIONS = {
    "bge_small_en_v1_5_frozen",
    "char_tfidf",
    "tfidf_truncated_svd_lsa",
    "tuned_word_char_tfidf",
    "v2c1_word_char_tfidf",
    "word_tfidf",
}
EXPECTED_CLASSIFIERS = {
    "LinearSVC",
    "LogisticRegression",
    "RidgeClassifier",
    "SGDClassifier",
}
EXPECTED_TIE_BREAKS = [
    "lower false_supported_rate",
    "higher unsupported_or_uncertain_recall",
    "lower protected_write_false_positive_rate",
    "lower macro_f1_standard_deviation",
    "lower inference_p95",
    "smaller artifact_model_size",
    "simpler_reversible_architecture",
]
REQUIRED_RECORD_FIELDS = {
    "consumed_evaluation_status",
    "current_role",
    "fresh_lockbox_eligible",
    "group_leakage_policy",
    "license_or_terms",
    "mapping_requirement",
    "model_selection_eligible",
    "protected_action_restrictions",
    "provenance",
    "rationale",
    "record_id",
    "semantic_review_requirement",
    "source_id",
    "source_split",
    "source_type",
    "training_eligible",
}


class ContractValidationError(ValueError):
    """Raised when the frozen V2-C3 configuration is internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def stable_json_bytes(value: Any) -> bytes:
    """Return the canonical serialization used by config-focused tests."""
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _records_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = registry.get("records")
    _require(isinstance(records, list), "registry records must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        _require(isinstance(record, dict), "every registry record must be an object")
        missing = REQUIRED_RECORD_FIELDS - set(record)
        _require(not missing, f"registry record missing fields: {sorted(missing)}")
        record_id = record["record_id"]
        _require(isinstance(record_id, str), "record_id must be a string")
        _require(record_id not in result, f"duplicate registry record_id: {record_id}")
        result[record_id] = record
    _require(set(result) == EXPECTED_RECORD_IDS, "registry record set is not frozen")
    return result


def _validate_taxonomy(contract: dict[str, Any]) -> None:
    _require(contract.get("intent_taxonomy") == EXPECTED_INTENTS, "intent mismatch")
    _require(contract.get("risk_taxonomy") == EXPECTED_RISKS, "risk mismatch")
    _require(
        contract.get("risk_by_intent") == EXPECTED_RISK_BY_INTENT,
        "intent-to-risk mapping mismatch",
    )
    supported = contract.get("supported_intents")
    _require(
        supported == EXPECTED_INTENTS[:-1],
        "supported intents must be the eight non-unsupported intents",
    )


def _validate_internal_dataset_facts(registry: dict[str, Any]) -> None:
    internal = registry["canonical_references"]["internal_dataset"]
    _require(internal["example_count"] == 603, "internal example count changed")
    _require(internal["intent_class_count"] == 9, "internal class count changed")
    _require(internal["examples_per_intent"] == 67, "class balance changed")
    _require(internal["group_count"] == 261, "internal group count changed")
    _require(
        internal["split_counts"]
        == {"train": 405, "validation": 108, "locked_test": 90},
        "internal split counts changed",
    )


def _validate_data_roles(
    registry: dict[str, Any],
    records: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    training_eligible = {
        record_id
        for record_id, record in records.items()
        if record["training_eligible"]
    }
    model_selection_eligible = {
        record_id
        for record_id, record in records.items()
        if record["model_selection_eligible"]
    }
    lockbox_eligible = {
        record_id
        for record_id, record in records.items()
        if record["fresh_lockbox_eligible"]
    }
    _require(
        training_eligible == EXPECTED_TRAINING_ELIGIBLE,
        "training eligibility set is not frozen",
    )
    _require(
        model_selection_eligible == EXPECTED_TRAINING_ELIGIBLE,
        "model-selection eligibility set is not frozen",
    )
    _require(
        lockbox_eligible == EXPECTED_LOCKBOX_ELIGIBLE,
        "fresh-lockbox eligibility set is not frozen",
    )
    prohibited = set(contract["prohibited_data"]["record_ids"])
    _require(prohibited == EXPECTED_PROHIBITED, "prohibited record set mismatch")
    for record_id in EXPECTED_PROHIBITED:
        record = records[record_id]
        _require(not record["training_eligible"], f"{record_id} training leak")
        _require(
            not record["model_selection_eligible"],
            f"{record_id} model-selection leak",
        )
        _require(
            not record["fresh_lockbox_eligible"],
            f"{record_id} fresh-lockbox leak",
        )

    for record_id, record in records.items():
        if str(record["consumed_evaluation_status"]).startswith("consumed"):
            _require(not record["training_eligible"], f"consumed {record_id} leak")
            _require(
                not record["model_selection_eligible"],
                f"consumed {record_id} selection leak",
            )

    _require(registry.get("runtime_authority") is False, "registry grants authority")
    _require(
        registry.get("training_or_evaluation_performed") is False,
        "registry claims experiment execution",
    )


def _validate_mapping_restrictions(records: dict[str, dict[str, Any]]) -> None:
    for record_id in (
        "banking77_train",
        "clinc_finance_train",
        "clinc_finance_val",
    ):
        record = records[record_id]
        review = record["semantic_review_requirement"]
        _require(isinstance(review, dict), f"{record_id} review policy missing")
        _require(
            review.get("automatically_eligible_mapping_statuses")
            == ["EXACT_MATCH", "UNSUPPORTED"],
            f"{record_id} automatic mapping statuses changed",
        )
        _require(
            review.get("utterance_review_required_mapping_statuses")
            == ["NEAR_MATCH", "AMBIGUOUS"],
            f"{record_id} review-required mapping statuses changed",
        )
        protected = str(record["protected_action_restrictions"])
        _require(
            "explicit current action request" in protected,
            f"{record_id} protected-write restriction missing",
        )

    for record_id in ("clinc_oos_train", "clinc_oos_val"):
        rule = str(records[record_id]["semantic_review_requirement"])
        _require(
            "Do not automatically label as unsupported_or_uncertain" in rule,
            f"{record_id} is blindly mapped to unsupported",
        )


def _validate_lockbox(
    registry: dict[str, Any], contract: dict[str, Any]
) -> None:
    registry_policy = registry["fresh_lockbox_design"]
    policy = contract["fresh_lockbox_policy"]
    _require(policy["materialized"] is False, "lockbox must not be materialized")
    _require(policy["seed"] == 20260928, "lockbox seed mismatch")
    _require(contract["random_seed"] == 20260928, "contract seed mismatch")
    _require(policy["development_fraction"] == 0.8, "development split changed")
    _require(policy["fresh_lockbox_fraction"] == 0.2, "lockbox split changed")
    _require(
        policy["development_fraction"] + policy["fresh_lockbox_fraction"] == 1.0,
        "lockbox fractions must sum to one",
    )
    _require(
        set(policy["eligible_registry_record_ids"]) == EXPECTED_LOCKBOX_ELIGIBLE,
        "contract lockbox source set mismatch",
    )
    _require(
        set(registry_policy["eligible_record_ids"]) == EXPECTED_LOCKBOX_ELIGIBLE,
        "registry lockbox source set mismatch",
    )
    for key in (
        "confidence_score_may_influence_membership",
        "lockbox_allowed_for_cv",
        "lockbox_allowed_for_model_selection",
        "lockbox_allowed_for_threshold_selection",
        "model_output_may_influence_membership",
        "normalized_duplicate_groups_may_cross",
    ):
        _require(policy[key] is False, f"unsafe lockbox policy: {key}")
    _require(
        policy["split_before_any_training_or_selection"] is True,
        "lockbox must be split before development",
    )


def _validate_cross_validation(contract: dict[str, Any]) -> None:
    policy = contract["cross_validation"]
    _require(policy["method"] == "StratifiedGroupKFold", "CV method changed")
    _require(
        policy["default"]
        == {"n_splits": 5, "shuffle": True, "random_state": 20260928},
        "default CV policy changed",
    )
    fallback = policy["fallback"]
    _require(fallback["n_splits"] == 3, "fallback must use three folds")
    _require(
        fallback["decision_recorded_before_model_scores"] is True,
        "fallback decision timing is not frozen",
    )
    _require(
        fallback["model_scores_may_change_fold_count"] is False,
        "scores cannot change fold count",
    )
    _require(
        policy["ordinary_stratified_kfold_allowed_when_related_examples_exist"]
        is False,
        "ordinary StratifiedKFold would permit group leakage",
    )
    _require(
        policy["group_rules"]["normalized_duplicates_share_group"] is True,
        "normalized duplicates must share a group",
    )


def _validate_candidates_and_search(contract: dict[str, Any]) -> None:
    representations = {row["id"] for row in contract["representation_candidates"]}
    classifiers = {row["id"] for row in contract["classifier_candidates"]}
    _require(
        representations == EXPECTED_REPRESENTATIONS,
        "representation candidate set changed",
    )
    _require(classifiers == EXPECTED_CLASSIFIERS, "classifier set changed")
    _require(
        {row["id"] for row in contract["architecture_candidates"]}
        == {"direct_9_way", "hierarchical_supported_then_intent"},
        "architecture candidate set changed",
    )
    bge = next(
        row
        for row in contract["representation_candidates"]
        if row["id"] == "bge_small_en_v1_5_frozen"
    )
    _require(bge["model_identifier"] == "BAAI/bge-small-en-v1.5", "BGE changed")
    _require(bge["fine_tuning_allowed"] is False, "embedding tuning enabled")
    _require(
        bge["external_api_inference_allowed"] is False,
        "external embedding API enabled",
    )

    search = contract["bounded_search_spaces"]
    expected_c = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    _require(search["LinearSVC"]["C"] == expected_c, "LinearSVC C changed")
    _require(
        search["LogisticRegression"]["C"] == expected_c,
        "LogisticRegression C changed",
    )
    _require(
        search["LogisticRegression"]["max_iter"] == [2000],
        "LogisticRegression max_iter must remain bounded",
    )
    _require(
        search["SGDClassifier"]["alpha"] == [0.00001, 0.0001, 0.001],
        "SGD alpha changed",
    )
    _require(
        search["RidgeClassifier"]["alpha"] == [0.1, 1.0, 10.0],
        "Ridge alpha changed",
    )
    for model_name in EXPECTED_CLASSIFIERS:
        _require(
            search[model_name]["class_weight"] == [None, "balanced"],
            f"{model_name} class weights changed",
        )
    finalists = contract["finalist_hyperparameter_search"]
    _require(finalists["maximum_finalist_combinations"] == 3, "too many finalists")
    _require(finalists["fresh_lockbox_allowed"] is False, "lockbox search enabled")
    _require(
        finalists["consumed_external_test_results_allowed"] is False,
        "historical test results enabled for search",
    )


def _validate_metrics_and_selection(contract: dict[str, Any]) -> None:
    required_metrics = {
        "accuracy",
        "artifact_model_size",
        "confusion_matrix",
        "false_supported_rate",
        "fold_to_fold_variance",
        "inference_p50",
        "inference_p90",
        "inference_p95",
        "per_class_precision_recall_f1",
        "protected_write_false_positive_count",
        "protected_write_false_positive_rate",
        "protected_write_recall",
        "source_domain_breakdown",
        "supported_intent_accuracy",
        "unsupported_or_uncertain_recall",
        "weighted_f1",
    }
    _require(
        set(contract["metrics"]["required_raw_classifier_metrics"])
        == required_metrics,
        "required metric set changed",
    )
    gates = contract["safety_gates"]
    _require(
        gates["protected_write_false_positive_rate_maximum"] == 0.01,
        "protected-write false-positive gate changed",
    )
    _require(
        gates["protected_write_recall_minimum"] == 0.8,
        "protected-write recall gate changed",
    )
    _require(
        gates["unsupported_or_uncertain_recall_minimum"] == 0.8,
        "unsupported recall gate changed",
    )
    _require(gates["all_gates_required"] is True, "safety gates not mandatory")
    winner = contract["winner_selection"]
    _require(winner["primary_ranking"] == "highest macro_f1_mean", "ranking changed")
    _require(
        winner["practical_tie_macro_f1_difference_below"] == 0.005,
        "practical tie threshold changed",
    )
    _require(winner["tie_break_order"] == EXPECTED_TIE_BREAKS, "tie-break changed")
    tuning = contract["test_set_tuning_rules"]
    _require(
        all(value is False for value in tuning.values()),
        "a test or historical benchmark can influence selection",
    )


def _validate_source_governance(sources: dict[str, Any]) -> None:
    _require(
        sources.get("schema_version") == "external-dataset-sources.v3",
        "source governance schema is not V2-C3",
    )
    by_id = {row["source_id"]: row for row in sources["sources"]}
    banking = by_id["banking77"]
    clinc = by_id["clinc150_oos"]
    _require(banking["allowed_for_future_training"] is True, "BANKING77 denied")
    _require(
        banking["v2c3_training_policy"]["approved_splits"] == ["train.csv"],
        "BANKING77 split approval changed",
    )
    _require(
        banking["v2c3_training_policy"]["prohibited_splits"] == ["test.csv"],
        "BANKING77 test is not prohibited",
    )
    _require(clinc["allowed_for_future_training"] is True, "CLINC denied")
    _require(
        clinc["v2c3_training_policy"]["approved_splits"] == ["train", "val"],
        "CLINC split approval changed",
    )
    _require(
        set(clinc["v2c3_training_policy"]["prohibited_splits"])
        == {"test", "oos_train", "oos_val", "oos_test"},
        "CLINC prohibited splits changed",
    )
    for source_id in (
        "bitext_retail_banking",
        "cfpb_consumer_complaint_narratives_archive",
        "bank_of_america_support_transcripts",
        "jpmorgan_chase_support_transcripts",
        "wells_fargo_support_transcripts",
    ):
        _require(
            by_id[source_id]["allowed_for_future_training"] is False,
            f"{source_id} unexpectedly training-approved",
        )


def validate_contract(
    registry: dict[str, Any],
    contract: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, int]:
    _require(
        registry.get("schema_version") == "v2c3-data-registry.v1",
        "registry schema mismatch",
    )
    _require(
        contract.get("schema_version") == "v2c3-experiment-contract.v1",
        "contract schema mismatch",
    )
    _require(contract.get("experiment_phase") == "V2-C3", "phase mismatch")
    _require(contract.get("advisory_only") is True, "contract is not advisory")
    _require(contract.get("runtime_authority") is False, "runtime authority granted")
    _require(
        contract.get("training_or_evaluation_performed") is False,
        "contract claims experiment execution",
    )
    _require(
        contract["data_registry"]["schema_version"]
        == registry["schema_version"],
        "contract registry schema reference mismatch",
    )
    _require(
        contract["data_registry"]["registry_version"]
        == registry["registry_version"],
        "contract registry version reference mismatch",
    )

    records = _records_by_id(registry)
    _validate_taxonomy(contract)
    _validate_internal_dataset_facts(registry)
    _validate_data_roles(registry, records, contract)
    _validate_mapping_restrictions(records)
    _validate_lockbox(registry, contract)
    _validate_cross_validation(contract)
    _validate_candidates_and_search(contract)
    _validate_metrics_and_selection(contract)
    _validate_source_governance(sources)
    return {
        "registry_records": len(records),
        "training_eligible_records": len(EXPECTED_TRAINING_ELIGIBLE),
        "fresh_lockbox_source_records": len(EXPECTED_LOCKBOX_ELIGIBLE),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--source-governance",
        type=Path,
        default=DEFAULT_SOURCE_GOVERNANCE_PATH,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = validate_contract(
        load_json(args.registry),
        load_json(args.contract),
        load_json(args.source_governance),
    )
    print(
        "V2-C3 contract valid:",
        summary["registry_records"],
        "registry records;",
        summary["training_eligible_records"],
        "training-eligible records;",
        summary["fresh_lockbox_source_records"],
        "fresh-lockbox sources; no training or evaluation performed.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
