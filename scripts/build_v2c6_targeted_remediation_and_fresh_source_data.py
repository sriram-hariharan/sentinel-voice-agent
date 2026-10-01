"""Prepare and build V2-C6 targeted training and fresh-source data.

The workflow creates empty local authoring slots and validates completed human
or AI-assisted authoring. It never generates text, executes models, or accesses
a final holdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = (
    "scripts/build_v2c6_targeted_remediation_and_fresh_source_data.py"
)

CONTRACT_SHA256 = (
    "339bd20f4bb94e73a42bd297c053aef8deaf55ee42a3b2145b8792172e7d5e91"
)
FREEZE_SHA256 = (
    "7f71dda024e5a947c997977f09ed675fa1f924c38d30f332422aa2032ecbc544"
)
FROZEN_DEVELOPMENT_SHA256 = (
    "d7f78d7a76799f47bfdc3c1291505d1b964b9d143245d2d353931e8cb4f4a493"
)
NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"
BUILDER_VERSION = "v2c6-targeted-remediation-fresh-source-builder.v1"
WORKFLOW_PHASE = "V2-C6 targeted remediation and fresh-source authoring"

TRAINING_WORKFILE_SCHEMA = "v2c6-r2-targeted-training-authoring-workfile.v1"
EVALUATION_WORKFILE_SCHEMA = (
    "v2c6-r2-fresh-source-evaluation-authoring-workfile.v1"
)
TRAINING_OUTPUT_SCHEMA = "v2c6-targeted-remediation-training-examples.v1"
TRAINING_MANIFEST_SCHEMA = (
    "v2c6-targeted-remediation-training-examples-manifest.v1"
)
COMBINED_OUTPUT_SCHEMA = (
    "v2c6-targeted-remediated-development-dataset.v1"
)
COMBINED_MANIFEST_SCHEMA = (
    "v2c6-targeted-remediated-development-dataset-manifest.v1"
)
EVALUATION_OUTPUT_SCHEMA = "v2c6-fresh-source-evaluation-dataset.v1"
EVALUATION_MANIFEST_SCHEMA = (
    "v2c6-fresh-source-evaluation-dataset-manifest.v1"
)

TRAINING_ROLE = "targeted_remediation_training"
EVALUATION_ROLE = "fresh_source_evaluation"
UNSUPPORTED_INTENT = "unsupported_or_uncertain"

TARGETED_INTENTS = (
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    UNSUPPORTED_INTENT,
)
PROTECTED_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)
RISK_BY_INTENT = {
    "account_blocked": "PRIVATE_READ",
    "cancel_transfer": "PROTECTED_WRITE",
    "close_account": "PROTECTED_WRITE",
    "create_dispute": "PROTECTED_WRITE",
    "freeze_card": "PROTECTED_WRITE",
    "transfer_failed_or_declined": "PRIVATE_READ",
    "transfer_pending": "PRIVATE_READ",
    UNSUPPORTED_INTENT: "ESCALATION_OR_UNCERTAIN",
}
UNSUPPORTED_SUBTYPES = (
    "truly_unsupported_banking_request",
    "ambiguous_or_insufficient_information",
    "adjacent_but_unsupported_intent",
    "supported_intent_hard_negative",
    "off_domain_or_noise",
)
REVIEW_METHODS = ("human_review", "ai_assisted_review")
AUTHORING_METHODS = (
    "human_authored",
    "controlled_llm_assisted",
    (
        "deterministic_scenario_or_template_generation_when_semantic_"
        "independence_is_preserved"
    ),
)
REVIEW_FLAG_FIELDS = (
    "reviewer_disagreement",
    "low_confidence_review",
    "unresolved_ambiguity",
    "provenance_inconsistency",
    "protected_write_ambiguity",
)
PROHIBITED_PREDICTION_FIELDS = frozenset(
    {
        "candidate_id",
        "predicted_intent",
        "decision_score",
        "model_score",
        "classifier_score",
        "classifier_confidence",
    }
)
HARD_NEGATIVE_PAIRS = (
    ("close_account", UNSUPPORTED_INTENT),
    ("transfer_failed_or_declined", UNSUPPORTED_INTENT),
    ("account_blocked", UNSUPPORTED_INTENT),
    ("cancel_transfer", UNSUPPORTED_INTENT),
    ("create_dispute", UNSUPPORTED_INTENT),
    ("freeze_card", UNSUPPORTED_INTENT),
    ("transfer_pending", UNSUPPORTED_INTENT),
    ("cancel_transfer", "transfer_pending"),
    ("transfer_failed_or_declined", "transfer_pending"),
    ("account_blocked", "transfer_failed_or_declined"),
)
TIER_ONE_PAIRS = (
    (UNSUPPORTED_INTENT, "cancel_transfer"),
    (UNSUPPORTED_INTENT, "close_account"),
    (UNSUPPORTED_INTENT, "create_dispute"),
    (UNSUPPORTED_INTENT, "freeze_card"),
    (UNSUPPORTED_INTENT, "transfer_pending"),
    (UNSUPPORTED_INTENT, "transfer_failed_or_declined"),
)
TIER_TWO_PAIRS = (
    (UNSUPPORTED_INTENT, "account_blocked"),
    ("cancel_transfer", "transfer_pending"),
    ("transfer_failed_or_declined", "transfer_pending"),
    ("account_blocked", "transfer_failed_or_declined"),
)
CONSUMED_SOURCE_FAMILIES = frozenset(
    {
        "v2c6_sf1_definition_direct",
        "v2c6_sf2_scenario_narrative",
        "v2c6_sf3_boundary_conversational",
    }
)
GOVERNANCE_FLAGS = {
    "builder_generated_example_text": False,
    "dataset_mutated_before_build": False,
    "embeddings_generated": False,
    "final_holdout_accessed": False,
    "model_fitting_performed": False,
    "model_inference_performed": False,
    "model_selection_performed": False,
    "runtime_behavior_changed": False,
    "step29i_authorized": False,
    "taxonomy_mutated": False,
    "threshold_tuning_performed": False,
}

BytesReader = Callable[[Path], bytes]


@dataclass(frozen=True)
class BuildPaths:
    repository_root: Path
    contract: Path
    freeze: Path
    frozen_development: Path
    training_workfile: Path
    evaluation_workfile: Path
    training_output: Path
    training_manifest: Path
    combined_output: Path
    combined_manifest: Path
    evaluation_output: Path
    evaluation_manifest: Path
    prohibited_holdout: Path
    builder: Path


DEFAULT_PATHS = BuildPaths(
    repository_root=REPOSITORY_ROOT,
    contract=(
        ML_DIRECTORY / "v2c6_targeted_remediation_design_contract.json"
    ),
    freeze=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.freeze.json"
    ),
    frozen_development=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.json"
    ),
    training_workfile=(
        ML_DIRECTORY / "local/v2c6_r2_targeted_training_authoring.json"
    ),
    evaluation_workfile=(
        ML_DIRECTORY
        / "local/v2c6_r2_fresh_source_evaluation_authoring.json"
    ),
    training_output=(
        ML_DIRECTORY / "v2c6_targeted_remediation_training_examples.json"
    ),
    training_manifest=(
        ML_DIRECTORY
        / "v2c6_targeted_remediation_training_examples.manifest.json"
    ),
    combined_output=(
        ML_DIRECTORY / "v2c6_targeted_remediated_development_dataset.json"
    ),
    combined_manifest=(
        ML_DIRECTORY
        / "v2c6_targeted_remediated_development_dataset.manifest.json"
    ),
    evaluation_output=(
        ML_DIRECTORY / "v2c6_fresh_source_evaluation_dataset.json"
    ),
    evaluation_manifest=(
        ML_DIRECTORY
        / "v2c6_fresh_source_evaluation_dataset.manifest.json"
    ),
    prohibited_holdout=ML_DIRECTORY / "v2c5_final_holdout.json",
    builder=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
)


@dataclass(frozen=True)
class FrozenSources:
    contract: dict[str, Any]
    contract_sha256: str
    freeze: dict[str, Any]
    development: dict[str, Any]
    development_records: tuple[dict[str, Any], ...]
    historical_ids: frozenset[str]
    historical_group_ids: frozenset[str]


@dataclass(frozen=True)
class AuthoringInputs:
    training_payload: dict[str, Any]
    training_bytes: bytes
    training_records: tuple[dict[str, Any], ...]
    evaluation_payload: dict[str, Any]
    evaluation_bytes: bytes
    evaluation_records: tuple[dict[str, Any], ...]
    validation_report: dict[str, Any]


@dataclass(frozen=True)
class BuiltArtifacts:
    contents: tuple[bytes, ...]
    payloads: tuple[dict[str, Any], ...]


def stable_json_bytes(payload: Any) -> bytes:
    value = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return bytes(value + "\n", "utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def filesystem_reader(path: Path) -> bytes:
    return path.read_bytes()


def display_path(path: Path, paths: BuildPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def guard_final_holdout(path: Path, paths: BuildPaths) -> None:
    resolved = path.resolve()
    if resolved == paths.prohibited_holdout.resolve():
        raise PermissionError("final holdout access is prohibited")
    ml_directory = (paths.repository_root / "data/evals/v2/ml").resolve()
    if resolved.is_relative_to(ml_directory) and "final_holdout" in resolved.name:
        raise PermissionError("final holdout access is prohibited")


def guarded_read_bytes(
    path: Path,
    paths: BuildPaths,
    reader: BytesReader = filesystem_reader,
) -> bytes:
    guard_final_holdout(path, paths)
    return reader(path)


def read_json_object(
    path: Path,
    paths: BuildPaths,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    payload = json.loads(guarded_read_bytes(path, paths, reader))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def normalized_text_sha256(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("normalized text cannot be empty")
    return text_sha256(normalized)


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise TypeError(f"{label} must be a list of objects")
    return value


def _nonempty_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def output_paths(paths: BuildPaths) -> tuple[Path, ...]:
    return (
        paths.training_output,
        paths.training_manifest,
        paths.combined_output,
        paths.combined_manifest,
        paths.evaluation_output,
        paths.evaluation_manifest,
    )


def validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version")
        != "v2c6-targeted-remediation-design-contract.v1"
        or contract.get("phase") != "V2-C6 Step 29H-C"
        or contract.get("status") != "FROZEN"
        or contract.get("next_required")
        != "v2c6_targeted_remediation_and_fresh_source_authoring"
    ):
        raise ValueError("unexpected targeted remediation contract identity")
    status = contract.get("contract_status", {})
    if (
        status.get("contract_frozen") is not True
        or status.get("design_only") is not True
        or status.get("step29i_authorized") is not False
        or status.get("new_training_records_authored") is not False
        or status.get("new_evaluation_records_authored") is not False
    ):
        raise ValueError("targeted remediation contract status changed")
    if tuple(contract.get("targeted_intents", [])) != TARGETED_INTENTS:
        raise ValueError("targeted intent set changed")
    training = contract.get("training_remediation_specification", {})
    evaluation = contract.get("fresh_evaluation_specification", {})
    if (
        training.get("record_count") != 600
        or training.get("family_count") != 3
        or training.get("per_family_record_count") != 200
        or evaluation.get("record_count") != 640
        or evaluation.get("family_count") != 2
        or evaluation.get("per_family_record_count") != 320
        or evaluation.get("per_intent_per_family_count") != 40
    ):
        raise ValueError("frozen authoring volume changed")
    pairs = tuple(
        tuple(pair)
        for pair in training.get("hard_negative_boundaries", {}).get(
            "required_pairs", []
        )
    )
    if pairs != HARD_NEGATIVE_PAIRS:
        raise ValueError("frozen hard-negative pair set changed")
    if tuple(
        training.get("unsupported_authoring", {}).get(
            "subtype_vocabulary", []
        )
    ) != UNSUPPORTED_SUBTYPES:
        raise ValueError("unsupported subtype vocabulary changed")
    final_policy = contract.get("final_holdout_policy", {})
    if (
        final_policy.get("existing_final_holdout_access_permitted") is not False
        or final_policy.get("future_v2c6_final_holdout_access_permitted")
        is not False
        or final_policy.get("future_v2c6_final_holdout_authoring_permitted")
        is not False
    ):
        raise ValueError("final holdout prohibition changed")


def validate_freeze(
    freeze: Mapping[str, Any],
    contract: Mapping[str, Any],
    paths: BuildPaths,
) -> None:
    specification = contract["source_artifacts"]["step29f_freeze"]
    if (
        specification["path"] != display_path(paths.freeze, paths)
        or specification["sha256"] != FREEZE_SHA256
        or freeze.get("schema_version") != specification["schema_version"]
        or freeze.get("phase") != "V2-C6 Step 29F"
        or freeze.get("freeze_status") != "FROZEN"
        or freeze.get("frozen_dataset_path")
        != display_path(paths.frozen_development, paths)
        or freeze.get("frozen_dataset_sha256") != FROZEN_DEVELOPMENT_SHA256
        or freeze.get("frozen_dataset_record_count") != 9008
    ):
        raise ValueError("frozen 9008-record development identity changed")


def validate_frozen_development(
    payload: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    frozenset[str],
    frozenset[str],
]:
    records = _object_list(payload.get("examples"), "development examples")
    if (
        payload.get("schema_version")
        != "v2c6-remediated-development-dataset.v1"
        or payload.get("example_count") != 9008
        or len(records) != 9008
        or payload.get("normalization_version") != NORMALIZATION_VERSION
    ):
        raise ValueError("frozen development dataset changed")
    identifiers: set[str] = set()
    groups: set[str] = set()
    validated: list[dict[str, Any]] = []
    for record in records:
        example_id = _nonempty_string(record, "example_id")
        group_id = _nonempty_string(record, "group_id")
        text = _nonempty_string(record, "text")
        _nonempty_string(record, "intent")
        if example_id in identifiers:
            raise ValueError(f"duplicate frozen example ID: {example_id}")
        identifiers.add(example_id)
        groups.add(group_id)
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError(f"frozen text hash mismatch: {example_id}")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError(f"frozen normalized hash mismatch: {example_id}")
        validated.append(copy.deepcopy(record))
    return tuple(validated), frozenset(identifiers), frozenset(groups)


def load_frozen_sources(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> FrozenSources:
    contract_bytes = guarded_read_bytes(paths.contract, paths, reader)
    contract_sha = sha256_bytes(contract_bytes)
    if contract_sha != CONTRACT_SHA256:
        raise ValueError(
            f"targeted remediation contract SHA-256 mismatch: {contract_sha}"
        )
    contract = json.loads(contract_bytes)
    if not isinstance(contract, dict):
        raise TypeError("targeted remediation contract must be an object")
    validate_contract(contract)
    freeze_bytes = guarded_read_bytes(paths.freeze, paths, reader)
    freeze_sha = sha256_bytes(freeze_bytes)
    if freeze_sha != FREEZE_SHA256:
        raise ValueError(f"Step 29F freeze SHA-256 mismatch: {freeze_sha}")
    freeze = json.loads(freeze_bytes)
    if not isinstance(freeze, dict):
        raise TypeError("Step 29F freeze must be an object")
    validate_freeze(freeze, contract, paths)
    development_bytes = guarded_read_bytes(
        paths.frozen_development, paths, reader
    )
    development_sha = sha256_bytes(development_bytes)
    if development_sha != FROZEN_DEVELOPMENT_SHA256:
        raise ValueError(
            f"frozen development SHA-256 mismatch: {development_sha}"
        )
    development = json.loads(development_bytes)
    if not isinstance(development, dict):
        raise TypeError("frozen development dataset must be an object")
    records, identifiers, groups = validate_frozen_development(development)
    return FrozenSources(
        contract=contract,
        contract_sha256=contract_sha,
        freeze=freeze,
        development=development,
        development_records=records,
        historical_ids=identifiers,
        historical_group_ids=groups,
    )


def boundary_targets(intent: str, dataset_role: str) -> list[str]:
    if dataset_role == TRAINING_ROLE:
        plan = {
            "account_blocked": [UNSUPPORTED_INTENT] * 7
            + ["transfer_failed_or_declined"] * 13,
            "cancel_transfer": [UNSUPPORTED_INTENT] * 15
            + ["transfer_pending"] * 5,
            "close_account": [UNSUPPORTED_INTENT] * 20,
            "create_dispute": [UNSUPPORTED_INTENT] * 20,
            "freeze_card": [UNSUPPORTED_INTENT] * 20,
            "transfer_failed_or_declined": [UNSUPPORTED_INTENT] * 15
            + ["transfer_pending"] * 2
            + ["account_blocked"] * 3,
            "transfer_pending": [UNSUPPORTED_INTENT] * 15
            + ["cancel_transfer"] * 3
            + ["transfer_failed_or_declined"] * 2,
            UNSUPPORTED_INTENT: [
                target
                for target in (
                    "cancel_transfer",
                    "close_account",
                    "create_dispute",
                    "freeze_card",
                    "transfer_pending",
                    "transfer_failed_or_declined",
                )
                for _ in range(9)
            ]
            + ["account_blocked"] * 6,
        }
    elif dataset_role == EVALUATION_ROLE:
        plan = {
            "account_blocked": [UNSUPPORTED_INTENT] * 10
            + ["transfer_failed_or_declined"] * 30,
            "cancel_transfer": [UNSUPPORTED_INTENT] * 30
            + ["transfer_pending"] * 10,
            "close_account": [UNSUPPORTED_INTENT] * 40,
            "create_dispute": [UNSUPPORTED_INTENT] * 40,
            "freeze_card": [UNSUPPORTED_INTENT] * 40,
            "transfer_failed_or_declined": [UNSUPPORTED_INTENT] * 30
            + ["transfer_pending"] * 5
            + ["account_blocked"] * 5,
            "transfer_pending": [UNSUPPORTED_INTENT] * 30
            + ["cancel_transfer"] * 5
            + ["transfer_failed_or_declined"] * 5,
            UNSUPPORTED_INTENT: [
                target
                for target in (
                    "cancel_transfer",
                    "close_account",
                    "create_dispute",
                    "freeze_card",
                    "transfer_pending",
                    "transfer_failed_or_declined",
                )
                for _ in range(6)
            ]
            + ["account_blocked"] * 4,
        }
    else:
        raise ValueError(f"unknown dataset role: {dataset_role}")
    return plan[intent]


def family_independence_basis(
    family_id: str,
    strategy: str,
    dataset_role: str,
) -> str:
    return (
        f"independently structured {dataset_role} source family "
        f"using the frozen {strategy} authoring strategy ({family_id})"
    )


def authoring_slot(
    *,
    record_id: str,
    group_id: str,
    intent: str,
    family_id: str,
    dataset_role: str,
    strategy: str,
    boundary_target: str,
    unsupported_index: int,
) -> dict[str, Any]:
    is_evaluation = dataset_role == EVALUATION_ROLE
    subtype = (
        UNSUPPORTED_SUBTYPES[unsupported_index % len(UNSUPPORTED_SUBTYPES)]
        if intent == UNSUPPORTED_INTENT
        else None
    )
    return {
        "authoring_provenance": {
            "authoring_batch_id": f"{family_id}_batch_001",
            "authoring_method": "",
            "source_family_independence_basis": family_independence_basis(
                family_id, strategy, dataset_role
            ),
            "source_revision": "v1",
        },
        "authoring_strategy": strategy,
        "boundary_target": boundary_target,
        "candidate_outputs_inspected_during_authoring": False,
        "consumed_old_source_family_used_as_paraphrase_template": False,
        "dataset_role": dataset_role,
        "excluded_from_candidate_fitting": is_evaluation,
        "group_id": group_id,
        "human_adjudication_completed": False,
        "human_adjudicator": "",
        "independently_authored": True,
        "intent": intent,
        "is_hard_negative": True,
        "not_training_data": is_evaluation,
        "prediction_informed_authoring": False,
        "record_id": record_id,
        "requires_human_adjudication": False,
        "review_flags": {field: False for field in REVIEW_FLAG_FIELDS},
        "review_method": "",
        "review_notes": "",
        "review_status": "unreviewed",
        "reviewer": "",
        "risk_level": RISK_BY_INTENT[intent],
        "source_family_id": family_id,
        "text": "",
        "unsupported_subtype": subtype,
    }


def workfile_template(
    contract: Mapping[str, Any],
    *,
    dataset_role: str,
) -> dict[str, Any]:
    if dataset_role == TRAINING_ROLE:
        specification = contract["training_remediation_specification"]
        schema = TRAINING_WORKFILE_SCHEMA
    elif dataset_role == EVALUATION_ROLE:
        specification = contract["fresh_evaluation_specification"]
        schema = EVALUATION_WORKFILE_SCHEMA
    else:
        raise ValueError(f"unknown dataset role: {dataset_role}")
    records: list[dict[str, Any]] = []
    for family in specification["family_specifications"]:
        family_id = str(family["source_family_id"])
        strategy = str(family["authoring_focus"])
        family_index = 0
        unsupported_index = 0
        for intent in TARGETED_INTENTS:
            count = int(family["intent_counts"][intent])
            targets = boundary_targets(intent, dataset_role)
            if len(targets) != count:
                raise ValueError(f"boundary plan count mismatch: {family_id}:{intent}")
            for target in targets:
                family_index += 1
                record_id = f"{family_id}_{family_index:04d}"
                group_id = f"{family_id}_group_{family_index:04d}"
                records.append(
                    authoring_slot(
                        record_id=record_id,
                        group_id=group_id,
                        intent=intent,
                        family_id=family_id,
                        dataset_role=dataset_role,
                        strategy=strategy,
                        boundary_target=target,
                        unsupported_index=unsupported_index,
                    )
                )
                if intent == UNSUPPORTED_INTENT:
                    unsupported_index += 1
    return {
        "authoring_complete": False,
        "contract": {
            "path": (
                "data/evals/v2/ml/"
                "v2c6_targeted_remediation_design_contract.json"
            ),
            "sha256": CONTRACT_SHA256,
        },
        "dataset_role": dataset_role,
        "governance": {
            **GOVERNANCE_FLAGS,
            "all_text_slots_initially_empty": True,
            "local_ignored_workfile": True,
            "text_generated_by_builder": False,
        },
        "phase": WORKFLOW_PHASE,
        "record_count": len(records),
        "records": records,
        "schema_version": schema,
    }


def create_authoring_templates(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        workfile_template(contract, dataset_role=TRAINING_ROLE),
        workfile_template(contract, dataset_role=EVALUATION_ROLE),
    )


def validate_workfile_structure(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    dataset_role: str,
    require_complete: bool,
) -> tuple[dict[str, Any], ...]:
    expected = workfile_template(contract, dataset_role=dataset_role)
    if (
        payload.get("schema_version") != expected["schema_version"]
        or payload.get("phase") != WORKFLOW_PHASE
        or payload.get("dataset_role") != dataset_role
        or payload.get("record_count") != expected["record_count"]
        or payload.get("contract") != expected["contract"]
        or payload.get("governance") != expected["governance"]
    ):
        raise ValueError(f"unexpected {dataset_role} workfile identity")
    if not isinstance(payload.get("authoring_complete"), bool):
        raise TypeError("authoring_complete must be boolean")
    if require_complete and payload.get("authoring_complete") is not True:
        raise ValueError(f"{dataset_role} authoring is not marked complete")
    records = _object_list(payload.get("records"), f"{dataset_role} records")
    if len(records) != expected["record_count"]:
        raise ValueError(f"{dataset_role} record count changed")
    expected_by_id = {
        str(record["record_id"]): record for record in expected["records"]
    }
    immutable_fields = (
        "authoring_strategy",
        "boundary_target",
        "candidate_outputs_inspected_during_authoring",
        "consumed_old_source_family_used_as_paraphrase_template",
        "dataset_role",
        "excluded_from_candidate_fitting",
        "group_id",
        "independently_authored",
        "intent",
        "is_hard_negative",
        "not_training_data",
        "prediction_informed_authoring",
        "record_id",
        "risk_level",
        "source_family_id",
        "unsupported_subtype",
    )
    validated: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    family_methods: dict[str, str] = {}
    for source_record in records:
        record = copy.deepcopy(source_record)
        record_id = _nonempty_string(record, "record_id")
        if record_id in observed_ids:
            raise ValueError(f"duplicate authoring record ID: {record_id}")
        observed_ids.add(record_id)
        expected_record = expected_by_id.get(record_id)
        if expected_record is None:
            raise ValueError(f"unexpected authoring record ID: {record_id}")
        if PROHIBITED_PREDICTION_FIELDS & _nested_keys(record):
            raise ValueError(f"prediction-informed metadata prohibited: {record_id}")
        for field in immutable_fields:
            if record.get(field) != expected_record[field]:
                raise ValueError(
                    f"frozen slot field changed: {record_id}:{field}"
                )
        provenance = record.get("authoring_provenance")
        expected_provenance = expected_record["authoring_provenance"]
        if not isinstance(provenance, dict):
            raise TypeError(f"authoring_provenance must be an object: {record_id}")
        if set(provenance) != set(expected_provenance):
            raise ValueError(f"authoring provenance schema changed: {record_id}")
        for field in (
            "authoring_batch_id",
            "source_family_independence_basis",
            "source_revision",
        ):
            if provenance.get(field) != expected_provenance[field]:
                raise ValueError(
                    f"frozen authoring provenance changed: {record_id}:{field}"
                )
        method = provenance.get("authoring_method")
        if require_complete:
            if method not in AUTHORING_METHODS:
                raise ValueError(f"invalid authoring method: {record_id}")
            family_id = str(record["source_family_id"])
            previous = family_methods.setdefault(family_id, str(method))
            if previous != method:
                raise ValueError(
                    f"source-family authoring method is inconsistent: {family_id}"
                )
        elif method not in {"", *AUTHORING_METHODS}:
            raise ValueError(f"invalid authoring method: {record_id}")
        flags = record.get("review_flags")
        if not isinstance(flags, dict) or set(flags) != set(REVIEW_FLAG_FIELDS):
            raise ValueError(f"review flag schema changed: {record_id}")
        if any(not isinstance(flags[field], bool) for field in REVIEW_FLAG_FIELDS):
            raise TypeError(f"review flags must be boolean: {record_id}")
        for field in (
            "requires_human_adjudication",
            "human_adjudication_completed",
        ):
            if not isinstance(record.get(field), bool):
                raise TypeError(f"{field} must be boolean: {record_id}")
        if not isinstance(record.get("text"), str):
            raise TypeError(f"text must be a string: {record_id}")
        if not isinstance(record.get("review_notes"), str):
            raise TypeError(f"review_notes must be a string: {record_id}")
        if not isinstance(record.get("reviewer"), str):
            raise TypeError(f"reviewer must be a string: {record_id}")
        if not isinstance(record.get("human_adjudicator"), str):
            raise TypeError(f"human_adjudicator must be a string: {record_id}")
        if record.get("review_status") not in {
            "unreviewed",
            "approved",
            "rejected",
            "needs_revision",
        }:
            raise ValueError(f"invalid review status: {record_id}")
        if record.get("review_method") not in {"", *REVIEW_METHODS}:
            raise ValueError(f"invalid review method: {record_id}")
        inferred_adjudication = (
            record.get("review_status") in {"rejected", "needs_revision"}
            or any(flags.values())
        )
        if inferred_adjudication and (
            record.get("requires_human_adjudication") is not True
        ):
            raise ValueError(f"human adjudication trigger is unmarked: {record_id}")
        if require_complete:
            _validate_completed_record(record)
        validated.append(record)
    if observed_ids != set(expected_by_id):
        raise ValueError(f"{dataset_role} authoring slots are incomplete")
    return tuple(validated)


def _validate_completed_record(record: Mapping[str, Any]) -> None:
    record_id = str(record["record_id"])
    text = record.get("text")
    if not isinstance(text, str) or not normalize_text(text):
        raise ValueError(f"authoring text is empty: {record_id}")
    review_method = record.get("review_method")
    if review_method not in REVIEW_METHODS:
        raise ValueError(f"invalid review method: {record_id}")
    if record.get("review_status") != "approved":
        raise ValueError(f"all records must be approved: {record_id}")
    if not str(record.get("review_notes", "")).strip():
        raise ValueError(f"approved review requires notes: {record_id}")
    if not str(record.get("reviewer", "")).strip():
        raise ValueError(f"approved review requires reviewer provenance: {record_id}")
    flags = record["review_flags"]
    inferred_adjudication = any(flags.values())
    required_adjudication = bool(
        record.get("requires_human_adjudication") or inferred_adjudication
    )
    if inferred_adjudication and record.get("requires_human_adjudication") is not True:
        raise ValueError(f"human adjudication trigger is unmarked: {record_id}")
    if required_adjudication:
        if record.get("human_adjudication_completed") is not True:
            raise ValueError(f"required human adjudication unresolved: {record_id}")
        if not str(record.get("human_adjudicator", "")).strip():
            raise ValueError(f"human adjudicator is missing: {record_id}")
    elif record.get("human_adjudication_completed") is not False:
        raise ValueError(f"unexpected human adjudication completion: {record_id}")
    if record.get("review_method") == "ai_assisted_review" and (
        record.get("human_adjudication_completed") is True
        and not str(record.get("human_adjudicator", "")).strip()
    ):
        raise ValueError(f"AI review cannot imply human adjudication: {record_id}")


def boundary_coverage_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    families = sorted({str(record["source_family_id"]) for record in records})
    required = {frozenset(pair) for pair in HARD_NEGATIVE_PAIRS}
    tier_one = {frozenset(pair) for pair in TIER_ONE_PAIRS}
    tier_two = {frozenset(pair) for pair in TIER_TWO_PAIRS}
    for family in families:
        family_records = [
            record
            for record in records
            if record["source_family_id"] == family
        ]
        counts: Counter[frozenset[str]] = Counter()
        side_counts: defaultdict[
            frozenset[str], Counter[tuple[str, str]]
        ] = defaultdict(Counter)
        for record in family_records:
            pair = frozenset(
                (str(record["intent"]), str(record["boundary_target"]))
            )
            if pair not in required or record["is_hard_negative"] is not True:
                raise ValueError(f"unfrozen boundary allocation: {record['record_id']}")
            counts[pair] += 1
            side_counts[pair][
                (str(record["intent"]), str(record["boundary_target"]))
            ] += 1
        pair_rows: list[dict[str, Any]] = []
        for side_a, side_b in HARD_NEGATIVE_PAIRS:
            pair = frozenset((side_a, side_b))
            a_to_b = side_counts[pair][(side_a, side_b)]
            b_to_a = side_counts[pair][(side_b, side_a)]
            pair_rows.append(
                {
                    "count": counts[pair],
                    "pair": [side_a, side_b],
                    "side_a_count": a_to_b,
                    "side_b_count": b_to_a,
                }
            )
            if a_to_b <= 0 or b_to_a <= 0:
                raise ValueError(
                    f"hard-negative pair lacks two-sided coverage: {family}"
                )
        minimum_tier_one = min(counts[pair] for pair in tier_one)
        maximum_tier_two = max(counts[pair] for pair in tier_two)
        if minimum_tier_one <= maximum_tier_two:
            raise ValueError(f"Tier-1 boundary allocation is not greater: {family}")
        by_family[family] = {
            "all_ten_pairs_covered_on_both_sides": True,
            "maximum_tier_2_pair_count": maximum_tier_two,
            "minimum_tier_1_pair_count": minimum_tier_one,
            "pair_count": 10,
            "pairs": pair_rows,
            "tier_1_allocation_greater_than_tier_2": True,
        }
    return {
        "all_families_cover_all_pairs": all(
            report["all_ten_pairs_covered_on_both_sides"]
            for report in by_family.values()
        ),
        "by_source_family": by_family,
        "required_pair_count": 10,
    }


def review_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = Counter(str(record["review_method"]) for record in records)
    required = sum(
        bool(record["requires_human_adjudication"]) for record in records
    )
    completed = sum(
        bool(record["human_adjudication_completed"]) for record in records
    )
    return {
        "ai_assisted_review_is_human_review": False,
        "approved_count": sum(
            record["review_status"] == "approved" for record in records
        ),
        "human_adjudication_completed_count": completed,
        "human_adjudication_required_count": required,
        "record_count": len(records),
        "review_methods": dict(sorted(methods.items())),
        "unresolved_human_adjudication_count": required - completed,
    }


def _digest_maps(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    defaultdict[str, list[Mapping[str, Any]]],
    defaultdict[str, list[Mapping[str, Any]]],
]:
    exact: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        text = str(record["text"])
        exact[text_sha256(text)].append(record)
        normalized[normalized_text_sha256(text)].append(record)
    return exact, normalized


def duplicate_and_leakage_report(
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training_exact, training_normalized = _digest_maps(training)
    evaluation_exact, evaluation_normalized = _digest_maps(evaluation)
    historical_exact = {str(record["text_sha256"]) for record in historical}
    historical_normalized = {
        str(record["normalized_text_sha256"]) for record in historical
    }
    consumed = [
        record
        for record in historical
        if record.get("source_family_id") in CONSUMED_SOURCE_FAMILIES
    ]
    consumed_exact = {str(record["text_sha256"]) for record in consumed}
    consumed_normalized = {
        str(record["normalized_text_sha256"]) for record in consumed
    }
    exact_within_training = {
        digest for digest, rows in training_exact.items() if len(rows) > 1
    }
    normalized_within_training = {
        digest for digest, rows in training_normalized.items() if len(rows) > 1
    }
    exact_within_evaluation = {
        digest for digest, rows in evaluation_exact.items() if len(rows) > 1
    }
    normalized_within_evaluation = {
        digest for digest, rows in evaluation_normalized.items() if len(rows) > 1
    }
    training_evaluation_exact = set(training_exact) & set(evaluation_exact)
    training_evaluation_normalized = set(training_normalized) & set(
        evaluation_normalized
    )
    training_historical_exact = set(training_exact) & historical_exact
    training_historical_normalized = (
        set(training_normalized) & historical_normalized
    )
    evaluation_historical_exact = set(evaluation_exact) & historical_exact
    evaluation_historical_normalized = (
        set(evaluation_normalized) & historical_normalized
    )
    evaluation_consumed_exact = set(evaluation_exact) & consumed_exact
    evaluation_consumed_normalized = (
        set(evaluation_normalized) & consumed_normalized
    )
    combined_new = [*training, *evaluation]
    cross_intent: set[str] = set()
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for record in combined_new:
        grouped[normalized_text_sha256(str(record["text"]))].add(
            str(record["intent"])
        )
    for digest, intents in grouped.items():
        if len(intents) > 1:
            cross_intent.add(digest)
    counts = {
        "cross_intent_normalized_duplicate_count": len(cross_intent),
        "evaluation_consumed_exact_overlap_count": len(
            evaluation_consumed_exact
        ),
        "evaluation_consumed_normalized_overlap_count": len(
            evaluation_consumed_normalized
        ),
        "evaluation_historical_exact_overlap_count": len(
            evaluation_historical_exact
        ),
        "evaluation_historical_normalized_overlap_count": len(
            evaluation_historical_normalized
        ),
        "evaluation_within_exact_duplicate_count": len(
            exact_within_evaluation
        ),
        "evaluation_within_normalized_duplicate_count": len(
            normalized_within_evaluation
        ),
        "training_evaluation_exact_overlap_count": len(
            training_evaluation_exact
        ),
        "training_evaluation_normalized_overlap_count": len(
            training_evaluation_normalized
        ),
        "training_historical_exact_overlap_count": len(
            training_historical_exact
        ),
        "training_historical_normalized_overlap_count": len(
            training_historical_normalized
        ),
        "training_within_exact_duplicate_count": len(exact_within_training),
        "training_within_normalized_duplicate_count": len(
            normalized_within_training
        ),
    }
    if any(counts.values()):
        failures = [name for name, count in counts.items() if count]
        raise ValueError("duplicate or leakage checks failed: " + ", ".join(failures))
    return {
        **counts,
        "all_duplicate_and_leakage_checks_passed": True,
        "embedding_or_semantic_similarity_used": False,
        "normalization_version": NORMALIZATION_VERSION,
        "raw_text_persisted_in_report": False,
    }


def validate_record_and_group_isolation(
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    sources: FrozenSources,
) -> dict[str, Any]:
    training_ids = {str(record["record_id"]) for record in training}
    evaluation_ids = {str(record["record_id"]) for record in evaluation}
    training_groups = {str(record["group_id"]) for record in training}
    evaluation_groups = {str(record["group_id"]) for record in evaluation}
    if len(training_ids) != len(training) or len(evaluation_ids) != len(evaluation):
        raise ValueError("new record IDs must be unique")
    if len(training_groups) != len(training) or len(evaluation_groups) != len(
        evaluation
    ):
        raise ValueError("new group IDs must be unique")
    if training_ids & evaluation_ids:
        raise ValueError("training and evaluation record IDs overlap")
    if training_groups & evaluation_groups:
        raise ValueError("training and evaluation group IDs overlap")
    if (training_ids | evaluation_ids) & set(sources.historical_ids):
        raise ValueError("new record ID collides with frozen development")
    if (training_groups | evaluation_groups) & set(
        sources.historical_group_ids
    ):
        raise ValueError("new group ID collides with frozen development")
    return {
        "cross_role_group_collision_count": 0,
        "cross_role_record_id_collision_count": 0,
        "historical_group_collision_count": 0,
        "historical_record_id_collision_count": 0,
        "new_group_ids_unique": True,
        "new_record_ids_unique": True,
    }


def load_workfile(
    path: Path,
    paths: BuildPaths,
    reader: BytesReader,
) -> tuple[dict[str, Any], bytes]:
    content = guarded_read_bytes(path, paths, reader)
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise TypeError(f"authoring workfile must be an object: {path}")
    return payload, content


def load_authoring_inputs(
    sources: FrozenSources,
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    *,
    require_complete: bool,
) -> AuthoringInputs:
    training_payload, training_bytes = load_workfile(
        paths.training_workfile, paths, reader
    )
    evaluation_payload, evaluation_bytes = load_workfile(
        paths.evaluation_workfile, paths, reader
    )
    training = validate_workfile_structure(
        training_payload,
        sources.contract,
        dataset_role=TRAINING_ROLE,
        require_complete=require_complete,
    )
    evaluation = validate_workfile_structure(
        evaluation_payload,
        sources.contract,
        dataset_role=EVALUATION_ROLE,
        require_complete=require_complete,
    )
    boundary = {
        "evaluation": boundary_coverage_report(evaluation),
        "training": boundary_coverage_report(training),
    }
    identity = validate_record_and_group_isolation(
        training, evaluation, sources
    )
    if require_complete:
        leakage = duplicate_and_leakage_report(
            training, evaluation, sources.development_records
        )
        training_subtypes = {
            str(record["unsupported_subtype"])
            for record in training
            if record["intent"] == UNSUPPORTED_INTENT
        }
        if training_subtypes != set(UNSUPPORTED_SUBTYPES):
            raise ValueError("training unsupported subtype coverage is incomplete")
        evaluation_subtypes = {
            str(record["unsupported_subtype"])
            for record in evaluation
            if record["intent"] == UNSUPPORTED_INTENT
        }
        if evaluation_subtypes != set(UNSUPPORTED_SUBTYPES):
            raise ValueError("evaluation unsupported subtype coverage is incomplete")
    else:
        leakage = {
            "all_duplicate_and_leakage_checks_passed": False,
            "not_run_until_text_is_complete": True,
        }
    report = {
        "boundary_coverage": boundary,
        "duplicate_and_leakage": leakage,
        "identity_and_group_isolation": identity,
        "review": {
            "evaluation": review_report(evaluation),
            "training": review_report(training),
        },
    }
    return AuthoringInputs(
        training_payload=training_payload,
        training_bytes=training_bytes,
        training_records=training,
        evaluation_payload=evaluation_payload,
        evaluation_bytes=evaluation_bytes,
        evaluation_records=evaluation,
        validation_report=report,
    )


def count_by(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def authored_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(record))
    output["normalized_text_sha256"] = normalized_text_sha256(
        str(record["text"])
    )
    output["text_sha256"] = text_sha256(str(record["text"]))
    return output


def combined_training_record(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record["authoring_provenance"]
    return {
        "authoring_batch_id": provenance["authoring_batch_id"],
        "authoring_method": provenance["authoring_method"],
        "authoring_strategy": record["authoring_strategy"],
        "boundary_target": record["boundary_target"],
        "data_role": "development",
        "dataset_role": TRAINING_ROLE,
        "example_id": record["record_id"],
        "group_id": record["group_id"],
        "independently_authored": record["independently_authored"],
        "intent": record["intent"],
        "is_hard_negative": record["is_hard_negative"],
        "normalized_text_sha256": normalized_text_sha256(str(record["text"])),
        "prediction_informed_authoring": record[
            "prediction_informed_authoring"
        ],
        "record_id": record["record_id"],
        "review_method": record["review_method"],
        "review_status": record["review_status"],
        "risk": record["risk_level"],
        "risk_level": record["risk_level"],
        "source_domain": "sentinelvoice_v2c6_targeted_remediation",
        "source_family_id": record["source_family_id"],
        "source_family_independence_basis": provenance[
            "source_family_independence_basis"
        ],
        "source_id": record["source_family_id"],
        "source_revision": provenance["source_revision"],
        "source_split": "development",
        "text": record["text"],
        "text_sha256": text_sha256(str(record["text"])),
        "unsupported_subtype": record.get("unsupported_subtype"),
        "v2c6_lineage": "TARGETED_REMEDIATION_R2_ADDITION",
    }


def build_training_payload(inputs: AuthoringInputs) -> dict[str, Any]:
    records = [
        authored_output_record(record)
        for record in sorted(
            inputs.training_records,
            key=lambda item: str(item["record_id"]),
        )
    ]
    return {
        "classifier_input_fields": ["text"],
        "dataset_role": TRAINING_ROLE,
        "example_count": len(records),
        "examples": records,
        "governance": {
            **GOVERNANCE_FLAGS,
            "all_records_approved": True,
            "fresh_evaluation_evidence": False,
        },
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "phase": WORKFLOW_PHASE,
        "schema_version": TRAINING_OUTPUT_SCHEMA,
    }


def build_evaluation_payload(inputs: AuthoringInputs) -> dict[str, Any]:
    records = [
        authored_output_record(record)
        for record in sorted(
            inputs.evaluation_records,
            key=lambda item: str(item["record_id"]),
        )
    ]
    family_counts: dict[str, Any] = {}
    for family_id in sorted({str(row["source_family_id"]) for row in records}):
        family = [row for row in records if row["source_family_id"] == family_id]
        protected = sum(row["intent"] in PROTECTED_INTENTS for row in family)
        non_protected = len(family) - protected
        family_counts[family_id] = {
            "maximum_passing_protected_false_positive_count": (
                non_protected // 100
            ),
            "non_protected_record_count": non_protected,
            "protected_false_positive_rate_threshold": 0.01,
            "protected_record_count": protected,
            "record_count": len(family),
        }
    return {
        "classifier_input_fields": ["text"],
        "dataset_role": EVALUATION_ROLE,
        "example_count": len(records),
        "examples": records,
        "fresh_evaluation_governance": {
            "candidate_outputs_inspected_during_authoring": False,
            "consumed_old_source_family_used_as_paraphrase_template": False,
            "excluded_from_candidate_fitting": True,
            "independently_authored": True,
            "not_training_data": True,
            "prediction_informed_authoring": False,
        },
        "governance": {
            **GOVERNANCE_FLAGS,
            "all_records_approved": True,
            "protected_fpr_calculated": False,
        },
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "phase": WORKFLOW_PHASE,
        "safety_composition_by_source_family": family_counts,
        "schema_version": EVALUATION_OUTPUT_SCHEMA,
    }


def build_combined_payload(
    sources: FrozenSources,
    inputs: AuthoringInputs,
    training_sha256: str,
) -> dict[str, Any]:
    old_records = [copy.deepcopy(record) for record in sources.development_records]
    new_records = [
        combined_training_record(record)
        for record in sorted(
            inputs.training_records,
            key=lambda item: str(item["record_id"]),
        )
    ]
    records = [*old_records, *new_records]
    return {
        "classifier_input_fields": ["text"],
        "dataset_version": "v2c6-targeted-remediated-development.v1",
        "example_count": len(records),
        "examples": records,
        "governance": {
            **GOVERNANCE_FLAGS,
            "existing_9008_records_mutated": False,
            "fresh_evaluation_records_included": False,
        },
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "ordering": "frozen_9008_order_then_new_training_record_id",
        "phase": WORKFLOW_PHASE,
        "schema_version": COMBINED_OUTPUT_SCHEMA,
        "source_artifacts": {
            "frozen_v2c6_development": {
                "path": (
                    "data/evals/v2/ml/"
                    "v2c6_remediated_development_dataset.json"
                ),
                "sha256": FROZEN_DEVELOPMENT_SHA256,
            },
            "targeted_training_examples": {
                "path": (
                    "data/evals/v2/ml/"
                    "v2c6_targeted_remediation_training_examples.json"
                ),
                "sha256": training_sha256,
            },
        },
    }


def common_manifest(
    inputs: AuthoringInputs,
    paths: BuildPaths,
    reader: BytesReader,
) -> dict[str, Any]:
    return {
        "builder": {
            "path": display_path(paths.builder, paths),
            "sha256": sha256_bytes(
                guarded_read_bytes(paths.builder, paths, reader)
            ),
            "version": BUILDER_VERSION,
        },
        "contract": {
            "path": display_path(paths.contract, paths),
            "sha256": CONTRACT_SHA256,
        },
        "freeze": {
            "path": display_path(paths.freeze, paths),
            "sha256": FREEZE_SHA256,
        },
        "frozen_development": {
            "path": display_path(paths.frozen_development, paths),
            "sha256": FROZEN_DEVELOPMENT_SHA256,
        },
        "governance": GOVERNANCE_FLAGS,
        "input_workfiles": {
            "fresh_evaluation": {
                "path": display_path(paths.evaluation_workfile, paths),
                "sha256": sha256_bytes(inputs.evaluation_bytes),
            },
            "targeted_training": {
                "path": display_path(paths.training_workfile, paths),
                "sha256": sha256_bytes(inputs.training_bytes),
            },
        },
        "phase": WORKFLOW_PHASE,
        "validation": inputs.validation_report,
    }


def build_manifests(
    sources: FrozenSources,
    inputs: AuthoringInputs,
    training_bytes: bytes,
    combined_bytes: bytes,
    evaluation_bytes: bytes,
    paths: BuildPaths,
    reader: BytesReader,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common = common_manifest(inputs, paths, reader)
    training_manifest = {
        **common,
        "artifact": {
            "path": display_path(paths.training_output, paths),
            "schema_version": TRAINING_OUTPUT_SCHEMA,
            "sha256": sha256_bytes(training_bytes),
        },
        "boundary_coverage": inputs.validation_report["boundary_coverage"][
            "training"
        ],
        "counts": {
            "by_intent": count_by(inputs.training_records, "intent"),
            "by_risk": count_by(inputs.training_records, "risk_level"),
            "by_source_family": count_by(
                inputs.training_records, "source_family_id"
            ),
            "record_count": len(inputs.training_records),
        },
        "schema_version": TRAINING_MANIFEST_SCHEMA,
    }
    combined_manifest = {
        **common,
        "artifact": {
            "path": display_path(paths.combined_output, paths),
            "schema_version": COMBINED_OUTPUT_SCHEMA,
            "sha256": sha256_bytes(combined_bytes),
        },
        "counts": {
            "fresh_evaluation_record_count": 0,
            "frozen_record_count": len(sources.development_records),
            "new_training_record_count": len(inputs.training_records),
            "total_record_count": (
                len(sources.development_records)
                + len(inputs.training_records)
            ),
        },
        "schema_version": COMBINED_MANIFEST_SCHEMA,
        "source_artifacts": {
            "frozen_development": {
                "path": display_path(paths.frozen_development, paths),
                "sha256": FROZEN_DEVELOPMENT_SHA256,
            },
            "targeted_training": {
                "path": display_path(paths.training_output, paths),
                "sha256": sha256_bytes(training_bytes),
            },
        },
    }
    evaluation_manifest = {
        **common,
        "artifact": {
            "path": display_path(paths.evaluation_output, paths),
            "schema_version": EVALUATION_OUTPUT_SCHEMA,
            "sha256": sha256_bytes(evaluation_bytes),
        },
        "boundary_coverage": inputs.validation_report["boundary_coverage"][
            "evaluation"
        ],
        "counts": {
            "by_intent": count_by(inputs.evaluation_records, "intent"),
            "by_risk": count_by(inputs.evaluation_records, "risk_level"),
            "by_source_family": count_by(
                inputs.evaluation_records, "source_family_id"
            ),
            "non_protected_record_count": 320,
            "protected_record_count": 320,
            "record_count": len(inputs.evaluation_records),
        },
        "fresh_evaluation_governance": {
            "excluded_from_candidate_fitting": True,
            "not_training_data": True,
            "protected_fpr_calculated": False,
        },
        "schema_version": EVALUATION_MANIFEST_SCHEMA,
    }
    return training_manifest, combined_manifest, evaluation_manifest


def validate_built_artifacts(
    artifacts: BuiltArtifacts,
    sources: FrozenSources,
    inputs: AuthoringInputs,
) -> None:
    (
        training,
        training_manifest,
        combined,
        combined_manifest,
        evaluation,
        evaluation_manifest,
    ) = artifacts.payloads
    if training["example_count"] != 600 or len(training["examples"]) != 600:
        raise ValueError("targeted training output must contain exactly 600 records")
    if combined["example_count"] != 9608 or len(combined["examples"]) != 9608:
        raise ValueError(
            "combined development output must contain exactly 9608 records"
        )
    if evaluation["example_count"] != 640 or len(evaluation["examples"]) != 640:
        raise ValueError("fresh evaluation output must contain exactly 640 records")
    old_count = len(sources.development_records)
    if combined["examples"][:old_count] != list(sources.development_records):
        raise ValueError("frozen 9008 development records changed")
    evaluation_ids = {row["record_id"] for row in evaluation["examples"]}
    combined_ids = {row["example_id"] for row in combined["examples"]}
    if evaluation_ids & combined_ids:
        raise ValueError("fresh evaluation records entered development training")
    evaluation_texts = {
        row["normalized_text_sha256"] for row in evaluation["examples"]
    }
    combined_texts = {
        row["normalized_text_sha256"] for row in combined["examples"]
    }
    if evaluation_texts & combined_texts:
        raise ValueError("fresh evaluation text entered development training")
    if any(
        row["risk_level"] != RISK_BY_INTENT[row["intent"]]
        for row in [*training["examples"], *evaluation["examples"]]
    ):
        raise ValueError("new record risk mapping changed")
    for manifest, content in zip(
        (training_manifest, combined_manifest, evaluation_manifest),
        (artifacts.contents[0], artifacts.contents[2], artifacts.contents[4]),
        strict=True,
    ):
        if manifest["artifact"]["sha256"] != sha256_bytes(content):
            raise ValueError("manifest artifact hash mismatch")
    if combined_manifest["counts"]["fresh_evaluation_record_count"] != 0:
        raise ValueError("combined manifest includes fresh evaluation records")
    if inputs.validation_report["duplicate_and_leakage"][
        "all_duplicate_and_leakage_checks_passed"
    ] is not True:
        raise ValueError("duplicate and leakage validation did not pass")


def build_artifacts(
    sources: FrozenSources,
    inputs: AuthoringInputs,
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> BuiltArtifacts:
    training_payload = build_training_payload(inputs)
    training_bytes = stable_json_bytes(training_payload)
    combined_payload = build_combined_payload(
        sources, inputs, sha256_bytes(training_bytes)
    )
    combined_bytes = stable_json_bytes(combined_payload)
    evaluation_payload = build_evaluation_payload(inputs)
    evaluation_bytes = stable_json_bytes(evaluation_payload)
    manifests = build_manifests(
        sources,
        inputs,
        training_bytes,
        combined_bytes,
        evaluation_bytes,
        paths,
        reader,
    )
    payloads = (
        training_payload,
        manifests[0],
        combined_payload,
        manifests[1],
        evaluation_payload,
        manifests[2],
    )
    contents = tuple(stable_json_bytes(payload) for payload in payloads)
    artifacts = BuiltArtifacts(contents=contents, payloads=payloads)
    validate_built_artifacts(artifacts, sources, inputs)
    return artifacts


def ensure_absent(paths_to_check: Sequence[Path], label: str) -> None:
    existing = [path for path in paths_to_check if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing {label}: "
            + ", ".join(str(path) for path in existing)
        )


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing file: {path}"
            ) from exc
        temporary_path.unlink()
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def preflight(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    loaded = sources or load_frozen_sources(paths, reader)
    training_present = paths.training_workfile.exists()
    evaluation_present = paths.evaluation_workfile.exists()
    ready = False
    validation: dict[str, Any] | None = None
    if training_present != evaluation_present:
        raise ValueError("authoring workfiles must either both exist or both be absent")
    if training_present:
        training_payload, _ = load_workfile(paths.training_workfile, paths, reader)
        evaluation_payload, _ = load_workfile(
            paths.evaluation_workfile, paths, reader
        )
        validate_workfile_structure(
            training_payload,
            loaded.contract,
            dataset_role=TRAINING_ROLE,
            require_complete=False,
        )
        validate_workfile_structure(
            evaluation_payload,
            loaded.contract,
            dataset_role=EVALUATION_ROLE,
            require_complete=False,
        )
        if (
            training_payload.get("authoring_complete") is True
            and evaluation_payload.get("authoring_complete") is True
        ):
            completed = load_authoring_inputs(
                loaded,
                paths,
                reader,
                require_complete=True,
            )
            ready = True
            validation = completed.validation_report
    return {
        "authoring_workfiles_present": training_present and evaluation_present,
        "files_written": False,
        "frozen_development_record_count": len(loaded.development_records),
        "governance": GOVERNANCE_FLAGS,
        "ready_for_build": ready,
        "tracked_outputs_present": any(path.exists() for path in output_paths(paths)),
        "validation": validation,
    }


def prepare_authoring_workfiles(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    loaded = sources or load_frozen_sources(paths, reader)
    workfiles = (paths.training_workfile, paths.evaluation_workfile)
    ensure_absent(workfiles, "authoring workfiles")
    training, evaluation = create_authoring_templates(loaded.contract)
    validate_workfile_structure(
        training,
        loaded.contract,
        dataset_role=TRAINING_ROLE,
        require_complete=False,
    )
    validate_workfile_structure(
        evaluation,
        loaded.contract,
        dataset_role=EVALUATION_ROLE,
        require_complete=False,
    )
    durable_create(paths.training_workfile, stable_json_bytes(training))
    durable_create(paths.evaluation_workfile, stable_json_bytes(evaluation))
    return {
        "evaluation_slot_count": evaluation["record_count"],
        "files_written": True,
        "governance": GOVERNANCE_FLAGS,
        "paths": [display_path(path, paths) for path in workfiles],
        "text_slots_populated": 0,
        "tracked_files_written": False,
        "training_slot_count": training["record_count"],
    }


def write_artifacts(artifacts: BuiltArtifacts, paths: BuildPaths) -> None:
    targets = output_paths(paths)
    ensure_absent(targets, "tracked outputs")
    for path, content in zip(targets, artifacts.contents, strict=True):
        durable_create(path, content)


def build(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    ensure_absent(output_paths(paths), "tracked outputs")
    loaded = sources or load_frozen_sources(paths, reader)
    inputs = load_authoring_inputs(
        loaded,
        paths,
        reader,
        require_complete=True,
    )
    artifacts = build_artifacts(loaded, inputs, paths, reader)
    write_artifacts(artifacts, paths)
    return {
        "combined_development_record_count": 9608,
        "files_written": True,
        "fresh_evaluation_record_count": 640,
        "governance": GOVERNANCE_FLAGS,
        "output_paths": [
            display_path(path, paths) for path in output_paths(paths)
        ],
        "targeted_training_record_count": 600,
    }


def check_results(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    loaded = sources or load_frozen_sources(paths, reader)
    inputs = load_authoring_inputs(
        loaded,
        paths,
        reader,
        require_complete=True,
    )
    expected = build_artifacts(loaded, inputs, paths, reader)
    actual = tuple(
        guarded_read_bytes(path, paths, reader) for path in output_paths(paths)
    )
    if actual != expected.contents:
        raise ValueError("existing outputs are not the deterministic build")
    return {
        "files_written": False,
        "governance": GOVERNANCE_FLAGS,
        "output_hashes": {
            display_path(path, paths): sha256_bytes(content)
            for path, content in zip(output_paths(paths), actual, strict=True)
        },
        "results_valid": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--prepare-authoring-workfiles", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check-results", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight:
        result = preflight()
    elif args.prepare_authoring_workfiles:
        result = prepare_authoring_workfiles()
    elif args.build:
        result = build()
    else:
        result = check_results()
    print(stable_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
