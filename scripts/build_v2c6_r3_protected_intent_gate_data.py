"""Prepare and build V2-C6 R3 protected-intent gate data.

The builder prepares empty, ignored authoring workfiles and later validates
completed authoring before creating tracked datasets.  It never authors text,
runs a model, or accesses a final holdout.
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
SCRIPT_RELATIVE_PATH = "scripts/build_v2c6_r3_protected_intent_gate_data.py"

CONTRACT_SHA256 = (
    "4a744a48e18d674c1f87bbd53f470cae95f0bf09fafd7920942310712c0c894e"
)
NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"
BUILDER_VERSION = "v2c6-r3-protected-intent-gate-data-builder.v1"
WORKFLOW_PHASE = "V2-C6 R3 protected-intent gate data authoring"

TRAINING_WORKFILE_SCHEMA = (
    "v2c6-r3-protected-intent-gate-training-authoring-workfile.v1"
)
EVALUATION_WORKFILE_SCHEMA = (
    "v2c6-r3-fresh-source-evaluation-authoring-workfile.v1"
)
TRAINING_OUTPUT_SCHEMA = "v2c6-r3-protected-intent-gate-training-examples.v1"
TRAINING_MANIFEST_SCHEMA = (
    "v2c6-r3-protected-intent-gate-training-examples-manifest.v1"
)
DEVELOPMENT_OUTPUT_SCHEMA = (
    "v2c6-r3-protected-intent-gate-development-dataset.v1"
)
DEVELOPMENT_MANIFEST_SCHEMA = (
    "v2c6-r3-protected-intent-gate-development-dataset-manifest.v1"
)
EVALUATION_OUTPUT_SCHEMA = "v2c6-r3-fresh-source-evaluation-dataset.v1"
EVALUATION_MANIFEST_SCHEMA = (
    "v2c6-r3-fresh-source-evaluation-dataset-manifest.v1"
)

TRAINING_ROLE = "protected_intent_gate_training"
EVALUATION_ROLE = "fresh_source_evaluation"
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
PROTECTED_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)
PRIMARY_INTENTS = (
    "account_blocked",
    *PROTECTED_INTENTS,
    "transfer_failed_or_declined",
    "transfer_pending",
    UNSUPPORTED_INTENT,
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
GOVERNANCE_FLAGS = {
    "builder_generated_example_text": False,
    "candidate_selected": False,
    "dataset_mutated_before_build": False,
    "embeddings_generated": False,
    "final_holdout_accessed": False,
    "inference_performed": False,
    "model_fitting_performed": False,
    "model_inference_performed": False,
    "model_selection_performed": False,
    "models_run": False,
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
    existing_development: Path
    consumed_r2_evaluation: Path
    training_workfile: Path
    evaluation_workfile: Path
    training_output: Path
    training_manifest: Path
    development_output: Path
    development_manifest: Path
    evaluation_output: Path
    evaluation_manifest: Path
    prohibited_holdout: Path
    builder: Path


DEFAULT_PATHS = BuildPaths(
    repository_root=REPOSITORY_ROOT,
    contract=(
        ML_DIRECTORY
        / "v2c6_protected_intent_gate_remediation_design_contract.json"
    ),
    existing_development=(
        ML_DIRECTORY / "v2c6_targeted_remediated_development_dataset.json"
    ),
    consumed_r2_evaluation=(
        ML_DIRECTORY / "v2c6_fresh_source_evaluation_dataset.json"
    ),
    training_workfile=(
        ML_DIRECTORY
        / "local/v2c6_r3_protected_intent_gate_training_authoring.json"
    ),
    evaluation_workfile=(
        ML_DIRECTORY
        / "local/v2c6_r3_fresh_source_evaluation_authoring.json"
    ),
    training_output=(
        ML_DIRECTORY / "v2c6_r3_protected_intent_gate_training_examples.json"
    ),
    training_manifest=(
        ML_DIRECTORY
        / "v2c6_r3_protected_intent_gate_training_examples.manifest.json"
    ),
    development_output=(
        ML_DIRECTORY
        / "v2c6_r3_protected_intent_gate_development_dataset.json"
    ),
    development_manifest=(
        ML_DIRECTORY
        / "v2c6_r3_protected_intent_gate_development_dataset.manifest.json"
    ),
    evaluation_output=(
        ML_DIRECTORY / "v2c6_r3_fresh_source_evaluation_dataset.json"
    ),
    evaluation_manifest=(
        ML_DIRECTORY
        / "v2c6_r3_fresh_source_evaluation_dataset.manifest.json"
    ),
    prohibited_holdout=ML_DIRECTORY / "v2c5_final_holdout.json",
    builder=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
)


@dataclass(frozen=True)
class FrozenSources:
    contract: dict[str, Any]
    contract_sha256: str
    development: dict[str, Any]
    development_records: tuple[dict[str, Any], ...]
    consumed_evaluation: dict[str, Any]
    consumed_evaluation_records: tuple[dict[str, Any], ...]
    source_lineage: dict[str, dict[str, Any]]
    historical_ids: frozenset[str]
    historical_group_ids: frozenset[str]


@dataclass(frozen=True)
class AuthoringInputs:
    training_bytes: bytes
    training_records: tuple[dict[str, Any], ...]
    evaluation_bytes: bytes
    evaluation_records: tuple[dict[str, Any], ...]
    validation_report: dict[str, Any]


@dataclass(frozen=True)
class BuiltArtifacts:
    contents: tuple[bytes, ...]
    payloads: tuple[dict[str, Any], ...]


def stable_json_bytes(payload: Any) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode()


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


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode())


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
        paths.development_output,
        paths.development_manifest,
        paths.evaluation_output,
        paths.evaluation_manifest,
    )


def validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version")
        != "v2c6-protected-intent-gate-remediation-design-contract.v1"
        or contract.get("phase")
        != "V2-C6 protected-intent safety-gate remediation design"
        or contract.get("status") != "FROZEN"
        or contract.get("next_required")
        != "v2c6_r3_targeted_addendum_and_fresh_evaluation_authoring"
    ):
        raise ValueError("unexpected R3 remediation contract identity")
    status = contract.get("contract_status", {})
    if (
        status.get("contract_frozen") is not True
        or status.get("data_authored") is not False
        or status.get("step29i_authorized") is not False
        or status.get("final_holdout_accessed") is not False
    ):
        raise ValueError("R3 remediation contract status changed")
    training = contract.get("targeted_training_addendum", {})
    evaluation = contract.get("fresh_evaluation_specification", {})
    if (
        training.get("record_count") != 480
        or training.get("family_count") != 3
        or evaluation.get("record_count") != 640
        or evaluation.get("family_count") != 2
        or evaluation.get("per_intent_per_family_count") != 40
    ):
        raise ValueError("frozen R3 authoring volume changed")
    architecture = contract.get("architecture_decision", {})
    if tuple(architecture.get("protected_intents", [])) != PROTECTED_INTENTS:
        raise ValueError("protected-intent set changed")
    final_policy = contract.get("final_holdout_policy", {})
    if any(
        final_policy.get(field) is not False
        for field in (
            "access_permitted",
            "hashing_permitted",
            "inspection_permitted",
            "parsing_permitted",
            "searching_permitted",
        )
    ):
        raise ValueError("final holdout prohibition changed")


def _validate_dataset(
    payload: Mapping[str, Any],
    *,
    schema: str,
    count: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    records = _object_list(payload.get("examples"), f"{label} examples")
    if (
        payload.get("schema_version") != schema
        or payload.get("example_count") != count
        or len(records) != count
        or payload.get("normalization_version") != NORMALIZATION_VERSION
    ):
        raise ValueError(f"{label} dataset identity changed")
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for record in records:
        identifier = record.get("example_id", record.get("record_id"))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{label} record ID is missing")
        if identifier in ids:
            raise ValueError(f"duplicate {label} record ID: {identifier}")
        ids.add(identifier)
        text = _nonempty_string(record, "text")
        _nonempty_string(record, "intent")
        _nonempty_string(record, "group_id")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError(f"{label} text hash mismatch: {identifier}")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError(f"{label} normalized hash mismatch: {identifier}")
        validated.append(copy.deepcopy(record))
    return tuple(validated)


def _validate_analysis_status(name: str, payload: Mapping[str, Any]) -> None:
    if name not in {"failure_analysis_result", "failure_analysis_manifest"}:
        return
    status = payload.get("governance", payload)
    execution = payload.get("execution_status", status.get("execution_status"))
    if execution != "COMPLETED":
        raise ValueError(f"{name} is not completed")
    for field in (
        "failure_analysis_executed",
        "final_holdout_accessed",
        "step29i_authorized",
    ):
        expected = field == "failure_analysis_executed"
        observed = payload.get(field, status.get(field))
        if observed is not expected:
            raise ValueError(f"{name} governance changed: {field}")


def load_frozen_sources(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> FrozenSources:
    contract_bytes = guarded_read_bytes(paths.contract, paths, reader)
    contract_sha = sha256_bytes(contract_bytes)
    if contract_sha != CONTRACT_SHA256:
        raise ValueError(f"R3 contract SHA-256 mismatch: {contract_sha}")
    contract = json.loads(contract_bytes)
    if not isinstance(contract, dict):
        raise TypeError("R3 contract must be an object")
    validate_contract(contract)

    lineage: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, specification in contract["source_artifacts"].items():
        source_path = paths.repository_root / str(specification["path"])
        content = guarded_read_bytes(source_path, paths, reader)
        digest = sha256_bytes(content)
        if digest != specification["sha256"]:
            raise ValueError(f"source artifact SHA-256 mismatch: {name}")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise TypeError(f"source artifact must be an object: {name}")
        if payload.get("schema_version") != specification["schema_version"]:
            raise ValueError(f"source artifact schema changed: {name}")
        _validate_analysis_status(name, payload)
        payloads[name] = payload
        lineage[name] = {
            "path": str(specification["path"]),
            "schema_version": str(specification["schema_version"]),
            "sha256": digest,
        }

    development = payloads["existing_development_dataset"]
    development_records = _validate_dataset(
        development,
        schema="v2c6-targeted-remediated-development-dataset.v1",
        count=9608,
        label="existing development",
    )
    consumed = payloads["consumed_r2_fresh_dataset"]
    consumed_records = _validate_dataset(
        consumed,
        schema="v2c6-fresh-source-evaluation-dataset.v1",
        count=640,
        label="consumed R2 evaluation",
    )
    historical_ids = {
        str(record.get("example_id", record.get("record_id")))
        for record in (*development_records, *consumed_records)
    }
    historical_groups = {
        str(record["group_id"])
        for record in (*development_records, *consumed_records)
    }
    return FrozenSources(
        contract=contract,
        contract_sha256=contract_sha,
        development=development,
        development_records=development_records,
        consumed_evaluation=consumed,
        consumed_evaluation_records=consumed_records,
        source_lineage=lineage,
        historical_ids=frozenset(historical_ids),
        historical_group_ids=frozenset(historical_groups),
    )


def _family_basis(family_id: str, strategy: str, role: str) -> str:
    return (
        f"independently structured {role} family using the frozen {strategy} "
        f"authoring strategy ({family_id})"
    )


def authoring_slot(
    *,
    record_id: str,
    intent: str,
    family_id: str,
    dataset_role: str,
    strategy: str,
    target: str | None,
    verifier_role: str,
    polarity: str,
) -> dict[str, Any]:
    evaluation = dataset_role == EVALUATION_ROLE
    return {
        "authoring_provenance": {
            "authoring_batch_id": f"{family_id}_batch_001",
            "authoring_method": "",
            "source_family_independence_basis": _family_basis(
                family_id, strategy, dataset_role
            ),
            "source_revision": "v1",
        },
        "authoring_strategy": strategy,
        "candidate_definitions_revised_from_this_evaluation": False,
        "candidate_outputs_inspected_during_authoring": False,
        "consumed_r2_evaluation_used_as_paraphrase_source": False,
        "dataset_role": dataset_role,
        "excluded_from_candidate_fitting": evaluation,
        "other_r3_fresh_family_wording_consulted": False,
        "group_id": f"{record_id}_group",
        "human_adjudication_completed": False,
        "human_adjudicator": "",
        "independently_authored": True,
        "intent": intent,
        "not_training_data": evaluation,
        "polarity": polarity,
        "prediction_informed_authoring": False,
        "protected_boundary_target": target,
        "r3_training_used_as_paraphrase_source": False,
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
        "text_sha256": "",
        "normalized_text_sha256": "",
        "verifier_role": verifier_role,
    }


def _training_slots(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    families = contract["targeted_training_addendum"]["family_specifications"]
    for family in families:
        family_id = str(family["source_family_id"])
        strategy = str(family["authoring_style"])
        index = 0
        for intent in PROTECTED_INTENTS:
            for _ in range(int(family["positive_intent_counts"][intent])):
                index += 1
                record_id = f"{family_id}_{index:04d}"
                records.append(
                    authoring_slot(
                        record_id=record_id,
                        intent=intent,
                        family_id=family_id,
                        dataset_role=TRAINING_ROLE,
                        strategy=strategy,
                        target=intent,
                        verifier_role="protected_positive",
                        polarity="positive",
                    )
                )
        for target in PROTECTED_INTENTS:
            for _ in range(int(family["unsupported_target_counts"][target])):
                index += 1
                record_id = f"{family_id}_{index:04d}"
                records.append(
                    authoring_slot(
                        record_id=record_id,
                        intent=UNSUPPORTED_INTENT,
                        family_id=family_id,
                        dataset_role=TRAINING_ROLE,
                        strategy=strategy,
                        target=target,
                        verifier_role="targeted_unsupported_negative",
                        polarity="negative",
                    )
                )
        if index != 160:
            raise ValueError(f"training slot count changed: {family_id}")
    return records


def _evaluation_slots(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    families = contract["fresh_evaluation_specification"][
        "family_specifications"
    ]
    for family in families:
        family_id = str(family["source_family_id"])
        strategy = family_id.rsplit("_", maxsplit=1)[-1]
        index = 0
        for intent in PRIMARY_INTENTS:
            count = int(family["intent_counts"][intent])
            if intent == UNSUPPORTED_INTENT:
                targets = [
                    target
                    for target in PROTECTED_INTENTS
                    for _ in range(
                        int(family["unsupported_boundary_target_counts"][target])
                    )
                ]
            else:
                targets = [intent if intent in PROTECTED_INTENTS else None] * count
            if len(targets) != count:
                raise ValueError(f"evaluation target count changed: {family_id}")
            for target in targets:
                index += 1
                record_id = f"{family_id}_{index:04d}"
                if intent in PROTECTED_INTENTS:
                    role, polarity = "protected_positive", "positive"
                elif intent == UNSUPPORTED_INTENT:
                    role, polarity = "targeted_unsupported_negative", "negative"
                else:
                    role, polarity = "non_protected_control", "not_applicable"
                records.append(
                    authoring_slot(
                        record_id=record_id,
                        intent=intent,
                        family_id=family_id,
                        dataset_role=EVALUATION_ROLE,
                        strategy=strategy,
                        target=target,
                        verifier_role=role,
                        polarity=polarity,
                    )
                )
        if index != 320:
            raise ValueError(f"evaluation slot count changed: {family_id}")
    return records


def workfile_template(
    contract: Mapping[str, Any], *, dataset_role: str
) -> dict[str, Any]:
    if dataset_role == TRAINING_ROLE:
        records = _training_slots(contract)
        schema = TRAINING_WORKFILE_SCHEMA
    elif dataset_role == EVALUATION_ROLE:
        records = _evaluation_slots(contract)
        schema = EVALUATION_WORKFILE_SCHEMA
    else:
        raise ValueError(f"unknown dataset role: {dataset_role}")
    return {
        "authoring_complete": False,
        "contract": {
            "path": (
                "data/evals/v2/ml/"
                "v2c6_protected_intent_gate_remediation_design_contract.json"
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


def _validate_completed_record(record: Mapping[str, Any]) -> None:
    record_id = str(record["record_id"])
    text = record.get("text")
    if not isinstance(text, str) or not normalize_text(text):
        raise ValueError(f"authoring text is empty: {record_id}")
    if record.get("text_sha256") != text_sha256(text):
        raise ValueError(f"authoring text hash mismatch: {record_id}")
    if record.get("normalized_text_sha256") != normalized_text_sha256(text):
        raise ValueError(f"authoring normalized hash mismatch: {record_id}")
    if record.get("review_status") != "approved":
        raise ValueError(f"all records must be approved: {record_id}")
    if record.get("review_method") not in REVIEW_METHODS:
        raise ValueError(f"invalid review method: {record_id}")
    for field in ("reviewer", "review_notes"):
        if not str(record.get(field, "")).strip():
            raise ValueError(f"approved review requires {field}: {record_id}")
    flags = record["review_flags"]
    required = bool(record["requires_human_adjudication"] or any(flags.values()))
    if any(flags.values()) and record["requires_human_adjudication"] is not True:
        raise ValueError(f"human adjudication trigger is unmarked: {record_id}")
    if required:
        if record["human_adjudication_completed"] is not True:
            raise ValueError(f"required human adjudication unresolved: {record_id}")
        if not str(record["human_adjudicator"]).strip():
            raise ValueError(f"human adjudicator is missing: {record_id}")
    elif record["human_adjudication_completed"] is not False:
        raise ValueError(f"unexpected human adjudication completion: {record_id}")


def validate_workfile_structure(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    dataset_role: str,
    require_complete: bool,
) -> tuple[dict[str, Any], ...]:
    expected = workfile_template(contract, dataset_role=dataset_role)
    for field in (
        "schema_version",
        "phase",
        "dataset_role",
        "record_count",
        "contract",
        "governance",
    ):
        if payload.get(field) != expected[field]:
            raise ValueError(f"unexpected {dataset_role} workfile identity")
    if not isinstance(payload.get("authoring_complete"), bool):
        raise TypeError("authoring_complete must be boolean")
    if require_complete and payload["authoring_complete"] is not True:
        raise ValueError(f"{dataset_role} authoring is not marked complete")
    records = _object_list(payload.get("records"), f"{dataset_role} records")
    expected_by_id = {
        str(record["record_id"]): record for record in expected["records"]
    }
    immutable = (
        "authoring_strategy",
        "candidate_definitions_revised_from_this_evaluation",
        "candidate_outputs_inspected_during_authoring",
        "consumed_r2_evaluation_used_as_paraphrase_source",
        "dataset_role",
        "excluded_from_candidate_fitting",
        "group_id",
        "independently_authored",
        "intent",
        "not_training_data",
        "other_r3_fresh_family_wording_consulted",
        "polarity",
        "prediction_informed_authoring",
        "protected_boundary_target",
        "r3_training_used_as_paraphrase_source",
        "record_id",
        "risk_level",
        "source_family_id",
        "verifier_role",
    )
    validated: list[dict[str, Any]] = []
    observed: set[str] = set()
    family_methods: dict[str, str] = {}
    for source in records:
        record = copy.deepcopy(source)
        record_id = _nonempty_string(record, "record_id")
        expected_record = expected_by_id.get(record_id)
        if expected_record is None or record_id in observed:
            raise ValueError(f"unexpected or duplicate authoring ID: {record_id}")
        observed.add(record_id)
        if PROHIBITED_PREDICTION_FIELDS & _nested_keys(record):
            raise ValueError(f"prediction metadata prohibited: {record_id}")
        for field in immutable:
            if record.get(field) != expected_record[field]:
                raise ValueError(f"frozen slot changed: {record_id}:{field}")
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
                raise ValueError(f"frozen provenance changed: {record_id}:{field}")
        method = provenance.get("authoring_method")
        allowed_methods = set(AUTHORING_METHODS)
        if require_complete and method not in allowed_methods:
            raise ValueError(f"invalid authoring method: {record_id}")
        if not require_complete and method not in {"", *allowed_methods}:
            raise ValueError(f"invalid authoring method: {record_id}")
        if method:
            family_id = str(record["source_family_id"])
            previous = family_methods.setdefault(family_id, str(method))
            if previous != method:
                raise ValueError(f"family authoring method inconsistent: {family_id}")
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
        for field in (
            "text",
            "text_sha256",
            "normalized_text_sha256",
            "review_notes",
            "reviewer",
            "human_adjudicator",
        ):
            if not isinstance(record.get(field), str):
                raise TypeError(f"{field} must be a string: {record_id}")
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
            raise ValueError(
                f"human adjudication trigger is unmarked: {record_id}"
            )
        if require_complete:
            _validate_completed_record(record)
        validated.append(record)
    if observed != set(expected_by_id):
        raise ValueError(f"{dataset_role} authoring slots are incomplete")
    return tuple(validated)


def verifier_composition(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for target in PROTECTED_INTENTS:
        relevant = [
            row for row in records if row["protected_boundary_target"] == target
        ]
        positives = sum(
            row["intent"] == target and row["verifier_role"] == "protected_positive"
            for row in relevant
        )
        negatives = sum(
            row["intent"] == UNSUPPORTED_INTENT
            and row["verifier_role"] == "targeted_unsupported_negative"
            for row in relevant
        )
        report[target] = {
            "negative_record_count": negatives,
            "positive_record_count": positives,
            "record_count": positives + negatives,
        }
    return report


def validate_allocations(
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training_families = Counter(str(row["source_family_id"]) for row in training)
    evaluation_families = Counter(
        str(row["source_family_id"]) for row in evaluation
    )
    if set(training_families.values()) != {160} or len(training_families) != 3:
        raise ValueError("training family allocation changed")
    if set(evaluation_families.values()) != {320} or len(evaluation_families) != 2:
        raise ValueError("evaluation family allocation changed")
    for family in training_families:
        rows = [row for row in training if row["source_family_id"] == family]
        for target in PROTECTED_INTENTS:
            positive = sum(row["intent"] == target for row in rows)
            negative = sum(
                row["intent"] == UNSUPPORTED_INTENT
                and row["protected_boundary_target"] == target
                for row in rows
            )
            if (positive, negative) != (20, 20):
                raise ValueError(f"training verifier allocation changed: {family}")
    for family in evaluation_families:
        rows = [row for row in evaluation if row["source_family_id"] == family]
        if Counter(str(row["intent"]) for row in rows) != Counter(
            {intent: 40 for intent in PRIMARY_INTENTS}
        ):
            raise ValueError(f"evaluation intent allocation changed: {family}")
        unsupported = [row for row in rows if row["intent"] == UNSUPPORTED_INTENT]
        if Counter(
            str(row["protected_boundary_target"]) for row in unsupported
        ) != Counter({target: 10 for target in PROTECTED_INTENTS}):
            raise ValueError(f"evaluation boundary allocation changed: {family}")
    composition = verifier_composition(training)
    if any(
        row != {
            "negative_record_count": 60,
            "positive_record_count": 60,
            "record_count": 120,
        }
        for row in composition.values()
    ):
        raise ValueError("verifier training composition changed")
    return {
        "evaluation_counts_by_source_family": dict(sorted(evaluation_families.items())),
        "training_counts_by_source_family": dict(sorted(training_families.items())),
        "verifier_training_composition": composition,
    }


def review_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = sum(bool(row["requires_human_adjudication"]) for row in records)
    completed = sum(bool(row["human_adjudication_completed"]) for row in records)
    return {
        "ai_assisted_review_is_human_review": False,
        "approved_count": sum(row["review_status"] == "approved" for row in records),
        "human_adjudication_completed_count": completed,
        "human_adjudication_required_count": required,
        "record_count": len(records),
        "review_methods": dict(
            sorted(Counter(str(row["review_method"]) for row in records).items())
        ),
        "unresolved_human_adjudication_count": required - completed,
    }


def duplicate_and_leakage_report(
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    consumed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def digests(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[
        defaultdict[str, list[Mapping[str, Any]]],
        defaultdict[str, list[Mapping[str, Any]]],
    ]:
        exact: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        normalized: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            exact[text_sha256(str(row["text"]))].append(row)
            normalized[normalized_text_sha256(str(row["text"]))].append(row)
        return exact, normalized

    training_exact, training_normalized = digests(training)
    evaluation_exact, evaluation_normalized = digests(evaluation)
    development_exact = {str(row["text_sha256"]) for row in development}
    development_normalized = {
        str(row["normalized_text_sha256"]) for row in development
    }
    consumed_exact = {str(row["text_sha256"]) for row in consumed}
    consumed_normalized = {
        str(row["normalized_text_sha256"]) for row in consumed
    }
    combined = [*training, *evaluation]
    intents_by_normalized: defaultdict[str, set[str]] = defaultdict(set)
    for row in combined:
        intents_by_normalized[normalized_text_sha256(str(row["text"]))].add(
            str(row["intent"])
        )
    counts = {
        "cross_intent_normalized_collision_count": sum(
            len(intents) > 1 for intents in intents_by_normalized.values()
        ),
        "evaluation_consumed_exact_overlap_count": len(
            set(evaluation_exact) & consumed_exact
        ),
        "evaluation_consumed_normalized_overlap_count": len(
            set(evaluation_normalized) & consumed_normalized
        ),
        "evaluation_development_exact_overlap_count": len(
            set(evaluation_exact) & development_exact
        ),
        "evaluation_development_normalized_overlap_count": len(
            set(evaluation_normalized) & development_normalized
        ),
        "evaluation_within_exact_duplicate_count": sum(
            len(rows) > 1 for rows in evaluation_exact.values()
        ),
        "evaluation_within_normalized_duplicate_count": sum(
            len(rows) > 1 for rows in evaluation_normalized.values()
        ),
        "training_consumed_exact_overlap_count": len(
            set(training_exact) & consumed_exact
        ),
        "training_consumed_normalized_overlap_count": len(
            set(training_normalized) & consumed_normalized
        ),
        "training_development_exact_overlap_count": len(
            set(training_exact) & development_exact
        ),
        "training_development_normalized_overlap_count": len(
            set(training_normalized) & development_normalized
        ),
        "training_evaluation_exact_overlap_count": len(
            set(training_exact) & set(evaluation_exact)
        ),
        "training_evaluation_normalized_overlap_count": len(
            set(training_normalized) & set(evaluation_normalized)
        ),
        "training_within_exact_duplicate_count": sum(
            len(rows) > 1 for rows in training_exact.values()
        ),
        "training_within_normalized_duplicate_count": sum(
            len(rows) > 1 for rows in training_normalized.values()
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


def validate_identity_and_groups(
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    sources: FrozenSources,
) -> dict[str, Any]:
    training_ids = [str(row["record_id"]) for row in training]
    evaluation_ids = [str(row["record_id"]) for row in evaluation]
    training_groups = [str(row["group_id"]) for row in training]
    evaluation_groups = [str(row["group_id"]) for row in evaluation]
    if len(training_ids) != len(set(training_ids)) or len(evaluation_ids) != len(
        set(evaluation_ids)
    ):
        raise ValueError("new record IDs must be unique")
    if len(training_groups) != len(set(training_groups)) or len(
        evaluation_groups
    ) != len(set(evaluation_groups)):
        raise ValueError("new group IDs must be unique")
    if set(training_ids) & set(evaluation_ids):
        raise ValueError("training and evaluation record IDs overlap")
    if set(training_groups) & set(evaluation_groups):
        raise ValueError("training and evaluation group IDs overlap")
    if (set(training_ids) | set(evaluation_ids)) & sources.historical_ids:
        raise ValueError("new record ID collides with historical evidence")
    if (
        set(training_groups) | set(evaluation_groups)
    ) & sources.historical_group_ids:
        raise ValueError("new group ID collides with historical evidence")
    return {
        "cross_role_group_collision_count": 0,
        "cross_role_record_id_collision_count": 0,
        "historical_group_collision_count": 0,
        "historical_record_id_collision_count": 0,
        "new_group_ids_unique": True,
        "new_record_ids_unique": True,
    }


def load_workfile(
    path: Path, paths: BuildPaths, reader: BytesReader
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
    allocation = validate_allocations(training, evaluation)
    identity = validate_identity_and_groups(training, evaluation, sources)
    if require_complete:
        leakage = duplicate_and_leakage_report(
            training,
            evaluation,
            sources.development_records,
            sources.consumed_evaluation_records,
        )
    else:
        leakage = {
            "all_duplicate_and_leakage_checks_passed": False,
            "not_run_until_text_is_complete": True,
        }
    return AuthoringInputs(
        training_bytes=training_bytes,
        training_records=training,
        evaluation_bytes=evaluation_bytes,
        evaluation_records=evaluation,
        validation_report={
            "allocation": allocation,
            "duplicate_and_leakage": leakage,
            "identity_and_group_isolation": identity,
            "review": {
                "evaluation": review_report(evaluation),
                "training": review_report(training),
            },
        },
    )


def count_by(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in records).items()))


def authored_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(record))


def development_training_record(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record["authoring_provenance"]
    return {
        "authoring_batch_id": provenance["authoring_batch_id"],
        "authoring_method": provenance["authoring_method"],
        "authoring_strategy": record["authoring_strategy"],
        "data_role": "development",
        "dataset_role": TRAINING_ROLE,
        "example_id": record["record_id"],
        "group_id": record["group_id"],
        "independently_authored": record["independently_authored"],
        "intent": record["intent"],
        "normalized_text_sha256": record["normalized_text_sha256"],
        "polarity": record["polarity"],
        "prediction_informed_authoring": record["prediction_informed_authoring"],
        "protected_boundary_target": record["protected_boundary_target"],
        "record_id": record["record_id"],
        "review_method": record["review_method"],
        "review_status": record["review_status"],
        "risk": record["risk_level"],
        "risk_level": record["risk_level"],
        "source_domain": "sentinelvoice_v2c6_r3_protected_intent_gate",
        "source_family_id": record["source_family_id"],
        "source_family_independence_basis": provenance[
            "source_family_independence_basis"
        ],
        "source_id": record["source_family_id"],
        "source_revision": provenance["source_revision"],
        "source_split": "development",
        "text": record["text"],
        "text_sha256": record["text_sha256"],
        "v2c6_lineage": "PROTECTED_INTENT_GATE_R3_ADDITION",
        "verifier_role": record["verifier_role"],
    }


def build_training_payload(inputs: AuthoringInputs) -> dict[str, Any]:
    records = [
        authored_output_record(row)
        for row in sorted(inputs.training_records, key=lambda row: row["record_id"])
    ]
    return {
        "classifier_input_fields": ["text"],
        "dataset_role": TRAINING_ROLE,
        "example_count": len(records),
        "examples": records,
        "governance": {**GOVERNANCE_FLAGS, "all_records_approved": True},
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "phase": WORKFLOW_PHASE,
        "schema_version": TRAINING_OUTPUT_SCHEMA,
    }


def build_development_payload(
    sources: FrozenSources, inputs: AuthoringInputs, training_sha256: str
) -> dict[str, Any]:
    existing = [copy.deepcopy(row) for row in sources.development_records]
    additions = [
        development_training_record(row)
        for row in sorted(inputs.training_records, key=lambda row: row["record_id"])
    ]
    records = [*existing, *additions]
    return {
        "classifier_input_fields": ["text"],
        "dataset_version": "v2c6-r3-protected-intent-gate-development.v1",
        "example_count": len(records),
        "examples": records,
        "governance": {
            **GOVERNANCE_FLAGS,
            "existing_9608_records_mutated": False,
            "fresh_evaluation_records_included": False,
        },
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "ordering": "frozen_9608_order_then_new_training_record_id",
        "phase": WORKFLOW_PHASE,
        "schema_version": DEVELOPMENT_OUTPUT_SCHEMA,
        "source_artifacts": {
            "existing_development": sources.source_lineage[
                "existing_development_dataset"
            ],
            "r3_training_examples": {
                "path": (
                    "data/evals/v2/ml/"
                    "v2c6_r3_protected_intent_gate_training_examples.json"
                ),
                "sha256": training_sha256,
            },
        },
    }


def build_evaluation_payload(inputs: AuthoringInputs) -> dict[str, Any]:
    records = [
        authored_output_record(row)
        for row in sorted(inputs.evaluation_records, key=lambda row: row["record_id"])
    ]
    return {
        "classifier_input_fields": ["text"],
        "dataset_role": EVALUATION_ROLE,
        "example_count": len(records),
        "examples": records,
        "fresh_evaluation_governance": {
            "candidate_definitions_revised_from_this_evaluation": False,
            "candidate_outputs_inspected_during_authoring": False,
            "consumed_r2_evaluation_used_as_paraphrase_source": False,
            "excluded_from_candidate_fitting": True,
            "fresh_families_consulted_each_other": False,
            "independently_authored": True,
            "not_training_data": True,
            "other_r3_fresh_family_wording_consulted": False,
            "prediction_informed_authoring": False,
            "r3_training_used_as_paraphrase_source": False,
        },
        "governance": {**GOVERNANCE_FLAGS, "all_records_approved": True},
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "phase": WORKFLOW_PHASE,
        "schema_version": EVALUATION_OUTPUT_SCHEMA,
    }


def common_manifest(
    sources: FrozenSources,
    inputs: AuthoringInputs,
    paths: BuildPaths,
    reader: BytesReader,
) -> dict[str, Any]:
    return {
        "builder": {
            "path": display_path(paths.builder, paths),
            "sha256": sha256_bytes(guarded_read_bytes(paths.builder, paths, reader)),
            "version": BUILDER_VERSION,
        },
        "contract": {
            "path": display_path(paths.contract, paths),
            "schema_version": (
                "v2c6-protected-intent-gate-remediation-design-contract.v1"
            ),
            "sha256": sources.contract_sha256,
        },
        "consumed_r2_lineage": sources.source_lineage[
            "consumed_r2_fresh_dataset"
        ],
        "existing_development": sources.source_lineage[
            "existing_development_dataset"
        ],
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
        "source_lineage": sources.source_lineage,
        "validation": inputs.validation_report,
    }


def build_artifacts(
    sources: FrozenSources,
    inputs: AuthoringInputs,
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> BuiltArtifacts:
    training = build_training_payload(inputs)
    training_bytes = stable_json_bytes(training)
    development = build_development_payload(
        sources, inputs, sha256_bytes(training_bytes)
    )
    development_bytes = stable_json_bytes(development)
    evaluation = build_evaluation_payload(inputs)
    evaluation_bytes = stable_json_bytes(evaluation)
    common = common_manifest(sources, inputs, paths, reader)
    training_manifest = {
        **common,
        "artifact": {
            "path": display_path(paths.training_output, paths),
            "schema_version": TRAINING_OUTPUT_SCHEMA,
            "sha256": sha256_bytes(training_bytes),
        },
        "counts": {
            "by_intent": count_by(inputs.training_records, "intent"),
            "by_source_family": count_by(inputs.training_records, "source_family_id"),
            "record_count": 480,
        },
        "schema_version": TRAINING_MANIFEST_SCHEMA,
        "verifier_training_composition": verifier_composition(
            inputs.training_records
        ),
    }
    development_manifest = {
        **common,
        "artifact": {
            "path": display_path(paths.development_output, paths),
            "schema_version": DEVELOPMENT_OUTPUT_SCHEMA,
            "sha256": sha256_bytes(development_bytes),
        },
        "counts": {
            "existing_record_count": 9608,
            "fresh_evaluation_record_count": 0,
            "new_training_record_count": 480,
            "total_record_count": 10088,
        },
        "schema_version": DEVELOPMENT_MANIFEST_SCHEMA,
    }
    evaluation_manifest = {
        **common,
        "artifact": {
            "path": display_path(paths.evaluation_output, paths),
            "schema_version": EVALUATION_OUTPUT_SCHEMA,
            "sha256": sha256_bytes(evaluation_bytes),
        },
        "counts": {
            "by_intent": count_by(inputs.evaluation_records, "intent"),
            "by_source_family": count_by(inputs.evaluation_records, "source_family_id"),
            "record_count": 640,
        },
        "fresh_evaluation_governance": evaluation[
            "fresh_evaluation_governance"
        ],
        "schema_version": EVALUATION_MANIFEST_SCHEMA,
    }
    payloads = (
        training,
        training_manifest,
        development,
        development_manifest,
        evaluation,
        evaluation_manifest,
    )
    contents = tuple(stable_json_bytes(payload) for payload in payloads)
    artifacts = BuiltArtifacts(contents=contents, payloads=payloads)
    validate_built_artifacts(artifacts, sources)
    return artifacts


def validate_built_artifacts(
    artifacts: BuiltArtifacts, sources: FrozenSources
) -> None:
    (
        training,
        training_manifest,
        development,
        development_manifest,
        evaluation,
        evaluation_manifest,
    ) = artifacts.payloads
    if training["example_count"] != 480 or len(training["examples"]) != 480:
        raise ValueError("R3 training output must contain exactly 480 records")
    if (
        development["example_count"] != 10088
        or len(development["examples"]) != 10088
    ):
        raise ValueError("R3 development output must contain exactly 10088 records")
    if evaluation["example_count"] != 640 or len(evaluation["examples"]) != 640:
        raise ValueError("R3 evaluation output must contain exactly 640 records")
    if development["examples"][:9608] != list(sources.development_records):
        raise ValueError("existing 9608 development records changed")
    evaluation_ids = {row["record_id"] for row in evaluation["examples"]}
    development_ids = {
        row.get("example_id", row.get("record_id"))
        for row in development["examples"]
    }
    if evaluation_ids & development_ids:
        raise ValueError("fresh evaluation records entered development training")
    for manifest, content in zip(
        (training_manifest, development_manifest, evaluation_manifest),
        (artifacts.contents[0], artifacts.contents[2], artifacts.contents[4]),
        strict=True,
    ):
        if manifest["artifact"]["sha256"] != sha256_bytes(content):
            raise ValueError("manifest artifact hash mismatch")


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
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing file: {path}"
            ) from exc
        temporary.unlink()
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def preflight(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    loaded = sources or load_frozen_sources(paths, reader)
    training_present = paths.training_workfile.exists()
    evaluation_present = paths.evaluation_workfile.exists()
    if training_present != evaluation_present:
        raise ValueError("authoring workfiles must both exist or both be absent")
    ready = False
    validation: dict[str, Any] | None = None
    if training_present:
        inputs = load_authoring_inputs(
            loaded, paths, reader, require_complete=False
        )
        training_payload, _ = load_workfile(paths.training_workfile, paths, reader)
        evaluation_payload, _ = load_workfile(
            paths.evaluation_workfile, paths, reader
        )
        if (
            training_payload["authoring_complete"] is True
            and evaluation_payload["authoring_complete"] is True
        ):
            inputs = load_authoring_inputs(
                loaded, paths, reader, require_complete=True
            )
            ready = True
        validation = inputs.validation_report
    return {
        "authoring_workfiles_present": training_present and evaluation_present,
        "files_written": False,
        "existing_development_record_count": len(loaded.development_records),
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
        "evaluation_slot_count": 640,
        "files_written": True,
        "governance": GOVERNANCE_FLAGS,
        "paths": [display_path(path, paths) for path in workfiles],
        "text_slots_populated": 0,
        "tracked_files_written": False,
        "training_slot_count": 480,
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
        loaded, paths, reader, require_complete=True
    )
    artifacts = build_artifacts(loaded, inputs, paths, reader)
    write_artifacts(artifacts, paths)
    return {
        "expanded_development_record_count": 10088,
        "files_written": True,
        "fresh_evaluation_record_count": 640,
        "governance": GOVERNANCE_FLAGS,
        "output_paths": [display_path(path, paths) for path in output_paths(paths)],
        "targeted_training_record_count": 480,
    }


def check_results(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    loaded = sources or load_frozen_sources(paths, reader)
    inputs = load_authoring_inputs(
        loaded, paths, reader, require_complete=True
    )
    expected = build_artifacts(loaded, inputs, paths, reader)
    actual = tuple(
        guarded_read_bytes(path, paths, reader) for path in output_paths(paths)
    )
    if actual != expected.contents:
        raise ValueError("existing outputs are not the deterministic R3 build")
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
    arguments = build_parser().parse_args(argv)
    if arguments.preflight:
        result = preflight()
    elif arguments.prepare_authoring_workfiles:
        result = prepare_authoring_workfiles()
    elif arguments.build:
        result = build()
    else:
        result = check_results()
    print(stable_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
