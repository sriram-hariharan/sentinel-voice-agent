"""Validate and build the deterministic V2-C6 remediated development data.

The consumed V2-C5 final holdout is prohibited on every code path. This module
contains no model, embedding, inference, selection, or evaluation behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = "scripts/build_v2c6_remediated_development_dataset.py"

DATASET_CONTRACT_SHA256 = (
    "2f251cb814f06058dd24a8b86b207fa85b14df98a5c17f2853bed50ffea40155"
)
DESIGN_CONTRACT_SHA256 = (
    "ffa8ec7f20b99cc22da3473dd50d32c7aa611cf548ecb205d5c53b9e5409a442"
)
DIAGNOSIS_SHA256 = (
    "58177bd3aacf52bbff2268fd834e883457e8241f36e35f5e84943f196f688536"
)
DIAGNOSIS_MANIFEST_SHA256 = (
    "bb04f4bf82704331ab4cd0c1dc5ed3a9d87761a30013315e8c52055b3c550305"
)
DEVELOPMENT_DATASET_SHA256 = (
    "dce97c0a3bfcdf0dee93d784ded24d3568f7df8f513871995428ff35d030bb07"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "89fea89d793cf5e027b059c331533ec45c6d59fabbd49e9245e0d055ae946a16"
)
TAXONOMY_SHA256 = (
    "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
)
TAXONOMY_MANIFEST_SHA256 = (
    "359eb38e5ff63af0c9cc0f5db19951b8626f9e15cb136f23046648d655a0f7a9"
)

AUTHORING_INPUT_SCHEMA = "v2c6-remediation-authoring-input.v1"
REMEDIATION_SCHEMA = "v2c6-remediation-examples.v1"
REMEDIATION_MANIFEST_SCHEMA = "v2c6-remediation-examples-manifest.v1"
COMBINED_DATASET_SCHEMA = "v2c6-remediated-development-dataset.v1"
COMBINED_MANIFEST_SCHEMA = "v2c6-remediated-development-dataset-manifest.v1"
BUILDER_VERSION = "v2c6-remediated-development-dataset-builder.v1"
NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"
EXPECTED_DEVELOPMENT_COUNT = 8198
UNSUPPORTED_INTENT = "unsupported_or_uncertain"

PRIMARY_INTENTS = (
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    UNSUPPORTED_INTENT,
)
PROTECTED_WRITE_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)
UNSUPPORTED_SUBTYPES = (
    "truly_unsupported_banking_request",
    "ambiguous_or_insufficient_information",
    "adjacent_but_unsupported_intent",
    "supported_intent_hard_negative",
    "off_domain_or_noise",
)
REVIEW_STATUSES = ("unreviewed", "approved", "rejected", "needs_revision")
APPROVED_AUTHORING_METHODS = (
    "human_authored",
    "controlled_llm_assisted",
    (
        "deterministic_scenario_or_template_generation_when_semantic_"
        "independence_is_preserved"
    ),
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
REQUIRED_RECORD_FIELDS = (
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
)
PROHIBITED_INDEPENDENCE_BASIS_PHRASES = (
    "duplicate template",
    "paraphrase",
    "punctuation change",
    "random seed",
    "same prompt",
    "shuffle",
    "superficial style",
)
GOVERNANCE_FLAGS = {
    "embeddings_generated": False,
    "model_inference_performed": False,
    "model_selection_performed": False,
    "model_training_performed": False,
    "runtime_behavior_changed": False,
    "threshold_tuning_performed": False,
    "v2c5_raw_final_holdout_accessed": False,
}

BytesReader = Callable[[Path], bytes]


@dataclass(frozen=True)
class BuildPaths:
    repository_root: Path
    dataset_contract: Path
    design_contract: Path
    diagnosis: Path
    diagnosis_manifest: Path
    development_dataset: Path
    development_manifest: Path
    taxonomy: Path
    taxonomy_manifest: Path
    authoring_input: Path
    remediation_output: Path
    remediation_manifest_output: Path
    combined_output: Path
    combined_manifest_output: Path
    prohibited_holdout: Path
    builder: Path


DEFAULT_PATHS = BuildPaths(
    repository_root=REPOSITORY_ROOT,
    dataset_contract=ML_DIRECTORY / "v2c6_remediation_dataset_contract.json",
    design_contract=ML_DIRECTORY / "v2c6_remediation_design_contract.json",
    diagnosis=ML_DIRECTORY / "v2c6_generalization_diagnosis.json",
    diagnosis_manifest=(
        ML_DIRECTORY / "v2c6_generalization_diagnosis.manifest.json"
    ),
    development_dataset=(
        ML_DIRECTORY / "v2c5_expanded_development_dataset.json"
    ),
    development_manifest=(
        ML_DIRECTORY / "v2c5_expanded_development_dataset.manifest.json"
    ),
    taxonomy=ML_DIRECTORY / "v2c5_taxonomy_freeze.json",
    taxonomy_manifest=ML_DIRECTORY / "v2c5_taxonomy_freeze.manifest.json",
    authoring_input=(
        ML_DIRECTORY / "local/v2c6_remediation_authoring_workfile.json"
    ),
    remediation_output=ML_DIRECTORY / "v2c6_remediation_examples.json",
    remediation_manifest_output=(
        ML_DIRECTORY / "v2c6_remediation_examples.manifest.json"
    ),
    combined_output=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.json"
    ),
    combined_manifest_output=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.manifest.json"
    ),
    prohibited_holdout=ML_DIRECTORY / "v2c5_final_holdout.json",
    builder=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
)


@dataclass(frozen=True)
class FrozenSources:
    dataset_contract: dict[str, Any]
    design_contract: dict[str, Any]
    diagnosis: dict[str, Any]
    diagnosis_manifest: dict[str, Any]
    development_dataset: dict[str, Any]
    development_manifest: dict[str, Any]
    taxonomy: dict[str, Any]
    taxonomy_manifest: dict[str, Any]
    development_examples: tuple[dict[str, Any], ...]
    intent_labels: tuple[str, ...]
    risk_by_intent: dict[str, str]
    source_artifacts: dict[str, dict[str, str]]


@dataclass(frozen=True)
class AuthoringValidation:
    input_payload: dict[str, Any]
    input_bytes: bytes
    records: tuple[dict[str, Any], ...]
    approved_records: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class BuiltArtifacts:
    remediation_bytes: bytes
    remediation_manifest_bytes: bytes
    combined_bytes: bytes
    combined_manifest_bytes: bytes
    remediation_payload: dict[str, Any]
    remediation_manifest_payload: dict[str, Any]
    combined_payload: dict[str, Any]
    combined_manifest_payload: dict[str, Any]


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def filesystem_reader(path: Path) -> bytes:
    return path.read_bytes()


def guard_consumed_holdout(path: Path, paths: BuildPaths) -> None:
    if path.resolve() == paths.prohibited_holdout.resolve():
        raise PermissionError("consumed V2-C5 final holdout access is prohibited")


def guarded_read_bytes(
    path: Path,
    paths: BuildPaths,
    reader: BytesReader = filesystem_reader,
) -> bytes:
    guard_consumed_holdout(path, paths)
    return reader(path)


def sha256_file(
    path: Path,
    paths: BuildPaths,
    reader: BytesReader = filesystem_reader,
) -> str:
    return sha256_bytes(guarded_read_bytes(path, paths, reader))


def read_json_object(
    path: Path,
    paths: BuildPaths,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    payload = json.loads(guarded_read_bytes(path, paths, reader))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def display_path(path: Path, paths: BuildPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


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
    return sha256_bytes(normalized.encode("utf-8"))


def _require_nonempty_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{label} must be a list of objects")
    return value


def source_path_items(
    paths: BuildPaths,
) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("dataset_contract", paths.dataset_contract, DATASET_CONTRACT_SHA256),
        ("design_contract", paths.design_contract, DESIGN_CONTRACT_SHA256),
        ("diagnosis", paths.diagnosis, DIAGNOSIS_SHA256),
        (
            "diagnosis_manifest",
            paths.diagnosis_manifest,
            DIAGNOSIS_MANIFEST_SHA256,
        ),
        (
            "development_dataset",
            paths.development_dataset,
            DEVELOPMENT_DATASET_SHA256,
        ),
        (
            "development_manifest",
            paths.development_manifest,
            DEVELOPMENT_MANIFEST_SHA256,
        ),
        ("taxonomy", paths.taxonomy, TAXONOMY_SHA256),
        ("taxonomy_manifest", paths.taxonomy_manifest, TAXONOMY_MANIFEST_SHA256),
    )


def validate_dataset_contract(contract: Mapping[str, Any], paths: BuildPaths) -> None:
    if (
        contract.get("schema_version") != "v2c6-remediation-dataset-contract.v1"
        or contract.get("contract_version")
        != "v2c6-remediation-dataset-contract.v1"
        or contract.get("phase") != "V2-C6 Step 29D"
    ):
        raise ValueError("unexpected V2-C6 remediation dataset contract identity")
    if tuple(contract.get("primary_remediation_intents", [])) != PRIMARY_INTENTS:
        raise ValueError("primary remediation intent set changed")
    if tuple(contract.get("protected_write_policy", {}).get("intents", [])) != (
        PROTECTED_WRITE_INTENTS
    ):
        raise ValueError("protected-write intent set changed")
    if tuple(contract.get("unsupported_subtypes", {}).get("categories", [])) != (
        UNSUPPORTED_SUBTYPES
    ):
        raise ValueError("unsupported subtype set changed")
    if tuple(contract.get("human_review_policy", {}).get(
        "review_status_allowlist", []
    )) != REVIEW_STATUSES:
        raise ValueError("review status allowlist changed")
    if tuple(contract.get("authoring_method_policy", {}).get(
        "allowed_methods", []
    )) != APPROVED_AUTHORING_METHODS:
        raise ValueError("authoring method allowlist changed")
    pairs = tuple(
        tuple(pair) for pair in contract.get("hard_negative_policy", {}).get(
            "required_pairs", []
        )
    )
    if pairs != HARD_NEGATIVE_PAIRS:
        raise ValueError("hard-negative pair set changed")
    required = tuple(contract.get("record_schema", {}).get("required_fields", []))
    if required != REQUIRED_RECORD_FIELDS:
        raise ValueError("required remediation record schema changed")
    status = contract.get("contract_status", {})
    expected_status = {
        "authoring_started": False,
        "contract_frozen": True,
        "development_dataset_frozen": False,
        "embeddings_generated": False,
        "examples_authored": False,
        "examples_reviewed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "next_required": "v2c6_remediation_authoring_and_build",
        "remediation_dataset_built": False,
        "runtime_behavior_changed": False,
        "v2c5_raw_final_holdout_accessed": False,
    }
    if status != expected_status:
        raise ValueError("Step 29D contract status changed")
    if contract.get("duplication_policy", {}).get(
        "embedding_or_semantic_near_duplicate_threshold_authorized"
    ) is not False:
        raise ValueError("embedding duplicate checks are not authorized")
    authoring = contract.get("authoring_method_policy", {})
    if (
        authoring.get("initial_authoring_may_use_current_classifier_predictions")
        is not False
        or authoring.get(
            "iterative_generation_until_current_classifier_is_right_or_wrong"
        )
        is not False
    ):
        raise ValueError("model-in-the-loop authoring became permitted")
    expected_outputs = {
        display_path(paths.remediation_output, paths),
        display_path(paths.remediation_manifest_output, paths),
        display_path(paths.combined_output, paths),
        display_path(paths.combined_manifest_output, paths),
    }
    outputs = {
        item.get("path")
        for item in contract.get("future_outputs", {}).get("tracked_outputs", [])
    }
    if outputs != expected_outputs:
        raise ValueError("future output paths changed")
    workfile = contract.get("future_outputs", {}).get("authoring_workfile", {})
    if (
        workfile.get("path") != display_path(paths.authoring_input, paths)
        or workfile.get("tracked") is not False
    ):
        raise ValueError("local authoring workfile contract changed")


def validate_design_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != "v2c6-remediation-design-contract.v1"
        or contract.get("phase") != "V2-C6 Step 29C"
        or tuple(contract.get("primary_remediation_intents", []))
        != PRIMARY_INTENTS
        or tuple(contract.get("protected_write_policy", {}).get("intents", []))
        != PROTECTED_WRITE_INTENTS
    ):
        raise ValueError("frozen remediation design changed")


def validate_diagnosis_lineage(
    diagnosis: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if (
        diagnosis.get("schema_version") != "v2c6-generalization-diagnosis.v1"
        or diagnosis.get("phase") != "V2-C6 Step 29B"
        or manifest.get("schema_version")
        != "v2c6-generalization-diagnosis-manifest.v1"
        or manifest.get("phase") != "V2-C6 Step 29B"
    ):
        raise ValueError("frozen diagnosis lineage changed")


def validate_taxonomy(
    taxonomy: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, str]]:
    labels = taxonomy.get("intent_label_order")
    risks = taxonomy.get("risk_by_intent")
    if (
        taxonomy.get("schema_version") != "v2c5-taxonomy-freeze.v1"
        or taxonomy.get("final_taxonomy_frozen") is not True
        or not isinstance(labels, list)
        or len(labels) != 16
        or len(set(labels)) != 16
        or tuple(taxonomy.get("protected_write_intents", []))
        != PROTECTED_WRITE_INTENTS
        or not isinstance(risks, dict)
        or set(risks) != set(labels)
    ):
        raise ValueError("frozen V2-C5 taxonomy changed")
    if manifest.get("taxonomy_freeze") != {
        "path": "data/evals/v2/ml/v2c5_taxonomy_freeze.json",
        "sha256": TAXONOMY_SHA256,
    }:
        raise ValueError("taxonomy manifest binding changed")
    return tuple(str(value) for value in labels), {
        str(key): str(value) for key, value in risks.items()
    }


def validate_development_dataset(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    labels: Sequence[str],
    risks: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    examples = _require_object_list(dataset.get("examples"), "development examples")
    if (
        dataset.get("schema_version") != "v2c5-expanded-development-dataset.v1"
        or dataset.get("example_count") != EXPECTED_DEVELOPMENT_COUNT
        or len(examples) != EXPECTED_DEVELOPMENT_COUNT
        or dataset.get("normalization_version") != NORMALIZATION_VERSION
    ):
        raise ValueError("frozen expanded development dataset changed")
    identifiers: set[str] = set()
    counts: Counter[str] = Counter()
    for index, record in enumerate(examples):
        example_id = _require_nonempty_string(record, "example_id")
        _require_nonempty_string(record, "group_id")
        text = _require_nonempty_string(record, "text")
        intent = record.get("intent")
        if example_id in identifiers:
            raise ValueError(f"duplicate frozen development ID: {example_id}")
        if intent not in labels or record.get("risk") != risks.get(str(intent)):
            raise ValueError(f"development record {index} taxonomy binding changed")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError(f"development record {index} text hash changed")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError(f"development record {index} normalized hash changed")
        identifiers.add(example_id)
        counts[str(intent)] += 1
    if manifest.get("dataset") != {
        "path": "data/evals/v2/ml/v2c5_expanded_development_dataset.json",
        "schema_version": "v2c5-expanded-development-dataset.v1",
        "sha256": DEVELOPMENT_DATASET_SHA256,
        "version": "v2c5-expanded-development.v1",
    }:
        raise ValueError("development manifest binding changed")
    if manifest.get("counts", {}).get("final_intent_counts") != dict(counts):
        raise ValueError("development intent counts changed")
    return tuple(examples)


def load_frozen_sources(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> FrozenSources:
    payloads: dict[str, dict[str, Any]] = {}
    source_artifacts: dict[str, dict[str, str]] = {}
    for name, path, expected_hash in source_path_items(paths):
        value = guarded_read_bytes(path, paths, reader)
        if sha256_bytes(value) != expected_hash:
            raise ValueError(f"frozen source hash changed: {name}")
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError(f"frozen source must be an object: {name}")
        payloads[name] = payload
        source_artifacts[name] = {
            "path": display_path(path, paths),
            "sha256": expected_hash,
        }
    validate_dataset_contract(payloads["dataset_contract"], paths)
    validate_design_contract(payloads["design_contract"])
    validate_diagnosis_lineage(
        payloads["diagnosis"], payloads["diagnosis_manifest"]
    )
    labels, risks = validate_taxonomy(
        payloads["taxonomy"], payloads["taxonomy_manifest"]
    )
    examples = validate_development_dataset(
        payloads["development_dataset"],
        payloads["development_manifest"],
        labels,
        risks,
    )
    return FrozenSources(
        dataset_contract=payloads["dataset_contract"],
        design_contract=payloads["design_contract"],
        diagnosis=payloads["diagnosis"],
        diagnosis_manifest=payloads["diagnosis_manifest"],
        development_dataset=payloads["development_dataset"],
        development_manifest=payloads["development_manifest"],
        taxonomy=payloads["taxonomy"],
        taxonomy_manifest=payloads["taxonomy_manifest"],
        development_examples=examples,
        intent_labels=labels,
        risk_by_intent=risks,
        source_artifacts=source_artifacts,
    )


def validate_authoring_input_identity(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != AUTHORING_INPUT_SCHEMA:
        raise ValueError("unexpected remediation authoring input schema")
    if payload.get("phase") != "V2-C6 Step 29E2":
        raise ValueError("unexpected remediation authoring phase")
    return _require_object_list(payload.get("records"), "authoring records")


def validate_record_structure(
    records: Sequence[Mapping[str, Any]],
    sources: FrozenSources,
) -> tuple[dict[str, Any], ...]:
    record_ids: set[str] = set()
    frozen_ids = {
        str(record["example_id"]) for record in sources.development_examples
    }
    family_descriptors: dict[str, tuple[str, str, str, str]] = {}
    descriptor_to_family: dict[tuple[str, str, str, str], str] = {}
    group_bindings: dict[str, tuple[str, str]] = {}
    validated: list[dict[str, Any]] = []
    allowed_pairs = {frozenset(pair) for pair in HARD_NEGATIVE_PAIRS}

    for index, source_record in enumerate(records):
        missing = [
            field for field in REQUIRED_RECORD_FIELDS if field not in source_record
        ]
        if missing:
            raise ValueError(f"authoring record {index} missing fields: {missing}")
        record = copy.deepcopy(dict(source_record))
        strings = {
            field: _require_nonempty_string(record, field)
            for field in REQUIRED_RECORD_FIELDS
            if field not in {"is_hard_negative"}
        }
        record_id = strings["record_id"]
        if record_id in record_ids:
            raise ValueError(f"duplicate remediation record ID: {record_id}")
        if record_id in frozen_ids:
            raise ValueError(f"remediation ID collides with development: {record_id}")
        record_ids.add(record_id)
        intent = strings["intent"]
        if intent not in PRIMARY_INTENTS:
            raise ValueError(f"remediation intent is not a primary target: {intent}")
        if strings["risk_level"] != sources.risk_by_intent[intent]:
            raise ValueError(f"risk_level disagrees with frozen taxonomy: {record_id}")
        if strings["review_status"] not in REVIEW_STATUSES:
            raise ValueError(f"invalid review status: {record_id}")
        if strings["authoring_method"] not in APPROVED_AUTHORING_METHODS:
            raise ValueError(f"invalid authoring method: {record_id}")
        if not isinstance(record.get("is_hard_negative"), bool):
            raise TypeError(f"is_hard_negative must be boolean: {record_id}")
        if intent == UNSUPPORTED_INTENT:
            subtype = record.get("unsupported_subtype")
            if subtype not in UNSUPPORTED_SUBTYPES:
                raise ValueError(f"invalid unsupported subtype: {record_id}")
        elif record.get("unsupported_subtype") not in {None, ""}:
            raise ValueError(f"supported record has unsupported subtype: {record_id}")
        if record["is_hard_negative"]:
            target = strings["boundary_target"]
            if target == intent or frozenset((intent, target)) not in allowed_pairs:
                raise ValueError(f"hard-negative pair is not frozen: {record_id}")
        normalized_text_sha256(strings["text"])

        family_id = strings["source_family_id"]
        normalized_basis = (
            strings["source_family_independence_basis"]
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        if any(
            phrase in normalized_basis
            for phrase in PROHIBITED_INDEPENDENCE_BASIS_PHRASES
        ):
            raise ValueError(
                f"source family basis is not independent: {family_id}"
            )
        descriptor = (
            strings["source_family_independence_basis"],
            strings["source_revision"],
            strings["authoring_batch_id"],
            strings["authoring_method"],
        )
        previous = family_descriptors.setdefault(family_id, descriptor)
        if previous != descriptor:
            raise ValueError(f"source family metadata is inconsistent: {family_id}")
        aliased_family = descriptor_to_family.setdefault(descriptor, family_id)
        if aliased_family != family_id:
            raise ValueError("multiple source family IDs use identical provenance")

        group_id = strings["group_id"]
        group_binding = (intent, family_id)
        previous_group = group_bindings.setdefault(group_id, group_binding)
        if previous_group != group_binding:
            raise ValueError(
                f"group crosses intent or source-family boundary: {group_id}"
            )
        validated.append(record)
    return tuple(validated)


def _audit_entries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "group_id": str(record["group_id"]),
            "intent": str(record["intent"]),
            "record_id": str(record.get("record_id", record.get("example_id"))),
            "source_family_id": str(
                record.get(
                    "source_family_id",
                    record.get("source_id", "frozen_v2c5_development"),
                )
            ),
        }
        for record in sorted(
            records,
            key=lambda item: str(item.get("record_id", item.get("example_id"))),
        )
    ]


def duplicate_report(
    approved_records: Sequence[Mapping[str, Any]],
    development_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_new: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized_new: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    exact_development: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized_development: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for record in approved_records:
        text = str(record["text"])
        exact_new[text_sha256(text)].append(record)
        normalized_new[normalized_text_sha256(text)].append(record)
    for record in development_examples:
        exact_development[str(record["text_sha256"])].append(record)
        normalized_development[str(record["normalized_text_sha256"])].append(record)

    exact_within = {
        digest: records for digest, records in exact_new.items() if len(records) > 1
    }
    normalized_within = {
        digest: records
        for digest, records in normalized_new.items()
        if len(records) > 1
    }
    exact_overlap = set(exact_new) & set(exact_development)
    normalized_overlap = set(normalized_new) & set(normalized_development)
    cross_intent: dict[str, list[Mapping[str, Any]]] = {}
    for digest, records in normalized_new.items():
        combined = list(records) + list(normalized_development.get(digest, []))
        if len({str(record["intent"]) for record in combined}) > 1:
            cross_intent[digest] = combined

    def new_cluster_payload(
        clusters: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "normalized_or_exact_sha256": digest,
                "records": _audit_entries(records),
            }
            for digest, records in sorted(clusters.items())
        ]

    def overlap_payload(digests: set[str]) -> list[dict[str, Any]]:
        return [
            {
                "development_example_ids": sorted(
                    str(record["example_id"])
                    for record in (
                        exact_development.get(digest, [])
                        or normalized_development.get(digest, [])
                    )
                ),
                "new_records": _audit_entries(
                    exact_new.get(digest, []) or normalized_new.get(digest, [])
                ),
                "normalized_or_exact_sha256": digest,
            }
            for digest in sorted(digests)
        ]

    passed = not (
        exact_within
        or normalized_within
        or exact_overlap
        or normalized_overlap
        or cross_intent
    )
    return {
        "cross_intent_normalized_conflict_count": len(cross_intent),
        "cross_intent_normalized_conflicts": new_cluster_payload(cross_intent),
        "duplicate_gates_passed": passed,
        "exact_development_overlap_count": len(exact_overlap),
        "exact_development_overlaps": overlap_payload(exact_overlap),
        "exact_within_new_cluster_count": len(exact_within),
        "exact_within_new_clusters": new_cluster_payload(exact_within),
        "normalized_development_overlap_count": len(normalized_overlap),
        "normalized_development_overlaps": overlap_payload(normalized_overlap),
        "normalized_within_new_cluster_count": len(normalized_within),
        "normalized_within_new_clusters": new_cluster_payload(normalized_within),
        "normalization_version": NORMALIZATION_VERSION,
        "raw_text_persisted_in_diagnostics": False,
        "semantic_or_embedding_similarity_used": False,
    }


def lexical_diversity_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    token_sequences = [
        re.findall(
            r"[^\W_]+(?:['’][^\W_]+)*",
            normalize_text(str(record["text"])),
            flags=re.UNICODE,
        )
        for record in records
    ]
    tokens = [token for sequence in token_sequences for token in sequence]
    bigrams = Counter(
        (sequence[index], sequence[index + 1])
        for sequence in token_sequences
        for index in range(len(sequence) - 1)
    )
    normalized_count = len(
        {normalized_text_sha256(str(record["text"])) for record in records}
    )
    group_count = len({str(record["group_id"]) for record in records})
    repeated_bigram_occurrences = sum(
        count for count in bigrams.values() if count > 1
    )
    return {
        "bigram_concentration": {
            "repeated_occurrence_share": (
                repeated_bigram_occurrences / sum(bigrams.values())
                if bigrams
                else 0.0
            ),
            "top_bigram_share": (
                max(bigrams.values()) / sum(bigrams.values()) if bigrams else 0.0
            ),
            "total_bigram_count": sum(bigrams.values()),
            "unique_bigram_count": len(bigrams),
        },
        "record_count": len(records),
        "token_diversity": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "total_token_count": len(tokens),
        "unique_group_ratio": group_count / len(records) if records else 0.0,
        "unique_normalized_text_ratio": (
            normalized_count / len(records) if records else 0.0
        ),
        "unique_token_count": len(set(tokens)),
    }


def source_family_report(
    approved_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for intent in PRIMARY_INTENTS:
        records = [record for record in approved_records if record["intent"] == intent]
        family_counts = Counter(str(record["source_family_id"]) for record in records)
        normalized = {normalized_text_sha256(str(record["text"])) for record in records}
        diversity_by_family = {
            family_id: lexical_diversity_report(
                [
                    record
                    for record in records
                    if record["source_family_id"] == family_id
                ]
            )
            for family_id in sorted(family_counts)
        }
        result[intent] = {
            "diversity_by_source_family": diversity_by_family,
            "dominant_wording_review": "human_review_required_no_automatic_threshold",
            "families_meeting_30_record_target": sum(
                count >= 30 for count in family_counts.values()
            ),
            "intent_diversity": lexical_diversity_report(records),
            "largest_source_family_share": (
                max(family_counts.values()) / len(records) if records else 0.0
            ),
            "per_family_30_record_target_met": bool(family_counts)
            and all(count >= 30 for count in family_counts.values()),
            "records_by_source_family": dict(sorted(family_counts.items())),
            "source_family_minimum_met": len(family_counts) >= 3,
            "total_new_records": len(records),
            "unique_group_count": len({str(record["group_id"]) for record in records}),
            "unique_normalized_text_count": len(normalized),
            "unique_source_family_count": len(family_counts),
        }
    return result


def group_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record["group_id"]) for record in records)
    values = list(counts.values())
    return {
        "group_semantic_independence_proven_automatically": False,
        "human_review_still_required": True,
        "maximum_records_per_group": max(values, default=0),
        "mean_records_per_group": statistics.fmean(values) if values else 0.0,
        "multi_record_group_count": sum(value > 1 for value in values),
        "record_count": len(records),
        "singleton_group_count": sum(value == 1 for value in values),
        "unique_group_count": len(counts),
    }


def hard_negative_report(
    approved_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for side_a, side_b in HARD_NEGATIVE_PAIRS:
        a_records = [
            record
            for record in approved_records
            if record["is_hard_negative"] is True
            and record["intent"] == side_a
            and record["boundary_target"] == side_b
        ]
        b_records = [
            record
            for record in approved_records
            if record["is_hard_negative"] is True
            and record["intent"] == side_b
            and record["boundary_target"] == side_a
        ]
        pairs.append(
            {
                "approved_count": len(a_records) + len(b_records),
                "coverage_complete": bool(a_records) and bool(b_records),
                "side_a": side_a,
                "side_a_count": len(a_records),
                "side_a_source_family_count": len(
                    {str(record["source_family_id"]) for record in a_records}
                ),
                "side_b": side_b,
                "side_b_count": len(b_records),
                "side_b_source_family_count": len(
                    {str(record["source_family_id"]) for record in b_records}
                ),
            }
        )
    return {
        "all_required_pairs_complete": all(
            pair["coverage_complete"] for pair in pairs
        ),
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def planning_volume_report(
    approved_records: Sequence[Mapping[str, Any]],
    source_report: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter(str(record["intent"]) for record in approved_records)
    supported = {
        intent: {
            "minimum_90_met": counts[intent] >= 90,
            "per_family_30_target_met": source_report[intent][
                "per_family_30_record_target_met"
            ],
            "record_count": counts[intent],
        }
        for intent in PRIMARY_INTENTS
        if intent != UNSUPPORTED_INTENT
    }
    return {
        "all_planning_targets_met": (
            all(value["minimum_90_met"] for value in supported.values())
            and all(
                value["per_family_30_target_met"] for value in supported.values()
            )
            and counts[UNSUPPORTED_INTENT] == 180
            and len(approved_records) == 810
        ),
        "planning_target_is_mandatory_build_gate": False,
        "supported_primary": supported,
        "supported_primary_count": sum(
            counts[intent]
            for intent in PRIMARY_INTENTS
            if intent != UNSUPPORTED_INTENT
        ),
        "supported_primary_target": 630,
        "total_approved_count": len(approved_records),
        "total_target": 810,
        "unsupported_count": counts[UNSUPPORTED_INTENT],
        "unsupported_target": 180,
    }


def source_aware_readiness_report(
    approved_records: Sequence[Mapping[str, Any]],
    source_report: Mapping[str, Mapping[str, Any]],
    duplicates: Mapping[str, Any],
    review_status_counts: Mapping[str, int],
) -> dict[str, Any]:
    family_to_intents: defaultdict[str, set[str]] = defaultdict(set)
    for record in approved_records:
        family_to_intents[str(record["source_family_id"])].add(str(record["intent"]))
    holdout_candidates = []
    for family_id in sorted(family_to_intents):
        remaining = {
            intent
            for intent in PRIMARY_INTENTS
            if any(
                record["intent"] == intent
                and record["source_family_id"] != family_id
                for record in approved_records
            )
        }
        if remaining == set(PRIMARY_INTENTS):
            holdout_candidates.append(family_id)
    source_minimum_met = all(
        source_report[intent]["source_family_minimum_met"]
        for intent in PRIMARY_INTENTS
    )
    review_complete = (
        review_status_counts.get("unreviewed", 0) == 0
        and review_status_counts.get("needs_revision", 0) == 0
    )
    return {
        "duplicate_gates_passed": duplicates["duplicate_gates_passed"],
        "group_ids_do_not_cross_prohibited_split_boundaries": True,
        "group_aware_cv_ready": bool(approved_records),
        "minimum_three_source_families_every_primary_intent": source_minimum_met,
        "only_approved_records_included": True,
        "review_gates_passed": review_complete,
        "source_family_distribution_per_intent": source_report,
        "source_family_independence_semantically_proven": False,
        "step29g_ready": (
            source_minimum_met
            and bool(holdout_candidates)
            and duplicates["duplicate_gates_passed"]
            and review_complete
        ),
        "whole_family_holdout_candidate_count": len(holdout_candidates),
        "whole_family_holdout_feasible_without_dropping_primary_intent": bool(
            holdout_candidates
        ),
    }


def enforce_mandatory_gates(report: Mapping[str, Any]) -> None:
    failures: list[str] = []
    if not report["duplicate_validation"]["duplicate_gates_passed"]:
        failures.append("duplicate_validation")
    if not all(
        value["source_family_minimum_met"]
        for value in report["source_family_statistics"].values()
    ):
        failures.append("minimum_source_families")
    if not report["hard_negative_coverage"]["all_required_pairs_complete"]:
        failures.append("hard_negative_coverage")
    if not report["unsupported_subtype_coverage_complete"]:
        failures.append("unsupported_subtype_coverage")
    if failures:
        raise ValueError("mandatory remediation gates failed: " + ", ".join(failures))


def validate_authoring_payload(
    payload: dict[str, Any],
    input_bytes: bytes,
    sources: FrozenSources,
    *,
    enforce_build_requirements: bool,
) -> AuthoringValidation:
    records = validate_record_structure(
        validate_authoring_input_identity(payload), sources
    )
    approved = tuple(
        sorted(
            (
                record
                for record in records
                if record["review_status"] == "approved"
            ),
            key=lambda record: str(record["record_id"]),
        )
    )
    review_counts = Counter(str(record["review_status"]) for record in records)
    review_summary = {
        status: review_counts.get(status, 0) for status in REVIEW_STATUSES
    }
    subtype_counts = Counter(
        str(record["unsupported_subtype"])
        for record in approved
        if record["intent"] == UNSUPPORTED_INTENT
    )
    subtype_summary = {
        subtype: subtype_counts.get(subtype, 0) for subtype in UNSUPPORTED_SUBTYPES
    }
    sources_report = source_family_report(approved)
    duplicates = duplicate_report(approved, sources.development_examples)
    hard_negatives = hard_negative_report(approved)
    report: dict[str, Any] = {
        "approved_record_count": len(approved),
        "duplicate_validation": duplicates,
        "excluded_record_count": len(records) - len(approved),
        "group_statistics": group_report(approved),
        "hard_negative_coverage": hard_negatives,
        "included_record_count": len(approved),
        "input_record_count": len(records),
        "planning_volume": planning_volume_report(approved, sources_report),
        "review_status_counts": review_summary,
        "source_family_statistics": sources_report,
        "source_independence_limitation": (
            "Structural provenance checks cannot prove true semantic independence; "
            "human review remains required."
        ),
        "unsupported_subtype_counts": subtype_summary,
        "unsupported_subtype_coverage_complete": all(
            subtype_summary[subtype] > 0 for subtype in UNSUPPORTED_SUBTYPES
        ),
    }
    report["source_aware_readiness"] = source_aware_readiness_report(
        approved, sources_report, duplicates, review_summary
    )
    report["mandatory_build_gates_passed"] = (
        duplicates["duplicate_gates_passed"]
        and all(
            value["source_family_minimum_met"]
            for value in sources_report.values()
        )
        and hard_negatives["all_required_pairs_complete"]
        and report["unsupported_subtype_coverage_complete"]
    )
    if enforce_build_requirements:
        enforce_mandatory_gates(report)
    return AuthoringValidation(
        input_payload=payload,
        input_bytes=input_bytes,
        records=records,
        approved_records=approved,
        report=report,
    )


def load_authoring_validation(
    sources: FrozenSources,
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    *,
    enforce_build_requirements: bool,
) -> AuthoringValidation:
    input_bytes = guarded_read_bytes(paths.authoring_input, paths, reader)
    payload = json.loads(input_bytes)
    if not isinstance(payload, dict):
        raise TypeError("remediation authoring input must be an object")
    return validate_authoring_payload(
        payload,
        input_bytes,
        sources,
        enforce_build_requirements=enforce_build_requirements,
    )


def approved_remediation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(record))
    output["normalized_text_sha256"] = normalized_text_sha256(str(record["text"]))
    output["text_sha256"] = text_sha256(str(record["text"]))
    return output


def combined_remediation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "authoring_batch_id": record["authoring_batch_id"],
        "authoring_method": record["authoring_method"],
        "boundary_target": record["boundary_target"],
        "data_role": "development",
        "example_id": record["record_id"],
        "group_id": record["group_id"],
        "intent": record["intent"],
        "is_hard_negative": record["is_hard_negative"],
        "normalized_text_sha256": normalized_text_sha256(str(record["text"])),
        "record_id": record["record_id"],
        "review_status": record["review_status"],
        "risk": record["risk_level"],
        "risk_level": record["risk_level"],
        "source_domain": "sentinelvoice_v2c6_remediation",
        "source_family_id": record["source_family_id"],
        "source_family_independence_basis": record[
            "source_family_independence_basis"
        ],
        "source_id": record["source_family_id"],
        "source_revision": record["source_revision"],
        "source_split": "development",
        "text": record["text"],
        "text_sha256": text_sha256(str(record["text"])),
        "unsupported_subtype": record.get("unsupported_subtype"),
        "v2c6_lineage": "REMEDIATION_ADDITION",
    }


def count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def build_remediation_payload(
    validation: AuthoringValidation,
    sources: FrozenSources,
) -> dict[str, Any]:
    records = [
        approved_remediation_record(record)
        for record in validation.approved_records
    ]
    return {
        "classifier_input_fields": ["text"],
        "example_count": len(records),
        "examples": records,
        "governance": {
            **GOVERNANCE_FLAGS,
            "authoring_completed_by_builder": False,
            "human_review_required": True,
            "only_approved_records_included": True,
        },
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "phase": "V2-C6 Step 29E2",
        "schema_version": REMEDIATION_SCHEMA,
        "source_contract": {
            "path": sources.source_artifacts["dataset_contract"]["path"],
            "sha256": DATASET_CONTRACT_SHA256,
        },
    }


def build_combined_payload(
    validation: AuthoringValidation,
    sources: FrozenSources,
    remediation_sha256: str,
) -> dict[str, Any]:
    old_records = [copy.deepcopy(record) for record in sources.development_examples]
    new_records = [
        combined_remediation_record(record)
        for record in validation.approved_records
    ]
    examples = old_records + new_records
    return {
        "classifier_input_fields": ["text"],
        "dataset_version": "v2c6-remediated-development.v1",
        "example_count": len(examples),
        "examples": examples,
        "governance": {
            **GOVERNANCE_FLAGS,
            "existing_v2c5_records_mutated": False,
            "future_final_holdout_eligible": False,
        },
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "ordering": "frozen_v2c5_order_then_v2c6_record_id",
        "phase": "V2-C6 Step 29E2",
        "schema_version": COMBINED_DATASET_SCHEMA,
        "source_artifacts": {
            "approved_remediation_examples": {
                "path": "data/evals/v2/ml/v2c6_remediation_examples.json",
                "sha256": remediation_sha256,
            },
            "frozen_v2c5_development": {
                "path": "data/evals/v2/ml/v2c5_expanded_development_dataset.json",
                "sha256": DEVELOPMENT_DATASET_SHA256,
            },
        },
        "taxonomy_version": "v2c5-taxonomy.v1",
    }


def build_remediation_manifest(
    validation: AuthoringValidation,
    remediation_bytes: bytes,
    paths: BuildPaths,
    reader: BytesReader,
) -> dict[str, Any]:
    records = validation.approved_records
    return {
        "authoring_input": {
            "path": display_path(paths.authoring_input, paths),
            "sha256": sha256_bytes(validation.input_bytes),
        },
        "builder": {
            "path": display_path(paths.builder, paths),
            "sha256": sha256_file(paths.builder, paths, reader),
            "version": BUILDER_VERSION,
        },
        "counts": {
            "intent_counts": count_by(records, "intent"),
            "record_count": len(records),
            "risk_counts": count_by(records, "risk_level"),
        },
        "duplicate_validation": validation.report["duplicate_validation"],
        "governance": GOVERNANCE_FLAGS,
        "group_statistics": validation.report["group_statistics"],
        "hard_negative_coverage": validation.report["hard_negative_coverage"],
        "phase": "V2-C6 Step 29E2",
        "remediation_design_contract": {
            "path": "data/evals/v2/ml/v2c6_remediation_design_contract.json",
            "sha256": DESIGN_CONTRACT_SHA256,
        },
        "remediation_examples": {
            "path": display_path(paths.remediation_output, paths),
            "schema_version": REMEDIATION_SCHEMA,
            "sha256": sha256_bytes(remediation_bytes),
        },
        "review_status_summary": validation.report["review_status_counts"],
        "schema_version": REMEDIATION_MANIFEST_SCHEMA,
        "source_family_counts_per_intent": validation.report[
            "source_family_statistics"
        ],
        "step29d_contract": {
            "path": "data/evals/v2/ml/v2c6_remediation_dataset_contract.json",
            "sha256": DATASET_CONTRACT_SHA256,
        },
        "unsupported_subtype_counts": validation.report[
            "unsupported_subtype_counts"
        ],
    }


def build_combined_manifest(
    validation: AuthoringValidation,
    sources: FrozenSources,
    remediation_bytes: bytes,
    combined_bytes: bytes,
    combined_payload: Mapping[str, Any],
    paths: BuildPaths,
    reader: BytesReader,
) -> dict[str, Any]:
    records = combined_payload["examples"]
    return {
        "builder": {
            "path": display_path(paths.builder, paths),
            "sha256": sha256_file(paths.builder, paths, reader),
            "version": BUILDER_VERSION,
        },
        "counts": {
            "new_record_count": len(validation.approved_records),
            "old_record_count": len(sources.development_examples),
            "per_intent": count_by(records, "intent"),
            "per_risk": count_by(records, "risk"),
            "total_combined_count": len(records),
        },
        "duplicate_gates": validation.report["duplicate_validation"],
        "governance": GOVERNANCE_FLAGS,
        "group_statistics": group_report(records),
        "next_required": "v2c6_remediated_development_dataset_validation_and_freeze",
        "phase": "V2-C6 Step 29E2",
        "remediated_development_dataset": {
            "path": display_path(paths.combined_output, paths),
            "schema_version": COMBINED_DATASET_SCHEMA,
            "sha256": sha256_bytes(combined_bytes),
        },
        "remediation_examples": {
            "path": display_path(paths.remediation_output, paths),
            "sha256": sha256_bytes(remediation_bytes),
        },
        "schema_version": COMBINED_MANIFEST_SCHEMA,
        "source_aware_readiness": validation.report["source_aware_readiness"],
        "source_artifacts": {
            "frozen_v2c5_development": sources.source_artifacts[
                "development_dataset"
            ],
            "step29d_contract": sources.source_artifacts["dataset_contract"],
            "taxonomy": sources.source_artifacts["taxonomy"],
        },
    }


def validate_built_artifacts(
    artifacts: BuiltArtifacts,
    validation: AuthoringValidation,
    sources: FrozenSources,
) -> None:
    remediation_records = artifacts.remediation_payload["examples"]
    combined_records = artifacts.combined_payload["examples"]
    old_count = len(sources.development_examples)
    if [record["record_id"] for record in remediation_records] != sorted(
        record["record_id"] for record in remediation_records
    ):
        raise ValueError("remediation record ordering is not deterministic")
    if combined_records[:old_count] != list(sources.development_examples):
        raise ValueError("frozen V2-C5 development records changed")
    new_ids = [record["example_id"] for record in combined_records[old_count:]]
    if new_ids != sorted(new_ids):
        raise ValueError("combined remediation ordering changed")
    if len(combined_records) != old_count + len(validation.approved_records):
        raise ValueError("combined record count does not reconcile")
    if len({record["example_id"] for record in combined_records}) != len(
        combined_records
    ):
        raise ValueError("combined dataset contains duplicate record IDs")
    if any(
        record["risk"] != sources.risk_by_intent[record["intent"]]
        for record in combined_records
    ):
        raise ValueError("combined dataset risk mapping changed")


def build_artifacts(
    sources: FrozenSources,
    validation: AuthoringValidation,
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> BuiltArtifacts:
    enforce_mandatory_gates(validation.report)
    remediation_payload = build_remediation_payload(validation, sources)
    remediation_bytes = stable_json_bytes(remediation_payload)
    combined_payload = build_combined_payload(
        validation, sources, sha256_bytes(remediation_bytes)
    )
    combined_bytes = stable_json_bytes(combined_payload)
    remediation_manifest_payload = build_remediation_manifest(
        validation, remediation_bytes, paths, reader
    )
    combined_manifest_payload = build_combined_manifest(
        validation,
        sources,
        remediation_bytes,
        combined_bytes,
        combined_payload,
        paths,
        reader,
    )
    artifacts = BuiltArtifacts(
        remediation_bytes=remediation_bytes,
        remediation_manifest_bytes=stable_json_bytes(
            remediation_manifest_payload
        ),
        combined_bytes=combined_bytes,
        combined_manifest_bytes=stable_json_bytes(combined_manifest_payload),
        remediation_payload=remediation_payload,
        remediation_manifest_payload=remediation_manifest_payload,
        combined_payload=combined_payload,
        combined_manifest_payload=combined_manifest_payload,
    )
    validate_built_artifacts(artifacts, validation, sources)
    return artifacts


def output_paths(paths: BuildPaths) -> tuple[Path, ...]:
    return (
        paths.remediation_output,
        paths.remediation_manifest_output,
        paths.combined_output,
        paths.combined_manifest_output,
    )


def ensure_outputs_absent(paths: BuildPaths) -> None:
    existing = [path for path in output_paths(paths) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing outputs: "
            + ", ".join(display_path(path, paths) for path in existing)
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
    if not paths.authoring_input.exists():
        return {
            "authoring_input_present": False,
            "files_written": False,
            "frozen_development_record_count": len(loaded.development_examples),
            "governance": GOVERNANCE_FLAGS,
            "ready_for_build": False,
        }
    validation = load_authoring_validation(
        loaded,
        paths,
        reader,
        enforce_build_requirements=False,
    )
    return {
        "authoring_input_present": True,
        "files_written": False,
        "frozen_development_record_count": len(loaded.development_examples),
        "governance": GOVERNANCE_FLAGS,
        "ready_for_build": validation.report["mandatory_build_gates_passed"],
        "validation": validation.report,
    }


def write_artifacts(artifacts: BuiltArtifacts, paths: BuildPaths) -> None:
    ensure_outputs_absent(paths)
    for path, content in zip(
        output_paths(paths),
        (
            artifacts.remediation_bytes,
            artifacts.remediation_manifest_bytes,
            artifacts.combined_bytes,
            artifacts.combined_manifest_bytes,
        ),
        strict=True,
    ):
        durable_create(path, content)


def build(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    ensure_outputs_absent(paths)
    loaded = sources or load_frozen_sources(paths, reader)
    validation = load_authoring_validation(
        loaded,
        paths,
        reader,
        enforce_build_requirements=True,
    )
    artifacts = build_artifacts(loaded, validation, paths, reader)
    write_artifacts(artifacts, paths)
    return {
        "combined_record_count": artifacts.combined_payload["example_count"],
        "files_written": True,
        "output_paths": [display_path(path, paths) for path in output_paths(paths)],
        "remediation_record_count": artifacts.remediation_payload["example_count"],
    }


def check_results(
    paths: BuildPaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
    sources: FrozenSources | None = None,
) -> dict[str, Any]:
    loaded = sources or load_frozen_sources(paths, reader)
    validation = load_authoring_validation(
        loaded,
        paths,
        reader,
        enforce_build_requirements=True,
    )
    expected = build_artifacts(loaded, validation, paths, reader)
    expected_bytes = (
        expected.remediation_bytes,
        expected.remediation_manifest_bytes,
        expected.combined_bytes,
        expected.combined_manifest_bytes,
    )
    actual_bytes = tuple(
        guarded_read_bytes(path, paths, reader) for path in output_paths(paths)
    )
    if actual_bytes != expected_bytes:
        raise ValueError("existing outputs are not the deterministic build")
    return {
        "files_written": False,
        "governance": GOVERNANCE_FLAGS,
        "output_hashes": {
            display_path(path, paths): sha256_bytes(content)
            for path, content in zip(output_paths(paths), actual_bytes, strict=True)
        },
        "results_valid": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check-results", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight:
        result = preflight()
    elif args.build:
        result = build()
    else:
        result = check_results()
    print(stable_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
