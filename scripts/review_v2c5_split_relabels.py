"""Manage local human relabeling for frozen V2-C5 split clusters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
LOCAL_ROOT = ML_ROOT / "local"

DEVELOPMENT_DATASET_SHA256 = (
    "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca"
)
DISCOVERY_CORPUS_SHA256 = (
    "0a873fe5797fdd93689e7d5054b029cd7b9068017b0ecff54af000641e282e01"
)
DISCOVERY_ASSIGNMENTS_SHA256 = (
    "aa91a38a3dde57747a8694c404ef00105dcdb0e4233c995749edc1d129596ac0"
)
TAXONOMY_ADJUDICATION_SHA256 = (
    "e63e6a9afc575c603587f2a2119badd597bebbed3e57febc8f5384c16ea99d5d"
)
TAXONOMY_FREEZE_SHA256 = (
    "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
)
TAXONOMY_FREEZE_MANIFEST_SHA256 = (
    "359eb38e5ff63af0c9cc0f5db19951b8626f9e15cb136f23046648d655a0f7a9"
)

EXPECTED_MANUAL_COUNT = 578
EXPECTED_DETERMINISTIC_COUNT = 397
EXPECTED_SPLIT_COVERAGE = 975
EXPECTED_ASSIGNMENT_COUNT = 6372
PRIMARY_CONFIG_ID = "mcs30_ms10"
INTERNAL_OR_NULL_LABEL = "__INTERNAL_OR_NULL__"

WORKFILE_SCHEMA_VERSION = "v2c5-split-relabel-workfile.v1"
ADJUDICATION_SCHEMA_VERSION = "v2c5-split-relabel-adjudication.v1"
MANIFEST_SCHEMA_VERSION = "v2c5-split-relabel-adjudication-manifest.v1"

CONFIDENCE_VALUES = ("HIGH", "MEDIUM", "LOW")
MANUAL_ALLOWED_FINAL_INTENTS = (
    "informational_policy",
    "unsupported_or_uncertain",
)
PROTECTED_INTENTS = frozenset(
    {"cancel_transfer", "close_account", "create_dispute", "freeze_card"}
)

SPLIT_CLUSTER_IDS = (
    "cluster-25a4740ae330f948",
    "cluster-2b49d15c7f2e42b9",
    "cluster-3db20683a0becc1c",
    "cluster-783d0d28bd4c4992",
    "cluster-a077797c96f814ce",
    "cluster-bab7911ea80f612b",
    "cluster-c29614d54555e30c",
    "cluster-d8ae664886c213a2",
    "cluster-e92864f71372fce3",
)

EXPECTED_SPLIT_CLUSTER_COUNTS = {
    "cluster-25a4740ae330f948": 237,
    "cluster-2b49d15c7f2e42b9": 66,
    "cluster-3db20683a0becc1c": 90,
    "cluster-783d0d28bd4c4992": 72,
    "cluster-a077797c96f814ce": 43,
    "cluster-bab7911ea80f612b": 221,
    "cluster-c29614d54555e30c": 46,
    "cluster-d8ae664886c213a2": 75,
    "cluster-e92864f71372fce3": 125,
}
EXPECTED_MANUAL_COUNTS = {
    "cluster-25a4740ae330f948": 161,
    "cluster-2b49d15c7f2e42b9": 44,
    "cluster-3db20683a0becc1c": 86,
    "cluster-783d0d28bd4c4992": 66,
    "cluster-bab7911ea80f612b": 221,
}
EXPECTED_DETERMINISTIC_COUNTS = {
    "cluster-25a4740ae330f948": 76,
    "cluster-2b49d15c7f2e42b9": 22,
    "cluster-3db20683a0becc1c": 4,
    "cluster-783d0d28bd4c4992": 6,
    "cluster-a077797c96f814ce": 43,
    "cluster-bab7911ea80f612b": 0,
    "cluster-c29614d54555e30c": 46,
    "cluster-d8ae664886c213a2": 75,
    "cluster-e92864f71372fce3": 125,
}

DETERMINISTIC_NATIVE_LABEL_MAPPINGS: dict[str, dict[str, str]] = {
    "cluster-25a4740ae330f948": {
        "disposable_card_limits": "informational_policy",
    },
    "cluster-2b49d15c7f2e42b9": {
        "card_delivery_estimate": "informational_policy",
    },
    "cluster-3db20683a0becc1c": {
        INTERNAL_OR_NULL_LABEL: "unsupported_or_uncertain",
    },
    "cluster-783d0d28bd4c4992": {
        "topping_up_by_card": "unsupported_or_uncertain",
    },
    "cluster-a077797c96f814ce": {
        "pending_transfer": "transfer_pending",
        "transfer_not_received_by_recipient": "transfer_pending",
        "transfer_timing": "informational_policy",
    },
    "cluster-bab7911ea80f612b": {},
    "cluster-c29614d54555e30c": {
        "account_blocked": "account_blocked",
        "freeze_account": "unsupported_or_uncertain",
    },
    "cluster-d8ae664886c213a2": {
        INTERNAL_OR_NULL_LABEL: "unsupported_or_uncertain",
        "credit_limit": "unsupported_or_uncertain",
        "credit_limit_change": "unsupported_or_uncertain",
    },
    "cluster-e92864f71372fce3": {
        "unable_to_verify_identity": "unsupported_or_uncertain",
        "verify_my_identity": "informational_policy",
        "verify_top_up": "unsupported_or_uncertain",
    },
}

MANUAL_NATIVE_LABELS_BY_CLUSTER: dict[str, frozenset[str]] = {
    "cluster-25a4740ae330f948": frozenset(
        {
            "get_disposable_virtual_card",
            "getting_spare_card",
            "getting_virtual_card",
            "new_card",
            "order_physical_card",
        }
    ),
    "cluster-2b49d15c7f2e42b9": frozenset({"card_arrival"}),
    "cluster-3db20683a0becc1c": frozenset(
        {"edit_personal_details", "verify_my_identity"}
    ),
    "cluster-783d0d28bd4c4992": frozenset(
        {"transfer", "transfer_into_account"}
    ),
    "cluster-bab7911ea80f612b": frozenset(
        {"change_pin", "get_physical_card", "pin_change"}
    ),
}

ALLOWED_DEVELOPMENT_SOURCE_SPLITS = frozenset(
    {
        ("sentinelvoice_v2c1_internal", "train"),
        ("sentinelvoice_v2c1_internal", "validation"),
        ("banking77", "train"),
        ("clinc150_oos", "train"),
        ("clinc150_oos", "val"),
    }
)

FORBIDDEN_INPUT_PATH_TOKENS = (
    "challenge",
    "cfpb",
    "final",
    "holdout",
    "lockbox",
    "test",
)

MANUAL_RECORD_METADATA_FIELDS = (
    "allowed_final_intents",
    "canonical_cluster_id",
    "discovery_id",
    "has_internal_or_null_label",
    "native_external_labels",
    "native_label_key",
    "normalized_text_sha256",
    "source_datasets",
    "source_example_ids",
    "source_occurrence_count",
)
MANUAL_REVIEW_FIELDS = (
    "confidence",
    "review_status",
    "reviewer_note",
    "selected_final_intent",
)
WORKFILE_RECORD_FIELDS = frozenset(
    (*MANUAL_RECORD_METADATA_FIELDS, *MANUAL_REVIEW_FIELDS)
)


@dataclass(frozen=True)
class ReviewPaths:
    development_dataset: Path
    discovery_corpus: Path
    discovery_assignments: Path
    taxonomy_adjudication: Path
    taxonomy_freeze: Path
    taxonomy_freeze_manifest: Path
    workfile: Path
    adjudication_output: Path
    adjudication_manifest_output: Path


DEFAULT_PATHS = ReviewPaths(
    development_dataset=ML_ROOT / "v2c3_development_dataset.json",
    discovery_corpus=ML_ROOT / "v2c5_discovery_corpus.json",
    discovery_assignments=ML_ROOT / "v2c5_intent_discovery_assignments.json",
    taxonomy_adjudication=ML_ROOT / "v2c5_taxonomy_adjudication.json",
    taxonomy_freeze=ML_ROOT / "v2c5_taxonomy_freeze.json",
    taxonomy_freeze_manifest=ML_ROOT / "v2c5_taxonomy_freeze.manifest.json",
    workfile=LOCAL_ROOT / "v2c5_split_relabel_workfile.json",
    adjudication_output=ML_ROOT / "v2c5_split_relabel_adjudication.json",
    adjudication_manifest_output=(
        ML_ROOT / "v2c5_split_relabel_adjudication.manifest.json"
    ),
)


@dataclass(frozen=True)
class FrozenSources:
    development_dataset: dict[str, Any]
    discovery_corpus: dict[str, Any]
    discovery_assignments: dict[str, Any]
    taxonomy_adjudication: dict[str, Any]
    taxonomy_freeze: dict[str, Any]
    taxonomy_freeze_manifest: dict[str, Any]
    corpus_by_id: dict[str, dict[str, Any]]
    split_records: list[dict[str, Any]]
    manual_records: list[dict[str, Any]]
    deterministic_records: list[dict[str, Any]]
    split_rules_by_cluster: dict[str, dict[str, Any]]
    source_artifacts: dict[str, dict[str, Any]]


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def source_path_items(paths: ReviewPaths) -> tuple[tuple[str, Path, str], ...]:
    return (
        (
            "development_dataset",
            paths.development_dataset,
            DEVELOPMENT_DATASET_SHA256,
        ),
        (
            "discovery_corpus",
            paths.discovery_corpus,
            DISCOVERY_CORPUS_SHA256,
        ),
        (
            "discovery_assignments",
            paths.discovery_assignments,
            DISCOVERY_ASSIGNMENTS_SHA256,
        ),
        (
            "taxonomy_adjudication",
            paths.taxonomy_adjudication,
            TAXONOMY_ADJUDICATION_SHA256,
        ),
        (
            "taxonomy_freeze",
            paths.taxonomy_freeze,
            TAXONOMY_FREEZE_SHA256,
        ),
        (
            "taxonomy_freeze_manifest",
            paths.taxonomy_freeze_manifest,
            TAXONOMY_FREEZE_MANIFEST_SHA256,
        ),
    )


def allowed_source_filenames() -> frozenset[str]:
    return frozenset(path.name for _, path, _ in source_path_items(DEFAULT_PATHS))


def guard_source_path(path: Path) -> None:
    normalized = path.as_posix().lower()
    if path.name not in allowed_source_filenames():
        raise ValueError(f"Step 21A source input is not allowed: {path.name}")
    if any(token in normalized for token in FORBIDDEN_INPUT_PATH_TOKENS):
        raise ValueError(f"prohibited non-development input path: {path}")


def read_source_bytes(path: Path) -> bytes:
    guard_source_path(path)
    return path.read_bytes()


def load_json_object(value: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def require_sha256(value: bytes, expected: str, label: str) -> None:
    if sha256_bytes(value) != expected:
        raise ValueError(f"frozen {label} SHA-256 changed")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def source_artifact_metadata(paths: ReviewPaths) -> dict[str, dict[str, Any]]:
    return {
        label: {"path": display_path(path), "sha256": expected_hash}
        for label, path, expected_hash in source_path_items(paths)
    }


def validate_development_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "v2c3-development-dataset.v1":
        raise ValueError("unexpected V2-C3 development dataset schema")
    records = payload.get("examples")
    if not isinstance(records, list) or payload.get("example_count") != len(records):
        raise ValueError("V2-C3 development example count changed")
    by_id: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("development records must be objects")
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or example_id in by_id:
            raise ValueError("development example IDs must be unique strings")
        if record.get("data_role") != "development":
            raise ValueError("non-development record entered Step 21A")
        source_key = (record.get("source_id"), record.get("source_split"))
        if source_key not in ALLOWED_DEVELOPMENT_SOURCE_SPLITS:
            raise ValueError("prohibited source split entered Step 21A")
        by_id[example_id] = record
    return by_id


def validate_source_contracts(
    corpus: Mapping[str, Any],
    assignments: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    taxonomy_freeze: Mapping[str, Any],
    taxonomy_freeze_manifest: Mapping[str, Any],
    paths: ReviewPaths,
) -> None:
    if corpus.get("schema_version") != "v2c5-discovery-corpus.v1":
        raise ValueError("unexpected discovery-corpus schema")
    if assignments.get("schema_version") != (
        "v2c5-intent-discovery-assignments.v1"
    ):
        raise ValueError("unexpected discovery-assignment schema")
    if assignments.get("primary_config_id") != PRIMARY_CONFIG_ID:
        raise ValueError("Step 21A requires primary mcs30_ms10 assignments")
    if assignments.get("assignment_count") != EXPECTED_ASSIGNMENT_COUNT:
        raise ValueError("discovery assignment count changed")
    if adjudication.get("schema_version") != "v2c5-taxonomy-adjudication.v1":
        raise ValueError("unexpected Step 19 adjudication schema")
    if adjudication.get("phase") != "V2-C5 Step 19":
        raise ValueError("unexpected Step 19 adjudication phase")
    if adjudication.get("cluster_count") != 35 or adjudication.get(
        "reviewed_count"
    ) != 35:
        raise ValueError("Step 19 adjudication is incomplete")
    if taxonomy_freeze.get("schema_version") != "v2c5-taxonomy-freeze.v1":
        raise ValueError("unexpected Step 20 taxonomy-freeze schema")
    if taxonomy_freeze.get("phase") != "V2-C5 Step 20":
        raise ValueError("unexpected Step 20 taxonomy-freeze phase")
    if taxonomy_freeze.get("final_intent_count") != 16:
        raise ValueError("Step 20 final intent count changed")
    if taxonomy_freeze.get("final_taxonomy_frozen") is not True:
        raise ValueError("Step 20 taxonomy is not frozen")
    expected_freeze_flags = {
        "classifier_training_performed": False,
        "dataset_relabeling_performed": False,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "step21_required": True,
    }
    for field, expected in expected_freeze_flags.items():
        if taxonomy_freeze.get(field) != expected:
            raise ValueError(f"Step 20 freeze governance changed: {field}")
    expected_step19 = {
        "path": display_path(paths.taxonomy_adjudication),
        "sha256": TAXONOMY_ADJUDICATION_SHA256,
    }
    if taxonomy_freeze.get("source_step19_adjudication") != expected_step19:
        raise ValueError("Step 20 freeze Step 19 binding changed")
    if taxonomy_freeze_manifest.get("schema_version") != (
        "v2c5-taxonomy-freeze-manifest.v1"
    ):
        raise ValueError("unexpected Step 20 freeze-manifest schema")
    if taxonomy_freeze_manifest.get("phase") != "V2-C5 Step 20":
        raise ValueError("unexpected Step 20 freeze-manifest phase")
    if taxonomy_freeze_manifest.get("final_intent_count") != 16:
        raise ValueError("Step 20 manifest final intent count changed")
    expected_freeze = {
        "path": display_path(paths.taxonomy_freeze),
        "sha256": TAXONOMY_FREEZE_SHA256,
    }
    if taxonomy_freeze_manifest.get("taxonomy_freeze") != expected_freeze:
        raise ValueError("Step 20 manifest freeze binding changed")
    if taxonomy_freeze_manifest.get("source_step19_adjudication") != expected_step19:
        raise ValueError("Step 20 manifest Step 19 binding changed")
    expected_flags = {
        "classifier_training_performed": False,
        "final_taxonomy_frozen": True,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "step21_required": True,
    }
    for field, expected in expected_flags.items():
        if taxonomy_freeze_manifest.get(field) != expected:
            raise ValueError(f"Step 20 manifest governance changed: {field}")


def split_rules_from_sources(
    adjudication: Mapping[str, Any],
    taxonomy_freeze: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    adjudications = adjudication.get("adjudications")
    if not isinstance(adjudications, list):
        raise TypeError("Step 19 adjudications must be a list")
    source_split_ids = {
        str(record["canonical_cluster_id"])
        for record in adjudications
        if isinstance(record, dict)
        and record.get("decision") == "NEEDS_SPLIT_REVIEW"
    }
    if source_split_ids != set(SPLIT_CLUSTER_IDS):
        raise ValueError("Step 19 split-review cluster set changed")

    values = taxonomy_freeze.get("split_cluster_resolutions")
    if not isinstance(values, list) or len(values) != len(SPLIT_CLUSTER_IDS):
        raise ValueError("Step 20 split-resolution count changed")
    by_cluster: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("Step 20 split resolution must be an object")
        cluster_id = value.get("canonical_cluster_id")
        if not isinstance(cluster_id, str) or cluster_id in by_cluster:
            raise ValueError("Step 20 split cluster IDs must be unique")
        if value.get("resolution") != "BRANCH_FOR_STEP21_RELABELING":
            raise ValueError("Step 20 split resolution policy changed")
        branches = value.get("branches")
        if not isinstance(branches, list) or len(branches) != 2:
            raise ValueError("Step 20 split cluster must have two branches")
        by_cluster[cluster_id] = copy.deepcopy(value)
    if set(by_cluster) != set(SPLIT_CLUSTER_IDS):
        raise ValueError("Step 20 split cluster set changed")
    for cluster_id in MANUAL_NATIVE_LABELS_BY_CLUSTER:
        targets = {
            str(branch["final_intent"])
            for branch in by_cluster[cluster_id]["branches"]
        }
        if targets != set(MANUAL_ALLOWED_FINAL_INTENTS):
            raise ValueError("manual cluster branch targets changed")
    for cluster_id, native_mappings in DETERMINISTIC_NATIVE_LABEL_MAPPINGS.items():
        branch_targets = {
            str(branch["final_intent"])
            for branch in by_cluster[cluster_id]["branches"]
        }
        if not set(native_mappings.values()).issubset(branch_targets):
            raise ValueError("deterministic mapping exceeds frozen branch targets")
    return by_cluster


def occurrence_native_label_key(record: Mapping[str, Any]) -> str:
    occurrences = record.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError("discovery record must contain source occurrences")
    labels = {
        (
            str(occurrence["native_external_label"])
            if occurrence.get("native_external_label") is not None
            else INTERNAL_OR_NULL_LABEL
        )
        for occurrence in occurrences
        if isinstance(occurrence, dict)
    }
    if len(labels) != 1:
        raise ValueError("split discovery record has ambiguous native labels")
    return next(iter(labels))


def validate_corpus_occurrences(
    record: Mapping[str, Any],
    development_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    for occurrence in record["occurrences"]:
        example_id = occurrence.get("example_id")
        if not isinstance(example_id, str) or example_id not in development_by_id:
            raise ValueError("discovery occurrence is absent from development data")
        development = development_by_id[example_id]
        expected_fields = {
            "data_role": "development",
            "normalized_text_sha256": record["normalized_text_sha256"],
            "source_id": occurrence.get("source_dataset"),
            "source_row_index": occurrence.get("source_row_index"),
            "source_split": occurrence.get("source_split"),
            "text_sha256": occurrence.get("text_sha256"),
        }
        for field, expected in expected_fields.items():
            if development.get(field) != expected:
                raise ValueError(f"development lineage changed: {field}")
        native_label = occurrence.get("native_external_label")
        if native_label is not None and development.get("source_label") != native_label:
            raise ValueError("native external label lineage changed")


def manual_record_metadata(
    record: Mapping[str, Any],
    cluster_id: str,
    native_label_key: str,
) -> dict[str, Any]:
    occurrences = record["occurrences"]
    return {
        "allowed_final_intents": list(MANUAL_ALLOWED_FINAL_INTENTS),
        "canonical_cluster_id": cluster_id,
        "discovery_id": record["discovery_id"],
        "has_internal_or_null_label": (
            native_label_key == INTERNAL_OR_NULL_LABEL
        ),
        "native_external_labels": sorted(
            {
                str(value["native_external_label"])
                for value in occurrences
                if value.get("native_external_label") is not None
            }
        ),
        "native_label_key": native_label_key,
        "normalized_text_sha256": record["normalized_text_sha256"],
        "source_datasets": sorted(
            {str(value["source_dataset"]) for value in occurrences}
        ),
        "source_example_ids": sorted(
            {str(value["example_id"]) for value in occurrences}
        ),
        "source_occurrence_count": len(occurrences),
    }


def record_sort_key(value: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(value["canonical_cluster_id"]),
        str(value["discovery_id"]),
    )


def partition_split_records(
    corpus: Mapping[str, Any],
    assignments: Mapping[str, Any],
    development_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    corpus_values = corpus.get("primary_cluster_population")
    assignment_values = assignments.get("assignments")
    if not isinstance(corpus_values, list) or not isinstance(
        assignment_values,
        list,
    ):
        raise TypeError("corpus and assignments must contain record lists")
    if len(corpus_values) != EXPECTED_ASSIGNMENT_COUNT or len(
        assignment_values
    ) != EXPECTED_ASSIGNMENT_COUNT:
        raise ValueError("corpus or assignment population changed")

    assignments_by_id: dict[str, dict[str, Any]] = {}
    for assignment in assignment_values:
        if not isinstance(assignment, dict):
            raise TypeError("assignment must be an object")
        discovery_id = assignment.get("discovery_id")
        if not isinstance(discovery_id, str) or discovery_id in assignments_by_id:
            raise ValueError("assignment discovery IDs must be unique strings")
        assignments_by_id[discovery_id] = assignment

    split_records: list[dict[str, Any]] = []
    manual_records: list[dict[str, Any]] = []
    deterministic_records: list[dict[str, Any]] = []
    corpus_ids: set[str] = set()
    for record in corpus_values:
        if not isinstance(record, dict):
            raise TypeError("discovery corpus record must be an object")
        discovery_id = record.get("discovery_id")
        if not isinstance(discovery_id, str) or discovery_id in corpus_ids:
            raise ValueError("corpus discovery IDs must be unique strings")
        corpus_ids.add(discovery_id)
        assignment = assignments_by_id.get(discovery_id)
        if assignment is None:
            raise ValueError("corpus record is absent from assignments")
        cluster_id = assignment.get("primary_canonical_cluster_id")
        if assignment.get("primary_is_noise") is True:
            continue
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError("primary canonical cluster ID must be a non-empty string")
        if cluster_id not in SPLIT_CLUSTER_IDS:
            continue
        if assignment.get("primary_is_noise") is not False:
            raise ValueError("split cluster record cannot be noise")
        if assignment.get("normalized_text_sha256") != record.get(
            "normalized_text_sha256"
        ):
            raise ValueError("corpus and assignment normalized hashes differ")
        validate_corpus_occurrences(record, development_by_id)
        native_label_key = occurrence_native_label_key(record)
        expected_labels = set(DETERMINISTIC_NATIVE_LABEL_MAPPINGS[cluster_id]) | set(
            MANUAL_NATIVE_LABELS_BY_CLUSTER.get(cluster_id, frozenset())
        )
        if native_label_key not in expected_labels:
            raise ValueError("unexpected native label in split cluster")
        derived_record = copy.deepcopy(dict(record))
        derived_record["canonical_cluster_id"] = cluster_id
        split_records.append(derived_record)
        deterministic_target = DETERMINISTIC_NATIVE_LABEL_MAPPINGS[
            cluster_id
        ].get(native_label_key)
        if deterministic_target is not None:
            deterministic_records.append(
                {
                    "canonical_cluster_id": cluster_id,
                    "discovery_id": discovery_id,
                    "final_intent": deterministic_target,
                    "native_label_key": native_label_key,
                    "normalized_text_sha256": record["normalized_text_sha256"],
                }
            )
        else:
            manual_records.append(
                manual_record_metadata(record, cluster_id, native_label_key)
            )

    return (
        sorted(split_records, key=record_sort_key),
        sorted(manual_records, key=record_sort_key),
        sorted(deterministic_records, key=record_sort_key),
    )


def validate_partition_counts(
    split_records: Sequence[Mapping[str, Any]],
    manual_records: Sequence[Mapping[str, Any]],
    deterministic_records: Sequence[Mapping[str, Any]],
) -> None:
    if len(split_records) != EXPECTED_SPLIT_COVERAGE:
        raise ValueError("split-cluster coverage must equal exactly 975")
    if len(manual_records) != EXPECTED_MANUAL_COUNT:
        raise ValueError("manual review population must equal exactly 578")
    if len(deterministic_records) != EXPECTED_DETERMINISTIC_COUNT:
        raise ValueError("deterministic mapping population must equal exactly 397")
    split_counts = Counter(
        str(value["canonical_cluster_id"]) for value in split_records
    )
    manual_counts = Counter(
        str(value["canonical_cluster_id"]) for value in manual_records
    )
    deterministic_counts = Counter(
        str(value["canonical_cluster_id"]) for value in deterministic_records
    )
    if dict(split_counts) != EXPECTED_SPLIT_CLUSTER_COUNTS:
        raise ValueError("split-cluster member counts changed")
    if dict(manual_counts) != EXPECTED_MANUAL_COUNTS:
        raise ValueError("manual per-cluster counts changed")
    normalized_deterministic = {
        cluster_id: deterministic_counts[cluster_id]
        for cluster_id in SPLIT_CLUSTER_IDS
    }
    if normalized_deterministic != EXPECTED_DETERMINISTIC_COUNTS:
        raise ValueError("deterministic per-cluster counts changed")
    if any(
        value["final_intent"] in PROTECTED_INTENTS
        for value in deterministic_records
    ):
        raise ValueError("protected intents are forbidden for split relabeling")
    manual_ids = {str(value["discovery_id"]) for value in manual_records}
    deterministic_ids = {
        str(value["discovery_id"]) for value in deterministic_records
    }
    if len(manual_ids) != len(manual_records):
        raise ValueError("duplicate manual discovery ID")
    if len(deterministic_ids) != len(deterministic_records):
        raise ValueError("duplicate deterministic discovery ID")
    if manual_ids & deterministic_ids:
        raise ValueError("manual and deterministic populations overlap")
    split_ids = {str(value["discovery_id"]) for value in split_records}
    if manual_ids | deterministic_ids != split_ids:
        raise ValueError("manual and deterministic populations do not cover splits")


def load_frozen_sources(paths: ReviewPaths = DEFAULT_PATHS) -> FrozenSources:
    payloads: dict[str, dict[str, Any]] = {}
    for label, path, expected_hash in source_path_items(paths):
        value = read_source_bytes(path)
        require_sha256(value, expected_hash, label.replace("_", " "))
        payloads[label] = load_json_object(value, label.replace("_", " "))

    development_by_id = validate_development_dataset(
        payloads["development_dataset"]
    )
    validate_source_contracts(
        payloads["discovery_corpus"],
        payloads["discovery_assignments"],
        payloads["taxonomy_adjudication"],
        payloads["taxonomy_freeze"],
        payloads["taxonomy_freeze_manifest"],
        paths,
    )
    split_rules = split_rules_from_sources(
        payloads["taxonomy_adjudication"],
        payloads["taxonomy_freeze"],
    )
    split_records, manual_records, deterministic_records = partition_split_records(
        payloads["discovery_corpus"],
        payloads["discovery_assignments"],
        development_by_id,
    )
    validate_partition_counts(
        split_records,
        manual_records,
        deterministic_records,
    )
    corpus_values = payloads["discovery_corpus"]["primary_cluster_population"]
    corpus_by_id = {
        str(value["discovery_id"]): value for value in corpus_values
    }
    return FrozenSources(
        development_dataset=payloads["development_dataset"],
        discovery_corpus=payloads["discovery_corpus"],
        discovery_assignments=payloads["discovery_assignments"],
        taxonomy_adjudication=payloads["taxonomy_adjudication"],
        taxonomy_freeze=payloads["taxonomy_freeze"],
        taxonomy_freeze_manifest=payloads["taxonomy_freeze_manifest"],
        corpus_by_id=corpus_by_id,
        split_records=split_records,
        manual_records=manual_records,
        deterministic_records=deterministic_records,
        split_rules_by_cluster=split_rules,
        source_artifacts=source_artifact_metadata(paths),
    )


def initial_review_fields() -> dict[str, Any]:
    return {
        "confidence": None,
        "review_status": "UNREVIEWED",
        "reviewer_note": "",
        "selected_final_intent": None,
    }


def workfile_governance() -> dict[str, Any]:
    return {
        "classifier_training_performed": False,
        "development_data_only": True,
        "development_dataset_built": False,
        "external_labels_are_metadata_only": True,
        "fresh_holdout_created": False,
        "native_labels_are_automatic_truth_for_manual_rows": False,
        "runtime_behavior_changed": False,
        "step21_complete": False,
    }


def build_workfile_payload(sources: FrozenSources) -> dict[str, Any]:
    records = [
        {**copy.deepcopy(value), **initial_review_fields()}
        for value in sources.manual_records
    ]
    payload = {
        "deterministic_mapping_count": len(sources.deterministic_records),
        "manual_review_count": len(records),
        "ordering": "canonical_cluster_id_then_discovery_id",
        "phase": "V2-C5 Step 21A",
        "records": records,
        "schema_version": WORKFILE_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "split_coverage_count": len(sources.split_records),
        "governance": workfile_governance(),
    }
    validate_workfile_payload(payload, sources)
    return payload


def expected_manual_metadata(sources: FrozenSources) -> list[dict[str, Any]]:
    return [copy.deepcopy(value) for value in sources.manual_records]


def validate_review_record(
    record: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> None:
    if set(record) != WORKFILE_RECORD_FIELDS:
        raise ValueError("manual review record fields changed")
    for field in MANUAL_RECORD_METADATA_FIELDS:
        if record.get(field) != expected_metadata.get(field):
            raise ValueError(f"frozen manual metadata changed: {field}")
    if record.get("allowed_final_intents") != list(
        MANUAL_ALLOWED_FINAL_INTENTS
    ):
        raise ValueError("manual allowed-label set changed")
    selected = record.get("selected_final_intent")
    if selected in PROTECTED_INTENTS:
        raise ValueError("protected intents are forbidden for split relabeling")
    status = record.get("review_status")
    if status == "UNREVIEWED":
        if (
            selected is not None
            or record.get("confidence") is not None
            or record.get("reviewer_note") != ""
        ):
            raise ValueError("UNREVIEWED record contains a human decision")
        return
    if status != "REVIEWED":
        raise ValueError("review status must be UNREVIEWED or REVIEWED")
    if selected not in record["allowed_final_intents"]:
        raise ValueError("selected final intent is not allowed for this cluster")
    if record.get("confidence") not in CONFIDENCE_VALUES:
        raise ValueError("reviewed record requires HIGH, MEDIUM, or LOW confidence")
    note = record.get("reviewer_note")
    if not isinstance(note, str) or note != note.strip():
        raise ValueError("reviewer note must be a trimmed string")


def validate_workfile_payload(
    payload: Mapping[str, Any],
    sources: FrozenSources,
) -> None:
    required_fields = {
        "deterministic_mapping_count",
        "governance",
        "manual_review_count",
        "ordering",
        "phase",
        "records",
        "schema_version",
        "source_artifacts",
        "split_coverage_count",
    }
    if set(payload) != required_fields:
        raise ValueError("split-relabel workfile fields changed")
    expected_values = {
        "deterministic_mapping_count": EXPECTED_DETERMINISTIC_COUNT,
        "manual_review_count": EXPECTED_MANUAL_COUNT,
        "ordering": "canonical_cluster_id_then_discovery_id",
        "phase": "V2-C5 Step 21A",
        "schema_version": WORKFILE_SCHEMA_VERSION,
        "source_artifacts": sources.source_artifacts,
        "split_coverage_count": EXPECTED_SPLIT_COVERAGE,
        "governance": workfile_governance(),
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise ValueError(f"split-relabel workfile changed: {field}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_MANUAL_COUNT:
        raise ValueError("workfile must contain exactly 578 manual records")
    expected_metadata = expected_manual_metadata(sources)
    actual_ids: list[str] = []
    for record, metadata in zip(records, expected_metadata, strict=True):
        if not isinstance(record, dict):
            raise TypeError("manual review record must be an object")
        validate_review_record(record, metadata)
        actual_ids.append(str(record["discovery_id"]))
    expected_ids = [str(value["discovery_id"]) for value in expected_metadata]
    if actual_ids != expected_ids or len(set(actual_ids)) != EXPECTED_MANUAL_COUNT:
        raise ValueError("manual record set or deterministic order changed")


def load_workfile(
    sources: FrozenSources,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    payload = load_json_object(paths.workfile.read_bytes(), "split relabel workfile")
    validate_workfile_payload(payload, sources)
    return payload


def progress_summary(
    payload: Mapping[str, Any],
    sources: FrozenSources,
) -> dict[str, Any]:
    records = payload["records"]
    statuses = Counter(str(value["review_status"]) for value in records)
    selected = Counter(
        str(value["selected_final_intent"])
        for value in records
        if value["review_status"] == "REVIEWED"
    )
    confidences = Counter(
        str(value["confidence"])
        for value in records
        if value["review_status"] == "REVIEWED"
    )
    cluster_counts: dict[str, dict[str, int]] = {}
    for cluster_id in EXPECTED_MANUAL_COUNTS:
        cluster_records = [
            value
            for value in records
            if value["canonical_cluster_id"] == cluster_id
        ]
        reviewed = sum(
            value["review_status"] == "REVIEWED" for value in cluster_records
        )
        cluster_counts[cluster_id] = {
            "reviewed": reviewed,
            "total": len(cluster_records),
            "unreviewed": len(cluster_records) - reviewed,
        }
    return {
        "confidence_counts": {
            value: confidences[value] for value in CONFIDENCE_VALUES
        },
        "counts_by_cluster": cluster_counts,
        "deterministic_mapping_count": len(sources.deterministic_records),
        "final_label_counts": {
            value: selected[value] for value in MANUAL_ALLOWED_FINAL_INTENTS
        },
        "reviewed": statuses["REVIEWED"],
        "split_coverage_count": len(sources.split_records),
        "total": len(records),
        "unreviewed": statuses["UNREVIEWED"],
    }


def find_record(payload: Mapping[str, Any], discovery_id: str) -> dict[str, Any]:
    matches = [
        value for value in payload["records"] if value["discovery_id"] == discovery_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown manual discovery ID: {discovery_id}")
    return matches[0]


def display_review_record(
    record: Mapping[str, Any],
    sources: FrozenSources,
) -> dict[str, Any]:
    discovery_id = str(record["discovery_id"])
    corpus_record = sources.corpus_by_id.get(discovery_id)
    if corpus_record is None:
        raise ValueError("manual discovery record is absent from frozen corpus")
    cluster_id = str(record["canonical_cluster_id"])
    return {
        "branch_semantics": copy.deepcopy(
            sources.split_rules_by_cluster[cluster_id]["branches"]
        ),
        "cluster_theme": sources.split_rules_by_cluster[cluster_id]["theme"],
        "metadata_notice": (
            "Source and native-label values are metadata/evidence only. "
            "Choose from the frozen branch targets based on the utterance's "
            "speech act; the label grants no runtime authority."
        ),
        "record": copy.deepcopy(dict(record)),
        "text": corpus_record["text"],
    }


def build_workfile(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    sources = load_frozen_sources(paths)
    if paths.workfile.exists():
        raise FileExistsError(
            "split-relabel workfile already exists; build will not overwrite it"
        )
    payload = build_workfile_payload(sources)
    atomic_write_bytes(paths.workfile, stable_json_bytes(payload))
    return payload


def show_record(
    discovery_id: str,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    sources = load_frozen_sources(paths)
    workfile = load_workfile(sources, paths)
    return display_review_record(find_record(workfile, discovery_id), sources)


def next_record(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    sources = load_frozen_sources(paths)
    workfile = load_workfile(sources, paths)
    record = next(
        (
            value
            for value in workfile["records"]
            if value["review_status"] == "UNREVIEWED"
        ),
        None,
    )
    if record is None:
        return {
            "message": "All split-cluster records have human labels.",
            "review_complete": True,
            "summary": progress_summary(workfile, sources),
        }
    return display_review_record(record, sources)


def apply_review_update(
    payload: Mapping[str, Any],
    sources: FrozenSources,
    *,
    discovery_id: str,
    selected_final_intent: str,
    confidence: str,
    reviewer_note: str | None,
) -> dict[str, Any]:
    validate_workfile_payload(payload, sources)
    updated = copy.deepcopy(dict(payload))
    record = find_record(updated, discovery_id)
    record.update(
        {
            "confidence": confidence,
            "review_status": "REVIEWED",
            "reviewer_note": reviewer_note.strip() if reviewer_note else "",
            "selected_final_intent": selected_final_intent,
        }
    )
    validate_workfile_payload(updated, sources)
    return updated


def set_record_review(
    *,
    discovery_id: str,
    selected_final_intent: str,
    confidence: str,
    reviewer_note: str | None,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    sources = load_frozen_sources(paths)
    workfile = load_workfile(sources, paths)
    updated = apply_review_update(
        workfile,
        sources,
        discovery_id=discovery_id,
        selected_final_intent=selected_final_intent,
        confidence=confidence,
        reviewer_note=reviewer_note,
    )
    atomic_write_bytes(paths.workfile, stable_json_bytes(updated))
    return updated


def deterministic_mapping_summary(
    sources: FrozenSources,
) -> dict[str, Any]:
    by_cluster: dict[str, dict[str, Any]] = {}
    for cluster_id in SPLIT_CLUSTER_IDS:
        values = [
            value
            for value in sources.deterministic_records
            if value["canonical_cluster_id"] == cluster_id
        ]
        label_counts = Counter(str(value["native_label_key"]) for value in values)
        intent_counts = Counter(str(value["final_intent"]) for value in values)
        by_cluster[cluster_id] = {
            "count": len(values),
            "final_intent_counts": dict(sorted(intent_counts.items())),
            "native_label_counts": dict(sorted(label_counts.items())),
        }
    return {
        "by_cluster": by_cluster,
        "mapping_table": copy.deepcopy(DETERMINISTIC_NATIVE_LABEL_MAPPINGS),
        "total": len(sources.deterministic_records),
    }


def validate_text_free(payload: Any) -> None:
    prohibited_keys = {"text", "normalized_text", "utterance", "examples"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in prohibited_keys:
                raise ValueError("split-relabel export cannot contain raw text")
            validate_text_free(value)
    elif isinstance(payload, list):
        for value in payload:
            validate_text_free(value)


def build_adjudication_payload(
    workfile: Mapping[str, Any],
    sources: FrozenSources,
) -> dict[str, Any]:
    validate_workfile_payload(workfile, sources)
    summary = progress_summary(workfile, sources)
    if summary["unreviewed"] != 0:
        raise ValueError("export requires all 578 manual records to be reviewed")
    manual_adjudications = [
        {
            "canonical_cluster_id": value["canonical_cluster_id"],
            "confidence": value["confidence"],
            "discovery_id": value["discovery_id"],
            "reviewer_note": value["reviewer_note"],
            "selected_final_intent": value["selected_final_intent"],
        }
        for value in workfile["records"]
    ]
    payload = {
        "all_reviewed": True,
        "classifier_training_performed": False,
        "deterministic_mapping_count": EXPECTED_DETERMINISTIC_COUNT,
        "deterministic_mapping_summary": deterministic_mapping_summary(sources),
        "deterministic_mappings": copy.deepcopy(sources.deterministic_records),
        "development_dataset_built": False,
        "fresh_holdout_created": False,
        "manual_adjudications": manual_adjudications,
        "manual_review_count": EXPECTED_MANUAL_COUNT,
        "next_required": (
            "expanded_dataset_and_fresh_final_holdout_construction"
        ),
        "phase": "V2-C5 Step 21A",
        "runtime_behavior_changed": False,
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "split_coverage_count": EXPECTED_SPLIT_COVERAGE,
        "step21_complete": False,
    }
    validate_text_free(payload)
    return payload


def build_adjudication_manifest(
    adjudication_bytes: bytes,
    sources: FrozenSources,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    payload = {
        "adjudication": {
            "path": display_path(paths.adjudication_output),
            "sha256": sha256_bytes(adjudication_bytes),
        },
        "all_reviewed": True,
        "classifier_training_performed": False,
        "deterministic_mapping_count": EXPECTED_DETERMINISTIC_COUNT,
        "development_dataset_built": False,
        "fresh_holdout_created": False,
        "manual_review_count": EXPECTED_MANUAL_COUNT,
        "phase": "V2-C5 Step 21A",
        "review_script": {
            "path": display_path(script_path),
            "sha256": sha256_bytes(script_path.read_bytes()),
        },
        "runtime_behavior_changed": False,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "split_coverage_count": EXPECTED_SPLIT_COVERAGE,
        "step21_complete": False,
    }
    validate_text_free(payload)
    return payload


def expected_export_bytes(
    workfile: Mapping[str, Any],
    sources: FrozenSources,
    paths: ReviewPaths,
) -> tuple[bytes, bytes]:
    adjudication = build_adjudication_payload(workfile, sources)
    adjudication_bytes = stable_json_bytes(adjudication)
    manifest = build_adjudication_manifest(adjudication_bytes, sources, paths)
    return adjudication_bytes, stable_json_bytes(manifest)


def validate_existing_exports(
    workfile: Mapping[str, Any],
    sources: FrozenSources,
    paths: ReviewPaths,
) -> None:
    adjudication_exists = paths.adjudication_output.exists()
    manifest_exists = paths.adjudication_manifest_output.exists()
    if adjudication_exists != manifest_exists:
        raise ValueError("split-relabel exports must exist as a pair")
    if not adjudication_exists:
        return
    expected_adjudication, expected_manifest = expected_export_bytes(
        workfile,
        sources,
        paths,
    )
    if paths.adjudication_output.read_bytes() != expected_adjudication:
        raise ValueError("existing split-relabel adjudication changed")
    if paths.adjudication_manifest_output.read_bytes() != expected_manifest:
        raise ValueError("existing split-relabel manifest changed")


def check_workfile(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    sources = load_frozen_sources(paths)
    workfile = load_workfile(sources, paths)
    validate_existing_exports(workfile, sources, paths)
    return progress_summary(workfile, sources)


def export_adjudication(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    sources = load_frozen_sources(paths)
    workfile = load_workfile(sources, paths)
    validate_existing_exports(workfile, sources, paths)
    adjudication_bytes, manifest_bytes = expected_export_bytes(
        workfile,
        sources,
        paths,
    )
    atomic_write_bytes(paths.adjudication_output, adjudication_bytes)
    atomic_write_bytes(paths.adjudication_manifest_output, manifest_bytes)
    return {
        "adjudication_path": display_path(paths.adjudication_output),
        "adjudication_sha256": sha256_bytes(adjudication_bytes),
        "manifest_path": display_path(paths.adjudication_manifest_output),
        "summary": progress_summary(workfile, sources),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("summary")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--discovery-id", required=True)
    subparsers.add_parser("next")

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("--discovery-id", required=True)
    set_parser.add_argument(
        "--final-intent",
        choices=MANUAL_ALLOWED_FINAL_INTENTS,
        required=True,
    )
    set_parser.add_argument("--confidence", choices=CONFIDENCE_VALUES, required=True)
    set_parser.add_argument("--reviewer-note")

    subparsers.add_parser("export")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            payload = build_workfile()
            sources = load_frozen_sources()
            result: Any = {
                "path": display_path(DEFAULT_PATHS.workfile),
                "summary": progress_summary(payload, sources),
            }
        elif args.command == "check":
            result = {"status": "valid", "summary": check_workfile()}
        elif args.command == "summary":
            result = check_workfile()
        elif args.command == "show":
            result = show_record(args.discovery_id)
        elif args.command == "next":
            result = next_record()
        elif args.command == "set":
            updated = set_record_review(
                discovery_id=args.discovery_id,
                selected_final_intent=args.final_intent,
                confidence=args.confidence,
                reviewer_note=args.reviewer_note,
            )
            sources = load_frozen_sources()
            result = progress_summary(updated, sources)
        elif args.command == "export":
            result = export_adjudication()
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(f"unhandled command: {args.command}")
    except (
        FileNotFoundError,
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
