"""Build the deterministic V2-C5 expanded development dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"

DEVELOPMENT_DATASET_SHA256 = (
    "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca"
)
DISCOVERY_CORPUS_SHA256 = (
    "0a873fe5797fdd93689e7d5054b029cd7b9068017b0ecff54af000641e282e01"
)
DISCOVERY_CORPUS_MANIFEST_SHA256 = (
    "8917c6b27b59c672286672ea3ad915b0195d9ef937605662a01e4209bca4162d"
)
DISCOVERY_ASSIGNMENTS_SHA256 = (
    "aa91a38a3dde57747a8694c404ef00105dcdb0e4233c995749edc1d129596ac0"
)
DISCOVERY_REPORT_SHA256 = (
    "ca007ab1ade5e07e377022a5add49c641b209ae2812c3d4217734c0aeaa7c3c1"
)
DISCOVERY_MANIFEST_SHA256 = (
    "a0ea3f8353c12b341fbb4142b67dff507c227d94a55c73993f9fca9645162fb9"
)
TAXONOMY_FREEZE_SHA256 = (
    "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
)
TAXONOMY_FREEZE_MANIFEST_SHA256 = (
    "359eb38e5ff63af0c9cc0f5db19951b8626f9e15cb136f23046648d655a0f7a9"
)
SPLIT_ADJUDICATION_SHA256 = (
    "4ddf3df61e371de0be82affa4bcd15272ac598cf432d85ce876a6b9a18e1ee65"
)
SPLIT_ADJUDICATION_MANIFEST_SHA256 = (
    "3d92dbea5a3a7751e45ab8c8698528a4d0da446310afd05814810373700ade44"
)

EXPECTED_DEVELOPMENT_COUNT = 8198
EXPECTED_SUPPORTED_COUNT = 1825
EXPECTED_UNSUPPORTED_COUNT = 6373
EXPECTED_DISCOVERY_COUNT = 6372
EXPECTED_SPLIT_COUNT = 975
EXPECTED_MANUAL_SPLIT_COUNT = 578
EXPECTED_DETERMINISTIC_SPLIT_COUNT = 397
EXPECTED_PRIMARY_CLUSTER_COUNT = 35
EXPECTED_PRIMARY_NOISE_COUNT = 3278
PRIMARY_CONFIG_ID = "mcs30_ms10"
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"

DATASET_SCHEMA_VERSION = "v2c5-expanded-development-dataset.v1"
MANIFEST_SCHEMA_VERSION = "v2c5-expanded-development-dataset-manifest.v1"
DATASET_VERSION = "v2c5-expanded-development.v1"
BUILDER_VERSION = "v2c5-expanded-development-dataset-builder.v1"

RETAINED_EXISTING_INTENT = "RETAINED_EXISTING_INTENT"
PRIMARY_NOISE = "PRIMARY_NOISE"
FROZEN_CLUSTER_RESOLUTION = "FROZEN_CLUSTER_RESOLUTION"
SPLIT_ADJUDICATION = "SPLIT_ADJUDICATION"
RELABEL_SOURCES = (
    RETAINED_EXISTING_INTENT,
    PRIMARY_NOISE,
    FROZEN_CLUSTER_RESOLUTION,
    SPLIT_ADJUDICATION,
)

PROTECTED_WRITE_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)

ALLOWED_SOURCE_SPLITS = frozenset(
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
    "selection_probe",
    "test",
)


@dataclass(frozen=True)
class BuildPaths:
    development_dataset: Path
    discovery_corpus: Path
    discovery_corpus_manifest: Path
    discovery_assignments: Path
    discovery_report: Path
    discovery_manifest: Path
    taxonomy_freeze: Path
    taxonomy_freeze_manifest: Path
    split_adjudication: Path
    split_adjudication_manifest: Path
    dataset_output: Path
    manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    development_dataset=ML_ROOT / "v2c3_development_dataset.json",
    discovery_corpus=ML_ROOT / "v2c5_discovery_corpus.json",
    discovery_corpus_manifest=(
        ML_ROOT / "v2c5_discovery_corpus.manifest.json"
    ),
    discovery_assignments=(
        ML_ROOT / "v2c5_intent_discovery_assignments.json"
    ),
    discovery_report=ML_ROOT / "v2c5_intent_discovery_report.json",
    discovery_manifest=ML_ROOT / "v2c5_intent_discovery.manifest.json",
    taxonomy_freeze=ML_ROOT / "v2c5_taxonomy_freeze.json",
    taxonomy_freeze_manifest=ML_ROOT / "v2c5_taxonomy_freeze.manifest.json",
    split_adjudication=ML_ROOT / "v2c5_split_relabel_adjudication.json",
    split_adjudication_manifest=(
        ML_ROOT / "v2c5_split_relabel_adjudication.manifest.json"
    ),
    dataset_output=ML_ROOT / "v2c5_expanded_development_dataset.json",
    manifest_output=(
        ML_ROOT / "v2c5_expanded_development_dataset.manifest.json"
    ),
)


@dataclass(frozen=True)
class FrozenSources:
    development_dataset: dict[str, Any]
    discovery_corpus: dict[str, Any]
    discovery_assignments: dict[str, Any]
    taxonomy_freeze: dict[str, Any]
    split_adjudication: dict[str, Any]
    development_examples: list[dict[str, Any]]
    corpus_by_discovery_id: dict[str, dict[str, Any]]
    corpus_by_normalized_hash: dict[str, dict[str, Any]]
    primary_assignments: dict[str, dict[str, Any]]
    frozen_cluster_resolutions: dict[str, str]
    split_cluster_ids: frozenset[str]
    split_allowed_targets: dict[str, frozenset[str]]
    split_decisions: dict[str, dict[str, Any]]
    intent_labels: tuple[str, ...]
    risk_by_intent: dict[str, str]
    source_artifacts: dict[str, dict[str, str]]


@dataclass(frozen=True)
class BuiltArtifacts:
    dataset_bytes: bytes
    manifest_bytes: bytes
    dataset_payload: dict[str, Any]
    manifest_payload: dict[str, Any]


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def normalized_text_sha256(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("normalized text cannot be empty")
    return sha256_bytes(normalized.encode("utf-8"))


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def source_path_items(paths: BuildPaths) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("development_dataset", paths.development_dataset, DEVELOPMENT_DATASET_SHA256),
        ("discovery_corpus", paths.discovery_corpus, DISCOVERY_CORPUS_SHA256),
        (
            "discovery_corpus_manifest",
            paths.discovery_corpus_manifest,
            DISCOVERY_CORPUS_MANIFEST_SHA256,
        ),
        (
            "discovery_assignments",
            paths.discovery_assignments,
            DISCOVERY_ASSIGNMENTS_SHA256,
        ),
        ("discovery_report", paths.discovery_report, DISCOVERY_REPORT_SHA256),
        (
            "discovery_manifest",
            paths.discovery_manifest,
            DISCOVERY_MANIFEST_SHA256,
        ),
        ("taxonomy_freeze", paths.taxonomy_freeze, TAXONOMY_FREEZE_SHA256),
        (
            "taxonomy_freeze_manifest",
            paths.taxonomy_freeze_manifest,
            TAXONOMY_FREEZE_MANIFEST_SHA256,
        ),
        (
            "split_adjudication",
            paths.split_adjudication,
            SPLIT_ADJUDICATION_SHA256,
        ),
        (
            "split_adjudication_manifest",
            paths.split_adjudication_manifest,
            SPLIT_ADJUDICATION_MANIFEST_SHA256,
        ),
    )


def allowed_source_filenames() -> frozenset[str]:
    return frozenset(path.name for _, path, _ in source_path_items(DEFAULT_PATHS))


def guard_source_path(path: Path) -> None:
    normalized = path.as_posix().lower()
    if path.name not in allowed_source_filenames():
        raise ValueError(f"Step 21B source input is not allowed: {path.name}")
    if any(token in normalized for token in FORBIDDEN_INPUT_PATH_TOKENS):
        raise ValueError(f"prohibited non-development input path: {path}")


def load_json_object(value: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


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


def source_artifact_metadata(paths: BuildPaths) -> dict[str, dict[str, str]]:
    return {
        label: {"path": display_path(path), "sha256": expected_hash}
        for label, path, expected_hash in source_path_items(paths)
    }


def require_object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{label} must be a list of objects")
    return value


def validate_development_dataset(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != "v2c3-development-dataset.v1":
        raise ValueError("unexpected V2-C3 development dataset schema")
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("V2-C3 normalization contract changed")
    if payload.get("training_or_evaluation_performed") is not False:
        raise ValueError("V2-C3 development execution policy changed")
    records = require_object_list(payload.get("examples"), "development examples")
    if payload.get("example_count") != len(records):
        raise ValueError("V2-C3 development example count mismatch")
    if len(records) != EXPECTED_DEVELOPMENT_COUNT:
        raise ValueError("development population must contain exactly 8198 records")

    example_ids: set[str] = set()
    intent_counts: Counter[str] = Counter()
    for record in records:
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in example_ids:
            raise ValueError("development example IDs must be unique non-empty strings")
        example_ids.add(example_id)
        if record.get("data_role") != "development":
            raise ValueError("final, test, or holdout role entered Step 21B")
        source_key = (record.get("source_id"), record.get("source_split"))
        if source_key not in ALLOWED_SOURCE_SPLITS:
            raise ValueError(f"prohibited development source or split: {source_key}")
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("development text must be a non-empty string")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError("development text hash changed")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError("development normalized-text hash changed")
        intent = record.get("intent")
        if not isinstance(intent, str) or not intent:
            raise ValueError("development intent must be a non-empty string")
        intent_counts[intent] += 1
    if intent_counts[UNSUPPORTED_INTENT] != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError("historical unsupported count must equal exactly 6373")
    if len(records) - intent_counts[UNSUPPORTED_INTENT] != EXPECTED_SUPPORTED_COUNT:
        raise ValueError("historical supported count must equal exactly 1825")
    return records


def validate_discovery_corpus(
    payload: Mapping[str, Any],
    development_examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if payload.get("schema_version") != "v2c5-discovery-corpus.v1":
        raise ValueError("unexpected Step 17 discovery corpus schema")
    if payload.get("data_role") != "v2c5_primary_intent_discovery_corpus":
        raise ValueError("unexpected Step 17 discovery corpus role")
    records = require_object_list(
        payload.get("primary_cluster_population"),
        "primary discovery population",
    )
    if len(records) != EXPECTED_DISCOVERY_COUNT:
        raise ValueError("discovery population must contain exactly 6372 records")

    unsupported_by_id = {
        str(record["example_id"]): record
        for record in development_examples
        if record["intent"] == UNSUPPORTED_INTENT
    }
    by_discovery_id: dict[str, dict[str, Any]] = {}
    by_normalized_hash: dict[str, dict[str, Any]] = {}
    occurrence_ids: list[str] = []
    occurrence_counts: Counter[int] = Counter()
    for record in records:
        discovery_id = record.get("discovery_id")
        normalized_hash = record.get("normalized_text_sha256")
        if not isinstance(normalized_hash, str) or not normalized_hash:
            raise ValueError("discovery normalized hash must be a non-empty string")
        if discovery_id != f"v2c5-discovery:{normalized_hash}":
            raise ValueError("discovery ID does not match normalized-text hash")
        if discovery_id in by_discovery_id or normalized_hash in by_normalized_hash:
            raise ValueError("duplicate discovery identity")
        if record.get("current_sentinelvoice_intent") != UNSUPPORTED_INTENT:
            raise ValueError("supported intent entered primary discovery population")
        normalized = record.get("normalized_text")
        if not isinstance(normalized, str) or sha256_bytes(
            normalized.encode("utf-8")
        ) != normalized_hash:
            raise ValueError("discovery normalized text hash changed")
        if normalize_text(str(record.get("text"))) != normalized:
            raise ValueError("discovery normalization contract changed")
        occurrences = require_object_list(
            record.get("occurrences"),
            "discovery occurrences",
        )
        if record.get("source_occurrence_count") != len(occurrences):
            raise ValueError("discovery source occurrence count changed")
        occurrence_counts[len(occurrences)] += 1
        for occurrence in occurrences:
            example_id = occurrence.get("example_id")
            if not isinstance(example_id, str) or example_id not in unsupported_by_id:
                raise ValueError("discovery occurrence is absent from development data")
            source = unsupported_by_id[example_id]
            expected = {
                "data_role": "development",
                "group_id": source.get("group_id"),
                "source_dataset": source.get("source_id"),
                "source_row_index": source.get("source_row_index"),
                "source_split": source.get("source_split"),
                "text_sha256": source.get("text_sha256"),
            }
            if any(occurrence.get(key) != value for key, value in expected.items()):
                raise ValueError("discovery occurrence lineage changed")
            if source.get("normalized_text_sha256") != normalized_hash:
                raise ValueError("discovery occurrence normalized hash changed")
            occurrence_ids.append(example_id)
        by_discovery_id[str(discovery_id)] = record
        by_normalized_hash[normalized_hash] = record

    if len(occurrence_ids) != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError("discovery occurrence coverage must equal exactly 6373")
    if len(set(occurrence_ids)) != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError("a development occurrence appears more than once in discovery")
    if set(occurrence_ids) != set(unsupported_by_id):
        raise ValueError("discovery occurrences do not cover historical unsupported data")
    if occurrence_counts != Counter({1: 6371, 2: 1}):
        raise ValueError("known one-occurrence normalization duplicate changed")
    return by_discovery_id, by_normalized_hash


def validate_discovery_corpus_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-discovery-corpus-manifest.v1":
        raise ValueError("unexpected Step 17 discovery manifest schema")
    expected_corpus = {
        "path": "data/evals/v2/ml/v2c5_discovery_corpus.json",
        "sha256": DISCOVERY_CORPUS_SHA256,
    }
    if payload.get("corpus") != expected_corpus:
        raise ValueError("Step 17 corpus binding changed")
    source_hashes = payload.get("source_input_hashes")
    if not isinstance(source_hashes, dict) or source_hashes.get(
        "v2c3_development_dataset"
    ) != DEVELOPMENT_DATASET_SHA256:
        raise ValueError("Step 17 development-dataset binding changed")
    counts = payload.get("counts")
    expected_counts = {
        "duplicate_group_count": 1,
        "duplicate_occurrence_count_removed_from_density": 1,
        "primary_unique_normalized_text_count": EXPECTED_DISCOVERY_COUNT,
        "primary_unsupported_occurrence_count_before_dedup": (
            EXPECTED_UNSUPPORTED_COUNT
        ),
        "raw_eligible_occurrence_count": EXPECTED_DEVELOPMENT_COUNT,
        "reference_anchor_count": EXPECTED_SUPPORTED_COUNT,
    }
    if not isinstance(counts, dict) or any(
        counts.get(key) != value for key, value in expected_counts.items()
    ):
        raise ValueError("Step 17 discovery counts changed")
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict) or normalization.get(
        "version"
    ) != NORMALIZATION_VERSION:
        raise ValueError("Step 17 normalization contract changed")
    checks = payload.get("exclusion_and_integrity_checks")
    required_zero = {
        "banking77_test_records",
        "cfpb_records",
        "clinc_test_records",
        "consumed_v2c3_challenge_records",
        "consumed_v2c3_external_lockbox_records",
        "final_or_test_role_records",
        "v2c3_internal_locked_test_records",
        "v2c4_selection_probe_records",
    }
    if not isinstance(checks, dict) or any(checks.get(key) != 0 for key in required_zero):
        raise ValueError("Step 17 manifest admits a prohibited source")
    if checks.get("sealed_v2c4_holdout_accessed") is not False:
        raise ValueError("Step 17 manifest accessed the sealed V2-C4 holdout")


def build_primary_assignment_index(
    payload: Mapping[str, Any],
    corpus_by_discovery_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != "v2c5-intent-discovery-assignments.v1":
        raise ValueError("unexpected Step 18 assignment schema")
    if payload.get("primary_config_id") != PRIMARY_CONFIG_ID:
        raise ValueError("Step 21B requires primary mcs30_ms10 assignments")
    if payload.get("corpus_sha256") != DISCOVERY_CORPUS_SHA256:
        raise ValueError("Step 18 assignment corpus binding changed")
    if payload.get("raw_text_persisted") is not False:
        raise ValueError("Step 18 assignment raw-text policy changed")
    if payload.get("semantic_intent_names_assigned") is not False:
        raise ValueError("Step 18 assignments cannot contain semantic intent names")
    records = require_object_list(payload.get("assignments"), "primary assignments")
    if payload.get("assignment_count") != len(records) or len(records) != (
        EXPECTED_DISCOVERY_COUNT
    ):
        raise ValueError("Step 18 assignment count must equal exactly 6372")

    result: dict[str, dict[str, Any]] = {}
    for record in records:
        discovery_id = record.get("discovery_id")
        if not isinstance(discovery_id, str) or not discovery_id:
            raise ValueError("assignment discovery ID must be a non-empty string")
        if discovery_id in result:
            raise ValueError("duplicate primary assignment discovery ID")
        corpus_record = corpus_by_discovery_id.get(discovery_id)
        if corpus_record is None:
            raise ValueError("primary assignment is absent from discovery corpus")
        if record.get("normalized_text_sha256") != corpus_record.get(
            "normalized_text_sha256"
        ):
            raise ValueError("primary assignment normalized hash changed")
        cluster_id = record.get("primary_canonical_cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError("primary canonical cluster ID must be a non-empty string")
        is_noise = record.get("primary_is_noise")
        if not isinstance(is_noise, bool):
            raise TypeError("primary noise flag must be boolean")
        if not isinstance(record.get("sensitivity_assignments"), dict):
            raise TypeError("sensitivity assignments must be a diagnostic object")
        result[discovery_id] = {
            "discovery_id": discovery_id,
            "normalized_text_sha256": record["normalized_text_sha256"],
            "primary_canonical_cluster_id": cluster_id,
            "primary_is_noise": is_noise,
        }
    if set(result) != set(corpus_by_discovery_id):
        raise ValueError("primary assignments do not exactly cover discovery corpus")
    return result


def validate_discovery_report(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-intent-discovery-report.v1":
        raise ValueError("unexpected Step 18 report schema")
    if payload.get("corpus_sha256") != DISCOVERY_CORPUS_SHA256:
        raise ValueError("Step 18 report corpus binding changed")
    primary = payload.get("primary")
    if not isinstance(primary, dict):
        raise TypeError("Step 18 primary report must be an object")
    expected = {
        "cluster_count_excluding_noise": EXPECTED_PRIMARY_CLUSTER_COUNT,
        "config_id": PRIMARY_CONFIG_ID,
        "example_count": EXPECTED_DISCOVERY_COUNT,
        "noise_count": EXPECTED_PRIMARY_NOISE_COUNT,
        "primary_result_replaceable_post_hoc": False,
    }
    if any(primary.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 18 primary report changed")
    if payload.get("taxonomy_changed") is not False:
        raise ValueError("Step 18 report cannot change the taxonomy")
    if payload.get("supervised_training_performed") is not False:
        raise ValueError("Step 18 report cannot contain supervised training")


def validate_discovery_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-intent-discovery-manifest.v1":
        raise ValueError("unexpected Step 18 manifest schema")
    expected_bindings = {
        "assignments": (
            "data/evals/v2/ml/v2c5_intent_discovery_assignments.json",
            DISCOVERY_ASSIGNMENTS_SHA256,
        ),
        "report": (
            "data/evals/v2/ml/v2c5_intent_discovery_report.json",
            DISCOVERY_REPORT_SHA256,
        ),
    }
    for key, (path, digest) in expected_bindings.items():
        if payload.get(key) != {"path": path, "sha256": digest}:
            raise ValueError(f"Step 18 {key} binding changed")
    frozen = payload.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise TypeError("Step 18 frozen inputs must be an object")
    if frozen.get("corpus") != {
        "path": "data/evals/v2/ml/v2c5_discovery_corpus.json",
        "sha256": DISCOVERY_CORPUS_SHA256,
    }:
        raise ValueError("Step 18 corpus binding changed")
    if frozen.get("corpus_manifest") != {
        "path": "data/evals/v2/ml/v2c5_discovery_corpus.manifest.json",
        "sha256": DISCOVERY_CORPUS_MANIFEST_SHA256,
    }:
        raise ValueError("Step 18 corpus-manifest binding changed")
    hdbscan = payload.get("hdbscan")
    if not isinstance(hdbscan, dict) or hdbscan.get(
        "primary_config_id"
    ) != PRIMARY_CONFIG_ID:
        raise ValueError("Step 18 primary assignment provenance changed")


def build_cluster_resolution_indexes(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, frozenset[str]],
    tuple[str, ...],
    dict[str, str],
]:
    if payload.get("schema_version") != "v2c5-taxonomy-freeze.v1":
        raise ValueError("unexpected Step 20 taxonomy schema")
    expected_governance = {
        "classifier_training_performed": False,
        "dataset_relabeling_performed": False,
        "final_intent_count": 16,
        "final_taxonomy_frozen": True,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "phase": "V2-C5 Step 20",
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "step21_required": True,
        "taxonomy_version": "v2c5-taxonomy.v1",
    }
    if any(payload.get(key) != value for key, value in expected_governance.items()):
        raise ValueError("Step 20 taxonomy governance changed")
    labels_value = payload.get("intent_label_order")
    if not isinstance(labels_value, list) or any(
        not isinstance(label, str) or not label for label in labels_value
    ):
        raise TypeError("Step 20 intent label order must contain strings")
    labels = tuple(labels_value)
    if len(labels) != 16 or tuple(sorted(set(labels))) != labels:
        raise ValueError("Step 20 label space must be 16 unique sorted intents")
    if tuple(payload.get("protected_write_intents", [])) != PROTECTED_WRITE_INTENTS:
        raise ValueError("protected-write intent set changed")
    risks_value = payload.get("risk_by_intent")
    if not isinstance(risks_value, dict) or set(risks_value) != set(labels):
        raise ValueError("Step 20 risk mapping must cover the exact label space")
    risks = {str(key): str(value) for key, value in risks_value.items()}

    fixed: dict[str, str] = {}
    for field in ("candidate_cluster_resolutions", "carry_forward_cluster_resolutions"):
        for record in require_object_list(payload.get(field), field):
            cluster_id = record.get("canonical_cluster_id")
            final_intent = record.get("final_intent")
            if not isinstance(cluster_id, str) or not cluster_id or cluster_id in fixed:
                raise ValueError("duplicate or invalid frozen cluster resolution")
            if final_intent not in labels:
                raise ValueError("frozen cluster resolution is outside the taxonomy")
            fixed[cluster_id] = str(final_intent)

    split_targets: dict[str, frozenset[str]] = {}
    for record in require_object_list(
        payload.get("split_cluster_resolutions"),
        "split cluster resolutions",
    ):
        cluster_id = record.get("canonical_cluster_id")
        if (
            not isinstance(cluster_id, str)
            or not cluster_id
            or cluster_id in fixed
            or cluster_id in split_targets
        ):
            raise ValueError("duplicate or invalid split-cluster resolution")
        if record.get("resolution") != "BRANCH_FOR_STEP21_RELABELING":
            raise ValueError("Step 20 split-cluster policy changed")
        branches = require_object_list(record.get("branches"), "split branches")
        if not branches or any(branch.get("final_intent") not in labels for branch in branches):
            raise ValueError("split branch target is outside the frozen taxonomy")
        split_targets[cluster_id] = frozenset(
            str(branch["final_intent"]) for branch in branches
        )
    if len(fixed) + len(split_targets) != EXPECTED_PRIMARY_CLUSTER_COUNT:
        raise ValueError("Step 20 must resolve all 35 primary clusters")
    return fixed, split_targets, labels, risks


def validate_taxonomy_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-taxonomy-freeze-manifest.v1":
        raise ValueError("unexpected Step 20 taxonomy manifest schema")
    if payload.get("taxonomy_freeze") != {
        "path": "data/evals/v2/ml/v2c5_taxonomy_freeze.json",
        "sha256": TAXONOMY_FREEZE_SHA256,
    }:
        raise ValueError("Step 20 taxonomy manifest binding changed")
    expected = {
        "classifier_training_performed": False,
        "final_intent_count": 16,
        "final_taxonomy_frozen": True,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "step21_required": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 20 taxonomy manifest governance changed")


def build_split_decision_index(
    payload: Mapping[str, Any],
    split_allowed_targets: Mapping[str, frozenset[str]],
    primary_assignments: Mapping[str, Mapping[str, Any]],
    taxonomy_labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != "v2c5-split-relabel-adjudication.v1":
        raise ValueError("unexpected Step 21A adjudication schema")
    expected = {
        "all_reviewed": True,
        "classifier_training_performed": False,
        "deterministic_mapping_count": EXPECTED_DETERMINISTIC_SPLIT_COUNT,
        "development_dataset_built": False,
        "fresh_holdout_created": False,
        "manual_review_count": EXPECTED_MANUAL_SPLIT_COUNT,
        "phase": "V2-C5 Step 21A",
        "runtime_behavior_changed": False,
        "split_coverage_count": EXPECTED_SPLIT_COUNT,
        "step21_complete": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 21A adjudication contract changed")
    manual = require_object_list(payload.get("manual_adjudications"), "manual splits")
    deterministic = require_object_list(
        payload.get("deterministic_mappings"),
        "deterministic splits",
    )
    if len(manual) != EXPECTED_MANUAL_SPLIT_COUNT or len(deterministic) != (
        EXPECTED_DETERMINISTIC_SPLIT_COUNT
    ):
        raise ValueError("Step 21A split record counts changed")

    decisions: dict[str, dict[str, Any]] = {}
    for records, is_manual in ((manual, True), (deterministic, False)):
        for record in records:
            discovery_id = record.get("discovery_id")
            cluster_id = record.get("canonical_cluster_id")
            final_intent = (
                record.get("selected_final_intent")
                if is_manual
                else record.get("final_intent")
            )
            if not isinstance(discovery_id, str) or discovery_id in decisions:
                raise ValueError("duplicate or invalid Step 21A discovery ID")
            assignment = primary_assignments.get(discovery_id)
            if assignment is None:
                raise ValueError("Step 21A record is absent from primary assignments")
            if assignment["primary_is_noise"] is not False:
                raise ValueError("Step 21A record cannot be primary noise")
            if cluster_id != assignment["primary_canonical_cluster_id"]:
                raise ValueError("Step 21A cluster does not match primary assignment")
            if cluster_id not in split_allowed_targets:
                raise ValueError("Step 21A record is not in a frozen split cluster")
            if final_intent not in taxonomy_labels:
                raise ValueError("Step 21A final intent is outside the frozen taxonomy")
            if final_intent not in split_allowed_targets[str(cluster_id)]:
                raise ValueError("Step 21A intent violates its frozen split branches")
            confidence = record.get("confidence") if is_manual else None
            if is_manual and confidence not in {"HIGH", "MEDIUM", "LOW"}:
                raise ValueError("manual Step 21A record has invalid confidence")
            decisions[discovery_id] = {
                "canonical_cluster_id": cluster_id,
                "confidence": confidence,
                "discovery_id": discovery_id,
                "final_intent": final_intent,
                "resolution_kind": "HUMAN" if is_manual else "DETERMINISTIC",
            }
    expected_ids = {
        discovery_id
        for discovery_id, assignment in primary_assignments.items()
        if assignment["primary_canonical_cluster_id"] in split_allowed_targets
    }
    if set(decisions) != expected_ids or len(decisions) != EXPECTED_SPLIT_COUNT:
        raise ValueError("Step 21A does not exactly cover all split discovery IDs")
    return decisions


def validate_split_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != (
        "v2c5-split-relabel-adjudication-manifest.v1"
    ):
        raise ValueError("unexpected Step 21A adjudication manifest schema")
    if payload.get("adjudication") != {
        "path": "data/evals/v2/ml/v2c5_split_relabel_adjudication.json",
        "sha256": SPLIT_ADJUDICATION_SHA256,
    }:
        raise ValueError("Step 21A adjudication binding changed")
    source_artifacts = payload.get("source_artifacts")
    expected_sources = {
        "development_dataset": DEVELOPMENT_DATASET_SHA256,
        "discovery_assignments": DISCOVERY_ASSIGNMENTS_SHA256,
        "discovery_corpus": DISCOVERY_CORPUS_SHA256,
        "taxonomy_freeze": TAXONOMY_FREEZE_SHA256,
        "taxonomy_freeze_manifest": TAXONOMY_FREEZE_MANIFEST_SHA256,
    }
    if not isinstance(source_artifacts, dict):
        raise TypeError("Step 21A source artifacts must be an object")
    for name, expected_hash in expected_sources.items():
        artifact = source_artifacts.get(name)
        if not isinstance(artifact, dict) or artifact.get("sha256") != expected_hash:
            raise ValueError(f"Step 21A source binding changed: {name}")
    expected = {
        "all_reviewed": True,
        "classifier_training_performed": False,
        "deterministic_mapping_count": EXPECTED_DETERMINISTIC_SPLIT_COUNT,
        "development_dataset_built": False,
        "fresh_holdout_created": False,
        "manual_review_count": EXPECTED_MANUAL_SPLIT_COUNT,
        "phase": "V2-C5 Step 21A",
        "runtime_behavior_changed": False,
        "split_coverage_count": EXPECTED_SPLIT_COUNT,
        "step21_complete": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 21A adjudication manifest governance changed")


def load_frozen_sources(paths: BuildPaths = DEFAULT_PATHS) -> FrozenSources:
    payloads: dict[str, dict[str, Any]] = {}
    for label, path, expected_hash in source_path_items(paths):
        guard_source_path(path)
        value = path.read_bytes()
        if sha256_bytes(value) != expected_hash:
            raise ValueError(f"frozen {label.replace('_', ' ')} SHA-256 changed")
        payloads[label] = load_json_object(value, label.replace("_", " "))

    development_examples = validate_development_dataset(
        payloads["development_dataset"]
    )
    corpus_by_id, corpus_by_hash = validate_discovery_corpus(
        payloads["discovery_corpus"],
        development_examples,
    )
    validate_discovery_corpus_manifest(payloads["discovery_corpus_manifest"])
    primary_assignments = build_primary_assignment_index(
        payloads["discovery_assignments"],
        corpus_by_id,
    )
    validate_discovery_report(payloads["discovery_report"])
    validate_discovery_manifest(payloads["discovery_manifest"])
    fixed, split_targets, labels, risks = build_cluster_resolution_indexes(
        payloads["taxonomy_freeze"]
    )
    validate_taxonomy_manifest(payloads["taxonomy_freeze_manifest"])
    non_noise_clusters = {
        str(assignment["primary_canonical_cluster_id"])
        for assignment in primary_assignments.values()
        if assignment["primary_is_noise"] is False
    }
    if set(fixed) | set(split_targets) != non_noise_clusters:
        raise ValueError("Step 20 resolutions do not match primary non-noise clusters")
    split_decisions = build_split_decision_index(
        payloads["split_adjudication"],
        split_targets,
        primary_assignments,
        labels,
    )
    validate_split_manifest(payloads["split_adjudication_manifest"])
    return FrozenSources(
        development_dataset=payloads["development_dataset"],
        discovery_corpus=payloads["discovery_corpus"],
        discovery_assignments=payloads["discovery_assignments"],
        taxonomy_freeze=payloads["taxonomy_freeze"],
        split_adjudication=payloads["split_adjudication"],
        development_examples=development_examples,
        corpus_by_discovery_id=corpus_by_id,
        corpus_by_normalized_hash=corpus_by_hash,
        primary_assignments=primary_assignments,
        frozen_cluster_resolutions=fixed,
        split_cluster_ids=frozenset(split_targets),
        split_allowed_targets=split_targets,
        split_decisions=split_decisions,
        intent_labels=labels,
        risk_by_intent=risks,
        source_artifacts=source_artifact_metadata(paths),
    )


def resolve_unsupported_example(
    example: Mapping[str, Any],
    sources: FrozenSources,
) -> tuple[str, str, str, str, str | None]:
    normalized_hash = normalized_text_sha256(str(example["text"]))
    corpus_record = sources.corpus_by_normalized_hash.get(normalized_hash)
    if corpus_record is None:
        raise ValueError("unsupported occurrence has no Step 17 discovery record")
    discovery_id = str(corpus_record["discovery_id"])
    assignment = sources.primary_assignments.get(discovery_id)
    if assignment is None:
        raise ValueError("discovery record has no unique primary assignment")
    cluster_id = str(assignment["primary_canonical_cluster_id"])
    if assignment["primary_is_noise"] is True:
        return UNSUPPORTED_INTENT, PRIMARY_NOISE, discovery_id, cluster_id, None
    if cluster_id in sources.frozen_cluster_resolutions:
        return (
            sources.frozen_cluster_resolutions[cluster_id],
            FROZEN_CLUSTER_RESOLUTION,
            discovery_id,
            cluster_id,
            None,
        )
    decision = sources.split_decisions.get(discovery_id)
    if decision is None or cluster_id not in sources.split_cluster_ids:
        raise ValueError("non-noise discovery record lacks a frozen resolution")
    return (
        str(decision["final_intent"]),
        SPLIT_ADJUDICATION,
        discovery_id,
        cluster_id,
        decision.get("confidence"),
    )


def build_expanded_examples(sources: FrozenSources) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in sources.development_examples:
        previous_intent = str(source["intent"])
        if previous_intent != UNSUPPORTED_INTENT:
            final_intent = previous_intent
            relabel_source = RETAINED_EXISTING_INTENT
            discovery_id = None
            cluster_id = None
            confidence = None
        else:
            (
                final_intent,
                relabel_source,
                discovery_id,
                cluster_id,
                confidence,
            ) = resolve_unsupported_example(source, sources)
        if final_intent not in sources.intent_labels:
            raise ValueError("derived final intent is outside the frozen taxonomy")
        risk = sources.risk_by_intent.get(final_intent)
        if risk is None:
            raise ValueError("derived final intent has no frozen risk mapping")
        record = copy.deepcopy(source)
        record.update(
            {
                "intent": final_intent,
                "previous_intent": previous_intent,
                "risk": risk,
                "v2c5_discovery_id": discovery_id,
                "v2c5_primary_canonical_cluster_id": cluster_id,
                "v2c5_relabel_source": relabel_source,
                "v2c5_split_adjudication_confidence": confidence,
            }
        )
        output.append(record)
    validate_expanded_examples(output, sources)
    return output


def validate_expanded_examples(
    records: Sequence[Mapping[str, Any]],
    sources: FrozenSources,
) -> None:
    if len(records) != EXPECTED_DEVELOPMENT_COUNT:
        raise ValueError("expanded development dataset must contain 8198 records")
    source_ids = [str(value["example_id"]) for value in sources.development_examples]
    output_ids = [str(value.get("example_id")) for value in records]
    if output_ids != source_ids or len(set(output_ids)) != EXPECTED_DEVELOPMENT_COUNT:
        raise ValueError("expanded dataset changed the original example population")

    unsupported_discovery_ids: list[str] = []
    for source, output in zip(sources.development_examples, records, strict=True):
        for field, value in source.items():
            if field not in {"intent", "risk"} and output.get(field) != value:
                raise ValueError(f"historical development field changed: {field}")
        previous_intent = source["intent"]
        if output.get("previous_intent") != previous_intent:
            raise ValueError("previous intent provenance changed")
        final_intent = output.get("intent")
        if final_intent not in sources.intent_labels:
            raise ValueError("output intent is outside the frozen 16-intent taxonomy")
        if output.get("risk") != sources.risk_by_intent[final_intent]:
            raise ValueError("output risk disagrees with the Step 20 freeze")
        if output.get("v2c5_relabel_source") not in RELABEL_SOURCES:
            raise ValueError("unknown V2-C5 relabel source")
        if previous_intent != UNSUPPORTED_INTENT:
            if final_intent != previous_intent:
                raise ValueError("historically supported intent was reinterpreted")
            if output.get("v2c5_relabel_source") != RETAINED_EXISTING_INTENT:
                raise ValueError("supported record has wrong relabel provenance")
            if any(
                output.get(field) is not None
                for field in (
                    "v2c5_discovery_id",
                    "v2c5_primary_canonical_cluster_id",
                    "v2c5_split_adjudication_confidence",
                )
            ):
                raise ValueError("supported record contains discovery provenance")
        else:
            discovery_id = output.get("v2c5_discovery_id")
            if not isinstance(discovery_id, str) or discovery_id not in (
                sources.corpus_by_discovery_id
            ):
                raise ValueError("unsupported output lacks discovery provenance")
            expected_resolution = resolve_unsupported_example(source, sources)
            actual_resolution = (
                final_intent,
                output.get("v2c5_relabel_source"),
                discovery_id,
                output.get("v2c5_primary_canonical_cluster_id"),
                output.get("v2c5_split_adjudication_confidence"),
            )
            if actual_resolution != expected_resolution:
                raise ValueError("unsupported output disagrees with frozen evidence")
            unsupported_discovery_ids.append(discovery_id)

    if len(unsupported_discovery_ids) != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError("expanded unsupported occurrence count changed")
    discovery_counts = Counter(unsupported_discovery_ids)
    if len(discovery_counts) != EXPECTED_DISCOVERY_COUNT:
        raise ValueError("expanded unsupported unique discovery count changed")
    duplicate_counts = sorted(discovery_counts.values())
    if duplicate_counts.count(2) != 1 or duplicate_counts.count(1) != 6371:
        raise ValueError("normalized duplicate occurrence was not preserved")
    duplicate_id = next(key for key, value in discovery_counts.items() if value == 2)
    duplicate_records = [
        record for record in records if record.get("v2c5_discovery_id") == duplicate_id
    ]
    duplicate_labels = {
        (
            record["intent"],
            record["v2c5_primary_canonical_cluster_id"],
            record["v2c5_relabel_source"],
        )
        for record in duplicate_records
    }
    if len(duplicate_labels) != 1:
        raise ValueError("normalized duplicate received conflicting relabel decisions")


def execution_status() -> dict[str, Any]:
    return {
        "classifier_evaluation_performed": False,
        "classifier_training_performed": False,
        "expanded_development_dataset_built": True,
        "fresh_holdout_created": False,
        "runtime_behavior_changed": False,
        "step21_complete": False,
        "step22_permitted": False,
    }


def build_dataset_payload(
    records: Sequence[Mapping[str, Any]],
    sources: FrozenSources,
) -> dict[str, Any]:
    return {
        "classifier_input_fields": ["text"],
        "dataset_version": DATASET_VERSION,
        "example_count": len(records),
        "examples": [copy.deepcopy(dict(record)) for record in records],
        "execution_status": execution_status(),
        "metadata_fields_are_classifier_features": False,
        "normalization_version": NORMALIZATION_VERSION,
        "phase": "V2-C5 Step 21B",
        "relabel_provenance_fields": [
            "previous_intent",
            "v2c5_discovery_id",
            "v2c5_primary_canonical_cluster_id",
            "v2c5_relabel_source",
            "v2c5_split_adjudication_confidence",
        ],
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "taxonomy_version": "v2c5-taxonomy.v1",
    }


def count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record[field]) for record in records)
    return dict(sorted(counts.items()))


def build_manifest_payload(
    dataset_bytes: bytes,
    records: Sequence[Mapping[str, Any]],
    sources: FrozenSources,
    paths: BuildPaths,
    script_path: Path,
) -> dict[str, Any]:
    relabel_counts = count_by(records, "v2c5_relabel_source")
    return {
        "builder": {
            "path": display_path(script_path),
            "sha256": sha256_bytes(script_path.read_bytes()),
            "version": BUILDER_VERSION,
        },
        "counts": {
            "final_intent_counts": count_by(records, "intent"),
            "historical_supported_occurrences": EXPECTED_SUPPORTED_COUNT,
            "historical_unsupported_occurrences": EXPECTED_UNSUPPORTED_COUNT,
            "output_occurrences": len(records),
            "relabel_source_counts": relabel_counts,
            "risk_counts": count_by(records, "risk"),
            "split_discovery_texts": EXPECTED_SPLIT_COUNT,
            "unique_unsupported_discovery_texts": EXPECTED_DISCOVERY_COUNT,
        },
        "dataset": {
            "path": display_path(paths.dataset_output),
            "schema_version": DATASET_SCHEMA_VERSION,
            "sha256": sha256_bytes(dataset_bytes),
            "version": DATASET_VERSION,
        },
        "exclusion_and_integrity_checks": {
            "banking77_test_records": 0,
            "cfpb_records": 0,
            "clinc_test_records": 0,
            "consumed_challenge_or_final_evaluation_records": 0,
            "consumed_v2c3_external_lockbox_records": 0,
            "classifier_scores_used_as_target_evidence": False,
            "duplicate_occurrence_difference": 1,
            "embedding_similarity_used_as_target_evidence": False,
            "every_original_example_preserved_exactly_once": True,
            "final_or_test_role_records": 0,
            "llm_outputs_used_as_target_evidence": False,
            "model_predictions_used_as_target_evidence": False,
            "native_external_labels_used_as_target_evidence": False,
            "sealed_or_holdout_data_accessed": False,
            "sensitivity_assignments_used_as_target_evidence": False,
        },
        "execution_status": execution_status(),
        "next_required": "new_independently_authored_v2c5_final_holdout",
        "phase": "V2-C5 Step 21B",
        "protected_write_intents": list(PROTECTED_WRITE_INTENTS),
        "risk_by_intent": copy.deepcopy(sources.risk_by_intent),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "taxonomy_intent_labels": list(sources.intent_labels),
    }


def validate_dataset_payload(
    payload: Mapping[str, Any],
    sources: FrozenSources,
) -> None:
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unexpected expanded dataset schema")
    if payload.get("example_count") != EXPECTED_DEVELOPMENT_COUNT:
        raise ValueError("expanded dataset count changed")
    records = require_object_list(payload.get("examples"), "expanded examples")
    validate_expanded_examples(records, sources)
    if payload.get("classifier_input_fields") != ["text"]:
        raise ValueError("classifier feature contract changed")
    if payload.get("metadata_fields_are_classifier_features") is not False:
        raise ValueError("relabel provenance cannot become classifier features")
    if payload.get("execution_status") != execution_status():
        raise ValueError("expanded dataset execution status changed")


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    sources: FrozenSources | None = None,
    script_path: Path | None = None,
) -> BuiltArtifacts:
    frozen = sources or load_frozen_sources(paths)
    records = build_expanded_examples(frozen)
    dataset_payload = build_dataset_payload(records, frozen)
    validate_dataset_payload(dataset_payload, frozen)
    dataset_bytes = stable_json_bytes(dataset_payload)
    actual_script_path = script_path or Path(__file__).resolve()
    manifest_payload = build_manifest_payload(
        dataset_bytes,
        records,
        frozen,
        paths,
        actual_script_path,
    )
    return BuiltArtifacts(
        dataset_bytes=dataset_bytes,
        manifest_bytes=stable_json_bytes(manifest_payload),
        dataset_payload=dataset_payload,
        manifest_payload=manifest_payload,
    )


def check_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    dataset_exists = paths.dataset_output.exists()
    manifest_exists = paths.manifest_output.exists()
    if dataset_exists != manifest_exists:
        raise ValueError("expanded dataset outputs must exist as a pair")
    if not dataset_exists:
        return
    if paths.dataset_output.read_bytes() != artifacts.dataset_bytes:
        raise ValueError("existing expanded development dataset is not deterministic")
    if paths.manifest_output.read_bytes() != artifacts.manifest_bytes:
        raise ValueError("existing expanded development manifest is not deterministic")


def write_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    atomic_write_bytes(paths.dataset_output, artifacts.dataset_bytes)
    atomic_write_bytes(paths.manifest_output, artifacts.manifest_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifacts = build_artifacts()
        if args.write:
            write_artifacts(artifacts)
            action = "Wrote"
        else:
            check_artifacts(artifacts)
            action = "Validated"
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc
    counts = artifacts.manifest_payload["counts"]
    print(
        f"{action} V2-C5 expanded development artifacts: "
        f"occurrences={counts['output_occurrences']}, "
        f"historical_unsupported="
        f"{counts['historical_unsupported_occurrences']}, "
        f"unique_discovery="
        f"{counts['unique_unsupported_discovery_texts']}; "
        "no holdout, training, evaluation, or runtime change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
