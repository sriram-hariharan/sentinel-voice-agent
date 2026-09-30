"""Manage local human adjudication of frozen V2-C5 discovery clusters."""

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

CONTRACT_SHA256 = (
    "acc48bace76a5ae93ff9a1818856e3b2dc0ad3585c80e8ba7a3f582208a71427"
)
REPORT_SHA256 = (
    "ca007ab1ade5e07e377022a5add49c641b209ae2812c3d4217734c0aeaa7c3c1"
)
ASSIGNMENTS_SHA256 = (
    "aa91a38a3dde57747a8694c404ef00105dcdb0e4233c995749edc1d129596ac0"
)
STEP18_MANIFEST_SHA256 = (
    "a0ea3f8353c12b341fbb4142b67dff507c227d94a55c73993f9fca9645162fb9"
)
CORPUS_SHA256 = (
    "0a873fe5797fdd93689e7d5054b029cd7b9068017b0ecff54af000641e282e01"
)
STEP18_SCRIPT_SHA256 = (
    "85700b8f54c3bf5c0e00780af8fcd88d9093d6c0a4b2b993a86b3d5acfac5ce3"
)

EXPECTED_CLUSTER_COUNT = 35
EXPECTED_ASSIGNMENT_COUNT = 6372
PRIMARY_CONFIG_ID = "mcs30_ms10"

WORKFILE_SCHEMA_VERSION = "v2c5-taxonomy-review-workfile.v1"
ADJUDICATION_SCHEMA_VERSION = "v2c5-taxonomy-adjudication.v1"
ADJUDICATION_MANIFEST_SCHEMA_VERSION = (
    "v2c5-taxonomy-adjudication-manifest.v1"
)

DECISIONS = (
    "MAP_TO_EXISTING_INTENT",
    "CANDIDATE_NEW_INTENT",
    "REMAIN_UNSUPPORTED",
    "NEEDS_SPLIT_REVIEW",
    "MIXED_OR_INCOHERENT",
    "INSUFFICIENT_EVIDENCE",
)
CONFIDENCE_VALUES = ("HIGH", "MEDIUM", "LOW")
EXISTING_INTENTS = (
    "informational_policy",
    "account_balance",
    "recent_transactions",
    "transaction_details",
    "card_status",
    "freeze_card",
    "create_dispute",
    "escalation",
)
PROTECTED_WRITE_INTENTS = frozenset({"freeze_card", "create_dispute"})

FORBIDDEN_INPUT_FILENAMES = frozenset(
    {
        "v2c4_safety_holdout.json",
        "v2c4_safety_holdout_seed.json",
    }
)

CLUSTER_METADATA_FIELDS = (
    "canonical_cluster_id",
    "member_count",
    "representative_discovery_ids",
    "boundary_discovery_ids",
    "source_dataset_concentration",
    "native_external_label_concentration",
)
REVIEW_FIELDS = (
    "review_status",
    "human_theme",
    "decision",
    "target_existing_intent",
    "candidate_intent_name",
    "confidence",
    "runtime_change_required",
    "human_rationale",
)
WORKFILE_CLUSTER_FIELDS = frozenset((*CLUSTER_METADATA_FIELDS, *REVIEW_FIELDS))


@dataclass(frozen=True)
class ReviewPaths:
    contract: Path
    report: Path
    assignments: Path
    step18_manifest: Path
    corpus: Path
    step18_script: Path
    workfile: Path
    adjudication_output: Path
    adjudication_manifest_output: Path


DEFAULT_PATHS = ReviewPaths(
    contract=ML_ROOT / "v2c5_intent_discovery_contract.json",
    report=ML_ROOT / "v2c5_intent_discovery_report.json",
    assignments=ML_ROOT / "v2c5_intent_discovery_assignments.json",
    step18_manifest=ML_ROOT / "v2c5_intent_discovery.manifest.json",
    corpus=ML_ROOT / "v2c5_discovery_corpus.json",
    step18_script=REPOSITORY_ROOT / "scripts/run_v2c5_intent_discovery.py",
    workfile=LOCAL_ROOT / "v2c5_taxonomy_review_workfile.json",
    adjudication_output=ML_ROOT / "v2c5_taxonomy_adjudication.json",
    adjudication_manifest_output=(
        ML_ROOT / "v2c5_taxonomy_adjudication.manifest.json"
    ),
)


@dataclass(frozen=True)
class FrozenStep18:
    contract: dict[str, Any]
    report: dict[str, Any]
    assignments: dict[str, Any]
    step18_manifest: dict[str, Any]
    corpus: dict[str, Any]
    clusters: list[dict[str, Any]]
    source_artifacts: dict[str, dict[str, Any]]


def stable_json_bytes(payload: Any) -> bytes:
    """Serialize deterministic, human-readable JSON with a trailing newline."""
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


def guard_input_path(path: Path) -> None:
    if path.name in FORBIDDEN_INPUT_FILENAMES:
        raise ValueError(f"sealed V2-C4 holdout input is forbidden: {path.name}")


def read_input_bytes(path: Path) -> bytes:
    guard_input_path(path)
    return path.read_bytes()


def load_json_object_bytes(value: bytes, label: str) -> dict[str, Any]:
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


def source_artifact_metadata(paths: ReviewPaths) -> dict[str, dict[str, Any]]:
    return {
        "contract": {
            "path": display_path(paths.contract),
            "sha256": CONTRACT_SHA256,
        },
        "discovery_assignments": {
            "path": display_path(paths.assignments),
            "sha256": ASSIGNMENTS_SHA256,
        },
        "discovery_corpus": {
            "path": display_path(paths.corpus),
            "sha256": CORPUS_SHA256,
        },
        "discovery_report": {
            "path": display_path(paths.report),
            "sha256": REPORT_SHA256,
        },
        "step18_manifest": {
            "path": display_path(paths.step18_manifest),
            "sha256": STEP18_MANIFEST_SHA256,
        },
        "step18_script": {
            "path": display_path(paths.step18_script),
            "sha256": STEP18_SCRIPT_SHA256,
        },
    }


def require_hash(value: bytes, expected: str, label: str) -> None:
    if sha256_bytes(value) != expected:
        raise ValueError(f"frozen {label} SHA-256 changed")


def validate_concentration(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    required = {"counts", "dominant", "dominant_proportion", "proportions"}
    if not required.issubset(value):
        raise ValueError(f"{label} is missing concentration fields")
    if not isinstance(value["counts"], dict) or not isinstance(
        value["proportions"],
        dict,
    ):
        raise TypeError(f"{label} counts and proportions must be objects")


def extract_primary_clusters(
    report: Mapping[str, Any],
    assignments: Mapping[str, Any],
    corpus: Mapping[str, Any],
    *,
    expected_cluster_count: int = EXPECTED_CLUSTER_COUNT,
    expected_assignment_count: int = EXPECTED_ASSIGNMENT_COUNT,
) -> list[dict[str, Any]]:
    primary = report.get("primary")
    if not isinstance(primary, dict):
        raise TypeError("Step 18 report primary result must be an object")
    if primary.get("config_id") != PRIMARY_CONFIG_ID:
        raise ValueError("Step 19 accepts only primary mcs30_ms10 clusters")
    if primary.get("primary_result_replaceable_post_hoc") is not False:
        raise ValueError("Step 18 primary result became replaceable post hoc")
    if primary.get("cluster_count_excluding_noise") != expected_cluster_count:
        raise ValueError("Step 18 primary cluster count changed")
    diagnostics_value = primary.get("cluster_diagnostics")
    if not isinstance(diagnostics_value, list):
        raise TypeError("Step 18 cluster diagnostics must be a list")
    if len(diagnostics_value) != expected_cluster_count:
        raise ValueError("Step 18 cluster-diagnostic count changed")

    clusters: list[dict[str, Any]] = []
    cluster_ids: set[str] = set()
    for value in diagnostics_value:
        if not isinstance(value, dict):
            raise TypeError("Step 18 cluster diagnostic must be an object")
        cluster = dict(value)
        cluster_id = cluster.get("canonical_cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id.startswith("cluster-"):
            raise ValueError("invalid canonical primary cluster ID")
        if cluster_id in cluster_ids:
            raise ValueError("duplicate canonical primary cluster ID")
        cluster_ids.add(cluster_id)
        member_count = cluster.get("member_count")
        if not isinstance(member_count, int) or member_count < 1:
            raise ValueError("primary cluster member count must be positive")
        for field in (
            "representative_discovery_ids",
            "boundary_discovery_ids",
        ):
            identifiers = cluster.get(field)
            if not isinstance(identifiers, list) or not all(
                isinstance(identifier, str) for identifier in identifiers
            ):
                raise TypeError(f"{field} must be a list of discovery IDs")
            if len(identifiers) > 5 or len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{field} must contain up to five unique IDs")
        validate_concentration(
            cluster.get("source_dataset_concentration"),
            "source-dataset concentration",
        )
        validate_concentration(
            cluster.get("native_external_label_concentration"),
            "native-label concentration",
        )
        clusters.append(cluster)

    if assignments.get("primary_config_id") != PRIMARY_CONFIG_ID:
        raise ValueError("assignment artifact is not the primary configuration")
    if assignments.get("assignment_count") != expected_assignment_count:
        raise ValueError("Step 18 assignment count changed")
    assignment_values = assignments.get("assignments")
    if not isinstance(assignment_values, list) or len(assignment_values) != (
        expected_assignment_count
    ):
        raise ValueError("Step 18 assignment rows changed")
    assignment_counts: Counter[str] = Counter()
    assignment_cluster_by_id: dict[str, str] = {}
    assignment_ids: list[str] = []
    for value in assignment_values:
        if not isinstance(value, dict):
            raise TypeError("Step 18 assignment must be an object")
        cluster_id = value.get("primary_canonical_cluster_id")
        is_noise = value.get("primary_is_noise")
        if is_noise is True:
            if cluster_id != "noise":
                raise ValueError("noise assignment has a non-noise cluster ID")
        elif is_noise is False:
            if cluster_id not in cluster_ids:
                raise ValueError("assignment references a non-primary cluster")
            assignment_counts[str(cluster_id)] += 1
        else:
            raise TypeError("primary noise flag must be boolean")
        discovery_id = value.get("discovery_id")
        if not isinstance(discovery_id, str):
            raise TypeError("assignment discovery ID must be a string")
        if discovery_id in assignment_cluster_by_id:
            raise ValueError("duplicate assignment discovery ID")
        assignment_cluster_by_id[discovery_id] = str(cluster_id)
        assignment_ids.append(discovery_id)

    expected_counts = {
        str(cluster["canonical_cluster_id"]): int(cluster["member_count"])
        for cluster in clusters
    }
    if dict(assignment_counts) != expected_counts:
        raise ValueError("primary assignment counts differ from diagnostics")

    corpus_values = corpus.get("primary_cluster_population")
    if not isinstance(corpus_values, list) or len(corpus_values) != (
        expected_assignment_count
    ):
        raise ValueError("frozen discovery corpus population changed")
    corpus_ids = [
        value.get("discovery_id") if isinstance(value, dict) else None
        for value in corpus_values
    ]
    if any(not isinstance(identifier, str) for identifier in corpus_ids):
        raise TypeError("corpus discovery IDs must be strings")
    if len(set(corpus_ids)) != expected_assignment_count:
        raise ValueError("duplicate corpus discovery ID")
    if assignment_ids != corpus_ids:
        raise ValueError("assignment and corpus discovery-ID ordering differs")
    corpus_id_set = set(corpus_ids)
    for cluster in clusters:
        review_ids = [
            *cluster["representative_discovery_ids"],
            *cluster["boundary_discovery_ids"],
        ]
        if any(identifier not in corpus_id_set for identifier in review_ids):
            raise ValueError("cluster review ID is absent from frozen corpus")
        if any(
            assignment_cluster_by_id[identifier]
            != cluster["canonical_cluster_id"]
            for identifier in review_ids
        ):
            raise ValueError("cluster review ID is not assigned to that cluster")
    return clusters


def validate_step18_governance(report: Mapping[str, Any]) -> None:
    expected = {
        "analysis_role": "unsupervised_taxonomy_discovery_evidence",
        "final_acceptance_evidence": False,
        "human_adjudication_required": True,
        "new_intents_created": False,
        "runtime_behavior_changed": False,
        "supervised_training_performed": False,
        "taxonomy_changed": False,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"Step 18 governance changed: {field}")


def validate_step18_manifest(
    manifest: Mapping[str, Any],
    paths: ReviewPaths,
) -> None:
    if manifest.get("schema_version") != (
        "v2c5-intent-discovery-manifest.v1"
    ):
        raise ValueError("unexpected Step 18 manifest schema")
    if manifest.get("report") != {
        "path": display_path(paths.report),
        "sha256": REPORT_SHA256,
    }:
        raise ValueError("Step 18 manifest report binding changed")
    if manifest.get("assignments") != {
        "path": display_path(paths.assignments),
        "sha256": ASSIGNMENTS_SHA256,
    }:
        raise ValueError("Step 18 manifest assignment binding changed")

    frozen_inputs = manifest.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict):
        raise TypeError("Step 18 manifest frozen inputs must be an object")
    expected_inputs = {
        "contract": {
            "path": display_path(paths.contract),
            "sha256": CONTRACT_SHA256,
        },
        "corpus": {
            "path": display_path(paths.corpus),
            "sha256": CORPUS_SHA256,
        },
    }
    for label, expected in expected_inputs.items():
        if frozen_inputs.get(label) != expected:
            raise ValueError(f"Step 18 manifest {label} binding changed")

    if manifest.get("script") != {
        "path": display_path(paths.step18_script),
        "sha256": STEP18_SCRIPT_SHA256,
    }:
        raise ValueError("Step 18 manifest script binding changed")
    hdbscan = manifest.get("hdbscan")
    if not isinstance(hdbscan, dict) or hdbscan.get("primary_config_id") != (
        PRIMARY_CONFIG_ID
    ):
        raise ValueError("Step 18 manifest primary configuration changed")
    execution_status = manifest.get("execution_status")
    if not isinstance(execution_status, dict):
        raise TypeError("Step 18 manifest execution status must be an object")
    expected_status = {
        "clustering_performed": True,
        "embeddings_generated_or_exact_cache_verified": True,
        "supervised_training_performed": False,
        "taxonomy_changed": False,
    }
    for field, expected in expected_status.items():
        if execution_status.get(field) != expected:
            raise ValueError(f"Step 18 execution status changed: {field}")


def load_frozen_step18(paths: ReviewPaths = DEFAULT_PATHS) -> FrozenStep18:
    raw_values = {
        "contract": read_input_bytes(paths.contract),
        "report": read_input_bytes(paths.report),
        "assignments": read_input_bytes(paths.assignments),
        "step18_manifest": read_input_bytes(paths.step18_manifest),
        "corpus": read_input_bytes(paths.corpus),
        "step18_script": read_input_bytes(paths.step18_script),
    }
    expected_hashes = {
        "contract": CONTRACT_SHA256,
        "report": REPORT_SHA256,
        "assignments": ASSIGNMENTS_SHA256,
        "step18_manifest": STEP18_MANIFEST_SHA256,
        "corpus": CORPUS_SHA256,
        "step18_script": STEP18_SCRIPT_SHA256,
    }
    for label, expected in expected_hashes.items():
        require_hash(raw_values[label], expected, label.replace("_", " "))

    contract = load_json_object_bytes(raw_values["contract"], "contract")
    report = load_json_object_bytes(raw_values["report"], "Step 18 report")
    assignments = load_json_object_bytes(
        raw_values["assignments"],
        "Step 18 assignments",
    )
    step18_manifest = load_json_object_bytes(
        raw_values["step18_manifest"],
        "Step 18 manifest",
    )
    corpus = load_json_object_bytes(raw_values["corpus"], "discovery corpus")

    if contract.get("schema_version") != (
        "v2c5-intent-discovery-contract.v1"
    ):
        raise ValueError("unexpected V2-C5 discovery-contract schema")
    if contract.get("status") != "frozen":
        raise ValueError("V2-C5 discovery contract is not frozen")
    if report.get("schema_version") != "v2c5-intent-discovery-report.v1":
        raise ValueError("unexpected Step 18 report schema")
    if assignments.get("schema_version") != (
        "v2c5-intent-discovery-assignments.v1"
    ):
        raise ValueError("unexpected Step 18 assignments schema")
    if corpus.get("schema_version") != "v2c5-discovery-corpus.v1":
        raise ValueError("unexpected discovery-corpus schema")

    validate_step18_governance(report)
    validate_step18_manifest(step18_manifest, paths)
    clusters = extract_primary_clusters(report, assignments, corpus)
    return FrozenStep18(
        contract=contract,
        report=report,
        assignments=assignments,
        step18_manifest=step18_manifest,
        corpus=corpus,
        clusters=clusters,
        source_artifacts=source_artifact_metadata(paths),
    )


def cluster_sort_key(cluster: Mapping[str, Any]) -> tuple[int, str]:
    return (-int(cluster["member_count"]), str(cluster["canonical_cluster_id"]))


def initial_review_fields() -> dict[str, Any]:
    return {
        "candidate_intent_name": None,
        "confidence": None,
        "decision": None,
        "human_rationale": "",
        "human_theme": "",
        "review_status": "UNREVIEWED",
        "runtime_change_required": None,
        "target_existing_intent": None,
    }


def expected_cluster_metadata(
    frozen: FrozenStep18,
) -> list[dict[str, Any]]:
    values = [
        {
            field: copy.deepcopy(cluster[field])
            for field in CLUSTER_METADATA_FIELDS
        }
        for cluster in frozen.clusters
    ]
    return sorted(values, key=cluster_sort_key)


def workfile_governance() -> dict[str, Any]:
    return {
        "classifier_training_performed": False,
        "clusters_are_intents": False,
        "external_labels_are_metadata_only": True,
        "final_taxonomy_frozen": False,
        "human_decisions_prepopulated": False,
        "new_intents_created": False,
        "protected_action_intents_inferred_automatically": False,
        "runtime_behavior_changed": False,
        "taxonomy_changed": False,
    }


def build_workfile_payload(frozen: FrozenStep18) -> dict[str, Any]:
    clusters = [
        {**metadata, **initial_review_fields()}
        for metadata in expected_cluster_metadata(frozen)
    ]
    return {
        "cluster_count": len(clusters),
        "clusters": clusters,
        "data_role": "local_human_taxonomy_review_workfile",
        "governance": workfile_governance(),
        "ordering": "descending_member_count_then_canonical_cluster_id",
        "phase": "V2-C5 Step 19",
        "primary_config_id": PRIMARY_CONFIG_ID,
        "schema_version": WORKFILE_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(frozen.source_artifacts),
    }


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required for a reviewed cluster")
    return value.strip()


def validate_review_record(
    record: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> None:
    if set(record) != WORKFILE_CLUSTER_FIELDS:
        raise ValueError("workfile cluster fields changed")
    for field in CLUSTER_METADATA_FIELDS:
        if record.get(field) != expected_metadata.get(field):
            raise ValueError(f"frozen cluster metadata changed: {field}")

    status = record.get("review_status")
    if status == "UNREVIEWED":
        if any(
            (
                record.get("decision") is not None,
                record.get("target_existing_intent") is not None,
                record.get("candidate_intent_name") is not None,
                record.get("confidence") is not None,
                record.get("runtime_change_required") is not None,
                record.get("human_theme") != "",
                record.get("human_rationale") != "",
            )
        ):
            raise ValueError("UNREVIEWED cluster contains human decisions")
        return
    if status != "REVIEWED":
        raise ValueError("review status must be UNREVIEWED or REVIEWED")

    decision = record.get("decision")
    if decision not in DECISIONS:
        raise ValueError("reviewed cluster requires an allowed decision")
    require_nonempty_string(record.get("human_theme"), "human_theme")
    require_nonempty_string(record.get("human_rationale"), "human_rationale")
    if record.get("confidence") not in CONFIDENCE_VALUES:
        raise ValueError("reviewed cluster requires HIGH, MEDIUM, or LOW confidence")
    runtime_change_required = record.get("runtime_change_required")
    if runtime_change_required is not None and not isinstance(
        runtime_change_required,
        bool,
    ):
        raise TypeError(
            "reviewed cluster runtime change must be true, false, or null"
        )

    target = record.get("target_existing_intent")
    candidate = record.get("candidate_intent_name")
    if decision == "MAP_TO_EXISTING_INTENT":
        if target not in EXISTING_INTENTS:
            raise ValueError("mapping requires an allowed existing intent")
        if candidate is not None:
            raise ValueError("mapping cannot include a candidate intent name")
    elif decision == "CANDIDATE_NEW_INTENT":
        candidate_name = require_nonempty_string(
            candidate,
            "candidate_intent_name",
        )
        if candidate_name in EXISTING_INTENTS:
            raise ValueError("candidate intent name is already an existing intent")
        if target is not None:
            raise ValueError("new-intent candidate cannot target an existing intent")
    elif target is not None or candidate is not None:
        raise ValueError(
            "this decision requires empty existing and candidate intent fields"
        )


def validate_workfile_payload(
    payload: Mapping[str, Any],
    frozen: FrozenStep18,
) -> None:
    required_fields = {
        "cluster_count",
        "clusters",
        "data_role",
        "governance",
        "ordering",
        "phase",
        "primary_config_id",
        "schema_version",
        "source_artifacts",
    }
    if set(payload) != required_fields:
        raise ValueError("workfile top-level fields changed")
    if payload.get("schema_version") != WORKFILE_SCHEMA_VERSION:
        raise ValueError("unexpected taxonomy-review workfile schema")
    if payload.get("phase") != "V2-C5 Step 19":
        raise ValueError("unexpected taxonomy-review phase")
    if payload.get("data_role") != "local_human_taxonomy_review_workfile":
        raise ValueError("unexpected taxonomy-review data role")
    if payload.get("primary_config_id") != PRIMARY_CONFIG_ID:
        raise ValueError("workfile is not bound to primary mcs30_ms10")
    if payload.get("cluster_count") != EXPECTED_CLUSTER_COUNT:
        raise ValueError("workfile must contain exactly 35 clusters")
    if payload.get("ordering") != (
        "descending_member_count_then_canonical_cluster_id"
    ):
        raise ValueError("workfile ordering policy changed")
    if payload.get("source_artifacts") != frozen.source_artifacts:
        raise ValueError("workfile frozen-source bindings changed")
    if payload.get("governance") != workfile_governance():
        raise ValueError("workfile governance changed")

    records = payload.get("clusters")
    if not isinstance(records, list) or len(records) != EXPECTED_CLUSTER_COUNT:
        raise ValueError("workfile must contain exactly 35 cluster records")
    expected = expected_cluster_metadata(frozen)
    actual_ids: list[str] = []
    for record, metadata in zip(records, expected, strict=True):
        if not isinstance(record, dict):
            raise TypeError("workfile cluster record must be an object")
        validate_review_record(record, metadata)
        actual_ids.append(str(record["canonical_cluster_id"]))
    expected_ids = [str(value["canonical_cluster_id"]) for value in expected]
    if actual_ids != expected_ids:
        raise ValueError("workfile cluster set or deterministic order changed")


def load_workfile(
    frozen: FrozenStep18,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    payload = load_json_object_bytes(
        read_input_bytes(paths.workfile),
        "taxonomy-review workfile",
    )
    validate_workfile_payload(payload, frozen)
    return payload


def progress_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = payload["clusters"]
    statuses = Counter(record["review_status"] for record in records)
    decisions = Counter(
        record["decision"]
        for record in records
        if record["review_status"] == "REVIEWED"
    )
    confidences = Counter(
        record["confidence"]
        for record in records
        if record["review_status"] == "REVIEWED"
    )
    return {
        "confidence_counts": {
            value: confidences[value] for value in CONFIDENCE_VALUES
        },
        "decision_counts": {value: decisions[value] for value in DECISIONS},
        "primary_config_id": PRIMARY_CONFIG_ID,
        "reviewed": statuses["REVIEWED"],
        "total": len(records),
        "unreviewed": statuses["UNREVIEWED"],
    }


def build_workfile(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    frozen = load_frozen_step18(paths)
    if paths.workfile.exists():
        raise FileExistsError(
            "taxonomy-review workfile already exists; build will not overwrite it"
        )
    payload = build_workfile_payload(frozen)
    validate_workfile_payload(payload, frozen)
    atomic_write_bytes(paths.workfile, stable_json_bytes(payload))
    return payload


def check_workfile(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    frozen = load_frozen_step18(paths)
    payload = load_workfile(frozen, paths)
    return progress_summary(payload)


def corpus_by_discovery_id(frozen: FrozenStep18) -> dict[str, dict[str, Any]]:
    values = frozen.corpus["primary_cluster_population"]
    return {str(value["discovery_id"]): value for value in values}


def review_example(
    discovery_id: str,
    corpus_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    record = corpus_by_id[discovery_id]
    occurrences = record["occurrences"]
    return {
        "discovery_id": discovery_id,
        "native_external_labels": sorted(
            {
                str(value["native_external_label"])
                for value in occurrences
                if value.get("native_external_label") is not None
            }
        ),
        "source_datasets": sorted(
            {str(value["source_dataset"]) for value in occurrences}
        ),
        "text": record["text"],
    }


def display_cluster_payload(
    cluster: Mapping[str, Any],
    frozen: FrozenStep18,
) -> dict[str, Any]:
    corpus_by_id = corpus_by_discovery_id(frozen)
    representatives = [
        review_example(identifier, corpus_by_id)
        for identifier in cluster["representative_discovery_ids"]
    ]
    boundaries = [
        review_example(identifier, corpus_by_id)
        for identifier in cluster["boundary_discovery_ids"]
    ]
    return {
        "boundary_examples": boundaries,
        "cluster": copy.deepcopy(dict(cluster)),
        "metadata_notice": (
            "Source datasets and native external labels are post-hoc metadata "
            "only; they do not define a SentinelVoice intent. Protected-write "
            "mapping requires explicit current-action semantics."
        ),
        "representative_examples": representatives,
        "section_order": [
            "cluster",
            "representative_examples",
            "boundary_examples",
        ],
    }


def find_cluster(
    payload: Mapping[str, Any],
    cluster_id: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in payload["clusters"]
        if record["canonical_cluster_id"] == cluster_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown primary canonical cluster ID: {cluster_id}")
    return matches[0]


def show_cluster(
    cluster_id: str,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    frozen = load_frozen_step18(paths)
    workfile = load_workfile(frozen, paths)
    return display_cluster_payload(find_cluster(workfile, cluster_id), frozen)


def next_cluster(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    frozen = load_frozen_step18(paths)
    workfile = load_workfile(frozen, paths)
    cluster = next(
        (
            record
            for record in workfile["clusters"]
            if record["review_status"] == "UNREVIEWED"
        ),
        None,
    )
    if cluster is None:
        return {
            "message": "All primary clusters have human review decisions.",
            "review_complete": True,
            "summary": progress_summary(workfile),
        }
    return display_cluster_payload(cluster, frozen)


def apply_review_update(
    payload: Mapping[str, Any],
    frozen: FrozenStep18,
    *,
    cluster_id: str,
    decision: str,
    human_theme: str,
    confidence: str,
    runtime_change_required: bool | None,
    human_rationale: str,
    target_existing_intent: str | None,
    candidate_intent_name: str | None,
) -> dict[str, Any]:
    validate_workfile_payload(payload, frozen)
    updated = copy.deepcopy(dict(payload))
    record = find_cluster(updated, cluster_id)
    record.update(
        {
            "candidate_intent_name": (
                candidate_intent_name.strip()
                if isinstance(candidate_intent_name, str)
                and candidate_intent_name.strip()
                else None
            ),
            "confidence": confidence,
            "decision": decision,
            "human_rationale": human_rationale.strip(),
            "human_theme": human_theme.strip(),
            "review_status": "REVIEWED",
            "runtime_change_required": runtime_change_required,
            "target_existing_intent": target_existing_intent,
        }
    )
    validate_workfile_payload(updated, frozen)
    return updated


def set_cluster_review(
    *,
    cluster_id: str,
    decision: str,
    human_theme: str,
    confidence: str,
    runtime_change_required: bool | None,
    human_rationale: str,
    target_existing_intent: str | None,
    candidate_intent_name: str | None,
    paths: ReviewPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    frozen = load_frozen_step18(paths)
    payload = load_workfile(frozen, paths)
    updated = apply_review_update(
        payload,
        frozen,
        cluster_id=cluster_id,
        decision=decision,
        human_theme=human_theme,
        confidence=confidence,
        runtime_change_required=runtime_change_required,
        human_rationale=human_rationale,
        target_existing_intent=target_existing_intent,
        candidate_intent_name=candidate_intent_name,
    )
    atomic_write_bytes(paths.workfile, stable_json_bytes(updated))
    return updated


def export_governance() -> dict[str, Any]:
    return {
        "classifier_training_performed": False,
        "external_labels_are_metadata_only": True,
        "final_taxonomy_frozen": False,
        "human_adjudication_completed": True,
        "new_intents_created": False,
        "protected_action_intents_inferred_automatically": False,
        "runtime_behavior_changed": False,
        "step20_required": True,
        "taxonomy_changed": False,
    }


def validate_text_free_artifact(payload: Any) -> None:
    prohibited_keys = {"text", "normalized_text", "utterance", "examples"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in prohibited_keys:
                    raise ValueError("final adjudication cannot contain raw text")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def build_adjudication_payload(
    workfile: Mapping[str, Any],
    frozen: FrozenStep18,
) -> dict[str, Any]:
    validate_workfile_payload(workfile, frozen)
    summary = progress_summary(workfile)
    if summary["unreviewed"] != 0:
        raise ValueError("export requires all 35 clusters to be reviewed")
    payload = {
        **export_governance(),
        "adjudications": copy.deepcopy(workfile["clusters"]),
        "analysis_role": "human_taxonomy_adjudication_evidence_only",
        "cluster_count": EXPECTED_CLUSTER_COUNT,
        "decision_counts": summary["decision_counts"],
        "phase": "V2-C5 Step 19",
        "primary_config_id": PRIMARY_CONFIG_ID,
        "reviewed_count": summary["reviewed"],
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(frozen.source_artifacts),
    }
    validate_text_free_artifact(payload)
    return payload


def build_adjudication_manifest(
    adjudication_bytes: bytes,
    adjudication: Mapping[str, Any],
    paths: ReviewPaths,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    payload = {
        **export_governance(),
        "adjudication": {
            "path": display_path(paths.adjudication_output),
            "sha256": sha256_bytes(adjudication_bytes),
        },
        "cluster_count": adjudication["cluster_count"],
        "decision_counts": copy.deepcopy(adjudication["decision_counts"]),
        "phase": "V2-C5 Step 19",
        "review_script": {
            "path": display_path(script_path),
            "sha256": sha256_bytes(script_path.read_bytes()),
        },
        "reviewed_count": adjudication["reviewed_count"],
        "schema_version": ADJUDICATION_MANIFEST_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(adjudication["source_artifacts"]),
    }
    validate_text_free_artifact(payload)
    return payload


def export_adjudication(paths: ReviewPaths = DEFAULT_PATHS) -> dict[str, Any]:
    frozen = load_frozen_step18(paths)
    workfile = load_workfile(frozen, paths)
    adjudication = build_adjudication_payload(workfile, frozen)
    adjudication_bytes = stable_json_bytes(adjudication)
    manifest = build_adjudication_manifest(
        adjudication_bytes,
        adjudication,
        paths,
    )
    atomic_write_bytes(paths.adjudication_output, adjudication_bytes)
    atomic_write_bytes(
        paths.adjudication_manifest_output,
        stable_json_bytes(manifest),
    )
    return {
        "adjudication_path": display_path(paths.adjudication_output),
        "adjudication_sha256": sha256_bytes(adjudication_bytes),
        "manifest_path": display_path(paths.adjudication_manifest_output),
        "summary": progress_summary(workfile),
    }


def parse_nullable_boolean(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    raise argparse.ArgumentTypeError("value must be true, false, or null")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("summary")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--cluster-id", required=True)
    subparsers.add_parser("next")

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("--cluster-id", required=True)
    set_parser.add_argument("--decision", choices=DECISIONS, required=True)
    set_parser.add_argument("--human-theme", required=True)
    set_parser.add_argument("--confidence", choices=CONFIDENCE_VALUES, required=True)
    set_parser.add_argument(
        "--runtime-change-required",
        type=parse_nullable_boolean,
        required=True,
    )
    set_parser.add_argument("--human-rationale", required=True)
    set_parser.add_argument(
        "--target-existing-intent",
        choices=EXISTING_INTENTS,
    )
    set_parser.add_argument("--candidate-intent-name")

    subparsers.add_parser("export")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            payload = build_workfile()
            result: Any = {
                "path": display_path(DEFAULT_PATHS.workfile),
                "summary": progress_summary(payload),
            }
        elif args.command == "check":
            result = {"status": "valid", "summary": check_workfile()}
        elif args.command == "summary":
            result = check_workfile()
        elif args.command == "show":
            result = show_cluster(args.cluster_id)
        elif args.command == "next":
            result = next_cluster()
        elif args.command == "set":
            updated = set_cluster_review(
                cluster_id=args.cluster_id,
                decision=args.decision,
                human_theme=args.human_theme,
                confidence=args.confidence,
                runtime_change_required=args.runtime_change_required,
                human_rationale=args.human_rationale,
                target_existing_intent=args.target_existing_intent,
                candidate_intent_name=args.candidate_intent_name,
            )
            result = progress_summary(updated)
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
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
