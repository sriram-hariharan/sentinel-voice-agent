"""Validate and seal the independently authored V2-C5 final holdout."""

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

CONTRACT_SHA256 = "ed3b65870304bd9d4d89f80bd720021d9c46908264506a29ca3a37b876803791"
TAXONOMY_SHA256 = "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
TAXONOMY_MANIFEST_SHA256 = (
    "359eb38e5ff63af0c9cc0f5db19951b8626f9e15cb136f23046648d655a0f7a9"
)
DEVELOPMENT_SHA256 = "dce97c0a3bfcdf0dee93d784ded24d3568f7df8f513871995428ff35d030bb07"
DEVELOPMENT_MANIFEST_SHA256 = (
    "89fea89d793cf5e027b059c331533ec45c6d59fabbd49e9245e0d055ae946a16"
)
CHALLENGE_SHA256 = "4493b9baa8363408c917fe29de069dec82f925c8c3a605878a08eba06b051e5c"
CHALLENGE_MANIFEST_SHA256 = (
    "51389b7e693156433edd6e7f0c2fe5b2c3a1ab0f34a261b47de7a7cda6e48b38"
)
LOCKBOX_MANIFEST_SHA256 = (
    "651bbe2ebe61bb5b1e9139c24cacac340973172f5637278c12d50cf596334364"
)

EXPECTED_TOTAL = 640
EXPECTED_INTENT_COUNT = 16
EXPECTED_PER_INTENT = 40
EXPECTED_PROTECTED_COUNT = 160
EXPECTED_NON_PROTECTED_COUNT = 480
EXPECTED_DEVELOPMENT_COUNT = 8198
EXPECTED_CHALLENGE_COUNT = 270
EXPECTED_LOCKBOX_COUNT = 1922

NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"
SEED_SCHEMA_VERSION = "v2c5-final-holdout-seed.v1"
DATASET_SCHEMA_VERSION = "v2c5-final-holdout.v1"
MANIFEST_SCHEMA_VERSION = "v2c5-final-holdout-manifest.v1"
DATASET_VERSION = "v2c5-final-holdout.v1"
BUILDER_VERSION = "v2c5-final-holdout-builder.v1"
DATA_ROLE = "sealed_v2c5_final_holdout"

PROTECTED_WRITE_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)
REQUIRED_AUTHORSHIP = {
    "development_utterances_inspected": False,
    "external_dataset_examples_used": False,
    "kind": "independently_authored_synthetic",
    "model_outputs_used": False,
    "old_holdout_contents_inspected": False,
}
FORBIDDEN_SEED_FIELDS = frozenset(
    {
        "classifier_prediction",
        "embedding",
        "embedding_model",
        "model_prediction",
        "predicted_intent",
        "prediction",
        "risk",
        "similarity",
        "similarity_score",
        "vector",
    }
)
SEALED_V2C4_HOLDOUT = ML_ROOT / "v2c4_safety_holdout.json"


@dataclass(frozen=True)
class BuildPaths:
    contract: Path
    taxonomy: Path
    taxonomy_manifest: Path
    development: Path
    development_manifest: Path
    challenge: Path
    challenge_manifest: Path
    lockbox_manifest: Path
    seed: Path
    probe_directory: Path
    dataset_output: Path
    manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    contract=ML_ROOT / "v2c5_final_holdout_contract.json",
    taxonomy=ML_ROOT / "v2c5_taxonomy_freeze.json",
    taxonomy_manifest=ML_ROOT / "v2c5_taxonomy_freeze.manifest.json",
    development=ML_ROOT / "v2c5_expanded_development_dataset.json",
    development_manifest=ML_ROOT / "v2c5_expanded_development_dataset.manifest.json",
    challenge=ML_ROOT / "v2c3_challenge_set.json",
    challenge_manifest=ML_ROOT / "v2c3_challenge_set.manifest.json",
    lockbox_manifest=ML_ROOT / "v2c3_fresh_lockbox_manifest.json",
    seed=ML_ROOT / "local/v2c5_final_holdout_seed.json",
    probe_directory=ML_ROOT,
    dataset_output=ML_ROOT / "v2c5_final_holdout.json",
    manifest_output=ML_ROOT / "v2c5_final_holdout.manifest.json",
)


@dataclass(frozen=True)
class FrozenSources:
    contract: dict[str, Any]
    taxonomy: dict[str, Any]
    intent_labels: tuple[str, ...]
    risk_by_intent: dict[str, str]
    required_coverage_tags: frozenset[str]
    overlap_hashes: dict[str, frozenset[str]]
    source_artifacts: dict[str, dict[str, Any]]
    probe_record_count: int


@dataclass(frozen=True)
class SeedValidation:
    records: list[dict[str, Any]]
    raw_duplicate_count: int
    normalized_duplicate_count: int
    intent_counts: dict[str, int]


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


def load_json_object(value: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def require_object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{label} must be a list of objects")
    return value


def require_string_list(value: Any, label: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TypeError(f"{label} must be a list of non-empty strings")
    if nonempty and not value:
        raise ValueError(f"{label} cannot be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} cannot contain duplicates")
    return value


def source_path_items(paths: BuildPaths) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("contract", paths.contract, CONTRACT_SHA256),
        ("taxonomy", paths.taxonomy, TAXONOMY_SHA256),
        ("taxonomy_manifest", paths.taxonomy_manifest, TAXONOMY_MANIFEST_SHA256),
        ("development", paths.development, DEVELOPMENT_SHA256),
        (
            "development_manifest",
            paths.development_manifest,
            DEVELOPMENT_MANIFEST_SHA256,
        ),
        ("challenge", paths.challenge, CHALLENGE_SHA256),
        ("challenge_manifest", paths.challenge_manifest, CHALLENGE_MANIFEST_SHA256),
        ("lockbox_manifest", paths.lockbox_manifest, LOCKBOX_MANIFEST_SHA256),
    )


def guard_prohibited_source(path: Path) -> None:
    resolved = path.resolve()
    if resolved == SEALED_V2C4_HOLDOUT.resolve() or path.name == SEALED_V2C4_HOLDOUT.name:
        raise ValueError("the sealed V2-C4 final holdout is a prohibited source")
    if "v2c4" in path.as_posix().lower():
        raise ValueError("V2-C4 artifacts are prohibited Step 21C2B sources")


def read_pinned_json(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    guard_prohibited_source(path)
    value = path.read_bytes()
    if sha256_bytes(value) != expected_hash:
        raise ValueError(f"frozen {label.replace('_', ' ')} SHA-256 changed")
    return load_json_object(value, label.replace("_", " "))


def validate_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-final-holdout-contract.v1":
        raise ValueError("unexpected final holdout contract schema")
    if payload.get("phase") != "V2-C5 Step 21C1":
        raise ValueError("unexpected final holdout contract phase")
    design = payload.get("holdout_design")
    if not isinstance(design, dict):
        raise TypeError("holdout design must be an object")
    expected_design = {
        "examples_per_intent": EXPECTED_PER_INTENT,
        "intent_count": EXPECTED_INTENT_COUNT,
        "non_protected_example_count": EXPECTED_NON_PROTECTED_COUNT,
        "protected_write_example_count": EXPECTED_PROTECTED_COUNT,
        "total_example_count": EXPECTED_TOTAL,
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            raise ValueError(f"holdout contract {key} changed")
    overlap = payload.get("overlap_policy")
    if not isinstance(overlap, dict):
        raise TypeError("overlap policy must be an object")
    if overlap.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("holdout normalization contract changed")
    if overlap.get("sealed_v2c4_holdout_accessed") is not False:
        raise ValueError("sealed V2-C4 holdout access must remain false")
    if overlap.get("v2c4_holdout_hash_or_contents_required_for_overlap_check") is not False:
        raise ValueError("sealed V2-C4 holdout cannot become overlap evidence")
    status = payload.get("contract_status")
    if not isinstance(status, dict) or status.get("final_holdout_contract_frozen") is not True:
        raise ValueError("final holdout contract is not frozen")


def validate_taxonomy(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, str]]:
    if payload.get("schema_version") != "v2c5-taxonomy-freeze.v1":
        raise ValueError("unexpected taxonomy schema")
    labels = tuple(
        require_string_list(
            payload.get("intent_label_order"), "intent labels", nonempty=True
        )
    )
    if len(labels) != EXPECTED_INTENT_COUNT or labels != tuple(sorted(labels)):
        raise ValueError("taxonomy must contain 16 lexicographically ordered intents")
    if tuple(payload.get("protected_write_intents", [])) != PROTECTED_WRITE_INTENTS:
        raise ValueError("protected-write intent set changed")
    risk_payload = payload.get("risk_by_intent")
    if not isinstance(risk_payload, dict) or set(risk_payload) != set(labels):
        raise ValueError("taxonomy risk mapping is incomplete")
    risks = {str(key): str(value) for key, value in risk_payload.items()}
    frozen = contract.get("taxonomy")
    if not isinstance(frozen, dict):
        raise TypeError("contract taxonomy must be an object")
    if tuple(frozen.get("intent_label_order", [])) != labels:
        raise ValueError("contract and Step 20 label spaces differ")
    if frozen.get("risk_by_intent") != risks:
        raise ValueError("contract and Step 20 risk mappings differ")
    if tuple(frozen.get("protected_write_intents", [])) != PROTECTED_WRITE_INTENTS:
        raise ValueError("contract protected-write set changed")
    return labels, risks


def validate_taxonomy_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-taxonomy-freeze-manifest.v1":
        raise ValueError("unexpected taxonomy manifest schema")
    frozen = payload.get("taxonomy_freeze")
    if not isinstance(frozen, dict) or frozen.get("sha256") != TAXONOMY_SHA256:
        raise ValueError("taxonomy manifest binding changed")
    if payload.get("final_taxonomy_frozen") is not True:
        raise ValueError("taxonomy is not frozen")


def validate_development(
    payload: Mapping[str, Any], labels: Sequence[str]
) -> frozenset[str]:
    if payload.get("schema_version") != "v2c5-expanded-development-dataset.v1":
        raise ValueError("unexpected expanded development schema")
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("development normalization changed")
    records = require_object_list(payload.get("examples"), "development examples")
    if (
        payload.get("example_count") != EXPECTED_DEVELOPMENT_COUNT
        or len(records) != EXPECTED_DEVELOPMENT_COUNT
    ):
        raise ValueError("expanded development count changed")
    allowed = set(labels)
    hashes: set[str] = set()
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"development example {index} has empty text")
        if record.get("intent") not in allowed:
            raise ValueError(f"development example {index} has an unknown intent")
        expected = normalized_text_sha256(text)
        if record.get("normalized_text_sha256") != expected:
            raise ValueError(f"development example {index} normalized hash changed")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError(f"development example {index} raw hash changed")
        hashes.add(expected)
    return frozenset(hashes)


def validate_development_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c5-expanded-development-dataset-manifest.v1":
        raise ValueError("unexpected development manifest schema")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("sha256") != DEVELOPMENT_SHA256:
        raise ValueError("development manifest binding changed")
    status = payload.get("execution_status")
    if not isinstance(status, dict) or status.get("expanded_development_dataset_built") is not True:
        raise ValueError("expanded development dataset is not frozen")


def validate_challenge(payload: Mapping[str, Any]) -> frozenset[str]:
    if payload.get("schema_version") != "v2c3-challenge-set.v1":
        raise ValueError("unexpected consumed challenge schema")
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("challenge normalization changed")
    records = require_object_list(payload.get("examples"), "challenge examples")
    if (
        payload.get("example_count") != EXPECTED_CHALLENGE_COUNT
        or len(records) != EXPECTED_CHALLENGE_COUNT
    ):
        raise ValueError("consumed challenge count changed")
    hashes: list[str] = []
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"challenge example {index} has empty text")
        normalized_hash = normalized_text_sha256(text)
        if record.get("normalized_text_sha256") != normalized_hash:
            raise ValueError(f"challenge example {index} normalized hash changed")
        hashes.append(normalized_hash)
    if len(hashes) != len(set(hashes)):
        raise ValueError("consumed challenge contains normalized duplicates")
    return frozenset(hashes)


def validate_challenge_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "v2c3-challenge-set-manifest.v1":
        raise ValueError("unexpected challenge manifest schema")
    output_hashes = payload.get("output_hashes")
    if (
        not isinstance(output_hashes, dict)
        or output_hashes.get("v2c3_challenge_set.json") != CHALLENGE_SHA256
    ):
        raise ValueError("challenge manifest binding changed")
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("version") != NORMALIZATION_VERSION:
        raise ValueError("challenge manifest normalization changed")


def validate_lockbox_manifest(payload: Mapping[str, Any]) -> frozenset[str]:
    if payload.get("schema_version") != "v2c3-fresh-lockbox-manifest.v1":
        raise ValueError("unexpected external lockbox manifest schema")
    if payload.get("contains_text") is not False:
        raise ValueError("external lockbox overlap input must remain text-free")
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("external lockbox normalization changed")
    records = require_object_list(payload.get("members"), "external lockbox members")
    if (
        payload.get("member_count") != EXPECTED_LOCKBOX_COUNT
        or len(records) != EXPECTED_LOCKBOX_COUNT
    ):
        raise ValueError("external lockbox count changed")
    hashes: list[str] = []
    for index, record in enumerate(records):
        if "text" in record:
            raise ValueError("external lockbox manifest unexpectedly contains text")
        value = record.get("normalized_text_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"external lockbox member {index} has an invalid hash")
        hashes.append(value)
    if len(hashes) != len(set(hashes)):
        raise ValueError("external lockbox normalized hashes are not unique")
    return frozenset(hashes)


def discover_probe_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    paths = tuple(
        sorted(
            path
            for path in directory.glob("v2c5*selection*probe*.json")
            if not path.name.endswith(".manifest.json")
        )
    )
    for path in paths:
        guard_prohibited_source(path)
    return paths


def load_probe_hashes(paths: Sequence[Path]) -> tuple[frozenset[str], list[dict[str, Any]], int]:
    hashes: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    count = 0
    for path in paths:
        guard_prohibited_source(path)
        value = path.read_bytes()
        payload = load_json_object(value, "V2-C5 model-selection probe")
        candidates = payload.get("examples", payload.get("records", payload.get("members")))
        records = require_object_list(candidates, f"{path.name} records")
        for index, record in enumerate(records):
            normalized_hash = record.get("normalized_text_sha256")
            if normalized_hash is None and isinstance(record.get("text"), str):
                normalized_hash = normalized_text_sha256(record["text"])
            if not isinstance(normalized_hash, str) or len(normalized_hash) != 64:
                raise ValueError(f"{path.name} record {index} lacks a normalized hash")
            hashes.add(normalized_hash)
        count += len(records)
        artifacts.append(
            {
                "path": display_path(path),
                "record_count": len(records),
                "sha256": sha256_bytes(value),
            }
        )
    return frozenset(hashes), artifacts, count


def validate_contract_source_bindings(contract: Mapping[str, Any]) -> None:
    sources = contract.get("source_artifacts")
    if not isinstance(sources, dict):
        raise TypeError("contract source artifacts must be an object")
    expected = {
        "expanded_development_dataset": DEVELOPMENT_SHA256,
        "expanded_development_manifest": DEVELOPMENT_MANIFEST_SHA256,
        "taxonomy_freeze": TAXONOMY_SHA256,
        "taxonomy_freeze_manifest": TAXONOMY_MANIFEST_SHA256,
    }
    for label, expected_hash in expected.items():
        artifact = sources.get(label)
        if not isinstance(artifact, dict) or artifact.get("sha256") != expected_hash:
            raise ValueError(f"contract {label.replace('_', ' ')} binding changed")


def load_frozen_sources(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    probe_paths: Sequence[Path] | None = None,
) -> FrozenSources:
    payloads = {
        label: read_pinned_json(path, expected_hash, label)
        for label, path, expected_hash in source_path_items(paths)
    }
    contract = payloads["contract"]
    validate_contract(contract)
    validate_contract_source_bindings(contract)
    labels, risks = validate_taxonomy(payloads["taxonomy"], contract)
    validate_taxonomy_manifest(payloads["taxonomy_manifest"])
    development_hashes = validate_development(payloads["development"], labels)
    validate_development_manifest(payloads["development_manifest"])
    challenge_hashes = validate_challenge(payloads["challenge"])
    validate_challenge_manifest(payloads["challenge_manifest"])
    lockbox_hashes = validate_lockbox_manifest(payloads["lockbox_manifest"])
    selected_probe_paths = (
        tuple(probe_paths)
        if probe_paths is not None
        else discover_probe_paths(paths.probe_directory)
    )
    probe_hashes, probe_artifacts, probe_count = load_probe_hashes(selected_probe_paths)
    coverage = contract.get("boundary_coverage")
    if not isinstance(coverage, dict):
        raise TypeError("contract boundary coverage must be an object")
    required_tags = frozenset(
        require_string_list(
            coverage.get("required_qualitative_tags"),
            "required qualitative tags",
            nonempty=True,
        )
    )
    source_artifacts: dict[str, dict[str, Any]] = {
        label: {"path": display_path(path), "sha256": expected_hash}
        for label, path, expected_hash in source_path_items(paths)
    }
    source_artifacts["preexisting_v2c5_model_selection_probes"] = {
        "artifacts": probe_artifacts,
        "record_count": probe_count,
    }
    overlap_hashes = {
        "all_historical_development_and_training_text_used_in_v2c5": development_hashes,
        "consumed_v2c3_challenge": challenge_hashes,
        "consumed_v2c3_external_lockbox": lockbox_hashes,
        "preexisting_v2c5_model_selection_probe": probe_hashes,
        "v2c5_expanded_development_dataset": development_hashes,
    }
    return FrozenSources(
        contract=contract,
        taxonomy=payloads["taxonomy"],
        intent_labels=labels,
        risk_by_intent=risks,
        required_coverage_tags=required_tags,
        overlap_hashes={key: frozenset(value) for key, value in overlap_hashes.items()},
        source_artifacts=source_artifacts,
        probe_record_count=probe_count,
    )


def validate_seed(
    payload: Mapping[str, Any],
    intent_labels: Sequence[str],
    required_coverage_tags: frozenset[str],
) -> SeedValidation:
    if set(payload) != {"authorship", "records", "schema_version"}:
        raise ValueError("seed top-level fields changed")
    if payload.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError("unexpected final holdout seed schema")
    if payload.get("authorship") != REQUIRED_AUTHORSHIP:
        raise ValueError("seed independent-authorship declaration changed")
    records = require_object_list(payload.get("records"), "seed records")
    if len(records) != EXPECTED_TOTAL:
        raise ValueError(f"seed must contain exactly {EXPECTED_TOTAL} records")
    expected_ids = [f"v2c5-final-holdout-seed:{index:04d}" for index in range(1, 641)]
    actual_ids = [record.get("seed_id") for record in records]
    if actual_ids != expected_ids:
        raise ValueError("seed IDs must be unique and sequential from 0001 through 0640")
    allowed_labels = set(intent_labels)
    raw_texts: list[str] = []
    normalized_hashes: list[str] = []
    coverage_by_intent = {intent: set() for intent in intent_labels}
    for index, record in enumerate(records, start=1):
        if set(record) != {
            "boundary_tags",
            "coverage_tags",
            "intent",
            "seed_id",
            "text",
        }:
            unexpected = set(record) - {
                "boundary_tags",
                "coverage_tags",
                "intent",
                "seed_id",
                "text",
            }
            if unexpected & FORBIDDEN_SEED_FIELDS:
                raise ValueError(f"seed record {index} contains prohibited derived metadata")
            raise ValueError(f"seed record {index} fields changed")
        intent = record.get("intent")
        if intent not in allowed_labels:
            raise ValueError(f"seed record {index} has an unknown intent")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"seed record {index} text must be non-empty")
        coverage_tags = require_string_list(
            record.get("coverage_tags"), f"seed record {index} coverage tags", nonempty=True
        )
        unknown_tags = set(coverage_tags) - required_coverage_tags
        if unknown_tags:
            raise ValueError(
                f"seed record {index} has unknown coverage tags: "
                f"{sorted(unknown_tags)}"
            )
        require_string_list(
            record.get("boundary_tags"), f"seed record {index} boundary tags", nonempty=False
        )
        coverage_by_intent[str(intent)].update(coverage_tags)
        raw_texts.append(text)
        normalized_hashes.append(normalized_text_sha256(text))
    raw_duplicate_count = len(raw_texts) - len(set(raw_texts))
    normalized_duplicate_count = len(normalized_hashes) - len(set(normalized_hashes))
    if raw_duplicate_count:
        raise ValueError("seed contains exact raw-text duplicates")
    if normalized_duplicate_count:
        raise ValueError("seed contains normalized-text duplicates")
    counts = Counter(str(record["intent"]) for record in records)
    expected_counts = {intent: EXPECTED_PER_INTENT for intent in intent_labels}
    if dict(sorted(counts.items())) != expected_counts:
        raise ValueError("seed must contain exactly 40 records for every frozen intent")
    for intent, observed_tags in coverage_by_intent.items():
        if observed_tags != required_coverage_tags:
            missing = sorted(required_coverage_tags - observed_tags)
            raise ValueError(f"seed intent {intent} lacks required coverage tags: {missing}")
    return SeedValidation(
        records=copy.deepcopy(records),
        raw_duplicate_count=raw_duplicate_count,
        normalized_duplicate_count=normalized_duplicate_count,
        intent_counts=expected_counts,
    )


def validate_no_overlaps(
    records: Sequence[Mapping[str, Any]],
    overlap_hashes: Mapping[str, frozenset[str]],
) -> dict[str, int]:
    holdout_hashes = {normalized_text_sha256(str(record["text"])) for record in records}
    counts = {
        source: len(holdout_hashes & source_hashes)
        for source, source_hashes in sorted(overlap_hashes.items())
    }
    contaminated = {source: count for source, count in counts.items() if count}
    if contaminated:
        details = ", ".join(f"{source}={count}" for source, count in contaminated.items())
        raise ValueError(f"final holdout normalized-text overlap detected: {details}")
    return counts


def list_overlap_seed_records(
    records: Sequence[Mapping[str, Any]],
    overlap_hashes: Mapping[str, frozenset[str]],
) -> list[dict[str, Any]]:
    overlapping: list[dict[str, Any]] = []
    for record in records:
        normalized_hash = normalized_text_sha256(str(record["text"]))
        sources = sorted(
            source
            for source, source_hashes in overlap_hashes.items()
            if normalized_hash in source_hashes
        )
        if sources:
            overlapping.append(
                {
                    "intent": str(record["intent"]),
                    "overlap_sources": sources,
                    "seed_id": str(record["seed_id"]),
                }
            )
    return sorted(overlapping, key=lambda record: record["seed_id"])


def load_and_validate_seed(
    paths: BuildPaths, sources: FrozenSources
) -> tuple[SeedValidation, bytes]:
    if not paths.seed.exists():
        raise FileNotFoundError(f"independently authored seed is missing: {paths.seed}")
    guard_prohibited_source(paths.seed)
    seed_bytes = paths.seed.read_bytes()
    seed_payload = load_json_object(seed_bytes, "final holdout seed")
    return (
        validate_seed(
            seed_payload,
            sources.intent_labels,
            sources.required_coverage_tags,
        ),
        seed_bytes,
    )


def build_examples(
    seed_records: Sequence[Mapping[str, Any]], risk_by_intent: Mapping[str, str]
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, source in enumerate(seed_records, start=1):
        text = str(source["text"])
        intent = str(source["intent"])
        examples.append(
            {
                "boundary_tags": list(source["boundary_tags"]),
                "coverage_tags": list(source["coverage_tags"]),
                "data_role": DATA_ROLE,
                "example_id": f"v2c5-final-holdout:{index:04d}",
                "intent": intent,
                "normalized_text_sha256": normalized_text_sha256(text),
                "risk": risk_by_intent[intent],
                "source_seed_id": source["seed_id"],
                "text": text,
                "text_sha256": text_sha256(text),
            }
        )
    return examples


def evaluation_governance() -> dict[str, Any]:
    return {
        "augmentation_source_eligible": False,
        "error_analysis_eligible_before_final_evaluation": False,
        "final_evaluation_step": "V2-C5 Step 23",
        "model_selection_eligible": False,
        "single_final_evaluation_only": True,
        "step22_may_inspect_individual_holdout_examples": False,
        "step22_may_run_holdout_inference": False,
        "step22_may_tune_from_holdout": False,
        "step22_may_use_holdout_errors": False,
        "taxonomy_discovery_eligible": False,
        "threshold_selection_eligible": False,
        "training_eligible": False,
    }


def execution_status() -> dict[str, Any]:
    return {
        "classifier_evaluation_performed": False,
        "classifier_training_performed": False,
        "final_holdout_contract_frozen": True,
        "final_holdout_evaluated": False,
        "final_holdout_examples_created": True,
        "final_holdout_frozen": True,
        "model_selection_performed": False,
        "runtime_behavior_changed": False,
        "step21_complete": True,
        "step22_permitted": True,
    }


def build_dataset_payload(
    examples: Sequence[Mapping[str, Any]], sources: FrozenSources
) -> dict[str, Any]:
    return {
        "data_role": DATA_ROLE,
        "dataset_version": DATASET_VERSION,
        "evaluation_governance": evaluation_governance(),
        "example_count": len(examples),
        "examples": [copy.deepcopy(dict(example)) for example in examples],
        "gold_label_source": "independently_authored_seed_intent",
        "normalization_version": NORMALIZATION_VERSION,
        "phase": "V2-C5 Step 21C2B",
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "taxonomy_version": sources.taxonomy["taxonomy_version"],
    }


def count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def validate_output_examples(
    examples: Sequence[Mapping[str, Any]], sources: FrozenSources
) -> None:
    if len(examples) != EXPECTED_TOTAL:
        raise ValueError("final holdout output count changed")
    expected_ids = [f"v2c5-final-holdout:{index:04d}" for index in range(1, 641)]
    if [record.get("example_id") for record in examples] != expected_ids:
        raise ValueError("final holdout IDs are not deterministic")
    if count_by(examples, "intent") != {
        intent: EXPECTED_PER_INTENT for intent in sources.intent_labels
    }:
        raise ValueError("final holdout class balance changed")
    for index, record in enumerate(examples, start=1):
        intent = str(record["intent"])
        text = str(record["text"])
        if record.get("risk") != sources.risk_by_intent[intent]:
            raise ValueError(f"final holdout example {index} risk is not taxonomy-derived")
        if record.get("data_role") != DATA_ROLE:
            raise ValueError(f"final holdout example {index} has the wrong role")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError(f"final holdout example {index} raw hash changed")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError(f"final holdout example {index} normalized hash changed")


def build_manifest_payload(
    dataset_bytes: bytes,
    examples: Sequence[Mapping[str, Any]],
    seed_bytes: bytes,
    seed_validation: SeedValidation,
    overlap_counts: Mapping[str, int],
    sources: FrozenSources,
    paths: BuildPaths,
    script_path: Path,
) -> dict[str, Any]:
    risk_counts = count_by(examples, "risk")
    protected_count = sum(
        record["intent"] in PROTECTED_WRITE_INTENTS for record in examples
    )
    return {
        "builder": {
            "path": display_path(script_path),
            "sha256": sha256_bytes(script_path.read_bytes()),
            "version": BUILDER_VERSION,
        },
        "counts": {
            "intent_counts": count_by(examples, "intent"),
            "internal_exact_duplicate_count": seed_validation.raw_duplicate_count,
            "internal_normalized_duplicate_count": seed_validation.normalized_duplicate_count,
            "non_protected_count": len(examples) - protected_count,
            "protected_write_positive_count": protected_count,
            "risk_counts": risk_counts,
            "total_count": len(examples),
        },
        "dataset": {
            "path": display_path(paths.dataset_output),
            "schema_version": DATASET_SCHEMA_VERSION,
            "sha256": sha256_bytes(dataset_bytes),
            "version": DATASET_VERSION,
        },
        "evaluation_governance": evaluation_governance(),
        "execution_status": execution_status(),
        "integrity_and_overlap_checks": {
            "embeddings_used": False,
            "model_predictions_used": False,
            "normalized_overlap_counts": dict(sorted(overlap_counts.items())),
            "prohibited_source_counts": {
                "cfpb_records": 0,
                "sealed_v2c4_final_holdout_records": 0,
                "v2c4_postmortem_records": 0,
                "v2c4_selection_probe_records": 0,
                "v2c4_training_augmentation_records": 0,
            },
            "sealed_v2c4_holdout_accessed": False,
            "semantic_similarity_used": False,
            "vector_similarity_used": False,
        },
        "next_required": "v2c5_expanded_taxonomy_model_selection",
        "phase": "V2-C5 Step 21C2B",
        "protected_write_intents": list(PROTECTED_WRITE_INTENTS),
        "risk_by_intent": copy.deepcopy(sources.risk_by_intent),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": {
            "path": display_path(paths.seed),
            "schema_version": SEED_SCHEMA_VERSION,
            "sha256": sha256_bytes(seed_bytes),
        },
        "source_artifacts": copy.deepcopy(sources.source_artifacts),
        "taxonomy_intent_labels": list(sources.intent_labels),
    }


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    sources: FrozenSources | None = None,
    probe_paths: Sequence[Path] | None = None,
    script_path: Path | None = None,
) -> BuiltArtifacts:
    frozen = sources or load_frozen_sources(paths, probe_paths=probe_paths)
    seed_validation, seed_bytes = load_and_validate_seed(paths, frozen)
    overlap_counts = validate_no_overlaps(seed_validation.records, frozen.overlap_hashes)
    examples = build_examples(seed_validation.records, frozen.risk_by_intent)
    validate_output_examples(examples, frozen)
    dataset_payload = build_dataset_payload(examples, frozen)
    dataset_bytes = stable_json_bytes(dataset_payload)
    actual_script_path = script_path or Path(__file__).resolve()
    manifest_payload = build_manifest_payload(
        dataset_bytes,
        examples,
        seed_bytes,
        seed_validation,
        overlap_counts,
        frozen,
        paths,
        actual_script_path,
    )
    if manifest_payload["counts"]["protected_write_positive_count"] != EXPECTED_PROTECTED_COUNT:
        raise ValueError("protected-write positive count changed")
    if manifest_payload["counts"]["non_protected_count"] != EXPECTED_NON_PROTECTED_COUNT:
        raise ValueError("non-protected count changed")
    return BuiltArtifacts(
        dataset_bytes=dataset_bytes,
        manifest_bytes=stable_json_bytes(manifest_payload),
        dataset_payload=dataset_payload,
        manifest_payload=manifest_payload,
    )


def build_overlap_diagnostic(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    sources: FrozenSources | None = None,
    probe_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    frozen = sources or load_frozen_sources(paths, probe_paths=probe_paths)
    seed_validation, _ = load_and_validate_seed(paths, frozen)
    records = list_overlap_seed_records(
        seed_validation.records, frozen.overlap_hashes
    )
    return {
        "overlapping_seed_count": len(records),
        "records": records,
    }


def check_artifacts(
    artifacts: BuiltArtifacts, paths: BuildPaths = DEFAULT_PATHS
) -> None:
    dataset_exists = paths.dataset_output.exists()
    manifest_exists = paths.manifest_output.exists()
    if dataset_exists != manifest_exists:
        raise ValueError("final holdout outputs must exist as a pair")
    if not dataset_exists:
        return
    if paths.dataset_output.read_bytes() != artifacts.dataset_bytes:
        raise ValueError("existing final holdout is not the deterministic build")
    if paths.manifest_output.read_bytes() != artifacts.manifest_bytes:
        raise ValueError("existing final holdout manifest is not deterministic")


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


def write_artifacts(
    artifacts: BuiltArtifacts, paths: BuildPaths = DEFAULT_PATHS
) -> None:
    atomic_write_bytes(paths.dataset_output, artifacts.dataset_bytes)
    atomic_write_bytes(paths.manifest_output, artifacts.manifest_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--list-overlap-seed-ids", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_overlap_seed_ids:
            diagnostic = build_overlap_diagnostic()
            print(json.dumps(diagnostic, indent=2, sort_keys=True))
            return 0
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
        f"{action} sealed V2-C5 final holdout: "
        f"examples={counts['total_count']}, "
        f"protected={counts['protected_write_positive_count']}, "
        f"non_protected={counts['non_protected_count']}; "
        "no training, inference, evaluation, or runtime change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
