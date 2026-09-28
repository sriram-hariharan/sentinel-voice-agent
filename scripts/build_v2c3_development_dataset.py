"""Build the governed V2-C3 development dataset and text-free fresh lockbox."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"

BUILDER_VERSION = "v2c3-development-dataset-builder.v1"
DATASET_SCHEMA_VERSION = "v2c3-development-dataset.v1"
LOCKBOX_SCHEMA_VERSION = "v2c3-fresh-lockbox-manifest.v1"
MANIFEST_SCHEMA_VERSION = "v2c3-dataset-manifest.v1"
DATASET_VERSION = "2026-09-28.v2c3-development.v1"
LOCKBOX_VERSION = "2026-09-28.v2c3-fresh-lockbox.v1"
NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"
NORMALIZATION_STEPS = [
    "Unicode NFKC normalization",
    "Unicode lowercase",
    "trim leading and trailing whitespace",
    "collapse every whitespace run to one ASCII space",
]

INTERNAL_SOURCE_ID = "sentinelvoice_v2c1_internal"
BANKING77_SOURCE_ID = "banking77"
CLINC_SOURCE_ID = "clinc150_oos"
ALLOWED_SOURCE_IDS = {
    INTERNAL_SOURCE_ID,
    BANKING77_SOURCE_ID,
    CLINC_SOURCE_ID,
}
PROHIBITED_SOURCE_IDS = {
    "bitext_retail_banking",
    "cfpb_consumer_complaint_narratives_archive",
    "unverified_bank_support_transcript_candidates",
}
ELIGIBLE_INTERNAL_SPLITS = {"train", "validation"}
ELIGIBLE_EXTERNAL_STATUSES = {"EXACT_MATCH", "UNSUPPORTED"}
REVIEW_ONLY_STATUSES = {"NEAR_MATCH", "AMBIGUOUS"}
EXPECTED_MAPPING_STATUSES = ELIGIBLE_EXTERNAL_STATUSES | REVIEW_ONLY_STATUSES
PROTECTED_WRITE_INTENTS = {"freeze_card", "create_dispute"}
EXPECTED_REGISTRY_ELIGIBLE_RECORDS = {
    "banking77_train",
    "clinc_finance_train",
    "clinc_finance_val",
    "internal_train",
    "internal_validation",
}
EXPECTED_LOCKBOX_ELIGIBLE_RECORDS = {
    "banking77_train",
    "clinc_finance_train",
    "clinc_finance_val",
}


@dataclass(frozen=True)
class BuildPaths:
    internal_dataset: Path
    banking77_train: Path
    banking77_raw_manifest: Path
    banking77_mapping: Path
    clinc_data: Path
    clinc_raw_manifest: Path
    clinc_mapping: Path
    registry: Path
    contract: Path
    development_output: Path
    lockbox_output: Path
    manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    internal_dataset=ML_ROOT / "intent_risk_dataset.json",
    banking77_train=EXTERNAL_ROOT / "raw/banking77/train.csv",
    banking77_raw_manifest=EXTERNAL_ROOT / "raw/banking77/manifest.json",
    banking77_mapping=EXTERNAL_ROOT / "banking77_intent_mapping.json",
    clinc_data=EXTERNAL_ROOT / "raw/clinc_oos/data_full.json",
    clinc_raw_manifest=EXTERNAL_ROOT / "raw/clinc_oos/manifest.json",
    clinc_mapping=EXTERNAL_ROOT / "clinc_finance_intent_mapping.json",
    registry=ML_ROOT / "v2c3_data_registry.json",
    contract=ML_ROOT / "v2c3_experiment_contract.json",
    development_output=ML_ROOT / "v2c3_development_dataset.json",
    lockbox_output=ML_ROOT / "v2c3_fresh_lockbox_manifest.json",
    manifest_output=ML_ROOT / "v2c3_dataset.manifest.json",
)


@dataclass(frozen=True)
class SourceBuild:
    records: list[dict[str, Any]]
    exclusions: Counter[str]
    source_revision: str


@dataclass(frozen=True)
class DuplicateSummary:
    duplicate_groups_found: int
    duplicate_groups_forced_away_from_lockbox: int
    duplicate_records: int
    conflicting_duplicate_count: int


@dataclass(frozen=True)
class BuiltArtifacts:
    development_bytes: bytes
    lockbox_bytes: bytes
    manifest_bytes: bytes
    development_payload: dict[str, Any]
    lockbox_payload: dict[str, Any]
    manifest_payload: dict[str, Any]


def stable_json_bytes(payload: Any) -> bytes:
    """Serialize deterministic, human-readable JSON with a trailing newline."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    """Apply the frozen normalization used for duplicate boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


def normalized_text_sha256(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("normalized text cannot be empty")
    return sha256_bytes(normalized.encode("utf-8"))


def text_sha256(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("source text cannot be empty")
    return sha256_bytes(text.encode("utf-8"))


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_sha256(value: bytes, expected: Any, label: str) -> str:
    actual = sha256_bytes(value)
    if expected != actual:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _manifest_entry(
    entries: Sequence[Mapping[str, Any]], path_name: str
) -> Mapping[str, Any]:
    matches = [entry for entry in entries if entry.get("path") == path_name]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one {path_name} entry")
    return matches[0]


def validate_configured_source_ids(source_ids: set[str]) -> None:
    prohibited = source_ids & PROHIBITED_SOURCE_IDS
    if prohibited:
        raise ValueError(f"prohibited V2-C3 sources configured: {sorted(prohibited)}")
    unknown = source_ids - ALLOWED_SOURCE_IDS
    if unknown:
        raise ValueError(f"unapproved V2-C3 sources configured: {sorted(unknown)}")
    if source_ids != ALLOWED_SOURCE_IDS:
        raise ValueError("configured source set must contain exactly the approved sources")


def validate_governance(
    registry: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[list[str], dict[str, str], int, float]:
    """Validate the frozen taxonomy, risk map, source roles, and split policy."""
    if registry.get("schema_version") != "v2c3-data-registry.v1":
        raise ValueError("unexpected V2-C3 data-registry schema")
    if contract.get("schema_version") != "v2c3-experiment-contract.v1":
        raise ValueError("unexpected V2-C3 experiment-contract schema")
    reference = contract.get("data_registry", {})
    if reference.get("schema_version") != registry.get("schema_version"):
        raise ValueError("contract and registry schema versions differ")
    if reference.get("registry_version") != registry.get("registry_version"):
        raise ValueError("contract and registry versions differ")

    intents = contract.get("intent_taxonomy")
    risks = contract.get("risk_taxonomy")
    risk_by_intent = contract.get("risk_by_intent")
    if not isinstance(intents, list) or len(intents) != 9:
        raise ValueError("V2-C3 contract must freeze exactly nine intents")
    if len(intents) != len(set(intents)) or intents != sorted(intents):
        raise ValueError("V2-C3 intents must be unique and sorted")
    if not isinstance(risks, list) or len(risks) != 4:
        raise ValueError("V2-C3 contract must freeze exactly four risk labels")
    if not isinstance(risk_by_intent, dict) or set(risk_by_intent) != set(intents):
        raise ValueError("V2-C3 intent-to-risk mapping is incomplete")
    if set(risk_by_intent.values()) != set(risks):
        raise ValueError("V2-C3 intent-to-risk mapping differs from risk taxonomy")

    records = registry.get("records")
    if not isinstance(records, list):
        raise TypeError("V2-C3 registry records must be a list")
    eligible = {
        row.get("record_id") for row in records if row.get("training_eligible") is True
    }
    lockbox_eligible = {
        row.get("record_id")
        for row in records
        if row.get("fresh_lockbox_eligible") is True
    }
    if eligible != EXPECTED_REGISTRY_ELIGIBLE_RECORDS:
        raise ValueError("registry training eligibility differs from frozen policy")
    if lockbox_eligible != EXPECTED_LOCKBOX_ELIGIBLE_RECORDS:
        raise ValueError("registry lockbox eligibility differs from frozen policy")
    prohibited_eligible = {
        row.get("record_id")
        for row in records
        if row.get("record_id") in contract.get("prohibited_data", {}).get(
            "record_ids", []
        )
        and (
            row.get("training_eligible") is True
            or row.get("model_selection_eligible") is True
            or row.get("fresh_lockbox_eligible") is True
        )
    }
    if prohibited_eligible:
        raise ValueError(
            f"prohibited development records: {sorted(prohibited_eligible)}"
        )

    registry_lockbox = registry.get("fresh_lockbox_design", {})
    contract_lockbox = contract.get("fresh_lockbox_policy", {})
    seed = contract_lockbox.get("seed")
    fraction = contract_lockbox.get("fresh_lockbox_fraction")
    if seed != registry_lockbox.get("seed") or seed != 20260928:
        raise ValueError("fresh-lockbox seed differs from frozen policy")
    if fraction != registry_lockbox.get("fresh_lockbox_fraction") or fraction != 0.2:
        raise ValueError("fresh-lockbox fraction differs from frozen policy")
    if contract_lockbox.get("materialized") is not False:
        raise ValueError("contract must represent the pre-materialization decision")
    if contract_lockbox.get("lockbox_allowed_for_model_selection") is not False:
        raise ValueError("fresh lockbox cannot be model-selection eligible")
    if contract.get("runtime_authority") is not False:
        raise ValueError("V2-C3 data contract cannot grant runtime authority")

    validate_configured_source_ids(set(ALLOWED_SOURCE_IDS))
    return intents, risk_by_intent, seed, fraction


def _base_record(
    *,
    example_id: str,
    text: str,
    intent: str,
    risk_by_intent: Mapping[str, str],
    group_id: str,
    source_id: str,
    source_split: str,
    source_label: str,
    mapping_status: str,
    original_example_id: str | None,
    source_row_index: int | None,
    source_revision: str,
    source_domain: str | None,
    original_split: str,
    data_role: str,
) -> dict[str, Any]:
    try:
        risk = risk_by_intent[intent]
    except KeyError as exc:
        raise ValueError(f"intent is outside the frozen taxonomy: {intent}") from exc
    return {
        "data_role": data_role,
        "example_id": example_id,
        "group_id": group_id,
        "intent": intent,
        "mapping_status": mapping_status,
        "normalized_text_sha256": normalized_text_sha256(text),
        "original_example_id": original_example_id,
        "original_split": original_split,
        "risk": risk,
        "source_domain": source_domain,
        "source_id": source_id,
        "source_label": source_label,
        "source_revision": source_revision,
        "source_row_index": source_row_index,
        "source_split": source_split,
        "text": text,
        "text_sha256": text_sha256(text),
    }


def build_internal_records(
    dataset: Mapping[str, Any],
    intents: Sequence[str],
    risk_by_intent: Mapping[str, str],
) -> list[dict[str, Any]]:
    if dataset.get("schema_version") != "intent-risk.v1":
        raise ValueError("unexpected internal dataset schema")
    dataset_version = dataset.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("internal dataset version is missing")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("internal dataset examples must be a list")

    source_intents = {row.get("intent") for row in examples}
    if source_intents != set(intents):
        raise ValueError("internal intent taxonomy differs from V2-C3 contract")
    source_splits = {row.get("split") for row in examples}
    if source_splits != {"train", "validation", "locked_test"}:
        raise ValueError("internal dataset splits differ from the frozen contract")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_row_index, row in enumerate(examples, start=1):
        split = row.get("split")
        if split not in ELIGIBLE_INTERNAL_SPLITS:
            continue
        example_id = row.get("example_id")
        group_id = row.get("group_id")
        intent = row.get("intent")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("internal example_id must be non-empty")
        if example_id in seen_ids:
            raise ValueError(f"duplicate internal example_id: {example_id}")
        seen_ids.add(example_id)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"internal group_id missing for {example_id}")
        if row.get("risk") != risk_by_intent.get(intent):
            raise ValueError(f"internal risk differs from contract for {example_id}")
        record = _base_record(
            example_id=example_id,
            text=row.get("text"),
            intent=intent,
            risk_by_intent=risk_by_intent,
            group_id=group_id,
            source_id=INTERNAL_SOURCE_ID,
            source_split=split,
            source_label=intent,
            mapping_status="INTERNAL_FROZEN",
            original_example_id=example_id,
            source_row_index=source_row_index,
            source_revision=dataset_version,
            source_domain="sentinelvoice",
            original_split=split,
            data_role="development",
        )
        records.append(record)
    if any(row["source_split"] == "locked_test" for row in records):
        raise ValueError("internal locked_test entered V2-C3 development")
    return records


def load_frozen_mappings(
    mapping: Mapping[str, Any],
    expected_source_id: str,
    expected_intents: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if mapping.get("source_id") != expected_source_id:
        raise ValueError(f"mapping source_id must be {expected_source_id}")
    mapping_intents = mapping.get("sentinelvoice_intents")
    if not isinstance(mapping_intents, list) or set(mapping_intents) != set(
        expected_intents
    ):
        raise ValueError(
            f"{expected_source_id} mapping taxonomy differs from V2-C3 contract"
        )
    entries = mapping.get("mappings")
    if not isinstance(entries, list):
        raise TypeError("frozen mapping must contain a mappings list")
    result: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("each frozen mapping entry must be an object")
        source_intent = entry.get("source_intent")
        status = entry.get("mapping_status")
        if not isinstance(source_intent, str) or not source_intent:
            raise ValueError("mapping source_intent must be non-empty")
        if source_intent in result:
            raise ValueError(f"duplicate frozen mapping: {source_intent}")
        if status not in EXPECTED_MAPPING_STATUSES:
            raise ValueError(f"unexpected mapping status for {source_intent}: {status}")
        result[source_intent] = dict(entry)
        counts[status] += 1
    expected_counts = mapping.get("summary", {}).get("mapping_status_counts")
    if dict(counts) != expected_counts:
        raise ValueError("mapping status counts differ from frozen summary")
    return result


def _mapped_target(mapping: Mapping[str, Any]) -> str | None:
    status = mapping.get("mapping_status")
    source_intent = mapping.get("source_intent")
    if status in REVIEW_ONLY_STATUSES:
        return None
    if status == "UNSUPPORTED":
        if mapping.get("sentinelvoice_intent") != "unsupported_or_uncertain":
            raise ValueError(
                f"UNSUPPORTED {source_intent} must map to unsupported_or_uncertain"
            )
        return "unsupported_or_uncertain"
    if status != "EXACT_MATCH":
        raise ValueError(f"unrecognized mapping status for {source_intent}: {status}")
    target = mapping.get("sentinelvoice_intent")
    if not isinstance(target, str) or not target:
        raise ValueError(f"EXACT_MATCH {source_intent} has no frozen target")
    if target in PROTECTED_WRITE_INTENTS:
        raise ValueError(
            f"unsafe automatic protected-write mapping: {source_intent} -> {target}"
        )
    return target


def _parse_banking77_rows(source_bytes: bytes) -> list[dict[str, str]]:
    try:
        decoded = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("BANKING77 train.csv must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if reader.fieldnames != ["text", "category"]:
        raise ValueError("BANKING77 train.csv columns must be text,category")
    rows = list(reader)
    for row_index, row in enumerate(rows, start=1):
        if not row.get("text", "").strip() or not row.get("category", "").strip():
            raise ValueError(f"BANKING77 train row {row_index} is incomplete")
    return rows


def build_banking77_records(
    rows: Sequence[Mapping[str, str]],
    mappings: Mapping[str, Mapping[str, Any]],
    source_revision: str,
    risk_by_intent: Mapping[str, str],
) -> SourceBuild:
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row_index, row in enumerate(rows, start=1):
        source_label = row["category"]
        if source_label not in mappings:
            raise ValueError(f"BANKING77 train label missing mapping: {source_label}")
        mapping = mappings[source_label]
        target = _mapped_target(mapping)
        status = mapping["mapping_status"]
        if target is None:
            exclusions[status] += 1
            continue
        example_id = f"banking77:{source_revision}:train:{row_index:06d}"
        records.append(
            _base_record(
                example_id=example_id,
                text=row["text"],
                intent=target,
                risk_by_intent=risk_by_intent,
                group_id=f"v2c3-source:{example_id}",
                source_id=BANKING77_SOURCE_ID,
                source_split="train",
                source_label=source_label,
                mapping_status=status,
                original_example_id=None,
                source_row_index=row_index,
                source_revision=source_revision,
                source_domain="banking",
                original_split="train",
                data_role="external_candidate",
            )
        )
    return SourceBuild(records, exclusions, source_revision)


def _parse_clinc_source(source_bytes: bytes) -> dict[str, list[list[str]]]:
    try:
        source = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CLINC data_full.json must be valid UTF-8 JSON") from exc
    if not isinstance(source, dict):
        raise TypeError("CLINC data_full.json must contain a JSON object")
    expected_splits = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if set(source) != expected_splits:
        raise ValueError("CLINC source split names differ from frozen raw manifest")
    for split, rows in source.items():
        if not isinstance(rows, list):
            raise TypeError(f"CLINC {split} split must be a list")
        for row_index, row in enumerate(rows, start=1):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"CLINC {split} row {row_index} must be [text, label]")
            text, label = row
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"CLINC {split} row {row_index} has empty text")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"CLINC {split} row {row_index} has empty label")
    return source


def build_clinc_records(
    source: Mapping[str, Sequence[Sequence[str]]],
    mappings: Mapping[str, Mapping[str, Any]],
    source_revision: str,
    risk_by_intent: Mapping[str, str],
) -> SourceBuild:
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for split in ("train", "val"):
        for row_index, row in enumerate(source[split], start=1):
            text, source_label = row
            mapping = mappings.get(source_label)
            if mapping is None:
                exclusions["NON_FINANCE"] += 1
                continue
            target = _mapped_target(mapping)
            status = mapping["mapping_status"]
            if target is None:
                exclusions[status] += 1
                continue
            example_id = f"clinc:{source_revision}:{split}:{row_index:06d}"
            records.append(
                _base_record(
                    example_id=example_id,
                    text=text,
                    intent=target,
                    risk_by_intent=risk_by_intent,
                    group_id=f"v2c3-source:{example_id}",
                    source_id=CLINC_SOURCE_ID,
                    source_split=split,
                    source_label=source_label,
                    mapping_status=status,
                    original_example_id=None,
                    source_row_index=row_index,
                    source_revision=source_revision,
                    source_domain=mapping.get("source_domain"),
                    original_split=split,
                    data_role="external_candidate",
                )
            )
    return SourceBuild(records, exclusions, source_revision)


def apply_duplicate_groups(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], DuplicateSummary, set[str]]:
    """Propagate normalized duplicate boundaries and reject target conflicts."""
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_hash[row["normalized_text_sha256"]].append(row)

    conflicting: dict[str, list[str]] = {}
    for digest, rows in by_hash.items():
        intents = sorted({str(row["intent"]) for row in rows})
        if len(intents) > 1:
            conflicting[digest] = intents
    if conflicting:
        first_digest = min(conflicting)
        conflict_intents = conflicting[first_digest]
        raise ValueError(
            f"conflicting target intents for {first_digest}: {conflict_intents}"
        )

    duplicate_hashes = {digest for digest, rows in by_hash.items() if len(rows) > 1}
    forced_development: set[str] = set()
    rewritten: list[dict[str, Any]] = []
    for digest, rows in by_hash.items():
        internal_rows = [row for row in rows if row["source_id"] == INTERNAL_SOURCE_ID]
        if internal_rows:
            internal_groups = {str(row["group_id"]) for row in internal_rows}
            if len(internal_groups) != 1 and len(rows) > 1:
                raise ValueError(f"internal duplicate group mismatch: {digest}")
            shared_group = min(internal_groups)
            if any(row["source_id"] != INTERNAL_SOURCE_ID for row in rows):
                forced_development.add(shared_group)
        elif len(rows) > 1:
            shared_group = f"v2c3-duplicate:{digest}"
        else:
            shared_group = str(rows[0]["group_id"])

        for row in rows:
            updated = dict(row)
            if row["source_id"] != INTERNAL_SOURCE_ID:
                updated["group_id"] = shared_group
            rewritten.append(updated)

    summary = DuplicateSummary(
        duplicate_groups_found=len(duplicate_hashes),
        duplicate_groups_forced_away_from_lockbox=len(forced_development),
        duplicate_records=sum(len(by_hash[digest]) for digest in duplicate_hashes),
        conflicting_duplicate_count=0,
    )
    return rewritten, summary, forced_development


def _stable_group_score(seed: int, stratum: tuple[str, str], group_id: str) -> str:
    source_id, intent = stratum
    value = f"{seed}|{source_id}|{intent}|{group_id}".encode()
    return sha256_bytes(value)


def split_external_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    lockbox_fraction: float,
    forced_development_groups: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    internal = [dict(row) for row in records if row["source_id"] == INTERNAL_SOURCE_ID]
    external = [dict(row) for row in records if row["source_id"] != INTERNAL_SOURCE_ID]
    groups_by_stratum: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in external:
        groups_by_stratum[(row["source_id"], row["intent"])].add(row["group_id"])

    lockbox_groups: set[str] = set()
    stratum_plan: dict[str, dict[str, int]] = {}
    for stratum in sorted(groups_by_stratum):
        groups = groups_by_stratum[stratum]
        selectable = groups - forced_development_groups
        target = int(len(selectable) * lockbox_fraction + 0.5)
        if selectable and lockbox_fraction > 0 and target == 0:
            target = 1
        already_selected = len(selectable & lockbox_groups)
        needed = max(0, target - already_selected)
        available = selectable - lockbox_groups
        ranked = sorted(
            available,
            key=lambda group_id: (
                _stable_group_score(seed, stratum, group_id),
                group_id,
            ),
        )
        lockbox_groups.update(ranked[:needed])
        key = f"{stratum[0]}:{stratum[1]}"
        stratum_plan[key] = {
            "external_group_count": len(groups),
            "forced_development_group_count": len(groups & forced_development_groups),
            "lockbox_eligible_group_count": len(selectable),
            "target_lockbox_group_count": target,
        }

    development: list[dict[str, Any]] = []
    lockbox: list[dict[str, Any]] = []
    for row in internal:
        row["data_role"] = "development"
        development.append(row)
    for row in external:
        if row["group_id"] in lockbox_groups:
            row["data_role"] = "fresh_lockbox"
            lockbox.append(row)
        else:
            row["data_role"] = "development"
            development.append(row)

    for key, plan in stratum_plan.items():
        source_id, intent = key.split(":", maxsplit=1)
        plan["actual_lockbox_group_count"] = len(
            {
                row["group_id"]
                for row in lockbox
                if row["source_id"] == source_id and row["intent"] == intent
            }
        )
    return development, lockbox, stratum_plan


def _record_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    source_order = {
        INTERNAL_SOURCE_ID: 0,
        BANKING77_SOURCE_ID: 1,
        CLINC_SOURCE_ID: 2,
    }
    split_order = {"train": 0, "validation": 1, "val": 1}
    return (
        source_order.get(str(row["source_id"]), 99),
        split_order.get(str(row["source_split"]), 99),
        row["source_row_index"] or 0,
        row["example_id"],
    )


def lockbox_member(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the text-free source identity needed for later reconstruction."""
    member = {
        "data_role": "fresh_lockbox",
        "example_id": row["example_id"],
        "group_id": row["group_id"],
        "intent": row["intent"],
        "mapping_status": row["mapping_status"],
        "normalized_text_sha256": row["normalized_text_sha256"],
        "risk": row["risk"],
        "source_domain": row["source_domain"],
        "source_id": row["source_id"],
        "source_label": row["source_label"],
        "source_revision": row["source_revision"],
        "source_row_index": row["source_row_index"],
        "source_split": row["source_split"],
        "text_sha256": row["text_sha256"],
    }
    if "text" in member:
        raise ValueError("fresh-lockbox membership cannot contain text")
    return member


def validate_partition(
    development: Sequence[Mapping[str, Any]],
    lockbox: Sequence[Mapping[str, Any]],
    intents: Sequence[str],
    risk_by_intent: Mapping[str, str],
) -> None:
    all_records = [*development, *lockbox]
    example_ids = [row["example_id"] for row in all_records]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("canonical V2-C3 example_id values must be unique")
    if any(row["intent"] not in intents for row in all_records):
        raise ValueError("V2-C3 record uses intent outside frozen taxonomy")
    if any(row["risk"] != risk_by_intent[row["intent"]] for row in all_records):
        raise ValueError("V2-C3 record risk is not derived from frozen intent mapping")
    if any(row["source_split"] == "locked_test" for row in all_records):
        raise ValueError("internal locked_test contamination detected")
    if any(
        row["source_id"] == BANKING77_SOURCE_ID and row["source_split"] != "train"
        for row in all_records
    ):
        raise ValueError("BANKING77 non-train contamination detected")
    if any(
        row["source_id"] == CLINC_SOURCE_ID
        and row["source_split"] not in {"train", "val"}
        for row in all_records
    ):
        raise ValueError("CLINC test or OOS contamination detected")
    if any(row["source_id"] not in ALLOWED_SOURCE_IDS for row in all_records):
        raise ValueError("prohibited or unknown source contamination detected")
    if any(row["mapping_status"] in REVIEW_ONLY_STATUSES for row in all_records):
        raise ValueError("review-only mapping status entered V2-C3 data")

    development_groups = {row["group_id"] for row in development}
    lockbox_groups = {row["group_id"] for row in lockbox}
    if development_groups & lockbox_groups:
        raise ValueError("group_id overlap between development and fresh lockbox")
    development_hashes = {row["normalized_text_sha256"] for row in development}
    lockbox_hashes = {row["normalized_text_sha256"] for row in lockbox}
    if development_hashes & lockbox_hashes:
        raise ValueError("normalized-text overlap between development and lockbox")
    if any(row["source_id"] == INTERNAL_SOURCE_ID for row in lockbox):
        raise ValueError("internal examples cannot enter the external fresh lockbox")


def _role_counts(
    development: Sequence[Mapping[str, Any]],
    lockbox: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, int]]:
    development_counts = Counter(str(row[field]) for row in development)
    lockbox_counts = Counter(str(row[field]) for row in lockbox)
    values = sorted(set(development_counts) | set(lockbox_counts))
    return {
        value: {
            "development": development_counts[value],
            "fresh_lockbox": lockbox_counts[value],
            "total": development_counts[value] + lockbox_counts[value],
        }
        for value in values
    }


def _source_split_counts(
    development: Sequence[Mapping[str, Any]],
    lockbox: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    def key(row: Mapping[str, Any]) -> str:
        return f"{row['source_id']}:{row['source_split']}"

    development_counts = Counter(key(row) for row in development)
    lockbox_counts = Counter(key(row) for row in lockbox)
    values = sorted(set(development_counts) | set(lockbox_counts))
    return {
        value: {
            "development": development_counts[value],
            "fresh_lockbox": lockbox_counts[value],
            "total": development_counts[value] + lockbox_counts[value],
        }
        for value in values
    }


def _validate_input_hashes(
    *,
    paths: BuildPaths,
    registry: Mapping[str, Any],
    internal_bytes: bytes,
    banking_bytes: bytes,
    banking_mapping_bytes: bytes,
    banking_manifest: Mapping[str, Any],
    clinc_bytes: bytes,
    clinc_mapping_bytes: bytes,
    clinc_manifest: Mapping[str, Any],
) -> dict[str, str]:
    references = registry.get("canonical_references", {})
    internal_ref = references.get("internal_dataset", {})
    banking_ref = references.get("banking77", {})
    clinc_ref = references.get("clinc150", {})
    hashes = {
        "internal_dataset_sha256": _require_sha256(
            internal_bytes,
            internal_ref.get("dataset_sha256"),
            "internal dataset",
        ),
        "banking77_train_sha256": _require_sha256(
            banking_bytes,
            banking_ref.get("train_sha256"),
            "BANKING77 train.csv",
        ),
        "banking77_mapping_sha256": _require_sha256(
            banking_mapping_bytes,
            banking_ref.get("mapping_sha256"),
            "BANKING77 mapping",
        ),
        "clinc_data_full_sha256": _require_sha256(
            clinc_bytes,
            clinc_ref.get("data_full_sha256"),
            "CLINC data_full.json",
        ),
        "clinc_mapping_sha256": _require_sha256(
            clinc_mapping_bytes,
            clinc_ref.get("mapping_sha256"),
            "CLINC finance mapping",
        ),
    }
    banking_entry = _manifest_entry(banking_manifest.get("files", []), "train.csv")
    if banking_entry.get("sha256") != hashes["banking77_train_sha256"]:
        raise ValueError("BANKING77 raw manifest train hash differs")
    if banking_entry.get("row_count") != 10_003:
        raise ValueError("BANKING77 raw manifest train count must be 10003")
    clinc_entry = _manifest_entry(clinc_manifest.get("raw_files", []), "data_full.json")
    if clinc_entry.get("sha256") != hashes["clinc_data_full_sha256"]:
        raise ValueError("CLINC raw manifest data hash differs")
    if banking_manifest.get("source_revision") != banking_ref.get("source_revision"):
        raise ValueError("BANKING77 raw manifest revision differs from registry")
    if clinc_manifest.get("pinned_revision") != clinc_ref.get("source_revision"):
        raise ValueError("CLINC raw manifest revision differs from registry")
    hashes.update(
        {
            "banking77_raw_manifest_sha256": sha256_bytes(
                paths.banking77_raw_manifest.read_bytes()
            ),
            "clinc_raw_manifest_sha256": sha256_bytes(
                paths.clinc_raw_manifest.read_bytes()
            ),
        }
    )
    return hashes


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    script_path: Path | None = None,
) -> BuiltArtifacts:
    """Build all payloads in memory without writing tracked artifacts."""
    internal_bytes = paths.internal_dataset.read_bytes()
    banking_bytes = paths.banking77_train.read_bytes()
    banking_mapping_bytes = paths.banking77_mapping.read_bytes()
    clinc_bytes = paths.clinc_data.read_bytes()
    clinc_mapping_bytes = paths.clinc_mapping.read_bytes()
    registry_bytes = paths.registry.read_bytes()
    contract_bytes = paths.contract.read_bytes()

    internal_dataset = load_json_object(paths.internal_dataset)
    banking_mapping = load_json_object(paths.banking77_mapping)
    banking_manifest = load_json_object(paths.banking77_raw_manifest)
    clinc_mapping = load_json_object(paths.clinc_mapping)
    clinc_manifest = load_json_object(paths.clinc_raw_manifest)
    registry = load_json_object(paths.registry)
    contract = load_json_object(paths.contract)
    intents, risk_by_intent, seed, lockbox_fraction = validate_governance(
        registry, contract
    )
    input_hashes = _validate_input_hashes(
        paths=paths,
        registry=registry,
        internal_bytes=internal_bytes,
        banking_bytes=banking_bytes,
        banking_mapping_bytes=banking_mapping_bytes,
        banking_manifest=banking_manifest,
        clinc_bytes=clinc_bytes,
        clinc_mapping_bytes=clinc_mapping_bytes,
        clinc_manifest=clinc_manifest,
    )

    internal_records = build_internal_records(
        internal_dataset, intents, risk_by_intent
    )
    banking_mappings = load_frozen_mappings(
        banking_mapping, BANKING77_SOURCE_ID, intents
    )
    banking_rows = _parse_banking77_rows(banking_bytes)
    if len(banking_rows) != 10_003:
        raise ValueError("BANKING77 train.csv must contain 10003 rows")
    banking_build = build_banking77_records(
        banking_rows,
        banking_mappings,
        banking_manifest["source_revision"],
        risk_by_intent,
    )

    clinc_mappings = load_frozen_mappings(
        clinc_mapping, CLINC_SOURCE_ID, intents
    )
    clinc_source = _parse_clinc_source(clinc_bytes)
    expected_clinc_counts = clinc_manifest.get("all_split_counts")
    actual_clinc_counts = {key: len(rows) for key, rows in clinc_source.items()}
    if actual_clinc_counts != expected_clinc_counts:
        raise ValueError("CLINC split counts differ from frozen raw manifest")
    clinc_build = build_clinc_records(
        clinc_source,
        clinc_mappings,
        clinc_manifest["pinned_revision"],
        risk_by_intent,
    )

    eligible_records = [
        *internal_records,
        *banking_build.records,
        *clinc_build.records,
    ]
    grouped_records, duplicate_summary, forced_development_groups = (
        apply_duplicate_groups(eligible_records)
    )
    development, lockbox, stratum_plan = split_external_groups(
        grouped_records,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
        forced_development_groups=forced_development_groups,
    )
    development.sort(key=_record_sort_key)
    lockbox.sort(key=_record_sort_key)
    validate_partition(development, lockbox, intents, risk_by_intent)

    lockbox_members = [lockbox_member(row) for row in lockbox]
    development_payload = {
        "advisory_only": True,
        "dataset_version": DATASET_VERSION,
        "example_count": len(development),
        "examples": development,
        "normalization_version": NORMALIZATION_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "training_or_evaluation_performed": False,
    }
    lockbox_payload = {
        "contains_text": False,
        "lockbox_version": LOCKBOX_VERSION,
        "member_count": len(lockbox_members),
        "members": lockbox_members,
        "normalization_version": NORMALIZATION_VERSION,
        "schema_version": LOCKBOX_SCHEMA_VERSION,
        "seed": seed,
        "selection_or_evaluation_performed": False,
    }
    development_bytes = stable_json_bytes(development_payload)
    lockbox_bytes = stable_json_bytes(lockbox_payload)

    external_total = len(banking_build.records) + len(clinc_build.records)
    external_development_count = sum(
        row["source_id"] != INTERNAL_SOURCE_ID for row in development
    )
    exclusions = banking_build.exclusions + clinc_build.exclusions
    source_exclusions = {
        BANKING77_SOURCE_ID: dict(sorted(banking_build.exclusions.items())),
        CLINC_SOURCE_ID: dict(sorted(clinc_build.exclusions.items())),
    }
    manifest_payload = {
        "build_metadata": {
            "builder_version": BUILDER_VERSION,
            "deterministic": True,
            "wall_clock_timestamp_recorded": False,
        },
        "build_script_sha256": sha256_bytes(
            (script_path or Path(__file__).resolve()).read_bytes()
        ),
        "contract_sha256": sha256_bytes(contract_bytes),
        "counts": {
            "actual_external_lockbox_fraction": (
                len(lockbox) / external_total if external_total else 0.0
            ),
            "conflicting_duplicate_count": (
                duplicate_summary.conflicting_duplicate_count
            ),
            "development_total": len(development),
            "development_group_count": len(
                {row["group_id"] for row in development}
            ),
            "duplicate_groups_forced_away_from_lockbox": (
                duplicate_summary.duplicate_groups_forced_away_from_lockbox
            ),
            "duplicate_groups_found": duplicate_summary.duplicate_groups_found,
            "duplicate_records": duplicate_summary.duplicate_records,
            "excluded_ambiguous_count": exclusions["AMBIGUOUS"],
            "excluded_near_match_count": exclusions["NEAR_MATCH"],
            "excluded_non_finance_clinc_count": exclusions["NON_FINANCE"],
            "excluded_pending_semantic_review": (
                exclusions["AMBIGUOUS"] + exclusions["NEAR_MATCH"]
            ),
            "external_automatically_eligible_total": external_total,
            "external_development": external_development_count,
            "fresh_lockbox": len(lockbox),
            "fresh_lockbox_group_count": len(
                {row["group_id"] for row in lockbox}
            ),
            "internal_development": len(internal_records),
            "prohibited_source_count": 0,
            "total_group_count": len(
                {row["group_id"] for row in [*development, *lockbox]}
            ),
        },
        "counts_by_mapping_status": _role_counts(
            development, lockbox, "mapping_status"
        ),
        "counts_by_source": _role_counts(development, lockbox, "source_id"),
        "counts_by_source_split": _source_split_counts(development, lockbox),
        "counts_by_target_intent": _role_counts(development, lockbox, "intent"),
        "data_roles": {
            "development": "Eligible internal train/validation plus external groups not assigned to the fresh lockbox.",
            "fresh_lockbox": "Text-free tracked membership; unavailable to fitting, CV, tuning, or selection.",
        },
        "excluded_counts_by_source": source_exclusions,
        "input_hashes": {
            **input_hashes,
            "contract_sha256": sha256_bytes(contract_bytes),
            "registry_sha256": sha256_bytes(registry_bytes),
        },
        "input_paths": {
            "banking77_mapping": _display_path(paths.banking77_mapping),
            "banking77_raw_manifest": _display_path(paths.banking77_raw_manifest),
            "banking77_train": _display_path(paths.banking77_train),
            "clinc_data": _display_path(paths.clinc_data),
            "clinc_mapping": _display_path(paths.clinc_mapping),
            "clinc_raw_manifest": _display_path(paths.clinc_raw_manifest),
            "contract": _display_path(paths.contract),
            "internal_dataset": _display_path(paths.internal_dataset),
            "registry": _display_path(paths.registry),
        },
        "lockbox_policy": {
            "actual_external_lockbox_fraction": (
                len(lockbox) / external_total if external_total else 0.0
            ),
            "configured_fraction": lockbox_fraction,
            "group_level": True,
            "model_output_or_confidence_used": False,
            "seed": seed,
            "source_and_target_strata": stratum_plan,
            "text_tracked": False,
        },
        "normalization": {
            "hash_algorithm": "SHA-256",
            "steps": NORMALIZATION_STEPS,
            "version": NORMALIZATION_VERSION,
        },
        "output_hashes": {
            "v2c3_development_dataset.json": sha256_bytes(development_bytes),
            "v2c3_fresh_lockbox_manifest.json": sha256_bytes(lockbox_bytes),
        },
        "registry_sha256": sha256_bytes(registry_bytes),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "training_or_evaluation_performed": False,
        "validation": {
            "automatic_protected_write_targets_absent": not any(
                row["intent"] in PROTECTED_WRITE_INTENTS
                and row["source_id"] != INTERNAL_SOURCE_ID
                for row in [*development, *lockbox]
            ),
            "banking77_test_used": False,
            "cfpb_used": False,
            "clinc_oos_used": False,
            "clinc_test_used": False,
            "conflicting_normalized_targets": 0,
            "development_lockbox_group_overlap": 0,
            "development_lockbox_normalized_hash_overlap": 0,
            "internal_locked_test_used": False,
            "lockbox_contains_text": False,
            "near_or_ambiguous_included": False,
            "prohibited_sources_used": False,
            "risk_derived_from_frozen_contract": True,
            "taxonomy_matches_frozen_contract": True,
        },
    }
    manifest_bytes = stable_json_bytes(manifest_payload)
    return BuiltArtifacts(
        development_bytes=development_bytes,
        lockbox_bytes=lockbox_bytes,
        manifest_bytes=manifest_bytes,
        development_payload=development_payload,
        lockbox_payload=lockbox_payload,
        manifest_payload=manifest_payload,
    )


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably write bytes and atomically replace a same-filesystem target."""
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
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    atomic_write_bytes(paths.development_output, artifacts.development_bytes)
    atomic_write_bytes(paths.lockbox_output, artifacts.lockbox_bytes)
    atomic_write_bytes(paths.manifest_output, artifacts.manifest_bytes)


def check_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    expected = {
        paths.development_output: artifacts.development_bytes,
        paths.lockbox_output: artifacts.lockbox_bytes,
        paths.manifest_output: artifacts.manifest_bytes,
    }
    for path, payload in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"V2-C3 artifact does not exist: {path}; use --write")
        if path.read_bytes() != payload:
            raise ValueError(f"V2-C3 artifact differs from deterministic build: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write all artifacts")
    mode.add_argument("--check", action="store_true", help="verify all artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = build_artifacts()
    if args.write:
        write_artifacts(artifacts)
        action = "Wrote"
    else:
        check_artifacts(artifacts)
        action = "Validated"
    counts = artifacts.manifest_payload["counts"]
    print(
        action,
        "V2-C3 data artifacts:",
        f"development={counts['development_total']},",
        f"external_development={counts['external_development']},",
        f"fresh_lockbox={counts['fresh_lockbox']},",
        f"lockbox_fraction={counts['actual_external_lockbox_fraction']:.6f};",
        "no training or evaluation performed.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
