"""Build the deterministic BANKING77 test-only external evaluation datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
RAW_DATA_DIR = EXTERNAL_DATA_ROOT / "raw/banking77"
RAW_TEST_PATH = RAW_DATA_DIR / "test.csv"
RAW_MANIFEST_PATH = RAW_DATA_DIR / "manifest.json"
MAPPING_PATH = EXTERNAL_DATA_ROOT / "banking77_intent_mapping.json"
OUTPUT_DIR = EXTERNAL_DATA_ROOT / "processed/banking77"
EVALUATION_PATH = OUTPUT_DIR / "banking77_external_eval.json"
REVIEW_POOL_PATH = OUTPUT_DIR / "banking77_review_pool.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

BUILDER_VERSION = "banking77-external-eval-builder.v1"
DATASET_SCHEMA_VERSION = "banking77-external-eval.v1"
REVIEW_POOL_SCHEMA_VERSION = "banking77-review-pool.v1"
MANIFEST_SCHEMA_VERSION = "banking77-processed-manifest.v1"
EXPECTED_SOURCE_ID = "banking77"
EXPECTED_TEST_ROWS = 3_080
EXPECTED_MAPPING_STATUSES = {
    "EXACT_MATCH",
    "NEAR_MATCH",
    "UNSUPPORTED",
    "AMBIGUOUS",
}
PROTECTED_ACTION_INTENTS = {"freeze_card", "create_dispute"}


def stable_json_bytes(payload: Any) -> bytes:
    """Serialize JSON identically across repeated builds."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def raw_file_metadata(raw_manifest: dict[str, Any], file_name: str) -> dict[str, Any]:
    matches = [entry for entry in raw_manifest.get("files", []) if entry.get("path") == file_name]
    if len(matches) != 1:
        raise ValueError(f"raw manifest must contain exactly one {file_name} entry")
    return matches[0]


def validate_source_metadata(
    mapping: dict[str, Any],
    raw_manifest: dict[str, Any],
    test_bytes: bytes,
) -> tuple[str, str]:
    if mapping.get("source_id") != EXPECTED_SOURCE_ID:
        raise ValueError("frozen mapping source_id must be banking77")
    if raw_manifest.get("source_id") != EXPECTED_SOURCE_ID:
        raise ValueError("raw manifest source_id must be banking77")

    source_revision = mapping.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("frozen mapping source_revision must be present")
    if raw_manifest.get("source_revision") != source_revision:
        raise ValueError("raw manifest and frozen mapping revisions differ")

    test_metadata = raw_file_metadata(raw_manifest, "test.csv")
    test_hash = sha256_bytes(test_bytes)
    if test_metadata.get("sha256") != test_hash:
        raise ValueError("raw test.csv SHA-256 does not match its frozen manifest")
    if test_metadata.get("byte_size") != len(test_bytes):
        raise ValueError("raw test.csv byte size does not match its frozen manifest")
    if test_metadata.get("row_count") != EXPECTED_TEST_ROWS:
        raise ValueError(f"raw manifest test row count must be {EXPECTED_TEST_ROWS}")

    return source_revision, test_hash


def load_frozen_mappings(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mappings = mapping.get("mappings")
    if not isinstance(mappings, list):
        raise TypeError("frozen mapping must contain a mappings list")

    by_source_intent: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for entry in mappings:
        if not isinstance(entry, dict):
            raise TypeError("each frozen mapping entry must be an object")
        source_intent = entry.get("source_intent")
        mapping_status = entry.get("mapping_status")
        if not isinstance(source_intent, str) or not source_intent:
            raise ValueError("each frozen mapping entry needs a source_intent")
        if source_intent in by_source_intent:
            raise ValueError(f"duplicate frozen mapping for {source_intent}")
        if mapping_status not in EXPECTED_MAPPING_STATUSES:
            raise ValueError(f"invalid mapping status for {source_intent}: {mapping_status}")
        by_source_intent[source_intent] = entry
        status_counts[mapping_status] += 1

    if len(by_source_intent) != mapping.get("source_intent_count"):
        raise ValueError("frozen mapping intent count does not match its mappings")
    expected_counts = mapping.get("summary", {}).get("mapping_status_counts")
    if dict(status_counts) != expected_counts:
        raise ValueError("frozen mapping status counts do not match its summary")
    return by_source_intent


def load_test_rows(test_bytes: bytes) -> list[dict[str, str]]:
    try:
        text = test_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("raw test.csv must be UTF-8") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != ["text", "category"]:
        raise ValueError("raw test.csv must have exactly text and category columns")

    rows = list(reader)
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"raw test.csv must contain {EXPECTED_TEST_ROWS} rows")
    for row_index, row in enumerate(rows, start=1):
        if not row["text"].strip():
            raise ValueError(f"raw test.csv row {row_index} has empty text")
        if not row["category"].strip():
            raise ValueError(f"raw test.csv row {row_index} has empty category")
    return rows


def partition_for_status(mapping_status: str) -> str:
    return {
        "EXACT_MATCH": "exact_match",
        "UNSUPPORTED": "unsupported",
        "NEAR_MATCH": "near_match_review",
        "AMBIGUOUS": "ambiguous_review",
    }[mapping_status]


def example_id(source_revision: str, row_index: int) -> str:
    return f"banking77:{source_revision}:test:{row_index:06d}"


def build_records(
    rows: list[dict[str, str]],
    mappings: dict[str, dict[str, Any]],
    source_revision: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    exact_match: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    review_pool: list[dict[str, Any]] = []
    partition_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for row_index, row in enumerate(rows, start=1):
        source_intent = row["category"]
        try:
            mapping = mappings[source_intent]
        except KeyError as exc:
            raise ValueError(f"test label missing from frozen mapping: {source_intent}") from exc

        mapping_status = mapping["mapping_status"]
        partition = partition_for_status(mapping_status)
        record_id = example_id(source_revision, row_index)
        if record_id in seen_ids:
            raise ValueError(f"duplicate generated example_id: {record_id}")
        seen_ids.add(record_id)
        partition_counts[partition] += 1

        provenance = {
            "example_id": record_id,
            "source_intent": source_intent,
            "source_revision": source_revision,
            "source_row_index": row_index,
            "source_split": "test",
            "text": row["text"],
        }

        if mapping_status == "EXACT_MATCH":
            expected_intent = mapping.get("sentinelvoice_intent")
            if not isinstance(expected_intent, str) or not expected_intent:
                raise ValueError(f"EXACT_MATCH {source_intent} needs a frozen target intent")
            if expected_intent in PROTECTED_ACTION_INTENTS:
                raise ValueError(
                    f"protected action intent cannot enter the clean external lane: {source_intent}"
                )
            exact_match.append(
                {
                    **provenance,
                    "expected_sentinelvoice_intent": expected_intent,
                    "mapping_status": mapping_status,
                }
            )
        elif mapping_status == "UNSUPPORTED":
            if mapping.get("sentinelvoice_intent") != "unsupported_or_uncertain":
                raise ValueError(
                    f"UNSUPPORTED {source_intent} must map to unsupported_or_uncertain"
                )
            unsupported.append(
                {
                    **provenance,
                    "expected_sentinelvoice_intent": "unsupported_or_uncertain",
                    "mapping_status": mapping_status,
                }
            )
        else:
            review_record = {
                **provenance,
                "candidate_sentinelvoice_intent": mapping.get("sentinelvoice_intent"),
                "mapping_status": mapping_status,
                "protected_action_risk": mapping["protected_action_risk"],
                "rationale": mapping["rationale"],
                "requires_utterance_review": mapping["requires_utterance_review"],
            }
            if "expected_sentinelvoice_intent" in review_record:
                raise ValueError("review-pool rows cannot have forced expected labels")
            review_pool.append(review_record)

    if len(seen_ids) != len(rows):
        raise ValueError("not every test row received one unique example_id")
    if sum(partition_counts.values()) != EXPECTED_TEST_ROWS:
        raise ValueError("test rows were lost or duplicated across output partitions")
    return exact_match, unsupported, review_pool, dict(partition_counts)


def source_intent_coverage(
    rows: list[dict[str, str]], mappings: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    counts = Counter(row["category"] for row in rows)
    return {
        source_intent: {
            "candidate_sentinelvoice_intent": mapping.get("sentinelvoice_intent"),
            "example_count": counts[source_intent],
            "mapping_status": mapping["mapping_status"],
            "output_partition": partition_for_status(mapping["mapping_status"]),
        }
        for source_intent, mapping in mappings.items()
    }


def sentinelvoice_intent_coverage(
    exact_match: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    review_pool: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    evaluation = Counter(
        record["expected_sentinelvoice_intent"]
        for record in [*exact_match, *unsupported]
    )
    review = Counter(
        record["candidate_sentinelvoice_intent"] or "unmapped"
        for record in review_pool
    )
    return {
        "primary_evaluation_expected_intents": dict(sorted(evaluation.items())),
        "review_pool_candidate_intents": dict(sorted(review.items())),
    }


def build_artifacts() -> tuple[bytes, bytes, bytes]:
    test_bytes = RAW_TEST_PATH.read_bytes()
    mapping_bytes = MAPPING_PATH.read_bytes()
    mapping = load_json_object(MAPPING_PATH)
    raw_manifest = load_json_object(RAW_MANIFEST_PATH)
    source_revision, test_hash = validate_source_metadata(
        mapping, raw_manifest, test_bytes
    )
    mappings = load_frozen_mappings(mapping)
    rows = load_test_rows(test_bytes)
    exact_match, unsupported, review_pool, partition_counts = build_records(
        rows, mappings, source_revision
    )

    evaluation_payload = {
        "dataset_version": "banking77-external-eval.2026-09-27.v1",
        "lanes": {
            "exact_match": {
                "description": (
                    "Clean external intent benchmark using only frozen EXACT_MATCH mappings."
                ),
                "example_count": len(exact_match),
                "examples": exact_match,
            },
            "unsupported": {
                "description": (
                    "External banking unsupported/OOS benchmark using only frozen "
                    "UNSUPPORTED mappings."
                ),
                "example_count": len(unsupported),
                "examples": unsupported,
            },
        },
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_split": "test",
    }
    review_payload = {
        "dataset_version": "banking77-review-pool.2026-09-27.v1",
        "description": (
            "Unscored NEAR_MATCH and AMBIGUOUS examples requiring later "
            "utterance-level review."
        ),
        "example_count": len(review_pool),
        "examples": review_pool,
        "schema_version": REVIEW_POOL_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_split": "test",
    }
    evaluation_bytes = stable_json_bytes(evaluation_payload)
    review_bytes = stable_json_bytes(review_payload)

    coverage = sentinelvoice_intent_coverage(
        exact_match, unsupported, review_pool
    )
    manifest_payload = {
        "build_metadata": {
            "builder_version": BUILDER_VERSION,
            "deterministic": True,
            "source_retrieval_date": raw_manifest["retrieval_date"],
            "wall_clock_timestamp_recorded": False,
        },
        "build_script_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "counts": {
            "ambiguous_review": partition_counts.get("ambiguous_review", 0),
            "exact_match": partition_counts.get("exact_match", 0),
            "near_match_review": partition_counts.get("near_match_review", 0),
            "total_review_pool": len(review_pool),
            "total_test_examples": len(rows),
            "unsupported": partition_counts.get("unsupported", 0),
        },
        "coverage_by_source_intent": source_intent_coverage(rows, mappings),
        "coverage_by_sentinelvoice_intent": coverage,
        "frozen_mapping_file_sha256": sha256_bytes(mapping_bytes),
        "processed_files": {
            EVALUATION_PATH.name: {
                "byte_size": len(evaluation_bytes),
                "sha256": sha256_bytes(evaluation_bytes),
            },
            REVIEW_POOL_PATH.name: {
                "byte_size": len(review_bytes),
                "sha256": sha256_bytes(review_bytes),
            },
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_test_file_sha256": test_hash,
        "train_csv_used": False,
        "validation": {
            "all_source_labels_in_frozen_mapping": True,
            "every_test_example_partitioned_once": True,
            "mapping_statuses_copied_unchanged": True,
            "no_duplicate_example_ids": True,
            "no_protected_action_labels_invented": True,
            "no_review_pool_expected_labels": True,
            "source_split": "test",
        },
    }
    manifest_bytes = stable_json_bytes(manifest_payload)
    return evaluation_bytes, review_bytes, manifest_bytes


def main() -> None:
    evaluation_bytes, review_bytes, manifest_bytes = build_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_bytes(evaluation_bytes)
    REVIEW_POOL_PATH.write_bytes(review_bytes)
    MANIFEST_PATH.write_bytes(manifest_bytes)

    manifest = json.loads(manifest_bytes)
    counts = manifest["counts"]
    print(
        "Built BANKING77 external evaluation artifacts: "
        f"exact_match={counts['exact_match']}, "
        f"unsupported={counts['unsupported']}, "
        f"near_match_review={counts['near_match_review']}, "
        f"ambiguous_review={counts['ambiguous_review']}"
    )


if __name__ == "__main__":
    main()
