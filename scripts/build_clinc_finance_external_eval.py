"""Build deterministic CLINC test-only finance evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
RAW_DATA_DIR = EXTERNAL_DATA_ROOT / "raw/clinc_oos"
RAW_DATA_PATH = RAW_DATA_DIR / "data_full.json"
RAW_MANIFEST_PATH = RAW_DATA_DIR / "manifest.json"
MAPPING_PATH = EXTERNAL_DATA_ROOT / "clinc_finance_intent_mapping.json"
OUTPUT_DIR = EXTERNAL_DATA_ROOT / "processed/clinc_finance"
EVALUATION_PATH = OUTPUT_DIR / "clinc_finance_external_eval.json"
REVIEW_POOL_PATH = OUTPUT_DIR / "clinc_finance_review_pool.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

BUILDER_VERSION = "clinc-finance-external-eval-builder.v1"
DATASET_SCHEMA_VERSION = "clinc-finance-external-eval.v1"
REVIEW_POOL_SCHEMA_VERSION = "clinc-finance-review-pool.v1"
MANIFEST_SCHEMA_VERSION = "clinc-finance-processed-manifest.v1"
DATASET_VERSION = "clinc-finance-external-eval.2026-09-27.v1"
REVIEW_POOL_VERSION = "clinc-finance-review-pool.2026-09-27.v1"
EXPECTED_SOURCE_ID = "clinc150_oos"
EXPECTED_SOURCE_SPLIT = "test"
EXPECTED_SOURCE_SPLIT_COUNT = 4_500
EXPECTED_FINANCE_INTENT_COUNT = 30
EXPECTED_EXAMPLES_PER_INTENT = 30
EXPECTED_FINANCE_EXAMPLE_COUNT = 900
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


def raw_file_metadata(raw_manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        entry
        for entry in raw_manifest.get("raw_files", [])
        if entry.get("path") == RAW_DATA_PATH.name
    ]
    if len(matches) != 1:
        raise ValueError("raw manifest must contain exactly one data_full.json entry")
    return matches[0]


def validate_source_metadata(
    mapping: dict[str, Any],
    raw_manifest: dict[str, Any],
    raw_bytes: bytes,
) -> tuple[str, str]:
    if mapping.get("source_id") != EXPECTED_SOURCE_ID:
        raise ValueError("frozen mapping source_id must be clinc150_oos")
    if raw_manifest.get("source_id") != EXPECTED_SOURCE_ID:
        raise ValueError("raw manifest source_id must be clinc150_oos")

    source_revision = mapping.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("frozen mapping source_revision must be present")
    if raw_manifest.get("pinned_revision") != source_revision:
        raise ValueError("raw manifest and frozen mapping revisions differ")

    metadata = raw_file_metadata(raw_manifest)
    raw_hash = sha256_bytes(raw_bytes)
    if metadata.get("sha256") != raw_hash:
        raise ValueError("raw data_full.json SHA-256 differs from its manifest")
    if metadata.get("byte_size") != len(raw_bytes):
        raise ValueError("raw data_full.json byte size differs from its manifest")
    if mapping.get("source_dataset_sha256") != raw_hash:
        raise ValueError("frozen mapping and raw data_full.json hashes differ")
    if raw_manifest.get("all_split_counts", {}).get(EXPECTED_SOURCE_SPLIT) != (
        EXPECTED_SOURCE_SPLIT_COUNT
    ):
        raise ValueError("raw manifest test count must be 4500")
    return source_revision, raw_hash


def load_frozen_mappings(
    mapping: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    mappings = mapping.get("mappings")
    if not isinstance(mappings, list):
        raise TypeError("frozen mapping must contain a mappings list")

    by_source_intent: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for entry in mappings:
        if not isinstance(entry, dict):
            raise TypeError("each frozen mapping entry must be an object")
        source_domain = entry.get("source_domain")
        source_intent = entry.get("source_intent")
        mapping_status = entry.get("mapping_status")
        if not isinstance(source_domain, str) or not source_domain:
            raise ValueError("each frozen mapping entry needs a source_domain")
        if not isinstance(source_intent, str) or not source_intent:
            raise ValueError("each frozen mapping entry needs a source_intent")
        if source_intent in by_source_intent:
            raise ValueError(f"duplicate frozen mapping for {source_intent}")
        if mapping_status not in EXPECTED_MAPPING_STATUSES:
            raise ValueError(
                f"invalid mapping status for {source_intent}: {mapping_status}"
            )
        if entry.get("sentinelvoice_intent") in PROTECTED_ACTION_INTENTS:
            raise ValueError(
                "frozen finance mapping cannot assign a protected-action intent: "
                f"{source_intent}"
            )
        by_source_intent[source_intent] = entry
        status_counts[mapping_status] += 1

    if len(by_source_intent) != EXPECTED_FINANCE_INTENT_COUNT:
        raise ValueError("frozen mapping must contain exactly 30 finance intents")
    if mapping.get("finance_intent_count") != EXPECTED_FINANCE_INTENT_COUNT:
        raise ValueError("frozen mapping finance_intent_count must be 30")
    expected_counts = mapping.get("summary", {}).get("mapping_status_counts")
    if dict(status_counts) != expected_counts:
        raise ValueError("frozen mapping status counts do not match its summary")
    return by_source_intent


def load_test_rows(raw_bytes: bytes) -> list[list[str]]:
    try:
        source = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw data_full.json must be valid UTF-8 JSON") from exc
    if not isinstance(source, dict):
        raise TypeError("raw data_full.json must contain a JSON object")

    rows = source.get(EXPECTED_SOURCE_SPLIT)
    if not isinstance(rows, list):
        raise TypeError("raw test split must be a list")
    if len(rows) != EXPECTED_SOURCE_SPLIT_COUNT:
        raise ValueError("raw test split must contain 4500 rows")
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"raw test row {row_index} must be [text, label]")
        text, label = row
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"raw test row {row_index} has empty text")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"raw test row {row_index} has empty label")
    return rows


def select_finance_rows(
    rows: list[list[str]], mappings: dict[str, dict[str, Any]]
) -> list[tuple[int, str, str]]:
    selected = [
        (row_index, text, source_intent)
        for row_index, (text, source_intent) in enumerate(rows, start=1)
        if source_intent in mappings
    ]
    if len(selected) != EXPECTED_FINANCE_EXAMPLE_COUNT:
        raise ValueError("raw test split must contain 900 finance examples")

    counts = Counter(source_intent for _, _, source_intent in selected)
    if set(counts) != set(mappings):
        raise ValueError("not every frozen finance intent appears in the test split")
    invalid_counts = {
        intent: count
        for intent, count in counts.items()
        if count != EXPECTED_EXAMPLES_PER_INTENT
    }
    if invalid_counts:
        raise ValueError(
            "each finance intent must have 30 test examples: "
            f"{dict(sorted(invalid_counts.items()))}"
        )

    source_rows = [(text, source_intent) for _, text, source_intent in selected]
    if len(source_rows) != len(set(source_rows)):
        raise ValueError("selected finance test rows contain duplicates")
    return selected


def partition_for_status(mapping_status: str) -> str:
    return {
        "EXACT_MATCH": "exact_match",
        "UNSUPPORTED": "unsupported",
        "NEAR_MATCH": "near_match_review",
        "AMBIGUOUS": "ambiguous_review",
    }[mapping_status]


def example_id(source_revision: str, row_index: int) -> str:
    return f"clinc:{source_revision}:test:{row_index:06d}"


def build_records(
    rows: list[tuple[int, str, str]],
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

    for row_index, text, source_intent in rows:
        mapping = mappings[source_intent]
        mapping_status = mapping["mapping_status"]
        partition = partition_for_status(mapping_status)
        record_id = example_id(source_revision, row_index)
        if record_id in seen_ids:
            raise ValueError(f"duplicate generated example_id: {record_id}")
        seen_ids.add(record_id)
        partition_counts[partition] += 1

        provenance = {
            "example_id": record_id,
            "mapping_status": mapping_status,
            "source_domain": mapping["source_domain"],
            "source_intent": source_intent,
            "source_revision": source_revision,
            "source_row_index": row_index,
            "source_split": EXPECTED_SOURCE_SPLIT,
            "text": text,
        }

        if mapping_status == "EXACT_MATCH":
            expected_intent = mapping.get("sentinelvoice_intent")
            if not isinstance(expected_intent, str) or not expected_intent:
                raise ValueError(
                    f"EXACT_MATCH {source_intent} needs a frozen target intent"
                )
            exact_match.append(
                {
                    **provenance,
                    "expected_sentinelvoice_intent": expected_intent,
                }
            )
        elif mapping_status == "UNSUPPORTED":
            if mapping.get("sentinelvoice_intent") != "unsupported_or_uncertain":
                raise ValueError(
                    f"UNSUPPORTED {source_intent} must map to "
                    "unsupported_or_uncertain"
                )
            unsupported.append(
                {
                    **provenance,
                    "expected_sentinelvoice_intent": "unsupported_or_uncertain",
                }
            )
        else:
            review_pool.append(
                {
                    **provenance,
                    "candidate_sentinelvoice_intent": mapping.get(
                        "sentinelvoice_intent"
                    ),
                    "protected_action_risk": mapping["protected_action_risk"],
                    "rationale": mapping["rationale"],
                    "requires_utterance_review": mapping[
                        "requires_utterance_review"
                    ],
                }
            )

    all_records = [*exact_match, *unsupported, *review_pool]
    if len(seen_ids) != EXPECTED_FINANCE_EXAMPLE_COUNT:
        raise ValueError("not every finance test row received one unique example_id")
    if len(all_records) != EXPECTED_FINANCE_EXAMPLE_COUNT:
        raise ValueError("finance rows were lost or duplicated across output partitions")
    if any("expected_sentinelvoice_intent" in row for row in review_pool):
        raise ValueError("review-pool rows cannot have forced expected labels")
    return exact_match, unsupported, review_pool, dict(partition_counts)


def source_intent_coverage(
    rows: list[tuple[int, str, str]],
    mappings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts = Counter(source_intent for _, _, source_intent in rows)
    return {
        source_intent: {
            "candidate_sentinelvoice_intent": mapping.get(
                "sentinelvoice_intent"
            ),
            "example_count": counts[source_intent],
            "mapping_status": mapping["mapping_status"],
            "output_partition": partition_for_status(mapping["mapping_status"]),
            "source_domain": mapping["source_domain"],
        }
        for source_intent, mapping in mappings.items()
    }


def sentinelvoice_intent_coverage(
    exact_match: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    review_pool: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    evaluation = Counter(
        row["expected_sentinelvoice_intent"]
        for row in [*exact_match, *unsupported]
    )
    review = Counter(
        row["candidate_sentinelvoice_intent"] or "unmapped"
        for row in review_pool
    )
    return {
        "scored_expected_intents": dict(sorted(evaluation.items())),
        "review_pool_candidate_intents": dict(sorted(review.items())),
    }


def build_artifacts() -> tuple[bytes, bytes, bytes]:
    raw_bytes = RAW_DATA_PATH.read_bytes()
    mapping_bytes = MAPPING_PATH.read_bytes()
    raw_manifest = load_json_object(RAW_MANIFEST_PATH)
    mapping = load_json_object(MAPPING_PATH)
    source_revision, raw_hash = validate_source_metadata(
        mapping, raw_manifest, raw_bytes
    )
    mappings = load_frozen_mappings(mapping)
    test_rows = load_test_rows(raw_bytes)
    finance_rows = select_finance_rows(test_rows, mappings)
    exact_match, unsupported, review_pool, partition_counts = build_records(
        finance_rows, mappings, source_revision
    )

    evaluation_payload = {
        "dataset_version": DATASET_VERSION,
        "lanes": {
            "exact_match": {
                "description": (
                    "Scored CLINC finance examples using only frozen "
                    "EXACT_MATCH mappings."
                ),
                "example_count": len(exact_match),
                "examples": exact_match,
            },
            "unsupported": {
                "description": (
                    "Scored CLINC finance examples using only frozen "
                    "UNSUPPORTED mappings."
                ),
                "example_count": len(unsupported),
                "examples": unsupported,
            },
        },
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_split": EXPECTED_SOURCE_SPLIT,
    }
    review_payload = {
        "dataset_version": REVIEW_POOL_VERSION,
        "description": (
            "Unscored CLINC finance NEAR_MATCH and AMBIGUOUS examples requiring "
            "utterance-level review."
        ),
        "example_count": len(review_pool),
        "examples": review_pool,
        "schema_version": REVIEW_POOL_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_split": EXPECTED_SOURCE_SPLIT,
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
            "total_finance_test_examples": len(finance_rows),
            "total_review_pool": len(review_pool),
            "unsupported": partition_counts.get("unsupported", 0),
        },
        "coverage_by_sentinelvoice_intent": coverage,
        "coverage_by_source_intent": source_intent_coverage(
            finance_rows, mappings
        ),
        "excluded_source_data": {
            "non_finance_test_examples_used": False,
            "oos_test_used": False,
            "oos_train_used": False,
            "oos_val_used": False,
            "train_used": False,
            "val_used": False,
        },
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
        "raw_data_full_sha256": raw_hash,
        "raw_manifest_sha256": sha256_bytes(RAW_MANIFEST_PATH.read_bytes()),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_split_count": len(test_rows),
        "source_split_used": EXPECTED_SOURCE_SPLIT,
        "validation": {
            "all_30_finance_intents_represented": True,
            "all_finance_intents_have_30_examples": True,
            "all_selected_rows_partitioned_once": True,
            "mapping_statuses_copied_unchanged": True,
            "no_duplicate_example_ids": True,
            "no_duplicate_selected_source_rows": True,
            "no_protected_action_labels_invented": True,
            "no_review_pool_expected_labels": True,
            "only_finance_test_examples_used": True,
        },
    }
    return evaluation_bytes, review_bytes, stable_json_bytes(manifest_payload)


def main() -> None:
    evaluation_bytes, review_bytes, manifest_bytes = build_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_bytes(evaluation_bytes)
    REVIEW_POOL_PATH.write_bytes(review_bytes)
    MANIFEST_PATH.write_bytes(manifest_bytes)

    manifest = json.loads(manifest_bytes)
    counts = manifest["counts"]
    print(
        "Built CLINC finance external evaluation artifacts: "
        f"total={counts['total_finance_test_examples']}, "
        f"exact_match={counts['exact_match']}, "
        f"unsupported={counts['unsupported']}, "
        f"near_match_review={counts['near_match_review']}, "
        f"ambiguous_review={counts['ambiguous_review']}"
    )


if __name__ == "__main__":
    main()
