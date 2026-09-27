"""Build the deterministic CLINC oos_test-only external evaluation dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
RAW_DATA_DIR = EXTERNAL_DATA_ROOT / "raw/clinc_oos"
RAW_DATA_PATH = RAW_DATA_DIR / "data_full.json"
RAW_MANIFEST_PATH = RAW_DATA_DIR / "manifest.json"
OUTPUT_DIR = EXTERNAL_DATA_ROOT / "processed/clinc_oos"
EVALUATION_PATH = OUTPUT_DIR / "clinc_oos_external_eval.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

BUILDER_VERSION = "clinc-oos-external-eval-builder.v1"
DATASET_SCHEMA_VERSION = "clinc-oos-external-eval.v1"
MANIFEST_SCHEMA_VERSION = "clinc-oos-processed-manifest.v1"
DATASET_VERSION = "clinc-oos-external-eval.2026-09-27.v1"
EXPECTED_SOURCE_ID = "clinc150_oos"
EXPECTED_SOURCE_SPLIT = "oos_test"
EXPECTED_SOURCE_LABEL = "oos"
EXPECTED_SENTINELVOICE_INTENT = "unsupported_or_uncertain"
EXPECTED_EXAMPLE_COUNT = 1_000


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
    raw_manifest: dict[str, Any], raw_bytes: bytes
) -> tuple[str, str]:
    if raw_manifest.get("source_id") != EXPECTED_SOURCE_ID:
        raise ValueError("raw manifest source_id must be clinc150_oos")
    source_revision = raw_manifest.get("pinned_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("raw manifest pinned_revision must be present")

    metadata = raw_file_metadata(raw_manifest)
    raw_hash = sha256_bytes(raw_bytes)
    if metadata.get("sha256") != raw_hash:
        raise ValueError("raw data_full.json SHA-256 differs from its manifest")
    if metadata.get("byte_size") != len(raw_bytes):
        raise ValueError("raw data_full.json byte size differs from its manifest")
    if raw_manifest.get("oos_counts_by_split", {}).get(EXPECTED_SOURCE_SPLIT) != (
        EXPECTED_EXAMPLE_COUNT
    ):
        raise ValueError("raw manifest oos_test count must be 1000")
    return source_revision, raw_hash


def load_oos_test_rows(raw_bytes: bytes) -> list[list[str]]:
    try:
        source = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw data_full.json must be valid UTF-8 JSON") from exc
    if not isinstance(source, dict):
        raise TypeError("raw data_full.json must contain a JSON object")
    if EXPECTED_SOURCE_SPLIT not in source:
        raise ValueError("raw data_full.json does not contain oos_test")

    rows = source[EXPECTED_SOURCE_SPLIT]
    if not isinstance(rows, list):
        raise TypeError("raw oos_test split must be a list")
    if len(rows) != EXPECTED_EXAMPLE_COUNT:
        raise ValueError(f"raw oos_test must contain {EXPECTED_EXAMPLE_COUNT} rows")

    seen_rows: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"raw oos_test row {row_index} must be [text, label]")
        text, label = row
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"raw oos_test row {row_index} has empty text")
        if label != EXPECTED_SOURCE_LABEL:
            raise ValueError(
                f"raw oos_test row {row_index} label must be {EXPECTED_SOURCE_LABEL}"
            )
        source_row = (text, label)
        if source_row in seen_rows:
            raise ValueError(f"duplicate raw oos_test row at index {row_index}")
        seen_rows.add(source_row)
    return rows


def build_examples(
    rows: list[list[str]], source_revision: str
) -> list[dict[str, Any]]:
    examples = [
        {
            "example_id": f"clinc_oos:oos_test:{row_index:06d}",
            "expected_sentinelvoice_intent": EXPECTED_SENTINELVOICE_INTENT,
            "source_label": source_label,
            "source_revision": source_revision,
            "source_row_index": row_index,
            "source_split": EXPECTED_SOURCE_SPLIT,
            "text": text,
        }
        for row_index, (text, source_label) in enumerate(rows, start=1)
    ]
    example_ids = [example["example_id"] for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("builder produced duplicate example IDs")
    if len(examples) != len(rows):
        raise ValueError("builder lost or duplicated source rows")
    return examples


def build_artifacts() -> tuple[bytes, bytes]:
    raw_bytes = RAW_DATA_PATH.read_bytes()
    raw_manifest = load_json_object(RAW_MANIFEST_PATH)
    source_revision, raw_hash = validate_source_metadata(raw_manifest, raw_bytes)
    rows = load_oos_test_rows(raw_bytes)
    examples = build_examples(rows, source_revision)

    dataset_payload = {
        "dataset_version": DATASET_VERSION,
        "description": (
            "Official CLINC oos_test examples for evaluating whether SentinelVoice "
            "rejects unrelated, non-banking requests."
        ),
        "example_count": len(examples),
        "examples": examples,
        "expected_sentinelvoice_intent": EXPECTED_SENTINELVOICE_INTENT,
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_revision": source_revision,
        "source_split": EXPECTED_SOURCE_SPLIT,
    }
    dataset_bytes = stable_json_bytes(dataset_payload)
    manifest_payload = {
        "build_metadata": {
            "builder_version": BUILDER_VERSION,
            "deterministic": True,
            "source_retrieval_date": raw_manifest["retrieval_date"],
            "wall_clock_timestamp_recorded": False,
        },
        "build_script_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "excluded_source_data": {
            "clinc_in_scope_intents_used": False,
            "oos_train_used": False,
            "oos_val_used": False,
        },
        "expected_sentinelvoice_intent": EXPECTED_SENTINELVOICE_INTENT,
        "pinned_source_revision": source_revision,
        "processed_example_count": len(examples),
        "processed_files": {
            EVALUATION_PATH.name: {
                "byte_size": len(dataset_bytes),
                "sha256": sha256_bytes(dataset_bytes),
            }
        },
        "raw_data_full_sha256": raw_hash,
        "raw_manifest_sha256": sha256_bytes(RAW_MANIFEST_PATH.read_bytes()),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": EXPECTED_SOURCE_ID,
        "source_split_count": len(rows),
        "source_split_used": EXPECTED_SOURCE_SPLIT,
        "validation": {
            "all_expected_labels_are_unsupported_or_uncertain": True,
            "all_source_labels_are_oos": True,
            "all_texts_non_empty": True,
            "no_duplicate_example_ids": True,
            "no_duplicate_source_rows": True,
            "only_oos_test_used": True,
        },
    }
    return dataset_bytes, stable_json_bytes(manifest_payload)


def main() -> None:
    dataset_bytes, manifest_bytes = build_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_bytes(dataset_bytes)
    MANIFEST_PATH.write_bytes(manifest_bytes)

    manifest = json.loads(manifest_bytes)
    print(
        "Built CLINC OOS external evaluation artifact: "
        f"source_split={manifest['source_split_used']}, "
        f"examples={manifest['processed_example_count']}"
    )


if __name__ == "__main__":
    main()
