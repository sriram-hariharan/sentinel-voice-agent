"""Build the frozen, synthetic V2-C3 final challenge evaluation set."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts import build_v2c3_development_dataset as development_builder
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import build_v2c3_development_dataset as development_builder


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"

BUILDER_VERSION = "v2c3-challenge-set-builder.v1"
SEED_SCHEMA_VERSION = "v2c3-challenge-seed.v1"
DATASET_SCHEMA_VERSION = "v2c3-challenge-set.v1"
MANIFEST_SCHEMA_VERSION = "v2c3-challenge-set-manifest.v1"
CHALLENGE_VERSION = "2026-09-28.v2c3-challenge.v1"
SOURCE_ID = "sentinelvoice_v2c3_synthetic_challenge"
DATA_ROLE = "final_challenge_evaluation"
EXAMPLES_PER_INTENT = 30
EXPECTED_TOTAL = 270
PROTECTED_WRITE_INTENTS = {"create_dispute", "freeze_card"}
REQUIRED_HARD_NEGATIVE_TAGS = {
    "historical_dispute",
    "lost_stolen_no_action",
    "prior_frozen_card",
    "unclear_no_current_request",
    "unfamiliar_transaction_no_dispute",
    "unsupported_operation",
    "vague_complaint",
}


@dataclass(frozen=True)
class BuildPaths:
    seed: Path
    contract: Path
    development_dataset: Path
    fresh_lockbox_manifest: Path
    internal_dataset: Path
    banking77_test: Path
    clinc_data: Path
    challenge_output: Path
    manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    seed=ML_ROOT / "v2c3_challenge_seed.json",
    contract=ML_ROOT / "v2c3_experiment_contract.json",
    development_dataset=ML_ROOT / "v2c3_development_dataset.json",
    fresh_lockbox_manifest=ML_ROOT / "v2c3_fresh_lockbox_manifest.json",
    internal_dataset=ML_ROOT / "intent_risk_dataset.json",
    banking77_test=EXTERNAL_ROOT / "raw/banking77/test.csv",
    clinc_data=EXTERNAL_ROOT / "raw/clinc_oos/data_full.json",
    challenge_output=ML_ROOT / "v2c3_challenge_set.json",
    manifest_output=ML_ROOT / "v2c3_challenge_set.manifest.json",
)


@dataclass(frozen=True)
class BuiltArtifacts:
    challenge_bytes: bytes
    manifest_bytes: bytes
    challenge_payload: dict[str, Any]
    manifest_payload: dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
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


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _reference_hashes(paths: BuildPaths) -> tuple[set[str], dict[str, int]]:
    development = _load_json_object(paths.development_dataset)
    lockbox = _load_json_object(paths.fresh_lockbox_manifest)
    internal = _load_json_object(paths.internal_dataset)
    clinc = _load_json_object(paths.clinc_data)

    development_hashes = {
        row["normalized_text_sha256"] for row in development.get("examples", [])
    }
    lockbox_hashes = {
        row["normalized_text_sha256"] for row in lockbox.get("members", [])
    }
    internal_locked_hashes = {
        development_builder.normalized_text_sha256(row["text"])
        for row in internal.get("examples", [])
        if row.get("split") == "locked_test"
    }

    banking_rows = list(
        csv.DictReader(io.StringIO(paths.banking77_test.read_text(encoding="utf-8")))
    )
    banking_test_hashes = {
        development_builder.normalized_text_sha256(row["text"])
        for row in banking_rows
    }
    clinc_test_rows = [*clinc.get("test", []), *clinc.get("oos_test", [])]
    clinc_test_hashes = {
        development_builder.normalized_text_sha256(row[0])
        for row in clinc_test_rows
    }

    named = {
        "development": development_hashes,
        "external_lockbox": lockbox_hashes,
        "internal_locked_test": internal_locked_hashes,
        "banking77_test": banking_test_hashes,
        "clinc_test_and_oos_test": clinc_test_hashes,
    }
    all_hashes: set[str] = set()
    for values in named.values():
        all_hashes.update(values)
    return all_hashes, {name: len(values) for name, values in named.items()}


def _validate_contract(contract: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    if contract.get("schema_version") != "v2c3-experiment-contract.v1":
        raise ValueError("unexpected V2-C3 experiment-contract schema")
    intents = contract.get("intent_taxonomy")
    risks = contract.get("risk_taxonomy")
    risk_by_intent = contract.get("risk_by_intent")
    if not isinstance(intents, list) or len(intents) != 9:
        raise ValueError("contract must freeze exactly nine intents")
    if intents != sorted(intents) or len(intents) != len(set(intents)):
        raise ValueError("contract intents must be unique and sorted")
    if not isinstance(risks, list) or len(risks) != 4:
        raise ValueError("contract must freeze exactly four risks")
    if not isinstance(risk_by_intent, dict) or set(risk_by_intent) != set(intents):
        raise ValueError("contract intent-to-risk mapping is incomplete")
    if set(risk_by_intent.values()) != set(risks):
        raise ValueError("contract intent-to-risk mapping differs from taxonomy")
    return intents, risk_by_intent


def _validate_source_policy(seed: Mapping[str, Any]) -> None:
    policy = seed.get("source_policy")
    if not isinstance(policy, dict):
        raise TypeError("challenge seed source_policy must be an object")
    false_fields = {
        "cfpb_used",
        "consumer_authored_text_used",
        "model_selection_eligible",
        "threshold_selection_eligible",
        "training_eligible",
    }
    if any(policy.get(field) is not False for field in false_fields):
        raise ValueError("challenge seed violates final-only synthetic-source policy")
    _require_string(policy.get("authorship"), "source_policy.authorship")


def _validate_protected_write(record: Mapping[str, Any]) -> None:
    intent = record["intent"]
    tags = set(record["tags"])
    text = record["text"].lower()
    if intent in PROTECTED_WRITE_INTENTS:
        required = {"explicit_current_action", "protected_write_positive"}
        if not required <= tags:
            raise ValueError(
                f"protected-write example lacks action tags: {record['example_id']}"
            )
        if intent == "freeze_card":
            card_context = re.search(r"\b(card|it)\b", text)
            action = re.search(
                r"\b(freeze|lock|block|disable)\b|shut off|turn .* off|"
                r"stop purchases|make .* unusable",
                text,
            )
            if not card_context or not action:
                raise ValueError(
                    f"freeze-card example lacks explicit current action: "
                    f"{record['example_id']}"
                )
        elif not re.search(r"\b(disput\w*|contest\w*|challeng\w*|claim)\b", text):
            raise ValueError(
                f"create-dispute example lacks explicit current action: "
                f"{record['example_id']}"
            )
    elif "protected_write_positive" in tags or "explicit_current_action" in tags:
        raise ValueError(
            f"non-protected example carries protected-action tags: "
            f"{record['example_id']}"
        )


def _build_records(
    seed: Mapping[str, Any],
    intents: Sequence[str],
    risk_by_intent: Mapping[str, str],
) -> list[dict[str, Any]]:
    if seed.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError("unexpected V2-C3 challenge-seed schema")
    if seed.get("challenge_version") != CHALLENGE_VERSION:
        raise ValueError("unexpected V2-C3 challenge version")
    _validate_source_policy(seed)
    by_intent = seed.get("examples_by_intent")
    if not isinstance(by_intent, dict) or set(by_intent) != set(intents):
        raise ValueError("challenge seed must contain exactly the frozen intents")

    records: list[dict[str, Any]] = []
    seen_lineage_intent: set[tuple[str, str]] = set()
    for intent in intents:
        examples = by_intent[intent]
        if not isinstance(examples, list) or len(examples) != EXAMPLES_PER_INTENT:
            raise ValueError(
                f"challenge intent {intent} must contain {EXAMPLES_PER_INTENT} examples"
            )
        for index, source in enumerate(examples, start=1):
            if not isinstance(source, dict):
                raise TypeError(f"challenge seed {intent}[{index}] must be an object")
            lineage_id = _require_string(
                source.get("lineage_id"), f"{intent}[{index}].lineage_id"
            )
            lineage_key = (intent, lineage_id)
            if lineage_key in seen_lineage_intent:
                raise ValueError(f"duplicate lineage within intent: {lineage_key}")
            seen_lineage_intent.add(lineage_key)
            text = _require_string(source.get("text"), f"{intent}[{index}].text")
            source_tags = source.get("tags")
            if not isinstance(source_tags, list) or not source_tags:
                raise ValueError(f"{intent}[{index}].tags must be a non-empty list")
            tags = sorted(
                {_require_string(tag, "challenge tag") for tag in source_tags}
            )
            if len(tags) != len(source_tags):
                raise ValueError(f"duplicate challenge tag in {intent}[{index}]")
            record = {
                "data_role": DATA_ROLE,
                "example_id": f"v2c3-challenge:{intent}:{index:03d}",
                "group_id": f"v2c3-challenge:{lineage_id}",
                "intent": intent,
                "lineage_id": lineage_id,
                "model_selection_eligible": False,
                "normalized_text_sha256": (
                    development_builder.normalized_text_sha256(text)
                ),
                "risk": risk_by_intent[intent],
                "source_id": SOURCE_ID,
                "tags": tags,
                "text": text,
                "text_sha256": development_builder.text_sha256(text),
                "threshold_selection_eligible": False,
                "training_eligible": False,
            }
            _validate_protected_write(record)
            records.append(record)

    if len(records) != EXPECTED_TOTAL:
        raise ValueError(f"challenge set must contain exactly {EXPECTED_TOTAL} records")
    normalized_hashes = [row["normalized_text_sha256"] for row in records]
    if len(normalized_hashes) != len(set(normalized_hashes)):
        raise ValueError("challenge set contains normalized-text duplicates")

    tag_counts = Counter(tag for row in records for tag in row["tags"])
    missing_tags = REQUIRED_HARD_NEGATIVE_TAGS - set(tag_counts)
    if missing_tags:
        raise ValueError(
            f"challenge seed lacks hard-negative tags: {sorted(missing_tags)}"
        )

    boundary_families: dict[str, set[str]] = {}
    for row in records:
        if "boundary_pair" in row["tags"]:
            boundary_families.setdefault(row["lineage_id"], set()).add(row["intent"])
    if not boundary_families or any(
        len(family_intents) < 2 for family_intents in boundary_families.values()
    ):
        raise ValueError(
            "every boundary-pair family must contrast at least two intents"
        )
    return records


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    script_path: Path | None = None,
) -> BuiltArtifacts:
    seed = _load_json_object(paths.seed)
    contract = _load_json_object(paths.contract)
    intents, risk_by_intent = _validate_contract(contract)
    records = _build_records(seed, intents, risk_by_intent)

    reference_hashes, reference_counts = _reference_hashes(paths)
    challenge_hashes = {row["normalized_text_sha256"] for row in records}
    overlap = challenge_hashes & reference_hashes
    if overlap:
        raise ValueError(
            "challenge text overlaps frozen development/evaluation data: "
            f"{len(overlap)}"
        )

    challenge_payload = {
        "advisory_only": True,
        "challenge_version": CHALLENGE_VERSION,
        "data_role": DATA_ROLE,
        "evaluation_timing": (
            "after model, representation, hyperparameters, and thresholds are frozen"
        ),
        "example_count": len(records),
        "examples": records,
        "model_selection_eligible": False,
        "normalization_version": development_builder.NORMALIZATION_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "threshold_selection_eligible": False,
        "training_eligible": False,
        "training_or_evaluation_performed": False,
    }
    challenge_bytes = development_builder.stable_json_bytes(challenge_payload)

    counts_by_intent = Counter(row["intent"] for row in records)
    counts_by_risk = Counter(row["risk"] for row in records)
    counts_by_tag = Counter(tag for row in records for tag in row["tags"])
    boundary_lineages = {
        row["lineage_id"] for row in records if "boundary_pair" in row["tags"]
    }
    input_paths = {
        "banking77_test": paths.banking77_test,
        "clinc_data": paths.clinc_data,
        "contract": paths.contract,
        "development_dataset": paths.development_dataset,
        "fresh_lockbox_manifest": paths.fresh_lockbox_manifest,
        "internal_dataset": paths.internal_dataset,
        "seed": paths.seed,
    }
    manifest_payload = {
        "build_metadata": {
            "builder_version": BUILDER_VERSION,
            "deterministic": True,
            "wall_clock_timestamp_recorded": False,
        },
        "build_script_sha256": development_builder.sha256_bytes(
            (script_path or Path(__file__).resolve()).read_bytes()
        ),
        "challenge_version": CHALLENGE_VERSION,
        "contract_sha256": development_builder.sha256_bytes(
            paths.contract.read_bytes()
        ),
        "counts": {
            "boundary_pair_family_count": len(boundary_lineages),
            "example_count": len(records),
            "group_count": len({row["group_id"] for row in records}),
        },
        "counts_by_intent": dict(sorted(counts_by_intent.items())),
        "counts_by_risk": dict(sorted(counts_by_risk.items())),
        "counts_by_tag": dict(sorted(counts_by_tag.items())),
        "evaluation_policy": {
            "data_role": DATA_ROLE,
            "evaluation_timing": (
                "after model, representation, hyperparameters, and thresholds "
                "are frozen"
            ),
            "model_selection_eligible": False,
            "threshold_selection_eligible": False,
            "training_eligible": False,
        },
        "input_hashes": {
            name: development_builder.sha256_bytes(path.read_bytes())
            for name, path in sorted(input_paths.items())
        },
        "input_paths": {
            name: _display_path(path) for name, path in sorted(input_paths.items())
        },
        "normalization": {
            "hash_algorithm": "SHA-256",
            "steps": development_builder.NORMALIZATION_STEPS,
            "version": development_builder.NORMALIZATION_VERSION,
        },
        "output_hashes": {
            "v2c3_challenge_set.json": development_builder.sha256_bytes(
                challenge_bytes
            )
        },
        "reference_normalized_hash_counts": reference_counts,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed_sha256": development_builder.sha256_bytes(paths.seed.read_bytes()),
        "source_policy": seed["source_policy"],
        "training_or_evaluation_performed": False,
        "validation": {
            "balanced_30_per_intent": all(
                counts_by_intent[intent] == EXAMPLES_PER_INTENT
                for intent in intents
            ),
            "banking77_test_normalized_hash_overlap": 0,
            "cfpb_used": False,
            "challenge_normalized_duplicate_count": 0,
            "clinc_test_and_oos_test_normalized_hash_overlap": 0,
            "development_normalized_hash_overlap": 0,
            "external_lockbox_normalized_hash_overlap": 0,
            "internal_locked_test_normalized_hash_overlap": 0,
            "protected_write_semantic_review": (
                "Seed positives are manually authored and tagged; deterministic "
                "lexical guards supplement but do not replace semantic review."
            ),
            "risk_derived_from_frozen_contract": True,
            "taxonomy_matches_frozen_contract": True,
        },
    }
    manifest_bytes = development_builder.stable_json_bytes(manifest_payload)
    return BuiltArtifacts(
        challenge_bytes=challenge_bytes,
        manifest_bytes=manifest_bytes,
        challenge_payload=challenge_payload,
        manifest_payload=manifest_payload,
    )


def write_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    development_builder.atomic_write_bytes(
        paths.challenge_output, artifacts.challenge_bytes
    )
    development_builder.atomic_write_bytes(
        paths.manifest_output, artifacts.manifest_bytes
    )


def check_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    expected = {
        paths.challenge_output: artifacts.challenge_bytes,
        paths.manifest_output: artifacts.manifest_bytes,
    }
    for path, payload in expected.items():
        if not path.exists():
            raise FileNotFoundError(
                f"V2-C3 challenge artifact is missing: {path}; use --write"
            )
        if path.read_bytes() != payload:
            raise ValueError(
                f"V2-C3 challenge artifact differs from deterministic build: {path}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write challenge artifacts")
    mode.add_argument("--check", action="store_true", help="verify challenge artifacts")
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
    print(
        f"{action} V2-C3 synthetic challenge artifacts: "
        f"examples={artifacts.challenge_payload['example_count']}, "
        "training/model/threshold selection eligible=false; "
        "no training or evaluation performed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
