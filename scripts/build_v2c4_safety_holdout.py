"""Build the frozen, synthetic V2-C4 final safety holdout."""

from __future__ import annotations

import argparse
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

BUILDER_VERSION = "v2c4-safety-holdout-builder.v1"
SEED_SCHEMA_VERSION = "v2c4-safety-holdout-seed.v1"
DATASET_SCHEMA_VERSION = "v2c4-safety-holdout.v1"
MANIFEST_SCHEMA_VERSION = "v2c4-safety-holdout-manifest.v1"
HOLDOUT_VERSION = "2026-09-29.v2c4-safety-holdout.v1"
SOURCE_ID = "sentinelvoice_v2c4_synthetic_safety_holdout"
DATA_ROLE = "v2c4_final_safety_holdout"
EXAMPLES_PER_INTENT = 40
EXPECTED_TOTAL = 360
PROTECTED_WRITE_INTENTS = {"create_dispute", "freeze_card"}
SAFETY_SENSITIVE_INTENTS = {
    "create_dispute",
    "freeze_card",
    "unsupported_or_uncertain",
}
REQUIRED_SAFETY_PATTERNS = {
    "explicit_current_action_request",
    "information_only_action_mention",
    "information_then_current_action",
    "past_tense_description",
    "hypothetical_request",
    "conditional_request",
    "negation_or_cancellation",
    "correction_or_change_of_mind",
    "ambiguous_reference",
    "adjacent_intent_wording",
    "incomplete_request",
}
PROTECTED_MENTION_HARD_NEGATIVE = "protected_mention_hard_negative"


@dataclass(frozen=True)
class BuildPaths:
    seed: Path
    contract: Path
    error_analysis_config: Path
    v2c3_development: Path
    v2c3_challenge: Path
    v2c3_challenge_manifest: Path
    v2c3_external_lockbox_manifest: Path
    holdout_output: Path
    manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    seed=ML_ROOT / "v2c4_safety_holdout_seed.json",
    contract=ML_ROOT / "v2c4_experiment_contract.json",
    error_analysis_config=ML_ROOT / "v2c4_error_analysis_config.json",
    v2c3_development=ML_ROOT / "v2c3_development_dataset.json",
    v2c3_challenge=ML_ROOT / "v2c3_challenge_set.json",
    v2c3_challenge_manifest=ML_ROOT / "v2c3_challenge_set.manifest.json",
    v2c3_external_lockbox_manifest=ML_ROOT / "v2c3_fresh_lockbox_manifest.json",
    holdout_output=ML_ROOT / "v2c4_safety_holdout.json",
    manifest_output=ML_ROOT / "v2c4_safety_holdout.manifest.json",
)


@dataclass(frozen=True)
class ReferenceMetadata:
    normalized_hashes: set[str]
    lineage_ids: set[str]
    hash_counts: dict[str, int]


@dataclass(frozen=True)
class BuiltArtifacts:
    holdout_bytes: bytes
    manifest_bytes: bytes
    holdout_payload: dict[str, Any]
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


def _validate_file_hash(path: Path, expected: str, label: str) -> None:
    actual = development_builder.sha256_bytes(path.read_bytes())
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _validate_contract(
    contract: Mapping[str, Any], paths: BuildPaths
) -> tuple[list[str], dict[str, str]]:
    if contract.get("schema_version") != "v2c4-experiment-contract.v1":
        raise ValueError("unexpected V2-C4 experiment-contract schema")
    if contract.get("contract_frozen_before_error_analysis") is not True:
        raise ValueError("V2-C4 contract must precede error analysis")
    if contract.get("error_analysis_performed") is not False:
        raise ValueError("V2-C4 error analysis must not precede holdout creation")
    if contract.get("training_or_evaluation_performed") is not False:
        raise ValueError("V2-C4 holdout must precede training and evaluation")
    if contract.get("runtime_authority") is not False:
        raise ValueError("V2-C4 contract cannot grant runtime authority")

    intents = contract.get("intent_taxonomy")
    risk_by_intent = contract.get("risk_by_intent")
    if not isinstance(intents, list) or len(intents) != 9:
        raise ValueError("V2-C4 must retain exactly nine intents")
    if intents != sorted(intents) or len(intents) != len(set(intents)):
        raise ValueError("V2-C4 intents must be unique and sorted")
    if not isinstance(risk_by_intent, dict) or set(risk_by_intent) != set(intents):
        raise ValueError("V2-C4 intent-to-risk mapping is incomplete")

    gates = contract.get("acceptance_gates")
    if gates != {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "all_gates_required": True,
        "post_hoc_relaxation_allowed": False,
        "macro_f1_or_accuracy_gate": False,
    }:
        raise ValueError("V2-C4 safety gates changed")

    consumed = contract.get("consumed_data_policy", {})
    if consumed.get("v2c3_challenge_set", {}).get("diagnostic_role") != (
        "consumed_v2c3_safety_challenge"
    ):
        raise ValueError("V2-C3 challenge is not marked consumed")
    if consumed.get("v2c3_external_lockbox", {}).get("diagnostic_role") != (
        "consumed_v2c3_external_regression"
    ):
        raise ValueError("V2-C3 external lockbox is not marked consumed")
    for policy in consumed.values():
        if (
            policy.get("v2c4_final_acceptance_eligible") is not False
            or policy.get("may_be_described_as_untouched") is not False
        ):
            raise ValueError("consumed V2-C3 data cannot be a fresh V2-C4 holdout")

    holdout_policy = contract.get("fresh_holdout_policy", {})
    expected_holdout_policy = {
        "data_role": DATA_ROLE,
        "expected_example_count": EXPECTED_TOTAL,
        "examples_per_intent": EXAMPLES_PER_INTENT,
        "training_eligible": False,
        "model_selection_eligible": False,
        "threshold_selection_eligible": False,
        "error_analysis_eligible_until_final_evaluation": False,
        "single_final_evaluation": True,
    }
    for field, expected in expected_holdout_policy.items():
        if holdout_policy.get(field) != expected:
            raise ValueError(f"V2-C4 fresh-holdout policy changed: {field}")
    if (
        contract.get("cfpb_policy", {}).get("read_by_v2c4_holdout_builder")
        is not False
    ):
        raise ValueError("CFPB must remain outside the V2-C4 holdout builder")

    parent_paths = {
        spec["path"]: spec["sha256"]
        for spec in contract.get("parent_v2c3_hashes", {}).values()
    }
    required_parent_paths = {
        _display_path(paths.v2c3_development),
        _display_path(paths.v2c3_challenge),
        _display_path(paths.v2c3_challenge_manifest),
        _display_path(paths.v2c3_external_lockbox_manifest),
    }
    if not required_parent_paths <= set(parent_paths):
        raise ValueError("V2-C4 contract omits required V2-C3 parent hashes")
    for relative_path, expected in parent_paths.items():
        _validate_file_hash(
            REPOSITORY_ROOT / relative_path,
            expected,
            f"V2-C3 parent {relative_path}",
        )
    return intents, risk_by_intent


def _validate_error_analysis_config(
    config: Mapping[str, Any], contract_path: Path
) -> None:
    if config.get("schema_version") != "v2c4-error-analysis-config.v1":
        raise ValueError("unexpected V2-C4 error-analysis schema")
    if config.get("frozen_before_error_analysis") is not True:
        raise ValueError("error-analysis categories must be frozen first")
    if config.get("error_analysis_performed") is not False:
        raise ValueError("error analysis has already been marked performed")
    contract_spec = config.get("v2c4_contract", {})
    if contract_spec.get("path") != _display_path(contract_path):
        raise ValueError("error-analysis config points to the wrong contract")
    _validate_file_hash(
        contract_path,
        str(contract_spec.get("sha256")),
        "V2-C4 contract",
    )
    category_ids = [row.get("id") for row in config.get("categories", [])]
    required = {
        "unsupported_to_supported",
        "supported_to_unsupported",
        "protected_to_wrong_protected_intent",
        "protected_to_non_protected",
        "non_protected_to_protected",
        "freeze_card_create_dispute_confusion",
        "informational_mention_vs_action_request",
        "ambiguity_or_insufficient_context",
        "lexical_trigger_over_reliance",
    }
    if set(category_ids) != required or len(category_ids) != len(required):
        raise ValueError("V2-C4 error-analysis categories changed")


def _reference_metadata(paths: BuildPaths) -> ReferenceMetadata:
    development = _load_json_object(paths.v2c3_development)
    challenge = _load_json_object(paths.v2c3_challenge)
    lockbox = _load_json_object(paths.v2c3_external_lockbox_manifest)

    named_hashes = {
        "v2c3_development": {
            row["normalized_text_sha256"]
            for row in development.get("examples", [])
        },
        "consumed_v2c3_challenge": {
            row["normalized_text_sha256"]
            for row in challenge.get("examples", [])
        },
        "consumed_v2c3_external_lockbox": {
            row["normalized_text_sha256"] for row in lockbox.get("members", [])
        },
    }
    all_hashes: set[str] = set()
    for hashes in named_hashes.values():
        all_hashes.update(hashes)
    old_lineages = {
        row["lineage_id"] for row in challenge.get("examples", [])
    }
    return ReferenceMetadata(
        normalized_hashes=all_hashes,
        lineage_ids=old_lineages,
        hash_counts={name: len(values) for name, values in named_hashes.items()},
    )


def _validate_source_policy(seed: Mapping[str, Any]) -> None:
    policy = seed.get("source_policy")
    if not isinstance(policy, dict):
        raise TypeError("V2-C4 seed source_policy must be an object")
    required_false = {
        "automatic_model_labeling_used",
        "cfpb_used",
        "consumer_authored_text_used",
        "model_predictions_used",
        "model_selection_eligible",
        "threshold_selection_eligible",
        "training_eligible",
    }
    if any(policy.get(field) is not False for field in required_false):
        raise ValueError("V2-C4 seed violates synthetic final-only policy")
    _require_string(policy.get("authorship"), "source_policy.authorship")


def _validate_protected_write(record: Mapping[str, Any]) -> None:
    intent = record["intent"]
    tags = set(record["tags"])
    text = record["text"].lower()
    if intent in PROTECTED_WRITE_INTENTS:
        if not {"explicit_current_action", "protected_write_positive"} <= tags:
            raise ValueError(
                f"protected-write example lacks action tags: {record['example_id']}"
            )
        if intent == "freeze_card":
            if not re.search(r"\b(card|it|one)\b", text) or not re.search(
                r"\b(freeze|lock|block|disable)\b", text
            ):
                raise ValueError(
                    f"freeze-card example lacks current action: "
                    f"{record['example_id']}"
                )
        elif not re.search(r"\b(disput\w*|contest\w*|challeng\w*)\b", text):
            raise ValueError(
                f"create-dispute example lacks current action: "
                f"{record['example_id']}"
            )
    elif {"explicit_current_action", "protected_write_positive"} & tags:
        raise ValueError(
            f"non-protected example carries protected-action tags: "
            f"{record['example_id']}"
        )


def _build_records(
    seed: Mapping[str, Any],
    intents: Sequence[str],
    risk_by_intent: Mapping[str, str],
    reference_metadata: ReferenceMetadata,
) -> list[dict[str, Any]]:
    if seed.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError("unexpected V2-C4 safety-holdout seed schema")
    if seed.get("holdout_version") != HOLDOUT_VERSION:
        raise ValueError("unexpected V2-C4 safety-holdout version")
    if seed.get("deterministic_seed") != 20260929:
        raise ValueError("unexpected V2-C4 deterministic seed")
    _validate_source_policy(seed)
    families = seed.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("V2-C4 seed families must be a non-empty list")

    records: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    ordinal_by_intent: Counter[str] = Counter()
    for family_index, family in enumerate(families, start=1):
        if not isinstance(family, dict):
            raise TypeError(f"seed family {family_index} must be an object")
        family_id = _require_string(
            family.get("family_id"), f"families[{family_index}].family_id"
        )
        if family_id in family_ids:
            raise ValueError(f"duplicate V2-C4 family ID: {family_id}")
        family_ids.add(family_id)
        intent = _require_string(
            family.get("intent"), f"families[{family_index}].intent"
        )
        if intent not in intents:
            raise ValueError(f"family intent is outside frozen taxonomy: {intent}")
        safety_pattern = _require_string(
            family.get("safety_pattern"),
            f"families[{family_index}].safety_pattern",
        )
        design_lane = _require_string(
            family.get("design_lane"), f"families[{family_index}].design_lane"
        )
        raw_tags = family.get("tags")
        if not isinstance(raw_tags, list) or not raw_tags:
            raise ValueError(f"family {family_id} tags must be non-empty")
        tags = sorted({_require_string(tag, "holdout tag") for tag in raw_tags})
        if len(tags) != len(raw_tags):
            raise ValueError(f"family {family_id} contains duplicate tags")
        texts = family.get("texts")
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"family {family_id} texts must be non-empty")
        for text_index, raw_text in enumerate(texts, start=1):
            text = _require_string(raw_text, f"family {family_id} text")
            ordinal_by_intent[intent] += 1
            ordinal = ordinal_by_intent[intent]
            lineage_id = f"v2c4-holdout:{family_id}:{text_index:02d}"
            record = {
                "data_role": DATA_ROLE,
                "design_lane": design_lane,
                "error_analysis_eligible": False,
                "example_id": f"v2c4-holdout:{intent}:{ordinal:03d}",
                "group_id": lineage_id,
                "intent": intent,
                "lineage_id": lineage_id,
                "model_selection_eligible": False,
                "normalized_text_sha256": (
                    development_builder.normalized_text_sha256(text)
                ),
                "risk": risk_by_intent[intent],
                "safety_pattern": safety_pattern,
                "source_id": SOURCE_ID,
                "tags": tags,
                "text": text,
                "text_sha256": development_builder.text_sha256(text),
                "threshold_selection_eligible": False,
                "training_eligible": False,
            }
            _validate_protected_write(record)
            records.append(record)

    counts = Counter(row["intent"] for row in records)
    if len(records) != EXPECTED_TOTAL:
        raise ValueError(f"V2-C4 holdout must contain exactly {EXPECTED_TOTAL} rows")
    if set(counts) != set(intents) or any(
        counts[intent] != EXAMPLES_PER_INTENT for intent in intents
    ):
        raise ValueError("V2-C4 holdout must contain exactly 40 rows per intent")

    normalized_hashes = [row["normalized_text_sha256"] for row in records]
    if len(normalized_hashes) != len(set(normalized_hashes)):
        raise ValueError("V2-C4 holdout contains normalized-text duplicates")
    overlap = set(normalized_hashes) & reference_metadata.normalized_hashes
    if overlap:
        raise ValueError(f"V2-C4 holdout overlaps V2-C3 data: {len(overlap)}")

    lineage_ids = [row["lineage_id"] for row in records]
    if len(lineage_ids) != len(set(lineage_ids)):
        raise ValueError("V2-C4 holdout contains duplicate lineage IDs")
    old_lineage_overlap = set(lineage_ids) & reference_metadata.lineage_ids
    if old_lineage_overlap:
        raise ValueError("V2-C4 lineage duplicates V2-C3 challenge lineage")

    sensitive_patterns = {
        row["safety_pattern"]
        for row in records
        if row["intent"] in SAFETY_SENSITIVE_INTENTS
    }
    missing_patterns = REQUIRED_SAFETY_PATTERNS - sensitive_patterns
    if missing_patterns:
        raise ValueError(
            f"V2-C4 safety lanes lack required patterns: {sorted(missing_patterns)}"
        )

    non_protected_intents = set(intents) - PROTECTED_WRITE_INTENTS
    hard_negative_counts = Counter(
        row["intent"]
        for row in records
        if PROTECTED_MENTION_HARD_NEGATIVE in row["tags"]
    )
    missing_hard_negatives = {
        intent for intent in non_protected_intents if hard_negative_counts[intent] < 4
    }
    if missing_hard_negatives:
        raise ValueError(
            "V2-C4 non-protected lanes lack protected-mention hard negatives: "
            f"{sorted(missing_hard_negatives)}"
        )
    return records


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS,
    *,
    script_path: Path | None = None,
) -> BuiltArtifacts:
    seed = _load_json_object(paths.seed)
    contract = _load_json_object(paths.contract)
    error_analysis = _load_json_object(paths.error_analysis_config)
    intents, risk_by_intent = _validate_contract(contract, paths)
    _validate_error_analysis_config(error_analysis, paths.contract)
    references = _reference_metadata(paths)
    records = _build_records(seed, intents, risk_by_intent, references)

    holdout_payload = {
        "advisory_only": True,
        "created_before_v2c4_error_analysis": True,
        "created_before_v2c4_model_improvement": True,
        "data_role": DATA_ROLE,
        "error_analysis_eligible": False,
        "evaluation_timing": "once after V2-C4 model selection is frozen",
        "example_count": len(records),
        "examples": records,
        "holdout_version": HOLDOUT_VERSION,
        "model_predictions_present": False,
        "model_selection_eligible": False,
        "normalization_version": development_builder.NORMALIZATION_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "threshold_selection_eligible": False,
        "training_eligible": False,
        "training_or_evaluation_performed": False,
    }
    holdout_bytes = development_builder.stable_json_bytes(holdout_payload)

    counts_by_intent = Counter(row["intent"] for row in records)
    counts_by_risk = Counter(row["risk"] for row in records)
    counts_by_pattern = Counter(row["safety_pattern"] for row in records)
    counts_by_tag = Counter(tag for row in records for tag in row["tags"])
    input_paths = {
        "error_analysis_config": paths.error_analysis_config,
        "seed": paths.seed,
        "v2c3_challenge": paths.v2c3_challenge,
        "v2c3_challenge_manifest": paths.v2c3_challenge_manifest,
        "v2c3_development": paths.v2c3_development,
        "v2c3_external_lockbox_manifest": paths.v2c3_external_lockbox_manifest,
        "v2c4_contract": paths.contract,
    }
    manifest_payload = {
        "build_metadata": {
            "builder_version": BUILDER_VERSION,
            "deterministic": True,
            "deterministic_seed": seed["deterministic_seed"],
            "wall_clock_timestamp_recorded": False,
        },
        "build_script_sha256": development_builder.sha256_bytes(
            (script_path or Path(__file__).resolve()).read_bytes()
        ),
        "counts": {
            "example_count": len(records),
            "group_count": len({row["group_id"] for row in records}),
            "lineage_count": len({row["lineage_id"] for row in records}),
        },
        "counts_by_intent": dict(sorted(counts_by_intent.items())),
        "counts_by_risk": dict(sorted(counts_by_risk.items())),
        "counts_by_safety_pattern": dict(sorted(counts_by_pattern.items())),
        "counts_by_tag": dict(sorted(counts_by_tag.items())),
        "evaluation_policy": {
            "data_role": DATA_ROLE,
            "error_analysis_eligible": False,
            "model_selection_eligible": False,
            "single_final_evaluation": True,
            "threshold_selection_eligible": False,
            "training_eligible": False,
        },
        "holdout_version": HOLDOUT_VERSION,
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
            "v2c4_safety_holdout.json": development_builder.sha256_bytes(
                holdout_bytes
            )
        },
        "reference_normalized_hash_counts": references.hash_counts,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed_sha256": development_builder.sha256_bytes(paths.seed.read_bytes()),
        "source_policy": seed["source_policy"],
        "training_or_evaluation_performed": False,
        "validation": {
            "balanced_40_per_intent": all(
                counts_by_intent[intent] == EXAMPLES_PER_INTENT
                for intent in intents
            ),
            "cfpb_used": False,
            "consumed_v2c3_challenge_normalized_hash_overlap": 0,
            "consumed_v2c3_external_lockbox_normalized_hash_overlap": 0,
            "independent_v2c4_lineage": True,
            "lineage_overlap_with_v2c3_challenge": 0,
            "model_predictions_used_for_authorship_or_membership": False,
            "normalized_duplicate_count": 0,
            "protected_write_semantic_review": (
                "Gold intents are explicit in the seed; protected positives are "
                "explicitly authored current-action requests and lexical checks "
                "supplement but do not replace human semantic review."
            ),
            "required_hard_negative_categories_present": True,
            "required_safety_patterns_present": True,
            "risk_derived_from_v2c4_contract": True,
            "taxonomy_matches_v2c4_contract": True,
            "v2c3_development_normalized_hash_overlap": 0,
        },
    }
    manifest_bytes = development_builder.stable_json_bytes(manifest_payload)
    return BuiltArtifacts(
        holdout_bytes=holdout_bytes,
        manifest_bytes=manifest_bytes,
        holdout_payload=holdout_payload,
        manifest_payload=manifest_payload,
    )


def write_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    development_builder.atomic_write_bytes(paths.holdout_output, artifacts.holdout_bytes)
    development_builder.atomic_write_bytes(
        paths.manifest_output, artifacts.manifest_bytes
    )


def check_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    expected = {
        paths.holdout_output: artifacts.holdout_bytes,
        paths.manifest_output: artifacts.manifest_bytes,
    }
    for path, payload in expected.items():
        if not path.exists():
            raise FileNotFoundError(
                f"V2-C4 safety holdout is missing: {path}; use --write"
            )
        if path.read_bytes() != payload:
            raise ValueError(
                f"V2-C4 safety holdout differs from deterministic build: {path}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write holdout artifacts")
    mode.add_argument("--check", action="store_true", help="verify holdout artifacts")
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
        f"{action} V2-C4 safety holdout artifacts: "
        f"examples={artifacts.holdout_payload['example_count']}, "
        "training/model/threshold/error-analysis eligible=false; "
        "no training, inference, or evaluation performed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
