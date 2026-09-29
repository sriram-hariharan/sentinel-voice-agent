"""Build frozen V2-C4 development augmentation and selection-probe artifacts."""

from __future__ import annotations

import argparse
import json
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

BUILDER_VERSION = "v2c4-development-data-builder.v1"
SEED_SCHEMA_VERSION = "v2c4-development-data-seed.v1"
DATASET_SCHEMA_VERSION = "v2c4-development-data.v1"
MANIFEST_SCHEMA_VERSION = "v2c4-development-data-manifest.v1"

AUGMENTATION_ROLE = "v2c4_targeted_training_augmentation"
PROBE_ROLE = "v2c4_model_selection_probe"
AUGMENTATION_TOTAL = 360
PROBE_TOTAL = 270
EXAMPLES_PER_PROBE_INTENT = 30

EXPECTED_INTENTS = [
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
    "unsupported_or_uncertain",
]
EXPECTED_RISK_BY_INTENT = {
    "account_balance": "PRIVATE_READ",
    "card_status": "PRIVATE_READ",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
EXPECTED_PROBE_RISKS = {
    "ESCALATION_OR_UNCERTAIN": 60,
    "PRIVATE_READ": 120,
    "PROTECTED_WRITE": 60,
    "PUBLIC": 30,
}
REQUIRED_PROVENANCE = {
    "independently_authored": True,
    "generated_after_v2c4_holdout_freeze": True,
    "consumed_v2c3_failure_categories_informed_design": True,
    "consumed_v2c3_raw_examples_used_for_authoring": False,
    "copied_consumed_examples": False,
    "paraphrased_consumed_examples": False,
    "cfpb_used": False,
    "banking77_test_used": False,
    "clinc_test_used": False,
    "v2c4_final_holdout_used_for_authoring": False,
}
REQUIRED_AUGMENTATION_TAGS_BY_LANE = {
    "protected_positive": {
        "correction",
        "contrast",
        "prior_card_status_context",
        "prior_transaction_details_context",
        "prior_policy_or_information_context",
        "conversational_or_noisy_phrasing",
        "long_form_phrasing",
        "lost_or_stolen_context_where_appropriate",
        "transaction_context",
        "natural_action_synonyms",
        "explicit_current_action_after_contextual_language",
    },
    "protected_action_hard_negative": {
        "information_only_mentions",
        "historical_protected_actions",
        "hypothetical_protected_actions",
        "explicit_negation",
        "cancellation",
        "protected_action_mention_with_no_current_request",
        "wrong_object_or_scope",
        "adjacent_private_read_or_public_policy_intents",
    },
    "unsupported_or_scope": {
        "unsupported_banking_operations",
        "unsupported_objects",
        "no_current_request_statements",
        "historical_statements",
        "hypotheticals",
        "vague_complaints",
        "fragments",
        "ambiguous_references",
        "wrong_object_requests",
        "scope_violations",
    },
}


@dataclass(frozen=True)
class BuildPaths:
    augmentation_seed: Path
    probe_seed: Path
    intervention_plan: Path
    experiment_contract: Path
    development_dataset: Path
    challenge_dataset: Path
    external_lockbox_manifest: Path
    final_holdout_manifest: Path
    augmentation_output: Path
    augmentation_manifest_output: Path
    probe_output: Path
    probe_manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    augmentation_seed=ML_ROOT / "v2c4_training_augmentation_seed.json",
    probe_seed=ML_ROOT / "v2c4_selection_probe_seed.json",
    intervention_plan=ML_ROOT / "v2c4_intervention_plan.json",
    experiment_contract=ML_ROOT / "v2c4_experiment_contract.json",
    development_dataset=ML_ROOT / "v2c3_development_dataset.json",
    challenge_dataset=ML_ROOT / "v2c3_challenge_set.json",
    external_lockbox_manifest=ML_ROOT / "v2c3_fresh_lockbox_manifest.json",
    final_holdout_manifest=ML_ROOT / "v2c4_safety_holdout.manifest.json",
    augmentation_output=ML_ROOT / "v2c4_training_augmentation.json",
    augmentation_manifest_output=(
        ML_ROOT / "v2c4_training_augmentation.manifest.json"
    ),
    probe_output=ML_ROOT / "v2c4_selection_probe.json",
    probe_manifest_output=ML_ROOT / "v2c4_selection_probe.manifest.json",
)


@dataclass(frozen=True)
class BuiltArtifacts:
    augmentation_bytes: bytes
    augmentation_manifest_bytes: bytes
    probe_bytes: bytes
    probe_manifest_bytes: bytes
    augmentation_payload: dict[str, Any]
    augmentation_manifest_payload: dict[str, Any]
    probe_payload: dict[str, Any]
    probe_manifest_payload: dict[str, Any]


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_plan_and_contract(
    plan: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, str]:
    if plan.get("schema_version") != "v2c4-intervention-plan.v1":
        raise ValueError("unexpected V2-C4 intervention-plan schema")
    if plan.get("step") != 9:
        raise ValueError("V2-C4 intervention plan must be frozen at Step 9")
    taxonomy = plan.get("current_taxonomy")
    if not isinstance(taxonomy, dict) or taxonomy.get("intents") != EXPECTED_INTENTS:
        raise ValueError("intervention plan does not freeze the expected taxonomy")
    if taxonomy.get("intent_taxonomy_expansion_deferred_to") != "V2-C5":
        raise ValueError("taxonomy expansion must remain deferred to V2-C5")
    augmentation = plan.get("targeted_training_augmentation")
    probe = plan.get("selection_probe")
    if not isinstance(augmentation, dict) or not isinstance(probe, dict):
        raise TypeError("intervention plan development contracts must be objects")
    if augmentation.get("target_example_count") != AUGMENTATION_TOTAL:
        raise ValueError("intervention plan augmentation count changed")
    if probe.get("target_example_count") != PROBE_TOTAL:
        raise ValueError("intervention plan selection-probe count changed")
    if probe.get("counts_by_intent") != dict.fromkeys(EXPECTED_INTENTS, 30):
        raise ValueError("intervention plan probe balance changed")
    risk_by_intent = contract.get("risk_by_intent")
    if contract.get("intent_taxonomy") != EXPECTED_INTENTS:
        raise ValueError("experiment contract taxonomy differs from Step 9")
    if risk_by_intent != EXPECTED_RISK_BY_INTENT:
        raise ValueError("experiment contract risk mapping differs from frozen mapping")
    return dict(risk_by_intent)


def validate_seed_header(
    seed: Mapping[str, Any], *, expected_role: str, expected_count: int
) -> None:
    if seed.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError(f"unexpected seed schema for {expected_role}")
    if seed.get("data_role") != expected_role:
        raise ValueError(f"unexpected data role for {expected_role}")
    if seed.get("expected_example_count") != expected_count:
        raise ValueError(f"unexpected example count for {expected_role}")
    if seed.get("authorship_provenance") != REQUIRED_PROVENANCE:
        raise ValueError(f"authorship provenance changed for {expected_role}")
    expected_eligibility = (
        {
            "training_eligible": True,
            "model_selection_eligible": False,
            "threshold_selection_eligible": False,
            "final_acceptance_evidence": False,
        }
        if expected_role == AUGMENTATION_ROLE
        else {
            "training_eligible": False,
            "model_selection_eligible": True,
            "threshold_selection_eligible": False,
            "final_acceptance_evidence": False,
        }
    )
    if seed.get("eligibility_policy") != expected_eligibility:
        raise ValueError(f"eligibility policy changed for {expected_role}")
    families = seed.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError(f"{expected_role} must contain authored families")


def build_records(
    seed: Mapping[str, Any], risk_by_intent: Mapping[str, str], prefix: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    for family_index, family_value in enumerate(seed["families"], start=1):
        if not isinstance(family_value, dict):
            raise TypeError(f"family {family_index} must be an object")
        family = family_value
        family_id = require_string(family.get("family_id"), "family_id")
        group_id = require_string(family.get("group_id"), "group_id")
        lineage_id = require_string(family.get("lineage_id"), "lineage_id")
        design_lane = require_string(family.get("design_lane"), "design_lane")
        intent = require_string(family.get("intent"), "intent")
        tags = family.get("tags")
        utterances = family.get("utterances")
        if family_id in family_ids:
            raise ValueError(f"duplicate family_id: {family_id}")
        family_ids.add(family_id)
        if intent not in risk_by_intent:
            raise ValueError(f"unknown intent in {family_id}: {intent}")
        if not isinstance(tags, list) or not tags or not all(
            isinstance(tag, str) and tag for tag in tags
        ):
            raise ValueError(f"{family_id} tags must be non-empty strings")
        if not isinstance(utterances, list) or not utterances:
            raise ValueError(f"{family_id} utterances must be a non-empty list")
        for utterance_index, text_value in enumerate(utterances, start=1):
            text = require_string(text_value, f"{family_id} utterance")
            if "protected action" in text.lower():
                raise ValueError(f"internal engineering phrase in {family_id}")
            ordinal = len(records) + 1
            records.append(
                {
                    "example_id": f"{prefix}:{ordinal:03d}",
                    "source_id": f"{prefix}-seed:{family_id}:{utterance_index:02d}",
                    "data_role": seed["data_role"],
                    "text": text,
                    "text_sha256": development_builder.text_sha256(text),
                    "normalized_text_sha256": (
                        development_builder.normalized_text_sha256(text)
                    ),
                    "intent": intent,
                    "risk": risk_by_intent[intent],
                    "family_id": family_id,
                    "group_id": group_id,
                    "lineage_id": lineage_id,
                    "design_lane": design_lane,
                    "tags": sorted(set(tags)),
                    **seed["eligibility_policy"],
                }
            )
    return records


def validate_record_integrity(records: list[dict[str, Any]], label: str) -> None:
    for field in ("example_id", "normalized_text_sha256"):
        values = [row[field] for row in records]
        if len(values) != len(set(values)):
            raise ValueError(f"{label} has duplicate {field}")
    family_to_group: dict[str, str] = {}
    family_to_lineage: dict[str, str] = {}
    for row in records:
        family_id = row["family_id"]
        group_id = row["group_id"]
        lineage_id = row["lineage_id"]
        if family_id in family_to_group and family_to_group[family_id] != group_id:
            raise ValueError(f"{family_id} maps to multiple groups")
        if family_id in family_to_lineage and (
            family_to_lineage[family_id] != lineage_id
        ):
            raise ValueError(f"{family_id} maps to multiple lineages")
        family_to_group[family_id] = group_id
        family_to_lineage[family_id] = lineage_id


def validate_augmentation(
    records: list[dict[str, Any]], seed: Mapping[str, Any]
) -> None:
    if len(records) != AUGMENTATION_TOTAL:
        raise ValueError("training augmentation must contain exactly 360 examples")
    lane_counts = Counter(row["design_lane"] for row in records)
    expected_lanes = {
        "protected_positive": 120,
        "protected_action_hard_negative": 120,
        "unsupported_or_scope": 120,
    }
    if lane_counts != expected_lanes:
        raise ValueError(f"augmentation lane counts differ: {dict(lane_counts)}")
    for lane, required_tags in REQUIRED_AUGMENTATION_TAGS_BY_LANE.items():
        actual_tags = {
            tag for row in records if row["design_lane"] == lane for tag in row["tags"]
        }
        if not required_tags.issubset(actual_tags):
            missing = sorted(required_tags - actual_tags)
            raise ValueError(f"{lane} omits required coverage tags: {missing}")
    positives = [row for row in records if row["design_lane"] == "protected_positive"]
    if Counter(row["intent"] for row in positives) != {
        "create_dispute": 60,
        "freeze_card": 60,
    }:
        raise ValueError("protected positives must be balanced 60/60")
    if any(row["risk"] != "PROTECTED_WRITE" for row in positives):
        raise ValueError("protected-positive risk mapping differs")
    hard_negatives = [
        row
        for row in records
        if row["design_lane"] == "protected_action_hard_negative"
    ]
    expected_hard_negative_counts = seed.get("hard_negative_counts_by_intent")
    if Counter(row["intent"] for row in hard_negatives) != expected_hard_negative_counts:
        raise ValueError("hard-negative intent distribution differs from seed metadata")
    if len({row["intent"] for row in hard_negatives}) < 4:
        raise ValueError("hard negatives must use at least four non-protected intents")
    required_hard_negative_intents = {
        "informational_policy",
        "card_status",
        "transaction_details",
        "unsupported_or_uncertain",
    }
    if not required_hard_negative_intents.issubset(
        {row["intent"] for row in hard_negatives}
    ):
        raise ValueError("hard negatives omit a required real intent")
    unsupported = [
        row for row in records if row["design_lane"] == "unsupported_or_scope"
    ]
    if any(row["intent"] != "unsupported_or_uncertain" for row in unsupported):
        raise ValueError("unsupported/scope lane contains another intent")
    expected_policy = {
        "training_eligible": True,
        "model_selection_eligible": False,
        "threshold_selection_eligible": False,
        "final_acceptance_evidence": False,
    }
    if any(
        {field: row[field] for field in expected_policy} != expected_policy
        for row in records
    ):
        raise ValueError("augmentation eligibility policy differs")


def validate_probe(records: list[dict[str, Any]]) -> None:
    if len(records) != PROBE_TOTAL:
        raise ValueError("selection probe must contain exactly 270 examples")
    if Counter(row["intent"] for row in records) != dict.fromkeys(
        EXPECTED_INTENTS, EXAMPLES_PER_PROBE_INTENT
    ):
        raise ValueError("selection probe must contain 30 examples per intent")
    if Counter(row["risk"] for row in records) != EXPECTED_PROBE_RISKS:
        raise ValueError("selection probe risk totals differ")
    expected_policy = {
        "training_eligible": False,
        "model_selection_eligible": True,
        "threshold_selection_eligible": False,
        "final_acceptance_evidence": False,
    }
    if any(
        {field: row[field] for field in expected_policy} != expected_policy
        for row in records
    ):
        raise ValueError("selection-probe eligibility policy differs")


def reference_index(
    payload: Mapping[str, Any], collection: str, reference_name: str
) -> dict[str, list[dict[str, str]]]:
    rows = payload.get(collection)
    if not isinstance(rows, list):
        raise TypeError(f"reference {collection} must be a list")
    index: dict[str, list[dict[str, str]]] = {}
    for ordinal, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"reference {collection} record must be an object")
        normalized_hash = require_string(
            row.get("normalized_text_sha256"),
            f"{reference_name} normalized_text_sha256",
        )
        reference_id = row.get("example_id", row.get("source_id"))
        if not isinstance(reference_id, str) or not reference_id:
            reference_id = f"record:{ordinal:06d}"
        reference_role = row.get("data_role")
        if not isinstance(reference_role, str) or not reference_role:
            reference_role = reference_name
        index.setdefault(normalized_hash, []).append(
            {
                "reference_id": reference_id,
                "reference_role": reference_role,
            }
        )
    return index


def reference_indexes(
    paths: BuildPaths,
) -> tuple[dict[str, dict[str, list[dict[str, str]]]], dict[str, Any]]:
    development = load_json_object(paths.development_dataset)
    challenge = load_json_object(paths.challenge_dataset)
    lockbox = load_json_object(paths.external_lockbox_manifest)
    holdout_manifest = load_json_object(paths.final_holdout_manifest)
    if holdout_manifest.get("schema_version") != "v2c4-safety-holdout-manifest.v1":
        raise ValueError("unexpected sealed V2-C4 holdout-manifest schema")
    references = {
        "v2c3_development": reference_index(
            development, "examples", "v2c3_development"
        ),
        "consumed_v2c3_challenge": reference_index(
            challenge, "examples", "consumed_v2c3_challenge"
        ),
        "consumed_v2c3_external_lockbox": reference_index(
            lockbox, "members", "consumed_v2c3_external_lockbox"
        ),
    }
    holdout_hashes = holdout_manifest.get("normalized_text_sha256_values")
    if holdout_hashes is not None:
        raise ValueError(
            "sealed holdout manifest unexpectedly exposes record hashes; "
            "review the frozen boundary before changing this builder"
        )
    limitation = {
        "exact_check_performed": False,
        "reason": (
            "The frozen V2-C4 holdout manifest exposes aggregate hash counts "
            "but no record-level normalized hashes. The sealed holdout and seed "
            "are not opened; independence relies on post-freeze authorship."
        ),
        "sealed_holdout_or_seed_opened": False,
        "manifest_path": display_path(paths.final_holdout_manifest),
    }
    return references, limitation


def reference_collisions(
    records: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
) -> list[dict[str, str]]:
    collisions: list[dict[str, str]] = []
    for row in records:
        normalized_hash = row["normalized_text_sha256"]
        for reference_name, index in sorted(references.items()):
            for reference in index.get(normalized_hash, []):
                collisions.append(
                    {
                        "new_data_role": row["data_role"],
                        "new_example_id": row["example_id"],
                        "new_intent": row["intent"],
                        "new_text": row["text"],
                        "normalized_text_sha256": normalized_hash,
                        "reference_name": reference_name,
                        "reference_role": reference["reference_role"],
                        "reference_id": reference["reference_id"],
                    }
                )
    return collisions


def raise_for_reference_collisions(
    records: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
) -> None:
    collisions = reference_collisions(records, references)
    if not collisions:
        return
    lines = ["Leakage detected:"]
    for collision_number, collision in enumerate(collisions, start=1):
        if collision_number > 1:
            lines.append("")
        lines.extend(
            [
                f"collision={collision_number}",
                f"new={collision['new_data_role']}",
                f"example_id={collision['new_example_id']}",
                f"intent={collision['new_intent']}",
                f"new_text={json.dumps(collision['new_text'], ensure_ascii=False)}",
                (
                    "normalized_text_sha256="
                    f"{collision['normalized_text_sha256']}"
                ),
                f"reference={collision['reference_name']}",
                f"reference_role={collision['reference_role']}",
                f"reference_id={collision['reference_id']}",
            ]
        )
    raise ValueError("\n".join(lines))


def counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in records).items()))


def tag_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(tag for row in records for tag in row["tags"]).items()))


def build_dataset_payload(
    seed: Mapping[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_version": seed["dataset_version"],
        "data_role": seed["data_role"],
        "example_count": len(records),
        "normalization_version": development_builder.NORMALIZATION_VERSION,
        "eligibility_policy": seed["eligibility_policy"],
        "training_or_evaluation_performed": False,
        "examples": records,
    }


def build_manifest(
    *,
    seed: Mapping[str, Any],
    seed_path: Path,
    output_path: Path,
    output_bytes: bytes,
    records: list[dict[str, Any]],
    script_path: Path,
    paths: BuildPaths,
    plan_sha256: str,
    reference_indexes_by_name: Mapping[
        str, Mapping[str, Sequence[Mapping[str, str]]]
    ],
    other_records: list[dict[str, Any]],
    holdout_limitation: Mapping[str, Any],
) -> dict[str, Any]:
    record_hashes = {row["normalized_text_sha256"] for row in records}
    other_hashes = {row["normalized_text_sha256"] for row in other_records}
    reference_results = {
        name: {
            "normalized_hash_count": len(index),
            "normalized_text_overlap": len(record_hashes & set(index)),
        }
        for name, index in sorted(reference_indexes_by_name.items())
    }
    raise_for_reference_collisions(records, reference_indexes_by_name)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": seed["dataset_version"],
        "build_metadata": {
            "builder_version": BUILDER_VERSION,
            "deterministic": True,
            "wall_clock_timestamp_recorded": False,
        },
        "builder_sha256": development_builder.sha256_bytes(script_path.read_bytes()),
        "seed_sha256": development_builder.sha256_bytes(seed_path.read_bytes()),
        "parent_intervention_plan_sha256": plan_sha256,
        "input_paths": {
            "seed": display_path(seed_path),
            "parent_intervention_plan": display_path(paths.intervention_plan),
            "experiment_contract": display_path(paths.experiment_contract),
            "v2c3_development": display_path(paths.development_dataset),
            "consumed_v2c3_challenge": display_path(paths.challenge_dataset),
            "consumed_v2c3_external_lockbox": display_path(
                paths.external_lockbox_manifest
            ),
            "sealed_v2c4_holdout_manifest_only": display_path(
                paths.final_holdout_manifest
            ),
        },
        "input_sha256": {
            "seed": development_builder.sha256_bytes(seed_path.read_bytes()),
            "parent_intervention_plan": plan_sha256,
            "experiment_contract": development_builder.sha256_bytes(
                paths.experiment_contract.read_bytes()
            ),
            "v2c3_development": development_builder.sha256_bytes(
                paths.development_dataset.read_bytes()
            ),
            "consumed_v2c3_challenge": development_builder.sha256_bytes(
                paths.challenge_dataset.read_bytes()
            ),
            "consumed_v2c3_external_lockbox": development_builder.sha256_bytes(
                paths.external_lockbox_manifest.read_bytes()
            ),
            "sealed_v2c4_holdout_manifest_only": development_builder.sha256_bytes(
                paths.final_holdout_manifest.read_bytes()
            ),
        },
        "counts": {
            "example_count": len(records),
            "family_count": len({row["family_id"] for row in records}),
            "group_count": len({row["group_id"] for row in records}),
            "lineage_count": len({row["lineage_id"] for row in records}),
        },
        "counts_by_intent": counts(records, "intent"),
        "counts_by_risk": counts(records, "risk"),
        "counts_by_design_lane": counts(records, "design_lane"),
        "counts_by_tag": tag_counts(records),
        "normalization": {
            "version": development_builder.NORMALIZATION_VERSION,
            "steps": development_builder.NORMALIZATION_STEPS,
            "hash_algorithm": "SHA-256",
        },
        "normalized_text_uniqueness": {
            "unique_count": len(record_hashes),
            "duplicate_count": len(records) - len(record_hashes),
        },
        "cross_dataset_overlap": {
            "other_data_role": other_records[0]["data_role"],
            "normalized_text_overlap": len(record_hashes & other_hashes),
            "example_id_overlap": len(
                {row["example_id"] for row in records}
                & {row["example_id"] for row in other_records}
            ),
            "group_id_overlap": len(
                {row["group_id"] for row in records}
                & {row["group_id"] for row in other_records}
            ),
            "lineage_id_overlap": len(
                {row["lineage_id"] for row in records}
                & {row["lineage_id"] for row in other_records}
            ),
        },
        "reference_dataset_overlap_results": reference_results,
        "sealed_v2c4_holdout_overlap_check": dict(holdout_limitation),
        "source_policy": seed["authorship_provenance"],
        "eligibility_policy": seed["eligibility_policy"],
        "execution_status": {
            "training_performed": False,
            "embedding_performed": False,
            "model_loading_performed": False,
            "inference_performed": False,
            "evaluation_performed": False,
        },
        "training_or_evaluation_performed": False,
        "output_sha256": development_builder.sha256_bytes(output_bytes),
        "output_path": display_path(output_path),
    }


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS, script_path: Path | None = None
) -> BuiltArtifacts:
    augmentation_seed = load_json_object(paths.augmentation_seed)
    probe_seed = load_json_object(paths.probe_seed)
    plan = load_json_object(paths.intervention_plan)
    contract = load_json_object(paths.experiment_contract)
    risk_by_intent = validate_plan_and_contract(plan, contract)
    validate_seed_header(
        augmentation_seed,
        expected_role=AUGMENTATION_ROLE,
        expected_count=AUGMENTATION_TOTAL,
    )
    validate_seed_header(
        probe_seed, expected_role=PROBE_ROLE, expected_count=PROBE_TOTAL
    )
    plan_sha256 = development_builder.sha256_bytes(paths.intervention_plan.read_bytes())
    for seed, label in (
        (augmentation_seed, "augmentation"),
        (probe_seed, "selection probe"),
    ):
        if seed.get("parent_intervention_plan_sha256") != plan_sha256:
            raise ValueError(f"{label} seed parent intervention-plan hash differs")

    augmentation_records = build_records(augmentation_seed, risk_by_intent, "v2c4-aug")
    probe_records = build_records(probe_seed, risk_by_intent, "v2c4-probe")
    validate_record_integrity(augmentation_records, "augmentation")
    validate_record_integrity(probe_records, "selection probe")
    validate_augmentation(augmentation_records, augmentation_seed)
    validate_probe(probe_records)

    cross_checks = {
        "normalized text": (
            {row["normalized_text_sha256"] for row in augmentation_records},
            {row["normalized_text_sha256"] for row in probe_records},
        ),
        "example ID": (
            {row["example_id"] for row in augmentation_records},
            {row["example_id"] for row in probe_records},
        ),
        "group ID": (
            {row["group_id"] for row in augmentation_records},
            {row["group_id"] for row in probe_records},
        ),
        "lineage ID": (
            {row["lineage_id"] for row in augmentation_records},
            {row["lineage_id"] for row in probe_records},
        ),
    }
    for label, (augmentation_values, probe_values) in cross_checks.items():
        if augmentation_values & probe_values:
            raise ValueError(f"augmentation/probe {label} overlap")

    references, holdout_limitation = reference_indexes(paths)
    augmentation_payload = build_dataset_payload(
        augmentation_seed, augmentation_records
    )
    probe_payload = build_dataset_payload(probe_seed, probe_records)
    augmentation_bytes = development_builder.stable_json_bytes(augmentation_payload)
    probe_bytes = development_builder.stable_json_bytes(probe_payload)
    actual_script = script_path or Path(__file__).resolve()
    augmentation_manifest = build_manifest(
        seed=augmentation_seed,
        seed_path=paths.augmentation_seed,
        output_path=paths.augmentation_output,
        output_bytes=augmentation_bytes,
        records=augmentation_records,
        script_path=actual_script,
        paths=paths,
        plan_sha256=plan_sha256,
        reference_indexes_by_name=references,
        other_records=probe_records,
        holdout_limitation=holdout_limitation,
    )
    probe_manifest = build_manifest(
        seed=probe_seed,
        seed_path=paths.probe_seed,
        output_path=paths.probe_output,
        output_bytes=probe_bytes,
        records=probe_records,
        script_path=actual_script,
        paths=paths,
        plan_sha256=plan_sha256,
        reference_indexes_by_name=references,
        other_records=augmentation_records,
        holdout_limitation=holdout_limitation,
    )
    return BuiltArtifacts(
        augmentation_bytes=augmentation_bytes,
        augmentation_manifest_bytes=development_builder.stable_json_bytes(
            augmentation_manifest
        ),
        probe_bytes=probe_bytes,
        probe_manifest_bytes=development_builder.stable_json_bytes(probe_manifest),
        augmentation_payload=augmentation_payload,
        augmentation_manifest_payload=augmentation_manifest,
        probe_payload=probe_payload,
        probe_manifest_payload=probe_manifest,
    )


def write_artifacts(
    artifacts: BuiltArtifacts, paths: BuildPaths = DEFAULT_PATHS
) -> None:
    outputs = {
        paths.augmentation_output: artifacts.augmentation_bytes,
        paths.augmentation_manifest_output: artifacts.augmentation_manifest_bytes,
        paths.probe_output: artifacts.probe_bytes,
        paths.probe_manifest_output: artifacts.probe_manifest_bytes,
    }
    for path, payload in outputs.items():
        development_builder.atomic_write_bytes(path, payload)


def check_artifacts(
    artifacts: BuiltArtifacts, paths: BuildPaths = DEFAULT_PATHS
) -> None:
    expected = {
        paths.augmentation_output: artifacts.augmentation_bytes,
        paths.augmentation_manifest_output: artifacts.augmentation_manifest_bytes,
        paths.probe_output: artifacts.probe_bytes,
        paths.probe_manifest_output: artifacts.probe_manifest_bytes,
    }
    for path, payload in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"V2-C4 artifact is missing: {path}; use --write")
        if path.read_bytes() != payload:
            raise ValueError(f"V2-C4 artifact differs from deterministic build: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write all four artifacts")
    mode.add_argument("--check", action="store_true", help="verify all four artifacts")
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
        f"{action} V2-C4 development artifacts: "
        f"augmentation={artifacts.augmentation_payload['example_count']}, "
        f"selection_probe={artifacts.probe_payload['example_count']}; "
        "no training, inference, or evaluation performed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
