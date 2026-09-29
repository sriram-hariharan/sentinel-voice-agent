"""Build the deterministic V2-C5 intent-discovery corpus and manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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

BUILDER_VERSION = "v2c5-discovery-corpus-builder.v1"
CORPUS_SCHEMA_VERSION = "v2c5-discovery-corpus.v1"
MANIFEST_SCHEMA_VERSION = "v2c5-discovery-corpus-manifest.v1"
CONTRACT_SCHEMA_VERSION = "v2c5-intent-discovery-contract.v1"
CONTRACT_SHA256 = (
    "acc48bace76a5ae93ff9a1818856e3b2dc0ad3585c80e8ba7a3f582208a71427"
)
DEVELOPMENT_DATASET_SHA256 = (
    "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "abe82919c24855d8d35a7c383843aaab4a5a2048ac868ad6a7b6be0791d34030"
)
DATA_REGISTRY_SHA256 = (
    "978dd7b04fcad23e3a01add65598ffda73aee9323118ffcc382dcd2bdce01316"
)

PRIMARY_INTENT = "unsupported_or_uncertain"
SUPPORTED_INTENTS = [
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
]
EXPECTED_INTENTS = [*SUPPORTED_INTENTS, PRIMARY_INTENT]
PRIMARY_DATA_ROLE = "v2c5_primary_intent_discovery_corpus"
REFERENCE_DATA_ROLE = "v2c5_reference_anchor_population"

EXPECTED_ELIGIBLE_SOURCE_ROLES = [
    {
        "development_assignment_required": True,
        "materialized_path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "record_id": "internal_train",
        "registry_role": "v2c3_development",
        "source_id": "sentinelvoice_v2c1_internal",
        "source_split": "train",
    },
    {
        "development_assignment_required": True,
        "materialized_path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "record_id": "internal_validation",
        "registry_role": "v2c3_development",
        "source_id": "sentinelvoice_v2c1_internal",
        "source_split": "validation",
    },
    {
        "development_assignment_required": True,
        "materialized_path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "record_id": "banking77_train",
        "registry_role": "v2c3_conditional_development_and_fresh_lockbox_source",
        "source_id": "banking77",
        "source_split": "train",
    },
    {
        "development_assignment_required": True,
        "materialized_path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "record_id": "clinc_finance_train",
        "registry_role": "v2c3_conditional_development_and_fresh_lockbox_source",
        "source_id": "clinc150_oos",
        "source_split": "train",
    },
    {
        "development_assignment_required": True,
        "materialized_path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "record_id": "clinc_finance_val",
        "registry_role": "v2c3_conditional_development_and_fresh_lockbox_source",
        "source_id": "clinc150_oos",
        "source_split": "val",
    },
]
EXPECTED_REGISTRY_SPLITS = {
    "internal_train": "train",
    "internal_validation": "validation",
    "banking77_train": "train.csv",
    "clinc_finance_train": "train (finance domains only)",
    "clinc_finance_val": "val (finance domains only)",
}
SOURCE_PRECEDENCE = {
    "sentinelvoice_v2c1_internal": 0,
    "banking77": 1,
    "clinc150_oos": 2,
}
SPLIT_PRECEDENCE = {"train": 0, "validation": 1, "val": 1}
FORBIDDEN_INPUT_FILENAMES = {
    "v2c4_safety_holdout.json",
    "v2c4_safety_holdout_seed.json",
}
REQUIRED_EXCLUSIONS = {
    "all_test_only_or_final_evaluation_sources",
    "banking77_test",
    "cfpb",
    "clinc_test",
    "consumed_v2c3_challenge",
    "consumed_v2c3_external_lockbox",
    "v2c3_internal_locked_test",
    "v2c4_final_safety_holdout",
    "v2c4_postmortem_review_examples",
    "v2c4_selection_probe",
    "v2c4_targeted_synthetic_augmentation",
}


@dataclass(frozen=True)
class BuildPaths:
    contract: Path
    development_dataset: Path
    development_manifest: Path
    data_registry: Path
    corpus_output: Path
    manifest_output: Path


DEFAULT_PATHS = BuildPaths(
    contract=ML_ROOT / "v2c5_intent_discovery_contract.json",
    development_dataset=ML_ROOT / "v2c3_development_dataset.json",
    development_manifest=ML_ROOT / "v2c3_dataset.manifest.json",
    data_registry=ML_ROOT / "v2c3_data_registry.json",
    corpus_output=ML_ROOT / "v2c5_discovery_corpus.json",
    manifest_output=ML_ROOT / "v2c5_discovery_corpus.manifest.json",
)


@dataclass(frozen=True)
class BuiltArtifacts:
    corpus_bytes: bytes
    manifest_bytes: bytes
    corpus_payload: dict[str, Any]
    manifest_payload: dict[str, Any]


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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_lines(values: Sequence[str]) -> str:
    encoded = "".join(f"{value}\n" for value in values).encode("utf-8")
    return development_builder.sha256_bytes(encoded)


def source_role_by_key() -> dict[tuple[str, str], str]:
    return {
        (entry["source_id"], entry["source_split"]): entry["registry_role"]
        for entry in EXPECTED_ELIGIBLE_SOURCE_ROLES
    }


def validate_contract(
    contract: Mapping[str, Any],
    contract_bytes: bytes,
) -> None:
    if development_builder.sha256_bytes(contract_bytes) != CONTRACT_SHA256:
        raise ValueError("frozen V2-C5 contract SHA-256 changed")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected V2-C5 contract schema")
    if contract.get("phase") != "V2-C5":
        raise ValueError("V2-C5 contract phase changed")
    if contract.get("status") != "frozen":
        raise ValueError("V2-C5 contract is not frozen")
    execution = contract.get("execution_status")
    if not isinstance(execution, dict) or execution.get("contract_frozen") is not True:
        raise ValueError("V2-C5 contract frozen execution state is not true")

    eligibility = contract.get("source_eligibility")
    if not isinstance(eligibility, dict):
        raise TypeError("V2-C5 source eligibility must be an object")
    if eligibility.get("eligible_source_roles") != EXPECTED_ELIGIBLE_SOURCE_ROLES:
        raise ValueError("V2-C5 eligible source roles changed")
    primary = eligibility.get("primary_cluster_population")
    if not isinstance(primary, dict):
        raise TypeError("primary cluster population must be an object")
    if primary.get("current_v2_taxonomy_label") != PRIMARY_INTENT:
        raise ValueError("primary discovery intent changed")
    if primary.get("dataset_sha256") != DEVELOPMENT_DATASET_SHA256:
        raise ValueError("primary discovery dataset hash changed")
    if primary.get("supported_intents_included") is not False:
        raise ValueError("supported intents cannot enter primary density")
    if primary.get("unique_normalized_text_only") is not True:
        raise ValueError("primary discovery population must be deduplicated")

    exclusions = contract.get("source_exclusions")
    if not isinstance(exclusions, dict) or set(exclusions) != REQUIRED_EXCLUSIONS:
        raise ValueError("V2-C5 source exclusions changed")
    for name in REQUIRED_EXCLUSIONS:
        specification = exclusions[name]
        if not isinstance(specification, dict):
            raise TypeError(f"source exclusion {name} must be an object")
        eligibility_flags = [
            value for key, value in specification.items() if "eligible" in key
        ]
        if not eligibility_flags or any(
            value is not False for value in eligibility_flags
        ):
            raise ValueError(f"source exclusion {name} is not fail-closed")

    labels = eligibility.get("label_metadata_policy")
    if not isinstance(labels, dict):
        raise TypeError("label metadata policy must be an object")
    if (
        labels.get("native_external_labels_supplied_as_clustering_features")
        is not False
    ):
        raise ValueError("native labels cannot be clustering features")
    if (
        labels.get(
            "current_sentinelvoice_labels_supplied_as_clustering_features"
        )
        is not False
    ):
        raise ValueError("current intent labels cannot be clustering features")


def validate_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != "v2c3-data-registry.v1":
        raise ValueError("unexpected V2-C3 data-registry schema")
    records_value = registry.get("records")
    if not isinstance(records_value, list):
        raise TypeError("V2-C3 data-registry records must be a list")
    records = {
        row["record_id"]: row
        for row in records_value
        if isinstance(row, dict) and isinstance(row.get("record_id"), str)
    }
    expected_ids = {
        entry["record_id"] for entry in EXPECTED_ELIGIBLE_SOURCE_ROLES
    }
    eligible_ids = {
        record_id
        for record_id, row in records.items()
        if row.get("training_eligible") is True
        and row.get("model_selection_eligible") is True
    }
    if eligible_ids != expected_ids:
        raise ValueError("V2-C3 eligible registry roles changed")

    for expected in EXPECTED_ELIGIBLE_SOURCE_ROLES:
        record = records.get(expected["record_id"])
        if not isinstance(record, dict):
            raise TypeError(f"missing registry record: {expected['record_id']}")
        checks = {
            "source_id": expected["source_id"],
            "source_split": EXPECTED_REGISTRY_SPLITS[expected["record_id"]],
            "current_role": expected["registry_role"],
            "training_eligible": True,
            "model_selection_eligible": True,
        }
        if any(record.get(key) != value for key, value in checks.items()):
            raise ValueError(
                f"registry eligibility changed for {expected['record_id']}"
            )


def validate_development_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "v2c3-dataset-manifest.v1":
        raise ValueError("unexpected V2-C3 dataset-manifest schema")
    output_hashes = manifest.get("output_hashes")
    if not isinstance(output_hashes, dict):
        raise TypeError("V2-C3 output hashes must be an object")
    if (
        output_hashes.get("v2c3_development_dataset.json")
        != DEVELOPMENT_DATASET_SHA256
    ):
        raise ValueError("V2-C3 development dataset hash changed in manifest")
    expected_normalization = {
        "hash_algorithm": "SHA-256",
        "steps": development_builder.NORMALIZATION_STEPS,
        "version": development_builder.NORMALIZATION_VERSION,
    }
    if manifest.get("normalization") != expected_normalization:
        raise ValueError("V2-C3 normalization convention changed")
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise TypeError("V2-C3 manifest validation must be an object")
    required_false = {
        "banking77_test_used",
        "cfpb_used",
        "clinc_oos_used",
        "clinc_test_used",
        "internal_locked_test_used",
        "prohibited_sources_used",
    }
    if any(validation.get(name) is not False for name in required_false):
        raise ValueError("V2-C3 development manifest admits excluded sources")
    if validation.get("development_lockbox_normalized_hash_overlap") != 0:
        raise ValueError("V2-C3 development/lockbox text overlap changed")


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _record_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    source_id = str(row.get("source_id"))
    source_split = str(row.get("source_split"))
    row_index = row.get("source_row_index")
    index_key = (
        (0, row_index)
        if isinstance(row_index, int) and not isinstance(row_index, bool)
        else (1, 0)
    )
    return (
        SOURCE_PRECEDENCE.get(source_id, 99),
        SPLIT_PRECEDENCE.get(source_split, 99),
        index_key,
        str(row.get("example_id")),
    )


def validate_source_record(
    row: Mapping[str, Any],
    roles: Mapping[tuple[str, str], str],
) -> None:
    if row.get("data_role") != "development":
        raise ValueError("non-development or final/test role entered discovery")
    key = (str(row.get("source_id")), str(row.get("source_split")))
    if key not in roles:
        raise ValueError(f"ineligible source role entered discovery: {key}")
    intent = row.get("intent")
    if intent not in EXPECTED_INTENTS:
        raise ValueError(f"unknown current SentinelVoice intent: {intent}")
    text = _require_nonempty_string(row.get("text"), "source text")
    example_id = _require_nonempty_string(row.get("example_id"), "example_id")
    group_id = _require_nonempty_string(row.get("group_id"), "group_id")
    del example_id, group_id
    if row.get("text_sha256") != development_builder.text_sha256(text):
        raise ValueError("source text SHA-256 mismatch")
    normalized_digest = development_builder.normalized_text_sha256(text)
    if row.get("normalized_text_sha256") != normalized_digest:
        raise ValueError("source normalized-text SHA-256 mismatch")
    mapping_status = row.get("mapping_status")
    if not isinstance(mapping_status, str) or not mapping_status:
        raise ValueError("source mapping status is missing")
    source_label = row.get("source_label")
    if not isinstance(source_label, str) or not source_label:
        raise ValueError("source label is missing")


def validate_development_dataset(
    dataset: Mapping[str, Any],
    roles: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    if dataset.get("schema_version") != "v2c3-development-dataset.v1":
        raise ValueError("unexpected V2-C3 development-dataset schema")
    if (
        dataset.get("normalization_version")
        != development_builder.NORMALIZATION_VERSION
    ):
        raise ValueError("V2-C3 dataset normalization version changed")
    if dataset.get("training_or_evaluation_performed") is not False:
        raise ValueError("V2-C3 source execution policy changed")
    examples_value = dataset.get("examples")
    if not isinstance(examples_value, list) or not examples_value:
        raise ValueError("V2-C3 development examples must be a non-empty list")
    if dataset.get("example_count") != len(examples_value):
        raise ValueError("V2-C3 development example count mismatch")

    examples: list[dict[str, Any]] = []
    example_ids: set[str] = set()
    normalized_intents: dict[str, str] = {}
    for row_value in examples_value:
        if not isinstance(row_value, dict):
            raise TypeError("V2-C3 development example must be an object")
        validate_source_record(row_value, roles)
        example_id = str(row_value["example_id"])
        if example_id in example_ids:
            raise ValueError(f"duplicate source example ID: {example_id}")
        example_ids.add(example_id)
        digest = str(row_value["normalized_text_sha256"])
        intent = str(row_value["intent"])
        previous = normalized_intents.setdefault(digest, intent)
        if previous != intent:
            raise ValueError("normalized duplicate has conflicting current intent")
        annotated = dict(row_value)
        source_key = (
            str(annotated["source_id"]),
            str(annotated["source_split"]),
        )
        annotated["source_dataset"] = annotated["source_id"]
        annotated["source_role"] = roles[source_key]
        examples.append(annotated)
    return examples


def occurrence_metadata(
    row: Mapping[str, Any],
    roles: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    source_id = str(row["source_id"])
    source_split = str(row["source_split"])
    is_external = source_id != "sentinelvoice_v2c1_internal"
    tags_value = row.get("tags")
    tags = (
        sorted(str(tag) for tag in tags_value)
        if isinstance(tags_value, list)
        else []
    )
    design_value = row.get("design_metadata")
    design_metadata = dict(design_value) if isinstance(design_value, dict) else {}
    return {
        "data_role": row["data_role"],
        "design_metadata": design_metadata,
        "example_id": row["example_id"],
        "group_id": row["group_id"],
        "lineage_id": row.get("lineage_id"),
        "mapping_status": row["mapping_status"],
        "native_external_label": row["source_label"] if is_external else None,
        "native_external_label_clustering_feature": False,
        "native_external_mapping_status": (
            row["mapping_status"] if is_external else None
        ),
        "original_example_id": row.get("original_example_id"),
        "original_split": row.get("original_split"),
        "source_dataset": source_id,
        "source_domain": row.get("source_domain"),
        "source_revision": row.get("source_revision"),
        "source_role": roles[(source_id, source_split)],
        "source_row_index": row.get("source_row_index"),
        "source_split": source_split,
        "tags": tags,
        "text_sha256": row["text_sha256"],
    }


def build_primary_population(
    examples: Sequence[Mapping[str, Any]],
    roles: Mapping[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    role_map = dict(roles or source_role_by_key())
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in examples:
        validate_source_record(row, role_map)
        if row["intent"] == PRIMARY_INTENT:
            grouped[str(row["normalized_text_sha256"])].append(row)

    primary: list[dict[str, Any]] = []
    discovery_ids: set[str] = set()
    for digest in sorted(grouped):
        occurrences = sorted(grouped[digest], key=_record_sort_key)
        if {row["intent"] for row in occurrences} != {PRIMARY_INTENT}:
            raise ValueError("supported intent entered primary population")
        canonical = occurrences[0]
        normalized_text = development_builder.normalize_text(canonical["text"])
        if (
            development_builder.sha256_bytes(normalized_text.encode("utf-8"))
            != digest
        ):
            raise ValueError("canonical normalized text hash mismatch")
        discovery_id = f"v2c5-discovery:{digest}"
        if discovery_id in discovery_ids:
            raise ValueError(f"discovery ID collision: {discovery_id}")
        discovery_ids.add(discovery_id)
        occurrence_values = [
            occurrence_metadata(row, role_map) for row in occurrences
        ]
        primary.append(
            {
                "all_original_example_ids": sorted(
                    {
                        str(row["original_example_id"])
                        for row in occurrences
                        if row.get("original_example_id") is not None
                    }
                ),
                "all_source_datasets": sorted(
                    {str(row["source_id"]) for row in occurrences}
                ),
                "current_risk": canonical.get("risk"),
                "current_sentinelvoice_intent": PRIMARY_INTENT,
                "current_sentinelvoice_intent_clustering_feature": False,
                "discovery_id": discovery_id,
                "normalized_text": normalized_text,
                "normalized_text_sha256": digest,
                "occurrences": occurrence_values,
                "source_occurrence_count": len(occurrence_values),
                "text": canonical["text"],
                "text_sha256": canonical["text_sha256"],
            }
        )
    return primary


def counts_by(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = ":".join(str(row[field]) for field in fields)
        counts[key] += 1
    return dict(sorted(counts.items()))


def reference_anchor_summary(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchors = sorted(
        (row for row in examples if row["intent"] in SUPPORTED_INTENTS),
        key=_record_sort_key,
    )
    return {
        "counts_by_intent": counts_by(anchors, ["intent"]),
        "counts_by_source_dataset": counts_by(anchors, ["source_id"]),
        "data_role": REFERENCE_DATA_ROLE,
        "density_driving_cluster_members": False,
        "example_count": len(anchors),
        "materialization": "hash_pinned_source_reference_only",
        "ordered_example_id_sha256": sha256_lines(
            [str(row["example_id"]) for row in anchors]
        ),
        "ordered_normalized_text_sha256": sha256_lines(
            [str(row["normalized_text_sha256"]) for row in anchors]
        ),
        "records_copied_into_corpus": False,
        "selection": {
            "current_sentinelvoice_intents": SUPPORTED_INTENTS,
            "data_role": "development",
            "eligible_source_roles": EXPECTED_ELIGIBLE_SOURCE_ROLES,
        },
        "source_dataset_path": (
            "data/evals/v2/ml/v2c3_development_dataset.json"
        ),
        "source_dataset_sha256": DEVELOPMENT_DATASET_SHA256,
    }


def execution_status() -> dict[str, bool]:
    return {
        "classifier_training_performed": False,
        "clustering_performed": False,
        "discovery_corpus_built": True,
        "embeddings_generated": False,
        "human_adjudication_performed": False,
        "taxonomy_changed": False,
        "umap_performed": False,
        "v2c5_holdout_created": False,
    }


def build_corpus_payload(
    primary: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_sha256": CONTRACT_SHA256,
        "data_role": PRIMARY_DATA_ROLE,
        "embedding_input_policy": {
            "fields": ["text"],
            "labels_concatenated_with_text": False,
            "native_labels_are_features": False,
            "source_metadata_concatenated_with_text": False,
        },
        "execution_status": execution_status(),
        "phase": "V2-C5",
        "primary_cluster_population": list(primary),
        "reference_anchor_population": dict(anchors),
        "runtime_scope": {
            "classifier_routing_role": "advisory_only",
            "new_intents_created": False,
            "new_protected_actions_inferred": False,
            "new_runtime_tools_added": False,
            "runtime_behavior_changed": False,
        },
        "schema_version": CORPUS_SCHEMA_VERSION,
    }


def primary_occurrences(
    primary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(occurrence)
        for record in primary
        for occurrence in record["occurrences"]
    ]


def source_count_summary(
    all_examples: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, dict[str, int]]:
    occurrences = primary_occurrences(primary)
    canonical_by_key: Counter[str] = Counter()
    for record in primary:
        canonical = record["occurrences"][0]
        key = ":".join(str(canonical[field]) for field in fields)
        canonical_by_key[key] += 1
    eligible = counts_by(all_examples, fields)
    unsupported = counts_by(occurrences, fields)
    keys = sorted(set(eligible) | set(unsupported) | set(canonical_by_key))
    return {
        key: {
            "eligible_occurrences": eligible.get(key, 0),
            "primary_canonical_records": canonical_by_key.get(key, 0),
            "primary_unsupported_occurrences": unsupported.get(key, 0),
        }
        for key in keys
    }


def build_manifest_payload(
    *,
    all_examples: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Any],
    corpus_bytes: bytes,
    paths: BuildPaths,
    script_path: Path,
) -> dict[str, Any]:
    occurrence_counts = Counter(
        int(record["source_occurrence_count"]) for record in primary
    )
    duplicate_records = [
        record for record in primary if record["source_occurrence_count"] > 1
    ]
    cross_source_duplicate_groups = sum(
        len(record["all_source_datasets"]) > 1 for record in duplicate_records
    )
    primary_occurrence_values = primary_occurrences(primary)
    external_native_count = sum(
        occurrence["native_external_label"] is not None
        for occurrence in primary_occurrence_values
    )
    normalized_order = [
        str(record["normalized_text_sha256"]) for record in primary
    ]
    discovery_id_order = [str(record["discovery_id"]) for record in primary]
    corpus_sha256 = development_builder.sha256_bytes(corpus_bytes)
    return {
        "builder": {
            "path": display_path(script_path),
            "sha256": development_builder.sha256_bytes(script_path.read_bytes()),
            "version": BUILDER_VERSION,
        },
        "canonical_representative_rule": {
            "input_iteration_order_used": False,
            "source_precedence": [
                "sentinelvoice_v2c1_internal",
                "banking77",
                "clinc150_oos",
            ],
            "tie_breaking": [
                "source_precedence",
                "train before validation or val",
                "source_row_index ascending with nulls last",
                "example_id lexicographic",
            ],
        },
        "contract": {
            "path": display_path(paths.contract),
            "sha256": CONTRACT_SHA256,
            "status": "frozen",
        },
        "corpus": {
            "path": display_path(paths.corpus_output),
            "sha256": corpus_sha256,
        },
        "counts": {
            "cross_source_duplicate_group_count": cross_source_duplicate_groups,
            "duplicate_group_count": len(duplicate_records),
            "duplicate_occurrence_count_removed_from_density": (
                len(primary_occurrence_values) - len(primary)
            ),
            "primary_unique_normalized_text_count": len(primary),
            "primary_unsupported_occurrence_count_before_dedup": len(
                primary_occurrence_values
            ),
            "raw_eligible_occurrence_count": len(all_examples),
            "reference_anchor_count": anchors["example_count"],
        },
        "counts_by_source_dataset": source_count_summary(
            all_examples,
            primary,
            ["source_dataset"],
        )
        if primary_occurrence_values
        else {},
        "counts_by_source_role_and_split": source_count_summary(
            all_examples,
            primary,
            ["source_role", "source_split"],
        )
        if primary_occurrence_values
        else {},
        "execution_status": execution_status(),
        "exclusion_and_integrity_checks": {
            "banking77_test_records": 0,
            "cfpb_records": 0,
            "clinc_test_records": 0,
            "consumed_v2c3_challenge_records": 0,
            "consumed_v2c3_external_lockbox_records": 0,
            "current_intent_labels_are_clustering_features": False,
            "discovery_id_collisions": 0,
            "duplicate_normalized_text_vector_records": 0,
            "final_or_test_role_records": 0,
            "native_external_labels_are_clustering_features": False,
            "sealed_v2c4_holdout_accessed": False,
            "supported_intents_in_primary_population": 0,
            "v2c3_internal_locked_test_records": 0,
            "v2c4_postmortem_review_records": 0,
            "v2c4_selection_probe_records": 0,
            "v2c4_targeted_augmentation_records": 0,
            "verification_basis": (
                "Pinned V2-C3 development artifact and fail-closed source-role "
                "validation; excluded raw datasets were not read."
            ),
        },
        "native_label_availability": {
            "primary_occurrences_with_native_external_label": (
                external_native_count
            ),
            "primary_occurrences_without_native_external_label": (
                len(primary_occurrence_values) - external_native_count
            ),
            "retained_as_metadata_only": True,
        },
        "normalization": {
            "hash_algorithm": "SHA-256",
            "steps": development_builder.NORMALIZATION_STEPS,
            "version": development_builder.NORMALIZATION_VERSION,
        },
        "occurrence_count_distribution": {
            str(count): record_count
            for count, record_count in sorted(occurrence_counts.items())
        },
        "ordered_hashes": {
            "algorithm": (
                "SHA-256 over ordered UTF-8 values, one per line, with "
                "trailing newline"
            ),
            "canonical_discovery_id_ordered_sha256": sha256_lines(
                discovery_id_order
            ),
            "normalized_text_ordered_sha256": sha256_lines(normalized_order),
        },
        "reference_anchor_population": dict(anchors),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_input_hashes": {
            "v2c3_data_registry": DATA_REGISTRY_SHA256,
            "v2c3_dataset_manifest": DEVELOPMENT_MANIFEST_SHA256,
            "v2c3_development_dataset": DEVELOPMENT_DATASET_SHA256,
            "v2c5_intent_discovery_contract": CONTRACT_SHA256,
        },
        "source_input_paths": {
            "v2c3_data_registry": display_path(paths.data_registry),
            "v2c3_dataset_manifest": display_path(paths.development_manifest),
            "v2c3_development_dataset": display_path(paths.development_dataset),
            "v2c5_intent_discovery_contract": display_path(paths.contract),
        },
        "source_roles": EXPECTED_ELIGIBLE_SOURCE_ROLES,
    }


def build_artifacts(
    paths: BuildPaths = DEFAULT_PATHS,
    script_path: Path | None = None,
) -> BuiltArtifacts:
    contract_bytes = read_input_bytes(paths.contract)
    development_bytes = read_input_bytes(paths.development_dataset)
    manifest_bytes = read_input_bytes(paths.development_manifest)
    registry_bytes = read_input_bytes(paths.data_registry)
    input_hashes = {
        "development dataset": (
            development_builder.sha256_bytes(development_bytes),
            DEVELOPMENT_DATASET_SHA256,
        ),
        "development manifest": (
            development_builder.sha256_bytes(manifest_bytes),
            DEVELOPMENT_MANIFEST_SHA256,
        ),
        "data registry": (
            development_builder.sha256_bytes(registry_bytes),
            DATA_REGISTRY_SHA256,
        ),
    }
    for label, (actual, expected) in input_hashes.items():
        if actual != expected:
            raise ValueError(f"frozen {label} SHA-256 changed")

    contract = load_json_object_bytes(contract_bytes, "V2-C5 contract")
    development = load_json_object_bytes(
        development_bytes,
        "V2-C3 development dataset",
    )
    development_manifest = load_json_object_bytes(
        manifest_bytes,
        "V2-C3 development manifest",
    )
    registry = load_json_object_bytes(registry_bytes, "V2-C3 data registry")
    validate_contract(contract, contract_bytes)
    validate_registry(registry)
    validate_development_manifest(development_manifest)
    roles = source_role_by_key()
    examples = validate_development_dataset(development, roles)
    primary = build_primary_population(examples, roles)
    anchors = reference_anchor_summary(examples)
    corpus_payload = build_corpus_payload(primary, anchors)
    corpus_bytes = development_builder.stable_json_bytes(corpus_payload)
    actual_script_path = script_path or Path(__file__).resolve()
    manifest_payload = build_manifest_payload(
        all_examples=examples,
        primary=primary,
        anchors=anchors,
        corpus_bytes=corpus_bytes,
        paths=paths,
        script_path=actual_script_path,
    )
    return BuiltArtifacts(
        corpus_bytes=corpus_bytes,
        manifest_bytes=development_builder.stable_json_bytes(manifest_payload),
        corpus_payload=corpus_payload,
        manifest_payload=manifest_payload,
    )


def write_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    development_builder.atomic_write_bytes(
        paths.corpus_output,
        artifacts.corpus_bytes,
    )
    development_builder.atomic_write_bytes(
        paths.manifest_output,
        artifacts.manifest_bytes,
    )


def check_artifacts(
    artifacts: BuiltArtifacts,
    paths: BuildPaths = DEFAULT_PATHS,
) -> None:
    expected = {
        paths.corpus_output: artifacts.corpus_bytes,
        paths.manifest_output: artifacts.manifest_bytes,
    }
    for path, payload in expected.items():
        if not path.exists():
            raise FileNotFoundError(
                f"V2-C5 discovery artifact is missing: {path}; use --write"
            )
        if path.read_bytes() != payload:
            raise ValueError(
                "V2-C5 discovery artifact differs from deterministic build: "
                f"{path}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="write the deterministic corpus and manifest",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the corpus and manifest byte-for-byte",
    )
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
        f"{action} V2-C5 discovery artifacts: "
        f"primary_occurrences="
        f"{counts['primary_unsupported_occurrence_count_before_dedup']}, "
        f"primary_unique="
        f"{counts['primary_unique_normalized_text_count']}, "
        f"reference_anchors={counts['reference_anchor_count']}; "
        "no embeddings, clustering, UMAP, taxonomy change, or training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
