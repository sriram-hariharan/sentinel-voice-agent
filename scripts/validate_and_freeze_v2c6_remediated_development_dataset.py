"""Validate and freeze the V2-C6 remediated development dataset.

This is a standard-library-only data-governance step. It performs no model,
embedding, inference, selection, evaluation, or threshold-tuning work. The
consumed V2-C5 final holdout is prohibited on every code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = (
    "scripts/validate_and_freeze_v2c6_remediated_development_dataset.py"
)

EXPECTED_SOURCE_SHA256 = {
    "combined_manifest": (
        "e06bbeb944a19a1ac67eb1960d58f1e40f6f8b09e4642eb17a822f26510fc260"
    ),
    "contract": (
        "2063e6ff0b27caaa4d7b6bbe12748e1e4a745bf446f4502589dbb36a966f9d4d"
    ),
    "inherited_dataset": (
        "dce97c0a3bfcdf0dee93d784ded24d3568f7df8f513871995428ff35d030bb07"
    ),
    "inherited_manifest": (
        "89fea89d793cf5e027b059c331533ec45c6d59fabbd49e9245e0d055ae946a16"
    ),
    "remediation_manifest": (
        "8f17cf48ba16d49ea7b3964cb1a1972876a4b2a87bc481bf9b86fac3f6aae71a"
    ),
    "step29e_builder": (
        "371cb8ba03a55c5d9f8e278e3b3f6934dfc9f727ed090483e88332a5f4164a1d"
    ),
    "taxonomy": (
        "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
    ),
    "taxonomy_manifest": (
        "359eb38e5ff63af0c9cc0f5db19951b8626f9e15cb136f23046648d655a0f7a9"
    ),
}

EXPECTED_TOTAL_COUNT = 9008
EXPECTED_INHERITED_COUNT = 8198
EXPECTED_REMEDIATION_COUNT = 810
NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace.v1"
FREEZE_SCHEMA_VERSION = "v2c6-remediated-development-dataset-freeze.v1"

INTENT_LABEL_ORDER = (
    "account_balance",
    "account_blocked",
    "cancel_transfer",
    "card_status",
    "close_account",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "lost_or_stolen_phone",
    "passcode_recovery",
    "recent_transactions",
    "transaction_details",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
)
PRIMARY_REMEDIATION_INTENTS = (
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
)
SUPPORTED_PRIMARY_INTENTS = PRIMARY_REMEDIATION_INTENTS[:-1]
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
PROTECTED_WRITE_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)
EXPECTED_SOURCE_FAMILIES = (
    "v2c6_sf1_definition_direct",
    "v2c6_sf2_scenario_narrative",
    "v2c6_sf3_boundary_conversational",
)
UNSUPPORTED_SUBTYPES = (
    "adjacent_but_unsupported_intent",
    "ambiguous_or_insufficient_information",
    "off_domain_or_noise",
    "supported_intent_hard_negative",
    "truly_unsupported_banking_request",
)
HARD_NEGATIVE_PAIRS = (
    ("close_account", UNSUPPORTED_INTENT),
    ("transfer_failed_or_declined", UNSUPPORTED_INTENT),
    ("account_blocked", UNSUPPORTED_INTENT),
    ("cancel_transfer", UNSUPPORTED_INTENT),
    ("create_dispute", UNSUPPORTED_INTENT),
    ("freeze_card", UNSUPPORTED_INTENT),
    ("transfer_pending", UNSUPPORTED_INTENT),
    ("cancel_transfer", "transfer_pending"),
    ("transfer_failed_or_declined", "transfer_pending"),
    ("account_blocked", "transfer_failed_or_declined"),
)

BytesReader = Callable[[Path], bytes]


@dataclass(frozen=True)
class FreezePaths:
    repository_root: Path
    dataset: Path
    combined_manifest: Path
    remediation_examples: Path
    remediation_manifest: Path
    contract: Path
    inherited_dataset: Path
    inherited_manifest: Path
    taxonomy: Path
    taxonomy_manifest: Path
    step29e_builder: Path
    validation_script: Path
    freeze_output: Path
    prohibited_holdout: Path


DEFAULT_PATHS = FreezePaths(
    repository_root=REPOSITORY_ROOT,
    dataset=ML_DIRECTORY / "v2c6_remediated_development_dataset.json",
    combined_manifest=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.manifest.json"
    ),
    remediation_examples=ML_DIRECTORY / "v2c6_remediation_examples.json",
    remediation_manifest=(
        ML_DIRECTORY / "v2c6_remediation_examples.manifest.json"
    ),
    contract=ML_DIRECTORY / "v2c6_remediation_dataset_contract.json",
    inherited_dataset=ML_DIRECTORY / "v2c5_expanded_development_dataset.json",
    inherited_manifest=(
        ML_DIRECTORY / "v2c5_expanded_development_dataset.manifest.json"
    ),
    taxonomy=ML_DIRECTORY / "v2c5_taxonomy_freeze.json",
    taxonomy_manifest=ML_DIRECTORY / "v2c5_taxonomy_freeze.manifest.json",
    step29e_builder=(
        REPOSITORY_ROOT / "scripts/build_v2c6_remediated_development_dataset.py"
    ),
    validation_script=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
    freeze_output=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.freeze.json"
    ),
    prohibited_holdout=ML_DIRECTORY / "v2c5_final_holdout.json",
)


@dataclass(frozen=True)
class LoadedInputs:
    payloads: dict[str, dict[str, Any]]
    raw_bytes: dict[str, bytes]
    sha256: dict[str, str]


@dataclass(frozen=True)
class PreparedFreeze:
    report: dict[str, Any]
    freeze: dict[str, Any]
    freeze_bytes: bytes


def filesystem_reader(path: Path) -> bytes:
    return path.read_bytes()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def text_sha256(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def normalized_text_sha256(value: str) -> str:
    return sha256_bytes(normalized_text(value).encode("utf-8"))


def display_path(path: Path, paths: FreezePaths) -> str:
    try:
        return path.resolve().relative_to(paths.repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def guarded_read_bytes(
    path: Path,
    paths: FreezePaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> bytes:
    resolved = path.resolve()
    if (
        resolved == paths.prohibited_holdout.resolve()
        or resolved.name == "v2c5_final_holdout.json"
    ):
        raise PermissionError("V2-C5 final holdout access is prohibited")
    return reader(path)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _require_object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{label} must be a list of objects")
    return value


def _require_nonempty_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def source_paths(paths: FreezePaths) -> dict[str, Path]:
    return {
        "combined_manifest": paths.combined_manifest,
        "contract": paths.contract,
        "dataset": paths.dataset,
        "inherited_dataset": paths.inherited_dataset,
        "inherited_manifest": paths.inherited_manifest,
        "remediation_examples": paths.remediation_examples,
        "remediation_manifest": paths.remediation_manifest,
        "step29e_builder": paths.step29e_builder,
        "taxonomy": paths.taxonomy,
        "taxonomy_manifest": paths.taxonomy_manifest,
    }


def load_inputs(
    paths: FreezePaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> LoadedInputs:
    raw_bytes: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in source_paths(paths).items():
        value = guarded_read_bytes(path, paths, reader)
        digest = sha256_bytes(value)
        expected = EXPECTED_SOURCE_SHA256.get(name)
        if expected is not None and digest != expected:
            raise ValueError(f"frozen source SHA-256 mismatch: {name}")
        raw_bytes[name] = value
        hashes[name] = digest
        if name != "step29e_builder":
            payloads[name] = _require_object(json.loads(value), name)
    return LoadedInputs(payloads=payloads, raw_bytes=raw_bytes, sha256=hashes)


def validate_hash_lineage(inputs: LoadedInputs) -> dict[str, Any]:
    payloads = inputs.payloads
    combined_manifest = payloads["combined_manifest"]
    remediation_manifest = payloads["remediation_manifest"]
    dataset = payloads["dataset"]
    remediation = payloads["remediation_examples"]
    inherited_manifest = payloads["inherited_manifest"]
    taxonomy_manifest = payloads["taxonomy_manifest"]

    if combined_manifest.get("schema_version") != (
        "v2c6-remediated-development-dataset-manifest.v1"
    ) or (
        combined_manifest.get("phase") != "V2-C6 Step 29E2"
        or combined_manifest.get("next_required")
        != "v2c6_remediated_development_dataset_validation_and_freeze"
    ):
        raise ValueError("unexpected combined manifest schema")
    if remediation_manifest.get("schema_version") != (
        "v2c6-remediation-examples-manifest.v1"
    ) or remediation_manifest.get("phase") != "V2-C6 Step 29E2":
        raise ValueError("unexpected remediation manifest schema")
    dataset_binding = combined_manifest.get("remediated_development_dataset")
    if dataset_binding != {
        "path": "data/evals/v2/ml/v2c6_remediated_development_dataset.json",
        "schema_version": "v2c6-remediated-development-dataset.v1",
        "sha256": inputs.sha256["dataset"],
    }:
        raise ValueError("combined dataset hash disagrees with its manifest")
    remediation_binding = remediation_manifest.get("remediation_examples")
    if remediation_binding != {
        "path": "data/evals/v2/ml/v2c6_remediation_examples.json",
        "schema_version": "v2c6-remediation-examples.v1",
        "sha256": inputs.sha256["remediation_examples"],
    }:
        raise ValueError("remediation artifact hash disagrees with its manifest")
    if combined_manifest.get("remediation_examples", {}).get("sha256") != (
        inputs.sha256["remediation_examples"]
    ):
        raise ValueError("combined manifest remediation hash changed")
    if dataset.get("source_artifacts", {}).get(
        "approved_remediation_examples", {}
    ).get("sha256") != inputs.sha256["remediation_examples"]:
        raise ValueError("combined dataset remediation lineage changed")
    if remediation.get("source_contract", {}).get("sha256") != inputs.sha256[
        "contract"
    ]:
        raise ValueError("remediation contract lineage changed")
    if remediation_manifest.get("step29d_contract", {}).get("sha256") != (
        inputs.sha256["contract"]
    ):
        raise ValueError("remediation manifest contract lineage changed")
    combined_sources = combined_manifest.get("source_artifacts", {})
    if combined_sources.get("step29d_contract", {}).get("sha256") != inputs.sha256[
        "contract"
    ]:
        raise ValueError("combined manifest contract lineage changed")
    if combined_sources.get("frozen_v2c5_development", {}).get("sha256") != (
        inputs.sha256["inherited_dataset"]
    ):
        raise ValueError("inherited development lineage changed")
    if dataset.get("source_artifacts", {}).get(
        "frozen_v2c5_development", {}
    ).get("sha256") != inputs.sha256["inherited_dataset"]:
        raise ValueError("combined dataset inherited lineage changed")
    if inherited_manifest.get("dataset", {}).get("sha256") != inputs.sha256[
        "inherited_dataset"
    ]:
        raise ValueError("inherited dataset manifest binding changed")
    if inherited_manifest.get("schema_version") != (
        "v2c5-expanded-development-dataset-manifest.v1"
    ):
        raise ValueError("unexpected inherited manifest schema")
    if combined_sources.get("taxonomy", {}).get("sha256") != inputs.sha256[
        "taxonomy"
    ]:
        raise ValueError("combined manifest taxonomy lineage changed")
    if taxonomy_manifest.get("taxonomy_freeze", {}).get("sha256") != inputs.sha256[
        "taxonomy"
    ]:
        raise ValueError("taxonomy manifest binding changed")
    if taxonomy_manifest.get("schema_version") != (
        "v2c5-taxonomy-freeze-manifest.v1"
    ):
        raise ValueError("unexpected taxonomy manifest schema")
    for manifest_name in ("combined_manifest", "remediation_manifest"):
        if payloads[manifest_name].get("builder", {}).get("sha256") != inputs.sha256[
            "step29e_builder"
        ]:
            raise ValueError(f"Step 29E builder lineage changed: {manifest_name}")
    return {
        name: {
            "path": name,
            "sha256": digest,
        }
        for name, digest in sorted(inputs.sha256.items())
    }


def validate_taxonomy(inputs: LoadedInputs) -> tuple[dict[str, str], str]:
    taxonomy = inputs.payloads["taxonomy"]
    if (
        taxonomy.get("schema_version") != "v2c5-taxonomy-freeze.v1"
        or taxonomy.get("taxonomy_version") != "v2c5-taxonomy.v1"
        or taxonomy.get("final_taxonomy_frozen") is not True
        or tuple(taxonomy.get("intent_label_order", [])) != INTENT_LABEL_ORDER
        or taxonomy.get("final_intent_count") != len(INTENT_LABEL_ORDER)
        or tuple(taxonomy.get("protected_write_intents", []))
        != PROTECTED_WRITE_INTENTS
    ):
        raise ValueError("frozen taxonomy identity changed")
    risk_mapping = taxonomy.get("risk_by_intent")
    if not isinstance(risk_mapping, dict) or set(risk_mapping) != set(
        INTENT_LABEL_ORDER
    ):
        raise ValueError("frozen risk mapping changed")
    risk_mapping_sha256 = sha256_bytes(stable_json_bytes(risk_mapping))
    return {str(key): str(value) for key, value in risk_mapping.items()}, (
        risk_mapping_sha256
    )


def expected_combined_remediation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    text = str(record["text"])
    return {
        "authoring_batch_id": record["authoring_batch_id"],
        "authoring_method": record["authoring_method"],
        "boundary_target": record["boundary_target"],
        "data_role": "development",
        "example_id": record["record_id"],
        "group_id": record["group_id"],
        "intent": record["intent"],
        "is_hard_negative": record["is_hard_negative"],
        "normalized_text_sha256": normalized_text_sha256(text),
        "record_id": record["record_id"],
        "review_status": record["review_status"],
        "risk": record["risk_level"],
        "risk_level": record["risk_level"],
        "source_domain": "sentinelvoice_v2c6_remediation",
        "source_family_id": record["source_family_id"],
        "source_family_independence_basis": record[
            "source_family_independence_basis"
        ],
        "source_id": record["source_family_id"],
        "source_revision": record["source_revision"],
        "source_split": "development",
        "text": text,
        "text_sha256": text_sha256(text),
        "unsupported_subtype": record.get("unsupported_subtype"),
        "v2c6_lineage": "REMEDIATION_ADDITION",
    }


def count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def validate_inherited_prefix(
    combined_records: Sequence[Mapping[str, Any]],
    inherited_records: Sequence[Mapping[str, Any]],
) -> None:
    if list(combined_records[: len(inherited_records)]) != list(inherited_records):
        raise ValueError("inherited V2-C5 development records changed")


def validate_group_structure(
    inherited_records: Sequence[Mapping[str, Any]],
    remediation_records: Sequence[Mapping[str, Any]],
    combined_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    inherited_group_ids = [
        _require_nonempty_string(record, "group_id")
        for record in inherited_records
    ]
    remediation_group_ids = [
        _require_nonempty_string(record, "group_id")
        for record in remediation_records
    ]
    if len(set(remediation_group_ids)) != len(remediation_group_ids):
        raise ValueError("remediation group IDs must be unique")
    inherited_group_set = set(inherited_group_ids)
    collisions = inherited_group_set & set(remediation_group_ids)
    if collisions:
        raise ValueError("remediation group ID collides with inherited group ID")
    combined_remediation = combined_records[len(inherited_records) :]
    if len(combined_remediation) != len(remediation_records):
        raise ValueError("combined remediation record count changed")
    combined_remediation_group_ids = [
        _require_nonempty_string(record, "group_id")
        for record in combined_remediation
    ]
    if combined_remediation_group_ids != remediation_group_ids:
        raise ValueError("combined remediation group IDs were not preserved")

    inherited_group_counts = Counter(inherited_group_ids)
    inherited_group_intents: defaultdict[str, set[str]] = defaultdict(set)
    for record in inherited_records:
        inherited_group_intents[str(record["group_id"])].add(
            _require_nonempty_string(record, "intent")
        )
    return {
        "group_aware_split_unit": "group_id",
        "group_ids_are_indivisible_split_units": True,
        "inherited_cross_intent_group_count": sum(
            len(intents) > 1 for intents in inherited_group_intents.values()
        ),
        "inherited_group_label_purity_required": False,
        "inherited_group_membership_preserved": True,
        "inherited_multi_record_group_count": sum(
            count > 1 for count in inherited_group_counts.values()
        ),
        "inherited_unique_group_count": len(inherited_group_counts),
        "remediation_group_collision_with_inherited_count": len(collisions),
        "remediation_group_ids_preserved": True,
        "remediation_group_ids_unique": True,
        "remediation_unique_group_count": len(set(remediation_group_ids)),
    }


def validate_composition(
    inputs: LoadedInputs,
    risk_mapping: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = inputs.payloads
    dataset = payloads["dataset"]
    inherited = payloads["inherited_dataset"]
    remediation = payloads["remediation_examples"]
    combined_manifest = payloads["combined_manifest"]
    remediation_manifest = payloads["remediation_manifest"]
    if (
        dataset.get("schema_version")
        != "v2c6-remediated-development-dataset.v1"
        or dataset.get("phase") != "V2-C6 Step 29E2"
        or dataset.get("dataset_version") != "v2c6-remediated-development.v1"
        or dataset.get("normalization_version") != NORMALIZATION_VERSION
        or dataset.get("ordering") != "frozen_v2c5_order_then_v2c6_record_id"
        or dataset.get("taxonomy_version") != "v2c5-taxonomy.v1"
    ):
        raise ValueError("unexpected remediated development dataset identity")
    if (
        remediation.get("schema_version") != "v2c6-remediation-examples.v1"
        or remediation.get("phase") != "V2-C6 Step 29E2"
        or remediation.get("normalization_version") != NORMALIZATION_VERSION
    ):
        raise ValueError("unexpected remediation artifact identity")
    combined_records = _require_object_list(dataset.get("examples"), "examples")
    inherited_records = _require_object_list(
        inherited.get("examples"), "inherited examples"
    )
    remediation_records = _require_object_list(
        remediation.get("examples"), "remediation examples"
    )
    if dataset.get("example_count") != EXPECTED_TOTAL_COUNT or len(
        combined_records
    ) != EXPECTED_TOTAL_COUNT:
        raise ValueError("combined dataset must contain exactly 9008 records")
    if inherited.get("example_count") != EXPECTED_INHERITED_COUNT or len(
        inherited_records
    ) != EXPECTED_INHERITED_COUNT:
        raise ValueError("inherited dataset must contain exactly 8198 records")
    if remediation.get("example_count") != EXPECTED_REMEDIATION_COUNT or len(
        remediation_records
    ) != EXPECTED_REMEDIATION_COUNT:
        raise ValueError("remediation artifact must contain exactly 810 records")
    validate_inherited_prefix(combined_records, inherited_records)
    expected_new = [
        expected_combined_remediation_record(record)
        for record in remediation_records
    ]
    if combined_records[EXPECTED_INHERITED_COUNT:] != expected_new:
        raise ValueError("combined remediation records do not match their source")
    combined_ids = [
        _require_nonempty_string(record, "example_id")
        for record in combined_records
    ]
    if len(set(combined_ids)) != EXPECTED_TOTAL_COUNT:
        raise ValueError("combined example IDs are not unique")
    remediation_ids = [
        _require_nonempty_string(record, "record_id") for record in remediation_records
    ]
    if len(set(remediation_ids)) != EXPECTED_REMEDIATION_COUNT:
        raise ValueError("remediation record IDs are not unique")
    if remediation_ids != sorted(remediation_ids):
        raise ValueError("remediation record ordering changed")
    for record in combined_records:
        _require_nonempty_string(record, "group_id")
        intent = _require_nonempty_string(record, "intent")
        text = _require_nonempty_string(record, "text")
        if intent not in risk_mapping or record.get("risk") != risk_mapping[intent]:
            raise ValueError("combined record risk mapping changed")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError("combined record text hash changed")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError("combined record normalized text hash changed")
    expected_counts = combined_manifest.get("counts", {})
    if expected_counts != {
        "new_record_count": EXPECTED_REMEDIATION_COUNT,
        "old_record_count": EXPECTED_INHERITED_COUNT,
        "per_intent": count_by(combined_records, "intent"),
        "per_risk": count_by(combined_records, "risk"),
        "total_combined_count": EXPECTED_TOTAL_COUNT,
    }:
        raise ValueError("combined manifest counts do not reconcile")
    if remediation_manifest.get("counts") != {
        "intent_counts": count_by(remediation_records, "intent"),
        "record_count": EXPECTED_REMEDIATION_COUNT,
        "risk_counts": count_by(remediation_records, "risk_level"),
    }:
        raise ValueError("remediation manifest counts do not reconcile")
    return combined_records, inherited_records, remediation_records


def expected_review_provenance() -> dict[str, Any]:
    return {
        "ai_assisted_review_record_count": 810,
        "approved_count": 810,
        "human_adjudication_completed_count": 0,
        "human_adjudication_required_count": 0,
        "human_review_record_count": 0,
        "low_confidence_review_count": 0,
        "needs_revision_count": 0,
        "provenance_inconsistency_count": 0,
        "rejected_count": 0,
        "review_method": "ai_assisted_review",
        "review_record_count": 810,
        "reviewer_disagreement_count": 0,
        "reviewer_type": "AI-assisted semantic reviewer",
        "unresolved_protected_write_ambiguity_count": 0,
        "unresolved_taxonomy_ambiguity_count": 0,
    }


def validate_review_provenance(
    inputs: LoadedInputs,
    remediation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payloads = inputs.payloads
    expected = expected_review_provenance()
    observed = (
        payloads["dataset"].get("remediation_review_provenance"),
        payloads["combined_manifest"].get("review_provenance"),
        payloads["remediation_examples"].get("review_provenance"),
        payloads["remediation_manifest"].get("review_provenance"),
        payloads["contract"].get("review_governance_amendment", {}).get(
            "executed_review_summary"
        ),
    )
    if any(value != expected for value in observed):
        raise ValueError("AI-assisted review provenance does not reconcile")
    if any(record.get("review_status") != "approved" for record in remediation_records):
        raise ValueError("all remediation records must be approved")
    status_summary = payloads["remediation_manifest"].get("review_status_summary")
    if status_summary != {
        "approved": EXPECTED_REMEDIATION_COUNT,
        "needs_revision": 0,
        "rejected": 0,
        "unreviewed": 0,
    }:
        raise ValueError("remediation review statuses do not reconcile")
    governance = payloads["remediation_manifest"].get("review_governance")
    if governance != {
        "all_records_semantically_reviewed": True,
        "approved_status_implies_human_review": False,
        "human_adjudication_complete": True,
        "human_review_is_universal_build_gate": False,
        "needs_revision_resolved": True,
        "review_gates_passed": True,
        "semantic_review_required_for_all_records": True,
    }:
        raise ValueError("semantic-review governance changed")
    if payloads["combined_manifest"].get("review_governance") != governance:
        raise ValueError("review governance differs across manifests")
    for artifact_name in ("dataset", "remediation_examples"):
        artifact_governance = payloads[artifact_name].get("governance", {})
        if artifact_governance.get("human_review_is_universal_build_gate") is not False:
            raise ValueError("artifact incorrectly claims universal human review")
    if expected["human_adjudication_required_count"] != (
        expected["human_adjudication_completed_count"]
    ):
        raise ValueError("required human adjudication is incomplete")
    return expected


def authoring_batch_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_family[str(record["source_family_id"])][
            str(record["authoring_batch_id"])
        ] += 1
    return {
        family_id: {
            "counts_by_authoring_batch_id": dict(sorted(counts.items())),
            "unique_authoring_batch_count": len(counts),
        }
        for family_id, counts in sorted(by_family.items())
    }


def validate_source_provenance(
    inputs: LoadedInputs,
    remediation_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, Any], dict[str, dict[str, int]]]:
    family_descriptors: dict[str, tuple[str, str, str]] = {}
    descriptor_to_family: dict[tuple[str, str, str], str] = {}
    group_bindings: dict[str, tuple[str, str]] = {}
    source_family_counts: Counter[str] = Counter()
    by_intent: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in remediation_records:
        family_id = _require_nonempty_string(record, "source_family_id")
        _require_nonempty_string(record, "authoring_batch_id")
        descriptor = (
            _require_nonempty_string(record, "source_revision"),
            _require_nonempty_string(record, "authoring_method"),
            _require_nonempty_string(record, "source_family_independence_basis"),
        )
        previous = family_descriptors.setdefault(family_id, descriptor)
        if previous != descriptor:
            raise ValueError("source-family provenance is inconsistent")
        previous_family = descriptor_to_family.setdefault(descriptor, family_id)
        if previous_family != family_id:
            raise ValueError("multiple source-family IDs use identical provenance")
        intent = _require_nonempty_string(record, "intent")
        group_id = _require_nonempty_string(record, "group_id")
        binding = (intent, family_id)
        if group_bindings.setdefault(group_id, binding) != binding:
            raise ValueError("remediation group crosses intent or family boundary")
        source_family_counts[family_id] += 1
        by_intent[intent][family_id] += 1
    if tuple(sorted(source_family_counts)) != tuple(sorted(EXPECTED_SOURCE_FAMILIES)):
        raise ValueError("exact remediation source-family set changed")
    expected_by_intent = {
        intent: {
            family_id: (60 if intent == UNSUPPORTED_INTENT else 30)
            for family_id in EXPECTED_SOURCE_FAMILIES
        }
        for intent in PRIMARY_REMEDIATION_INTENTS
    }
    actual_by_intent = {
        intent: dict(sorted(by_intent[intent].items()))
        for intent in PRIMARY_REMEDIATION_INTENTS
    }
    if actual_by_intent != expected_by_intent:
        raise ValueError("per-intent source-family coverage changed")
    batches = authoring_batch_report(remediation_records)
    payloads = inputs.payloads
    if payloads["remediation_manifest"].get(
        "authoring_batch_statistics_by_source_family"
    ) != batches or payloads["combined_manifest"].get(
        "authoring_batch_statistics_by_source_family"
    ) != batches:
        raise ValueError("authoring-batch provenance does not reconcile")
    source_report = payloads["remediation_manifest"].get(
        "source_family_counts_per_intent", {}
    )
    combined_source_report = payloads["combined_manifest"].get(
        "source_aware_readiness", {}
    ).get("source_family_distribution_per_intent")
    if combined_source_report != source_report:
        raise ValueError("source-family reports differ across manifests")
    for intent, counts in expected_by_intent.items():
        observed = source_report.get(intent, {})
        if (
            observed.get("records_by_source_family") != counts
            or observed.get("total_new_records") != sum(counts.values())
            or observed.get("unique_source_family_count") != 3
            or observed.get("source_family_minimum_met") is not True
            or observed.get("per_family_30_record_target_met") is not True
            or observed.get("dominant_wording_review")
            != "semantic_review_required_no_automatic_threshold"
        ):
            raise ValueError(f"source-family report changed for {intent}")
    return dict(sorted(source_family_counts.items())), batches, actual_by_intent


def hard_negative_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for side_a, side_b in HARD_NEGATIVE_PAIRS:
        a_records = [
            record
            for record in records
            if record.get("is_hard_negative") is True
            and record.get("intent") == side_a
            and record.get("boundary_target") == side_b
        ]
        b_records = [
            record
            for record in records
            if record.get("is_hard_negative") is True
            and record.get("intent") == side_b
            and record.get("boundary_target") == side_a
        ]
        pairs.append(
            {
                "approved_count": len(a_records) + len(b_records),
                "coverage_complete": bool(a_records) and bool(b_records),
                "side_a": side_a,
                "side_a_count": len(a_records),
                "side_a_source_family_count": len(
                    {str(record["source_family_id"]) for record in a_records}
                ),
                "side_b": side_b,
                "side_b_count": len(b_records),
                "side_b_source_family_count": len(
                    {str(record["source_family_id"]) for record in b_records}
                ),
            }
        )
    return {
        "all_required_pairs_complete": all(
            pair["coverage_complete"] for pair in pairs
        ),
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def validate_unsupported_and_hard_negatives(
    inputs: LoadedInputs,
    remediation_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, Any]]:
    unsupported_counts = Counter(
        str(record.get("unsupported_subtype"))
        for record in remediation_records
        if record.get("intent") == UNSUPPORTED_INTENT
    )
    expected_unsupported = {subtype: 36 for subtype in UNSUPPORTED_SUBTYPES}
    if dict(sorted(unsupported_counts.items())) != expected_unsupported:
        raise ValueError("unsupported subtype coverage changed")
    if inputs.payloads["remediation_manifest"].get(
        "unsupported_subtype_counts"
    ) != expected_unsupported:
        raise ValueError("unsupported subtype manifest counts changed")
    hard_negatives = hard_negative_report(remediation_records)
    if (
        hard_negatives.get("all_required_pairs_complete") is not True
        or hard_negatives != inputs.payloads["remediation_manifest"].get(
            "hard_negative_coverage"
        )
    ):
        raise ValueError("hard-negative coverage changed")
    return expected_unsupported, hard_negatives


def duplicate_summary(
    inherited_records: Sequence[Mapping[str, Any]],
    remediation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_old = {str(record["text_sha256"]) for record in inherited_records}
    normalized_old = {
        str(record["normalized_text_sha256"]) for record in inherited_records
    }
    exact_new: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized_new: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized_old_records: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for record in inherited_records:
        normalized_old_records[str(record["normalized_text_sha256"])].append(record)
    for record in remediation_records:
        text = _require_nonempty_string(record, "text")
        if record.get("text_sha256") != text_sha256(text):
            raise ValueError("remediation text hash changed")
        if record.get("normalized_text_sha256") != normalized_text_sha256(text):
            raise ValueError("remediation normalized text hash changed")
        exact_new[text_sha256(text)].append(record)
        normalized_new[normalized_text_sha256(text)].append(record)
    exact_within = sum(len(records) > 1 for records in exact_new.values())
    normalized_within = sum(len(records) > 1 for records in normalized_new.values())
    exact_overlap = len(set(exact_new) & exact_old)
    normalized_overlap = len(set(normalized_new) & normalized_old)
    cross_intent = 0
    for digest, records in normalized_new.items():
        combined = list(records) + list(normalized_old_records.get(digest, []))
        if len({str(record["intent"]) for record in combined}) > 1:
            cross_intent += 1
    return {
        "cross_intent_normalized_conflict_count": cross_intent,
        "duplicate_gates_passed": not any(
            (
                exact_within,
                normalized_within,
                exact_overlap,
                normalized_overlap,
                cross_intent,
            )
        ),
        "exact_development_overlap_count": exact_overlap,
        "exact_within_new_cluster_count": exact_within,
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_development_overlap_count": normalized_overlap,
        "normalized_within_new_cluster_count": normalized_within,
        "raw_text_persisted_in_diagnostics": False,
        "semantic_or_embedding_similarity_used": False,
    }


def validate_duplicates(
    inputs: LoadedInputs,
    inherited_records: Sequence[Mapping[str, Any]],
    remediation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = duplicate_summary(inherited_records, remediation_records)
    for manifest_name, field in (
        ("remediation_manifest", "duplicate_validation"),
        ("combined_manifest", "duplicate_gates"),
    ):
        manifest_report = inputs.payloads[manifest_name].get(field, {})
        for key, value in summary.items():
            if manifest_report.get(key) != value:
                raise ValueError(f"duplicate report changed: {manifest_name}.{key}")
    if summary["duplicate_gates_passed"] is not True:
        raise ValueError("duplicate or normalized-conflict gate failed")
    return summary


def validate_group_and_source_readiness(
    inputs: LoadedInputs,
    combined_records: Sequence[Mapping[str, Any]],
    remediation_records: Sequence[Mapping[str, Any]],
    duplicate_validation: Mapping[str, Any],
) -> dict[str, Any]:
    groups = Counter(str(record["group_id"]) for record in combined_records)
    values = list(groups.values())
    expected_group_statistics = {
        "group_semantic_independence_proven_automatically": False,
        "maximum_records_per_group": max(values, default=0),
        "mean_records_per_group": statistics.fmean(values) if values else 0.0,
        "multi_record_group_count": sum(value > 1 for value in values),
        "record_count": len(combined_records),
        "semantic_review_still_required": True,
        "singleton_group_count": sum(value == 1 for value in values),
        "unique_group_count": len(groups),
    }
    if inputs.payloads["combined_manifest"].get("group_statistics") != (
        expected_group_statistics
    ):
        raise ValueError("combined group statistics changed")
    remediation_groups = Counter(
        str(record["group_id"]) for record in remediation_records
    )
    remediation_values = list(remediation_groups.values())
    expected_remediation_group_statistics = {
        "group_semantic_independence_proven_automatically": False,
        "maximum_records_per_group": max(remediation_values, default=0),
        "mean_records_per_group": (
            statistics.fmean(remediation_values) if remediation_values else 0.0
        ),
        "multi_record_group_count": sum(
            value > 1 for value in remediation_values
        ),
        "record_count": len(remediation_records),
        "semantic_review_still_required": True,
        "singleton_group_count": sum(
            value == 1 for value in remediation_values
        ),
        "unique_group_count": len(remediation_groups),
    }
    if inputs.payloads["remediation_manifest"].get("group_statistics") != (
        expected_remediation_group_statistics
    ):
        raise ValueError("remediation group statistics changed")
    family_to_intents: defaultdict[str, set[str]] = defaultdict(set)
    for record in remediation_records:
        family_to_intents[str(record["source_family_id"])].add(str(record["intent"]))
    holdout_candidates = [
        family_id
        for family_id in sorted(family_to_intents)
        if all(
            any(
                record["intent"] == intent
                and record["source_family_id"] != family_id
                for record in remediation_records
            )
            for intent in PRIMARY_REMEDIATION_INTENTS
        )
    ]
    groups_by_intent: defaultdict[str, set[str]] = defaultdict(set)
    for record in combined_records:
        groups_by_intent[str(record["intent"])].add(str(record["group_id"]))
    group_aware_cv_ready = all(
        len(groups_by_intent[intent]) >= 2 for intent in INTENT_LABEL_ORDER
    )
    readiness = {
        "duplicate_gates_passed": duplicate_validation["duplicate_gates_passed"],
        "group_aware_cv_ready": group_aware_cv_ready,
        "group_ids_do_not_cross_prohibited_split_boundaries": True,
        "minimum_three_source_families_every_primary_intent": True,
        "only_approved_records_included": True,
        "review_gates_passed": True,
        "source_family_independence_semantically_proven": False,
        "step29g_ready": bool(holdout_candidates) and group_aware_cv_ready,
        "whole_family_holdout_candidate_count": len(holdout_candidates),
        "whole_family_holdout_feasible_without_dropping_primary_intent": bool(
            holdout_candidates
        ),
    }
    manifest_readiness = inputs.payloads["combined_manifest"].get(
        "source_aware_readiness", {}
    )
    for key, value in readiness.items():
        if manifest_readiness.get(key) != value:
            raise ValueError(f"source-aware readiness changed: {key}")
    if not readiness["step29g_ready"]:
        raise ValueError("dataset is not ready for Step 29G")
    return readiness


def validate_governance(inputs: LoadedInputs) -> dict[str, bool]:
    expected = {
        "embeddings_generated": False,
        "model_inference_performed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "runtime_behavior_changed": False,
        "threshold_tuning_performed": False,
        "v2c5_raw_final_holdout_accessed": False,
    }
    for name in ("combined_manifest", "remediation_manifest"):
        if inputs.payloads[name].get("governance") != expected:
            raise ValueError(f"data-governance flags changed: {name}")
    for name in ("dataset", "remediation_examples"):
        governance = inputs.payloads[name].get("governance", {})
        for key, value in expected.items():
            if governance.get(key) is not value:
                raise ValueError(f"data-governance flag changed: {name}.{key}")
    return expected


def validate_inputs(inputs: LoadedInputs) -> dict[str, Any]:
    validate_hash_lineage(inputs)
    risk_mapping, risk_mapping_sha256 = validate_taxonomy(inputs)
    combined, inherited, remediation = validate_composition(inputs, risk_mapping)
    grouping_evidence = validate_group_structure(
        inherited, remediation, combined
    )
    review_provenance = validate_review_provenance(inputs, remediation)
    source_counts, batch_counts, source_counts_by_intent = (
        validate_source_provenance(inputs, remediation)
    )
    unsupported_counts, hard_negatives = validate_unsupported_and_hard_negatives(
        inputs, remediation
    )
    duplicates = validate_duplicates(inputs, inherited, remediation)
    readiness = validate_group_and_source_readiness(
        inputs, combined, remediation, duplicates
    )
    grouping_evidence["group_aware_cv_ready"] = readiness[
        "group_aware_cv_ready"
    ]
    governance = validate_governance(inputs)
    return {
        "authoring_batch_statistics_by_source_family": batch_counts,
        "duplicate_validation": duplicates,
        "governance": governance,
        "group_and_source_aware_readiness": readiness,
        "structural_grouping_evidence": grouping_evidence,
        "hard_negative_coverage": hard_negatives,
        "inherited_record_count": len(inherited),
        "intent_counts": count_by(combined, "intent"),
        "protected_write_intents": list(PROTECTED_WRITE_INTENTS),
        "remediation_intent_counts": count_by(remediation, "intent"),
        "remediation_record_count": len(remediation),
        "remediation_source_family_counts": source_counts,
        "remediation_source_family_counts_by_intent": source_counts_by_intent,
        "review_provenance": review_provenance,
        "risk_counts": count_by(combined, "risk"),
        "risk_mapping": dict(sorted(risk_mapping.items())),
        "risk_mapping_sha256": risk_mapping_sha256,
        "taxonomy_intent_labels": list(INTENT_LABEL_ORDER),
        "total_record_count": len(combined),
        "unsupported_subtype_counts": unsupported_counts,
        "validation_gates_passed": True,
    }


def build_freeze_payload(
    inputs: LoadedInputs,
    report: Mapping[str, Any],
    paths: FreezePaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    validation_script_bytes = guarded_read_bytes(
        paths.validation_script, paths, reader
    )
    parent_paths = source_paths(paths)
    parent_lineage = {
        name: {
            "path": display_path(parent_paths[name], paths),
            "sha256": inputs.sha256[name],
        }
        for name in sorted(parent_paths)
    }
    return {
        "freeze_status": "FROZEN",
        "frozen_dataset_path": display_path(paths.dataset, paths),
        "frozen_dataset_record_count": report["total_record_count"],
        "frozen_dataset_sha256": inputs.sha256["dataset"],
        "governance": {
            **report["governance"],
            "approved_status_implies_human_review": False,
            "development_dataset_frozen": True,
            "final_evaluation_performed": False,
            "human_review_is_universal_build_gate": False,
            "independent_human_annotation_claimed": False,
            "semantic_review_completed": True,
        },
        "group_and_source_aware_readiness": report[
            "group_and_source_aware_readiness"
        ],
        "structural_grouping_evidence": report[
            "structural_grouping_evidence"
        ],
        "hard_negative_coverage": report["hard_negative_coverage"],
        "inherited_v2c5_record_count": report["inherited_record_count"],
        "intent_counts": report["intent_counts"],
        "known_limitations": [
            "AI-assisted semantic review is not independent human annotation.",
            (
                "Structural provenance checks do not prove semantic "
                "independence."
            ),
            (
                "Group-aware evaluation treats group IDs as indivisible split "
                "units; inherited historical group IDs are not required to be "
                "label-pure."
            ),
            (
                "Independent generalization must be measured later using "
                "source-aware development evaluation and ultimately a fresh "
                "untouched final holdout."
            ),
        ],
        "next_required": "v2c6_source_aware_model_selection_contract",
        "parent_artifact_lineage": parent_lineage,
        "phase": "V2-C6 Step 29F",
        "protected_write_intents": report["protected_write_intents"],
        "remediation_authoring_batch_counts": report[
            "authoring_batch_statistics_by_source_family"
        ],
        "remediation_intent_counts": report["remediation_intent_counts"],
        "remediation_record_count": report["remediation_record_count"],
        "remediation_source_family_counts": report[
            "remediation_source_family_counts"
        ],
        "remediation_source_family_counts_by_intent": report[
            "remediation_source_family_counts_by_intent"
        ],
        "review_provenance_summary": report["review_provenance"],
        "risk_level_counts": report["risk_counts"],
        "schema_version": FREEZE_SCHEMA_VERSION,
        "taxonomy": {
            "intent_count": len(INTENT_LABEL_ORDER),
            "intent_label_order": report["taxonomy_intent_labels"],
            "path": display_path(paths.taxonomy, paths),
            "risk_mapping_sha256": report["risk_mapping_sha256"],
            "sha256": inputs.sha256["taxonomy"],
            "version": "v2c5-taxonomy.v1",
        },
        "unsupported_subtype_counts": report["unsupported_subtype_counts"],
        "validation_script": {
            "path": display_path(paths.validation_script, paths),
            "sha256": sha256_bytes(validation_script_bytes),
        },
        "validation_summary": {
            "duplicate_validation": report["duplicate_validation"],
            "hashes_and_manifests_reconciled": True,
            "validation_gates_passed": True,
        },
    }


def prepare_freeze(
    paths: FreezePaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> PreparedFreeze:
    inputs = load_inputs(paths, reader)
    report = validate_inputs(inputs)
    freeze = build_freeze_payload(inputs, report, paths, reader)
    return PreparedFreeze(
        report=report,
        freeze=freeze,
        freeze_bytes=stable_json_bytes(freeze),
    )


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing freeze artifact: {path}"
            ) from exc
        temporary_path.unlink()
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def check(
    paths: FreezePaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    prepared = prepare_freeze(paths, reader)
    freeze_present = paths.freeze_output.exists()
    if freeze_present:
        actual = guarded_read_bytes(paths.freeze_output, paths, reader)
        if actual != prepared.freeze_bytes:
            raise ValueError("existing freeze artifact is not deterministic")
    return {
        "files_written": False,
        "freeze_artifact_matches": freeze_present,
        "freeze_artifact_present": freeze_present,
        "frozen_dataset_record_count": prepared.report["total_record_count"],
        "ready_to_freeze": True,
        "remediation_record_count": prepared.report["remediation_record_count"],
    }


def freeze(
    paths: FreezePaths = DEFAULT_PATHS,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    if paths.freeze_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing freeze artifact: {paths.freeze_output}"
        )
    prepared = prepare_freeze(paths, reader)
    durable_create(paths.freeze_output, prepared.freeze_bytes)
    return {
        "files_written": True,
        "freeze_artifact": display_path(paths.freeze_output, paths),
        "freeze_artifact_sha256": sha256_bytes(prepared.freeze_bytes),
        "frozen_dataset_record_count": prepared.report["total_record_count"],
        "next_required": prepared.freeze["next_required"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the V2-C6 remediated development dataset."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check",
        action="store_true",
        help="Validate without writing; verify an existing freeze if present.",
    )
    action.add_argument(
        "--freeze",
        action="store_true",
        help="Validate and create the freeze artifact without overwriting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = check() if args.check else freeze()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
