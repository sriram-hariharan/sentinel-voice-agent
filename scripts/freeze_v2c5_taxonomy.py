"""Validate and freeze the human-adjudicated V2-C5 classifier taxonomy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"

STEP19_ADJUDICATION_SHA256 = (
    "e63e6a9afc575c603587f2a2119badd597bebbed3e57febc8f5384c16ea99d5d"
)
STEP19_MANIFEST_SHA256 = (
    "443501b3cd6d0614a089b8acd2a420674ee43b4e5821db9fce4df6e7a2db379f"
)

EXPECTED_CLUSTER_COUNT = 35
EXPECTED_DECISION_COUNTS = {
    "CANDIDATE_NEW_INTENT": 9,
    "INSUFFICIENT_EVIDENCE": 0,
    "MAP_TO_EXISTING_INTENT": 4,
    "MIXED_OR_INCOHERENT": 0,
    "NEEDS_SPLIT_REVIEW": 9,
    "REMAIN_UNSUPPORTED": 13,
}

FREEZE_SCHEMA_VERSION = "v2c5-taxonomy-freeze.v1"
FREEZE_MANIFEST_SCHEMA_VERSION = "v2c5-taxonomy-freeze-manifest.v1"
TAXONOMY_VERSION = "v2c5-taxonomy.v1"

RETAINED_INTENTS = (
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
    "unsupported_or_uncertain",
)
NEW_INTENTS = (
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "lost_or_stolen_phone",
    "passcode_recovery",
    "transfer_failed_or_declined",
    "transfer_pending",
)
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
RISK_BY_INTENT = {
    "account_balance": "PRIVATE_READ",
    "account_blocked": "PRIVATE_READ",
    "cancel_transfer": "PROTECTED_WRITE",
    "card_status": "PRIVATE_READ",
    "close_account": "PROTECTED_WRITE",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "lost_or_stolen_phone": "ESCALATION_OR_UNCERTAIN",
    "passcode_recovery": "ESCALATION_OR_UNCERTAIN",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "transfer_failed_or_declined": "PRIVATE_READ",
    "transfer_pending": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
PROTECTED_WRITE_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)

CANDIDATE_CLUSTER_RESOLUTIONS: tuple[dict[str, Any], ...] = (
    {
        "candidate_intent_name": "transfer_failed_or_declined",
        "canonical_cluster_id": "cluster-3c52f1b1fc6eccb5",
        "final_intent": "transfer_failed_or_declined",
        "rationale": (
            "Failed or declined transfers require a distinct classifier and "
            "evaluation distinction from pending transfers and generic "
            "transaction lookup."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
    {
        "candidate_intent_name": "cancel_transfer",
        "canonical_cluster_id": "cluster-9c182bcd6203a6a5",
        "final_intent": "cancel_transfer",
        "rationale": (
            "Transfer cancellation is a coherent protected-write goal, but "
            "recognition creates no transfer-cancellation runtime tool."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
    {
        "candidate_intent_name": "transfer_fee_charged",
        "canonical_cluster_id": "cluster-9d9626407e037b50",
        "final_intent": "transaction_details",
        "rationale": (
            "The customer is investigating a charge attached to a specific "
            "transfer. A separate classifier class adds little operational "
            "value; policy may supplement transaction lookup."
        ),
        "resolution": "MERGE_TO_EXISTING_INTENT",
    },
    {
        "candidate_intent_name": "passcode_recovery",
        "canonical_cluster_id": "cluster-cfbf7fb0669b0ecf",
        "final_intent": "passcode_recovery",
        "rationale": (
            "Credential recovery is a distinct security-sensitive support "
            "goal that requires recovery guidance or escalation rather than "
            "credential disclosure."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
    {
        "candidate_intent_name": "lost_or_stolen_phone",
        "canonical_cluster_id": "cluster-f9a5115f146f8ccb",
        "final_intent": "lost_or_stolen_phone",
        "rationale": (
            "A compromised phone or app session is a distinct security event "
            "and does not by itself imply a card-freeze request."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
    {
        "candidate_intent_name": "close_account",
        "canonical_cluster_id": "cluster-de04bcd8a7332e73",
        "final_intent": "close_account",
        "rationale": (
            "Account closure is a distinct protected-write goal; Step 20 "
            "recognizes it without adding an account-closure runtime tool."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
    {
        "candidate_intent_name": "transfer_pending",
        "canonical_cluster_id": "cluster-21a4b755f7c96f2c",
        "final_intent": "transfer_pending",
        "rationale": (
            "Investigation of an already pending transfer is distinct from "
            "failed transfers, cancellation, and generic timing guidance."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
    {
        "candidate_intent_name": "card_retained_by_atm",
        "canonical_cluster_id": "cluster-7dab4f9ae124477e",
        "final_intent": "informational_policy",
        "rationale": (
            "ATM-retained-card incidents primarily require recovery and "
            "security guidance. freeze_card remains an explicit protected "
            "action and is never inferred from ATM retention alone."
        ),
        "resolution": "MERGE_TO_EXISTING_INTENT",
    },
    {
        "candidate_intent_name": "account_blocked",
        "canonical_cluster_id": "cluster-a37f29dd8989408e",
        "final_intent": "account_blocked",
        "rationale": (
            "Investigation of an already blocked bank account is a private "
            "read and is distinct from freezing a card or requesting a new "
            "account freeze."
        ),
        "resolution": "ACCEPT_NEW_INTENT",
    },
)

SPLIT_CLUSTER_RESOLUTIONS: tuple[dict[str, Any], ...] = (
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "informational, capability, or card-acquisition guidance",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "direct unsupported card acquisition, order, or management",
            },
        ],
        "canonical_cluster_id": "cluster-25a4740ae330f948",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "theme": "Card acquisition and virtual/disposable card management",
    },
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "PIN information or how-to guidance",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "direct PIN-changing operation",
            },
        ],
        "canonical_cluster_id": "cluster-bab7911ea80f612b",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "theme": "PIN information and PIN change requests",
    },
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "generic identity-verification guidance",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "customer-specific failed verification or recovery problem",
            },
        ],
        "canonical_cluster_id": "cluster-e92864f71372fce3",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "safety_constraints": [
            (
                "Verification failure never becomes escalation unless the "
                "utterance requests human escalation."
            )
        ],
        "theme": "Identity verification guidance and verification failures",
    },
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "generic personal-details update guidance",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "direct personal-details update operation",
            },
        ],
        "canonical_cluster_id": "cluster-3db20683a0becc1c",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "theme": "Personal-details change guidance and direct update requests",
    },
    {
        "branches": [
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "customer-specific credit-limit lookup",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "direct credit-limit change or increase",
            },
        ],
        "canonical_cluster_id": "cluster-d8ae664886c213a2",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "safety_constraints": [
            "No credit-account capability exists in the current V1 runtime."
        ],
        "theme": "Credit-limit lookup and credit-limit change requests",
    },
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "generic money-transfer guidance",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "direct transfer execution",
            },
        ],
        "canonical_cluster_id": "cluster-783d0d28bd4c4992",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "safety_constraints": [
            "Direct transfer execution does not create or imply a transfer-money runtime tool."
        ],
        "theme": "Money transfer guidance and execution requests",
    },
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "generic card-delivery timing guidance",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "customer-specific delayed or missing-card investigation",
            },
        ],
        "canonical_cluster_id": "cluster-2b49d15c7f2e42b9",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "theme": "Card delivery estimates and delayed-card investigation",
    },
    {
        "branches": [
            {
                "final_intent": "account_blocked",
                "when": "already blocked or frozen bank-account investigation",
            },
            {
                "final_intent": "unsupported_or_uncertain",
                "when": "direct request to freeze or block the bank account",
            },
        ],
        "canonical_cluster_id": "cluster-c29614d54555e30c",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "safety_constraints": [
            "A bank-account freeze request never becomes freeze_card."
        ],
        "theme": "Frozen-account investigation and account-freeze requests",
    },
    {
        "branches": [
            {
                "final_intent": "informational_policy",
                "when": "generic transfer timing or how-long guidance",
            },
            {
                "final_intent": "transfer_pending",
                "when": "already delayed or pending transfer investigation",
            },
        ],
        "canonical_cluster_id": "cluster-a077797c96f814ce",
        "resolution": "BRANCH_FOR_STEP21_RELABELING",
        "theme": "Transfer timing guidance and delayed-transfer investigation",
    },
)

FORBIDDEN_SOURCE_DECISIONS = frozenset(
    {"MIXED_OR_INCOHERENT", "INSUFFICIENT_EVIDENCE"}
)
FORBIDDEN_INPUT_FILENAMES = frozenset(
    {
        "v2c4_safety_holdout.json",
        "v2c4_safety_holdout_seed.json",
    }
)


@dataclass(frozen=True)
class FreezePaths:
    step19_adjudication: Path
    step19_manifest: Path
    freeze_output: Path
    freeze_manifest_output: Path


DEFAULT_PATHS = FreezePaths(
    step19_adjudication=ML_ROOT / "v2c5_taxonomy_adjudication.json",
    step19_manifest=ML_ROOT / "v2c5_taxonomy_adjudication.manifest.json",
    freeze_output=ML_ROOT / "v2c5_taxonomy_freeze.json",
    freeze_manifest_output=ML_ROOT / "v2c5_taxonomy_freeze.manifest.json",
)


@dataclass(frozen=True)
class Step19Evidence:
    adjudication: dict[str, Any]
    manifest: dict[str, Any]
    adjudication_bytes: bytes
    manifest_bytes: bytes


@dataclass(frozen=True)
class PreparedFreeze:
    freeze: dict[str, Any]
    freeze_bytes: bytes
    manifest: dict[str, Any]
    manifest_bytes: bytes


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def guard_input_path(path: Path) -> None:
    if path.name in FORBIDDEN_INPUT_FILENAMES:
        raise ValueError(f"sealed V2-C4 holdout input is forbidden: {path.name}")
    allowed = {
        DEFAULT_PATHS.step19_adjudication.name,
        DEFAULT_PATHS.step19_manifest.name,
    }
    if path.name not in allowed:
        raise ValueError(f"Step 20 input artifact is not allowed: {path.name}")


def read_input_bytes(path: Path) -> bytes:
    guard_input_path(path)
    return path.read_bytes()


def load_json_object(value: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def require_sha256(value: bytes, expected: str, label: str) -> None:
    if sha256_bytes(value) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


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


def validate_step19_manifest(
    manifest: Mapping[str, Any],
    paths: FreezePaths,
) -> None:
    if manifest.get("schema_version") != (
        "v2c5-taxonomy-adjudication-manifest.v1"
    ):
        raise ValueError("unexpected Step 19 manifest schema")
    if manifest.get("phase") != "V2-C5 Step 19":
        raise ValueError("unexpected Step 19 manifest phase")
    expected_reference = {
        "path": display_path(paths.step19_adjudication),
        "sha256": STEP19_ADJUDICATION_SHA256,
    }
    if manifest.get("adjudication") != expected_reference:
        raise ValueError("Step 19 manifest adjudication binding changed")
    if manifest.get("cluster_count") != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 19 manifest cluster count changed")
    if manifest.get("reviewed_count") != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 19 manifest reviewed count changed")
    if manifest.get("decision_counts") != EXPECTED_DECISION_COUNTS:
        raise ValueError("Step 19 manifest decision counts changed")
    expected_governance = {
        "classifier_training_performed": False,
        "final_taxonomy_frozen": False,
        "human_adjudication_completed": True,
        "new_intents_created": False,
        "runtime_behavior_changed": False,
        "step20_required": True,
        "taxonomy_changed": False,
    }
    for field, expected in expected_governance.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Step 19 manifest governance changed: {field}")


def candidate_source_pairs(
    adjudications: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for record in adjudications:
        if record.get("decision") != "CANDIDATE_NEW_INTENT":
            continue
        cluster_id = record.get("canonical_cluster_id")
        candidate = record.get("candidate_intent_name")
        if not isinstance(cluster_id, str) or not isinstance(candidate, str):
            raise TypeError("candidate cluster ID and name must be strings")
        if cluster_id in pairs:
            raise ValueError("duplicate Step 19 candidate cluster")
        pairs[cluster_id] = candidate
    return pairs


def validate_step19_adjudication(adjudication: Mapping[str, Any]) -> None:
    if adjudication.get("schema_version") != "v2c5-taxonomy-adjudication.v1":
        raise ValueError("unexpected Step 19 adjudication schema")
    if adjudication.get("phase") != "V2-C5 Step 19":
        raise ValueError("unexpected Step 19 adjudication phase")
    if adjudication.get("cluster_count") != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 19 adjudication cluster count changed")
    if adjudication.get("reviewed_count") != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 19 adjudication is incomplete")
    if adjudication.get("decision_counts") != EXPECTED_DECISION_COUNTS:
        raise ValueError("Step 19 adjudication decision counts changed")
    expected_governance = {
        "classifier_training_performed": False,
        "final_taxonomy_frozen": False,
        "human_adjudication_completed": True,
        "new_intents_created": False,
        "runtime_behavior_changed": False,
        "step20_required": True,
        "taxonomy_changed": False,
    }
    for field, expected in expected_governance.items():
        if adjudication.get(field) != expected:
            raise ValueError(f"Step 19 adjudication governance changed: {field}")

    records = adjudication.get("adjudications")
    if not isinstance(records, list) or len(records) != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 19 must contain exactly 35 adjudications")
    if any(not isinstance(record, dict) for record in records):
        raise TypeError("Step 19 adjudications must be objects")
    cluster_ids = [record.get("canonical_cluster_id") for record in records]
    if any(not isinstance(cluster_id, str) for cluster_id in cluster_ids):
        raise TypeError("Step 19 cluster IDs must be strings")
    if len(set(cluster_ids)) != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 19 cluster IDs must be unique")
    if any(record.get("review_status") != "REVIEWED" for record in records):
        raise ValueError("Step 19 contains an unreviewed cluster")

    if any(record.get("decision") in FORBIDDEN_SOURCE_DECISIONS for record in records):
        raise ValueError("Step 20 cannot resolve mixed or insufficient evidence")
    actual_counts = Counter(str(record.get("decision")) for record in records)
    normalized_counts = {
        decision: actual_counts[decision]
        for decision in EXPECTED_DECISION_COUNTS
    }
    if (
        normalized_counts != EXPECTED_DECISION_COUNTS
        or set(actual_counts) - set(EXPECTED_DECISION_COUNTS)
    ):
        raise ValueError("Step 19 record decision counts changed")

    expected_candidates = {
        str(item["canonical_cluster_id"]): str(item["candidate_intent_name"])
        for item in CANDIDATE_CLUSTER_RESOLUTIONS
    }
    if candidate_source_pairs(records) != expected_candidates:
        raise ValueError("Step 19 candidate cluster IDs or names changed")
    expected_split_ids = {
        str(item["canonical_cluster_id"])
        for item in SPLIT_CLUSTER_RESOLUTIONS
    }
    actual_split_ids = {
        str(record["canonical_cluster_id"])
        for record in records
        if record.get("decision") == "NEEDS_SPLIT_REVIEW"
    }
    if actual_split_ids != expected_split_ids:
        raise ValueError("Step 19 split-review cluster IDs changed")


def load_step19(paths: FreezePaths = DEFAULT_PATHS) -> Step19Evidence:
    adjudication_bytes = read_input_bytes(paths.step19_adjudication)
    manifest_bytes = read_input_bytes(paths.step19_manifest)
    require_sha256(
        adjudication_bytes,
        STEP19_ADJUDICATION_SHA256,
        "Step 19 adjudication",
    )
    require_sha256(
        manifest_bytes,
        STEP19_MANIFEST_SHA256,
        "Step 19 adjudication manifest",
    )
    adjudication = load_json_object(
        adjudication_bytes,
        "Step 19 adjudication",
    )
    manifest = load_json_object(manifest_bytes, "Step 19 manifest")
    validate_step19_adjudication(adjudication)
    validate_step19_manifest(manifest, paths)
    return Step19Evidence(
        adjudication=adjudication,
        manifest=manifest,
        adjudication_bytes=adjudication_bytes,
        manifest_bytes=manifest_bytes,
    )


def build_carry_forward_resolutions(
    adjudication: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for record in adjudication["adjudications"]:
        decision = record["decision"]
        if decision == "MAP_TO_EXISTING_INTENT":
            final_intent = record.get("target_existing_intent")
            if final_intent not in RETAINED_INTENTS:
                raise ValueError("invalid Step 19 existing-intent target")
            resolution = "CARRY_FORWARD_EXISTING_INTENT"
        elif decision == "REMAIN_UNSUPPORTED":
            final_intent = "unsupported_or_uncertain"
            resolution = "REMAIN_UNSUPPORTED"
        else:
            continue
        values.append(
            {
                "canonical_cluster_id": record["canonical_cluster_id"],
                "final_intent": final_intent,
                "source_decision": decision,
                "resolution": resolution,
            }
        )
    return sorted(values, key=lambda value: str(value["canonical_cluster_id"]))


def all_resolution_ids(payload: Mapping[str, Any]) -> list[str]:
    groups = (
        payload["candidate_cluster_resolutions"],
        payload["split_cluster_resolutions"],
        payload["carry_forward_cluster_resolutions"],
    )
    return [
        str(record["canonical_cluster_id"])
        for group in groups
        for record in group
    ]


def validate_no_runtime_tool_mapping(payload: Any) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if "tool" in normalized and "mapping" in normalized:
                raise ValueError("Step 20 cannot introduce a runtime tool mapping")
            validate_no_runtime_tool_mapping(value)
    elif isinstance(payload, list):
        for value in payload:
            validate_no_runtime_tool_mapping(value)


def validate_no_raw_utterance_text(payload: Any) -> None:
    prohibited_keys = {"text", "normalized_text", "utterance", "examples"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in prohibited_keys:
                raise ValueError("taxonomy freeze cannot contain utterance text")
            validate_no_raw_utterance_text(value)
    elif isinstance(payload, list):
        for value in payload:
            validate_no_raw_utterance_text(value)


def validate_freeze_payload(
    payload: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    paths: FreezePaths = DEFAULT_PATHS,
) -> None:
    if payload.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("unexpected Step 20 freeze schema")
    if payload.get("phase") != "V2-C5 Step 20":
        raise ValueError("unexpected Step 20 freeze phase")
    if payload.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError("unexpected V2-C5 taxonomy version")
    if payload.get("prior_intent_count") != 9:
        raise ValueError("prior intent count must remain nine")
    if payload.get("new_intent_count") != 7:
        raise ValueError("new intent count must be seven")
    if payload.get("final_intent_count") != 16:
        raise ValueError("final intent count must be sixteen")
    if payload.get("retained_intents") != list(RETAINED_INTENTS):
        raise ValueError("retained intent list changed")
    if payload.get("new_intents") != list(NEW_INTENTS):
        raise ValueError("new intent list changed")
    if payload.get("intent_label_order") != list(INTENT_LABEL_ORDER):
        raise ValueError("final intent label order changed")
    labels = payload["intent_label_order"]
    if labels != sorted(labels) or len(labels) != len(set(labels)):
        raise ValueError("final intent labels must be unique and sorted")
    if set(labels) != set(RETAINED_INTENTS) | set(NEW_INTENTS):
        raise ValueError("final intent label set changed")
    if payload.get("risk_by_intent") != RISK_BY_INTENT:
        raise ValueError("V2-C5 risk mapping changed")
    if set(payload["risk_by_intent"]) != set(labels):
        raise ValueError("risk mapping must cover every final intent exactly")
    if payload.get("protected_write_intents") != list(PROTECTED_WRITE_INTENTS):
        raise ValueError("protected-write intent set changed")

    candidate_resolutions = payload.get("candidate_cluster_resolutions")
    split_resolutions = payload.get("split_cluster_resolutions")
    if candidate_resolutions != [
        copy.deepcopy(value) for value in CANDIDATE_CLUSTER_RESOLUTIONS
    ]:
        raise ValueError("candidate cluster resolutions changed")
    if split_resolutions != [
        copy.deepcopy(value) for value in SPLIT_CLUSTER_RESOLUTIONS
    ]:
        raise ValueError("split cluster resolutions changed")
    if sum(
        value["resolution"] == "ACCEPT_NEW_INTENT"
        for value in candidate_resolutions
    ) != 7:
        raise ValueError("exactly seven candidate intents must be accepted")
    if sum(
        value["resolution"] == "MERGE_TO_EXISTING_INTENT"
        for value in candidate_resolutions
    ) != 2:
        raise ValueError("exactly two candidate intents must be merged")
    accepted_intents = {
        str(value["final_intent"])
        for value in candidate_resolutions
        if value["resolution"] == "ACCEPT_NEW_INTENT"
    }
    if accepted_intents != set(NEW_INTENTS):
        raise ValueError("accepted candidates must equal the seven new intents")
    for resolution in split_resolutions:
        for branch in resolution["branches"]:
            if branch["final_intent"] not in INTENT_LABEL_ORDER:
                raise ValueError("split branch references an unknown intent")

    expected_carry = build_carry_forward_resolutions(adjudication)
    if payload.get("carry_forward_cluster_resolutions") != expected_carry:
        raise ValueError("carry-forward cluster resolutions changed")
    resolved_ids = all_resolution_ids(payload)
    source_ids = [
        str(record["canonical_cluster_id"])
        for record in adjudication["adjudications"]
    ]
    if len(resolved_ids) != EXPECTED_CLUSTER_COUNT:
        raise ValueError("Step 20 must contain exactly 35 cluster resolutions")
    if len(set(resolved_ids)) != EXPECTED_CLUSTER_COUNT:
        raise ValueError("a Step 19 cluster is resolved more than once")
    if set(resolved_ids) != set(source_ids):
        raise ValueError("Step 20 does not account for every Step 19 cluster")

    expected_source = {
        "path": display_path(paths.step19_adjudication),
        "sha256": STEP19_ADJUDICATION_SHA256,
    }
    source_reference = payload.get("source_step19_adjudication")
    if not isinstance(source_reference, dict):
        raise TypeError("Step 20 source adjudication reference must be an object")
    if source_reference != expected_source:
        raise ValueError("Step 20 source adjudication binding changed")
    input_scope = payload.get("input_scope")
    if not isinstance(input_scope, dict):
        raise TypeError("Step 20 input scope must be an object")
    if input_scope.get("prohibited_artifacts_consumed") != []:
        raise ValueError("Step 20 consumed a prohibited artifact")
    if input_scope.get("consumed_artifacts") != [
        source_reference["path"],
        display_path(paths.step19_manifest),
    ]:
        raise ValueError("Step 20 input scope changed")

    expected_flags = {
        "classifier_training_performed": False,
        "dataset_relabeling_performed": False,
        "final_taxonomy_frozen": True,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "runtime_authority": False,
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "step21_required": True,
    }
    for field, expected in expected_flags.items():
        if payload.get(field) != expected:
            raise ValueError(f"Step 20 governance changed: {field}")
    validate_no_runtime_tool_mapping(payload)
    validate_no_raw_utterance_text(payload)


def build_freeze_payload(
    adjudication: Mapping[str, Any],
    paths: FreezePaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    validate_step19_adjudication(adjudication)
    payload = {
        "candidate_cluster_resolutions": [
            copy.deepcopy(value) for value in CANDIDATE_CLUSTER_RESOLUTIONS
        ],
        "carry_forward_cluster_resolutions": (
            build_carry_forward_resolutions(adjudication)
        ),
        "carry_forward_resolution_policy": {
            "MAP_TO_EXISTING_INTENT": (
                "Carry forward target_existing_intent exactly."
            ),
            "REMAIN_UNSUPPORTED": "Resolve to unsupported_or_uncertain.",
            "individual_utterance_relabeling_performed": False,
        },
        "classifier_training_performed": False,
        "dataset_relabeling_performed": False,
        "final_intent_count": 16,
        "final_taxonomy_frozen": True,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "input_scope": {
            "consumed_artifacts": [
                display_path(paths.step19_adjudication),
                display_path(paths.step19_manifest),
            ],
            "prohibited_artifacts_consumed": [],
        },
        "intent_label_order": list(INTENT_LABEL_ORDER),
        "new_intent_count": 7,
        "new_intents": list(NEW_INTENTS),
        "phase": "V2-C5 Step 20",
        "prior_intent_count": 9,
        "protected_write_intents": list(PROTECTED_WRITE_INTENTS),
        "retained_intents": list(RETAINED_INTENTS),
        "risk_by_intent": copy.deepcopy(RISK_BY_INTENT),
        "risk_mapping_role": "classifier_and_evaluation_metadata_only",
        "runtime_authority": False,
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "schema_version": FREEZE_SCHEMA_VERSION,
        "source_step19_adjudication": {
            "path": display_path(paths.step19_adjudication),
            "sha256": STEP19_ADJUDICATION_SHA256,
        },
        "split_cluster_resolutions": [
            copy.deepcopy(value) for value in SPLIT_CLUSTER_RESOLUTIONS
        ],
        "step21_required": True,
        "taxonomy_version": TAXONOMY_VERSION,
    }
    validate_freeze_payload(payload, adjudication, paths)
    return payload


def build_freeze_manifest(
    freeze_bytes: bytes,
    paths: FreezePaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "classifier_training_performed": False,
        "final_intent_count": 16,
        "final_taxonomy_frozen": True,
        "fresh_holdout_created": False,
        "fresh_holdout_required": True,
        "new_intent_count": 7,
        "phase": "V2-C5 Step 20",
        "runtime_behavior_changed": False,
        "runtime_tools_changed": False,
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "source_step19_adjudication": {
            "path": display_path(paths.step19_adjudication),
            "sha256": STEP19_ADJUDICATION_SHA256,
        },
        "source_step19_manifest": {
            "path": display_path(paths.step19_manifest),
            "sha256": STEP19_MANIFEST_SHA256,
        },
        "step20_script": {
            "path": display_path(script_path),
            "sha256": sha256_bytes(script_path.read_bytes()),
        },
        "step21_required": True,
        "taxonomy_freeze": {
            "path": display_path(paths.freeze_output),
            "sha256": sha256_bytes(freeze_bytes),
        },
    }


def validate_freeze_manifest(
    manifest: Mapping[str, Any],
    freeze_bytes: bytes,
    paths: FreezePaths = DEFAULT_PATHS,
) -> None:
    expected = build_freeze_manifest(freeze_bytes, paths)
    if manifest != expected:
        raise ValueError("Step 20 freeze manifest is not deterministic")
    validate_no_runtime_tool_mapping(manifest)


def validate_existing_outputs(
    prepared: PreparedFreeze,
    paths: FreezePaths,
) -> None:
    freeze_exists = paths.freeze_output.exists()
    manifest_exists = paths.freeze_manifest_output.exists()
    if freeze_exists != manifest_exists:
        raise ValueError("Step 20 output artifacts must exist as a pair")
    if not freeze_exists:
        return
    if paths.freeze_output.read_bytes() != prepared.freeze_bytes:
        raise ValueError("existing Step 20 taxonomy freeze is not deterministic")
    if paths.freeze_manifest_output.read_bytes() != prepared.manifest_bytes:
        raise ValueError("existing Step 20 freeze manifest is not deterministic")


def prepare_freeze(
    paths: FreezePaths = DEFAULT_PATHS,
    *,
    validate_existing: bool = True,
) -> PreparedFreeze:
    evidence = load_step19(paths)
    freeze = build_freeze_payload(evidence.adjudication, paths)
    freeze_bytes = stable_json_bytes(freeze)
    manifest = build_freeze_manifest(freeze_bytes, paths)
    validate_freeze_manifest(manifest, freeze_bytes, paths)
    manifest_bytes = stable_json_bytes(manifest)
    prepared = PreparedFreeze(
        freeze=freeze,
        freeze_bytes=freeze_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
    )
    if validate_existing:
        validate_existing_outputs(prepared, paths)
    return prepared


def check_freeze(paths: FreezePaths = DEFAULT_PATHS) -> dict[str, Any]:
    prepared = prepare_freeze(paths)
    return {
        "accounted_cluster_count": len(all_resolution_ids(prepared.freeze)),
        "final_intent_count": prepared.freeze["final_intent_count"],
        "new_intent_count": prepared.freeze["new_intent_count"],
        "status": "valid",
        "writes_performed": False,
    }


def export_freeze(paths: FreezePaths = DEFAULT_PATHS) -> dict[str, Any]:
    prepared = prepare_freeze(paths)
    atomic_write_bytes(paths.freeze_output, prepared.freeze_bytes)
    atomic_write_bytes(paths.freeze_manifest_output, prepared.manifest_bytes)
    return {
        "freeze_path": display_path(paths.freeze_output),
        "freeze_sha256": sha256_bytes(prepared.freeze_bytes),
        "manifest_path": display_path(paths.freeze_manifest_output),
        "status": "exported",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("export")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = check_freeze()
        elif args.command == "export":
            result = export_freeze()
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(f"unhandled command: {args.command}")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
