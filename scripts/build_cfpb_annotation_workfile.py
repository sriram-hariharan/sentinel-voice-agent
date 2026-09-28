"""Initialize and export the CFPB semantic annotation workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
REVIEW_POOL_MANIFEST_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/review_pool_manifest.json"
)
LOCAL_REVIEW_POOL_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/local/cfpb_review_pool.jsonl"
)
LOCAL_WORKFILE_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/local/cfpb_semantic_review.jsonl"
)
LOCAL_FIRST_PASS_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/local/cfpb_llm_first_pass.jsonl"
)
LABEL_ARTIFACT_PATH = EXTERNAL_ROOT / "processed/cfpb/cfpb_semantic_labels.json"

WORKFLOW_VERSION = "cfpb-semantic-review-workflow.2026-09-27.v2"
WORKFILE_SCHEMA_VERSION = "cfpb-semantic-review-workfile.v1"
LABEL_SCHEMA_VERSION = "cfpb-semantic-labels.v2"
REVIEW_POOL_SCHEMA_VERSION = "cfpb-review-pool-manifest.v1"

SEMANTIC_REVIEW_STATUSES = ("NEAR_MATCH", "AMBIGUOUS")
DEFAULT_POOL_LANE_COUNTS = {
    "NEAR_MATCH": 600,
    "AMBIGUOUS": 1_200,
    "UNSUPPORTED": 2_000,
}
REVIEW_CATEGORIES = (
    "SINGLE_SUPPORTED_INTENT",
    "MULTI_SUPPORTED_INTENT",
    "UNSUPPORTED",
    "UNCLEAR_OR_INSUFFICIENT",
    "NO_CURRENT_REQUEST",
)
SUPPORTED_INTENTS = (
    "informational_policy",
    "account_balance",
    "recent_transactions",
    "transaction_details",
    "card_status",
    "freeze_card",
    "create_dispute",
    "escalation",
)
PROTECTED_WRITE_INTENTS = frozenset({"freeze_card", "create_dispute"})
ANNOTATION_CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
ADJUDICATION_STATUSES = (
    "UNREVIEWED",
    "REVIEWED",
    "NEEDS_ADJUDICATION",
    "ADJUDICATED",
)
INCOMPLETE_ADJUDICATION_STATUSES = frozenset(
    {"UNREVIEWED", "NEEDS_ADJUDICATION"}
)
ANNOTATION_FIELDS = (
    "review_category",
    "supported_intents",
    "annotation_confidence",
    "annotation_note",
    "reviewer_id",
    "adjudication_status",
    "secondary_review_required",
)
FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "abstention_result",
        "classifier_abstention",
        "classifier_prediction",
        "classifier_score",
        "logistic_probability",
        "predicted_intent",
        "prediction",
        "svm_score",
    }
)
TRACKED_LABEL_FIELDS = (
    "complaint_id",
    "narrative_sha256",
    "original_mapping_status",
    "candidate_sentinelvoice_intents",
    "review_category",
    "supported_intents",
    "annotation_confidence",
    "annotation_note",
    "adjudication_status",
    "annotation_provenance",
    "human_review_required",
    "human_review_reasons",
    "codex_annotator_model",
    "codex_prompt_version",
    "codex_batch_id",
    "qc_sample_selected",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def stable_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value, raw


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path} line {line_number} must be a JSON object")
        rows.append(value)
    return rows, raw


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def reconcile_review_pool_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_lane_counts: Mapping[str, int] = DEFAULT_POOL_LANE_COUNTS,
) -> dict[str, Any]:
    if manifest.get("schema_version") != REVIEW_POOL_SCHEMA_VERSION:
        raise ValueError("unexpected CFPB review-pool manifest schema")
    actual_counts = manifest.get("summary", {}).get(
        "selected_by_mapping_status"
    )
    expected_counts = dict(expected_lane_counts)
    if actual_counts != expected_counts:
        raise ValueError(
            "review-pool lane counts differ from the frozen annotation scope"
        )
    semantic_count = sum(
        expected_counts[status] for status in SEMANTIC_REVIEW_STATUSES
    )
    if manifest.get("summary", {}).get("total_selected") != sum(
        expected_counts.values()
    ):
        raise ValueError("review-pool total does not reconcile with lane counts")
    return {
        "excluded_unsupported_count": expected_counts["UNSUPPORTED"],
        "semantic_review_count": semantic_count,
        "semantic_review_statuses": list(SEMANTIC_REVIEW_STATUSES),
    }


def semantic_manifest_records(
    manifest: Mapping[str, Any],
    *,
    expected_lane_counts: Mapping[str, int] = DEFAULT_POOL_LANE_COUNTS,
) -> list[dict[str, Any]]:
    reconciliation = reconcile_review_pool_manifest(
        manifest,
        expected_lane_counts=expected_lane_counts,
    )
    records = manifest.get("selected_records")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise TypeError("review-pool manifest must contain selected_records")
    if len(records) != sum(expected_lane_counts.values()):
        raise ValueError("review-pool selected-record total does not reconcile")
    semantic_records = [
        record
        for record in records
        if record.get("mapping_status") in SEMANTIC_REVIEW_STATUSES
    ]
    if len(semantic_records) != reconciliation["semantic_review_count"]:
        raise ValueError("semantic-review record count does not reconcile")
    hashes = [record.get("narrative_sha256") for record in semantic_records]
    if not all(isinstance(value, str) and len(value) == 64 for value in hashes):
        raise ValueError("semantic-review records require narrative SHA-256 values")
    if len(hashes) != len(set(hashes)):
        raise ValueError("semantic-review narrative hashes must be unique")
    return semantic_records


def validate_source_row(row: Mapping[str, Any]) -> None:
    if FORBIDDEN_MODEL_FIELDS.intersection(row):
        raise ValueError("annotation input contains forbidden classifier output fields")
    if row.get("mapping_status") not in {
        *SEMANTIC_REVIEW_STATUSES,
        "UNSUPPORTED",
    }:
        raise ValueError("annotation input has an unexpected mapping status")
    narrative = row.get("narrative")
    digest = row.get("narrative_sha256")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("annotation input requires non-empty narrative text")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("annotation input requires a narrative SHA-256")
    if sha256_bytes(narrative.encode("utf-8")) != digest:
        raise ValueError("annotation input narrative does not match its SHA-256")


def source_rows_by_hash(
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_hash: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        validate_source_row(row)
        digest = row["narrative_sha256"]
        if digest in by_hash:
            raise ValueError("annotation input contains duplicate narrative hashes")
        by_hash[digest] = row
    return by_hash


def validate_source_against_manifest(
    source_rows: Sequence[Mapping[str, Any]],
    manifest_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_hash = source_rows_by_hash(source_rows)
    manifest_by_hash = {
        record["narrative_sha256"]: record for record in manifest_records
    }
    if set(by_hash) != set(manifest_by_hash):
        raise ValueError("local semantic-review rows do not match holdout hashes")
    for digest, source in by_hash.items():
        manifest_record = manifest_by_hash[digest]
        comparisons = {
            "candidate_sentinelvoice_intents": "candidate_sentinelvoice_intents",
            "complaint_id": "complaint_id",
            "mapping_status": "mapping_status",
        }
        for source_field, manifest_field in comparisons.items():
            if source.get(source_field) != manifest_record.get(manifest_field):
                raise ValueError(
                    f"local source field {source_field} differs from manifest"
                )
    return by_hash


def initialized_annotation_fields() -> dict[str, Any]:
    return {
        "adjudication_status": "UNREVIEWED",
        "annotation_confidence": None,
        "annotation_note": "",
        "review_category": None,
        "reviewer_id": "",
        "secondary_review_required": False,
        "supported_intents": [],
    }


def validate_annotation_fields(row: Mapping[str, Any]) -> None:
    status = row.get("adjudication_status")
    category = row.get("review_category")
    confidence = row.get("annotation_confidence")
    supported = row.get("supported_intents")
    note = row.get("annotation_note")
    reviewer_id = row.get("reviewer_id")
    secondary_required = row.get("secondary_review_required")

    if status not in ADJUDICATION_STATUSES:
        raise ValueError(f"invalid adjudication_status: {status}")
    if not isinstance(supported, list) or not all(
        isinstance(intent, str) for intent in supported
    ):
        raise ValueError("supported_intents must be a list of strings")
    if len(supported) != len(set(supported)) or supported != sorted(supported):
        raise ValueError("supported_intents must be unique and sorted")
    invalid_intents = set(supported) - set(SUPPORTED_INTENTS)
    if invalid_intents:
        raise ValueError(f"unsupported supported_intents: {sorted(invalid_intents)}")
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError("annotation_note must be a string of at most 500 characters")
    if not isinstance(reviewer_id, str):
        raise TypeError("reviewer_id must be a string")
    if not isinstance(secondary_required, bool):
        raise TypeError("secondary_review_required must be boolean")

    if status == "UNREVIEWED":
        if any(
            (
                category is not None,
                bool(supported),
                confidence is not None,
                bool(note),
                bool(reviewer_id),
                secondary_required,
            )
        ):
            raise ValueError("UNREVIEWED rows must retain empty annotation fields")
        return

    if category not in REVIEW_CATEGORIES:
        raise ValueError(f"invalid review_category: {category}")
    if confidence not in ANNOTATION_CONFIDENCES:
        raise ValueError(f"invalid annotation_confidence: {confidence}")
    if not reviewer_id.strip():
        raise ValueError("reviewed rows require reviewer_id")
    if not note.strip():
        raise ValueError("reviewed rows require a concise annotation_note")
    if status == "NEEDS_ADJUDICATION" and not secondary_required:
        raise ValueError("NEEDS_ADJUDICATION requires secondary_review_required")

    if category == "SINGLE_SUPPORTED_INTENT" and len(supported) != 1:
        raise ValueError("SINGLE_SUPPORTED_INTENT requires exactly one intent")
    if category == "MULTI_SUPPORTED_INTENT" and len(supported) < 2:
        raise ValueError("MULTI_SUPPORTED_INTENT requires at least two intents")
    if category in {
        "UNSUPPORTED",
        "UNCLEAR_OR_INSUFFICIENT",
        "NO_CURRENT_REQUEST",
    } and supported:
        raise ValueError(f"{category} requires an empty supported_intents list")


def merge_existing_annotations(
    initialized_rows: Sequence[Mapping[str, Any]],
    existing_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    initialized_by_hash = {
        row["narrative_sha256"]: row for row in initialized_rows
    }
    existing_by_hash: dict[str, Mapping[str, Any]] = {}
    for row in existing_rows:
        digest = row.get("narrative_sha256")
        if not isinstance(digest, str) or digest in existing_by_hash:
            raise ValueError("existing workfile has invalid or duplicate hashes")
        existing_by_hash[digest] = row
    if set(existing_by_hash) != set(initialized_by_hash):
        raise ValueError("existing workfile hashes differ from the frozen holdout")

    merged: list[dict[str, Any]] = []
    for initialized in initialized_rows:
        digest = initialized["narrative_sha256"]
        existing = existing_by_hash[digest]
        expected_fields = set(initialized)
        if set(existing) != expected_fields:
            raise ValueError("existing workfile schema differs from expected schema")
        for field_name in expected_fields - set(ANNOTATION_FIELDS):
            if existing[field_name] != initialized[field_name]:
                raise ValueError(
                    f"existing workfile changed immutable field {field_name}"
                )
        validate_annotation_fields(existing)
        merged.append(
            {
                **initialized,
                **{
                    field_name: existing[field_name]
                    for field_name in ANNOTATION_FIELDS
                },
            }
        )
    return merged


def build_annotation_workfile(
    source_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    output_path: Path,
    reset_existing: bool = False,
    expected_lane_counts: Mapping[str, int] = DEFAULT_POOL_LANE_COUNTS,
) -> list[dict[str, Any]]:
    manifest_records = semantic_manifest_records(
        manifest,
        expected_lane_counts=expected_lane_counts,
    )
    semantic_source_rows = [
        row
        for row in source_rows
        if row.get("mapping_status") in SEMANTIC_REVIEW_STATUSES
    ]
    source_by_hash = validate_source_against_manifest(
        semantic_source_rows,
        manifest_records,
    )
    initialized_rows = [
        {
            **source_by_hash[manifest_record["narrative_sha256"]],
            **initialized_annotation_fields(),
        }
        for manifest_record in manifest_records
    ]
    if output_path.exists() and not reset_existing:
        existing_rows, _ = load_jsonl(output_path)
        output_rows = merge_existing_annotations(initialized_rows, existing_rows)
    else:
        output_rows = initialized_rows
    atomic_write(output_path, stable_jsonl_bytes(output_rows))
    return output_rows


def validate_workfile_against_manifest(
    workfile_rows: Sequence[Mapping[str, Any]],
    manifest_records: Sequence[Mapping[str, Any]],
) -> None:
    workfile_by_hash: dict[str, Mapping[str, Any]] = {}
    for row in workfile_rows:
        validate_source_row(row)
        digest = row.get("narrative_sha256")
        if not isinstance(digest, str) or digest in workfile_by_hash:
            raise ValueError("workfile has invalid or duplicate narrative hashes")
        workfile_by_hash[digest] = row
        validate_annotation_fields(row)
    manifest_by_hash = {
        record["narrative_sha256"]: record for record in manifest_records
    }
    if set(workfile_by_hash) != set(manifest_by_hash):
        raise ValueError("workfile hashes differ from the frozen evaluation holdout")
    for digest, row in workfile_by_hash.items():
        manifest_record = manifest_by_hash[digest]
        if row.get("complaint_id") != manifest_record.get("complaint_id"):
            raise ValueError("workfile complaint ID differs from review-pool manifest")
        if row.get("mapping_status") != manifest_record.get("mapping_status"):
            raise ValueError("workfile mapping status differs from review-pool manifest")
        if row.get("candidate_sentinelvoice_intents") != manifest_record.get(
            "candidate_sentinelvoice_intents"
        ):
            raise ValueError("workfile candidate intents differ from manifest")


def count_values(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                record[field_name]
                for record in records
                if record.get(field_name) is not None
            ).items()
        )
    )


def semantic_decision(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return fields whose change constitutes a human semantic override."""
    return (
        row.get("review_category"),
        tuple(row.get("supported_intents", [])),
        row.get("annotation_confidence"),
    )


def validate_codex_first_pass_for_export(
    first_pass_rows: Sequence[Mapping[str, Any]],
    workfile_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Validate text-free Codex first-pass linkage."""
    workfile_by_hash = {row["narrative_sha256"]: row for row in workfile_rows}
    by_hash: dict[str, Mapping[str, Any]] = {}
    for row in first_pass_rows:
        digest = row.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in workfile_by_hash:
            raise ValueError("Codex first-pass row has a non-holdout hash")
        if digest in by_hash:
            raise ValueError("Codex first-pass rows contain duplicate hashes")
        if row.get("status") not in {"SUCCEEDED", "INVALID", "MISSING"}:
            raise ValueError("Codex first-pass row has an invalid status")
        if row.get("annotation_source") != "CODEX_FIRST_PASS":
            raise ValueError("Codex first-pass row has an invalid source")
        if row.get("schema_version") != "cfpb-codex-first-pass.v1":
            raise ValueError("Codex first-pass row has an invalid schema")
        if not isinstance(row.get("annotator_model"), str) or not row[
            "annotator_model"
        ].strip():
            raise ValueError("Codex first-pass row requires an annotator model")
        if not isinstance(row.get("prompt_version"), str) or not row[
            "prompt_version"
        ].strip():
            raise ValueError("Codex first-pass row requires a prompt version")
        if not isinstance(row.get("batch_id"), str) or not row["batch_id"].strip():
            raise ValueError("Codex first-pass row requires a batch identifier")
        source = workfile_by_hash[digest]
        if row.get("complaint_id") != source.get("complaint_id"):
            raise ValueError("Codex first-pass complaint ID differs from workfile")
        if row.get("original_mapping_status") != source.get("mapping_status"):
            raise ValueError("Codex first-pass mapping status differs from workfile")
        if not isinstance(row.get("human_review_required"), bool):
            raise TypeError("Codex human_review_required must be boolean")
        reasons = row.get("human_review_reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) for reason in reasons
        ):
            raise TypeError("Codex human_review_reasons must be a list of strings")
        if not isinstance(row.get("qc_sample_selected"), bool):
            raise TypeError("Codex qc_sample_selected must be boolean")
        if row["status"] == "SUCCEEDED":
            validation_row = {
                field: row.get(field) for field in ANNOTATION_FIELDS
            }
            validation_row.update(
                {
                    "adjudication_status": "REVIEWED",
                    "reviewer_id": "CODEX_FIRST_PASS",
                }
            )
            validate_annotation_fields(validation_row)
        elif not row["human_review_required"]:
            raise ValueError("unresolved Codex first pass must require human review")
        by_hash[digest] = row
    return by_hash


def hybrid_label_record(
    workfile_row: Mapping[str, Any],
    first_pass_row: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Resolve one eventual label and its internal provenance."""
    human_complete = workfile_row.get("adjudication_status") in {
        "REVIEWED",
        "ADJUDICATED",
    }
    human_started = workfile_row.get("adjudication_status") != "UNREVIEWED"
    if first_pass_row is None:
        if human_complete:
            selected = workfile_row
            provenance = "HUMAN_ONLY_LEGACY"
            unresolved = False
        else:
            selected = workfile_row
            provenance = "MISSING_CODEX_FIRST_PASS"
            unresolved = True
        review_required = True
        review_reasons = ["MISSING_CODEX_FIRST_PASS"]
        annotator_model = None
        prompt_version = None
        batch_id = None
        qc_selected = False
    else:
        review_required = bool(first_pass_row["human_review_required"])
        review_reasons = list(first_pass_row["human_review_reasons"])
        annotator_model = first_pass_row.get("annotator_model")
        prompt_version = first_pass_row.get("prompt_version")
        batch_id = first_pass_row.get("batch_id")
        qc_selected = bool(first_pass_row.get("qc_sample_selected", False))
        if first_pass_row["status"] in {"INVALID", "MISSING"}:
            if human_complete:
                selected = workfile_row
                provenance = "HUMAN_RESOLVED_CODEX_FAILURE"
                unresolved = False
            else:
                selected = workfile_row
                provenance = "CODEX_FAILURE_UNRESOLVED"
                unresolved = True
        elif human_complete:
            selected = workfile_row
            if semantic_decision(workfile_row) == semantic_decision(first_pass_row):
                provenance = "HUMAN_CONFIRMED_CODEX_FIRST_PASS"
            else:
                provenance = "HUMAN_OVERRULED_CODEX_FIRST_PASS"
            unresolved = False
        elif human_started:
            selected = workfile_row
            provenance = "HUMAN_ADJUDICATION_PENDING"
            unresolved = True
        elif review_required:
            selected = workfile_row
            provenance = "HUMAN_REVIEW_PENDING"
            unresolved = True
        else:
            selected = first_pass_row
            provenance = "CODEX_FIRST_PASS_ACCEPTED"
            unresolved = False

    adjudication_status = selected.get("adjudication_status")
    if provenance == "CODEX_FIRST_PASS_ACCEPTED":
        adjudication_status = "CODEX_ACCEPTED"
    return (
        {
            "adjudication_status": adjudication_status,
            "annotation_confidence": selected.get("annotation_confidence"),
            "annotation_note": selected.get("annotation_note"),
            "annotation_provenance": provenance,
            "candidate_sentinelvoice_intents": workfile_row[
                "candidate_sentinelvoice_intents"
            ],
            "complaint_id": workfile_row["complaint_id"],
            "human_review_reasons": review_reasons,
            "human_review_required": review_required,
            "codex_annotator_model": annotator_model,
            "codex_prompt_version": prompt_version,
            "codex_batch_id": batch_id,
            "narrative_sha256": workfile_row["narrative_sha256"],
            "original_mapping_status": workfile_row["mapping_status"],
            "qc_sample_selected": qc_selected,
            "review_category": selected.get("review_category"),
            "supported_intents": list(selected.get("supported_intents", [])),
        },
        unresolved,
    )


def export_semantic_labels(
    workfile_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    workfile_sha256: str,
    review_pool_manifest_sha256: str,
    output_path: Path,
    allow_partial: bool = False,
    codex_first_pass_rows: Sequence[Mapping[str, Any]] | None = None,
    codex_first_pass_sha256: str | None = None,
    expected_lane_counts: Mapping[str, int] = DEFAULT_POOL_LANE_COUNTS,
) -> dict[str, Any]:
    manifest_records = semantic_manifest_records(
        manifest,
        expected_lane_counts=expected_lane_counts,
    )
    validate_workfile_against_manifest(workfile_rows, manifest_records)
    workfile_by_hash = {
        row["narrative_sha256"]: row for row in workfile_rows
    }
    ordered_workfile_rows = [
        workfile_by_hash[record["narrative_sha256"]]
        for record in manifest_records
    ]
    first_pass_by_hash = (
        validate_codex_first_pass_for_export(
            codex_first_pass_rows,
            ordered_workfile_rows,
        )
        if codex_first_pass_rows is not None
        else {}
    )
    resolved = [
        hybrid_label_record(
            row,
            first_pass_by_hash.get(row["narrative_sha256"]),
        )
        for row in ordered_workfile_rows
    ]
    records = [record for record, _ in resolved]
    incomplete = [record for record, unresolved in resolved if unresolved]
    if incomplete and not allow_partial:
        raise ValueError(
            f"cannot freeze labels with {len(incomplete)} unresolved reviews; "
            "use --allow-partial only for an explicitly partial artifact"
        )

    supported_intent_counts: Counter[str] = Counter()
    for record in records:
        supported_intent_counts.update(record["supported_intents"])
    artifact = {
        "annotation_protocol": {
            "allowed_confidences": list(ANNOTATION_CONFIDENCES),
            "labeling_method": (
                "codex_first_pass_targeted_human_review_and_deterministic_qc"
                if codex_first_pass_rows is not None
                else "legacy_human_only"
            ),
            "protected_write_intents_requiring_explicit_current_action": sorted(
                PROTECTED_WRITE_INTENTS
            ),
            "review_categories": list(REVIEW_CATEGORIES),
            "supported_intents": list(SUPPORTED_INTENTS),
            "workfile_schema_version": WORKFILE_SCHEMA_VERSION,
        },
        "artifact_boundary": {
            "classifier_outputs_visible_to_reviewers": False,
            "contains_consumer_narrative_text": False,
            "external_evaluation_holdout": True,
            "codex_labels_used_for_training": False,
            "model_training_or_retraining_run": False,
            "private_reviewer_identity_included": False,
        },
        "holdout_policy": {
            "final_external_evaluation_allowed_after_labels_are_frozen": True,
            "hyperparameter_selection_prohibited": True,
            "model_selection_prohibited": True,
            "narrative_hashes_must_not_be_used_for_training": True,
            "later_cfpb_training_data_must_use_disjoint_narratives": True,
        },
        "input_integrity": {
            "review_pool_manifest_sha256": review_pool_manifest_sha256,
            "semantic_review_workfile_sha256": workfile_sha256,
            "codex_first_pass_sha256": codex_first_pass_sha256,
        },
        "partial": bool(incomplete),
        "records": records,
        "schema_version": LABEL_SCHEMA_VERSION,
        "summary": {
            "adjudication_status_counts": count_values(
                records,
                "adjudication_status",
            ),
            "annotation_confidence_counts": count_values(
                records,
                "annotation_confidence",
            ),
            "annotation_provenance_counts": count_values(
                records,
                "annotation_provenance",
            ),
            "candidate_intents_are_context_not_labels": True,
            "completed_count": len(records) - len(incomplete),
            "original_mapping_status_counts": count_values(
                records,
                "original_mapping_status",
            ),
            "review_category_counts": count_values(records, "review_category"),
            "supported_intent_counts_may_overlap": True,
            "supported_intent_counts": dict(sorted(supported_intent_counts.items())),
            "total_holdout_records": len(records),
            "unresolved_count": len(incomplete),
        },
        "workflow_version": WORKFLOW_VERSION,
    }
    serialized = stable_json_bytes(artifact)
    if any("narrative" in record for record in artifact["records"]):
        raise AssertionError("tracked label record contains narrative text")
    if any(set(record) != set(TRACKED_LABEL_FIELDS) for record in artifact["records"]):
        raise AssertionError("tracked label record schema changed unexpectedly")
    serialized_text = serialized.decode("utf-8")
    for row in ordered_workfile_rows:
        narrative = row.get("narrative")
        if isinstance(narrative, str) and narrative and narrative in serialized_text:
            raise ValueError("tracked label artifact would contain a full narrative")
    if any(FORBIDDEN_MODEL_FIELDS.intersection(record) for record in artifact["records"]):
        raise AssertionError("tracked label artifact contains classifier output fields")
    atomic_write(output_path, serialized)
    return artifact


def initialize_from_default_paths(*, reset_existing: bool) -> None:
    if reset_existing and LOCAL_WORKFILE_PATH.exists():
        raise ValueError(
            "direct reset is disabled for the real workfile; use "
            "review_cfpb_semantic_annotations.py --reset-reviewed with its "
            "required confirmation so a backup is created first"
        )
    manifest, _ = load_json_object(REVIEW_POOL_MANIFEST_PATH)
    source_rows, source_bytes = load_jsonl(LOCAL_REVIEW_POOL_PATH)
    expected_source_hash = manifest.get("local_review_material", {}).get("sha256")
    if sha256_bytes(source_bytes) != expected_source_hash:
        raise ValueError("local review pool does not match its tracked manifest hash")
    rows = build_annotation_workfile(
        source_rows,
        manifest,
        output_path=LOCAL_WORKFILE_PATH,
        reset_existing=reset_existing,
    )
    print(
        f"Initialized CFPB semantic-review workfile: rows={len(rows)}, "
        f"path={LOCAL_WORKFILE_PATH}",
        flush=True,
    )


def export_from_default_paths(*, allow_partial: bool) -> None:
    manifest, manifest_bytes = load_json_object(REVIEW_POOL_MANIFEST_PATH)
    workfile_rows, workfile_bytes = load_jsonl(LOCAL_WORKFILE_PATH)
    first_pass_rows, first_pass_bytes = load_jsonl(LOCAL_FIRST_PASS_PATH)
    artifact = export_semantic_labels(
        workfile_rows,
        manifest,
        workfile_sha256=sha256_bytes(workfile_bytes),
        review_pool_manifest_sha256=sha256_bytes(manifest_bytes),
        output_path=LABEL_ARTIFACT_PATH,
        allow_partial=allow_partial,
        codex_first_pass_rows=first_pass_rows,
        codex_first_pass_sha256=sha256_bytes(first_pass_bytes),
    )
    print(
        f"Exported CFPB semantic labels: records={len(artifact['records'])}, "
        f"partial={artifact['partial']}, path={LABEL_ARTIFACT_PATH}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser(
        "init",
        help="initialize or safely refresh the local annotation workfile",
    )
    initialize.add_argument(
        "--reset-existing",
        action="store_true",
        help="only for first initialization; real-workfile reset is disabled",
    )
    export = subparsers.add_parser(
        "export",
        help="validate and export the text-free tracked label artifact",
    )
    export.add_argument(
        "--allow-partial",
        action="store_true",
        help="permit an explicitly partial artifact with unresolved reviews",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        initialize_from_default_paths(reset_existing=args.reset_existing)
    elif args.command == "export":
        export_from_default_paths(allow_partial=args.allow_partial)
    else:  # pragma: no cover - argparse restricts the command choices.
        raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    main()
