"""Coordinate blind dual-pass Codex CFPB annotation and adjudication."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import build_cfpb_annotation_workfile as contract
    from scripts import review_cfpb_semantic_annotations as reviewer
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    import build_cfpb_annotation_workfile as contract
    import review_cfpb_semantic_annotations as reviewer


DEFAULT_INPUT_PATH = contract.LOCAL_WORKFILE_PATH
DEFAULT_FIRST_PASS_PATH = (
    contract.EXTERNAL_ROOT
    / "processed/cfpb/local/cfpb_llm_first_pass.jsonl"
)
DEFAULT_SECOND_PASS_PATH = (
    contract.EXTERNAL_ROOT
    / "processed/cfpb/local/cfpb_codex_second_pass.jsonl"
)
DEFAULT_ADJUDICATION_PATH = (
    contract.EXTERNAL_ROOT
    / "processed/cfpb/local/cfpb_codex_adjudication.jsonl"
)
DEFAULT_BATCH_DIRECTORY = (
    contract.EXTERNAL_ROOT / "processed/cfpb/local/codex_batches"
)
DEFAULT_SECOND_PASS_BATCH_DIRECTORY = (
    contract.EXTERNAL_ROOT / "processed/cfpb/local/codex_second_pass_batches"
)
DEFAULT_ADJUDICATION_BATCH_DIRECTORY = (
    contract.EXTERNAL_ROOT / "processed/cfpb/local/codex_adjudication_batches"
)
FIRST_PASS_SCHEMA_VERSION = "cfpb-codex-first-pass.v1"
BATCH_SCHEMA_VERSION = "cfpb-codex-annotation-batch.v1"
PROMPT_VERSION = "cfpb-codex-semantic-prompt.v1"
ANNOTATION_SOURCE = "CODEX_FIRST_PASS"
SECOND_PASS_SCHEMA_VERSION = "cfpb-codex-second-pass.v1"
SECOND_PASS_BATCH_SCHEMA_VERSION = "cfpb-codex-second-pass-batch.v1"
SECOND_PASS_PROMPT_VERSION = "cfpb-codex-semantic-second-pass.v1"
SECOND_PASS_ANNOTATION_SOURCE = "CODEX_SECOND_PASS"
ADJUDICATION_SCHEMA_VERSION = "cfpb-codex-adjudication.v1"
ADJUDICATION_BATCH_SCHEMA_VERSION = "cfpb-codex-adjudication-batch.v1"
ADJUDICATION_PROMPT_VERSION = "cfpb-codex-semantic-adjudication.v1"
ADJUDICATION_ANNOTATION_SOURCE = "CODEX_ADJUDICATOR"
DEFAULT_ANNOTATOR_MODEL = "codex"
QC_SAMPLE_PERCENT = 10
QC_SAMPLE_SALT = "sentinelvoice-cfpb-codex-qc-v1"
MAX_BATCH_SIZE = 200

SEMANTIC_FIELDS = (
    "review_category",
    "supported_intents",
    "annotation_confidence",
    "annotation_note",
    "secondary_review_required",
)
CODEX_RESULT_FIELDS = (
    "annotation_source",
    "annotator_model",
    "prompt_version",
    "batch_id",
    "complaint_id",
    "narrative_sha256",
    *SEMANTIC_FIELDS,
)
FIRST_PASS_FIELDS = (
    "schema_version",
    "status",
    "annotation_source",
    "annotator_model",
    "prompt_version",
    "batch_id",
    "complaint_id",
    "narrative_sha256",
    "original_mapping_status",
    *SEMANTIC_FIELDS,
    "codex_requested_secondary_review",
    "human_review_required",
    "human_review_reasons",
    "qc_sample_selected",
    "validation_error_type",
)
BATCH_FIELDS = (
    "schema_version",
    "annotation_source",
    "annotator_model",
    "prompt_version",
    "batch_id",
    "batch_position",
    "complaint_id",
    "narrative_sha256",
    "narrative",
    "original_mapping_status",
    "taxonomy_candidate_intents_hints_only",
    "cfpb_product",
    "cfpb_sub_product",
    "cfpb_issue",
    "cfpb_sub_issue",
)
FIRST_PASS_STATUSES = frozenset({"SUCCEEDED", "INVALID", "MISSING"})
SECOND_PASS_STATUSES = FIRST_PASS_STATUSES
ADJUDICATION_STORE_STATUSES = FIRST_PASS_STATUSES
ADJUDICATION_RESOLUTION_STATUSES = frozenset({"RESOLVED", "UNRESOLVED"})
SECOND_PASS_FIELDS = (
    "schema_version",
    "status",
    "annotation_source",
    "annotator_model",
    "prompt_version",
    "batch_id",
    "complaint_id",
    "narrative_sha256",
    "original_mapping_status",
    *SEMANTIC_FIELDS,
    "validation_error_type",
)
ADJUDICATION_BATCH_FIELDS = (
    "schema_version",
    "annotation_source",
    "annotator_model",
    "prompt_version",
    "batch_id",
    "batch_position",
    "complaint_id",
    "narrative_sha256",
    "narrative",
    "original_mapping_status",
    "taxonomy_candidate_intents_hints_only",
    "cfpb_product",
    "cfpb_sub_product",
    "cfpb_issue",
    "cfpb_sub_issue",
    "pass_a",
    "pass_b",
    "adjudication_reasons",
)
ADJUDICATION_SEMANTIC_FIELDS = (
    "final_review_category",
    "final_supported_intents",
    "final_annotation_confidence",
    "final_annotation_note",
)
ADJUDICATION_RESULT_FIELDS = (
    "annotation_source",
    "annotator_model",
    "prompt_version",
    "batch_id",
    "complaint_id",
    "narrative_sha256",
    "resolution_status",
    *ADJUDICATION_SEMANTIC_FIELDS,
    "unresolved_reason",
)
ADJUDICATION_STORE_FIELDS = (
    "schema_version",
    "status",
    *ADJUDICATION_RESULT_FIELDS,
    "validation_error_type",
)


def deterministic_qc_selected(narrative_sha256: str) -> bool:
    """Select a stable 10% QC sample without classifier-derived information."""
    digest = contract.sha256_bytes(
        f"{QC_SAMPLE_SALT}:{narrative_sha256}".encode()
    )
    return int(digest[:8], 16) % 100 < QC_SAMPLE_PERCENT


def human_review_reasons(
    annotation: Mapping[str, Any],
    *,
    narrative_sha256: str,
) -> list[str]:
    """Return deterministic mandatory-review reasons for a valid annotation."""
    reasons: list[str] = []
    if annotation["annotation_confidence"] == "LOW":
        reasons.append("LOW_CONFIDENCE")
    if annotation["review_category"] == "UNCLEAR_OR_INSUFFICIENT":
        reasons.append("UNCLEAR_OR_INSUFFICIENT")
    if annotation["review_category"] == "MULTI_SUPPORTED_INTENT":
        reasons.append("MULTI_SUPPORTED_INTENT")
    supported = set(annotation["supported_intents"])
    if "freeze_card" in supported:
        reasons.append("PROTECTED_WRITE_FREEZE_CARD")
    if "create_dispute" in supported:
        reasons.append("PROTECTED_WRITE_CREATE_DISPUTE")
    if annotation["secondary_review_required"]:
        reasons.append("CODEX_REQUESTED_SECONDARY_REVIEW")
    if not reasons and deterministic_qc_selected(narrative_sha256):
        reasons.append("DETERMINISTIC_QC_SAMPLE")
    return reasons


def batch_source_row(
    source: Mapping[str, Any],
    *,
    batch_id: str,
    position: int,
    annotator_model: str,
) -> dict[str, Any]:
    """Allowlist batch fields so V2-C1 and prior human labels cannot leak."""
    row = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "annotation_source": ANNOTATION_SOURCE,
        "annotator_model": annotator_model,
        "prompt_version": PROMPT_VERSION,
        "batch_id": batch_id,
        "batch_position": position,
        "complaint_id": source.get("complaint_id"),
        "narrative_sha256": source.get("narrative_sha256"),
        "narrative": source.get("narrative"),
        "original_mapping_status": source.get("mapping_status"),
        "taxonomy_candidate_intents_hints_only": source.get(
            "candidate_sentinelvoice_intents",
            [],
        ),
        "cfpb_product": source.get("product"),
        "cfpb_sub_product": source.get("sub_product"),
        "cfpb_issue": source.get("issue"),
        "cfpb_sub_issue": source.get("sub_issue"),
    }
    if any(
        reviewer.is_model_related_field(field)
        for field in row
        if field not in {"annotator_model"}
    ):
        raise AssertionError("Codex batch allowlist contains a classifier field")
    return row


def second_pass_batch_source_row(
    source: Mapping[str, Any],
    *,
    batch_id: str,
    position: int,
    annotator_model: str,
) -> dict[str, Any]:
    """Build a blind Pass-B row from source-only fields."""
    row = batch_source_row(
        source,
        batch_id=batch_id,
        position=position,
        annotator_model=annotator_model,
    )
    row.update(
        {
            "schema_version": SECOND_PASS_BATCH_SCHEMA_VERSION,
            "annotation_source": SECOND_PASS_ANNOTATION_SOURCE,
            "prompt_version": SECOND_PASS_PROMPT_VERSION,
        }
    )
    if set(row) != set(BATCH_FIELDS):
        raise AssertionError("Pass-B batch row differs from its source allowlist")
    if set(row).intersection(SEMANTIC_FIELDS):
        raise AssertionError("Pass-B batch row exposes Pass-A semantic fields")
    return row


def _validate_semantic_result(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
    *,
    expected_source: str,
) -> dict[str, Any]:
    """Validate one semantic result against its exact prepared batch row."""
    if set(result) != set(CODEX_RESULT_FIELDS):
        raise ValueError("Codex result fields differ from the required schema")
    comparisons = (
        ("annotation_source", expected_source),
        ("annotator_model", batch_row["annotator_model"]),
        ("prompt_version", batch_row["prompt_version"]),
        ("batch_id", batch_row["batch_id"]),
        ("complaint_id", batch_row["complaint_id"]),
        ("narrative_sha256", batch_row["narrative_sha256"]),
    )
    for field, expected in comparisons:
        if result.get(field) != expected:
            raise ValueError(f"Codex result {field} differs from its batch")
    annotator_model = result["annotator_model"]
    if not isinstance(annotator_model, str) or not annotator_model.strip():
        raise ValueError("annotator_model must be a non-empty string")

    validation_row = {field: result[field] for field in SEMANTIC_FIELDS}
    validation_row.update(
        {
            "adjudication_status": "REVIEWED",
            "reviewer_id": expected_source,
        }
    )
    contract.validate_annotation_fields(validation_row)
    return dict(result)


def validate_codex_result(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one Codex result against its batch and the frozen C2P rules."""
    return _validate_semantic_result(
        result,
        batch_row,
        expected_source=ANNOTATION_SOURCE,
    )


def successful_first_pass_record(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a validated, text-free first-pass row with review flags."""
    validated = validate_codex_result(result, batch_row)
    digest = validated["narrative_sha256"]
    reasons = human_review_reasons(validated, narrative_sha256=digest)
    requested_secondary = validated["secondary_review_required"]
    return {
        "schema_version": FIRST_PASS_SCHEMA_VERSION,
        "status": "SUCCEEDED",
        "annotation_source": ANNOTATION_SOURCE,
        "annotator_model": validated["annotator_model"],
        "prompt_version": validated["prompt_version"],
        "batch_id": validated["batch_id"],
        "complaint_id": validated["complaint_id"],
        "narrative_sha256": digest,
        "original_mapping_status": batch_row["original_mapping_status"],
        "review_category": validated["review_category"],
        "supported_intents": list(validated["supported_intents"]),
        "annotation_confidence": validated["annotation_confidence"],
        "annotation_note": validated["annotation_note"],
        "secondary_review_required": bool(requested_secondary or reasons),
        "codex_requested_secondary_review": requested_secondary,
        "human_review_required": bool(reasons),
        "human_review_reasons": reasons,
        "qc_sample_selected": "DETERMINISTIC_QC_SAMPLE" in reasons,
        "validation_error_type": None,
    }


def unresolved_first_pass_record(
    batch_row: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    validation_error: BaseException | None = None,
) -> dict[str, Any]:
    """Record missing/invalid output without accepting a semantic label."""
    if status not in {"INVALID", "MISSING"}:
        raise ValueError("unresolved first-pass status must be INVALID or MISSING")
    return {
        "schema_version": FIRST_PASS_SCHEMA_VERSION,
        "status": status,
        "annotation_source": ANNOTATION_SOURCE,
        "annotator_model": batch_row["annotator_model"],
        "prompt_version": batch_row["prompt_version"],
        "batch_id": batch_row["batch_id"],
        "complaint_id": batch_row["complaint_id"],
        "narrative_sha256": batch_row["narrative_sha256"],
        "original_mapping_status": batch_row["original_mapping_status"],
        "review_category": None,
        "supported_intents": [],
        "annotation_confidence": None,
        "annotation_note": "",
        "secondary_review_required": True,
        "codex_requested_secondary_review": False,
        "human_review_required": True,
        "human_review_reasons": [reason],
        "qc_sample_selected": False,
        "validation_error_type": (
            type(validation_error).__name__ if validation_error is not None else None
        ),
    }


def validate_first_pass_record(record: Mapping[str, Any]) -> None:
    """Validate one persisted Codex success or unresolved row."""
    if set(record) != set(FIRST_PASS_FIELDS):
        raise ValueError("Codex first-pass fields differ from the required schema")
    if record.get("schema_version") != FIRST_PASS_SCHEMA_VERSION:
        raise ValueError("unexpected Codex first-pass schema")
    status = record.get("status")
    if status not in FIRST_PASS_STATUSES:
        raise ValueError("invalid Codex first-pass status")
    if record.get("annotation_source") != ANNOTATION_SOURCE:
        raise ValueError("invalid Codex annotation source")
    if record.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("invalid Codex prompt version")
    for field in ("annotator_model", "batch_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    digest = record.get("narrative_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Codex first-pass row requires a narrative SHA-256")
    reasons = record.get("human_review_reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(reason, str) for reason in reasons
    ):
        raise TypeError("human_review_reasons must be a list of strings")
    for field in (
        "secondary_review_required",
        "codex_requested_secondary_review",
        "human_review_required",
        "qc_sample_selected",
    ):
        if not isinstance(record.get(field), bool):
            raise TypeError(f"{field} must be boolean")

    if status != "SUCCEEDED":
        expected_reason = f"CODEX_ANNOTATION_{status}"
        if reasons != [expected_reason] or not record["human_review_required"]:
            raise ValueError("unresolved Codex row has inconsistent review flags")
        if any(
            (
                record["review_category"] is not None,
                bool(record["supported_intents"]),
                record["annotation_confidence"] is not None,
                bool(record["annotation_note"]),
                not record["secondary_review_required"],
                record["codex_requested_secondary_review"],
                record["qc_sample_selected"],
            )
        ):
            raise ValueError("unresolved Codex row must not contain a semantic label")
        error_type = record["validation_error_type"]
        if status == "INVALID" and not isinstance(error_type, str):
            raise TypeError("INVALID Codex rows require validation_error_type")
        if status == "MISSING" and error_type is not None:
            raise ValueError("MISSING Codex rows cannot have validation_error_type")
        return

    validation_row = {
        field: record.get(field) for field in SEMANTIC_FIELDS
    }
    validation_row.update(
        {
            "adjudication_status": "REVIEWED",
            "reviewer_id": ANNOTATION_SOURCE,
        }
    )
    contract.validate_annotation_fields(validation_row)
    raw_annotation = {
        **validation_row,
        "secondary_review_required": record[
            "codex_requested_secondary_review"
        ],
    }
    expected_reasons = human_review_reasons(
        raw_annotation,
        narrative_sha256=digest,
    )
    if reasons != expected_reasons:
        raise ValueError("Codex human-review reasons are not reproducible")
    if record["human_review_required"] != bool(expected_reasons):
        raise ValueError("Codex human-review flag is inconsistent")
    if record["qc_sample_selected"] != (
        "DETERMINISTIC_QC_SAMPLE" in expected_reasons
    ):
        raise ValueError("Codex QC flag is inconsistent")


def load_first_pass_records(path: Path) -> list[dict[str, Any]]:
    records, _ = contract.load_jsonl(path)
    seen: set[str] = set()
    for record in records:
        validate_first_pass_record(record)
        digest = record["narrative_sha256"]
        if digest in seen:
            raise ValueError("Codex first-pass rows contain duplicate hashes")
        seen.add(digest)
    return records


def validate_first_pass_source_linkage(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
) -> None:
    source_by_hash = {
        record["narrative_sha256"]: record for record in source_records
    }
    for first_pass in first_pass_records:
        source = source_by_hash.get(first_pass["narrative_sha256"])
        if source is None:
            raise ValueError("Codex first pass contains a non-holdout hash")
        if first_pass.get("complaint_id") != source.get("complaint_id"):
            raise ValueError("Codex complaint ID differs from source")
        if first_pass.get("original_mapping_status") != source.get(
            "mapping_status"
        ):
            raise ValueError("Codex mapping status differs from source")


def first_pass_progress(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    statuses = Counter(record["status"] for record in first_pass_records)
    return {
        "total": len(source_records),
        "succeeded": statuses["SUCCEEDED"],
        "invalid": statuses["INVALID"],
        "missing": statuses["MISSING"],
        "pending": len(source_records) - len(first_pass_records),
        "human_review_required": sum(
            bool(record["human_review_required"])
            for record in first_pass_records
        ),
    }


def pending_source_indices(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Select absent or unresolved rows while never relabeling successes."""
    by_hash = {
        record["narrative_sha256"]: record for record in first_pass_records
    }
    return [
        index
        for index, source in enumerate(source_records)
        if by_hash.get(source["narrative_sha256"], {}).get("status")
        != "SUCCEEDED"
    ]


def prepare_batch(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    batch_id: str | None = None,
    annotator_model: str = DEFAULT_ANNOTATOR_MODEL,
) -> tuple[str, list[dict[str, Any]]]:
    """Create the next deterministic small batch without altering labels."""
    if not 1 <= limit <= MAX_BATCH_SIZE:
        raise ValueError(f"batch limit must be between 1 and {MAX_BATCH_SIZE}")
    if not annotator_model.strip():
        raise ValueError("annotator_model must be non-empty")
    validate_first_pass_source_linkage(source_records, first_pass_records)
    selected_indices = pending_source_indices(
        source_records,
        first_pass_records,
    )[:limit]
    if not selected_indices:
        return batch_id or "codex-batch-complete", []
    if batch_id is None:
        first_index = selected_indices[0]
        first_hash = source_records[first_index]["narrative_sha256"]
        batch_id = f"codex-batch-{first_index + 1:04d}-{first_hash[:8]}"
    if not batch_id.strip():
        raise ValueError("batch_id must be non-empty")
    return batch_id, [
        batch_source_row(
            source_records[index],
            batch_id=batch_id,
            position=position,
            annotator_model=annotator_model,
        )
        for position, index in enumerate(selected_indices, start=1)
    ]


def _validate_semantic_batch_records(
    batch_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    *,
    expected_schema: str,
    expected_source: str,
    expected_prompt: str,
) -> None:
    if not batch_records:
        raise ValueError("Codex annotation batch must not be empty")
    source_by_hash = {
        record["narrative_sha256"]: record for record in source_records
    }
    seen: set[str] = set()
    batch_ids: set[str] = set()
    positions: list[int] = []
    for batch_row in batch_records:
        if set(batch_row) != set(BATCH_FIELDS):
            raise ValueError("Codex batch fields differ from the required schema")
        if batch_row.get("schema_version") != expected_schema:
            raise ValueError("unexpected Codex batch schema")
        if batch_row.get("annotation_source") != expected_source:
            raise ValueError("Codex batch has an invalid annotation source")
        if batch_row.get("prompt_version") != expected_prompt:
            raise ValueError("Codex batch has an invalid prompt version")
        if not isinstance(batch_row.get("annotator_model"), str) or not batch_row[
            "annotator_model"
        ].strip():
            raise ValueError("Codex batch requires an annotator model")
        if not isinstance(batch_row.get("batch_position"), int) or batch_row[
            "batch_position"
        ] < 1:
            raise ValueError("Codex batch position must be positive")
        positions.append(batch_row["batch_position"])
        digest = batch_row.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in source_by_hash:
            raise ValueError("Codex batch contains a non-holdout hash")
        if digest in seen:
            raise ValueError("Codex batch contains duplicate hashes")
        seen.add(digest)
        batch_ids.add(str(batch_row.get("batch_id", "")))
        source = source_by_hash[digest]
        if batch_row.get("complaint_id") != source.get("complaint_id"):
            raise ValueError("Codex batch complaint ID differs from source")
        narrative = batch_row.get("narrative")
        if narrative != source.get("narrative"):
            raise ValueError("Codex batch narrative differs from source")
        if contract.sha256_bytes(str(narrative).encode("utf-8")) != digest:
            raise ValueError("Codex batch narrative does not match its SHA-256")
        comparisons = {
            "original_mapping_status": source.get("mapping_status"),
            "taxonomy_candidate_intents_hints_only": source.get(
                "candidate_sentinelvoice_intents",
                [],
            ),
            "cfpb_product": source.get("product"),
            "cfpb_sub_product": source.get("sub_product"),
            "cfpb_issue": source.get("issue"),
            "cfpb_sub_issue": source.get("sub_issue"),
        }
        for field, expected in comparisons.items():
            if batch_row.get(field) != expected:
                raise ValueError(f"Codex batch {field} differs from source")
    if len(batch_ids) != 1 or not next(iter(batch_ids)).strip():
        raise ValueError("Codex batch must contain one non-empty batch_id")
    if positions != list(range(1, len(batch_records) + 1)):
        raise ValueError("Codex batch positions must be contiguous and ordered")


def validate_batch_records(
    batch_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> None:
    _validate_semantic_batch_records(
        batch_records,
        source_records,
        expected_schema=BATCH_SCHEMA_VERSION,
        expected_source=ANNOTATION_SOURCE,
        expected_prompt=PROMPT_VERSION,
    )


def validate_second_pass_batch_records(
    batch_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> None:
    _validate_semantic_batch_records(
        batch_records,
        source_records,
        expected_schema=SECOND_PASS_BATCH_SCHEMA_VERSION,
        expected_source=SECOND_PASS_ANNOTATION_SOURCE,
        expected_prompt=SECOND_PASS_PROMPT_VERSION,
    )
    for row in batch_records:
        if set(row).intersection(SEMANTIC_FIELDS):
            raise ValueError("Pass-B batch exposes Pass-A semantic fields")


def merge_first_pass_records(
    source_records: Sequence[Mapping[str, Any]],
    existing_records: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_hash = {
        record["narrative_sha256"]: dict(record) for record in existing_records
    }
    for replacement in replacements:
        by_hash[replacement["narrative_sha256"]] = dict(replacement)
    source_hashes = {record["narrative_sha256"] for record in source_records}
    if set(by_hash) - source_hashes:
        raise ValueError("Codex first pass contains hashes outside the holdout")
    return [
        by_hash[source["narrative_sha256"]]
        for source in source_records
        if source["narrative_sha256"] in by_hash
    ]


def import_codex_results(
    source_records: Sequence[Mapping[str, Any]],
    existing_records: Sequence[Mapping[str, Any]],
    batch_records: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    *,
    replace_successful: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate and merge one batch, marking invalid/missing rows for review."""
    validate_first_pass_source_linkage(source_records, existing_records)
    validate_batch_records(batch_records, source_records)
    batch_by_hash = {
        row["narrative_sha256"]: row for row in batch_records
    }
    existing_by_hash = {
        row["narrative_sha256"]: row for row in existing_records
    }
    results_by_hash: dict[str, Mapping[str, Any]] = {}
    for result in result_records:
        digest = result.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in batch_by_hash:
            raise ValueError("Codex result contains a hash outside its batch")
        if digest in results_by_hash:
            raise ValueError("Codex results contain a duplicate hash")
        results_by_hash[digest] = result

    replacements: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for digest, batch_row in batch_by_hash.items():
        existing = existing_by_hash.get(digest)
        result = results_by_hash.get(digest)
        if existing is not None and existing["status"] == "SUCCEEDED":
            if result is None:
                continue
            if not replace_successful:
                raise ValueError(
                    "refusing to overwrite a successful Codex annotation"
                )
        if result is None:
            replacement = unresolved_first_pass_record(
                batch_row,
                status="MISSING",
                reason="CODEX_ANNOTATION_MISSING",
            )
        else:
            try:
                replacement = successful_first_pass_record(result, batch_row)
            except (TypeError, ValueError) as error:
                if existing is not None and existing["status"] == "SUCCEEDED":
                    raise ValueError(
                        "replacement Codex annotation is invalid; successful "
                        "annotation was preserved"
                    ) from error
                replacement = unresolved_first_pass_record(
                    batch_row,
                    status="INVALID",
                    reason="CODEX_ANNOTATION_INVALID",
                    validation_error=error,
                )
        validate_first_pass_record(replacement)
        replacements.append(replacement)
        counts[replacement["status"].lower()] += 1

    merged = merge_first_pass_records(
        source_records,
        existing_records,
        replacements,
    )
    return merged, {
        "succeeded": counts["succeeded"],
        "invalid": counts["invalid"],
        "missing": counts["missing"],
    }


def second_pass_required_hashes(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Derive Pass-B membership from validated Pass-A review flags."""
    validate_first_pass_source_linkage(source_records, first_pass_records)
    first_by_hash = {
        record["narrative_sha256"]: record for record in first_pass_records
    }
    if len(first_by_hash) != len(source_records):
        raise ValueError("Pass B requires one completed Pass-A row per source row")
    return [
        source["narrative_sha256"]
        for source in source_records
        if first_by_hash.get(source["narrative_sha256"], {}).get(
            "human_review_required"
        )
    ]


def successful_second_pass_record(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a validated, text-free Pass-B success row."""
    validated = _validate_semantic_result(
        result,
        batch_row,
        expected_source=SECOND_PASS_ANNOTATION_SOURCE,
    )
    return {
        "schema_version": SECOND_PASS_SCHEMA_VERSION,
        "status": "SUCCEEDED",
        "annotation_source": SECOND_PASS_ANNOTATION_SOURCE,
        "annotator_model": validated["annotator_model"],
        "prompt_version": validated["prompt_version"],
        "batch_id": validated["batch_id"],
        "complaint_id": validated["complaint_id"],
        "narrative_sha256": validated["narrative_sha256"],
        "original_mapping_status": batch_row["original_mapping_status"],
        **{field: validated[field] for field in SEMANTIC_FIELDS},
        "validation_error_type": None,
    }


def unresolved_second_pass_record(
    batch_row: Mapping[str, Any],
    *,
    status: str,
    validation_error: BaseException | None = None,
) -> dict[str, Any]:
    """Build a retryable Pass-B invalid or missing row without a label."""
    if status not in {"INVALID", "MISSING"}:
        raise ValueError("unresolved Pass-B status must be INVALID or MISSING")
    return {
        "schema_version": SECOND_PASS_SCHEMA_VERSION,
        "status": status,
        "annotation_source": SECOND_PASS_ANNOTATION_SOURCE,
        "annotator_model": batch_row["annotator_model"],
        "prompt_version": batch_row["prompt_version"],
        "batch_id": batch_row["batch_id"],
        "complaint_id": batch_row["complaint_id"],
        "narrative_sha256": batch_row["narrative_sha256"],
        "original_mapping_status": batch_row["original_mapping_status"],
        "review_category": None,
        "supported_intents": [],
        "annotation_confidence": None,
        "annotation_note": "",
        "secondary_review_required": True,
        "validation_error_type": (
            type(validation_error).__name__ if validation_error is not None else None
        ),
    }


def validate_second_pass_record(record: Mapping[str, Any]) -> None:
    """Validate one persisted Pass-B success or retryable row."""
    if set(record) != set(SECOND_PASS_FIELDS):
        raise ValueError("Pass-B fields differ from the required schema")
    if record.get("schema_version") != SECOND_PASS_SCHEMA_VERSION:
        raise ValueError("unexpected Pass-B schema")
    if record.get("annotation_source") != SECOND_PASS_ANNOTATION_SOURCE:
        raise ValueError("invalid Pass-B annotation source")
    if record.get("prompt_version") != SECOND_PASS_PROMPT_VERSION:
        raise ValueError("invalid Pass-B prompt version")
    status = record.get("status")
    if status not in SECOND_PASS_STATUSES:
        raise ValueError("invalid Pass-B status")
    for field in ("annotator_model", "batch_id", "complaint_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pass-B {field} must be a non-empty string")
    digest = record.get("narrative_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Pass-B row requires a narrative SHA-256")
    if record.get("original_mapping_status") not in (
        contract.SEMANTIC_REVIEW_STATUSES
    ):
        raise ValueError("Pass-B row has an invalid original mapping status")

    if status != "SUCCEEDED":
        if record.get("supported_intents") != []:
            raise ValueError("unresolved Pass-B supported_intents must be empty")
        if record.get("annotation_note") != "":
            raise ValueError("unresolved Pass-B annotation_note must be empty")
        if record.get("secondary_review_required") is not True:
            raise ValueError("unresolved Pass-B row must remain retryable")
        if any(
            (
                record["review_category"] is not None,
                record["annotation_confidence"] is not None,
            )
        ):
            raise ValueError("unresolved Pass-B row must not contain a label")
        error_type = record["validation_error_type"]
        if status == "INVALID" and not isinstance(error_type, str):
            raise TypeError("INVALID Pass-B rows require validation_error_type")
        if status == "MISSING" and error_type is not None:
            raise ValueError("MISSING Pass-B rows cannot have validation_error_type")
        return

    validation_row = {field: record[field] for field in SEMANTIC_FIELDS}
    validation_row.update(
        {
            "adjudication_status": "REVIEWED",
            "reviewer_id": SECOND_PASS_ANNOTATION_SOURCE,
        }
    )
    contract.validate_annotation_fields(validation_row)
    if record["validation_error_type"] is not None:
        raise ValueError("successful Pass-B rows cannot have validation errors")


def load_second_pass_records(path: Path) -> list[dict[str, Any]]:
    records, _ = contract.load_jsonl(path)
    seen: set[str] = set()
    for record in records:
        validate_second_pass_record(record)
        digest = record["narrative_sha256"]
        if digest in seen:
            raise ValueError("Pass-B rows contain duplicate hashes")
        seen.add(digest)
    return records


def validate_second_pass_source_linkage(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
) -> None:
    required = set(second_pass_required_hashes(source_records, first_pass_records))
    source_by_hash = {
        record["narrative_sha256"]: record for record in source_records
    }
    for second_pass in second_pass_records:
        digest = second_pass["narrative_sha256"]
        if digest not in required:
            raise ValueError("Pass-B store contains a hash not selected by Pass A")
        source = source_by_hash[digest]
        if second_pass.get("complaint_id") != source.get("complaint_id"):
            raise ValueError("Pass-B complaint ID differs from source")
        if second_pass.get("original_mapping_status") != source.get(
            "mapping_status"
        ):
            raise ValueError("Pass-B mapping status differs from source")


def prepare_second_pass_batch(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    batch_id: str | None = None,
    annotator_model: str = DEFAULT_ANNOTATOR_MODEL,
) -> tuple[str, list[dict[str, Any]]]:
    """Prepare the next deterministic blind Pass-B batch."""
    if not 1 <= limit <= MAX_BATCH_SIZE:
        raise ValueError(f"batch limit must be between 1 and {MAX_BATCH_SIZE}")
    if not annotator_model.strip():
        raise ValueError("annotator_model must be non-empty")
    validate_second_pass_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    required = set(second_pass_required_hashes(source_records, first_pass_records))
    second_by_hash = {
        record["narrative_sha256"]: record for record in second_pass_records
    }
    selected = [
        source
        for source in source_records
        if source["narrative_sha256"] in required
        and second_by_hash.get(source["narrative_sha256"], {}).get("status")
        != "SUCCEEDED"
    ][:limit]
    if not selected:
        return batch_id or "codex-second-pass-complete", []
    if batch_id is None:
        first_index = next(
            index
            for index, source in enumerate(source_records)
            if source["narrative_sha256"] == selected[0]["narrative_sha256"]
        )
        batch_id = (
            f"codex-second-pass-{first_index + 1:04d}-"
            f"{selected[0]['narrative_sha256'][:8]}"
        )
    if not batch_id.strip():
        raise ValueError("batch_id must be non-empty")
    return batch_id, [
        second_pass_batch_source_row(
            source,
            batch_id=batch_id,
            position=position,
            annotator_model=annotator_model,
        )
        for position, source in enumerate(selected, start=1)
    ]


def import_second_pass_results(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    existing_records: Sequence[Mapping[str, Any]],
    batch_records: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    *,
    replace_successful: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate and merge Pass-B results without exposing Pass-A semantics."""
    validate_second_pass_source_linkage(
        source_records,
        first_pass_records,
        existing_records,
    )
    validate_second_pass_batch_records(batch_records, source_records)
    required = set(second_pass_required_hashes(source_records, first_pass_records))
    batch_by_hash = {row["narrative_sha256"]: row for row in batch_records}
    if set(batch_by_hash) - required:
        raise ValueError("Pass-B batch contains a hash not selected by Pass A")
    existing_by_hash = {
        row["narrative_sha256"]: row for row in existing_records
    }
    results_by_hash: dict[str, Mapping[str, Any]] = {}
    for result in result_records:
        digest = result.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in batch_by_hash:
            raise ValueError("Pass-B result contains a hash outside its batch")
        if digest in results_by_hash:
            raise ValueError("Pass-B results contain a duplicate hash")
        results_by_hash[digest] = result

    replacements: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for digest, batch_row in batch_by_hash.items():
        existing = existing_by_hash.get(digest)
        result = results_by_hash.get(digest)
        if existing is not None and existing["status"] == "SUCCEEDED":
            if result is None:
                continue
            if not replace_successful:
                raise ValueError("refusing to overwrite a successful Pass-B result")
        if result is None:
            replacement = unresolved_second_pass_record(
                batch_row,
                status="MISSING",
            )
        else:
            try:
                replacement = successful_second_pass_record(result, batch_row)
            except (TypeError, ValueError) as error:
                if existing is not None and existing["status"] == "SUCCEEDED":
                    raise ValueError(
                        "replacement Pass-B result is invalid; successful result "
                        "was preserved"
                    ) from error
                replacement = unresolved_second_pass_record(
                    batch_row,
                    status="INVALID",
                    validation_error=error,
                )
        validate_second_pass_record(replacement)
        replacements.append(replacement)
        counts[replacement["status"].lower()] += 1

    by_hash = {
        record["narrative_sha256"]: dict(record) for record in existing_records
    }
    by_hash.update(
        {record["narrative_sha256"]: record for record in replacements}
    )
    merged = [
        by_hash[source["narrative_sha256"]]
        for source in source_records
        if source["narrative_sha256"] in by_hash
    ]
    validate_second_pass_source_linkage(
        source_records,
        first_pass_records,
        merged,
    )
    return merged, {
        "succeeded": counts["succeeded"],
        "invalid": counts["invalid"],
        "missing": counts["missing"],
    }


def normalize_supported_intents(value: Sequence[str]) -> tuple[str, ...]:
    """Return the stable comparison form for a validated intent collection."""
    return tuple(sorted(set(value)))


def pass_a_semantic_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only adjudication-safe Pass-A provenance and semantic fields."""
    succeeded = record.get("status") == "SUCCEEDED"
    return {
        "status": record.get("status"),
        "annotation_source": record.get("annotation_source"),
        "annotator_model": record.get("annotator_model"),
        "prompt_version": record.get("prompt_version"),
        "batch_id": record.get("batch_id"),
        "review_category": record.get("review_category") if succeeded else None,
        "supported_intents": (
            list(record.get("supported_intents", [])) if succeeded else []
        ),
        "annotation_confidence": (
            record.get("annotation_confidence") if succeeded else None
        ),
        "annotation_note": record.get("annotation_note", "") if succeeded else "",
        # The first-pass store elevates this workflow flag for every review
        # reason, including QC. Preserve the model's original semantic request.
        "secondary_review_required": (
            bool(record.get("codex_requested_secondary_review"))
            if succeeded
            else True
        ),
    }


def pass_b_semantic_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only adjudication-safe Pass-B provenance and semantic fields."""
    succeeded = record.get("status") == "SUCCEEDED"
    return {
        "status": record.get("status"),
        "annotation_source": record.get("annotation_source"),
        "annotator_model": record.get("annotator_model"),
        "prompt_version": record.get("prompt_version"),
        "batch_id": record.get("batch_id"),
        "review_category": record.get("review_category") if succeeded else None,
        "supported_intents": (
            list(record.get("supported_intents", [])) if succeeded else []
        ),
        "annotation_confidence": (
            record.get("annotation_confidence") if succeeded else None
        ),
        "annotation_note": record.get("annotation_note", "") if succeeded else "",
        "secondary_review_required": (
            bool(record.get("secondary_review_required")) if succeeded else True
        ),
    }


def compare_annotation_passes(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compare Pass A/B deterministically and identify Pass-C routing."""
    validate_second_pass_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    required = set(second_pass_required_hashes(source_records, first_pass_records))
    first_by_hash = {
        record["narrative_sha256"]: record for record in first_pass_records
    }
    second_by_hash = {
        record["narrative_sha256"]: record for record in second_pass_records
    }
    comparisons: list[dict[str, Any]] = []
    summary: Counter[str] = Counter()
    summary["pass_b_total"] = len(required)

    for source in source_records:
        digest = source["narrative_sha256"]
        if digest not in required:
            continue
        first = first_by_hash[digest]
        second = second_by_hash.get(digest)
        second_succeeded = second is not None and second["status"] == "SUCCEEDED"
        reasons: list[str] = []
        exact_semantic_agreement = False
        strong_agreement = False
        safe_agreement = False
        low_case = False
        unclear_case = False
        multi_case = False
        freeze_case = False
        dispute_case = False
        secondary_case = False

        if not second_succeeded:
            summary["pass_b_pending"] += 1
        else:
            summary["pass_b_complete"] += 1
            if first["status"] != "SUCCEEDED":
                reasons.append(f"PASS_A_{first['status']}")
            else:
                first_intents = normalize_supported_intents(
                    first["supported_intents"]
                )
                second_intents = normalize_supported_intents(
                    second["supported_intents"]
                )
                exact_semantic_agreement = (
                    first["review_category"] == second["review_category"]
                    and first_intents == second_intents
                )
                low_case = (
                    first["annotation_confidence"] == "LOW"
                    or second["annotation_confidence"] == "LOW"
                )
                strong_agreement = exact_semantic_agreement and not low_case
                categories = {
                    first["review_category"],
                    second["review_category"],
                }
                combined_intents = set(first_intents) | set(second_intents)
                unclear_case = "UNCLEAR_OR_INSUFFICIENT" in categories
                multi_case = "MULTI_SUPPORTED_INTENT" in categories
                freeze_case = "freeze_card" in combined_intents
                dispute_case = "create_dispute" in combined_intents
                secondary_case = bool(
                    first.get("codex_requested_secondary_review")
                    or second["secondary_review_required"]
                )

                if not exact_semantic_agreement:
                    reasons.append("SEMANTIC_DISAGREEMENT")
                    summary["ab_disagreements"] += 1
                if low_case:
                    reasons.append("LOW_CONFIDENCE")
                if unclear_case:
                    reasons.append("UNCLEAR_OR_INSUFFICIENT")
                if multi_case:
                    reasons.append("MULTI_SUPPORTED_INTENT")
                if freeze_case:
                    reasons.append("PROTECTED_WRITE_FREEZE_CARD")
                if dispute_case:
                    reasons.append("PROTECTED_WRITE_CREATE_DISPUTE")
                if secondary_case:
                    reasons.append("SECONDARY_REVIEW_REQUIRED")
                safe_agreement = strong_agreement and not reasons

            if strong_agreement:
                summary["exact_strong_agreements"] += 1
            if low_case:
                summary["low_confidence_cases"] += 1
            if unclear_case:
                summary["unclear_cases"] += 1
            if multi_case:
                summary["multi_cases"] += 1
            if freeze_case:
                summary["freeze_card_cases"] += 1
            if dispute_case:
                summary["create_dispute_cases"] += 1
            if secondary_case:
                summary["secondary_review_cases"] += 1
            if safe_agreement:
                summary["safe_agreements"] += 1
            else:
                summary["pass_c_required"] += 1

        comparisons.append(
            {
                "complaint_id": source["complaint_id"],
                "narrative_sha256": digest,
                "pass_a_status": first["status"],
                "pass_b_status": (
                    second["status"] if second is not None else "PENDING"
                ),
                "exact_semantic_agreement": exact_semantic_agreement,
                "strong_agreement": strong_agreement,
                "safe_agreement": safe_agreement,
                "pass_c_required": second_succeeded and not safe_agreement,
                "adjudication_reasons": reasons,
            }
        )

    keys = (
        "pass_b_total",
        "pass_b_complete",
        "pass_b_pending",
        "exact_strong_agreements",
        "safe_agreements",
        "ab_disagreements",
        "low_confidence_cases",
        "unclear_cases",
        "multi_cases",
        "freeze_card_cases",
        "create_dispute_cases",
        "secondary_review_cases",
        "pass_c_required",
    )
    return comparisons, {key: summary[key] for key in keys}


def _contains_model_related_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            reviewer.is_model_related_field(str(key))
            or _contains_model_related_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_model_related_field(child) for child in value)
    return False


def adjudication_batch_source_row(
    source: Mapping[str, Any],
    first_pass: Mapping[str, Any],
    second_pass: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    batch_id: str,
    position: int,
    annotator_model: str,
) -> dict[str, Any]:
    """Build one Pass-C row with source text and concise A/B context."""
    row = {
        "schema_version": ADJUDICATION_BATCH_SCHEMA_VERSION,
        "annotation_source": ADJUDICATION_ANNOTATION_SOURCE,
        "annotator_model": annotator_model,
        "prompt_version": ADJUDICATION_PROMPT_VERSION,
        "batch_id": batch_id,
        "batch_position": position,
        "complaint_id": source.get("complaint_id"),
        "narrative_sha256": source.get("narrative_sha256"),
        "narrative": source.get("narrative"),
        "original_mapping_status": source.get("mapping_status"),
        "taxonomy_candidate_intents_hints_only": source.get(
            "candidate_sentinelvoice_intents", []
        ),
        "cfpb_product": source.get("product"),
        "cfpb_sub_product": source.get("sub_product"),
        "cfpb_issue": source.get("issue"),
        "cfpb_sub_issue": source.get("sub_issue"),
        "pass_a": pass_a_semantic_snapshot(first_pass),
        "pass_b": pass_b_semantic_snapshot(second_pass),
        "adjudication_reasons": list(comparison["adjudication_reasons"]),
    }
    if _contains_model_related_field(row):
        raise AssertionError("Pass-C batch contains a classifier-output field")
    return row


def validate_adjudication_batch_records(
    batch_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
) -> None:
    if not batch_records:
        raise ValueError("Pass-C batch must not be empty")
    comparisons, _ = compare_annotation_passes(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    comparison_by_hash = {
        row["narrative_sha256"]: row
        for row in comparisons
        if row["pass_c_required"]
    }
    source_by_hash = {
        row["narrative_sha256"]: row for row in source_records
    }
    first_by_hash = {
        row["narrative_sha256"]: row for row in first_pass_records
    }
    second_by_hash = {
        row["narrative_sha256"]: row for row in second_pass_records
    }
    seen: set[str] = set()
    batch_ids: set[str] = set()
    positions: list[int] = []
    for row in batch_records:
        if set(row) != set(ADJUDICATION_BATCH_FIELDS):
            raise ValueError("Pass-C batch fields differ from the required schema")
        if row.get("schema_version") != ADJUDICATION_BATCH_SCHEMA_VERSION:
            raise ValueError("unexpected Pass-C batch schema")
        if row.get("annotation_source") != ADJUDICATION_ANNOTATION_SOURCE:
            raise ValueError("invalid Pass-C annotation source")
        if row.get("prompt_version") != ADJUDICATION_PROMPT_VERSION:
            raise ValueError("invalid Pass-C prompt version")
        if _contains_model_related_field(row):
            raise ValueError("Pass-C batch contains a classifier-output field")
        digest = row.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in comparison_by_hash:
            raise ValueError("Pass-C batch contains a hash not requiring Pass C")
        if digest in seen:
            raise ValueError("Pass-C batch contains duplicate hashes")
        seen.add(digest)
        source = source_by_hash[digest]
        expected = adjudication_batch_source_row(
            source,
            first_by_hash[digest],
            second_by_hash[digest],
            comparison_by_hash[digest],
            batch_id=row["batch_id"],
            position=row["batch_position"],
            annotator_model=row["annotator_model"],
        )
        if dict(row) != expected:
            raise ValueError("Pass-C batch row differs from deterministic context")
        positions.append(row["batch_position"])
        batch_ids.add(row["batch_id"])
    if len(batch_ids) != 1 or not next(iter(batch_ids)).strip():
        raise ValueError("Pass-C batch must contain one non-empty batch_id")
    if positions != list(range(1, len(batch_records) + 1)):
        raise ValueError("Pass-C batch positions must be contiguous and ordered")


def prepare_adjudication_batch(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    adjudication_records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    batch_id: str | None = None,
    annotator_model: str = DEFAULT_ANNOTATOR_MODEL,
) -> tuple[str, list[dict[str, Any]]]:
    """Prepare the next deterministic Pass-C adjudication batch."""
    if not 1 <= limit <= MAX_BATCH_SIZE:
        raise ValueError(f"batch limit must be between 1 and {MAX_BATCH_SIZE}")
    if not annotator_model.strip():
        raise ValueError("annotator_model must be non-empty")
    validate_adjudication_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )
    comparisons, _ = compare_annotation_passes(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    comparison_by_hash = {
        row["narrative_sha256"]: row
        for row in comparisons
        if row["pass_c_required"]
    }
    adjudication_by_hash = {
        row["narrative_sha256"]: row for row in adjudication_records
    }
    selected = [
        source
        for source in source_records
        if source["narrative_sha256"] in comparison_by_hash
        and adjudication_by_hash.get(
            source["narrative_sha256"], {}
        ).get("status")
        != "SUCCEEDED"
    ][:limit]
    if not selected:
        return batch_id or "codex-adjudication-complete", []
    if batch_id is None:
        first_index = next(
            index
            for index, source in enumerate(source_records)
            if source["narrative_sha256"] == selected[0]["narrative_sha256"]
        )
        batch_id = (
            f"codex-adjudication-{first_index + 1:04d}-"
            f"{selected[0]['narrative_sha256'][:8]}"
        )
    if not batch_id.strip():
        raise ValueError("batch_id must be non-empty")
    first_by_hash = {
        row["narrative_sha256"]: row for row in first_pass_records
    }
    second_by_hash = {
        row["narrative_sha256"]: row for row in second_pass_records
    }
    rows = [
        adjudication_batch_source_row(
            source,
            first_by_hash[source["narrative_sha256"]],
            second_by_hash[source["narrative_sha256"]],
            comparison_by_hash[source["narrative_sha256"]],
            batch_id=batch_id,
            position=position,
            annotator_model=annotator_model,
        )
        for position, source in enumerate(selected, start=1)
    ]
    validate_adjudication_batch_records(
        rows,
        source_records,
        first_pass_records,
        second_pass_records,
    )
    return batch_id, rows


def validate_adjudication_result(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one Pass-C decision against its exact prepared row."""
    if set(result) != set(ADJUDICATION_RESULT_FIELDS):
        raise ValueError("Pass-C result fields differ from the required schema")
    comparisons = (
        ("annotation_source", ADJUDICATION_ANNOTATION_SOURCE),
        ("annotator_model", batch_row["annotator_model"]),
        ("prompt_version", batch_row["prompt_version"]),
        ("batch_id", batch_row["batch_id"]),
        ("complaint_id", batch_row["complaint_id"]),
        ("narrative_sha256", batch_row["narrative_sha256"]),
    )
    for field, expected in comparisons:
        if result.get(field) != expected:
            raise ValueError(f"Pass-C result {field} differs from its batch")
    if not isinstance(result["annotator_model"], str) or not result[
        "annotator_model"
    ].strip():
        raise ValueError("Pass-C annotator_model must be non-empty")
    resolution_status = result.get("resolution_status")
    if resolution_status not in ADJUDICATION_RESOLUTION_STATUSES:
        raise ValueError("invalid Pass-C resolution_status")

    if resolution_status == "RESOLVED":
        validation_row = {
            "review_category": result["final_review_category"],
            "supported_intents": result["final_supported_intents"],
            "annotation_confidence": result["final_annotation_confidence"],
            "annotation_note": result["final_annotation_note"],
            "secondary_review_required": False,
            "adjudication_status": "ADJUDICATED",
            "reviewer_id": ADJUDICATION_ANNOTATION_SOURCE,
        }
        contract.validate_annotation_fields(validation_row)
        if result["unresolved_reason"] != "":
            raise ValueError("RESOLVED Pass-C result cannot have unresolved_reason")
    else:
        if result.get("final_supported_intents") != []:
            raise ValueError(
                "UNRESOLVED Pass-C final_supported_intents must be empty"
            )
        if result.get("final_annotation_note") != "":
            raise ValueError(
                "UNRESOLVED Pass-C final_annotation_note must be empty"
            )
        if any(
            (
                result["final_review_category"] is not None,
                result["final_annotation_confidence"] is not None,
            )
        ):
            raise ValueError("UNRESOLVED Pass-C result cannot contain a final label")
        reason = result.get("unresolved_reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError(
                "UNRESOLVED Pass-C result requires a concise unresolved_reason"
            )
    return dict(result)


def successful_adjudication_record(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_adjudication_result(result, batch_row)
    return {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "status": "SUCCEEDED",
        **validated,
        "validation_error_type": None,
    }


def unresolved_adjudication_record(
    batch_row: Mapping[str, Any],
    *,
    status: str,
    validation_error: BaseException | None = None,
) -> dict[str, Any]:
    if status not in {"INVALID", "MISSING"}:
        raise ValueError("unresolved Pass-C status must be INVALID or MISSING")
    return {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "status": status,
        "annotation_source": ADJUDICATION_ANNOTATION_SOURCE,
        "annotator_model": batch_row["annotator_model"],
        "prompt_version": batch_row["prompt_version"],
        "batch_id": batch_row["batch_id"],
        "complaint_id": batch_row["complaint_id"],
        "narrative_sha256": batch_row["narrative_sha256"],
        "resolution_status": None,
        "final_review_category": None,
        "final_supported_intents": [],
        "final_annotation_confidence": None,
        "final_annotation_note": "",
        "unresolved_reason": "",
        "validation_error_type": (
            type(validation_error).__name__ if validation_error is not None else None
        ),
    }


def validate_adjudication_record(record: Mapping[str, Any]) -> None:
    """Validate one persisted Pass-C success or retryable row."""
    if set(record) != set(ADJUDICATION_STORE_FIELDS):
        raise ValueError("Pass-C store fields differ from the required schema")
    if record.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
        raise ValueError("unexpected Pass-C store schema")
    if record.get("annotation_source") != ADJUDICATION_ANNOTATION_SOURCE:
        raise ValueError("invalid Pass-C store annotation source")
    if record.get("prompt_version") != ADJUDICATION_PROMPT_VERSION:
        raise ValueError("invalid Pass-C store prompt version")
    status = record.get("status")
    if status not in ADJUDICATION_STORE_STATUSES:
        raise ValueError("invalid Pass-C store status")
    for field in ("annotator_model", "batch_id", "complaint_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pass-C {field} must be a non-empty string")
    digest = record.get("narrative_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Pass-C row requires a narrative SHA-256")
    if status != "SUCCEEDED":
        if record.get("final_supported_intents") != []:
            raise ValueError("retryable Pass-C intents must be empty")
        if record.get("final_annotation_note") != "":
            raise ValueError("retryable Pass-C note must be empty")
        if record.get("unresolved_reason") != "":
            raise ValueError("retryable Pass-C reason must be empty")
        if any(
            (
                record["resolution_status"] is not None,
                record["final_review_category"] is not None,
                record["final_annotation_confidence"] is not None,
            )
        ):
            raise ValueError("retryable Pass-C row cannot contain a decision")
        error_type = record["validation_error_type"]
        if status == "INVALID" and not isinstance(error_type, str):
            raise TypeError("INVALID Pass-C rows require validation_error_type")
        if status == "MISSING" and error_type is not None:
            raise ValueError("MISSING Pass-C rows cannot have validation_error_type")
        return
    result = {field: record[field] for field in ADJUDICATION_RESULT_FIELDS}
    synthetic_batch = {
        field: record[field]
        for field in (
            "annotator_model",
            "prompt_version",
            "batch_id",
            "complaint_id",
            "narrative_sha256",
        )
    }
    validate_adjudication_result(result, synthetic_batch)
    if record["validation_error_type"] is not None:
        raise ValueError("successful Pass-C rows cannot have validation errors")


def load_adjudication_records(path: Path) -> list[dict[str, Any]]:
    records, _ = contract.load_jsonl(path)
    seen: set[str] = set()
    for record in records:
        validate_adjudication_record(record)
        digest = record["narrative_sha256"]
        if digest in seen:
            raise ValueError("Pass-C store contains duplicate hashes")
        seen.add(digest)
    return records


def validate_adjudication_source_linkage(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    adjudication_records: Sequence[Mapping[str, Any]],
) -> None:
    comparisons, _ = compare_annotation_passes(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    required = {
        row["narrative_sha256"]
        for row in comparisons
        if row["pass_c_required"]
    }
    source_by_hash = {
        row["narrative_sha256"]: row for row in source_records
    }
    for adjudication in adjudication_records:
        digest = adjudication["narrative_sha256"]
        if digest not in required:
            raise ValueError("Pass-C store contains a hash not requiring Pass C")
        if adjudication.get("complaint_id") != source_by_hash[digest].get(
            "complaint_id"
        ):
            raise ValueError("Pass-C complaint ID differs from source")


def import_adjudication_results(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    existing_records: Sequence[Mapping[str, Any]],
    batch_records: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    *,
    replace_successful: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate and atomically merge externally written Pass-C decisions."""
    validate_adjudication_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
        existing_records,
    )
    validate_adjudication_batch_records(
        batch_records,
        source_records,
        first_pass_records,
        second_pass_records,
    )
    batch_by_hash = {row["narrative_sha256"]: row for row in batch_records}
    existing_by_hash = {
        row["narrative_sha256"]: row for row in existing_records
    }
    results_by_hash: dict[str, Mapping[str, Any]] = {}
    for result in result_records:
        digest = result.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in batch_by_hash:
            raise ValueError("Pass-C result contains a hash outside its batch")
        if digest in results_by_hash:
            raise ValueError("Pass-C results contain a duplicate hash")
        results_by_hash[digest] = result

    replacements: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for digest, batch_row in batch_by_hash.items():
        existing = existing_by_hash.get(digest)
        result = results_by_hash.get(digest)
        if existing is not None and existing["status"] == "SUCCEEDED":
            if result is None:
                continue
            if not replace_successful:
                raise ValueError("refusing to overwrite a successful Pass-C result")
        if result is None:
            replacement = unresolved_adjudication_record(
                batch_row,
                status="MISSING",
            )
        else:
            try:
                replacement = successful_adjudication_record(result, batch_row)
            except (TypeError, ValueError) as error:
                if existing is not None and existing["status"] == "SUCCEEDED":
                    raise ValueError(
                        "replacement Pass-C result is invalid; successful result "
                        "was preserved"
                    ) from error
                replacement = unresolved_adjudication_record(
                    batch_row,
                    status="INVALID",
                    validation_error=error,
                )
        validate_adjudication_record(replacement)
        replacements.append(replacement)
        counts[replacement["status"].lower()] += 1

    by_hash = {
        record["narrative_sha256"]: dict(record) for record in existing_records
    }
    by_hash.update(
        {record["narrative_sha256"]: record for record in replacements}
    )
    merged = [
        by_hash[source["narrative_sha256"]]
        for source in source_records
        if source["narrative_sha256"] in by_hash
    ]
    validate_adjudication_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
        merged,
    )
    return merged, {
        "succeeded": counts["succeeded"],
        "invalid": counts["invalid"],
        "missing": counts["missing"],
    }


def build_provisional_final_labels(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    adjudication_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Construct a text-free preview without freezing or exporting labels."""
    validate_adjudication_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )
    required_b = set(second_pass_required_hashes(source_records, first_pass_records))
    comparisons, _ = compare_annotation_passes(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    comparison_by_hash = {
        row["narrative_sha256"]: row for row in comparisons
    }
    first_by_hash = {
        row["narrative_sha256"]: row for row in first_pass_records
    }
    adjudication_by_hash = {
        row["narrative_sha256"]: row for row in adjudication_records
    }
    preview: list[dict[str, Any]] = []
    for source in source_records:
        digest = source["narrative_sha256"]
        first = first_by_hash.get(digest)
        selected: Mapping[str, Any] | None = None
        label_source: str | None = None
        if digest not in required_b:
            if first is not None and first["status"] == "SUCCEEDED":
                selected = first
                label_source = ANNOTATION_SOURCE
        else:
            comparison = comparison_by_hash[digest]
            if comparison["safe_agreement"]:
                selected = first
                label_source = "CODEX_DUAL_PASS_AGREEMENT"
            elif comparison["pass_c_required"]:
                adjudication = adjudication_by_hash.get(digest)
                if (
                    adjudication is not None
                    and adjudication["status"] == "SUCCEEDED"
                    and adjudication["resolution_status"] == "RESOLVED"
                ):
                    selected = {
                        "review_category": adjudication[
                            "final_review_category"
                        ],
                        "supported_intents": adjudication[
                            "final_supported_intents"
                        ],
                        "annotation_confidence": adjudication[
                            "final_annotation_confidence"
                        ],
                        "annotation_note": adjudication["final_annotation_note"],
                    }
                    label_source = ADJUDICATION_ANNOTATION_SOURCE
        if selected is None or label_source is None:
            continue
        preview.append(
            {
                "complaint_id": source["complaint_id"],
                "narrative_sha256": digest,
                "original_mapping_status": source["mapping_status"],
                "review_category": selected["review_category"],
                "supported_intents": list(selected["supported_intents"]),
                "annotation_confidence": selected["annotation_confidence"],
                "annotation_note": selected["annotation_note"],
                "provisional_label_source": label_source,
            }
        )
    return preview


def workflow_status(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    adjudication_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return deterministic end-to-end annotation workflow progress."""
    first_progress = first_pass_progress(source_records, first_pass_records)
    comparisons, comparison_summary = compare_annotation_passes(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    pass_c_required = {
        row["narrative_sha256"]
        for row in comparisons
        if row["pass_c_required"]
    }
    adjudication_by_hash = {
        row["narrative_sha256"]: row for row in adjudication_records
    }
    pass_c_resolved = sum(
        adjudication_by_hash.get(digest, {}).get("status") == "SUCCEEDED"
        and adjudication_by_hash[digest]["resolution_status"] == "RESOLVED"
        for digest in pass_c_required
    )
    pass_c_unresolved = sum(
        adjudication_by_hash.get(digest, {}).get("status") == "SUCCEEDED"
        and adjudication_by_hash[digest]["resolution_status"] == "UNRESOLVED"
        for digest in pass_c_required
    )
    preview = build_provisional_final_labels(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )
    preview_sources = Counter(
        row["provisional_label_source"] for row in preview
    )
    return {
        "total_holdout": len(source_records),
        "pass_a_complete": (
            first_progress["succeeded"] == len(source_records)
            and first_progress["invalid"] == 0
            and first_progress["missing"] == 0
            and first_progress["pending"] == 0
        ),
        "pass_a_succeeded": first_progress["succeeded"],
        "pass_b_required": comparison_summary["pass_b_total"],
        "pass_b_complete": comparison_summary["pass_b_complete"],
        "pass_b_pending": comparison_summary["pass_b_pending"],
        "strong_ab_agreements": comparison_summary[
            "exact_strong_agreements"
        ],
        "safe_ab_agreements": comparison_summary["safe_agreements"],
        "pass_c_required": comparison_summary["pass_c_required"],
        "pass_c_resolved": pass_c_resolved,
        "pass_c_unresolved": pass_c_unresolved,
        "pass_c_pending": (
            comparison_summary["pass_c_required"]
            - pass_c_resolved
            - pass_c_unresolved
        ),
        "final_labels_currently_available": len(preview),
        "final_label_sources": dict(sorted(preview_sources.items())),
        "human_review_remaining": pass_c_unresolved,
    }


def print_status(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
) -> None:
    progress = first_pass_progress(source_records, first_pass_records)
    print(
        " | ".join(
            (
                f"Total: {progress['total']}",
                f"Succeeded: {progress['succeeded']}",
                f"Invalid: {progress['invalid']}",
                f"Missing: {progress['missing']}",
                f"Pending: {progress['pending']}",
                f"Human review required: {progress['human_review_required']}",
            )
        )
    )


def load_optional_first_pass(path: Path) -> list[dict[str, Any]]:
    return load_first_pass_records(path) if path.exists() else []


def load_optional_second_pass(path: Path) -> list[dict[str, Any]]:
    return load_second_pass_records(path) if path.exists() else []


def load_optional_adjudication(path: Path) -> list[dict[str, Any]]:
    return load_adjudication_records(path) if path.exists() else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--first-pass",
        type=Path,
        default=DEFAULT_FIRST_PASS_PATH,
    )
    parser.add_argument(
        "--second-pass",
        type=Path,
        default=DEFAULT_SECOND_PASS_PATH,
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=DEFAULT_ADJUDICATION_PATH,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write the next local batch")
    prepare.add_argument("--limit", type=int, default=25)
    prepare.add_argument("--batch-id")
    prepare.add_argument("--annotator-model", default=DEFAULT_ANNOTATOR_MODEL)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--overwrite-batch", action="store_true")

    import_results = subparsers.add_parser(
        "import",
        help="validate and atomically import Codex result JSONL",
    )
    import_results.add_argument("--batch", type=Path, required=True)
    import_results.add_argument("--annotations", type=Path, required=True)
    import_results.add_argument("--replace-successful", action="store_true")

    subparsers.add_parser("status", help="show first-pass progress")

    second_pass = subparsers.add_parser(
        "second-pass",
        help="prepare, import, or inspect blind Pass-B annotations",
    )
    second_commands = second_pass.add_subparsers(
        dest="second_pass_command",
        required=True,
    )
    second_prepare = second_commands.add_parser(
        "prepare",
        help="write the next blind Pass-B batch",
    )
    second_prepare.add_argument("--limit", type=int, default=25)
    second_prepare.add_argument("--batch-id")
    second_prepare.add_argument(
        "--annotator-model",
        default=DEFAULT_ANNOTATOR_MODEL,
    )
    second_prepare.add_argument("--output", type=Path)
    second_prepare.add_argument("--overwrite-batch", action="store_true")
    second_import = second_commands.add_parser(
        "import",
        help="validate and atomically import Pass-B result JSONL",
    )
    second_import.add_argument("--batch", type=Path, required=True)
    second_import.add_argument("--annotations", type=Path, required=True)
    second_import.add_argument("--replace-successful", action="store_true")
    second_commands.add_parser("status", help="show Pass-B progress")

    compare = subparsers.add_parser(
        "compare",
        help="compare Pass A and Pass B without narratives",
    )
    compare.add_argument("--output", type=Path)
    compare.add_argument("--overwrite", action="store_true")

    adjudication = subparsers.add_parser(
        "adjudication",
        help="prepare, import, or inspect Pass-C adjudication",
    )
    adjudication_commands = adjudication.add_subparsers(
        dest="adjudication_command",
        required=True,
    )
    adjudication_prepare = adjudication_commands.add_parser(
        "prepare",
        help="write the next Pass-C adjudication batch",
    )
    adjudication_prepare.add_argument("--limit", type=int, default=25)
    adjudication_prepare.add_argument("--batch-id")
    adjudication_prepare.add_argument(
        "--annotator-model",
        default=DEFAULT_ANNOTATOR_MODEL,
    )
    adjudication_prepare.add_argument("--output", type=Path)
    adjudication_prepare.add_argument(
        "--overwrite-batch",
        action="store_true",
    )
    adjudication_import = adjudication_commands.add_parser(
        "import",
        help="validate and atomically import Pass-C result JSONL",
    )
    adjudication_import.add_argument("--batch", type=Path, required=True)
    adjudication_import.add_argument(
        "--annotations",
        type=Path,
        required=True,
    )
    adjudication_import.add_argument(
        "--replace-successful",
        action="store_true",
    )
    adjudication_commands.add_parser("status", help="show Pass-C progress")

    workflow = subparsers.add_parser(
        "workflow-status",
        help="show end-to-end Pass A/B/C progress",
    )
    workflow.add_argument("--preview-output", type=Path)
    workflow.add_argument("--overwrite-preview", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_records = reviewer.load_records(
        args.input,
        expected_mapping_counts=reviewer.EXPECTED_MAPPING_COUNTS,
    )
    first_pass_records = load_optional_first_pass(args.first_pass)
    validate_first_pass_source_linkage(source_records, first_pass_records)

    if args.command == "status":
        print_status(source_records, first_pass_records)
        return 0

    if args.command in {"prepare", "import"}:
        existing_human_reviews = sum(
            record["adjudication_status"] != "UNREVIEWED"
            for record in source_records
        )
        if existing_human_reviews:
            parser.error(
                f"canonical workfile contains {existing_human_reviews} existing "
                "human reviews; inspect and explicitly back up/reset them with "
                "review_cfpb_semantic_annotations.py before preparing/importing "
                "Codex batches"
            )

    if args.command == "prepare":
        batch_id, batch_records = prepare_batch(
            source_records,
            first_pass_records,
            limit=args.limit,
            batch_id=args.batch_id,
            annotator_model=args.annotator_model,
        )
        if not batch_records:
            print("All CFPB rows have successful Codex first-pass annotations.")
            return 0
        output_path = args.output or DEFAULT_BATCH_DIRECTORY / f"{batch_id}.jsonl"
        if output_path.exists() and not args.overwrite_batch:
            parser.error("batch output exists; use --overwrite-batch explicitly")
        reviewer.atomic_write_records(output_path, batch_records)
        print(
            f"Prepared Codex batch: id={batch_id}, rows={len(batch_records)}, "
            f"path={output_path}"
        )
        return 0

    if args.command == "import":
        batch_records, _ = contract.load_jsonl(args.batch)
        result_records, _ = contract.load_jsonl(args.annotations)
        merged, summary = import_codex_results(
            source_records,
            first_pass_records,
            batch_records,
            result_records,
            replace_successful=args.replace_successful,
        )
        reviewer.atomic_write_records(args.first_pass, merged)
        print(
            "Imported Codex batch atomically: "
            f"succeeded={summary['succeeded']}, "
            f"invalid={summary['invalid']}, missing={summary['missing']}"
        )
        return 0

    second_pass_records = load_optional_second_pass(args.second_pass)
    validate_second_pass_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
    )

    if args.command == "second-pass":
        if args.second_pass_command == "status":
            required = second_pass_required_hashes(
                source_records,
                first_pass_records,
            )
            statuses = Counter(record["status"] for record in second_pass_records)
            succeeded_required = sum(
                record["status"] == "SUCCEEDED"
                for record in second_pass_records
            )
            print(
                " | ".join(
                    (
                        f"Pass B required: {len(required)}",
                        f"Succeeded: {succeeded_required}",
                        f"Invalid: {statuses['INVALID']}",
                        f"Missing: {statuses['MISSING']}",
                        f"Pending: {len(required) - succeeded_required}",
                    )
                )
            )
            return 0
        if args.second_pass_command == "prepare":
            batch_id, batch_records = prepare_second_pass_batch(
                source_records,
                first_pass_records,
                second_pass_records,
                limit=args.limit,
                batch_id=args.batch_id,
                annotator_model=args.annotator_model,
            )
            if not batch_records:
                print("All required CFPB rows have successful Pass-B annotations.")
                return 0
            output_path = args.output or (
                DEFAULT_SECOND_PASS_BATCH_DIRECTORY / f"{batch_id}.jsonl"
            )
            if output_path.exists() and not args.overwrite_batch:
                parser.error(
                    "Pass-B batch output exists; use --overwrite-batch explicitly"
                )
            reviewer.atomic_write_records(output_path, batch_records)
            print(
                f"Prepared Pass-B batch: id={batch_id}, "
                f"rows={len(batch_records)}, path={output_path}"
            )
            return 0
        if args.second_pass_command == "import":
            batch_records, _ = contract.load_jsonl(args.batch)
            result_records, _ = contract.load_jsonl(args.annotations)
            merged, summary = import_second_pass_results(
                source_records,
                first_pass_records,
                second_pass_records,
                batch_records,
                result_records,
                replace_successful=args.replace_successful,
            )
            reviewer.atomic_write_records(args.second_pass, merged)
            print(
                "Imported Pass-B batch atomically: "
                f"succeeded={summary['succeeded']}, "
                f"invalid={summary['invalid']}, missing={summary['missing']}"
            )
            return 0

    if args.command == "compare":
        comparisons, summary = compare_annotation_passes(
            source_records,
            first_pass_records,
            second_pass_records,
        )
        if args.output is not None:
            if args.output.exists() and not args.overwrite:
                parser.error("comparison output exists; use --overwrite explicitly")
            reviewer.atomic_write_records(args.output, comparisons)
        print(
            " | ".join(
                (
                    f"Pass-B total: {summary['pass_b_total']}",
                    f"Exact strong agreements: {summary['exact_strong_agreements']}",
                    f"A/B disagreements: {summary['ab_disagreements']}",
                    f"LOW-confidence cases: {summary['low_confidence_cases']}",
                    f"UNCLEAR cases: {summary['unclear_cases']}",
                    f"MULTI cases: {summary['multi_cases']}",
                    f"freeze_card cases: {summary['freeze_card_cases']}",
                    f"create_dispute cases: {summary['create_dispute_cases']}",
                    f"secondary-review cases: {summary['secondary_review_cases']}",
                    f"Pass-C required: {summary['pass_c_required']}",
                )
            )
        )
        return 0

    adjudication_records = load_optional_adjudication(args.adjudication)
    validate_adjudication_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )

    if args.command == "adjudication":
        if args.adjudication_command == "prepare":
            batch_id, batch_records = prepare_adjudication_batch(
                source_records,
                first_pass_records,
                second_pass_records,
                adjudication_records,
                limit=args.limit,
                batch_id=args.batch_id,
                annotator_model=args.annotator_model,
            )
            if not batch_records:
                print("No pending CFPB rows currently require Pass C.")
                return 0
            output_path = args.output or (
                DEFAULT_ADJUDICATION_BATCH_DIRECTORY / f"{batch_id}.jsonl"
            )
            if output_path.exists() and not args.overwrite_batch:
                parser.error(
                    "Pass-C batch output exists; use --overwrite-batch explicitly"
                )
            reviewer.atomic_write_records(output_path, batch_records)
            print(
                f"Prepared Pass-C batch: id={batch_id}, "
                f"rows={len(batch_records)}, path={output_path}"
            )
            return 0
        if args.adjudication_command == "import":
            batch_records, _ = contract.load_jsonl(args.batch)
            result_records, _ = contract.load_jsonl(args.annotations)
            merged, summary = import_adjudication_results(
                source_records,
                first_pass_records,
                second_pass_records,
                adjudication_records,
                batch_records,
                result_records,
                replace_successful=args.replace_successful,
            )
            reviewer.atomic_write_records(args.adjudication, merged)
            print(
                "Imported Pass-C batch atomically: "
                f"succeeded={summary['succeeded']}, "
                f"invalid={summary['invalid']}, missing={summary['missing']}"
            )
            return 0
        if args.adjudication_command == "status":
            status = workflow_status(
                source_records,
                first_pass_records,
                second_pass_records,
                adjudication_records,
            )
            print(
                " | ".join(
                    (
                        f"Pass C required: {status['pass_c_required']}",
                        f"Resolved: {status['pass_c_resolved']}",
                        f"Unresolved: {status['pass_c_unresolved']}",
                        f"Pending: {status['pass_c_pending']}",
                        f"Human review remaining: {status['human_review_remaining']}",
                    )
                )
            )
            return 0

    if args.command == "workflow-status":
        status = workflow_status(
            source_records,
            first_pass_records,
            second_pass_records,
            adjudication_records,
        )
        if args.preview_output is not None:
            if args.preview_output.exists() and not args.overwrite_preview:
                parser.error(
                    "preview output exists; use --overwrite-preview explicitly"
                )
            preview = build_provisional_final_labels(
                source_records,
                first_pass_records,
                second_pass_records,
                adjudication_records,
            )
            reviewer.atomic_write_records(args.preview_output, preview)
        final_labels_available = status["final_labels_currently_available"]
        print(
            " | ".join(
                (
                    f"Total holdout: {status['total_holdout']}",
                    f"Pass A complete: {status['pass_a_complete']}",
                    f"Pass B required: {status['pass_b_required']}",
                    f"Pass B complete: {status['pass_b_complete']}",
                    f"Pass B pending: {status['pass_b_pending']}",
                    f"Strong A/B agreements: {status['strong_ab_agreements']}",
                    f"Pass C required: {status['pass_c_required']}",
                    f"Pass C resolved: {status['pass_c_resolved']}",
                    f"Pass C unresolved: {status['pass_c_unresolved']}",
                    f"Final labels currently available: {final_labels_available}",
                    f"Human review remaining: {status['human_review_remaining']}",
                )
            )
        )
        return 0

    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
