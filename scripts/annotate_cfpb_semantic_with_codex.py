"""Prepare, validate, import, and track direct-Codex CFPB annotations."""

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
DEFAULT_BATCH_DIRECTORY = (
    contract.EXTERNAL_ROOT / "processed/cfpb/local/codex_batches"
)
FIRST_PASS_SCHEMA_VERSION = "cfpb-codex-first-pass.v1"
BATCH_SCHEMA_VERSION = "cfpb-codex-annotation-batch.v1"
PROMPT_VERSION = "cfpb-codex-semantic-prompt.v1"
ANNOTATION_SOURCE = "CODEX_FIRST_PASS"
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


def validate_codex_result(
    result: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one Codex result against its batch and the frozen C2P rules."""
    if set(result) != set(CODEX_RESULT_FIELDS):
        raise ValueError("Codex result fields differ from the required schema")
    comparisons = (
        ("annotation_source", ANNOTATION_SOURCE),
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

    validation_row = {
        field: result[field] for field in SEMANTIC_FIELDS
    }
    validation_row.update(
        {
            "adjudication_status": "REVIEWED",
            "reviewer_id": ANNOTATION_SOURCE,
        }
    )
    contract.validate_annotation_fields(validation_row)
    return dict(result)


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


def validate_batch_records(
    batch_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
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
        if batch_row.get("schema_version") != BATCH_SCHEMA_VERSION:
            raise ValueError("unexpected Codex batch schema")
        if batch_row.get("annotation_source") != ANNOTATION_SOURCE:
            raise ValueError("Codex batch has an invalid annotation source")
        if batch_row.get("prompt_version") != PROMPT_VERSION:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--first-pass",
        type=Path,
        default=DEFAULT_FIRST_PASS_PATH,
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

    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
