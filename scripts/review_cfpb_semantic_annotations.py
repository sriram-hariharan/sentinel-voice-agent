"""Review the CFPB holdout without exposing V2-C1 classifier outputs."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

try:
    from scripts import build_cfpb_annotation_workfile as contract
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    import build_cfpb_annotation_workfile as contract


DEFAULT_INPUT_PATH = contract.LOCAL_WORKFILE_PATH
DEFAULT_FIRST_PASS_PATH = (
    contract.EXTERNAL_ROOT
    / "processed/cfpb/local/cfpb_llm_first_pass.jsonl"
)
EXPECTED_MAPPING_COUNTS = {"AMBIGUOUS": 1_200, "NEAR_MATCH": 600}
RESET_CONFIRMATION = "RESET_REVIEWED_ANNOTATIONS"
SAVEABLE_ADJUDICATION_STATUSES = (
    "REVIEWED",
    "NEEDS_ADJUDICATION",
    "ADJUDICATED",
)

# The display is constructed from an explicit allowlist. These fragments provide
# a second guard against accidentally adding model-derived fields later.
MODEL_FIELD_FRAGMENTS = (
    "abstention",
    "classifier",
    "logistic",
    "margin",
    "model_output",
    "model_score",
    "predicted",
    "prediction",
    "probability",
    "svm",
)


def is_model_related_field(field_name: str) -> bool:
    """Return whether a field name resembles a prohibited model output."""
    normalized = field_name.casefold().replace("-", "_")
    return normalized in contract.FORBIDDEN_MODEL_FIELDS or any(
        fragment in normalized for fragment in MODEL_FIELD_FRAGMENTS
    )


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_mapping_counts: Mapping[str, int] | None = None,
) -> None:
    """Validate workfile structure without consuming unknown/model fields."""
    if not records:
        raise ValueError("semantic-review workfile must not be empty")

    seen_hashes: set[str] = set()
    mapping_counts: Counter[str] = Counter()
    for row in records:
        mapping_status = row.get("mapping_status")
        if mapping_status not in contract.SEMANTIC_REVIEW_STATUSES:
            raise ValueError(
                "semantic-review CLI accepts only NEAR_MATCH and AMBIGUOUS rows"
            )
        mapping_counts[mapping_status] += 1

        narrative = row.get("narrative")
        digest = row.get("narrative_sha256")
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("semantic-review row requires non-empty narrative text")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("semantic-review row requires a narrative SHA-256")
        if contract.sha256_bytes(narrative.encode("utf-8")) != digest:
            raise ValueError("semantic-review narrative does not match its SHA-256")
        if digest in seen_hashes:
            raise ValueError("semantic-review narrative hashes must be unique")
        seen_hashes.add(digest)
        contract.validate_annotation_fields(row)

    if expected_mapping_counts is not None and dict(mapping_counts) != dict(
        expected_mapping_counts
    ):
        raise ValueError(
            "semantic-review workfile does not contain the frozen "
            "600 NEAR_MATCH + 1,200 AMBIGUOUS rows"
        )


def load_records(
    path: Path,
    *,
    expected_mapping_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate a JSONL workfile while preserving row/key order."""
    records, _ = contract.load_jsonl(path)
    validate_records(
        records,
        expected_mapping_counts=expected_mapping_counts,
    )
    return records


def serialize_records(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize JSONL without sorting or dropping existing fields."""
    return (
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in records
        )
    ).encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably write bytes and atomically replace a same-filesystem target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Durably write a complete JSONL file and atomically replace the original."""
    atomic_write_bytes(path, serialize_records(records))


def reset_reviewed_annotations(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    confirmation: str,
    backup_path: Path | None = None,
) -> tuple[list[dict[str, Any]], int, Path | None]:
    """Explicitly back up and reset only non-UNREVIEWED annotation fields."""
    if confirmation != RESET_CONFIRMATION:
        raise ValueError(f"reset requires confirmation {RESET_CONFIRMATION!r}")
    reviewed_count = sum(
        record.get("adjudication_status") != "UNREVIEWED" for record in records
    )
    if reviewed_count == 0:
        return [dict(record) for record in records], 0, None

    if backup_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = path.with_name(f"{path.stem}.backup-{timestamp}{path.suffix}")
    if backup_path.exists():
        raise FileExistsError(f"reset backup already exists: {backup_path}")

    atomic_write_bytes(backup_path, path.read_bytes())
    reset_records = [
        {
            **record,
            **(
                contract.initialized_annotation_fields()
                if record.get("adjudication_status") != "UNREVIEWED"
                else {}
            ),
        }
        for record in records
    ]
    atomic_write_records(path, reset_records)
    return reset_records, reviewed_count, backup_path


def progress_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count initial-review and adjudication progress."""
    statuses = Counter(row.get("adjudication_status") for row in records)
    unreviewed = statuses["UNREVIEWED"]
    return {
        "total": len(records),
        "reviewed": len(records) - unreviewed,
        "unreviewed": unreviewed,
        "needs_adjudication": statuses["NEEDS_ADJUDICATION"],
        "adjudicated": statuses["ADJUDICATED"],
    }


def next_unreviewed_index(
    records: Sequence[Mapping[str, Any]],
    *,
    start_index: int = 0,
) -> int | None:
    """Find the first row still requiring initial review, with one wrap."""
    if not records:
        return None
    if start_index < 0 or start_index >= len(records):
        raise IndexError("start index is outside the workfile")
    return next(
        (
            index
            for index in chain(
                range(start_index, len(records)),
                range(start_index),
            )
            if records[index].get("adjudication_status") == "UNREVIEWED"
        ),
        None,
    )


def next_unreviewed_in_indices(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    start_position: int = 0,
) -> int | None:
    """Find an initially unreviewed row within a deterministic review subset."""
    if not indices:
        return None
    if start_position < 0 or start_position >= len(indices):
        raise IndexError("review-subset start position is outside the subset")
    positions = chain(
        range(start_position, len(indices)),
        range(start_position),
    )
    return next(
        (
            indices[position]
            for position in positions
            if records[indices[position]].get("adjudication_status")
            == "UNREVIEWED"
        ),
        None,
    )


def load_first_pass_context(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load separate Codex context without merging human annotations."""
    first_pass_records, _ = contract.load_jsonl(path)
    source_hashes = {record["narrative_sha256"] for record in records}
    by_hash: dict[str, dict[str, Any]] = {}
    for first_pass in first_pass_records:
        digest = first_pass.get("narrative_sha256")
        if not isinstance(digest, str) or digest not in source_hashes:
            raise ValueError("Codex first-pass context contains a non-holdout hash")
        if digest in by_hash:
            raise ValueError("Codex first-pass context contains duplicate hashes")
        if first_pass.get("status") not in {"SUCCEEDED", "INVALID", "MISSING"}:
            raise ValueError("Codex first-pass context has an invalid status")
        if first_pass.get("annotation_source") != "CODEX_FIRST_PASS":
            raise ValueError("Codex first-pass context has an invalid source")
        if not isinstance(first_pass.get("human_review_required"), bool):
            raise TypeError("Codex human_review_required must be boolean")
        by_hash[digest] = first_pass
    return by_hash


def flagged_review_indices(
    records: Sequence[Mapping[str, Any]],
    first_pass_by_hash: Mapping[str, Mapping[str, Any]],
) -> list[int]:
    """Return source-order rows marked for targeted human review."""
    return [
        index
        for index, record in enumerate(records)
        if first_pass_by_hash.get(record["narrative_sha256"], {}).get(
            "human_review_required"
        )
    ]


def next_index(current_index: int, total: int) -> int:
    """Move forward one row, wrapping at the end."""
    if total <= 0:
        raise ValueError("cannot navigate an empty workfile")
    return (current_index + 1) % total


def previous_index(current_index: int, total: int) -> int:
    """Move backward one row, wrapping at the beginning."""
    if total <= 0:
        raise ValueError("cannot navigate an empty workfile")
    return (current_index - 1) % total


def jump_index(one_based_index: int, total: int) -> int:
    """Convert a reviewer-facing one-based position to a list index."""
    if not 1 <= one_based_index <= total:
        raise IndexError(f"record index must be between 1 and {total}")
    return one_based_index - 1


def build_annotation(
    *,
    review_category: str,
    supported_intents: Sequence[str],
    annotation_confidence: str,
    annotation_note: str,
    reviewer_id: str,
    adjudication_status: str,
    secondary_review_required: bool,
) -> dict[str, Any]:
    """Build annotation-only fields and validate them against the C2P contract."""
    annotation = {
        "adjudication_status": adjudication_status,
        "annotation_confidence": annotation_confidence,
        "annotation_note": annotation_note,
        "review_category": review_category,
        "reviewer_id": reviewer_id,
        "secondary_review_required": secondary_review_required,
        "supported_intents": sorted(supported_intents),
    }
    contract.validate_annotation_fields(annotation)
    return annotation


def apply_annotation(
    records: Sequence[Mapping[str, Any]],
    index: int,
    annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replace only the selected row's frozen annotation fields."""
    if set(annotation) != set(contract.ANNOTATION_FIELDS):
        raise ValueError("annotation must contain exactly the frozen annotation fields")
    updated_row = dict(records[index])
    updated_row.update(annotation)
    contract.validate_annotation_fields(updated_row)
    updated_records = [dict(row) for row in records]
    updated_records[index] = updated_row
    return updated_records


def save_annotation(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    index: int,
    annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate, atomically persist, and return one explicit annotation update."""
    updated_records = apply_annotation(records, index, annotation)
    atomic_write_records(path, updated_records)
    return updated_records


def display_safe_metadata(
    record: Mapping[str, Any],
    *,
    index: int,
    progress: Mapping[str, int],
) -> dict[str, Any]:
    """Construct only reviewer-approved context, excluding model-like fields."""
    metadata = {
        "position": index + 1,
        "total": progress["total"],
        "reviewed": progress["reviewed"],
        "remaining": progress["unreviewed"],
        "complaint_id": record.get("complaint_id"),
        "narrative_sha256_abbreviated": str(
            record.get("narrative_sha256", "")
        )[:12],
        "original_mapping_status": record.get("mapping_status"),
        "candidate_intents_hints_only": list(
            record.get("candidate_sentinelvoice_intents", [])
        ),
        "product": record.get("product"),
        "issue": record.get("issue"),
        "sub_issue": record.get("sub_issue"),
        "narrative_length": record.get("narrative_length"),
        "current_annotation": {
            field_name: record.get(field_name)
            for field_name in contract.ANNOTATION_FIELDS
        },
    }
    if any(is_model_related_field(field_name) for field_name in metadata):
        raise AssertionError("display metadata includes a model-related field")
    return metadata


def print_progress(
    records: Sequence[Mapping[str, Any]],
    *,
    output: Callable[[str], None] = print,
) -> None:
    """Print aggregate progress without narrative text."""
    progress = progress_summary(records)
    output(
        " | ".join(
            (
                f"Total: {progress['total']}",
                f"Reviewed: {progress['reviewed']}",
                f"Unreviewed: {progress['unreviewed']}",
                f"Needs adjudication: {progress['needs_adjudication']}",
                f"Adjudicated: {progress['adjudicated']}",
            )
        )
    )


def display_record(
    record: Mapping[str, Any],
    *,
    index: int,
    progress: Mapping[str, int],
    codex_first_pass: Mapping[str, Any] | None = None,
    output: Callable[[str], None] = print,
) -> None:
    """Display one narrative plus an explicitly allowlisted header."""
    metadata = display_safe_metadata(record, index=index, progress=progress)
    hints = metadata["candidate_intents_hints_only"]
    output("\n" + "=" * 72)
    output(
        f"Record {metadata['position']}/{metadata['total']} | "
        f"Reviewed: {metadata['reviewed']} | Remaining: {metadata['remaining']}"
    )
    output(
        f"Complaint: {metadata['complaint_id']} | "
        f"SHA-256: {metadata['narrative_sha256_abbreviated']}... | "
        f"Mapping: {metadata['original_mapping_status']}"
    )
    output(f"Candidate intents (HINTS only): {', '.join(hints) or '(none)'}")
    output(
        f"CFPB: product={metadata['product']} | issue={metadata['issue']} | "
        f"sub-issue={metadata['sub_issue']}"
    )
    output(f"Narrative length: {metadata['narrative_length']}")
    output(
        "Current annotation: "
        + json.dumps(metadata["current_annotation"], ensure_ascii=False)
    )
    if codex_first_pass is not None:
        codex_display = {
            "status": codex_first_pass.get("status"),
            "annotation_source": codex_first_pass.get("annotation_source"),
            "annotator_model": codex_first_pass.get("annotator_model"),
            "review_category": codex_first_pass.get("review_category"),
            "supported_intents": codex_first_pass.get("supported_intents"),
            "annotation_confidence": codex_first_pass.get(
                "annotation_confidence"
            ),
            "annotation_note": codex_first_pass.get("annotation_note"),
            "human_review_reasons": codex_first_pass.get(
                "human_review_reasons"
            ),
        }
        output(
            "Codex first pass (adjudication context; NOT V2-C1): "
            + json.dumps(codex_display, ensure_ascii=False)
        )
    output("-" * 72)
    output(str(record["narrative"]))
    output("=" * 72)


def prompt_choice(
    label: str,
    choices: Sequence[str],
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> str:
    """Prompt for a numbered or exact frozen value."""
    for position, choice in enumerate(choices, start=1):
        output(f"  {position}. {choice}")
    while True:
        answer = input_fn(f"{label}: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        normalized = answer.upper()
        if normalized in choices:
            return normalized
        output("Invalid choice; enter a number or exact value shown above.")


def prompt_supported_intents(
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> list[str]:
    """Prompt for explicit human selections; never derive them from the row."""
    for position, intent in enumerate(contract.SUPPORTED_INTENTS, start=1):
        output(f"  {position}. {intent}")
    while True:
        answer = input_fn("Supported intents (comma-separated numbers/names): ")
        selected: list[str] = []
        invalid = False
        for item in (part.strip() for part in answer.split(",")):
            if not item:
                continue
            if item.isdigit() and 1 <= int(item) <= len(contract.SUPPORTED_INTENTS):
                selected.append(contract.SUPPORTED_INTENTS[int(item) - 1])
            elif item in contract.SUPPORTED_INTENTS:
                selected.append(item)
            else:
                invalid = True
                break
        if not invalid and len(selected) == len(set(selected)):
            return sorted(selected)
        output("Invalid or duplicate intent selection; choose only listed intents.")


def prompt_boolean(
    label: str,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> bool:
    """Prompt for an explicit yes/no decision."""
    while True:
        answer = input_fn(f"{label} [y/n]: ").strip().casefold()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        output("Enter y or n.")


def collect_annotation(
    reviewer_id: str,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Collect a human decision without inspecting narrative or metadata."""
    while True:
        category = prompt_choice(
            "Review category",
            contract.REVIEW_CATEGORIES,
            input_fn=input_fn,
            output=output,
        )
        if category in {
            "SINGLE_SUPPORTED_INTENT",
            "MULTI_SUPPORTED_INTENT",
        }:
            intents = prompt_supported_intents(input_fn=input_fn, output=output)
        else:
            intents = []
        confidence = prompt_choice(
            "Annotation confidence",
            contract.ANNOTATION_CONFIDENCES,
            input_fn=input_fn,
            output=output,
        )
        note = input_fn("Concise annotation note (required, max 500 chars): ")
        status = prompt_choice(
            "Adjudication status",
            SAVEABLE_ADJUDICATION_STATUSES,
            input_fn=input_fn,
            output=output,
        )
        secondary = prompt_boolean(
            "Secondary review required",
            input_fn=input_fn,
            output=output,
        )
        try:
            return build_annotation(
                review_category=category,
                supported_intents=intents,
                annotation_confidence=confidence,
                annotation_note=note,
                reviewer_id=reviewer_id,
                adjudication_status=status,
                secondary_review_required=secondary,
            )
        except (TypeError, ValueError) as error:
            output(f"Annotation not saved: {error}")
            output("Please enter the annotation again.")


def review_loop(
    path: Path,
    records: list[dict[str, Any]],
    *,
    reviewer_id: str,
    start_index: int,
    review_indices: Sequence[int] | None = None,
    first_pass_by_hash: Mapping[str, Mapping[str, Any]] | None = None,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> None:
    """Run the thin terminal interaction around the tested data helpers."""
    navigation_indices = list(
        range(len(records)) if review_indices is None else review_indices
    )
    if start_index not in navigation_indices:
        raise ValueError("start index is outside the active review subset")
    current_index = start_index
    while True:
        current_record = records[current_index]
        display_record(
            current_record,
            index=current_index,
            progress=progress_summary(records),
            codex_first_pass=(
                first_pass_by_hash.get(current_record["narrative_sha256"])
                if first_pass_by_hash is not None
                else None
            ),
            output=output,
        )
        command = input_fn(
            "[s] save  [k] skip  [b] back  [j] jump  [p] progress  [q] quit: "
        ).strip().casefold()
        if command == "q":
            output("Exited safely. Unsaved input was not written.")
            return
        if command == "p":
            print_progress(records, output=output)
            continue
        if command == "k":
            position = navigation_indices.index(current_index)
            current_index = navigation_indices[
                next_index(position, len(navigation_indices))
            ]
            continue
        if command == "b":
            position = navigation_indices.index(current_index)
            current_index = navigation_indices[
                previous_index(position, len(navigation_indices))
            ]
            continue
        if command == "j":
            try:
                requested = int(input_fn("Record number: ").strip())
                requested_index = jump_index(requested, len(records))
                if requested_index not in navigation_indices:
                    raise ValueError("record is outside the active review subset")
                current_index = requested_index
            except (ValueError, IndexError) as error:
                output(f"Invalid jump: {error}")
            continue
        if command == "s":
            annotation = collect_annotation(
                reviewer_id,
                input_fn=input_fn,
                output=output,
            )
            records = save_annotation(
                path,
                records,
                current_index,
                annotation,
            )
            output(f"Saved record {current_index + 1} atomically.")
            position = navigation_indices.index(current_index)
            following = next_unreviewed_in_indices(
                records,
                navigation_indices,
                start_position=next_index(position, len(navigation_indices)),
            )
            if following is None:
                output("All records in the active review subset are reviewed.")
                print_progress(records, output=output)
                return
            current_index = following
            continue
        output("Unknown command. Choose s, k, b, j, p, or q.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review/adjudicate the local CFPB semantic holdout. Candidate "
            "intents are hints only; V2-C1 outputs are never displayed."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="local semantic-review JSONL workfile",
    )
    parser.add_argument(
        "--reviewer-id",
        help="reviewer identity (required unless using --status or reset)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="one-based record index; defaults to the first UNREVIEWED row",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print progress counts without displaying narratives",
    )
    parser.add_argument(
        "--first-pass",
        type=Path,
        default=DEFAULT_FIRST_PASS_PATH,
        help="separate local Codex first-pass JSONL used as adjudication context",
    )
    parser.add_argument(
        "--flagged-only",
        action="store_true",
        help="review only rows deterministically flagged by the Codex workflow",
    )
    parser.add_argument(
        "--reset-reviewed",
        action="store_true",
        help="back up the workfile and reset only existing reviewed rows",
    )
    parser.add_argument(
        "--confirm-reset-reviewed",
        metavar="PHRASE",
        help=f"required confirmation phrase: {RESET_CONFIRMATION}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.status and args.reset_reviewed:
        parser.error("--status and --reset-reviewed cannot be combined")
    if (
        not args.status
        and not args.reset_reviewed
        and not (args.reviewer_id and args.reviewer_id.strip())
    ):
        parser.error("--reviewer-id is required for interactive review")

    try:
        records = load_records(
            args.input,
            expected_mapping_counts=EXPECTED_MAPPING_COUNTS,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: unable to load semantic-review workfile: {error}\n")

    if args.reset_reviewed:
        try:
            _, reset_count, backup_path = reset_reviewed_annotations(
                args.input,
                records,
                confirmation=args.confirm_reset_reviewed or "",
            )
        except (OSError, TypeError, ValueError) as error:
            parser.exit(2, f"error: reset not performed: {error}\n")
        if backup_path is None:
            print("No reviewed annotations found; no backup or reset was needed.")
        else:
            print(
                f"Backed up {args.input} to {backup_path} and reset "
                f"{reset_count} reviewed rows."
            )
        return 0

    first_pass_by_hash: dict[str, dict[str, Any]] = {}
    if args.first_pass.exists():
        try:
            first_pass_by_hash = load_first_pass_context(
                args.first_pass,
                records,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            parser.exit(2, f"error: unable to load Codex first pass: {error}\n")
    elif args.flagged_only:
        parser.error("--flagged-only requires an existing --first-pass file")
    if args.flagged_only and len(first_pass_by_hash) != len(records):
        parser.error(
            "--flagged-only requires a complete first pass for every holdout row"
        )

    review_indices = (
        flagged_review_indices(records, first_pass_by_hash)
        if args.flagged_only
        else list(range(len(records)))
    )

    if args.status:
        print_progress(records)
        if first_pass_by_hash:
            print(
                f"Codex first pass: {len(first_pass_by_hash)} rows | "
                "Flagged for human review: "
                f"{len(flagged_review_indices(records, first_pass_by_hash))}"
            )
        return 0

    if not review_indices:
        print("No records are flagged for human review.")
        return 0

    if args.start_index is not None:
        try:
            start_index = jump_index(args.start_index, len(records))
            if start_index not in review_indices:
                parser.error("--start-index is outside the active review subset")
        except IndexError as error:
            parser.error(str(error))
    else:
        next_index_value = next_unreviewed_in_indices(records, review_indices)
        if next_index_value is None:
            print(
                "All records in the active review subset have an initial review. "
                "Use --start-index to edit one."
            )
            print_progress(records)
            return 0
        start_index = next_index_value

    try:
        review_loop(
            args.input,
            records,
            reviewer_id=args.reviewer_id.strip(),
            start_index=start_index,
            review_indices=review_indices,
            first_pass_by_hash=first_pass_by_hash,
        )
    except (EOFError, KeyboardInterrupt):
        print("\nExited safely. Unsaved input was not written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
