"""Build a deterministic CFPB narrative review/evaluation-candidate pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.plan_cfpb_narrative_candidates import (
        COMPLAINT_ID_FIELD,
        ISSUE_FIELD,
        MAPPING_STATUSES,
        NARRATIVE_FIELD,
        PRODUCT_FIELD,
        SUB_ISSUE_FIELD,
        SUB_PRODUCT_FIELD,
        FrozenAssignment,
        TaxonomyKey,
        length_bin,
        load_frozen_assignments,
        normalize_fieldnames,
        privacy_observation,
        production_archive_paths,
        stable_json_bytes,
        taxonomy_key_from_values,
    )
    from scripts.plan_cfpb_narrative_candidates import (
        REQUIRED_FIELDS as PLANNER_REQUIRED_FIELDS,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from plan_cfpb_narrative_candidates import (
        COMPLAINT_ID_FIELD,
        ISSUE_FIELD,
        MAPPING_STATUSES,
        NARRATIVE_FIELD,
        PRODUCT_FIELD,
        SUB_ISSUE_FIELD,
        SUB_PRODUCT_FIELD,
        FrozenAssignment,
        TaxonomyKey,
        length_bin,
        load_frozen_assignments,
        normalize_fieldnames,
        privacy_observation,
        production_archive_paths,
        stable_json_bytes,
        taxonomy_key_from_values,
    )
    from plan_cfpb_narrative_candidates import (
        REQUIRED_FIELDS as PLANNER_REQUIRED_FIELDS,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
RAW_DIR = EXTERNAL_ROOT / "raw/cfpb_narratives"
RAW_MANIFEST_PATH = RAW_DIR / "manifest.json"
MAPPING_PATH = EXTERNAL_ROOT / "mappings/cfpb_to_sentinel_intents.json"
CANDIDATE_PLAN_PATH = EXTERNAL_ROOT / "processed/cfpb/candidate_plan.json"
MANIFEST_OUTPUT_PATH = EXTERNAL_ROOT / "processed/cfpb/review_pool_manifest.json"
LOCAL_OUTPUT_PATH = EXTERNAL_ROOT / "processed/cfpb/local/cfpb_review_pool.jsonl"

DATE_FIELD = "Date received"
REQUIRED_FIELDS = PLANNER_REQUIRED_FIELDS | {DATE_FIELD}
SELECTABLE_STATUSES = ("NEAR_MATCH", "AMBIGUOUS", "UNSUPPORTED")
SEMANTIC_REVIEW_STATUSES = ("NEAR_MATCH", "AMBIGUOUS")
OWNERSHIP_PRECEDENCE = {
    "AMBIGUOUS": 0,
    "NEAR_MATCH": 1,
    "UNSUPPORTED": 2,
    "EXACT_MATCH": 3,
}
DEFAULT_LANE_SIZES = {
    "NEAR_MATCH": 600,
    "AMBIGUOUS": 1_200,
    "UNSUPPORTED": 2_000,
}
REVIEW_LENGTH_GROUPS = ("lte_500", "501_1000", "1001_2000", "gt_2000")
REVIEW_LENGTH_WEIGHTS = {
    "lte_500": 0.35,
    "501_1000": 0.30,
    "1001_2000": 0.25,
    "gt_2000": 0.10,
}
ISSUE_CAP_FRACTION = 0.20
SELECTION_NAMESPACE = "sentinelvoice-cfpb-review-pool-2026-09-27-v1"
BUILDER_VERSION = "cfpb-review-pool-builder.v1"
SELECTION_ALGORITHM_VERSION = "cfpb-review-pool-selection.2026-09-27.v1"
MANIFEST_SCHEMA_VERSION = "cfpb-review-pool-manifest.v1"
LOCAL_SCHEMA_VERSION = "cfpb-review-pool-local-jsonl.v1"
SQL_BATCH_SIZE = 10_000


@dataclass(frozen=True)
class TaxonomyMetadata:
    taxonomy_id: int
    key: TaxonomyKey
    assignment: FrozenAssignment


@dataclass(frozen=True, slots=True)
class Candidate:
    digest: bytes
    mapping_status: str
    complaint_id: str
    taxonomy_id: int
    archive_index: int
    date_received: str
    narrative_length: int
    review_length_group: str
    selection_key: bytes


@dataclass
class ScanStatistics:
    total_csv_rows: int = 0
    narrative_records_by_status: Counter[str] = field(default_factory=Counter)
    privacy_excluded_by_status: Counter[str] = field(default_factory=Counter)
    clean_records_by_status: Counter[str] = field(default_factory=Counter)


@dataclass
class SelectionResult:
    candidates: list[Candidate]
    metadata: dict[str, Any]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_components(namespace: str, components: Sequence[str]) -> bytes:
    payload = json.dumps(
        [namespace, *components],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def review_length_group(character_length: int) -> str:
    if character_length < 1:
        raise ValueError("narrative length must be positive")
    if character_length <= 500:
        return "lte_500"
    if character_length <= 1_000:
        return "501_1000"
    if character_length <= 2_000:
        return "1001_2000"
    return "gt_2000"


def allocate_length_targets(requested_size: int) -> dict[str, int]:
    if requested_size < 0:
        raise ValueError("requested sample size cannot be negative")
    raw = {
        group: requested_size * REVIEW_LENGTH_WEIGHTS[group]
        for group in REVIEW_LENGTH_GROUPS
    }
    targets = {group: math.floor(raw[group]) for group in REVIEW_LENGTH_GROUPS}
    remaining = requested_size - sum(targets.values())
    remainder_order = sorted(
        REVIEW_LENGTH_GROUPS,
        key=lambda group: (
            -(raw[group] - targets[group]),
            REVIEW_LENGTH_GROUPS.index(group),
        ),
    )
    for group in remainder_order[:remaining]:
        targets[group] += 1
    return targets


def taxonomy_sort_key(key: TaxonomyKey) -> tuple[str, str, str, str]:
    return (
        key.product,
        key.sub_product or "",
        key.issue,
        key.sub_issue or "",
    )


def taxonomy_catalog(
    assignments: Mapping[TaxonomyKey, FrozenAssignment],
) -> tuple[dict[TaxonomyKey, TaxonomyMetadata], dict[int, TaxonomyMetadata]]:
    by_key: dict[TaxonomyKey, TaxonomyMetadata] = {}
    by_id: dict[int, TaxonomyMetadata] = {}
    for taxonomy_id, key in enumerate(sorted(assignments, key=taxonomy_sort_key)):
        metadata = TaxonomyMetadata(
            taxonomy_id=taxonomy_id,
            key=key,
            assignment=assignments[key],
        )
        by_key[key] = metadata
        by_id[taxonomy_id] = metadata
    return by_key, by_id


def selection_key_for_record(
    *,
    narrative_sha256: str,
    mapping_status: str,
    complaint_id: str,
    source_archive: str,
    date_received: str,
    key: TaxonomyKey,
) -> bytes:
    return stable_hash_components(
        SELECTION_NAMESPACE,
        (
            narrative_sha256,
            mapping_status,
            complaint_id,
            source_archive,
            date_received,
            key.product,
            key.sub_product or "",
            key.issue,
            key.sub_issue or "",
        ),
    )


def semantic_stratum_key(
    metadata: TaxonomyMetadata,
    source_archive: str,
) -> bytes:
    return stable_hash_components(
        f"{SELECTION_NAMESPACE}:semantic-stratum",
        (
            "|".join(metadata.assignment.candidate_intents),
            metadata.key.issue,
            metadata.key.product,
            source_archive,
        ),
    )


def candidate_intent_set_key(metadata: TaxonomyMetadata) -> bytes:
    return stable_hash_components(
        f"{SELECTION_NAMESPACE}:candidate-intent-set",
        metadata.assignment.candidate_intents,
    )


def unsupported_stratum_key(
    metadata: TaxonomyMetadata,
    length_group: str,
) -> bytes:
    return stable_hash_components(
        f"{SELECTION_NAMESPACE}:unsupported-stratum",
        (
            metadata.key.product,
            metadata.key.issue,
            metadata.key.sub_issue or "",
            length_group,
        ),
    )


def product_key(product: str) -> bytes:
    return stable_hash_components(
        f"{SELECTION_NAMESPACE}:product",
        (product,),
    )


def create_index() -> tuple[Path, sqlite3.Connection]:
    with tempfile.NamedTemporaryFile(
        prefix="cfpb_review_pool_",
        suffix=".sqlite3",
        delete=False,
    ) as temporary:
        path = Path(temporary.name)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        """
        CREATE TABLE eligible_hash_lanes (
            digest BLOB NOT NULL,
            mapping_status TEXT NOT NULL,
            status_precedence INTEGER NOT NULL,
            occurrence_count INTEGER NOT NULL,
            complaint_id TEXT NOT NULL,
            taxonomy_id INTEGER NOT NULL,
            archive_index INTEGER NOT NULL,
            date_received TEXT NOT NULL,
            narrative_length INTEGER NOT NULL,
            review_length_group TEXT NOT NULL,
            selection_key BLOB NOT NULL,
            candidate_intent_set BLOB NOT NULL,
            semantic_stratum BLOB NOT NULL,
            unsupported_stratum BLOB NOT NULL,
            product_key BLOB NOT NULL,
            PRIMARY KEY (digest, mapping_status)
        ) WITHOUT ROWID
        """
    )
    return path, connection


def store_index_batch(
    connection: sqlite3.Connection,
    rows: list[tuple[Any, ...]],
) -> None:
    connection.executemany(
        """
        INSERT INTO eligible_hash_lanes (
            digest,
            mapping_status,
            status_precedence,
            occurrence_count,
            complaint_id,
            taxonomy_id,
            archive_index,
            date_received,
            narrative_length,
            review_length_group,
            selection_key,
            candidate_intent_set,
            semantic_stratum,
            unsupported_stratum,
            product_key
        ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(digest, mapping_status) DO UPDATE SET
            occurrence_count = eligible_hash_lanes.occurrence_count + 1,
            complaint_id = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.complaint_id ELSE eligible_hash_lanes.complaint_id END,
            taxonomy_id = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.taxonomy_id ELSE eligible_hash_lanes.taxonomy_id END,
            archive_index = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.archive_index ELSE eligible_hash_lanes.archive_index END,
            date_received = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.date_received ELSE eligible_hash_lanes.date_received END,
            narrative_length = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.narrative_length ELSE eligible_hash_lanes.narrative_length END,
            review_length_group = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.review_length_group
                ELSE eligible_hash_lanes.review_length_group END,
            candidate_intent_set = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.candidate_intent_set
                ELSE eligible_hash_lanes.candidate_intent_set END,
            semantic_stratum = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.semantic_stratum ELSE eligible_hash_lanes.semantic_stratum END,
            unsupported_stratum = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.unsupported_stratum
                ELSE eligible_hash_lanes.unsupported_stratum END,
            product_key = CASE
                WHEN excluded.selection_key < eligible_hash_lanes.selection_key
                THEN excluded.product_key ELSE eligible_hash_lanes.product_key END,
            selection_key = MIN(
                eligible_hash_lanes.selection_key,
                excluded.selection_key
            )
        """,
        rows,
    )


def archive_csv_reader(archive_path: Path) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".csv")
        ]
        if len(members) != 1:
            raise ValueError(f"{archive_path} must contain exactly one CSV")
        with archive.open(members[0]) as binary_member:
            text_member = io.TextIOWrapper(
                binary_member,
                encoding="utf-8-sig",
                newline="",
            )
            reader = csv.DictReader(text_member)
            fields = normalize_fieldnames(reader.fieldnames)
            if reader.fieldnames != fields:
                reader.fieldnames = fields
            missing = REQUIRED_FIELDS - set(fields)
            if missing:
                raise ValueError(
                    f"{archive_path.name} is missing fields: {sorted(missing)}"
                )
            yield from reader


def scan_eligible_candidates(
    archive_paths: Sequence[Path],
    catalog_by_key: Mapping[TaxonomyKey, TaxonomyMetadata],
    connection: sqlite3.Connection,
) -> ScanStatistics:
    statistics = ScanStatistics()
    batch: list[tuple[Any, ...]] = []
    for archive_index, archive_path in enumerate(archive_paths):
        print(f"Indexing {archive_path.name}", flush=True)
        for row_number, row in enumerate(archive_csv_reader(archive_path), start=1):
            statistics.total_csv_rows += 1
            narrative = row.get(NARRATIVE_FIELD) or ""
            if not narrative.strip():
                continue
            key = taxonomy_key_from_values(
                row.get(PRODUCT_FIELD),
                row.get(SUB_PRODUCT_FIELD),
                row.get(ISSUE_FIELD),
                row.get(SUB_ISSUE_FIELD),
            )
            try:
                taxonomy = catalog_by_key[key]
            except KeyError as exc:
                raise ValueError(
                    "narrative row is absent from the frozen V2-C2M mapping: "
                    f"{key}"
                ) from exc
            status = taxonomy.assignment.mapping_status
            statistics.narrative_records_by_status[status] += 1
            _, has_privacy_signal = privacy_observation(narrative)
            if has_privacy_signal:
                statistics.privacy_excluded_by_status[status] += 1
                continue
            statistics.clean_records_by_status[status] += 1

            complaint_id = (row.get(COMPLAINT_ID_FIELD) or "").strip()
            date_received = (row.get(DATE_FIELD) or "").strip()
            if not complaint_id or not date_received:
                raise ValueError(
                    f"missing complaint ID or date at row {row_number} in "
                    f"{archive_path.name}"
                )
            digest = hashlib.sha256(narrative.encode("utf-8")).digest()
            digest_hex = digest.hex()
            character_length = len(narrative)
            length_group = review_length_group(character_length)
            selection_key = selection_key_for_record(
                narrative_sha256=digest_hex,
                mapping_status=status,
                complaint_id=complaint_id,
                source_archive=archive_path.name,
                date_received=date_received,
                key=key,
            )
            batch.append(
                (
                    digest,
                    status,
                    OWNERSHIP_PRECEDENCE[status],
                    complaint_id,
                    taxonomy.taxonomy_id,
                    archive_index,
                    date_received,
                    character_length,
                    length_group,
                    selection_key,
                    candidate_intent_set_key(taxonomy),
                    semantic_stratum_key(taxonomy, archive_path.name),
                    unsupported_stratum_key(taxonomy, length_group),
                    product_key(key.product),
                )
            )
            if len(batch) == SQL_BATCH_SIZE:
                store_index_batch(connection, batch)
                batch.clear()
    if batch:
        store_index_batch(connection, batch)
    connection.commit()
    return statistics


def create_owned_candidates(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE owned_candidates AS
        SELECT lanes.*
        FROM eligible_hash_lanes AS lanes
        WHERE lanes.status_precedence = (
            SELECT MIN(other.status_precedence)
            FROM eligible_hash_lanes AS other
            WHERE other.digest = lanes.digest
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX owned_candidates_digest ON owned_candidates (digest)"
    )
    connection.execute(
        """
        CREATE INDEX owned_candidates_status_length
        ON owned_candidates (mapping_status, review_length_group)
        """
    )
    connection.commit()


def count_by_status(
    connection: sqlite3.Connection,
    *,
    table: str,
) -> dict[str, int]:
    if table not in {"eligible_hash_lanes", "owned_candidates"}:
        raise ValueError("unexpected count table")
    counts = {status: 0 for status in MAPPING_STATUSES}
    for status, count in connection.execute(
        f"SELECT mapping_status, COUNT(*) FROM {table} GROUP BY mapping_status"
    ):
        counts[status] = count
    return counts


def validate_against_candidate_plan(
    candidate_plan: Mapping[str, Any],
    statistics: ScanStatistics,
    unique_clean_by_status: Mapping[str, int],
) -> None:
    if candidate_plan.get("schema_version") != "cfpb-narrative-candidate-plan.v1":
        raise ValueError("unexpected CFPB candidate-plan schema version")
    expected_csv_rows = candidate_plan.get("overall", {}).get(
        "total_full_records_rows_scanned"
    )
    if statistics.total_csv_rows != expected_csv_rows:
        raise ValueError("full-record row count differs from candidate plan")
    expected_unique = candidate_plan.get("potential_evaluation_populations", {}).get(
        "exact_deduplicated_without_privacy_signal",
        {},
    ).get("by_mapping_status", {})
    for status in MAPPING_STATUSES:
        lane = candidate_plan.get("by_mapping_status", {}).get(status, {})
        expected_records = lane.get("record_count")
        expected_privacy = lane.get("privacy_screening", {}).get(
            "any_privacy_screening_signal"
        )
        if statistics.narrative_records_by_status[status] != expected_records:
            raise ValueError(f"{status} narrative count differs from candidate plan")
        if statistics.privacy_excluded_by_status[status] != expected_privacy:
            raise ValueError(f"{status} privacy count differs from candidate plan")
        if unique_clean_by_status[status] != expected_unique.get(status):
            raise ValueError(
                f"{status} clean deduplicated count differs from candidate plan"
            )


def duplicate_statistics(connection: sqlite3.Connection) -> dict[str, Any]:
    within_lane = {status: 0 for status in MAPPING_STATUSES}
    for status, repeated in connection.execute(
        """
        SELECT mapping_status, COALESCE(SUM(occurrence_count - 1), 0)
        FROM eligible_hash_lanes
        GROUP BY mapping_status
        """
    ):
        within_lane[status] = repeated

    clean_record_count, global_unique_count = connection.execute(
        """
        SELECT
            COALESCE(SUM(occurrence_count), 0),
            COUNT(DISTINCT digest)
        FROM eligible_hash_lanes
        """
    ).fetchone()
    cross_lane_hashes, cross_lane_records = connection.execute(
        """
        WITH cross_lane AS (
            SELECT digest, SUM(occurrence_count) AS record_count
            FROM eligible_hash_lanes
            GROUP BY digest
            HAVING COUNT(*) > 1
        )
        SELECT COUNT(*), COALESCE(SUM(record_count), 0)
        FROM cross_lane
        """
    ).fetchone()

    owner_hashes = {status: 0 for status in MAPPING_STATUSES}
    for status, count in connection.execute(
        """
        SELECT owned.mapping_status, COUNT(*)
        FROM owned_candidates AS owned
        WHERE EXISTS (
            SELECT 1
            FROM eligible_hash_lanes AS lanes
            WHERE lanes.digest = owned.digest
            GROUP BY lanes.digest
            HAVING COUNT(*) > 1
        )
        GROUP BY owned.mapping_status
        """
    ):
        owner_hashes[status] = count

    losing_records = {status: 0 for status in MAPPING_STATUSES}
    for status, count in connection.execute(
        """
        SELECT lanes.mapping_status, COALESCE(SUM(lanes.occurrence_count), 0)
        FROM eligible_hash_lanes AS lanes
        JOIN owned_candidates AS owned ON owned.digest = lanes.digest
        WHERE lanes.mapping_status != owned.mapping_status
        GROUP BY lanes.mapping_status
        """
    ):
        losing_records[status] = count

    return {
        "clean_duplicate_occurrences_excluded": (
            clean_record_count - global_unique_count
        ),
        "clean_records_before_global_deduplication": clean_record_count,
        "cross_lane_duplicate_hashes_resolved": cross_lane_hashes,
        "cross_lane_records_involved": cross_lane_records,
        "cross_lane_selected_owner_hashes_by_status": owner_hashes,
        "cross_lane_records_excluded_from_losing_status": losing_records,
        "global_unique_clean_narrative_hashes": global_unique_count,
        "within_lane_repeated_occurrences_beyond_first": within_lane,
    }


CANDIDATE_COLUMNS = """
    digest,
    mapping_status,
    complaint_id,
    taxonomy_id,
    archive_index,
    date_received,
    narrative_length,
    review_length_group,
    selection_key
"""


def candidate_from_row(row: Sequence[Any]) -> Candidate:
    return Candidate(
        digest=row[0],
        mapping_status=row[1],
        complaint_id=row[2],
        taxonomy_id=row[3],
        archive_index=row[4],
        date_received=row[5],
        narrative_length=row[6],
        review_length_group=row[7],
        selection_key=row[8],
    )


def semantic_candidate_rows(
    connection: sqlite3.Connection,
    status: str,
    length_group: str | None,
) -> Iterable[Candidate]:
    where_length = "" if length_group is None else "AND review_length_group = ?"
    parameters: tuple[str, ...] = (
        (status,) if length_group is None else (status, length_group)
    )
    rows = connection.execute(
        f"""
        WITH stratum_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY semantic_stratum, review_length_group
                    ORDER BY selection_key, digest
                ) AS stratum_rank
            FROM owned_candidates
            WHERE mapping_status = ? {where_length}
        ),
        intent_set_fair AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY candidate_intent_set
                    ORDER BY stratum_rank, selection_key, digest
                ) AS intent_set_round
            FROM stratum_ranked
        )
        SELECT {CANDIDATE_COLUMNS}
        FROM intent_set_fair
        ORDER BY intent_set_round, stratum_rank, selection_key, digest
        """,
        parameters,
    )
    for row in rows:
        yield candidate_from_row(row)


def add_candidate(
    candidate: Candidate,
    selected: list[Candidate],
    selected_digests: set[bytes],
    achieved_lengths: Counter[str],
    issue_counts: Counter[str],
    catalog_by_id: Mapping[int, TaxonomyMetadata],
) -> None:
    selected.append(candidate)
    selected_digests.add(candidate.digest)
    achieved_lengths[candidate.review_length_group] += 1
    issue_counts[catalog_by_id[candidate.taxonomy_id].key.issue] += 1


def fill_semantic_candidates(
    candidates: Iterable[Candidate],
    *,
    desired: int,
    enforce_issue_cap: bool,
    issue_cap: int,
    selected: list[Candidate],
    selected_digests: set[bytes],
    achieved_lengths: Counter[str],
    issue_counts: Counter[str],
    catalog_by_id: Mapping[int, TaxonomyMetadata],
) -> int:
    added = 0
    for candidate in candidates:
        if added == desired:
            break
        if candidate.digest in selected_digests:
            continue
        issue = catalog_by_id[candidate.taxonomy_id].key.issue
        if enforce_issue_cap and issue_counts[issue] >= issue_cap:
            continue
        add_candidate(
            candidate,
            selected,
            selected_digests,
            achieved_lengths,
            issue_counts,
            catalog_by_id,
        )
        added += 1
    return added


def select_semantic_lane(
    connection: sqlite3.Connection,
    status: str,
    requested_size: int,
    catalog_by_id: Mapping[int, TaxonomyMetadata],
) -> SelectionResult:
    targets = allocate_length_targets(requested_size)
    issue_cap = max(1, math.floor(requested_size * ISSUE_CAP_FRACTION))
    selected: list[Candidate] = []
    selected_digests: set[bytes] = set()
    achieved_lengths: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    issue_cap_relaxation_selected = 0

    for group in REVIEW_LENGTH_GROUPS:
        desired = targets[group]
        added = fill_semantic_candidates(
            semantic_candidate_rows(connection, status, group),
            desired=desired,
            enforce_issue_cap=True,
            issue_cap=issue_cap,
            selected=selected,
            selected_digests=selected_digests,
            achieved_lengths=achieved_lengths,
            issue_counts=issue_counts,
            catalog_by_id=catalog_by_id,
        )
        remaining = desired - added
        if remaining:
            relaxed = fill_semantic_candidates(
                semantic_candidate_rows(connection, status, group),
                desired=remaining,
                enforce_issue_cap=False,
                issue_cap=issue_cap,
                selected=selected,
                selected_digests=selected_digests,
                achieved_lengths=achieved_lengths,
                issue_counts=issue_counts,
                catalog_by_id=catalog_by_id,
            )
            issue_cap_relaxation_selected += relaxed

    deficits_before_reallocation = {
        group: max(0, targets[group] - achieved_lengths[group])
        for group in REVIEW_LENGTH_GROUPS
    }
    remaining_total = requested_size - len(selected)
    if remaining_total:
        added = fill_semantic_candidates(
            semantic_candidate_rows(connection, status, None),
            desired=remaining_total,
            enforce_issue_cap=True,
            issue_cap=issue_cap,
            selected=selected,
            selected_digests=selected_digests,
            achieved_lengths=achieved_lengths,
            issue_counts=issue_counts,
            catalog_by_id=catalog_by_id,
        )
        remaining_total -= added
    if remaining_total:
        relaxed = fill_semantic_candidates(
            semantic_candidate_rows(connection, status, None),
            desired=remaining_total,
            enforce_issue_cap=False,
            issue_cap=issue_cap,
            selected=selected,
            selected_digests=selected_digests,
            achieved_lengths=achieved_lengths,
            issue_counts=issue_counts,
            catalog_by_id=catalog_by_id,
        )
        issue_cap_relaxation_selected += relaxed

    achieved = {
        group: achieved_lengths[group] for group in REVIEW_LENGTH_GROUPS
    }
    shortfalls = {
        group: max(0, targets[group] - achieved[group])
        for group in REVIEW_LENGTH_GROUPS
    }
    excess = {
        group: max(0, achieved[group] - targets[group])
        for group in REVIEW_LENGTH_GROUPS
    }
    maximum_issue_count = max(issue_counts.values(), default=0)
    return SelectionResult(
        candidates=selected,
        metadata={
            "achieved_length_targets": achieved,
            "issue_cap_count": issue_cap,
            "issue_cap_exceeded": maximum_issue_count > issue_cap,
            "issue_cap_fraction": ISSUE_CAP_FRACTION,
            "issue_cap_relaxation_selected": issue_cap_relaxation_selected,
            "maximum_achieved_issue_count": maximum_issue_count,
            "length_quota_excess_after_reallocation": excess,
            "length_quota_reallocated_count": sum(excess.values()),
            "length_quota_shortfalls_after_reallocation": shortfalls,
            "length_quota_shortfalls_before_reallocation": (
                deficits_before_reallocation
            ),
            "requested_length_targets": targets,
            "requested_size": requested_size,
            "selected_size": len(selected),
            "total_shortfall": requested_size - len(selected),
        },
    )


def select_unsupported_lane(
    connection: sqlite3.Connection,
    requested_size: int,
) -> SelectionResult:
    rows = connection.execute(
        f"""
        WITH fine_stratum_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY unsupported_stratum
                    ORDER BY selection_key, digest
                ) AS fine_stratum_rank
            FROM owned_candidates
            WHERE mapping_status = 'UNSUPPORTED'
        ),
        product_fair AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY product_key
                    ORDER BY fine_stratum_rank, selection_key, digest
                ) AS product_round
            FROM fine_stratum_ranked
        )
        SELECT {CANDIDATE_COLUMNS}
        FROM product_fair
        ORDER BY product_round, selection_key, digest
        LIMIT ?
        """,
        (requested_size,),
    )
    selected = [candidate_from_row(row) for row in rows]
    return SelectionResult(
        candidates=selected,
        metadata={
            "diversity_order": (
                "round-robin product coverage over product/issue/sub-issue/"
                "review-length strata"
            ),
            "requested_size": requested_size,
            "selected_size": len(selected),
            "total_shortfall": requested_size - len(selected),
        },
    )


def record_metadata(
    candidate: Candidate,
    *,
    selection_rank: int,
    catalog_by_id: Mapping[int, TaxonomyMetadata],
    archive_paths: Sequence[Path],
) -> dict[str, Any]:
    taxonomy = catalog_by_id[candidate.taxonomy_id]
    return {
        "candidate_sentinelvoice_intents": list(
            taxonomy.assignment.candidate_intents
        ),
        "complaint_id": candidate.complaint_id,
        "date_received": candidate.date_received,
        "issue": taxonomy.key.issue,
        "length_bin": length_bin(candidate.narrative_length),
        "mapping_status": candidate.mapping_status,
        "narrative_length": candidate.narrative_length,
        "narrative_sha256": candidate.digest.hex(),
        "privacy_screening_signal": False,
        "product": taxonomy.key.product,
        "review_length_group": candidate.review_length_group,
        "selection_key": candidate.selection_key.hex(),
        "selection_rank": selection_rank,
        "source_archive": archive_paths[candidate.archive_index].name,
        "sub_issue": taxonomy.key.sub_issue,
        "sub_product": taxonomy.key.sub_product,
    }


def local_record(manifest_record: Mapping[str, Any], narrative: str) -> dict[str, Any]:
    return {
        "candidate_sentinelvoice_intents": manifest_record[
            "candidate_sentinelvoice_intents"
        ],
        "complaint_id": manifest_record["complaint_id"],
        "date_received": manifest_record["date_received"],
        "issue": manifest_record["issue"],
        "length_bin": manifest_record["length_bin"],
        "mapping_status": manifest_record["mapping_status"],
        "narrative": narrative,
        "narrative_length": manifest_record["narrative_length"],
        "narrative_sha256": manifest_record["narrative_sha256"],
        "product": manifest_record["product"],
        "source_archive": manifest_record["source_archive"],
        "sub_issue": manifest_record["sub_issue"],
        "sub_product": manifest_record["sub_product"],
    }


def materialize_local_rows(
    archive_paths: Sequence[Path],
    manifest_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    wanted = {
        (
            record["source_archive"],
            record["complaint_id"],
            record["narrative_sha256"],
        ): index
        for index, record in enumerate(manifest_records)
    }
    if len(wanted) != len(manifest_records):
        raise ValueError("selected record identities are not unique")
    narratives: dict[int, str] = {}
    for archive_path in archive_paths:
        if not any(identity[0] == archive_path.name for identity in wanted):
            continue
        print(f"Materializing selected text from {archive_path.name}", flush=True)
        for row in archive_csv_reader(archive_path):
            narrative = row.get(NARRATIVE_FIELD) or ""
            if not narrative.strip():
                continue
            complaint_id = (row.get(COMPLAINT_ID_FIELD) or "").strip()
            digest = hashlib.sha256(narrative.encode("utf-8")).hexdigest()
            identity = (archive_path.name, complaint_id, digest)
            index = wanted.get(identity)
            if index is None:
                continue
            _, has_privacy_signal = privacy_observation(narrative)
            if has_privacy_signal:
                raise ValueError("selected narrative failed the privacy recheck")
            if len(narrative) != manifest_records[index]["narrative_length"]:
                raise ValueError("selected narrative length changed during materialization")
            narratives[index] = narrative
    if len(narratives) != len(manifest_records):
        missing = len(manifest_records) - len(narratives)
        raise ValueError(f"failed to materialize {missing} selected narratives")
    return [
        local_record(record, narratives[index])
        for index, record in enumerate(manifest_records)
    ]


def stable_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def count_distribution(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
) -> list[dict[str, Any]]:
    counts: Counter[Any] = Counter(record[field_name] for record in records)
    return [
        {field_name: value, "count": count}
        for value, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0] or ""),
        )
    ]


def selected_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_counts: Counter[str] = Counter()
    for record in records:
        candidate_counts.update(record["candidate_sentinelvoice_intents"])
    dates = [record["date_received"] for record in records]
    return {
        "archive_coverage": count_distribution(records, "source_archive"),
        "candidate_intent_counts_may_overlap": True,
        "candidate_sentinelvoice_intent_coverage": dict(
            sorted(candidate_counts.items())
        ),
        "date_received_maximum": max(dates) if dates else None,
        "date_received_minimum": min(dates) if dates else None,
        "issue_distribution": count_distribution(records, "issue"),
        "length_bin_distribution": count_distribution(records, "length_bin"),
        "product_distribution": count_distribution(records, "product"),
        "review_length_group_distribution": count_distribution(
            records,
            "review_length_group",
        ),
        "selected_count": len(records),
    }


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


def display_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def build_review_pool(
    archive_paths: Iterable[Path],
    mapping: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    *,
    requested_sizes: Mapping[str, int],
    source_manifest_sha256: str,
    candidate_plan_sha256: str,
    taxonomy_mapping_sha256: str,
    builder_script_sha256: str,
    manifest_output_path: Path,
    local_output_path: Path,
) -> dict[str, Any]:
    archive_paths = tuple(sorted(archive_paths, key=lambda path: path.name))
    if set(requested_sizes) != set(SELECTABLE_STATUSES):
        raise ValueError("requested sizes must cover exactly the selectable statuses")
    if any(size < 0 for size in requested_sizes.values()):
        raise ValueError("requested sizes cannot be negative")

    assignments = load_frozen_assignments(mapping)
    catalog_by_key, catalog_by_id = taxonomy_catalog(assignments)
    database_path, connection = create_index()
    try:
        statistics = scan_eligible_candidates(
            archive_paths,
            catalog_by_key,
            connection,
        )
        unique_clean_by_status = count_by_status(
            connection,
            table="eligible_hash_lanes",
        )
        validate_against_candidate_plan(
            candidate_plan,
            statistics,
            unique_clean_by_status,
        )
        create_owned_candidates(connection)
        owned_by_status = count_by_status(connection, table="owned_candidates")
        duplicate_summary = duplicate_statistics(connection)

        lane_results = {
            "NEAR_MATCH": select_semantic_lane(
                connection,
                "NEAR_MATCH",
                requested_sizes["NEAR_MATCH"],
                catalog_by_id,
            ),
            "AMBIGUOUS": select_semantic_lane(
                connection,
                "AMBIGUOUS",
                requested_sizes["AMBIGUOUS"],
                catalog_by_id,
            ),
            "UNSUPPORTED": select_unsupported_lane(
                connection,
                requested_sizes["UNSUPPORTED"],
            ),
        }
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    records: list[dict[str, Any]] = []
    for status in SELECTABLE_STATUSES:
        for rank, candidate in enumerate(
            lane_results[status].candidates,
            start=1,
        ):
            records.append(
                record_metadata(
                    candidate,
                    selection_rank=rank,
                    catalog_by_id=catalog_by_id,
                    archive_paths=archive_paths,
                )
            )

    local_rows = materialize_local_rows(archive_paths, records)
    local_bytes = stable_jsonl_bytes(local_rows)
    by_status_records = {
        status: [record for record in records if record["mapping_status"] == status]
        for status in SELECTABLE_STATUSES
    }
    manifest = {
        "artifact_boundary": {
            "classifier_predictions_run": False,
            "contains_complaint_ids": True,
            "contains_narrative_hash_values": True,
            "contains_narrative_text": False,
            "final_evaluation_set_created": False,
            "final_or_gold_intent_labels_assigned": False,
            "model_training_or_retraining_run": False,
            "review_or_evaluation_candidate_pool_only": True,
        },
        "builder_version": BUILDER_VERSION,
        "duplicate_exclusions": duplicate_summary,
        "eligibility": {
            "cross_lane_ownership_precedence": list(OWNERSHIP_PRECEDENCE),
            "exact_narrative_hash": "SHA-256 over exact UTF-8 narrative field text",
            "globally_deduplicated_owned_population_by_status": owned_by_status,
            "privacy_excluded_records_by_status": {
                status: statistics.privacy_excluded_by_status[status]
                for status in MAPPING_STATUSES
            },
            "privacy_excluded_records_total": sum(
                statistics.privacy_excluded_by_status.values()
            ),
            "privacy_screening_reference": (
                "scripts/profile_cfpb_narratives.py::PRIVACY_PATTERNS"
            ),
            "unique_privacy_free_hashes_before_cross_lane_ownership": (
                unique_clean_by_status
            ),
        },
        "input_integrity": {
            "builder_script_sha256": builder_script_sha256,
            "candidate_plan_sha256": candidate_plan_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "taxonomy_mapping_sha256": taxonomy_mapping_sha256,
        },
        "local_review_material": {
            "contains_narrative_text": True,
            "git_ignored": True,
            "path": display_output_path(local_output_path),
            "row_count": len(local_rows),
            "schema_version": LOCAL_SCHEMA_VERSION,
            "sha256": sha256_bytes(local_bytes),
        },
        "quota_results": {
            status: lane_results[status].metadata for status in SELECTABLE_STATUSES
        },
        "reconciliation": {
            "candidate_plan_counts_match": True,
            "clean_records_by_status": {
                status: statistics.clean_records_by_status[status]
                for status in MAPPING_STATUSES
            },
            "narrative_records_by_status": {
                status: statistics.narrative_records_by_status[status]
                for status in MAPPING_STATUSES
            },
            "selected_lane_count_sum": len(records),
            "total_full_records_rows_scanned": statistics.total_csv_rows,
        },
        "requested_lane_sizes": {
            status: requested_sizes[status] for status in SELECTABLE_STATUSES
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selected_records": records,
        "selection_algorithm": {
            "candidate_intents_are_overlapping_not_ground_truth": True,
            "issue_cap_fraction": ISSUE_CAP_FRACTION,
            "length_target_weights": REVIEW_LENGTH_WEIGHTS,
            "selection_key": (
                "SHA-256 over a frozen namespace plus narrative hash, mapping lane, "
                "complaint ID, source archive, date, and full CFPB taxonomy tuple"
            ),
            "selection_namespace": SELECTION_NAMESPACE,
            "version": SELECTION_ALGORITHM_VERSION,
        },
        "summary": {
            "by_mapping_status": {
                status: selected_summary(by_status_records[status])
                for status in SELECTABLE_STATUSES
            },
            "overall": selected_summary(records),
            "selected_by_mapping_status": {
                status: len(by_status_records[status])
                for status in SELECTABLE_STATUSES
            },
            "total_selected": len(records),
        },
    }
    serialized_manifest = stable_json_bytes(manifest)
    if any("narrative" in record for record in manifest["selected_records"]):
        raise AssertionError("tracked manifest record contains narrative text")
    forbidden_label_fields = {"final_intent", "gold_label", "ground_truth_intent"}
    manifest_text = serialized_manifest.decode("utf-8")
    if any(field_name in manifest_text for field_name in forbidden_label_fields):
        raise AssertionError("tracked manifest contains a forbidden label field")

    atomic_write(local_output_path, local_bytes)
    atomic_write(manifest_output_path, serialized_manifest)
    return manifest


def validate_archive_files(
    raw_manifest: Mapping[str, Any],
    archive_paths: Sequence[Path],
) -> None:
    metadata_by_name = {
        entry["local_filename"]: entry for entry in raw_manifest.get("archives", [])
    }
    for archive_path in archive_paths:
        metadata = metadata_by_name.get(archive_path.name)
        if metadata is None:
            raise ValueError(f"archive is absent from raw manifest: {archive_path.name}")
        if sha256_path(archive_path) != metadata.get("sha256"):
            raise ValueError(f"archive SHA-256 differs from manifest: {archive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-size", type=int, default=DEFAULT_LANE_SIZES["NEAR_MATCH"])
    parser.add_argument(
        "--ambiguous-size",
        type=int,
        default=DEFAULT_LANE_SIZES["AMBIGUOUS"],
    )
    parser.add_argument(
        "--unsupported-size",
        type=int,
        default=DEFAULT_LANE_SIZES["UNSUPPORTED"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_manifest_bytes = RAW_MANIFEST_PATH.read_bytes()
    mapping_bytes = MAPPING_PATH.read_bytes()
    candidate_plan_bytes = CANDIDATE_PLAN_PATH.read_bytes()
    builder_script_bytes = Path(__file__).read_bytes()
    raw_manifest = json.loads(raw_manifest_bytes)
    mapping = json.loads(mapping_bytes)
    candidate_plan = json.loads(candidate_plan_bytes)

    source_manifest_sha256 = sha256_bytes(raw_manifest_bytes)
    taxonomy_mapping_sha256 = sha256_bytes(mapping_bytes)
    candidate_plan_sha256 = sha256_bytes(candidate_plan_bytes)
    candidate_input = candidate_plan.get("input_integrity", {})
    if candidate_input.get("cfpb_manifest_sha256") != source_manifest_sha256:
        raise ValueError("candidate plan does not match the current CFPB manifest")
    if candidate_input.get("cfpb_taxonomy_mapping_sha256") != taxonomy_mapping_sha256:
        raise ValueError("candidate plan does not match the current taxonomy mapping")

    archive_paths = tuple(production_archive_paths(raw_manifest))
    validate_archive_files(raw_manifest, archive_paths)
    manifest = build_review_pool(
        archive_paths,
        mapping,
        candidate_plan,
        requested_sizes={
            "NEAR_MATCH": args.near_size,
            "AMBIGUOUS": args.ambiguous_size,
            "UNSUPPORTED": args.unsupported_size,
        },
        source_manifest_sha256=source_manifest_sha256,
        candidate_plan_sha256=candidate_plan_sha256,
        taxonomy_mapping_sha256=taxonomy_mapping_sha256,
        builder_script_sha256=sha256_bytes(builder_script_bytes),
        manifest_output_path=MANIFEST_OUTPUT_PATH,
        local_output_path=LOCAL_OUTPUT_PATH,
    )
    print(
        "Built CFPB review pool: "
        f"selected={manifest['summary']['total_selected']}, "
        f"manifest={MANIFEST_OUTPUT_PATH}, local={LOCAL_OUTPUT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
