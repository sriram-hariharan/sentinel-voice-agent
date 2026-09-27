"""Plan CFPB narrative evaluation candidates without selecting or labeling them."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.profile_cfpb_narratives import PRIVACY_PATTERNS
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from profile_cfpb_narratives import PRIVACY_PATTERNS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
RAW_DIR = EXTERNAL_ROOT / "raw/cfpb_narratives"
RAW_FILES_DIR = RAW_DIR / "files"
RAW_MANIFEST_PATH = RAW_DIR / "manifest.json"
MAPPING_PATH = EXTERNAL_ROOT / "mappings/cfpb_to_sentinel_intents.json"
OUTPUT_PATH = EXTERNAL_ROOT / "processed/cfpb/candidate_plan.json"

PRODUCT_FIELD = "Product"
SUB_PRODUCT_FIELD = "Sub-product"
ISSUE_FIELD = "Issue"
SUB_ISSUE_FIELD = "Sub-issue"
NARRATIVE_FIELD = "Consumer complaint narrative"
COMPLAINT_ID_FIELD = "Complaint ID"
REQUIRED_FIELDS = {
    PRODUCT_FIELD,
    SUB_PRODUCT_FIELD,
    ISSUE_FIELD,
    SUB_ISSUE_FIELD,
    NARRATIVE_FIELD,
    COMPLAINT_ID_FIELD,
}

MAPPING_STATUSES = ("EXACT_MATCH", "NEAR_MATCH", "AMBIGUOUS", "UNSUPPORTED")
PROTECTED_WRITE_INTENTS = frozenset({"freeze_card", "create_dispute"})
PLANNING_VERSION = "cfpb-narrative-candidate-plan.2026-09-27.v1"
SCHEMA_VERSION = "cfpb-narrative-candidate-plan.v1"
PRIVACY_RULE_VERSION = "cfpb-profile-privacy-patterns.v1"
SQL_BATCH_SIZE = 10_000

LENGTH_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("1-200", 1, 200),
    ("201-500", 201, 500),
    ("501-1000", 501, 1000),
    ("1001-2000", 1001, 2000),
    ("2001-4000", 2001, 4000),
    ("4001+", 4001, None),
)
LENGTH_QUALIFIED_KEYS = ("lte_500", "lte_1000", "lte_2000", "gt_2000")


@dataclass(frozen=True, order=True)
class TaxonomyKey:
    product: str
    sub_product: str | None
    issue: str
    sub_issue: str | None


@dataclass(frozen=True)
class FrozenAssignment:
    mapping_status: str
    candidate_intents: tuple[str, ...]
    protected_action_risk: str
    matched_rule_id: str


@dataclass
class LaneAggregates:
    record_count: int = 0
    length_counts: Counter[int] = field(default_factory=Counter)
    length_bins: Counter[str] = field(default_factory=Counter)
    length_qualified: Counter[str] = field(default_factory=Counter)
    privacy_records: Counter[str] = field(default_factory=Counter)
    privacy_matches: Counter[str] = field(default_factory=Counter)
    product_counts: Counter[str] = field(default_factory=Counter)
    sub_product_counts: Counter[str | None] = field(default_factory=Counter)
    issue_counts: Counter[str] = field(default_factory=Counter)
    sub_issue_counts: Counter[str | None] = field(default_factory=Counter)
    full_taxonomy_counts: Counter[TaxonomyKey] = field(default_factory=Counter)
    candidate_intent_counts: Counter[str] = field(default_factory=Counter)


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise ValueError("CFPB CSV has no header")
    return [field_name.lstrip("\ufeff").strip() for field_name in fieldnames]


def nullable(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def taxonomy_key_from_values(
    product: str | None,
    sub_product: str | None,
    issue: str | None,
    sub_issue: str | None,
) -> TaxonomyKey:
    normalized_product = (product or "").strip()
    normalized_issue = (issue or "").strip()
    if not normalized_product or not normalized_issue:
        raise ValueError("narrative-bearing rows require Product and Issue")
    return TaxonomyKey(
        product=normalized_product,
        sub_product=nullable(sub_product),
        issue=normalized_issue,
        sub_issue=nullable(sub_issue),
    )


def taxonomy_key_from_mapping(entry: Mapping[str, Any]) -> TaxonomyKey:
    return taxonomy_key_from_values(
        entry.get("product"),
        entry.get("sub_product"),
        entry.get("issue"),
        entry.get("sub_issue"),
    )


def load_frozen_assignments(
    mapping: Mapping[str, Any],
) -> dict[TaxonomyKey, FrozenAssignment]:
    if mapping.get("schema_version") != "cfpb-to-sentinel-intents.v1":
        raise ValueError("unexpected CFPB mapping schema version")
    if tuple(mapping.get("allowed_mapping_statuses", ())) != MAPPING_STATUSES:
        raise ValueError("frozen mapping status order or values changed")
    artifact_boundary = mapping.get("artifact_boundary", {})
    if any(artifact_boundary.values()):
        raise ValueError("frozen taxonomy mapping crossed its no-execution boundary")

    rows = mapping.get("mapped_taxonomy_combinations")
    if not isinstance(rows, list):
        raise TypeError("frozen mapping must contain mapped_taxonomy_combinations")
    assignments: dict[TaxonomyKey, FrozenAssignment] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each frozen taxonomy assignment must be an object")
        key = taxonomy_key_from_mapping(row)
        if key in assignments:
            raise ValueError(f"duplicate frozen taxonomy assignment: {key}")
        status = row.get("mapping_status")
        candidates = row.get("candidate_sentinelvoice_intents")
        if status not in MAPPING_STATUSES:
            raise ValueError(f"invalid frozen mapping status for {key}: {status}")
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, str) and candidate for candidate in candidates
        ):
            raise ValueError(f"invalid frozen candidate intents for {key}")
        if status == "EXACT_MATCH" and PROTECTED_WRITE_INTENTS.intersection(candidates):
            raise ValueError(f"protected-write EXACT_MATCH is forbidden: {key}")
        assignments[key] = FrozenAssignment(
            mapping_status=status,
            candidate_intents=tuple(candidates),
            protected_action_risk=str(row.get("protected_action_risk", "NONE")),
            matched_rule_id=str(row.get("matched_rule_id", "")),
        )

    expected_count = mapping.get("coverage", {}).get("total_mapped_taxonomy_tuples")
    if len(assignments) != expected_count:
        raise ValueError("frozen mapping tuple count does not reconcile")
    return assignments


def length_bin(character_length: int) -> str:
    if character_length < 1:
        raise ValueError("narrative length must be positive")
    for label, lower, upper in LENGTH_BINS:
        if character_length >= lower and (upper is None or character_length <= upper):
            return label
    raise AssertionError("length bin definitions do not cover the input")


def update_length_qualified(counter: Counter[str], character_length: int) -> None:
    if character_length <= 500:
        counter["lte_500"] += 1
    if character_length <= 1000:
        counter["lte_1000"] += 1
    if character_length <= 2000:
        counter["lte_2000"] += 1
    if character_length > 2000:
        counter["gt_2000"] += 1


def rank_value(length_counts: Counter[int], rank: int) -> int:
    if rank < 1 or rank > sum(length_counts.values()):
        raise ValueError("rank is outside the observed length distribution")
    cumulative = 0
    for character_length in sorted(length_counts):
        cumulative += length_counts[character_length]
        if cumulative >= rank:
            return character_length
    raise AssertionError("weighted rank traversal did not resolve")


def weighted_length_statistics(length_counts: Counter[int]) -> dict[str, int | float | None]:
    count = sum(length_counts.values())
    if count == 0:
        return {
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "p50_nearest_rank": None,
            "p75_nearest_rank": None,
            "p90_nearest_rank": None,
            "p95_nearest_rank": None,
            "p99_nearest_rank": None,
        }
    if count % 2:
        median: int | float = rank_value(length_counts, (count + 1) // 2)
    else:
        left = rank_value(length_counts, count // 2)
        right = rank_value(length_counts, count // 2 + 1)
        median = (left + right) / 2
    return {
        "maximum": max(length_counts),
        "mean": round(
            sum(length * frequency for length, frequency in length_counts.items())
            / count,
            3,
        ),
        "median": median,
        "minimum": min(length_counts),
        "p50_nearest_rank": rank_value(length_counts, math.ceil(0.50 * count)),
        "p75_nearest_rank": rank_value(length_counts, math.ceil(0.75 * count)),
        "p90_nearest_rank": rank_value(length_counts, math.ceil(0.90 * count)),
        "p95_nearest_rank": rank_value(length_counts, math.ceil(0.95 * count)),
        "p99_nearest_rank": rank_value(length_counts, math.ceil(0.99 * count)),
    }


def distribution(
    counter: Counter[str | None], *, value_key: str
) -> list[dict[str, Any]]:
    return [
        {"count": count, value_key: value}
        for value, count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0] or "")
        )
    ]


def taxonomy_distribution(
    counter: Counter[TaxonomyKey],
) -> list[dict[str, Any]]:
    return [
        {
            "count": count,
            "issue": key.issue,
            "product": key.product,
            "sub_issue": key.sub_issue,
            "sub_product": key.sub_product,
        }
        for key, count in sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                item[0].product,
                item[0].sub_product or "",
                item[0].issue,
                item[0].sub_issue or "",
            ),
        )
    ]


def create_index() -> tuple[Path, sqlite3.Connection]:
    with tempfile.NamedTemporaryFile(
        prefix="cfpb_candidate_plan_", suffix=".sqlite3", delete=False
    ) as temporary:
        path = Path(temporary.name)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        """
        CREATE TABLE narrative_hash_lanes (
            digest BLOB NOT NULL,
            mapping_status TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            has_privacy_signal INTEGER NOT NULL,
            character_length INTEGER NOT NULL,
            PRIMARY KEY (digest, mapping_status)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE complaint_id_lanes (
            complaint_id TEXT NOT NULL,
            mapping_status TEXT NOT NULL,
            PRIMARY KEY (complaint_id, mapping_status)
        ) WITHOUT ROWID
        """
    )
    return path, connection


def store_index_batch(
    connection: sqlite3.Connection,
    hash_rows: list[tuple[bytes, str, int, int]],
    complaint_rows: list[tuple[str, str]],
) -> None:
    connection.executemany(
        """
        INSERT INTO narrative_hash_lanes (
            digest,
            mapping_status,
            occurrence_count,
            has_privacy_signal,
            character_length
        ) VALUES (?, ?, 1, ?, ?)
        ON CONFLICT(digest, mapping_status) DO UPDATE SET
            occurrence_count = occurrence_count + 1,
            has_privacy_signal = MAX(
                has_privacy_signal,
                excluded.has_privacy_signal
            )
        """,
        hash_rows,
    )
    connection.executemany(
        """
        INSERT OR IGNORE INTO complaint_id_lanes (complaint_id, mapping_status)
        VALUES (?, ?)
        """,
        complaint_rows,
    )


def privacy_observation(narrative: str) -> tuple[dict[str, int], bool]:
    match_counts = {
        pattern_name: len(pattern.findall(narrative))
        for pattern_name, pattern in PRIVACY_PATTERNS.items()
    }
    return match_counts, any(match_counts.values())


def update_lane(
    lane: LaneAggregates,
    *,
    key: TaxonomyKey,
    assignment: FrozenAssignment,
    character_length: int,
    privacy_matches: Mapping[str, int],
    has_privacy_signal: bool,
) -> None:
    lane.record_count += 1
    lane.length_counts[character_length] += 1
    lane.length_bins[length_bin(character_length)] += 1
    update_length_qualified(lane.length_qualified, character_length)
    lane.product_counts[key.product] += 1
    lane.sub_product_counts[key.sub_product] += 1
    lane.issue_counts[key.issue] += 1
    lane.sub_issue_counts[key.sub_issue] += 1
    lane.full_taxonomy_counts[key] += 1
    for pattern_name, match_count in privacy_matches.items():
        if match_count:
            lane.privacy_records[pattern_name] += 1
            lane.privacy_matches[pattern_name] += match_count
    if has_privacy_signal:
        lane.privacy_records["any_privacy_screening_signal"] += 1
    if assignment.mapping_status in {"NEAR_MATCH", "AMBIGUOUS"}:
        for candidate_intent in assignment.candidate_intents:
            lane.candidate_intent_counts[candidate_intent] += 1


def scan_archives(
    archive_paths: Iterable[Path],
    assignments: Mapping[TaxonomyKey, FrozenAssignment],
    connection: sqlite3.Connection,
) -> tuple[dict[str, LaneAggregates], int]:
    lanes = {status: LaneAggregates() for status in MAPPING_STATUSES}
    total_csv_rows = 0
    hash_batch: list[tuple[bytes, str, int, int]] = []
    complaint_batch: list[tuple[str, str]] = []
    for archive_path in archive_paths:
        print(f"Planning {archive_path.name}", flush=True)
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
                    binary_member, encoding="utf-8-sig", newline=""
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
                for row_number, row in enumerate(reader, start=1):
                    total_csv_rows += 1
                    narrative = row.get(NARRATIVE_FIELD) or ""
                    if not narrative.strip():
                        continue
                    complaint_id = (row.get(COMPLAINT_ID_FIELD) or "").strip()
                    if not complaint_id:
                        raise ValueError(
                            f"missing complaint ID at row {row_number} in "
                            f"{archive_path.name}"
                        )
                    key = taxonomy_key_from_values(
                        row.get(PRODUCT_FIELD),
                        row.get(SUB_PRODUCT_FIELD),
                        row.get(ISSUE_FIELD),
                        row.get(SUB_ISSUE_FIELD),
                    )
                    try:
                        assignment = assignments[key]
                    except KeyError as exc:
                        raise ValueError(
                            "narrative row is absent from the frozen V2-C2M mapping: "
                            f"{key}"
                        ) from exc

                    pattern_matches, has_privacy_signal = privacy_observation(narrative)
                    character_length = len(narrative)
                    digest = hashlib.sha256(narrative.encode("utf-8")).digest()
                    lane = lanes[assignment.mapping_status]
                    update_lane(
                        lane,
                        key=key,
                        assignment=assignment,
                        character_length=character_length,
                        privacy_matches=pattern_matches,
                        has_privacy_signal=has_privacy_signal,
                    )
                    hash_batch.append(
                        (
                            digest,
                            assignment.mapping_status,
                            int(has_privacy_signal),
                            character_length,
                        )
                    )
                    complaint_batch.append((complaint_id, assignment.mapping_status))
                    if len(hash_batch) == SQL_BATCH_SIZE:
                        store_index_batch(connection, hash_batch, complaint_batch)
                        hash_batch.clear()
                        complaint_batch.clear()
    if hash_batch:
        store_index_batch(connection, hash_batch, complaint_batch)
    connection.commit()
    return lanes, total_csv_rows


def hash_statistics(connection: sqlite3.Connection) -> tuple[dict[str, Any], dict[str, Any]]:
    by_status = {
        status: {
            "duplicate_narrative_groups": 0,
            "repeated_narrative_occurrences_beyond_first": 0,
            "unique_narrative_hashes": 0,
        }
        for status in MAPPING_STATUSES
    }
    rows = connection.execute(
        """
        SELECT
            mapping_status,
            COUNT(*),
            COALESCE(SUM(CASE WHEN occurrence_count > 1 THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(occurrence_count - 1), 0)
        FROM narrative_hash_lanes
        GROUP BY mapping_status
        """
    )
    for status, unique_hashes, duplicate_groups, repeated in rows:
        by_status[status] = {
            "duplicate_narrative_groups": duplicate_groups,
            "repeated_narrative_occurrences_beyond_first": repeated,
            "unique_narrative_hashes": unique_hashes,
        }

    unique_hashes, duplicate_groups, repeated = connection.execute(
        """
        WITH overall AS (
            SELECT digest, SUM(occurrence_count) AS occurrence_count
            FROM narrative_hash_lanes
            GROUP BY digest
        )
        SELECT
            COUNT(*),
            COALESCE(SUM(CASE WHEN occurrence_count > 1 THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(occurrence_count - 1), 0)
        FROM overall
        """
    ).fetchone()
    cross_lane_hashes, cross_lane_records = connection.execute(
        """
        WITH cross_lane AS (
            SELECT digest, SUM(occurrence_count) AS record_count
            FROM narrative_hash_lanes
            GROUP BY digest
            HAVING COUNT(*) > 1
        )
        SELECT COUNT(*), COALESCE(SUM(record_count), 0)
        FROM cross_lane
        """
    ).fetchone()
    overall = {
        "duplicate_narrative_groups": duplicate_groups,
        "hashes_appearing_in_multiple_mapping_lanes": cross_lane_hashes,
        "records_involved_in_cross_lane_duplicate_hashes": cross_lane_records,
        "repeated_narrative_occurrences_beyond_first": repeated,
        "unique_exact_narrative_hashes": unique_hashes,
    }
    return overall, by_status


def complaint_id_statistics(
    connection: sqlite3.Connection,
) -> tuple[int, dict[str, int]]:
    by_status = {status: 0 for status in MAPPING_STATUSES}
    for status, count in connection.execute(
        "SELECT mapping_status, COUNT(*) FROM complaint_id_lanes GROUP BY mapping_status"
    ):
        by_status[status] = count
    overall = connection.execute(
        "SELECT COUNT(DISTINCT complaint_id) FROM complaint_id_lanes"
    ).fetchone()[0]
    return overall, by_status


def length_qualified_sql(prefix: str = "") -> str:
    return ", ".join(
        (
            f"SUM(CASE WHEN {prefix}character_length <= 500 THEN 1 ELSE 0 END)",
            f"SUM(CASE WHEN {prefix}character_length <= 1000 THEN 1 ELSE 0 END)",
            f"SUM(CASE WHEN {prefix}character_length <= 2000 THEN 1 ELSE 0 END)",
            f"SUM(CASE WHEN {prefix}character_length > 2000 THEN 1 ELSE 0 END)",
        )
    )


def qualified_row(values: Iterable[int]) -> dict[str, int]:
    return dict(zip(LENGTH_QUALIFIED_KEYS, values, strict=True))


def unique_population_statistics(
    connection: sqlite3.Connection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_status = {
        status: {
            "exact_deduplicated": 0,
            "exact_deduplicated_length_qualified": {
                key: 0 for key in LENGTH_QUALIFIED_KEYS
            },
            "exact_deduplicated_without_privacy_signal": 0,
            "privacy_free_exact_deduplicated_length_qualified": {
                key: 0 for key in LENGTH_QUALIFIED_KEYS
            },
        }
        for status in MAPPING_STATUSES
    }
    rows = connection.execute(
        f"""
        SELECT
            mapping_status,
            COUNT(*),
            {length_qualified_sql()},
            SUM(CASE WHEN has_privacy_signal = 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length <= 500 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length <= 1000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length <= 2000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length > 2000 THEN 1 ELSE 0 END)
        FROM narrative_hash_lanes
        GROUP BY mapping_status
        """
    )
    for row in rows:
        status = row[0]
        by_status[status] = {
            "exact_deduplicated": row[1],
            "exact_deduplicated_length_qualified": qualified_row(row[2:6]),
            "exact_deduplicated_without_privacy_signal": row[6],
            "privacy_free_exact_deduplicated_length_qualified": qualified_row(
                row[7:11]
            ),
        }

    overall_row = connection.execute(
        f"""
        WITH overall AS (
            SELECT
                digest,
                MIN(character_length) AS character_length,
                MAX(has_privacy_signal) AS has_privacy_signal
            FROM narrative_hash_lanes
            GROUP BY digest
        )
        SELECT
            COUNT(*),
            {length_qualified_sql()},
            SUM(CASE WHEN has_privacy_signal = 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length <= 500 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length <= 1000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length <= 2000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_privacy_signal = 0 AND character_length > 2000 THEN 1 ELSE 0 END)
        FROM overall
        """
    ).fetchone()
    overall = {
        "exact_deduplicated": overall_row[0],
        "exact_deduplicated_length_qualified": qualified_row(overall_row[1:5]),
        "exact_deduplicated_without_privacy_signal": overall_row[5],
        "privacy_free_exact_deduplicated_length_qualified": qualified_row(
            overall_row[6:10]
        ),
    }
    return overall, by_status


def privacy_payload(lane: LaneAggregates) -> dict[str, Any]:
    patterns = {
        pattern_name: {
            "records_with_pattern": lane.privacy_records[pattern_name],
            "total_matches": lane.privacy_matches[pattern_name],
        }
        for pattern_name in PRIVACY_PATTERNS
    }
    any_signal = lane.privacy_records["any_privacy_screening_signal"]
    return {
        "any_privacy_screening_signal": any_signal,
        "eligible_without_privacy_signal": lane.record_count - any_signal,
        "patterns": patterns,
    }


def lane_payload(
    lane: LaneAggregates,
    *,
    unique_complaint_ids: int,
    hash_stats: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "candidate_sentinelvoice_intent_record_counts": dict(
            sorted(lane.candidate_intent_counts.items())
        ),
        "candidate_intent_counts_may_overlap": True,
        "duplicate_statistics": dict(hash_stats),
        "length_analysis": {
            "bins": {
                label: lane.length_bins[label] for label, _, _ in LENGTH_BINS
            },
            "statistics": weighted_length_statistics(lane.length_counts),
        },
        "length_qualified_record_counts": {
            key: lane.length_qualified[key] for key in LENGTH_QUALIFIED_KEYS
        },
        "privacy_screening": privacy_payload(lane),
        "record_count": lane.record_count,
        "taxonomy_distribution": {
            "full_taxonomy_tuple": taxonomy_distribution(
                lane.full_taxonomy_counts
            ),
            "issue": distribution(lane.issue_counts, value_key="issue"),
            "product": distribution(lane.product_counts, value_key="product"),
            "sub_issue": distribution(
                lane.sub_issue_counts, value_key="sub_issue"
            ),
            "sub_product": distribution(
                lane.sub_product_counts, value_key="sub_product"
            ),
        },
        "unique_complaint_ids": unique_complaint_ids,
    }


def aggregate_lanes(lanes: Mapping[str, LaneAggregates]) -> LaneAggregates:
    overall = LaneAggregates()
    for lane in lanes.values():
        overall.record_count += lane.record_count
        for target, source in (
            (overall.length_counts, lane.length_counts),
            (overall.length_bins, lane.length_bins),
            (overall.length_qualified, lane.length_qualified),
            (overall.privacy_records, lane.privacy_records),
            (overall.privacy_matches, lane.privacy_matches),
            (overall.product_counts, lane.product_counts),
            (overall.sub_product_counts, lane.sub_product_counts),
            (overall.issue_counts, lane.issue_counts),
            (overall.sub_issue_counts, lane.sub_issue_counts),
            (overall.full_taxonomy_counts, lane.full_taxonomy_counts),
        ):
            target.update(source)
    return overall


def build_plan(
    archive_paths: Iterable[Path],
    mapping: Mapping[str, Any],
    *,
    manifest_sha256: str,
    mapping_sha256: str,
) -> dict[str, Any]:
    archive_paths = tuple(archive_paths)
    assignments = load_frozen_assignments(mapping)
    database_path, connection = create_index()
    try:
        lanes, total_csv_rows = scan_archives(archive_paths, assignments, connection)
        overall_hashes, lane_hashes = hash_statistics(connection)
        unique_ids, lane_unique_ids = complaint_id_statistics(connection)
        unique_population, lane_unique_population = unique_population_statistics(
            connection
        )
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    overall_lane = aggregate_lanes(lanes)
    by_status = {
        status: lane_payload(
            lanes[status],
            unique_complaint_ids=lane_unique_ids[status],
            hash_stats=lane_hashes[status],
        )
        for status in MAPPING_STATUSES
    }
    expected_coverage = mapping.get("coverage", {}).get("by_status", {})
    for status in MAPPING_STATUSES:
        expected = expected_coverage.get(status, {}).get("narrative_count")
        if expected != lanes[status].record_count:
            raise ValueError(
                f"{status} record count differs from frozen mapping: "
                f"{lanes[status].record_count} != {expected}"
            )
    if overall_lane.record_count != mapping.get("coverage", {}).get(
        "total_narrative_bearing_records"
    ):
        raise ValueError("total narrative count differs from frozen mapping")

    all_record_lengths = {
        "overall": {
            key: overall_lane.length_qualified[key]
            for key in LENGTH_QUALIFIED_KEYS
        },
        "by_mapping_status": {
            status: {
                key: lanes[status].length_qualified[key]
                for key in LENGTH_QUALIFIED_KEYS
            }
            for status in MAPPING_STATUSES
        },
    }
    potential_populations = {
        "all_narrative_bearing_records": {
            "by_mapping_status": {
                status: lanes[status].record_count for status in MAPPING_STATUSES
            },
            "overall": overall_lane.record_count,
        },
        "exact_deduplicated_narratives": {
            "by_mapping_status": {
                status: lane_unique_population[status]["exact_deduplicated"]
                for status in MAPPING_STATUSES
            },
            "overall": unique_population["exact_deduplicated"],
        },
        "exact_deduplicated_without_privacy_signal": {
            "by_mapping_status": {
                status: lane_unique_population[status][
                    "exact_deduplicated_without_privacy_signal"
                ]
                for status in MAPPING_STATUSES
            },
            "overall": unique_population[
                "exact_deduplicated_without_privacy_signal"
            ],
        },
        "length_qualified_counts": {
            "all_narrative_bearing_records": all_record_lengths,
            "exact_deduplicated_narratives": {
                "by_mapping_status": {
                    status: lane_unique_population[status][
                        "exact_deduplicated_length_qualified"
                    ]
                    for status in MAPPING_STATUSES
                },
                "overall": unique_population[
                    "exact_deduplicated_length_qualified"
                ],
            },
            "exact_deduplicated_without_privacy_signal": {
                "by_mapping_status": {
                    status: lane_unique_population[status][
                        "privacy_free_exact_deduplicated_length_qualified"
                    ]
                    for status in MAPPING_STATUSES
                },
                "overall": unique_population[
                    "privacy_free_exact_deduplicated_length_qualified"
                ],
            },
        },
    }
    unsupported = lanes["UNSUPPORTED"]
    return {
        "artifact_boundary": {
            "aggregate_only": True,
            "classifier_predictions_run": False,
            "contains_complaint_ids": False,
            "contains_narrative_hash_values": False,
            "contains_narrative_text": False,
            "final_evaluation_sample_selected": False,
            "individual_narrative_intents_assigned": False,
            "model_training_or_retraining_run": False,
        },
        "builder_version": "cfpb-narrative-candidate-planner.v1",
        "by_mapping_status": by_status,
        "generation_parameters": {
            "candidate_intent_counts_may_overlap": True,
            "duplicate_hash": "SHA-256 over the exact UTF-8 narrative field text",
            "empty_narrative_check": "outer whitespace trim is used only to identify empty fields",
            "length_bins": [
                {
                    "inclusive_lower": lower,
                    "inclusive_upper": upper,
                    "label": label,
                }
                for label, lower, upper in LENGTH_BINS
            ],
            "length_qualified_counts_are_cumulative_except_gt_2000": True,
            "mapping_statuses": list(MAPPING_STATUSES),
            "percentile_method": "nearest rank; median uses conventional midpoint",
            "privacy_screening_reference": (
                "scripts/profile_cfpb_narratives.py::PRIVACY_PATTERNS"
            ),
            "privacy_screening_rule_version": PRIVACY_RULE_VERSION,
            "temporary_index": (
                "SQLite outside committed artifacts; stores hashes/counts/lengths/privacy "
                "booleans and complaint IDs, never narrative text; removed after the run"
            ),
        },
        "input_integrity": {
            "cfpb_manifest_sha256": manifest_sha256,
            "cfpb_taxonomy_mapping_sha256": mapping_sha256,
            "cfpb_taxonomy_mapping_version": mapping.get("mapping_version"),
            "cfpb_taxonomy_inventory_sha256": mapping.get("inventory_sha256"),
        },
        "overall": {
            **overall_hashes,
            "length_analysis": {
                "bins": {
                    label: overall_lane.length_bins[label]
                    for label, _, _ in LENGTH_BINS
                },
                "statistics": weighted_length_statistics(
                    overall_lane.length_counts
                ),
            },
            "privacy_screening": privacy_payload(overall_lane),
            "total_full_records_rows_scanned": total_csv_rows,
            "total_narrative_bearing_records": overall_lane.record_count,
            "unique_complaint_ids": unique_ids,
        },
        "planning_version": PLANNING_VERSION,
        "potential_evaluation_populations": potential_populations,
        "reconciliation": {
            "frozen_mapping_lane_counts_match": True,
            "lane_record_count_sum": sum(
                lanes[status].record_count for status in MAPPING_STATUSES
            ),
            "overall_record_count": overall_lane.record_count,
            "status_count": len(MAPPING_STATUSES),
        },
        "schema_version": SCHEMA_VERSION,
        "source_archive_partition_count": len(archive_paths),
        "source_id": "cfpb_consumer_complaint_narratives_archive",
        "unsupported_ood_diversity": {
            "distinct_full_taxonomy_tuples": len(
                unsupported.full_taxonomy_counts
            ),
            "distinct_issues": len(unsupported.issue_counts),
            "distinct_products": len(unsupported.product_counts),
            "distinct_sub_issues_including_missing": len(
                unsupported.sub_issue_counts
            ),
            "distinct_nonempty_sub_issues": len(
                {
                    sub_issue
                    for sub_issue in unsupported.sub_issue_counts
                    if sub_issue is not None
                }
            ),
            "length_analysis_reference": "/by_mapping_status/UNSUPPORTED/length_analysis",
            "taxonomy_distribution_reference": (
                "/by_mapping_status/UNSUPPORTED/taxonomy_distribution"
            ),
        },
    }


def production_archive_paths(manifest: Mapping[str, Any]) -> list[Path]:
    archives = manifest.get("archives")
    if not isinstance(archives, list) or len(archives) != 21:
        raise ValueError("raw CFPB manifest must contain 21 archives")
    paths: list[Path] = []
    for archive in archives:
        path = RAW_FILES_DIR / archive["local_filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != archive["compressed_byte_size"]:
            raise ValueError(f"archive size differs from manifest: {path}")
        paths.append(path)
    return paths


def main() -> None:
    manifest_bytes = RAW_MANIFEST_PATH.read_bytes()
    mapping_bytes = MAPPING_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    mapping = json.loads(mapping_bytes)
    plan = build_plan(
        production_archive_paths(manifest),
        mapping,
        manifest_sha256=sha256_bytes(manifest_bytes),
        mapping_sha256=sha256_bytes(mapping_bytes),
    )
    expected_rows = manifest.get("summary", {}).get("total_rows")
    if plan["overall"]["total_full_records_rows_scanned"] != expected_rows:
        raise ValueError("planner full-record count differs from raw manifest")
    expected_narratives = manifest.get("summary", {}).get(
        "total_nonempty_narratives"
    )
    if plan["overall"]["total_narrative_bearing_records"] != expected_narratives:
        raise ValueError("planner narrative count differs from raw manifest")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(stable_json_bytes(plan))
    print(
        "Built CFPB narrative candidate plan: "
        f"records={plan['overall']['total_narrative_bearing_records']}, "
        f"unique_hashes={plan['overall']['unique_exact_narrative_hashes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
