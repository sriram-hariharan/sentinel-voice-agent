"""Deterministically profile the complete local CFPB narratives archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPOSITORY_ROOT / "data/evals/v2/external/raw/cfpb_narratives"
MANIFEST_PATH = ARCHIVE_DIR / "manifest.json"
PROFILE_PATH = ARCHIVE_DIR / "profile.json"

DATE_FIELD = "Date received"
PRODUCT_FIELD = "Product"
SUB_PRODUCT_FIELD = "Sub-product"
ISSUE_FIELD = "Issue"
SUB_ISSUE_FIELD = "Sub-issue"
NARRATIVE_FIELD = "Consumer complaint narrative"
COMPANY_FIELD = "Company"
STATE_FIELD = "State"
COMPLAINT_ID_FIELD = "Complaint ID"

REQUIRED_FIELDS = {
    DATE_FIELD,
    PRODUCT_FIELD,
    SUB_PRODUCT_FIELD,
    ISSUE_FIELD,
    SUB_ISSUE_FIELD,
    NARRATIVE_FIELD,
    COMPANY_FIELD,
    STATE_FIELD,
    COMPLAINT_ID_FIELD,
}

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"
)
LONG_DIGIT_PATTERN = re.compile(r"(?<!\d)\d{9,}(?!\d)")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
PRIVACY_PATTERNS = {
    "email_like": EMAIL_PATTERN,
    "phone_like": PHONE_PATTERN,
    "long_digit_sequence_9_plus": LONG_DIGIT_PATTERN,
    "url_like": URL_PATTERN,
}

ACCOUNT_PRODUCTS = {"Bank account or service", "Checking or savings account"}
CARD_PRODUCTS = {"Credit card", "Credit card or prepaid card", "Prepaid card"}
TRANSFER_PRODUCTS = {
    "Money transfer, virtual currency, or money service",
    "Money transfers",
    "Virtual currency",
}
OTHER_BANKING_PRODUCTS = {"Other financial service"}
BANKING_PRODUCTS = ACCOUNT_PRODUCTS | CARD_PRODUCTS | TRANSFER_PRODUCTS | OTHER_BANKING_PRODUCTS


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def normalized_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise ValueError("CFPB CSV has no header")
    return [field.lstrip("\ufeff").strip() for field in fieldnames]


def normalized_date(value: str) -> str:
    parts = value.split("/")
    if len(parts) == 3:
        month, day, year = parts
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    parts = value.split("-")
    if len(parts) == 3:
        year, month, day = parts
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    raise ValueError(f"unsupported CFPB date format: {value!r}")


def distribution(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"count": count, "value": value}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 6) if denominator else 0.0


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot calculate a percentile for an empty collection")
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def relevance_groups(
    product: str, sub_product: str, issue: str, sub_issue: str
) -> set[str]:
    """Map structured CFPB taxonomy values to broad, overlapping relevance groups."""
    issue_taxonomy = f"{issue} {sub_issue}".casefold()
    groups: set[str] = set()

    if product in ACCOUNT_PRODUCTS:
        groups.add("checking_savings_accounts")
    if product in CARD_PRODUCTS or "debit card" in sub_product.casefold():
        groups.add("credit_debit_cards")
    if product in TRANSFER_PRODUCTS or "transfer" in issue_taxonomy:
        groups.add("transfers")
    if product in OTHER_BANKING_PRODUCTS:
        groups.add("other_banking_related_categories")

    if product in CARD_PRODUCTS and any(
        term in issue_taxonomy
        for term in ("cash advance", "charge", "payment", "purchase", "transaction")
    ):
        groups.add("card_transactions")
    if any(
        term in issue_taxonomy
        for term in ("fraud", "scam", "stolen", "unauthorized", "you do not recognize")
    ):
        groups.add("unauthorized_transactions_fraud_related")
    if "cash withdrawal" in issue_taxonomy or "atm" in issue_taxonomy:
        groups.add("cash_withdrawals_atm")
    if product in BANKING_PRODUCTS and any(
        term in issue_taxonomy
        for term in ("chargeback", "dispute", "investigation", "purchase shown on your statement")
    ):
        groups.add("disputes_chargebacks")
    if product in ACCOUNT_PRODUCTS and any(
        term in issue_taxonomy
        for term in ("access", "blocked", "cash deposit", "deposit", "funds", "withdraw")
    ):
        groups.add("account_access")

    return groups


def make_duplicate_database() -> tuple[Path, sqlite3.Connection]:
    with tempfile.NamedTemporaryFile(
        prefix="cfpb_profile_", suffix=".sqlite3", delete=False
    ) as temporary:
        path = Path(temporary.name)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        """
        CREATE TABLE narrative_hashes (
            digest BLOB PRIMARY KEY,
            occurrence_count INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    return path, connection


def store_narrative_hashes(
    connection: sqlite3.Connection, rows: list[tuple[bytes]]
) -> None:
    connection.executemany(
        """
        INSERT INTO narrative_hashes (digest, occurrence_count)
        VALUES (?, 1)
        ON CONFLICT(digest)
        DO UPDATE SET occurrence_count = occurrence_count + 1
        """,
        rows,
    )


def profile_archive(manifest: dict[str, Any]) -> dict[str, Any]:
    fields: list[str] | None = None
    total_rows = 0
    narrative_lengths: list[int] = []
    minimum_date: str | None = None
    maximum_date: str | None = None
    product_counts: Counter[str] = Counter()
    sub_product_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    sub_issue_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    companies: set[str] = set()
    relevance_row_counts: Counter[str] = Counter()
    relevance_narrative_counts: Counter[str] = Counter()
    relevant_issue_counts: Counter[str] = Counter()
    relevant_issue_narrative_counts: Counter[str] = Counter()
    sensitive_narrative_counts: Counter[str] = Counter()
    sensitive_match_counts: Counter[str] = Counter()
    narratives_with_any_sensitive_pattern = 0

    database_path, duplicate_database = make_duplicate_database()
    hash_batch: list[tuple[bytes]] = []
    try:
        for archive_metadata in manifest["archives"]:
            archive_path = ARCHIVE_DIR / "files" / archive_metadata["local_filename"]
            print(f"Profiling {archive_path.name}", flush=True)
            if archive_path.stat().st_size != archive_metadata["compressed_byte_size"]:
                raise ValueError(f"archive size changed: {archive_path}")
            with zipfile.ZipFile(archive_path) as archive:
                csv_members = [
                    member
                    for member in archive.infolist()
                    if not member.is_dir() and member.filename.lower().endswith(".csv")
                ]
                if len(csv_members) != 1:
                    raise ValueError(f"{archive_path} must contain exactly one CSV")
                with archive.open(csv_members[0]) as binary_member:
                    text_member = io.TextIOWrapper(
                        binary_member, encoding="utf-8-sig", newline=""
                    )
                    reader = csv.DictReader(text_member)
                    current_fields = normalized_fieldnames(reader.fieldnames)
                    if reader.fieldnames != current_fields:
                        reader.fieldnames = current_fields
                    if fields is None:
                        fields = current_fields
                        missing_required = REQUIRED_FIELDS - set(fields)
                        if missing_required:
                            raise ValueError(
                                f"CFPB CSV is missing required fields: {sorted(missing_required)}"
                            )
                    elif current_fields != fields:
                        raise ValueError(f"CFPB schema changed in {archive_path.name}")

                    partition_rows = 0
                    for row in reader:
                        partition_rows += 1
                        total_rows += 1
                        stripped = {field: (row.get(field) or "").strip() for field in fields}
                        for field, value in stripped.items():
                            if not value:
                                missing_counts[field] += 1

                        complaint_id = stripped[COMPLAINT_ID_FIELD]
                        if not complaint_id:
                            raise ValueError(
                                f"missing complaint ID at row {partition_rows} in {archive_path.name}"
                            )

                        received = stripped[DATE_FIELD]
                        if received:
                            received = normalized_date(received)
                            minimum_date = min(minimum_date, received) if minimum_date else received
                            maximum_date = max(maximum_date, received) if maximum_date else received

                        product = stripped[PRODUCT_FIELD]
                        sub_product = stripped[SUB_PRODUCT_FIELD]
                        issue = stripped[ISSUE_FIELD]
                        sub_issue = stripped[SUB_ISSUE_FIELD]
                        if product:
                            product_counts[product] += 1
                        if sub_product:
                            sub_product_counts[sub_product] += 1
                        if issue:
                            issue_counts[issue] += 1
                        if sub_issue:
                            sub_issue_counts[sub_issue] += 1
                        if stripped[STATE_FIELD]:
                            state_counts[stripped[STATE_FIELD]] += 1
                        if stripped[COMPANY_FIELD]:
                            companies.add(stripped[COMPANY_FIELD])

                        groups = relevance_groups(product, sub_product, issue, sub_issue)
                        for group in groups:
                            relevance_row_counts[group] += 1
                        if groups and issue:
                            relevant_issue_counts[issue] += 1

                        narrative = stripped[NARRATIVE_FIELD]
                        if not narrative:
                            continue
                        narrative_lengths.append(len(narrative))
                        digest = hashlib.sha256(narrative.encode("utf-8")).digest()
                        hash_batch.append((digest,))
                        if len(hash_batch) == 10_000:
                            store_narrative_hashes(duplicate_database, hash_batch)
                            hash_batch.clear()

                        for group in groups:
                            relevance_narrative_counts[group] += 1
                        if groups and issue:
                            relevant_issue_narrative_counts[issue] += 1

                        matched_any = False
                        for pattern_name, pattern in PRIVACY_PATTERNS.items():
                            matches = pattern.findall(narrative)
                            if matches:
                                matched_any = True
                                sensitive_narrative_counts[pattern_name] += 1
                                sensitive_match_counts[pattern_name] += len(matches)
                        if matched_any:
                            narratives_with_any_sensitive_pattern += 1

                    if partition_rows != archive_metadata["row_count"]:
                        raise ValueError(
                            f"row-count mismatch for {archive_path.name}: "
                            f"{partition_rows} != {archive_metadata['row_count']}"
                        )

        if hash_batch:
            store_narrative_hashes(duplicate_database, hash_batch)
        duplicate_database.commit()
        unique_narrative_count = duplicate_database.execute(
            "SELECT COUNT(*) FROM narrative_hashes"
        ).fetchone()[0]
        duplicate_narrative_groups = duplicate_database.execute(
            "SELECT COUNT(*) FROM narrative_hashes WHERE occurrence_count > 1"
        ).fetchone()[0]
        narratives_in_duplicate_groups = duplicate_database.execute(
            "SELECT COALESCE(SUM(occurrence_count), 0) "
            "FROM narrative_hashes WHERE occurrence_count > 1"
        ).fetchone()[0]
    finally:
        duplicate_database.close()
        database_path.unlink(missing_ok=True)

    if fields is None or not narrative_lengths:
        raise ValueError("CFPB archive contained no profileable data")
    manifest_summary = manifest["summary"]
    if total_rows != manifest_summary["total_rows"]:
        raise ValueError("profile row total differs from manifest")
    if len(narrative_lengths) != manifest_summary["total_nonempty_narratives"]:
        raise ValueError("profile narrative total differs from manifest")
    if maximum_date != manifest["observed_full_records_date_received_max"]:
        raise ValueError("observed maximum Date received differs from manifest")

    narrative_lengths.sort()
    narratives = len(narrative_lengths)
    relevance_groups_output = {}
    all_group_names = (
        "checking_savings_accounts",
        "credit_debit_cards",
        "card_transactions",
        "unauthorized_transactions_fraud_related",
        "cash_withdrawals_atm",
        "transfers",
        "disputes_chargebacks",
        "account_access",
        "other_banking_related_categories",
    )
    for group in all_group_names:
        relevance_groups_output[group] = {
            "rows": relevance_row_counts[group],
            "rows_with_nonempty_narrative": relevance_narrative_counts[group],
        }

    missingness = {
        field: {
            "missing_or_empty": missing_counts[field],
            "missing_or_empty_percent": percentage(missing_counts[field], total_rows),
            "present": total_rows - missing_counts[field],
        }
        for field in fields
    }

    return {
        "archive_date_metadata": {
            "declared_archive_start_date": manifest["archive_start_date"],
            "minimum_date_received_in_records": minimum_date,
            "observed_full_records_date_received_max": maximum_date,
            "official_archive_declared_narrative_coverage_end": manifest[
                "official_archive_declared_narrative_coverage_end"
            ],
            "source_metadata_qualification_note": (
                "The official archive page declares previously published narrative coverage "
                "through 2026-08-14, while the downloaded official Full Records contain Date "
                "received values through 2026-08-31. The profile retains all records and does "
                "not treat August 31 as the archive page's narrative-coverage end."
            ),
        },
        "company_count": len(companies),
        "distributions": {
            "issue": distribution(issue_counts),
            "product": distribution(product_counts),
            "sub_issue": distribution(sub_issue_counts),
            "sub_product": distribution(sub_product_counts),
        },
        "duplicate_statistics": {
            "complaint_id_duplicate_occurrences": manifest_summary[
                "duplicate_complaint_id_count"
            ],
            "complaint_ids_with_duplicates": manifest_summary[
                "complaint_ids_with_duplicates"
            ],
            "duplicate_narrative_groups_by_sha256": duplicate_narrative_groups,
            "duplicate_narrative_occurrences_beyond_first": narratives
            - unique_narrative_count,
            "narratives_in_duplicate_groups": narratives_in_duplicate_groups,
            "unique_complaint_ids": manifest_summary["unique_complaint_id_count"],
            "unique_narratives_by_sha256": unique_narrative_count,
        },
        "field_names": fields,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "missingness": missingness,
        "narrative_length_characters": {
            "maximum": narrative_lengths[-1],
            "mean": round(fmean(narrative_lengths), 3),
            "median": median(narrative_lengths),
            "minimum": narrative_lengths[0],
            "p90_nearest_rank": nearest_rank(narrative_lengths, 0.90),
            "p95_nearest_rank": nearest_rank(narrative_lengths, 0.95),
        },
        "privacy_pattern_aggregates": {
            "method": (
                "Regex heuristics over non-empty narratives; counts are aggregate signals, "
                "not confirmation of personal information. No matched strings are retained."
            ),
            "narratives_with_any_pattern": narratives_with_any_sensitive_pattern,
            "patterns": {
                name: {
                    "narratives_with_pattern": sensitive_narrative_counts[name],
                    "total_matches": sensitive_match_counts[name],
                }
                for name in PRIVACY_PATTERNS
            },
        },
        "profile_scope": (
            "Complete official CFPB Consumer Complaint Database Narratives Archive Full "
            "Records ZIP collection; acquisition/profiling only."
        ),
        "schema_version": "cfpb-narratives-profile.v1",
        "sentinelvoice_relevance": {
            "group_counts_overlap": True,
            "groups": relevance_groups_output,
            "method": (
                "Deterministic rules over CFPB Product, Sub-product, Issue, and Sub-issue "
                "values only; narrative text is not used for relevance classification."
            ),
            "relevant_issue_distribution": [
                {
                    "issue": item["value"],
                    "rows": item["count"],
                    "rows_with_nonempty_narrative": relevant_issue_narrative_counts[
                        item["value"]
                    ],
                }
                for item in distribution(relevant_issue_counts)
            ],
        },
        "source_id": manifest["source_id"],
        "state_availability": {
            "distinct_nonempty_values": len(state_counts),
            "distribution": distribution(state_counts),
            "rows_missing_or_empty": missing_counts[STATE_FIELD],
            "rows_with_state": total_rows - missing_counts[STATE_FIELD],
        },
        "summary": {
            "rows_with_nonempty_narrative": narratives,
            "rows_without_narrative": total_rows - narratives,
            "total_rows": total_rows,
            "unique_complaint_ids": manifest_summary["unique_complaint_id_count"],
        },
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("archive_partition_count") != 21 or len(manifest.get("archives", [])) != 21:
        raise ValueError("expected a complete 21-part CFPB manifest")
    profile = profile_archive(manifest)
    PROFILE_PATH.write_bytes(stable_json_bytes(profile))
    print(
        "Built CFPB profile: "
        f"rows={profile['summary']['total_rows']}, "
        f"narratives={profile['summary']['rows_with_nonempty_narrative']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
