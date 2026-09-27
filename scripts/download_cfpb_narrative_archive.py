"""Download and inventory the complete official CFPB narratives archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, BinaryIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY_ROOT / "data/evals/v2/external/raw/cfpb_narratives"
FILES_DIR = OUTPUT_DIR / "files"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SOURCE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/"
    "foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
OFFICIAL_FILE_ORIGIN = "https://files.consumerfinance.gov"
OFFICIAL_FILE_PREFIX = "/f/documents/CCDB_Export_"
EXPECTED_PARTITION_COUNT = 21
ARCHIVE_START_DATE = "2011-12-01"
OFFICIAL_ARCHIVE_DECLARED_NARRATIVE_COVERAGE_END = "2026-08-14"
USER_AGENT = "curl/8.7.1"
CHUNK_SIZE = 1024 * 1024

COMPLAINT_ID_FIELD = "Complaint ID"
NARRATIVE_FIELD = "Consumer complaint narrative"
DATE_RECEIVED_FIELD = "Date received"


@dataclass(frozen=True)
class ArchiveSource:
    label: str
    url: str
    filename: str
    expected_bytes: int
    source_last_modified: str | None


class ArchivePageParser(HTMLParser):
    """Extract official ZIP links and page modification metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.page_datetimes: list[str] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "a":
            href = attributes.get("href")
            if (
                href
                and href.startswith(OFFICIAL_FILE_ORIGIN + OFFICIAL_FILE_PREFIX)
                and href.endswith(".zip")
            ):
                self._active_href = href
                self._active_text = []
        elif tag == "time" and attributes.get("datetime"):
            self.page_datetimes.append(attributes["datetime"] or "")

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_href is not None:
            label = " ".join("".join(self._active_text).split())
            self.links.append((self._active_href, html.unescape(label)))
            self._active_href = None
            self._active_text = []


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def request(url: str, *, method: str = "GET", range_start: int | None = None):
    headers = {"User-Agent": USER_AGENT}
    if range_start is not None:
        headers["Range"] = f"bytes={range_start}-"
    return urllib.request.Request(url, headers=headers, method=method)


def read_url(url: str) -> bytes:
    with urllib.request.urlopen(request(url), timeout=60) as response:
        return response.read()


def validate_official_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "files.consumerfinance.gov"
        or not parsed.path.startswith(OFFICIAL_FILE_PREFIX)
        or not parsed.path.endswith(".zip")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"unexpected CFPB archive URL: {url}")
    return Path(parsed.path).name


def discover_archive_links(page_bytes: bytes) -> tuple[list[tuple[str, str]], str]:
    try:
        page_text = page_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CFPB archive page must be UTF-8") from exc
    parser = ArchivePageParser()
    parser.feed(page_text)

    unique_links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for url, label in parser.links:
        validate_official_url(url)
        if url not in seen:
            unique_links.append((url, label))
            seen.add(url)
    if len(unique_links) != EXPECTED_PARTITION_COUNT:
        raise ValueError(
            "official CFPB page must expose exactly "
            f"{EXPECTED_PARTITION_COUNT} Full Records ZIPs; found "
            f"{len(unique_links)}"
        )
    if len(parser.page_datetimes) < 2:
        raise ValueError("CFPB archive page last-modified datetime was not found")
    return unique_links, parser.page_datetimes[-1]


def head_metadata(url: str) -> tuple[int, str | None]:
    with urllib.request.urlopen(request(url, method="HEAD"), timeout=60) as response:
        if response.status != 200:
            raise ValueError(f"HEAD failed for {url}: HTTP {response.status}")
        content_length = response.headers.get("Content-Length")
        if content_length is None or not content_length.isdigit():
            raise ValueError(f"official ZIP has no numeric Content-Length: {url}")
        return int(content_length), response.headers.get("Last-Modified")


def discover_sources() -> tuple[list[ArchiveSource], bytes, str]:
    page_bytes = read_url(SOURCE_PAGE)
    links, page_last_modified = discover_archive_links(page_bytes)
    sources = []
    for url, label in links:
        expected_bytes, source_last_modified = head_metadata(url)
        sources.append(
            ArchiveSource(
                label=label,
                url=url,
                filename=validate_official_url(url),
                expected_bytes=expected_bytes,
                source_last_modified=source_last_modified,
            )
        )
    return sources, page_bytes, page_last_modified


def download_source(source: ArchiveSource) -> Path:
    destination = FILES_DIR / source.filename
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if destination.stat().st_size != source.expected_bytes:
            raise ValueError(
                f"existing {destination} has an unexpected size; refusing overwrite"
            )
        print(f"Verified existing {source.filename} ({source.expected_bytes} bytes)")
        return destination

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        start = partial.stat().st_size if partial.exists() else 0
        if start > source.expected_bytes:
            raise ValueError(f"partial file exceeds expected size: {partial}")
        try:
            with urllib.request.urlopen(
                request(source.url, range_start=start if start else None),
                timeout=120,
            ) as response:
                append = start > 0 and response.status == 206
                mode = "ab" if append else "wb"
                if start and not append:
                    start = 0
                with partial.open(mode) as output:
                    copy_response(response, output)
            if partial.stat().st_size != source.expected_bytes:
                raise ValueError(
                    f"downloaded size mismatch for {source.filename}: "
                    f"{partial.stat().st_size} != {source.expected_bytes}"
                )
            partial.replace(destination)
            print(f"Downloaded {source.filename} ({source.expected_bytes} bytes)")
            return destination
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"failed to download {source.url} after {attempt} attempts"
                ) from exc
            print(f"Retrying {source.filename} after attempt {attempt}: {exc}")
            time.sleep(2**attempt)
    raise AssertionError("download retry loop exited unexpectedly")


def copy_response(response: BinaryIO, output: BinaryIO) -> None:
    while True:
        chunk = response.read(CHUNK_SIZE)
        if not chunk:
            break
        output.write(chunk)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise ValueError("CFPB CSV has no header")
    return [field.lstrip("\ufeff").strip() for field in fieldnames]


def scan_csv_member(
    member: BinaryIO,
    complaint_ids: sqlite3.Connection,
) -> dict[str, Any]:
    text = io.TextIOWrapper(member, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text)
    fieldnames = normalize_fieldnames(reader.fieldnames)
    if reader.fieldnames != fieldnames:
        reader.fieldnames = fieldnames
    required = {COMPLAINT_ID_FIELD, NARRATIVE_FIELD, DATE_RECEIVED_FIELD}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"CFPB CSV is missing required fields: {sorted(missing)}")

    row_count = 0
    narrative_count = 0
    minimum_date: str | None = None
    maximum_date: str | None = None
    id_batch: list[tuple[str]] = []
    for row in reader:
        row_count += 1
        complaint_id = (row.get(COMPLAINT_ID_FIELD) or "").strip()
        if not complaint_id:
            raise ValueError(f"CFPB row {row_count} has no complaint ID")
        id_batch.append((complaint_id,))
        if len(id_batch) == 10_000:
            upsert_complaint_ids(complaint_ids, id_batch)
            id_batch.clear()

        if (row.get(NARRATIVE_FIELD) or "").strip():
            narrative_count += 1
        received = (row.get(DATE_RECEIVED_FIELD) or "").strip()
        if received:
            normalized = normalize_date(received)
            minimum_date = min(minimum_date, normalized) if minimum_date else normalized
            maximum_date = max(maximum_date, normalized) if maximum_date else normalized
    if id_batch:
        upsert_complaint_ids(complaint_ids, id_batch)
    return {
        "field_names": fieldnames,
        "row_count": row_count,
        "rows_with_nonempty_consumer_narrative": narrative_count,
        "source_date_range": {
            "minimum_date_received": minimum_date,
            "maximum_date_received": maximum_date,
        },
    }


def normalize_date(value: str) -> str:
    for date_format in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(value, date_format).replace(tzinfo=UTC)
            return parsed.date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unsupported CFPB date format: {value!r}")


def upsert_complaint_ids(
    connection: sqlite3.Connection, rows: list[tuple[str]]
) -> None:
    connection.executemany(
        """
        INSERT INTO complaint_ids (complaint_id, occurrence_count)
        VALUES (?, 1)
        ON CONFLICT(complaint_id)
        DO UPDATE SET occurrence_count = occurrence_count + 1
        """,
        rows,
    )


def scan_archive(
    source: ArchiveSource,
    path: Path,
    complaint_ids: sqlite3.Connection,
) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt ZIP member in {path}")
        members = [member for member in archive.infolist() if not member.is_dir()]
        csv_members = [
            member for member in members if member.filename.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(f"{path} must contain exactly one CSV member")
        with archive.open(csv_members[0]) as member:
            csv_summary = scan_csv_member(member, complaint_ids)
        return {
            "compressed_byte_size": path.stat().st_size,
            "extracted_byte_size": sum(member.file_size for member in members),
            "extracted_members": [
                {
                    "compressed_byte_size": member.compress_size,
                    "extracted_byte_size": member.file_size,
                    "name": member.filename,
                }
                for member in members
            ],
            "field_names": csv_summary["field_names"],
            "local_filename": path.name,
            "official_archive_label": source.label,
            "official_file_url": source.url,
            "official_last_modified": source.source_last_modified,
            "row_count": csv_summary["row_count"],
            "rows_with_nonempty_consumer_narrative": csv_summary[
                "rows_with_nonempty_consumer_narrative"
            ],
            "sha256": sha256_path(path),
            "source_date_range": csv_summary["source_date_range"],
        }


def build_manifest(
    sources: list[ArchiveSource],
    page_bytes: bytes,
    page_last_modified: str,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        prefix="cfpb_manifest_", suffix=".sqlite3", delete=False
    ) as temporary:
        database_path = Path(temporary.name)
    try:
        with sqlite3.connect(database_path) as complaint_ids:
            complaint_ids.execute("PRAGMA journal_mode=OFF")
            complaint_ids.execute("PRAGMA synchronous=OFF")
            complaint_ids.execute(
                """
                CREATE TABLE complaint_ids (
                    complaint_id TEXT PRIMARY KEY,
                    occurrence_count INTEGER NOT NULL
                )
                """
            )
            archives = []
            for source in sources:
                print(f"Inventorying {source.filename}")
                archives.append(
                    scan_archive(source, FILES_DIR / source.filename, complaint_ids)
                )
            complaint_ids.commit()
            unique_count = complaint_ids.execute(
                "SELECT COUNT(*) FROM complaint_ids"
            ).fetchone()[0]
            duplicate_count = complaint_ids.execute(
                "SELECT COALESCE(SUM(occurrence_count - 1), 0) FROM complaint_ids"
            ).fetchone()[0]
            duplicated_id_count = complaint_ids.execute(
                "SELECT COUNT(*) FROM complaint_ids WHERE occurrence_count > 1"
            ).fetchone()[0]
    finally:
        database_path.unlink(missing_ok=True)

    return {
        "archive_partition_count": len(archives),
        "archive_start_date": ARCHIVE_START_DATE,
        "archives": archives,
        "license_and_use_notes": (
            "Previously published narratives are public CFPB records; use and "
            "redistribution terms remain subject to the documented V2-C2K review."
        ),
        "privacy_and_provenance_notes": [
            "This is real consumer-authored complaint data prepared by CFPB for public release, not synthetic application data.",
            "Narratives were subject to publication consent and personal-information scrubbing, but residual privacy risk is not assumed to be zero.",
            "Complaint narratives are unverified, one-sided allegations and are not a representative sample.",
            "Complaint counts must not be used to rank or judge financial institutions.",
            "The local ZIP payloads are immutable, Git-ignored source files for offline evaluation only.",
        ],
        "publisher": "Consumer Financial Protection Bureau",
        "retrieval_date": datetime.now(tz=UTC).date().isoformat(),
        "official_archive_declared_narrative_coverage_end": (
            OFFICIAL_ARCHIVE_DECLARED_NARRATIVE_COVERAGE_END
        ),
        "observed_full_records_date_received_max": max(
            archive["source_date_range"]["maximum_date_received"]
            for archive in archives
        ),
        "schema_version": "cfpb-narratives-raw-manifest.v1",
        "source_id": "cfpb_consumer_complaint_narratives_archive",
        "source_page": SOURCE_PAGE,
        "source_page_last_modified": page_last_modified,
        "source_page_sha256": hashlib.sha256(page_bytes).hexdigest(),
        "source_metadata_qualification_note": (
            "The official archive page declares previously published narrative coverage "
            "through 2026-08-14, while the downloaded official Full Records contain Date "
            "received values through 2026-08-31. No records are filtered or discarded."
        ),
        "source_type": "real consumer-authored complaint data",
        "summary": {
            "complaint_ids_with_duplicates": duplicated_id_count,
            "duplicate_complaint_id_count": duplicate_count,
            "total_compressed_bytes": sum(
                archive["compressed_byte_size"] for archive in archives
            ),
            "total_extracted_bytes": sum(
                archive["extracted_byte_size"] for archive in archives
            ),
            "total_nonempty_narratives": sum(
                archive["rows_with_nonempty_consumer_narrative"]
                for archive in archives
            ),
            "total_rows": sum(archive["row_count"] for archive in archives),
            "unique_complaint_id_count": unique_count,
        },
    }


def print_plan(sources: list[ArchiveSource], page_last_modified: str) -> None:
    for source in sources:
        print(f"{source.expected_bytes:>12}  {source.filename}")
    total = sum(source.expected_bytes for source in sources)
    print(f"Official partitions: {len(sources)}")
    print(f"Total compressed bytes: {total}")
    print(f"Total compressed GiB: {total / (1024**3):.3f}")
    print(f"Archive page last modified: {page_last_modified}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover official partitions and sizes without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources, page_bytes, page_last_modified = discover_sources()
    print_plan(sources, page_last_modified)
    if args.dry_run:
        return

    for source in sources:
        download_source(source)
    manifest = build_manifest(sources, page_bytes, page_last_modified)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_bytes(stable_json_bytes(manifest))
    summary = manifest["summary"]
    print(
        "Built CFPB raw manifest: "
        f"rows={summary['total_rows']}, "
        f"narratives={summary['total_nonempty_narratives']}, "
        f"unique_complaint_ids={summary['unique_complaint_id_count']}"
    )


if __name__ == "__main__":
    main()
