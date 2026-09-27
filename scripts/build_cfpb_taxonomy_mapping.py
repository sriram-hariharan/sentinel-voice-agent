"""Inventory CFPB narrative taxonomy and freeze conservative intent mappings."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
RAW_DIR = EXTERNAL_ROOT / "raw/cfpb_narratives"
RAW_FILES_DIR = RAW_DIR / "files"
RAW_MANIFEST_PATH = RAW_DIR / "manifest.json"
RAW_PROFILE_PATH = RAW_DIR / "profile.json"
MAPPINGS_DIR = EXTERNAL_ROOT / "mappings"
INVENTORY_PATH = MAPPINGS_DIR / "cfpb_taxonomy_inventory.json"
MAPPING_PATH = MAPPINGS_DIR / "cfpb_to_sentinel_intents.json"

DATE_FIELD = "Date received"
PRODUCT_FIELD = "Product"
SUB_PRODUCT_FIELD = "Sub-product"
ISSUE_FIELD = "Issue"
SUB_ISSUE_FIELD = "Sub-issue"
NARRATIVE_FIELD = "Consumer complaint narrative"
COMPLAINT_ID_FIELD = "Complaint ID"
REQUIRED_FIELDS = {
    DATE_FIELD,
    PRODUCT_FIELD,
    SUB_PRODUCT_FIELD,
    ISSUE_FIELD,
    SUB_ISSUE_FIELD,
    NARRATIVE_FIELD,
    COMPLAINT_ID_FIELD,
}

ALLOWED_STATUSES = ("EXACT_MATCH", "NEAR_MATCH", "AMBIGUOUS", "UNSUPPORTED")
SENTINELVOICE_INTENTS = (
    "informational_policy",
    "account_balance",
    "recent_transactions",
    "transaction_details",
    "card_status",
    "freeze_card",
    "create_dispute",
    "escalation",
    "unsupported_or_uncertain",
)
RISK_BY_INTENT = {
    "informational_policy": "PUBLIC",
    "account_balance": "PRIVATE_READ",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "card_status": "PRIVATE_READ",
    "freeze_card": "PROTECTED_WRITE",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
PROTECTED_WRITE_INTENTS = frozenset({"freeze_card", "create_dispute"})
PROTECTED_EXACT_RULE_ALLOWLIST: frozenset[str] = frozenset()
PROTECTED_ACTION_RISKS = (
    "NONE",
    "POTENTIAL_PROTECTED_WRITE",
    "EXPLICIT_UNSUPPORTED_WRITE",
)

UNSUPPORTED_PRODUCTS = (
    "Consumer Loan",
    "Credit reporting",
    "Credit reporting or other personal consumer reports",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Money transfers",
    "Mortgage",
    "Other financial service",
    "Payday loan",
    "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan",
    "Student loan",
    "Vehicle loan or lease",
    "Virtual currency",
)
ACCOUNT_PRODUCTS = ("Bank account or service", "Checking or savings account")
CARD_PRODUCTS = ("Credit card", "Credit card or prepaid card", "Prepaid card")


@dataclass(frozen=True, order=True)
class TaxonomyKey:
    product: str
    sub_product: str | None
    issue: str
    sub_issue: str | None


@dataclass
class Counts:
    row_count: int = 0
    narrative_count: int = 0
    complaint_ids: set[int] = field(default_factory=set)
    earliest_date_received: str | None = None
    latest_date_received: str | None = None

    def add(
        self,
        *,
        complaint_id: int,
        date_received: str,
        row_count: int = 1,
        narrative_count: int = 1,
    ) -> None:
        self.row_count += row_count
        self.narrative_count += narrative_count
        self.complaint_ids.add(complaint_id)
        self.earliest_date_received = (
            min(self.earliest_date_received, date_received)
            if self.earliest_date_received
            else date_received
        )
        self.latest_date_received = (
            max(self.latest_date_received, date_received)
            if self.latest_date_received
            else date_received
        )

    def merge(self, other: Counts) -> None:
        self.row_count += other.row_count
        self.narrative_count += other.narrative_count
        self.complaint_ids.update(other.complaint_ids)
        if other.earliest_date_received:
            self.earliest_date_received = (
                min(self.earliest_date_received, other.earliest_date_received)
                if self.earliest_date_received
                else other.earliest_date_received
            )
        if other.latest_date_received:
            self.latest_date_received = (
                max(self.latest_date_received, other.latest_date_received)
                if self.latest_date_received
                else other.latest_date_received
            )


@dataclass(frozen=True)
class MappingRule:
    rule_id: str
    mapping_status: str
    candidate_intents: tuple[str, ...]
    rationale: str
    protected_action_risk: str = "NONE"
    product: str | None = None
    sub_product: str | None = None
    issue: str | None = None
    sub_issue: str | None = None

    @property
    def specificity(self) -> int:
        values = (self.product, self.sub_product, self.issue, self.sub_issue)
        return sum(value is not None for value in values)

    def matches(self, key: TaxonomyKey) -> bool:
        return all(
            expected is None or expected == actual
            for expected, actual in (
                (self.product, key.product),
                (self.sub_product, key.sub_product),
                (self.issue, key.issue),
                (self.sub_issue, key.sub_issue),
            )
        )


@dataclass(frozen=True)
class ScanResult:
    full_counts: dict[TaxonomyKey, Counts]
    archive_filenames: tuple[str, ...]
    total_csv_rows: int


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise ValueError("CFPB CSV has no header")
    return [field_name.lstrip("\ufeff").strip() for field_name in fieldnames]


def normalize_date(value: str) -> str:
    separator = "/" if "/" in value else "-"
    parts = value.split(separator)
    if len(parts) != 3:
        raise ValueError(f"unsupported CFPB date format: {value!r}")
    if separator == "/":
        month, day, year = parts
    else:
        year, month, day = parts
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def scan_archives(archive_paths: Iterable[Path]) -> ScanResult:
    full_counts: dict[TaxonomyKey, Counts] = {}
    filenames: list[str] = []
    total_csv_rows = 0
    for archive_path in archive_paths:
        filenames.append(archive_path.name)
        print(f"Inventorying {archive_path.name}", flush=True)
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
                    narrative = (row.get(NARRATIVE_FIELD) or "").strip()
                    if not narrative:
                        continue
                    product = (row.get(PRODUCT_FIELD) or "").strip()
                    issue = (row.get(ISSUE_FIELD) or "").strip()
                    if not product or not issue:
                        raise ValueError(
                            f"narrative row {row_number} in {archive_path.name} "
                            "has no product or issue"
                        )
                    complaint_id_text = (row.get(COMPLAINT_ID_FIELD) or "").strip()
                    if not complaint_id_text.isdigit():
                        raise ValueError(
                            f"invalid complaint ID at row {row_number} in "
                            f"{archive_path.name}"
                        )
                    date_received = normalize_date(
                        (row.get(DATE_FIELD) or "").strip()
                    )
                    key = TaxonomyKey(
                        product=product,
                        sub_product=(row.get(SUB_PRODUCT_FIELD) or "").strip() or None,
                        issue=issue,
                        sub_issue=(row.get(SUB_ISSUE_FIELD) or "").strip() or None,
                    )
                    full_counts.setdefault(key, Counts()).add(
                        complaint_id=int(complaint_id_text),
                        date_received=date_received,
                    )
    return ScanResult(
        full_counts=full_counts,
        archive_filenames=tuple(filenames),
        total_csv_rows=total_csv_rows,
    )


def marginalize(
    full_counts: dict[TaxonomyKey, Counts],
    dimensions: tuple[str, ...],
) -> dict[tuple[str | None, ...], Counts]:
    result: dict[tuple[str | None, ...], Counts] = {}
    for key, counts in full_counts.items():
        marginal_key = tuple(getattr(key, dimension) for dimension in dimensions)
        result.setdefault(marginal_key, Counts()).merge(counts)
    return result


def count_fields(counts: Counts) -> dict[str, Any]:
    return {
        "distinct_complaint_ids": len(counts.complaint_ids),
        "earliest_date_received": counts.earliest_date_received,
        "latest_date_received": counts.latest_date_received,
        "narrative_count": counts.narrative_count,
        "row_count": counts.row_count,
    }


def sorted_count_entries(
    counts_by_key: dict[tuple[str | None, ...], Counts],
    dimensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    def sort_key(item: tuple[tuple[str | None, ...], Counts]) -> tuple[str, ...]:
        return tuple(value or "" for value in item[0])

    return [
        {
            **dict(zip(dimensions, key, strict=True)),
            **count_fields(counts),
        }
        for key, counts in sorted(counts_by_key.items(), key=sort_key)
    ]


def build_inventory(
    scan: ScanResult,
    *,
    manifest_sha256: str,
    profile_sha256: str,
) -> dict[str, Any]:
    full_dimensions = ("product", "sub_product", "issue", "sub_issue")
    full_as_tuples = {
        (key.product, key.sub_product, key.issue, key.sub_issue): counts
        for key, counts in scan.full_counts.items()
    }
    products = marginalize(scan.full_counts, ("product",))
    product_issues = marginalize(scan.full_counts, ("product", "issue"))
    product_sub_product_issues = marginalize(
        scan.full_counts, ("product", "sub_product", "issue")
    )
    total_narratives = sum(counts.narrative_count for counts in scan.full_counts.values())
    all_complaint_ids: set[int] = set()
    for counts in scan.full_counts.values():
        all_complaint_ids.update(counts.complaint_ids)

    return {
        "archive_partition_count": len(scan.archive_filenames),
        "builder_version": "cfpb-taxonomy-mapping-builder.v1",
        "input_integrity": {
            "raw_manifest_sha256": manifest_sha256,
            "raw_profile_sha256": profile_sha256,
        },
        "marginal_counts": {
            "full_product_sub_product_issue_sub_issue": sorted_count_entries(
                full_as_tuples, full_dimensions
            ),
            "product": sorted_count_entries(products, ("product",)),
            "product_issue": sorted_count_entries(
                product_issues, ("product", "issue")
            ),
            "product_sub_product_issue": sorted_count_entries(
                product_sub_product_issues,
                ("product", "sub_product", "issue"),
            ),
        },
        "qualification_phase": "V2-C2M",
        "schema_version": "cfpb-taxonomy-inventory.v1",
        "scope": (
            "Complete observed CFPB Product/Sub-product/Issue/Sub-issue taxonomy "
            "for rows with a non-empty Consumer complaint narrative."
        ),
        "source_archive_filenames": sorted(scan.archive_filenames),
        "source_id": "cfpb_consumer_complaint_narratives_archive",
        "summary": {
            "distinct_complaint_ids": len(all_complaint_ids),
            "narrative_bearing_rows": total_narratives,
            "narrative_count": total_narratives,
            "total_full_records_rows_scanned": scan.total_csv_rows,
            "unique_full_taxonomy_tuples": len(scan.full_counts),
            "unique_issues": len({key.issue for key in scan.full_counts}),
            "unique_products": len({key.product for key in scan.full_counts}),
            "unique_sub_issues": len(
                {key.sub_issue for key in scan.full_counts if key.sub_issue is not None}
            ),
            "unique_sub_products": len(
                {
                    key.sub_product
                    for key in scan.full_counts
                    if key.sub_product is not None
                }
            ),
        },
    }


def rule(
    rule_id: str,
    status: str,
    candidates: tuple[str, ...],
    rationale: str,
    *,
    product: str | None = None,
    issue: str | None = None,
    protected_action_risk: str = "NONE",
) -> MappingRule:
    return MappingRule(
        rule_id=rule_id,
        mapping_status=status,
        candidate_intents=candidates,
        rationale=rationale,
        protected_action_risk=protected_action_risk,
        product=product,
        issue=issue,
    )


def frozen_rules() -> tuple[MappingRule, ...]:
    rules: list[MappingRule] = []
    for index, product in enumerate(UNSUPPORTED_PRODUCTS, start=1):
        rules.append(
            rule(
                f"unsupported_product_{index:02d}",
                "UNSUPPORTED",
                ("unsupported_or_uncertain",),
                "The product is outside SentinelVoice's supported account/card scope.",
                product=product,
            )
        )

    account_near = {
        "Deposits and withdrawals": ("recent_transactions", "transaction_details"),
        "Problem caused by your funds being low": ("account_balance",),
        "Problems caused by my funds being low": ("account_balance",),
    }
    account_ambiguous = {
        "Account opening, closing, or management": (
            "account_balance",
            "recent_transactions",
            "escalation",
        ),
        "Fees or interest": ("informational_policy", "transaction_details"),
        "Managing an account": (
            "account_balance",
            "recent_transactions",
            "transaction_details",
            "escalation",
        ),
        "Problem with a lender or other company charging your account": (
            "transaction_details",
            "create_dispute",
        ),
        "Unauthorized withdrawals or charges": (
            "transaction_details",
            "create_dispute",
        ),
        "Using a debit or ATM card": ("card_status", "transaction_details"),
    }
    account_unsupported = {
        "Closing an account",
        "Closing your account",
        "Getting a line of credit",
        "Opening an account",
    }
    for product in ACCOUNT_PRODUCTS:
        slug = "legacy_account" if product == "Bank account or service" else "account"
        for index, (issue, candidates) in enumerate(account_near.items(), start=1):
            rules.append(
                rule(
                    f"{slug}_near_{index:02d}",
                    "NEAR_MATCH",
                    candidates,
                    "The structured issue overlaps a supported private read, but a "
                    "complaint topic does not establish the user's exact read request.",
                    product=product,
                    issue=issue,
                )
            )
        for index, (issue, candidates) in enumerate(account_ambiguous.items(), start=1):
            rules.append(
                rule(
                    f"{slug}_ambiguous_{index:02d}",
                    "AMBIGUOUS",
                    candidates,
                    "The complaint issue can describe multiple read, remedy, or escalation "
                    "goals and requires narrative-level semantic review.",
                    product=product,
                    issue=issue,
                    protected_action_risk=(
                        "POTENTIAL_PROTECTED_WRITE"
                        if "create_dispute" in candidates
                        else "NONE"
                    ),
                )
            )
        for index, issue in enumerate(sorted(account_unsupported), start=1):
            rules.append(
                rule(
                    f"{slug}_unsupported_{index:02d}",
                    "UNSUPPORTED",
                    ("unsupported_or_uncertain",),
                    "The issue concerns an unsupported account lifecycle or credit action.",
                    product=product,
                    issue=issue,
                    protected_action_risk="EXPLICIT_UNSUPPORTED_WRITE",
                )
            )
        rules.append(
            rule(
                f"{slug}_default_ambiguous",
                "AMBIGUOUS",
                (
                    "informational_policy",
                    "account_balance",
                    "recent_transactions",
                    "transaction_details",
                    "card_status",
                    "escalation",
                ),
                "An account-product complaint alone does not identify one conversational "
                "intent or distinguish supported reads from unsupported remedies.",
                product=product,
            )
        )

    card_near = {
        "APR or interest rate": ("informational_policy",),
        "Billing statement": ("recent_transactions",),
        "Cash advance": ("transaction_details",),
        "Other transaction issues": ("transaction_details",),
        "Problem with cash advance": ("transaction_details",),
        "Transaction issue": ("transaction_details",),
        "Trouble using the card": ("card_status",),
        "Trouble using your card": ("card_status",),
        "Wrong amount charged or received": ("transaction_details",),
    }
    card_ambiguous = {
        "Billing disputes": ("transaction_details", "create_dispute"),
        "Charged fees or interest I didn't expect": (
            "informational_policy",
            "transaction_details",
            "create_dispute",
        ),
        "Charged fees or interest you didn't expect": (
            "informational_policy",
            "transaction_details",
            "create_dispute",
        ),
        "Customer service / Customer relations": ("escalation",),
        "Fees or interest": (
            "informational_policy",
            "transaction_details",
            "create_dispute",
        ),
        "Fraud or scam": (
            "card_status",
            "freeze_card",
            "transaction_details",
            "create_dispute",
            "escalation",
        ),
        "Identity theft / Fraud / Embezzlement": (
            "card_status",
            "freeze_card",
            "create_dispute",
            "escalation",
        ),
        "Other transaction problem": ("transaction_details", "create_dispute"),
        "Problem with a purchase shown on your statement": (
            "transaction_details",
            "create_dispute",
        ),
        "Unauthorized transactions or other transaction problem": (
            "transaction_details",
            "create_dispute",
        ),
        "Unauthorized transactions/trans. issues": (
            "transaction_details",
            "create_dispute",
        ),
    }
    card_unsupported = {
        "Closing/Cancelling account",
        "Credit line increase/decrease",
        "Getting a credit card",
        "Problem getting a card or closing an account",
        "Rewards",
        "Unsolicited issuance of credit card",
    }
    for product_index, product in enumerate(CARD_PRODUCTS, start=1):
        slug = f"card_product_{product_index:02d}"
        for index, (issue, candidates) in enumerate(card_near.items(), start=1):
            rules.append(
                rule(
                    f"{slug}_near_{index:02d}",
                    "NEAR_MATCH",
                    candidates,
                    "The issue overlaps a supported informational or private-read "
                    "capability, but complaint framing can include unsupported remediation.",
                    product=product,
                    issue=issue,
                )
            )
        for index, (issue, candidates) in enumerate(card_ambiguous.items(), start=1):
            rules.append(
                rule(
                    f"{slug}_ambiguous_{index:02d}",
                    "AMBIGUOUS",
                    candidates,
                    "The issue does not distinguish information lookup, protected action, "
                    "or escalation semantics.",
                    product=product,
                    issue=issue,
                    protected_action_risk=(
                        "POTENTIAL_PROTECTED_WRITE"
                        if PROTECTED_WRITE_INTENTS.intersection(candidates)
                        else "NONE"
                    ),
                )
            )
        for index, issue in enumerate(sorted(card_unsupported), start=1):
            rules.append(
                rule(
                    f"{slug}_unsupported_{index:02d}",
                    "UNSUPPORTED",
                    ("unsupported_or_uncertain",),
                    "The issue concerns unsupported card issuance, account lifecycle, "
                    "credit-line, or rewards behavior.",
                    product=product,
                    issue=issue,
                    protected_action_risk="EXPLICIT_UNSUPPORTED_WRITE",
                )
            )
        rules.append(
            rule(
                f"{slug}_default_ambiguous",
                "AMBIGUOUS",
                (
                    "informational_policy",
                    "transaction_details",
                    "card_status",
                    "freeze_card",
                    "create_dispute",
                    "escalation",
                ),
                "A card-product complaint alone does not establish a supported read, "
                "protected action, or escalation request.",
                product=product,
                protected_action_risk="POTENTIAL_PROTECTED_WRITE",
            )
        )

    rules.append(
        rule(
            "explicit_default_unsupported",
            "UNSUPPORTED",
            ("unsupported_or_uncertain",),
            "An unrecognized product has no qualified SentinelVoice mapping.",
        )
    )
    return tuple(rules)


def validate_rules(rules: tuple[MappingRule, ...]) -> None:
    if len({mapping_rule.rule_id for mapping_rule in rules}) != len(rules):
        raise ValueError("mapping rule IDs must be unique")
    defaults = [mapping_rule for mapping_rule in rules if mapping_rule.specificity == 0]
    if len(defaults) != 1 or defaults[0].mapping_status != "UNSUPPORTED":
        raise ValueError("mapping rules require one explicit UNSUPPORTED default")

    for mapping_rule in rules:
        if mapping_rule.mapping_status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status in {mapping_rule.rule_id}")
        if mapping_rule.protected_action_risk not in PROTECTED_ACTION_RISKS:
            raise ValueError(f"invalid protected-action risk in {mapping_rule.rule_id}")
        if not mapping_rule.candidate_intents:
            raise ValueError(f"{mapping_rule.rule_id} needs candidate intents")
        if not set(mapping_rule.candidate_intents).issubset(SENTINELVOICE_INTENTS):
            raise ValueError(f"invalid intent in {mapping_rule.rule_id}")
        if mapping_rule.mapping_status == "UNSUPPORTED" and (
            mapping_rule.candidate_intents != ("unsupported_or_uncertain",)
        ):
            raise ValueError("UNSUPPORTED rules must map to unsupported_or_uncertain")
        if (
            mapping_rule.mapping_status == "EXACT_MATCH"
            and PROTECTED_WRITE_INTENTS.intersection(mapping_rule.candidate_intents)
            and mapping_rule.rule_id not in PROTECTED_EXACT_RULE_ALLOWLIST
        ):
            raise ValueError(
                "protected-write EXACT_MATCH requires an explicit rule allowlist entry"
            )
        shape = (
            mapping_rule.product is not None,
            mapping_rule.sub_product is not None,
            mapping_rule.issue is not None,
            mapping_rule.sub_issue is not None,
        )
        allowed_shapes = {
            (False, False, False, False),
            (True, False, False, False),
            (True, False, True, False),
            (True, True, True, False),
            (True, True, True, True),
        }
        if shape not in allowed_shapes:
            raise ValueError(f"invalid selector shape in {mapping_rule.rule_id}")

    for index, left in enumerate(rules):
        for right in rules[index + 1 :]:
            if left.specificity != right.specificity:
                continue
            selectors = (
                (left.product, right.product),
                (left.sub_product, right.sub_product),
                (left.issue, right.issue),
                (left.sub_issue, right.sub_issue),
            )
            overlap = all(
                left_value is None
                or right_value is None
                or left_value == right_value
                for left_value, right_value in selectors
            )
            if overlap:
                raise ValueError(
                    "same-precedence rules overlap: "
                    f"{left.rule_id}, {right.rule_id}"
                )


def select_rule(key: TaxonomyKey, rules: tuple[MappingRule, ...]) -> MappingRule:
    matches = [mapping_rule for mapping_rule in rules if mapping_rule.matches(key)]
    if not matches:
        raise ValueError(f"no mapping rule matched {key}")
    highest_specificity = max(mapping_rule.specificity for mapping_rule in matches)
    winners = [
        mapping_rule
        for mapping_rule in matches
        if mapping_rule.specificity == highest_specificity
    ]
    if len(winners) != 1:
        raise ValueError(f"conflicting mapping rules for {key}")
    return winners[0]


def selector_payload(mapping_rule: MappingRule) -> dict[str, str]:
    return {
        field_name: value
        for field_name, value in (
            ("product", mapping_rule.product),
            ("sub_product", mapping_rule.sub_product),
            ("issue", mapping_rule.issue),
            ("sub_issue", mapping_rule.sub_issue),
        )
        if value is not None
    }


def build_mapping(inventory: dict[str, Any]) -> dict[str, Any]:
    rules = frozen_rules()
    validate_rules(rules)
    combinations = inventory["marginal_counts"][
        "full_product_sub_product_issue_sub_issue"
    ]
    assignments: list[dict[str, Any]] = []
    rule_tuple_counts: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = {
        status: Counter() for status in ALLOWED_STATUSES
    }
    status_product_counts: dict[str, Counter[str]] = {
        status: Counter() for status in ALLOWED_STATUSES
    }
    candidate_counts: dict[str, Counter[str]] = {
        status: Counter() for status in ALLOWED_STATUSES
    }
    exact_by_intent: dict[str, Counter[str]] = {}

    for combination in combinations:
        key = TaxonomyKey(
            product=combination["product"],
            sub_product=combination["sub_product"],
            issue=combination["issue"],
            sub_issue=combination["sub_issue"],
        )
        matched = select_rule(key, rules)
        rule_tuple_counts[matched.rule_id] += 1
        for metric in ("row_count", "narrative_count", "distinct_complaint_ids"):
            status_counts[matched.mapping_status][metric] += combination[metric]
        status_product_counts[matched.mapping_status][key.product] += combination[
            "narrative_count"
        ]
        for candidate in matched.candidate_intents:
            candidate_counts[matched.mapping_status][candidate] += combination[
                "narrative_count"
            ]
        if matched.mapping_status == "EXACT_MATCH":
            intent = matched.candidate_intents[0]
            exact_by_intent.setdefault(intent, Counter())
            for metric in ("row_count", "narrative_count", "distinct_complaint_ids"):
                exact_by_intent[intent][metric] += combination[metric]

        assignments.append(
            {
                "candidate_risk_groups": sorted(
                    {RISK_BY_INTENT[intent] for intent in matched.candidate_intents}
                ),
                "candidate_sentinelvoice_intents": list(matched.candidate_intents),
                "distinct_complaint_ids": combination["distinct_complaint_ids"],
                "earliest_date_received": combination["earliest_date_received"],
                "issue": key.issue,
                "latest_date_received": combination["latest_date_received"],
                "mapping_status": matched.mapping_status,
                "matched_rule_id": matched.rule_id,
                "narrative_count": combination["narrative_count"],
                "product": key.product,
                "protected_action_risk": matched.protected_action_risk,
                "requires_narrative_semantic_review": matched.mapping_status
                in {"NEAR_MATCH", "AMBIGUOUS"},
                "row_count": combination["row_count"],
                "sentinelvoice_intent": (
                    matched.candidate_intents[0]
                    if len(matched.candidate_intents) == 1
                    else None
                ),
                "sub_issue": key.sub_issue,
                "sub_product": key.sub_product,
            }
        )

    total_narratives = inventory["summary"]["narrative_count"]
    if sum(counts["narrative_count"] for counts in status_counts.values()) != (
        total_narratives
    ):
        raise ValueError("mapping status counts do not reconcile to inventory")
    protected_exact = sum(
        assignment["narrative_count"]
        for assignment in assignments
        if assignment["mapping_status"] == "EXACT_MATCH"
        and PROTECTED_WRITE_INTENTS.intersection(
            assignment["candidate_sentinelvoice_intents"]
        )
    )
    if protected_exact:
        raise ValueError("protected-write EXACT_MATCH count must remain zero")

    active_rules = [
        mapping_rule
        for mapping_rule in rules
        if rule_tuple_counts[mapping_rule.rule_id] > 0
    ]
    rule_payloads = [
        {
            "candidate_risk_groups": sorted(
                {
                    RISK_BY_INTENT[intent]
                    for intent in mapping_rule.candidate_intents
                }
            ),
            "candidate_sentinelvoice_intents": list(mapping_rule.candidate_intents),
            "mapping_status": mapping_rule.mapping_status,
            "matched_observed_taxonomy_tuples": rule_tuple_counts[
                mapping_rule.rule_id
            ],
            "precedence": mapping_rule.specificity,
            "protected_action_risk": mapping_rule.protected_action_risk,
            "rationale": mapping_rule.rationale,
            "requires_narrative_semantic_review": mapping_rule.mapping_status
            in {"NEAR_MATCH", "AMBIGUOUS"},
            "rule_id": mapping_rule.rule_id,
            "selector": selector_payload(mapping_rule),
            "sentinelvoice_intent": (
                mapping_rule.candidate_intents[0]
                if len(mapping_rule.candidate_intents) == 1
                else None
            ),
        }
        for mapping_rule in active_rules
    ]

    coverage_by_status = {
        status: {
            "candidate_intent_narrative_counts": dict(
                sorted(candidate_counts[status].items())
            ),
            "distinct_complaint_ids": status_counts[status][
                "distinct_complaint_ids"
            ],
            "major_products_by_narrative_count": [
                {"narrative_count": count, "product": product}
                for product, count in status_product_counts[status].most_common()
            ],
            "narrative_count": status_counts[status]["narrative_count"],
            "row_count": status_counts[status]["row_count"],
        }
        for status in ALLOWED_STATUSES
    }

    return {
        "allowed_mapping_statuses": list(ALLOWED_STATUSES),
        "artifact_boundary": {
            "classifier_predictions_run": False,
            "final_evaluation_sample_created": False,
            "individual_narratives_labeled": False,
            "model_training_or_retraining_run": False,
        },
        "coverage": {
            "by_status": coverage_by_status,
            "exact_match_by_expected_sentinelvoice_intent": {
                intent: dict(counts)
                for intent, counts in sorted(exact_by_intent.items())
            },
            "protected_write_exact_match": {
                "narrative_count": protected_exact,
                "rule_count": sum(
                    mapping_rule.mapping_status == "EXACT_MATCH"
                    and bool(
                        PROTECTED_WRITE_INTENTS.intersection(
                            mapping_rule.candidate_intents
                        )
                    )
                    for mapping_rule in active_rules
                ),
            },
            "total_mapped_taxonomy_tuples": len(assignments),
            "total_narrative_bearing_records": total_narratives,
        },
        "inventory_sha256": sha256_bytes(stable_json_bytes(inventory)),
        "long_form_qualification": {
            "future_lanes": [
                "taxonomy-derived scored examples",
                "narrative-semantic review examples",
                "multi-intent / ambiguity robustness examples",
                "unsupported / OOD examples",
            ],
            "limitations": [
                "CFPB narratives are often long-form and retrospective.",
                "A narrative may contain multiple events, intents, and requested remedies.",
                "A narrative may describe actions a company already took.",
                (
                    "A structured issue label describes complaint subject matter, not "
                    "necessarily a direct conversational intent."
                ),
                "Narrative-level semantic qualification is deferred to a later phase.",
            ],
        },
        "mapped_taxonomy_combinations": assignments,
        "mapping_rules": rule_payloads,
        "mapping_version": "2026-09-27.v1",
        "precedence_policy": (
            "full product/sub-product/issue/sub-issue > product/sub-product/issue > "
            "product/issue > product > explicit default; same-precedence overlaps fail."
        ),
        "protected_action_policy": (
            "CFPB fraud, unauthorized-transaction, lost/stolen-card, dispute, and "
            "chargeback topics never establish freeze_card or create_dispute as an "
            "EXACT_MATCH. Protected actions require later narrative-level evidence of a "
            "current explicit user request plus existing authorization and confirmation."
        ),
        "qualification_phase": "V2-C2M",
        "schema_version": "cfpb-to-sentinel-intents.v1",
        "semantic_boundary": (
            "CFPB structured taxonomy represents a complaint's product and issue, not the "
            "exact conversational action a SentinelVoice user requests."
        ),
        "sentinelvoice_intents": list(SENTINELVOICE_INTENTS),
        "sentinelvoice_risk_groups": [
            "PUBLIC",
            "PRIVATE_READ",
            "PROTECTED_WRITE",
            "ESCALATION_OR_UNCERTAIN",
        ],
        "sentinelvoice_risk_group_by_intent": RISK_BY_INTENT,
        "sentinelvoice_taxonomy_version": "sentinelvoice-v2c1-intent-taxonomy.v1",
        "source_id": "cfpb_consumer_complaint_narratives_archive",
    }


def build_artifacts(
    archive_paths: Iterable[Path],
    *,
    manifest_sha256: str,
    profile_sha256: str,
) -> tuple[bytes, bytes]:
    scan = scan_archives(archive_paths)
    inventory = build_inventory(
        scan,
        manifest_sha256=manifest_sha256,
        profile_sha256=profile_sha256,
    )
    mapping = build_mapping(inventory)
    return stable_json_bytes(inventory), stable_json_bytes(mapping)


def production_archive_paths(manifest: dict[str, Any]) -> list[Path]:
    archives = manifest.get("archives")
    if not isinstance(archives, list) or len(archives) != 21:
        raise ValueError("raw CFPB manifest must contain 21 archives")
    paths = [RAW_FILES_DIR / archive["local_filename"] for archive in archives]
    for path, archive in zip(paths, archives, strict=True):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != archive["compressed_byte_size"]:
            raise ValueError(f"archive size differs from manifest: {path}")
    return paths


def main() -> None:
    manifest_bytes = RAW_MANIFEST_PATH.read_bytes()
    profile_bytes = RAW_PROFILE_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    profile = json.loads(profile_bytes)
    inventory_bytes, mapping_bytes = build_artifacts(
        production_archive_paths(manifest),
        manifest_sha256=sha256_bytes(manifest_bytes),
        profile_sha256=sha256_bytes(profile_bytes),
    )
    inventory = json.loads(inventory_bytes)
    if inventory["summary"]["narrative_count"] != profile["summary"][
        "rows_with_nonempty_narrative"
    ]:
        raise ValueError("taxonomy inventory narrative count differs from raw profile")
    if inventory["summary"]["total_full_records_rows_scanned"] != profile[
        "summary"
    ]["total_rows"]:
        raise ValueError("taxonomy inventory total row count differs from raw profile")

    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_bytes(inventory_bytes)
    MAPPING_PATH.write_bytes(mapping_bytes)
    mapping = json.loads(mapping_bytes)
    print(
        "Built CFPB taxonomy artifacts: "
        f"narratives={inventory['summary']['narrative_count']}, "
        f"tuples={inventory['summary']['unique_full_taxonomy_tuples']}, "
        f"active_rules={len(mapping['mapping_rules'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
