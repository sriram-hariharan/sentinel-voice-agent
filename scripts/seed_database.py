import argparse
import asyncio
import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import delete, select

from backend.app.db.models import (
    Account,
    Card,
    Customer,
    Dispute,
    SupportCase,
    Transaction,
)
from backend.app.db.session import get_session_factory

FIXTURE_PATH = Path("data/fixtures/banking.json")
V2_FIXTURE_PATH = Path("data/fixtures/v2/banking.generated.json")
FIXTURE_COLLECTIONS = (
    "customers",
    "accounts",
    "cards",
    "transactions",
    "disputes",
    "support_cases",
)


def load_fixtures(profile: str = "v1") -> dict:
    canonical = json.loads(FIXTURE_PATH.read_text())
    if profile == "v1":
        return canonical
    if profile != "v2":
        raise ValueError(f"unsupported seed profile: {profile}")

    generated = json.loads(V2_FIXTURE_PATH.read_text())
    return {
        collection: [
            *canonical.get(collection, []),
            *generated.get(collection, []),
        ]
        for collection in FIXTURE_COLLECTIONS
    }


def parse_uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value)


async def seed(reset: bool, profile: str = "v1") -> None:
    fixture = load_fixtures(profile)
    session_factory = get_session_factory()

    async with session_factory() as session:
        existing = await session.scalar(select(Customer.customer_id).limit(1))

        if existing is not None and not reset:
            raise RuntimeError(
                "Database already contains synthetic customers. "
                "Run with --reset to restore deterministic fixtures."
            )

        if reset:
            for model in (
                Dispute,
                Transaction,
                Card,
                SupportCase,
                Account,
                Customer,
            ):
                await session.execute(delete(model))

        session.add_all(
            Customer(
                customer_id=parse_uuid(row["customer_id"]),
                first_name=row["first_name"],
                last_name=row["last_name"],
                email=row["email"],
                phone=row["phone"],
                date_of_birth=date.fromisoformat(row["date_of_birth"]),
                status=row["status"],
                authentication_profile=row["authentication_profile"],
            )
            for row in fixture["customers"]
        )
        await session.flush()

        session.add_all(
            Account(
                account_id=parse_uuid(row["account_id"]),
                customer_id=parse_uuid(row["customer_id"]),
                account_type=row["account_type"],
                masked_account_number=row["masked_account_number"],
                current_balance=Decimal(row["current_balance"]),
                available_balance=Decimal(row["available_balance"]),
                currency=row["currency"],
                status=row["status"],
            )
            for row in fixture["accounts"]
        )
        await session.flush()

        session.add_all(
            Card(
                card_id=parse_uuid(row["card_id"]),
                customer_id=parse_uuid(row["customer_id"]),
                account_id=parse_uuid(row["account_id"]),
                masked_card_number=row["masked_card_number"],
                card_type=row["card_type"],
                status=row["status"],
                expiration_month=row["expiration_month"],
                expiration_year=row["expiration_year"],
            )
            for row in fixture["cards"]
        )
        await session.flush()

        session.add_all(
            Transaction(
                transaction_id=parse_uuid(row["transaction_id"]),
                account_id=parse_uuid(row["account_id"]),
                card_id=parse_uuid(row["card_id"]) if row["card_id"] else None,
                merchant_name=row["merchant_name"],
                merchant_category=row["merchant_category"],
                amount=Decimal(row["amount"]),
                currency=row["currency"],
                transaction_timestamp=datetime.fromisoformat(
                    row["transaction_timestamp"]
                ),
                posted_timestamp=(
                    datetime.fromisoformat(row["posted_timestamp"])
                    if row["posted_timestamp"]
                    else None
                ),
                status=row["status"],
                transaction_type=row["transaction_type"],
                location=row["location"],
            )
            for row in fixture["transactions"]
        )
        await session.flush()

        session.add_all(
            Dispute(
                dispute_id=parse_uuid(row["dispute_id"]),
                customer_id=parse_uuid(row["customer_id"]),
                transaction_id=parse_uuid(row["transaction_id"]),
                reason_code=row["reason_code"],
                status=row["status"],
                notes=row["notes"],
            )
            for row in fixture["disputes"]
        )
        await session.flush()

        await session.commit()

    print(
        "Seeded "
        f"{len(fixture['customers'])} customers, "
        f"{len(fixture['accounts'])} accounts, "
        f"{len(fixture['cards'])} cards, "
        f"{len(fixture['transactions'])} transactions, "
        f"{len(fixture['disputes'])} disputes."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing synthetic banking data before seeding.",
    )
    parser.add_argument(
        "--profile",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "Seed the canonical V1 fixture, or explicitly compose it with "
            "the generated V2 add-on."
        ),
    )
    args = parser.parse_args()

    asyncio.run(seed(reset=args.reset, profile=args.profile))


if __name__ == "__main__":
    main()
