from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Account, Card, Transaction
from backend.app.tools.banking import (
    get_account_balance,
    get_card_status,
    get_recent_transactions,
    get_transaction_details,
)
from backend.app.tools.definitions import (
    GET_ACCOUNT_BALANCE,
    GET_CARD_STATUS,
    GET_RECENT_TRANSACTIONS,
    GET_TRANSACTION_DETAILS,
    PermissionLevel,
)
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolResourceNotFoundError,
)
from backend.app.tools.schemas import (
    AccountBalanceInput,
    CardStatusInput,
    RecentTransactionsInput,
    ToolExecutionContext,
    TransactionDetailsInput,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
ACCOUNT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")
TRANSACTION_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeee1")


@pytest.mark.parametrize(
    "definition",
    [
        GET_ACCOUNT_BALANCE,
        GET_RECENT_TRANSACTIONS,
        GET_TRANSACTION_DETAILS,
        GET_CARD_STATUS,
    ],
)
def test_private_read_tool_definitions(definition) -> None:
    assert definition.permission_level == PermissionLevel.PRIVATE_READ
    assert definition.requires_authentication is True
    assert definition.requires_confirmation is False
    assert definition.idempotent is True


@pytest.mark.asyncio
async def test_get_account_balance_requires_authentication() -> None:
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolAuthenticationError):
        await get_account_balance(
            AccountBalanceInput(account_id=ACCOUNT_ID),
            ToolExecutionContext(),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_account_balance_returns_owned_account() -> None:
    account = Account(
        account_id=ACCOUNT_ID,
        customer_id=CUSTOMER_ID,
        account_type="checking",
        masked_account_number="****4101",
        current_balance=Decimal("2847.63"),
        available_balance=Decimal("2612.44"),
        currency="USD",
        status="ACTIVE",
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = account

    result = await get_account_balance(
        AccountBalanceInput(account_id=ACCOUNT_ID),
        ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
        session,
    )

    assert result.current_balance == Decimal("2847.63")
    assert result.available_balance == Decimal("2612.44")


@pytest.mark.asyncio
async def test_get_account_balance_hides_unowned_account() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    with pytest.raises(ToolResourceNotFoundError, match="Account not found"):
        await get_account_balance(
            AccountBalanceInput(account_id=ACCOUNT_ID),
            ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
            session,
        )


@pytest.mark.asyncio
async def test_get_recent_transactions_returns_owned_account_transactions() -> None:
    account = Account(
        account_id=ACCOUNT_ID,
        customer_id=CUSTOMER_ID,
        account_type="checking",
        masked_account_number="****4101",
        current_balance=Decimal("2847.63"),
        available_balance=Decimal("2612.44"),
        currency="USD",
        status="ACTIVE",
    )

    transaction = Transaction(
        transaction_id=TRANSACTION_ID,
        account_id=ACCOUNT_ID,
        card_id=CARD_ID,
        merchant_name="ABC Electronics",
        merchant_category="Electronics",
        amount=Decimal("274.19"),
        currency="USD",
        transaction_timestamp=datetime(2026, 9, 20, tzinfo=UTC),
        posted_timestamp=datetime(2026, 9, 21, tzinfo=UTC),
        status="POSTED",
        transaction_type="CARD_PURCHASE",
        location="Newark, NJ",
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = account

    scalar_result = MagicMock()
    scalar_result.all.return_value = [transaction]
    session.scalars.return_value = scalar_result

    result = await get_recent_transactions(
        RecentTransactionsInput(account_id=ACCOUNT_ID, limit=10),
        ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
        session,
    )

    assert len(result.transactions) == 1
    assert result.transactions[0].merchant_name == "ABC Electronics"
    assert result.transactions[0].amount == Decimal("274.19")


@pytest.mark.asyncio
async def test_get_recent_transactions_hides_unowned_account() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    with pytest.raises(ToolResourceNotFoundError, match="Account not found"):
        await get_recent_transactions(
            RecentTransactionsInput(account_id=ACCOUNT_ID),
            ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
            session,
        )

    session.scalars.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_transaction_details_returns_owned_transaction() -> None:
    transaction = Transaction(
        transaction_id=TRANSACTION_ID,
        account_id=ACCOUNT_ID,
        card_id=CARD_ID,
        merchant_name="ABC Electronics",
        merchant_category="Electronics",
        amount=Decimal("274.19"),
        currency="USD",
        transaction_timestamp=datetime(2026, 9, 20, tzinfo=UTC),
        posted_timestamp=datetime(2026, 9, 21, tzinfo=UTC),
        status="POSTED",
        transaction_type="CARD_PURCHASE",
        location="Newark, NJ",
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = transaction

    result = await get_transaction_details(
        TransactionDetailsInput(transaction_id=TRANSACTION_ID),
        ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
        session,
    )

    assert result.transaction_id == TRANSACTION_ID
    assert result.merchant_name == "ABC Electronics"
    assert result.amount == Decimal("274.19")


@pytest.mark.asyncio
async def test_get_transaction_details_hides_unowned_transaction() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    with pytest.raises(ToolResourceNotFoundError, match="Transaction not found"):
        await get_transaction_details(
            TransactionDetailsInput(transaction_id=TRANSACTION_ID),
            ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
            session,
        )


@pytest.mark.asyncio
async def test_get_card_status_returns_owned_card() -> None:
    card = Card(
        card_id=CARD_ID,
        customer_id=CUSTOMER_ID,
        account_id=ACCOUNT_ID,
        masked_card_number="****1842",
        card_type="DEBIT",
        status="ACTIVE",
        expiration_month=8,
        expiration_year=2029,
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = card

    result = await get_card_status(
        CardStatusInput(card_id=CARD_ID),
        ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
        session,
    )

    assert result.card_id == CARD_ID
    assert result.status == "ACTIVE"
    assert result.masked_card_number == "****1842"


@pytest.mark.asyncio
async def test_get_card_status_hides_unowned_card() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    with pytest.raises(ToolResourceNotFoundError, match="Card not found"):
        await get_card_status(
            CardStatusInput(card_id=CARD_ID),
            ToolExecutionContext(customer_id=CUSTOMER_ID, authenticated=True),
            session,
        )


@pytest.mark.asyncio
async def test_freeze_card_requires_matching_confirmation() -> None:
    from backend.app.tools.banking import freeze_card
    from backend.app.tools.errors import ToolConfirmationError
    from backend.app.tools.schemas import FreezeCardInput

    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolConfirmationError):
        await freeze_card(
            FreezeCardInput(card_id=CARD_ID),
            ToolExecutionContext(
                customer_id=CUSTOMER_ID,
                authenticated=True,
            ),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_freeze_card_rejects_confirmation_for_different_card() -> None:
    from backend.app.tools.banking import freeze_card
    from backend.app.tools.errors import ToolConfirmationError
    from backend.app.tools.schemas import (
        ActionConfirmation,
        FreezeCardInput,
    )

    other_card_id = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc2")
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolConfirmationError):
        await freeze_card(
            FreezeCardInput(card_id=CARD_ID),
            ToolExecutionContext(
                customer_id=CUSTOMER_ID,
                authenticated=True,
                confirmation=ActionConfirmation(
                    action="freeze_card",
                    resource_id=other_card_id,
                    confirmed=True,
                ),
            ),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_freeze_card_freezes_owned_active_card() -> None:
    from backend.app.tools.banking import freeze_card
    from backend.app.tools.schemas import (
        ActionConfirmation,
        FreezeCardInput,
    )

    card = Card(
        card_id=CARD_ID,
        customer_id=CUSTOMER_ID,
        account_id=ACCOUNT_ID,
        masked_card_number="****1842",
        card_type="DEBIT",
        status="ACTIVE",
        expiration_month=8,
        expiration_year=2029,
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = card

    result = await freeze_card(
        FreezeCardInput(card_id=CARD_ID),
        ToolExecutionContext(
            customer_id=CUSTOMER_ID,
            authenticated=True,
            confirmation=ActionConfirmation(
                action="freeze_card",
                resource_id=CARD_ID,
                confirmed=True,
            ),
        ),
        session,
    )

    assert result.previous_status == "ACTIVE"
    assert result.status == "FROZEN"
    assert result.changed is True
    assert card.status == "FROZEN"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_freeze_card_is_idempotent_when_already_frozen() -> None:
    from backend.app.tools.banking import freeze_card
    from backend.app.tools.schemas import (
        ActionConfirmation,
        FreezeCardInput,
    )

    card = Card(
        card_id=CARD_ID,
        customer_id=CUSTOMER_ID,
        account_id=ACCOUNT_ID,
        masked_card_number="****1842",
        card_type="DEBIT",
        status="FROZEN",
        expiration_month=8,
        expiration_year=2029,
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = card

    result = await freeze_card(
        FreezeCardInput(card_id=CARD_ID),
        ToolExecutionContext(
            customer_id=CUSTOMER_ID,
            authenticated=True,
            confirmation=ActionConfirmation(
                action="freeze_card",
                resource_id=CARD_ID,
                confirmed=True,
            ),
        ),
        session,
    )

    assert result.status == "FROZEN"
    assert result.changed is False
    session.commit.assert_not_awaited()


def test_freeze_card_definition_is_protected() -> None:
    from backend.app.tools.definitions import FREEZE_CARD

    assert FREEZE_CARD.permission_level == PermissionLevel.PROTECTED_WRITE
    assert FREEZE_CARD.requires_authentication is True
    assert FREEZE_CARD.requires_confirmation is True
    assert FREEZE_CARD.idempotent is True
