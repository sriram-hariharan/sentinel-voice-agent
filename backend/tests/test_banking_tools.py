from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Account
from backend.app.tools.banking import get_account_balance
from backend.app.tools.definitions import (
    GET_ACCOUNT_BALANCE,
    PermissionLevel,
)
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolResourceNotFoundError,
)
from backend.app.tools.schemas import (
    AccountBalanceInput,
    ToolExecutionContext,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
ACCOUNT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")


def test_get_account_balance_definition() -> None:
    assert GET_ACCOUNT_BALANCE.permission_level == PermissionLevel.PRIVATE_READ
    assert GET_ACCOUNT_BALANCE.requires_authentication is True
    assert GET_ACCOUNT_BALANCE.requires_confirmation is False
    assert GET_ACCOUNT_BALANCE.idempotent is True


@pytest.mark.asyncio
async def test_get_account_balance_requires_authentication() -> None:
    session = AsyncMock(spec=AsyncSession)

    request = AccountBalanceInput(account_id=ACCOUNT_ID)
    context = ToolExecutionContext()

    with pytest.raises(ToolAuthenticationError):
        await get_account_balance(request, context, session)

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

    request = AccountBalanceInput(account_id=ACCOUNT_ID)
    context = ToolExecutionContext(
        customer_id=CUSTOMER_ID,
        authenticated=True,
    )

    result = await get_account_balance(request, context, session)

    assert result.account_id == ACCOUNT_ID
    assert result.current_balance == Decimal("2847.63")
    assert result.available_balance == Decimal("2612.44")
    assert result.masked_account_number == "****4101"

    session.scalar.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_account_balance_hides_unowned_account() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    request = AccountBalanceInput(account_id=ACCOUNT_ID)
    context = ToolExecutionContext(
        customer_id=CUSTOMER_ID,
        authenticated=True,
    )

    with pytest.raises(ToolResourceNotFoundError, match="Account not found"):
        await get_account_balance(request, context, session)

    statement = session.scalar.await_args.args[0]
    statement_text = str(statement)

    assert "accounts.account_id" in statement_text
    assert "accounts.customer_id" in statement_text
