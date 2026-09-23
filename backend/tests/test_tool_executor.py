import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Account
from backend.app.tools.definitions import ToolDefinition
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolConfirmationError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import (
    TOOL_REGISTRY,
    RegisteredTool,
)
from backend.app.tools.schemas import (
    ActionConfirmation,
    ToolExecutionContext,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
ACCOUNT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")


def test_registry_contains_exactly_seven_v1_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "get_account_balance",
        "get_recent_transactions",
        "get_transaction_details",
        "get_card_status",
        "freeze_card",
        "create_dispute",
        "escalate_to_human",
    }


@pytest.mark.asyncio
async def test_executor_rejects_unknown_tool() -> None:
    executor = ToolExecutor()
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolNotFoundError):
        await executor.execute(
            "delete_everything",
            {},
            ToolExecutionContext(),
            session,
        )


@pytest.mark.asyncio
async def test_executor_rejects_unauthenticated_private_read() -> None:
    executor = ToolExecutor()
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolAuthenticationError):
        await executor.execute(
            "get_account_balance",
            {"account_id": str(ACCOUNT_ID)},
            ToolExecutionContext(),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_validates_model_tool_arguments() -> None:
    executor = ToolExecutor()
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolValidationError):
        await executor.execute(
            "get_account_balance",
            {"account_id": "not-a-uuid"},
            ToolExecutionContext(
                customer_id=CUSTOMER_ID,
                authenticated=True,
            ),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_dispatches_private_read() -> None:
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

    executor = ToolExecutor()

    result = await executor.execute(
        "get_account_balance",
        {"account_id": str(ACCOUNT_ID)},
        ToolExecutionContext(
            customer_id=CUSTOMER_ID,
            authenticated=True,
        ),
        session,
    )

    assert result.account_id == ACCOUNT_ID
    assert result.current_balance == Decimal("2847.63")


@pytest.mark.asyncio
async def test_executor_requires_confirmation_for_protected_write() -> None:
    executor = ToolExecutor()
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolConfirmationError):
        await executor.execute(
            "freeze_card",
            {"card_id": str(CARD_ID)},
            ToolExecutionContext(
                customer_id=CUSTOMER_ID,
                authenticated=True,
            ),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_rejects_confirmation_for_other_action() -> None:
    executor = ToolExecutor()
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolConfirmationError):
        await executor.execute(
            "freeze_card",
            {"card_id": str(CARD_ID)},
            ToolExecutionContext(
                customer_id=CUSTOMER_ID,
                authenticated=True,
                confirmation=ActionConfirmation(
                    action="create_dispute",
                    resource_id=CARD_ID,
                    confirmed=True,
                ),
            ),
            session,
        )

    session.scalar.assert_not_awaited()


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    ok: bool


@pytest.mark.asyncio
async def test_executor_enforces_tool_timeout() -> None:
    async def slow_handler(
        request,
        context,
        session,
    ) -> BaseModel:
        await asyncio.sleep(0.05)
        return EmptyOutput(ok=True)

    definition = ToolDefinition(
        name="slow_tool",
        permission_level=TOOL_REGISTRY[
            "get_account_balance"
        ].definition.permission_level,
        requires_authentication=False,
        requires_confirmation=False,
        timeout_seconds=0.001,
        idempotent=True,
        audit_event="test.slow",
        error_types=("timeout",),
    )

    executor = ToolExecutor(
        registry={
            "slow_tool": RegisteredTool(
                definition=definition,
                input_model=EmptyInput,
                handler=slow_handler,
            )
        }
    )

    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolTimeoutError):
        await executor.execute(
            "slow_tool",
            {},
            ToolExecutionContext(),
            session,
        )
