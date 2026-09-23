from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import SupportCase
from backend.app.tools.definitions import (
    ESCALATE_TO_HUMAN,
    PermissionLevel,
)
from backend.app.tools.errors import ToolSessionError
from backend.app.tools.schemas import (
    EscalateToHumanInput,
    ToolExecutionContext,
)
from backend.app.tools.support import escalate_to_human

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
CASE_ID = UUID("99999999-9999-4999-8999-999999999999")
TRANSACTION_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeee1")


def _request() -> EscalateToHumanInput:
    return EscalateToHumanInput(
        category="suspected_card_fraud",
        priority="HIGH",
        summary="Customer reports an unrecognized card transaction.",
        handoff_reason="Fraud review required.",
        transaction_id=TRANSACTION_ID,
        transaction_amount=Decimal("274.19"),
        merchant="ABC Electronics",
        actions_completed=["retrieved transaction details"],
        actions_not_completed=["manual fraud review"],
        conversation_summary="Customer does not recognize the transaction.",
    )


def test_escalate_to_human_is_safety_path() -> None:
    assert (
        ESCALATE_TO_HUMAN.permission_level
        == PermissionLevel.SAFETY_ESCALATION
    )
    assert ESCALATE_TO_HUMAN.requires_authentication is False
    assert ESCALATE_TO_HUMAN.requires_confirmation is False
    assert ESCALATE_TO_HUMAN.idempotent is True


@pytest.mark.asyncio
async def test_escalation_requires_server_side_session() -> None:
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(ToolSessionError):
        await escalate_to_human(
            _request(),
            ToolExecutionContext(),
            session,
        )

    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthenticated_escalation_creates_case_without_customer() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    result = await escalate_to_human(
        _request(),
        ToolExecutionContext(
            session_id="session-auth-failure-001",
            customer_id=CUSTOMER_ID,
            authenticated=False,
        ),
        session,
    )

    assert result.created is True
    assert result.status == "ESCALATED"
    assert result.handoff.authenticated is False
    assert result.handoff.customer_id is None
    assert result.handoff.transaction_id == TRANSACTION_ID

    support_case = session.add.call_args.args[0]
    assert support_case.customer_id is None
    assert support_case.session_id == "session-auth-failure-001"

    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_authenticated_escalation_keeps_verified_customer() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    result = await escalate_to_human(
        _request(),
        ToolExecutionContext(
            session_id="session-authenticated-001",
            customer_id=CUSTOMER_ID,
            authenticated=True,
        ),
        session,
    )

    assert result.created is True
    assert result.handoff.authenticated is True
    assert result.handoff.customer_id == CUSTOMER_ID

    support_case = session.add.call_args.args[0]
    assert support_case.customer_id == CUSTOMER_ID


@pytest.mark.asyncio
async def test_escalation_retry_returns_existing_case() -> None:
    existing_case = SupportCase(
        case_id=CASE_ID,
        customer_id=CUSTOMER_ID,
        session_id="session-retry-001",
        category="suspected_card_fraud",
        priority="HIGH",
        status="ESCALATED",
        summary="Customer reports an unrecognized card transaction.",
        handoff_reason="Fraud review required.",
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = existing_case

    result = await escalate_to_human(
        _request(),
        ToolExecutionContext(
            session_id="session-retry-001",
            customer_id=CUSTOMER_ID,
            authenticated=True,
        ),
        session,
    )

    assert result.case_id == CASE_ID
    assert result.created is False
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_simultaneous_escalation_retry_recovers_existing_case() -> None:
    existing_case = SupportCase(
        case_id=CASE_ID,
        customer_id=None,
        session_id="session-race-001",
        category="identity_verification_failure",
        priority="HIGH",
        status="ESCALATED",
        summary="Customer could not complete identity verification.",
        handoff_reason="Identity verification failed.",
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.side_effect = [None, existing_case]
    session.flush.side_effect = IntegrityError(
        "insert support case",
        {},
        Exception("duplicate case"),
    )

    request = EscalateToHumanInput(
        category="identity_verification_failure",
        priority="HIGH",
        summary="Customer could not complete identity verification.",
        handoff_reason="Identity verification failed.",
        actions_completed=[],
        actions_not_completed=["identity verification"],
        conversation_summary="Verification could not be completed.",
    )

    result = await escalate_to_human(
        request,
        ToolExecutionContext(
            session_id="session-race-001",
            authenticated=False,
        ),
        session,
    )

    assert result.case_id == CASE_ID
    assert result.created is False
    assert result.handoff.authenticated is False
    session.rollback.assert_awaited_once()
