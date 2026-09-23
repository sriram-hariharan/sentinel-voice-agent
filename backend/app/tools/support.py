from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import SupportCase
from backend.app.tools.errors import ToolSessionError
from backend.app.tools.schemas import (
    EscalateToHumanInput,
    EscalateToHumanOutput,
    HandoffSummary,
    ToolExecutionContext,
)


def _case_id_for_session(session_id: str) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"sentinelvoice:support-case:{session_id}",
    )


def _build_handoff(
    request: EscalateToHumanInput,
    customer_id: UUID | None,
    authenticated: bool,
    category: str,
    priority: str,
    summary: str,
    handoff_reason: str,
) -> HandoffSummary:
    return HandoffSummary(
        customer_id=customer_id,
        authenticated=authenticated,
        category=category,
        priority=priority,
        summary=summary,
        transaction_id=request.transaction_id,
        transaction_amount=request.transaction_amount,
        merchant=request.merchant,
        actions_completed=request.actions_completed,
        actions_not_completed=request.actions_not_completed,
        reason_for_handoff=handoff_reason,
        conversation_summary=request.conversation_summary,
    )


def _output_from_existing_case(
    case: SupportCase,
    request: EscalateToHumanInput,
) -> EscalateToHumanOutput:
    return EscalateToHumanOutput(
        case_id=case.case_id,
        status=case.status,
        created=False,
        handoff=_build_handoff(
            request=request,
            customer_id=case.customer_id,
            authenticated=case.customer_id is not None,
            category=case.category,
            priority=case.priority,
            summary=case.summary,
            handoff_reason=case.handoff_reason,
        ),
    )


async def escalate_to_human(
    request: EscalateToHumanInput,
    context: ToolExecutionContext,
    session: AsyncSession,
) -> EscalateToHumanOutput:
    if context.session_id is None:
        raise ToolSessionError("Server-side session ID is required for escalation")

    case_id = _case_id_for_session(context.session_id)

    existing_case = await session.scalar(
        select(SupportCase).where(SupportCase.case_id == case_id)
    )

    if existing_case is not None:
        return _output_from_existing_case(existing_case, request)

    verified_customer_id = (
        context.customer_id
        if context.authenticated and context.customer_id is not None
        else None
    )

    support_case = SupportCase(
        case_id=case_id,
        customer_id=verified_customer_id,
        session_id=context.session_id,
        category=request.category,
        priority=request.priority,
        status="ESCALATED",
        summary=request.summary,
        handoff_reason=request.handoff_reason,
    )

    session.add(support_case)

    try:
        await session.flush()
        await session.commit()
    except IntegrityError:
        # A simultaneous retry for the same session may have created
        # the deterministic case ID first.
        await session.rollback()

        existing_case = await session.scalar(
            select(SupportCase).where(SupportCase.case_id == case_id)
        )

        if existing_case is None:
            raise

        return _output_from_existing_case(existing_case, request)

    return EscalateToHumanOutput(
        case_id=support_case.case_id,
        status=support_case.status,
        created=True,
        handoff=_build_handoff(
            request=request,
            customer_id=verified_customer_id,
            authenticated=verified_customer_id is not None,
            category=support_case.category,
            priority=support_case.priority,
            summary=support_case.summary,
            handoff_reason=support_case.handoff_reason,
        ),
    )
