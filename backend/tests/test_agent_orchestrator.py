from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.orchestrator import (
    AgentOrchestrator,
    AgentTurnStatus,
)
from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationPhase,
    ConversationState,
)
from backend.app.providers.llm import (
    LLMResponse,
    LLMToolCall,
)
from backend.app.tools.schemas import (
    AccountBalanceOutput,
    FreezeCardOutput,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
ACCOUNT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")


class SequenceLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(
        self,
        *,
        messages,
        tools=None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
            }
        )

        return self.responses.pop(0)


def _authenticated_state() -> ConversationState:
    return ConversationState(
        session_id="session-001",
        customer_id=CUSTOMER_ID,
        authentication_level=AuthenticationLevel.AUTHENTICATED,
    )


@pytest.mark.asyncio
async def test_unauthenticated_session_exposes_only_escalation() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="How can I help?",
                model="test-model",
            )
        ]
    )

    orchestrator = AgentOrchestrator(llm=llm)
    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="Hello",
        state=ConversationState(session_id="session-001"),
        db=db,
    )

    assert result.text == "How can I help?"

    tool_names = {
        schema["function"]["name"]
        for schema in llm.calls[0]["tools"]
    }

    assert tool_names == {"escalate_to_human"}


@pytest.mark.asyncio
async def test_authenticated_session_exposes_all_v1_tools() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="How can I help?",
                model="test-model",
            )
        ]
    )

    orchestrator = AgentOrchestrator(llm=llm)
    db = AsyncMock(spec=AsyncSession)

    await orchestrator.handle_text_turn(
        user_text="Hello",
        state=_authenticated_state(),
        db=db,
    )

    tool_names = {
        schema["function"]["name"]
        for schema in llm.calls[0]["tools"]
    }

    assert tool_names == {
        "get_account_balance",
        "get_recent_transactions",
        "get_transaction_details",
        "get_card_status",
        "freeze_card",
        "create_dispute",
        "escalate_to_human",
    }


@pytest.mark.asyncio
async def test_protected_tool_proposal_waits_for_confirmation() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="",
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-001",
                        name="freeze_card",
                        arguments={
                            "card_id": str(CARD_ID),
                        },
                    )
                ],
            )
        ]
    )

    executor = AsyncMock()
    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
    )

    state = _authenticated_state()
    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="Freeze my card",
        state=state,
        db=db,
    )

    assert result.status == AgentTurnStatus.WAITING_FOR_CONFIRMATION
    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION

    assert state.pending_action is not None
    assert state.pending_action.action == "freeze_card"
    assert state.pending_action.resource_id == CARD_ID
    assert state.pending_action.arguments == {
        "card_id": str(CARD_ID),
    }

    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_yes_executes_stored_protected_action_once() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="Your card is now frozen.",
                model="test-model",
            )
        ]
    )

    executor = AsyncMock()
    executor.execute.return_value = FreezeCardOutput(
        card_id=CARD_ID,
        masked_card_number="****1842",
        previous_status="ACTIVE",
        status="FROZEN",
        changed=True,
    )

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
    )

    state = _authenticated_state()
    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={
            "card_id": str(CARD_ID),
        },
    )

    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="Yes",
        state=state,
        db=db,
    )

    assert result.text == "Your card is now frozen."
    assert result.executed_tools == ["freeze_card"]
    assert state.pending_action is None

    executor.execute.assert_awaited_once()

    call = executor.execute.await_args

    assert call.args[0] == "freeze_card"
    assert call.args[1] == {
        "card_id": str(CARD_ID),
    }

    context = call.args[2]

    assert context.confirmation is not None
    assert context.confirmation.action == "freeze_card"
    assert context.confirmation.resource_id == CARD_ID


@pytest.mark.asyncio
async def test_no_cancels_protected_action_without_execution() -> None:
    llm = SequenceLLM([])

    executor = AsyncMock()
    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
    )

    state = _authenticated_state()
    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={
            "card_id": str(CARD_ID),
        },
    )

    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="Actually don't",
        state=state,
        db=db,
    )

    assert result.text == "Okay, I won’t perform that action."
    assert state.pending_action is None
    executor.execute.assert_not_awaited()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_correction_invalidates_old_confirmation_and_is_reprocessed() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="Which card would you like help with?",
                model="test-model",
            )
        ]
    )

    executor = AsyncMock()

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
    )

    state = _authenticated_state()
    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={
            "card_id": str(CARD_ID),
        },
    )

    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="I meant my other card",
        state=state,
        db=db,
    )

    assert (
        result.text
        == "Which card would you like help with?"
    )
    assert state.pending_action is None
    executor.execute.assert_not_awaited()
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_private_read_tool_result_returns_to_llm() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="",
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-balance",
                        name="get_account_balance",
                        arguments={
                            "account_id": str(ACCOUNT_ID),
                        },
                    )
                ],
            ),
            LLMResponse(
                content="Your available balance is $2,612.44.",
                model="test-model",
            ),
        ]
    )

    executor = AsyncMock()
    executor.execute.return_value = AccountBalanceOutput(
        account_id=ACCOUNT_ID,
        account_type="checking",
        masked_account_number="****4101",
        current_balance=Decimal("2847.63"),
        available_balance=Decimal("2612.44"),
        currency="USD",
        status="ACTIVE",
    )

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
    )

    state = _authenticated_state()
    state.active_account_id = ACCOUNT_ID

    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="What is my checking balance?",
        state=state,
        db=db,
    )

    assert result.text == "Your available balance is $2,612.44."
    assert result.executed_tools == ["get_account_balance"]

    assert state.last_tool_result is not None
    assert (
        state.last_tool_result["account_id"]
        == str(ACCOUNT_ID)
    )

    assert len(llm.calls) == 2

    second_messages = llm.calls[1]["messages"]

    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call-balance"
