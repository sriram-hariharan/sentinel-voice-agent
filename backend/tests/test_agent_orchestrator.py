from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.orchestrator import (
    AgentOrchestrator,
    AgentTurnStatus,
)
from backend.app.agent.resource_resolver import ResourceResolution
from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationPhase,
    ConversationState,
    EscalationStatus,
)
from backend.app.providers.llm import (
    LLMResponse,
    LLMToolCall,
)
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolBackendError,
    ToolConfirmationError,
    ToolValidationError,
)
from backend.app.tools.schemas import (
    AccountBalanceOutput,
    EscalateToHumanOutput,
    FreezeCardOutput,
    HandoffSummary,
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


class NoopResourceResolver:
    async def resolve(self, **kwargs) -> ResourceResolution:
        return ResourceResolution()


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
        resource_resolver=NoopResourceResolver(),
    )

    state = _authenticated_state()
    state.active_card_id = CARD_ID
    state.active_intent = "freeze_card"
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
        resource_resolver=NoopResourceResolver(),
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
        resource_resolver=NoopResourceResolver(),
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
        resource_resolver=NoopResourceResolver(),
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
        resource_resolver=NoopResourceResolver(),
    )

    state = _authenticated_state()
    state.active_account_id = ACCOUNT_ID
    state.active_intent = "get_account_balance"

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


def _balance_tool_call(call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        model="test-model",
        tool_calls=[
            LLMToolCall(
                id=call_id,
                name="get_account_balance",
                arguments={"account_id": str(ACCOUNT_ID)},
            )
        ],
    )


def _escalation_output() -> EscalateToHumanOutput:
    return EscalateToHumanOutput(
        case_id=UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd"),
        status="ESCALATED",
        created=True,
        handoff=HandoffSummary(
            customer_id=CUSTOMER_ID,
            authenticated=True,
            category="repeated_backend_failure",
            priority="HIGH",
            summary="Repeated backend failures.",
            transaction_id=None,
            transaction_amount=None,
            merchant=None,
            actions_completed=[],
            actions_not_completed=["get_account_balance"],
            reason_for_handoff="Repeated backend failure.",
            conversation_summary="Customer asked for their balance.",
        ),
    )


@pytest.mark.asyncio
async def test_first_backend_failure_does_not_auto_escalate() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="I couldn't retrieve that right now. Please try again.",
                model="test-model",
            )
        ]
    )
    executor = AsyncMock()
    executor.execute.side_effect = ToolBackendError("banking backend unavailable")

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
        resource_resolver=NoopResourceResolver(),
    )
    state = _authenticated_state()
    state.active_account_id = ACCOUNT_ID
    state.active_intent = "get_account_balance"
    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator._run_model_loop(
        messages=[{"role": "user", "content": "What is my balance?"}],
        state=state,
        db=db,
        initial_response=_balance_tool_call("call-backend-1"),
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert state.consecutive_backend_failures == 1
    assert state.escalation_status == EscalationStatus.NONE
    assert executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_second_consecutive_backend_failure_auto_escalates() -> None:
    llm = SequenceLLM([])
    executor = AsyncMock()
    executor.execute.side_effect = [
        ToolBackendError("banking backend unavailable"),
        _escalation_output(),
    ]

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
        resource_resolver=NoopResourceResolver(),
    )
    state = _authenticated_state()
    state.active_account_id = ACCOUNT_ID
    state.active_intent = "get_account_balance"
    state.consecutive_backend_failures = 1
    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator._run_model_loop(
        messages=[{"role": "user", "content": "What is my balance?"}],
        state=state,
        db=db,
        initial_response=_balance_tool_call("call-backend-2"),
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert result.executed_tools == ["escalate_to_human"]
    assert state.escalation_status == EscalationStatus.ESCALATED
    assert state.consecutive_backend_failures == 0
    assert executor.execute.await_count == 2

    escalation_call = executor.execute.await_args_list[1]
    assert escalation_call.args[0] == "escalate_to_human"
    assert escalation_call.args[1]["category"] == "repeated_backend_failure"
    assert escalation_call.args[1]["priority"] == "HIGH"
    assert escalation_call.args[1]["actions_not_completed"] == [
        "get_account_balance"
    ]
    assert escalation_call.args[1]["conversation_summary"] == (
        "Customer request could not be completed due to repeated "
        "backend failures."
    )
    assert "What is my balance?" not in escalation_call.args[1][
        "conversation_summary"
    ]


@pytest.mark.parametrize(
    "error",
    [
        ToolAuthenticationError("authentication required"),
        ToolConfirmationError("confirmation required"),
        ToolValidationError("invalid arguments"),
    ],
)
@pytest.mark.asyncio
async def test_non_backend_tool_errors_do_not_trigger_escalation(
    error: Exception,
) -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="I need different information to continue.",
                model="test-model",
            )
        ]
    )
    executor = AsyncMock()
    executor.execute.side_effect = error

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
        resource_resolver=NoopResourceResolver(),
    )
    state = _authenticated_state()
    state.active_account_id = ACCOUNT_ID
    state.active_intent = "get_account_balance"
    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator._run_model_loop(
        messages=[{"role": "user", "content": "What is my balance?"}],
        state=state,
        db=db,
        initial_response=_balance_tool_call("call-non-backend"),
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert state.consecutive_backend_failures == 0
    assert state.escalation_status == EscalationStatus.NONE
    assert executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_second_protected_backend_failure_escalates_without_retry() -> None:
    llm = SequenceLLM([])
    executor = AsyncMock()
    executor.execute.side_effect = [
        ToolBackendError("banking backend unavailable"),
        _escalation_output(),
    ]

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
        resource_resolver=NoopResourceResolver(),
    )

    state = _authenticated_state()
    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={"card_id": str(CARD_ID)},
    )
    state.consecutive_backend_failures = 1

    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator.handle_text_turn(
        user_text="Yes",
        state=state,
        db=db,
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert result.executed_tools == ["escalate_to_human"]
    assert state.escalation_status == EscalationStatus.ESCALATED
    assert state.pending_action is None
    assert state.consecutive_backend_failures == 0

    assert executor.execute.await_count == 2
    assert executor.execute.await_args_list[0].args[0] == "freeze_card"
    assert executor.execute.await_args_list[1].args[0] == "escalate_to_human"


@pytest.mark.asyncio
async def test_auto_escalation_failure_does_not_false_mark_escalated() -> None:
    llm = SequenceLLM([])
    executor = AsyncMock()
    executor.execute.side_effect = [
        ToolBackendError("banking backend unavailable"),
        ToolBackendError("support backend unavailable"),
    ]

    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
        resource_resolver=NoopResourceResolver(),
    )

    state = _authenticated_state()
    state.active_account_id = ACCOUNT_ID
    state.active_intent = "get_account_balance"
    state.consecutive_backend_failures = 1

    db = AsyncMock(spec=AsyncSession)

    result = await orchestrator._run_model_loop(
        messages=[{"role": "user", "content": "What is my balance?"}],
        state=state,
        db=db,
        initial_response=_balance_tool_call("call-escalation-failure"),
    )

    assert result.status == AgentTurnStatus.TOOL_ERROR
    assert state.escalation_status == EscalationStatus.NONE
    assert executor.execute.await_count == 2
    assert executor.execute.await_args_list[0].args[0] == "get_account_balance"
    assert executor.execute.await_args_list[1].args[0] == "escalate_to_human"
