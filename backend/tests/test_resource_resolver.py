from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.orchestrator import AgentOrchestrator, AgentTurnStatus
from backend.app.agent.resource_resolver import ResourceResolver
from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationPhase,
    ConversationState,
    ResourceType,
    VoicePlaybackStatus,
)
from backend.app.db.models import Account, Card, Transaction
from backend.app.providers.llm import LLMResponse, LLMToolCall
from backend.app.tools.schemas import (
    AccountBalanceOutput,
    FreezeCardOutput,
    TransactionDetailsOutput,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_CUSTOMER_ID = UUID("22222222-2222-4222-8222-222222222222")
CHECKING_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
SAVINGS_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")
FROZEN_CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc2")
OTHER_CARD_ID = UUID("dddddddd-dddd-4ddd-8ddd-ddddddddddd1")
TRANSACTION_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeee1")
OTHER_TRANSACTION_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeee2")


class SequenceLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


def _state() -> ConversationState:
    return ConversationState(
        session_id="session-001",
        customer_id=CUSTOMER_ID,
        authentication_level=AuthenticationLevel.AUTHENTICATED,
    )


def _account(
    account_id: UUID,
    account_type: str,
    masked_number: str,
    *,
    customer_id: UUID = CUSTOMER_ID,
) -> Account:
    return Account(
        account_id=account_id,
        customer_id=customer_id,
        account_type=account_type,
        masked_account_number=masked_number,
        current_balance=Decimal("2847.63"),
        available_balance=Decimal("2612.44"),
        currency="USD",
        status="ACTIVE",
    )


def _card(
    card_id: UUID,
    masked_number: str,
    status: str,
    *,
    customer_id: UUID = CUSTOMER_ID,
) -> Card:
    return Card(
        card_id=card_id,
        customer_id=customer_id,
        account_id=CHECKING_ID,
        masked_card_number=masked_number,
        card_type="DEBIT",
        status=status,
        expiration_month=8,
        expiration_year=2029,
    )


def _transaction(
    transaction_id: UUID,
    amount: str,
    day: int,
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id,
        account_id=CHECKING_ID,
        card_id=CARD_ID,
        merchant_name="ABC Electronics",
        merchant_category="Electronics",
        amount=Decimal(amount),
        currency="USD",
        transaction_timestamp=datetime(2026, 9, day, 14, 12, tzinfo=UTC),
        posted_timestamp=datetime(2026, 9, day + 1, 6, 0, tzinfo=UTC),
        status="POSTED",
        transaction_type="CARD_PURCHASE",
        location="Newark, NJ",
    )


def _db_with_scalar_results(*collections: list[object]) -> AsyncMock:
    db = AsyncMock(spec=AsyncSession)
    results = []

    for collection in collections:
        result = MagicMock()
        result.all.return_value = collection
        results.append(result)

    db.scalars.side_effect = results
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("account_type", "expected_id"),
    [
        ("checking", CHECKING_ID),
        ("savings", SAVINGS_ID),
    ],
)
async def test_account_type_resolves_owned_account(
    account_type: str,
    expected_id: UUID,
) -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _account(CHECKING_ID, "checking", "****4101"),
            _account(SAVINGS_ID, "savings", "****9204"),
        ]
    )

    result = await ResourceResolver().resolve(
        user_text=f"What is my {account_type} balance?",
        state=state,
        db=db,
    )

    assert result.clarification is None
    assert state.active_account_id == expected_id
    assert state.active_intent == "get_account_balance"


@pytest.mark.asyncio
async def test_barge_in_correction_replaces_active_checking_with_savings() -> None:
    state = _state()
    state.active_intent = "get_account_balance"
    state.active_account_id = CHECKING_ID
    state.record_voice_playback(
        speech_id="speech-checking",
        voice_turn_id="turn-checking",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=60,
    )
    db = _db_with_scalar_results(
        [
            _account(CHECKING_ID, "checking", "****4101"),
            _account(SAVINGS_ID, "savings", "****9204"),
        ]
    )

    result = await ResourceResolver().resolve(
        user_text="No, I meant savings.",
        state=state,
        db=db,
    )

    assert result.clarification is None
    assert state.active_account_id == SAVINGS_ID
    assert state.active_intent == "get_account_balance"


@pytest.mark.asyncio
async def test_ambiguous_card_does_not_reuse_or_select_a_card() -> None:
    state = _state()
    state.active_card_id = CARD_ID
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )

    result = await ResourceResolver().resolve(
        user_text="Freeze my card",
        state=state,
        db=db,
    )

    assert state.active_card_id is None
    assert state.active_intent == "freeze_card"
    assert result.clarification == (
        "I found two debit cards ending in 1842 and 6620. Which one do you mean?"
    )
    assert state.pending_resource_resolution is not None
    assert state.pending_resource_resolution.resource_type == ResourceType.CARD
    assert state.pending_resource_resolution.intent == "freeze_card"
    assert {
        candidate.resource_id
        for candidate in state.pending_resource_resolution.candidates
    } == {CARD_ID, FROZEN_CARD_ID}
    assert all(
        "ending in" in candidate.label
        for candidate in state.pending_resource_resolution.candidates
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [
        "1842",
        "the one ending in 1842",
        "card ending 1842",
        "the first one",
    ],
)
async def test_stored_card_candidate_selection_creates_pending_action(
    selection: str,
) -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )
    resolver = ResourceResolver()

    await resolver.resolve(
        user_text="Freeze my card",
        state=state,
        db=db,
    )
    result = await resolver.resolve(
        user_text=selection,
        state=state,
        db=db,
    )

    assert result.clarification is None
    assert db.scalars.await_count == 1
    assert state.pending_resource_resolution is None
    assert state.active_card_id == CARD_ID
    assert state.pending_action is not None
    assert state.pending_action.action == "freeze_card"
    assert state.pending_action.resource_id == CARD_ID
    assert state.pending_action.arguments == {"card_id": str(CARD_ID)}
    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION


@pytest.mark.asyncio
async def test_masked_suffix_resolves_one_owned_card() -> None:
    state = _state()
    state.active_intent = "freeze_card"
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )

    result = await ResourceResolver().resolve(
        user_text="The one ending in 1842",
        state=state,
        db=db,
    )

    assert result.clarification is None
    assert state.active_card_id == CARD_ID
    assert state.active_intent == "freeze_card"


@pytest.mark.asyncio
async def test_cross_customer_cards_are_never_considered() -> None:
    state = _state()
    state.active_intent = "freeze_card"
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(
                OTHER_CARD_ID,
                "****4407",
                "ACTIVE",
                customer_id=OTHER_CUSTOMER_ID,
            ),
        ]
    )

    result = await ResourceResolver().resolve(
        user_text="The card ending in 4407",
        state=state,
        db=db,
    )

    assert state.active_card_id is None
    assert result.clarification == (
        "I couldn’t find a card ending in 4407 for you."
    )


@pytest.mark.asyncio
async def test_duplicate_merchant_transactions_trigger_clarification() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [_account(CHECKING_ID, "checking", "****4101")],
        [
            _transaction(TRANSACTION_ID, "274.19", 20),
            _transaction(OTHER_TRANSACTION_ID, "276.04", 19),
        ],
    )

    result = await ResourceResolver().resolve(
        user_text="What was that ABC Electronics charge?",
        state=state,
        db=db,
    )

    assert state.active_transaction_id is None
    assert result.clarification is not None
    assert "$274.19 on September 20, 2026" in result.clarification
    assert "$276.04 on September 19, 2026" in result.clarification
    assert "UUID" not in result.clarification
    assert state.pending_resource_resolution is not None
    assert (
        state.pending_resource_resolution.resource_type
        == ResourceType.TRANSACTION
    )


@pytest.mark.asyncio
async def test_account_clarification_follow_up_uses_stored_candidates() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _account(CHECKING_ID, "checking", "****4101"),
            _account(SAVINGS_ID, "savings", "****9204"),
        ]
    )
    resolver = ResourceResolver()

    first = await resolver.resolve(
        user_text="Show my recent transactions",
        state=state,
        db=db,
    )
    second = await resolver.resolve(
        user_text="Checking",
        state=state,
        db=db,
    )

    assert first.clarification is not None
    assert second.clarification is None
    assert db.scalars.await_count == 1
    assert state.pending_resource_resolution is None
    assert state.active_account_id == CHECKING_ID
    assert state.active_intent == "get_recent_transactions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "expected_id"),
    [
        ("$274.19", TRANSACTION_ID),
        ("274.19", TRANSACTION_ID),
        ("the $274.19 one", TRANSACTION_ID),
        ("September 20", TRANSACTION_ID),
        ("September 20, 2026", TRANSACTION_ID),
        ("the first one", TRANSACTION_ID),
        ("the second one", OTHER_TRANSACTION_ID),
    ],
)
async def test_transaction_follow_up_uses_stored_candidates(
    selection: str,
    expected_id: UUID,
) -> None:
    state = _state()
    db = _db_with_scalar_results(
        [_account(CHECKING_ID, "checking", "****4101")],
        [
            _transaction(TRANSACTION_ID, "274.19", 20),
            _transaction(OTHER_TRANSACTION_ID, "276.04", 19),
        ],
    )
    resolver = ResourceResolver()

    await resolver.resolve(
        user_text="What was that ABC Electronics charge?",
        state=state,
        db=db,
    )
    result = await resolver.resolve(
        user_text=selection,
        state=state,
        db=db,
    )

    assert result.clarification is None
    assert result.selected_from_pending is True
    assert db.scalars.await_count == 2
    assert state.pending_resource_resolution is None
    assert state.active_transaction_id == expected_id
    assert state.active_intent == "get_transaction_details"


@pytest.mark.asyncio
async def test_transaction_selection_cannot_escape_stored_candidates() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [_account(CHECKING_ID, "checking", "****4101")],
        [
            _transaction(TRANSACTION_ID, "274.19", 20),
            _transaction(OTHER_TRANSACTION_ID, "276.04", 19),
        ],
    )
    resolver = ResourceResolver()

    first = await resolver.resolve(
        user_text="What was that ABC Electronics charge?",
        state=state,
        db=db,
    )
    candidate_ids = {
        candidate.resource_id
        for candidate in state.pending_resource_resolution.candidates
    }
    second = await resolver.resolve(
        user_text="$999.99",
        state=state,
        db=db,
    )

    assert second.clarification == first.clarification
    assert db.scalars.await_count == 2
    assert state.active_transaction_id is None
    assert state.pending_resource_resolution is not None
    assert {
        candidate.resource_id
        for candidate in state.pending_resource_resolution.candidates
    } == candidate_ids


@pytest.mark.asyncio
async def test_unrelated_new_turn_clears_stale_resource_clarification() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )
    resolver = ResourceResolver()

    await resolver.resolve(
        user_text="Freeze my card",
        state=state,
        db=db,
    )
    state.record_voice_playback(
        speech_id="speech-clarification",
        voice_turn_id="turn-clarification",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=60,
    )
    assert state.pending_resource_resolution is not None
    result = await resolver.resolve(
        user_text="Hello there",
        state=state,
        db=db,
    )

    assert result.kind is None
    assert result.clarification is None
    assert db.scalars.await_count == 1
    assert state.pending_resource_resolution is None
    assert state.active_intent is None
    assert state.active_card_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_text", "expected_id"),
    [
        ("The ABC Electronics charge for $274.19", TRANSACTION_ID),
        ("ABC Electronics on September 19, 2026", OTHER_TRANSACTION_ID),
    ],
)
async def test_merchant_with_amount_or_date_resolves_transaction(
    user_text: str,
    expected_id: UUID,
) -> None:
    state = _state()
    db = _db_with_scalar_results(
        [_account(CHECKING_ID, "checking", "****4101")],
        [
            _transaction(TRANSACTION_ID, "274.19", 20),
            _transaction(OTHER_TRANSACTION_ID, "276.04", 19),
        ],
    )

    result = await ResourceResolver().resolve(
        user_text=user_text,
        state=state,
        db=db,
    )

    assert result.clarification is None
    assert state.active_transaction_id == expected_id
    assert state.active_account_id == CHECKING_ID


@pytest.mark.asyncio
async def test_resolved_account_feeds_balance_tool_with_authoritative_id() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _account(CHECKING_ID, "checking", "****4101"),
            _account(SAVINGS_ID, "savings", "****9204"),
        ]
    )
    llm = SequenceLLM(
        [
            LLMResponse(
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-balance",
                        name="get_account_balance",
                        arguments={"account_id": str(SAVINGS_ID)},
                    )
                ],
            ),
            LLMResponse(content="Your available balance is $2,612.44.", model="test-model"),
        ]
    )
    executor = AsyncMock()
    executor.execute.return_value = AccountBalanceOutput(
        account_id=CHECKING_ID,
        account_type="checking",
        masked_account_number="****4101",
        current_balance=Decimal("2847.63"),
        available_balance=Decimal("2612.44"),
        currency="USD",
        status="ACTIVE",
    )
    orchestrator = AgentOrchestrator(llm=llm, tool_executor=executor)

    result = await orchestrator.handle_text_turn(
        user_text="What is my checking balance?",
        state=state,
        db=db,
    )

    assert result.executed_tools == ["get_account_balance"]
    assert executor.execute.await_args.args[1]["account_id"] == str(CHECKING_ID)


@pytest.mark.asyncio
async def test_resolved_card_feeds_protected_proposal_and_still_waits() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )
    llm = SequenceLLM(
        [
            LLMResponse(
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-freeze",
                        name="freeze_card",
                        arguments={"card_id": str(FROZEN_CARD_ID)},
                    )
                ],
            )
        ]
    )
    executor = AsyncMock()
    orchestrator = AgentOrchestrator(llm=llm, tool_executor=executor)

    result = await orchestrator.handle_text_turn(
        user_text="Freeze the card ending in 1842",
        state=state,
        db=db,
    )

    assert result.status == AgentTurnStatus.WAITING_FOR_CONFIRMATION
    assert state.pending_action is not None
    assert state.pending_action.resource_id == CARD_ID
    assert state.pending_action.arguments == {"card_id": str(CARD_ID)}
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_stored_transaction_selection_executes_details_tool() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [_account(CHECKING_ID, "checking", "****4101")],
        [
            _transaction(TRANSACTION_ID, "274.19", 20),
            _transaction(OTHER_TRANSACTION_ID, "276.04", 19),
        ],
    )
    llm = SequenceLLM(
        [
            LLMResponse(
                content=(
                    "The ABC Electronics transaction was $274.19 "
                    "on September 20, 2026."
                ),
                model="test-model",
            )
        ]
    )
    executor = AsyncMock()
    executor.execute.return_value = TransactionDetailsOutput(
        transaction_id=TRANSACTION_ID,
        account_id=CHECKING_ID,
        card_id=CARD_ID,
        merchant_name="ABC Electronics",
        merchant_category="Electronics",
        amount=Decimal("274.19"),
        currency="USD",
        transaction_timestamp=datetime(
            2026,
            9,
            20,
            14,
            12,
            tzinfo=UTC,
        ),
        posted_timestamp=datetime(
            2026,
            9,
            21,
            6,
            0,
            tzinfo=UTC,
        ),
        status="POSTED",
        transaction_type="CARD_PURCHASE",
        location="Newark, NJ",
    )
    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=executor,
    )

    clarification = await orchestrator.handle_text_turn(
        user_text="What was that ABC Electronics charge?",
        state=state,
        db=db,
    )
    result = await orchestrator.handle_text_turn(
        user_text="$274.19",
        state=state,
        db=db,
    )

    assert clarification.text.startswith(
        "I found two ABC Electronics transactions"
    )
    assert result.executed_tools == ["get_transaction_details"]
    assert str(TRANSACTION_ID) not in result.text
    assert state.pending_resource_resolution is None
    executor.execute.assert_awaited_once()
    call = executor.execute.await_args
    assert call.args[0] == "get_transaction_details"
    assert call.args[1] == {"transaction_id": str(TRANSACTION_ID)}
    assert len(llm.calls) == 1
    assert "$274.19" not in llm.calls[0]["messages"][2]["content"]
    assert any(
        message["role"] == "tool"
        for message in llm.calls[0]["messages"]
    )


@pytest.mark.asyncio
async def test_customer_facing_response_never_requests_uuid() -> None:
    state = _state()
    db = AsyncMock(spec=AsyncSession)
    llm = SequenceLLM(
        [
            LLMResponse(
                content="Please provide your card UUID.",
                model="test-model",
            )
        ]
    )

    result = await AgentOrchestrator(llm=llm).handle_text_turn(
        user_text="Hello",
        state=state,
        db=db,
    )

    assert "UUID" not in result.text
    assert "customer-friendly terms" in result.text


@pytest.mark.asyncio
async def test_freeze_clarification_selection_and_cancel_are_deterministic() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )
    llm = SequenceLLM([])
    executor = AsyncMock()
    orchestrator = AgentOrchestrator(llm=llm, tool_executor=executor)

    clarification = await orchestrator.handle_text_turn(
        user_text="Freeze my card",
        state=state,
        db=db,
    )
    confirmation_prompt = await orchestrator.handle_text_turn(
        user_text="1842",
        state=state,
        db=db,
    )

    assert clarification.text == (
        "I found two debit cards ending in 1842 and 6620. Which one do you mean?"
    )
    assert confirmation_prompt.status == AgentTurnStatus.WAITING_FOR_CONFIRMATION
    assert "freeze" in confirmation_prompt.text.casefold()
    assert state.pending_action is not None
    assert state.pending_action.resource_id == CARD_ID
    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION
    assert llm.calls == []

    cancellation = await orchestrator.handle_text_turn(
        user_text="Cancel",
        state=state,
        db=db,
    )

    assert cancellation.text == "Okay, I won’t perform that action."
    assert state.pending_action is None
    assert state.active_card_id is None
    executor.execute.assert_not_awaited()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_confirm_executes_exactly_the_selected_card() -> None:
    state = _state()
    db = _db_with_scalar_results(
        [
            _card(CARD_ID, "****1842", "ACTIVE"),
            _card(FROZEN_CARD_ID, "****6620", "FROZEN"),
        ]
    )
    llm = SequenceLLM(
        [
            LLMResponse(
                content="Your card ending in 1842 is now frozen.",
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
    orchestrator = AgentOrchestrator(llm=llm, tool_executor=executor)

    await orchestrator.handle_text_turn(
        user_text="Freeze my card",
        state=state,
        db=db,
    )
    await orchestrator.handle_text_turn(
        user_text="the one ending in 1842",
        state=state,
        db=db,
    )
    result = await orchestrator.handle_text_turn(
        user_text="Confirm",
        state=state,
        db=db,
    )

    assert result.executed_tools == ["freeze_card"]
    assert state.pending_action is None
    executor.execute.assert_awaited_once()
    call = executor.execute.await_args
    assert call.args[0] == "freeze_card"
    assert call.args[1] == {"card_id": str(CARD_ID)}

    context = call.args[2]
    assert context.confirmation is not None
    assert context.confirmation.action == "freeze_card"
    assert context.confirmation.resource_id == CARD_ID
    assert context.confirmation.resource_id != FROZEN_CARD_ID


@pytest.mark.asyncio
async def test_confirmation_language_is_blocked_without_pending_action() -> None:
    state = _state()
    db = AsyncMock(spec=AsyncSession)
    llm = SequenceLLM(
        [
            LLMResponse(
                content=(
                    "I just want to confirm—do you want to freeze "
                    "the card ending in 1842?"
                ),
                model="test-model",
            )
        ]
    )

    result = await AgentOrchestrator(llm=llm).handle_text_turn(
        user_text="Can you help me?",
        state=state,
        db=db,
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert state.pending_action is None
    assert "confirm" not in result.text.casefold()
    assert "would you like" not in result.text.casefold()
    assert "do you want" not in result.text.casefold()


@pytest.mark.asyncio
async def test_post_action_llm_cannot_create_false_confirmation_state() -> None:
    state = _state()
    state.active_card_id = CARD_ID
    state.active_intent = "freeze_card"
    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={"card_id": str(CARD_ID)},
    )
    db = AsyncMock(spec=AsyncSession)
    llm = SequenceLLM(
        [
            LLMResponse(
                content=(
                    "The card is frozen.\n"
                    "Would you like me to freeze another card?"
                ),
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
        resource_resolver=AsyncMock(),
    )

    result = await orchestrator.handle_text_turn(
        user_text="Confirm",
        state=state,
        db=db,
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert state.pending_action is None
    assert "would you like" not in result.text.casefold()
