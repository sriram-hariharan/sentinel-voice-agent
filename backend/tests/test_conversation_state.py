from uuid import UUID

import pytest

from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationPhase,
    ConversationState,
    ConversationStateError,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")
OTHER_CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc2")


def test_conversation_phase_contains_required_v1_states() -> None:
    assert {state.value for state in ConversationPhase} == {
        "LISTENING",
        "USER_SPEAKING",
        "PROCESSING",
        "TOOL_EXECUTION",
        "AGENT_SPEAKING",
        "INTERRUPTED",
        "WAITING_FOR_CONFIRMATION",
        "ESCALATING",
        "ENDED",
        "FAILED",
    }


def test_conversation_starts_listening_and_unauthenticated() -> None:
    state = ConversationState(session_id="session-001")

    assert state.phase == ConversationPhase.LISTENING
    assert state.authenticated is False
    assert state.pending_action is None


def test_confirmation_is_bound_to_pending_action_and_resource() -> None:
    state = ConversationState(
        session_id="session-001",
        customer_id=CUSTOMER_ID,
        authentication_level=AuthenticationLevel.AUTHENTICATED,
    )

    state.request_action("freeze_card", CARD_ID)

    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION
    assert state.to_tool_context().confirmation is None

    state.confirm_pending_action()

    context = state.to_tool_context()

    assert context.confirmation is not None
    assert context.confirmation.action == "freeze_card"
    assert context.confirmation.resource_id == CARD_ID
    assert context.confirmation.confirmed is True


def test_replacing_pending_action_invalidates_old_confirmation() -> None:
    state = ConversationState(
        session_id="session-001",
        customer_id=CUSTOMER_ID,
        authentication_level=AuthenticationLevel.AUTHENTICATED,
    )

    state.request_action("freeze_card", CARD_ID)
    state.confirm_pending_action()

    assert state.to_tool_context().confirmation is not None

    state.request_action("freeze_card", OTHER_CARD_ID)

    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION
    assert state.to_tool_context().confirmation is None
    assert state.pending_action is not None
    assert state.pending_action.resource_id == OTHER_CARD_ID
    assert state.pending_action.confirmation_received is False


def test_stale_tool_completion_cannot_clear_new_pending_action() -> None:
    state = ConversationState(session_id="session-001")

    state.request_action("freeze_card", CARD_ID)
    state.confirm_pending_action()

    state.request_action("freeze_card", OTHER_CARD_ID)

    with pytest.raises(ConversationStateError):
        state.complete_pending_action(
            "freeze_card",
            CARD_ID,
        )

    assert state.pending_action is not None
    assert state.pending_action.resource_id == OTHER_CARD_ID


def test_interruption_cancels_action_not_yet_executing() -> None:
    state = ConversationState(session_id="session-001")

    state.request_action("freeze_card", CARD_ID)

    state.interrupt()

    assert state.phase == ConversationPhase.INTERRUPTED
    assert state.pending_action is None


def test_interruption_tracks_action_already_executing() -> None:
    state = ConversationState(session_id="session-001")

    state.request_action("freeze_card", CARD_ID)
    state.confirm_pending_action()
    state.phase = ConversationPhase.TOOL_EXECUTION

    state.interrupt()

    assert state.phase == ConversationPhase.INTERRUPTED
    assert state.pending_action is not None
    assert state.pending_action.resource_id == CARD_ID


def test_tool_context_never_marks_unverified_identity_authenticated() -> None:
    state = ConversationState(
        session_id="session-001",
        customer_id=CUSTOMER_ID,
    )

    context = state.to_tool_context()

    assert context.customer_id == CUSTOMER_ID
    assert context.authenticated is False


def test_pending_action_preserves_validated_arguments() -> None:
    state = ConversationState(session_id="session-001")

    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={
            "card_id": str(CARD_ID),
        },
    )

    assert state.pending_action is not None
    assert state.pending_action.arguments == {
        "card_id": str(CARD_ID),
    }

    state.confirm_pending_action()

    context = state.to_tool_context()

    assert context.confirmation is not None
    assert context.confirmation.resource_id == CARD_ID
