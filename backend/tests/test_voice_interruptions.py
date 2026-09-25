from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from backend.app.auth.sessions import InMemorySessionStore, get_session_store
from backend.app.conversation.state import (
    ConversationPhase,
    ConversationState,
    VoicePlaybackStatus,
)
from backend.app.main import app

CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")


@pytest.fixture
def playback_api():
    store = InMemorySessionStore()
    app.dependency_overrides[get_session_store] = lambda: store

    with TestClient(app) as client:
        yield client, store

    app.dependency_overrides.clear()


def _scheduled_payload(*, sequence: int = 1) -> dict[str, object]:
    return {
        "speech_id": f"speech-{sequence}",
        "voice_turn_id": f"turn-{sequence}",
        "sequence": sequence,
        "status": "SCHEDULED",
        "response_phase": "AGENT_SPEAKING",
    }


def test_scheduled_playback_is_recorded_authoritatively(playback_api) -> None:
    client, store = playback_api
    state = store.create()
    state.phase = ConversationPhase.AGENT_SPEAKING

    response = client.post(
        f"/sessions/{state.session_id}/voice/playback",
        json=_scheduled_payload(),
    )

    assert response.status_code == 200
    assert response.json()["voice_playback"] == {
        "speech_id": "speech-1",
        "voice_turn_id": "turn-1",
        "sequence": 1,
        "status": "SCHEDULED",
        "interruption_stop_latency_ms": None,
    }


def test_interrupted_playback_sets_phase_and_latency(playback_api) -> None:
    client, store = playback_api
    state = store.create()
    state.phase = ConversationPhase.AGENT_SPEAKING
    client.post(
        f"/sessions/{state.session_id}/voice/playback",
        json=_scheduled_payload(),
    )

    response = client.post(
        f"/sessions/{state.session_id}/voice/playback",
        json={
            "speech_id": "speech-1",
            "voice_turn_id": "turn-1",
            "sequence": 1,
            "status": "INTERRUPTED",
            "interruption_stop_latency_ms": 87.5,
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_phase"] == "INTERRUPTED"
    assert response.json()["voice_playback"]["status"] == "INTERRUPTED"
    assert response.json()["voice_playback"][
        "interruption_stop_latency_ms"
    ] == 87.5


def test_voice_interruption_preserves_unconfirmed_protected_action() -> None:
    state = ConversationState(session_id="session-1")
    state.request_action("freeze_card", CARD_ID)

    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=50,
    )

    assert state.phase == ConversationPhase.INTERRUPTED
    assert state.pending_action is not None
    assert state.pending_action.action == "freeze_card"
    assert state.pending_action.confirmation_received is False


def test_interruption_after_completed_action_does_not_restore_action() -> None:
    state = ConversationState(session_id="session-1")
    state.request_action("freeze_card", CARD_ID)
    state.confirm_pending_action()
    state.complete_pending_action("freeze_card", CARD_ID)
    state.last_tool_result = {"status": "FROZEN"}

    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=50,
    )

    assert state.pending_action is None
    assert state.last_tool_result == {"status": "FROZEN"}


def test_completed_playback_returns_agent_speaking_to_listening() -> None:
    state = ConversationState(
        session_id="session-1",
        phase=ConversationPhase.AGENT_SPEAKING,
    )
    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.SCHEDULED,
        response_phase=ConversationPhase.AGENT_SPEAKING,
    )

    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.COMPLETED,
    )

    assert state.phase == ConversationPhase.LISTENING


def test_completed_confirmation_prompt_keeps_waiting_phase() -> None:
    state = ConversationState(session_id="session-1")
    state.request_action("freeze_card", CARD_ID)
    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.SCHEDULED,
        response_phase=ConversationPhase.WAITING_FOR_CONFIRMATION,
    )

    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.COMPLETED,
    )

    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION
    assert state.pending_action is not None


def test_duplicate_terminal_event_is_an_exact_no_op() -> None:
    state = ConversationState(session_id="session-1")
    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=50,
    )

    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.COMPLETED,
    )

    assert state.phase == ConversationPhase.INTERRUPTED
    assert state.voice_playback is not None
    assert state.voice_playback.status == VoicePlaybackStatus.INTERRUPTED
    assert state.voice_playback.interruption_stop_latency_ms == 50


def test_stale_interruption_cannot_overwrite_newer_speech() -> None:
    state = ConversationState(
        session_id="session-1",
        phase=ConversationPhase.AGENT_SPEAKING,
    )
    state.record_voice_playback(
        speech_id="speech-2",
        voice_turn_id="turn-2",
        sequence=2,
        status=VoicePlaybackStatus.SCHEDULED,
        response_phase=ConversationPhase.AGENT_SPEAKING,
    )

    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=50,
    )

    assert state.phase == ConversationPhase.AGENT_SPEAKING
    assert state.voice_playback is not None
    assert state.voice_playback.speech_id == "speech-2"


def test_new_corrected_speech_recovers_from_crossed_interruption() -> None:
    state = ConversationState(session_id="session-1")
    state.record_voice_playback(
        speech_id="speech-1",
        voice_turn_id="turn-1",
        sequence=1,
        status=VoicePlaybackStatus.INTERRUPTED,
        interruption_stop_latency_ms=50,
    )

    state.record_voice_playback(
        speech_id="speech-2",
        voice_turn_id="turn-2",
        sequence=2,
        status=VoicePlaybackStatus.SCHEDULED,
        response_phase=ConversationPhase.WAITING_FOR_CONFIRMATION,
    )

    assert state.phase == ConversationPhase.WAITING_FOR_CONFIRMATION
    assert state.voice_playback is not None
    assert state.voice_playback.speech_id == "speech-2"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "speech_id": "speech-1",
            "voice_turn_id": "turn-1",
            "sequence": 1,
            "status": "INTERRUPTED",
        },
        {
            "speech_id": "speech-1",
            "voice_turn_id": "turn-1",
            "sequence": 1,
            "status": "COMPLETED",
            "interruption_stop_latency_ms": 10,
        },
        {
            "speech_id": "speech-1",
            "voice_turn_id": "turn-1",
            "sequence": 1,
            "status": "SCHEDULED",
        },
    ],
)
def test_playback_endpoint_rejects_inconsistent_events(
    playback_api,
    payload,
) -> None:
    client, store = playback_api
    state = store.create()

    response = client.post(
        f"/sessions/{state.session_id}/voice/playback",
        json=payload,
    )

    assert response.status_code == 422
    assert state.voice_playback is None


def test_playback_endpoint_rejects_unknown_session(playback_api) -> None:
    client, _ = playback_api

    response = client.post(
        "/sessions/not-found/voice/playback",
        json=_scheduled_payload(),
    )

    assert response.status_code == 404


def test_playback_endpoint_rejects_extra_identity_data(playback_api) -> None:
    client, store = playback_api
    state = store.create()
    payload = _scheduled_payload()
    payload["customer_id"] = str(CARD_ID)

    response = client.post(
        f"/sessions/{state.session_id}/voice/playback",
        json=payload,
    )

    assert response.status_code == 422
    assert state.voice_playback is None
