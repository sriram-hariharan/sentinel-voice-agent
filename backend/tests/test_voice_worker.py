import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from livekit.agents import StopResponse, llm
from pydantic import SecretStr

from backend.app.config.settings import Settings
from backend.app.voice.bridge import VoiceBridgeError, VoiceTurnResult
from backend.app.voice.worker import (
    VOICE_BACKEND_ERROR_MESSAGE,
    VOICE_PLAYBACK_ERROR_MESSAGE,
    VOICE_TURN_HANDLING,
    SentinelVoiceAgent,
    VoiceWorkerConfigurationError,
    _groq_api_key,
    _TrackedSpeech,
    _voice_room_options,
    parse_voice_session_metadata,
)


class FakeTextOutput:
    def __init__(self) -> None:
        self.segments: list[str] = []
        self.flush_count = 0

    async def capture_text(self, text: str) -> None:
        self.segments.append(text)

    def flush(self) -> None:
        self.flush_count += 1


class FakeSpeechHandle:
    def __init__(self, *, speech_id: str) -> None:
        self.id = speech_id
        self.awaited = False
        self.done = False
        self.interrupted = False
        self.interrupt_sources: list[str] = []
        self._error: Exception | None = None
        self._callbacks = []

    def __await__(self):
        self.awaited = True

        async def unexpected_wait() -> None:
            raise AssertionError(
                "the finalized-turn hook must not await speech playout"
            )

        return unexpected_wait().__await__()

    def exception(self) -> Exception | None:
        assert self.done
        return self._error

    def add_done_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def interrupt(self, *, source: str) -> None:
        if self.done:
            return
        self.interrupted = True
        self.interrupt_sources.append(source)
        self.complete()

    def complete(self, error: Exception | None = None) -> None:
        self._error = error
        self.done = True
        for callback in self._callbacks:
            callback(self)


class FakeVoiceSession:
    def __init__(self, *, say_error: Exception | None = None) -> None:
        self.transcription = FakeTextOutput()
        self.output = SimpleNamespace(
            transcription=self.transcription,
            transcription_enabled=True,
        )
        self.say_error = say_error
        self.say_calls: list[dict[str, object]] = []
        self.assistant_transcripts: list[str] = []
        self.speech_handles: list[FakeSpeechHandle] = []

    def say(self, text: str, *, allow_interruptions: bool) -> FakeSpeechHandle:
        if self.say_error is not None:
            raise self.say_error

        self.say_calls.append(
            {
                "text": text,
                "allow_interruptions": allow_interruptions,
            }
        )
        # LiveKit's unsynchronized text output forwards this independently of
        # the TTS/playout completion represented by the unfinished handle.
        self.assistant_transcripts.append(text)
        handle = FakeSpeechHandle(
            speech_id=f"speech-{len(self.speech_handles) + 1}"
        )
        self.speech_handles.append(handle)
        return handle


def _agent_with_session(
    *,
    bridge: AsyncMock,
    session: FakeVoiceSession | None = None,
    publish_event: AsyncMock | None = None,
    clock=None,
) -> tuple[SentinelVoiceAgent, FakeVoiceSession]:
    active_session = session or FakeVoiceSession()
    agent = SentinelVoiceAgent(
        bridge=bridge,
        session_id="opaque-session-id",
        publish_event=publish_event,
        **({"clock": clock} if clock is not None else {}),
    )
    agent._activity = SimpleNamespace(session=active_session)
    return agent, active_session


def _voice_result() -> VoiceTurnResult:
    return VoiceTurnResult(
        session_id="opaque-session-id",
        message="Your checking balance is $125.00.",
        turn_status="RESPONDED",
        conversation_phase="AGENT_SPEAKING",
    )


async def _invoke_finalized_turn(
    agent: SentinelVoiceAgent,
    message: llm.ChatMessage,
) -> None:
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(llm.ChatContext.empty(), message)


def test_voice_worker_accepts_only_opaque_session_metadata() -> None:
    metadata = parse_voice_session_metadata(
        json.dumps({"sentinelvoice_session_id": "opaque-session-id"})
    )

    assert metadata.sentinelvoice_session_id == "opaque-session-id"


@pytest.mark.parametrize(
    "raw_metadata",
    [
        "not-json",
        json.dumps({}),
        json.dumps(
            {
                "sentinelvoice_session_id": "opaque-session-id",
                "customer_id": "must-not-cross-the-boundary",
            }
        ),
    ],
)
def test_voice_worker_rejects_unsafe_or_invalid_metadata(
    raw_metadata: str,
) -> None:
    with pytest.raises(
        VoiceWorkerConfigurationError,
        match="dispatch metadata is invalid",
    ):
        parse_voice_session_metadata(raw_metadata)


def test_voice_worker_requires_groq_configuration() -> None:
    settings = Settings(_env_file=None, groq_api_key=None)

    with pytest.raises(VoiceWorkerConfigurationError, match="not configured"):
        _groq_api_key(settings)


def test_voice_worker_reads_configured_groq_key() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key=SecretStr("test-groq-key"),
    )

    assert _groq_api_key(settings) == "test-groq-key"


def test_voice_worker_publishes_text_independently_of_tts() -> None:
    options = _voice_room_options()

    assert options.get_text_output_options() is not None
    assert options.get_text_output_options().sync_transcription is False


def test_voice_worker_uses_supported_vad_interruption_options() -> None:
    assert VOICE_TURN_HANDLING == {
        "interruption": {
            "enabled": True,
            "mode": "vad",
            "min_duration": 0.35,
            "min_words": 0,
            "resume_false_interruption": False,
        }
    }


@pytest.mark.asyncio
async def test_finalized_turn_calls_bridge_once_and_starts_speech() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    agent, session = _agent_with_session(bridge=bridge)
    message = llm.ChatMessage(
        id="voice-turn-1",
        role="user",
        content=["  What is my checking balance?  "],
    )

    await _invoke_finalized_turn(agent, message)
    await _invoke_finalized_turn(agent, message)

    bridge.handle_transcript.assert_awaited_once_with(
        session_id="opaque-session-id",
        transcript="What is my checking balance?",
    )
    assert session.say_calls == [
        {
            "text": "Your checking balance is $125.00.",
            "allow_interruptions": True,
        }
    ]
    assert session.assistant_transcripts == [
        "Your checking balance is $125.00."
    ]
    assert session.speech_handles[0].awaited is False
    assert session.speech_handles[0].done is False


@pytest.mark.asyncio
async def test_repeated_words_in_distinct_turns_are_not_deduplicated() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    agent, session = _agent_with_session(bridge=bridge)

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="confirmation-1", role="user", content=["Yes"]),
    )
    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="confirmation-2", role="user", content=["Yes"]),
    )

    assert bridge.handle_transcript.await_count == 2
    assert len(session.say_calls) == 2
    assert session.speech_handles[0].interrupted is True
    assert session.speech_handles[1].interrupted is False


@pytest.mark.asyncio
async def test_empty_finalized_turn_does_nothing() -> None:
    bridge = AsyncMock()
    agent, session = _agent_with_session(bridge=bridge)
    message = llm.ChatMessage(
        id="empty-turn",
        role="user",
        content=[" \n "],
    )

    await _invoke_finalized_turn(agent, message)

    bridge.handle_transcript.assert_not_awaited()
    assert session.say_calls == []


@pytest.mark.asyncio
async def test_bridge_failure_publishes_error_without_tts() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.side_effect = VoiceBridgeError("backend failed")
    agent, session = _agent_with_session(bridge=bridge)
    message = llm.ChatMessage(
        id="failed-turn",
        role="user",
        content=["Show my balance"],
    )

    await _invoke_finalized_turn(agent, message)

    bridge.handle_transcript.assert_awaited_once()
    assert session.say_calls == []
    assert session.transcription.segments == [VOICE_BACKEND_ERROR_MESSAGE]
    assert session.transcription.flush_count == 1


@pytest.mark.asyncio
async def test_tts_failure_preserves_text_and_publishes_playback_error() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    session = FakeVoiceSession()
    agent, session = _agent_with_session(bridge=bridge, session=session)
    message = llm.ChatMessage(
        id="tts-failed-turn",
        role="user",
        content=["Show my balance"],
    )

    await _invoke_finalized_turn(agent, message)

    bridge.handle_transcript.assert_awaited_once()
    assert session.say_calls[0]["text"] == _voice_result().message
    assert session.assistant_transcripts == [_voice_result().message]
    assert session.transcription.segments == []

    session.speech_handles[0].complete(RuntimeError("tts unavailable"))
    await asyncio.sleep(0)

    assert session.transcription.segments == [VOICE_PLAYBACK_ERROR_MESSAGE]
    assert session.transcription.flush_count == 1


@pytest.mark.asyncio
async def test_tts_failure_during_speech_signal_does_not_duplicate_turn() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    agent, session = _agent_with_session(bridge=bridge)

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    agent.note_user_speaking()
    session.speech_handles[0].complete(RuntimeError("tts unavailable"))
    await asyncio.sleep(0)

    assert bridge.handle_transcript.await_count == 1
    assert agent._interruption_candidate_id is None
    interrupted_reports = [
        call
        for call in bridge.report_playback.await_args_list
        if call.kwargs["status"] == "INTERRUPTED"
    ]
    assert interrupted_reports == []


@pytest.mark.asyncio
async def test_new_turn_cancels_stale_responses_and_schedules_in_order() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.side_effect = [
        _voice_result().model_copy(update={"message": "Response A"}),
        _voice_result().model_copy(update={"message": "Response B"}),
        _voice_result().model_copy(update={"message": "Response C"}),
    ]
    agent, session = _agent_with_session(bridge=bridge)

    for turn_id, transcript in (
        ("turn-a", "Question A"),
        ("turn-b", "Question B"),
        ("turn-c", "Question C"),
    ):
        await _invoke_finalized_turn(
            agent,
            llm.ChatMessage(id=turn_id, role="user", content=[transcript]),
        )

    assert [call["text"] for call in session.say_calls] == [
        "Response A",
        "Response B",
        "Response C",
    ]
    assert session.assistant_transcripts == [
        "Response A",
        "Response B",
        "Response C",
    ]
    assert all(not handle.awaited for handle in session.speech_handles)
    assert [handle.interrupted for handle in session.speech_handles] == [
        True,
        True,
        False,
    ]


@pytest.mark.asyncio
async def test_vad_signal_waits_for_livekit_before_committing_interruption() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    agent, session = _agent_with_session(bridge=bridge)

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    agent.note_user_speaking()
    await asyncio.sleep(0)

    assert session.speech_handles[0].interrupted is False
    bridge.report_playback.assert_awaited()


@pytest.mark.asyncio
async def test_interruption_reports_measured_latency_and_browser_event() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    publish_event = AsyncMock()
    clock_values = iter([10.0, 10.123])
    agent, session = _agent_with_session(
        bridge=bridge,
        publish_event=publish_event,
        clock=lambda: next(clock_values),
    )

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    sequence = agent._speech_sequence
    agent.note_user_speaking()
    session.speech_handles[0].interrupt(source="audio_activity")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    bridge.report_playback.assert_any_await(
        session_id="opaque-session-id",
        speech_id="speech-1",
        voice_turn_id="turn-a",
        sequence=sequence,
        status="INTERRUPTED",
        interruption_stop_latency_ms=pytest.approx(123.0),
        response_phase=None,
    )
    publish_event.assert_awaited_once_with(
        {
            "type": "speech_interrupted",
            "speech_id": "speech-1",
            "voice_turn_id": "turn-a",
            "interruption_stop_latency_ms": pytest.approx(123.0),
        }
    )


@pytest.mark.asyncio
async def test_interruption_cancels_all_stale_queued_speech() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    agent, session = _agent_with_session(bridge=bridge)

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    for queued_turn in ("turn-b", "turn-c"):
        speech = session.say("Queued response", allow_interruptions=True)
        agent._speech_sequence += 1
        agent._speech[speech.id] = _TrackedSpeech(
            handle=speech,
            turn_id=queued_turn,
            sequence=agent._speech_sequence,
        )

    agent.note_user_speaking()
    session.speech_handles[0].interrupt(source="audio_activity")
    await asyncio.sleep(0)

    assert all(handle.interrupted for handle in session.speech_handles)
    assert all(
        handle.interrupt_sources
        for handle in session.speech_handles
    )


@pytest.mark.asyncio
async def test_completed_speech_reports_completion_without_interruption() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    publish_event = AsyncMock()
    agent, session = _agent_with_session(
        bridge=bridge,
        publish_event=publish_event,
    )

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    sequence = agent._speech_sequence
    session.speech_handles[0].complete()
    await asyncio.sleep(0)

    bridge.report_playback.assert_any_await(
        session_id="opaque-session-id",
        speech_id="speech-1",
        voice_turn_id="turn-a",
        sequence=sequence,
        status="COMPLETED",
        interruption_stop_latency_ms=None,
        response_phase=None,
    )
    publish_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_stt_after_interruption_never_creates_backend_turn() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    agent, session = _agent_with_session(bridge=bridge)

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    bridge.handle_transcript.reset_mock()
    agent.note_user_speaking()
    session.speech_handles[0].interrupt(source="audio_activity")

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="empty-after-interrupt", role="user", content=[" "]),
    )

    bridge.handle_transcript.assert_not_awaited()


@pytest.mark.asyncio
async def test_interrupted_handle_cannot_resume_or_emit_duplicate_event() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    publish_event = AsyncMock()
    agent, session = _agent_with_session(
        bridge=bridge,
        publish_event=publish_event,
    )

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    agent.note_user_speaking()
    handle = session.speech_handles[0]
    handle.interrupt(source="audio_activity")
    handle.interrupt(source="audio_activity")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    publish_event.assert_awaited_once()
    interrupted_reports = [
        call
        for call in bridge.report_playback.await_args_list
        if call.kwargs["status"] == "INTERRUPTED"
    ]
    assert len(interrupted_reports) == 1


@pytest.mark.asyncio
async def test_backend_interruption_notification_failure_is_contained() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    bridge.report_playback.side_effect = VoiceBridgeError("unavailable")
    publish_event = AsyncMock()
    agent, session = _agent_with_session(
        bridge=bridge,
        publish_event=publish_event,
    )

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="turn-a", role="user", content=["Question A"]),
    )
    agent.note_user_speaking()
    session.speech_handles[0].interrupt(source="audio_activity")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert session.speech_handles[0].interrupted is True
    publish_event.assert_awaited_once()


def test_interruption_signal_without_agent_speech_is_a_no_op() -> None:
    bridge = AsyncMock()
    agent, session = _agent_with_session(bridge=bridge)

    agent.note_user_speaking()

    assert session.speech_handles == []
    assert agent._interruption_candidate_id is None


@pytest.mark.asyncio
async def test_synchronous_speech_scheduling_failure_keeps_text_visible() -> None:
    bridge = AsyncMock()
    bridge.handle_transcript.return_value = _voice_result()
    session = FakeVoiceSession(say_error=RuntimeError("output unavailable"))
    agent, session = _agent_with_session(bridge=bridge, session=session)

    await _invoke_finalized_turn(
        agent,
        llm.ChatMessage(id="schedule-failure", role="user", content=["Balance"]),
    )

    assert session.transcription.segments == [
        _voice_result().message,
        VOICE_PLAYBACK_ERROR_MESSAGE,
    ]
    assert session.transcription.flush_count == 2
