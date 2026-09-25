import asyncio
import json
import logging
import os
import time
from collections import deque
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    StopResponse,
    llm,
)
from livekit.agents.voice.room_io import RoomOptions, TextOutputOptions
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.config.settings import Settings, get_settings
from backend.app.providers.groq_speech import (
    GroqSpeechToTextProvider,
    GroqTextToSpeechProvider,
)
from backend.app.voice.adapters import GroqSTTAdapter, GroqTTSAdapter
from backend.app.voice.bridge import VoiceBridge, VoiceBridgeError

logger = logging.getLogger(__name__)

VOICE_BACKEND_ERROR_MESSAGE = (
    "Voice session error: I couldn’t complete that request. "
    "Please try again or use text chat."
)
VOICE_PLAYBACK_ERROR_MESSAGE = (
    "Voice session error: Audio playback failed. "
    "The text response above is still authoritative."
)
VOICE_EVENT_TOPIC = "sentinelvoice.voice"
VOICE_TURN_HANDLING = {
    "interruption": {
        "enabled": True,
        "mode": "vad",
        "min_duration": 0.35,
        "min_words": 0,
        "resume_false_interruption": False,
    }
}


class VoiceWorkerConfigurationError(RuntimeError):
    """Raised when the voice worker cannot start safely."""


class VoiceSessionMetadata(BaseModel):
    sentinelvoice_session_id: str = Field(min_length=1, max_length=128)

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class _TrackedSpeech:
    handle: Any
    turn_id: str
    sequence: int


def parse_voice_session_metadata(raw_metadata: str) -> VoiceSessionMetadata:
    try:
        return VoiceSessionMetadata.model_validate(json.loads(raw_metadata))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise VoiceWorkerConfigurationError(
            "Voice dispatch metadata is invalid"
        ) from exc


def _groq_api_key(settings: Settings) -> str:
    if settings.groq_api_key is None:
        raise VoiceWorkerConfigurationError("Groq is not configured")

    api_key = settings.groq_api_key.get_secret_value().strip()

    if not api_key:
        raise VoiceWorkerConfigurationError("Groq is not configured")

    return api_key


class SentinelVoiceAgent(Agent):
    """Media-facing agent that delegates every turn to FastAPI."""

    def __init__(
        self,
        *,
        bridge: VoiceBridge,
        session_id: str,
        publish_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(
            instructions=(
                "Relay each final user transcript to the authoritative "
                "SentinelVoice application and speak its response exactly."
            ),
            turn_handling=VOICE_TURN_HANDLING,
        )
        self._bridge = bridge
        self._session_id = session_id
        self._publish_event = publish_event
        self._clock = clock
        self._processed_turn_ids: set[str] = set()
        self._processed_turn_order: deque[str] = deque()
        # A wall-clock seed keeps ordering monotonic when a browser reconnects
        # and a new worker instance reports into the same FastAPI session.
        self._speech_sequence = time.time_ns() // 1_000
        self._speech: dict[str, _TrackedSpeech] = {}
        self._interruption_candidate_id: str | None = None
        self._interruption_started_at: float | None = None
        self._interruption_cutoff = 0
        self._reported_interruptions: set[str] = set()
        self._awaiting_corrected_transcript = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _claim_turn(self, turn_id: str) -> bool:
        if turn_id in self._processed_turn_ids:
            return False

        if len(self._processed_turn_order) >= 128:
            expired = self._processed_turn_order.popleft()
            self._processed_turn_ids.remove(expired)

        self._processed_turn_ids.add(turn_id)
        self._processed_turn_order.append(turn_id)
        return True

    async def _publish_text_only(self, text: str) -> None:
        output = self.session.output.transcription

        if output is None or not self.session.output.transcription_enabled:
            logger.error(
                "voice error text could not be published",
                extra={"sentinelvoice_session_id": self._session_id},
            )
            return

        await output.capture_text(text)
        output.flush()

    def _create_background_task(self, coroutine: Awaitable[Any]) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def _speech_is_done(speech: Any) -> bool:
        done = getattr(speech, "done", False)
        return bool(done() if callable(done) else done)

    def note_user_speaking(self) -> None:
        """Record the first VAD speech signal for interruption latency."""
        if self._interruption_candidate_id is not None:
            return

        unfinished = sorted(
            (
                tracked
                for tracked in self._speech.values()
                if not self._speech_is_done(tracked.handle)
            ),
            key=lambda tracked: tracked.sequence,
        )
        if not unfinished:
            return

        self._interruption_candidate_id = unfinished[0].handle.id
        self._interruption_started_at = self._clock()
        self._interruption_cutoff = unfinished[-1].sequence
        logger.info(
            "voice interruption detected",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "speech_id": self._interruption_candidate_id,
                "queued_speech_count": len(unfinished),
            },
        )

    def _interrupt_speech_through(self, sequence: int) -> None:
        stale = sorted(
            (
                tracked
                for tracked in self._speech.values()
                if tracked.sequence <= sequence
                and not self._speech_is_done(tracked.handle)
            ),
            key=lambda tracked: tracked.sequence,
        )
        for tracked in stale:
            tracked.handle.interrupt(source="user_turn")

    async def _report_playback(
        self,
        *,
        speech_id: str,
        turn_id: str,
        sequence: int,
        status: str,
        interruption_stop_latency_ms: float | None = None,
        response_phase: str | None = None,
    ) -> None:
        try:
            await self._bridge.report_playback(
                session_id=self._session_id,
                speech_id=speech_id,
                voice_turn_id=turn_id,
                sequence=sequence,
                status=status,
                interruption_stop_latency_ms=interruption_stop_latency_ms,
                response_phase=response_phase,
            )
        except VoiceBridgeError:
            # Text turns remain available even if this observability/state
            # callback cannot reach the application boundary.
            logger.exception(
                "voice playback state could not be synchronized",
                extra={
                    "sentinelvoice_session_id": self._session_id,
                    "speech_id": speech_id,
                    "voice_turn_id": turn_id,
                    "playback_status": status,
                },
            )

    async def _report_interruption(
        self,
        *,
        speech_id: str,
        turn_id: str,
        sequence: int,
        latency_ms: float,
    ) -> None:
        await self._report_playback(
            speech_id=speech_id,
            turn_id=turn_id,
            sequence=sequence,
            status="INTERRUPTED",
            interruption_stop_latency_ms=latency_ms,
        )
        if self._publish_event is not None:
            try:
                await self._publish_event(
                    {
                        "type": "speech_interrupted",
                        "speech_id": speech_id,
                        "voice_turn_id": turn_id,
                        "interruption_stop_latency_ms": latency_ms,
                    }
                )
            except Exception:
                logger.exception(
                    "voice interruption browser event could not be published",
                    extra={
                        "sentinelvoice_session_id": self._session_id,
                        "speech_id": speech_id,
                        "voice_turn_id": turn_id,
                    },
                )

    def _observe_speech_completion(
        self,
        speech: Any,
        *,
        turn_id: str,
    ) -> None:
        speech_id = str(getattr(speech, "id", ""))
        tracked = self._speech.pop(speech_id, None)
        interrupted = bool(getattr(speech, "interrupted", False))

        if interrupted:
            logger.info(
                "speech playback interrupted",
                extra={
                    "sentinelvoice_session_id": self._session_id,
                    "voice_turn_id": turn_id,
                    "speech_id": speech_id,
                },
            )
            if (
                speech_id == self._interruption_candidate_id
                and speech_id not in self._reported_interruptions
            ):
                started_at = self._interruption_started_at or self._clock()
                latency_ms = max(
                    0.0,
                    (self._clock() - started_at) * 1000,
                )
                self._reported_interruptions.add(speech_id)
                self._awaiting_corrected_transcript = True
                self._interrupt_speech_through(self._interruption_cutoff)
                logger.info(
                    "speech interruption stop measured",
                    extra={
                        "sentinelvoice_session_id": self._session_id,
                        "voice_turn_id": turn_id,
                        "speech_id": speech_id,
                        "interruption_stop_latency_ms": latency_ms,
                    },
                )
                self._create_background_task(
                    self._report_interruption(
                        speech_id=speech_id,
                        turn_id=turn_id,
                        sequence=(tracked.sequence if tracked else 1),
                        latency_ms=latency_ms,
                    )
                )
                self._interruption_candidate_id = None
                self._interruption_started_at = None
                self._interruption_cutoff = 0
            return

        exception = speech.exception()

        if exception is not None:
            logger.error(
                "tts synthesis or publication failed",
                exc_info=exception,
                extra={
                    "sentinelvoice_session_id": self._session_id,
                    "voice_turn_id": turn_id,
                    "speech_id": getattr(speech, "id", None),
                },
            )
            asyncio.create_task(
                self._publish_text_only(VOICE_PLAYBACK_ERROR_MESSAGE)
            )
            if speech_id == self._interruption_candidate_id:
                self._interruption_candidate_id = None
                self._interruption_started_at = None
                self._interruption_cutoff = 0
            return

        logger.info(
            "speech playback completed",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "voice_turn_id": turn_id,
                "speech_id": getattr(speech, "id", None),
            },
        )
        self._create_background_task(
            self._report_playback(
                speech_id=speech_id,
                turn_id=turn_id,
                sequence=(tracked.sequence if tracked else 1),
                status="COMPLETED",
            )
        )
        if speech_id == self._interruption_candidate_id:
            self._interruption_candidate_id = None
            self._interruption_started_at = None
            self._interruption_cutoff = 0

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        logger.info(
            "tts node entered",
            extra={"sentinelvoice_session_id": self._session_id},
        )
        audio = super().tts_node(text, model_settings)
        if not isinstance(audio, AsyncIterable):
            audio = await audio

        if audio is None:
            return

        frame_count = 0
        async for frame in audio:
            frame_count += 1
            yield frame

        logger.info(
            "tts audio frames yielded",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "audio_frame_count": frame_count,
            },
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        del turn_ctx
        transcript = " ".join((new_message.text_content or "").split())

        if not transcript:
            logger.info(
                "empty finalized voice transcript ignored",
                extra={"sentinelvoice_session_id": self._session_id},
            )
            raise StopResponse()

        if not self._claim_turn(new_message.id):
            logger.info(
                "duplicate finalized voice turn ignored",
                extra={
                    "sentinelvoice_session_id": self._session_id,
                    "voice_turn_id": new_message.id,
                },
            )
            raise StopResponse()

        # LiveKit normally interrupts the active handle from VAD before this
        # hook. The finalized-turn fallback also cancels every stale queued
        # response, while the corrected response receives a later sequence.
        if self._awaiting_corrected_transcript:
            unfinished_sequences = [
                tracked.sequence
                for tracked in self._speech.values()
                if not self._speech_is_done(tracked.handle)
            ]
            if unfinished_sequences:
                self._interrupt_speech_through(max(unfinished_sequences))
        else:
            self.note_user_speaking()
            if self._interruption_candidate_id is not None:
                self._interrupt_speech_through(self._interruption_cutoff)

        corrects_interruption = (
            self._awaiting_corrected_transcript
            or self._interruption_candidate_id is not None
        )

        logger.info(
            (
                "new corrected transcript finalized"
                if corrects_interruption
                else "voice transcript finalized"
            ),
            extra={
                "sentinelvoice_session_id": self._session_id,
                "voice_turn_id": new_message.id,
                "transcript_length": len(transcript),
            },
        )
        self._awaiting_corrected_transcript = False

        try:
            result = await self._bridge.handle_transcript(
                session_id=self._session_id,
                transcript=transcript,
            )
        except VoiceBridgeError:
            logger.exception(
                "voice bridge failed before speech synthesis",
                extra={
                    "sentinelvoice_session_id": self._session_id,
                    "voice_turn_id": new_message.id,
                },
            )
            await self._publish_text_only(VOICE_BACKEND_ERROR_MESSAGE)
            raise StopResponse()

        if result is None:
            raise StopResponse()

        logger.info(
            "assistant response received",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "voice_turn_id": new_message.id,
                "turn_status": result.turn_status,
                "assistant_text_length": len(result.message),
            },
        )
        try:
            speech = self.session.say(
                result.message,
                allow_interruptions=True,
            )
        except RuntimeError:
            logger.exception(
                "assistant speech could not be scheduled",
                extra={
                    "sentinelvoice_session_id": self._session_id,
                    "voice_turn_id": new_message.id,
                },
            )
            await self._publish_text_only(result.message)
            await self._publish_text_only(VOICE_PLAYBACK_ERROR_MESSAGE)
            raise StopResponse()

        logger.info(
            "assistant speech scheduled",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "voice_turn_id": new_message.id,
                "speech_id": speech.id,
            },
        )
        self._speech_sequence += 1
        self._speech[speech.id] = _TrackedSpeech(
            handle=speech,
            turn_id=new_message.id,
            sequence=self._speech_sequence,
        )
        self._create_background_task(
            self._report_playback(
                speech_id=speech.id,
                turn_id=new_message.id,
                sequence=self._speech_sequence,
                status="SCHEDULED",
                response_phase=result.conversation_phase,
            )
        )
        speech.add_done_callback(
            lambda completed: self._observe_speech_completion(
                completed,
                turn_id=new_message.id,
            )
        )
        raise StopResponse()


def _prewarm(proc: JobProcess) -> None:
    # Silero imports ONNX Runtime, whose telemetry cache otherwise creates a
    # ``:memory:.ses`` marker in the worker's current directory on macOS.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()


def _voice_room_options() -> RoomOptions:
    return RoomOptions(
        text_output=TextOutputOptions(sync_transcription=False),
    )


settings = get_settings()
server = AgentServer(
    setup_fnc=_prewarm,
    ws_url=settings.livekit_url,
    api_key=(
        settings.livekit_api_key.get_secret_value()
        if settings.livekit_api_key is not None
        else None
    ),
    api_secret=(
        settings.livekit_api_secret.get_secret_value()
        if settings.livekit_api_secret is not None
        else None
    ),
)


@server.rtc_session(agent_name=settings.livekit_agent_name)
async def voice_session(ctx: JobContext) -> None:
    metadata = parse_voice_session_metadata(ctx.job.metadata)
    api_key = _groq_api_key(settings)
    bridge = VoiceBridge(api_base_url=settings.api_base_url)
    ctx.add_shutdown_callback(bridge.aclose)

    stt_provider = GroqSpeechToTextProvider(
        api_key=api_key,
        model=settings.stt_model,
    )
    tts_provider = GroqTextToSpeechProvider(
        api_key=api_key,
        model=settings.tts_model,
        voice=settings.tts_voice,
    )
    session = AgentSession(
        stt=GroqSTTAdapter(
            provider=stt_provider,
            model=settings.stt_model,
        ),
        vad=ctx.proc.userdata["vad"],
        tts=GroqTTSAdapter(
            provider=tts_provider,
            model=settings.tts_model,
        ),
        turn_handling=VOICE_TURN_HANDLING,
    )

    async def _publish_voice_event(payload: dict[str, Any]) -> None:
        await ctx.room.local_participant.publish_data(
            json.dumps(payload),
            reliable=True,
            topic=VOICE_EVENT_TOPIC,
        )

    agent = SentinelVoiceAgent(
        bridge=bridge,
        session_id=metadata.sentinelvoice_session_id,
        publish_event=_publish_voice_event,
    )

    def _log_agent_state_change(event: Any) -> None:
        if event.new_state == "speaking":
            logger.info(
                "speech playback started",
                extra={
                    "sentinelvoice_session_id": metadata.sentinelvoice_session_id,
                },
            )

    def _handle_user_state_change(event: Any) -> None:
        if event.new_state == "speaking":
            agent.note_user_speaking()

    session.on("agent_state_changed", _log_agent_state_change)
    session.on("user_state_changed", _handle_user_state_change)

    logger.info(
        "sentinelvoice session resolved from voice dispatch metadata",
        extra={
            "sentinelvoice_session_id": metadata.sentinelvoice_session_id,
        },
    )
    await ctx.connect()
    await session.start(
        room=ctx.room,
        room_options=_voice_room_options(),
        agent=agent,
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
