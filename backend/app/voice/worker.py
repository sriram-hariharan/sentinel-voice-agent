import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import AsyncIterable
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


class VoiceWorkerConfigurationError(RuntimeError):
    """Raised when the voice worker cannot start safely."""


class VoiceSessionMetadata(BaseModel):
    sentinelvoice_session_id: str = Field(min_length=1, max_length=128)

    model_config = ConfigDict(extra="forbid")


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

    def __init__(self, *, bridge: VoiceBridge, session_id: str) -> None:
        super().__init__(
            instructions=(
                "Relay each final user transcript to the authoritative "
                "SentinelVoice application and speak its response exactly."
            ),
            allow_interruptions=False,
        )
        self._bridge = bridge
        self._session_id = session_id
        self._processed_turn_ids: set[str] = set()
        self._processed_turn_order: deque[str] = deque()

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

    def _observe_speech_completion(
        self,
        speech: Any,
        *,
        turn_id: str,
    ) -> None:
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
            return

        logger.info(
            "speech playback completed",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "voice_turn_id": turn_id,
                "speech_id": getattr(speech, "id", None),
            },
        )

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

        logger.info(
            "voice transcript finalized",
            extra={
                "sentinelvoice_session_id": self._session_id,
                "voice_turn_id": new_message.id,
                "transcript_length": len(transcript),
            },
        )

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
                allow_interruptions=False,
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
        allow_interruptions=False,
    )

    def _log_agent_state_change(event: Any) -> None:
        if event.new_state == "speaking":
            logger.info(
                "speech playback started",
                extra={
                    "sentinelvoice_session_id": metadata.sentinelvoice_session_id,
                },
            )

    session.on("agent_state_changed", _log_agent_state_change)

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
        agent=SentinelVoiceAgent(
            bridge=bridge,
            session_id=metadata.sentinelvoice_session_id,
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
