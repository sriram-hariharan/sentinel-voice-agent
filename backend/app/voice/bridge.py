import logging
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.app.observability.context import (
    TRACE_ID_HEADER,
    TURN_ID_HEADER,
    build_trace_context,
    trace_scope,
)
from backend.app.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class VoiceBridgeError(RuntimeError):
    """Raised when the authoritative application turn fails."""


class VoiceTurnResult(BaseModel):
    session_id: str
    trace_id: str | None = None
    turn_id: str | None = None
    message: str
    turn_status: str
    conversation_phase: str
    executed_tools: list[str] = Field(default_factory=list)
    pending_action: str | None = None
    policy_sources: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, extra="ignore")


class VoicePlaybackResult(BaseModel):
    session_id: str
    conversation_phase: str

    model_config = ConfigDict(frozen=True, extra="ignore")


class VoiceBridge:
    def __init__(
        self,
        *,
        api_base_url: str,
        client: Any | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=api_base_url.rstrip("/"),
            timeout=30,
        )
        self._owns_client = client is None

    async def handle_transcript(
        self,
        *,
        session_id: str,
        transcript: str,
        trace_id: str | None = None,
        turn_id: str | None = None,
    ) -> VoiceTurnResult | None:
        normalized = " ".join(transcript.split())

        if not normalized:
            return None

        correlation = build_trace_context(
            session_id=session_id,
            trace_id=trace_id,
            turn_id=turn_id,
        )

        with trace_scope(
            session_id=session_id,
            trace_id=correlation.trace_id,
            turn_id=correlation.turn_id,
        ):
            logger.info(
                "sentinelvoice backend turn started",
                extra={
                    "sentinelvoice_session_id": session_id,
                    "trace_id": correlation.trace_id,
                    "turn_id": correlation.turn_id,
                    "transcript_length": len(normalized),
                },
            )
            try:
                with trace_span(
                    "voice.backend_turn",
                    component="voice_bridge",
                ):
                    response = await self._client.post(
                        f"/sessions/{session_id}/messages",
                        json={"message": normalized},
                        headers={
                            TRACE_ID_HEADER: correlation.trace_id,
                            TURN_ID_HEADER: correlation.turn_id,
                        },
                    )
                    logger.info(
                        "sentinelvoice backend HTTP response received",
                        extra={
                            "sentinelvoice_session_id": session_id,
                            "trace_id": correlation.trace_id,
                            "turn_id": correlation.turn_id,
                            "http_status": response.status_code,
                        },
                    )
                    response.raise_for_status()
                    result = VoiceTurnResult.model_validate(response.json())
            except Exception as exc:
                status_code = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                logger.exception(
                    "sentinelvoice backend turn failed",
                    extra={
                        "sentinelvoice_session_id": session_id,
                        "trace_id": correlation.trace_id,
                        "turn_id": correlation.turn_id,
                        "http_status": status_code,
                    },
                )
                raise VoiceBridgeError(
                    "SentinelVoice message turn failed"
                ) from exc

        logger.info(
            "sentinelvoice backend turn completed",
            extra={
                "sentinelvoice_session_id": session_id,
                "trace_id": correlation.trace_id,
                "turn_id": correlation.turn_id,
                "turn_status": result.turn_status,
                "assistant_text_length": len(result.message),
            },
        )
        return result

    async def report_playback(
        self,
        *,
        session_id: str,
        speech_id: str,
        voice_turn_id: str,
        sequence: int,
        status: str,
        interruption_stop_latency_ms: float | None = None,
        response_phase: str | None = None,
        trace_id: str | None = None,
        turn_id: str | None = None,
    ) -> VoicePlaybackResult:
        payload: dict[str, str | float] = {
            "speech_id": speech_id,
            "voice_turn_id": voice_turn_id,
            "sequence": sequence,
            "status": status,
        }
        if interruption_stop_latency_ms is not None:
            payload["interruption_stop_latency_ms"] = (
                interruption_stop_latency_ms
            )
        if response_phase is not None:
            payload["response_phase"] = response_phase

        correlation = build_trace_context(
            session_id=session_id,
            trace_id=trace_id,
            turn_id=turn_id,
        )
        try:
            response = await self._client.post(
                f"/sessions/{session_id}/voice/playback",
                json=payload,
                headers={
                    TRACE_ID_HEADER: correlation.trace_id,
                    TURN_ID_HEADER: correlation.turn_id,
                },
            )
            response.raise_for_status()
            return VoicePlaybackResult.model_validate(response.json())
        except Exception as exc:
            status_code = (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError)
                else None
            )
            logger.exception(
                "sentinelvoice playback state report failed",
                extra={
                    "sentinelvoice_session_id": session_id,
                    "voice_turn_id": voice_turn_id,
                    "speech_id": speech_id,
                    "playback_status": status,
                    "http_status": status_code,
                },
            )
            raise VoiceBridgeError(
                "SentinelVoice playback state report failed"
            ) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
