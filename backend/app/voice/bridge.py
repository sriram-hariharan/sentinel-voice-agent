import logging
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class VoiceBridgeError(RuntimeError):
    """Raised when the authoritative application turn fails."""


class VoiceTurnResult(BaseModel):
    session_id: str
    message: str
    turn_status: str
    conversation_phase: str
    executed_tools: list[str] = Field(default_factory=list)
    pending_action: str | None = None

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
    ) -> VoiceTurnResult | None:
        normalized = " ".join(transcript.split())

        if not normalized:
            return None

        logger.info(
            "sentinelvoice backend turn started",
            extra={
                "sentinelvoice_session_id": session_id,
                "transcript_length": len(normalized),
            },
        )

        try:
            response = await self._client.post(
                f"/sessions/{session_id}/messages",
                json={"message": normalized},
            )
            logger.info(
                "sentinelvoice backend HTTP response received",
                extra={
                    "sentinelvoice_session_id": session_id,
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
                "turn_status": result.turn_status,
                "assistant_text_length": len(result.message),
            },
        )
        return result

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
