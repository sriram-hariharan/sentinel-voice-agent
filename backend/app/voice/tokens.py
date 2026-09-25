import json
from datetime import timedelta
from uuid import uuid4

from livekit import api
from pydantic import BaseModel, ConfigDict

from backend.app.config.settings import Settings


class VoiceConfigurationError(RuntimeError):
    """Raised when the voice transport is not configured."""


class VoiceConnectionToken(BaseModel):
    server_url: str
    participant_token: str

    model_config = ConfigDict(frozen=True)


def _secret_value(value: object) -> str:
    getter = getattr(value, "get_secret_value", None)
    return getter() if getter is not None else ""


def create_voice_connection_token(
    *,
    session_id: str,
    settings: Settings,
    max_ttl_seconds: int | None = None,
) -> VoiceConnectionToken:
    server_url = (settings.livekit_url or "").strip()
    api_key = _secret_value(settings.livekit_api_key)
    api_secret = _secret_value(settings.livekit_api_secret)

    if not server_url or not api_key or not api_secret:
        raise VoiceConfigurationError("LiveKit is not configured")

    room_name = f"sv-{uuid4().hex}"
    participant_identity = f"browser-{uuid4().hex}"
    session_ttl_cap = (
        settings.demo_session_ttl_seconds
        if max_ttl_seconds is None
        else max_ttl_seconds
    )

    if session_ttl_cap <= 0:
        raise VoiceConfigurationError("Voice token TTL is invalid")

    token_ttl_seconds = min(
        settings.voice_token_ttl_seconds,
        session_ttl_cap,
    )
    safe_metadata = json.dumps(
        {"sentinelvoice_session_id": session_id},
        separators=(",", ":"),
    )
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name("SentinelVoice customer")
        .with_metadata(safe_metadata)
        .with_ttl(timedelta(seconds=token_ttl_seconds))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=False,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=settings.livekit_agent_name,
                        metadata=safe_metadata,
                    )
                ]
            )
        )
        .to_jwt()
    )

    return VoiceConnectionToken(
        server_url=server_url,
        participant_token=token,
    )
