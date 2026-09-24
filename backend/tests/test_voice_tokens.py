import json
from unittest.mock import AsyncMock
from uuid import UUID

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.sessions import InMemorySessionStore, get_session_store
from backend.app.config.settings import Settings, get_settings
from backend.app.conversation.state import AuthenticationLevel
from backend.app.db.session import get_db_session
from backend.app.main import app

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
JWT_SECRET = "test-secret-that-is-at-least-32-bytes-long"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "livekit_url": "wss://sentinelvoice.test.livekit.cloud",
        "livekit_api_key": SecretStr("test-key"),
        "livekit_api_secret": SecretStr(JWT_SECRET),
        "groq_api_key": SecretStr("test-groq-key"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def voice_api_context():
    store = InMemorySessionStore()
    db = AsyncMock(spec=AsyncSession)
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_db_session] = lambda: db
    app.dependency_overrides[get_settings] = lambda: _settings()

    with TestClient(app) as client:
        yield client, store

    app.dependency_overrides.clear()


def _authenticated_state(store: InMemorySessionStore):
    state = store.create()
    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED
    return state


def test_voice_token_rejects_unknown_session(voice_api_context) -> None:
    client, _ = voice_api_context

    response = client.post("/sessions/does-not-exist/voice/token")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_voice_token_requires_authenticated_session(voice_api_context) -> None:
    client, store = voice_api_context
    state = store.create()

    response = client.post(f"/sessions/{state.session_id}/voice/token")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Authentication is required for voice"
    }


def test_voice_token_handles_missing_livekit_credentials(
    voice_api_context,
) -> None:
    client, store = voice_api_context
    state = _authenticated_state(store)
    app.dependency_overrides[get_settings] = lambda: _settings(
        livekit_api_secret=None
    )

    response = client.post(f"/sessions/{state.session_id}/voice/token")

    assert response.status_code == 503
    assert response.json() == {"detail": "LiveKit is not configured"}


def test_voice_token_handles_missing_groq_configuration(
    voice_api_context,
) -> None:
    client, store = voice_api_context
    state = _authenticated_state(store)
    app.dependency_overrides[get_settings] = lambda: _settings(
        groq_api_key=None
    )

    response = client.post(f"/sessions/{state.session_id}/voice/token")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Groq speech providers are not configured"
    }


def test_voice_token_contains_only_safe_session_linkage(
    voice_api_context,
) -> None:
    client, store = voice_api_context
    state = _authenticated_state(store)

    response = client.post(
        f"/sessions/{state.session_id}/voice/token",
        json={"customer_id": "99999999-9999-4999-8999-999999999999"},
    )

    assert response.status_code == 200
    payload = jwt.decode(
        response.json()["participant_token"],
        JWT_SECRET,
        algorithms=["HS256"],
        options={"verify_aud": False},
    )
    expected_metadata = {
        "sentinelvoice_session_id": state.session_id,
    }

    assert json.loads(payload["metadata"]) == expected_metadata
    assert json.loads(payload["roomConfig"]["agents"][0]["metadata"]) == (
        expected_metadata
    )
    assert payload["roomConfig"]["agents"][0]["agentName"] == (
        "sentinelvoice"
    )
    assert payload["sub"].startswith("browser-")
    assert payload["video"]["room"].startswith("sv-")
    assert payload["video"]["canPublishData"] is False
    assert "customer" not in payload["metadata"].lower()
    assert str(CUSTOMER_ID) not in response.text


def test_session_state_can_be_refreshed_after_voice_turn(
    voice_api_context,
) -> None:
    client, store = voice_api_context
    state = _authenticated_state(store)

    response = client.get(f"/sessions/{state.session_id}")

    assert response.status_code == 200
    assert response.json()["session_id"] == state.session_id
    assert response.json()["authenticated"] is True
