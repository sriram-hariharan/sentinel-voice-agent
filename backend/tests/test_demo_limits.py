from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.dependencies import get_agent_orchestrator
from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.agent.resource_resolver import ResourceResolution
from backend.app.auth.sessions import (
    InMemorySessionStore,
    SessionCapacityError,
    SessionNotFoundError,
    SessionTurnLimitError,
    get_session_store,
)
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.providers.llm import LLMResponse


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="Hello.",
            model="test-model",
        )


class NoopResourceResolver:
    async def resolve(self, **kwargs) -> ResourceResolution:
        return ResourceResolution()


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def test_session_store_rejects_creation_at_capacity() -> None:
    store = InMemorySessionStore(max_active_sessions=1)

    store.create()

    with pytest.raises(SessionCapacityError):
        store.create()


def test_expired_session_is_removed_and_capacity_is_reclaimed() -> None:
    now = [datetime(2026, 9, 25, tzinfo=UTC)]

    store = InMemorySessionStore(
        session_ttl_seconds=60,
        max_active_sessions=1,
        clock=lambda: now[0],
    )

    expired = store.create()

    now[0] += timedelta(seconds=61)

    replacement = store.create()

    with pytest.raises(SessionNotFoundError):
        store.get(expired.session_id)

    assert store.get(replacement.session_id) is replacement


def test_session_store_enforces_turn_budget() -> None:
    store = InMemorySessionStore(max_turns_per_session=2)
    state = store.create()

    assert store.claim_turn(state.session_id) == 1
    assert store.claim_turn(state.session_id) == 2

    with pytest.raises(SessionTurnLimitError):
        store.claim_turn(state.session_id)


def test_session_api_returns_429_at_capacity() -> None:
    store = InMemorySessionStore(max_active_sessions=1)
    app.dependency_overrides[get_session_store] = lambda: store

    with TestClient(app) as client:
        first = client.post("/sessions")
        second = client.post("/sessions")

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.json() == {"detail": "Demo session capacity reached"}


def test_message_api_stops_before_llm_after_turn_budget() -> None:
    store = InMemorySessionStore(max_turns_per_session=1)
    db = AsyncMock(spec=AsyncSession)
    llm = CountingLLM()

    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_db_session] = lambda: db
    app.dependency_overrides[get_agent_orchestrator] = lambda: AgentOrchestrator(
        llm=llm,
        resource_resolver=NoopResourceResolver(),
    )

    with TestClient(app) as client:
        created = client.post("/sessions")
        session_id = created.json()["session_id"]

        first = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Hello"},
        )
        second = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Hello again"},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json() == {"detail": "Session turn limit reached"}
    assert llm.calls == 1


def test_expired_session_is_unavailable_through_api() -> None:
    now = [datetime(2026, 9, 25, tzinfo=UTC)]

    store = InMemorySessionStore(
        session_ttl_seconds=60,
        clock=lambda: now[0],
    )
    app.dependency_overrides[get_session_store] = lambda: store

    with TestClient(app) as client:
        created = client.post("/sessions")
        session_id = created.json()["session_id"]

        now[0] += timedelta(seconds=61)

        response = client.get(f"/sessions/{session_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_invalid_trace_header_does_not_consume_turn_budget() -> None:
    store = InMemorySessionStore(max_turns_per_session=1)
    db = AsyncMock(spec=AsyncSession)
    llm = CountingLLM()

    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_db_session] = lambda: db
    app.dependency_overrides[get_agent_orchestrator] = lambda: AgentOrchestrator(
        llm=llm,
        resource_resolver=NoopResourceResolver(),
    )

    with TestClient(app) as client:
        created = client.post("/sessions")
        session_id = created.json()["session_id"]

        invalid = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Hello"},
            headers={"X-SentinelVoice-Trace-ID": "too-short"},
        )
        valid = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Hello"},
        )
        exhausted = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "One more"},
        )

    assert invalid.status_code == 400
    assert valid.status_code == 200
    assert exhausted.status_code == 429
    assert llm.calls == 1
