from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.dependencies import get_agent_orchestrator
from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.agent.resource_resolver import ResourceResolution
from backend.app.auth.sessions import InMemorySessionStore, get_session_store
from backend.app.config.settings import Settings, get_settings
from backend.app.conversation.state import AuthenticationLevel
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.providers.llm import LLMResponse, LLMToolCall
from backend.app.tools.schemas import FreezeCardOutput

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")


class SequenceLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


class NoopResourceResolver:
    async def resolve(self, **kwargs) -> ResourceResolution:
        return ResourceResolution()


@pytest.fixture
def api_context():
    store = InMemorySessionStore()
    db = AsyncMock(spec=AsyncSession)

    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_db_session] = lambda: db

    with TestClient(app) as client:
        yield client, store, db

    app.dependency_overrides.clear()


def _use_orchestrator(orchestrator: AgentOrchestrator) -> None:
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator


def _direct_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="test-model")


def test_message_rejects_unknown_session(api_context) -> None:
    client, _, _ = api_context
    _use_orchestrator(
        AgentOrchestrator(llm=SequenceLLM([_direct_response("Hello")]))
    )

    response = client.post(
        "/sessions/does-not-exist/messages",
        json={"message": "Hello"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_message_returns_503_without_groq_configuration(api_context) -> None:
    client, store, _ = api_context
    state = store.create()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        groq_api_key=None,
    )

    response = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Hello"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "LLM provider is not configured"}


def test_unauthenticated_session_can_make_normal_agent_turn(api_context) -> None:
    client, store, _ = api_context
    state = store.create()
    llm = SequenceLLM([_direct_response("How can I help?")])
    _use_orchestrator(AgentOrchestrator(llm=llm))

    response = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Hello"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": state.session_id,
        "message": "How can I help?",
        "turn_status": "RESPONDED",
        "conversation_phase": "AGENT_SPEAKING",
        "authenticated": False,
        "customer_id": None,
        "executed_tools": [],
        "pending_action": None,
        "policy_sources": [],
    }

    refreshed = client.get(f"/sessions/{state.session_id}")
    assert refreshed.status_code == 200
    assert refreshed.json()["turn_status"] == "RESPONDED"
    assert refreshed.json()["conversation_phase"] == "AGENT_SPEAKING"


def test_authenticated_turn_uses_server_side_identity(api_context) -> None:
    client, store, _ = api_context
    state = store.create()
    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED

    llm = SequenceLLM([_direct_response("Welcome back.")])
    _use_orchestrator(AgentOrchestrator(llm=llm))

    response = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Who am I?"},
    )

    assert response.status_code == 200
    assert response.json()["authenticated"] is True
    assert response.json()["customer_id"] == str(CUSTOMER_ID)
    assert "Authenticated session: true" in llm.calls[0]["messages"][1]["content"]


def test_protected_action_returns_waiting_for_confirmation(api_context) -> None:
    client, store, _ = api_context
    state = store.create()
    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED
    state.active_card_id = CARD_ID
    state.active_intent = "freeze_card"

    llm = SequenceLLM(
        [
            LLMResponse(
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-freeze",
                        name="freeze_card",
                        arguments={"card_id": str(CARD_ID)},
                    )
                ],
            )
        ]
    )
    executor = AsyncMock()
    _use_orchestrator(
        AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        )
    )

    response = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Freeze my card"},
    )

    assert response.status_code == 200
    assert response.json()["turn_status"] == "WAITING_FOR_CONFIRMATION"
    assert response.json()["conversation_phase"] == "WAITING_FOR_CONFIRMATION"
    assert response.json()["pending_action"] == "freeze_card"
    executor.execute.assert_not_awaited()


def test_confirmation_executes_stored_protected_action(api_context) -> None:
    client, store, _ = api_context
    state = store.create()
    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED
    state.active_card_id = CARD_ID
    state.active_intent = "freeze_card"

    llm = SequenceLLM(
        [
            LLMResponse(
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-freeze",
                        name="freeze_card",
                        arguments={"card_id": str(CARD_ID)},
                    )
                ],
            ),
            _direct_response("Your card is now frozen."),
        ]
    )
    executor = AsyncMock()
    executor.execute.return_value = FreezeCardOutput(
        card_id=CARD_ID,
        masked_card_number="****1842",
        previous_status="ACTIVE",
        status="FROZEN",
        changed=True,
    )
    _use_orchestrator(
        AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        )
    )

    proposal = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Freeze my card"},
    )
    confirmation = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Yes"},
    )

    assert proposal.json()["turn_status"] == "WAITING_FOR_CONFIRMATION"
    assert confirmation.status_code == 200
    assert confirmation.json()["executed_tools"] == ["freeze_card"]
    assert confirmation.json()["pending_action"] is None
    assert state.active_intent is None
    assert state.active_card_id is None
    executor.execute.assert_awaited_once()

    context = executor.execute.await_args.args[2]
    assert context.customer_id == CUSTOMER_ID
    assert context.confirmation is not None
    assert context.confirmation.resource_id == CARD_ID


def test_cancellation_does_not_execute_protected_action(api_context) -> None:
    client, store, _ = api_context
    state = store.create()
    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED
    state.active_card_id = CARD_ID
    state.active_intent = "freeze_card"

    llm = SequenceLLM(
        [
            LLMResponse(
                model="test-model",
                tool_calls=[
                    LLMToolCall(
                        id="call-freeze",
                        name="freeze_card",
                        arguments={"card_id": str(CARD_ID)},
                    )
                ],
            )
        ]
    )
    executor = AsyncMock()
    _use_orchestrator(
        AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        )
    )

    client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Freeze my card"},
    )
    cancellation = client.post(
        f"/sessions/{state.session_id}/messages",
        json={"message": "Cancel"},
    )

    assert cancellation.status_code == 200
    assert cancellation.json()["message"] == "Okay, I won’t perform that action."
    assert cancellation.json()["pending_action"] is None
    assert state.active_intent is None
    assert state.active_card_id is None
    executor.execute.assert_not_awaited()
