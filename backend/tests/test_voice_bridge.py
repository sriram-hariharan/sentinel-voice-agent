from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.dependencies import get_agent_orchestrator
from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.agent.resource_resolver import ResourceResolution
from backend.app.auth.sessions import InMemorySessionStore, get_session_store
from backend.app.conversation.state import AuthenticationLevel
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.providers.llm import LLMResponse
from backend.app.tools.schemas import FreezeCardOutput
from backend.app.voice.bridge import VoiceBridge

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


class PhaseRecordingResourceResolver:
    def __init__(self) -> None:
        self.phases = []

    async def resolve(self, *, state, **kwargs) -> ResourceResolution:
        self.phases.append(state.phase)
        return ResourceResolution()


def _pending_protected_action(store: InMemorySessionStore):
    state = store.create()
    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED
    state.active_intent = "freeze_card"
    state.active_card_id = CARD_ID
    state.request_action(
        "freeze_card",
        CARD_ID,
        arguments={"card_id": str(CARD_ID)},
    )
    return state


async def _bridge_for(
    *,
    store: InMemorySessionStore,
    orchestrator: AgentOrchestrator,
) -> tuple[VoiceBridge, httpx.AsyncClient]:
    db = AsyncMock(spec=AsyncSession)
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_db_session] = lambda: db
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://sentinelvoice.test",
    )
    return VoiceBridge(api_base_url="http://unused", client=client), client


@pytest.mark.asyncio
async def test_voice_bridge_uses_authoritative_message_boundary_and_session() -> None:
    store = InMemorySessionStore()
    state = store.create()
    llm = SequenceLLM(
        [LLMResponse(content="Your balance is $125.00.", model="test")]
    )
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            resource_resolver=NoopResourceResolver(),
        ),
    )

    try:
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="  What is my checking balance?  ",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert result.session_id == state.session_id
    assert result.message == "Your balance is $125.00."
    assert llm.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "What is my checking balance?",
    }


@pytest.mark.asyncio
async def test_voice_yes_uses_existing_confirmation_path() -> None:
    store = InMemorySessionStore()
    state = _pending_protected_action(store)
    executor = AsyncMock()
    executor.execute.return_value = FreezeCardOutput(
        card_id=CARD_ID,
        masked_card_number="****1842",
        previous_status="ACTIVE",
        status="FROZEN",
        changed=True,
    )
    llm = SequenceLLM(
        [LLMResponse(content="Your card is now frozen.", model="test")]
    )
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        ),
    )

    try:
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="Yes",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert result.executed_tools == ["freeze_card"]
    assert state.pending_action is None
    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_no_uses_existing_cancellation_path() -> None:
    store = InMemorySessionStore()
    state = _pending_protected_action(store)
    executor = AsyncMock()
    llm = SequenceLLM([])
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        ),
    )

    try:
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="No",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert result.message == "Okay, I won’t perform that action."
    assert state.pending_action is None
    executor.execute.assert_not_awaited()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_interrupted_confirmation_prompt_actually_no_cancels() -> None:
    store = InMemorySessionStore()
    state = _pending_protected_action(store)
    executor = AsyncMock()
    llm = SequenceLLM([])
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        ),
    )

    try:
        await bridge.report_playback(
            session_id=state.session_id,
            speech_id="speech-confirmation",
            voice_turn_id="turn-proposal",
            sequence=1,
            status="INTERRUPTED",
            interruption_stop_latency_ms=75,
        )
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="Actually no",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert result.message == "Okay, I won’t perform that action."
    assert state.pending_action is None
    executor.execute.assert_not_awaited()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_unrelated_barge_in_never_executes_pending_action() -> None:
    store = InMemorySessionStore()
    state = _pending_protected_action(store)
    executor = AsyncMock()
    llm = SequenceLLM(
        [LLMResponse(content="I can help with your balance.", model="test")]
    )
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        ),
    )

    try:
        await bridge.report_playback(
            session_id=state.session_id,
            speech_id="speech-confirmation",
            voice_turn_id="turn-proposal",
            sequence=1,
            status="INTERRUPTED",
            interruption_stop_latency_ms=75,
        )
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="What is my balance?",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert result.message == "I can help with your balance."
    assert state.pending_action is None
    executor.execute.assert_not_awaited()
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_interrupted_confirmation_still_executes_once_on_yes() -> None:
    store = InMemorySessionStore()
    state = _pending_protected_action(store)
    executor = AsyncMock()
    executor.execute.return_value = FreezeCardOutput(
        card_id=CARD_ID,
        masked_card_number="****1842",
        previous_status="ACTIVE",
        status="FROZEN",
        changed=True,
    )
    llm = SequenceLLM(
        [LLMResponse(content="Your card is now frozen.", model="test")]
    )
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            tool_executor=executor,
            resource_resolver=NoopResourceResolver(),
        ),
    )

    try:
        await bridge.report_playback(
            session_id=state.session_id,
            speech_id="speech-confirmation",
            voice_turn_id="turn-proposal",
            sequence=1,
            status="INTERRUPTED",
            interruption_stop_latency_ms=75,
        )
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="Yes",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert result.executed_tools == ["freeze_card"]
    assert state.pending_action is None
    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_corrected_turn_leaves_interrupted_state_for_processing() -> None:
    store = InMemorySessionStore()
    state = store.create()
    resolver = PhaseRecordingResourceResolver()
    llm = SequenceLLM(
        [LLMResponse(content="Corrected response.", model="test")]
    )
    bridge, client = await _bridge_for(
        store=store,
        orchestrator=AgentOrchestrator(
            llm=llm,
            resource_resolver=resolver,
        ),
    )

    try:
        await bridge.report_playback(
            session_id=state.session_id,
            speech_id="speech-old",
            voice_turn_id="turn-old",
            sequence=1,
            status="INTERRUPTED",
            interruption_stop_latency_ms=75,
        )
        assert state.phase.value == "INTERRUPTED"
        result = await bridge.handle_transcript(
            session_id=state.session_id,
            transcript="Use savings instead",
        )
    finally:
        await client.aclose()
        app.dependency_overrides.clear()

    assert result is not None
    assert resolver.phases == ["PROCESSING"]
    assert state.phase.value == "AGENT_SPEAKING"


@pytest.mark.asyncio
async def test_empty_voice_transcript_never_reaches_backend() -> None:
    client = AsyncMock()
    bridge = VoiceBridge(api_base_url="http://unused", client=client)

    result = await bridge.handle_transcript(
        session_id="session-123",
        transcript=" \n ",
    )

    assert result is None
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_speech_failure_cannot_execute_pending_action() -> None:
    state = _pending_protected_action(InMemorySessionStore())
    executor = AsyncMock()
    speech_provider = AsyncMock()
    speech_provider.transcribe.side_effect = RuntimeError("STT failed")
    bridge = AsyncMock()

    with pytest.raises(RuntimeError, match="STT failed"):
        await speech_provider.transcribe(b"audio")

    bridge.handle_transcript.assert_not_awaited()
    executor.execute.assert_not_awaited()
    assert state.pending_action is not None
