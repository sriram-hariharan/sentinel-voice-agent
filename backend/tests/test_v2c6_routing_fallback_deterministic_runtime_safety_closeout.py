from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.orchestrator import (
    SEMANTIC_CLARIFICATIONS,
    AgentOrchestrator,
    AgentTurnStatus,
)
from backend.app.agent.protected_action_verifier import (
    ProtectedActionSemanticDecision,
    ProtectedActionVerificationError,
    VerifierFailureCategory,
)
from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationState,
)
from backend.app.providers.llm import LLMResponse, LLMToolCall
from backend.app.tools.errors import ToolAuthenticationError
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.schemas import ActionConfirmation, ToolExecutionContext
from scripts import (
    run_v2c6_routing_fallback_deterministic_runtime_safety_closeout as closeout,
)

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
ACCOUNT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
CARD_ID = UUID("cccccccc-cccc-4ccc-8ccc-ccccccccccc1")
TRANSACTION_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeee1")


class SequenceLLM:
    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        del messages, tools
        return self.responses.pop(0)


class ScriptedVerifier:
    provider = "test"
    model = "test-verifier"

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str]] = []

    async def verify(self, *, user_text: str, proposed_action: str):
        self.calls.append((user_text, proposed_action))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _state() -> ConversationState:
    return ConversationState(
        session_id="v2c6-runtime-closeout",
        customer_id=CUSTOMER_ID,
        authentication_level=AuthenticationLevel.AUTHENTICATED,
    )


def _tool_call(name: str, **arguments: str) -> LLMResponse:
    return LLMResponse(
        model="test-model",
        tool_calls=[LLMToolCall(id=f"call-{name}", name=name, arguments=arguments)],
    )


@pytest.mark.parametrize(
    "failure",
    [
        ProtectedActionVerificationError(
            VerifierFailureCategory.TIMEOUT,
            "timeout",
        ),
        ProtectedActionVerificationError(
            VerifierFailureCategory.PROVIDER_ERROR,
            "provider failure",
        ),
        ProtectedActionVerificationError(
            VerifierFailureCategory.MALFORMED_STRUCTURED_OUTPUT,
            "malformed output",
        ),
        "EXPLICIT_CURRENT_ACTION",
    ],
)
@pytest.mark.asyncio
async def test_create_dispute_representative_verifier_failures_fail_closed(
    failure: object,
) -> None:
    db = AsyncMock(spec=AsyncSession)
    executor = AsyncMock()
    verifier = ScriptedVerifier(failure)
    state = _state()
    orchestrator = AgentOrchestrator(
        llm=SequenceLLM(
            _tool_call(
                "create_dispute",
                transaction_id=str(TRANSACTION_ID),
                reason_code="unauthorized",
            )
        ),
        tool_executor=executor,
        protected_action_verifier=verifier,
    )

    result = await orchestrator.handle_text_turn(
        user_text="Create a dispute for that charge now",
        state=state,
        db=db,
    )

    assert result.text == SEMANTIC_CLARIFICATIONS["create_dispute"]
    assert result.status == AgentTurnStatus.RESPONDED
    assert verifier.calls == [
        ("Create a dispute for that charge now", "create_dispute")
    ]
    db.scalars.assert_not_awaited()
    executor.execute.assert_not_awaited()
    assert state.pending_resource_resolution is None
    assert state.protected_action_semantic_context is None
    assert state.pending_action is None


@pytest.mark.asyncio
async def test_create_dispute_not_requested_cannot_advance() -> None:
    db = AsyncMock(spec=AsyncSession)
    executor = AsyncMock()
    state = _state()
    orchestrator = AgentOrchestrator(
        llm=SequenceLLM(
            _tool_call(
                "create_dispute",
                transaction_id=str(TRANSACTION_ID),
                reason_code="unauthorized",
            )
        ),
        tool_executor=executor,
        protected_action_verifier=ScriptedVerifier(
            ProtectedActionSemanticDecision.NOT_REQUESTED
        ),
    )

    result = await orchestrator.handle_text_turn(
        user_text="Do not dispute that charge",
        state=state,
        db=db,
    )

    assert result.text == SEMANTIC_CLARIFICATIONS["create_dispute"]
    assert result.status == AgentTurnStatus.RESPONDED
    db.scalars.assert_not_awaited()
    executor.execute.assert_not_awaited()
    assert state.pending_resource_resolution is None
    assert state.protected_action_semantic_context is None
    assert state.pending_action is None


@pytest.mark.parametrize(
    ("tool_name", "arguments", "resource_id"),
    [
        ("freeze_card", {"card_id": str(CARD_ID)}, CARD_ID),
        (
            "create_dispute",
            {
                "transaction_id": str(TRANSACTION_ID),
                "reason_code": "unauthorized",
            },
            TRANSACTION_ID,
        ),
    ],
)
@pytest.mark.asyncio
async def test_matching_confirmation_never_bypasses_authentication(
    tool_name: str,
    arguments: dict[str, str],
    resource_id: UUID,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    context = ToolExecutionContext(
        authenticated=False,
        confirmation=ActionConfirmation(
            action=tool_name,
            resource_id=resource_id,
            confirmed=True,
        ),
    )

    with pytest.raises(ToolAuthenticationError):
        await ToolExecutor().execute(tool_name, arguments, context, session)

    session.scalar.assert_not_awaited()


@pytest.mark.parametrize(
    ("tool_name", "arguments", "active_intent", "active_field", "resource_id"),
    [
        (
            "get_account_balance",
            {"account_id": str(ACCOUNT_ID)},
            "get_account_balance",
            "active_account_id",
            ACCOUNT_ID,
        ),
        (
            "get_recent_transactions",
            {"account_id": str(ACCOUNT_ID)},
            "get_recent_transactions",
            "active_account_id",
            ACCOUNT_ID,
        ),
        (
            "get_transaction_details",
            {"transaction_id": str(TRANSACTION_ID)},
            "get_transaction_details",
            "active_transaction_id",
            TRANSACTION_ID,
        ),
        (
            "get_card_status",
            {"card_id": str(CARD_ID)},
            "get_card_status",
            "active_card_id",
            CARD_ID,
        ),
    ],
)
@pytest.mark.asyncio
async def test_private_read_tools_never_invoke_protected_verifier(
    tool_name: str,
    arguments: dict[str, str],
    active_intent: str,
    active_field: str,
    resource_id: UUID,
) -> None:
    state = _state()
    state.active_intent = active_intent
    setattr(state, active_field, resource_id)
    verifier = ScriptedVerifier(ProtectedActionSemanticDecision.EXPLICIT_CURRENT_ACTION)
    executor = AsyncMock()
    output = MagicMock()
    output.model_dump.return_value = {"ok": True}
    executor.execute.return_value = output
    orchestrator = AgentOrchestrator(
        llm=SequenceLLM(LLMResponse(content="Done.", model="test-model")),
        tool_executor=executor,
        protected_action_verifier=verifier,
    )

    result = await orchestrator._run_model_loop(
        messages=[],
        state=state,
        db=AsyncMock(spec=AsyncSession),
        initial_response=_tool_call(tool_name, **arguments),
        user_text="Read my banking information",
    )

    assert result.status == AgentTurnStatus.RESPONDED
    assert verifier.calls == []
    executor.execute.assert_awaited_once()


def test_closeout_contract_freezes_all_invariants_and_source_hashes() -> None:
    contract_bytes = closeout.CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(contract_bytes).hexdigest() == (
        closeout.EXPECTED_CONTRACT_SHA256
    )
    contract = json.loads(contract_bytes)
    assert tuple(row["id"] for row in contract["invariants"]) == tuple(
        "ABCDEFGHIJKL"
    )
    assert all(
        row["coverage_after"] == "SUFFICIENT" for row in contract["invariants"]
    )
    for binding in contract["source_bindings"].values():
        path = closeout.ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    assert contract["semantic_evaluation_status"] == "FRESH_SEMANTIC_PASS"
    assert contract["governance"]["requires_real_provider"] is False
    assert contract["governance"]["runtime_configuration_change_authorized"] is (
        False
    )
    assert contract["governance"]["step29i_authorized"] is False
    assert contract["final_holdout_policy"]["final_holdout_access_permitted"] is (
        False
    )


def test_closeout_preflight_is_read_only() -> None:
    result_existed = closeout.RESULT_PATH.exists()
    result_hash = (
        hashlib.sha256(closeout.RESULT_PATH.read_bytes()).hexdigest()
        if result_existed
        else None
    )

    report = closeout.preflight()

    assert report["status"] == "READY"
    assert report["invariant_count"] == 12
    assert report["provider_calls_performed"] is False
    assert report["files_written"] is False
    assert closeout.RESULT_PATH.exists() is result_existed
    if result_existed:
        assert hashlib.sha256(closeout.RESULT_PATH.read_bytes()).hexdigest() == (
            result_hash
        )


def test_closeout_result_recomputes_strict_pass_and_governance() -> None:
    report = closeout.check_results()

    assert report["results_valid"] is True
    assert report["final_status"] == "RUNTIME_SAFETY_PASS"
    assert report["total_required_invariants"] == 12
    assert report["passed_invariants"] == 12
    assert report["failed_invariants"] == 0
    assert report["provider_calls_performed"] is False
    assert report["files_written"] is False
    assert report["final_holdout_accessed"] is False
    assert report["step29i_authorized"] is False


def test_closeout_refuses_result_overwrite() -> None:
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        closeout.run()


def test_closeout_prohibited_holdout_guard() -> None:
    with pytest.raises(PermissionError):
        closeout.read_bytes(closeout.PROHIBITED_HOLDOUT_PATH)
