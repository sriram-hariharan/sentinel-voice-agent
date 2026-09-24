import json
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.prompt import SYSTEM_PROMPT
from backend.app.agent.tool_schemas import build_llm_tool_schemas
from backend.app.conversation.state import (
    ConversationPhase,
    ConversationState,
)
from backend.app.providers.llm import (
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    LLMUsage,
)
from backend.app.tools.errors import ToolError
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import TOOL_REGISTRY


class AgentTurnStatus(StrEnum):
    RESPONDED = "RESPONDED"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    TOOL_ERROR = "TOOL_ERROR"


class AgentOrchestrationError(RuntimeError):
    """Raised when the agent cannot safely continue orchestration."""


class AgentTurnResult(BaseModel):
    text: str
    status: AgentTurnStatus
    executed_tools: list[str] = []
    usage: LLMUsage = LLMUsage()

    model_config = ConfigDict(frozen=True)


class ConfirmationDecision(StrEnum):
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    OTHER = "OTHER"


_AFFIRMATIVE_CONFIRMATIONS = {
    "yes",
    "yes please",
    "confirm",
    "confirmed",
    "do it",
    "go ahead",
    "please do",
    "please do it",
}

_NEGATIVE_CONFIRMATIONS = {
    "no",
    "no thanks",
    "cancel",
    "cancel it",
    "stop",
    "don't",
    "do not",
    "don't do it",
    "do not do it",
    "actually don't",
    "actually do not",
    "wait",
}


def _normalize_confirmation_text(text: str) -> str:
    return " ".join(
        text.strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
        .split()
    )


def _classify_confirmation(text: str) -> ConfirmationDecision:
    normalized = _normalize_confirmation_text(text)

    if normalized in _AFFIRMATIVE_CONFIRMATIONS:
        return ConfirmationDecision.CONFIRM

    if normalized in _NEGATIVE_CONFIRMATIONS:
        return ConfirmationDecision.CANCEL

    return ConfirmationDecision.OTHER


def _merge_usage(left: LLMUsage, right: LLMUsage) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=(
            left.completion_tokens + right.completion_tokens
        ),
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _confirmation_prompt(tool_name: str) -> str:
    if tool_name == "freeze_card":
        return (
            "I’m ready to freeze that card. "
            "Would you like me to freeze it now?"
        )

    if tool_name == "create_dispute":
        return (
            "I’m ready to create the dispute for that transaction. "
            "Would you like me to submit it now?"
        )

    return "Would you like me to proceed with that action?"


def _tool_call_message(
    response: LLMResponse,
    tool_call: LLMToolCall,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _tool_result_message(
    tool_call_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload),
    }


class AgentOrchestrator:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        tool_executor: ToolExecutor | None = None,
        max_tool_calls: int = 3,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")

        self._llm = llm
        self._tool_executor = tool_executor or ToolExecutor()
        self._max_tool_calls = max_tool_calls

    async def handle_text_turn(
        self,
        *,
        user_text: str,
        state: ConversationState,
        db: AsyncSession,
    ) -> AgentTurnResult:
        user_text = user_text.strip()

        if not user_text:
            raise ValueError("user_text must not be empty")

        if state.phase in {
            ConversationPhase.ENDED,
            ConversationPhase.FAILED,
        }:
            raise AgentOrchestrationError(
                "Cannot process a turn for a terminal conversation"
            )

        state.phase = ConversationPhase.USER_SPEAKING

        pending = state.pending_action

        if (
            pending is not None
            and pending.confirmation_required
            and not pending.confirmation_received
        ):
            decision = _classify_confirmation(user_text)

            if decision == ConfirmationDecision.CONFIRM:
                return await self._execute_confirmed_action(
                    user_text=user_text,
                    state=state,
                    db=db,
                )

            state.cancel_pending_action()

            if decision == ConfirmationDecision.CANCEL:
                state.phase = ConversationPhase.AGENT_SPEAKING

                return AgentTurnResult(
                    text="Okay, I won’t perform that action.",
                    status=AgentTurnStatus.RESPONDED,
                )

            # Any non-explicit confirmation invalidates the old pending
            # confirmation and is processed as a new user request.

        state.phase = ConversationPhase.PROCESSING

        messages = self._build_messages(
            user_text=user_text,
            state=state,
        )

        return await self._run_model_loop(
            messages=messages,
            state=state,
            db=db,
        )

    def _allowed_tool_names(
        self,
        state: ConversationState,
    ) -> set[str]:
        allowed = {"escalate_to_human"}

        if state.authenticated:
            allowed.update(
                {
                    "get_account_balance",
                    "get_recent_transactions",
                    "get_transaction_details",
                    "get_card_status",
                    "freeze_card",
                    "create_dispute",
                }
            )

        return allowed

    def _build_messages(
        self,
        *,
        user_text: str,
        state: ConversationState,
    ) -> list[dict[str, Any]]:
        context_lines = [
            f"Authenticated session: {str(state.authenticated).lower()}",
        ]

        if state.active_account_id is not None:
            context_lines.append(
                f"Active account ID: {state.active_account_id}"
            )

        if state.active_card_id is not None:
            context_lines.append(
                f"Active card ID: {state.active_card_id}"
            )

        if state.active_transaction_id is not None:
            context_lines.append(
                f"Active transaction ID: {state.active_transaction_id}"
            )

        if state.conversation_summary:
            context_lines.append(
                f"Conversation summary: {state.conversation_summary}"
            )

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "system",
                "content": "\n".join(context_lines),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

    async def _run_model_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        state: ConversationState,
        db: AsyncSession,
    ) -> AgentTurnResult:
        executed_tools: list[str] = []
        total_usage = LLMUsage()
        tool_calls_used = 0

        tool_schemas = build_llm_tool_schemas(
            allowed_names=self._allowed_tool_names(state)
        )

        while True:
            response = await self._llm.generate(
                messages=messages,
                tools=tool_schemas,
            )

            total_usage = _merge_usage(
                total_usage,
                response.usage,
            )

            if not response.tool_calls:
                if not response.content.strip():
                    state.phase = ConversationPhase.FAILED
                    raise AgentOrchestrationError(
                        "LLM returned neither text nor a tool call"
                    )

                state.phase = ConversationPhase.AGENT_SPEAKING

                return AgentTurnResult(
                    text=response.content.strip(),
                    status=AgentTurnStatus.RESPONDED,
                    executed_tools=executed_tools,
                    usage=total_usage,
                )

            if len(response.tool_calls) != 1:
                state.phase = ConversationPhase.FAILED
                raise AgentOrchestrationError(
                    "Only one tool call per model response is supported"
                )

            if tool_calls_used >= self._max_tool_calls:
                state.phase = ConversationPhase.FAILED
                raise AgentOrchestrationError(
                    "Maximum tool calls exceeded"
                )

            tool_calls_used += 1
            tool_call = response.tool_calls[0]

            registered = TOOL_REGISTRY.get(tool_call.name)

            if registered is None:
                messages.extend(
                    [
                        _tool_call_message(
                            response,
                            tool_call,
                            tool_call.arguments,
                        ),
                        _tool_result_message(
                            tool_call.id,
                            {
                                "ok": False,
                                "error": "unknown_tool",
                            },
                        ),
                    ]
                )
                continue

            try:
                validated_request = (
                    registered.input_model.model_validate(
                        tool_call.arguments
                    )
                )
            except ValidationError:
                messages.extend(
                    [
                        _tool_call_message(
                            response,
                            tool_call,
                            tool_call.arguments,
                        ),
                        _tool_result_message(
                            tool_call.id,
                            {
                                "ok": False,
                                "error": "invalid_tool_arguments",
                            },
                        ),
                    ]
                )
                continue

            normalized_arguments = validated_request.model_dump(
                mode="json"
            )

            if registered.definition.requires_confirmation:
                resource_field = (
                    registered.definition.confirmation_resource_field
                )

                if resource_field is None:
                    state.phase = ConversationPhase.FAILED
                    raise AgentOrchestrationError(
                        f"{tool_call.name} requires confirmation "
                        "but has no confirmation resource field"
                    )

                resource_value = getattr(
                    validated_request,
                    resource_field,
                    None,
                )

                if not isinstance(resource_value, UUID):
                    state.phase = ConversationPhase.FAILED
                    raise AgentOrchestrationError(
                        f"{tool_call.name} has invalid confirmation resource"
                    )

                state.request_action(
                    tool_call.name,
                    resource_value,
                    arguments=normalized_arguments,
                    confirmation_required=True,
                )

                return AgentTurnResult(
                    text=_confirmation_prompt(tool_call.name),
                    status=AgentTurnStatus.WAITING_FOR_CONFIRMATION,
                    executed_tools=executed_tools,
                    usage=total_usage,
                )

            state.phase = ConversationPhase.TOOL_EXECUTION

            try:
                tool_result = await self._tool_executor.execute(
                    tool_call.name,
                    normalized_arguments,
                    state.to_tool_context(),
                    db,
                )
            except ToolError as exc:
                state.phase = ConversationPhase.PROCESSING

                messages.extend(
                    [
                        _tool_call_message(
                            response,
                            tool_call,
                            normalized_arguments,
                        ),
                        _tool_result_message(
                            tool_call.id,
                            {
                                "ok": False,
                                "error": type(exc).__name__,
                                "message": str(exc),
                            },
                        ),
                    ]
                )
                continue

            executed_tools.append(tool_call.name)

            payload = tool_result.model_dump(mode="json")
            state.last_tool_result = payload

            if tool_call.name == "escalate_to_human":
                state.mark_escalated()
            else:
                state.phase = ConversationPhase.PROCESSING

            messages.extend(
                [
                    _tool_call_message(
                        response,
                        tool_call,
                        normalized_arguments,
                    ),
                    _tool_result_message(
                        tool_call.id,
                        {
                            "ok": True,
                            "result": payload,
                        },
                    ),
                ]
            )

    async def _execute_confirmed_action(
        self,
        *,
        user_text: str,
        state: ConversationState,
        db: AsyncSession,
    ) -> AgentTurnResult:
        pending = state.pending_action

        if pending is None:
            raise AgentOrchestrationError(
                "No pending action exists"
            )

        state.confirm_pending_action()
        state.phase = ConversationPhase.TOOL_EXECUTION

        try:
            tool_result = await self._tool_executor.execute(
                pending.action,
                pending.arguments,
                state.to_tool_context(),
                db,
            )
        except ToolError:
            # Confirmation is one-use. Never automatically retry an
            # uncertain protected write using a stale confirmation.
            state.cancel_pending_action()
            state.phase = ConversationPhase.AGENT_SPEAKING

            return AgentTurnResult(
                text=(
                    "I couldn’t complete that protected action safely. "
                    "Please try again or ask me to connect you to a human."
                ),
                status=AgentTurnStatus.TOOL_ERROR,
            )

        payload = tool_result.model_dump(mode="json")
        state.last_tool_result = payload

        action = pending.action
        resource_id = pending.resource_id
        arguments = dict(pending.arguments)

        state.complete_pending_action(
            action,
            resource_id,
        )

        synthetic_call = LLMToolCall(
            id="confirmed-protected-action",
            name=action,
            arguments=arguments,
        )

        synthetic_response = LLMResponse(
            content="",
            tool_calls=[synthetic_call],
            model="application-confirmed-action",
        )

        messages = self._build_messages(
            user_text=user_text,
            state=state,
        )

        messages.extend(
            [
                _tool_call_message(
                    synthetic_response,
                    synthetic_call,
                    arguments,
                ),
                _tool_result_message(
                    synthetic_call.id,
                    {
                        "ok": True,
                        "result": payload,
                    },
                ),
            ]
        )

        final_response = await self._llm.generate(
            messages=messages,
            tools=None,
        )

        if (
            final_response.tool_calls
            or not final_response.content.strip()
        ):
            state.phase = ConversationPhase.FAILED
            raise AgentOrchestrationError(
                "LLM did not return a final response "
                "after protected action execution"
            )

        state.phase = ConversationPhase.AGENT_SPEAKING

        return AgentTurnResult(
            text=final_response.content.strip(),
            status=AgentTurnStatus.RESPONDED,
            executed_tools=[action],
            usage=final_response.usage,
        )
