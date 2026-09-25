import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.observability.events import TraceStatus
from backend.app.observability.tracing import emit_trace_event, trace_span
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolBackendError,
    ToolConfirmationError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from backend.app.tools.registry import TOOL_REGISTRY, RegisteredTool
from backend.app.tools.schemas import ToolExecutionContext


class ToolExecutor:
    def __init__(
        self,
        registry: Mapping[str, RegisteredTool] | None = None,
    ) -> None:
        self._registry = registry or TOOL_REGISTRY

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        session: AsyncSession,
    ) -> BaseModel:
        registered = self._registry.get(tool_name)

        if registered is None:
            raise ToolNotFoundError(f"Unknown tool: {tool_name}")

        definition = registered.definition

        if (
            definition.requires_authentication
            and (
                not context.authenticated
                or context.customer_id is None
            )
        ):
            emit_trace_event(
                "authorization.checked",
                component="tools",
                status=TraceStatus.FAILED,
                error_category="authorization_denied",
                metadata={
                    "tool_name": tool_name,
                    "permission_level": definition.permission_level.value,
                    "authorization_decision": "denied",
                    "reason": "authentication_required",
                },
            )
            raise ToolAuthenticationError(
                f"{tool_name} requires an authenticated customer session"
            )

        if definition.requires_confirmation:
            confirmation = context.confirmation

            if (
                confirmation is None
                or not confirmation.confirmed
                or confirmation.action != tool_name
            ):
                emit_trace_event(
                    "authorization.checked",
                    component="tools",
                    status=TraceStatus.FAILED,
                    error_category="authorization_denied",
                    metadata={
                        "tool_name": tool_name,
                        "permission_level": definition.permission_level.value,
                        "authorization_decision": "denied",
                        "reason": "confirmation_required",
                    },
                )
                raise ToolConfirmationError(
                    f"{tool_name} requires explicit confirmation"
                )

        emit_trace_event(
            "authorization.checked",
            component="tools",
            status=TraceStatus.COMPLETED,
            metadata={
                "tool_name": tool_name,
                "permission_level": definition.permission_level.value,
                "authorization_decision": "allowed",
                "confirmation_present": context.confirmation is not None,
            },
        )

        try:
            async with trace_span(
                "tool.execution",
                component="tools",
                metadata={
                    "tool_name": tool_name,
                    "permission_level": definition.permission_level.value,
                    "arguments": arguments,
                },
            ):
                try:
                    request = registered.input_model.model_validate(arguments)
                except ValidationError as exc:
                    raise ToolValidationError(
                        f"Invalid arguments for {tool_name}"
                    ) from exc
                async with asyncio.timeout(definition.timeout_seconds):
                    return await registered.handler(
                        request,
                        context,
                        session,
                    )
        except TimeoutError as exc:
            await session.rollback()
            raise ToolTimeoutError(
                f"{tool_name} exceeded its "
                f"{definition.timeout_seconds:g}s timeout"
            ) from exc
        except SQLAlchemyError as exc:
            await session.rollback()
            raise ToolBackendError(
                f"{tool_name} backend execution failed"
            ) from exc
