import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.tools.errors import (
    ToolAuthenticationError,
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
                raise ToolConfirmationError(
                    f"{tool_name} requires explicit confirmation"
                )

        try:
            request = registered.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(
                f"Invalid arguments for {tool_name}"
            ) from exc

        try:
            async with asyncio.timeout(definition.timeout_seconds):
                return await registered.handler(
                    request,
                    context,
                    session,
                )
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"{tool_name} exceeded its "
                f"{definition.timeout_seconds:g}s timeout"
            ) from exc
