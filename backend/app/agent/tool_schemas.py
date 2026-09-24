from collections.abc import Mapping

from backend.app.tools.registry import (
    TOOL_REGISTRY,
    RegisteredTool,
)


def build_llm_tool_schemas(
    registry: Mapping[str, RegisteredTool] | None = None,
    *,
    allowed_names: set[str] | None = None,
) -> list[dict]:
    registry = registry or TOOL_REGISTRY

    schemas: list[dict] = []

    for name, registered in registry.items():
        if allowed_names is not None and name not in allowed_names:
            continue

        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": registered.definition.name,
                    "description": registered.definition.description,
                    "parameters": registered.input_model.model_json_schema(),
                },
            }
        )

    return schemas
