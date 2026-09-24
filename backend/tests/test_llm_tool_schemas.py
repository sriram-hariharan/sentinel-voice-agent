from backend.app.agent.tool_schemas import build_llm_tool_schemas


def test_all_seven_v1_tools_have_model_schemas() -> None:
    schemas = build_llm_tool_schemas()

    names = {
        schema["function"]["name"]
        for schema in schemas
    }

    assert names == {
        "get_account_balance",
        "get_recent_transactions",
        "get_transaction_details",
        "get_card_status",
        "freeze_card",
        "create_dispute",
        "escalate_to_human",
    }

    assert len(schemas) == 7


def test_model_tool_schemas_have_descriptions_and_json_schema() -> None:
    schemas = build_llm_tool_schemas()

    for schema in schemas:
        function = schema["function"]

        assert function["description"]
        assert function["parameters"]["type"] == "object"


def test_tool_schemas_can_be_permission_filtered() -> None:
    schemas = build_llm_tool_schemas(
        allowed_names={
            "get_account_balance",
            "escalate_to_human",
        }
    )

    assert {
        schema["function"]["name"]
        for schema in schemas
    } == {
        "get_account_balance",
        "escalate_to_human",
    }
