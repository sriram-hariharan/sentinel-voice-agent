from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.tools.banking import (
    create_dispute,
    freeze_card,
    get_account_balance,
    get_card_status,
    get_recent_transactions,
    get_transaction_details,
)
from backend.app.tools.definitions import (
    CREATE_DISPUTE,
    ESCALATE_TO_HUMAN,
    FREEZE_CARD,
    GET_ACCOUNT_BALANCE,
    GET_CARD_STATUS,
    GET_RECENT_TRANSACTIONS,
    GET_TRANSACTION_DETAILS,
    ToolDefinition,
)
from backend.app.tools.schemas import (
    AccountBalanceInput,
    CardStatusInput,
    CreateDisputeInput,
    EscalateToHumanInput,
    FreezeCardInput,
    RecentTransactionsInput,
    ToolExecutionContext,
    TransactionDetailsInput,
)
from backend.app.tools.support import escalate_to_human

ToolHandler = Callable[
    [BaseModel, ToolExecutionContext, AsyncSession],
    Awaitable[BaseModel],
]


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    input_model: type[BaseModel]
    handler: ToolHandler


TOOL_REGISTRY: dict[str, RegisteredTool] = {
    GET_ACCOUNT_BALANCE.name: RegisteredTool(
        definition=GET_ACCOUNT_BALANCE,
        input_model=AccountBalanceInput,
        handler=get_account_balance,
    ),
    GET_RECENT_TRANSACTIONS.name: RegisteredTool(
        definition=GET_RECENT_TRANSACTIONS,
        input_model=RecentTransactionsInput,
        handler=get_recent_transactions,
    ),
    GET_TRANSACTION_DETAILS.name: RegisteredTool(
        definition=GET_TRANSACTION_DETAILS,
        input_model=TransactionDetailsInput,
        handler=get_transaction_details,
    ),
    GET_CARD_STATUS.name: RegisteredTool(
        definition=GET_CARD_STATUS,
        input_model=CardStatusInput,
        handler=get_card_status,
    ),
    FREEZE_CARD.name: RegisteredTool(
        definition=FREEZE_CARD,
        input_model=FreezeCardInput,
        handler=freeze_card,
    ),
    CREATE_DISPUTE.name: RegisteredTool(
        definition=CREATE_DISPUTE,
        input_model=CreateDisputeInput,
        handler=create_dispute,
    ),
    ESCALATE_TO_HUMAN.name: RegisteredTool(
        definition=ESCALATE_TO_HUMAN,
        input_model=EscalateToHumanInput,
        handler=escalate_to_human,
    ),
}
