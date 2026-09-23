from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ToolExecutionContext(BaseModel):
    customer_id: UUID | None = None
    authenticated: bool = False

    model_config = ConfigDict(frozen=True)


class AccountBalanceInput(BaseModel):
    account_id: UUID


class AccountBalanceOutput(BaseModel):
    account_id: UUID
    account_type: str
    masked_account_number: str
    current_balance: Decimal
    available_balance: Decimal
    currency: str
    status: str
