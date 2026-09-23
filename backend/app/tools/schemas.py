from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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


class RecentTransactionsInput(BaseModel):
    account_id: UUID
    limit: int = Field(default=10, ge=1, le=50)


class TransactionSummary(BaseModel):
    transaction_id: UUID
    merchant_name: str
    amount: Decimal
    currency: str
    transaction_timestamp: datetime
    status: str


class RecentTransactionsOutput(BaseModel):
    account_id: UUID
    transactions: list[TransactionSummary]


class TransactionDetailsInput(BaseModel):
    transaction_id: UUID


class TransactionDetailsOutput(BaseModel):
    transaction_id: UUID
    account_id: UUID
    card_id: UUID | None
    merchant_name: str
    merchant_category: str | None
    amount: Decimal
    currency: str
    transaction_timestamp: datetime
    posted_timestamp: datetime | None
    status: str
    transaction_type: str
    location: str | None


class CardStatusInput(BaseModel):
    card_id: UUID


class CardStatusOutput(BaseModel):
    card_id: UUID
    account_id: UUID
    masked_card_number: str
    card_type: str
    status: str
    expiration_month: int
    expiration_year: int
