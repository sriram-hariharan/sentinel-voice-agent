from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionContext(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    customer_id: UUID | None = None
    authenticated: bool = False
    confirmation: "ActionConfirmation | None" = None

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


class ActionConfirmation(BaseModel):
    action: str
    resource_id: UUID
    confirmed: bool

    model_config = ConfigDict(frozen=True)


class FreezeCardInput(BaseModel):
    card_id: UUID


class FreezeCardOutput(BaseModel):
    card_id: UUID
    masked_card_number: str
    previous_status: str
    status: str
    changed: bool


ToolExecutionContext.model_rebuild()




class CreateDisputeInput(BaseModel):
    transaction_id: UUID
    reason_code: str = Field(min_length=1, max_length=64)
    notes: str | None = None


class CreateDisputeOutput(BaseModel):
    dispute_id: UUID
    transaction_id: UUID
    reason_code: str
    status: str
    created: bool


class EscalateToHumanInput(BaseModel):
    category: str = Field(min_length=1, max_length=64)
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"]
    summary: str = Field(min_length=1, max_length=1000)
    handoff_reason: str = Field(min_length=1, max_length=1000)
    transaction_id: UUID | None = None
    transaction_amount: Decimal | None = None
    merchant: str | None = Field(default=None, max_length=255)
    actions_completed: list[str] = Field(default_factory=list)
    actions_not_completed: list[str] = Field(default_factory=list)
    conversation_summary: str = Field(min_length=1, max_length=4000)


class HandoffSummary(BaseModel):
    customer_id: UUID | None
    authenticated: bool
    category: str
    priority: str
    summary: str
    transaction_id: UUID | None
    transaction_amount: Decimal | None
    merchant: str | None
    actions_completed: list[str]
    actions_not_completed: list[str]
    reason_for_handoff: str
    conversation_summary: str


class EscalateToHumanOutput(BaseModel):
    case_id: UUID
    status: str
    created: bool
    handoff: HandoffSummary
