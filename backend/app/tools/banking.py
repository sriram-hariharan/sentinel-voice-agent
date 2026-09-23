from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Account, Card, Transaction
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolResourceNotFoundError,
)
from backend.app.tools.schemas import (
    AccountBalanceInput,
    AccountBalanceOutput,
    CardStatusInput,
    CardStatusOutput,
    RecentTransactionsInput,
    RecentTransactionsOutput,
    ToolExecutionContext,
    TransactionDetailsInput,
    TransactionDetailsOutput,
    TransactionSummary,
)


def _require_authenticated_customer(context: ToolExecutionContext) -> UUID:
    if not context.authenticated or context.customer_id is None:
        raise ToolAuthenticationError("Authenticated customer session required")

    return context.customer_id


async def _get_owned_account(
    account_id: UUID,
    customer_id: UUID,
    session: AsyncSession,
) -> Account:
    statement = select(Account).where(
        Account.account_id == account_id,
        Account.customer_id == customer_id,
    )

    account = await session.scalar(statement)

    if account is None:
        raise ToolResourceNotFoundError("Account not found")

    return account


async def get_account_balance(
    request: AccountBalanceInput,
    context: ToolExecutionContext,
    session: AsyncSession,
) -> AccountBalanceOutput:
    customer_id = _require_authenticated_customer(context)

    account = await _get_owned_account(
        request.account_id,
        customer_id,
        session,
    )

    return AccountBalanceOutput(
        account_id=account.account_id,
        account_type=account.account_type,
        masked_account_number=account.masked_account_number,
        current_balance=account.current_balance,
        available_balance=account.available_balance,
        currency=account.currency,
        status=account.status,
    )


async def get_recent_transactions(
    request: RecentTransactionsInput,
    context: ToolExecutionContext,
    session: AsyncSession,
) -> RecentTransactionsOutput:
    customer_id = _require_authenticated_customer(context)

    await _get_owned_account(
        request.account_id,
        customer_id,
        session,
    )

    statement = (
        select(Transaction)
        .where(Transaction.account_id == request.account_id)
        .order_by(Transaction.transaction_timestamp.desc())
        .limit(request.limit)
    )

    result = await session.scalars(statement)
    transactions = result.all()

    return RecentTransactionsOutput(
        account_id=request.account_id,
        transactions=[
            TransactionSummary(
                transaction_id=transaction.transaction_id,
                merchant_name=transaction.merchant_name,
                amount=transaction.amount,
                currency=transaction.currency,
                transaction_timestamp=transaction.transaction_timestamp,
                status=transaction.status,
            )
            for transaction in transactions
        ],
    )


async def get_transaction_details(
    request: TransactionDetailsInput,
    context: ToolExecutionContext,
    session: AsyncSession,
) -> TransactionDetailsOutput:
    customer_id = _require_authenticated_customer(context)

    statement = (
        select(Transaction)
        .join(Account, Transaction.account_id == Account.account_id)
        .where(
            Transaction.transaction_id == request.transaction_id,
            Account.customer_id == customer_id,
        )
    )

    transaction = await session.scalar(statement)

    if transaction is None:
        raise ToolResourceNotFoundError("Transaction not found")

    return TransactionDetailsOutput(
        transaction_id=transaction.transaction_id,
        account_id=transaction.account_id,
        card_id=transaction.card_id,
        merchant_name=transaction.merchant_name,
        merchant_category=transaction.merchant_category,
        amount=transaction.amount,
        currency=transaction.currency,
        transaction_timestamp=transaction.transaction_timestamp,
        posted_timestamp=transaction.posted_timestamp,
        status=transaction.status,
        transaction_type=transaction.transaction_type,
        location=transaction.location,
    )


async def get_card_status(
    request: CardStatusInput,
    context: ToolExecutionContext,
    session: AsyncSession,
) -> CardStatusOutput:
    customer_id = _require_authenticated_customer(context)

    statement = select(Card).where(
        Card.card_id == request.card_id,
        Card.customer_id == customer_id,
    )

    card = await session.scalar(statement)

    if card is None:
        raise ToolResourceNotFoundError("Card not found")

    return CardStatusOutput(
        card_id=card.card_id,
        account_id=card.account_id,
        masked_card_number=card.masked_card_number,
        card_type=card.card_type,
        status=card.status,
        expiration_month=card.expiration_month,
        expiration_year=card.expiration_year,
    )
