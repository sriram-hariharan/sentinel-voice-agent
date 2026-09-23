from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Account
from backend.app.tools.errors import (
    ToolAuthenticationError,
    ToolResourceNotFoundError,
)
from backend.app.tools.schemas import (
    AccountBalanceInput,
    AccountBalanceOutput,
    ToolExecutionContext,
)


async def get_account_balance(
    request: AccountBalanceInput,
    context: ToolExecutionContext,
    session: AsyncSession,
) -> AccountBalanceOutput:
    if not context.authenticated or context.customer_id is None:
        raise ToolAuthenticationError("Authenticated customer session required")

    statement = select(Account).where(
        Account.account_id == request.account_id,
        Account.customer_id == context.customer_id,
    )

    account = await session.scalar(statement)

    if account is None:
        raise ToolResourceNotFoundError("Account not found")

    return AccountBalanceOutput(
        account_id=account.account_id,
        account_type=account.account_type,
        masked_account_number=account.masked_account_number,
        current_balance=account.current_balance,
        available_balance=account.available_balance,
        currency=account.currency,
        status=account.status,
    )
