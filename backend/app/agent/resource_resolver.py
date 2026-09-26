import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.conversation.state import (
    ConversationState,
    PendingResourceResolution,
    ResourceCandidate,
    ResourceType,
)
from backend.app.db.models import Account, Card, Transaction


@dataclass(frozen=True)
class ResourceResolution:
    kind: ResourceType | None = None
    clarification: str | None = None
    selected_from_pending: bool = False


_ACCOUNT_INTENTS = {"get_account_balance", "get_recent_transactions"}
_CARD_INTENTS = {"get_card_status", "freeze_card"}
_TRANSACTION_INTENTS = {"get_transaction_details", "create_dispute"}

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_ORDINALS = {
    "first": 0,
    "first one": 0,
    "second": 1,
    "second one": 1,
    "third": 2,
    "third one": 2,
    "fourth": 3,
    "fourth one": 3,
}

_TRANSACTION_STATUSES = ("pending", "posted", "reversed", "declined")


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


def _masked_suffix(masked_number: str) -> str:
    digits = re.sub(r"\D", "", masked_number)
    return digits[-4:]


def _requested_suffix(text: str) -> str | None:
    match = re.search(
        r"(?:ending(?:\s+in)?|ends(?:\s+in)?|last\s+four(?:\s+digits)?(?:\s+are)?)"
        r"\s*(\d{4})\b",
        text,
    )
    if match is None:
        match = re.search(r"\bcards?\s+(\d{4})\b", text)
    return match.group(1) if match else None


def _amount_from_text(text: str) -> Decimal | None:
    match = re.search(r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)", text)

    if match is None:
        match = re.search(
            r"(?:amount|for)\s+(?:was\s+)?(?:\$\s*)?"
            r"(\d[\d,]*\.\d{2})\b",
            text,
        )

    if match is None:
        return None

    try:
        return Decimal(match.group(1).replace(",", "")).quantize(
            Decimal("0.01")
        )
    except InvalidOperation:
        return None


def _date_from_text(text: str) -> tuple[int | None, int, int] | None:
    iso_match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)

    if iso_match:
        return (
            int(iso_match.group(1)),
            int(iso_match.group(2)),
            int(iso_match.group(3)),
        )

    month_names = "|".join(_MONTHS)
    named_match = re.search(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        r"(?:,?\s+(20\d{2}))?\b",
        text,
    )

    if named_match is None:
        return None

    return (
        int(named_match.group(3)) if named_match.group(3) else None,
        _MONTHS[named_match.group(1)],
        int(named_match.group(2)),
    )


def _date_matches(value: date, requested: tuple[int | None, int, int]) -> bool:
    year, month, day = requested
    return (
        value.month == month
        and value.day == day
        and (year is None or value.year == year)
    )


def _number_word(count: int) -> str:
    words = {2: "two", 3: "three", 4: "four", 5: "five"}
    return words.get(count, str(count))


def _candidate_matches(text: str, candidate: ResourceCandidate) -> bool:
    for selector in candidate.selectors:
        normalized = _normalize(selector)

        if normalized.isdigit():
            if re.search(rf"\b{re.escape(normalized)}\b", text):
                return True
        elif re.search(rf"\b{re.escape(normalized)}\b", text):
            return True

    return False


def _transaction_merchant_from_text(text: str) -> str | None:
    match = re.search(
        r"\b(?:about|with)\s+(?:the\s+)?(?P<merchant>.+?)\s+"
        r"(?:transaction|charge|purchase|payment)\b",
        text,
    )

    if match is None:
        return None

    merchant = match.group("merchant")
    merchant = re.sub(
        rf"^(?:{'|'.join(_TRANSACTION_STATUSES)})\s+",
        "",
        merchant,
    )
    merchant = re.sub(
        rf"\s+(?:{'|'.join(_TRANSACTION_STATUSES)})$",
        "",
        merchant,
    )
    return merchant or None


def _dates_are_compatible(
    requested: tuple[int | None, int, int],
    candidate: tuple[int | None, int, int],
) -> bool:
    requested_year, requested_month, requested_day = requested
    candidate_year, candidate_month, candidate_day = candidate
    return (
        requested_month == candidate_month
        and requested_day == candidate_day
        and (
            requested_year is None
            or candidate_year is None
            or requested_year == candidate_year
        )
    )


def _transaction_candidate_matches(
    text: str,
    candidate: ResourceCandidate,
) -> bool:
    if not _candidate_matches(text, candidate):
        return False

    requested_amount = _amount_from_text(text)
    candidate_amounts = {
        amount
        for selector in candidate.selectors
        if (amount := _amount_from_text(selector)) is not None
    }
    if requested_amount is not None and requested_amount not in candidate_amounts:
        return False

    requested_date = _date_from_text(text)
    candidate_dates = {
        candidate_date
        for selector in candidate.selectors
        if (candidate_date := _date_from_text(selector)) is not None
    }
    if requested_date is not None and not any(
        _dates_are_compatible(requested_date, candidate_date)
        for candidate_date in candidate_dates
    ):
        return False

    requested_statuses = {
        status
        for status in _TRANSACTION_STATUSES
        if re.search(rf"\b{status}\b", text)
    }
    candidate_selectors = {_normalize(selector) for selector in candidate.selectors}
    if requested_statuses and requested_statuses.isdisjoint(candidate_selectors):
        return False

    requested_merchant = _transaction_merchant_from_text(text)
    candidate_merchant = _normalize(candidate.selectors[0])
    return requested_merchant is None or requested_merchant == candidate_merchant


def _candidate_matches_pending_selector(
    text: str,
    pending: PendingResourceResolution,
    candidate: ResourceCandidate,
) -> bool:
    if pending.resource_type == ResourceType.TRANSACTION:
        return _transaction_candidate_matches(text, candidate)

    return _candidate_matches(text, candidate)


def _matches_pending_candidate_selector(
    text: str,
    pending: PendingResourceResolution,
) -> bool:
    for phrase, index in _ORDINALS.items():
        if (
            re.search(rf"\b{re.escape(phrase)}\b", text)
            and index < len(pending.candidates)
        ):
            return True

    return any(
        _candidate_matches_pending_selector(text, pending, candidate)
        for candidate in pending.candidates
    )


def _selected_candidate(
    text: str,
    pending: PendingResourceResolution,
) -> ResourceCandidate | None:
    for phrase, index in _ORDINALS.items():
        if (
            re.search(rf"\b{re.escape(phrase)}\b", text)
            and index < len(pending.candidates)
        ):
            return pending.candidates[index]

    matches = [
        candidate
        for candidate in pending.candidates
        if _candidate_matches_pending_selector(text, pending, candidate)
    ]

    return matches[0] if len(matches) == 1 else None


class ResourceResolver:
    async def resolve(
        self,
        *,
        user_text: str,
        state: ConversationState,
        db: AsyncSession,
    ) -> ResourceResolution:
        if not state.authenticated or state.customer_id is None:
            state.clear_resource_resolution()
            return ResourceResolution()

        text = _normalize(user_text)
        pending_result = self._resolve_pending(text, state)

        if pending_result is not None:
            return pending_result

        if self._is_account_request(text, state):
            return await self._resolve_account(text, state, db)

        if self._is_card_request(text, state):
            return await self._resolve_card(text, state, db)

        if self._is_transaction_request(text, state):
            return await self._resolve_transaction(text, state, db)

        return ResourceResolution()

    def _resolve_pending(
        self,
        text: str,
        state: ConversationState,
    ) -> ResourceResolution | None:
        pending = state.pending_resource_resolution

        if pending is None:
            return None

        selected = _selected_candidate(text, pending)

        if selected is not None:
            self._activate_candidate(
                state,
                pending.resource_type,
                pending.intent,
                selected,
            )
            return ResourceResolution(
                kind=pending.resource_type,
                selected_from_pending=True,
            )

        new_type = self._explicit_resource_type(text)

        if new_type is not None and new_type != pending.resource_type:
            self._clear_pending_context(state, pending.resource_type)
            return None

        if _matches_pending_candidate_selector(text, pending):
            return ResourceResolution(
                kind=pending.resource_type,
                clarification=pending.clarification,
            )

        self._clear_pending_context(state, pending.resource_type)
        return None

    @staticmethod
    def _explicit_resource_type(text: str) -> ResourceType | None:
        if re.search(r"\bcards?\b", text) or "freeze" in text:
            return ResourceType.CARD

        if "recent" in text and "transaction" in text:
            return ResourceType.ACCOUNT

        if any(word in text for word in ("checking", "savings", "balance", "account")):
            return ResourceType.ACCOUNT

        if any(
            word in text
            for word in ("charge", "purchase", "payment", "dispute", "transaction")
        ):
            return ResourceType.TRANSACTION

        return None

    @staticmethod
    def _is_account_request(text: str, state: ConversationState) -> bool:
        if "checking" in text or "savings" in text or "account" in text:
            return True

        if "balance" in text:
            return True

        if "recent" in text and "transaction" in text:
            return True

        return (
            state.active_intent in _ACCOUNT_INTENTS
            and any(word in text for word in ("account", "ending"))
        )

    @staticmethod
    def _is_card_request(text: str, state: ConversationState) -> bool:
        if re.search(r"\bcards?\b", text):
            return True

        return (
            state.active_intent in _CARD_INTENTS
            and (
                _requested_suffix(text) is not None
                or any(word in text for word in ("active", "frozen", "debit"))
            )
        )

    @staticmethod
    def _is_transaction_request(text: str, state: ConversationState) -> bool:
        if "recent" in text and "transaction" in text:
            return False

        if any(
            word in text
            for word in ("charge", "purchase", "payment", "dispute", "transaction")
        ):
            return True

        if _date_from_text(text) is not None or _amount_from_text(text) is not None:
            return True

        return state.active_intent in _TRANSACTION_INTENTS

    async def _resolve_account(
        self,
        text: str,
        state: ConversationState,
        db: AsyncSession,
    ) -> ResourceResolution:
        state.active_account_id = None

        if "balance" in text:
            state.active_intent = "get_account_balance"
        elif "transaction" in text:
            state.active_intent = "get_recent_transactions"

        intent = state.active_intent or "get_account_balance"
        result = await db.scalars(
            select(Account)
            .where(Account.customer_id == state.customer_id)
            .order_by(Account.account_type, Account.masked_account_number)
        )
        accounts = [
            account
            for account in result.all()
            if account.customer_id == state.customer_id
        ]

        requested_type = next(
            (kind for kind in ("checking", "savings") if kind in text),
            None,
        )
        suffix = _requested_suffix(text)

        if requested_type is not None:
            accounts = [
                account
                for account in accounts
                if account.account_type.casefold() == requested_type
            ]

        if suffix is not None:
            accounts = [
                account
                for account in accounts
                if _masked_suffix(account.masked_account_number) == suffix
            ]

        candidates = [self._account_candidate(account) for account in accounts]

        if len(candidates) == 1:
            self._activate_candidate(
                state,
                ResourceType.ACCOUNT,
                intent,
                candidates[0],
            )
            return ResourceResolution(kind=ResourceType.ACCOUNT)

        if not candidates:
            description = requested_type or "matching"
            return ResourceResolution(
                kind=ResourceType.ACCOUNT,
                clarification=f"I couldn’t find a {description} account for you.",
            )

        choices = " and ".join(f"a {candidate.label}" for candidate in candidates)
        clarification = f"I found {choices}. Which one do you mean?"
        state.request_resource_resolution(
            ResourceType.ACCOUNT,
            intent,
            candidates,
            clarification,
        )
        return ResourceResolution(
            kind=ResourceType.ACCOUNT,
            clarification=clarification,
        )

    async def _resolve_card(
        self,
        text: str,
        state: ConversationState,
        db: AsyncSession,
    ) -> ResourceResolution:
        state.active_card_id = None

        if "freeze" in text:
            state.active_intent = "freeze_card"
        elif state.active_intent not in _CARD_INTENTS:
            state.active_intent = "get_card_status"

        intent = state.active_intent or "get_card_status"
        result = await db.scalars(
            select(Card)
            .where(Card.customer_id == state.customer_id)
            .order_by(Card.card_type, Card.masked_card_number)
        )
        cards = [
            card
            for card in result.all()
            if card.customer_id == state.customer_id
        ]

        suffix = _requested_suffix(text)
        requested_type = next(
            (kind for kind in ("debit", "credit") if kind in text),
            None,
        )
        requested_status = next(
            (status for status in ("active", "frozen") if status in text),
            None,
        )

        if suffix is not None:
            cards = [
                card
                for card in cards
                if _masked_suffix(card.masked_card_number) == suffix
            ]

        if requested_type is not None:
            cards = [
                card
                for card in cards
                if card.card_type.casefold() == requested_type
            ]

        if requested_status is not None:
            cards = [
                card
                for card in cards
                if card.status.casefold() == requested_status
            ]

        candidates = [self._card_candidate(card) for card in cards]

        if len(candidates) == 1:
            self._activate_candidate(
                state,
                ResourceType.CARD,
                intent,
                candidates[0],
            )
            return ResourceResolution(kind=ResourceType.CARD)

        if not candidates:
            if suffix:
                description = f"card ending in {suffix}"
            elif requested_status:
                description = f"{requested_status} card"
            else:
                description = "matching card"

            return ResourceResolution(
                kind=ResourceType.CARD,
                clarification=f"I couldn’t find a {description} for you.",
            )

        suffixes = " and ".join(
            _masked_suffix(card.masked_card_number) for card in cards
        )
        card_type = (
            cards[0].card_type.casefold()
            if all(card.card_type == cards[0].card_type for card in cards)
            else ""
        )
        description = f" {card_type}" if card_type else ""
        clarification = (
            f"I found {_number_word(len(cards))}{description} cards ending in "
            f"{suffixes}. Which one do you mean?"
        )
        state.request_resource_resolution(
            ResourceType.CARD,
            intent,
            candidates,
            clarification,
        )
        return ResourceResolution(
            kind=ResourceType.CARD,
            clarification=clarification,
        )

    async def _resolve_transaction(
        self,
        text: str,
        state: ConversationState,
        db: AsyncSession,
    ) -> ResourceResolution:
        state.active_transaction_id = None

        if "dispute" in text:
            state.active_intent = "create_dispute"
        elif state.active_intent not in _TRANSACTION_INTENTS:
            state.active_intent = "get_transaction_details"

        intent = state.active_intent or "get_transaction_details"
        account_result = await db.scalars(
            select(Account).where(Account.customer_id == state.customer_id)
        )
        owned_account_ids = {
            account.account_id
            for account in account_result.all()
            if account.customer_id == state.customer_id
        }

        if not owned_account_ids:
            return ResourceResolution(
                kind=ResourceType.TRANSACTION,
                clarification="I couldn’t find any accounts for this session.",
            )

        transaction_result = await db.scalars(
            select(Transaction)
            .where(Transaction.account_id.in_(owned_account_ids))
            .order_by(Transaction.transaction_timestamp.desc())
        )
        transactions = [
            transaction
            for transaction in transaction_result.all()
            if transaction.account_id in owned_account_ids
        ]

        requested_merchants = {
            transaction.merchant_name.casefold()
            for transaction in transactions
            if transaction.merchant_name.casefold() in text
        }

        if requested_merchants:
            transactions = [
                transaction
                for transaction in transactions
                if transaction.merchant_name.casefold() in requested_merchants
            ]

        requested_amount = _amount_from_text(text)
        if requested_amount is not None:
            transactions = [
                transaction
                for transaction in transactions
                if transaction.amount == requested_amount
            ]

        requested_date = _date_from_text(text)
        if requested_date is not None:
            transactions = [
                transaction
                for transaction in transactions
                if _date_matches(
                    transaction.transaction_timestamp.date(),
                    requested_date,
                )
            ]

        candidates = [
            self._transaction_candidate(transaction)
            for transaction in transactions
        ]
        has_selector = bool(
            requested_merchants
            or requested_amount is not None
            or requested_date is not None
        )

        if len(candidates) == 1:
            self._activate_candidate(
                state,
                ResourceType.TRANSACTION,
                intent,
                candidates[0],
            )
            return ResourceResolution(kind=ResourceType.TRANSACTION)

        if not candidates:
            return ResourceResolution(
                kind=ResourceType.TRANSACTION,
                clarification=(
                    "I couldn’t find a matching transaction. Try the merchant, "
                    "amount, or date."
                ),
            )

        if has_selector and len(requested_merchants) == 1:
            merchant = transactions[0].merchant_name
            choices = " and ".join(
                f"${transaction.amount:.2f} on "
                f"{transaction.transaction_timestamp.strftime('%B %-d, %Y')}"
                for transaction in transactions
            )
            clarification = (
                f"I found {_number_word(len(transactions))} {merchant} transactions: "
                f"{choices}. Which one do you mean?"
            )
        else:
            clarification = (
                "Which transaction do you mean? You can identify it by "
                "merchant, amount, or date."
            )

        state.request_resource_resolution(
            ResourceType.TRANSACTION,
            intent,
            candidates,
            clarification,
        )
        return ResourceResolution(
            kind=ResourceType.TRANSACTION,
            clarification=clarification,
        )

    @staticmethod
    def _account_candidate(account: Account) -> ResourceCandidate:
        suffix = _masked_suffix(account.masked_account_number)
        return ResourceCandidate(
            resource_id=account.account_id,
            label=f"{account.account_type} account ending in {suffix}",
            selectors=(
                account.account_type.casefold(),
                suffix,
                f"ending in {suffix}",
            ),
        )

    @staticmethod
    def _card_candidate(card: Card) -> ResourceCandidate:
        suffix = _masked_suffix(card.masked_card_number)
        card_type = card.card_type.casefold()
        status = card.status.casefold()
        return ResourceCandidate(
            resource_id=card.card_id,
            label=f"{status} {card_type} card ending in {suffix}",
            selectors=(
                suffix,
                f"ending in {suffix}",
                status,
                f"{status} card",
                card_type,
            ),
            related_account_id=card.account_id,
        )

    @staticmethod
    def _transaction_candidate(transaction: Transaction) -> ResourceCandidate:
        timestamp = transaction.transaction_timestamp
        display_date = timestamp.strftime("%B %-d, %Y")
        amount = f"{transaction.amount:.2f}"
        status = transaction.status.casefold()
        return ResourceCandidate(
            resource_id=transaction.transaction_id,
            label=f"{transaction.merchant_name}, ${amount} on {display_date}",
            selectors=(
                transaction.merchant_name.casefold(),
                amount,
                f"${amount}",
                display_date.casefold(),
                timestamp.strftime("%B %-d").casefold(),
                timestamp.date().isoformat(),
                status,
                f"{status} transaction",
            ),
            related_account_id=transaction.account_id,
        )

    @staticmethod
    def _activate_candidate(
        state: ConversationState,
        resource_type: ResourceType,
        intent: str,
        candidate: ResourceCandidate,
    ) -> None:
        state.pending_action = None
        state.clear_resource_resolution()
        state.active_intent = intent

        if resource_type == ResourceType.ACCOUNT:
            state.active_account_id = candidate.resource_id
        elif resource_type == ResourceType.CARD:
            state.active_card_id = candidate.resource_id
            state.active_account_id = candidate.related_account_id
        else:
            state.active_transaction_id = candidate.resource_id
            state.active_account_id = candidate.related_account_id

        if (
            resource_type == ResourceType.CARD
            and intent == "freeze_card"
        ):
            state.request_action(
                "freeze_card",
                candidate.resource_id,
                arguments={"card_id": str(candidate.resource_id)},
            )

    @staticmethod
    def _clear_pending_context(
        state: ConversationState,
        resource_type: ResourceType,
    ) -> None:
        state.clear_resource_resolution()
        state.active_intent = None

        if resource_type == ResourceType.ACCOUNT:
            state.active_account_id = None
        elif resource_type == ResourceType.CARD:
            state.active_card_id = None
        else:
            state.active_transaction_id = None
