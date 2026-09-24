from dataclasses import dataclass
from enum import StrEnum


class PermissionLevel(StrEnum):
    PUBLIC_INFORMATIONAL = "PUBLIC_INFORMATIONAL"
    PRIVATE_READ = "PRIVATE_READ"
    PROTECTED_WRITE = "PROTECTED_WRITE"
    SAFETY_ESCALATION = "SAFETY_ESCALATION"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    permission_level: PermissionLevel
    requires_authentication: bool
    requires_confirmation: bool
    timeout_seconds: float
    idempotent: bool
    audit_event: str
    error_types: tuple[str, ...]
    description: str = ""
    confirmation_resource_field: str | None = None


_PRIVATE_READ_ERRORS = (
    "authentication_required",
    "resource_not_found",
    "database_error",
    "timeout",
)

_PROTECTED_WRITE_ERRORS = (
    "authentication_required",
    "confirmation_required",
    "resource_not_found",
    "invalid_state",
    "database_error",
    "timeout",
)

_ESCALATION_ERRORS = (
    "session_required",
    "database_error",
    "timeout",
)


GET_ACCOUNT_BALANCE = ToolDefinition(
    name="get_account_balance",
    permission_level=PermissionLevel.PRIVATE_READ,
    requires_authentication=True,
    requires_confirmation=False,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.account_balance.read",
    error_types=_PRIVATE_READ_ERRORS,
    description=(
        "Get the current and available balance for one account owned by "
        "the authenticated customer."
    ),
)

GET_RECENT_TRANSACTIONS = ToolDefinition(
    name="get_recent_transactions",
    permission_level=PermissionLevel.PRIVATE_READ,
    requires_authentication=True,
    requires_confirmation=False,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.transactions.read",
    error_types=_PRIVATE_READ_ERRORS,
    description=(
        "Get recent transactions for one account owned by the "
        "authenticated customer."
    ),
)

GET_TRANSACTION_DETAILS = ToolDefinition(
    name="get_transaction_details",
    permission_level=PermissionLevel.PRIVATE_READ,
    requires_authentication=True,
    requires_confirmation=False,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.transaction_details.read",
    error_types=_PRIVATE_READ_ERRORS,
    description=(
        "Get detailed information about one transaction belonging to "
        "the authenticated customer."
    ),
)

GET_CARD_STATUS = ToolDefinition(
    name="get_card_status",
    permission_level=PermissionLevel.PRIVATE_READ,
    requires_authentication=True,
    requires_confirmation=False,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.card_status.read",
    error_types=_PRIVATE_READ_ERRORS,
    description=(
        "Get status and masked details for one card owned by the "
        "authenticated customer."
    ),
)

FREEZE_CARD = ToolDefinition(
    name="freeze_card",
    permission_level=PermissionLevel.PROTECTED_WRITE,
    requires_authentication=True,
    requires_confirmation=True,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.card.freeze",
    error_types=_PROTECTED_WRITE_ERRORS,
    description=(
        "Freeze a specific card owned by the authenticated customer. "
        "This is a protected action requiring explicit confirmation."
    ),
    confirmation_resource_field="card_id",
)

CREATE_DISPUTE = ToolDefinition(
    name="create_dispute",
    permission_level=PermissionLevel.PROTECTED_WRITE,
    requires_authentication=True,
    requires_confirmation=True,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.dispute.create",
    error_types=_PROTECTED_WRITE_ERRORS,
    description=(
        "Create a synthetic dispute for a specific transaction owned by "
        "the authenticated customer. This is a protected action requiring "
        "explicit confirmation."
    ),
    confirmation_resource_field="transaction_id",
)

ESCALATE_TO_HUMAN = ToolDefinition(
    name="escalate_to_human",
    permission_level=PermissionLevel.SAFETY_ESCALATION,
    requires_authentication=False,
    requires_confirmation=False,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="support.escalation.create",
    error_types=_ESCALATION_ERRORS,
    description=(
        "Create a human-support escalation with a structured handoff. "
        "This safety path remains available even before authentication."
    ),
)
