from dataclasses import dataclass
from enum import StrEnum


class PermissionLevel(StrEnum):
    PUBLIC_INFORMATIONAL = "PUBLIC_INFORMATIONAL"
    PRIVATE_READ = "PRIVATE_READ"
    PROTECTED_WRITE = "PROTECTED_WRITE"


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


GET_ACCOUNT_BALANCE = ToolDefinition(
    name="get_account_balance",
    permission_level=PermissionLevel.PRIVATE_READ,
    requires_authentication=True,
    requires_confirmation=False,
    timeout_seconds=2.0,
    idempotent=True,
    audit_event="banking.account_balance.read",
    error_types=(
        "authentication_required",
        "resource_not_found",
        "database_error",
        "timeout",
    ),
)
