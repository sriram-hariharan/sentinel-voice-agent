class ToolError(Exception):
    """Base exception for deterministic tool failures."""


class ToolAuthenticationError(ToolError):
    """Raised when a private tool is called without authentication."""


class ToolConfirmationError(ToolError):
    """Raised when a protected tool lacks matching explicit confirmation."""


class ToolResourceNotFoundError(ToolError):
    """Raised when the requested customer-owned resource cannot be found."""


class ToolInvalidStateError(ToolError):
    """Raised when a resource cannot perform the requested state transition."""
