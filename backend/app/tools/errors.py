class ToolError(Exception):
    """Base exception for deterministic tool failures."""


class ToolAuthenticationError(ToolError):
    """Raised when a private tool is called without authentication."""


class ToolResourceNotFoundError(ToolError):
    """Raised when the requested customer-owned resource cannot be found."""
