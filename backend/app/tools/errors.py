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

class ToolSessionError(ToolError):
    """Raised when a tool requires server-side session context."""


class ToolNotFoundError(ToolError):
    """Raised when an unknown tool name is requested."""


class ToolValidationError(ToolError):
    """Raised when model-provided tool arguments fail validation."""


class ToolBackendError(ToolError):
    """Raised when a tool backend fails independently of user input."""


class ToolTimeoutError(ToolBackendError):
    """Raised when tool execution exceeds its configured timeout."""
