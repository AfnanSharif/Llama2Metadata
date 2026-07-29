class ApplicationError(Exception):
    """Base exception for the application."""
    pass

class ConfigurationError(ApplicationError):
    """Raised when there is a configuration issue."""
    pass

class ProcessingError(ApplicationError):
    """Raised when data processing fails."""
    pass
