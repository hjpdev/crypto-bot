class CryptoBotError(Exception):
    """Base class for all crypto bot exceptions."""

    pass


class ConfigError(CryptoBotError):
    """Raised when there's an error in the configuration."""

    pass


class APIError(CryptoBotError):
    """Raised when there's an error with API calls."""

    pass


class ExchangeError(CryptoBotError):
    """Raised when there's an error with the exchange."""

    pass


class DatabaseError(CryptoBotError):
    """Raised when there's a database-related error."""

    pass


class StrategyError(CryptoBotError):
    """Raised when there's an error with a trading strategy."""

    pass


class DataError(CryptoBotError):
    """Raised when there's an error with market data."""

    pass


class ValidationError(CryptoBotError):
    """Raised when validation fails."""

    pass


class AuthenticationError(CryptoBotError):
    """Raised when authentication fails."""

    pass


class InsufficientFundsError(CryptoBotError):
    """Raised when there are insufficient funds to execute a trade."""

    pass


class OrderError(CryptoBotError):
    """Raised when there's an error with an order."""

    pass
