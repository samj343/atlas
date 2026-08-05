"""Atlas exception hierarchy.

Every failure mode raises a specific, catchable exception rather than a bare
``Exception`` or a silently-swallowed error. Callers can catch
:class:`AtlasError` to handle anything Atlas raises deliberately.
"""

from __future__ import annotations

__all__ = [
    "AtlasError",
    "BrokerError",
    "ConfigurationError",
    "DataError",
    "DataValidationError",
    "ExecutionBlocked",
    "InsufficientDataError",
    "KillSwitchActive",
    "OrderRejected",
    "PortfolioConstructionError",
    "ReconciliationError",
    "RiskLimitBreached",
    "StaleDataError",
    "StrategyError",
]


class AtlasError(Exception):
    """Base class for every error raised deliberately by Atlas."""


class ConfigurationError(AtlasError):
    """Raised when configuration is missing, malformed or internally inconsistent."""


class DataError(AtlasError):
    """Base class for data-layer problems."""


class DataValidationError(DataError):
    """Raised when market data fails validation and cannot be used safely."""


class InsufficientDataError(DataError):
    """Raised when there is not enough history to compute a required quantity."""


class StaleDataError(DataError):
    """Raised when the most recent observation is older than the configured limit."""


class StrategyError(AtlasError):
    """Raised when a strategy cannot produce valid signals."""


class PortfolioConstructionError(AtlasError):
    """Raised when target weights cannot be constructed."""


class RiskLimitBreached(AtlasError):
    """Raised when a hard risk limit blocks an action.

    Parameters
    ----------
    message:
        Human-readable description.
    limit_name:
        Machine-readable identifier of the breached limit.
    observed, limit:
        The observed value and the configured limit, when applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        limit_name: str = "",
        observed: float | None = None,
        limit: float | None = None,
    ) -> None:
        super().__init__(message)
        self.limit_name = limit_name
        self.observed = observed
        self.limit = limit


class KillSwitchActive(AtlasError):
    """Raised when the global kill switch blocks all trading activity."""


class ExecutionBlocked(AtlasError):
    """Raised when a pre-trade safety check prevents an execution cycle."""


class BrokerError(AtlasError):
    """Raised on broker connectivity or API failures."""


class OrderRejected(BrokerError):
    """Raised when an order is rejected, locally or by the broker."""

    def __init__(self, message: str, *, order_id: str | None = None, reason: str = "") -> None:
        super().__init__(message)
        self.order_id = order_id
        self.reason = reason


class ReconciliationError(AtlasError):
    """Raised when broker and local positions cannot be reconciled."""
