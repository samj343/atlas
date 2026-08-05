"""Interactive Brokers paper-trading integration.

Live trading is not implemented. ``EXECUTION_MODE`` supports ``backtest``,
``dry_run`` and ``paper`` only; ``live`` exists as a disabled placeholder that is
rejected both at configuration load and by the safety gate.
"""

from atlas.execution.ibkr_client import (
    AccountSummary,
    BrokerClient,
    BrokerPosition,
    IBKRClient,
    MockBrokerClient,
)
from atlas.execution.order_manager import ExecutionCycleResult, OrderManager, TradePlan
from atlas.execution.reconciliation import (
    PositionMismatch,
    Reconciler,
    ReconciliationReport,
)
from atlas.execution.safety import SafetyCheck, SafetyGate, SafetyReport

__all__ = [
    "AccountSummary",
    "BrokerClient",
    "BrokerPosition",
    "ExecutionCycleResult",
    "IBKRClient",
    "MockBrokerClient",
    "OrderManager",
    "PositionMismatch",
    "Reconciler",
    "ReconciliationReport",
    "SafetyCheck",
    "SafetyGate",
    "SafetyReport",
    "TradePlan",
]
