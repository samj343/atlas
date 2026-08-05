"""Event-driven backtesting: engine, simulated broker, costs and metrics."""

from atlas.backtest.broker import ExecutionContext, SimulatedBroker
from atlas.backtest.costs import CostBreakdown, TransactionCostModel
from atlas.backtest.engine import BacktestEngine, BacktestResult
from atlas.backtest.metrics import PerformanceMetrics, compute_metrics
from atlas.backtest.orders import Fill, Order, OrderSide, OrderStatus, OrderType
from atlas.backtest.positions import Portfolio, PortfolioSnapshot, Position

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CostBreakdown",
    "ExecutionContext",
    "Fill",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PerformanceMetrics",
    "Portfolio",
    "PortfolioSnapshot",
    "Position",
    "SimulatedBroker",
    "TransactionCostModel",
    "compute_metrics",
]
