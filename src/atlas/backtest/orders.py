"""Orders, fills and the events that drive the backtest engine.

These types are shared by the simulated broker and the live (paper) execution
path, so an order created by the research stack is structurally identical to one
sent to Interactive Brokers. That symmetry is what makes it possible to test the
execution logic without a broker connection.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd

__all__ = [
    "Event",
    "EventType",
    "Fill",
    "FillEvent",
    "MarketEvent",
    "Order",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PortfolioEvent",
    "RebalanceEvent",
    "RiskEventRecord",
    "SignalEvent",
    "TimeInForce",
    "new_order_id",
]


def new_order_id(prefix: str = "ATL") -> str:
    """Generate a unique, human-readable order identifier.

    Uniqueness matters for duplicate suppression and for reconciling local
    orders against broker records, so a UUID4 suffix is used rather than a
    counter that would reset between processes.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "BUY"
    SELL = "SELL"

    @classmethod
    def from_quantity(cls, quantity: float) -> OrderSide:
        """Infer the side from a signed quantity."""
        return cls.BUY if quantity >= 0 else cls.SELL


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "MKT"
    LIMIT = "LMT"


class TimeInForce(str, Enum):
    """Order lifetime."""

    DAY = "DAY"
    GTC = "GTC"


class OrderStatus(str, Enum):
    """Order lifecycle states."""

    CREATED = "created"
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """An instruction to trade a quantity of one symbol.

    ``quantity`` is always positive; direction is carried by ``side``. This
    matches broker APIs and removes a whole class of sign errors.
    """

    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    order_id: str = field(default_factory=new_order_id)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    reject_reason: str = ""
    broker_order_id: str | None = None
    reference_price: float | None = None
    target_weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).upper()
        if self.quantity < 0:
            raise ValueError("Order.quantity must be non-negative; use `side` for direction")

    @classmethod
    def from_signed_quantity(cls, symbol: str, quantity: float, **kwargs: Any) -> Order:
        """Build an order from a signed quantity."""
        return cls(
            symbol=symbol,
            quantity=abs(float(quantity)),
            side=OrderSide.from_quantity(quantity),
            **kwargs,
        )

    @property
    def signed_quantity(self) -> float:
        """Quantity with the sign implied by the side."""
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def remaining_quantity(self) -> float:
        """Quantity still to be filled."""
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        """True when the order can no longer change state."""
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

    @property
    def is_open(self) -> bool:
        """True when the order is live at the broker."""
        return self.status in {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }

    @property
    def notional(self) -> float:
        """Estimated notional value using the reference price, when known."""
        price = self.reference_price or self.limit_price or 0.0
        return abs(self.quantity * price)

    def apply_fill(self, fill: Fill) -> None:
        """Update the order state from a fill."""
        if fill.quantity <= 0:
            return
        total = self.filled_quantity + fill.quantity
        # Volume-weighted average across partial fills.
        self.average_fill_price = (
            (self.average_fill_price * self.filled_quantity + fill.price * fill.quantity) / total
        )
        self.filled_quantity = total
        self.status = (
            OrderStatus.FILLED
            if self.remaining_quantity <= 1e-9
            else OrderStatus.PARTIALLY_FILLED
        )

    def reject(self, reason: str) -> None:
        """Mark the order rejected with a reason."""
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason

    def cancel(self, reason: str = "") -> None:
        """Mark the order cancelled."""
        self.status = OrderStatus.CANCELLED
        if reason:
            self.metadata["cancel_reason"] = reason

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "reject_reason": self.reject_reason,
            "created_at": self.created_at,
            "reference_price": self.reference_price,
            "target_weight": self.target_weight,
        }


@dataclass
class Fill:
    """An execution against an order."""

    order_id: str
    symbol: str
    quantity: float           # always positive
    price: float              # the effective fill price, inclusive of price-based costs
    side: OrderSide
    timestamp: pd.Timestamp
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    impact_cost: float = 0.0
    fill_id: str = field(default_factory=lambda: f"FILL-{uuid.uuid4().hex[:12].upper()}")
    reference_price: float | None = None

    @property
    def signed_quantity(self) -> float:
        """Quantity with the sign implied by the side."""
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def notional(self) -> float:
        """Absolute traded value at the fill price."""
        return abs(self.quantity * self.price)

    @property
    def cash_flow(self) -> float:
        """Signed change in cash, including commission.

        Buying consumes cash (negative), selling releases it (positive); the
        commission is always an outflow.
        """
        return -self.signed_quantity * self.price - self.commission

    @property
    def total_cost(self) -> float:
        """All costs charged on this fill."""
        return self.commission + self.spread_cost + self.slippage_cost + self.impact_cost

    @property
    def implementation_shortfall(self) -> float:
        """Cost of executing away from the reference price, in currency."""
        if self.reference_price is None:
            return self.spread_cost + self.slippage_cost + self.impact_cost
        return abs(self.price - self.reference_price) * self.quantity

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "reference_price": self.reference_price,
            "timestamp": self.timestamp,
            "commission": self.commission,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "impact_cost": self.impact_cost,
            "total_cost": self.total_cost,
            "notional": self.notional,
            "cash_flow": self.cash_flow,
        }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """The event types the engine processes, in causal order."""

    MARKET = "market"
    SIGNAL = "signal"
    REBALANCE = "rebalance"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    PORTFOLIO = "portfolio"


@dataclass
class Event:
    """Base event carried on the engine's queue."""

    type: EventType
    timestamp: pd.Timestamp

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {"type": self.type.value, "timestamp": self.timestamp}


@dataclass
class MarketEvent(Event):
    """New market data has been revealed for a bar."""

    symbols: list[str] = field(default_factory=list)
    is_rebalance_date: bool = False

    def __init__(
        self, timestamp: pd.Timestamp, symbols: list[str], *, is_rebalance_date: bool = False
    ) -> None:
        super().__init__(EventType.MARKET, timestamp)
        self.symbols = symbols
        self.is_rebalance_date = is_rebalance_date


@dataclass
class SignalEvent(Event):
    """Strategy signals have been produced for a date."""

    signals: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    regime: str | None = None

    def __init__(
        self, timestamp: pd.Timestamp, signals: pd.Series, *, regime: str | None = None
    ) -> None:
        super().__init__(EventType.SIGNAL, timestamp)
        self.signals = signals
        self.regime = regime


@dataclass
class RebalanceEvent(Event):
    """Target weights have been computed."""

    target_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    risk_multiplier: float = 1.0

    def __init__(
        self, timestamp: pd.Timestamp, target_weights: pd.Series, *, risk_multiplier: float = 1.0
    ) -> None:
        super().__init__(EventType.REBALANCE, timestamp)
        self.target_weights = target_weights
        self.risk_multiplier = risk_multiplier


@dataclass
class OrderEvent(Event):
    """An order has been created."""

    order: Order | None = None

    def __init__(self, timestamp: pd.Timestamp, order: Order) -> None:
        super().__init__(EventType.ORDER, timestamp)
        self.order = order


@dataclass
class FillEvent(Event):
    """An order has been (partially) executed."""

    fill: Fill | None = None

    def __init__(self, timestamp: pd.Timestamp, fill: Fill) -> None:
        super().__init__(EventType.FILL, timestamp)
        self.fill = fill


@dataclass
class RiskEventRecord(Event):
    """A risk control fired."""

    control: str = ""
    action: str = ""
    message: str = ""

    def __init__(self, timestamp: pd.Timestamp, control: str, action: str, message: str) -> None:
        super().__init__(EventType.RISK, timestamp)
        self.control = control
        self.action = action
        self.message = message


@dataclass
class PortfolioEvent(Event):
    """The portfolio has been marked to market."""

    equity: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0

    def __init__(
        self, timestamp: pd.Timestamp, equity: float, cash: float, positions_value: float
    ) -> None:
        super().__init__(EventType.PORTFOLIO, timestamp)
        self.equity = equity
        self.cash = cash
        self.positions_value = positions_value
