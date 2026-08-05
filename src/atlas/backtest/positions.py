"""Position and portfolio accounting.

:class:`Portfolio` is the single source of truth for cash, holdings and equity
during a backtest or a paper-trading session. It is deliberately explicit about
cash: every fill moves cash, every dividend credits cash, and equity is always
``cash + sum(position market values)``. Nothing is inferred from returns.

Short positions are supported (negative quantity); the cost basis and realised
P&L handle direction changes correctly, including a trade that flips a position
from long to short in one fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.orders import Fill
from atlas.exceptions import AtlasError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["Portfolio", "PortfolioSnapshot", "Position"]


@dataclass
class Position:
    """A holding in one symbol.

    Attributes
    ----------
    symbol:
        Ticker.
    quantity:
        Signed share count; negative for a short.
    average_price:
        Average cost per share of the *current* position, inclusive of the
        price-based execution costs actually paid.
    realized_pnl:
        Cumulative realised profit and loss from closed quantity.
    total_commission:
        Cumulative commission attributed to this symbol.
    """

    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    last_price: float = 0.0
    opened_at: pd.Timestamp | None = None
    last_trade_at: pd.Timestamp | None = None

    @property
    def is_open(self) -> bool:
        """True when the position holds a non-trivial quantity."""
        return abs(self.quantity) > 1e-9

    @property
    def is_long(self) -> bool:
        """True for a long position."""
        return self.quantity > 1e-9

    @property
    def cost_basis(self) -> float:
        """Signed capital committed at the average price."""
        return self.quantity * self.average_price

    def market_value(self, price: float | None = None) -> float:
        """Mark-to-market value at ``price`` (defaults to the last known price)."""
        p = float(price) if price is not None and np.isfinite(price) else self.last_price
        return self.quantity * p

    def unrealized_pnl(self, price: float | None = None) -> float:
        """Unrealised profit and loss at ``price``."""
        p = float(price) if price is not None and np.isfinite(price) else self.last_price
        return (p - self.average_price) * self.quantity

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill and return the realised P&L it generated.

        Handles four cases: opening, adding to an existing position, reducing or
        closing it, and reversing through zero.
        """
        signed = fill.signed_quantity
        price = float(fill.price)
        realized = 0.0

        if not self.is_open:
            # Opening a new position.
            self.quantity = signed
            self.average_price = price
            self.opened_at = fill.timestamp
        elif np.sign(signed) == np.sign(self.quantity):
            # Adding: blend the cost basis.
            total = self.quantity + signed
            self.average_price = (
                self.average_price * self.quantity + price * signed
            ) / total
            self.quantity = total
        else:
            closing = min(abs(signed), abs(self.quantity))
            # Realised P&L is the price difference on the closed quantity,
            # signed by the direction of the position being closed.
            realized = (price - self.average_price) * closing * np.sign(self.quantity)
            remaining = self.quantity + signed
            if abs(remaining) <= 1e-9:
                self.quantity = 0.0
                self.average_price = 0.0
                self.opened_at = None
            elif np.sign(remaining) == np.sign(self.quantity):
                # Partial close: the average price is unchanged.
                self.quantity = remaining
            else:
                # Reversal through zero: the residual opens at the fill price.
                self.quantity = remaining
                self.average_price = price
                self.opened_at = fill.timestamp

        self.realized_pnl += realized
        self.total_commission += fill.commission
        self.last_price = price
        self.last_trade_at = fill.timestamp
        return realized

    def as_dict(self, price: float | None = None) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "last_price": price if price is not None else self.last_price,
            "market_value": self.market_value(price),
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl(price),
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "opened_at": self.opened_at,
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state at one point in time."""

    timestamp: pd.Timestamp
    equity: float
    cash: float
    positions_value: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    weights: dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    cumulative_costs: float = 0.0
    turnover: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation (weights excluded)."""
        return {
            "timestamp": self.timestamp,
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "cash_weight": 1.0 - self.net_exposure,
            "n_positions": self.n_positions,
            "daily_pnl": self.daily_pnl,
            "cumulative_costs": self.cumulative_costs,
            "turnover": self.turnover,
        }


class Portfolio:
    """Cash, positions and equity accounting.

    Parameters
    ----------
    initial_capital:
        Starting cash.
    allow_negative_cash:
        When False (the default for paper trading), a fill that would drive cash
        negative raises. Backtests set this True and rely on the leverage limits
        instead, so that a margin-financed book is representable.
    """

    def __init__(self, initial_capital: float, *, allow_negative_cash: bool = True) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.allow_negative_cash = allow_negative_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.cumulative_costs = 0.0
        self.cumulative_commission = 0.0
        self.dividends_received = 0.0
        self._last_equity = float(initial_capital)
        self._last_prices: dict[str, float] = {}

    # -- state ----------------------------------------------------------------

    def position(self, symbol: str) -> Position:
        """Return (creating if needed) the position for ``symbol``."""
        key = symbol.upper()
        if key not in self.positions:
            self.positions[key] = Position(symbol=key)
        return self.positions[key]

    def quantity(self, symbol: str) -> float:
        """Signed share count held in ``symbol``."""
        pos = self.positions.get(symbol.upper())
        return pos.quantity if pos else 0.0

    def open_positions(self) -> dict[str, Position]:
        """All positions with a non-trivial quantity."""
        return {s: p for s, p in self.positions.items() if p.is_open}

    def positions_value(self, prices: pd.Series | None = None) -> float:
        """Total mark-to-market value of all holdings."""
        lookup = self._price_map(prices)
        total = 0.0
        for symbol, pos in self.positions.items():
            if not pos.is_open:
                continue
            total += pos.quantity * self._resolve(symbol, lookup)
        return float(total)

    def equity(self, prices: pd.Series | None = None) -> float:
        """Total account value: cash plus holdings."""
        return float(self.cash + self.positions_value(prices))

    def weights(self, prices: pd.Series | None = None) -> pd.Series:
        """Position weights as a fraction of equity."""
        total = self.equity(prices)
        if abs(total) < 1e-9:
            return pd.Series(dtype="float64")
        lookup = self._price_map(prices)
        data = {
            symbol: (pos.quantity * self._resolve(symbol, lookup)) / total
            for symbol, pos in self.positions.items()
            if pos.is_open
        }
        return pd.Series(data, dtype="float64").sort_index()

    def gross_exposure(self, prices: pd.Series | None = None) -> float:
        """Sum of absolute position weights."""
        w = self.weights(prices)
        return float(w.abs().sum()) if len(w) else 0.0

    def net_exposure(self, prices: pd.Series | None = None) -> float:
        """Sum of signed position weights."""
        w = self.weights(prices)
        return float(w.sum()) if len(w) else 0.0

    def buying_power(self, prices: pd.Series | None = None, leverage: float = 1.0) -> float:
        """Cash available for new long positions."""
        return float(max(0.0, self.cash + (leverage - 1.0) * self.equity(prices)))

    # -- mutation -------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill: update the position, cash and cost accumulators.

        Returns the realised P&L generated by the fill.

        Raises
        ------
        AtlasError
            If the fill would drive cash negative and that is disallowed.
        """
        prospective_cash = self.cash + fill.cash_flow
        if prospective_cash < -1e-6 and not self.allow_negative_cash:
            raise AtlasError(
                f"fill of {fill.quantity:.4f} {fill.symbol} @ {fill.price:.4f} would drive cash to "
                f"${prospective_cash:,.2f}; increase capital or reduce the target weight"
            )

        realized = self.position(fill.symbol).apply_fill(fill)
        self.cash = prospective_cash
        self.cumulative_costs += fill.total_cost
        self.cumulative_commission += fill.commission
        self.fills.append(fill)
        self._last_prices[fill.symbol.upper()] = fill.price
        return realized

    def credit_dividends(self, dividends: pd.Series, timestamp: pd.Timestamp) -> float:
        """Credit cash dividends on held positions. Returns the amount credited."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if not pos.is_open:
                continue
            per_share = float(dividends.get(symbol, 0.0) or 0.0)
            if per_share > 0:
                total += pos.quantity * per_share
        if abs(total) > 1e-12:
            self.cash += total
            self.dividends_received += total
            log.debug(
                "dividends credited",
                extra={"context": {"date": str(pd.Timestamp(timestamp).date()), "amount": round(total, 2)}},
            )
        return float(total)

    def mark_to_market(self, prices: pd.Series) -> None:
        """Record the latest prices on each position."""
        lookup = self._price_map(prices)
        for symbol, pos in self.positions.items():
            price = lookup.get(symbol)
            if price is not None and np.isfinite(price):
                pos.last_price = float(price)
                self._last_prices[symbol] = float(price)

    def snapshot(
        self, timestamp: pd.Timestamp, prices: pd.Series | None = None, *, turnover: float = 0.0
    ) -> PortfolioSnapshot:
        """Record and return a snapshot of the current state."""
        if prices is not None:
            self.mark_to_market(prices)
        equity = self.equity(prices)
        weights = self.weights(prices)
        snap = PortfolioSnapshot(
            timestamp=pd.Timestamp(timestamp),
            equity=equity,
            cash=self.cash,
            positions_value=self.positions_value(prices),
            gross_exposure=float(weights.abs().sum()) if len(weights) else 0.0,
            net_exposure=float(weights.sum()) if len(weights) else 0.0,
            n_positions=len(weights),
            weights=weights.to_dict(),
            daily_pnl=equity - self._last_equity,
            cumulative_costs=self.cumulative_costs,
            turnover=turnover,
        )
        self._last_equity = equity
        self.snapshots.append(snap)
        return snap

    # -- reporting ------------------------------------------------------------

    def equity_curve(self) -> pd.Series:
        """Equity over time from the recorded snapshots."""
        if not self.snapshots:
            return pd.Series(dtype="float64")
        return pd.Series(
            [s.equity for s in self.snapshots],
            index=pd.DatetimeIndex([s.timestamp for s in self.snapshots], name="date"),
            name="equity",
        )

    def snapshot_frame(self) -> pd.DataFrame:
        """All snapshots as a DataFrame indexed by date."""
        if not self.snapshots:
            return pd.DataFrame()
        frame = pd.DataFrame([s.as_dict() for s in self.snapshots])
        return frame.set_index("timestamp").rename_axis("date")

    def weight_history(self) -> pd.DataFrame:
        """Position weights over time (date x symbol)."""
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame(
            [s.weights for s in self.snapshots],
            index=pd.DatetimeIndex([s.timestamp for s in self.snapshots], name="date"),
        ).fillna(0.0)

    def fills_frame(self) -> pd.DataFrame:
        """All fills as a DataFrame."""
        if not self.fills:
            return pd.DataFrame(
                columns=["fill_id", "order_id", "symbol", "side", "quantity", "price", "timestamp"]
            )
        return pd.DataFrame([f.as_dict() for f in self.fills])

    def positions_frame(self, prices: pd.Series | None = None) -> pd.DataFrame:
        """Current open positions as a DataFrame."""
        lookup = self._price_map(prices)
        rows = [
            pos.as_dict(self._resolve(symbol, lookup))
            for symbol, pos in sorted(self.positions.items())
            if pos.is_open
        ]
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "quantity", "average_price", "last_price", "market_value"]
            )
        return pd.DataFrame(rows).set_index("symbol")

    def realized_pnl(self) -> float:
        """Total realised P&L across all symbols."""
        return float(sum(p.realized_pnl for p in self.positions.values()))

    def summary(self, prices: pd.Series | None = None) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        equity = self.equity(prices)
        return {
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(self.positions_value(prices), 2),
            "n_positions": len(self.open_positions()),
            "gross_exposure": round(self.gross_exposure(prices), 4),
            "net_exposure": round(self.net_exposure(prices), 4),
            "total_return": round(equity / self.initial_capital - 1.0, 4),
            "cumulative_costs": round(self.cumulative_costs, 2),
            "realized_pnl": round(self.realized_pnl(), 2),
            "n_fills": len(self.fills),
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _price_map(prices: pd.Series | None) -> dict[str, float]:
        """Convert a price Series to a dict once.

        Repeated ``Series.get`` calls dominate the backtest profile; converting
        to a dict up front turns an O(n) inner loop into O(1) lookups.
        """
        if prices is None or len(prices) == 0:
            return {}
        return {str(k): float(v) for k, v in prices.items()}

    def _resolve(self, symbol: str, lookup: dict[str, float]) -> float:
        """Best available price for ``symbol``: supplied, else last known."""
        value = lookup.get(symbol)
        if value is not None and np.isfinite(value):
            return value
        position = self.positions.get(symbol)
        return float(self._last_prices.get(symbol.upper(), position.last_price if position else 0.0))

    def _price_for(self, symbol: str, prices: pd.Series | None) -> float:
        """Best available price for ``symbol``: supplied, else last known."""
        return self._resolve(symbol, self._price_map(prices))
