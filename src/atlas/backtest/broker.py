"""Simulated broker for backtesting.

:class:`SimulatedBroker` turns orders into fills using the price data available
on the execution bar and the configured :class:`~atlas.backtest.costs.TransactionCostModel`.
It models the failure modes that matter for a daily-frequency system:

* **rejection** when no price is available on the execution bar;
* **partial fills** when the order would consume more than ``max_participation``
  of that day's volume;
* **whole-share rounding** when fractional shares are disabled;
* **costs** applied through the fill price and as separately billed commission.

The broker is deterministic: given the same orders and the same data it produces
the same fills, which is what makes backtests reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.costs import TransactionCostModel
from atlas.backtest.orders import Fill, Order, OrderSide, OrderStatus
from atlas.config import ExecutionTiming, FillConfig
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ExecutionContext", "SimulatedBroker"]


@dataclass(frozen=True)
class ExecutionContext:
    """Everything the broker needs to price a fill on one bar.

    Attributes
    ----------
    timestamp:
        The execution bar's date.
    prices:
        Execution reference price per symbol (the open or close of the bar,
        depending on the configured timing).
    volumes:
        Average daily volume per symbol, used for participation limits and
        market impact.
    volatilities:
        Daily return volatility per symbol, used for market impact.
    """

    timestamp: pd.Timestamp
    prices: pd.Series
    volumes: pd.Series | None = None
    volatilities: pd.Series | None = None

    def price(self, symbol: str) -> float | None:
        """Execution price for ``symbol``, or None when unavailable."""
        if symbol not in self.prices.index:
            return None
        value = self.prices.get(symbol)
        if value is None or not np.isfinite(value) or value <= 0:
            return None
        return float(value)

    def volume(self, symbol: str) -> float | None:
        """Average daily volume for ``symbol``, or None."""
        if self.volumes is None or symbol not in self.volumes.index:
            return None
        value = self.volumes.get(symbol)
        if value is None or not np.isfinite(value) or value <= 0:
            return None
        return float(value)

    def volatility(self, symbol: str) -> float | None:
        """Daily volatility for ``symbol``, or None."""
        if self.volatilities is None or symbol not in self.volatilities.index:
            return None
        value = self.volatilities.get(symbol)
        if value is None or not np.isfinite(value) or value <= 0:
            return None
        return float(value)


class SimulatedBroker:
    """Execute orders against historical bars.

    Parameters
    ----------
    cost_model:
        The transaction-cost model.
    fill_config:
        Fill behaviour (partial fills, rejection rules).
    fractional_shares:
        When False, order quantities are rounded down to whole shares.
    timing:
        Execution timing, recorded on each fill for auditability.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        fill_config: FillConfig,
        *,
        fractional_shares: bool = True,
        timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN,
    ) -> None:
        self.cost_model = cost_model
        self.fill_config = fill_config
        self.fractional_shares = fractional_shares
        self.timing = timing
        self.rejected_orders: list[Order] = []

    # -- execution ------------------------------------------------------------

    def execute(self, order: Order, context: ExecutionContext) -> Fill | None:
        """Attempt to execute ``order`` on the bar described by ``context``.

        Returns the :class:`~atlas.backtest.orders.Fill`, or ``None`` when the
        order could not be filled at all (it is then marked rejected).
        """
        symbol = order.symbol
        price = context.price(symbol)

        if price is None:
            if self.fill_config.reject_if_no_price:
                order.reject(f"no execution price available for {symbol} on {context.timestamp.date()}")
                self.rejected_orders.append(order)
                log.warning(
                    "order rejected: no price",
                    extra={"context": {"symbol": symbol, "date": str(context.timestamp.date())}},
                )
                return None
            return None

        quantity = self._round_quantity(order.remaining_quantity)
        if quantity <= 0:
            order.reject(
                "order quantity rounds to zero shares (whole-share mode); "
                "increase capital or the minimum trade size"
            )
            self.rejected_orders.append(order)
            return None

        # --- limit-price check ---
        if order.limit_price is not None:
            crosses = (
                price <= order.limit_price
                if order.side is OrderSide.BUY
                else price >= order.limit_price
            )
            if not crosses:
                order.status = OrderStatus.CANCELLED
                order.metadata["cancel_reason"] = (
                    f"limit {order.limit_price:.4f} not reached (bar price {price:.4f})"
                )
                return None

        # --- participation limit -> partial fill ---
        adv = context.volume(symbol)
        max_shares = self.cost_model.max_fillable_shares(adv)
        if self.fill_config.allow_partial_fills and quantity > max_shares:
            log.info(
                "partial fill: order exceeds the participation limit",
                extra={
                    "context": {
                        "symbol": symbol,
                        "requested": round(quantity, 2),
                        "fillable": round(max_shares, 2),
                    }
                },
            )
            quantity = self._round_quantity(max_shares)
            if quantity <= 0:
                order.reject(f"participation limit permits no shares of {symbol} on this bar")
                self.rejected_orders.append(order)
                return None

        signed = quantity if order.side is OrderSide.BUY else -quantity
        fill_price, costs = self.cost_model.effective_price(
            signed,
            price,
            adv_shares=adv,
            daily_volatility=context.volatility(symbol),
        )

        fill = Fill(
            order_id=order.order_id,
            symbol=symbol,
            quantity=quantity,
            price=fill_price,
            side=order.side,
            timestamp=context.timestamp,
            commission=costs.commission,
            spread_cost=costs.spread,
            slippage_cost=costs.slippage,
            impact_cost=costs.market_impact,
            reference_price=price,
        )
        order.status = OrderStatus.SUBMITTED
        order.apply_fill(fill)
        return fill

    def execute_batch(
        self, orders: list[Order], context: ExecutionContext
    ) -> tuple[list[Fill], list[Order]]:
        """Execute a batch of orders. Returns ``(fills, rejected_orders)``.

        Sells are executed before buys so the cash released by a sale is
        available to fund the purchases on the same bar - matching how a real
        rebalance is sequenced.
        """
        ordered = sorted(orders, key=lambda o: 0 if o.side is OrderSide.SELL else 1)
        fills: list[Fill] = []
        rejected: list[Order] = []
        for order in ordered:
            fill = self.execute(order, context)
            if fill is not None:
                fills.append(fill)
            elif order.status is OrderStatus.REJECTED:
                rejected.append(order)
        return fills, rejected

    # -- helpers --------------------------------------------------------------

    def _round_quantity(self, quantity: float) -> float:
        """Round an order quantity per the share-granularity setting."""
        q = abs(float(quantity))
        if self.fractional_shares:
            # Round to 4 decimals: brokers that support fractional shares still
            # quantise, and this keeps the arithmetic stable.
            return float(np.floor(q * 10_000) / 10_000)
        return float(np.floor(q))

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        return {
            "timing": self.timing.value,
            "fractional_shares": self.fractional_shares,
            "allow_partial_fills": self.fill_config.allow_partial_fills,
            "n_rejected": len(self.rejected_orders),
        }

    def rejected_frame(self) -> pd.DataFrame:
        """Rejected orders as a DataFrame."""
        if not self.rejected_orders:
            return pd.DataFrame(columns=["order_id", "symbol", "side", "quantity", "reject_reason"])
        return pd.DataFrame([o.as_dict() for o in self.rejected_orders])
