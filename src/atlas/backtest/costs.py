"""Transaction-cost model.

Every simulated trade pays four separable costs, tracked individually so the
report can show where the money went:

.. math::

    \\text{Cost} = \\underbrace{c_{fixed} + c_{share} q}_{\\text{commission}}
                 + \\underbrace{\\tfrac{1}{2} s \\cdot P q}_{\\text{spread}}
                 + \\underbrace{\\kappa \\cdot P q}_{\\text{slippage}}
                 + \\underbrace{\\lambda \\sigma_d \\sqrt{q / V} \\cdot P q}_{\\text{impact}}

Commission
    IBKR-like: a per-share rate with a per-order minimum and a cap expressed as a
    percentage of trade value.
Half spread
    Crossing the bid-ask spread costs half of it in each direction. For liquid
    ETFs 1 bp is a reasonable assumption; it should be raised for anything else.
Slippage
    A flat adverse move representing the difference between the decision price
    and the achievable price.
Market impact
    A square-root law in participation rate - the standard functional form from
    the market-microstructure literature (Almgren et al. 2005). Impact grows with
    the fraction of daily volume consumed, so large trades in thin names are
    penalised super-linearly in notional terms.

All costs are charged as cash, and the fill price is adjusted so that the
position's cost basis reflects the spread, slippage and impact. Commission is
charged separately as a cash outflow, matching how a broker actually bills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from atlas.config import CostConfig

__all__ = ["CostBreakdown", "TransactionCostModel"]


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised cost of a single trade, in account currency."""

    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0

    @property
    def total(self) -> float:
        """Total cost of the trade."""
        return self.commission + self.spread + self.slippage + self.market_impact

    @property
    def price_costs(self) -> float:
        """Costs expressed through the fill price rather than billed separately."""
        return self.spread + self.slippage + self.market_impact

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            commission=self.commission + other.commission,
            spread=self.spread + other.spread,
            slippage=self.slippage + other.slippage,
            market_impact=self.market_impact + other.market_impact,
        )

    def as_dict(self) -> dict[str, float]:
        """Flat representation including the total."""
        return {
            "commission": self.commission,
            "spread": self.spread,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "total": self.total,
        }


class TransactionCostModel:
    """Compute the cost of a simulated trade.

    Parameters
    ----------
    config:
        A :class:`~atlas.config.CostConfig`.
    """

    def __init__(self, config: CostConfig) -> None:
        self.config = config

    # -- components -----------------------------------------------------------

    def commission(self, quantity: float, price: float) -> float:
        """Broker commission for a trade of ``quantity`` shares at ``price``.

        Applies the per-order fee, the per-share rate, the per-order minimum and
        the percentage-of-value cap, in that order.
        """
        cfg = self.config
        shares = abs(float(quantity))
        if shares <= 0:
            return 0.0
        raw = cfg.commission_per_order + cfg.commission_per_share * shares
        charged = max(raw, cfg.min_commission)
        notional = shares * abs(float(price))
        if cfg.max_commission_pct > 0:
            charged = min(charged, cfg.max_commission_pct * notional)
        # The cap must never push the charge below zero.
        return float(max(charged, 0.0))

    def spread_cost(self, quantity: float, price: float) -> float:
        """Half-spread cost of crossing the quoted market."""
        notional = abs(float(quantity) * float(price))
        return notional * self.config.half_spread_bps / 10_000.0

    def slippage_cost(self, quantity: float, price: float) -> float:
        """Flat adverse price movement between decision and execution."""
        notional = abs(float(quantity) * float(price))
        return notional * self.config.slippage_bps / 10_000.0

    def market_impact_cost(
        self,
        quantity: float,
        price: float,
        *,
        adv_shares: float | None = None,
        daily_volatility: float | None = None,
    ) -> float:
        """Square-root market impact.

        Parameters
        ----------
        adv_shares:
            Average daily volume in shares. When unknown, impact is zero -
            Atlas prefers a stated assumption over an invented one, and the
            validation report flags symbols without volume data.
        daily_volatility:
            Daily return volatility of the asset; defaults to 1.5% when unknown.
        """
        cfg = self.config
        if not cfg.market_impact_enabled or not adv_shares or adv_shares <= 0:
            return 0.0
        shares = abs(float(quantity))
        if shares <= 0:
            return 0.0
        participation = min(shares / float(adv_shares), 1.0)
        sigma = float(daily_volatility) if daily_volatility and daily_volatility > 0 else 0.015
        notional = shares * abs(float(price))
        return float(cfg.market_impact_coefficient * sigma * np.sqrt(participation) * notional)

    # -- aggregation ----------------------------------------------------------

    def compute(
        self,
        quantity: float,
        price: float,
        *,
        adv_shares: float | None = None,
        daily_volatility: float | None = None,
    ) -> CostBreakdown:
        """Return the full :class:`CostBreakdown` for one trade."""
        if quantity == 0 or not np.isfinite(price) or price <= 0:
            return CostBreakdown()
        return CostBreakdown(
            commission=self.commission(quantity, price),
            spread=self.spread_cost(quantity, price),
            slippage=self.slippage_cost(quantity, price),
            market_impact=self.market_impact_cost(
                quantity, price, adv_shares=adv_shares, daily_volatility=daily_volatility
            ),
        )

    def effective_price(
        self,
        quantity: float,
        price: float,
        *,
        adv_shares: float | None = None,
        daily_volatility: float | None = None,
    ) -> tuple[float, CostBreakdown]:
        """Return ``(fill_price, costs)`` for a trade.

        The fill price moves *against* the trader: a buy fills above the
        reference price, a sell below it. Commission is excluded from the price
        because brokers bill it separately.
        """
        costs = self.compute(
            quantity, price, adv_shares=adv_shares, daily_volatility=daily_volatility
        )
        shares = abs(float(quantity))
        if shares <= 0:
            return float(price), costs
        adverse_per_share = costs.price_costs / shares
        direction = 1.0 if quantity > 0 else -1.0
        fill = float(price) + direction * adverse_per_share
        # A pathological cost model must never produce a non-positive price.
        return float(max(fill, 1e-6)), costs

    def max_fillable_shares(self, adv_shares: float | None) -> float:
        """Largest order size that fills completely on one bar.

        Orders above ``max_participation * ADV`` are partially filled; the
        remainder is not carried over, which is the conservative assumption for
        a daily-frequency system.
        """
        if not adv_shares or adv_shares <= 0:
            return float("inf")
        return float(self.config.max_participation * adv_shares)

    def describe(self) -> dict[str, Any]:
        """Human-readable summary of the cost assumptions, for reports."""
        cfg = self.config
        return {
            "commission_per_order": f"${cfg.commission_per_order:.2f}",
            "commission_per_share": f"${cfg.commission_per_share:.4f}",
            "min_commission": f"${cfg.min_commission:.2f}",
            "max_commission_pct": f"{cfg.max_commission_pct:.2%} of trade value",
            "half_spread": f"{cfg.half_spread_bps:.1f} bps",
            "slippage": f"{cfg.slippage_bps:.1f} bps",
            "market_impact": (
                f"sqrt-law, coefficient {cfg.market_impact_coefficient:.2f}"
                if cfg.market_impact_enabled
                else "disabled"
            ),
            "max_participation": f"{cfg.max_participation:.1%} of ADV",
        }
