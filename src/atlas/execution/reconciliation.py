"""Position reconciliation between Atlas and the broker.

After every execution cycle Atlas compares what it *thinks* it holds with what
the broker *says* it holds. Disagreement means one of them is wrong, and trading
on a wrong position is how small problems become large ones - so an unresolved
material mismatch blocks the next cycle.

Tolerances exist because exact equality is the wrong test:

* **Fractional shares** - brokers round differently; a 0.01-share difference is
  noise.
* **Market value** - a small dollar difference on a large position is not
  actionable.
* **Percentage** - a difference proportional to position size catches the case a
  flat share tolerance misses.

A difference must exceed *all three* tolerances to count as a mismatch. A symbol
present on only one side is always a mismatch regardless of size, because that
indicates a structural problem rather than rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import ReconciliationConfig
from atlas.execution.ibkr_client import BrokerPosition
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["PositionMismatch", "Reconciler", "ReconciliationReport"]


@dataclass(frozen=True)
class PositionMismatch:
    """A disagreement about one symbol."""

    symbol: str
    local_quantity: float
    broker_quantity: float
    price: float
    kind: str  # quantity_difference | missing_locally | missing_at_broker

    @property
    def share_difference(self) -> float:
        """Broker quantity minus local quantity."""
        return self.broker_quantity - self.local_quantity

    @property
    def value_difference(self) -> float:
        """Share difference valued at the current price."""
        price = self.price if np.isfinite(self.price) else 0.0
        return self.share_difference * price

    @property
    def pct_difference(self) -> float:
        """Share difference relative to the larger of the two positions."""
        base = max(abs(self.local_quantity), abs(self.broker_quantity))
        return abs(self.share_difference) / base if base > 1e-9 else float("inf")

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "symbol": self.symbol,
            "kind": self.kind,
            "local_quantity": self.local_quantity,
            "broker_quantity": self.broker_quantity,
            "share_difference": self.share_difference,
            "value_difference": self.value_difference,
            "pct_difference": self.pct_difference,
            "price": self.price,
        }


@dataclass
class ReconciliationReport:
    """The outcome of one reconciliation."""

    mismatches: list[PositionMismatch] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    within_tolerance: list[PositionMismatch] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    blocking: bool = False

    @property
    def reconciled(self) -> bool:
        """True when no material mismatch was found."""
        return not self.mismatches

    @property
    def status(self) -> str:
        """``reconciled``, ``tolerated`` or ``mismatch``."""
        if self.mismatches:
            return "mismatch"
        return "tolerated" if self.within_tolerance else "reconciled"

    @property
    def n_checked(self) -> int:
        """Number of distinct symbols compared."""
        return len(self.matched) + len(self.mismatches) + len(self.within_tolerance)

    @property
    def max_share_difference(self) -> float:
        """Largest absolute share difference among the mismatches."""
        return max((abs(m.share_difference) for m in self.mismatches), default=0.0)

    @property
    def max_value_difference(self) -> float:
        """Largest absolute value difference among the mismatches."""
        return max((abs(m.value_difference) for m in self.mismatches), default=0.0)

    def mismatch_records(self) -> list[dict[str, Any]]:
        """Mismatches as plain dictionaries."""
        return [m.as_dict() for m in self.mismatches]

    def to_frame(self) -> pd.DataFrame:
        """All differences (material and tolerated) as a DataFrame."""
        rows = [m.as_dict() for m in self.mismatches + self.within_tolerance]
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "kind", "local_quantity", "broker_quantity", "share_difference"]
            )
        frame = pd.DataFrame(rows)
        frame["material"] = [True] * len(self.mismatches) + [False] * len(self.within_tolerance)
        return frame

    def render(self) -> str:
        """Human-readable report."""
        lines = [
            "Atlas position reconciliation",
            "=" * 60,
            f"  checked at:  {self.checked_at.isoformat()}",
            f"  status:      {self.status.upper()}",
            f"  symbols:     {self.n_checked}",
            f"  matched:     {len(self.matched)}",
            f"  tolerated:   {len(self.within_tolerance)}",
            f"  mismatches:  {len(self.mismatches)}",
            f"  blocking:    {self.blocking}",
        ]
        if self.mismatches:
            lines.append("-" * 60)
            for mismatch in self.mismatches:
                lines.append(
                    f"  {mismatch.symbol:<8} {mismatch.kind:<20} "
                    f"local {mismatch.local_quantity:>12,.4f}  "
                    f"broker {mismatch.broker_quantity:>12,.4f}  "
                    f"diff {mismatch.share_difference:>+12,.4f} "
                    f"(${mismatch.value_difference:+,.2f})"
                )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs."""
        return {
            "status": self.status,
            "n_checked": self.n_checked,
            "n_mismatches": len(self.mismatches),
            "n_tolerated": len(self.within_tolerance),
            "max_share_difference": round(self.max_share_difference, 6),
            "max_value_difference": round(self.max_value_difference, 2),
            "blocking": self.blocking,
        }


class Reconciler:
    """Compare local and broker positions under configured tolerances."""

    def __init__(self, config: ReconciliationConfig) -> None:
        self.config = config

    def reconcile(
        self,
        local_positions: dict[str, float] | pd.Series,
        broker_positions: list[BrokerPosition] | dict[str, float] | pd.Series,
        *,
        prices: pd.Series | None = None,
    ) -> ReconciliationReport:
        """Compare two position views.

        Parameters
        ----------
        local_positions:
            Atlas's view, as ``{symbol: signed_quantity}``.
        broker_positions:
            The broker's view, as :class:`~atlas.execution.ibkr_client.BrokerPosition`
            objects or a mapping.
        prices:
            Current prices, used to value the differences.

        Returns
        -------
        ReconciliationReport
        """
        local = self._as_mapping(local_positions)
        broker = self._as_mapping(broker_positions)
        price_map = self._price_map(prices, broker_positions)

        symbols = sorted(set(local) | set(broker))
        mismatches: list[PositionMismatch] = []
        tolerated: list[PositionMismatch] = []
        matched: list[str] = []

        for symbol in symbols:
            local_qty = float(local.get(symbol, 0.0))
            broker_qty = float(broker.get(symbol, 0.0))
            price = float(price_map.get(symbol, float("nan")))

            if abs(local_qty - broker_qty) <= 1e-12:
                if abs(local_qty) > 1e-9:
                    matched.append(symbol)
                continue

            if abs(local_qty) <= 1e-9:
                kind = "missing_locally"
            elif abs(broker_qty) <= 1e-9:
                kind = "missing_at_broker"
            else:
                kind = "quantity_difference"

            mismatch = PositionMismatch(
                symbol=symbol,
                local_quantity=local_qty,
                broker_quantity=broker_qty,
                price=price,
                kind=kind,
            )
            # A position on only one side is structural, never rounding.
            if kind == "quantity_difference" and self._within_tolerance(mismatch):
                tolerated.append(mismatch)
            else:
                mismatches.append(mismatch)

        report = ReconciliationReport(
            mismatches=mismatches,
            matched=matched,
            within_tolerance=tolerated,
            blocking=bool(mismatches) and self.config.block_on_mismatch,
        )
        log.info("reconciliation complete", extra={"context": report.summary()})
        for mismatch in mismatches:
            log.error(
                "position mismatch",
                extra={"context": mismatch.as_dict()},
            )
        return report

    # -- helpers --------------------------------------------------------------

    def _within_tolerance(self, mismatch: PositionMismatch) -> bool:
        """True when a difference is small on *every* configured measure."""
        cfg = self.config
        share_ok = abs(mismatch.share_difference) <= cfg.tolerance_shares
        value_ok = (
            abs(mismatch.value_difference) <= cfg.tolerance_value
            if np.isfinite(mismatch.value_difference)
            else False
        )
        pct_ok = mismatch.pct_difference <= cfg.tolerance_pct
        return share_ok and value_ok and pct_ok

    @staticmethod
    def _as_mapping(positions: Any) -> dict[str, float]:
        """Normalise any accepted position representation to a mapping."""
        if isinstance(positions, pd.Series):
            return {str(k).upper(): float(v) for k, v in positions.items() if abs(float(v)) > 1e-12}
        if isinstance(positions, dict):
            return {str(k).upper(): float(v) for k, v in positions.items() if abs(float(v)) > 1e-12}
        if isinstance(positions, list):
            return {
                p.symbol.upper(): float(p.quantity)
                for p in positions
                if abs(float(p.quantity)) > 1e-12
            }
        raise TypeError(f"unsupported position container: {type(positions)!r}")

    @staticmethod
    def _price_map(prices: pd.Series | None, broker_positions: Any) -> dict[str, float]:
        """Build a symbol -> price map from the supplied prices and broker marks."""
        out: dict[str, float] = {}
        if isinstance(broker_positions, list):
            for position in broker_positions:
                mark = position.market_price or position.average_cost
                if mark and np.isfinite(mark):
                    out[position.symbol.upper()] = float(mark)
        if prices is not None:
            for symbol, price in prices.items():
                if price is not None and np.isfinite(price):
                    out[str(symbol).upper()] = float(price)
        return out
