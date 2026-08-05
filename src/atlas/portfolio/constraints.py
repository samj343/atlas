"""Portfolio constraint application.

Constraints are applied by *projection*, not by optimisation: each limit is
enforced in a defined order, and every adjustment is recorded with a reason.
This is deliberately simple and auditable - when a weight ends up where it is,
the reason is in the log rather than inside a solver.

Order of application, and why
-----------------------------
1. **Long-only** - drop shorts first, since every subsequent normalisation
   depends on the sign structure.
2. **Per-asset cap** - cap and redistribute the excess proportionally to the
   uncapped names. Repeated until the caps hold or the redistribution converges.
3. **Asset-class cap** - scale down every member of an over-weight class.
4. **Position count** - keep the largest ``max_active_positions`` names.
5. **Gross / net / leverage caps** - scale the whole book down.
6. **Cash reserve** - scale the book so the required cash buffer is free.
7. **Turnover cap** - blend toward the previous weights so the day's trading
   stays within budget.
8. **Small-trade threshold** - revert trades too small to be worth the costs.

Steps 5-7 are pure scalings of an already feasible vector, so they cannot
reintroduce a violation of steps 1-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import PortfolioConfig
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ConstraintAdjustment", "ConstraintResult", "PortfolioConstraints"]


@dataclass(frozen=True)
class ConstraintAdjustment:
    """A single recorded constraint action."""

    constraint: str
    reason: str
    before: float
    after: float
    symbols: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "constraint": self.constraint,
            "reason": self.reason,
            "before": round(float(self.before), 6),
            "after": round(float(self.after), 6),
            "symbols": ",".join(self.symbols),
        }


@dataclass
class ConstraintResult:
    """Constrained weights plus the audit trail that produced them."""

    weights: pd.Series
    adjustments: list[ConstraintAdjustment] = field(default_factory=list)

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute weights."""
        return float(self.weights.abs().sum())

    @property
    def net_exposure(self) -> float:
        """Sum of signed weights."""
        return float(self.weights.sum())

    @property
    def cash_weight(self) -> float:
        """Weight left in cash (may be negative when leveraged)."""
        return 1.0 - self.net_exposure

    @property
    def n_positions(self) -> int:
        """Number of non-zero positions."""
        return int((self.weights.abs() > 1e-9).sum())

    def was_adjusted(self) -> bool:
        """True when any constraint changed the weights."""
        return bool(self.adjustments)

    def to_frame(self) -> pd.DataFrame:
        """Audit trail as a DataFrame."""
        if not self.adjustments:
            return pd.DataFrame(columns=["constraint", "reason", "before", "after", "symbols"])
        return pd.DataFrame([a.as_dict() for a in self.adjustments])

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        return {
            "gross_exposure": round(self.gross_exposure, 4),
            "net_exposure": round(self.net_exposure, 4),
            "cash_weight": round(self.cash_weight, 4),
            "n_positions": self.n_positions,
            "n_adjustments": len(self.adjustments),
        }


class PortfolioConstraints:
    """Apply position, exposure and turnover limits to a weight vector.

    Parameters
    ----------
    config:
        A :class:`~atlas.config.PortfolioConfig`.
    asset_classes:
        Mapping of symbol -> asset class, used for the class-level cap.
    """

    def __init__(self, config: PortfolioConfig, asset_classes: dict[str, str] | None = None) -> None:
        self.config = config
        self.asset_classes = dict(asset_classes or {})
        # Grouping symbols by class is pure configuration, so it is cached per
        # symbol tuple rather than recomputed on every rebalance.
        self._class_cache: dict[tuple[str, ...], dict[str, list[str]]] = {}
        self._position_cache: dict[tuple[str, ...], dict[str, np.ndarray]] = {}

    def _class_groups(self, symbols: pd.Index) -> dict[str, list[str]]:
        """Return ``{asset_class: [symbol, ...]}`` for ``symbols``, memoised."""
        key = tuple(str(s) for s in symbols)
        cached = self._class_cache.get(key)
        if cached is None:
            cached = {}
            for symbol in key:
                cached.setdefault(self.asset_classes.get(symbol, "unclassified"), []).append(symbol)
            self._class_cache[key] = cached
        return cached

    def _class_positions(self, symbols: pd.Index) -> dict[str, np.ndarray]:
        """Return ``{asset_class: positional indices}``, memoised.

        Label-based ``.loc[list]`` indexing costs about a millisecond per call on
        an object-dtype index; a backtest performs tens of thousands of them.
        Working with positional arrays keeps the constraint layer in NumPy.
        """
        key = tuple(str(s) for s in symbols)
        cached = self._position_cache.get(key)
        if cached is None:
            lookup = {s: i for i, s in enumerate(key)}
            cached = {
                name: np.array([lookup[s] for s in members], dtype=int)
                for name, members in self._class_groups(symbols).items()
            }
            self._position_cache[key] = cached
        return cached

    # -- main entry point -----------------------------------------------------

    def apply(
        self,
        weights: pd.Series,
        *,
        previous_weights: pd.Series | None = None,
        enforce_turnover: bool = True,
    ) -> ConstraintResult:
        """Return ``weights`` projected onto the feasible set.

        Parameters
        ----------
        weights:
            Proposed target weights as a fraction of portfolio equity.
        previous_weights:
            Current weights, needed for the turnover and small-trade rules.
        enforce_turnover:
            Set False when the caller wants the unconstrained-turnover target
            (for example when comparing intended vs achievable trading).
        """
        adjustments: list[ConstraintAdjustment] = []
        w = weights.astype("float64").fillna(0.0).copy()
        prev = (
            previous_weights.reindex(w.index).fillna(0.0)
            if previous_weights is not None
            else pd.Series(0.0, index=w.index)
        )

        w = self._apply_long_only(w, adjustments)
        w = self._apply_asset_cap(w, adjustments)
        w = self._apply_class_cap(w, adjustments)
        w = self._apply_position_count(w, adjustments)
        w = self._apply_exposure_caps(w, adjustments)
        w = self._apply_cash_reserve(w, adjustments)
        if enforce_turnover:
            w = self._apply_turnover_cap(w, prev, adjustments)
        w = self._apply_min_trade(w, prev, adjustments)

        # The turnover blend and small-trade revert can pull weights back toward
        # `prev`, which may itself sit at a cap. Re-run the hard caps once so the
        # final vector is guaranteed feasible.
        w = self._apply_asset_cap(w, adjustments)
        w = self._apply_exposure_caps(w, adjustments)

        result = ConstraintResult(weights=w.round(10), adjustments=adjustments)
        if adjustments:
            log.debug("constraints applied", extra={"context": result.summary()})
        return result

    # -- individual constraints -----------------------------------------------

    def _apply_long_only(
        self, w: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Drop negative weights when the portfolio is long-only."""
        if not self.config.long_only:
            return w
        shorts = w[w < -1e-12]
        if shorts.empty:
            return w
        adjustments.append(
            ConstraintAdjustment(
                "long_only",
                f"removed {len(shorts)} short position(s) in long-only mode",
                float(shorts.sum()),
                0.0,
                tuple(str(s) for s in shorts.index),
            )
        )
        return w.clip(lower=0.0)

    def _apply_asset_cap(
        self, w: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Cap each asset and redistribute the excess to uncapped names."""
        cap = self.config.max_asset_weight
        breaches = w[w.abs() > cap + 1e-12]
        if breaches.empty:
            return w

        before_gross = float(w.abs().sum())
        capped = w.clip(-cap, cap)
        excess = float(w.abs().sum() - capped.abs().sum())

        # Redistribute proportionally among names with headroom, iterating a
        # bounded number of times so the redistribution cannot itself breach.
        for _ in range(10):
            if excess <= 1e-12:
                break
            headroom = (cap - capped.abs()).clip(lower=0.0)
            # Only names already holding a position receive the redistribution;
            # spilling into unheld assets would change the strategy's intent.
            eligible = headroom[(capped.abs() > 1e-12) & (headroom > 1e-12)]
            if eligible.empty:
                break
            share = eligible / eligible.sum()
            add = (share * excess).clip(upper=eligible)
            capped.loc[add.index] = capped.loc[add.index] + np.sign(capped.loc[add.index]) * add
            excess -= float(add.sum())

        adjustments.append(
            ConstraintAdjustment(
                "max_asset_weight",
                f"capped {len(breaches)} position(s) at {cap:.1%} of equity",
                before_gross,
                float(capped.abs().sum()),
                tuple(str(s) for s in breaches.index),
            )
        )
        return capped

    def _apply_class_cap(
        self, w: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Scale down asset classes that exceed their exposure cap."""
        cap = self.config.max_asset_class_weight
        if not self.asset_classes:
            return w

        values = w.to_numpy(dtype="float64").copy()
        changed = False
        for name, members in self._class_groups(w.index).items():
            positions = self._class_positions(w.index)[name]
            exposure = float(np.abs(values[positions]).sum())
            if exposure <= cap + 1e-12:
                continue
            scale = cap / exposure
            values[positions] *= scale
            changed = True
            adjustments.append(
                ConstraintAdjustment(
                    "max_asset_class_weight",
                    f"asset class {name!r} scaled by {scale:.3f} to respect the {cap:.0%} cap",
                    exposure,
                    cap,
                    tuple(members),
                )
            )
        return pd.Series(values, index=w.index) if changed else w

    def _apply_position_count(
        self, w: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Keep only the largest ``max_active_positions`` holdings."""
        limit = self.config.max_active_positions
        active = w[w.abs() > 1e-9]
        if len(active) <= limit:
            return w
        keep = set(active.abs().nlargest(limit).index)
        dropped = [str(s) for s in active.index if s not in keep]
        drop_set = set(dropped)
        out = pd.Series(
            np.where([str(s) in drop_set for s in w.index], 0.0, w.to_numpy()), index=w.index
        )
        adjustments.append(
            ConstraintAdjustment(
                "max_active_positions",
                f"dropped {len(dropped)} smallest position(s) to respect the {limit}-position limit",
                float(len(active)),
                float(limit),
                tuple(dropped),
            )
        )
        return out

    def _apply_exposure_caps(
        self, w: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Scale the book to respect gross, net and leverage caps."""
        cfg = self.config
        out = w.copy()

        gross = float(out.abs().sum())
        gross_cap = min(cfg.max_gross_exposure, cfg.max_leverage)
        if gross > gross_cap + 1e-12:
            scale = gross_cap / gross
            out = out * scale
            adjustments.append(
                ConstraintAdjustment(
                    "max_gross_exposure",
                    f"gross exposure scaled by {scale:.3f} to reach the {gross_cap:.2f}x cap",
                    gross,
                    gross_cap,
                )
            )

        net = float(out.sum())
        if abs(net) > cfg.max_net_exposure + 1e-12:
            scale = cfg.max_net_exposure / abs(net)
            out = out * scale
            adjustments.append(
                ConstraintAdjustment(
                    "max_net_exposure",
                    f"net exposure scaled by {scale:.3f} to reach the {cfg.max_net_exposure:.2f}x cap",
                    net,
                    float(out.sum()),
                )
            )
        return out

    def _apply_cash_reserve(
        self, w: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Ensure the minimum cash weight remains unencumbered."""
        min_cash = self.config.minimum_cash_weight
        if min_cash <= 0.0:
            return w
        investable = 1.0 - min_cash
        gross = float(w.abs().sum())
        if gross <= investable + 1e-12:
            return w
        scale = investable / gross
        adjustments.append(
            ConstraintAdjustment(
                "minimum_cash_weight",
                f"book scaled by {scale:.3f} to hold {min_cash:.0%} in cash",
                gross,
                investable,
            )
        )
        return w * scale

    def _apply_turnover_cap(
        self, w: pd.Series, prev: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Blend toward the previous weights so turnover stays within budget.

        Turnover is measured as one-way traded notional: ``sum |w_new - w_old|``.
        Blending is linear, so the realised turnover of the blended vector is
        exactly ``lambda * requested_turnover``.
        """
        limit = self.config.maximum_daily_turnover
        turnover = float((w - prev).abs().sum())
        if turnover <= limit + 1e-12 or turnover <= 0.0:
            return w
        lam = limit / turnover
        adjustments.append(
            ConstraintAdjustment(
                "maximum_daily_turnover",
                f"trade blended at {lam:.3f} to cap turnover at {limit:.0%}",
                turnover,
                limit,
            )
        )
        return prev + lam * (w - prev)

    def _apply_min_trade(
        self, w: pd.Series, prev: pd.Series, adjustments: list[ConstraintAdjustment]
    ) -> pd.Series:
        """Revert trades too small to justify their transaction costs.

        A trade that closes a position entirely is always allowed through, since
        leaving a residual sliver would defeat the intent of exiting.
        """
        threshold = self.config.min_trade_weight
        if threshold <= 0.0:
            return w
        delta = w - prev
        small = delta.abs() < threshold
        closing = (w.abs() < 1e-9) & (prev.abs() > 1e-9)
        revert = small & ~closing
        if not revert.any():
            return w
        out = pd.Series(
            np.where(revert.to_numpy(), prev.to_numpy(), w.to_numpy()), index=w.index
        )
        adjustments.append(
            ConstraintAdjustment(
                "min_trade_weight",
                f"reverted {int(revert.sum())} trade(s) smaller than {threshold:.2%} of equity",
                float(delta.abs().sum()),
                float((out - prev).abs().sum()),
                tuple(str(s) for s in delta.index[revert]),
            )
        )
        return out

    # -- diagnostics ----------------------------------------------------------

    def check(self, weights: pd.Series) -> dict[str, bool]:
        """Return a pass/fail map for every configured limit.

        Useful for tests and for the risk dashboard, where the question is "does
        this vector satisfy the limits" rather than "make it satisfy them".
        """
        cfg = self.config
        w = weights.fillna(0.0)
        gross = float(w.abs().sum())
        checks = {
            "long_only": bool((w >= -1e-9).all()) if cfg.long_only else True,
            "max_asset_weight": bool((w.abs() <= cfg.max_asset_weight + 1e-9).all()),
            "max_gross_exposure": gross <= min(cfg.max_gross_exposure, cfg.max_leverage) + 1e-9,
            "max_net_exposure": abs(float(w.sum())) <= cfg.max_net_exposure + 1e-9,
            "minimum_cash_weight": gross <= (1.0 - cfg.minimum_cash_weight) + 1e-9,
            "max_active_positions": int((w.abs() > 1e-9).sum()) <= cfg.max_active_positions,
        }
        if self.asset_classes:
            checks["max_asset_class_weight"] = all(
                float(w[members].abs().sum()) <= cfg.max_asset_class_weight + 1e-9
                for members in self._class_groups(w.index).values()
            )
        return checks
