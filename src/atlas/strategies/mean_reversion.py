"""Short-term mean reversion.

Economic rationale
------------------
Over horizons of a few days, liquid index-tracking instruments exhibit partial
reversal of large moves. The standard explanations are inventory and liquidity
provision (a market maker or ETF authorised participant absorbing a one-sided
order flow demands compensation, which is repaid as the imbalance clears) and
short-horizon over-reaction to news (Lehmann 1990; Lo & MacKinlay 1990; Nagel,
*Evaporating Liquidity*, RFS 2012). The effect is strongest exactly when
liquidity provision is expensive - which is also when it is most dangerous, so
the strategy is explicitly filtered.

Signal construction
-------------------
Two components are averaged:

* the negative z-score of the trailing ``return_lookback``-day return, measured
  against that asset's own trailing distribution;
* a rescaled RSI, mapped so that oversold (< 50) is positive and overbought
  (> 50) is negative.

Entry, exit and holding rules
-----------------------------
* A position opens once ``|z| >= entry_threshold``.
* It is held while ``|z| > exit_threshold``, and closes as the z-score reverts
  toward zero - the strategy takes profit into the reversion rather than waiting
  for a full sign flip.
* A position is force-closed after ``max_holding_days`` days, so a signal that
  never reverts cannot become a permanent position.

Filters
-------
``use_trend_filter``
    Suppresses signals that fight a strong long-term trend (fading a persistent
    downtrend is the classic way this strategy loses money).
``use_volatility_filter``
    Suppresses signals when trailing volatility is in an extreme percentile of
    its own history - during a crisis, "cheap" keeps getting cheaper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import MeanReversionConfig
from atlas.data.features import (
    FeatureSet,
    moving_average,
    realized_volatility,
    relative_strength_index,
    rolling_zscore,
    trailing_return,
)
from atlas.strategies.base import SignalResult, Strategy, register_strategy

__all__ = ["MeanReversionStrategy"]


@register_strategy
class MeanReversionStrategy(Strategy):
    """Short-horizon reversal strategy with trend, volatility and holding filters."""

    name = "mean_reversion"

    def __init__(
        self, config: MeanReversionConfig, *, asset_classes: dict[str, str] | None = None
    ) -> None:
        super().__init__(config, asset_classes=asset_classes)
        self.config: MeanReversionConfig = config

    @property
    def required_lookback(self) -> int:
        """History needed before the slowest filter is defined."""
        cfg = self.config
        return max(
            cfg.zscore_lookback + cfg.return_lookback,
            cfg.rsi_lookback * 2,
            cfg.trend_filter_ma if cfg.use_trend_filter else 0,
            cfg.volatility_lookback,
        )

    # -- computation ----------------------------------------------------------

    def _compute(self, prices: pd.DataFrame, features: FeatureSet) -> SignalResult:
        cfg = self.config

        # --- component 1: z-score of the short-horizon return -----------------
        short_return = self._lookup(
            features,
            f"return_{cfg.return_lookback}d",
            lambda: trailing_return(prices, cfg.return_lookback),
        )
        zscore = self._lookup(
            features,
            f"zscore_return_5d_{cfg.zscore_lookback}"
            if cfg.return_lookback == 5
            else f"__zscore_{cfg.return_lookback}_{cfg.zscore_lookback}",
            lambda: rolling_zscore(short_return, cfg.zscore_lookback),
        )

        # --- component 2: deviation from the short moving average --------------
        short_ma = moving_average(prices, cfg.zscore_lookback)
        deviation = (prices / short_ma) - 1.0
        deviation_z = rolling_zscore(deviation, cfg.zscore_lookback)

        # --- component 3: RSI --------------------------------------------------
        rsi = self._lookup(
            features, "rsi", lambda: relative_strength_index(prices, cfg.rsi_lookback)
        )
        # Map RSI in [0, 100] -> [-1, 1] with the sign flipped: oversold is bullish.
        rsi_signal = -((rsi - 50.0) / 50.0)

        # --- blended reversion score ------------------------------------------
        # Average the two z-score views, then flip the sign: a large positive
        # move produces a negative (short / reduce) signal.
        blended_z = 0.5 * (zscore.fillna(0.0) + deviation_z.fillna(0.0))
        z_component = -blended_z
        raw = (1.0 - cfg.rsi_weight) * z_component + cfg.rsi_weight * rsi_signal.fillna(0.0)
        raw = pd.DataFrame(raw, index=prices.index, columns=prices.columns)

        # --- entry / exit state machine ---------------------------------------
        magnitude = blended_z.abs()
        # `active` is True where a position should be open, computed with a
        # vectorised hysteresis: enter on |z| >= entry, stay while |z| > exit.
        active = self._hysteresis(
            enter=magnitude >= cfg.entry_threshold,
            hold=magnitude > cfg.exit_threshold,
            direction=-np.sign(blended_z.fillna(0.0)),
            max_holding_days=cfg.max_holding_days,
        )

        # Conviction scales with how far past the entry threshold the z-score is,
        # saturating at twice the threshold.
        conviction = ((magnitude - cfg.exit_threshold) / max(cfg.entry_threshold, 1e-9)).clip(0.0, 1.0)
        signal = np.sign(z_component) * conviction * active.astype(float)
        signal = pd.DataFrame(signal, index=prices.index, columns=prices.columns)

        # Blend in the RSI tilt so the signal is continuous rather than binary.
        signal = 0.75 * signal + 0.25 * raw.clip(-1.0, 1.0) * active.astype(float)

        # --- filters ------------------------------------------------------------
        trend_filter = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        if cfg.use_trend_filter:
            trend_ma = self._lookup(
                features,
                f"ma_{cfg.trend_filter_ma}",
                lambda: moving_average(prices, cfg.trend_filter_ma),
            )
            above_trend = prices > trend_ma
            # Block short signals in an uptrend and long signals in a downtrend:
            # only fade moves that run *against* the prevailing long-term trend.
            allow_long = above_trend.fillna(False)
            allow_short = (~above_trend).fillna(False)
            trend_filter = pd.DataFrame(
                np.where(signal > 0, allow_long, np.where(signal < 0, allow_short, True)).astype(float),
                index=prices.index,
                columns=prices.columns,
            )
            signal = signal * trend_filter

        vol_gate = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        if cfg.use_volatility_filter:
            vol = self._lookup(
                features,
                f"volatility_{cfg.volatility_lookback}d",
                lambda: realized_volatility(
                    prices.pct_change(fill_method=None), cfg.volatility_lookback
                ),
            )
            vol_gate = self._volatility_gate(
                vol,
                cfg.volatility_percentile_cap,
                min_periods=min(252, max(60, len(prices) // 4)),
                floor=0.0,
            )
            signal = signal * vol_gate

        if cfg.long_only:
            signal = signal.clip(lower=0.0)

        warm = pd.Series(
            (np.arange(len(prices)) >= self.required_lookback - 1).astype(float), index=prices.index
        )
        signal = signal.mul(warm, axis=0)

        # Confidence: distance past the entry threshold, gated by the filters.
        confidence = (conviction * trend_filter * vol_gate).clip(0.0, 1.0)

        return SignalResult(
            strategy=self.name,
            signal=signal.fillna(0.0).clip(-1.0, 1.0),
            raw=raw,
            confidence=confidence,
            diagnostics={
                "zscore": blended_z,
                "rsi": rsi,
                "position_active": active.astype(float),
                "trend_filter": trend_filter,
                "volatility_gate": vol_gate,
            },
        )

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _hysteresis(
        enter: pd.DataFrame,
        hold: pd.DataFrame,
        direction: pd.DataFrame,
        max_holding_days: int,
    ) -> pd.DataFrame:
        """Entry/exit state machine with a maximum holding period.

        A position opens on ``enter``, stays open while ``hold`` remains True and
        the direction has not flipped, and closes when ``hold`` turns False, the
        direction flips, or ``max_holding_days`` bars have elapsed.

        The loop is explicitly sequential because the state at *t* depends on the
        state at *t-1*; it therefore cannot look ahead by construction.
        """
        enter_a = enter.fillna(False).to_numpy()
        hold_a = hold.fillna(False).to_numpy()
        dir_a = direction.fillna(0.0).to_numpy()

        n_rows, n_cols = enter_a.shape
        out = np.zeros((n_rows, n_cols), dtype=bool)
        open_for = np.zeros(n_cols, dtype=int)
        open_dir = np.zeros(n_cols, dtype=float)

        for i in range(n_rows):
            currently_open = open_for > 0
            # Close where the hold condition failed, the direction flipped, or
            # the maximum holding period was reached.
            flipped = currently_open & (dir_a[i] * open_dir < 0)
            expired = currently_open & (open_for >= max_holding_days)
            close = currently_open & (~hold_a[i] | flipped | expired)
            open_for = np.where(close, 0, open_for)
            open_dir = np.where(close, 0.0, open_dir)

            # Open new positions. A position closed by the holding-period rule
            # may not reopen on the same bar: without this the rule would be
            # unobservable, because a persistent entry signal would re-enter
            # immediately and the position would never actually go flat.
            opening = (open_for == 0) & enter_a[i] & (dir_a[i] != 0.0) & ~expired
            open_for = np.where(opening, 1, open_for)
            open_dir = np.where(opening, dir_a[i], open_dir)

            # Age the positions that stayed open.
            still_open = (open_for > 0) & ~opening
            open_for = np.where(still_open, open_for + 1, open_for)

            out[i] = open_for > 0

        return pd.DataFrame(out, index=enter.index, columns=enter.columns)

    @staticmethod
    def _lookup(features: FeatureSet, name: str, fallback) -> pd.DataFrame:
        """Return a precomputed feature, or compute it if the set lacks it."""
        if name in features:
            return features.get(name)
        return fallback()
