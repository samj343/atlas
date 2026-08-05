"""Time-series trend following.

Economic rationale
------------------
Time-series momentum - the tendency for an asset's own past excess return to
predict its near-future return - is one of the most robustly documented
cross-asset premia (Moskowitz, Ooi & Pedersen, *Time Series Momentum*, JFE 2012;
Hurst, Ooi & Pedersen, 2017). Plausible drivers are slow diffusion of
information, investor under-reaction to news, and the risk-management behaviour
of leveraged holders, which forces trades in the direction of recent moves.

The strategy is deliberately simple and interpretable: it blends four
backward-looking measures of the same idea rather than optimising a single
clever one.

.. math::

    s_i = w_1 \\, \\mathrm{sign}\\!\\left(\\frac{P}{MA_{fast}} - 1\\right)
        + w_2 \\, \\mathrm{sign}\\!\\left(\\frac{P}{MA_{slow}} - 1\\right)
        + w_3 \\, \\tilde{r}^{(short)}
        + w_4 \\, \\tilde{r}^{(long)}

where :math:`\\tilde r` denotes a trailing return normalised by its own trailing
volatility and squashed into ``[-1, 1]``.

Turnover control
----------------
Raw sign functions flip on every marginal crossing of a moving average, which is
expensive. Two mechanisms damp this:

* the two momentum components are *continuous*, so they change gradually;
* a soft neutral zone (``neutral_threshold``) zeroes out small signals and
  rescales the rest continuously, so position size does not jump at the boundary.

Risk control
------------
When ``volatility_scaling`` is on, the signal is multiplied by a gate that
shrinks toward 0.25 as trailing volatility enters the top percentile of its own
expanding history. This reduces exposure during volatility spikes without
requiring a discretionary override.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import TrendConfig
from atlas.data.features import (
    FeatureSet,
    moving_average,
    realized_volatility,
    trailing_return,
)
from atlas.strategies.base import (
    SignalResult,
    Strategy,
    register_strategy,
    soft_threshold,
)

__all__ = ["TrendStrategy"]


@register_strategy
class TrendStrategy(Strategy):
    """Blended time-series trend-following strategy.

    Combines two moving-average position measures with two volatility-normalised
    trailing returns, applies a soft neutral zone, and optionally scales the
    result down when realised volatility is unusually high.
    """

    name = "trend"

    def __init__(self, config: TrendConfig, *, asset_classes: dict[str, str] | None = None) -> None:
        super().__init__(config, asset_classes=asset_classes)
        self.config: TrendConfig = config

    @property
    def required_lookback(self) -> int:
        """History needed before the slowest component is defined."""
        return max(
            self.config.slow_ma,
            self.config.momentum_lookback,
            self.config.short_momentum_lookback,
        )

    # -- computation ----------------------------------------------------------

    def _compute(self, prices: pd.DataFrame, features: FeatureSet) -> SignalResult:
        cfg = self.config
        weights = cfg.component_weights.normalized()

        # --- component 1 & 2: position relative to the moving averages --------
        ma_fast = self._lookup(features, f"ma_{cfg.fast_ma}", lambda: moving_average(prices, cfg.fast_ma))
        ma_slow = self._lookup(features, f"ma_{cfg.slow_ma}", lambda: moving_average(prices, cfg.slow_ma))

        # A smooth substitute for sign(): the raw distance from the moving
        # average scaled by trailing volatility, squashed by tanh. This keeps the
        # economic meaning of "above/below the average" while removing the
        # discontinuity at the crossing point.
        daily_vol = self._daily_volatility(prices, features)
        ma_horizon_vol_fast = daily_vol * np.sqrt(cfg.fast_ma / 2.0)
        ma_horizon_vol_slow = daily_vol * np.sqrt(cfg.slow_ma / 2.0)

        dist_fast = ((prices / ma_fast) - 1.0) / ma_horizon_vol_fast.replace(0.0, np.nan)
        dist_slow = ((prices / ma_slow) - 1.0) / ma_horizon_vol_slow.replace(0.0, np.nan)
        comp_ma_fast = np.tanh(dist_fast)
        comp_ma_slow = np.tanh(dist_slow)

        # --- component 3 & 4: volatility-normalised trailing returns ---------
        ret_short = self._lookup(
            features,
            f"return_{cfg.short_momentum_lookback}d",
            lambda: trailing_return(prices, cfg.short_momentum_lookback),
        )
        ret_long = self._lookup(
            features,
            f"return_{cfg.momentum_lookback}d",
            lambda: trailing_return(prices, cfg.momentum_lookback),
        )
        # Scale each trailing return by the volatility of a return over the same
        # horizon, so a 63-day and a 252-day measure are on a comparable scale.
        scale_short = (daily_vol * np.sqrt(cfg.short_momentum_lookback)).replace(0.0, np.nan)
        scale_long = (daily_vol * np.sqrt(cfg.momentum_lookback)).replace(0.0, np.nan)
        comp_mom_short = np.tanh(ret_short / scale_short)
        comp_mom_long = np.tanh(ret_long / scale_long)

        components = {
            "ma_fast": comp_ma_fast,
            "ma_slow": comp_ma_slow,
            "momentum_short": comp_mom_short,
            "momentum_long": comp_mom_long,
        }

        # --- blend ------------------------------------------------------------
        raw = sum(weights[k] * v.fillna(0.0) for k, v in components.items())
        raw = pd.DataFrame(raw, index=prices.index, columns=prices.columns)

        # Confidence reflects how much the four components agree. Four aligned
        # components -> 1.0; perfectly offsetting components -> 0.0.
        stacked = np.stack([np.sign(v.fillna(0.0).to_numpy()) for v in components.values()])
        agreement = np.abs(stacked.sum(axis=0)) / len(components)
        confidence = pd.DataFrame(agreement, index=prices.index, columns=prices.columns)

        # --- neutral zone -----------------------------------------------------
        signal = soft_threshold(raw.clip(-1.0, 1.0), cfg.neutral_threshold)

        # --- volatility gate --------------------------------------------------
        gate = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        if cfg.volatility_scaling:
            vol = self._lookup(
                features,
                f"volatility_{cfg.volatility_lookback}d",
                lambda: realized_volatility(prices.pct_change(fill_method=None), cfg.volatility_lookback),
            )
            gate = self._volatility_gate(
                vol, cfg.volatility_percentile_cap, min_periods=min(252, max(60, len(prices) // 4))
            )
            signal = signal * gate

        # --- direction constraint --------------------------------------------
        if cfg.long_only:
            signal = signal.clip(lower=0.0)

        # Only emit once every component has had a chance to warm up.
        warm = pd.Series(
            (np.arange(len(prices)) >= self.required_lookback - 1).astype(float), index=prices.index
        )
        signal = signal.mul(warm, axis=0)

        return SignalResult(
            strategy=self.name,
            signal=signal.fillna(0.0).clip(-1.0, 1.0),
            raw=raw,
            confidence=confidence,
            diagnostics={
                "component_ma_fast": comp_ma_fast,
                "component_ma_slow": comp_ma_slow,
                "component_momentum_short": comp_mom_short,
                "component_momentum_long": comp_mom_long,
                "volatility_gate": gate,
            },
        )

    # -- helpers --------------------------------------------------------------

    def _daily_volatility(self, prices: pd.DataFrame, features: FeatureSet) -> pd.DataFrame:
        """Trailing daily (not annualised) volatility used to scale distances."""
        annual = self._lookup(
            features,
            f"volatility_{self.config.volatility_lookback}d",
            lambda: realized_volatility(
                prices.pct_change(fill_method=None), self.config.volatility_lookback
            ),
        )
        daily = annual / np.sqrt(252.0)
        # Floor the volatility so a near-constant series (e.g. a T-bill ETF)
        # cannot produce an enormous normalised distance.
        return daily.clip(lower=1e-4)

    @staticmethod
    def _lookup(features: FeatureSet, name: str, fallback) -> pd.DataFrame:
        """Return a precomputed feature, or compute it if the set lacks it."""
        if name in features:
            return features.get(name)
        return fallback()
