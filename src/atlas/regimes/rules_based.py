"""Interpretable, rules-based market-regime detection.

The baseline detector answers two questions with two observable quantities:

**Direction** - is the benchmark trading above its long-term moving average?
    A simple, widely used and economically meaningful proxy for whether the
    market is in an uptrend. It is slow, which is the point: a regime label that
    flips weekly is not a regime.

**Turbulence** - is trailing realised volatility high relative to its own past?
    Volatility is measured over ``volatility_lookback`` days and then converted
    to a percentile against an *expanding* window of its own history, so the
    threshold adapts across decades without using future data. A fixed absolute
    threshold (e.g. "VIX > 20") would classify most of 2017 and most of 2008
    incorrectly at different points in the sample.

Crossing the two gives four regimes:

============  ======================  =======================
              Low volatility          High volatility
============  ======================  =======================
Above trend   ``bull_low_vol``        ``bull_high_vol``
Below trend   ``bear_low_vol``        ``bear_high_vol``
============  ======================  =======================

Three further indicators are recorded but do not drive the discrete label; they
feed the risk overlay and the dashboard:

* **breadth** - fraction of assets above their own 200-day moving average;
* **average pairwise correlation** - diversification tends to vanish in stress;
* **drawdown** - the benchmark's decline from its trailing peak.

A persistence filter (``min_regime_days``) requires a new classification to hold
for several consecutive days before it is accepted, which removes single-day
whipsaws. The filter is causal: it only ever delays a change, never anticipates
one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import RegimeConfig, RegimeLabel
from atlas.data.features import (
    FeatureSet,
    breadth_above_ma,
    rolling_average_correlation,
)
from atlas.data.loader import PricePanel
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger
from atlas.regimes.base import RegimeDetector, RegimeResult

log = get_logger(__name__)

__all__ = ["RulesBasedRegimeDetector"]


class RulesBasedRegimeDetector(RegimeDetector):
    """Two-factor (direction x volatility) regime classifier.

    Parameters
    ----------
    config:
        The :class:`~atlas.config.RegimeConfig` block.
    benchmark:
        Override the benchmark symbol from the config.
    """

    name = "rules_based"

    def __init__(self, config: RegimeConfig, *, benchmark: str | None = None) -> None:
        self.config = config
        self.detector_cfg = config.detector
        self.benchmark = (benchmark or self.detector_cfg.benchmark).upper()

    # -- detection ------------------------------------------------------------

    def detect(
        self,
        panel: PricePanel,
        features: FeatureSet | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> RegimeResult:
        """Classify every date in ``panel``. See :class:`RegimeDetector`."""
        if as_of_date is not None:
            panel = panel.as_of(as_of_date)
            if features is not None:
                features = features.as_of(as_of_date)

        prices = panel.adj_close
        if prices.empty:
            raise InsufficientDataError("cannot detect regimes from an empty price panel")

        benchmark_series = self._benchmark_series(prices)
        indicators = self._indicators(prices, benchmark_series, features)
        raw_labels = self._classify(indicators)
        labels = self._apply_persistence(raw_labels, self.detector_cfg.min_regime_days)

        result = RegimeResult(labels=labels, features=indicators, detector=self.name)
        log.info("regimes detected", extra={"context": result.summary()})
        return result

    # -- components -----------------------------------------------------------

    def _benchmark_series(self, prices: pd.DataFrame) -> pd.Series:
        """Return the benchmark price series, falling back to an equal-weight index."""
        if self.benchmark in prices.columns:
            series = prices[self.benchmark]
            if series.notna().sum() >= self.detector_cfg.trend_ma:
                return series
            log.warning(
                "benchmark has insufficient history; using an equal-weight index instead",
                extra={"context": {"benchmark": self.benchmark}},
            )
        else:
            log.warning(
                "benchmark not in the panel; using an equal-weight index instead",
                extra={"context": {"benchmark": self.benchmark}},
            )
        equal_weight_return = prices.pct_change(fill_method=None).mean(axis=1, skipna=True)
        return (1.0 + equal_weight_return.fillna(0.0)).cumprod().rename("equal_weight")

    def _indicators(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series,
        features: FeatureSet | None,
    ) -> pd.DataFrame:
        """Compute the continuous indicators behind the classification."""
        cfg = self.detector_cfg
        index = prices.index
        benchmark = benchmark.reindex(index)

        # --- direction ---
        trend_ma = benchmark.rolling(cfg.trend_ma, min_periods=cfg.trend_ma).mean()
        trend_score = (benchmark / trend_ma) - 1.0

        # --- volatility ---
        benchmark_returns = benchmark.pct_change(fill_method=None)
        volatility = benchmark_returns.rolling(
            cfg.volatility_lookback, min_periods=max(5, cfg.volatility_lookback // 2)
        ).std(ddof=1) * np.sqrt(252.0)
        vol_percentile = _expanding_percentile(
            volatility,
            window=cfg.volatility_percentile_window,
            min_obs=cfg.volatility_percentile_min_obs,
        )

        # --- breadth ---
        if features is not None and "breadth" in features.market.columns:
            breadth = features.market["breadth"].reindex(index)
        else:
            breadth = breadth_above_ma(prices, cfg.breadth_ma).reindex(index)

        # --- correlation ---
        if features is not None and "avg_correlation" in features.market.columns:
            avg_correlation = features.market["avg_correlation"].reindex(index)
        else:
            avg_correlation = rolling_average_correlation(
                prices.pct_change(fill_method=None), cfg.correlation_lookback
            ).reindex(index)

        # --- drawdown ---
        rolling_peak = benchmark.rolling(cfg.drawdown_lookback, min_periods=1).max()
        drawdown_series = (benchmark / rolling_peak) - 1.0

        return pd.DataFrame(
            {
                "benchmark_price": benchmark,
                "trend_ma": trend_ma,
                "trend_score": trend_score,
                "above_trend": (benchmark > trend_ma).astype(float).where(trend_ma.notna()),
                "volatility": volatility,
                "volatility_percentile": vol_percentile,
                "breadth": breadth,
                "weak_breadth": (breadth < cfg.weak_breadth_threshold).astype(float).where(
                    breadth.notna()
                ),
                "avg_correlation": avg_correlation,
                "drawdown": drawdown_series,
            },
            index=index,
        )

    def _classify(self, indicators: pd.DataFrame) -> pd.Series:
        """Map the indicators onto discrete regime labels."""
        cfg = self.detector_cfg
        above = indicators["above_trend"]
        vol_pct = indicators["volatility_percentile"]

        # Before the volatility percentile has enough history, fall back to
        # comparing volatility with its own expanding median - conservative and
        # still strictly backward-looking.
        fallback_high = (
            indicators["volatility"] > indicators["volatility"].expanding(min_periods=20).median()
        )
        high_vol = vol_pct.gt(cfg.high_volatility_percentile).where(vol_pct.notna(), fallback_high)

        labels = pd.Series(RegimeLabel.UNKNOWN, index=indicators.index, dtype=object)
        known = above.notna()
        bull = known & (above > 0.5)
        bear = known & (above <= 0.5)
        hot = high_vol.fillna(False).astype(bool)

        labels[bull & ~hot] = RegimeLabel.BULL_LOW_VOL
        labels[bull & hot] = RegimeLabel.BULL_HIGH_VOL
        labels[bear & ~hot] = RegimeLabel.BEAR_LOW_VOL
        labels[bear & hot] = RegimeLabel.BEAR_HIGH_VOL
        return labels

    @staticmethod
    def _apply_persistence(labels: pd.Series, min_days: int) -> pd.Series:
        """Require a new label to persist ``min_days`` days before accepting it.

        Strictly causal: the accepted label at *t* depends only on labels at
        ``<= t``. A candidate that has not yet held long enough leaves the
        previously accepted label in force.
        """
        if min_days <= 1 or labels.empty:
            return labels

        values = labels.to_numpy(dtype=object)
        out = np.empty(len(values), dtype=object)
        accepted = RegimeLabel.UNKNOWN
        candidate = None
        streak = 0

        for i, value in enumerate(values):
            if value is RegimeLabel.UNKNOWN or value is None:
                out[i] = accepted
                continue
            if accepted is RegimeLabel.UNKNOWN:
                # First real classification is accepted immediately.
                accepted = value
                candidate, streak = None, 0
                out[i] = accepted
                continue
            if value == accepted:
                candidate, streak = None, 0
            elif value == candidate:
                streak += 1
                if streak >= min_days:
                    accepted = value
                    candidate, streak = None, 0
            else:
                candidate, streak = value, 1
            out[i] = accepted

        return pd.Series(out, index=labels.index, dtype=object)


def _expanding_percentile(series: pd.Series, *, window: int, min_obs: int) -> pd.Series:
    """Percentile rank of each observation within its own trailing history.

    Uses a trailing window of at most ``window`` observations and requires at
    least ``min_obs`` before emitting a value. Implemented with an incremental
    sorted list so the cost is O(n log n) rather than O(n * window).
    """
    values = series.to_numpy(dtype="float64")
    out = np.full(values.shape, np.nan)
    buffer: list[float] = []
    order: list[int] = []  # indices into `values`, in arrival order

    import bisect

    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        # Drop observations that have fallen out of the trailing window.
        while order and (i - order[0]) >= window:
            old_index = order.pop(0)
            old_value = values[old_index]
            pos = bisect.bisect_left(buffer, old_value)
            if pos < len(buffer) and buffer[pos] == old_value:
                buffer.pop(pos)
        pos = bisect.bisect_left(buffer, v)
        bisect.insort(buffer, v)
        order.append(i)
        if len(buffer) >= min_obs:
            out[i] = pos / max(len(buffer) - 1, 1)
    return pd.Series(out, index=series.index, name="volatility_percentile")
