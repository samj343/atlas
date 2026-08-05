"""Feature engineering with a strict point-in-time contract.

Every feature in this module obeys one rule:

    ``feature.loc[t, symbol]`` may depend only on observations dated ``<= t``.

Rolling windows in pandas are already backward-looking, so a feature computed
here is *observable at the close of day t*. Consumers that want to trade on a
feature must additionally shift it, because a signal computed from the close of
day *t* cannot be executed until day *t+1*. That shift lives in
:class:`~atlas.backtest.engine.BacktestEngine` (execution timing), not here, so
that features stay usable for same-day analysis and reporting.

:func:`FeatureSet.shifted` provides an explicit, tested way to obtain the
strictly-lagged view when a caller wants one.

The look-ahead contract is enforced by ``tests/unit/test_lookahead.py``, which
perturbs future prices and asserts that no historical feature value changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.data.loader import PricePanel
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "DEFAULT_RETURN_HORIZONS",
    "FeatureEngineer",
    "FeatureSet",
    "average_true_range",
    "breadth_above_ma",
    "cross_sectional_rank",
    "drawdown",
    "ewm_volatility",
    "moving_average",
    "price_to_ma",
    "realized_volatility",
    "relative_strength_index",
    "rolling_average_correlation",
    "rolling_zscore",
    "trailing_return",
]

DEFAULT_RETURN_HORIZONS: tuple[int, ...] = (1, 5, 21, 63, 126, 252)
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Primitive feature functions (all backward-looking)
# ---------------------------------------------------------------------------


def trailing_return(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple return over the trailing ``window`` trading days.

    ``r_t = P_t / P_{t-window} - 1``. Uses only prices up to and including *t*.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    return prices / prices.shift(window) - 1.0


def moving_average(prices: pd.DataFrame, window: int, *, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing simple moving average over ``window`` observations."""
    if window < 1:
        raise ValueError("window must be >= 1")
    mp = window if min_periods is None else min_periods
    return prices.rolling(window=window, min_periods=mp).mean()


def price_to_ma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Ratio of price to its trailing moving average, minus one.

    Positive values mean the asset trades above its moving average.
    """
    ma = moving_average(prices, window)
    return (prices / ma) - 1.0


def realized_volatility(
    returns: pd.DataFrame,
    window: int,
    *,
    annualize: bool = True,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Trailing realised volatility of daily returns.

    Uses the sample standard deviation over ``window`` observations, annualised
    by ``sqrt(252)`` when ``annualize`` is True.
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")
    mp = max(2, window // 2) if min_periods is None else min_periods
    vol = returns.rolling(window=window, min_periods=mp).std(ddof=1)
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def ewm_volatility(
    returns: pd.DataFrame,
    halflife: float,
    *,
    annualize: bool = True,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Exponentially weighted volatility of daily returns."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    vol = returns.ewm(halflife=halflife, min_periods=min_periods, adjust=True).std(bias=False)
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def rolling_zscore(
    frame: pd.DataFrame, window: int, *, min_periods: int | None = None, clip: float | None = 5.0
) -> pd.DataFrame:
    """Rolling z-score of a series against its own trailing distribution.

    A zero trailing standard deviation yields ``NaN`` rather than an infinity.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    mp = max(2, window // 2) if min_periods is None else min_periods
    roll = frame.rolling(window=window, min_periods=mp)
    mean = roll.mean()
    std = roll.std(ddof=1)
    std = std.where(std > 1e-12)
    z = (frame - mean) / std
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def drawdown(prices: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Drawdown from the running (or rolling) maximum, as a negative fraction.

    ``window=None`` uses an expanding maximum from the start of the series.
    """
    if window is None:
        peak = prices.cummax()
    else:
        if window < 1:
            raise ValueError("window must be >= 1")
        peak = prices.rolling(window=window, min_periods=1).max()
    return (prices / peak) - 1.0


def average_true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Average true range over ``window`` days.

    True range is ``max(high-low, |high-prev_close|, |low-prev_close|)``. With
    ``normalize=True`` the result is divided by the close, giving a unit-free
    volatility proxy comparable across assets.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
        keys=["hl", "hc", "lc"],
    )
    # Element-wise max across the three candidate ranges, per symbol.
    true_range = tr.T.groupby(level=1).max().T
    true_range = true_range.reindex(columns=close.columns)
    atr = true_range.rolling(window=window, min_periods=max(2, window // 2)).mean()
    return atr / close if normalize else atr


def relative_strength_index(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's relative strength index, in ``[0, 100]``.

    Uses an exponentially weighted average of gains and losses with
    ``alpha = 1/window``, matching Wilder's original smoothing.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows have zero average loss -> RSI is 100 by definition.
    rsi = rsi.where(avg_loss > 0.0, other=100.0)
    return rsi.where(avg_gain.notna())


def cross_sectional_rank(frame: pd.DataFrame, *, min_assets: int = 2) -> pd.DataFrame:
    """Percentile rank of each asset within each row, scaled to ``[-1, 1]``.

    Rows with fewer than ``min_assets`` valid observations return ``NaN``, since
    a cross-sectional rank across one asset is meaningless. Ties receive the
    average rank, which keeps the cross-section mean at zero.
    """
    counts = frame.notna().sum(axis=1)
    ranks = frame.rank(axis=1, method="average", na_option="keep")
    n = counts.replace(0, np.nan)
    scaled = 2.0 * ((ranks.sub(0.5, axis=0)).div(n, axis=0)) - 1.0
    return scaled.where(counts >= min_assets, other=np.nan)


def rolling_average_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Average pairwise correlation of asset returns over a trailing window.

    A rising average correlation is a classic risk-off indicator: diversification
    disappears exactly when it is needed.
    """
    if window < 5:
        raise ValueError("window must be >= 5")
    n_assets = returns.shape[1]
    if n_assets < 2:
        return pd.Series(np.nan, index=returns.index, name="avg_correlation")

    out = pd.Series(np.nan, index=returns.index, dtype="float64", name="avg_correlation")
    values = returns.to_numpy(dtype="float64")
    for i in range(window - 1, len(returns)):
        block = values[i - window + 1 : i + 1]
        # Keep only columns with a full, non-degenerate window.
        valid = ~np.isnan(block).any(axis=0)
        if valid.sum() < 2:
            continue
        sub = block[:, valid]
        std = sub.std(axis=0, ddof=1)
        keep = std > 1e-12
        if keep.sum() < 2:
            continue
        corr = np.corrcoef(sub[:, keep], rowvar=False)
        upper = corr[np.triu_indices_from(corr, k=1)]
        if upper.size:
            out.iloc[i] = float(np.nanmean(upper))
    return out


def breadth_above_ma(prices: pd.DataFrame, window: int = 200) -> pd.Series:
    """Fraction of assets trading above their own ``window``-day moving average.

    Values lie in ``[0, 1]``. Assets without a full moving-average window are
    excluded from both numerator and denominator.
    """
    ma = moving_average(prices, window)
    above = (prices > ma)
    valid = prices.notna() & ma.notna()
    numerator = (above & valid).sum(axis=1)
    denominator = valid.sum(axis=1)
    breadth = numerator / denominator.replace(0, np.nan)
    return breadth.rename("breadth")


# ---------------------------------------------------------------------------
# FeatureSet container
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """A named collection of aligned per-asset feature frames.

    Attributes
    ----------
    features:
        Mapping of feature name -> wide ``DataFrame`` (index = date, columns = symbol).
    market:
        Market-level (single-series) features such as breadth and average
        correlation, stored as a ``DataFrame`` with one column per indicator.
    """

    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    market: pd.DataFrame = field(default_factory=pd.DataFrame)

    # -- mapping-like access --------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self.features

    def __iter__(self) -> Iterator[str]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def names(self) -> list[str]:
        """Names of the available per-asset features."""
        return sorted(self.features)

    @property
    def index(self) -> pd.DatetimeIndex:
        """The shared date index."""
        if not self.features:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(next(iter(self.features.values())).index)

    @property
    def symbols(self) -> list[str]:
        """The shared symbol columns."""
        if not self.features:
            return []
        return list(next(iter(self.features.values())).columns)

    def get(self, name: str, default: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the frame for ``name``.

        Raises
        ------
        KeyError
            If the feature is absent and no ``default`` is provided.
        """
        if name in self.features:
            return self.features[name]
        if default is not None:
            return default
        raise KeyError(
            f"feature {name!r} not available; known features: {', '.join(self.names)}"
        )

    def add(self, name: str, frame: pd.DataFrame) -> None:
        """Register (or replace) a feature frame."""
        self.features[name] = frame

    def market_series(self, name: str) -> pd.Series:
        """Return a market-level indicator by name."""
        if name not in self.market.columns:
            raise KeyError(
                f"market feature {name!r} not available; known: {list(self.market.columns)}"
            )
        return self.market[name]

    # -- views ----------------------------------------------------------------

    def shifted(self, periods: int = 1) -> FeatureSet:
        """Return a copy with every feature lagged by ``periods`` days.

        Use this when a consumer needs values that were observable *strictly
        before* the current bar.
        """
        return FeatureSet(
            features={k: v.shift(periods) for k, v in self.features.items()},
            market=self.market.shift(periods) if not self.market.empty else self.market,
        )

    def as_of(self, timestamp: pd.Timestamp | str) -> FeatureSet:
        """Return a copy truncated at ``timestamp`` (inclusive)."""
        ts = pd.Timestamp(timestamp)
        return FeatureSet(
            features={k: v.loc[v.index <= ts] for k, v in self.features.items()},
            market=self.market.loc[self.market.index <= ts] if not self.market.empty else self.market,
        )

    def slice(self, start: pd.Timestamp | str | None, end: pd.Timestamp | str | None) -> FeatureSet:
        """Return a copy restricted to ``[start, end]``."""

        def _cut(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame
            if start is not None:
                out = out.loc[out.index >= pd.Timestamp(start)]
            if end is not None:
                out = out.loc[out.index <= pd.Timestamp(end)]
            return out

        return FeatureSet(
            features={k: _cut(v) for k, v in self.features.items()},
            market=_cut(self.market) if not self.market.empty else self.market,
        )

    def select(self, symbols: Iterable[str]) -> FeatureSet:
        """Return a copy restricted to ``symbols``."""
        cols = list(symbols)
        return FeatureSet(
            features={k: v.reindex(columns=cols) for k, v in self.features.items()},
            market=self.market,
        )

    def to_wide(self) -> pd.DataFrame:
        """Return every feature as one frame with ``(feature, symbol)`` columns.

        This is the ``pd.DataFrame`` form referenced by the ``Strategy`` protocol
        and is convenient for persistence and inspection.
        """
        if not self.features:
            return pd.DataFrame()
        return pd.concat(self.features, axis=1)

    def row(self, timestamp: pd.Timestamp | str) -> pd.DataFrame:
        """Return a ``symbol x feature`` frame for a single date."""
        ts = pd.Timestamp(timestamp)
        data = {}
        for name, frame in self.features.items():
            if ts in frame.index:
                data[name] = frame.loc[ts]
        return pd.DataFrame(data)

    def summary(self) -> dict[str, Any]:
        """Compact description of the feature set."""
        return {
            "n_features": len(self.features),
            "n_market_features": int(self.market.shape[1]) if not self.market.empty else 0,
            "n_symbols": len(self.symbols),
            "n_dates": len(self.index),
            "start": str(self.index.min().date()) if len(self.index) else None,
            "end": str(self.index.max().date()) if len(self.index) else None,
        }


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Compute the standard Atlas feature set from a price panel.

    Parameters
    ----------
    return_horizons:
        Trailing-return horizons in trading days.
    volatility_windows:
        Rolling realised-volatility windows in trading days.
    ma_windows:
        Moving-average windows in trading days.
    ewm_halflife:
        Half-life for the exponentially weighted volatility estimate.
    zscore_windows:
        Windows used for rolling z-scores of the 5-day return and of the
        price-to-moving-average ratio.
    breadth_window, correlation_window, drawdown_window, atr_window, rsi_window:
        Windows for the corresponding indicators.
    price_field:
        Which price series features are computed from. ``adj_close`` (total
        return) is the default and is what strategies should use.
    """

    def __init__(
        self,
        *,
        return_horizons: Iterable[int] = DEFAULT_RETURN_HORIZONS,
        volatility_windows: Iterable[int] = (20, 63, 252),
        ma_windows: Iterable[int] = (20, 50, 100, 200),
        ewm_halflife: float = 21.0,
        zscore_windows: Iterable[int] = (20, 63),
        breadth_window: int = 200,
        correlation_window: int = 63,
        drawdown_window: int = 252,
        atr_window: int = 14,
        rsi_window: int = 14,
        price_field: str = "adj_close",
    ) -> None:
        self.return_horizons = tuple(sorted({int(h) for h in return_horizons}))
        self.volatility_windows = tuple(sorted({int(w) for w in volatility_windows}))
        self.ma_windows = tuple(sorted({int(w) for w in ma_windows}))
        self.ewm_halflife = float(ewm_halflife)
        self.zscore_windows = tuple(sorted({int(w) for w in zscore_windows}))
        self.breadth_window = int(breadth_window)
        self.correlation_window = int(correlation_window)
        self.drawdown_window = int(drawdown_window)
        self.atr_window = int(atr_window)
        self.rsi_window = int(rsi_window)
        self.price_field = price_field

    @property
    def max_lookback(self) -> int:
        """Longest window used, i.e. the required warm-up in trading days."""
        return max(
            [
                *self.return_horizons,
                *self.volatility_windows,
                *self.ma_windows,
                *self.zscore_windows,
                self.breadth_window,
                self.correlation_window,
                self.drawdown_window,
                self.atr_window,
                self.rsi_window,
                int(self.ewm_halflife * 4),
            ]
        )

    @classmethod
    def from_config(cls, config: Any) -> FeatureEngineer:
        """Build an engineer whose windows cover everything the config needs."""
        strategies = config.strategies.strategies
        regime = config.regimes.detector
        ma_windows = {
            20,
            50,
            strategies.trend.fast_ma,
            strategies.trend.slow_ma,
            strategies.mean_reversion.trend_filter_ma,
            regime.trend_ma,
            regime.breadth_ma,
        }
        vol_windows = {
            20,
            strategies.trend.volatility_lookback,
            strategies.mean_reversion.volatility_lookback,
            strategies.cross_sectional_momentum.volatility_lookback,
            regime.volatility_lookback,
            config.risk.volatility.lookback,
        }
        horizons = {
            1,
            5,
            21,
            63,
            126,
            252,
            strategies.trend.momentum_lookback,
            strategies.trend.short_momentum_lookback,
            strategies.cross_sectional_momentum.lookback,
            strategies.mean_reversion.return_lookback,
        }
        zscores = {20, 63, strategies.mean_reversion.zscore_lookback}
        return cls(
            return_horizons=horizons,
            volatility_windows=vol_windows,
            ma_windows=ma_windows,
            ewm_halflife=config.risk.volatility.halflife,
            zscore_windows=zscores,
            breadth_window=regime.breadth_ma,
            correlation_window=regime.correlation_lookback,
            drawdown_window=regime.drawdown_lookback,
            rsi_window=strategies.mean_reversion.rsi_lookback,
            price_field=config.backtest.price_field,
        )

    # -- computation ----------------------------------------------------------

    def build(self, panel: PricePanel) -> FeatureSet:
        """Compute the full feature set for ``panel``.

        Raises
        ------
        InsufficientDataError
            If the panel has fewer rows than the shortest usable window.
        """
        prices = panel.field(self.price_field)
        if prices.empty:
            raise InsufficientDataError("cannot build features from an empty price panel")
        if len(prices) < 3:
            raise InsufficientDataError(
                f"need at least 3 observations to build features, got {len(prices)}"
            )

        returns = prices.pct_change(fill_method=None)
        log_returns = np.log(prices).diff()

        features: dict[str, pd.DataFrame] = {
            "price": prices,
            "return_1d": returns,
            "log_return_1d": log_returns,
        }

        # --- trailing returns ---
        for h in self.return_horizons:
            features[f"return_{h}d"] = trailing_return(prices, h)

        # --- volatility ---
        for w in self.volatility_windows:
            features[f"volatility_{w}d"] = realized_volatility(returns, w)
        features["volatility_ewm"] = ewm_volatility(returns, self.ewm_halflife)

        # --- moving averages and price ratios ---
        for w in self.ma_windows:
            features[f"ma_{w}"] = moving_average(prices, w)
            features[f"price_to_ma_{w}"] = price_to_ma(prices, w)

        # --- z-scores ---
        for w in self.zscore_windows:
            features[f"zscore_return_5d_{w}"] = rolling_zscore(
                features.get("return_5d", trailing_return(prices, 5)), w
            )
            base_ma = min(self.ma_windows, key=lambda m: abs(m - w))
            features[f"zscore_price_to_ma_{w}"] = rolling_zscore(
                features[f"price_to_ma_{base_ma}"], w
            )
        # Price z-score against its own trailing distribution.
        for w in self.zscore_windows:
            features[f"zscore_price_{w}"] = rolling_zscore(prices, w)

        # --- drawdown ---
        features["drawdown"] = drawdown(prices, None)
        features[f"drawdown_{self.drawdown_window}d"] = drawdown(prices, self.drawdown_window)

        # --- range / oscillator ---
        features["atr"] = average_true_range(
            panel.high, panel.low, panel.close, self.atr_window, normalize=True
        )
        features["rsi"] = relative_strength_index(prices, self.rsi_window)

        # --- risk-adjusted momentum, used by cross-sectional ranking ---
        for h in (63, 126, 252):
            if f"return_{h}d" not in features:
                features[f"return_{h}d"] = trailing_return(prices, h)
            vol_window = min(self.volatility_windows, key=lambda w: abs(w - min(h, 63)))
            vol = features[f"volatility_{vol_window}d"]
            features[f"risk_adjusted_momentum_{h}d"] = _safe_divide(
                features[f"return_{h}d"], vol
            )

        # --- cross-sectional ranks ---
        for h in (21, 63, 126, 252):
            key = f"return_{h}d"
            if key in features:
                features[f"rank_return_{h}d"] = cross_sectional_rank(features[key])
        if "risk_adjusted_momentum_252d" in features:
            features["rank_risk_adjusted_momentum"] = cross_sectional_rank(
                features["risk_adjusted_momentum_252d"]
            )

        # --- rolling correlation to the equal-weight market ---
        features["correlation_to_market"] = self._correlation_to_market(returns)

        # --- market-level indicators ---
        market = self._market_features(prices, returns)

        fs = FeatureSet(features=features, market=market)
        log.info("features built", extra={"context": fs.summary()})
        return fs

    def _correlation_to_market(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Rolling correlation of each asset to the equal-weight cross-section."""
        window = self.correlation_window
        market = returns.mean(axis=1, skipna=True)
        min_periods = max(5, window // 2)
        out = {}
        for col in returns.columns:
            out[col] = returns[col].rolling(window, min_periods=min_periods).corr(market)
        return pd.DataFrame(out, index=returns.index)

    def _market_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute market-level (single-series) indicators."""
        market = pd.DataFrame(index=prices.index)
        market["breadth"] = breadth_above_ma(prices, self.breadth_window)
        market["breadth_50"] = breadth_above_ma(prices, 50)
        market["avg_correlation"] = rolling_average_correlation(returns, self.correlation_window)
        equal_weight = returns.mean(axis=1, skipna=True)
        market["equal_weight_return"] = equal_weight
        equity = (1.0 + equal_weight.fillna(0.0)).cumprod()
        market["equal_weight_drawdown"] = (equity / equity.cummax()) - 1.0
        market["dispersion"] = returns.std(axis=1, ddof=1)
        market["n_valid_assets"] = returns.notna().sum(axis=1).astype("float64")
        return market


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame, floor: float = 1e-8) -> pd.DataFrame:
    """Element-wise division that returns ``NaN`` instead of ``inf``."""
    denom = denominator.where(denominator.abs() > floor)
    return numerator / denom


def build_features(panel: PricePanel, config: Any | None = None) -> FeatureSet:
    """Convenience wrapper: build the feature set for ``panel``."""
    engineer = FeatureEngineer.from_config(config) if config is not None else FeatureEngineer()
    return engineer.build(panel)
