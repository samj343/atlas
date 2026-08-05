"""Benchmark portfolios.

A strategy result means nothing without a reference. Atlas computes eight:

1. **SPY buy-and-hold** - the honest hurdle for any US-centric system.
2. **60/40 SPY/IEF** - the classic balanced portfolio, rebalanced monthly.
3. **Equal-weight universe** - what naive diversification alone achieves.
4-6. **Each strategy standalone** - trend, mean reversion, cross-sectional
   momentum, each run through the same portfolio, cost and risk machinery.
7. **Equal-weight strategy ensemble** - the ensemble without regime weighting.
8. **Regime-aware ensemble** - the full Atlas system.

Benchmarks 4-8 all use identical costs, constraints and execution timing, so a
difference between them is attributable to the signal rather than to the
plumbing. Passive benchmarks (1-3) are charged rebalancing costs where they
rebalance, and are stated to be gross of any fund fee.

Every benchmark is evaluated over the same date range. Where an asset lacks
history for part of the range, the shortfall is reported explicitly rather than
being silently back-filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from atlas.backtest.engine import BacktestEngine, BacktestResult
from atlas.backtest.metrics import PerformanceMetrics, comparison_table, compute_metrics
from atlas.config import AtlasConfig, RebalanceFrequency
from atlas.data.features import FeatureEngineer, FeatureSet
from atlas.data.loader import PricePanel
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger
from atlas.strategies.base import build_strategies

log = get_logger(__name__)

__all__ = ["BenchmarkResult", "BenchmarkSuite"]


@dataclass
class BenchmarkResult:
    """Return series and metrics for every benchmark."""

    returns: dict[str, pd.Series] = field(default_factory=dict)
    metrics: dict[str, PerformanceMetrics] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def equity_curves(self, initial: float = 1.0) -> pd.DataFrame:
        """Compounded equity curves for every benchmark, aligned on one index."""
        curves = {
            name: initial * (1.0 + series).cumprod() for name, series in self.returns.items()
        }
        return pd.DataFrame(curves).sort_index()

    def table(self, keys: list[str] | None = None) -> pd.DataFrame:
        """Side-by-side comparison of the headline metrics."""
        return comparison_table(self.metrics, keys)

    def correlation(self) -> pd.DataFrame:
        """Correlation matrix of the benchmark return series."""
        frame = pd.DataFrame(self.returns).dropna(how="all")
        return frame.corr() if not frame.empty else pd.DataFrame()

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        return {
            "n_benchmarks": len(self.returns),
            "names": sorted(self.returns),
            "start": str(self.date_range[0].date()) if self.date_range else None,
            "end": str(self.date_range[1].date()) if self.date_range else None,
            "n_notes": len(self.notes),
        }


class BenchmarkSuite:
    """Build and evaluate the benchmark set."""

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.settings = config.validation.benchmarks

    # -- main entry point -----------------------------------------------------

    def run(
        self,
        panel: PricePanel,
        *,
        atlas_result: BacktestResult | None = None,
        features: FeatureSet | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        include_strategy_variants: bool = True,
    ) -> BenchmarkResult:
        """Compute every benchmark over the same window as ``atlas_result``.

        Parameters
        ----------
        panel:
            Price history.
        atlas_result:
            The full-system result, included in the comparison when supplied and
            used to define the common evaluation window.
        include_strategy_variants:
            Run the single-strategy and equal-weight-ensemble variants. These are
            full backtests and dominate the runtime.
        """
        features = features or FeatureEngineer.from_config(self.config).build(panel)

        # The comparison window is whatever Atlas actually traded, so nothing
        # gets an unfair head start.
        if atlas_result is not None and not atlas_result.returns.empty:
            window_start = atlas_result.returns.index.min()
            window_end = atlas_result.returns.index.max()
        else:
            window_start = pd.Timestamp(start) if start else panel.dates.min()
            window_end = pd.Timestamp(end) if end else panel.dates.max()

        returns: dict[str, pd.Series] = {}
        notes: list[str] = []

        # --- passive benchmarks ---
        for symbol in self.settings.buy_and_hold:
            series, note = self._buy_and_hold(panel, symbol, window_start, window_end)
            if series is not None:
                returns[f"Buy & hold {symbol}"] = series
            if note:
                notes.append(note)

        sixty_forty, note = self._fixed_weight(
            panel, self.settings.sixty_forty, window_start, window_end, "60/40"
        )
        if sixty_forty is not None:
            returns["60/40 SPY/IEF"] = sixty_forty
        if note:
            notes.append(note)

        if self.settings.equal_weight_universe:
            weights = {s: 1.0 / len(self.config.symbols) for s in self.config.symbols}
            equal, note = self._fixed_weight(
                panel, weights, window_start, window_end, "equal-weight universe"
            )
            if equal is not None:
                returns["Equal-weight ETF portfolio"] = equal
            if note:
                notes.append(note)

        # --- strategy variants ---
        if include_strategy_variants:
            for name, series, note in self._strategy_variants(
                panel, features, window_start, window_end
            ):
                if series is not None:
                    returns[name] = series
                if note:
                    notes.append(note)

        # --- the full system ---
        if atlas_result is not None and not atlas_result.returns.empty:
            returns["Atlas regime-aware ensemble"] = atlas_result.returns

        risk_free = self.settings.risk_free_rate
        metrics = {
            name: compute_metrics(series, risk_free_rate=risk_free, label=name)
            for name, series in returns.items()
        }

        result = BenchmarkResult(
            returns=returns,
            metrics=metrics,
            notes=notes,
            date_range=(window_start, window_end),
        )
        log.info("benchmarks complete", extra={"context": result.summary()})
        for note in notes:
            log.info("benchmark note", extra={"context": {"detail": note}})
        return result

    # -- passive benchmarks ---------------------------------------------------

    def _buy_and_hold(
        self, panel: PricePanel, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.Series | None, str]:
        """Total-return buy-and-hold of one symbol."""
        symbol = symbol.upper()
        if symbol not in panel.symbols:
            return None, f"buy-and-hold {symbol}: symbol is not in the loaded universe"
        prices = panel.adj_close[symbol].loc[start:end].dropna()
        if len(prices) < 20:
            return None, f"buy-and-hold {symbol}: only {len(prices)} observations in the window"
        series = prices.pct_change(fill_method=None).dropna()
        note = ""
        if prices.index.min() > start + pd.Timedelta(days=5):
            note = (
                f"buy-and-hold {symbol}: history starts {prices.index.min().date()}, after the "
                f"comparison window opens on {start.date()}"
            )
        return series.rename(f"buy_and_hold_{symbol}"), note

    def _fixed_weight(
        self,
        panel: PricePanel,
        weights: dict[str, float],
        start: pd.Timestamp,
        end: pd.Timestamp,
        label: str,
    ) -> tuple[pd.Series | None, str]:
        """Fixed-weight portfolio, rebalanced on the configured schedule.

        Weights drift between rebalances (the portfolio is *not* continuously
        rebalanced, which would be both unrealistic and free), and a rebalance
        charges the same round-trip cost assumption the strategy pays.
        """
        available = [s for s in weights if s.upper() in panel.symbols]
        missing = [s for s in weights if s.upper() not in panel.symbols]
        if not available:
            return None, f"{label}: none of {sorted(weights)} are in the universe"

        prices = panel.adj_close[[s.upper() for s in available]].loc[start:end]
        prices = prices.dropna(how="all")
        if len(prices) < 20:
            return None, f"{label}: only {len(prices)} observations in the window"

        note = ""
        if missing:
            note = (
                f"{label}: {sorted(missing)} unavailable; the remaining weights were renormalised"
            )
        first_valid = prices.apply(lambda c: c.first_valid_index()).max()
        if pd.notna(first_valid) and first_valid > start + pd.Timedelta(days=5):
            extra = f"{label}: full composition only available from {pd.Timestamp(first_valid).date()}"
            note = f"{note}; {extra}" if note else extra
            prices = prices.loc[first_valid:]

        target = pd.Series({s.upper(): weights[s] for s in available}, dtype="float64")
        target = target / target.sum()

        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        rebalance_dates = set(self._rebalance_dates(pd.DatetimeIndex(prices.index)))

        # Simulate drifting weights explicitly rather than assuming a constant mix.
        cost_rate = (
            self.config.costs.half_spread_bps + self.config.costs.slippage_bps
        ) / 10_000.0
        current = target.copy()
        portfolio_returns: list[float] = []
        index: list[pd.Timestamp] = []

        for date in prices.index[1:]:
            day_return = float((current * asset_returns.loc[date]).sum())
            # Let the weights drift with the assets' returns.
            grown = current * (1.0 + asset_returns.loc[date])
            total = float(grown.sum())
            current = grown / total if abs(total) > 1e-12 else target.copy()

            if date in rebalance_dates:
                turnover = float((target - current).abs().sum())
                day_return -= turnover * cost_rate
                current = target.copy()

            portfolio_returns.append(day_return)
            index.append(date)

        return pd.Series(portfolio_returns, index=pd.DatetimeIndex(index), name=label), note

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Rebalance dates for the passive benchmarks."""
        frequency = self.settings.rebalance_frequency
        if frequency is RebalanceFrequency.DAILY:
            return list(dates)
        period = "W" if frequency is RebalanceFrequency.WEEKLY else "M"
        frame = pd.DataFrame(index=dates)
        frame["period"] = dates.to_period(period)
        return [group.index[-1] for _, group in frame.groupby("period", sort=True)]

    # -- strategy variants ----------------------------------------------------

    def _strategy_variants(
        self,
        panel: PricePanel,
        features: FeatureSet,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[tuple[str, pd.Series | None, str]]:
        """Run each strategy alone, plus the equal-weight ensemble."""
        out: list[tuple[str, pd.Series | None, str]] = []
        labels = {
            "trend": "Trend only",
            "mean_reversion": "Mean reversion only",
            "cross_sectional_momentum": "Cross-sectional momentum only",
        }

        for name, label in labels.items():
            config = self.config.model_copy(deep=True)
            block = config.strategies.strategies
            for other in labels:
                getattr(block, other).enabled = other == name
            if not getattr(block, name).enabled:
                out.append((label, None, f"{label}: strategy is disabled in configuration"))
                continue
            config.strategies.ensemble.mode = "fixed"
            config.strategies.regime.enabled = False
            out.append(self._run_variant(config, panel, features, start, end, label))

        # Equal-weight ensemble: all strategies, no regime tilt.
        equal_config = self.config.model_copy(deep=True)
        equal_config.strategies.ensemble.mode = "equal"
        equal_config.strategies.regime.enabled = False
        out.append(
            self._run_variant(
                equal_config, panel, features, start, end, "Equal-weight strategy ensemble"
            )
        )
        return out

    def _run_variant(
        self,
        config: AtlasConfig,
        panel: PricePanel,
        features: FeatureSet,
        start: pd.Timestamp,
        end: pd.Timestamp,
        label: str,
    ) -> tuple[str, pd.Series | None, str]:
        """Run one strategy variant through the full engine."""
        try:
            strategies = build_strategies(config)
            result = BacktestEngine(config, strategies=strategies).run(
                panel, features=features, start=start, end=end
            )
            return label, result.returns.rename(label), ""
        except (InsufficientDataError, ValueError) as exc:
            log.warning(
                "benchmark variant failed",
                extra={"context": {"variant": label, "error": str(exc)}},
            )
            return label, None, f"{label}: {exc}"
