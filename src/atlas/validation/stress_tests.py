"""Stress testing over historical crisis windows.

A strategy that performed well in one crisis has told you almost nothing: crises
differ in mechanism (a credit freeze, a liquidity shock, a rate repricing) and a
system tuned to one can fail badly in another. These tests are therefore
presented as a *panel* - the interesting question is whether behaviour is
consistent across windows, not whether one window looks good.

Each window reports return, drawdown, volatility, worst day, exposures, the
regime mix that prevailed, the risk controls that fired, and the asset and
strategy contributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestResult
from atlas.backtest.metrics import compute_metrics, max_drawdown
from atlas.config import StressPeriod, ValidationConfig
from atlas.data.loader import PricePanel
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["StressTestResult", "StressTester"]


@dataclass
class StressTestResult:
    """Results across all configured stress windows."""

    periods: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_contributions: dict[str, pd.Series] = field(default_factory=dict)
    strategy_contributions: dict[str, pd.Series] = field(default_factory=dict)
    risk_actions: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def worst_period(self) -> str | None:
        """Name of the window with the deepest drawdown."""
        if self.periods.empty or "max_drawdown" not in self.periods.columns:
            return None
        return str(self.periods.loc[self.periods["max_drawdown"].idxmax(), "name"])

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        if self.periods.empty:
            return {"n_periods": 0, "skipped": len(self.skipped)}
        return {
            "n_periods": len(self.periods),
            "skipped": len(self.skipped),
            "worst_period": self.worst_period,
            "worst_drawdown": round(float(self.periods["max_drawdown"].max()), 4),
            "median_return": round(float(self.periods["total_return"].median()), 4),
            "n_positive": int((self.periods["total_return"] > 0).sum()),
        }


class StressTester:
    """Evaluate a completed backtest over historical stress windows."""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def run(
        self,
        result: BacktestResult,
        *,
        panel: PricePanel | None = None,
        periods: list[StressPeriod] | None = None,
    ) -> StressTestResult:
        """Evaluate ``result`` over every configured stress window.

        Windows falling entirely outside the backtest's date range are skipped
        and reported, never silently dropped.
        """
        windows = periods if periods is not None else self.config.stress_tests.periods
        returns = result.returns
        rows: list[dict[str, Any]] = []
        asset_contributions: dict[str, pd.Series] = {}
        strategy_contributions: dict[str, pd.Series] = {}
        risk_actions: dict[str, pd.DataFrame] = {}
        skipped: list[str] = []

        if returns.empty:
            return StressTestResult(skipped=[p.name for p in windows])

        for period in windows:
            start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
            window_returns = returns.loc[(returns.index >= start) & (returns.index <= end)]
            if len(window_returns) < 5:
                skipped.append(
                    f"{period.name} ({start.date()}..{end.date()}): "
                    f"only {len(window_returns)} trading day(s) inside the backtest range"
                )
                continue

            equity = (1.0 + window_returns).cumprod()
            metrics = compute_metrics(
                window_returns,
                equity=equity,
                risk_free_rate=self.config.benchmarks.risk_free_rate,
                label=period.name,
            )

            snapshots = result.snapshots
            in_window = (
                (snapshots.index >= start) & (snapshots.index <= end)
                if not snapshots.empty
                else None
            )
            gross = net = cash = turnover = float("nan")
            if in_window is not None and in_window.any():
                block = snapshots.loc[in_window]
                gross = float(block["gross_exposure"].mean())
                net = float(block["net_exposure"].mean())
                cash = float(block.get("cash_weight", pd.Series(dtype=float)).mean())
                turnover = float(block.get("turnover", pd.Series(dtype=float)).sum())

            regime_mix = ""
            if result.regimes is not None:
                distribution = result.regimes.slice(start, end).distribution()
                if len(distribution):
                    regime_mix = ", ".join(
                        f"{k} {v:.0%}" for k, v in distribution.sort_values(ascending=False).items()
                    )

            controls = ""
            if not result.risk_events.empty and "timestamp" in result.risk_events.columns:
                stamps = pd.DatetimeIndex(result.risk_events["timestamp"])
                mask = (stamps >= start) & (stamps <= end)
                events = result.risk_events.loc[mask]
                if not events.empty:
                    risk_actions[period.name] = events
                    controls = ", ".join(
                        f"{k} x{v}" for k, v in events["control"].value_counts().head(4).items()
                    )

            rows.append(
                {
                    "name": period.name,
                    "start": start.date(),
                    "end": end.date(),
                    "trading_days": len(window_returns),
                    "total_return": metrics.values["total_return"],
                    "annualized_volatility": metrics.values["annualized_volatility"],
                    "max_drawdown": max_drawdown(equity),
                    "worst_day": metrics.values["worst_day"],
                    "best_day": metrics.values["best_day"],
                    "sharpe_ratio": metrics.sharpe,
                    "avg_gross_exposure": gross,
                    "avg_net_exposure": net,
                    "avg_cash_weight": cash,
                    "turnover": turnover,
                    "regime_mix": regime_mix,
                    "risk_controls_fired": controls,
                }
            )

            asset_contributions[period.name] = self._asset_contributions(
                result, panel, start, end
            )
            strategy_contributions[period.name] = self._strategy_contributions(result, start, end)

        stress = StressTestResult(
            periods=pd.DataFrame(rows),
            asset_contributions=asset_contributions,
            strategy_contributions=strategy_contributions,
            risk_actions=risk_actions,
            skipped=skipped,
        )
        log.info("stress tests complete", extra={"context": stress.summary()})
        for note in skipped:
            log.info("stress window skipped", extra={"context": {"detail": note}})
        return stress

    # -- contributions --------------------------------------------------------

    @staticmethod
    def _asset_contributions(
        result: BacktestResult,
        panel: PricePanel | None,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        """Return contribution per asset over the window.

        Contribution on day *t* is the weight held going into the day multiplied
        by that day's return, so nothing is attributed to a position that was not
        yet held.
        """
        if panel is None or result.weights.empty:
            return pd.Series(dtype="float64")
        asset_returns = panel.returns("adj_close")
        weights = result.weights
        mask = (weights.index >= start) & (weights.index <= end)
        window_weights = weights.loc[mask]
        if window_weights.empty:
            return pd.Series(dtype="float64")
        aligned = asset_returns.reindex(
            index=window_weights.index, columns=window_weights.columns
        )
        contributions = window_weights.shift(1).fillna(0.0) * aligned.fillna(0.0)
        return contributions.sum().sort_values()

    @staticmethod
    def _strategy_contributions(
        result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Average share of the combined signal held by each strategy."""
        if result.strategy_contributions.empty:
            return pd.Series(dtype="float64")
        frame = result.strategy_contributions
        mask = (frame.index >= start) & (frame.index <= end)
        window = frame.loc[mask]
        return window.mean() if not window.empty else pd.Series(dtype="float64")


def block_bootstrap(
    returns: pd.Series,
    *,
    n_simulations: int = 1000,
    block_size: int = 21,
    horizon: int = 252,
    seed: int = 7,
    stationary: bool = False,
) -> pd.DataFrame:
    """Block-bootstrap a return series.

    Resampling individual days independently destroys the serial dependence that
    drives drawdowns - it would make almost any strategy look far safer than it
    is. Sampling contiguous *blocks* preserves short-run autocorrelation and
    volatility clustering.

    Parameters
    ----------
    returns:
        Historical daily returns to resample.
    n_simulations:
        Number of simulated paths.
    block_size:
        Expected block length in trading days.
    horizon:
        Length of each simulated path in trading days.
    seed:
        Random seed; results are fully reproducible.
    stationary:
        Use Politis-Romano stationary bootstrap (geometric block lengths)
        instead of fixed-length blocks. Geometric lengths remove the artefacts
        that a fixed block boundary can introduce.

    Returns
    -------
    pandas.DataFrame
        One column per simulation, ``horizon`` rows of simulated daily returns.

    Notes
    -----
    This is a **statistical stress test, not a forecast**. It answers "given a
    return distribution with this shape and this dependence structure, how wide
    is the range of outcomes?" - it does not predict what will happen next.
    """
    clean = pd.Series(returns).astype("float64").dropna()
    if len(clean) < max(block_size * 2, 30):
        raise ValueError(
            f"need at least {max(block_size * 2, 30)} observations to bootstrap, got {len(clean)}"
        )

    values = clean.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    paths = np.empty((horizon, n_simulations), dtype="float64")

    for sim in range(n_simulations):
        path: list[float] = []
        while len(path) < horizon:
            start = int(rng.integers(0, n))
            length = (
                int(rng.geometric(1.0 / block_size)) if stationary else block_size
            )
            length = max(1, min(length, horizon - len(path)))
            # Wrap around the end of the sample so every observation is equally
            # likely to appear, including those near the end.
            indices = [(start + k) % n for k in range(length)]
            path.extend(values[i] for i in indices)
        paths[:, sim] = path[:horizon]

    return pd.DataFrame(paths, columns=[f"sim_{i}" for i in range(n_simulations)])


def bootstrap_statistics(
    paths: pd.DataFrame, percentiles: list[float] | None = None
) -> pd.DataFrame:
    """Summarise bootstrapped paths.

    Reports the distribution of annualised return, maximum drawdown, ending
    wealth, Sharpe ratio and the worst rolling one-year outcome.
    """
    from atlas.backtest.metrics import annualized_return, sharpe_ratio

    pct = percentiles or [0.05, 0.25, 0.50, 0.75, 0.95]
    rows = []
    for column in paths.columns:
        series = paths[column]
        equity = (1.0 + series).cumprod()
        drawdown = float(-((equity / equity.cummax()) - 1.0).min())
        rows.append(
            {
                "annualized_return": annualized_return(series),
                "max_drawdown": drawdown,
                "ending_wealth": float(equity.iloc[-1]),
                "sharpe_ratio": sharpe_ratio(series),
                "worst_252d_return": float(
                    (1.0 + series).rolling(min(252, len(series))).apply(np.prod, raw=True).min() - 1.0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary = frame.quantile(pct).T
    summary.columns = [f"p{int(p * 100)}" for p in pct]
    summary.insert(0, "mean", frame.mean())
    summary.insert(1, "std", frame.std(ddof=1))
    return summary
