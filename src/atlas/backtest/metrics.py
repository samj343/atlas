"""Performance and risk metrics.

Every metric here states its convention explicitly, because most disagreements
about a Sharpe ratio are really disagreements about conventions:

* Returns are **simple daily** returns unless stated otherwise.
* Annualisation uses **252 trading days**.
* The Sharpe ratio is computed on **excess** returns over the stated risk-free
  rate, which is de-annualised geometrically before subtraction, and is reported
  alongside that rate. :meth:`PerformanceMetrics.describe_conventions` prints the
  assumptions with the numbers.
* Drawdowns are measured on the **equity curve**, not on returns, so a drawdown
  survives a period of missing observations.
* Value at Risk and expected shortfall are **historical** (empirical quantiles),
  not parametric, and are reported as positive loss magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "TRADING_DAYS",
    "PerformanceMetrics",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "compute_metrics",
    "drawdown_series",
    "expected_shortfall",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
]

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _clean(returns: pd.Series) -> pd.Series:
    """Drop non-finite observations from a return series."""
    if returns is None or len(returns) == 0:
        return pd.Series(dtype="float64")
    series = pd.Series(returns).astype("float64")
    return series[np.isfinite(series)]


def total_return(returns: pd.Series) -> float:
    """Cumulative simple return over the whole series."""
    clean = _clean(returns)
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric (compound) annualised return, i.e. CAGR."""
    clean = _clean(returns)
    if clean.empty:
        return 0.0
    growth = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0:
        return 0.0
    if growth <= 0:
        # Total loss: the geometric return is -100% however long it took.
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    clean = _clean(returns)
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio of excess returns.

    Parameters
    ----------
    returns:
        Simple periodic returns.
    risk_free_rate:
        **Annualised** risk-free rate; de-annualised geometrically before
        subtraction so that compounding is handled consistently.
    """
    clean = _clean(returns)
    if len(clean) < 2:
        return 0.0
    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - periodic_rf
    std = excess.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio: excess return over downside deviation.

    Downside deviation uses the *full* sample in the denominator (returns above
    the target contribute zero), which is the standard definition and avoids
    inflating the ratio when few observations fall below the target.
    """
    clean = _clean(returns)
    if len(clean) < 2:
        return 0.0
    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - periodic_rf
    downside = np.minimum(excess - target, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    if downside_dev <= 1e-12:
        # No observation fell below the target, so the ratio has no denominator.
        # Returning 0.0 here would rank a series that never lost *worst*, which
        # is the opposite of the truth; NaN says "undefined" and reports render
        # it as n/a. A non-positive mean excess with no downside is a flat
        # series, for which zero is the correct answer.
        return float("nan") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound a return series into an equity curve."""
    clean = _clean(returns)
    if clean.empty:
        return pd.Series(dtype="float64")
    return initial * (1.0 + clean).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a negative fraction."""
    if equity is None or len(equity) == 0:
        return pd.Series(dtype="float64")
    curve = pd.Series(equity).astype("float64").dropna()
    if curve.empty:
        return pd.Series(dtype="float64")
    peak = curve.cummax()
    return (curve / peak) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a positive fraction."""
    dd = drawdown_series(equity)
    return float(-dd.min()) if len(dd) else 0.0


def calmar_ratio(returns: pd.Series, equity: pd.Series | None = None) -> float:
    """Annualised return divided by maximum drawdown."""
    curve = equity if equity is not None else equity_from_returns(returns)
    mdd = max_drawdown(curve)
    if mdd <= 1e-9:
        return 0.0
    return float(annualized_return(returns) / mdd)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk as a positive loss magnitude.

    ``confidence=0.95`` returns the loss exceeded on 5% of days.
    """
    clean = _clean(returns)
    if clean.empty:
        return 0.0
    quantile = float(np.quantile(clean, 1.0 - confidence))
    return float(max(0.0, -quantile))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical expected shortfall (average loss beyond the VaR threshold)."""
    clean = _clean(returns)
    if clean.empty:
        return 0.0
    threshold = float(np.quantile(clean, 1.0 - confidence))
    tail = clean[clean <= threshold]
    if tail.empty:
        return float(max(0.0, -threshold))
    return float(max(0.0, -tail.mean()))


def drawdown_stats(equity: pd.Series) -> dict[str, float]:
    """Depth and duration statistics for every drawdown episode."""
    dd = drawdown_series(equity)
    if dd.empty:
        return {
            "max_drawdown": 0.0,
            "average_drawdown": 0.0,
            "max_drawdown_duration_days": 0.0,
            "average_drawdown_duration_days": 0.0,
            "max_recovery_duration_days": 0.0,
            "n_drawdown_episodes": 0.0,
            "time_in_drawdown_pct": 0.0,
        }

    underwater = dd < -1e-12
    episodes: list[dict[str, float]] = []
    start_idx: int | None = None
    trough = 0.0
    trough_idx: int | None = None

    values = dd.to_numpy()
    for i, is_under in enumerate(underwater.to_numpy()):
        if is_under and start_idx is None:
            start_idx, trough, trough_idx = i, values[i], i
        elif is_under:
            if values[i] < trough:
                trough, trough_idx = values[i], i
        elif start_idx is not None:
            episodes.append(
                {
                    "depth": -trough,
                    "duration": float(i - start_idx),
                    "recovery": float(i - (trough_idx if trough_idx is not None else start_idx)),
                }
            )
            start_idx, trough, trough_idx = None, 0.0, None
    if start_idx is not None:  # still under water at the end of the sample
        episodes.append(
            {
                "depth": -trough,
                "duration": float(len(values) - start_idx),
                "recovery": float("nan"),
            }
        )

    if not episodes:
        return {
            "max_drawdown": float(-dd.min()),
            "average_drawdown": 0.0,
            "max_drawdown_duration_days": 0.0,
            "average_drawdown_duration_days": 0.0,
            "max_recovery_duration_days": 0.0,
            "n_drawdown_episodes": 0.0,
            "time_in_drawdown_pct": 0.0,
        }

    depths = [e["depth"] for e in episodes]
    durations = [e["duration"] for e in episodes]
    recoveries = [e["recovery"] for e in episodes if np.isfinite(e["recovery"])]
    return {
        "max_drawdown": float(max(depths)),
        "average_drawdown": float(np.mean(depths)),
        "max_drawdown_duration_days": float(max(durations)),
        "average_drawdown_duration_days": float(np.mean(durations)),
        "max_recovery_duration_days": float(max(recoveries)) if recoveries else 0.0,
        "n_drawdown_episodes": float(len(episodes)),
        "time_in_drawdown_pct": float(underwater.mean()),
    }


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Compound daily returns into calendar-month returns."""
    clean = _clean(returns)
    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64")
    return clean.resample("ME").apply(lambda r: float((1.0 + r).prod() - 1.0))


def annual_returns(returns: pd.Series) -> pd.Series:
    """Compound daily returns into calendar-year returns."""
    clean = _clean(returns)
    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64")
    return clean.resample("YE").apply(lambda r: float((1.0 + r).prod() - 1.0))


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month table of returns, convenient for a heatmap."""
    monthly = monthly_returns(returns)
    if monthly.empty:
        return pd.DataFrame()
    frame = monthly.to_frame("ret")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    table = frame.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
    table.columns = [
        pd.Timestamp(2000, int(m), 1).strftime("%b") for m in table.columns
    ]
    return table


# ---------------------------------------------------------------------------
# Aggregate container
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """A full performance report for one return series.

    Build with :func:`compute_metrics`. Access individual numbers as attributes
    of :attr:`values`, or use :meth:`to_series` / :meth:`to_frame` for display.
    """

    values: dict[str, float] = field(default_factory=dict)
    risk_free_rate: float = 0.0
    periods_per_year: int = TRADING_DAYS
    label: str = "strategy"

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = float("nan")) -> float:
        """Return a metric by name."""
        return self.values.get(key, default)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio."""
        return self.values.get("sharpe_ratio", 0.0)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a positive fraction."""
        return self.values.get("max_drawdown", 0.0)

    @property
    def cagr(self) -> float:
        """Compound annual growth rate."""
        return self.values.get("cagr", 0.0)

    def to_series(self) -> pd.Series:
        """Metrics as a Series."""
        return pd.Series(self.values, name=self.label)

    def to_frame(self) -> pd.DataFrame:
        """Metrics as a single-column DataFrame."""
        return self.to_series().to_frame()

    def describe_conventions(self) -> dict[str, str]:
        """The conventions behind these numbers, for inclusion in reports."""
        return {
            "annualization": f"{self.periods_per_year} trading days per year",
            "risk_free_rate": (
                f"{self.risk_free_rate:.2%} annualised, de-annualised geometrically "
                "and subtracted from each period's return before computing Sharpe and Sortino"
            ),
            "returns": "simple (arithmetic) daily returns, net of all modelled transaction costs",
            "drawdown": "measured on the equity curve from its running peak",
            "var_es": "historical (empirical) at 95% confidence, reported as positive losses",
        }

    def render(self) -> str:
        """Human-readable text block."""
        lines = [f"Performance: {self.label}", "=" * 52]
        formats = {
            "total_return": "{:.2%}", "cagr": "{:.2%}", "annualized_return": "{:.2%}",
            "annualized_volatility": "{:.2%}", "sharpe_ratio": "{:.2f}",
            "sortino_ratio": "{:.2f}", "calmar_ratio": "{:.2f}", "max_drawdown": "{:.2%}",
            "average_drawdown": "{:.2%}", "best_day": "{:.2%}", "worst_day": "{:.2%}",
            "best_month": "{:.2%}", "worst_month": "{:.2%}", "monthly_win_rate": "{:.1%}",
            "value_at_risk_95": "{:.2%}", "expected_shortfall_95": "{:.2%}",
            "skewness": "{:.2f}", "kurtosis": "{:.2f}",
        }
        for key, value in self.values.items():
            fmt = formats.get(key, "{:.4f}")
            try:
                rendered = fmt.format(value)
            except (TypeError, ValueError):
                rendered = str(value)
            lines.append(f"  {key:<32} {rendered:>12}")
        lines.append("-" * 52)
        for key, value in self.describe_conventions().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def compute_metrics(
    returns: pd.Series,
    *,
    equity: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    label: str = "strategy",
    turnover: pd.Series | None = None,
    costs: float | None = None,
    gross_exposure: pd.Series | None = None,
    net_exposure: pd.Series | None = None,
    cash_weight: pd.Series | None = None,
    n_trades: int | None = None,
    trade_values: pd.Series | None = None,
    holding_periods: pd.Series | None = None,
) -> PerformanceMetrics:
    """Compute the full metric set for a return series.

    Parameters
    ----------
    returns:
        Simple periodic (normally daily) returns, net of costs.
    equity:
        The equity curve. Reconstructed from ``returns`` when omitted.
    risk_free_rate:
        Annualised risk-free rate used for Sharpe and Sortino.
    turnover, costs, gross_exposure, net_exposure, cash_weight, n_trades,
    trade_values, holding_periods:
        Optional portfolio statistics; the corresponding metrics are only
        included when the input is supplied, so a metric is never invented.
    """
    clean = _clean(returns)
    curve = equity if equity is not None else equity_from_returns(clean)

    values: dict[str, float] = {}
    values["n_observations"] = float(len(clean))
    values["start_date_ordinal"] = float(clean.index[0].toordinal()) if len(clean) and isinstance(
        clean.index, pd.DatetimeIndex
    ) else float("nan")
    values["years"] = float(len(clean) / periods_per_year)

    # --- return ---
    values["total_return"] = total_return(clean)
    values["cagr"] = annualized_return(clean, periods_per_year)
    values["annualized_return"] = values["cagr"]
    values["arithmetic_annual_return"] = float(clean.mean() * periods_per_year) if len(clean) else 0.0

    # --- risk ---
    values["annualized_volatility"] = annualized_volatility(clean, periods_per_year)
    values["downside_volatility"] = (
        float(np.sqrt(np.mean(np.minimum(clean, 0.0) ** 2)) * np.sqrt(periods_per_year))
        if len(clean)
        else 0.0
    )
    values["sharpe_ratio"] = sharpe_ratio(clean, risk_free_rate, periods_per_year)
    values["sortino_ratio"] = sortino_ratio(clean, risk_free_rate, periods_per_year)
    values["calmar_ratio"] = calmar_ratio(clean, curve)

    # --- drawdown ---
    values.update(drawdown_stats(curve))

    # --- distribution ---
    if len(clean):
        values["best_day"] = float(clean.max())
        values["worst_day"] = float(clean.min())
        values["positive_days_pct"] = float((clean > 0).mean())
    else:
        values["best_day"] = values["worst_day"] = values["positive_days_pct"] = 0.0

    monthly = monthly_returns(clean)
    if len(monthly):
        values["best_month"] = float(monthly.max())
        values["worst_month"] = float(monthly.min())
        values["monthly_win_rate"] = float((monthly > 0).mean())
        values["n_months"] = float(len(monthly))
    else:
        values["best_month"] = values["worst_month"] = values["monthly_win_rate"] = 0.0
        values["n_months"] = 0.0

    values["value_at_risk_95"] = value_at_risk(clean, 0.95)
    values["expected_shortfall_95"] = expected_shortfall(clean, 0.95)
    values["value_at_risk_99"] = value_at_risk(clean, 0.99)
    values["expected_shortfall_99"] = expected_shortfall(clean, 0.99)

    if len(clean) >= 3:
        values["skewness"] = float(stats.skew(clean, bias=False))
        values["kurtosis"] = float(stats.kurtosis(clean, fisher=True, bias=False))
    else:
        values["skewness"] = values["kurtosis"] = 0.0

    # --- portfolio statistics (only when supplied) ---
    if turnover is not None and len(turnover):
        clean_turnover = _clean(turnover)
        values["average_daily_turnover"] = float(clean_turnover.mean())
        values["annual_turnover"] = float(clean_turnover.sum() / max(values["years"], 1e-9))
    if costs is not None:
        values["total_transaction_costs"] = float(costs)
    if gross_exposure is not None and len(gross_exposure):
        values["average_gross_exposure"] = float(_clean(gross_exposure).mean())
    if net_exposure is not None and len(net_exposure):
        values["average_net_exposure"] = float(_clean(net_exposure).mean())
    if cash_weight is not None and len(cash_weight):
        values["average_cash_weight"] = float(_clean(cash_weight).mean())
    if n_trades is not None:
        values["n_trades"] = float(n_trades)
    if trade_values is not None and len(trade_values):
        values["average_trade_value"] = float(_clean(trade_values).mean())
    if holding_periods is not None and len(holding_periods):
        values["average_holding_period_days"] = float(_clean(holding_periods).mean())

    return PerformanceMetrics(
        values=values,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        label=label,
    )


def rolling_sharpe(
    returns: pd.Series, window: int = 252, risk_free_rate: float = 0.0
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    clean = _clean(returns)
    if len(clean) < window:
        return pd.Series(dtype="float64")
    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = clean - periodic_rf
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std.replace(0.0, np.nan)) * np.sqrt(TRADING_DAYS)


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualised volatility."""
    clean = _clean(returns)
    return clean.rolling(window, min_periods=max(2, window // 2)).std(ddof=1) * np.sqrt(TRADING_DAYS)


def return_contribution(
    weights: pd.DataFrame, asset_returns: pd.DataFrame
) -> pd.Series:
    """Total return contribution per asset.

    Contribution on day *t* is ``w_{t-1} * r_t`` - the weight held going into the
    day multiplied by that day's return. Weights are lagged, so this cannot
    attribute a return to a position that was not yet held.
    """
    aligned_returns = asset_returns.reindex(index=weights.index, columns=weights.columns)
    contributions = weights.shift(1).fillna(0.0) * aligned_returns.fillna(0.0)
    return contributions.sum().sort_values(ascending=False)


def comparison_table(reports: dict[str, PerformanceMetrics], keys: list[str] | None = None) -> pd.DataFrame:
    """Side-by-side comparison of several metric sets."""
    if not reports:
        return pd.DataFrame()
    default_keys = [
        "total_return", "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
        "calmar_ratio", "max_drawdown", "worst_day", "monthly_win_rate",
        "value_at_risk_95", "skewness", "kurtosis",
    ]
    selected = keys or default_keys
    data = {
        name: {k: report.values.get(k, float("nan")) for k in selected}
        for name, report in reports.items()
    }
    return pd.DataFrame(data)
