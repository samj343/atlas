"""Plotly chart builders.

Every chart in Atlas comes from this module so styling, axis conventions and
hover behaviour stay consistent between the research reports and the dashboard.

Conventions enforced here:

* **No truncated value axes.** Percentage and return axes include zero; an
  equity-curve axis starts at zero or uses a log scale. A chart that starts the
  y-axis at 95% to make a 2% move look dramatic is misleading, so the helpers
  simply do not offer it.
* **Every axis is labelled with its unit**, and every trace has a hover template.
* **Titles state what and over what period.**
* Colours come from one palette, and drawdown/loss is consistently red.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas.backtest.metrics import (
    drawdown_series,
    monthly_return_table,
    rolling_sharpe,
    rolling_volatility,
)

__all__ = [
    "PALETTE",
    "apply_layout",
    "asset_correlation_chart",
    "drawdown_chart",
    "equity_curve_chart",
    "exposure_chart",
    "monthly_heatmap",
    "parameter_sensitivity_chart",
    "regime_timeline_chart",
    "return_distribution_chart",
    "risk_contribution_chart",
    "rolling_sharpe_chart",
    "rolling_volatility_chart",
    "strategy_correlation_chart",
    "turnover_cost_chart",
    "walk_forward_chart",
    "weights_chart",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito), plus semantic colours.
PALETTE: dict[str, Any] = {
    "series": [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#CC79A7",  # pink
        "#56B4E9",  # sky
        "#D55E00",  # vermillion
        "#F0E442",  # yellow
        "#666666",  # grey
    ],
    "primary": "#0072B2",
    "loss": "#D55E00",
    "gain": "#009E73",
    "neutral": "#666666",
    "grid": "rgba(128,128,128,0.20)",
}

_REGIME_COLORS = {
    "bull_low_vol": "#009E73",
    "bull_high_vol": "#F0E442",
    "bear_low_vol": "#E69F00",
    "bear_high_vol": "#D55E00",
    "unknown": "#BBBBBB",
}


def apply_layout(
    figure: go.Figure,
    *,
    title: str,
    xaxis_title: str = "Date",
    yaxis_title: str = "",
    height: int = 420,
    legend: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
    """Apply the house style to a figure."""
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        hovermode=hovermode,
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        font={"size": 12},
    )
    figure.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"])
    figure.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=True)
    return figure


def _date_range_label(index: pd.Index) -> str:
    """A ``2007-01-03 to 2024-12-31`` label for a chart title."""
    if len(index) == 0:
        return "no data"
    return f"{pd.Timestamp(index.min()).date()} to {pd.Timestamp(index.max()).date()}"


# ---------------------------------------------------------------------------
# Performance charts
# ---------------------------------------------------------------------------


def equity_curve_chart(
    curves: dict[str, pd.Series],
    *,
    title: str = "Portfolio versus benchmarks",
    log_scale: bool = False,
    normalize: bool = True,
) -> go.Figure:
    """Equity curves for the portfolio and its benchmarks.

    With ``normalize`` every series starts at 100, so relative growth is
    comparable regardless of starting capital. The y-axis is never truncated: it
    is either anchored at zero or shown on a log scale, where equal vertical
    distances mean equal percentage moves.
    """
    figure = go.Figure()
    for i, (name, series) in enumerate(curves.items()):
        clean = pd.Series(series).dropna()
        if clean.empty:
            continue
        values = 100.0 * clean / float(clean.iloc[0]) if normalize else clean
        figure.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(),
                name=name,
                mode="lines",
                line={"color": PALETTE["series"][i % len(PALETTE["series"])], "width": 2},
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:,.1f}}<extra></extra>",
            )
        )
    index = next((pd.Series(s).dropna().index for s in curves.values() if len(pd.Series(s).dropna())), pd.Index([]))
    unit = "Index (start = 100)" if normalize else "Equity"
    apply_layout(figure, title=f"{title} - {_date_range_label(index)}", yaxis_title=unit, height=460)
    if log_scale:
        figure.update_yaxes(type="log")
    else:
        figure.update_yaxes(rangemode="tozero")
    return figure


def drawdown_chart(
    equity: pd.Series | dict[str, pd.Series], *, title: str = "Underwater drawdown"
) -> go.Figure:
    """Underwater plot: drawdown from the running peak."""
    figure = go.Figure()
    series_map = equity if isinstance(equity, dict) else {"Portfolio": equity}
    index = pd.Index([])
    for i, (name, curve) in enumerate(series_map.items()):
        dd = drawdown_series(pd.Series(curve))
        if dd.empty:
            continue
        index = dd.index
        colour = PALETTE["loss"] if i == 0 else PALETTE["series"][i % len(PALETTE["series"])]
        figure.add_trace(
            go.Scatter(
                x=dd.index,
                y=100.0 * dd.to_numpy(),
                name=name,
                mode="lines",
                fill="tozeroy" if i == 0 else None,
                line={"color": colour, "width": 1.5},
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>",
            )
        )
    apply_layout(
        figure,
        title=f"{title} - {_date_range_label(index)}",
        yaxis_title="Drawdown (%)",
        height=340,
    )
    figure.update_yaxes(rangemode="tozero")
    return figure


def rolling_sharpe_chart(
    returns: pd.Series, *, window: int = 252, risk_free_rate: float = 0.0
) -> go.Figure:
    """Rolling annualised Sharpe ratio."""
    series = rolling_sharpe(returns, window=window, risk_free_rate=risk_free_rate).dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            name=f"{window}-day Sharpe",
            mode="lines",
            line={"color": PALETTE["primary"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe %{y:.2f}<extra></extra>",
        )
    )
    figure.add_hline(y=0.0, line={"color": PALETTE["neutral"], "dash": "dash", "width": 1})
    apply_layout(
        figure,
        title=(
            f"Rolling {window}-day Sharpe ratio (annualised, risk-free "
            f"{risk_free_rate:.2%}) - {_date_range_label(series.index)}"
        ),
        yaxis_title="Sharpe ratio (annualised)",
        height=340,
    )
    return figure


def rolling_volatility_chart(
    returns: pd.Series, *, window: int = 63, target: float | None = None
) -> go.Figure:
    """Rolling annualised volatility, with the target overlaid when supplied."""
    series = rolling_volatility(returns, window=window).dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series.index,
            y=100.0 * series.to_numpy(),
            name=f"{window}-day realised volatility",
            mode="lines",
            line={"color": PALETTE["primary"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        )
    )
    if target is not None:
        figure.add_hline(
            y=100.0 * target,
            line={"color": PALETTE["gain"], "dash": "dash", "width": 1.5},
            annotation_text=f"target {target:.1%}",
            annotation_position="top left",
        )
    apply_layout(
        figure,
        title=f"Rolling {window}-day annualised volatility - {_date_range_label(series.index)}",
        yaxis_title="Annualised volatility (%)",
        height=340,
    )
    figure.update_yaxes(rangemode="tozero")
    return figure


def monthly_heatmap(returns: pd.Series, *, title: str = "Monthly returns") -> go.Figure:
    """Year x month heatmap of returns, on a symmetric diverging scale."""
    table = monthly_return_table(returns)
    if table.empty:
        return apply_layout(go.Figure(), title=f"{title} - no data", xaxis_title="Month")

    values = 100.0 * table.to_numpy(dtype="float64")
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=list(table.columns),
            y=[str(y) for y in table.index],
            colorscale="RdYlGn",
            # Symmetric about zero so +3% and -3% are equally saturated.
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Return (%)"},
            hovertemplate="%{y} %{x}<br>%{z:.2f}%<extra></extra>",
        )
    )
    apply_layout(
        figure,
        title=f"{title} (%)",
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(300, 26 * len(table) + 140),
        legend=False,
        hovermode="closest",
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def return_distribution_chart(
    returns: pd.Series, *, benchmark: pd.Series | None = None, bins: int = 80
) -> go.Figure:
    """Histogram of daily returns with the mean and 5% VaR marked."""
    clean = pd.Series(returns).dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=100.0 * clean.to_numpy(),
            nbinsx=bins,
            name="Atlas",
            marker={"color": PALETTE["primary"], "opacity": 0.75},
            hovertemplate="Return %{x:.2f}%<br>Days %{y}<extra></extra>",
        )
    )
    if benchmark is not None:
        other = pd.Series(benchmark).dropna()
        figure.add_trace(
            go.Histogram(
                x=100.0 * other.to_numpy(),
                nbinsx=bins,
                name="Benchmark",
                marker={"color": PALETTE["series"][1], "opacity": 0.45},
                hovertemplate="Return %{x:.2f}%<br>Days %{y}<extra></extra>",
            )
        )
    if len(clean):
        var95 = float(np.quantile(clean, 0.05)) * 100.0
        figure.add_vline(
            x=var95,
            line={"color": PALETTE["loss"], "dash": "dash", "width": 1.5},
            annotation_text=f"5% VaR {var95:.2f}%",
        )
        figure.add_vline(
            x=float(clean.mean()) * 100.0,
            line={"color": PALETTE["neutral"], "dash": "dot", "width": 1},
            annotation_text="mean",
        )
    apply_layout(
        figure,
        title=f"Daily return distribution - {_date_range_label(clean.index)}",
        xaxis_title="Daily return (%)",
        yaxis_title="Number of days",
        height=360,
        hovermode="closest",
    )
    figure.update_layout(barmode="overlay")
    return figure


# ---------------------------------------------------------------------------
# Portfolio charts
# ---------------------------------------------------------------------------


def weights_chart(weights: pd.DataFrame, *, title: str = "Portfolio weights over time") -> go.Figure:
    """Stacked area chart of position weights."""
    figure = go.Figure()
    frame = weights.fillna(0.0)
    for i, column in enumerate(frame.columns):
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=100.0 * frame[column].to_numpy(),
                name=str(column),
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": PALETTE["series"][i % len(PALETTE["series"])]},
                hovertemplate=f"<b>{column}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>",
            )
        )
    apply_layout(
        figure,
        title=f"{title} - {_date_range_label(frame.index)}",
        yaxis_title="Weight (% of equity)",
        height=420,
    )
    return figure


def exposure_chart(snapshots: pd.DataFrame, *, title: str = "Portfolio exposure") -> go.Figure:
    """Gross exposure, net exposure and cash weight over time."""
    figure = go.Figure()
    series = {
        "Gross exposure": ("gross_exposure", PALETTE["series"][0]),
        "Net exposure": ("net_exposure", PALETTE["series"][2]),
        "Cash weight": ("cash_weight", PALETTE["neutral"]),
    }
    for name, (column, colour) in series.items():
        if column not in snapshots.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=snapshots.index,
                y=100.0 * snapshots[column].to_numpy(),
                name=name,
                mode="lines",
                line={"color": colour, "width": 1.8},
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )
    apply_layout(
        figure,
        title=f"{title} - {_date_range_label(snapshots.index)}",
        yaxis_title="Percent of equity (%)",
        height=340,
    )
    return figure


def turnover_cost_chart(
    snapshots: pd.DataFrame, fills: pd.DataFrame, *, title: str = "Turnover and transaction costs"
) -> go.Figure:
    """Daily turnover with cumulative transaction costs on a secondary axis."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if "turnover" in snapshots.columns:
        figure.add_trace(
            go.Bar(
                x=snapshots.index,
                y=100.0 * snapshots["turnover"].to_numpy(),
                name="Daily turnover",
                marker={"color": PALETTE["series"][0], "opacity": 0.7},
                hovertemplate="%{x|%Y-%m-%d}<br>Turnover %{y:.2f}%<extra></extra>",
            ),
            secondary_y=False,
        )
    if not fills.empty and "total_cost" in fills.columns:
        costs = (
            fills.assign(date=pd.DatetimeIndex(fills["timestamp"]).normalize())
            .groupby("date")["total_cost"]
            .sum()
            .cumsum()
        )
        figure.add_trace(
            go.Scatter(
                x=costs.index,
                y=costs.to_numpy(),
                name="Cumulative costs",
                mode="lines",
                line={"color": PALETTE["loss"], "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>Cumulative cost $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
    apply_layout(figure, title=f"{title} - {_date_range_label(snapshots.index)}", height=380)
    figure.update_yaxes(title_text="Daily turnover (% of equity)", secondary_y=False, rangemode="tozero")
    figure.update_yaxes(title_text="Cumulative cost ($)", secondary_y=True, rangemode="tozero")
    return figure


def risk_contribution_chart(
    contributions: pd.DataFrame, *, title: str = "Risk contribution by asset"
) -> go.Figure:
    """Stacked area chart of each asset's share of portfolio risk."""
    figure = go.Figure()
    frame = contributions.fillna(0.0)
    for i, column in enumerate(frame.columns):
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=100.0 * frame[column].to_numpy(),
                name=str(column),
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": PALETTE["series"][i % len(PALETTE["series"])]},
                hovertemplate=f"<b>{column}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )
    apply_layout(
        figure,
        title=f"{title} - {_date_range_label(frame.index)}",
        yaxis_title="Share of portfolio risk (%)",
        height=380,
    )
    return figure


def _correlation_figure(matrix: pd.DataFrame, title: str) -> go.Figure:
    """Shared implementation for the correlation heatmaps."""
    if matrix.empty:
        return apply_layout(go.Figure(), title=f"{title} - no data", xaxis_title="")
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale="RdBu",
            reversescale=True,
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "Correlation"},
            hovertemplate="%{y} / %{x}<br>corr %{z:.2f}<extra></extra>",
        )
    )
    apply_layout(
        figure,
        title=title,
        xaxis_title="",
        yaxis_title="",
        height=max(360, 30 * len(matrix) + 150),
        legend=False,
        hovermode="closest",
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def asset_correlation_chart(returns: pd.DataFrame, *, window_label: str = "") -> go.Figure:
    """Correlation matrix of asset returns."""
    suffix = f" ({window_label})" if window_label else ""
    return _correlation_figure(returns.corr(), f"Asset return correlation{suffix}")


def strategy_correlation_chart(strategy_returns: pd.DataFrame) -> go.Figure:
    """Correlation matrix of strategy return series."""
    return _correlation_figure(strategy_returns.corr(), "Strategy return correlation")


# ---------------------------------------------------------------------------
# Regime and research charts
# ---------------------------------------------------------------------------


def regime_timeline_chart(
    regime_labels: pd.Series,
    *,
    equity: pd.Series | None = None,
    title: str = "Market regime timeline",
) -> go.Figure:
    """Regime bands over time, optionally with the equity curve overlaid."""
    figure = go.Figure()
    labels = regime_labels.dropna()
    if labels.empty:
        return apply_layout(figure, title=f"{title} - no data")

    strings = labels.map(lambda v: getattr(v, "value", str(v)))
    block = (strings != strings.shift(1)).cumsum()
    for _, group in strings.groupby(block):
        name = group.iloc[0]
        figure.add_vrect(
            x0=group.index[0],
            x1=group.index[-1],
            fillcolor=_REGIME_COLORS.get(name, "#CCCCCC"),
            opacity=0.30,
            line_width=0,
            layer="below",
        )
    # One legend entry per regime, drawn off-screen so the bands stay clean.
    for name, colour in _REGIME_COLORS.items():
        if (strings == name).any():
            figure.add_trace(
                go.Scatter(
                    x=[labels.index[0]],
                    y=[None],
                    name=name,
                    mode="markers",
                    marker={"color": colour, "size": 12, "symbol": "square"},
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
    if equity is not None:
        curve = pd.Series(equity).dropna()
        if len(curve):
            figure.add_trace(
                go.Scatter(
                    x=curve.index,
                    y=100.0 * curve / float(curve.iloc[0]),
                    name="Portfolio",
                    mode="lines",
                    line={"color": "#111111", "width": 1.8},
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
                )
            )
    apply_layout(
        figure,
        title=f"{title} - {_date_range_label(labels.index)}",
        yaxis_title="Index (start = 100)" if equity is not None else "",
        height=380,
    )
    return figure


def parameter_sensitivity_chart(
    sweep: pd.DataFrame,
    parameter: str,
    *,
    metric: str = "sharpe",
    gross_metric: str = "gross_sharpe",
) -> go.Figure:
    """Performance across a parameter sweep, net and gross of costs."""
    data = sweep[sweep["parameter"] == parameter].sort_values("value")
    figure = go.Figure()
    if data.empty:
        return apply_layout(figure, title=f"Parameter sensitivity: {parameter} - no data")

    figure.add_trace(
        go.Scatter(
            x=data["value"],
            y=data[metric],
            name="Net of costs",
            mode="lines+markers",
            line={"color": PALETTE["primary"], "width": 2.5},
            marker={"size": 9},
            hovertemplate="%{x}<br>Sharpe %{y:.3f}<extra></extra>",
        )
    )
    if gross_metric in data.columns and data[gross_metric].notna().any():
        figure.add_trace(
            go.Scatter(
                x=data["value"],
                y=data[gross_metric],
                name="Gross of costs",
                mode="lines+markers",
                line={"color": PALETTE["series"][1], "width": 2, "dash": "dash"},
                marker={"size": 7},
                hovertemplate="%{x}<br>Gross Sharpe %{y:.3f}<extra></extra>",
            )
        )
    median = float(data[metric].median())
    figure.add_hline(
        y=median,
        line={"color": PALETTE["neutral"], "dash": "dot", "width": 1},
        annotation_text=f"median {median:.2f}",
    )
    apply_layout(
        figure,
        title=f"Parameter sensitivity: {parameter}",
        xaxis_title=f"{parameter} value",
        yaxis_title="Sharpe ratio (annualised)",
        height=360,
        hovermode="closest",
    )
    return figure


def walk_forward_chart(selections: pd.DataFrame) -> go.Figure:
    """Train, validation and test Sharpe per walk-forward window.

    The visual point is the *gap*: bars that shrink left to right mean the
    selection process was fitting noise.
    """
    figure = go.Figure()
    if selections.empty:
        return apply_layout(figure, title="Walk-forward results - no windows")

    labels = [
        f"W{int(row['window'])}\n{row['test_start']}" for _, row in selections.iterrows()
    ]
    for i, (column, name) in enumerate(
        [("train_sharpe", "Train"), ("validation_sharpe", "Validation"), ("test_sharpe", "Test (out-of-sample)")]
    ):
        if column not in selections.columns:
            continue
        figure.add_trace(
            go.Bar(
                x=labels,
                y=selections[column],
                name=name,
                marker={"color": PALETTE["series"][i]},
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>Sharpe %{{y:.2f}}<extra></extra>",
            )
        )
    figure.add_hline(y=0.0, line={"color": PALETTE["neutral"], "width": 1})
    apply_layout(
        figure,
        title="Walk-forward Sharpe by window: in-sample, validation, out-of-sample",
        xaxis_title="Walk-forward window (test period start)",
        yaxis_title="Sharpe ratio (annualised)",
        height=420,
        hovermode="closest",
    )
    figure.update_layout(barmode="group")
    return figure


def strategy_contribution_chart(contributions: pd.DataFrame) -> go.Figure:
    """Each strategy's share of the combined signal over time."""
    figure = go.Figure()
    frame = contributions.fillna(0.0)
    for i, column in enumerate(frame.columns):
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=100.0 * frame[column].to_numpy(),
                name=str(column),
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": PALETTE["series"][i % len(PALETTE["series"])]},
                hovertemplate=f"<b>{column}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )
    apply_layout(
        figure,
        title=f"Strategy contribution to the combined signal - {_date_range_label(frame.index)}",
        yaxis_title="Share of total absolute signal (%)",
        height=380,
    )
    return figure


def bootstrap_distribution_chart(paths: pd.DataFrame, metric: str = "annualized_return") -> go.Figure:
    """Distribution of a bootstrapped statistic."""
    from atlas.validation.stress_tests import bootstrap_statistics

    equity = (1.0 + paths).cumprod()
    if metric == "annualized_return":
        values = equity.iloc[-1] ** (252.0 / len(paths)) - 1.0
        label = "Simulated annualised return"
    elif metric == "max_drawdown":
        values = -((equity / equity.cummax()) - 1.0).min()
        label = "Simulated maximum drawdown"
    else:
        summary = bootstrap_statistics(paths)
        values = pd.Series(summary.loc[metric]) if metric in summary.index else pd.Series(dtype=float)
        label = metric

    figure = go.Figure(
        go.Histogram(
            x=100.0 * values.to_numpy(),
            nbinsx=60,
            marker={"color": PALETTE["primary"], "opacity": 0.8},
            hovertemplate="%{x:.1f}%<br>%{y} simulations<extra></extra>",
        )
    )
    if len(values):
        for quantile, colour in ((0.05, PALETTE["loss"]), (0.5, PALETTE["neutral"]), (0.95, PALETTE["gain"])):
            figure.add_vline(
                x=100.0 * float(np.quantile(values, quantile)),
                line={"color": colour, "dash": "dash", "width": 1.5},
                annotation_text=f"p{int(quantile * 100)}",
            )
    apply_layout(
        figure,
        title=f"Block-bootstrap distribution: {label} (statistical stress test, not a forecast)",
        xaxis_title=f"{label} (%)",
        yaxis_title="Number of simulations",
        height=360,
        legend=False,
        hovermode="closest",
    )
    return figure
