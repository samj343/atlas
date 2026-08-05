"""Portfolio analysis page: weights, exposures, risk contributions and turnover."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import chart, metric_row, render_dataframe, section
from atlas.reporting import charts


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the portfolio-analysis page."""
    from atlas.dashboard.data_access import run_backtest

    result, panel, _ = run_backtest(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )
    weights = result.weights
    snapshots = result.snapshots

    section("Current weights")
    if not weights.empty:
        current = weights.iloc[-1]
        current = current[current.abs() > 1e-6].sort_values(ascending=False)
        frame = current.rename("weight").to_frame()
        frame["asset class"] = [config.asset_class_map.get(s, "-") for s in frame.index]
        frame["weight"] = frame["weight"].map("{:.2%}".format)
        render_dataframe(frame.reset_index().rename(columns={"index": "symbol"}))

        latest = snapshots.iloc[-1]
        metric_row(
            {
                "Gross exposure": f"{latest['gross_exposure']:.2%}",
                "Net exposure": f"{latest['net_exposure']:.2%}",
                "Cash": f"{latest['cash_weight']:.2%}",
                "Positions": int(latest["n_positions"]),
            }
        )

    st.markdown("---")
    chart(charts.weights_chart(weights))
    chart(charts.exposure_chart(snapshots))

    section("Asset-class exposure over time")
    if not weights.empty:
        classes: dict[str, list[str]] = {}
        for symbol in weights.columns:
            classes.setdefault(config.asset_class_map.get(str(symbol), "unclassified"), []).append(
                str(symbol)
            )
        by_class = pd.DataFrame(
            {name: weights[members].sum(axis=1) for name, members in classes.items()}
        )
        chart(charts.weights_chart(by_class, title="Asset-class exposure"))
        st.caption(
            f"The per-class cap is {config.portfolio.max_asset_class_weight:.0%} of equity."
        )

    st.markdown("---")
    section(
        "Risk contribution by asset",
        "Share of predicted portfolio variance, computed as w_i (Sigma w)_i normalised to sum "
        "to one. A large capital weight in a low-volatility asset contributes little risk.",
    )
    contributions = _risk_contributions(config, panel, weights)
    if not contributions.empty:
        chart(charts.risk_contribution_chart(contributions))
        latest = contributions.iloc[-1].sort_values(ascending=False)
        render_dataframe(
            latest.rename("risk share").to_frame().map(lambda v: f"{v:.1%}").reset_index().rename(
                columns={"index": "symbol"}
            )
        )

    st.markdown("---")
    section("Asset correlation")
    chart(charts.asset_correlation_chart(panel.returns(), window_label="full sample"))

    section("Turnover")
    if "turnover" in snapshots.columns:
        turnover = snapshots["turnover"]
        years = max(len(turnover) / 252.0, 1e-9)
        metric_row(
            {
                "Average daily turnover": f"{turnover.mean():.3%}",
                "Annual turnover": f"{turnover.sum() / years:.2f}x",
                "Maximum daily turnover": f"{turnover.max():.2%}",
                "Turnover cap": f"{config.portfolio.maximum_daily_turnover:.0%}",
            }
        )
        chart(charts.turnover_cost_chart(snapshots, result.fills))


@st.cache_data(show_spinner=False, ttl=1800)
def _risk_contributions_cached(_weights_hash: str) -> pd.DataFrame:  # pragma: no cover
    return pd.DataFrame()


def _risk_contributions(config, panel, weights: pd.DataFrame) -> pd.DataFrame:
    """Risk contribution per asset, sampled monthly to keep the page responsive."""
    if weights.empty:
        return pd.DataFrame()
    from atlas.portfolio.covariance import CovarianceEstimator, risk_contributions

    estimator = CovarianceEstimator(config.risk.covariance)
    returns = panel.returns("adj_close")
    # Monthly sampling: a daily covariance re-estimate would make the page slow
    # without changing the picture.
    sample_dates = weights.resample("ME").last().index
    rows: dict[pd.Timestamp, pd.Series] = {}
    for date in sample_dates:
        if date not in weights.index:
            candidates = weights.index[weights.index <= date]
            if not len(candidates):
                continue
            date = candidates[-1]
        window = returns.loc[:date].tail(config.risk.covariance.lookback)
        if len(window) < config.risk.covariance.min_observations:
            continue
        row = weights.loc[date]
        held = row[row.abs() > 1e-9]
        if held.empty:
            continue
        try:
            covariance = estimator.estimate_or_fallback(window, symbols=list(held.index))
            rows[date] = risk_contributions(held, covariance.matrix)
        except Exception:
            continue
    return pd.DataFrame(rows).T.fillna(0.0) if rows else pd.DataFrame()
