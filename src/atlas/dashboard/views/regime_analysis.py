"""Regime analysis page: labels, indicators, transitions and per-regime performance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import chart, metric_row, render_dataframe, section
from atlas.reporting import charts


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the regime-analysis page."""
    from atlas.dashboard.data_access import run_backtest

    result, _panel, _ = run_backtest(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )
    regimes = result.regimes
    if regimes is None:
        st.info("Regime detection is disabled in configs/strategies.yaml.")
        return

    labels = regimes.as_strings()
    indicators = regimes.features
    current = labels.iloc[-1]
    latest = indicators.iloc[-1]

    section("Current regime", f"As of {labels.index.max().date()}")
    metric_row(
        {
            "Regime": current.replace("_", " "),
            "Trend score": f"{latest.get('trend_score', np.nan):.2%}",
            "Volatility (20d)": f"{latest.get('volatility', np.nan):.2%}",
            "Volatility percentile": f"{latest.get('volatility_percentile', np.nan):.0%}",
        }
    )
    metric_row(
        {
            "Breadth (% above 200d MA)": f"{latest.get('breadth', np.nan):.0%}",
            "Average correlation": f"{latest.get('avg_correlation', np.nan):.2f}",
            "Benchmark drawdown": f"{latest.get('drawdown', np.nan):.2%}",
            "Detector": regimes.detector,
        }
    )

    st.markdown("---")
    chart(charts.regime_timeline_chart(regimes.labels, equity=result.equity_curve))

    left, right = st.columns(2)
    with left:
        section("Regime distribution")
        distribution = regimes.distribution().rename("share of days")
        render_dataframe(
            distribution.map("{:.1%}".format).to_frame().reset_index().rename(
                columns={"index": "regime"}
            )
        )
    with right:
        section("Regime transitions")
        transitions = regimes.transitions()
        render_dataframe(transitions, empty_message="No regime transition occurred.")

    st.markdown("---")
    section("Underlying indicators")
    columns = [
        c
        for c in ("trend_score", "volatility", "volatility_percentile", "breadth",
                  "avg_correlation", "drawdown")
        if c in indicators.columns
    ]
    st.line_chart(indicators[columns].tail(1500), height=320)
    st.caption(
        "The discrete label is driven only by trend score (above or below the moving average) "
        "and the volatility percentile. Breadth, correlation and drawdown are recorded for "
        "context and feed the defensive overlay."
    )

    st.markdown("---")
    section("Portfolio performance by regime")
    aligned = labels.reindex(result.returns.index).ffill()
    rows = []
    for regime, group in result.returns.groupby(aligned):
        if len(group) < 10:
            continue
        rows.append(
            {
                "regime": regime,
                "days": len(group),
                "share of sample": len(group) / len(result.returns),
                "annualised return": float((1 + group).prod() ** (252 / len(group)) - 1),
                "annualised volatility": float(group.std(ddof=1) * np.sqrt(252)),
                "worst day": float(group.min()),
                "best day": float(group.max()),
            }
        )
    if rows:
        render_dataframe(pd.DataFrame(rows).round(4))

    section("Regime episodes")
    episodes = regimes.episodes()
    if not episodes.empty:
        display = episodes.copy()
        display["start"] = display["start"].dt.date
        display["end"] = display["end"].dt.date
        render_dataframe(display.sort_values("days", ascending=False).head(25))

    st.markdown("---")
    section("Strategy weights by regime", "From configs/regimes.yaml - not fitted to the data.")
    rows = []
    for name, block in config.regimes.regime_weights.items():
        weights = block.strategy_weights()
        rows.append(
            {
                "regime": name,
                **{k: f"{v:.0%}" for k, v in weights.items()},
                "risk multiplier": f"{block.risk_multiplier:.2f}",
                "defensive weight": f"{block.defensive_weight:.0%}",
            }
        )
    render_dataframe(pd.DataFrame(rows))
