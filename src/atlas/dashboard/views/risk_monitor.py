"""Risk monitor page: limit status, drawdown, warnings and the risk-event log."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import (
    chart,
    error_banner,
    metric_row,
    render_dataframe,
    section,
    status_pill,
)
from atlas.reporting import charts


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the risk-monitor page."""
    from atlas.dashboard.data_access import run_backtest

    result, panel, _ = run_backtest(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )

    from atlas.backtest.metrics import drawdown_series, rolling_volatility

    equity = result.equity_curve
    drawdown = drawdown_series(equity)
    current_drawdown = abs(float(drawdown.iloc[-1])) if len(drawdown) else 0.0
    realised_vol = rolling_volatility(result.returns, 63).dropna()
    latest_vol = float(realised_vol.iloc[-1]) if len(realised_vol) else float("nan")
    snapshots = result.snapshots
    latest = snapshots.iloc[-1] if not snapshots.empty else None

    section("Kill switch")
    if config.risk.kill_switch.enabled:
        error_banner(
            f"KILL SWITCH ENGAGED — {config.risk.kill_switch.reason or 'no reason recorded'}. "
            "No component may create or submit an order."
        )
    else:
        st.success("Kill switch is off. Disable trading with `ATLAS_KILL_SWITCH=1`.")

    st.markdown("---")
    section("Risk-limit status")

    multiplier, breached = config.risk.drawdown_controls.multiplier_for(current_drawdown)
    checks = {
        "Maximum drawdown": (
            current_drawdown <= config.risk.limits.max_drawdown,
            f"{current_drawdown:.2%} of a {config.risk.limits.max_drawdown:.0%} limit",
        ),
        "Predicted volatility": (
            not np.isfinite(latest_vol) or latest_vol <= config.portfolio.max_portfolio_volatility,
            f"{latest_vol:.2%} against a {config.portfolio.max_portfolio_volatility:.0%} ceiling",
        ),
        "Gross exposure": (
            latest is None or latest["gross_exposure"] <= config.portfolio.max_gross_exposure,
            f"{latest['gross_exposure']:.2%} of {config.portfolio.max_gross_exposure:.0%}"
            if latest is not None
            else "-",
        ),
        "Net exposure": (
            latest is None or abs(latest["net_exposure"]) <= config.portfolio.max_net_exposure,
            f"{latest['net_exposure']:.2%} of {config.portfolio.max_net_exposure:.0%}"
            if latest is not None
            else "-",
        ),
        "Position count": (
            latest is None or latest["n_positions"] <= config.portfolio.max_active_positions,
            f"{int(latest['n_positions'])} of {config.portfolio.max_active_positions}"
            if latest is not None
            else "-",
        ),
        "Data freshness": (
            True,
            f"last observation {panel.dates.max().date()}",
        ),
    }
    for label, (ok, detail) in checks.items():
        status_pill(label, ok, detail, detail)

    st.markdown("---")
    section("Drawdown control")
    metric_row(
        {
            "Current drawdown": f"{current_drawdown:.2%}",
            "Risk multiplier in force": f"{multiplier:.2f}",
            "Threshold breached": f"{breached:.0%}" if breached is not None else "none",
            "Maximum drawdown to date": f"{result.metrics.max_drawdown:.2%}",
        }
    )
    ladder = pd.DataFrame(
        [
            {
                "drawdown at least": f"{t.drawdown:.0%}",
                "risk multiplier": f"{t.multiplier:.2f}",
                "in force": "yes" if current_drawdown >= t.drawdown else "no",
            }
            for t in config.risk.drawdown_controls.thresholds
        ]
    )
    render_dataframe(ladder)
    st.caption(
        "The ladder scales the portfolio risk budget. Atlas never liquidates automatically on "
        "drawdown unless `liquidate_on_breach` is explicitly enabled."
    )
    chart(charts.drawdown_chart(equity))

    st.markdown("---")
    section("Concentration warnings")
    if not result.weights.empty:
        current = result.weights.iloc[-1]
        held = current[current.abs() > 1e-6]
        if len(held) >= 2:
            correlation = panel.returns()[list(held.index)].tail(252).corr()
            values = correlation.to_numpy()
            upper = values[np.triu_indices_from(values, k=1)]
            worst = float(np.nanmax(upper)) if upper.size else float("nan")
            threshold = config.risk.limits.correlation_warning_threshold
            status_pill(
                "Held-position correlation",
                not np.isfinite(worst) or worst <= threshold,
                f"highest pairwise correlation {worst:.2f}, below the {threshold:.2f} threshold",
                f"highest pairwise correlation {worst:.2f} exceeds the {threshold:.2f} threshold",
            )
            chart(charts.asset_correlation_chart(panel.returns()[list(held.index)].tail(252),
                                                 window_label="held positions, last 252 days"))
        largest = held.abs().max() if len(held) else 0.0
        status_pill(
            "Largest position",
            largest <= config.portfolio.max_asset_weight + 1e-9,
            f"{largest:.2%} of a {config.portfolio.max_asset_weight:.0%} cap",
            f"{largest:.2%} exceeds the {config.portfolio.max_asset_weight:.0%} cap",
        )

    st.markdown("---")
    section("Risk events during the backtest")
    events = result.risk_events
    if events.empty:
        st.info("No risk control fired during this run.")
    else:
        counts = events["control"].value_counts().rename("times fired")
        render_dataframe(counts.to_frame().reset_index().rename(columns={"index": "control"}))
        st.caption("Most recent 100 events:")
        recent = events.sort_values("timestamp", ascending=False).head(100)
        render_dataframe(recent[["timestamp", "control", "severity", "action", "message"]])

    section("Configured limits")
    st.json(config.risk.limits.model_dump(mode="json"))
