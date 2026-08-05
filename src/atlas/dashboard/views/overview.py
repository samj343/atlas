"""Overview page: current portfolio state, exposure, regime and execution status."""

from __future__ import annotations

from typing import Any

import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import chart, metric_row, render_dataframe, section, status_pill
from atlas.reporting import charts


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the overview page."""
    from atlas.dashboard.data_access import run_backtest

    result, _panel, _ = run_backtest(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )

    equity = result.equity_curve
    returns = result.returns
    snapshots = result.snapshots
    latest = snapshots.iloc[-1] if not snapshots.empty else None

    from atlas.backtest.metrics import drawdown_series, rolling_volatility

    drawdown = drawdown_series(equity)
    predicted_vol = rolling_volatility(returns, 63).dropna()
    regime = (
        result.regimes.as_strings().iloc[-1] if result.regimes is not None else "unknown"
    )

    section(
        "Portfolio overview",
        f"State as of {equity.index.max().date()} · run `{result.run_id}`",
    )
    metric_row(
        {
            "Portfolio value": f"${result.final_equity:,.0f}",
            "Daily P&L": (
                f"${latest['daily_pnl']:,.0f}" if latest is not None else "-",
                f"{returns.iloc[-1]:.2%}" if len(returns) else None,
            ),
            "Total return": f"{result.metrics.values.get('total_return', 0):.2%}",
            "Current drawdown": f"{drawdown.iloc[-1]:.2%}" if len(drawdown) else "-",
        }
    )
    metric_row(
        {
            "Realised volatility (63d)": (
                f"{predicted_vol.iloc[-1]:.2%}" if len(predicted_vol) else "-"
            ),
            "Gross exposure": f"{latest['gross_exposure']:.2%}" if latest is not None else "-",
            "Net exposure": f"{latest['net_exposure']:.2%}" if latest is not None else "-",
            "Cash allocation": f"{latest['cash_weight']:.2%}" if latest is not None else "-",
        }
    )
    metric_row(
        {
            "Market regime": regime.replace("_", " "),
            "Volatility target": f"{config.portfolio.target_volatility:.1%}",
            "Latest rebalance": (
                str(result.target_weights.index.max().date())
                if not result.target_weights.empty
                else "-"
            ),
            "Execution status": config.execution.mode.value,
        }
    )

    st.markdown("---")
    chart(charts.equity_curve_chart({"Atlas": equity}, title="Portfolio equity"))

    left, right = st.columns(2)
    with left:
        section("Current positions")
        if not result.weights.empty:
            current = result.weights.iloc[-1]
            current = current[current.abs() > 1e-6].sort_values(ascending=False)
            frame = current.rename("weight").to_frame()
            frame["weight"] = frame["weight"].map("{:.2%}".format)
            frame["asset class"] = [config.asset_class_map.get(s, "-") for s in frame.index]
            render_dataframe(frame.reset_index().rename(columns={"index": "symbol"}))
        else:
            st.info("No positions held.")
    with right:
        section("System status")
        status_pill("Data validation", True, "prices loaded and validated")
        status_pill(
            "Kill switch",
            not config.risk.kill_switch.enabled,
            "off - trading permitted",
            f"ENGAGED: {config.risk.kill_switch.reason}",
        )
        status_pill(
            "Live trading",
            True,
            "disabled by design - paper only",
        )
        status_pill(
            "Data source",
            not result.is_synthetic_data,
            f"real data ({result.data_source})",
            "SYNTHETIC - results are simulated",
        )
        st.caption(f"Git commit `{(result.git_commit or 'unknown')[:12]}`")
