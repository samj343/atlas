"""Backtest results page: equity, drawdown, rolling statistics and costs."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import chart, metric_row, render_dataframe, section
from atlas.reporting import charts


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the backtest-results page."""
    from atlas.dashboard.data_access import get_benchmarks, run_backtest

    result, _panel, _features = run_backtest(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )
    metrics = result.metrics

    section("Headline performance", "All figures are net of the modelled transaction costs.")
    metric_row(
        {
            "Total return": f"{metrics.values.get('total_return', 0):.2%}",
            "CAGR": f"{metrics.cagr:.2%}",
            "Volatility": f"{metrics.values.get('annualized_volatility', 0):.2%}",
            "Sharpe": f"{metrics.sharpe:.2f}",
            "Sortino": f"{metrics.values.get('sortino_ratio', 0):.2f}",
            "Calmar": f"{metrics.values.get('calmar_ratio', 0):.2f}",
            "Max drawdown": f"{metrics.max_drawdown:.2%}",
            "Trades": f"{result.n_trades:,}",
        },
        columns=4,
    )
    st.caption(
        f"Sharpe and Sortino assume a {config.validation.benchmarks.risk_free_rate:.2%} "
        "annualised risk-free rate, de-annualised geometrically, annualised by sqrt(252)."
    )

    st.markdown("---")
    benchmarks = get_benchmarks(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )

    curves = {"Atlas": result.equity_curve}
    for name, series in benchmarks.returns.items():
        if name != "Atlas regime-aware ensemble":
            curves[name] = (1.0 + series).cumprod()
    chart(charts.equity_curve_chart(curves, title="Atlas versus benchmarks"))

    section("Benchmark comparison")
    render_dataframe(benchmarks.table().round(4).reset_index().rename(columns={"index": "metric"}))
    if benchmarks.notes:
        for note in benchmarks.notes:
            st.caption(f"Note: {note}")

    st.markdown("---")
    chart(charts.drawdown_chart(result.equity_curve))

    left, right = st.columns(2)
    with left:
        chart(
            charts.rolling_volatility_chart(
                result.returns, target=config.portfolio.target_volatility
            )
        )
    with right:
        chart(
            charts.rolling_sharpe_chart(
                result.returns, risk_free_rate=config.validation.benchmarks.risk_free_rate
            )
        )

    chart(charts.monthly_heatmap(result.returns))
    chart(charts.return_distribution_chart(result.returns))

    st.markdown("---")
    section("Annual returns")
    from atlas.backtest.metrics import annual_returns

    annual = annual_returns(result.returns)
    if len(annual):
        frame = pd.DataFrame(
            {"year": [d.year for d in annual.index], "return": annual.map("{:.2%}".format).to_numpy()}
        )
        render_dataframe(frame)

    section("Transaction costs")
    breakdown = result.cost_breakdown()
    drag = result.cost_drag()
    metric_row(
        {
            "Total costs": f"${breakdown.sum():,.0f}",
            "Commission": f"${breakdown.get('commission', 0):,.0f}",
            "Spread": f"${breakdown.get('spread_cost', 0):,.0f}",
            "Slippage": f"${breakdown.get('slippage_cost', 0):,.0f}",
            "Market impact": f"${breakdown.get('impact_cost', 0):,.0f}",
            "Gross CAGR": f"{drag.get('gross_cagr', 0):.2%}",
            "Net CAGR": f"{drag.get('net_cagr', 0):.2%}",
            "Annual cost drag": f"{drag.get('cost_drag_annual', 0):.2%}",
        },
        columns=4,
    )
    chart(charts.turnover_cost_chart(result.snapshots, result.fills))

    section("Full metric set")
    render_dataframe(
        metrics.to_series().rename("value").to_frame().reset_index().rename(
            columns={"index": "metric"}
        ),
        height=420,
    )
