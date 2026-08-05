"""Strategy analysis page: per-strategy behaviour, signals and contributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import chart, metric_row, render_dataframe, section
from atlas.reporting import charts


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the strategy-analysis page."""
    from atlas.dashboard.data_access import run_backtest

    result, panel, features = run_backtest(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )

    section(
        "Individual strategy performance",
        "Each strategy is run alone through the same portfolio, cost and risk machinery, "
        "so any difference comes from the signal rather than the plumbing.",
    )
    with st.spinner("Running each strategy standalone..."):
        variants = _variants(config, panel, features, result)

    if variants:
        table = pd.DataFrame(
            {
                name: {
                    "CAGR": metrics.cagr,
                    "Volatility": metrics.values.get("annualized_volatility", np.nan),
                    "Sharpe": metrics.sharpe,
                    "Max drawdown": metrics.max_drawdown,
                }
                for name, metrics in variants["metrics"].items()
            }
        ).T
        render_dataframe(table.round(4).reset_index().rename(columns={"index": "strategy"}))
        chart(charts.equity_curve_chart(variants["curves"], title="Strategy equity curves"))

        section("Strategy return correlation")
        frame = pd.DataFrame(variants["returns"]).dropna(how="all")
        if frame.shape[1] > 1:
            chart(charts.strategy_correlation_chart(frame))
            st.caption(
                "Low correlation between strategy returns is what makes combining them useful. "
                "High correlation means the ensemble is one bet wearing three hats."
            )

    st.markdown("---")
    section("Strategy contribution to the combined signal")
    if not result.strategy_contributions.empty:
        chart(charts.strategy_contribution_chart(result.strategy_contributions))
        average = result.strategy_contributions.mean()
        metric_row({k: f"{v:.1%}" for k, v in average.items()}, columns=3)

    st.markdown("---")
    section("Current signals by strategy")
    rows = []
    for name, frame in result.signals.items():
        if frame.empty:
            continue
        latest = frame.iloc[-1]
        for symbol, value in latest.items():
            rows.append({"strategy": name, "symbol": symbol, "signal": round(float(value), 4)})
    if rows:
        signals = pd.DataFrame(rows).pivot(index="symbol", columns="strategy", values="signal")
        render_dataframe(signals.reset_index(), hide_index=True)

    section("Strategy behaviour by regime")
    if result.regimes is not None and variants:
        labels = result.regimes.as_strings()
        rows = []
        for name, series in variants["returns"].items():
            aligned = labels.reindex(series.index).ffill()
            for regime, group in series.groupby(aligned):
                if len(group) < 20:
                    continue
                rows.append(
                    {
                        "strategy": name,
                        "regime": regime,
                        "days": len(group),
                        "annualised return": float((1 + group).prod() ** (252 / len(group)) - 1),
                        "annualised volatility": float(group.std(ddof=1) * np.sqrt(252)),
                    }
                )
        if rows:
            render_dataframe(pd.DataFrame(rows).round(4))
            st.caption(
                "This is the evidence behind the regime-conditional strategy weights in "
                "configs/regimes.yaml."
            )

    section("Strategy parameters")
    for name, strategy_config in config.strategies.strategies.as_dict().items():
        with st.expander(f"{name} ({'enabled' if strategy_config.enabled else 'disabled'})"):
            st.json(strategy_config.model_dump(mode="json"))


def _variants(config, panel, features, result):
    """Run each strategy standalone plus the equal-weight ensemble."""
    from atlas.backtest.engine import BacktestEngine
    from atlas.strategies.base import build_strategies

    names = ["trend", "mean_reversion", "cross_sectional_momentum"]
    curves: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    metrics: dict[str, Any] = {}

    for name in names:
        variant = config.model_copy(deep=True)
        for other in names:
            getattr(variant.strategies.strategies, other).enabled = other == name
        if not getattr(variant.strategies.strategies, name).enabled:
            continue
        variant.strategies.ensemble.mode = "fixed"
        variant.strategies.regime.enabled = False
        try:
            run = BacktestEngine(variant, strategies=build_strategies(variant)).run(
                panel, features=features
            )
        except Exception:
            continue
        curves[name] = run.equity_curve
        returns[name] = run.returns
        metrics[name] = run.metrics

    curves["ensemble (regime-aware)"] = result.equity_curve
    returns["ensemble (regime-aware)"] = result.returns
    metrics["ensemble (regime-aware)"] = result.metrics
    return {"curves": curves, "returns": returns, "metrics": metrics}
