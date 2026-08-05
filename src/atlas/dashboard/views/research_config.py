"""Research configuration page: inspect (but do not casually overwrite) settings."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig
from atlas.dashboard.components import metric_row, render_dataframe, section


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:  # noqa: ARG001
    """Render the research-configuration page.

    ``settings`` is unused here - this page shows configuration, not results -
    but the argument is part of the common render signature every view shares.
    """
    section(
        "Configuration",
        "This page is read-only by design. Configuration changes belong in version-controlled "
        "YAML so every stored result can be traced back to the exact settings that produced it.",
    )
    metric_row(
        {
            "Configuration hash": config.config_hash,
            "Universe": f"{config.universe.name} ({len(config.symbols)})",
            "Strategies enabled": len(config.strategies.enabled_strategies),
            "Execution mode": config.execution.mode.value,
        }
    )

    st.info(
        "To change a setting: edit the YAML file, re-run the command, and Atlas stores a **new** "
        "configuration version. Previous versions are never overwritten."
    )

    st.markdown("---")
    section("Asset universe")
    render_dataframe(
        pd.DataFrame(
            [
                {
                    "symbol": a.symbol,
                    "name": a.name,
                    "asset class": a.asset_class,
                    "tradable": a.tradable,
                }
                for a in config.universe.assets
            ]
        )
    )
    st.caption(
        f"Benchmark: **{config.universe.benchmark}** · "
        f"defensive asset: **{config.universe.defensive_asset}**"
    )

    st.markdown("---")
    tabs = st.tabs(
        [
            "Strategies",
            "Regimes",
            "Portfolio & risk",
            "Costs & execution",
            "Validation",
            "Stored versions",
        ]
    )

    with tabs[0]:
        for name, block in config.strategies.strategies.as_dict().items():
            with st.expander(f"{name} · {'enabled' if block.enabled else 'disabled'}", expanded=True):
                st.json(block.model_dump(mode="json"))
        st.subheader("Ensemble")
        st.json(config.strategies.ensemble.model_dump(mode="json"))
        st.subheader("Backtest")
        st.json(config.strategies.backtest.model_dump(mode="json"))

    with tabs[1]:
        st.subheader("Detector")
        st.json(config.regimes.detector.model_dump(mode="json"))
        st.subheader("Strategy weights by regime")
        render_dataframe(
            pd.DataFrame(
                [
                    {"regime": name, **block.model_dump(mode="json")}
                    for name, block in config.regimes.regime_weights.items()
                ]
            )
        )
        st.subheader("Breadth overlay")
        st.json(config.regimes.breadth_overlay.model_dump(mode="json"))

    with tabs[2]:
        st.subheader("Portfolio limits")
        st.json(config.portfolio.model_dump(mode="json"))
        st.subheader("Risk limits")
        st.json(config.risk.limits.model_dump(mode="json"))
        st.subheader("Drawdown controls")
        st.json(config.risk.drawdown_controls.model_dump(mode="json"))
        st.subheader("Covariance and volatility")
        st.json(
            {
                "covariance": config.risk.covariance.model_dump(mode="json"),
                "volatility": config.risk.volatility.model_dump(mode="json"),
            }
        )

    with tabs[3]:
        st.subheader("Transaction costs")
        from atlas.backtest.costs import TransactionCostModel

        render_dataframe(
            pd.DataFrame(
                TransactionCostModel(config.costs).describe().items(),
                columns=["assumption", "value"],
            )
        )
        st.subheader("Broker")
        broker = config.execution.broker.model_dump(mode="json")
        st.json(broker)
        st.caption(
            "No credential is ever stored in configuration. Connection settings come from the "
            "environment (see .env.example)."
        )
        st.subheader("Trading window")
        st.json(config.execution.trading_window.model_dump(mode="json"))

    with tabs[4]:
        st.subheader("Walk-forward")
        st.json(config.validation.walk_forward.model_dump(mode="json"))
        st.subheader("Stress periods")
        render_dataframe(
            pd.DataFrame(
                [
                    {"name": p.name, "start": p.start, "end": p.end}
                    for p in config.validation.stress_tests.periods
                ]
            )
        )
        st.subheader("Benchmarks")
        st.json(config.validation.benchmarks.model_dump(mode="json"))

    with tabs[5]:
        st.subheader("Stored configuration versions")
        try:
            from atlas.database import AtlasRepository, DatabaseManager

            repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
            render_dataframe(
                repository.list_configurations(),
                empty_message="No configuration version has been stored yet.",
            )
            st.subheader("Stored runs")
            render_dataframe(
                repository.list_runs(limit=25),
                empty_message="No backtest run has been stored yet.",
            )
        except Exception as exc:
            st.info(f"No database is available yet ({exc}).")

    st.markdown("---")
    section("Full configuration snapshot")
    with st.expander("Show the complete validated configuration"):
        st.json(config.snapshot())
    st.download_button(
        "Download configuration snapshot (JSON)",
        data=json.dumps(config.snapshot(), indent=2, default=str),
        file_name=f"atlas_config_{config.config_hash}.json",
        mime="application/json",
    )
