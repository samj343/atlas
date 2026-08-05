"""Execution page: current and target positions, required trades, orders and fills."""

from __future__ import annotations

import contextlib
from typing import Any

import pandas as pd
import streamlit as st

from atlas.config import AtlasConfig, ExecutionMode
from atlas.dashboard.components import (
    error_banner,
    metric_row,
    render_dataframe,
    section,
    status_pill,
    warning_banner,
)


def render(config: AtlasConfig, settings: dict[str, Any]) -> None:
    """Render the execution page."""

    section("Execution status")
    mode = config.execution.mode
    metric_row(
        {
            "Execution mode": mode.value,
            "Broker": config.execution.broker.provider,
            "Endpoint": f"{config.execution.broker.host}:{config.execution.broker.port}",
            "Client id": config.execution.broker.client_id,
        }
    )

    if mode is ExecutionMode.PAPER:
        warning_banner(
            "EXECUTION_MODE is `paper`. Running `atlas paper-trade` will submit orders to the "
            "configured Interactive Brokers **paper** account."
        )
    else:
        st.info(
            f"EXECUTION_MODE is `{mode.value}`: orders are previewed but never sent. "
            "Live trading is not implemented in Atlas."
        )

    st.markdown("---")
    section(
        "Dry-run order preview",
        "Computed exactly as the paper-trading cycle would, then stopped before submission.",
    )

    use_mock = st.checkbox(
        "Use the in-memory mock broker",
        value=True,
        help=(
            "The mock broker needs no TWS connection. Uncheck it to connect to a running "
            "TWS or IB Gateway instance."
        ),
    )

    if st.button("Generate order preview", type="primary"):
        _preview(config, settings, use_mock)
    else:
        st.caption("Press the button to compute the trades Atlas would place today.")

    st.markdown("---")
    section("Safety checks")
    from atlas.execution.safety import SafetyGate

    report = SafetyGate(config).run(
        account="DU0000000" if use_mock else None,
        connected=use_mock,
        risk_halted=False,
    )
    for check in report.checks:
        status_pill(check.name, check.passed, check.message, check.message)

    st.markdown("---")
    section("Recent orders and reconciliation")
    try:
        from atlas.database import AtlasRepository, DatabaseManager

        repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
        render_dataframe(
            repository.recent_orders(limit=50),
            empty_message="No orders have been recorded yet.",
        )
        latest = repository.latest_reconciliation()
        if latest:
            status_pill(
                "Last reconciliation",
                latest["status"] in {"reconciled", "tolerated"},
                f"{latest['status']} at {latest['checked_at']}",
                f"{latest['n_mismatches']} mismatch(es) - trading blocked",
            )
        else:
            st.info("No reconciliation has been recorded yet.")
    except Exception as exc:
        st.info(f"No execution database available yet ({exc}).")


def _preview(config: AtlasConfig, settings: dict[str, Any], use_mock: bool) -> None:
    """Build and display a dry-run trade plan."""
    from atlas.dashboard.data_access import get_panel
    from atlas.execution.ibkr_client import IBKRClient, MockBrokerClient
    from atlas.execution.order_manager import OrderManager

    panel, _ = get_panel(
        settings["provider"], settings["start"], settings["end"], config.config_hash
    )
    prices = panel.adj_close.iloc[-1]

    broker = (
        MockBrokerClient(prices=prices, initial_cash=config.portfolio.initial_capital)
        if use_mock
        else IBKRClient(config.execution)
    )
    try:
        with st.spinner("Computing target portfolio..."):
            manager = OrderManager(config, broker=broker)
            result = manager.run_cycle(panel, submit=False)
    except Exception as exc:
        error_banner(f"Could not build an order preview: {exc}")
        return
    finally:
        with contextlib.suppress(Exception):
            # A failed disconnect must not mask the result already computed.
            broker.disconnect()

    if result.blocked_reason:
        error_banner(f"Execution blocked: {result.blocked_reason}")

    plan = result.plan
    if plan is None:
        st.info("No trade plan was produced.")
        return

    metric_row(
        {
            "Orders": plan.n_orders,
            "Gross notional": f"${plan.gross_notional:,.0f}",
            "Estimated turnover": f"{plan.estimated_turnover:.2%}",
            "Regime": plan.regime.replace("_", " "),
        }
    )

    section("Required trades")
    render_dataframe(plan.to_frame(), empty_message="No trades are required.")

    left, right = st.columns(2)
    with left:
        section("Target positions")
        targets = plan.target_weights
        targets = targets[targets.abs() > 1e-6].sort_values(ascending=False)
        render_dataframe(
            targets.map("{:.2%}".format).rename("target weight").to_frame().reset_index().rename(
                columns={"index": "symbol"}
            ),
            empty_message="The target portfolio is entirely cash.",
        )
    with right:
        section("Current positions")
        current = plan.current_shares
        current = current[current.abs() > 1e-9]
        render_dataframe(
            current.rename("shares").to_frame().reset_index().rename(columns={"index": "symbol"}),
            empty_message="No positions are currently held.",
        )

    if plan.skipped:
        section("Skipped")
        render_dataframe(pd.DataFrame(plan.skipped))

    if result.reconciliation is not None:
        section("Reconciliation")
        status_pill(
            "Broker versus local positions",
            result.reconciliation.reconciled,
            result.reconciliation.status,
            f"{len(result.reconciliation.mismatches)} mismatch(es)",
        )
        render_dataframe(
            result.reconciliation.to_frame(), empty_message="No differences found."
        )

    st.code(plan.render(), language="text")
