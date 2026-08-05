"""Atlas Streamlit dashboard.

Run with ``atlas dashboard`` or ``streamlit run src/atlas/dashboard/app.py``.

The dashboard is a *viewer*, not a controller: it reads configuration, runs
research on demand and displays stored results. It deliberately cannot submit an
order or change a configuration file - those actions belong on the command line
where they are logged and auditable.

Pages
-----
Overview, Backtest Results, Strategy Analysis, Portfolio Analysis, Regime
Analysis, Execution, Risk Monitor, Research Configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow `streamlit run src/atlas/dashboard/app.py` from a source checkout.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:  # pragma: no cover - import-path shim
    sys.path.insert(0, str(_SRC))

from atlas import DISCLAIMER, __version__  # noqa: E402
from atlas.config import AtlasConfig  # noqa: E402
from atlas.dashboard.components import warning_banner  # noqa: E402
from atlas.dashboard.data_access import get_config  # noqa: E402

st.set_page_config(
    page_title="Atlas - Regime-Aware Multi-Strategy Trading System",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar(config: AtlasConfig) -> dict[str, Any]:
    """Render the sidebar and return the selected settings."""
    st.sidebar.markdown("## 🧭 Atlas")
    st.sidebar.caption(f"v{__version__} · regime-aware multi-strategy research")

    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Backtest Results",
            "Strategy Analysis",
            "Portfolio Analysis",
            "Regime Analysis",
            "Execution",
            "Risk Monitor",
            "Research Configuration",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data")
    provider = st.sidebar.selectbox(
        "Provider",
        ["synthetic", "yfinance", "csv"],
        index=["synthetic", "yfinance", "csv"].index(config.data.provider)
        if config.data.provider in {"synthetic", "yfinance", "csv"}
        else 0,
        help=(
            "'synthetic' generates simulated prices and needs no network access. "
            "Results from it are NOT market results."
        ),
    )
    start = st.sidebar.date_input("Start date", value=pd.Timestamp("2008-01-01").date())
    end = st.sidebar.date_input("End date", value=pd.Timestamp("2018-12-31").date())

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status")
    st.sidebar.write(f"**Execution mode:** `{config.execution.mode.value}`")
    if config.risk.kill_switch.enabled:
        st.sidebar.error("KILL SWITCH ENGAGED")
    else:
        st.sidebar.success("Kill switch off")
    st.sidebar.caption(f"Config hash `{config.config_hash}`")

    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)

    return {"page": page, "provider": provider, "start": str(start), "end": str(end)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the dashboard."""
    try:
        config = get_config()
    except Exception as exc:
        st.error(f"Could not load configuration: {exc}")
        st.stop()
        return

    settings = sidebar(config)
    page = settings["page"]

    st.title("Atlas")
    st.caption(
        "Regime-aware multi-strategy research and paper-trading platform · "
        "**educational use only**"
    )

    if settings["provider"] == "synthetic":
        warning_banner(
            "Synthetic data selected. Every number shown is computed from **simulated** prices "
            "and carries no information about real market performance. Switch the provider to "
            "`yfinance` (network required) for real history."
        )

    # Page modules live in `views/`, not `pages/`: Streamlit reserves a `pages/`
    # directory beside the entry point for its own automatic multipage
    # navigation, which would render a second, conflicting page switcher.
    # They are imported lazily so a failure in one does not break the others.
    from atlas.dashboard.views import (
        backtest_results,
        execution,
        overview,
        portfolio_analysis,
        regime_analysis,
        research_config,
        risk_monitor,
        strategy_analysis,
    )

    renderers = {
        "Overview": overview.render,
        "Backtest Results": backtest_results.render,
        "Strategy Analysis": strategy_analysis.render,
        "Portfolio Analysis": portfolio_analysis.render,
        "Regime Analysis": regime_analysis.render,
        "Execution": execution.render,
        "Risk Monitor": risk_monitor.render,
        "Research Configuration": research_config.render,
    }

    renderer = renderers.get(page)
    if renderer is None:  # pragma: no cover - defensive
        st.error(f"Unknown page: {page}")
        return

    try:
        renderer(config, settings)
    except Exception as exc:
        st.error(f"This page could not be rendered: {exc}")
        with st.expander("Details"):
            st.exception(exc)


# Streamlit executes this file as a script, so `main()` runs on import under
# `streamlit run`. It is deliberately NOT called when the module is imported by
# something else - that would render the sidebar a second time and Streamlit
# rejects the duplicate widget IDs.
if __name__ == "__main__":  # pragma: no cover
    main()
