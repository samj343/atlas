"""Dashboard page renderers.

Named ``views`` rather than ``pages`` because Streamlit treats a ``pages/``
directory beside the entry point as its own automatic multipage navigation,
which would conflict with the sidebar page selector Atlas uses.
"""

__all__ = [
    "backtest_results",
    "execution",
    "overview",
    "portfolio_analysis",
    "regime_analysis",
    "research_config",
    "risk_monitor",
    "strategy_analysis",
]
