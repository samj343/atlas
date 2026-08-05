"""Atlas - a regime-aware, multi-strategy research and paper-trading platform.

Atlas is an **educational research and paper-trading system**. Historical
performance does not guarantee future results, and paper-trading execution does
not replicate all real-world liquidity, slippage, latency and fill conditions.

The package is organised into layers that can be used independently:

``atlas.data``
    Provider-agnostic market data, caching, validation and feature engineering.
``atlas.strategies``
    Signal generators: trend following, mean reversion, cross-sectional momentum.
``atlas.regimes``
    Interpretable market-regime detection.
``atlas.portfolio``
    Volatility and covariance estimation, allocation, constraints, risk management.
``atlas.backtest``
    Event-driven backtesting engine, cost model and performance metrics.
``atlas.validation``
    Walk-forward validation, parameter stability, stress tests, bootstrapping.
``atlas.execution``
    Interactive Brokers paper-trading integration with pre-trade safety checks.
``atlas.reporting``
    Charts and automated research reports.
"""

from __future__ import annotations

__version__ = "0.1.0"

DISCLAIMER = (
    "Atlas is an educational research and paper-trading system. Historical "
    "performance does not guarantee future results. Paper-trading execution does "
    "not replicate all real-world liquidity, slippage, latency, and fill conditions."
)

__all__ = ["DISCLAIMER", "__version__"]
