"""Cached data access for the dashboard.

These helpers live in their own module rather than in ``app.py`` so that pages
can import them without re-executing the app script. Importing ``app`` from a
page would create a second module object, run its top-level code again, and
render the sidebar twice - which Streamlit rejects as duplicate widget IDs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from atlas.backtest.engine import BacktestEngine
from atlas.config import AtlasConfig, load_config
from atlas.data.features import FeatureEngineer
from atlas.data.loader import MarketDataLoader
from atlas.data.validator import DataValidator

__all__ = ["get_config", "get_panel", "run_backtest"]


@st.cache_resource(show_spinner=False)
def get_config() -> AtlasConfig:
    """Load the validated configuration once per session."""
    return load_config()


@st.cache_data(show_spinner="Loading market data...", ttl=3600)
def get_panel(provider: str, start: str, end: str | None, _config_hash: str):
    """Load and validate the price panel. Returns ``(panel, validation_report)``."""
    config = get_config().model_copy(deep=True)
    config.assets.data.provider = provider  # type: ignore[assignment]
    loader = MarketDataLoader.from_config(config)
    panel = loader.load(
        config.universe.symbols,
        pd.Timestamp(start).date(),
        pd.Timestamp(end).date() if end else None,
    )
    validator = DataValidator(
        max_staleness_days=config.data.max_staleness_days,
        min_history_days=config.data.min_history_days,
    )
    report = validator.validate(panel, expected_symbols=config.universe.symbols)
    return panel, report


@st.cache_data(show_spinner="Running backtest...", ttl=3600)
def run_backtest(provider: str, start: str, end: str | None, _config_hash: str):
    """Run a backtest. Returns ``(result, panel, features)``."""
    config = get_config().model_copy(deep=True)
    config.assets.data.provider = provider  # type: ignore[assignment]
    panel, _ = get_panel(provider, start, end, _config_hash)
    features = FeatureEngineer.from_config(config).build(panel)
    result = BacktestEngine(config).run(panel, features=features)
    return result, panel, features


@st.cache_data(show_spinner="Computing benchmarks...", ttl=3600)
def get_benchmarks(provider: str, start: str, end: str | None, _config_hash: str):
    """Compute the benchmark suite for the current settings."""
    from atlas.validation.benchmarks import BenchmarkSuite

    config = get_config()
    result, panel, features = run_backtest(provider, start, end, _config_hash)
    return BenchmarkSuite(config).run(panel, atlas_result=result, features=features)
