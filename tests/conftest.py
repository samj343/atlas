"""Shared pytest fixtures.

Every fixture is deterministic: the synthetic provider is seeded, and no test
touches the network or a broker. Fixtures that build a panel are session-scoped
because generating and validating price history dominates test runtime.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas.config import AtlasConfig, load_config
from atlas.data.features import FeatureEngineer, FeatureSet
from atlas.data.loader import MarketDataLoader, PricePanel
from atlas.data.provider import SyntheticProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A window long enough for a 252-day lookback plus several years of trading.
PANEL_START = dt.date(2006, 1, 2)
PANEL_END = dt.date(2014, 12, 31)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """The repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def base_config() -> AtlasConfig:
    """The shipped configuration, loaded once."""
    return load_config(PROJECT_ROOT / "configs")


@pytest.fixture
def config(base_config: AtlasConfig) -> AtlasConfig:
    """A fresh deep copy of the configuration for each test.

    Tests routinely mutate configuration to exercise a limit; copying keeps them
    independent of each other.
    """
    copy = base_config.model_copy(deep=True)
    copy.assets.data.provider = "synthetic"  # type: ignore[assignment]
    return copy


@pytest.fixture(scope="session")
def provider() -> SyntheticProvider:
    """A seeded synthetic data provider."""
    return SyntheticProvider(seed=20240101)


@pytest.fixture(scope="session")
def panel(provider: SyntheticProvider, base_config: AtlasConfig) -> PricePanel:
    """A price panel for the full configured universe.

    Built directly from the provider (no on-disk cache) so tests never depend on
    or pollute the developer's cache directory.
    """
    frame = provider.get_historical_prices(base_config.universe.symbols, PANEL_START, PANEL_END)
    from atlas.data.loader import long_to_panel

    return long_to_panel(frame)


@pytest.fixture(scope="session")
def short_panel(provider: SyntheticProvider) -> PricePanel:
    """A small three-asset panel, for fast unit tests."""
    from atlas.data.loader import long_to_panel

    frame = provider.get_historical_prices(
        ["SPY", "TLT", "GLD"], dt.date(2010, 1, 4), dt.date(2013, 12, 31)
    )
    return long_to_panel(frame)


@pytest.fixture(scope="session")
def features(panel: PricePanel, base_config: AtlasConfig) -> FeatureSet:
    """Features built from the session panel."""
    return FeatureEngineer.from_config(base_config).build(panel)


@pytest.fixture
def tmp_database(tmp_path: Path):
    """An isolated SQLite database for one test."""
    from atlas.database import DatabaseManager

    manager = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
    yield manager
    manager.dispose()


@pytest.fixture
def loader(provider: SyntheticProvider, tmp_path: Path) -> MarketDataLoader:
    """A loader whose cache lives in a temporary directory."""
    return MarketDataLoader(provider, tmp_path / "cache")


@pytest.fixture
def simple_prices() -> pd.DataFrame:
    """A tiny, fully deterministic price frame with known properties.

    ``UP`` rises 1% per day, ``DOWN`` falls 1% per day, ``FLAT`` never moves.
    Strategy sign conventions can be asserted exactly against this.
    """
    index = pd.bdate_range("2020-01-01", periods=400, name="date")
    n = len(index)
    return pd.DataFrame(
        {
            "UP": 100.0 * (1.01 ** np.arange(n)),
            "DOWN": 100.0 * (0.99 ** np.arange(n)),
            "FLAT": np.full(n, 100.0),
        },
        index=index,
    )


@pytest.fixture
def flat_returns() -> pd.Series:
    """A zero-return series, for degenerate-input tests."""
    index = pd.bdate_range("2020-01-01", periods=300, name="date")
    return pd.Series(0.0, index=index)


@pytest.fixture
def known_returns() -> pd.Series:
    """A return series with an exactly known total return and drawdown.

    Ten days of +10%, then five of -10%: total return
    ``1.1**10 * 0.9**5 - 1``, and a maximum drawdown of ``1 - 0.9**5``.
    """
    index = pd.bdate_range("2021-01-04", periods=15, name="date")
    values = [0.10] * 10 + [-0.10] * 5
    return pd.Series(values, index=index)
