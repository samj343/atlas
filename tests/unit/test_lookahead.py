"""Look-ahead-bias tests.

These are the most important tests in Atlas. Every one works the same way:
compute something, then *change the future*, recompute, and assert the past did
not move. If a rolling window is mis-shifted or a normalisation secretly uses
the full sample, these fail.

Covered:

* features are unaffected by future prices;
* strategy signals are unaffected by future prices;
* regime labels are unaffected by future prices;
* ``as_of`` truncation reproduces real-time computation exactly;
* walk-forward training windows never overlap validation or test windows;
* a signal formed on the close of day *t* is not executed at that same close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.engine import BacktestEngine
from atlas.config import ExecutionTiming
from atlas.data.features import FeatureEngineer
from atlas.data.loader import PricePanel
from atlas.regimes.rules_based import RulesBasedRegimeDetector
from atlas.strategies.base import build_strategies
from atlas.validation.walk_forward import WalkForwardValidator

# The point after which prices are perturbed, as a fraction of the sample.
SPLIT_FRACTION = 0.7


def _perturb_future(panel: PricePanel, split: pd.Timestamp, factor: float = 1.5) -> PricePanel:
    """Return a copy of ``panel`` with every price after ``split`` multiplied.

    A 50% shift in every future price is far larger than any real revision, so a
    feature that leaks even slightly will move detectably.
    """
    fields = ("open", "high", "low", "close", "adj_close", "volume")
    changed = {}
    for name in fields:
        frame = panel.field(name).copy()
        mask = frame.index > split
        if name == "volume":
            changed[name] = frame
            continue
        frame.loc[mask] = frame.loc[mask] * factor
        changed[name] = frame
    return PricePanel(
        **changed, source=dict(panel.source), downloaded_at=dict(panel.downloaded_at)
    )


@pytest.fixture(scope="module")
def split_point(panel: PricePanel) -> pd.Timestamp:
    """A date roughly 70% through the sample."""
    return panel.dates[int(len(panel.dates) * SPLIT_FRACTION)]


class TestFeatureLookahead:
    """Features must depend only on data at or before their own timestamp."""

    def test_features_ignore_future_prices(self, panel, base_config, split_point):
        engineer = FeatureEngineer.from_config(base_config)
        original = engineer.build(panel)
        perturbed = engineer.build(_perturb_future(panel, split_point))

        past = panel.dates <= split_point
        offenders: list[str] = []

        for name in original.names:
            a = original.get(name).loc[past]
            b = perturbed.get(name).loc[past]
            if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-12):
                offenders.append(name)

        assert not offenders, (
            f"these features changed in the past when future prices changed, which means they "
            f"leak future information: {offenders}"
        )

    def test_market_features_ignore_future_prices(self, panel, base_config, split_point):
        engineer = FeatureEngineer.from_config(base_config)
        original = engineer.build(panel).market
        perturbed = engineer.build(_perturb_future(panel, split_point)).market

        past = original.index <= split_point
        for column in original.columns:
            a = original.loc[past, column].to_numpy()
            b = perturbed.loc[past, column].to_numpy()
            assert np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12), (
                f"market feature {column!r} changed in the past when future prices changed"
            )

    def test_as_of_matches_full_sample(self, panel, base_config, split_point):
        """Truncating input at *t* must give the same values as slicing output at *t*."""
        engineer = FeatureEngineer.from_config(base_config)
        full = engineer.build(panel)
        truncated = engineer.build(panel.as_of(split_point))

        for name in ("return_21d", "volatility_20d", "price_to_ma_200", "rsi"):
            a = full.get(name).loc[full.index <= split_point]
            b = truncated.get(name)
            common = a.index.intersection(b.index)
            assert len(common) > 100
            assert np.allclose(
                a.loc[common].to_numpy(), b.loc[common].to_numpy(), equal_nan=True, atol=1e-10
            ), f"feature {name!r} differs between a full run and an as-of run"

    def test_rolling_windows_are_backward_looking(self, simple_prices):
        """A moving average at t must equal the mean of the last N closes up to t."""
        from atlas.data.features import moving_average

        ma = moving_average(simple_prices, 20)
        for position in (25, 100, 399):
            timestamp = simple_prices.index[position]
            expected = simple_prices["UP"].iloc[position - 19 : position + 1].mean()
            assert ma.loc[timestamp, "UP"] == pytest.approx(expected)

    def test_shifted_view_is_strictly_lagged(self, panel, base_config):
        engineer = FeatureEngineer.from_config(base_config)
        features = engineer.build(panel)
        lagged = features.shifted(1)
        name = "return_21d"
        original = features.get(name)
        shifted = lagged.get(name)
        assert np.allclose(
            original.iloc[:-1].to_numpy(), shifted.iloc[1:].to_numpy(), equal_nan=True
        )


class TestStrategyLookahead:
    """Strategy signals must not respond to future prices."""

    def test_signals_ignore_future_prices(self, panel, base_config, features, split_point):
        engineer = FeatureEngineer.from_config(base_config)
        perturbed_panel = _perturb_future(panel, split_point)
        perturbed_features = engineer.build(perturbed_panel)
        past = panel.dates <= split_point

        for strategy in build_strategies(base_config):
            original = strategy.compute(panel.adj_close, features).signal.loc[past]
            perturbed = strategy.compute(
                perturbed_panel.adj_close, perturbed_features
            ).signal.loc[past]
            assert np.allclose(
                original.to_numpy(), perturbed.to_numpy(), equal_nan=True, atol=1e-9
            ), (
                f"strategy {strategy.name!r} produced different historical signals after future "
                "prices changed, which means it leaks future information"
            )

    def test_as_of_signal_matches_full_run(self, panel, base_config, features, split_point):
        """A signal computed in real time on *t* equals the full-sample value at *t*."""
        for strategy in build_strategies(base_config):
            full = strategy.compute(panel.adj_close, features).signal
            realtime = strategy.compute(
                panel.adj_close, features, as_of_date=split_point
            ).signal
            assert realtime.index.max() == split_point
            common = full.index.intersection(realtime.index)
            assert len(common) > 100
            assert np.allclose(
                full.loc[common].to_numpy(),
                realtime.loc[common].to_numpy(),
                equal_nan=True,
                atol=1e-9,
            ), f"strategy {strategy.name!r} differs between a full run and an as-of run"

    def test_ensemble_ignores_future(self, panel, base_config, features, split_point):
        from atlas.strategies.ensemble import SignalEnsemble

        engineer = FeatureEngineer.from_config(base_config)
        perturbed_panel = _perturb_future(panel, split_point)
        perturbed_features = engineer.build(perturbed_panel)
        detector = RulesBasedRegimeDetector(base_config.regimes)
        strategies = build_strategies(base_config)

        def _combined(p, f):
            ensemble = SignalEnsemble(
                strategies, base_config.strategies.ensemble, regime_config=base_config.regimes
            )
            signals = {s.name: s.compute(p.adj_close, f) for s in strategies}
            return ensemble.combine(signals, regimes=detector.detect(p, f)).signal

        past = panel.dates <= split_point
        original = _combined(panel, features).loc[past]
        perturbed = _combined(perturbed_panel, perturbed_features).loc[past]
        assert np.allclose(original.to_numpy(), perturbed.to_numpy(), equal_nan=True, atol=1e-9)


class TestRegimeLookahead:
    """Regime labels must be causal."""

    def test_labels_ignore_future_prices(self, panel, base_config, split_point):
        detector = RulesBasedRegimeDetector(base_config.regimes)
        original = detector.detect(panel).as_strings()
        perturbed = detector.detect(_perturb_future(panel, split_point)).as_strings()

        past = original.index <= split_point
        mismatches = (original.loc[past] != perturbed.loc[past]).sum()
        assert mismatches == 0, (
            f"{mismatches} historical regime label(s) changed when future prices changed"
        )

    def test_volatility_percentile_is_expanding(self, panel, base_config):
        """The volatility percentile at *t* uses only observations up to *t*."""
        detector = RulesBasedRegimeDetector(base_config.regimes)
        full = detector.detect(panel).features["volatility_percentile"]
        cut = panel.dates[int(len(panel.dates) * 0.6)]
        truncated = detector.detect(panel.as_of(cut)).features["volatility_percentile"]
        common = full.index.intersection(truncated.index)
        assert np.allclose(
            full.loc[common].to_numpy(), truncated.loc[common].to_numpy(), equal_nan=True, atol=1e-12
        )

    def test_persistence_filter_only_delays(self, base_config):
        """The persistence filter may postpone a change, never anticipate it."""
        from atlas.config import RegimeLabel

        index = pd.bdate_range("2020-01-01", periods=20)
        raw = pd.Series([RegimeLabel.BULL_LOW_VOL] * 10 + [RegimeLabel.BEAR_HIGH_VOL] * 10, index=index)
        filtered = RulesBasedRegimeDetector._apply_persistence(raw, min_days=3)
        # The new regime is accepted no earlier than the raw switch.
        first_bear_raw = next(i for i, v in enumerate(raw) if v == RegimeLabel.BEAR_HIGH_VOL)
        first_bear_filtered = next(
            (i for i, v in enumerate(filtered) if v == RegimeLabel.BEAR_HIGH_VOL), len(filtered)
        )
        assert first_bear_filtered >= first_bear_raw


class TestWalkForwardIsolation:
    """Training data must never overlap validation or test data."""

    def test_windows_do_not_overlap(self, base_config, panel):
        validator = WalkForwardValidator(base_config)
        base_config.validation.walk_forward.train_years = 3
        base_config.validation.walk_forward.validation_years = 1
        base_config.validation.walk_forward.test_years = 1
        windows = validator.build_windows(panel.dates)
        assert windows, "expected at least one walk-forward window"

        for window in windows:
            assert window.train_end < window.validation_start
            assert window.validation_end < window.test_start
            assert window.test_start <= window.test_end

    def test_embargo_gap_is_applied(self, base_config, panel):
        base_config.validation.walk_forward.train_years = 3
        base_config.validation.walk_forward.embargo_days = 21
        validator = WalkForwardValidator(base_config)
        for window in validator.build_windows(panel.dates):
            gap = (window.validation_start - window.train_end).days
            assert gap >= 21, "the embargo gap between train and validation is missing"
            gap = (window.test_start - window.validation_end).days
            assert gap >= 21, "the embargo gap between validation and test is missing"

    def test_test_windows_advance_forward(self, base_config, panel):
        base_config.validation.walk_forward.train_years = 3
        validator = WalkForwardValidator(base_config)
        windows = validator.build_windows(panel.dates)
        starts = [w.test_start for w in windows]
        assert starts == sorted(starts), "walk-forward windows must move forward in time"


class TestExecutionTiming:
    """A signal from the close of day t must not be executed at that same close."""

    def test_orders_fill_after_the_signal_bar(self, config, short_panel):
        config.backtest.rebalance_frequency = "weekly"  # type: ignore[assignment]
        config.backtest.execution_timing = ExecutionTiming.NEXT_OPEN
        result = BacktestEngine(config).run(short_panel)

        assert not result.fills.empty, "expected at least one fill"
        fills = result.fills.copy()
        fills["fill_date"] = pd.DatetimeIndex(fills["timestamp"]).normalize()

        # Orders are created at a rebalance date's close and execute on the
        # following bar, so no fill may land on a rebalance date.
        rebalance_dates = set(pd.DatetimeIndex(result.target_weights.index))
        fill_dates = set(fills["fill_date"])
        assert not (fill_dates & rebalance_dates), (
            "a fill occurred on the same bar as the signal that produced it - "
            "this is same-close execution and is not achievable in practice"
        )

        # And each individual fill must be strictly later than its own signal bar.
        trading_days = list(short_panel.dates)
        for _, row in fills.iterrows():
            preceding = [d for d in rebalance_dates if d < row["fill_date"]]
            assert preceding, f"fill on {row['fill_date']} has no earlier rebalance date"
            signal_day = max(preceding)
            assert trading_days.index(row["fill_date"]) > trading_days.index(signal_day)

    def test_unrealistic_mode_is_labelled(self, config, short_panel):
        config.backtest.execution_timing = ExecutionTiming.SAME_CLOSE_UNREALISTIC
        result = BacktestEngine(config).run(short_panel)
        assert any("unrealistic" in w.lower() for w in result.warnings), (
            "same-close execution must be labelled as unrealistic in the result"
        )

    def test_next_open_uses_the_open_price(self, config, short_panel):
        """Fills reference the next bar's open, not the signal bar's close."""
        config.backtest.execution_timing = ExecutionTiming.NEXT_OPEN
        result = BacktestEngine(config).run(short_panel)
        assert not result.fills.empty

        row = result.fills.iloc[0]
        fill_date = pd.Timestamp(row["timestamp"]).normalize()
        expected_open = short_panel.open.loc[fill_date, row["symbol"]]
        # The reference price is the raw bar price before costs are applied.
        assert row["reference_price"] == pytest.approx(expected_open, rel=1e-6)
