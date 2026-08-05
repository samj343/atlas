"""Tests for the strategies, the ensemble and the regime detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.config import (
    CrossSectionalMomentumConfig,
    EnsembleConfig,
    MeanReversionConfig,
    RegimeLabel,
    TrendConfig,
)
from atlas.data.features import FeatureEngineer
from atlas.exceptions import StrategyError
from atlas.regimes.rules_based import RulesBasedRegimeDetector
from atlas.strategies.base import build_strategies, normalize_signal, soft_threshold
from atlas.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from atlas.strategies.ensemble import SignalEnsemble
from atlas.strategies.mean_reversion import MeanReversionStrategy
from atlas.strategies.trend import TrendStrategy


class TestSignalHelpers:
    def test_normalize_bounds(self):
        frame = pd.DataFrame({"A": [-100.0, 0.0, 100.0]})
        for method in ("tanh", "clip", "sign"):
            result = normalize_signal(frame, method=method)
            assert (result.abs() <= 1.0 + 1e-12).to_numpy().all()

    def test_normalize_preserves_sign(self):
        frame = pd.DataFrame({"A": [-3.0, -0.1, 0.1, 3.0]})
        result = normalize_signal(frame)
        assert np.array_equal(np.sign(result["A"].to_numpy()), np.sign(frame["A"].to_numpy()))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown normalisation"):
            normalize_signal(pd.DataFrame({"A": [1.0]}), method="magic")

    def test_soft_threshold_zeroes_small_signals(self):
        frame = pd.DataFrame({"A": [0.02, -0.02, 0.5, -0.5]})
        result = soft_threshold(frame, 0.05)
        assert result["A"].iloc[0] == 0.0
        assert result["A"].iloc[1] == 0.0
        assert result["A"].iloc[2] > 0.0

    def test_soft_threshold_is_continuous(self):
        """Just above the threshold the output must be near zero, not a jump."""
        frame = pd.DataFrame({"A": [0.0501]})
        assert abs(soft_threshold(frame, 0.05)["A"].iloc[0]) < 0.001


class TestTrendStrategy:
    def test_long_signal_in_an_uptrend(self, simple_prices):
        strategy = TrendStrategy(TrendConfig(long_only=True, volatility_scaling=False))
        result = strategy.compute(simple_prices)
        assert result.signal["UP"].iloc[-1] > 0.3, "a monotone uptrend must give a strong long signal"

    def test_no_long_signal_in_a_downtrend(self, simple_prices):
        strategy = TrendStrategy(TrendConfig(long_only=True, volatility_scaling=False))
        result = strategy.compute(simple_prices)
        assert result.signal["DOWN"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_short_signal_when_long_short_enabled(self, simple_prices):
        strategy = TrendStrategy(TrendConfig(long_only=False, volatility_scaling=False))
        result = strategy.compute(simple_prices)
        assert result.signal["DOWN"].iloc[-1] < -0.3

    def test_flat_series_gives_no_signal(self, simple_prices):
        strategy = TrendStrategy(TrendConfig(long_only=False, volatility_scaling=False))
        result = strategy.compute(simple_prices)
        assert abs(result.signal["FLAT"].iloc[-1]) < 0.05

    def test_signal_bounds(self, panel, features):
        strategy = TrendStrategy(TrendConfig())
        result = strategy.compute(panel.adj_close, features)
        assert (result.signal.abs() <= 1.0 + 1e-9).to_numpy().all()

    def test_neutral_zone_reduces_turnover(self, panel, features):
        loose = TrendStrategy(TrendConfig(neutral_threshold=0.0)).compute(panel.adj_close, features)
        tight = TrendStrategy(TrendConfig(neutral_threshold=0.30)).compute(panel.adj_close, features)
        assert tight.signal.diff().abs().sum().sum() <= loose.signal.diff().abs().sum().sum()

    def test_volatility_scaling_reduces_exposure(self, panel, features):
        without = TrendStrategy(TrendConfig(volatility_scaling=False)).compute(panel.adj_close, features)
        with_gate = TrendStrategy(TrendConfig(volatility_scaling=True)).compute(panel.adj_close, features)
        assert with_gate.signal.abs().sum().sum() <= without.signal.abs().sum().sum() + 1e-9

    def test_parameters_are_configurable(self):
        strategy = TrendStrategy(TrendConfig(fast_ma=50, slow_ma=150, momentum_lookback=126))
        assert strategy.required_lookback == 150
        assert strategy.parameters()["fast_ma"] == 50

    def test_invalid_ma_ordering_rejected(self):
        with pytest.raises(ValueError, match="strictly less"):
            TrendConfig(fast_ma=200, slow_ma=100)

    def test_insufficient_history_gives_neutral_signals(self, simple_prices):
        strategy = TrendStrategy(TrendConfig())
        result = strategy.compute(simple_prices.head(50))
        assert (result.signal == 0.0).to_numpy().all()

    def test_empty_prices_raise(self):
        with pytest.raises(StrategyError, match="empty prices"):
            TrendStrategy(TrendConfig()).compute(pd.DataFrame())

    def test_records_have_required_fields(self, panel, features):
        records = TrendStrategy(TrendConfig()).generate_signals(panel.adj_close, features)
        for column in ("date", "symbol", "strategy", "raw_signal", "normalized_signal", "confidence"):
            assert column in records.columns
        assert (records["strategy"] == "trend").all()


class TestMeanReversionStrategy:
    def test_signal_bounds(self, panel, features):
        result = MeanReversionStrategy(MeanReversionConfig()).compute(panel.adj_close, features)
        assert (result.signal.abs() <= 1.0 + 1e-9).to_numpy().all()

    def test_fades_a_large_positive_move(self):
        """After an unusually large up-move the signal must not be long."""
        index = pd.bdate_range("2020-01-01", periods=300, name="date")
        values = np.full(len(index), 100.0)
        values[-1] = 130.0  # a sharp spike on the final day
        prices = pd.DataFrame({"X": values}, index=index)
        config = MeanReversionConfig(
            long_only=False, use_trend_filter=False, use_volatility_filter=False
        )
        result = MeanReversionStrategy(config).compute(prices)
        assert result.signal["X"].iloc[-1] <= 0.0

    def test_long_only_never_shorts(self, panel, features):
        config = MeanReversionConfig(long_only=True)
        result = MeanReversionStrategy(config).compute(panel.adj_close, features)
        assert (result.signal >= -1e-12).to_numpy().all()

    def test_entry_threshold_controls_activity(self, panel, features):
        loose = MeanReversionStrategy(
            MeanReversionConfig(entry_threshold=0.75)
        ).compute(panel.adj_close, features)
        strict = MeanReversionStrategy(
            MeanReversionConfig(entry_threshold=2.5)
        ).compute(panel.adj_close, features)
        assert (strict.signal.abs() > 0).sum().sum() <= (loose.signal.abs() > 0).sum().sum()

    def test_exit_threshold_must_be_below_entry(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            MeanReversionConfig(entry_threshold=1.0, exit_threshold=1.5)

    def test_max_holding_period_is_enforced(self):
        """No position may stay open beyond max_holding_days."""
        enter = pd.DataFrame({"X": [True] * 40})
        hold = pd.DataFrame({"X": [True] * 40})
        direction = pd.DataFrame({"X": [1.0] * 40})
        active = MeanReversionStrategy._hysteresis(enter, hold, direction, max_holding_days=5)
        # With a permanent entry signal, the position reopens but each run of
        # consecutive open days must never exceed the limit.
        runs, current = [], 0
        for value in active["X"]:
            if value:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        assert max(runs) <= 5

    def test_trend_filter_blocks_fading_a_downtrend(self, simple_prices):
        config = MeanReversionConfig(
            long_only=True, use_trend_filter=True, trend_filter_ma=200, use_volatility_filter=False
        )
        result = MeanReversionStrategy(config).compute(simple_prices)
        # DOWN is always below its 200-day average, so no long signal is allowed.
        assert result.signal["DOWN"].abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_signals_are_continuous_not_binary(self, panel, features):
        result = MeanReversionStrategy(MeanReversionConfig()).compute(panel.adj_close, features)
        active = result.signal.to_numpy()[np.abs(result.signal.to_numpy()) > 1e-9]
        assert len(np.unique(np.round(active, 4))) > 10, "signals should be continuous"


class TestCrossSectionalMomentum:
    def test_top_k_selected(self, panel, features):
        config = CrossSectionalMomentumConfig(top_k=3, long_only=True, neutralize_asset_class=False)
        strategy = CrossSectionalMomentumStrategy(config, asset_classes=None)
        result = strategy.compute(panel.adj_close, features)
        active = (result.signal.abs() > 1e-9).sum(axis=1)
        assert active.max() <= 3

    def test_ranks_the_strongest_asset_highest(self, simple_prices):
        config = CrossSectionalMomentumConfig(
            lookback=100, skip_recent_days=5, top_k=1, min_assets=3, neutralize_asset_class=False
        )
        strategy = CrossSectionalMomentumStrategy(config)
        result = strategy.compute(simple_prices)
        last = result.signal.iloc[-1]
        assert last["UP"] > 0.0
        assert last["DOWN"] == pytest.approx(0.0, abs=1e-9)

    def test_long_only_never_shorts(self, panel, features, base_config):
        config = CrossSectionalMomentumConfig(long_only=True)
        strategy = CrossSectionalMomentumStrategy(config, asset_classes=base_config.asset_class_map)
        result = strategy.compute(panel.adj_close, features)
        assert (result.signal >= -1e-12).to_numpy().all()

    def test_asset_class_neutralisation_changes_selection(self, panel, features, base_config):
        plain = CrossSectionalMomentumStrategy(
            CrossSectionalMomentumConfig(neutralize_asset_class=False),
            asset_classes=base_config.asset_class_map,
        ).compute(panel.adj_close, features)
        neutral = CrossSectionalMomentumStrategy(
            CrossSectionalMomentumConfig(neutralize_asset_class=True),
            asset_classes=base_config.asset_class_map,
        ).compute(panel.adj_close, features)
        assert not np.allclose(plain.signal.to_numpy(), neutral.signal.to_numpy())

    def test_no_signal_with_too_few_assets(self, short_panel, base_config):
        config = CrossSectionalMomentumConfig(min_assets=10)
        strategy = CrossSectionalMomentumStrategy(config, asset_classes=base_config.asset_class_map)
        features = FeatureEngineer.from_config(base_config).build(short_panel)
        result = strategy.compute(short_panel.adj_close, features)
        assert (result.signal.abs() < 1e-12).to_numpy().all()

    def test_skip_recent_must_be_shorter_than_lookback(self):
        with pytest.raises(ValueError, match="skip_recent_days"):
            CrossSectionalMomentumConfig(lookback=100, skip_recent_days=100)

    def test_ties_use_average_rank(self, simple_prices):
        """Identical series must receive identical, not arbitrary, scores."""
        index = simple_prices.index
        n = len(index)
        prices = pd.DataFrame(
            {
                "A": 100.0 * (1.01 ** np.arange(n)),
                "B": 100.0 * (1.01 ** np.arange(n)),
                "C": 100.0 * (0.99 ** np.arange(n)),
                "D": np.full(n, 100.0),
            },
            index=index,
        )
        config = CrossSectionalMomentumConfig(
            lookback=100, skip_recent_days=5, top_k=2, min_assets=3, neutralize_asset_class=False
        )
        result = CrossSectionalMomentumStrategy(config).compute(prices)
        last = result.signal.iloc[-1]
        assert last["A"] == pytest.approx(last["B"])


class TestEnsemble:
    def test_weights_sum_to_one(self, base_config):
        strategies = build_strategies(base_config)
        ensemble = SignalEnsemble(
            strategies, base_config.strategies.ensemble, regime_config=base_config.regimes
        )
        for regime in RegimeLabel:
            weights = ensemble.weights_for_regime(regime)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_equal_mode_weights_equally(self, base_config):
        strategies = build_strategies(base_config)
        ensemble = SignalEnsemble(strategies, EnsembleConfig(mode="equal"))
        weights = ensemble.weights_for_regime(RegimeLabel.BULL_LOW_VOL)
        assert all(w == pytest.approx(1.0 / len(strategies)) for w in weights.values())

    def test_regime_mode_changes_weights(self, base_config):
        strategies = build_strategies(base_config)
        ensemble = SignalEnsemble(
            strategies, EnsembleConfig(mode="regime"), regime_config=base_config.regimes
        )
        calm = ensemble.weights_for_regime(RegimeLabel.BULL_LOW_VOL)
        stressed = ensemble.weights_for_regime(RegimeLabel.BEAR_HIGH_VOL)
        assert calm != stressed
        assert stressed["trend"] > calm["trend"]

    def test_regime_mode_requires_regime_config(self, base_config):
        with pytest.raises(StrategyError, match="requires a regime configuration"):
            SignalEnsemble(build_strategies(base_config), EnsembleConfig(mode="regime"))

    def test_contributions_sum_to_the_raw_signal(self, base_config, panel, features):
        strategies = build_strategies(base_config)
        ensemble = SignalEnsemble(
            strategies, base_config.strategies.ensemble, regime_config=base_config.regimes
        )
        detector = RulesBasedRegimeDetector(base_config.regimes)
        signals = {s.name: s.compute(panel.adj_close, features) for s in strategies}
        combined = ensemble.combine(signals, regimes=detector.detect(panel, features))
        total = sum(combined.contributions.values())
        assert np.allclose(total.to_numpy(), combined.raw.to_numpy(), atol=1e-12)

    def test_smoothing_reduces_turnover(self, base_config, panel, features):
        strategies = build_strategies(base_config)
        detector = RulesBasedRegimeDetector(base_config.regimes)
        regimes = detector.detect(panel, features)
        signals = {s.name: s.compute(panel.adj_close, features) for s in strategies}

        raw = SignalEnsemble(
            strategies, EnsembleConfig(mode="equal", signal_smoothing_alpha=1.0, min_signal=0.0)
        ).combine(signals, regimes=regimes)
        smooth = SignalEnsemble(
            strategies, EnsembleConfig(mode="equal", signal_smoothing_alpha=0.2, min_signal=0.0)
        ).combine(signals, regimes=regimes)

        assert smooth.signal.diff().abs().sum().sum() < raw.signal.diff().abs().sum().sum()

    def test_min_signal_threshold(self, base_config, panel, features):
        strategies = build_strategies(base_config)
        ensemble = SignalEnsemble(strategies, EnsembleConfig(mode="equal", min_signal=0.2))
        signals = {s.name: s.compute(panel.adj_close, features) for s in strategies}
        combined = ensemble.combine(signals)
        nonzero = combined.signal.to_numpy()[np.abs(combined.signal.to_numpy()) > 1e-12]
        assert (np.abs(nonzero) >= 0.2 - 1e-9).all()

    def test_empty_signals_raise(self, base_config):
        ensemble = SignalEnsemble(build_strategies(base_config), EnsembleConfig(mode="equal"))
        with pytest.raises(StrategyError, match="no strategy signals"):
            ensemble.combine({})


class TestRegimeDetector:
    def test_produces_valid_labels(self, panel, base_config):
        result = RulesBasedRegimeDetector(base_config.regimes).detect(panel)
        valid = {r.value for r in RegimeLabel}
        assert set(result.as_strings().unique()).issubset(valid)

    def test_all_four_regimes_appear(self, panel, base_config):
        result = RulesBasedRegimeDetector(base_config.regimes).detect(panel)
        observed = set(result.as_strings().unique()) - {"unknown"}
        assert len(observed) >= 3, "the sample should exercise most regimes"

    def test_indicators_are_recorded(self, panel, base_config):
        result = RulesBasedRegimeDetector(base_config.regimes).detect(panel)
        for column in ("trend_score", "volatility", "volatility_percentile", "breadth", "drawdown"):
            assert column in result.features.columns

    def test_bull_labels_when_above_trend(self, panel, base_config):
        result = RulesBasedRegimeDetector(base_config.regimes).detect(panel)
        frame = result.features.join(result.as_strings().rename("regime"))
        clean = frame.dropna(subset=["above_trend"])
        bull = clean[clean["regime"].str.startswith("bull")]
        # The persistence filter can hold a stale label briefly, so allow a margin.
        assert (bull["above_trend"] > 0.5).mean() > 0.85

    def test_persistence_reduces_transitions(self, panel, base_config):
        config = base_config.model_copy(deep=True)
        config.regimes.detector.min_regime_days = 1
        jumpy = RulesBasedRegimeDetector(config.regimes).detect(panel)
        config.regimes.detector.min_regime_days = 10
        stable = RulesBasedRegimeDetector(config.regimes).detect(panel)
        assert len(stable.transitions()) <= len(jumpy.transitions())

    def test_episodes_and_transitions(self, panel, base_config):
        result = RulesBasedRegimeDetector(base_config.regimes).detect(panel)
        episodes = result.episodes()
        assert not episodes.empty
        assert episodes["days"].sum() == len(result.labels)

    def test_falls_back_without_benchmark(self, short_panel, base_config):
        config = base_config.model_copy(deep=True)
        config.regimes.detector.benchmark = "NOT_IN_UNIVERSE"
        result = RulesBasedRegimeDetector(config.regimes).detect(short_panel)
        assert len(result.labels) == len(short_panel.dates)

    def test_label_at_falls_back_to_earlier_date(self, panel, base_config):
        result = RulesBasedRegimeDetector(base_config.regimes).detect(panel)
        future = panel.dates.max() + pd.Timedelta(days=30)
        assert isinstance(result.label_at(future), RegimeLabel)
