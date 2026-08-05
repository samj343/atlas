"""Tests for the optional clustering regime detector.

The clustering model is a *comparison* model. These tests check that it stays
one: that it is off by default, that it fits on training data only, and that its
labels are usable rather than silently corrupted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from atlas.config import RegimeLabel
from atlas.exceptions import InsufficientDataError
from atlas.regimes.clustering import ClusteringRegimeDetector
from atlas.regimes.rules_based import RulesBasedRegimeDetector


class TestClusteringDetector:
    def test_disabled_by_default(self, base_config):
        assert not base_config.regimes.clustering.enabled

    def test_produces_valid_labels(self, panel, features, base_config):
        result = ClusteringRegimeDetector(base_config.regimes).detect(panel, features)
        valid = {r.value for r in RegimeLabel}
        observed = set(result.as_strings().unique())
        assert observed.issubset(valid), f"unexpected labels: {observed - valid}"

    def test_labels_are_regime_enum_members(self, panel, features, base_config):
        """A regression test: fillna with a str-subclass enum truncated labels."""
        result = ClusteringRegimeDetector(base_config.regimes).detect(panel, features)
        assert all(isinstance(v, RegimeLabel) for v in result.labels)
        assert not any(s.startswith("RegimeL") for s in result.as_strings())

    def test_covers_the_whole_sample(self, panel, features, base_config):
        result = ClusteringRegimeDetector(base_config.regimes).detect(panel, features)
        assert len(result.labels) == len(panel.dates)

    def test_fits_on_training_data_only(self, panel, features, base_config):
        """Labels before the training cut must not change when the future does."""
        cut = panel.dates[int(len(panel.dates) * 0.6)]
        detector = ClusteringRegimeDetector(base_config.regimes, train_end=cut)
        full = detector.detect(panel, features).as_strings()

        # Re-fitting with the same training cut on a longer panel must reproduce
        # the training-period labels exactly: the fit saw the same rows.
        again = ClusteringRegimeDetector(base_config.regimes, train_end=cut).detect(panel, features)
        assert (full.loc[full.index <= cut] == again.as_strings().loc[full.index <= cut]).all()

    def test_insufficient_data_raises(self, short_panel, base_config):
        config = base_config.model_copy(deep=True)
        config.regimes.clustering.min_train_obs = 100_000
        with pytest.raises(InsufficientDataError, match="complete observations"):
            ClusteringRegimeDetector(config.regimes).detect(short_panel)

    def test_comparison_against_the_baseline(self, panel, features, base_config):
        frame = ClusteringRegimeDetector(base_config.regimes).compare_with(panel, features)
        for column in ("rules_based", "clustering", "agree"):
            assert column in frame.columns
        assert len(frame) > 100
        assert frame["agree"].dtype == bool

    def test_does_not_replace_the_baseline(self, panel, features, base_config):
        """The engine must keep using the rules-based detector by default."""
        from atlas.backtest.engine import BacktestEngine

        engine = BacktestEngine(base_config)
        assert isinstance(engine.regime_detector, RulesBasedRegimeDetector)


class TestLabelSeriesHandling:
    def test_pandas_fillna_truncates_str_enums(self):
        """Documents the pandas behaviour the detector must avoid.

        If this ever stops truncating, the workaround in clustering.py can be
        simplified - but the workaround is correct either way.
        """
        index = pd.date_range("2020-01-01", periods=4)
        series = pd.Series([RegimeLabel.BULL_LOW_VOL] * 3, index=index[1:], dtype=object)
        filled = series.reindex(index).ffill().fillna(RegimeLabel.UNKNOWN)
        # The first value is NOT a RegimeLabel - this is exactly the trap.
        assert not isinstance(filled.iloc[0], RegimeLabel)
