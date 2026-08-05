"""Tests for the data layer: providers, the caching loader and validation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from atlas.data.loader import MarketDataLoader, PricePanel, long_to_panel, panel_to_long
from atlas.data.provider import (
    CANONICAL_COLUMNS,
    CSVProvider,
    SyntheticProvider,
    build_provider,
    empty_price_frame,
)
from atlas.data.validator import DataValidator, Severity
from atlas.exceptions import DataError


class TestProvider:
    def test_canonical_schema(self, provider):
        frame = provider.get_historical_prices(["SPY", "TLT"], dt.date(2015, 1, 1), dt.date(2015, 6, 30))
        assert list(frame.columns) == list(CANONICAL_COLUMNS)
        assert frame["symbol"].nunique() == 2
        assert frame["date"].is_monotonic_increasing or frame.groupby("symbol")["date"].is_monotonic_increasing.all()
        assert (frame["close"] > 0).all()

    def test_no_duplicate_symbol_dates(self, provider):
        frame = provider.get_historical_prices(["SPY"], dt.date(2015, 1, 1), dt.date(2016, 1, 1))
        assert not frame.duplicated(subset=["symbol", "date"]).any()

    def test_dates_are_bounded(self, provider):
        start, end = dt.date(2015, 3, 2), dt.date(2015, 9, 30)
        frame = provider.get_historical_prices(["SPY"], start, end)
        assert frame["date"].min().date() >= start
        assert frame["date"].max().date() <= end

    def test_deterministic_across_calls(self, provider):
        a = provider.get_historical_prices(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 12, 31))
        b = provider.get_historical_prices(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 12, 31))
        assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())

    def test_symbols_are_independent(self, provider):
        """A regression test: symbols once shared one path because of re-seeding."""
        frame = provider.get_historical_prices(
            ["SPY", "TLT", "GLD", "EEM"], dt.date(2010, 1, 1), dt.date(2015, 1, 1)
        )
        wide = frame.pivot(index="date", columns="symbol", values="adj_close")
        correlations = wide.pct_change().corr().to_numpy()
        off_diagonal = correlations[np.triu_indices_from(correlations, k=1)]
        assert (off_diagonal < 0.999).all(), "symbols must not share an identical price path"

    def test_batching_does_not_change_output(self, provider):
        """A regression test: window/batching must not alter a symbol's path."""
        together = provider.get_historical_prices(
            ["SPY", "TLT"], dt.date(2012, 1, 1), dt.date(2013, 1, 1)
        )
        alone = provider.get_historical_prices(["SPY"], dt.date(2012, 1, 1), dt.date(2013, 1, 1))
        assert np.allclose(
            together[together["symbol"] == "SPY"]["close"].to_numpy(), alone["close"].to_numpy()
        )

    def test_sub_window_is_a_slice_of_the_full_path(self, provider):
        """A regression test: an incremental refresh must not splice a new path."""
        full = provider.get_historical_prices(["SPY"], dt.date(2010, 1, 1), dt.date(2015, 1, 1))
        part = provider.get_historical_prices(["SPY"], dt.date(2013, 1, 1), dt.date(2015, 1, 1))
        merged = full.merge(part, on="date", suffixes=("_full", "_part"))
        assert len(merged) > 200
        assert np.allclose(merged["close_full"].to_numpy(), merged["close_part"].to_numpy())

    def test_synthetic_is_labelled(self, provider):
        frame = provider.get_historical_prices(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 3, 1))
        assert (frame["source"] == "synthetic").all()
        assert provider.metadata.is_synthetic

    def test_empty_symbol_list(self, provider):
        assert provider.get_historical_prices([], dt.date(2015, 1, 1), dt.date(2015, 2, 1)).empty

    def test_reversed_dates_raise(self, provider):
        with pytest.raises(DataError, match="precedes"):
            provider.get_historical_prices(["SPY"], dt.date(2015, 6, 1), dt.date(2015, 1, 1))

    def test_build_provider_factory(self):
        assert isinstance(build_provider("synthetic", seed=3), SyntheticProvider)
        with pytest.raises(DataError, match="unknown data provider"):
            build_provider("no_such_provider")

    def test_csv_provider_round_trip(self, tmp_path, provider):
        frame = provider.get_historical_prices(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 6, 1))
        frame.drop(columns=["symbol"]).to_csv(tmp_path / "SPY.csv", index=False)
        loaded = CSVProvider(tmp_path).get_historical_prices(
            ["SPY"], dt.date(2015, 1, 1), dt.date(2015, 6, 1)
        )
        assert len(loaded) == len(frame)
        assert np.allclose(loaded["close"].to_numpy(), frame["close"].to_numpy())

    def test_csv_provider_missing_directory(self):
        with pytest.raises(DataError, match="does not exist"):
            CSVProvider("/nonexistent/path").get_historical_prices(
                ["SPY"], dt.date(2015, 1, 1), dt.date(2015, 2, 1)
            )


class TestPanel:
    def test_round_trip(self, short_panel):
        rebuilt = long_to_panel(panel_to_long(short_panel))
        assert rebuilt.symbols == short_panel.symbols
        assert np.allclose(
            rebuilt.adj_close.to_numpy(), short_panel.adj_close.to_numpy(), equal_nan=True
        )

    def test_object_dtype_columns(self, short_panel):
        """Symbol columns are object-dtype for backtest-loop performance."""
        assert short_panel.close.columns.dtype == object

    def test_as_of_truncates(self, short_panel):
        cut = short_panel.dates[100]
        assert short_panel.as_of(cut).dates.max() == cut

    def test_select_reorders(self, short_panel):
        selected = short_panel.select(["GLD", "SPY"])
        assert selected.symbols == ["GLD", "SPY"]

    def test_returns_do_not_fabricate_zeros(self, short_panel):
        """A missing price must produce NaN, never a fabricated 0% return."""
        frame = short_panel.adj_close.copy()
        frame.iloc[50, 0] = np.nan
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        returns = panel.returns()
        assert np.isnan(returns.iloc[50, 0])
        assert np.isnan(returns.iloc[51, 0])

    def test_implied_dividends_non_negative(self, short_panel):
        dividends = short_panel.implied_dividends()
        assert (dividends.fillna(0.0) >= 0.0).to_numpy().all()

    def test_tradable_mask_requires_history(self, short_panel):
        mask = short_panel.tradable_mask(min_history=50)
        assert not mask.iloc[10].any()
        assert mask.iloc[100].all()

    def test_unknown_field_raises(self, short_panel):
        with pytest.raises(DataError, match="unknown price field"):
            short_panel.field("nonsense")

    def test_empty_long_frame(self):
        panel = long_to_panel(empty_price_frame())
        assert panel.symbols == []
        assert len(panel.dates) == 0


class TestLoaderCaching:
    def test_writes_and_reads_cache(self, loader):
        panel = loader.load(["SPY", "TLT"], dt.date(2015, 1, 1), dt.date(2016, 1, 1))
        assert loader.cache_path("SPY").is_file()
        again = loader.load(["SPY", "TLT"], dt.date(2015, 1, 1), dt.date(2016, 1, 1))
        assert np.allclose(
            panel.adj_close.to_numpy(), again.adj_close.to_numpy(), equal_nan=True
        )

    def test_incremental_extension_is_consistent(self, loader):
        """Extending a cached range must not create a discontinuity at the seam."""
        loader.load(["SPY"], dt.date(2012, 1, 1), dt.date(2014, 1, 1))
        extended = loader.load(["SPY"], dt.date(2012, 1, 1), dt.date(2016, 1, 1))
        returns = extended.returns()["SPY"].dropna()
        assert returns.abs().max() < 0.25, (
            "a cache-extension seam introduced an implausible one-day move"
        )

    def test_clear_cache(self, loader):
        loader.load(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 6, 1))
        assert loader.clear_cache() >= 1
        assert not loader.cache_path("SPY").is_file()

    def test_corrupt_cache_is_ignored(self, loader):
        loader.load(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 6, 1))
        loader.cache_path("SPY").write_text("this is not a valid cache file")
        panel = loader.load(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 6, 1))
        assert len(panel.dates) > 50

    def test_no_data_raises(self, tmp_path):
        """A provider that returns nothing for every symbol must fail loudly.

        The synthetic provider invents a path for any ticker, so this uses a CSV
        provider pointed at an empty directory instead.
        """
        (tmp_path / "csv").mkdir()
        csv_loader = MarketDataLoader(CSVProvider(tmp_path / "csv"), tmp_path / "cache")
        with pytest.raises(DataError, match="no data could be loaded"):
            csv_loader.load(["ZZZZ"], dt.date(2015, 1, 1), dt.date(2015, 2, 1))


class TestValidator:
    def test_clean_panel_passes(self, short_panel):
        report = DataValidator(min_history_days=100).validate(short_panel)
        assert report.is_valid
        assert not report.errors

    def test_detects_negative_prices(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[100, 0] = -5.0
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        assert not report.is_valid
        assert any(i.code == "non_positive_price" for i in report.errors)

    def test_detects_zero_prices(self, short_panel):
        frame = short_panel.close.copy()
        frame.iloc[80, 1] = 0.0
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=frame, adj_close=short_panel.adj_close, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        assert any(i.code == "non_positive_price" for i in report.errors)

    def test_detects_abnormal_gaps(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[120:, 0] = frame.iloc[120:, 0] * 0.4
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        assert any(i.code == "abnormal_return" for i in report.warnings)

    def test_detects_inverted_high_low(self, short_panel):
        high = short_panel.high.copy()
        high.iloc[60, 0] = short_panel.low.iloc[60, 0] - 10.0
        panel = PricePanel(
            open=short_panel.open, high=high, low=short_panel.low,
            close=short_panel.close, adj_close=short_panel.adj_close, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        assert any(i.code == "inverted_high_low" for i in report.errors)

    def test_detects_missing_observations(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[200:203, 0] = np.nan
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        assert any(i.code == "missing_observations" for i in report.issues)

    def test_long_gap_is_an_error(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[200:230, 0] = np.nan
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        report = DataValidator(max_gap_days=10).validate(panel)
        assert any(
            i.code == "missing_observations" and i.severity is Severity.ERROR for i in report.issues
        )

    def test_detects_unsupported_symbol(self, short_panel):
        report = DataValidator().validate(short_panel, expected_symbols=["SPY", "NOPE"])
        assert any(i.code == "unsupported_symbol" for i in report.errors)

    def test_detects_stale_series(self, short_panel):
        report = DataValidator(max_staleness_days=2).validate(
            short_panel, as_of=dt.date(2030, 1, 1)
        )
        assert any(i.code == "stale_series" for i in report.warnings)

    def test_detects_flat_series(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[100:130, 0] = frame.iloc[99, 0]
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        assert any(i.code == "flat_series" for i in report.warnings)

    def test_empty_panel_is_an_error(self):
        panel = long_to_panel(empty_price_frame())
        report = DataValidator().validate(panel)
        assert not report.is_valid
        assert any(i.code == "empty_panel" for i in report.errors)

    def test_clean_fills_short_gaps_only(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[100:102, 0] = np.nan   # short: fillable
        frame.iloc[200:220, 0] = np.nan   # long: must remain NaN
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        report = DataValidator().validate(panel)
        cleaned = DataValidator(max_fill_days=3).clean(panel, report)
        assert cleaned.adj_close.iloc[100:102, 0].notna().all()
        assert cleaned.adj_close.iloc[210, 0] != cleaned.adj_close.iloc[210, 0]  # NaN
        assert any(i.code == "gaps_filled" for i in report.issues)

    def test_clean_does_not_backfill_before_inception(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[:50, 0] = np.nan
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        cleaned = DataValidator().clean(panel)
        assert cleaned.adj_close.iloc[:50, 0].isna().all()

    def test_report_raises_on_error(self, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[10, 0] = -1.0
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        from atlas.exceptions import DataValidationError

        report = DataValidator().validate(panel)
        with pytest.raises(DataValidationError):
            report.raise_if_invalid()

    def test_report_renders(self, short_panel):
        report = DataValidator().validate(short_panel)
        text = report.render()
        assert "Atlas data validation report" in text
        assert "symbols_checked" in text
