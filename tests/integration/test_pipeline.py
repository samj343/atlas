"""End-to-end integration tests.

These exercise whole workflows rather than single functions:

* data -> features -> signals -> portfolio -> backtest -> metrics -> report;
* strategy -> portfolio weights, with every constraint respected on every day;
* backtest -> database -> retrieval;
* walk-forward across several windows.

They are slower than the unit tests and are marked accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.engine import BacktestEngine
from atlas.data.features import FeatureEngineer
from atlas.data.validator import DataValidator
from atlas.database import AtlasRepository
from atlas.reporting.export import export_backtest, load_manifest
from atlas.reporting.performance_report import ReportBundle, ResearchReport
from atlas.validation.benchmarks import BenchmarkSuite
from atlas.validation.stress_tests import StressTester, block_bootstrap, bootstrap_statistics
from atlas.validation.walk_forward import WalkForwardValidator, apply_overrides, expand_grid


@pytest.fixture(scope="module")
def backtest_result(base_config, panel):
    """One backtest shared across the integration tests in this module."""
    features = FeatureEngineer.from_config(base_config).build(panel)
    return BacktestEngine(base_config).run(panel, features=features)


class TestDataToBacktest:
    def test_full_workflow(self, base_config, panel):
        # 1. validate
        report = DataValidator(min_history_days=100).validate(
            panel, expected_symbols=base_config.universe.symbols
        )
        assert report.is_valid

        # 2. features
        features = FeatureEngineer.from_config(base_config).build(panel)
        assert len(features) > 20

        # 3. backtest
        result = BacktestEngine(base_config).run(panel, features=features)

        # 4. metrics
        assert result.metrics is not None
        assert len(result.equity_curve) > 500
        assert result.final_equity > 0
        assert np.isfinite(result.metrics.sharpe)

    def test_equity_accounting_is_consistent(self, backtest_result):
        """Equity must always equal cash plus position value."""
        snapshots = backtest_result.snapshots
        difference = (snapshots["equity"] - snapshots["cash"] - snapshots["positions_value"]).abs()
        assert difference.max() < 1e-6

    def test_returns_match_the_equity_curve(self, backtest_result):
        rebuilt = backtest_result.equity_curve.pct_change(fill_method=None).dropna()
        assert np.allclose(rebuilt.to_numpy(), backtest_result.returns.to_numpy(), atol=1e-12)

    def test_every_day_respects_the_limits(self, backtest_result, base_config):
        """Not one day may breach a position or exposure limit."""
        from atlas.portfolio.constraints import PortfolioConstraints

        constraints = PortfolioConstraints(base_config.portfolio, base_config.asset_class_map)
        weights = backtest_result.weights
        # Realised weights drift between rebalances, so allow a small tolerance
        # above the target caps for market movement.
        tolerance = 1.15
        assert (weights.abs().max().max()) <= base_config.portfolio.max_asset_weight * tolerance
        gross = weights.abs().sum(axis=1)
        cap = min(base_config.portfolio.max_gross_exposure, base_config.portfolio.max_leverage)
        assert gross.max() <= cap * tolerance
        del constraints

    def test_costs_are_charged(self, backtest_result):
        assert backtest_result.total_costs > 0
        breakdown = backtest_result.cost_breakdown()
        assert breakdown["commission"] > 0
        assert breakdown.sum() == pytest.approx(backtest_result.total_costs)

    def test_gross_beats_net(self, backtest_result):
        drag = backtest_result.cost_drag()
        assert drag["gross_cagr"] >= drag["net_cagr"] - 1e-12
        assert drag["cost_drag_annual"] >= 0

    def test_no_trades_before_the_warmup(self, backtest_result, base_config):
        first_fill = pd.Timestamp(backtest_result.fills["timestamp"].min())
        assert first_fill >= backtest_result.equity_curve.index.min()

    def test_regimes_cover_the_backtest(self, backtest_result):
        assert backtest_result.regimes is not None
        labels = backtest_result.regimes.labels
        assert labels.index.min() <= backtest_result.equity_curve.index.min()

    def test_strategy_attribution_is_available(self, backtest_result):
        share = backtest_result.strategy_contributions
        assert not share.empty
        assert set(share.columns) == {"trend", "mean_reversion", "cross_sectional_momentum"}
        # Shares are fractions of the total absolute signal.
        assert share.sum(axis=1).dropna().max() <= 1.0 + 1e-9


class TestBenchmarks:
    def test_benchmark_suite_runs(self, base_config, panel, backtest_result):
        features = FeatureEngineer.from_config(base_config).build(panel)
        suite = BenchmarkSuite(base_config).run(
            panel, atlas_result=backtest_result, features=features
        )
        expected = {
            "Buy & hold SPY",
            "60/40 SPY/IEF",
            "Equal-weight ETF portfolio",
            "Trend only",
            "Mean reversion only",
            "Cross-sectional momentum only",
            "Equal-weight strategy ensemble",
            "Atlas regime-aware ensemble",
        }
        assert expected.issubset(set(suite.returns))

    def test_benchmarks_share_a_date_range(self, base_config, panel, backtest_result):
        features = FeatureEngineer.from_config(base_config).build(panel)
        suite = BenchmarkSuite(base_config).run(
            panel, atlas_result=backtest_result, features=features, include_strategy_variants=False
        )
        starts = [s.index.min() for s in suite.returns.values()]
        ends = [s.index.max() for s in suite.returns.values()]
        assert (max(starts) - min(starts)).days < 400, "benchmarks should broadly align"
        assert (max(ends) - min(ends)).days < 10

    def test_passive_benchmarks_differ(self, base_config, panel):
        """A regression test: the three passive benchmarks were once identical."""
        suite = BenchmarkSuite(base_config)
        spy, _ = suite._buy_and_hold(panel, "SPY", panel.dates.min(), panel.dates.max())
        sixty_forty, _ = suite._fixed_weight(
            panel, {"SPY": 0.6, "IEF": 0.4}, panel.dates.min(), panel.dates.max(), "60/40"
        )
        assert not np.allclose(spy.to_numpy(), sixty_forty.to_numpy())

    def test_comparison_table(self, base_config, panel, backtest_result):
        features = FeatureEngineer.from_config(base_config).build(panel)
        suite = BenchmarkSuite(base_config).run(
            panel, atlas_result=backtest_result, features=features, include_strategy_variants=False
        )
        table = suite.table()
        assert "sharpe_ratio" in table.index
        assert "max_drawdown" in table.index


class TestStressAndBootstrap:
    def test_stress_periods(self, base_config, panel, backtest_result):
        result = StressTester(base_config.validation).run(backtest_result, panel=panel)
        assert not result.periods.empty
        for column in ("name", "total_return", "max_drawdown", "regime_mix"):
            assert column in result.periods.columns

    def test_out_of_range_windows_are_reported(self, base_config, backtest_result):
        from atlas.config import StressPeriod

        far_future = [StressPeriod(name="2099", start="2099-01-01", end="2099-06-01")]
        result = StressTester(base_config.validation).run(backtest_result, periods=far_future)
        assert result.periods.empty
        assert len(result.skipped) == 1
        assert "2099" in result.skipped[0]

    def test_block_bootstrap_preserves_dependence(self, backtest_result):
        paths = block_bootstrap(
            backtest_result.returns, n_simulations=100, block_size=21, horizon=252, seed=7
        )
        assert paths.shape == (252, 100)
        # Block resampling keeps volatility roughly comparable to the original.
        original = backtest_result.returns.std()
        assert 0.5 * original < paths.stack().std() < 2.0 * original

    def test_bootstrap_is_reproducible(self, backtest_result):
        a = block_bootstrap(backtest_result.returns, n_simulations=20, seed=42)
        b = block_bootstrap(backtest_result.returns, n_simulations=20, seed=42)
        assert np.allclose(a.to_numpy(), b.to_numpy())

    def test_bootstrap_statistics(self, backtest_result):
        paths = block_bootstrap(backtest_result.returns, n_simulations=100, seed=1)
        summary = bootstrap_statistics(paths, [0.05, 0.5, 0.95])
        for row in ("annualized_return", "max_drawdown", "ending_wealth", "sharpe_ratio"):
            assert row in summary.index
        assert (summary.loc["max_drawdown", "p95"] >= summary.loc["max_drawdown", "p5"])

    def test_bootstrap_needs_enough_history(self):
        with pytest.raises(ValueError, match="at least"):
            block_bootstrap(pd.Series([0.01, 0.02]), block_size=21)


class TestWalkForward:
    @pytest.mark.slow
    def test_walk_forward_produces_out_of_sample_results(self, base_config, panel):
        config = base_config.model_copy(deep=True)
        config.validation.walk_forward.train_years = 3
        config.validation.walk_forward.validation_years = 1
        config.validation.walk_forward.test_years = 1
        config.validation.walk_forward.step_years = 2
        config.validation.walk_forward.parameter_grid = {
            "trend.slow_ma": [100, 200],
            "portfolio.target_volatility": [0.08, 0.12],
        }
        validator = WalkForwardValidator(config)
        result = validator.run(panel, progress=False)

        assert len(result.windows) >= 1
        assert not result.selections.empty
        assert len(result.test_returns) > 100
        assert result.test_metrics is not None

    @pytest.mark.slow
    def test_out_of_sample_periods_do_not_overlap(self, base_config, panel):
        config = base_config.model_copy(deep=True)
        config.validation.walk_forward.train_years = 3
        config.validation.walk_forward.step_years = 2
        config.validation.walk_forward.parameter_grid = {"trend.slow_ma": [200]}
        result = WalkForwardValidator(config).run(panel, progress=False)
        assert not result.test_returns.index.duplicated().any()

    @pytest.mark.slow
    def test_degradation_is_reported(self, base_config, panel):
        config = base_config.model_copy(deep=True)
        config.validation.walk_forward.train_years = 3
        config.validation.walk_forward.step_years = 2
        config.validation.walk_forward.parameter_grid = {"trend.slow_ma": [100, 200]}
        result = WalkForwardValidator(config).run(panel, progress=False)
        degradation = result.degradation()
        assert "train_minus_test_sharpe" in degradation
        assert "mean_test_sharpe" in degradation

    def test_grid_expansion(self):
        grid = {"a": [1, 2], "b": [3, 4, 5]}
        combos = expand_grid(grid)
        assert len(combos) == 6
        assert {"a": 1, "b": 3} in combos

    def test_empty_grid_gives_one_candidate(self):
        assert expand_grid({}) == [{}]

    def test_overrides_do_not_mutate_the_original(self, base_config):
        before = base_config.strategies.strategies.trend.slow_ma
        modified = apply_overrides(base_config, {"trend.slow_ma": 123})
        assert modified.strategies.strategies.trend.slow_ma == 123
        assert base_config.strategies.strategies.trend.slow_ma == before

    def test_override_prefixes(self, base_config):
        modified = apply_overrides(
            base_config,
            {
                "portfolio.target_volatility": 0.05,
                "mean_reversion.entry_threshold": 2.0,
                "covariance.lookback": 100,
            },
        )
        assert modified.portfolio.target_volatility == 0.05
        assert modified.strategies.strategies.mean_reversion.entry_threshold == 2.0
        assert modified.risk.covariance.lookback == 100

    def test_unknown_override_raises(self, base_config):
        with pytest.raises(ValueError, match="unknown override prefix"):
            apply_overrides(base_config, {"nonsense.value": 1})
        with pytest.raises(ValueError, match="dotted path"):
            apply_overrides(base_config, {"trend": 1})


class TestPersistence:
    def test_backtest_round_trip(self, tmp_database, backtest_result, base_config):
        repository = AtlasRepository(tmp_database)
        repository.save_configuration(base_config.config_hash, base_config.snapshot(), "test")
        row_id = repository.save_backtest(backtest_result, label="integration test")
        assert row_id > 0

        stored = repository.get_run(backtest_result.run_id)
        assert stored is not None
        assert stored["run_id"] == backtest_result.run_id
        assert stored["is_synthetic_data"] == backtest_result.is_synthetic_data
        assert "sharpe_ratio" in stored["metrics"]

    def test_equity_curve_round_trip(self, tmp_database, backtest_result):
        repository = AtlasRepository(tmp_database)
        repository.save_backtest(backtest_result)
        curve = repository.get_equity_curve(backtest_result.run_id)
        assert len(curve) == len(backtest_result.equity_curve)
        assert curve.iloc[-1] == pytest.approx(backtest_result.final_equity, rel=1e-6)

    def test_prices_round_trip(self, tmp_database, short_panel):
        from atlas.data.loader import panel_to_long

        repository = AtlasRepository(tmp_database)
        frame = panel_to_long(short_panel)
        written = repository.save_prices(frame)
        assert written == len(frame)

        loaded = repository.load_prices(["SPY"])
        assert len(loaded) == len(frame[frame["symbol"] == "SPY"])

    def test_prices_are_idempotent(self, tmp_database, short_panel):
        from atlas.data.loader import panel_to_long

        repository = AtlasRepository(tmp_database)
        frame = panel_to_long(short_panel)
        repository.save_prices(frame)
        repository.save_prices(frame)
        loaded = repository.load_prices()
        assert len(loaded) == len(frame), "re-importing must update, not duplicate"

    def test_configuration_versions_accumulate(self, tmp_database, base_config):
        repository = AtlasRepository(tmp_database)
        first = repository.save_configuration("hash_a", {"a": 1}, "first")
        again = repository.save_configuration("hash_a", {"a": 1}, "first")
        second = repository.save_configuration("hash_b", {"a": 2}, "second")
        assert first == again, "an identical configuration must not be duplicated"
        assert second != first
        assert len(repository.list_configurations()) == 2

    def test_run_listing(self, tmp_database, backtest_result):
        repository = AtlasRepository(tmp_database)
        repository.save_backtest(backtest_result)
        listing = repository.list_runs()
        assert len(listing) == 1
        assert listing.iloc[0]["run_id"] == backtest_result.run_id


class TestReporting:
    def test_export_writes_every_artefact(self, tmp_path, backtest_result, base_config):
        destination = export_backtest(backtest_result, tmp_path / "run", config=base_config)
        manifest = load_manifest(destination)
        assert manifest["run_id"] == backtest_result.run_id
        assert "config_snapshot.json" in manifest["files"]
        for name in ("equity_curve.csv", "snapshots.csv", "fills.csv", "metrics.json"):
            assert (destination / name).is_file()

    def test_manifest_records_provenance(self, tmp_path, backtest_result, base_config):
        destination = export_backtest(backtest_result, tmp_path / "run", config=base_config)
        manifest = load_manifest(destination)
        assert manifest["config_hash"] == base_config.config_hash
        assert "cost_breakdown" in manifest
        assert manifest["is_synthetic_data"] is True

    def test_html_report_is_generated(self, tmp_path, backtest_result, base_config, panel):
        bundle = ReportBundle(
            result=backtest_result, config=base_config, panel=panel, label="Integration test"
        )
        path = ResearchReport(bundle).to_html(tmp_path / "report.html")
        html = path.read_text()
        assert len(html) > 50_000
        assert "Executive summary" in html
        assert "Limitations" in html
        assert "Atlas is an educational research" in html

    def test_report_flags_synthetic_data(self, tmp_path, backtest_result, base_config):
        bundle = ReportBundle(result=backtest_result, config=base_config)
        html = ResearchReport(bundle).to_html(tmp_path / "report.html").read_text()
        assert "SYNTHETIC DATA" in html

    def test_report_states_missing_walk_forward(self, tmp_path, backtest_result, base_config):
        bundle = ReportBundle(result=backtest_result, config=base_config)
        html = ResearchReport(bundle).to_html(tmp_path / "report.html").read_text()
        assert "No walk-forward validation was run" in html

    def test_markdown_report(self, tmp_path, backtest_result, base_config):
        bundle = ReportBundle(result=backtest_result, config=base_config)
        path = ResearchReport(bundle).to_markdown(tmp_path / "report.md")
        text = path.read_text()
        assert text.startswith("# ")
        assert "Limitations" in text

    def test_charts_build(self, backtest_result, panel):
        from atlas.reporting import charts

        figures = {
            "equity": charts.equity_curve_chart({"Atlas": backtest_result.equity_curve}),
            "drawdown": charts.drawdown_chart(backtest_result.equity_curve),
            "rolling_sharpe": charts.rolling_sharpe_chart(backtest_result.returns),
            "monthly": charts.monthly_heatmap(backtest_result.returns),
            "weights": charts.weights_chart(backtest_result.weights),
            "exposure": charts.exposure_chart(backtest_result.snapshots),
            "regimes": charts.regime_timeline_chart(backtest_result.regimes.labels),
            "correlation": charts.asset_correlation_chart(panel.returns()),
            "turnover": charts.turnover_cost_chart(backtest_result.snapshots, backtest_result.fills),
        }
        for name, figure in figures.items():
            assert figure.layout.title.text, f"chart {name} has no title"

    def test_charts_label_axes(self, backtest_result):
        from atlas.reporting import charts

        figure = charts.rolling_volatility_chart(backtest_result.returns, target=0.10)
        assert figure.layout.yaxis.title.text
        assert "%" in figure.layout.yaxis.title.text

    def test_equity_axis_is_not_truncated(self, backtest_result):
        from atlas.reporting import charts

        figure = charts.equity_curve_chart({"Atlas": backtest_result.equity_curve})
        assert figure.layout.yaxis.rangemode == "tozero"

    def test_export_figures(self, tmp_path, backtest_result):
        from atlas.reporting import charts
        from atlas.reporting.export import export_figures

        figures = {"equity": charts.equity_curve_chart({"Atlas": backtest_result.equity_curve})}
        written = export_figures(figures, tmp_path / "figures", fmt="html")
        assert len(written) == 1
        assert written[0].is_file()


class TestWalkForwardGridValidation:
    """An unusable grid must fail immediately, not silently shrink the search."""

    def test_shipped_grid_is_valid(self, base_config):
        WalkForwardValidator(base_config).validate_grid()

    def test_invalid_combination_is_rejected_up_front(self, base_config):
        from atlas.exceptions import ConfigurationError

        config = base_config.model_copy(deep=True)
        # fast_ma is 100, so slow_ma=100 cannot produce a valid configuration.
        config.validation.walk_forward.parameter_grid = {"trend.slow_ma": [100, 200]}
        with pytest.raises(ConfigurationError, match="cannot produce a valid configuration"):
            WalkForwardValidator(config).validate_grid()

    def test_error_names_the_offending_combination(self, base_config):
        from atlas.exceptions import ConfigurationError

        config = base_config.model_copy(deep=True)
        config.validation.walk_forward.parameter_grid = {"portfolio.target_volatility": [0.10, 5.0]}
        with pytest.raises(ConfigurationError) as excinfo:
            WalkForwardValidator(config).validate_grid()
        assert "5.0" in str(excinfo.value)


class TestStabilitySweepValidation:
    """The same rule for the one-at-a-time sweep.

    A dropped sweep point does not fail the run; it shrinks the neighbourhood
    the verdict is computed over, which is a quieter and more misleading
    outcome than an error.
    """

    def test_shipped_sweep_is_valid(self, base_config):
        from atlas.validation.parameter_stability import ParameterStabilityAnalyzer

        analyzer = ParameterStabilityAnalyzer(base_config)
        analyzer.validate_sweep(base_config.validation.parameter_stability.parameters)

    def test_impossible_value_is_rejected_up_front(self, base_config):
        from atlas.exceptions import ConfigurationError
        from atlas.validation.parameter_stability import ParameterStabilityAnalyzer

        # fast_ma is 100, so slow_ma=100 can never construct.
        sweep = {"trend.slow_ma": [100, 150, 200]}
        with pytest.raises(ConfigurationError, match="cannot produce a valid configuration"):
            ParameterStabilityAnalyzer(base_config).validate_sweep(sweep)

    def test_error_names_the_offending_value(self, base_config):
        from atlas.exceptions import ConfigurationError
        from atlas.validation.parameter_stability import ParameterStabilityAnalyzer

        sweep = {"portfolio.target_volatility": [0.10, 5.0]}
        with pytest.raises(ConfigurationError) as excinfo:
            ParameterStabilityAnalyzer(base_config).validate_sweep(sweep)
        assert "5.0" in str(excinfo.value)
