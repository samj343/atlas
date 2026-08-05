"""Edge-case and degenerate-input tests.

Real market data is messy. Each test here feeds the system something broken and
asserts it either handles it deliberately or fails with a clear, specific error -
never that it silently produces a plausible-looking wrong number.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.broker import ExecutionContext, SimulatedBroker
from atlas.backtest.costs import TransactionCostModel
from atlas.backtest.engine import BacktestEngine
from atlas.backtest.metrics import compute_metrics, sharpe_ratio
from atlas.backtest.orders import Order, OrderSide
from atlas.backtest.positions import Portfolio
from atlas.config import CostConfig, CovarianceConfig, FillConfig, VolatilityConfig
from atlas.data.features import FeatureEngineer
from atlas.data.loader import PricePanel, long_to_panel
from atlas.data.provider import empty_price_frame
from atlas.exceptions import (
    BrokerError,
    InsufficientDataError,
    OrderRejected,
    StrategyError,
)
from atlas.execution.ibkr_client import MockBrokerClient
from atlas.execution.reconciliation import Reconciler
from atlas.portfolio.allocator import PortfolioAllocator
from atlas.portfolio.covariance import CovarianceEstimator
from atlas.portfolio.volatility import VolatilityEstimator
from atlas.strategies.base import build_strategies


class TestEmptyAndTinyInputs:
    def test_empty_panel_features_raise(self):
        panel = long_to_panel(empty_price_frame())
        with pytest.raises(InsufficientDataError, match="empty price panel"):
            FeatureEngineer().build(panel)

    def test_two_row_panel_raises(self, short_panel):
        tiny = short_panel.slice(short_panel.dates[0], short_panel.dates[1])
        with pytest.raises(InsufficientDataError, match="at least 3 observations"):
            FeatureEngineer().build(tiny)

    def test_backtest_needs_two_days(self, config, short_panel):
        tiny = short_panel.slice(short_panel.dates[0], short_panel.dates[0])
        with pytest.raises(InsufficientDataError):
            BacktestEngine(config).run(tiny)

    def test_strategies_reject_empty_prices(self, base_config):
        for strategy in build_strategies(base_config):
            with pytest.raises(StrategyError, match="empty prices"):
                strategy.compute(pd.DataFrame())

    def test_no_symbols_in_panel(self, config, short_panel):
        """A panel sharing no symbols with the universe must fail explicitly."""
        empty = short_panel.select(["NOT_IN_UNIVERSE"])
        with pytest.raises(InsufficientDataError, match="none of the configured symbols"):
            BacktestEngine(config).run(empty)


class TestInsufficientLookback:
    def test_strategies_return_neutral_signals(self, base_config, short_panel):
        tiny = short_panel.slice(short_panel.dates[0], short_panel.dates[30])
        for strategy in build_strategies(base_config):
            result = strategy.compute(tiny.adj_close)
            assert (result.signal.abs() < 1e-12).to_numpy().all()

    def test_covariance_raises_below_minimum(self, short_panel):
        returns = short_panel.returns().head(20)
        estimator = CovarianceEstimator(CovarianceConfig(min_observations=60))
        with pytest.raises(InsufficientDataError):
            estimator.estimate(returns)

    def test_allocator_holds_cash_without_volatility_estimates(self, config, short_panel):
        """With too little history to size anything, the correct answer is cash."""
        allocator = PortfolioAllocator(config)
        returns = short_panel.returns().head(10)
        signals = pd.Series(0.5, index=short_panel.symbols)
        result = allocator.allocate(signals, returns)
        assert result.gross_exposure == 0.0
        assert "cash" in result.diagnostics.get("reason", "")

    def test_allocator_falls_back_to_diagonal_covariance(self, config, short_panel):
        """When covariance cannot be estimated, a diagonal matrix is used."""
        config.risk.covariance.min_observations = 100_000
        allocator = PortfolioAllocator(config)
        returns = short_panel.returns().tail(400)
        signals = pd.Series(0.5, index=short_panel.symbols)
        result = allocator.allocate(signals, returns)
        assert result.covariance is not None
        assert result.covariance.method == "diagonal_fallback"


class TestDegenerateNumerics:
    def test_zero_volatility_asset(self, config):
        index = pd.bdate_range("2018-01-01", periods=400)
        frame = pd.DataFrame(
            {"FLAT": np.full(len(index), 100.0), "MOVER": 100 * (1.001 ** np.arange(len(index)))},
            index=index,
        )
        returns = frame.pct_change(fill_method=None)
        estimates = VolatilityEstimator(VolatilityConfig()).estimate_latest(returns)
        assert estimates["FLAT"] > 0, "a floor must prevent a zero-volatility divide"
        assert np.isfinite(estimates["FLAT"])

    def test_singular_covariance_is_usable(self):
        index = pd.bdate_range("2018-01-01", periods=400)
        rng = np.random.default_rng(11)
        base = rng.standard_normal(len(index)) * 0.01
        returns = pd.DataFrame({"A": base, "B": base, "C": base}, index=index)
        result = CovarianceEstimator(CovarianceConfig(method="sample")).estimate(returns)
        assert np.linalg.eigvalsh(result.matrix.to_numpy()).min() > 0

    def test_all_nan_returns(self):
        index = pd.bdate_range("2020-01-01", periods=100)
        returns = pd.Series(np.nan, index=index)
        metrics = compute_metrics(returns)
        assert metrics.values["total_return"] == 0.0
        assert np.isfinite(metrics.sharpe)

    def test_single_observation(self):
        index = pd.bdate_range("2020-01-01", periods=1)
        assert sharpe_ratio(pd.Series([0.01], index=index)) == 0.0

    def test_extreme_price_change_does_not_crash(self, config, short_panel):
        frame = short_panel.adj_close.copy()
        frame.iloc[300:, 0] = frame.iloc[300:, 0] * 20.0
        panel = PricePanel(
            open=short_panel.open, high=short_panel.high, low=short_panel.low,
            close=short_panel.close, adj_close=frame, volume=short_panel.volume,
        )
        result = BacktestEngine(config).run(panel)
        assert np.isfinite(result.final_equity)
        assert result.final_equity > 0

    def test_negative_cash_is_representable_in_backtest(self):
        portfolio = Portfolio(1_000.0, allow_negative_cash=True)
        from atlas.backtest.orders import Fill

        portfolio.apply_fill(
            Fill("o", "SPY", 100.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01"))
        )
        assert portfolio.cash < 0
        assert np.isfinite(portfolio.equity(pd.Series({"SPY": 100.0})))


class TestMissingData:
    def test_asset_with_late_inception(self, config, base_config, provider):
        """An asset that starts late must simply be untradable until it exists."""
        frame = provider.get_historical_prices(
            base_config.universe.symbols, dt.date(2008, 1, 1), dt.date(2013, 12, 31)
        )
        panel = long_to_panel(frame)
        late = panel.adj_close.copy()
        late.iloc[:600, late.columns.get_loc("EEM")] = np.nan
        panel = PricePanel(
            open=panel.open, high=panel.high, low=panel.low, close=panel.close,
            adj_close=late, volume=panel.volume,
        )
        result = BacktestEngine(config).run(panel)
        if "EEM" in result.weights.columns:
            early = result.weights["EEM"].loc[result.weights.index < panel.dates[600]]
            assert (early.abs() < 1e-9).all(), "an asset was traded before it had any price"

    def test_missing_price_blocks_that_order_only(self, config, short_panel):
        marks = short_panel.adj_close.iloc[-1].copy()
        marks.iloc[0] = np.nan
        from atlas.portfolio.risk_manager import RiskManager

        manager = RiskManager(config)
        weights = pd.Series(0.1, index=short_panel.symbols)
        assessment = manager.assess(
            short_panel.dates[-1], weights, equity=100_000.0, prices=marks,
            last_data_date=short_panel.dates[-1],
        )
        assert assessment.weights.iloc[0] == 0.0
        assert assessment.weights.iloc[1] > 0.0

    def test_duplicate_dates_are_flagged(self, short_panel):
        from atlas.data.validator import DataValidator

        def _double(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.concat([frame, frame.tail(5)])

        panel = PricePanel(
            open=_double(short_panel.open),
            high=_double(short_panel.high),
            low=_double(short_panel.low),
            close=_double(short_panel.close),
            adj_close=_double(short_panel.adj_close),
            volume=_double(short_panel.volume),
        )
        report = DataValidator().validate(panel)
        assert any(i.code in {"duplicate_dates", "unsorted_calendar"} for i in report.errors)

    def test_provider_deduplicates_repeated_bars(self, provider):
        """A vendor emitting a provisional and a final bar must not double-count."""
        frame = provider.get_historical_prices(["SPY"], dt.date(2015, 1, 1), dt.date(2015, 3, 1))
        assert not frame.duplicated(subset=["symbol", "date"]).any()


class TestOrderEdgeCases:
    @pytest.fixture
    def broker(self) -> SimulatedBroker:
        return SimulatedBroker(TransactionCostModel(CostConfig()), FillConfig())

    def test_zero_volume_bar(self, broker):
        context = ExecutionContext(
            timestamp=pd.Timestamp("2020-01-01"),
            prices=pd.Series({"SPY": 100.0}),
            volumes=pd.Series({"SPY": 0.0}),
        )
        order = Order(symbol="SPY", quantity=100.0, side=OrderSide.BUY)
        fill = broker.execute(order, context)
        # Zero volume means no ADV estimate, so the participation limit does not
        # apply and the order fills; costs simply carry no impact term.
        assert fill is not None
        assert fill.quantity == pytest.approx(100.0)

    def test_negative_price_rejected(self, broker):
        context = ExecutionContext(
            timestamp=pd.Timestamp("2020-01-01"), prices=pd.Series({"SPY": -5.0})
        )
        order = Order(symbol="SPY", quantity=10.0, side=OrderSide.BUY)
        assert broker.execute(order, context) is None
        assert order.status.value == "rejected"

    def test_duplicate_orders_rejected(self, config):
        from atlas.portfolio.risk_manager import RiskManager

        manager = RiskManager(config)
        allowed_first, _ = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 100.0, 100.0)
        allowed_second, reason = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 100.0, 100.0)
        assert allowed_first and not allowed_second and "duplicate" in reason

    def test_broker_rejection_is_raised(self):
        broker = MockBrokerClient(prices=pd.Series({"SPY": 100.0}), reject_symbols={"SPY"})
        broker.connect()
        order = Order(symbol="SPY", quantity=10.0, side=OrderSide.BUY)
        with pytest.raises(OrderRejected):
            broker.submit_order(order)
        assert order.status.value == "rejected"

    def test_partial_fill_from_broker(self):
        broker = MockBrokerClient(prices=pd.Series({"SPY": 100.0}), fill_ratio=0.5)
        broker.connect()
        order = Order(symbol="SPY", quantity=100.0, side=OrderSide.BUY)
        broker.submit_order(order)
        assert order.filled_quantity == pytest.approx(50.0)
        assert order.status.value == "partially_filled"

    def test_order_without_price_rejected(self):
        broker = MockBrokerClient(prices=pd.Series(dtype="float64"))
        broker.connect()
        order = Order(symbol="SPY", quantity=10.0, side=OrderSide.BUY)
        with pytest.raises(OrderRejected, match="no market price"):
            broker.submit_order(order)


class TestBrokerDisconnection:
    def test_reads_fail_when_disconnected(self):
        broker = MockBrokerClient()
        with pytest.raises(BrokerError, match="not connected"):
            broker.get_positions()

    def test_api_failures_are_simulated(self):
        broker = MockBrokerClient(fail_reads=2)
        broker.connect()
        with pytest.raises(BrokerError, match="simulated"):
            broker.get_positions()
        with pytest.raises(BrokerError):
            broker.get_positions()
        assert broker.get_positions() == []

    def test_repeated_failures_halt_trading(self, config):
        from atlas.portfolio.risk_manager import RiskManager

        manager = RiskManager(config)
        broker = MockBrokerClient(fail_reads=10)
        broker.connect()
        for _ in range(config.risk.limits.max_consecutive_api_failures):
            with pytest.raises(BrokerError):
                broker.get_positions()
            manager.record_api_failure("get_positions")
        assert manager.halted


class TestReconciliationEdgeCases:
    @pytest.fixture
    def reconciler(self, config) -> Reconciler:
        return Reconciler(config.execution.reconciliation)

    def test_identical_positions_reconcile(self, reconciler):
        report = reconciler.reconcile({"SPY": 100.0}, {"SPY": 100.0})
        assert report.reconciled
        assert report.status == "reconciled"

    def test_fractional_difference_is_tolerated(self, reconciler):
        report = reconciler.reconcile(
            {"SPY": 100.0}, {"SPY": 100.005}, prices=pd.Series({"SPY": 100.0})
        )
        assert report.reconciled
        assert report.status == "tolerated"

    def test_material_difference_blocks(self, reconciler):
        report = reconciler.reconcile(
            {"SPY": 100.0}, {"SPY": 150.0}, prices=pd.Series({"SPY": 100.0})
        )
        assert not report.reconciled
        assert report.blocking

    def test_position_missing_locally_is_always_material(self, reconciler):
        """A position on only one side is structural, not rounding."""
        report = reconciler.reconcile({}, {"SPY": 0.001}, prices=pd.Series({"SPY": 1.0}))
        assert not report.reconciled
        assert report.mismatches[0].kind == "missing_locally"

    def test_position_missing_at_broker(self, reconciler):
        report = reconciler.reconcile({"SPY": 10.0}, {}, prices=pd.Series({"SPY": 100.0}))
        assert report.mismatches[0].kind == "missing_at_broker"

    def test_empty_both_sides(self, reconciler):
        report = reconciler.reconcile({}, {})
        assert report.reconciled
        assert report.n_checked == 0

    def test_report_renders(self, reconciler):
        report = reconciler.reconcile({"SPY": 10.0}, {"SPY": 20.0}, prices=pd.Series({"SPY": 5.0}))
        text = report.render()
        assert "reconciliation" in text.lower()
        assert "SPY" in text

    def test_unsupported_container_raises(self, reconciler):
        with pytest.raises(TypeError, match="unsupported position container"):
            reconciler.reconcile("not a container", {})
