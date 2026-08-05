"""Integration tests for the paper-trading execution path.

Every test uses :class:`~atlas.execution.ibkr_client.MockBrokerClient`. No test in
Atlas ever opens a socket to a broker; the mock implements the same interface, so
the code under test is the code that would run against TWS.
"""

from __future__ import annotations

import pandas as pd
import pytest

from atlas.backtest.orders import Order, OrderSide, OrderStatus
from atlas.config import ExecutionMode
from atlas.database import AtlasRepository
from atlas.exceptions import ExecutionBlocked, OrderRejected
from atlas.execution.ibkr_client import MockBrokerClient
from atlas.execution.order_manager import OrderManager
from atlas.execution.reconciliation import Reconciler


@pytest.fixture
def prices(short_panel) -> pd.Series:
    # Raw, because that is the scale a broker quotes on.
    return short_panel.close.iloc[-1]


@pytest.fixture
def broker(prices) -> MockBrokerClient:
    client = MockBrokerClient(prices=prices, initial_cash=100_000.0)
    client.connect()
    return client


@pytest.fixture
def repository(tmp_database) -> AtlasRepository:
    return AtlasRepository(tmp_database)


@pytest.fixture
def paper_config(config):
    """Configuration permitting submission, whatever time the suite runs at.

    The trading-window control legitimately rejects every order outside
    09:35-15:55 New York time, which would make these tests pass or fail
    depending on the clock. The window itself is tested directly in
    tests/unit/test_risk.py; here it is widened so the submission path is what
    is under test.
    """
    config.execution.mode = ExecutionMode.PAPER
    config.execution.broker.account_allowlist = ["DU1234567"]
    config.execution.trading_window.start = "00:00"
    config.execution.trading_window.end = "23:59"
    config.execution.trading_window.trading_days = [0, 1, 2, 3, 4, 5, 6]
    return config


class TestMockBroker:
    def test_connection_lifecycle(self, prices):
        client = MockBrokerClient(prices=prices)
        assert not client.connected
        client.connect()
        assert client.connected
        assert client.account.startswith("DU")
        client.disconnect()
        assert not client.connected

    def test_context_manager(self, prices):
        with MockBrokerClient(prices=prices) as client:
            assert client.connected

    def test_account_summary(self, broker):
        summary = broker.get_account_summary()
        assert summary.net_liquidation == pytest.approx(100_000.0)
        assert summary.account.startswith("DU")

    def test_order_updates_positions_and_cash(self, broker, prices):
        symbol = prices.index[0]
        order = Order(symbol=symbol, quantity=10.0, side=OrderSide.BUY)
        broker.submit_order(order)

        assert order.status is OrderStatus.FILLED
        positions = {p.symbol: p.quantity for p in broker.get_positions()}
        assert positions[symbol] == pytest.approx(10.0)
        assert broker.cash < 100_000.0

    def test_round_trip_closes_the_position(self, broker, prices):
        symbol = prices.index[0]
        broker.submit_order(Order(symbol=symbol, quantity=10.0, side=OrderSide.BUY))
        broker.submit_order(Order(symbol=symbol, quantity=10.0, side=OrderSide.SELL))
        assert not [p for p in broker.get_positions() if p.symbol == symbol]

    def test_fills_are_recorded(self, broker, prices):
        broker.submit_order(Order(symbol=prices.index[0], quantity=5.0, side=OrderSide.BUY))
        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].commission > 0

    def test_cancel_open_order(self, prices):
        broker = MockBrokerClient(prices=prices, fill_ratio=0.0)
        broker.connect()
        order = Order(symbol=prices.index[0], quantity=10.0, side=OrderSide.BUY)
        broker.submit_order(order)
        assert broker.cancel_order(order)


class TestExecutionCycle:
    def test_dry_run_plans_but_does_not_submit(self, config, short_panel, broker):
        config.execution.mode = ExecutionMode.DRY_RUN
        manager = OrderManager(config, broker=broker)
        result = manager.run_cycle(short_panel)

        assert result.dry_run
        assert result.plan is not None
        assert result.plan.n_orders > 0
        assert not result.submitted
        assert not broker.submitted_orders

    def test_paper_mode_submits(self, paper_config, short_panel, prices):
        broker = MockBrokerClient(prices=prices, initial_cash=100_000.0, account="DU1234567")
        broker.connect()

        manager = OrderManager(paper_config, broker=broker)
        result = manager.run_cycle(short_panel, submit=True)

        assert result.executed
        assert len(broker.submitted_orders) == len(result.submitted)
        assert result.fills

    def test_paper_mode_blocked_without_allowlist(self, config, short_panel, prices):
        config.execution.mode = ExecutionMode.PAPER
        config.execution.broker.account_allowlist = []
        broker = MockBrokerClient(prices=prices, account="DU1234567")
        broker.connect()

        result = OrderManager(config, broker=broker).run_cycle(short_panel, submit=True)
        assert result.blocked_reason is not None
        assert not result.submitted

    def test_non_paper_account_blocked(self, config, short_panel, prices):
        config.execution.mode = ExecutionMode.PAPER
        config.execution.broker.account_allowlist = ["U7654321"]
        broker = MockBrokerClient(prices=prices, account="U7654321")
        broker.connect()

        result = OrderManager(config, broker=broker).run_cycle(short_panel, submit=True)
        assert result.blocked_reason is not None
        assert "paper" in result.blocked_reason.lower()

    def test_kill_switch_blocks_the_cycle(self, config, short_panel, broker):
        config.risk.kill_switch.enabled = True
        config.risk.kill_switch.reason = "testing"
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        assert result.blocked_reason is not None
        assert not result.submitted

    def test_stale_data_blocks_the_cycle(self, config, short_panel, broker):
        far_future = short_panel.dates.max() + pd.Timedelta(days=90)
        result = OrderManager(config, broker=broker).run_cycle(short_panel, as_of=far_future)
        assert result.blocked_reason is not None

    def test_plan_respects_the_minimum_order_size(self, config, short_panel, broker):
        config.risk.limits.min_order_notional = 1_000_000.0
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        assert result.plan.n_orders == 0
        assert result.plan.skipped

    def test_plan_renders(self, config, short_panel, broker):
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        text = result.plan.render()
        assert "Atlas order preview" in text
        assert "SYMBOL" in text or "no orders" in text

    def test_report_renders(self, config, short_panel, broker):
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        text = result.render()
        assert "Atlas execution report" in text
        assert "DRY RUN" in text

    def test_orders_carry_unique_ids(self, config, short_panel, broker):
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        ids = [o.order_id for o in result.plan.orders]
        assert len(ids) == len(set(ids))

    def test_target_weights_are_risk_checked(self, config, short_panel, broker):
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        weights = result.plan.target_weights
        assert weights.abs().max() <= config.portfolio.max_asset_weight + 1e-9
        assert float(weights.abs().sum()) <= config.portfolio.max_gross_exposure + 1e-9

    def test_rejected_orders_do_not_stop_the_cycle(self, paper_config, panel):
        """One broker rejection must not prevent the other orders being sent."""
        full_prices = panel.adj_close.iloc[-1]
        broker = MockBrokerClient(
            prices=full_prices, initial_cash=100_000.0, account="DU1234567",
            reject_symbols={str(full_prices.index[0])},
        )
        broker.connect()
        result = OrderManager(paper_config, broker=broker).run_cycle(panel, submit=True)
        assert result.rejected
        assert result.submitted, "one rejection must not block the remaining orders"

    def test_partial_fills_are_recorded(self, paper_config, short_panel, prices):
        broker = MockBrokerClient(
            prices=prices, initial_cash=100_000.0, account="DU1234567", fill_ratio=0.5
        )
        broker.connect()
        result = OrderManager(paper_config, broker=broker).run_cycle(short_panel, submit=True)
        assert result.submitted
        assert all(o.status is OrderStatus.PARTIALLY_FILLED for o in result.submitted)


    def test_trading_window_blocks_out_of_hours_orders(self, config, short_panel, prices):
        """Outside the configured window every order is skipped with a reason."""
        config.execution.mode = ExecutionMode.PAPER
        config.execution.broker.account_allowlist = ["DU1234567"]
        config.execution.trading_window.start = "09:35"
        config.execution.trading_window.end = "09:36"
        config.execution.trading_window.trading_days = [0]
        broker = MockBrokerClient(prices=prices, initial_cash=100_000.0, account="DU1234567")
        broker.connect()

        result = OrderManager(config, broker=broker).run_cycle(short_panel, submit=True)
        assert result.plan.n_orders == 0
        assert any("trading window" in s["reason"] for s in result.plan.skipped)


class TestReconciliationInCycle:
    def test_mismatch_blocks_the_next_cycle(self, config, short_panel, prices):
        """A broker position Atlas does not know about must stop trading."""
        broker = MockBrokerClient(prices=prices, initial_cash=100_000.0)
        broker.connect()
        broker.set_position(prices.index[0], 500.0, float(prices.iloc[0]))

        manager = OrderManager(config, broker=broker)
        result = manager.run_cycle(short_panel)

        assert result.reconciliation is not None
        assert not result.reconciliation.reconciled
        assert result.reconciliation.blocking
        assert result.blocked_reason is not None
        assert not result.submitted

    def test_clean_state_reconciles(self, config, short_panel, broker):
        result = OrderManager(config, broker=broker).run_cycle(short_panel)
        assert result.reconciliation.reconciled
        assert result.blocked_reason is None

    def test_reconciliation_is_persisted(self, config, short_panel, broker, repository):
        manager = OrderManager(config, broker=broker, repository=repository)
        manager.run_cycle(short_panel)
        stored = repository.latest_reconciliation()
        assert stored is not None
        assert stored["status"] in {"reconciled", "tolerated", "mismatch"}

    def test_tolerances_are_applied(self, config, prices):
        reconciler = Reconciler(config.execution.reconciliation)
        symbol = prices.index[0]
        price = float(prices.iloc[0])
        # A tiny fractional difference on a small position is tolerated.
        report = reconciler.reconcile(
            {symbol: 10.0}, {symbol: 10.005}, prices=pd.Series({symbol: price})
        )
        assert report.reconciled


class TestPersistedExecution:
    def test_orders_and_fills_are_stored(self, paper_config, short_panel, prices, repository):
        broker = MockBrokerClient(prices=prices, initial_cash=100_000.0, account="DU1234567")
        broker.connect()

        manager = OrderManager(paper_config, broker=broker, repository=repository)
        result = manager.run_cycle(short_panel, submit=True)
        assert result.submitted

        stored = repository.recent_orders(session_id=manager.session_id)
        assert len(stored) == len(result.submitted)
        assert set(stored["execution_mode"]) == {"paper"}


class TestPriceScaleAgainstTheBroker:
    """Marks used against a broker must be on the broker's own price scale.

    Vendors report OHLC raw and supply a separate adjusted close; a broker
    quotes and settles raw. Sizing orders off an adjusted close overstates the
    share count by the whole accumulated dividend adjustment.
    """

    def test_marks_fall_back_to_the_raw_close(self, config, short_panel):
        manager = OrderManager(config, broker=None)
        reference = short_panel.dates.max()
        marks = manager._current_prices(short_panel, reference)

        expected = short_panel.close.loc[:reference].ffill().iloc[-1]
        pd.testing.assert_series_equal(marks, expected, check_names=False)

    def test_a_quote_gap_is_filled_on_the_quote_scale(self, config, short_panel, prices):
        """A symbol the broker cannot quote must not be marked on a second scale."""
        missing = prices.index[0]
        broker = MockBrokerClient(prices=prices.drop(missing), initial_cash=100_000.0)
        broker.connect()
        manager = OrderManager(config, broker=broker)
        reference = short_panel.dates.max()

        marks = manager._current_prices(short_panel, reference)

        raw = short_panel.close.loc[:reference].ffill().iloc[-1]
        assert marks[missing] == pytest.approx(raw[missing])
        # And the gap-filled name sits on the same scale as the quoted ones.
        quoted = [s for s in marks.index if s != missing]
        pd.testing.assert_series_equal(
            marks[quoted],
            prices.drop(missing)[quoted],
            check_names=False,
            check_index_type=False,
        )

    def test_the_mock_broker_agrees_with_the_local_marks(self, config, short_panel, prices):
        """No position mismatch can arise purely from a valuation scale."""
        broker = MockBrokerClient(prices=prices, initial_cash=100_000.0)
        broker.connect()
        manager = OrderManager(config, broker=broker)
        reference = short_panel.dates.max()

        marks = manager._current_prices(short_panel, reference)
        quotes = broker.get_market_prices(list(short_panel.symbols))
        pd.testing.assert_series_equal(marks, quotes, check_names=False)


class TestSafetyGuarantees:
    def test_live_mode_cannot_be_selected(self, config):
        from atlas.execution.safety import SafetyGate

        gate = SafetyGate(config)
        gate.execution.__dict__["mode"] = ExecutionMode.LIVE
        with pytest.raises(ExecutionBlocked):
            gate.assert_not_live()

    def test_submission_requires_paper_mode(self, config, short_panel, broker):
        config.execution.mode = ExecutionMode.DRY_RUN
        manager = OrderManager(config, broker=broker)
        assert not manager.gate.may_submit_orders
        result = manager.run_cycle(short_panel)
        assert not result.submitted

    def test_no_broker_means_no_submission(self, config, short_panel):
        manager = OrderManager(config, broker=None)
        plan, _ = manager.build_plan(short_panel)
        with pytest.raises(ExecutionBlocked, match="without a broker"):
            manager._submit(plan)

    def test_order_submission_is_never_retried(self, config, prices):
        """A failed submission must surface, not be re-sent."""
        symbol = prices.index[0]
        broker = MockBrokerClient(prices=prices, reject_symbols={symbol})
        broker.connect()
        order = Order(symbol=symbol, quantity=10.0, side=OrderSide.BUY)
        with pytest.raises(OrderRejected):
            broker.submit_order(order)
        assert len(broker.submitted_orders) == 1, "the order must not be re-submitted"
        assert config.execution.broker.order_retry_attempts == 0
