"""Tests for costs, orders, fills, position accounting and the simulated broker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.broker import ExecutionContext, SimulatedBroker
from atlas.backtest.costs import CostBreakdown, TransactionCostModel
from atlas.backtest.orders import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    new_order_id,
)
from atlas.backtest.positions import Portfolio, Position
from atlas.config import CostConfig, FillConfig
from atlas.exceptions import AtlasError


@pytest.fixture
def cost_model() -> TransactionCostModel:
    return TransactionCostModel(
        CostConfig(
            commission_per_order=0.0,
            commission_per_share=0.005,
            min_commission=1.0,
            max_commission_pct=0.005,
            half_spread_bps=1.0,
            slippage_bps=2.0,
            market_impact_enabled=True,
            market_impact_coefficient=0.10,
            max_participation=0.05,
        )
    )


class TestCostModel:
    def test_per_share_commission(self, cost_model):
        assert cost_model.commission(1000, 100.0) == pytest.approx(5.0)

    def test_minimum_commission(self, cost_model):
        assert cost_model.commission(10, 100.0) == pytest.approx(1.0)

    def test_percentage_cap(self, cost_model):
        # 100 shares at $1 = $100 notional; the 0.5% cap gives $0.50, below the
        # $1.00 minimum, and the cap wins.
        assert cost_model.commission(100, 1.0) == pytest.approx(0.50)

    def test_spread_and_slippage_in_bps(self, cost_model):
        notional = 100 * 50.0
        assert cost_model.spread_cost(100, 50.0) == pytest.approx(notional * 0.0001)
        assert cost_model.slippage_cost(100, 50.0) == pytest.approx(notional * 0.0002)

    def test_impact_grows_with_participation(self, cost_model):
        small = cost_model.market_impact_cost(1_000, 100.0, adv_shares=1_000_000)
        large = cost_model.market_impact_cost(50_000, 100.0, adv_shares=1_000_000)
        assert large > small
        # Square-root law: 50x the size gives more than 50x but less than 50^2 cost.
        assert large / small > 50
        assert large / small < 2500

    def test_impact_zero_without_volume(self, cost_model):
        assert cost_model.market_impact_cost(1000, 100.0, adv_shares=None) == 0.0

    def test_impact_can_be_disabled(self):
        model = TransactionCostModel(CostConfig(market_impact_enabled=False))
        assert model.market_impact_cost(1000, 100.0, adv_shares=1_000_000) == 0.0

    def test_costs_are_itemised(self, cost_model):
        costs = cost_model.compute(1000, 100.0, adv_shares=1_000_000)
        assert costs.commission > 0
        assert costs.spread > 0
        assert costs.slippage > 0
        assert costs.market_impact > 0
        assert costs.total == pytest.approx(
            costs.commission + costs.spread + costs.slippage + costs.market_impact
        )

    def test_buy_fills_above_the_reference(self, cost_model):
        price, _ = cost_model.effective_price(1000, 100.0, adv_shares=1_000_000)
        assert price > 100.0

    def test_sell_fills_below_the_reference(self, cost_model):
        price, _ = cost_model.effective_price(-1000, 100.0, adv_shares=1_000_000)
        assert price < 100.0

    def test_commission_excluded_from_the_fill_price(self, cost_model):
        price, costs = cost_model.effective_price(1000, 100.0, adv_shares=1_000_000)
        expected = 100.0 + costs.price_costs / 1000
        assert price == pytest.approx(expected)

    def test_zero_quantity_costs_nothing(self, cost_model):
        assert cost_model.compute(0, 100.0).total == 0.0

    def test_participation_limit(self, cost_model):
        assert cost_model.max_fillable_shares(1_000_000) == pytest.approx(50_000)
        assert cost_model.max_fillable_shares(None) == float("inf")

    def test_breakdown_addition(self):
        a = CostBreakdown(commission=1.0, spread=2.0)
        b = CostBreakdown(commission=3.0, slippage=4.0)
        total = a + b
        assert total.commission == 4.0
        assert total.total == 10.0

    def test_describe_is_human_readable(self, cost_model):
        described = cost_model.describe()
        assert "bps" in described["half_spread"]
        assert "ADV" in described["max_participation"]


class TestOrders:
    def test_ids_are_unique(self):
        assert len({new_order_id() for _ in range(500)}) == 500

    def test_from_signed_quantity(self):
        buy = Order.from_signed_quantity("SPY", 100.0)
        sell = Order.from_signed_quantity("SPY", -100.0)
        assert buy.side is OrderSide.BUY and buy.quantity == 100.0
        assert sell.side is OrderSide.SELL and sell.quantity == 100.0
        assert sell.signed_quantity == -100.0

    def test_negative_quantity_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Order(symbol="SPY", quantity=-10.0, side=OrderSide.BUY)

    def test_symbol_is_upper_cased(self):
        assert Order(symbol="spy", quantity=1.0, side=OrderSide.BUY).symbol == "SPY"

    def test_partial_then_full_fill(self):
        order = Order(symbol="SPY", quantity=100.0, side=OrderSide.BUY)
        order.apply_fill(Fill("x", "SPY", 40.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == pytest.approx(60.0)
        order.apply_fill(Fill("x", "SPY", 60.0, 110.0, OrderSide.BUY, pd.Timestamp("2020-01-02")))
        assert order.status is OrderStatus.FILLED
        assert order.average_fill_price == pytest.approx(0.4 * 100.0 + 0.6 * 110.0)

    def test_reject_and_cancel(self):
        order = Order(symbol="SPY", quantity=10.0, side=OrderSide.BUY)
        order.reject("no price")
        assert order.status is OrderStatus.REJECTED and order.is_terminal
        other = Order(symbol="SPY", quantity=10.0, side=OrderSide.BUY)
        other.cancel("changed mind")
        assert other.status is OrderStatus.CANCELLED

    def test_fill_cash_flow_signs(self):
        buy = Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01"), commission=1.0)
        sell = Fill("o", "SPY", 10.0, 100.0, OrderSide.SELL, pd.Timestamp("2020-01-01"), commission=1.0)
        assert buy.cash_flow == pytest.approx(-1001.0)
        assert sell.cash_flow == pytest.approx(999.0)


class TestPositionAccounting:
    def test_open_and_value(self):
        position = Position("SPY")
        position.apply_fill(Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        assert position.quantity == 10.0
        assert position.average_price == 100.0
        assert position.market_value(110.0) == pytest.approx(1100.0)
        assert position.unrealized_pnl(110.0) == pytest.approx(100.0)

    def test_adding_blends_the_cost_basis(self):
        position = Position("SPY")
        position.apply_fill(Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        position.apply_fill(Fill("o", "SPY", 10.0, 120.0, OrderSide.BUY, pd.Timestamp("2020-01-02")))
        assert position.average_price == pytest.approx(110.0)

    def test_partial_close_realises_pnl(self):
        position = Position("SPY")
        position.apply_fill(Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        realized = position.apply_fill(
            Fill("o", "SPY", 4.0, 130.0, OrderSide.SELL, pd.Timestamp("2020-01-02"))
        )
        assert realized == pytest.approx(120.0)
        assert position.quantity == pytest.approx(6.0)
        assert position.average_price == pytest.approx(100.0)

    def test_full_close(self):
        position = Position("SPY")
        position.apply_fill(Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        position.apply_fill(Fill("o", "SPY", 10.0, 90.0, OrderSide.SELL, pd.Timestamp("2020-01-02")))
        assert not position.is_open
        assert position.realized_pnl == pytest.approx(-100.0)

    def test_reversal_through_zero(self):
        position = Position("SPY")
        position.apply_fill(Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        position.apply_fill(Fill("o", "SPY", 25.0, 110.0, OrderSide.SELL, pd.Timestamp("2020-01-02")))
        assert position.quantity == pytest.approx(-15.0)
        assert position.average_price == pytest.approx(110.0)
        assert position.realized_pnl == pytest.approx(100.0)

    def test_short_position_pnl(self):
        position = Position("SPY")
        position.apply_fill(Fill("o", "SPY", 10.0, 100.0, OrderSide.SELL, pd.Timestamp("2020-01-01")))
        assert position.quantity == -10.0
        assert position.unrealized_pnl(90.0) == pytest.approx(100.0)


class TestPortfolio:
    def test_equity_identity(self):
        portfolio = Portfolio(100_000.0)
        portfolio.apply_fill(
            Fill("o", "SPY", 100.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01"), commission=1.0)
        )
        prices = pd.Series({"SPY": 105.0})
        assert portfolio.equity(prices) == pytest.approx(
            portfolio.cash + portfolio.positions_value(prices)
        )
        assert portfolio.cash == pytest.approx(100_000.0 - 10_000.0 - 1.0)

    def test_weights_sum_correctly(self):
        portfolio = Portfolio(100_000.0)
        portfolio.apply_fill(Fill("o", "SPY", 100.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        portfolio.apply_fill(Fill("o", "TLT", 100.0, 50.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        prices = pd.Series({"SPY": 100.0, "TLT": 50.0})
        weights = portfolio.weights(prices)
        assert weights["SPY"] == pytest.approx(0.10)
        assert weights["TLT"] == pytest.approx(0.05)

    def test_negative_cash_can_be_forbidden(self):
        portfolio = Portfolio(1_000.0, allow_negative_cash=False)
        with pytest.raises(AtlasError, match="drive cash"):
            portfolio.apply_fill(
                Fill("o", "SPY", 100.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01"))
            )

    def test_dividends_credit_cash(self):
        portfolio = Portfolio(100_000.0)
        portfolio.apply_fill(Fill("o", "SPY", 100.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01")))
        before = portfolio.cash
        credited = portfolio.credit_dividends(pd.Series({"SPY": 0.50}), pd.Timestamp("2020-03-01"))
        assert credited == pytest.approx(50.0)
        assert portfolio.cash == pytest.approx(before + 50.0)

    def test_snapshots_and_equity_curve(self):
        portfolio = Portfolio(100_000.0)
        for i in range(5):
            portfolio.snapshot(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), pd.Series(dtype=float))
        curve = portfolio.equity_curve()
        assert len(curve) == 5
        assert curve.iloc[0] == pytest.approx(100_000.0)

    def test_costs_accumulate(self):
        portfolio = Portfolio(100_000.0)
        portfolio.apply_fill(
            Fill("o", "SPY", 10.0, 100.0, OrderSide.BUY, pd.Timestamp("2020-01-01"),
                 commission=1.0, spread_cost=0.5, slippage_cost=0.25)
        )
        assert portfolio.cumulative_costs == pytest.approx(1.75)
        assert portfolio.cumulative_commission == pytest.approx(1.0)

    def test_invalid_capital_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            Portfolio(0.0)


class TestSimulatedBroker:
    @pytest.fixture
    def broker(self, cost_model) -> SimulatedBroker:
        return SimulatedBroker(cost_model, FillConfig(), fractional_shares=True)

    @pytest.fixture
    def context(self) -> ExecutionContext:
        return ExecutionContext(
            timestamp=pd.Timestamp("2020-06-01"),
            prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
            volumes=pd.Series({"SPY": 1_000_000.0, "TLT": 500_000.0}),
            volatilities=pd.Series({"SPY": 0.012, "TLT": 0.008}),
        )

    def test_fills_a_valid_order(self, broker, context):
        order = Order(symbol="SPY", quantity=100.0, side=OrderSide.BUY)
        fill = broker.execute(order, context)
        assert fill is not None
        assert fill.quantity == pytest.approx(100.0)
        assert fill.price > 100.0
        assert order.status is OrderStatus.FILLED

    def test_rejects_when_no_price(self, broker, context):
        order = Order(symbol="GLD", quantity=100.0, side=OrderSide.BUY)
        assert broker.execute(order, context) is None
        assert order.status is OrderStatus.REJECTED
        assert "no execution price" in order.reject_reason

    def test_partial_fill_on_participation_limit(self, broker, context):
        # 5% of 1,000,000 ADV = 50,000 shares maximum.
        order = Order(symbol="SPY", quantity=200_000.0, side=OrderSide.BUY)
        fill = broker.execute(order, context)
        assert fill is not None
        assert fill.quantity == pytest.approx(50_000.0)
        assert order.status is OrderStatus.PARTIALLY_FILLED

    def test_whole_share_mode_truncates(self, cost_model, context):
        broker = SimulatedBroker(cost_model, FillConfig(), fractional_shares=False)
        order = Order(symbol="SPY", quantity=10.7, side=OrderSide.BUY)
        fill = broker.execute(order, context)
        assert fill.quantity == pytest.approx(10.0)

    def test_whole_share_mode_rejects_sub_share_orders(self, cost_model, context):
        broker = SimulatedBroker(cost_model, FillConfig(), fractional_shares=False)
        order = Order(symbol="SPY", quantity=0.4, side=OrderSide.BUY)
        assert broker.execute(order, context) is None
        assert order.status is OrderStatus.REJECTED

    def test_limit_order_not_crossed(self, broker, context):
        order = Order(
            symbol="SPY", quantity=10.0, side=OrderSide.BUY,
            order_type=OrderType.LIMIT, limit_price=90.0,
        )
        assert broker.execute(order, context) is None
        assert order.status is OrderStatus.CANCELLED

    def test_limit_order_crossed(self, broker, context):
        order = Order(
            symbol="SPY", quantity=10.0, side=OrderSide.BUY,
            order_type=OrderType.LIMIT, limit_price=110.0,
        )
        assert broker.execute(order, context) is not None

    def test_sells_execute_before_buys(self, broker, context):
        buy = Order(symbol="SPY", quantity=10.0, side=OrderSide.BUY)
        sell = Order(symbol="TLT", quantity=10.0, side=OrderSide.SELL)
        fills, _ = broker.execute_batch([buy, sell], context)
        assert fills[0].symbol == "TLT", "sells must settle first to fund the buys"

    def test_deterministic(self, cost_model, context):
        results = []
        for _ in range(3):
            broker = SimulatedBroker(cost_model, FillConfig())
            order = Order(symbol="SPY", quantity=1234.0, side=OrderSide.BUY)
            results.append(broker.execute(order, context).price)
        assert len(set(results)) == 1

    def test_context_handles_missing_data(self):
        context = ExecutionContext(
            timestamp=pd.Timestamp("2020-01-01"), prices=pd.Series({"SPY": np.nan})
        )
        assert context.price("SPY") is None
        assert context.price("MISSING") is None
        assert context.volume("SPY") is None
