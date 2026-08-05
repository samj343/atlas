"""Tests for the performance and risk metrics.

Where possible each metric is checked against a closed-form value on a
hand-constructed series, so a refactor that changes a convention fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.metrics import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    comparison_table,
    compute_metrics,
    drawdown_series,
    drawdown_stats,
    equity_from_returns,
    expected_shortfall,
    max_drawdown,
    monthly_return_table,
    monthly_returns,
    return_contribution,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    value_at_risk,
)


class TestReturnMetrics:
    def test_total_return_compounds(self, known_returns):
        expected = 1.1**10 * 0.9**5 - 1.0
        assert total_return(known_returns) == pytest.approx(expected)

    def test_cagr_of_a_constant_series(self):
        """A series returning exactly r each day for a year has CAGR (1+r)^252 - 1."""
        index = pd.bdate_range("2020-01-01", periods=TRADING_DAYS)
        returns = pd.Series(0.0004, index=index)
        assert annualized_return(returns) == pytest.approx(1.0004**TRADING_DAYS - 1.0, rel=1e-6)

    def test_annualized_volatility(self):
        rng = np.random.default_rng(0)
        index = pd.bdate_range("2020-01-01", periods=2520)
        returns = pd.Series(rng.standard_normal(2520) * 0.01, index=index)
        expected = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        assert annualized_volatility(returns) == pytest.approx(expected)

    def test_total_loss_gives_minus_one(self):
        index = pd.bdate_range("2020-01-01", periods=10)
        returns = pd.Series([-1.0] + [0.0] * 9, index=index)
        assert annualized_return(returns) == pytest.approx(-1.0)

    def test_empty_series(self):
        empty = pd.Series(dtype="float64")
        assert total_return(empty) == 0.0
        assert annualized_return(empty) == 0.0
        assert annualized_volatility(empty) == 0.0


class TestRiskAdjustedRatios:
    def test_sharpe_with_zero_risk_free(self):
        rng = np.random.default_rng(1)
        index = pd.bdate_range("2015-01-01", periods=2520)
        returns = pd.Series(rng.standard_normal(2520) * 0.01 + 0.0005, index=index)
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        assert sharpe_ratio(returns, 0.0) == pytest.approx(expected)

    def test_risk_free_rate_lowers_sharpe(self):
        rng = np.random.default_rng(2)
        index = pd.bdate_range("2015-01-01", periods=1260)
        returns = pd.Series(rng.standard_normal(1260) * 0.01 + 0.0005, index=index)
        assert sharpe_ratio(returns, 0.05) < sharpe_ratio(returns, 0.0)

    def test_risk_free_is_de_annualised_geometrically(self):
        index = pd.bdate_range("2020-01-01", periods=TRADING_DAYS)
        periodic = 1.05 ** (1 / TRADING_DAYS) - 1.0
        returns = pd.Series(periodic, index=index)
        # A portfolio earning exactly the risk-free rate has zero excess return.
        assert sharpe_ratio(returns, 0.05) == pytest.approx(0.0, abs=1e-9)

    def test_zero_volatility_gives_zero_sharpe(self, flat_returns):
        assert sharpe_ratio(flat_returns) == 0.0
        assert sortino_ratio(flat_returns) == 0.0

    def test_sortino_exceeds_sharpe_for_positive_skew(self):
        """Rare large gains inflate volatility but not downside deviation."""
        index = pd.bdate_range("2020-01-01", periods=500)
        values = np.full(500, -0.0002)
        values[::50] = 0.05  # occasional large gains, only small losses
        returns = pd.Series(values, index=index)
        assert sortino_ratio(returns) > sharpe_ratio(returns)

    def test_sortino_is_undefined_without_downside(self):
        """A series that never falls below the target has no denominator.

        Returning 0.0 would rank such a series worst, so it returns NaN instead.
        """
        index = pd.bdate_range("2020-01-01", periods=100)
        assert np.isnan(sortino_ratio(pd.Series(0.001, index=index), 0.0))

    def test_calmar_ratio(self, known_returns):
        equity = equity_from_returns(known_returns)
        expected = annualized_return(known_returns) / max_drawdown(equity)
        assert calmar_ratio(known_returns, equity) == pytest.approx(expected)

    def test_calmar_with_no_drawdown(self):
        index = pd.bdate_range("2020-01-01", periods=100)
        returns = pd.Series(0.001, index=index)
        assert calmar_ratio(returns) == 0.0


class TestDrawdown:
    def test_known_drawdown(self, known_returns):
        equity = equity_from_returns(known_returns)
        assert max_drawdown(equity) == pytest.approx(1.0 - 0.9**5)

    def test_drawdown_series_is_non_positive(self, known_returns):
        dd = drawdown_series(equity_from_returns(known_returns))
        assert (dd <= 1e-12).all()

    def test_monotonic_growth_has_no_drawdown(self):
        index = pd.bdate_range("2020-01-01", periods=100)
        equity = pd.Series(np.linspace(100, 200, 100), index=index)
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_drawdown_stats_shape(self, known_returns):
        stats = drawdown_stats(equity_from_returns(known_returns))
        for key in (
            "max_drawdown", "average_drawdown", "max_drawdown_duration_days",
            "n_drawdown_episodes", "time_in_drawdown_pct",
        ):
            assert key in stats
        assert stats["n_drawdown_episodes"] >= 1

    def test_drawdown_survives_missing_observations(self):
        index = pd.bdate_range("2020-01-01", periods=50)
        values = np.linspace(100, 120, 50)
        values[20:25] = np.nan
        equity = pd.Series(values, index=index)
        assert np.isfinite(max_drawdown(equity))

    def test_empty_equity(self):
        assert max_drawdown(pd.Series(dtype="float64")) == 0.0
        assert drawdown_series(pd.Series(dtype="float64")).empty


class TestTailRisk:
    def test_var_is_a_positive_loss(self):
        index = pd.bdate_range("2020-01-01", periods=1000)
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.standard_normal(1000) * 0.01, index=index)
        var = value_at_risk(returns, 0.95)
        assert var > 0
        assert (returns < -var).mean() == pytest.approx(0.05, abs=0.02)

    def test_expected_shortfall_exceeds_var(self):
        index = pd.bdate_range("2020-01-01", periods=1000)
        rng = np.random.default_rng(4)
        returns = pd.Series(rng.standard_normal(1000) * 0.01, index=index)
        assert expected_shortfall(returns, 0.95) >= value_at_risk(returns, 0.95)

    def test_higher_confidence_gives_larger_var(self):
        index = pd.bdate_range("2020-01-01", periods=2000)
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.standard_normal(2000) * 0.01, index=index)
        assert value_at_risk(returns, 0.99) > value_at_risk(returns, 0.95)

    def test_all_positive_returns_give_zero_var(self):
        index = pd.bdate_range("2020-01-01", periods=100)
        assert value_at_risk(pd.Series(0.01, index=index)) == 0.0


class TestPeriodAggregation:
    def test_monthly_returns_compound(self):
        index = pd.bdate_range("2020-01-01", "2020-03-31")
        returns = pd.Series(0.001, index=index)
        monthly = monthly_returns(returns)
        assert len(monthly) == 3
        january = returns.loc["2020-01"]
        assert monthly.iloc[0] == pytest.approx((1.001 ** len(january)) - 1.0)

    def test_monthly_table_shape(self):
        index = pd.bdate_range("2020-01-01", "2022-12-31")
        rng = np.random.default_rng(6)
        returns = pd.Series(rng.standard_normal(len(index)) * 0.005, index=index)
        table = monthly_return_table(returns)
        assert list(table.index) == [2020, 2021, 2022]
        assert len(table.columns) == 12

    def test_rolling_sharpe(self):
        index = pd.bdate_range("2015-01-01", periods=1000)
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.standard_normal(1000) * 0.01, index=index)
        rolling = rolling_sharpe(returns, window=252)
        assert rolling.iloc[:251].isna().all()
        assert rolling.iloc[252:].notna().all()


class TestComputeMetrics:
    def test_all_required_metrics_present(self, known_returns):
        metrics = compute_metrics(known_returns, risk_free_rate=0.02)
        required = [
            "total_return", "cagr", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
            "average_drawdown", "max_drawdown_duration_days", "max_recovery_duration_days",
            "best_day", "worst_day", "best_month", "worst_month", "monthly_win_rate",
            "value_at_risk_95", "expected_shortfall_95", "skewness", "kurtosis",
        ]
        missing = [k for k in required if k not in metrics.values]
        assert not missing, f"missing metrics: {missing}"

    def test_optional_metrics_only_when_supplied(self, known_returns):
        without = compute_metrics(known_returns)
        assert "n_trades" not in without.values
        assert "annual_turnover" not in without.values

        turnover = pd.Series(0.05, index=known_returns.index)
        with_extras = compute_metrics(known_returns, turnover=turnover, n_trades=42, costs=100.0)
        assert with_extras.values["n_trades"] == 42
        assert with_extras.values["total_transaction_costs"] == 100.0
        assert "annual_turnover" in with_extras.values

    def test_risk_free_rate_is_stated(self, known_returns):
        metrics = compute_metrics(known_returns, risk_free_rate=0.03)
        conventions = metrics.describe_conventions()
        assert "3.00%" in conventions["risk_free_rate"]
        assert "252" in conventions["annualization"]

    def test_render_includes_conventions(self, known_returns):
        text = compute_metrics(known_returns, risk_free_rate=0.02).render()
        assert "risk_free_rate" in text
        assert "annualization" in text

    def test_comparison_table(self, known_returns):
        a = compute_metrics(known_returns, label="A")
        b = compute_metrics(known_returns * 0.5, label="B")
        table = comparison_table({"A": a, "B": b})
        assert list(table.columns) == ["A", "B"]
        assert "sharpe_ratio" in table.index

    def test_empty_returns(self):
        metrics = compute_metrics(pd.Series(dtype="float64"))
        assert metrics.values["total_return"] == 0.0
        assert metrics.sharpe == 0.0


class TestAttribution:
    def test_return_contribution_uses_lagged_weights(self):
        """Day t's contribution uses the weight held going *into* day t."""
        index = pd.bdate_range("2020-01-01", periods=4)
        # A is first held at the close of day 1, so it earns day 2, 3 and 4's
        # returns; the weight at day t-1 multiplies the return at day t.
        weights = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.0, 0.5, 0.5, 0.5]}, index=index)
        returns = pd.DataFrame(
            {"A": [0.10, 0.10, 0.10, 0.10], "B": [0.20, 0.20, 0.20, 0.20]}, index=index
        )
        contribution = return_contribution(weights, returns)
        # A: weights at days 1-3 apply to returns on days 2-4 -> 3 * 0.5 * 0.10.
        assert contribution["A"] == pytest.approx(0.15)
        # B: weight is zero going into day 2, so it earns only days 3 and 4.
        assert contribution["B"] == pytest.approx(0.20)

    def test_no_contribution_before_a_position_exists(self):
        index = pd.bdate_range("2020-01-01", periods=3)
        weights = pd.DataFrame({"A": [0.0, 0.0, 1.0]}, index=index)
        returns = pd.DataFrame({"A": [0.50, 0.50, 0.10]}, index=index)
        assert return_contribution(weights, returns)["A"] == pytest.approx(0.0)
