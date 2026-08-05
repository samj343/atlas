"""Tests for volatility, covariance, constraints and allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.config import CovarianceConfig, PortfolioConfig, VolatilityConfig
from atlas.exceptions import InsufficientDataError
from atlas.portfolio.allocator import PortfolioAllocator
from atlas.portfolio.constraints import PortfolioConstraints
from atlas.portfolio.covariance import (
    CovarianceEstimator,
    correlation_from_covariance,
    nearest_positive_definite,
    portfolio_volatility,
    risk_contributions,
)
from atlas.portfolio.volatility import VolatilityEstimator, inverse_volatility_weights


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Four assets with known, distinct volatilities and a common factor."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2018-01-01", periods=600, name="date")
    factor = rng.standard_normal(len(index)) * 0.01
    data = {
        "LOW": 0.3 * factor + rng.standard_normal(len(index)) * 0.002,
        "MID": 0.8 * factor + rng.standard_normal(len(index)) * 0.005,
        "HIGH": 1.5 * factor + rng.standard_normal(len(index)) * 0.012,
        "CASH": rng.standard_normal(len(index)) * 0.0002,
    }
    return pd.DataFrame(data, index=index)


class TestVolatilityEstimator:
    def test_annualisation(self, sample_returns):
        estimator = VolatilityEstimator(VolatilityConfig(method="rolling", lookback=252, min_periods=60))
        estimates = estimator.estimate(sample_returns)
        realised = sample_returns["HIGH"].iloc[-252:].std(ddof=1) * np.sqrt(252)
        assert estimates["HIGH"].iloc[-1] == pytest.approx(realised, rel=1e-9)

    def test_ordering_is_preserved(self, sample_returns):
        estimator = VolatilityEstimator(VolatilityConfig())
        latest = estimator.estimate_latest(sample_returns)
        assert latest["CASH"] < latest["LOW"] < latest["MID"] < latest["HIGH"]

    def test_floor_is_applied(self, sample_returns):
        config = VolatilityConfig(min_annual_volatility=0.05)
        latest = VolatilityEstimator(config).estimate_latest(sample_returns)
        assert (latest >= 0.05 - 1e-12).all()

    def test_zero_volatility_series(self):
        index = pd.bdate_range("2020-01-01", periods=200)
        flat = pd.DataFrame({"FLAT": np.zeros(len(index))}, index=index)
        latest = VolatilityEstimator(VolatilityConfig()).estimate_latest(flat)
        assert latest["FLAT"] == pytest.approx(VolatilityConfig().min_annual_volatility)

    def test_empty_frame_raises(self):
        with pytest.raises(InsufficientDataError):
            VolatilityEstimator(VolatilityConfig()).estimate_latest(pd.DataFrame())

    def test_inverse_volatility_weights(self):
        signals = pd.Series({"A": 1.0, "B": 1.0})
        volatilities = pd.Series({"A": 0.10, "B": 0.20})
        weights = inverse_volatility_weights(signals, volatilities)
        # Equal signals, double the volatility -> half the weight.
        assert weights["A"] == pytest.approx(2.0 * weights["B"])

    def test_missing_volatility_gives_zero_weight(self):
        signals = pd.Series({"A": 1.0, "B": 1.0})
        volatilities = pd.Series({"A": 0.10, "B": np.nan})
        weights = inverse_volatility_weights(signals, volatilities)
        assert weights["B"] == 0.0


class TestCovarianceEstimator:
    @pytest.mark.parametrize("method", ["sample", "shrinkage", "ewma"])
    def test_estimators_produce_valid_matrices(self, sample_returns, method):
        estimator = CovarianceEstimator(CovarianceConfig(method=method, lookback=252))
        result = estimator.estimate(sample_returns)
        matrix = result.matrix.to_numpy()
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix, matrix.T), "covariance must be symmetric"
        assert np.linalg.eigvalsh(matrix).min() > 0, "covariance must be positive definite"

    def test_shrinkage_is_better_conditioned(self, sample_returns):
        short = sample_returns.tail(70)
        sample = CovarianceEstimator(
            CovarianceConfig(method="sample", lookback=70, min_observations=60)
        ).estimate(short)
        shrunk = CovarianceEstimator(
            CovarianceConfig(method="shrinkage", lookback=70, min_observations=60)
        ).estimate(short)
        assert shrunk.condition_number() <= sample.condition_number() * 1.01

    def test_insufficient_observations_raise(self, sample_returns):
        estimator = CovarianceEstimator(CovarianceConfig(min_observations=500, lookback=100))
        with pytest.raises(InsufficientDataError, match="complete observations"):
            estimator.estimate(sample_returns)

    def test_fallback_to_diagonal(self, sample_returns):
        estimator = CovarianceEstimator(CovarianceConfig(min_observations=5000))
        result = estimator.estimate_or_fallback(sample_returns)
        assert result.method == "diagonal_fallback"
        off_diagonal = result.matrix.to_numpy()[np.triu_indices(4, k=1)]
        assert np.allclose(off_diagonal, 0.0)

    def test_singular_matrix_is_repaired(self):
        """Perfectly collinear assets produce a singular matrix; it must be fixed."""
        index = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(1)
        base = rng.standard_normal(len(index)) * 0.01
        frame = pd.DataFrame({"A": base, "B": base, "C": base * 2.0}, index=index)
        result = CovarianceEstimator(CovarianceConfig(method="sample")).estimate(frame)
        assert np.linalg.eigvalsh(result.matrix.to_numpy()).min() > 0
        assert result.repaired

    def test_nearest_positive_definite(self):
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
        repaired = nearest_positive_definite(matrix)
        assert np.linalg.eigvalsh(repaired).min() > 0

    def test_volatility_floor_applied(self, sample_returns):
        config = CovarianceConfig(method="sample", min_annual_volatility=0.10)
        result = CovarianceEstimator(config).estimate(sample_returns)
        assert (result.volatilities >= 0.10 - 1e-9).all()

    def test_correlation_from_covariance(self, sample_returns):
        result = CovarianceEstimator(CovarianceConfig()).estimate(sample_returns)
        correlation = correlation_from_covariance(result.matrix)
        assert np.allclose(np.diag(correlation.to_numpy()), 1.0)
        assert (correlation.to_numpy() <= 1.0 + 1e-9).all()
        assert (correlation.to_numpy() >= -1.0 - 1e-9).all()

    def test_portfolio_volatility_matches_formula(self, sample_returns):
        result = CovarianceEstimator(CovarianceConfig()).estimate(sample_returns)
        weights = pd.Series(0.25, index=result.matrix.columns)
        expected = float(
            np.sqrt(weights.to_numpy() @ result.matrix.to_numpy() @ weights.to_numpy())
        )
        assert portfolio_volatility(weights, result.matrix) == pytest.approx(expected)

    def test_portfolio_volatility_handles_misaligned_index(self, sample_returns):
        result = CovarianceEstimator(CovarianceConfig()).estimate(sample_returns)
        weights = pd.Series({"HIGH": 0.5, "LOW": 0.5, "NOT_IN_MATRIX": 1.0})
        assert portfolio_volatility(weights, result.matrix) > 0

    def test_risk_contributions_sum_to_one(self, sample_returns):
        result = CovarianceEstimator(CovarianceConfig()).estimate(sample_returns)
        weights = pd.Series({"LOW": 0.4, "MID": 0.3, "HIGH": 0.2, "CASH": 0.1})
        contributions = risk_contributions(weights, result.matrix)
        assert contributions.sum() == pytest.approx(1.0)
        # The most volatile asset should not have the smallest contribution.
        assert contributions["HIGH"] > contributions["CASH"]


class TestConstraints:
    @pytest.fixture
    def asset_classes(self) -> dict[str, str]:
        return {"A": "equity", "B": "equity", "C": "bond", "D": "commodity"}

    @pytest.fixture
    def constraints(self, asset_classes) -> PortfolioConstraints:
        config = PortfolioConfig(
            max_asset_weight=0.20,
            max_asset_class_weight=0.50,
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
            max_leverage=1.0,
            minimum_cash_weight=0.05,
            maximum_daily_turnover=0.30,
            min_trade_weight=0.005,
            max_active_positions=10,
            long_only=True,
        )
        return PortfolioConstraints(config, asset_classes)

    def test_asset_cap_enforced(self, constraints):
        weights = pd.Series({"A": 0.60, "B": 0.10, "C": 0.10, "D": 0.10})
        result = constraints.apply(weights)
        assert (result.weights <= 0.20 + 1e-9).all()

    def test_class_cap_enforced(self, constraints):
        weights = pd.Series({"A": 0.20, "B": 0.20, "C": 0.10, "D": 0.10})
        # equity = 0.40, under the 0.50 cap
        assert constraints.check(constraints.apply(weights).weights)["max_asset_class_weight"]

    def test_class_cap_binds(self, asset_classes):
        config = PortfolioConfig(max_asset_class_weight=0.25, max_asset_weight=0.5)
        constraints = PortfolioConstraints(config, asset_classes)
        weights = pd.Series({"A": 0.30, "B": 0.30, "C": 0.10, "D": 0.10})
        result = constraints.apply(weights)
        equity = result.weights[["A", "B"]].abs().sum()
        assert equity <= 0.25 + 1e-9
        assert any(a.constraint == "max_asset_class_weight" for a in result.adjustments)

    def test_long_only_removes_shorts(self, constraints):
        weights = pd.Series({"A": 0.20, "B": -0.20, "C": 0.10, "D": 0.10})
        result = constraints.apply(weights)
        assert (result.weights >= 0.0).all()
        assert any(a.constraint == "long_only" for a in result.adjustments)

    def test_cash_reserve_enforced(self, constraints):
        weights = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20})
        result = constraints.apply(weights)
        assert result.gross_exposure <= 0.95 + 1e-9

    def test_gross_exposure_cap(self, asset_classes):
        config = PortfolioConfig(
            max_gross_exposure=0.6, max_leverage=0.6, max_net_exposure=0.6,
            minimum_cash_weight=0.0, max_asset_weight=1.0, max_asset_class_weight=1.0,
            # The turnover cap is disabled here so the exposure cap is what binds;
            # it is tested separately in test_turnover_cap.
            maximum_daily_turnover=5.0,
        )
        constraints = PortfolioConstraints(config, asset_classes)
        weights = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5})
        result = constraints.apply(weights)
        assert result.gross_exposure == pytest.approx(0.6, abs=1e-9)

    def test_position_count_limit(self, asset_classes):
        config = PortfolioConfig(max_active_positions=2, max_asset_weight=1.0, max_asset_class_weight=1.0)
        constraints = PortfolioConstraints(config, asset_classes)
        weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
        result = constraints.apply(weights)
        assert result.n_positions == 2
        assert result.weights["D"] == 0.0

    def test_turnover_cap(self, constraints):
        previous = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
        target = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20})
        result = constraints.apply(target, previous_weights=previous)
        turnover = float((result.weights - previous).abs().sum())
        assert turnover <= 0.30 + 1e-9

    def test_min_trade_threshold_reverts_small_trades(self, constraints):
        previous = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.10})
        target = pd.Series({"A": 0.1005, "B": 0.10, "C": 0.10, "D": 0.10})
        result = constraints.apply(target, previous_weights=previous)
        assert result.weights["A"] == pytest.approx(0.10)

    def test_exit_trades_are_never_reverted(self, constraints):
        previous = pd.Series({"A": 0.002, "B": 0.10, "C": 0.10, "D": 0.10})
        target = pd.Series({"A": 0.0, "B": 0.10, "C": 0.10, "D": 0.10})
        result = constraints.apply(target, previous_weights=previous)
        assert result.weights["A"] == 0.0, "a full exit must not be blocked by the small-trade rule"

    def test_final_weights_satisfy_all_limits(self, constraints):
        rng = np.random.default_rng(7)
        for _ in range(25):
            weights = pd.Series(rng.uniform(-1.0, 1.0, 4), index=["A", "B", "C", "D"])
            result = constraints.apply(weights)
            checks = constraints.check(result.weights)
            assert all(checks.values()), f"constraint violation: {checks}"

    def test_empty_weights(self, constraints):
        result = constraints.apply(pd.Series(dtype="float64"))
        assert result.weights.empty
        assert result.gross_exposure == 0.0

    def test_adjustments_are_recorded(self, constraints):
        weights = pd.Series({"A": 0.90, "B": 0.10, "C": 0.0, "D": 0.0})
        result = constraints.apply(weights)
        assert result.was_adjusted()
        frame = result.to_frame()
        assert "reason" in frame.columns
        assert frame["reason"].str.len().min() > 0


class TestAllocator:
    def test_targets_volatility(self, config, panel):
        allocator = PortfolioAllocator(config)
        returns = panel.returns("adj_close").tail(500)
        signals = pd.Series(1.0, index=panel.symbols)
        result = allocator.allocate(signals, returns)
        predicted = result.realized_predicted_volatility()
        # The target may be reduced by the constraint layer but must not be exceeded.
        assert predicted <= config.portfolio.max_portfolio_volatility + 1e-6

    def test_zero_signals_hold_cash(self, config, panel):
        allocator = PortfolioAllocator(config)
        returns = panel.returns("adj_close").tail(400)
        result = allocator.allocate(pd.Series(0.0, index=panel.symbols), returns)
        assert result.gross_exposure == 0.0
        assert "cash" in result.diagnostics.get("reason", "")

    def test_risk_multiplier_reduces_exposure(self, config, panel):
        # Raise the turnover cap so the risk multiplier, not the cap, decides the
        # resulting exposure. From an all-cash book the cap would bind on both.
        config.portfolio.maximum_daily_turnover = 5.0
        allocator = PortfolioAllocator(config)
        returns = panel.returns("adj_close").tail(500)
        signals = pd.Series(0.8, index=panel.symbols)
        full = allocator.allocate(signals, returns, risk_multiplier=1.0)
        halved = allocator.allocate(signals, returns, risk_multiplier=0.5)
        assert halved.gross_exposure < full.gross_exposure

    def test_turnover_cap_binds_from_an_all_cash_start(self, config, panel):
        """Reaching a target from cash takes several rebalances, by design."""
        allocator = PortfolioAllocator(config)
        returns = panel.returns("adj_close").tail(500)
        signals = pd.Series(0.8, index=panel.symbols)
        result = allocator.allocate(signals, returns)
        assert result.gross_exposure == pytest.approx(
            config.portfolio.maximum_daily_turnover, abs=1e-6
        )

    def test_defensive_allocation(self, config, panel):
        allocator = PortfolioAllocator(config)
        returns = panel.returns("adj_close").tail(500)
        config.portfolio.maximum_daily_turnover = 5.0
        signals = pd.Series(0.5, index=panel.symbols)
        result = allocator.allocate(signals, returns, defensive_weight=0.20)
        assert result.weights.get("BIL", 0.0) > 0.0

    def test_constraints_are_satisfied(self, config, panel):
        allocator = PortfolioAllocator(config)
        returns = panel.returns("adj_close").tail(500)
        signals = pd.Series(np.linspace(-1, 1, len(panel.symbols)), index=panel.symbols)
        result = allocator.allocate(signals, returns)
        checks = allocator.constraints.check(result.weights)
        assert all(checks.values()), f"constraint violation: {checks}"

    def test_risk_posture_by_regime(self, config):
        from atlas.config import RegimeLabel

        allocator = PortfolioAllocator(config)
        calm, calm_defensive = allocator.risk_posture(RegimeLabel.BULL_LOW_VOL)
        stressed, stressed_defensive = allocator.risk_posture(RegimeLabel.BEAR_HIGH_VOL)
        assert stressed < calm, "a bearish, high-volatility regime must reduce the risk budget"
        assert stressed_defensive > calm_defensive

    def test_drawdown_multiplier_compounds_with_regime(self, config):
        from atlas.config import RegimeLabel

        allocator = PortfolioAllocator(config)
        base, _ = allocator.risk_posture(RegimeLabel.BULL_LOW_VOL, drawdown_multiplier=1.0)
        reduced, _ = allocator.risk_posture(RegimeLabel.BULL_LOW_VOL, drawdown_multiplier=0.5)
        assert reduced == pytest.approx(base * 0.5)

    def test_breadth_overlay_adds_defensive_weight(self, config):
        from atlas.config import RegimeLabel

        allocator = PortfolioAllocator(config)
        _, normal = allocator.risk_posture(RegimeLabel.BULL_LOW_VOL, weak_breadth=False)
        _, weak = allocator.risk_posture(RegimeLabel.BULL_LOW_VOL, weak_breadth=True)
        assert weak > normal
