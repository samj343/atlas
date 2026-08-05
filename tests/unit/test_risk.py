"""Tests for the risk-management engine and the safety gate."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from atlas.config import DrawdownControlConfig, DrawdownThreshold, ExecutionMode
from atlas.exceptions import ExecutionBlocked, KillSwitchActive, RiskLimitBreached
from atlas.execution.safety import SafetyGate
from atlas.portfolio.risk_manager import RiskAction, RiskManager, RiskSeverity


@pytest.fixture
def manager(config) -> RiskManager:
    return RiskManager(config)


@pytest.fixture
def weights(config) -> pd.Series:
    symbols = config.symbols[:6]
    return pd.Series(0.10, index=symbols)


class TestDrawdownLadder:
    def test_multiplier_steps(self):
        ladder = DrawdownControlConfig(
            enabled=True,
            thresholds=[
                DrawdownThreshold(drawdown=0.05, multiplier=0.75),
                DrawdownThreshold(drawdown=0.10, multiplier=0.50),
                DrawdownThreshold(drawdown=0.15, multiplier=0.25),
            ],
        )
        assert ladder.multiplier_for(0.02)[0] == 1.00
        assert ladder.multiplier_for(0.05)[0] == 0.75
        assert ladder.multiplier_for(0.07)[0] == 0.75
        assert ladder.multiplier_for(0.10)[0] == 0.50
        assert ladder.multiplier_for(0.20)[0] == 0.25

    def test_disabled_ladder_returns_one(self):
        ladder = DrawdownControlConfig(enabled=False, thresholds=[
            DrawdownThreshold(drawdown=0.05, multiplier=0.5)
        ])
        assert ladder.multiplier_for(0.50)[0] == 1.0

    def test_non_monotonic_thresholds_rejected(self):
        with pytest.raises(ValueError, match="non-increasing"):
            DrawdownControlConfig(
                thresholds=[
                    DrawdownThreshold(drawdown=0.05, multiplier=0.50),
                    DrawdownThreshold(drawdown=0.10, multiplier=0.75),
                ]
            )

    def test_manager_tracks_drawdown(self, manager):
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        manager.update_equity(pd.Timestamp("2020-01-02"), 110_000.0)
        manager.update_equity(pd.Timestamp("2020-01-03"), 99_000.0)
        assert manager.current_drawdown() == pytest.approx(0.10)
        assert manager.drawdown_multiplier()[0] == 0.50

    def test_drawdown_scales_the_assessment(self, manager, weights):
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        assessment = manager.assess(
            pd.Timestamp("2020-06-01"), weights, equity=88_000.0
        )
        assert assessment.risk_multiplier < 1.0
        assert any(e.control == "drawdown_control" for e in assessment.events)


class TestPositionControls:
    def test_asset_weight_cap(self, manager, config):
        weights = pd.Series({config.symbols[0]: 0.90})
        assessment = manager.assess(pd.Timestamp("2020-01-01"), weights, equity=100_000.0)
        assert assessment.weights.abs().max() <= config.portfolio.max_asset_weight + 1e-9
        assert any(e.control == "max_asset_weight" for e in assessment.events)

    def test_asset_class_cap(self, manager, config):
        equities = [s for s, c in config.asset_class_map.items() if c == "us_equity"]
        weights = pd.Series(0.20, index=equities)
        assessment = manager.assess(pd.Timestamp("2020-01-01"), weights, equity=100_000.0)
        exposure = assessment.weights[equities].abs().sum()
        assert exposure <= config.portfolio.max_asset_class_weight + 1e-9

    def test_gross_exposure_cap(self, manager, config):
        weights = pd.Series(0.20, index=config.symbols)
        assessment = manager.assess(pd.Timestamp("2020-01-01"), weights, equity=100_000.0)
        cap = min(config.portfolio.max_gross_exposure, config.portfolio.max_leverage)
        assert float(assessment.weights.abs().sum()) <= cap + 1e-9

    def test_position_count_limit(self, manager, config):
        config.portfolio.max_active_positions = 3
        weights = pd.Series(0.05, index=config.symbols)
        assessment = RiskManager(config).assess(
            pd.Timestamp("2020-01-01"), weights, equity=100_000.0
        )
        assert int((assessment.weights.abs() > 1e-9).sum()) <= 3

    def test_predicted_volatility_ceiling(self, manager, config, panel):
        from atlas.portfolio.covariance import CovarianceEstimator, portfolio_volatility

        covariance = CovarianceEstimator(config.risk.covariance).estimate(
            panel.returns("adj_close").tail(300)
        )
        weights = pd.Series(0.20, index=covariance.matrix.columns[:6])
        assessment = manager.assess(
            pd.Timestamp("2020-01-01"),
            weights,
            equity=100_000.0,
            covariance=covariance.matrix,
        )
        predicted = portfolio_volatility(assessment.weights, covariance.matrix)
        assert predicted <= config.portfolio.max_portfolio_volatility + 1e-6


class TestPortfolioControls:
    def test_daily_loss_limit_halts(self, manager, weights):
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        assessment = manager.assess(pd.Timestamp("2020-01-02"), weights, equity=90_000.0)
        assert not assessment.approved
        assert manager.halted
        assert "daily loss" in manager.halt_reason.lower()

    def test_weekly_loss_limit_halts(self, config):
        manager = RiskManager(config)
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        for day in range(1, 6):
            manager.update_equity(pd.Timestamp("2020-01-01") + pd.Timedelta(days=day), 98_000.0 - day * 300)
        assessment = manager.assess(
            pd.Timestamp("2020-01-08"),
            pd.Series(0.1, index=config.symbols[:4]),
            equity=90_000.0,
        )
        assert not assessment.approved

    def test_max_drawdown_blocks_risk_increases(self, config):
        config.risk.limits.max_drawdown = 0.10
        config.risk.limits.daily_loss_limit = 0.90
        config.risk.limits.weekly_loss_limit = 0.90
        manager = RiskManager(config)
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        previous = pd.Series({config.symbols[0]: 0.05})
        target = pd.Series({config.symbols[0]: 0.20})
        assessment = manager.assess(
            pd.Timestamp("2021-01-01"), target, equity=80_000.0, previous_weights=previous
        )
        assert assessment.weights[config.symbols[0]] <= 0.05 + 1e-9
        assert any(e.control == "max_drawdown" for e in assessment.events)

    def test_turnover_cap(self, manager, config):
        previous = pd.Series(0.0, index=config.symbols[:6])
        target = pd.Series(0.15, index=config.symbols[:6])
        assessment = manager.assess(
            pd.Timestamp("2020-01-01"), target, equity=100_000.0, previous_weights=previous
        )
        turnover = float((assessment.weights - previous).abs().sum())
        assert turnover <= config.portfolio.maximum_daily_turnover + 1e-9

    def test_correlation_warning(self, manager, config):
        symbols = config.symbols[:3]
        matrix = pd.DataFrame(
            [[0.04, 0.039, 0.038], [0.039, 0.04, 0.039], [0.038, 0.039, 0.04]],
            index=symbols, columns=symbols,
        )
        assessment = manager.assess(
            pd.Timestamp("2020-01-01"),
            pd.Series(0.1, index=symbols),
            equity=100_000.0,
            covariance=matrix,
        )
        assert any(e.control == "correlation_concentration" for e in assessment.events)


class TestOperationalControls:
    def test_stale_data_blocks(self, manager, weights):
        assessment = manager.assess(
            pd.Timestamp("2020-01-20"),
            weights,
            equity=100_000.0,
            last_data_date=pd.Timestamp("2020-01-01"),
        )
        assert not assessment.approved
        assert any(e.control == "stale_data" for e in assessment.events)

    def test_missing_price_drops_the_position(self, manager, config):
        symbols = config.symbols[:3]
        prices = pd.Series({symbols[0]: 100.0, symbols[1]: np.nan, symbols[2]: 50.0})
        assessment = manager.assess(
            pd.Timestamp("2020-01-01"),
            pd.Series(0.1, index=symbols),
            equity=100_000.0,
            prices=prices,
            last_data_date=pd.Timestamp("2020-01-01"),
        )
        assert assessment.weights[symbols[1]] == 0.0
        assert any(e.control == "missing_prices" for e in assessment.events)

    def test_order_notional_bounds(self, manager, config):
        allowed, reason = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 0.1, 100.0)
        assert not allowed and "minimum" in reason

        allowed, reason = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 100_000.0, 100.0)
        assert not allowed and "maximum" in reason

        allowed, _ = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0)
        assert allowed

    def test_buying_power_check(self, manager):
        allowed, reason = manager.check_order(
            pd.Timestamp("2020-01-01"), "SPY", 100.0, 100.0, buying_power=500.0
        )
        assert not allowed and "buying power" in reason

    def test_invalid_price_rejected(self, manager):
        for price in (0.0, -10.0, float("nan")):
            allowed, reason = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 50.0, price)
            assert not allowed and "invalid price" in reason

    def test_duplicate_order_rejected(self, manager):
        now = datetime(2020, 1, 1, 15, 0, tzinfo=UTC)
        allowed, _ = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0, now=now)
        assert allowed
        allowed, reason = manager.check_order(
            pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0, now=now
        )
        assert not allowed and "duplicate" in reason

    def test_duplicate_window_expires(self, manager, config):
        window = config.execution.order_defaults.duplicate_window_seconds
        first = datetime(2020, 1, 1, 15, 0, tzinfo=UTC)
        later = datetime(2020, 1, 1, 15, 0, tzinfo=UTC) + pd.Timedelta(
            seconds=window + 10
        ).to_pytimedelta()
        manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0, now=first)
        allowed, _ = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0, now=later)
        assert allowed

    def test_trading_window_enforced(self, manager):
        outside = datetime(2020, 1, 1, 3, 0, tzinfo=UTC)  # 22:00 previous day in NY
        allowed, reason = manager.check_order(
            pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0,
            now=outside, enforce_trading_window=True,
        )
        assert not allowed and "trading window" in reason

    def test_weekend_rejected(self, manager):
        saturday = datetime(2020, 1, 4, 16, 0, tzinfo=UTC)
        allowed, _ = manager.check_order(
            pd.Timestamp("2020-01-04"), "SPY", 50.0, 100.0,
            now=saturday, enforce_trading_window=True,
        )
        assert not allowed

    def test_api_failures_halt(self, manager, config):
        for _ in range(config.risk.limits.max_consecutive_api_failures - 1):
            assert not manager.record_api_failure("read")
        assert manager.record_api_failure("read")
        assert manager.halted

    def test_api_success_resets_counter(self, manager):
        manager.record_api_failure("read")
        manager.record_api_success()
        assert manager._api_failures == 0

    def test_reconciliation_failure_halts(self, manager):
        manager.record_reconciliation_failure("2 mismatches")
        assert manager.halted
        with pytest.raises(RiskLimitBreached):
            manager.raise_if_halted()


class TestKillSwitch:
    def test_blocks_assessment(self, config, weights):
        config.risk.kill_switch.enabled = True
        config.risk.kill_switch.reason = "maintenance"
        manager = RiskManager(config)
        assessment = manager.assess(pd.Timestamp("2020-01-01"), weights, equity=100_000.0)
        assert not assessment.approved
        assert float(assessment.weights.abs().sum()) == 0.0

    def test_blocks_orders(self, config):
        config.risk.kill_switch.enabled = True
        manager = RiskManager(config)
        allowed, reason = manager.check_order(pd.Timestamp("2020-01-01"), "SPY", 50.0, 100.0)
        assert not allowed and "kill switch" in reason

    def test_raises_on_check(self, config):
        config.risk.kill_switch.enabled = True
        with pytest.raises(KillSwitchActive):
            RiskManager(config).check_kill_switch()

    def test_cannot_resume_while_engaged(self, config):
        config.risk.kill_switch.enabled = True
        with pytest.raises(KillSwitchActive, match="cannot resume"):
            RiskManager(config).resume()

    def test_disabled_by_default(self, base_config):
        assert not base_config.risk.kill_switch.enabled


class TestSafetyGate:
    def test_live_mode_rejected_at_config(self, base_config):
        from atlas.config import ExecutionConfig

        with pytest.raises(ValueError, match="not implemented"):
            ExecutionConfig(mode="live")

    def test_live_mode_rejected_at_gate(self, config):
        gate = SafetyGate(config)
        # Bypass the config validator to prove the gate is an independent barrier.
        object.__setattr__(gate, "execution", config.execution.model_copy(deep=True))
        gate.execution.__dict__["mode"] = ExecutionMode.LIVE
        with pytest.raises(ExecutionBlocked, match="not implemented"):
            gate.assert_not_live()

    def test_dry_run_never_submits(self, config):
        config.execution.mode = ExecutionMode.DRY_RUN
        gate = SafetyGate(config)
        assert gate.is_dry_run
        assert not gate.may_submit_orders

    def test_paper_mode_may_submit(self, config):
        config.execution.mode = ExecutionMode.PAPER
        assert SafetyGate(config).may_submit_orders

    def test_non_paper_account_rejected(self, config):
        config.execution.broker.require_paper_account = True
        checks = SafetyGate(config).check_account("U1234567")
        paper = next(c for c in checks if c.name == "paper_account")
        assert not paper.passed
        assert "paper account" in paper.message

    def test_paper_account_accepted(self, config):
        checks = SafetyGate(config).check_account("DU1234567")
        assert next(c for c in checks if c.name == "paper_account").passed

    def test_allowlist_enforced(self, config):
        config.execution.broker.account_allowlist = ["DU9999999"]
        checks = SafetyGate(config).check_account("DU1234567")
        allowlist = next(c for c in checks if c.name == "account_allowlist")
        assert not allowlist.passed

    def test_allowlist_permits_listed_account(self, config):
        config.execution.broker.account_allowlist = ["DU1234567"]
        checks = SafetyGate(config).check_account("DU1234567")
        assert next(c for c in checks if c.name == "account_allowlist").passed

    def test_missing_allowlist_blocks_paper_submission(self, config):
        config.execution.mode = ExecutionMode.PAPER
        config.execution.broker.account_allowlist = []
        report = SafetyGate(config).run(account="DU1234567", connected=True)
        assert not report.passed

    def test_kill_switch_fails_the_gate(self, config):
        config.risk.kill_switch.enabled = True
        config.risk.kill_switch.reason = "testing"
        report = SafetyGate(config).run(connected=True, account="DU1234567")
        assert not report.passed
        assert any(c.name == "kill_switch" and not c.passed for c in report.checks)

    def test_guard_raises_when_blocked(self, config):
        config.risk.kill_switch.enabled = True
        with pytest.raises(KillSwitchActive):
            SafetyGate(config).guard(connected=True, account="DU1234567")

    def test_report_renders(self, config):
        report = SafetyGate(config).run(connected=True, account="DU1234567")
        text = report.render()
        assert "Atlas pre-trade safety report" in text
        assert "overall:" in text


class TestRiskEventRecords:
    def test_events_are_recorded(self, manager, config):
        manager.assess(
            pd.Timestamp("2020-01-01"),
            pd.Series({config.symbols[0]: 0.90}),
            equity=100_000.0,
        )
        frame = manager.events_frame()
        assert not frame.empty
        for column in ("timestamp", "control", "severity", "action", "message"):
            assert column in frame.columns

    def test_status_summary(self, manager):
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        status = manager.status()
        assert status["halted"] is False
        assert status["equity"] == 100_000.0

    def test_reset_clears_state(self, manager, weights):
        manager.update_equity(pd.Timestamp("2020-01-01"), 100_000.0)
        manager.assess(pd.Timestamp("2020-01-02"), weights, equity=80_000.0)
        manager.reset()
        assert not manager.halted
        assert manager.events == []
        assert manager.current_drawdown() == 0.0

    def test_severity_and_action_enums(self):
        assert RiskSeverity.CRITICAL.value == "critical"
        assert RiskAction.HALTED.value == "halted"


class TestHaltCooldown:
    """Loss-limit halts are circuit breakers, not permanent shutdowns."""

    def _breach_daily_limit(self, manager: RiskManager, weights: pd.Series) -> pd.Timestamp:
        start = pd.Timestamp("2020-01-01")
        manager.update_equity(start, 100_000.0)
        manager.assess(start + pd.Timedelta(days=1), weights, equity=90_000.0)
        assert manager.halted
        return start + pd.Timedelta(days=1)

    def test_loss_limit_halt_lifts_after_cooldown(self, config, weights):
        config.risk.limits.loss_limit_cooldown_days = 5
        manager = RiskManager(config)
        breached_at = self._breach_daily_limit(manager, weights)

        # Still halted the next day.
        assessment = manager.assess(breached_at + pd.Timedelta(days=1), weights, equity=90_000.0)
        assert not assessment.approved

        # Lifted once the cool-off has elapsed.
        later = breached_at + pd.tseries.offsets.BDay(6)
        manager.assess(later, weights, equity=90_000.0)
        assert not manager.halted

    def test_zero_cooldown_lifts_immediately(self, config, weights):
        config.risk.limits.loss_limit_cooldown_days = 0
        manager = RiskManager(config)
        breached_at = self._breach_daily_limit(manager, weights)
        # Equity recovers, so no limit is still breached when the halt lifts.
        assessment = manager.assess(
            breached_at + pd.Timedelta(days=1), weights, equity=100_500.0
        )
        assert not manager.halted
        assert assessment.approved

    def test_halt_persists_while_the_loss_condition_holds(self, config, weights):
        """Lifting the timer does not excuse a limit that is still breached."""
        config.risk.limits.loss_limit_cooldown_days = 0
        manager = RiskManager(config)
        breached_at = self._breach_daily_limit(manager, weights)
        # Equity has not recovered, so the weekly limit re-halts immediately.
        assessment = manager.assess(
            breached_at + pd.Timedelta(days=1), weights, equity=90_000.0
        )
        assert not assessment.approved
        assert manager.halted
        assert "weekly" in manager.halt_reason.lower()

    def test_reconciliation_halt_never_auto_resumes(self, config, weights):
        """A halt needing human judgement must latch until someone resumes it."""
        manager = RiskManager(config)
        manager.record_reconciliation_failure("3 mismatches")
        assert manager.halted
        far_future = pd.Timestamp("2030-01-01")
        assessment = manager.assess(far_future, weights, equity=100_000.0)
        assert not assessment.approved
        assert manager.halted

    def test_api_failure_halt_never_auto_resumes(self, config, weights):
        manager = RiskManager(config)
        for _ in range(config.risk.limits.max_consecutive_api_failures):
            manager.record_api_failure("read")
        assert manager.halted
        manager.assess(pd.Timestamp("2030-01-01"), weights, equity=100_000.0)
        assert manager.halted

    def test_manual_resume_clears_the_timer(self, config, weights):
        manager = RiskManager(config)
        self._breach_daily_limit(manager, weights)
        manager.resume("operator cleared the breach")
        assert not manager.halted
        assert manager.status()["halt_until"] is None


class TestDrawdownReferencePeak:
    """The drawdown ladder must not become a one-way ratchet."""

    def test_rolling_peak_decays(self, config):
        """An old high stops counting once it leaves the lookback window."""
        config.risk.drawdown_controls.peak_lookback_days = 250
        manager = RiskManager(config)
        start = pd.Timestamp("2020-01-01")

        # A high, then a long flat stretch below it.
        manager.update_equity(start, 200_000.0)
        for i in range(1, 400):
            manager.update_equity(start + pd.tseries.offsets.BDay(i), 100_000.0)

        # Against the all-time peak this is a 50% drawdown forever.
        assert manager.all_time_drawdown() == pytest.approx(0.50)
        # Against the rolling peak the old high has aged out, so risk recovers.
        assert manager.current_drawdown() == pytest.approx(0.0)
        assert manager.drawdown_multiplier()[0] == 1.0

    def test_all_time_peak_when_lookback_disabled(self, config):
        config.risk.drawdown_controls.peak_lookback_days = 0
        manager = RiskManager(config)
        start = pd.Timestamp("2020-01-01")
        manager.update_equity(start, 200_000.0)
        for i in range(1, 400):
            manager.update_equity(start + pd.tseries.offsets.BDay(i), 100_000.0)
        assert manager.current_drawdown() == pytest.approx(0.50)

    def test_recent_drawdown_still_reduces_risk(self, config):
        """The rolling peak must not stop the control working on a fresh loss."""
        config.risk.drawdown_controls.peak_lookback_days = 504
        manager = RiskManager(config)
        start = pd.Timestamp("2020-01-01")
        manager.update_equity(start, 100_000.0)
        manager.update_equity(start + pd.tseries.offsets.BDay(1), 88_000.0)
        assert manager.current_drawdown() == pytest.approx(0.12)
        assert manager.drawdown_multiplier()[0] == 0.50

    def test_history_is_long_enough_for_the_lookback(self, config):
        config.risk.drawdown_controls.peak_lookback_days = 504
        manager = RiskManager(config)
        assert manager._equity_history.maxlen >= 504
