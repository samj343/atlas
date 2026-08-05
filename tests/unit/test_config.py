"""Tests for configuration loading, validation and reproducibility."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from atlas.config import (
    AtlasConfig,
    CrossSectionalMomentumConfig,
    ExecutionConfig,
    ExecutionMode,
    PortfolioConfig,
    RegimeConfig,
    RegimeLabel,
    UniverseConfig,
    WalkForwardConfig,
    find_project_root,
    load_config,
    load_yaml,
)


class TestLoading:
    def test_shipped_config_is_valid(self, base_config):
        assert isinstance(base_config, AtlasConfig)
        assert len(base_config.symbols) == 14
        assert base_config.universe.benchmark == "SPY"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path)

    def test_unknown_key_is_rejected(self, tmp_path, project_root):
        """A typo in YAML must fail loudly rather than being silently ignored."""
        for name in ("assets", "strategies", "regimes", "risk", "execution", "validation"):
            source = project_root / "configs" / f"{name}.yaml"
            (tmp_path / f"{name}.yaml").write_text(source.read_text())

        risk = yaml.safe_load((tmp_path / "risk.yaml").read_text())
        risk["portfolio"]["target_volatilty"] = 0.10  # deliberate typo
        (tmp_path / "risk.yaml").write_text(yaml.safe_dump(risk))

        with pytest.raises(ValidationError, match="target_volatilty"):
            load_config(tmp_path)

    def test_overrides_are_applied(self, project_root):
        config = load_config(
            project_root / "configs",
            overrides={"risk": {"portfolio": {"target_volatility": 0.06}}},
        )
        assert config.portfolio.target_volatility == 0.06

    def test_load_yaml_rejects_non_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml(path)

    def test_find_project_root(self, project_root):
        assert find_project_root() == project_root
        assert (find_project_root() / "configs").is_dir()


class TestUniverse:
    def test_duplicate_symbols_rejected(self):
        with pytest.raises(ValidationError, match="duplicate symbols"):
            UniverseConfig(
                benchmark="SPY",
                defensive_asset=None,
                assets=[{"symbol": "SPY"}, {"symbol": "SPY"}],
            )

    def test_benchmark_must_be_in_universe(self):
        with pytest.raises(ValidationError, match="benchmark"):
            UniverseConfig(benchmark="QQQ", defensive_asset=None, assets=[{"symbol": "SPY"}])

    def test_defensive_asset_must_be_in_universe(self):
        with pytest.raises(ValidationError, match="defensive_asset"):
            UniverseConfig(benchmark="SPY", defensive_asset="BIL", assets=[{"symbol": "SPY"}])

    def test_empty_universe_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            UniverseConfig(benchmark="SPY", assets=[])

    def test_symbols_are_upper_cased(self):
        universe = UniverseConfig(
            benchmark="SPY", defensive_asset=None, assets=[{"symbol": " spy "}]
        )
        assert universe.symbols == ["SPY"]

    def test_asset_class_grouping(self, base_config):
        classes = base_config.universe.asset_classes
        assert "us_equity" in classes
        assert "SPY" in classes["us_equity"]
        assert base_config.asset_class_map["TLT"] == "fixed_income"


class TestPortfolioConfig:
    def test_target_cannot_exceed_ceiling(self):
        with pytest.raises(ValidationError, match="exceeds max_portfolio_volatility"):
            PortfolioConfig(target_volatility=0.20, max_portfolio_volatility=0.15)

    def test_net_cannot_exceed_gross(self):
        with pytest.raises(ValidationError, match="cannot exceed max_gross_exposure"):
            PortfolioConfig(max_gross_exposure=1.0, max_net_exposure=1.5)

    def test_bounds_are_enforced(self):
        with pytest.raises(ValidationError):
            PortfolioConfig(target_volatility=-0.1)
        with pytest.raises(ValidationError):
            PortfolioConfig(max_asset_weight=1.5)

    def test_investable_weight_accounts_for_cash(self):
        config = PortfolioConfig(max_gross_exposure=1.0, max_leverage=1.0, minimum_cash_weight=0.05)
        assert config.max_investable_weight == pytest.approx(0.95)


class TestExecutionConfig:
    def test_live_mode_rejected(self):
        with pytest.raises(ValidationError, match="not implemented"):
            ExecutionConfig(mode="live")

    def test_supported_modes_accepted(self):
        for mode in ("backtest", "dry_run", "paper"):
            assert ExecutionConfig(mode=mode).mode.value == mode

    def test_default_is_dry_run(self):
        assert ExecutionConfig().mode is ExecutionMode.DRY_RUN

    def test_order_retries_are_forbidden(self):
        from atlas.config import BrokerConfig

        with pytest.raises(ValidationError):
            BrokerConfig(order_retry_attempts=1)

    def test_trading_window_format(self):
        from atlas.config import TradingWindowConfig

        with pytest.raises(ValidationError, match="HH:MM"):
            TradingWindowConfig(start="9am")
        with pytest.raises(ValidationError, match="out of range"):
            TradingWindowConfig(start="25:00")

    def test_environment_overrides(self, project_root, monkeypatch):
        monkeypatch.setenv("EXECUTION_MODE", "backtest")
        monkeypatch.setenv("ATLAS_IBKR_ACCOUNT_ALLOWLIST", "DU111,DU222")
        config = load_config(project_root / "configs")
        assert config.execution.mode is ExecutionMode.BACKTEST
        assert config.execution.broker.account_allowlist == ["DU111", "DU222"]

    def test_kill_switch_environment_override(self, project_root, monkeypatch):
        monkeypatch.setenv("ATLAS_KILL_SWITCH", "true")
        config = load_config(project_root / "configs")
        assert config.risk.kill_switch.enabled

    def test_no_credentials_in_config(self, base_config):
        """Configuration must never carry secrets."""
        blob = str(base_config.snapshot()).lower()
        for token in ("password", "api_key", "secret", "token"):
            assert token not in blob


class TestStrategyConfig:
    def test_skip_must_be_shorter_than_lookback(self):
        with pytest.raises(ValidationError, match="skip_recent_days"):
            CrossSectionalMomentumConfig(lookback=60, skip_recent_days=120)

    def test_bottom_k_ignored_when_long_only(self):
        config = CrossSectionalMomentumConfig(long_only=True, bottom_k=3)
        assert config.bottom_k == 0

    def test_base_weights_normalise(self, base_config):
        weights = base_config.strategies.strategies.base_weights()
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_max_lookback(self, base_config):
        assert base_config.strategies.max_lookback >= 252

    def test_component_weights_normalise(self):
        from atlas.config import TrendComponentWeights

        weights = TrendComponentWeights(ma_fast=2, ma_slow=2, momentum_short=2, momentum_long=2)
        assert sum(weights.normalized().values()) == pytest.approx(1.0)


class TestRegimeConfig:
    def test_all_regimes_required(self):
        with pytest.raises(ValidationError, match="missing regimes"):
            RegimeConfig(regime_weights={"bull_low_vol": {"trend": 1.0}})

    def test_unknown_regime_rejected(self, base_config):
        blocks = {k: v.model_dump() for k, v in base_config.regimes.regime_weights.items()}
        blocks["sideways"] = {"trend": 1.0}
        with pytest.raises(ValidationError, match="unknown regime labels"):
            RegimeConfig(regime_weights=blocks)

    def test_weights_for_unknown_regime_are_conservative(self, base_config):
        block = base_config.regimes.weights_for(RegimeLabel.UNKNOWN)
        multipliers = [b.risk_multiplier for b in base_config.regimes.regime_weights.values()]
        assert block.risk_multiplier == min(multipliers)

    def test_strategy_weights_normalise(self, base_config):
        for block in base_config.regimes.regime_weights.values():
            assert sum(block.strategy_weights().values()) == pytest.approx(1.0)


class TestValidationConfig:
    def test_large_grids_rejected(self):
        grid = {f"p{i}": [1, 2, 3, 4] for i in range(6)}  # 4096 combinations
        with pytest.raises(ValidationError, match="brute-force"):
            WalkForwardConfig(parameter_grid=grid)

    def test_empty_grid_values_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            WalkForwardConfig(parameter_grid={"trend.slow_ma": []})

    def test_shipped_grid_is_small(self, base_config):
        grid = base_config.validation.walk_forward.parameter_grid
        size = 1
        for values in grid.values():
            size *= len(values)
        assert size <= 24, "the default grid should stay small to avoid parameter mining"

    def test_sixty_forty_weights_must_sum_to_one(self):
        from atlas.config import BenchmarkConfig

        with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
            BenchmarkConfig(sixty_forty={"SPY": 0.6, "IEF": 0.5})

    def test_stress_periods_ordered(self):
        from atlas.config import StressPeriod

        with pytest.raises(ValidationError, match="end must be after start"):
            StressPeriod(name="bad", start="2020-01-01", end="2019-01-01")

    def test_percentiles_bounded(self):
        from atlas.config import MonteCarloConfig

        with pytest.raises(ValidationError, match="strictly between"):
            MonteCarloConfig(percentiles=[0.0, 1.5])


class TestReproducibility:
    def test_hash_is_stable(self, base_config):
        assert base_config.config_hash == base_config.config_hash
        assert len(base_config.config_hash) == 16

    def test_hash_changes_with_configuration(self, base_config):
        before = base_config.config_hash
        modified = base_config.model_copy(deep=True)
        modified.risk.portfolio.target_volatility = 0.07
        assert modified.config_hash != before

    def test_snapshot_is_json_serialisable(self, base_config):
        import json

        blob = json.dumps(base_config.snapshot(), default=str)
        assert len(blob) > 1000

    def test_snapshot_excludes_local_paths(self, base_config):
        snapshot = base_config.snapshot()
        assert "project_root" not in snapshot

    def test_resolve_path(self, base_config):
        resolved = base_config.resolve_path("data/raw")
        assert resolved.is_absolute()
        absolute = Path("/tmp/somewhere")
        assert base_config.resolve_path(absolute) == absolute
