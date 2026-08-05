"""Typed, validated configuration for Atlas.

Every YAML file under ``configs/`` maps onto a pydantic model in this module.
Configuration is loaded once, validated eagerly, and then passed explicitly to
the components that need it - Atlas keeps no global mutable configuration state.

The public entry point is :func:`load_config`, which returns an
:class:`AtlasConfig` aggregating the asset, strategy, risk, execution and
validation configuration.

Example
-------
>>> from atlas.config import load_config
>>> cfg = load_config()                      # doctest: +SKIP
>>> cfg.portfolio.target_volatility          # doctest: +SKIP
0.1
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AssetConfig",
    "AssetsConfig",
    "AtlasConfig",
    "BacktestConfig",
    "CostConfig",
    "CovarianceConfig",
    "CrossSectionalMomentumConfig",
    "DataConfig",
    "DrawdownControlConfig",
    "EnsembleConfig",
    "ExecutionConfig",
    "ExecutionMode",
    "LimitsConfig",
    "MeanReversionConfig",
    "MonteCarloConfig",
    "PortfolioConfig",
    "RegimeConfig",
    "RegimeLabel",
    "RiskConfig",
    "StrategiesConfig",
    "TrendConfig",
    "ValidationConfig",
    "VolatilityConfig",
    "WalkForwardConfig",
    "find_project_root",
    "load_config",
    "load_yaml",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    """Supported execution modes.

    ``LIVE`` is intentionally present but disabled everywhere in the codebase;
    Atlas is documented and built as a paper-trading system.
    """

    BACKTEST = "backtest"
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"  # disabled placeholder - see execution.safety


class RegimeLabel(str, Enum):
    """Discrete market regimes produced by the regime detectors."""

    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    UNKNOWN = "unknown"


class RebalanceFrequency(str, Enum):
    """Supported rebalance schedules."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ExecutionTiming(str, Enum):
    """When an order generated from a signal is filled.

    ``SAME_CLOSE_UNREALISTIC`` uses the same close that produced the signal and
    exists only as a clearly labelled research comparison. It is never a default.
    """

    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"
    SAME_CLOSE_UNREALISTIC = "same_close_unrealistic"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    """Base model: strict about unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# ---------------------------------------------------------------------------
# Assets and data
# ---------------------------------------------------------------------------


class AssetConfig(_Base):
    """A single tradable instrument."""

    symbol: str
    name: str = ""
    asset_class: str = "unclassified"
    tradable: bool = True

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol must not be empty")
        return v


class DataConfig(_Base):
    """Historical data provider and caching settings."""

    provider: Literal["yfinance", "csv", "synthetic"] = "yfinance"
    start_date: date = date(2005, 1, 1)
    end_date: date | None = None
    cache_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    min_history_days: int = Field(260, ge=0)
    max_staleness_days: int = Field(5, ge=0)


class UniverseConfig(_Base):
    """The tradable universe and its metadata."""

    name: str = "default"
    description: str = ""
    base_currency: str = "USD"
    trading_calendar: str = "NYSE"
    benchmark: str = "SPY"
    defensive_asset: str | None = "BIL"
    assets: list[AssetConfig]

    @field_validator("assets")
    @classmethod
    def _non_empty_unique(cls, v: list[AssetConfig]) -> list[AssetConfig]:
        if not v:
            raise ValueError("universe.assets must not be empty")
        symbols = [a.symbol for a in v]
        dupes = {s for s in symbols if symbols.count(s) > 1}
        if dupes:
            raise ValueError(f"duplicate symbols in universe: {sorted(dupes)}")
        return v

    @model_validator(mode="after")
    def _benchmark_in_universe(self) -> UniverseConfig:
        symbols = {a.symbol for a in self.assets}
        if self.benchmark.upper() not in symbols:
            raise ValueError(
                f"benchmark {self.benchmark!r} must be part of the universe {sorted(symbols)}"
            )
        if self.defensive_asset and self.defensive_asset.upper() not in symbols:
            raise ValueError(
                f"defensive_asset {self.defensive_asset!r} must be part of the universe"
            )
        return self

    @property
    def symbols(self) -> list[str]:
        """All symbols in the universe, in configuration order."""
        return [a.symbol for a in self.assets]

    @property
    def tradable_symbols(self) -> list[str]:
        """Symbols flagged as tradable."""
        return [a.symbol for a in self.assets if a.tradable]

    @property
    def asset_class_map(self) -> dict[str, str]:
        """Mapping of symbol -> asset class."""
        return {a.symbol: a.asset_class for a in self.assets}

    @property
    def asset_classes(self) -> dict[str, list[str]]:
        """Mapping of asset class -> list of symbols."""
        out: dict[str, list[str]] = {}
        for a in self.assets:
            out.setdefault(a.asset_class, []).append(a.symbol)
        return out


class AssetsConfig(_Base):
    """Top-level model for ``configs/assets.yaml``."""

    universe: UniverseConfig
    data: DataConfig = Field(default_factory=DataConfig)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class TrendComponentWeights(_Base):
    """Relative weights of the four trend sub-signals."""

    ma_fast: float = 0.25
    ma_slow: float = 0.25
    momentum_short: float = 0.25
    momentum_long: float = 0.25

    def normalized(self) -> dict[str, float]:
        """Return the component weights scaled to sum to one."""
        raw = self.model_dump()
        total = sum(abs(v) for v in raw.values())
        if total <= 0:
            n = len(raw)
            return dict.fromkeys(raw, 1.0 / n)
        return {k: v / total for k, v in raw.items()}


class TrendConfig(_Base):
    """Time-series trend-following parameters."""

    enabled: bool = True
    weight: float = Field(0.40, ge=0.0)
    fast_ma: int = Field(100, ge=2)
    slow_ma: int = Field(200, ge=3)
    momentum_lookback: int = Field(252, ge=5)
    short_momentum_lookback: int = Field(63, ge=2)
    neutral_threshold: float = Field(0.05, ge=0.0, lt=1.0)
    long_only: bool = True
    volatility_scaling: bool = True
    volatility_lookback: int = Field(63, ge=5)
    volatility_percentile_cap: float = Field(0.90, gt=0.0, le=1.0)
    component_weights: TrendComponentWeights = Field(default_factory=TrendComponentWeights)

    @model_validator(mode="after")
    def _ma_ordering(self) -> TrendConfig:
        if self.fast_ma >= self.slow_ma:
            raise ValueError("trend.fast_ma must be strictly less than trend.slow_ma")
        return self

    @property
    def max_lookback(self) -> int:
        """Longest lookback used, for warm-up computation."""
        return max(
            self.fast_ma,
            self.slow_ma,
            self.momentum_lookback,
            self.short_momentum_lookback,
            self.volatility_lookback,
        )


class MeanReversionConfig(_Base):
    """Short-term mean-reversion parameters."""

    enabled: bool = True
    weight: float = Field(0.25, ge=0.0)
    zscore_lookback: int = Field(20, ge=3)
    return_lookback: int = Field(5, ge=1)
    entry_threshold: float = Field(1.5, gt=0.0)
    exit_threshold: float = Field(0.25, ge=0.0)
    rsi_lookback: int = Field(14, ge=2)
    rsi_weight: float = Field(0.35, ge=0.0, le=1.0)
    use_trend_filter: bool = True
    trend_filter_ma: int = Field(200, ge=5)
    use_volatility_filter: bool = True
    volatility_lookback: int = Field(20, ge=5)
    volatility_percentile_cap: float = Field(0.95, gt=0.0, le=1.0)
    max_holding_days: int = Field(10, ge=1)
    long_only: bool = True

    @model_validator(mode="after")
    def _thresholds(self) -> MeanReversionConfig:
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError(
                "mean_reversion.exit_threshold must be below entry_threshold "
                "(exit happens as the z-score reverts toward zero)"
            )
        return self

    @property
    def max_lookback(self) -> int:
        """Longest lookback used, for warm-up computation."""
        return max(
            self.zscore_lookback,
            self.return_lookback,
            self.rsi_lookback,
            self.trend_filter_ma if self.use_trend_filter else 0,
            self.volatility_lookback,
        )


class CrossSectionalMomentumConfig(_Base):
    """Cross-sectional (relative strength) momentum parameters."""

    enabled: bool = True
    weight: float = Field(0.35, ge=0.0)
    lookback: int = Field(252, ge=10)
    skip_recent_days: int = Field(21, ge=0)
    volatility_lookback: int = Field(63, ge=5)
    top_k: int = Field(4, ge=1)
    bottom_k: int = Field(0, ge=0)
    long_only: bool = True
    neutralize_asset_class: bool = True
    min_assets: int = Field(4, ge=2)
    scoring: Literal["rank", "topk"] = "rank"

    @model_validator(mode="after")
    def _skip_lt_lookback(self) -> CrossSectionalMomentumConfig:
        if self.skip_recent_days >= self.lookback:
            raise ValueError(
                "cross_sectional_momentum.skip_recent_days must be smaller than lookback"
            )
        if self.long_only and self.bottom_k > 0:
            # Not an error: bottom_k is simply ignored. Normalise it for clarity.
            self.bottom_k = 0
        return self

    @property
    def max_lookback(self) -> int:
        """Longest lookback used, for warm-up computation."""
        return max(self.lookback, self.volatility_lookback)


class EnsembleConfig(_Base):
    """Signal-combination settings."""

    mode: Literal["regime", "fixed", "equal"] = "regime"
    use_confidence: bool = True
    signal_smoothing_alpha: float = Field(0.5, gt=0.0, le=1.0)
    min_signal: float = Field(0.02, ge=0.0)
    clip: float = Field(1.0, gt=0.0)


class RegimeSettingsRef(_Base):
    """Pointer to the regime configuration file plus an on/off switch."""

    config_file: Path = Path("configs/regimes.yaml")
    enabled: bool = True


class BacktestConfig(_Base):
    """Backtest run settings."""

    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = Field(100_000.0, gt=0)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    rebalance_weekday: int = Field(4, ge=0, le=4)
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN
    fractional_shares: bool = True
    price_field: Literal["adj_close", "close"] = "adj_close"
    benchmark: str = "SPY"
    warmup_days: int = Field(300, ge=0)
    random_seed: int = 7


class StrategiesConfig(_Base):
    """Top-level model for ``configs/strategies.yaml``."""

    strategies: _StrategyBlock
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    regime: RegimeSettingsRef = Field(default_factory=RegimeSettingsRef)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @property
    def enabled_strategies(self) -> list[str]:
        """Names of the enabled strategies."""
        return [
            name
            for name, cfg in self.strategies.as_dict().items()
            if getattr(cfg, "enabled", False)
        ]

    @property
    def max_lookback(self) -> int:
        """Longest lookback across enabled strategies (used for warm-up)."""
        lookbacks = [
            cfg.max_lookback
            for name, cfg in self.strategies.as_dict().items()
            if getattr(cfg, "enabled", False)
        ]
        return max(lookbacks) if lookbacks else 0


class _StrategyBlock(_Base):
    """Container for the individual strategy configurations."""

    trend: TrendConfig = Field(default_factory=TrendConfig)
    mean_reversion: MeanReversionConfig = Field(default_factory=MeanReversionConfig)
    cross_sectional_momentum: CrossSectionalMomentumConfig = Field(
        default_factory=CrossSectionalMomentumConfig
    )

    def as_dict(self) -> dict[str, Any]:
        """Return ``{strategy_name: config}`` for all defined strategies."""
        return {
            "trend": self.trend,
            "mean_reversion": self.mean_reversion,
            "cross_sectional_momentum": self.cross_sectional_momentum,
        }

    def base_weights(self) -> dict[str, float]:
        """Static (non-regime) strategy weights, normalised over enabled strategies."""
        raw = {n: c.weight for n, c in self.as_dict().items() if c.enabled}
        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return dict.fromkeys(raw, 1.0 / n) if n else {}
        return {k: v / total for k, v in raw.items()}


StrategiesConfig.model_rebuild()


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


class RegimeDetectorConfig(_Base):
    """Rules-based regime detector settings."""

    type: Literal["rules_based", "clustering"] = "rules_based"
    benchmark: str = "SPY"
    trend_ma: int = Field(200, ge=20)
    volatility_lookback: int = Field(20, ge=5)
    volatility_percentile_window: int = Field(756, ge=100)
    volatility_percentile_min_obs: int = Field(252, ge=20)
    high_volatility_percentile: float = Field(0.70, gt=0.0, lt=1.0)
    breadth_ma: int = Field(200, ge=20)
    correlation_lookback: int = Field(63, ge=10)
    drawdown_lookback: int = Field(252, ge=20)
    weak_breadth_threshold: float = Field(0.40, ge=0.0, le=1.0)
    min_regime_days: int = Field(3, ge=1)


class ClusteringRegimeConfig(_Base):
    """Optional clustering-based regime model (comparison only)."""

    enabled: bool = False
    n_clusters: int = Field(4, ge=2, le=10)
    features: list[str] = Field(
        default_factory=lambda: [
            "trend_score",
            "volatility",
            "breadth",
            "avg_correlation",
            "drawdown",
        ]
    )
    standardize: bool = True
    random_state: int = 7
    min_train_obs: int = Field(500, ge=50)


class RegimeWeightBlock(_Base):
    """Strategy weights and risk posture for a single regime."""

    trend: float = Field(0.0, ge=0.0)
    mean_reversion: float = Field(0.0, ge=0.0)
    cross_sectional_momentum: float = Field(0.0, ge=0.0)
    risk_multiplier: float = Field(1.0, gt=0.0, le=2.0)
    defensive_weight: float = Field(0.0, ge=0.0, le=1.0)

    def strategy_weights(self) -> dict[str, float]:
        """Strategy weights normalised to sum to one."""
        raw = {
            "trend": self.trend,
            "mean_reversion": self.mean_reversion,
            "cross_sectional_momentum": self.cross_sectional_momentum,
        }
        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return dict.fromkeys(raw, 1.0 / n)
        return {k: v / total for k, v in raw.items()}


class BreadthOverlayConfig(_Base):
    """Extra defensive allocation triggered by deteriorating breadth."""

    enabled: bool = True
    additional_defensive_weight: float = Field(0.10, ge=0.0, le=1.0)


class RegimeConfig(_Base):
    """Top-level model for ``configs/regimes.yaml``."""

    detector: RegimeDetectorConfig = Field(default_factory=RegimeDetectorConfig)
    clustering: ClusteringRegimeConfig = Field(default_factory=ClusteringRegimeConfig)
    regime_weights: dict[str, RegimeWeightBlock]
    breadth_overlay: BreadthOverlayConfig = Field(default_factory=BreadthOverlayConfig)

    @field_validator("regime_weights")
    @classmethod
    def _known_regimes(cls, v: dict[str, RegimeWeightBlock]) -> dict[str, RegimeWeightBlock]:
        valid = {r.value for r in RegimeLabel}
        unknown = set(v) - valid
        if unknown:
            raise ValueError(f"unknown regime labels in regime_weights: {sorted(unknown)}")
        required = {
            RegimeLabel.BULL_LOW_VOL.value,
            RegimeLabel.BULL_HIGH_VOL.value,
            RegimeLabel.BEAR_LOW_VOL.value,
            RegimeLabel.BEAR_HIGH_VOL.value,
        }
        missing = required - set(v)
        if missing:
            raise ValueError(f"regime_weights is missing regimes: {sorted(missing)}")
        return v

    def weights_for(self, regime: RegimeLabel | str) -> RegimeWeightBlock:
        """Return the weight block for ``regime``, falling back to a neutral block."""
        key = regime.value if isinstance(regime, RegimeLabel) else str(regime)
        if key in self.regime_weights:
            return self.regime_weights[key]
        # UNKNOWN or an unconfigured regime: average the configured blocks and
        # take the most conservative risk multiplier.
        blocks = list(self.regime_weights.values())
        n = len(blocks)
        return RegimeWeightBlock(
            trend=sum(b.trend for b in blocks) / n,
            mean_reversion=sum(b.mean_reversion for b in blocks) / n,
            cross_sectional_momentum=sum(b.cross_sectional_momentum for b in blocks) / n,
            risk_multiplier=min(b.risk_multiplier for b in blocks),
            defensive_weight=max(b.defensive_weight for b in blocks),
        )


# ---------------------------------------------------------------------------
# Portfolio and risk
# ---------------------------------------------------------------------------


class PortfolioConfig(_Base):
    """Portfolio construction limits and the volatility target."""

    initial_capital: float = Field(100_000.0, gt=0)
    target_volatility: float = Field(0.10, gt=0.0, le=1.0)
    max_portfolio_volatility: float = Field(0.15, gt=0.0, le=2.0)
    max_asset_weight: float = Field(0.20, gt=0.0, le=1.0)
    max_asset_class_weight: float = Field(0.50, gt=0.0, le=1.0)
    max_gross_exposure: float = Field(1.25, gt=0.0, le=5.0)
    max_net_exposure: float = Field(1.00, gt=0.0, le=5.0)
    max_leverage: float = Field(1.25, gt=0.0, le=5.0)
    minimum_cash_weight: float = Field(0.05, ge=0.0, lt=1.0)
    maximum_daily_turnover: float = Field(0.30, gt=0.0, le=5.0)
    min_trade_weight: float = Field(0.005, ge=0.0, lt=1.0)
    max_active_positions: int = Field(12, ge=1)
    long_only: bool = True
    max_vol_target_scaler: float = Field(3.0, gt=0.0)
    min_vol_target_scaler: float = Field(0.1, gt=0.0)

    @model_validator(mode="after")
    def _coherent(self) -> PortfolioConfig:
        if self.target_volatility > self.max_portfolio_volatility:
            raise ValueError("portfolio.target_volatility exceeds max_portfolio_volatility")
        if self.min_vol_target_scaler > self.max_vol_target_scaler:
            raise ValueError("min_vol_target_scaler must not exceed max_vol_target_scaler")
        # Net exposure can never exceed gross exposure by construction.
        if self.max_net_exposure > self.max_gross_exposure:
            raise ValueError("portfolio.max_net_exposure cannot exceed max_gross_exposure")
        return self

    @property
    def max_investable_weight(self) -> float:
        """Gross exposure available after the minimum cash reserve."""
        return min(self.max_gross_exposure, self.max_leverage) * (1.0 - self.minimum_cash_weight)


class CovarianceConfig(_Base):
    """Covariance estimator settings."""

    method: Literal["sample", "shrinkage", "ewma"] = "shrinkage"
    lookback: int = Field(252, ge=20)
    ewma_halflife: int = Field(63, ge=2)
    shrinkage_intensity: float | None = Field(None, ge=0.0, le=1.0)
    min_observations: int = Field(60, ge=5)
    min_annual_volatility: float = Field(0.005, gt=0.0)
    ridge: float = Field(1e-6, ge=0.0)


class VolatilityConfig(_Base):
    """Per-asset volatility estimator settings."""

    method: Literal["rolling", "ewma"] = "ewma"
    lookback: int = Field(63, ge=5)
    halflife: int = Field(21, ge=1)
    min_periods: int = Field(20, ge=2)
    annualization_factor: int = Field(252, ge=1)
    min_annual_volatility: float = Field(0.005, gt=0.0)


class DrawdownThreshold(_Base):
    """A single drawdown bucket and its risk multiplier."""

    drawdown: float = Field(..., ge=0.0, le=1.0)
    multiplier: float = Field(..., ge=0.0, le=1.0)


class DrawdownControlConfig(_Base):
    """Drawdown-based risk reduction ladder."""

    enabled: bool = True
    thresholds: list[DrawdownThreshold] = Field(default_factory=list)
    liquidate_on_breach: bool = False
    # Trading days over which the reference peak is measured. Set to 0 to use
    # the all-time high.
    #
    # This is not a cosmetic choice. Measured against an all-time peak, a
    # portfolio that falls 15% early is de-risked to a quarter of its budget
    # *forever*: with a quarter of the risk it cannot earn its way back to the
    # old peak, so the multiplier never lifts. A rolling peak lets the reference
    # decay, so the control de-risks a drawdown without permanently disabling
    # the strategy.
    peak_lookback_days: int = Field(504, ge=0)

    @field_validator("thresholds")
    @classmethod
    def _sorted_monotonic(cls, v: list[DrawdownThreshold]) -> list[DrawdownThreshold]:
        v = sorted(v, key=lambda t: t.drawdown)
        for a, b in pairwise(v):
            if b.multiplier > a.multiplier:
                raise ValueError(
                    "drawdown_controls.thresholds must have non-increasing multipliers "
                    "as drawdown deepens"
                )
        return v

    @property
    def max_history_days(self) -> int:
        """Equity history the manager must retain to evaluate this ladder."""
        return max(60, self.peak_lookback_days + 10)

    def multiplier_for(self, drawdown: float) -> tuple[float, float | None]:
        """Return ``(multiplier, breached_threshold)`` for a current ``drawdown``.

        ``drawdown`` is expressed as a positive fraction (0.12 == 12% below the
        running peak). Returns a multiplier of 1.0 and ``None`` when no threshold
        is breached or when the control is disabled.
        """
        if not self.enabled or not self.thresholds:
            return 1.0, None
        dd = abs(float(drawdown))
        multiplier = 1.0
        breached: float | None = None
        # A tolerance matters here: a drawdown computed as 1 - 99000/110000 is
        # 0.09999999999999998, and without it a portfolio exactly 10% below its
        # peak would sit in the 5% bucket rather than the 10% one.
        tolerance = 1e-9
        for t in self.thresholds:
            if dd >= t.drawdown - tolerance:
                multiplier = t.multiplier
                breached = t.drawdown
        return multiplier, breached


class LimitsConfig(_Base):
    """Hard risk limits enforced by the risk manager."""

    max_order_notional: float = Field(50_000.0, gt=0)
    min_order_notional: float = Field(100.0, ge=0)
    max_drawdown: float = Field(0.25, gt=0.0, le=1.0)
    daily_loss_limit: float = Field(0.04, gt=0.0, le=1.0)
    weekly_loss_limit: float = Field(0.08, gt=0.0, le=1.0)
    # Trading days a loss-limit halt lasts before it lifts automatically. A
    # loss limit is a circuit breaker, not a permanent shutdown: leaving it
    # latched would freeze the remainder of a multi-year backtest and make the
    # result meaningless. Halts that need a human - unreconciled positions,
    # repeated API failures, the kill switch - never auto-resume.
    loss_limit_cooldown_days: int = Field(5, ge=0)
    correlation_warning_threshold: float = Field(0.80, gt=0.0, le=1.0)
    max_data_staleness_days: int = Field(5, ge=0)
    max_consecutive_api_failures: int = Field(3, ge=1)
    reconciliation_tolerance_shares: float = Field(0.01, ge=0.0)
    reconciliation_tolerance_value: float = Field(50.0, ge=0.0)
    reconciliation_tolerance_pct: float = Field(0.005, ge=0.0)


class KillSwitchConfig(_Base):
    """Global trading kill switch."""

    enabled: bool = False
    reason: str = ""


class RiskConfig(_Base):
    """Top-level model for ``configs/risk.yaml``."""

    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    covariance: CovarianceConfig = Field(default_factory=CovarianceConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    drawdown_controls: DrawdownControlConfig = Field(default_factory=DrawdownControlConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class CostConfig(_Base):
    """Transaction-cost model parameters."""

    commission_per_order: float = Field(0.0, ge=0.0)
    commission_per_share: float = Field(0.005, ge=0.0)
    min_commission: float = Field(1.0, ge=0.0)
    max_commission_pct: float = Field(0.005, ge=0.0, le=1.0)
    half_spread_bps: float = Field(1.0, ge=0.0)
    slippage_bps: float = Field(2.0, ge=0.0)
    market_impact_enabled: bool = True
    market_impact_coefficient: float = Field(0.10, ge=0.0)
    adv_lookback: int = Field(21, ge=1)
    max_participation: float = Field(0.05, gt=0.0, le=1.0)


class FillConfig(_Base):
    """Simulated broker fill behaviour."""

    allow_partial_fills: bool = True
    reject_if_no_price: bool = True
    deterministic: bool = True


class BrokerConfig(_Base):
    """Interactive Brokers connectivity settings."""

    provider: Literal["ibkr", "mock"] = "ibkr"
    host: str = "127.0.0.1"
    port: int = Field(7497, ge=1, le=65535)
    client_id: int = Field(17, ge=0)
    account_allowlist: list[str] = Field(default_factory=list)
    require_paper_account: bool = True
    connect_timeout_seconds: float = Field(15.0, gt=0)
    request_timeout_seconds: float = Field(30.0, gt=0)
    market_data_type: int = Field(3, ge=1, le=4)
    read_retry_attempts: int = Field(3, ge=0)
    read_retry_backoff_seconds: float = Field(2.0, ge=0)
    order_retry_attempts: int = Field(0, ge=0, le=0)  # order submissions are never retried


class TradingWindowConfig(_Base):
    """Permitted order-submission window."""

    timezone: str = "America/New_York"
    start: str = "09:35"
    end: str = "15:55"
    trading_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"time must be 'HH:MM', got {v!r}")
        hh, mm = int(parts[0]), int(parts[1])
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"time out of range: {v!r}")
        return v


class OrderDefaultsConfig(_Base):
    """Default order attributes."""

    order_type: Literal["MKT", "LMT"] = "MKT"
    time_in_force: Literal["DAY", "GTC"] = "DAY"
    limit_offset_bps: float = Field(5.0, ge=0.0)
    duplicate_window_seconds: int = Field(300, ge=0)


class ReconciliationConfig(_Base):
    """Broker/local position reconciliation settings."""

    enabled: bool = True
    block_on_mismatch: bool = True
    tolerance_shares: float = Field(0.01, ge=0.0)
    tolerance_value: float = Field(50.0, ge=0.0)
    tolerance_pct: float = Field(0.005, ge=0.0)
    settle_delay_seconds: float = Field(5.0, ge=0.0)


class ExecutionConfig(_Base):
    """Top-level model for ``configs/execution.yaml``."""

    mode: ExecutionMode = ExecutionMode.DRY_RUN
    costs: CostConfig = Field(default_factory=CostConfig)
    fills: FillConfig = Field(default_factory=FillConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    trading_window: TradingWindowConfig = Field(default_factory=TradingWindowConfig)
    order_defaults: OrderDefaultsConfig = Field(default_factory=OrderDefaultsConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)

    @model_validator(mode="after")
    def _live_disabled(self) -> ExecutionConfig:
        if self.mode == ExecutionMode.LIVE:
            raise ValueError(
                "EXECUTION_MODE=live is not implemented. Atlas is a paper-trading system; "
                "supported modes are backtest, dry_run and paper."
            )
        return self


# ---------------------------------------------------------------------------
# Validation / research
# ---------------------------------------------------------------------------


class WalkForwardObjective(_Base):
    """Multi-objective parameter-selection score."""

    w_sharpe: float = 1.0
    w_drawdown: float = 1.0
    w_turnover: float = 0.10
    w_stability: float = 0.50
    max_acceptable_drawdown: float = Field(0.35, gt=0.0, le=1.0)
    max_acceptable_annual_turnover: float = Field(25.0, gt=0.0)


class WalkForwardConfig(_Base):
    """Walk-forward window layout and parameter grid."""

    train_years: float = Field(5.0, gt=0)
    validation_years: float = Field(1.0, gt=0)
    test_years: float = Field(1.0, gt=0)
    step_years: float = Field(1.0, gt=0)
    embargo_days: int = Field(21, ge=0)
    min_train_observations: int = Field(500, ge=50)
    parameter_grid: dict[str, list[Any]] = Field(default_factory=dict)
    objective: WalkForwardObjective = Field(default_factory=WalkForwardObjective)

    @field_validator("parameter_grid")
    @classmethod
    def _grid_size(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        size = 1
        for values in v.values():
            if not values:
                raise ValueError("parameter_grid entries must be non-empty lists")
            size *= len(values)
        if size > 500:
            raise ValueError(
                f"parameter_grid expands to {size} combinations; Atlas deliberately avoids "
                "large brute-force searches (limit 500)"
            )
        return v


class ParameterStabilityConfig(_Base):
    """One-at-a-time parameter sweep settings."""

    parameters: dict[str, list[Any]] = Field(default_factory=dict)
    fragility_sharpe_gap: float = Field(0.50, ge=0.0)
    cv_threshold: float = Field(0.60, ge=0.0)
    include_gross: bool = True


class StressPeriod(_Base):
    """A named historical stress window."""

    name: str
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> StressPeriod:
        if self.end <= self.start:
            raise ValueError(f"stress period {self.name!r}: end must be after start")
        return self


class StressTestConfig(_Base):
    """Stress-test period list."""

    periods: list[StressPeriod] = Field(default_factory=list)


class MonteCarloConfig(_Base):
    """Block-bootstrap resampling settings."""

    enabled: bool = True
    method: Literal["block_bootstrap", "stationary_bootstrap"] = "block_bootstrap"
    n_simulations: int = Field(1000, ge=10, le=100_000)
    block_size_days: int = Field(21, ge=1)
    horizon_days: int = Field(252, ge=20)
    random_seed: int = 7
    percentiles: list[float] = Field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])

    @field_validator("percentiles")
    @classmethod
    def _in_range(cls, v: list[float]) -> list[float]:
        if any(not (0.0 < p < 1.0) for p in v):
            raise ValueError("percentiles must lie strictly between 0 and 1")
        return sorted(v)


class BenchmarkConfig(_Base):
    """Benchmark definitions used for comparison."""

    buy_and_hold: list[str] = Field(default_factory=lambda: ["SPY"])
    sixty_forty: dict[str, float] = Field(default_factory=lambda: {"SPY": 0.6, "IEF": 0.4})
    equal_weight_universe: bool = True
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    risk_free_rate: float = Field(0.02, ge=-0.05, le=0.25)

    @field_validator("sixty_forty")
    @classmethod
    def _weights_sum(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sixty_forty weights must sum to 1.0, got {total}")
        return v


class ValidationConfig(_Base):
    """Top-level model for ``configs/validation.yaml``."""

    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    parameter_stability: ParameterStabilityConfig = Field(default_factory=ParameterStabilityConfig)
    stress_tests: StressTestConfig = Field(default_factory=StressTestConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


# ---------------------------------------------------------------------------
# Aggregate configuration
# ---------------------------------------------------------------------------


class AtlasConfig(_Base):
    """Aggregate configuration object handed to every Atlas component."""

    assets: AssetsConfig
    strategies: StrategiesConfig
    regimes: RegimeConfig
    risk: RiskConfig
    execution: ExecutionConfig
    validation: ValidationConfig
    project_root: Path = Field(default_factory=lambda: find_project_root())
    source_files: dict[str, str] = Field(default_factory=dict)

    # -- convenience accessors ------------------------------------------------

    @property
    def universe(self) -> UniverseConfig:
        """The configured asset universe."""
        return self.assets.universe

    @property
    def data(self) -> DataConfig:
        """Data provider settings."""
        return self.assets.data

    @property
    def portfolio(self) -> PortfolioConfig:
        """Portfolio construction settings."""
        return self.risk.portfolio

    @property
    def backtest(self) -> BacktestConfig:
        """Backtest run settings."""
        return self.strategies.backtest

    @property
    def costs(self) -> CostConfig:
        """Transaction-cost settings."""
        return self.execution.costs

    @property
    def symbols(self) -> list[str]:
        """Tradable symbols in the universe."""
        return self.universe.tradable_symbols

    @property
    def asset_class_map(self) -> dict[str, str]:
        """Mapping of symbol -> asset class."""
        return self.universe.asset_class_map

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` relative to the project root when it is not absolute."""
        p = Path(path)
        return p if p.is_absolute() else (self.project_root / p)

    # -- reproducibility ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the full configuration.

        Used to make backtest runs reproducible: the snapshot is stored with each
        run alongside the git commit hash.
        """
        payload = self.model_dump(mode="json", exclude={"project_root", "source_files"})
        payload["_source_files"] = self.source_files
        return payload

    @property
    def config_hash(self) -> str:
        """Stable SHA-256 hash of the configuration snapshot."""
        blob = json.dumps(self.snapshot(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking up for a ``configs`` directory.

    Falls back to the current working directory when no marker is found, so the
    package remains importable from an installed wheel.
    """
    env_root = os.environ.get("ATLAS_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "configs").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the document is not a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file {p} must contain a mapping at the top level")
    return data


def load_config(
    config_dir: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> AtlasConfig:
    """Load and validate the full Atlas configuration.

    Parameters
    ----------
    config_dir:
        Directory containing ``assets.yaml``, ``strategies.yaml``, ``regimes.yaml``,
        ``risk.yaml``, ``execution.yaml`` and ``validation.yaml``. Defaults to
        ``<project_root>/configs``.
    overrides:
        Optional nested dictionary merged over the loaded YAML before validation,
        keyed by file stem, e.g. ``{"risk": {"portfolio": {"target_volatility": 0.08}}}``.

    Returns
    -------
    AtlasConfig
        A fully validated configuration object.
    """
    root = find_project_root()
    cdir = Path(config_dir) if config_dir is not None else root / "configs"
    cdir = cdir if cdir.is_absolute() else root / cdir

    names = {
        "assets": "assets.yaml",
        "strategies": "strategies.yaml",
        "regimes": "regimes.yaml",
        "risk": "risk.yaml",
        "execution": "execution.yaml",
        "validation": "validation.yaml",
    }
    raw: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}
    for key, fname in names.items():
        path = cdir / fname
        raw[key] = load_yaml(path)
        source_files[key] = str(path)

    if overrides:
        for key, patch in overrides.items():
            if key in raw and isinstance(patch, dict):
                raw[key] = _deep_merge(raw[key], patch)

    # Environment overrides for the few settings that are deployment-specific.
    _apply_env_overrides(raw)

    cfg = AtlasConfig(
        assets=AssetsConfig.model_validate(raw["assets"]),
        strategies=StrategiesConfig.model_validate(raw["strategies"]),
        regimes=RegimeConfig.model_validate(raw["regimes"]),
        risk=RiskConfig.model_validate(raw["risk"]),
        execution=ExecutionConfig.model_validate(raw["execution"]),
        validation=ValidationConfig.model_validate(raw["validation"]),
        project_root=root,
        source_files=source_files,
    )
    return cfg


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into a copy of ``base``."""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(raw: dict[str, dict[str, Any]]) -> None:
    """Apply the small set of environment-variable overrides, in place.

    Only deployment-specific settings are overridable from the environment;
    research parameters must live in version-controlled YAML so results stay
    reproducible. Credentials are never read into the configuration object.
    """
    exec_block = raw.setdefault("execution", {})

    mode = os.environ.get("EXECUTION_MODE") or os.environ.get("ATLAS_EXECUTION_MODE")
    if mode:
        exec_block["mode"] = mode.strip().lower()

    broker = exec_block.setdefault("broker", {})
    if host := os.environ.get("ATLAS_IBKR_HOST"):
        broker["host"] = host
    if port := os.environ.get("ATLAS_IBKR_PORT"):
        broker["port"] = int(port)
    if client_id := os.environ.get("ATLAS_IBKR_CLIENT_ID"):
        broker["client_id"] = int(client_id)
    if allowlist := os.environ.get("ATLAS_IBKR_ACCOUNT_ALLOWLIST"):
        broker["account_allowlist"] = [s.strip() for s in allowlist.split(",") if s.strip()]

    risk_block = raw.setdefault("risk", {})
    if os.environ.get("ATLAS_KILL_SWITCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        risk_block.setdefault("kill_switch", {})["enabled"] = True
        risk_block["kill_switch"]["reason"] = "enabled via ATLAS_KILL_SWITCH"
