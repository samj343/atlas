"""SQLAlchemy models for Atlas persistence.

The schema is written against SQLAlchemy 2.0's typed declarative API and uses
only portable column types, so the same models run on SQLite locally and on
PostgreSQL in a deployment. Notable choices:

* Money and prices use ``Numeric`` rather than ``Float`` so repeated arithmetic
  does not accumulate binary rounding error in stored records.
* Every research artefact hangs off a :class:`BacktestRun`, which carries the
  configuration snapshot, the git commit and the data range. That is what makes
  a stored result reproducible rather than merely archived.
* Time-series tables carry ``(symbol, date)`` unique constraints so a re-import
  updates rather than duplicates.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Asset",
    "BacktestMetric",
    "BacktestRun",
    "Base",
    "ConfigurationVersion",
    "FeatureValue",
    "FillRecord",
    "OrderRecord",
    "PortfolioSnapshotRecord",
    "PositionRecord",
    "PriceBar",
    "ReconciliationRecord",
    "RegimeObservation",
    "RiskEventRecord",
    "StrategySignal",
    "TargetWeight",
]

# Prices need more precision than money; both need more than a float.
PRICE = Numeric(18, 6)
MONEY = Numeric(20, 4)
QUANTITY = Numeric(20, 6)
WEIGHT = Numeric(12, 8)


def _utcnow() -> datetime:
    """Timezone-aware current UTC timestamp."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every Atlas table."""

    type_annotation_map = {dict[str, Any]: JSON, Decimal: MONEY}


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


class Asset(Base):
    """A tradable instrument in the universe."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    asset_class: Mapped[str] = mapped_column(String(50), default="unclassified", index=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    tradable: Mapped[bool] = mapped_column(Boolean, default=True)
    first_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    bars: Mapped[list[PriceBar]] = relationship(back_populates="asset", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Asset {self.symbol} ({self.asset_class})>"


class PriceBar(Base):
    """One daily OHLCV bar, with its provenance."""

    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "bar_date", name="uq_price_symbol_date"),
        Index("ix_price_symbol_date", "symbol", "bar_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("assets.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    bar_date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    high: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    low: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    close: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    adj_close: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="unknown")
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    asset: Mapped[Asset | None] = relationship(back_populates="bars")

    def __repr__(self) -> str:
        return f"<PriceBar {self.symbol} {self.bar_date} close={self.close}>"


class FeatureValue(Base):
    """A computed feature value for one symbol on one date."""

    __tablename__ = "feature_values"
    __table_args__ = (
        UniqueConstraint("symbol", "feature_date", "feature_name", name="uq_feature"),
        Index("ix_feature_lookup", "feature_name", "feature_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    feature_date: Mapped[date] = mapped_column(Date, index=True)
    feature_name: Mapped[str] = mapped_column(String(80))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)


# ---------------------------------------------------------------------------
# Research artefacts
# ---------------------------------------------------------------------------


class ConfigurationVersion(Base):
    """An immutable snapshot of a configuration.

    Configuration changes create a new row rather than overwriting the previous
    one, so a stored result can always be traced back to the exact settings that
    produced it.
    """

    __tablename__ = "configuration_versions"
    __table_args__ = (UniqueConstraint("config_hash", name="uq_config_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str] = mapped_column(String(100), default="atlas")

    def __repr__(self) -> str:
        return f"<ConfigurationVersion {self.config_hash} {self.label!r}>"


class BacktestRun(Base):
    """A single backtest execution and everything needed to reproduce it."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    run_type: Mapped[str] = mapped_column(String(30), default="backtest")
    label: Mapped[str] = mapped_column(String(200), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    universe: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    strategy_parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cost_assumptions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    data_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    data_source: Mapped[str] = mapped_column(String(50), default="unknown")
    is_synthetic_data: Mapped[bool] = mapped_column(Boolean, default=False)
    initial_capital: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    final_equity: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    result_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")

    metrics: Mapped[list[BacktestMetric]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    snapshots: Mapped[list[PortfolioSnapshotRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<BacktestRun {self.run_id} {self.data_start}..{self.data_end}>"


class BacktestMetric(Base):
    """One named metric from a run."""

    __tablename__ = "backtest_metrics"
    __table_args__ = (
        UniqueConstraint("run_id", "series", "metric_name", name="uq_metric"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    series: Mapped[str] = mapped_column(String(60), default="portfolio")
    metric_name: Mapped[str] = mapped_column(String(80))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[BacktestRun] = relationship(back_populates="metrics")


class StrategySignal(Base):
    """A strategy's signal for one symbol on one date."""

    __tablename__ = "strategy_signals"
    __table_args__ = (
        UniqueConstraint("run_id", "signal_date", "symbol", "strategy", name="uq_signal"),
        Index("ix_signal_lookup", "run_id", "signal_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    signal_date: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    strategy: Mapped[str] = mapped_column(String(60), index=True)
    raw_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    normalized_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class RegimeObservation(Base):
    """The regime label and its underlying indicators for one date."""

    __tablename__ = "regime_observations"
    __table_args__ = (
        UniqueConstraint("run_id", "observation_date", "detector", name="uq_regime"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    observation_date: Mapped[date] = mapped_column(Date, index=True)
    detector: Mapped[str] = mapped_column(String(40), default="rules_based")
    regime: Mapped[str] = mapped_column(String(40), index=True)
    trend_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    breadth: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_correlation: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)


class TargetWeight(Base):
    """A requested target weight for one symbol on one rebalance date."""

    __tablename__ = "target_weights"
    __table_args__ = (
        UniqueConstraint("run_id", "weight_date", "symbol", name="uq_target_weight"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    weight_date: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    target_weight: Mapped[Decimal | None] = mapped_column(WEIGHT, nullable=True)
    signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(40), nullable=True)
    risk_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)


# ---------------------------------------------------------------------------
# Execution artefacts
# ---------------------------------------------------------------------------


class OrderRecord(Base):
    """An order, whether simulated or sent to the broker."""

    __tablename__ = "orders"
    __table_args__ = (Index("ix_order_lookup", "session_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[Decimal] = mapped_column(QUANTITY)
    order_type: Mapped[str] = mapped_column(String(8), default="MKT")
    time_in_force: Mapped[str] = mapped_column(String(8), default="DAY")
    limit_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    reference_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    target_weight: Mapped[Decimal | None] = mapped_column(WEIGHT, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="created", index=True)
    filled_quantity: Mapped[Decimal] = mapped_column(QUANTITY, default=Decimal(0))
    average_fill_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    reject_reason: Mapped[str] = mapped_column(Text, default="")
    execution_mode: Mapped[str] = mapped_column(String(20), default="backtest")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    fills: Mapped[list[FillRecord]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Order {self.order_id} {self.side} {self.quantity} {self.symbol} [{self.status}]>"


class FillRecord(Base):
    """An execution against an order, with its itemised costs."""

    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fill_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[Decimal] = mapped_column(QUANTITY)
    price: Mapped[Decimal] = mapped_column(PRICE)
    reference_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    commission: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    spread_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    slippage_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    impact_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))

    order: Mapped[OrderRecord | None] = relationship(back_populates="fills")


class PositionRecord(Base):
    """A position held at a point in time."""

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("run_id", "session_id", "position_date", "symbol", name="uq_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    position_date: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    quantity: Mapped[Decimal] = mapped_column(QUANTITY)
    average_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    market_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    market_value: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    realized_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    weight: Mapped[Decimal | None] = mapped_column(WEIGHT, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="local")


class PortfolioSnapshotRecord(Base):
    """Portfolio state on one date."""

    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "session_id", "snapshot_date", name="uq_snapshot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    snapshot_date: Mapped[date] = mapped_column(Date, index=True)
    equity: Mapped[Decimal] = mapped_column(MONEY)
    cash: Mapped[Decimal] = mapped_column(MONEY)
    positions_value: Mapped[Decimal] = mapped_column(MONEY)
    gross_exposure: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_exposure: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_positions: Mapped[int] = mapped_column(Integer, default=0)
    daily_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    cumulative_costs: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(40), nullable=True)

    run: Mapped[BacktestRun | None] = relationship(back_populates="snapshots")


class RiskEventRecord(Base):
    """A risk-control decision."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    control: Mapped[str] = mapped_column(String(60), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="info")
    action: Mapped[str] = mapped_column(String(20), default="allowed")
    message: Mapped[str] = mapped_column(Text, default="")
    observed: Mapped[float | None] = mapped_column(Float, nullable=True)
    limit_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ReconciliationRecord(Base):
    """The outcome of one broker/local reconciliation."""

    __tablename__ = "reconciliations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    account: Mapped[str] = mapped_column(String(40), default="")
    status: Mapped[str] = mapped_column(String(20), default="unknown", index=True)
    n_symbols_checked: Mapped[int] = mapped_column(Integer, default=0)
    n_mismatches: Mapped[int] = mapped_column(Integer, default=0)
    max_share_difference: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_value_difference: Mapped[float | None] = mapped_column(Float, nullable=True)
    blocking: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    def __repr__(self) -> str:
        return f"<Reconciliation {self.session_id} {self.status} mismatches={self.n_mismatches}>"
