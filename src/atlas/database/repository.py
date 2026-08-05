"""Repository layer: the only place that knows about the ORM.

Callers pass and receive pandas objects and Atlas dataclasses; the repository
translates. That keeps SQLAlchemy out of the research code and means the
storage backend could be swapped without touching a strategy.

Writes are idempotent where it matters: re-persisting the same price history or
the same backtest run updates rather than duplicates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select

from atlas.database.models import (
    Asset,
    BacktestMetric,
    BacktestRun,
    ConfigurationVersion,
    FillRecord,
    OrderRecord,
    PortfolioSnapshotRecord,
    PositionRecord,
    PriceBar,
    ReconciliationRecord,
    RegimeObservation,
    RiskEventRecord,
    StrategySignal,
    TargetWeight,
)
from atlas.database.session import DatabaseManager
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["AtlasRepository"]


def _decimal(value: Any) -> Decimal | None:
    """Convert a value to Decimal, returning None for missing/non-finite input."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return Decimal(str(round(number, 8)))


def _float(value: Any) -> float | None:
    """Convert to a finite float or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class AtlasRepository:
    """Persist and retrieve Atlas artefacts."""

    def __init__(self, database: DatabaseManager) -> None:
        self.db = database

    # -- reference data -------------------------------------------------------

    def upsert_assets(self, assets: list[dict[str, Any]]) -> int:
        """Insert or update universe metadata. Returns the number written."""
        written = 0
        with self.db.session() as session:
            for spec in assets:
                symbol = str(spec["symbol"]).upper()
                existing = session.scalar(select(Asset).where(Asset.symbol == symbol))
                if existing is None:
                    existing = Asset(symbol=symbol)
                    session.add(existing)
                existing.name = spec.get("name", existing.name or "")
                existing.asset_class = spec.get("asset_class", existing.asset_class)
                existing.tradable = bool(spec.get("tradable", True))
                if spec.get("first_date") is not None:
                    existing.first_date = pd.Timestamp(spec["first_date"]).date()
                if spec.get("last_date") is not None:
                    existing.last_date = pd.Timestamp(spec["last_date"]).date()
                written += 1
        return written

    def save_prices(self, frame: pd.DataFrame, *, replace: bool = False) -> int:
        """Persist a canonical long price frame.

        Existing ``(symbol, date)`` rows are updated in place, so re-importing
        overlapping history never duplicates bars.
        """
        if frame.empty:
            return 0

        written = 0
        with self.db.session() as session:
            asset_ids: dict[str, int] = dict(
                session.execute(select(Asset.symbol, Asset.id)).all()  # type: ignore[arg-type]
            )
            for symbol, group in frame.groupby("symbol"):
                symbol = str(symbol)
                if replace:
                    session.execute(delete(PriceBar).where(PriceBar.symbol == symbol))
                    existing: dict[Any, PriceBar] = {}
                else:
                    rows = session.scalars(
                        select(PriceBar).where(PriceBar.symbol == symbol)
                    ).all()
                    existing = {row.bar_date: row for row in rows}

                for record in group.to_dict("records"):
                    bar_date = pd.Timestamp(record["date"]).date()
                    bar = existing.get(bar_date)
                    if bar is None:
                        bar = PriceBar(symbol=symbol, bar_date=bar_date)
                        session.add(bar)
                    bar.asset_id = asset_ids.get(symbol)
                    bar.open = _decimal(record.get("open"))
                    bar.high = _decimal(record.get("high"))
                    bar.low = _decimal(record.get("low"))
                    bar.close = _decimal(record.get("close"))
                    bar.adj_close = _decimal(record.get("adj_close"))
                    bar.volume = _float(record.get("volume"))
                    bar.source = str(record.get("source", "unknown"))
                    downloaded = record.get("downloaded_at")
                    bar.downloaded_at = (
                        pd.Timestamp(downloaded).to_pydatetime()
                        if downloaded is not None and pd.notna(downloaded)
                        else None
                    )
                    written += 1
        log.info("prices persisted", extra={"context": {"rows": written}})
        return written

    def load_prices(
        self,
        symbols: list[str] | None = None,
        start: Any = None,
        end: Any = None,
    ) -> pd.DataFrame:
        """Load stored prices back into the canonical long frame."""
        with self.db.session() as session:
            query = select(PriceBar)
            if symbols:
                query = query.where(PriceBar.symbol.in_([s.upper() for s in symbols]))
            if start is not None:
                query = query.where(PriceBar.bar_date >= pd.Timestamp(start).date())
            if end is not None:
                query = query.where(PriceBar.bar_date <= pd.Timestamp(end).date())
            rows = session.scalars(query.order_by(PriceBar.symbol, PriceBar.bar_date)).all()

            records = [
                {
                    "date": pd.Timestamp(row.bar_date),
                    "symbol": row.symbol,
                    "open": _float(row.open),
                    "high": _float(row.high),
                    "low": _float(row.low),
                    "close": _float(row.close),
                    "adj_close": _float(row.adj_close),
                    "volume": row.volume,
                    "source": row.source,
                    "downloaded_at": row.downloaded_at,
                }
                for row in rows
            ]
        return pd.DataFrame(records)

    # -- configuration --------------------------------------------------------

    def save_configuration(self, config_hash: str, payload: dict[str, Any], label: str = "") -> int:
        """Store a configuration snapshot, returning its row id.

        A configuration with an identical hash is never overwritten - versions
        accumulate so historical results stay interpretable.
        """
        with self.db.session() as session:
            existing = session.scalar(
                select(ConfigurationVersion).where(ConfigurationVersion.config_hash == config_hash)
            )
            if existing is not None:
                return int(existing.id)
            version = ConfigurationVersion(config_hash=config_hash, payload=payload, label=label)
            session.add(version)
            session.flush()
            return int(version.id)

    def list_configurations(self) -> pd.DataFrame:
        """All stored configuration versions, newest first."""
        with self.db.session() as session:
            rows = session.scalars(
                select(ConfigurationVersion).order_by(ConfigurationVersion.created_at.desc())
            ).all()
            return pd.DataFrame(
                [
                    {
                        "id": row.id,
                        "config_hash": row.config_hash,
                        "label": row.label,
                        "created_at": row.created_at,
                    }
                    for row in rows
                ]
            )

    def get_configuration(self, config_hash: str) -> dict[str, Any] | None:
        """Retrieve a stored configuration payload by hash."""
        with self.db.session() as session:
            row = session.scalar(
                select(ConfigurationVersion).where(ConfigurationVersion.config_hash == config_hash)
            )
            return dict(row.payload) if row else None

    # -- backtest runs --------------------------------------------------------

    def save_backtest(
        self,
        result: Any,
        *,
        label: str = "",
        run_type: str = "backtest",
        persist_signals: bool = False,
        persist_regimes: bool = True,
    ) -> int:
        """Persist a :class:`~atlas.backtest.engine.BacktestResult`.

        Returns the run's row id. Snapshots, target weights, orders, fills, risk
        events and metrics are all written; per-strategy signals are optional
        because they are large.
        """
        snapshot = result.config_snapshot or {}
        config_hash = str(snapshot.get("_config_hash") or "")

        with self.db.session() as session:
            existing = session.scalar(select(BacktestRun).where(BacktestRun.run_id == result.run_id))
            if existing is not None:
                log.info("replacing existing run", extra={"context": {"run_id": result.run_id}})
                session.delete(existing)
                session.flush()

            equity = result.equity_curve
            run = BacktestRun(
                run_id=result.run_id,
                run_type=run_type,
                label=label,
                started_at=result.started_at,
                finished_at=result.finished_at,
                git_commit=result.git_commit,
                config_hash=config_hash or None,
                config_snapshot=snapshot,
                universe=snapshot.get("assets", {}).get("universe", {}) if snapshot else {},
                strategy_parameters=snapshot.get("strategies", {}) if snapshot else {},
                cost_assumptions=snapshot.get("execution", {}).get("costs", {}) if snapshot else {},
                data_start=equity.index.min().date() if len(equity) else None,
                data_end=equity.index.max().date() if len(equity) else None,
                data_source=result.data_source,
                is_synthetic_data=bool(result.is_synthetic_data),
                initial_capital=_decimal(result.initial_capital),
                final_equity=_decimal(result.final_equity),
                result_summary=result.summary(),
                notes="\n".join(result.warnings),
            )
            session.add(run)
            session.flush()
            run_id = int(run.id)

            if result.metrics is not None:
                session.add_all(
                    BacktestMetric(
                        run_id=run_id, series="portfolio", metric_name=name, value=_float(value)
                    )
                    for name, value in result.metrics.values.items()
                )

            self._write_snapshots(session, run_id, result)
            self._write_target_weights(session, run_id, result)
            self._write_orders_and_fills(session, run_id, result)
            self._write_risk_events(session, run_id, result)
            if persist_regimes and result.regimes is not None:
                self._write_regimes(session, run_id, result)
            if persist_signals:
                self._write_signals(session, run_id, result)

        log.info(
            "backtest persisted",
            extra={"context": {"run_id": result.run_id, "row_id": run_id}},
        )
        return run_id

    @staticmethod
    def _write_snapshots(session, run_id: int, result: Any) -> None:
        """Write the daily portfolio snapshots."""
        snapshots = result.snapshots
        if snapshots.empty:
            return
        regimes = result.regimes.as_strings() if result.regimes is not None else None
        session.add_all(
            PortfolioSnapshotRecord(
                run_id=run_id,
                snapshot_date=pd.Timestamp(index).date(),
                equity=_decimal(row.get("equity")) or Decimal(0),
                cash=_decimal(row.get("cash")) or Decimal(0),
                positions_value=_decimal(row.get("positions_value")) or Decimal(0),
                gross_exposure=_float(row.get("gross_exposure")),
                net_exposure=_float(row.get("net_exposure")),
                n_positions=int(row.get("n_positions", 0) or 0),
                daily_pnl=_decimal(row.get("daily_pnl")),
                turnover=_float(row.get("turnover")),
                cumulative_costs=_decimal(row.get("cumulative_costs")),
                regime=(
                    str(regimes.get(index)) if regimes is not None and index in regimes.index else None
                ),
            )
            for index, row in snapshots.iterrows()
        )

    @staticmethod
    def _write_target_weights(session, run_id: int, result: Any) -> None:
        """Write requested target weights on each rebalance date."""
        targets = result.target_weights
        if targets.empty:
            return
        records = []
        for index, row in targets.iterrows():
            for symbol, weight in row.items():
                if abs(float(weight)) < 1e-9:
                    continue
                records.append(
                    TargetWeight(
                        run_id=run_id,
                        weight_date=pd.Timestamp(index).date(),
                        symbol=str(symbol),
                        target_weight=_decimal(weight),
                    )
                )
        session.add_all(records)

    @staticmethod
    def _write_orders_and_fills(session, run_id: int, result: Any) -> None:
        """Write orders and their fills, linking each fill to its order row."""
        order_rows: dict[str, OrderRecord] = {}
        frames = [result.orders, result.rejected_orders]
        for frame in frames:
            if frame is None or frame.empty:
                continue
            for record in frame.to_dict("records"):
                order_id = str(record.get("order_id", ""))
                if not order_id or order_id in order_rows:
                    continue
                row = OrderRecord(
                    order_id=order_id,
                    broker_order_id=record.get("broker_order_id"),
                    run_id=run_id,
                    symbol=str(record.get("symbol", "")),
                    side=str(record.get("side", "")),
                    quantity=_decimal(record.get("quantity")) or Decimal(0),
                    order_type=str(record.get("order_type", "MKT")),
                    time_in_force=str(record.get("time_in_force", "DAY")),
                    limit_price=_decimal(record.get("limit_price")),
                    reference_price=_decimal(record.get("reference_price")),
                    target_weight=_decimal(record.get("target_weight")),
                    status=str(record.get("status", "created")),
                    filled_quantity=_decimal(record.get("filled_quantity")) or Decimal(0),
                    average_fill_price=_decimal(record.get("average_fill_price")),
                    reject_reason=str(record.get("reject_reason", "") or ""),
                    execution_mode="backtest",
                )
                order_rows[order_id] = row
                session.add(row)
        session.flush()

        if result.fills is None or result.fills.empty:
            return
        session.add_all(
            FillRecord(
                fill_id=str(record.get("fill_id")),
                order_id=(
                    order_rows[str(record.get("order_id"))].id
                    if str(record.get("order_id")) in order_rows
                    else None
                ),
                run_id=run_id,
                symbol=str(record.get("symbol", "")),
                side=str(record.get("side", "")),
                quantity=_decimal(record.get("quantity")) or Decimal(0),
                price=_decimal(record.get("price")) or Decimal(0),
                reference_price=_decimal(record.get("reference_price")),
                filled_at=pd.Timestamp(record.get("timestamp")).to_pydatetime(),
                commission=_decimal(record.get("commission")) or Decimal(0),
                spread_cost=_decimal(record.get("spread_cost")) or Decimal(0),
                slippage_cost=_decimal(record.get("slippage_cost")) or Decimal(0),
                impact_cost=_decimal(record.get("impact_cost")) or Decimal(0),
            )
            for record in result.fills.to_dict("records")
        )

    @staticmethod
    def _write_risk_events(session, run_id: int, result: Any) -> None:
        """Write the risk-event audit trail."""
        events = result.risk_events
        if events is None or events.empty:
            return
        session.add_all(
            RiskEventRecord(
                run_id=run_id,
                occurred_at=pd.Timestamp(record.get("timestamp")).to_pydatetime(),
                control=str(record.get("control", "")),
                severity=str(record.get("severity", "info")),
                action=str(record.get("action", "allowed")),
                message=str(record.get("message", "")),
                observed=_float(record.get("observed")),
                limit_value=_float(record.get("limit")),
            )
            for record in events.to_dict("records")
        )

    @staticmethod
    def _write_regimes(session, run_id: int, result: Any) -> None:
        """Write regime labels and their underlying indicators."""
        regimes = result.regimes
        labels = regimes.as_strings()
        features = regimes.features
        session.add_all(
            RegimeObservation(
                run_id=run_id,
                observation_date=pd.Timestamp(index).date(),
                detector=regimes.detector,
                regime=str(label),
                trend_score=_float(features.at[index, "trend_score"])
                if "trend_score" in features.columns and index in features.index
                else None,
                volatility=_float(features.at[index, "volatility"])
                if "volatility" in features.columns and index in features.index
                else None,
                volatility_percentile=_float(features.at[index, "volatility_percentile"])
                if "volatility_percentile" in features.columns and index in features.index
                else None,
                breadth=_float(features.at[index, "breadth"])
                if "breadth" in features.columns and index in features.index
                else None,
                avg_correlation=_float(features.at[index, "avg_correlation"])
                if "avg_correlation" in features.columns and index in features.index
                else None,
                drawdown=_float(features.at[index, "drawdown"])
                if "drawdown" in features.columns and index in features.index
                else None,
            )
            for index, label in labels.items()
        )

    @staticmethod
    def _write_signals(session, run_id: int, result: Any) -> None:
        """Write per-strategy signals (large; opt-in)."""
        records = []
        for strategy, frame in result.signals.items():
            stacked = frame.stack(future_stack=True).dropna()
            for (index, symbol), value in stacked.items():
                if abs(float(value)) < 1e-9:
                    continue
                records.append(
                    StrategySignal(
                        run_id=run_id,
                        signal_date=pd.Timestamp(index).date(),
                        symbol=str(symbol),
                        strategy=str(strategy),
                        normalized_signal=float(value),
                    )
                )
        session.add_all(records)

    # -- run queries ----------------------------------------------------------

    def list_runs(self, limit: int = 50) -> pd.DataFrame:
        """Recent backtest runs, newest first."""
        with self.db.session() as session:
            rows = session.scalars(
                select(BacktestRun).order_by(BacktestRun.started_at.desc()).limit(limit)
            ).all()
            return pd.DataFrame(
                [
                    {
                        "run_id": row.run_id,
                        "run_type": row.run_type,
                        "label": row.label,
                        "started_at": row.started_at,
                        "data_start": row.data_start,
                        "data_end": row.data_end,
                        "data_source": row.data_source,
                        "synthetic": row.is_synthetic_data,
                        "final_equity": _float(row.final_equity),
                        "git_commit": (row.git_commit or "")[:8],
                    }
                    for row in rows
                ]
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch one run's header and metrics by ``run_id``."""
        with self.db.session() as session:
            row = session.scalar(select(BacktestRun).where(BacktestRun.run_id == run_id))
            if row is None:
                return None
            metrics = session.scalars(
                select(BacktestMetric).where(BacktestMetric.run_id == row.id)
            ).all()
            return {
                "run_id": row.run_id,
                "run_type": row.run_type,
                "label": row.label,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "git_commit": row.git_commit,
                "config_hash": row.config_hash,
                "config_snapshot": dict(row.config_snapshot or {}),
                "data_start": row.data_start,
                "data_end": row.data_end,
                "data_source": row.data_source,
                "is_synthetic_data": row.is_synthetic_data,
                "initial_capital": _float(row.initial_capital),
                "final_equity": _float(row.final_equity),
                "result_summary": dict(row.result_summary or {}),
                "notes": row.notes,
                "metrics": {m.metric_name: m.value for m in metrics},
            }

    def get_equity_curve(self, run_id: str) -> pd.Series:
        """Equity curve for a stored run."""
        with self.db.session() as session:
            run = session.scalar(select(BacktestRun).where(BacktestRun.run_id == run_id))
            if run is None:
                return pd.Series(dtype="float64")
            rows = session.scalars(
                select(PortfolioSnapshotRecord)
                .where(PortfolioSnapshotRecord.run_id == run.id)
                .order_by(PortfolioSnapshotRecord.snapshot_date)
            ).all()
            if not rows:
                return pd.Series(dtype="float64")
            return pd.Series(
                [float(r.equity) for r in rows],
                index=pd.DatetimeIndex([r.snapshot_date for r in rows], name="date"),
                name="equity",
            )

    # -- live session artefacts ----------------------------------------------

    def save_order(self, order: Any, *, session_id: str, execution_mode: str) -> None:
        """Persist a live/dry-run order."""
        with self.db.session() as session:
            existing = session.scalar(
                select(OrderRecord).where(OrderRecord.order_id == order.order_id)
            )
            row = existing or OrderRecord(order_id=order.order_id)
            row.broker_order_id = order.broker_order_id
            row.session_id = session_id
            row.symbol = order.symbol
            row.side = order.side.value
            row.quantity = _decimal(order.quantity) or Decimal(0)
            row.order_type = order.order_type.value
            row.time_in_force = order.time_in_force.value
            row.limit_price = _decimal(order.limit_price)
            row.reference_price = _decimal(order.reference_price)
            row.target_weight = _decimal(order.target_weight)
            row.status = order.status.value
            row.filled_quantity = _decimal(order.filled_quantity) or Decimal(0)
            row.average_fill_price = _decimal(order.average_fill_price)
            row.reject_reason = order.reject_reason
            row.execution_mode = execution_mode
            if existing is None:
                session.add(row)

    def save_fill(self, fill: Any, *, session_id: str) -> None:  # noqa: ARG002
        """Persist a live/dry-run fill."""
        with self.db.session() as session:
            order = session.scalar(select(OrderRecord).where(OrderRecord.order_id == fill.order_id))
            session.add(
                FillRecord(
                    fill_id=fill.fill_id,
                    order_id=order.id if order else None,
                    symbol=fill.symbol,
                    side=fill.side.value,
                    quantity=_decimal(fill.quantity) or Decimal(0),
                    price=_decimal(fill.price) or Decimal(0),
                    reference_price=_decimal(fill.reference_price),
                    filled_at=pd.Timestamp(fill.timestamp).to_pydatetime(),
                    commission=_decimal(fill.commission) or Decimal(0),
                )
            )

    def save_positions(
        self, positions: pd.DataFrame, *, session_id: str, as_of: Any, source: str = "local"
    ) -> int:
        """Persist a snapshot of positions."""
        if positions.empty:
            return 0
        as_of_date = pd.Timestamp(as_of).date()
        frame = positions.reset_index() if positions.index.name == "symbol" else positions
        with self.db.session() as session:
            session.execute(
                delete(PositionRecord).where(
                    PositionRecord.session_id == session_id,
                    PositionRecord.position_date == as_of_date,
                    PositionRecord.source == source,
                )
            )
            session.add_all(
                PositionRecord(
                    session_id=session_id,
                    position_date=as_of_date,
                    symbol=str(record.get("symbol")),
                    quantity=_decimal(record.get("quantity")) or Decimal(0),
                    average_price=_decimal(record.get("average_price")),
                    market_price=_decimal(record.get("last_price")),
                    market_value=_decimal(record.get("market_value")),
                    unrealized_pnl=_decimal(record.get("unrealized_pnl")),
                    realized_pnl=_decimal(record.get("realized_pnl")),
                    weight=_decimal(record.get("weight")),
                    source=source,
                )
                for record in frame.to_dict("records")
            )
        return len(frame)

    def latest_positions(
        self, *, session_id: str | None = None, source: str = "local"
    ) -> dict[str, float]:
        """Return the most recently stored position snapshot as ``{symbol: qty}``.

        An empty mapping means Atlas holds no record - which is the correct
        answer for a fresh installation, and is why reconciliation treats a
        broker position Atlas does not know about as a material mismatch.
        """
        with self.db.session() as session:
            query = select(PositionRecord).where(PositionRecord.source == source)
            if session_id:
                query = query.where(PositionRecord.session_id == session_id)
            latest_date = session.scalars(
                query.order_by(PositionRecord.position_date.desc()).limit(1)
            ).first()
            if latest_date is None:
                return {}
            rows = session.scalars(
                query.where(PositionRecord.position_date == latest_date.position_date)
            ).all()
            return {
                r.symbol: float(r.quantity)
                for r in rows
                if r.quantity is not None and abs(float(r.quantity)) > 1e-12
            }

    def save_risk_events(self, events: pd.DataFrame, *, session_id: str) -> int:
        """Persist risk events from a live session."""
        if events.empty:
            return 0
        with self.db.session() as session:
            session.add_all(
                RiskEventRecord(
                    session_id=session_id,
                    occurred_at=pd.Timestamp(record.get("timestamp")).to_pydatetime(),
                    control=str(record.get("control", "")),
                    severity=str(record.get("severity", "info")),
                    action=str(record.get("action", "allowed")),
                    message=str(record.get("message", "")),
                    observed=_float(record.get("observed")),
                    limit_value=_float(record.get("limit")),
                )
                for record in events.to_dict("records")
            )
        return len(events)

    def save_reconciliation(self, report: Any, *, session_id: str, account: str = "") -> int:
        """Persist a reconciliation outcome, returning its row id."""
        with self.db.session() as session:
            row = ReconciliationRecord(
                session_id=session_id,
                checked_at=datetime.now(UTC),
                account=account,
                status=report.status,
                n_symbols_checked=int(report.n_checked),
                n_mismatches=len(report.mismatches),
                max_share_difference=_float(report.max_share_difference),
                max_value_difference=_float(report.max_value_difference),
                blocking=bool(report.blocking),
                details={"mismatches": report.mismatch_records()},
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def latest_reconciliation(self, session_id: str | None = None) -> dict[str, Any] | None:
        """Most recent reconciliation record."""
        with self.db.session() as session:
            query = select(ReconciliationRecord).order_by(ReconciliationRecord.checked_at.desc())
            if session_id:
                query = query.where(ReconciliationRecord.session_id == session_id)
            row = session.scalars(query.limit(1)).first()
            if row is None:
                return None
            return {
                "session_id": row.session_id,
                "checked_at": row.checked_at,
                "account": row.account,
                "status": row.status,
                "n_mismatches": row.n_mismatches,
                "blocking": row.blocking,
                "details": dict(row.details or {}),
            }

    def recent_orders(self, session_id: str | None = None, limit: int = 100) -> pd.DataFrame:
        """Recent orders, newest first."""
        with self.db.session() as session:
            query = select(OrderRecord).order_by(OrderRecord.created_at.desc()).limit(limit)
            if session_id:
                query = query.where(OrderRecord.session_id == session_id)
            rows = session.scalars(query).all()
            return pd.DataFrame(
                [
                    {
                        "order_id": r.order_id,
                        "broker_order_id": r.broker_order_id,
                        "symbol": r.symbol,
                        "side": r.side,
                        "quantity": _float(r.quantity),
                        "status": r.status,
                        "filled_quantity": _float(r.filled_quantity),
                        "average_fill_price": _float(r.average_fill_price),
                        "reject_reason": r.reject_reason,
                        "execution_mode": r.execution_mode,
                        "created_at": r.created_at,
                    }
                    for r in rows
                ]
            )
