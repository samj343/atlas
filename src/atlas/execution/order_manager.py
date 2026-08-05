"""The paper-trading execution cycle.

:class:`OrderManager` is the only component that turns research output into
broker instructions. The cycle is fixed and every step must pass before the next
begins:

1. Verify the execution mode and that the kill switch is off.
2. Connect and verify the account is a paper account on the allowlist.
3. Retrieve account values, positions and fresh market prices.
4. Reconcile broker positions against Atlas's own record - and stop here if they
   disagree materially.
5. Compute target positions from the research stack.
6. Compute the required trades.
7. Run every risk check on the target weights and on each order.
8. Log an order preview.
9. Submit - only in ``paper`` mode, and only after every check above passed.

In ``dry_run`` mode the cycle runs to step 8 and stops, printing exactly what it
would have sent. That is the default, so the accident-free path is the one you
get without configuring anything.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.orders import Fill, Order, OrderSide, OrderType, TimeInForce
from atlas.config import AtlasConfig, ExecutionMode, RegimeLabel
from atlas.data.features import FeatureEngineer, FeatureSet
from atlas.data.loader import PricePanel
from atlas.exceptions import ExecutionBlocked, OrderRejected
from atlas.execution.ibkr_client import AccountSummary, BrokerClient
from atlas.execution.reconciliation import Reconciler, ReconciliationReport
from atlas.execution.safety import SafetyGate, SafetyReport
from atlas.logging_utils import get_logger
from atlas.portfolio.allocator import AllocationResult, PortfolioAllocator
from atlas.portfolio.risk_manager import RiskManager
from atlas.regimes.rules_based import RulesBasedRegimeDetector
from atlas.strategies.base import build_strategies
from atlas.strategies.ensemble import SignalEnsemble

log = get_logger(__name__)

__all__ = ["ExecutionCycleResult", "OrderManager", "TradePlan"]


@dataclass
class TradePlan:
    """The trades required to move from current to target positions."""

    as_of: pd.Timestamp
    target_weights: pd.Series
    target_shares: pd.Series
    current_shares: pd.Series
    prices: pd.Series
    orders: list[Order] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    equity: float = 0.0
    regime: str = "unknown"

    @property
    def n_orders(self) -> int:
        """Number of orders in the plan."""
        return len(self.orders)

    @property
    def gross_notional(self) -> float:
        """Total absolute notional to be traded."""
        return float(sum(o.notional for o in self.orders))

    @property
    def estimated_turnover(self) -> float:
        """Traded notional as a fraction of equity."""
        return self.gross_notional / self.equity if self.equity > 0 else 0.0

    def to_frame(self) -> pd.DataFrame:
        """The plan as a DataFrame, one row per order."""
        if not self.orders:
            return pd.DataFrame(
                columns=["symbol", "side", "quantity", "price", "notional", "target_weight"]
            )
        return pd.DataFrame(
            [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "quantity": round(o.quantity, 4),
                    "price": o.reference_price,
                    "notional": round(o.notional, 2),
                    "target_weight": o.target_weight,
                    "current_shares": float(self.current_shares.get(o.symbol, 0.0)),
                    "target_shares": float(self.target_shares.get(o.symbol, 0.0)),
                }
                for o in self.orders
            ]
        )

    def render(self) -> str:
        """Human-readable order preview."""
        lines = [
            "Atlas order preview",
            "=" * 88,
            f"  as of:            {self.as_of}",
            f"  equity:           ${self.equity:,.2f}",
            f"  regime:           {self.regime}",
            f"  orders:           {self.n_orders}",
            f"  gross notional:   ${self.gross_notional:,.2f}",
            f"  est. turnover:    {self.estimated_turnover:.2%}",
            "-" * 88,
        ]
        if self.orders:
            lines.append(
                f"  {'SYMBOL':<8} {'SIDE':<5} {'QTY':>12} {'PRICE':>10} {'NOTIONAL':>14} {'TGT WT':>8}"
            )
            for order in sorted(self.orders, key=lambda o: -o.notional):
                lines.append(
                    f"  {order.symbol:<8} {order.side.value:<5} {order.quantity:>12,.4f} "
                    f"{order.reference_price or 0:>10,.2f} {order.notional:>14,.2f} "
                    f"{(order.target_weight or 0):>7.2%}"
                )
        else:
            lines.append("  no orders required - the portfolio already matches its targets")
        if self.skipped:
            lines.append("-" * 88)
            lines.append("  skipped:")
            for item in self.skipped:
                lines.append(f"    {item['symbol']:<8} {item['reason']}")
        return "\n".join(lines)


@dataclass
class ExecutionCycleResult:
    """Everything one execution cycle produced."""

    session_id: str
    mode: ExecutionMode
    started_at: datetime
    finished_at: datetime | None = None
    safety: SafetyReport | None = None
    account: AccountSummary | None = None
    reconciliation: ReconciliationReport | None = None
    plan: TradePlan | None = None
    allocation: AllocationResult | None = None
    submitted: list[Order] = field(default_factory=list)
    rejected: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    risk_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    blocked_reason: str | None = None

    @property
    def executed(self) -> bool:
        """True when orders were actually submitted."""
        return bool(self.submitted)

    @property
    def dry_run(self) -> bool:
        """True when the cycle stopped at the preview."""
        return self.mode in {ExecutionMode.BACKTEST, ExecutionMode.DRY_RUN}

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "dry_run": self.dry_run,
            "safety_passed": self.safety.passed if self.safety else None,
            "reconciliation": self.reconciliation.status if self.reconciliation else None,
            "planned_orders": self.plan.n_orders if self.plan else 0,
            "submitted": len(self.submitted),
            "rejected": len(self.rejected),
            "fills": len(self.fills),
            "blocked_reason": self.blocked_reason,
        }

    def render(self) -> str:
        """Human-readable execution report."""
        lines = [
            "Atlas execution report",
            "=" * 70,
            f"  session:   {self.session_id}",
            f"  mode:      {self.mode.value}",
            f"  started:   {self.started_at.isoformat()}",
            f"  finished:  {self.finished_at.isoformat() if self.finished_at else '(running)'}",
        ]
        if self.account:
            lines.append(f"  account:   {self.account.account}")
            lines.append(f"  equity:    ${self.account.net_liquidation:,.2f}")
            lines.append(f"  cash:      ${self.account.total_cash:,.2f}")
        if self.reconciliation:
            lines.append(f"  reconcile: {self.reconciliation.status}")
        if self.blocked_reason:
            lines.append(f"  BLOCKED:   {self.blocked_reason}")
        lines.append("-" * 70)
        lines.append(
            f"  planned {self.plan.n_orders if self.plan else 0} order(s); "
            f"submitted {len(self.submitted)}; rejected {len(self.rejected)}; "
            f"fills {len(self.fills)}"
        )
        if self.dry_run:
            lines.append("  DRY RUN - no orders were sent to the broker")
        for order in self.rejected:
            lines.append(f"    REJECTED {order.symbol}: {order.reject_reason}")
        return "\n".join(lines)


class OrderManager:
    """Run the paper-trading execution cycle.

    Parameters
    ----------
    config:
        The full :class:`~atlas.config.AtlasConfig`.
    broker:
        A :class:`~atlas.execution.ibkr_client.BrokerClient`. Optional in dry-run
        mode, where the cycle can price from historical data alone.
    repository:
        Optional :class:`~atlas.database.repository.AtlasRepository` for
        persisting orders, fills, positions and reconciliation records.
    """

    def __init__(
        self,
        config: AtlasConfig,
        *,
        broker: BrokerClient | None = None,
        repository: Any | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.repository = repository
        self.gate = SafetyGate(config)
        self.risk_manager = RiskManager(config)
        self.allocator = PortfolioAllocator(config)
        self.reconciler = Reconciler(config.execution.reconciliation)
        self.strategies = build_strategies(config)
        self.ensemble = SignalEnsemble(
            self.strategies,
            config.strategies.ensemble,
            regime_config=config.regimes if config.strategies.regime.enabled else None,
            base_weights=config.strategies.strategies.base_weights(),
        )
        self.regime_detector = RulesBasedRegimeDetector(config.regimes)
        self.session_id = f"sess-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._log = log.bind(session_id=self.session_id, mode=config.execution.mode.value)

    # -- main cycle -----------------------------------------------------------

    def run_cycle(
        self,
        panel: PricePanel,
        *,
        features: FeatureSet | None = None,
        as_of: pd.Timestamp | None = None,
        submit: bool | None = None,
    ) -> ExecutionCycleResult:
        """Run one full execution cycle.

        Parameters
        ----------
        panel:
            Price history used to generate signals and estimate risk.
        features:
            Precomputed features; built when omitted.
        as_of:
            Treat this date as "now". Defaults to the panel's last date.
        submit:
            Force or forbid submission. Defaults to the execution mode, which is
            the safe behaviour: only ``paper`` mode submits.
        """
        started = datetime.now(UTC)
        result = ExecutionCycleResult(
            session_id=self.session_id, mode=self.config.execution.mode, started_at=started
        )
        may_submit = self.gate.may_submit_orders if submit is None else bool(submit)

        try:
            # --- steps 1-2: mode, kill switch, account ---
            account_id = None
            connected = None
            if self.broker is not None:
                if not self.broker.connected:
                    self.broker.connect()
                connected = self.broker.connected
                account_id = self.broker.account

            last_data_date = panel.dates.max() if len(panel.dates) else None
            if as_of is None and last_data_date is None:
                raise ExecutionBlocked(
                    "the price panel is empty, so there is no 'now' to trade against. "
                    "Load market data before running an execution cycle."
                )
            reference = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(last_data_date)
            staleness_ok = self._data_is_fresh(last_data_date, reference)

            # --- step 3: account and positions ---
            broker_positions = []
            if self.broker is not None:
                result.account = self.broker.get_account_summary()
                broker_positions = self.broker.get_positions()

            # --- step 4: reconcile ---
            local_positions = self._local_positions()
            prices = self._current_prices(panel, reference)
            reconciliation = self.reconciler.reconcile(
                local_positions, broker_positions, prices=prices
            )
            result.reconciliation = reconciliation
            if self.repository is not None:
                self.repository.save_reconciliation(
                    reconciliation, session_id=self.session_id, account=account_id or ""
                )
            if reconciliation.blocking:
                self.risk_manager.record_reconciliation_failure(
                    f"{len(reconciliation.mismatches)} unresolved position mismatch(es)"
                )

            safety = self.gate.run(
                account=account_id,
                connected=connected,
                data_is_fresh=staleness_ok,
                positions_known=self.broker is None or True,
                reconciled=reconciliation.reconciled if self.broker is not None else None,
                risk_halted=self.risk_manager.halted,
                risk_halt_reason=self.risk_manager.halt_reason,
            )
            result.safety = safety
            if not safety.passed:
                result.blocked_reason = "; ".join(f"{c.name}: {c.message}" for c in safety.failures)
                self._log.error(
                    "execution cycle blocked", extra={"context": {"reason": result.blocked_reason}}
                )
                result.finished_at = datetime.now(UTC)
                return result

            # --- steps 5-7: targets, trades, risk ---
            equity = (
                result.account.net_liquidation
                if result.account is not None
                else self.config.portfolio.initial_capital
            )
            plan, allocation = self.build_plan(
                panel,
                features=features,
                as_of=reference,
                equity=equity,
                current_shares=pd.Series(local_positions, dtype="float64"),
                prices=prices,
            )
            result.plan = plan
            result.allocation = allocation

            # --- step 8: preview ---
            self._log.info("order preview\n%s", plan.render())

            # --- step 9: submit ---
            if may_submit and plan.orders:
                submitted, rejected, fills = self._submit(plan)
                result.submitted, result.rejected, result.fills = submitted, rejected, fills
                # Record what Atlas now believes it holds, so the next cycle has
                # something to reconcile the broker against.
                if self.broker is not None:
                    self._persist_positions(
                        {p.symbol: p.quantity for p in self.broker.get_positions()}, reference
                    )
            elif plan.orders:
                self._log.info(
                    "dry run: orders were not submitted",
                    extra={"context": {"n_orders": plan.n_orders}},
                )

            result.risk_events = self.risk_manager.events_frame()
            if self.repository is not None and not result.risk_events.empty:
                self.repository.save_risk_events(result.risk_events, session_id=self.session_id)

        except ExecutionBlocked as exc:
            result.blocked_reason = str(exc)
            self._log.error("execution blocked", extra={"context": {"error": str(exc)}})
        finally:
            result.finished_at = datetime.now(UTC)

        self._log.info("execution cycle complete", extra={"context": result.summary()})
        return result

    # -- planning -------------------------------------------------------------

    def build_plan(
        self,
        panel: PricePanel,
        *,
        features: FeatureSet | None = None,
        as_of: pd.Timestamp | None = None,
        equity: float | None = None,
        current_shares: pd.Series | None = None,
        prices: pd.Series | None = None,
    ) -> tuple[TradePlan, AllocationResult | None]:
        """Compute target positions and the trades required to reach them."""
        cfg = self.config
        reference = pd.Timestamp(as_of) if as_of is not None else panel.dates.max()
        panel = panel.as_of(reference)
        features = (
            features.as_of(reference).select(panel.symbols)
            if features is not None
            else FeatureEngineer.from_config(cfg).build(panel)
        )
        equity = float(equity if equity is not None else cfg.portfolio.initial_capital)
        current = (
            current_shares.copy()
            if current_shares is not None
            else pd.Series(0.0, index=panel.symbols)
        )
        marks = prices if prices is not None else self._current_prices(panel, reference)

        # --- signals and regime ---
        regimes = (
            self.regime_detector.detect(panel, features) if cfg.strategies.regime.enabled else None
        )
        signal_results = {s.name: s.compute(panel.adj_close, features) for s in self.strategies}
        combined = self.ensemble.combine(signal_results, regimes=regimes)
        signals = combined.last()
        regime_label = (
            regimes.label_at(reference) if regimes is not None else RegimeLabel.UNKNOWN
        )

        # --- eligibility ---
        history = panel.adj_close.notna().sum()
        tradable = [
            s
            for s in panel.symbols
            if int(history.get(s, 0)) >= cfg.data.min_history_days
            and np.isfinite(marks.get(s, np.nan))
        ]

        # --- risk posture ---
        self.risk_manager.update_equity(reference, equity)
        dd_multiplier, _ = self.risk_manager.drawdown_multiplier(equity)
        weak_breadth = bool(
            "breadth" in features.market.columns
            and len(features.market)
            and features.market["breadth"].iloc[-1] < cfg.regimes.detector.weak_breadth_threshold
        )
        risk_multiplier, defensive_weight = self.allocator.risk_posture(
            regime_label, drawdown_multiplier=dd_multiplier, weak_breadth=weak_breadth
        )

        current_weights = pd.Series(
            {
                s: float(current.get(s, 0.0)) * float(marks.get(s, 0.0)) / equity
                for s in tradable
            },
            dtype="float64",
        ).fillna(0.0)

        allocation = self.allocator.allocate(
            signals.reindex(tradable).fillna(0.0),
            panel.returns("adj_close"),
            previous_weights=current_weights,
            risk_multiplier=risk_multiplier,
            defensive_weight=defensive_weight,
            tradable=tradable,
        )

        assessment = self.risk_manager.assess(
            reference,
            allocation.weights,
            equity=equity,
            previous_weights=current_weights,
            covariance=allocation.covariance.matrix if allocation.covariance else None,
            last_data_date=panel.dates.max() if len(panel.dates) else None,
            prices=marks,
        )
        targets = assessment.weights if assessment.approved else current_weights
        if not assessment.approved:
            self._log.error(
                "risk manager blocked the target portfolio; holding current positions",
                extra={"context": {"reason": assessment.blocked_reason}},
            )

        # --- required trades ---
        orders, skipped, target_shares = self._orders_from_targets(
            targets, current, marks, equity, reference
        )
        plan = TradePlan(
            as_of=reference,
            target_weights=targets,
            target_shares=target_shares,
            current_shares=current,
            prices=marks,
            orders=orders,
            skipped=skipped,
            equity=equity,
            regime=str(getattr(regime_label, "value", regime_label)),
        )
        return plan, allocation

    def _orders_from_targets(
        self,
        targets: pd.Series,
        current: pd.Series,
        prices: pd.Series,
        equity: float,
        reference: pd.Timestamp,
    ) -> tuple[list[Order], list[dict[str, Any]], pd.Series]:
        """Convert target weights into orders, recording why any were skipped."""
        cfg = self.config
        defaults = cfg.execution.order_defaults
        orders: list[Order] = []
        skipped: list[dict[str, Any]] = []
        target_shares: dict[str, float] = {}
        buying_power = equity * cfg.portfolio.max_leverage

        for raw_symbol, weight in targets.items():
            symbol = str(raw_symbol)
            price = prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                skipped.append({"symbol": symbol, "reason": "no current market price"})
                continue

            desired = float(weight) * equity / float(price)
            if not cfg.backtest.fractional_shares:
                desired = float(np.trunc(desired))
            target_shares[symbol] = desired

            delta = desired - float(current.get(symbol, 0.0))
            if abs(delta) < 1e-6:
                continue

            notional = abs(delta * float(price))
            if notional < cfg.risk.limits.min_order_notional:
                skipped.append(
                    {
                        "symbol": symbol,
                        "reason": f"trade of ${notional:,.2f} is below the "
                        f"${cfg.risk.limits.min_order_notional:,.2f} minimum",
                    }
                )
                continue

            limit_price = None
            if defaults.order_type == "LMT":
                offset = defaults.limit_offset_bps / 10_000.0
                limit_price = float(price) * (1.0 + offset if delta > 0 else 1.0 - offset)

            order = Order.from_signed_quantity(
                symbol,
                delta,
                order_type=OrderType(defaults.order_type),
                time_in_force=TimeInForce(defaults.time_in_force),
                limit_price=limit_price,
                reference_price=float(price),
                target_weight=float(weight),
                metadata={"session_id": self.session_id},
            )

            allowed, reason = self.risk_manager.check_order(
                reference,
                symbol,
                order.signed_quantity,
                float(price),
                buying_power=buying_power,
                enforce_trading_window=self.gate.may_submit_orders,
            )
            if not allowed:
                order.reject(reason)
                skipped.append({"symbol": symbol, "reason": reason})
                continue
            orders.append(order)

        return orders, skipped, pd.Series(target_shares, dtype="float64")

    # -- submission -----------------------------------------------------------

    def _submit(self, plan: TradePlan) -> tuple[list[Order], list[Order], list[Fill]]:
        """Submit the plan's orders, sells first."""
        if self.broker is None:
            raise ExecutionBlocked("cannot submit orders without a broker connection")
        self.gate.assert_not_live()

        submitted: list[Order] = []
        rejected: list[Order] = []
        ordered = sorted(plan.orders, key=lambda o: 0 if o.side is OrderSide.SELL else 1)

        for order in ordered:
            try:
                self.broker.submit_order(order)
                submitted.append(order)
            except OrderRejected as exc:
                rejected.append(order)
                self._log.error(
                    "order rejected",
                    extra={
                        "context": {
                            "symbol": order.symbol,
                            "order_id": order.order_id,
                            "reason": str(exc),
                        }
                    },
                )
                # A submission failure is never retried; see ibkr_client.
                continue
            finally:
                if self.repository is not None:
                    self.repository.save_order(
                        order,
                        session_id=self.session_id,
                        execution_mode=self.config.execution.mode.value,
                    )

        fills = self.broker.get_fills(since=plan.as_of.to_pydatetime())
        if self.repository is not None:
            for fill in fills:
                self.repository.save_fill(fill, session_id=self.session_id)
        return submitted, rejected, fills

    # -- helpers --------------------------------------------------------------

    def _local_positions(self) -> dict[str, float]:
        """Atlas's own view of current positions.

        Read from the most recent persisted position snapshot. A fresh
        installation legitimately has none, and an empty record is *not* quietly
        treated as agreement: reconciliation reports any broker position Atlas
        does not know about as a material mismatch, which blocks trading until a
        human confirms it. That is the intended behaviour when the system is
        first pointed at an account that already holds something.
        """
        if self.repository is None:
            return {}
        try:
            return self.repository.latest_positions(source="local")
        except Exception as exc:
            # A persistence failure must not silently become "no positions",
            # which would look like agreement with an empty broker account.
            self._log.error(
                "could not read stored positions; treating the local record as unknown",
                extra={"context": {"error": str(exc)}},
            )
            return {}

    def _persist_positions(
        self, positions: dict[str, float], as_of: pd.Timestamp
    ) -> None:
        """Store Atlas's post-cycle view of its positions for the next cycle."""
        if self.repository is None or not positions:
            return
        frame = pd.DataFrame(
            [{"symbol": symbol, "quantity": quantity} for symbol, quantity in positions.items()]
        )
        try:
            self.repository.save_positions(
                frame, session_id=self.session_id, as_of=as_of, source="local"
            )
        except Exception as exc:
            self._log.warning(
                "could not persist positions", extra={"context": {"error": str(exc)}}
            )

    def _current_prices(self, panel: PricePanel, reference: pd.Timestamp) -> pd.Series:
        """Prices for the cycle: live quotes when connected, else the last close."""
        if self.broker is not None and self.broker.connected:
            try:
                quotes = self.broker.get_market_prices(list(panel.symbols))
                if quotes.notna().any():
                    # Fill any gaps from the last close rather than dropping the symbol.
                    closes = panel.adj_close.loc[:reference].ffill().iloc[-1]
                    return quotes.fillna(closes)
            except Exception as exc:
                self._log.warning(
                    "live prices unavailable; falling back to the last close",
                    extra={"context": {"error": str(exc)}},
                )
                self.risk_manager.record_api_failure("get_market_prices")
        frame = panel.adj_close.loc[:reference]
        if frame.empty:
            return pd.Series(dtype="float64")
        return frame.ffill().iloc[-1]

    def _data_is_fresh(
        self, last_data_date: pd.Timestamp | None, reference: pd.Timestamp | None
    ) -> bool:
        """True when the newest observation is inside the staleness limit."""
        if last_data_date is None or reference is None:
            return False
        age = (pd.Timestamp(reference) - pd.Timestamp(last_data_date)).days
        return age <= self.config.risk.limits.max_data_staleness_days
