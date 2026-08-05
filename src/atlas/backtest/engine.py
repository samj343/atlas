"""Event-driven backtesting engine.

The engine advances one trading day at a time and processes a fixed causal
sequence of events. Nothing is vectorised across time, so a bug that reaches
forward in time has nowhere to hide.

Per bar *t*:

1. **MarketEvent** - the bar's data becomes visible. Data after *t* is never
   passed to any component.
2. Pending orders created on *t-1* are executed at *t*'s **open** (the default),
   producing **FillEvents**. Cash and positions update.
3. Dividends are credited (when running on raw prices).
4. Positions are marked to *t*'s **close**; a **PortfolioEvent** is recorded.
5. On a rebalance date: a **SignalEvent** is generated from data through *t*'s
   close, the allocator produces target weights (**RebalanceEvent**), the risk
   manager vets them (**RiskEvent**), and the resulting **OrderEvents** are
   queued for execution on *t+1*.

That ordering is the whole point: **a signal computed from the close of day t is
never executed at the close of day t**. The only way to get that behaviour is to
explicitly select ``execution_timing='same_close_unrealistic'``, which the
engine logs as a warning on every run and marks in the results.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.broker import ExecutionContext, SimulatedBroker
from atlas.backtest.costs import TransactionCostModel
from atlas.backtest.metrics import PerformanceMetrics, compute_metrics
from atlas.backtest.orders import (
    Event,
    FillEvent,
    MarketEvent,
    Order,
    PortfolioEvent,
    RebalanceEvent,
    RiskEventRecord,
    SignalEvent,
)
from atlas.backtest.positions import Portfolio
from atlas.config import (
    AtlasConfig,
    ExecutionTiming,
    RebalanceFrequency,
    RegimeLabel,
)
from atlas.data.features import FeatureEngineer, FeatureSet
from atlas.data.loader import PricePanel
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger
from atlas.portfolio.allocator import PortfolioAllocator
from atlas.portfolio.risk_manager import RiskManager
from atlas.regimes.base import RegimeResult
from atlas.regimes.rules_based import RulesBasedRegimeDetector
from atlas.strategies.base import Strategy, build_strategies
from atlas.strategies.ensemble import CombinedSignal, SignalEnsemble

log = get_logger(__name__)

__all__ = ["BacktestEngine", "BacktestResult"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Everything a backtest produced.

    Attributes
    ----------
    run_id:
        Unique identifier for this run.
    equity_curve:
        Portfolio equity indexed by date.
    returns:
        Daily portfolio returns net of costs.
    snapshots:
        Daily portfolio state (cash, exposures, turnover, costs).
    weights:
        Realised position weights over time (date x symbol).
    target_weights:
        Requested target weights on rebalance dates.
    fills, orders, rejected_orders, risk_events:
        Execution and risk audit trails.
    metrics:
        Computed performance metrics.
    """

    run_id: str
    config_snapshot: dict[str, Any]
    equity_curve: pd.Series
    returns: pd.Series
    snapshots: pd.DataFrame
    weights: pd.DataFrame
    target_weights: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    rejected_orders: pd.DataFrame
    risk_events: pd.DataFrame
    signals: dict[str, pd.DataFrame] = field(default_factory=dict)
    strategy_contributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    regimes: RegimeResult | None = None
    metrics: PerformanceMetrics | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    git_commit: str | None = None
    data_source: str = "unknown"
    is_synthetic_data: bool = False
    warnings: list[str] = field(default_factory=list)

    # -- derived ---------------------------------------------------------------

    @property
    def initial_capital(self) -> float:
        """Starting equity."""
        return float(self.equity_curve.iloc[0]) if len(self.equity_curve) else 0.0

    @property
    def final_equity(self) -> float:
        """Ending equity."""
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else 0.0

    @property
    def total_costs(self) -> float:
        """Total transaction costs charged."""
        if self.fills.empty or "total_cost" not in self.fills.columns:
            return 0.0
        return float(self.fills["total_cost"].sum())

    @property
    def n_trades(self) -> int:
        """Number of fills."""
        return len(self.fills)

    def cost_breakdown(self) -> pd.Series:
        """Transaction costs by category."""
        columns = ["commission", "spread_cost", "slippage_cost", "impact_cost"]
        if self.fills.empty:
            return pd.Series(dict.fromkeys(columns, 0.0))
        available = [c for c in columns if c in self.fills.columns]
        return self.fills[available].sum()

    def cost_drag(self) -> dict[str, float]:
        """Estimate the performance given up to trading costs.

        The gross return series adds each day's costs back to that day's P&L,
        expressed as a fraction of the previous day's equity, so gross and net
        are compared on the same capital base.
        """
        if self.fills.empty or self.equity_curve.empty:
            return {"gross_cagr": self.metrics.cagr if self.metrics else 0.0, "net_cagr": 0.0, "cost_drag_annual": 0.0}
        daily_costs = (
            self.fills.groupby(pd.DatetimeIndex(self.fills["timestamp"]).normalize())["total_cost"]
            .sum()
            .reindex(self.equity_curve.index)
            .fillna(0.0)
        )
        base = self.equity_curve.shift(1)
        gross_returns = (self.returns + (daily_costs / base).fillna(0.0)).dropna()
        from atlas.backtest.metrics import annualized_return

        gross = annualized_return(gross_returns)
        net = annualized_return(self.returns)
        gross_total = float((1.0 + gross_returns).prod() - 1.0)
        net_total = float((1.0 + self.returns.dropna()).prod() - 1.0)
        return {
            "gross_cagr": gross,
            "net_cagr": net,
            "cost_drag_annual": gross - net,
            "gross_total_return": gross_total,
            "net_total_return": net_total,
            "cost_share_of_gross_return": (
                (gross_total - net_total) / abs(gross_total) if abs(gross_total) > 1e-9 else float("nan")
            ),
            "total_costs": self.total_costs,
        }

    def turnover_series(self) -> pd.Series:
        """Daily one-way turnover as a fraction of equity."""
        if self.snapshots.empty or "turnover" not in self.snapshots.columns:
            return pd.Series(dtype="float64")
        return self.snapshots["turnover"]

    def strategy_contribution_summary(self) -> pd.Series:
        """Average share of the combined signal contributed by each strategy."""
        if self.strategy_contributions.empty:
            return pd.Series(dtype="float64")
        return self.strategy_contributions.mean()

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        out: dict[str, Any] = {
            "run_id": self.run_id,
            "start": str(self.equity_curve.index.min().date()) if len(self.equity_curve) else None,
            "end": str(self.equity_curve.index.max().date()) if len(self.equity_curve) else None,
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "n_trades": self.n_trades,
            "total_costs": round(self.total_costs, 2),
            "data_source": self.data_source,
            "synthetic_data": self.is_synthetic_data,
        }
        if self.metrics:
            out.update(
                {
                    "total_return": round(self.metrics.values.get("total_return", 0.0), 4),
                    "cagr": round(self.metrics.cagr, 4),
                    "volatility": round(self.metrics.values.get("annualized_volatility", 0.0), 4),
                    "sharpe": round(self.metrics.sharpe, 3),
                    "max_drawdown": round(self.metrics.max_drawdown, 4),
                }
            )
        return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Run an event-driven backtest of the Atlas system.

    Parameters
    ----------
    config:
        The full :class:`~atlas.config.AtlasConfig`.
    strategies:
        Strategy instances; built from configuration when omitted.
    regime_detector:
        Regime detector; a rules-based one is built when omitted.
    """

    def __init__(
        self,
        config: AtlasConfig,
        *,
        strategies: list[Strategy] | None = None,
        regime_detector: RulesBasedRegimeDetector | None = None,
    ) -> None:
        self.config = config
        self.strategies = strategies if strategies is not None else build_strategies(config)
        self.ensemble = SignalEnsemble(
            self.strategies,
            config.strategies.ensemble,
            regime_config=config.regimes if config.strategies.regime.enabled else None,
            base_weights=config.strategies.strategies.base_weights(),
        )
        self.regime_detector = regime_detector or RulesBasedRegimeDetector(config.regimes)
        self.allocator = PortfolioAllocator(config)
        self.risk_manager = RiskManager(config)
        self.cost_model = TransactionCostModel(config.costs)
        self.broker = SimulatedBroker(
            self.cost_model,
            config.execution.fills,
            fractional_shares=config.backtest.fractional_shares,
            timing=config.backtest.execution_timing,
        )
        self.events: list[Event] = []
        self._rng = np.random.default_rng(config.backtest.random_seed)

    # -- public API -----------------------------------------------------------

    def run(
        self,
        panel: PricePanel,
        *,
        features: FeatureSet | None = None,
        start: pd.Timestamp | date | str | None = None,
        end: pd.Timestamp | date | str | None = None,
        record_events: bool = False,
        progress: bool = False,
    ) -> BacktestResult:
        """Run the backtest over ``panel``.

        Parameters
        ----------
        panel:
            Price history for the universe.
        features:
            Precomputed features; built from ``panel`` when omitted.
        start, end:
            Restrict the *trading* window. History before ``start`` is still used
            for warm-up, which is why the warm-up is not a source of look-ahead:
            it only ever looks backwards.
        record_events:
            Retain every event object. Useful for debugging, memory-hungry for
            long runs.
        progress:
            Log progress every 250 bars.
        """
        cfg = self.config
        bt = cfg.backtest

        if bt.execution_timing is ExecutionTiming.SAME_CLOSE_UNREALISTIC:
            log.warning(
                "UNREALISTIC EXECUTION MODE: orders fill at the same close that produced the "
                "signal. Results are not achievable in practice and are for research "
                "comparison only.",
            )

        panel = panel.select([s for s in cfg.symbols if s in panel.symbols])
        if not panel.symbols:
            raise InsufficientDataError("none of the configured symbols are present in the panel")

        if features is None:
            features = FeatureEngineer.from_config(cfg).build(panel)
        else:
            # A caller-supplied feature set may have been built from a panel with
            # a different column order (the loader sorts symbols, the config does
            # not). Realign it so every frame shares the panel's exact columns.
            features = features.select(panel.symbols)
        regimes = (
            self.regime_detector.detect(panel, features)
            if cfg.strategies.regime.enabled
            else None
        )

        # Precompute per-strategy signals over the whole panel. Every underlying
        # feature is backward-looking, so a value at t equals what would have
        # been computed on t in real time; `tests/unit/test_lookahead.py` asserts
        # this by perturbing future prices.
        signal_results = {
            s.name: s.compute(panel.adj_close, features) for s in self.strategies
        }
        combined = self.ensemble.combine(signal_results, regimes=regimes)

        trading_dates = self._trading_dates(panel, start, end)
        if len(trading_dates) < 2:
            raise InsufficientDataError(
                f"backtest needs at least 2 trading days, got {len(trading_dates)}"
            )
        rebalance_dates = set(self._rebalance_dates(trading_dates))

        return self._loop(
            panel=panel,
            features=features,
            combined=combined,
            signal_results=signal_results,
            regimes=regimes,
            trading_dates=trading_dates,
            rebalance_dates=rebalance_dates,
            record_events=record_events,
            progress=progress,
        )

    # -- the event loop -------------------------------------------------------

    def _loop(
        self,
        *,
        panel: PricePanel,
        features: FeatureSet,
        combined: CombinedSignal,
        signal_results: dict[str, Any],
        regimes: RegimeResult | None,
        trading_dates: pd.DatetimeIndex,
        rebalance_dates: set[pd.Timestamp],
        record_events: bool,
        progress: bool,
    ) -> BacktestResult:
        cfg = self.config
        bt = cfg.backtest
        run_id = f"bt-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        started = datetime.now(UTC)

        portfolio = Portfolio(bt.initial_capital, allow_negative_cash=True)
        self.risk_manager.reset()

        open_prices = panel.open
        close_prices = panel.close if bt.price_field == "close" else panel.adj_close
        returns = panel.returns("adj_close")
        adv = panel.volume.rolling(cfg.costs.adv_lookback, min_periods=1).mean()
        daily_vol = returns.rolling(21, min_periods=5).std(ddof=1)
        dividends = panel.implied_dividends() if bt.price_field == "close" else None

        pending_orders: list[Order] = []
        all_orders: list[Order] = []
        target_weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
        events: list[Event] = []
        prev_target = pd.Series(0.0, index=panel.symbols)
        min_history = max(1, cfg.data.min_history_days)

        # Precomputed once rather than re-derived on every rebalance: the number
        # of observations available for each symbol up to and including each
        # date. Slicing `notna().sum()` per rebalance is quadratic in run length.
        history_counts = panel.adj_close.notna().cumsum()
        # Longest trailing window any estimator needs. Passing only this slice to
        # the allocator keeps each rebalance O(window) instead of O(elapsed run).
        estimation_window = max(
            cfg.risk.covariance.lookback,
            cfg.risk.volatility.lookback,
            cfg.risk.volatility.halflife * 8,
            cfg.risk.covariance.ewma_halflife * 8,
        ) + 20

        for i, today in enumerate(trading_dates):
            # ---- 1. market event -------------------------------------------
            market = MarketEvent(
                today, list(panel.symbols), is_rebalance_date=today in rebalance_dates
            )
            if record_events:
                events.append(market)

            # ---- 2. execute orders queued on the previous bar ---------------
            turnover_today = 0.0
            if pending_orders:
                exec_prices = self._execution_prices(
                    today, open_prices, close_prices, timing=bt.execution_timing
                )
                context = ExecutionContext(
                    timestamp=today,
                    prices=exec_prices,
                    volumes=adv.loc[today] if today in adv.index else None,
                    volatilities=daily_vol.loc[today] if today in daily_vol.index else None,
                )
                equity_before = portfolio.equity(exec_prices)
                fills, rejected = self.broker.execute_batch(pending_orders, context)
                for fill in fills:
                    portfolio.apply_fill(fill)
                    turnover_today += fill.notional
                    if record_events:
                        events.append(FillEvent(today, fill))
                for order in rejected:
                    log.debug(
                        "order rejected by the simulated broker",
                        extra={"context": {"symbol": order.symbol, "reason": order.reject_reason}},
                    )
                if equity_before > 0:
                    turnover_today /= equity_before
                pending_orders = []

            # ---- 3. dividends ----------------------------------------------
            if dividends is not None and today in dividends.index:
                portfolio.credit_dividends(dividends.loc[today], today)

            # ---- 4. mark to market ------------------------------------------
            marks = close_prices.loc[today]
            snapshot = portfolio.snapshot(today, marks, turnover=turnover_today)
            self.risk_manager.update_equity(today, snapshot.equity)
            if record_events:
                events.append(
                    PortfolioEvent(today, snapshot.equity, snapshot.cash, snapshot.positions_value)
                )

            # ---- 5. rebalance -----------------------------------------------
            is_last_bar = i >= len(trading_dates) - 1
            if today in rebalance_dates and not is_last_bar:
                orders = self._rebalance(
                    today=today,
                    panel=panel,
                    features=features,
                    combined=combined,
                    regimes=regimes,
                    returns=returns,
                    marks=marks,
                    portfolio=portfolio,
                    prev_target=prev_target,
                    min_history=min_history,
                    history_counts=history_counts,
                    estimation_window=estimation_window,
                    target_weight_rows=target_weight_rows,
                    events=events if record_events else None,
                )
                pending_orders = orders
                all_orders.extend(orders)

            if progress and i % 250 == 0 and i:
                log.info(
                    "backtest progress",
                    extra={
                        "context": {
                            "date": str(today.date()),
                            "bar": f"{i}/{len(trading_dates)}",
                            "equity": round(snapshot.equity, 2),
                        }
                    },
                )

        return self._build_result(
            run_id=run_id,
            started=started,
            portfolio=portfolio,
            panel=panel,
            combined=combined,
            signal_results=signal_results,
            regimes=regimes,
            all_orders=all_orders,
            target_weight_rows=target_weight_rows,
            events=events,
        )

    # -- rebalance ------------------------------------------------------------

    def _rebalance(
        self,
        *,
        today: pd.Timestamp,
        panel: PricePanel,
        features: FeatureSet,
        combined: CombinedSignal,
        regimes: RegimeResult | None,
        returns: pd.DataFrame,
        marks: pd.Series,
        portfolio: Portfolio,
        prev_target: pd.Series,
        min_history: int,
        history_counts: pd.DataFrame,
        estimation_window: int,
        target_weight_rows: dict[pd.Timestamp, dict[str, float]],
        events: list[Event] | None,
    ) -> list[Order]:
        """Produce the orders for one rebalance date."""
        cfg = self.config

        if today not in combined.signal.index:
            return []
        signals = combined.signal.loc[today]

        # --- eligibility: live price plus enough history ---
        history = history_counts.loc[today]
        tradable = [
            s
            for s in panel.symbols
            if int(history.get(s, 0)) >= min_history and np.isfinite(marks.get(s, np.nan))
        ]
        if not tradable:
            log.warning("no tradable assets", extra={"context": {"date": str(today.date())}})
            return []

        regime_label = regimes.label_at(today) if regimes is not None else RegimeLabel.UNKNOWN
        if events is not None:
            events.append(SignalEvent(today, signals, regime=str(getattr(regime_label, "value", regime_label))))

        # --- risk posture from the regime, the drawdown ladder and breadth ---
        equity = portfolio.equity(marks)
        dd_multiplier, _ = self.risk_manager.drawdown_multiplier(equity)
        weak_breadth = bool(
            "breadth" in features.market.columns
            and today in features.market.index
            and features.market.loc[today, "breadth"]
            < cfg.regimes.detector.weak_breadth_threshold
        )
        risk_multiplier, defensive_weight = self.allocator.risk_posture(
            regime_label, drawdown_multiplier=dd_multiplier, weak_breadth=weak_breadth
        )

        # --- allocate: strictly historical returns only ---
        # `.loc[:today]` is the look-ahead guard; `.tail()` is a performance
        # bound. Trimming to the longest window any estimator uses cannot change
        # the estimates, only the time taken to compute them.
        history_returns = returns.loc[:today].tail(estimation_window)
        current_weights = portfolio.weights(marks).reindex(tradable).fillna(0.0)
        allocation = self.allocator.allocate(
            signals.reindex(tradable),
            history_returns,
            previous_weights=current_weights,
            risk_multiplier=risk_multiplier,
            defensive_weight=defensive_weight,
            tradable=tradable,
        )

        # --- risk assessment ---
        assessment = self.risk_manager.assess(
            today,
            allocation.weights,
            equity=equity,
            previous_weights=current_weights,
            covariance=allocation.covariance.matrix if allocation.covariance else None,
            last_data_date=today,
            prices=marks,
        )
        if events is not None:
            for event in assessment.events:
                events.append(
                    RiskEventRecord(today, event.control, event.action.value, event.message)
                )
        if not assessment.approved:
            log.warning(
                "rebalance blocked by the risk manager",
                extra={"context": {"date": str(today.date()), "reason": assessment.blocked_reason}},
            )
            return []

        targets = assessment.weights
        target_weight_rows[today] = targets.to_dict()
        prev_target.loc[targets.index] = targets.to_numpy()
        if events is not None:
            events.append(RebalanceEvent(today, targets, risk_multiplier=assessment.risk_multiplier))

        return self._orders_from_targets(today, targets, portfolio, marks, equity)

    def _orders_from_targets(
        self,
        today: pd.Timestamp,
        targets: pd.Series,
        portfolio: Portfolio,
        marks: pd.Series,
        equity: float,
    ) -> list[Order]:
        """Convert target weights into orders.

        The reference price is today's close (what is known now); the order will
        actually fill at tomorrow's price, and that difference is precisely the
        execution risk the backtest is meant to include.
        """
        cfg = self.config
        orders: list[Order] = []
        min_notional = cfg.risk.limits.min_order_notional
        max_notional = cfg.risk.limits.max_order_notional

        for symbol, target_weight in targets.items():
            price = marks.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            target_value = float(target_weight) * equity
            target_shares = target_value / float(price)
            if not cfg.backtest.fractional_shares:
                target_shares = float(np.trunc(target_shares))
            delta = target_shares - portfolio.quantity(symbol)
            if abs(delta) < 1e-9:
                continue

            notional = abs(delta * float(price))
            if notional < min_notional:
                continue
            if notional > max_notional:
                # Trim rather than reject: partial progress toward the target is
                # better than none, and the remainder trades at the next rebalance.
                delta = np.sign(delta) * (max_notional / float(price))

            order = Order.from_signed_quantity(
                symbol,
                delta,
                reference_price=float(price),
                target_weight=float(target_weight),
                metadata={"rebalance_date": str(today.date())},
            )
            allowed, reason = self.risk_manager.check_order(
                today, symbol, order.signed_quantity, float(price)
            )
            if not allowed:
                order.reject(reason)
                self.broker.rejected_orders.append(order)
                log.debug(
                    "order blocked pre-trade",
                    extra={"context": {"symbol": symbol, "reason": reason}},
                )
                continue
            orders.append(order)

        return orders

    # -- scheduling -----------------------------------------------------------

    def _trading_dates(
        self, panel: PricePanel, start: Any, end: Any
    ) -> pd.DatetimeIndex:
        """Trading dates in the run window, after the warm-up."""
        bt = self.config.backtest
        dates = panel.dates
        window_start = start if start is not None else bt.start_date
        window_end = end if end is not None else bt.end_date

        if window_start is not None:
            dates = dates[dates >= pd.Timestamp(window_start)]
        if window_end is not None:
            dates = dates[dates <= pd.Timestamp(window_end)]

        # Ensure enough history precedes the first trading day.
        warmup = max(bt.warmup_days, self.config.strategies.max_lookback)
        full = panel.dates
        if len(dates) and warmup > 0:
            first_allowed_position = min(warmup, max(len(full) - 2, 0))
            first_allowed = full[first_allowed_position]
            dates = dates[dates >= first_allowed]
        return pd.DatetimeIndex(dates)

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Dates on which a rebalance is scheduled."""
        bt = self.config.backtest
        if len(dates) == 0:
            return []
        frequency = bt.rebalance_frequency

        if frequency is RebalanceFrequency.DAILY:
            return list(dates)
        if frequency is RebalanceFrequency.WEEKLY:
            frame = pd.DataFrame(index=dates)
            frame["week"] = dates.isocalendar().week.to_numpy()
            frame["year"] = dates.isocalendar().year.to_numpy()
            frame["weekday"] = dates.weekday
            chosen: list[pd.Timestamp] = []
            for _, group in frame.groupby(["year", "week"], sort=True):
                # The configured weekday, or the last available day of that week
                # when the market was closed on it (a holiday).
                exact = group.index[group["weekday"] == bt.rebalance_weekday]
                chosen.append(exact[0] if len(exact) else group.index[-1])
            return sorted(chosen)
        # monthly: the last trading day of each month
        frame = pd.DataFrame(index=dates)
        frame["period"] = dates.to_period("M")
        return sorted(group.index[-1] for _, group in frame.groupby("period", sort=True))

    @staticmethod
    def _execution_prices(
        today: pd.Timestamp,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        *,
        timing: ExecutionTiming,
    ) -> pd.Series:
        """Select the execution price series for the bar.

        ``next_open`` and ``same_close_unrealistic`` both resolve to a price on
        the *current* bar here, because the caller has already advanced a day for
        the realistic modes - orders queued on t-1 execute during bar t.
        """
        if timing is ExecutionTiming.NEXT_OPEN:
            prices = open_prices.loc[today].copy()
            # Fall back to the close for symbols with no open price that day.
            missing = prices.isna()
            if missing.any():
                prices[missing] = close_prices.loc[today][missing]
            return prices
        return close_prices.loc[today].copy()

    # -- result assembly ------------------------------------------------------

    def _build_result(
        self,
        *,
        run_id: str,
        started: datetime,
        portfolio: Portfolio,
        panel: PricePanel,
        combined: CombinedSignal,
        signal_results: dict[str, Any],
        regimes: RegimeResult | None,
        all_orders: list[Order],
        target_weight_rows: dict[pd.Timestamp, dict[str, float]],
        events: list[Event],
    ) -> BacktestResult:
        """Assemble the :class:`BacktestResult`."""
        cfg = self.config
        equity = portfolio.equity_curve()
        returns = equity.pct_change(fill_method=None).dropna()
        snapshots = portfolio.snapshot_frame()
        weights = portfolio.weight_history()

        target_frame = (
            pd.DataFrame.from_dict(target_weight_rows, orient="index").sort_index().fillna(0.0)
            if target_weight_rows
            else pd.DataFrame()
        )
        fills_frame = portfolio.fills_frame()

        holding_periods = self._holding_periods(portfolio)
        metrics = compute_metrics(
            returns,
            equity=equity,
            risk_free_rate=cfg.validation.benchmarks.risk_free_rate,
            label="Atlas",
            turnover=snapshots["turnover"] if "turnover" in snapshots.columns else None,
            costs=portfolio.cumulative_costs,
            gross_exposure=snapshots.get("gross_exposure"),
            net_exposure=snapshots.get("net_exposure"),
            cash_weight=snapshots.get("cash_weight"),
            n_trades=len(portfolio.fills),
            trade_values=fills_frame["notional"] if "notional" in fills_frame.columns else None,
            holding_periods=holding_periods,
        )

        warnings: list[str] = []
        if panel.is_synthetic:
            warnings.append(
                "Results were computed from SYNTHETIC simulated prices, not market data. "
                "They demonstrate the machinery only and carry no information about real performance."
            )
        if cfg.backtest.execution_timing is ExecutionTiming.SAME_CLOSE_UNREALISTIC:
            warnings.append(
                "Execution timing is 'same_close_unrealistic': orders fill at the same close that "
                "produced the signal. These results are not achievable in practice."
            )

        result = BacktestResult(
            run_id=run_id,
            config_snapshot=cfg.snapshot(),
            equity_curve=equity,
            returns=returns,
            snapshots=snapshots,
            weights=weights,
            target_weights=target_frame,
            fills=fills_frame,
            orders=pd.DataFrame([o.as_dict() for o in all_orders])
            if all_orders
            else pd.DataFrame(),
            rejected_orders=self.broker.rejected_frame(),
            risk_events=self.risk_manager.events_frame(),
            signals={name: r.signal for name, r in signal_results.items()},
            strategy_contributions=combined.contribution_share(),
            regimes=regimes,
            metrics=metrics,
            started_at=started,
            finished_at=datetime.now(UTC),
            git_commit=_git_commit(),
            data_source=next(iter(panel.source.values()), "unknown"),
            is_synthetic_data=panel.is_synthetic,
            warnings=warnings,
        )
        self.events = events
        log.info("backtest complete", extra={"context": result.summary()})
        for warning in warnings:
            log.warning(warning)
        return result

    @staticmethod
    def _holding_periods(portfolio: Portfolio) -> pd.Series:
        """Approximate holding period per round trip, in calendar days.

        Pairs each opening fill with the fill that later reduced the position to
        zero, per symbol. Positions still open at the end of the run are excluded.
        """
        by_symbol: dict[str, list] = {}
        for fill in portfolio.fills:
            by_symbol.setdefault(fill.symbol, []).append(fill)

        durations: list[float] = []
        for fills in by_symbol.values():
            position = 0.0
            opened_at: pd.Timestamp | None = None
            for fill in fills:
                before = position
                position += fill.signed_quantity
                if abs(before) <= 1e-9 and abs(position) > 1e-9:
                    opened_at = fill.timestamp
                elif abs(position) <= 1e-9 and opened_at is not None:
                    durations.append(float((fill.timestamp - opened_at).days))
                    opened_at = None
        return pd.Series(durations, dtype="float64")


def _git_commit() -> str | None:
    """Return the current git commit hash, when available.

    Recorded with each run so a result can always be traced back to the code
    that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None

    candidate = out.stdout.strip()
    # In a repository with no commits `git rev-parse HEAD` fails and echoes the
    # literal string "HEAD"; recording that as a commit would be worse than
    # recording nothing.
    if out.returncode != 0 or len(candidate) != 40 or not all(c in "0123456789abcdef" for c in candidate):
        return None
    return candidate
