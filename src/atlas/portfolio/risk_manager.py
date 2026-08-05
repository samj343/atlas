"""The risk-management engine.

The risk manager is deliberately **independent of strategy logic**. It receives
a proposed portfolio state and either passes it, scales it, or blocks it - and
records a :class:`RiskEvent` for every decision. Nothing in this module knows
what a moving average is, which is the point: a bug or an overfit in a strategy
cannot disable a risk control.

Three families of control are enforced:

Position controls
    Per-symbol weight, asset-class exposure, gross and net leverage, position
    count, minimum and maximum order notional.
Portfolio controls
    Volatility target and ceiling, drawdown ladder, maximum drawdown halt, daily
    and weekly loss limits, turnover cap, correlation-concentration warning.
Operational controls
    Stale data, missing prices, duplicate orders, trading-hours window, buying
    power, reconciliation status, consecutive API failures, and a global kill
    switch.

The kill switch defaults to off but blocks *everything* when engaged, and live
trading is disabled unconditionally in :mod:`atlas.execution.safety`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import AtlasConfig, RiskConfig
from atlas.exceptions import KillSwitchActive, RiskLimitBreached
from atlas.logging_utils import get_logger
from atlas.portfolio.covariance import correlation_from_covariance, portfolio_volatility

log = get_logger(__name__)

__all__ = [
    "RiskAction",
    "RiskAssessment",
    "RiskEvent",
    "RiskManager",
    "RiskSeverity",
]


class RiskSeverity(str, Enum):
    """How serious a risk finding is."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskAction(str, Enum):
    """What the risk manager did about a finding."""

    ALLOWED = "allowed"
    SCALED = "scaled"
    REJECTED = "rejected"
    HALTED = "halted"


@dataclass(frozen=True)
class RiskEvent:
    """A single recorded risk decision."""

    timestamp: pd.Timestamp
    control: str
    severity: RiskSeverity
    action: RiskAction
    message: str
    observed: float | None = None
    limit: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "timestamp": self.timestamp,
            "control": self.control,
            "severity": self.severity.value,
            "action": self.action.value,
            "message": self.message,
            "observed": self.observed,
            "limit": self.limit,
            **self.details,
        }


@dataclass
class RiskAssessment:
    """Outcome of a pre-trade risk assessment."""

    approved: bool
    weights: pd.Series
    risk_multiplier: float
    events: list[RiskEvent] = field(default_factory=list)
    blocked_reason: str | None = None

    @property
    def scaled(self) -> bool:
        """True when the risk manager reduced the proposed exposure."""
        return any(e.action is RiskAction.SCALED for e in self.events)

    def critical_events(self) -> list[RiskEvent]:
        """Events with critical severity."""
        return [e for e in self.events if e.severity is RiskSeverity.CRITICAL]

    def to_frame(self) -> pd.DataFrame:
        """Events as a DataFrame."""
        if not self.events:
            return pd.DataFrame(
                columns=["timestamp", "control", "severity", "action", "message", "observed", "limit"]
            )
        return pd.DataFrame([e.as_dict() for e in self.events])

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs."""
        return {
            "approved": self.approved,
            "risk_multiplier": round(self.risk_multiplier, 4),
            "gross_exposure": round(float(self.weights.abs().sum()), 4),
            "n_events": len(self.events),
            "n_critical": len(self.critical_events()),
            "blocked_reason": self.blocked_reason,
        }


class RiskManager:
    """Enforce Atlas's risk limits.

    Parameters
    ----------
    config:
        The full :class:`~atlas.config.AtlasConfig`.

    Notes
    -----
    The manager keeps a small amount of state - the equity peak, a rolling
    window of recent equity values, an order-fingerprint window for duplicate
    detection, and an API-failure counter. This state is per-instance, so a
    backtest and a live session never share it.
    """

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.risk: RiskConfig = config.risk
        self.limits = config.risk.limits
        self.portfolio_cfg = config.portfolio
        self.asset_classes = config.asset_class_map

        self._peak_equity: float | None = None
        self._equity_history: deque[tuple[pd.Timestamp, float]] = deque(
            maxlen=config.risk.drawdown_controls.max_history_days
        )
        self._recent_orders: deque[tuple[datetime, str]] = deque(maxlen=500)
        self._api_failures = 0
        self._halted = False
        self._halt_reason = ""
        self._halt_until: pd.Timestamp | None = None
        self.events: list[RiskEvent] = []
        self._class_cache: dict[tuple[str, ...], dict[str, list[str]]] = {}
        self._position_cache: dict[tuple[str, ...], dict[str, np.ndarray]] = {}

    def _class_positions(self, symbols: pd.Index) -> dict[str, np.ndarray]:
        """Return ``{asset_class: positional indices}``, memoised (see constraints)."""
        key = tuple(str(s) for s in symbols)
        cached = self._position_cache.get(key)
        if cached is None:
            lookup = {s: i for i, s in enumerate(key)}
            cached = {
                name: np.array([lookup[s] for s in members], dtype=int)
                for name, members in self._class_groups(symbols).items()
            }
            self._position_cache[key] = cached
        return cached

    def _class_groups(self, symbols: pd.Index) -> dict[str, list[str]]:
        """Return ``{asset_class: [symbol, ...]}`` for ``symbols``, memoised."""
        key = tuple(str(s) for s in symbols)
        cached = self._class_cache.get(key)
        if cached is None:
            cached = {}
            for symbol in key:
                cached.setdefault(self.asset_classes.get(symbol, "unclassified"), []).append(symbol)
            self._class_cache[key] = cached
        return cached

    # -- state ----------------------------------------------------------------

    @property
    def halted(self) -> bool:
        """True when trading has been halted by a control or the kill switch."""
        return self._halted or self.risk.kill_switch.enabled

    @property
    def halt_reason(self) -> str:
        """Why trading is halted."""
        if self.risk.kill_switch.enabled:
            return self.risk.kill_switch.reason or "kill switch enabled in configuration"
        return self._halt_reason

    def reset(self) -> None:
        """Clear all accumulated state. Used between backtest runs."""
        self._peak_equity = None
        self._equity_history.clear()
        self._recent_orders.clear()
        self._api_failures = 0
        self._halted = False
        self._halt_reason = ""
        self._halt_until = None
        self.events.clear()

    def resume(self, reason: str = "manual resume") -> None:
        """Clear a halt set by a portfolio control (not the kill switch)."""
        if self.risk.kill_switch.enabled:
            raise KillSwitchActive(
                "cannot resume while the kill switch is enabled; disable it in configs/risk.yaml "
                "or unset ATLAS_KILL_SWITCH"
            )
        self._halted = False
        self._halt_reason = ""
        self._halt_until = None
        log.warning("trading resumed", extra={"context": {"reason": reason}})

    def check_kill_switch(self) -> None:
        """Raise :class:`KillSwitchActive` when the global kill switch is on."""
        if self.risk.kill_switch.enabled:
            raise KillSwitchActive(
                f"kill switch active: {self.risk.kill_switch.reason or 'no reason given'}"
            )

    def update_equity(self, timestamp: pd.Timestamp, equity: float) -> None:
        """Record the latest portfolio equity, updating the running peak."""
        value = float(equity)
        self._peak_equity = value if self._peak_equity is None else max(self._peak_equity, value)
        self._equity_history.append((pd.Timestamp(timestamp), value))

    # -- derived risk measures ------------------------------------------------

    def reference_peak(self) -> float | None:
        """The peak the drawdown control measures against.

        Uses the highest equity in the trailing ``peak_lookback_days`` window
        when one is configured, else the all-time high. See
        :class:`~atlas.config.DrawdownControlConfig` for why this matters.
        """
        lookback = self.risk.drawdown_controls.peak_lookback_days
        if lookback <= 0 or not self._equity_history:
            return self._peak_equity
        window = list(self._equity_history)[-lookback:]
        return max(value for _, value in window) if window else self._peak_equity

    def current_drawdown(self, equity: float | None = None) -> float:
        """Drawdown from the reference peak, as a positive fraction."""
        value = float(equity) if equity is not None else (
            self._equity_history[-1][1] if self._equity_history else 0.0
        )
        peak = self.reference_peak()
        if not peak or peak <= 0:
            return 0.0
        return max(0.0, 1.0 - value / peak)

    def all_time_drawdown(self, equity: float | None = None) -> float:
        """Drawdown from the all-time peak, for reporting."""
        value = float(equity) if equity is not None else (
            self._equity_history[-1][1] if self._equity_history else 0.0
        )
        if not self._peak_equity or self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - value / self._peak_equity)

    def drawdown_multiplier(self, equity: float | None = None) -> tuple[float, float | None]:
        """Return ``(multiplier, breached_threshold)`` from the drawdown ladder."""
        dd = self.current_drawdown(equity)
        return self.risk.drawdown_controls.multiplier_for(dd)

    def period_return(self, days: int) -> float:
        """Return over the trailing ``days`` calendar days from stored equity."""
        if len(self._equity_history) < 2:
            return 0.0
        latest_ts, latest = self._equity_history[-1]
        cutoff = latest_ts - pd.Timedelta(days=days)
        earlier = [v for ts, v in self._equity_history if ts <= cutoff]
        base = earlier[-1] if earlier else self._equity_history[0][1]
        if base <= 0:
            return 0.0
        return latest / base - 1.0

    # -- main pre-trade assessment --------------------------------------------

    def assess(
        self,
        timestamp: pd.Timestamp,
        weights: pd.Series,
        *,
        equity: float,
        previous_weights: pd.Series | None = None,
        covariance: pd.DataFrame | None = None,
        last_data_date: pd.Timestamp | None = None,
        prices: pd.Series | None = None,
    ) -> RiskAssessment:
        """Assess a proposed target-weight vector before it becomes orders.

        Returns an assessment carrying possibly-reduced weights and the full list
        of risk events. When ``approved`` is False the caller must not trade.
        """
        ts = pd.Timestamp(timestamp)
        events: list[RiskEvent] = []
        w = weights.astype("float64").fillna(0.0).copy()
        prev = (
            previous_weights.reindex(w.index).fillna(0.0)
            if previous_weights is not None
            else pd.Series(0.0, index=w.index)
        )
        multiplier = 1.0

        # --- global blocks ----------------------------------------------------
        if self.risk.kill_switch.enabled:
            return self._blocked(
                ts, w, "kill_switch", f"kill switch active: {self.halt_reason}", events
            )
        self._maybe_lift_halt(ts)
        if self._halted:
            return self._blocked(ts, w, "trading_halted", self._halt_reason, events)

        self.update_equity(ts, equity)

        # --- operational: stale data -------------------------------------------
        if last_data_date is not None:
            staleness = (ts - pd.Timestamp(last_data_date)).days
            if staleness > self.limits.max_data_staleness_days:
                events.append(
                    RiskEvent(
                        ts,
                        "stale_data",
                        RiskSeverity.CRITICAL,
                        RiskAction.REJECTED,
                        f"market data is {staleness} day(s) old, limit "
                        f"{self.limits.max_data_staleness_days}",
                        float(staleness),
                        float(self.limits.max_data_staleness_days),
                    )
                )
                return self._finish(w, 0.0, events, approved=False, reason="stale market data")

        # --- operational: missing prices ---------------------------------------
        if prices is not None:
            held = w[w.abs() > 1e-9].index
            missing = [str(s) for s in held if pd.isna(prices.get(s, np.nan))]
            if missing:
                events.append(
                    RiskEvent(
                        ts,
                        "missing_prices",
                        RiskSeverity.CRITICAL,
                        RiskAction.REJECTED,
                        f"no price available for {len(missing)} target position(s): "
                        f"{', '.join(missing[:5])}",
                        details={"symbols": missing},
                    )
                )
                # Rather than blocking the whole rebalance, drop the unpriceable
                # names; the rest of the portfolio can still be traded safely.
                w.loc[missing] = 0.0

        # --- portfolio: drawdown ladder ----------------------------------------
        dd = self.current_drawdown(equity)
        dd_multiplier, breached = self.drawdown_multiplier(equity)
        if dd_multiplier < 1.0:
            multiplier *= dd_multiplier
            events.append(
                RiskEvent(
                    ts,
                    "drawdown_control",
                    RiskSeverity.WARNING,
                    RiskAction.SCALED,
                    f"drawdown {dd:.2%} breached the {breached:.0%} threshold; "
                    f"risk budget scaled to {dd_multiplier:.0%}",
                    dd,
                    breached,
                )
            )

        # --- portfolio: maximum drawdown halt ----------------------------------
        if dd > self.limits.max_drawdown:
            events.append(
                RiskEvent(
                    ts,
                    "max_drawdown",
                    RiskSeverity.CRITICAL,
                    RiskAction.HALTED,
                    f"drawdown {dd:.2%} exceeds the {self.limits.max_drawdown:.0%} limit; "
                    "risk-increasing trades are blocked",
                    dd,
                    self.limits.max_drawdown,
                )
            )
            # Allow risk-reducing trades only: the target may not exceed the
            # current book. Liquidation is never automatic.
            w = self._cap_to_risk_reduction(w, prev)

        # --- portfolio: loss limits ---------------------------------------------
        daily = self.period_return(1)
        if daily < -self.limits.daily_loss_limit:
            events.append(
                RiskEvent(
                    ts,
                    "daily_loss_limit",
                    RiskSeverity.CRITICAL,
                    RiskAction.HALTED,
                    f"daily loss {daily:.2%} exceeds the {self.limits.daily_loss_limit:.0%} limit",
                    daily,
                    -self.limits.daily_loss_limit,
                )
            )
            self._halt(
                f"daily loss limit breached ({daily:.2%})",
                until=ts + pd.tseries.offsets.BDay(self.limits.loss_limit_cooldown_days),
            )
            return self._finish(prev, 0.0, events, approved=False, reason="daily loss limit")

        weekly = self.period_return(7)
        if weekly < -self.limits.weekly_loss_limit:
            events.append(
                RiskEvent(
                    ts,
                    "weekly_loss_limit",
                    RiskSeverity.CRITICAL,
                    RiskAction.HALTED,
                    f"weekly loss {weekly:.2%} exceeds the {self.limits.weekly_loss_limit:.0%} limit",
                    weekly,
                    -self.limits.weekly_loss_limit,
                )
            )
            self._halt(
                f"weekly loss limit breached ({weekly:.2%})",
                until=ts + pd.tseries.offsets.BDay(self.limits.loss_limit_cooldown_days),
            )
            return self._finish(prev, 0.0, events, approved=False, reason="weekly loss limit")

        # --- apply the accumulated risk multiplier ------------------------------
        if multiplier < 1.0:
            w = w * multiplier

        # --- position controls ---------------------------------------------------
        w = self._enforce_position_limits(ts, w, events)

        # --- portfolio: predicted volatility ceiling ------------------------------
        if covariance is not None and not covariance.empty:
            predicted = portfolio_volatility(w, covariance)
            ceiling = self.portfolio_cfg.max_portfolio_volatility
            if predicted > ceiling > 0:
                shrink = ceiling / predicted
                w = w * shrink
                events.append(
                    RiskEvent(
                        ts,
                        "max_portfolio_volatility",
                        RiskSeverity.WARNING,
                        RiskAction.SCALED,
                        f"predicted volatility {predicted:.2%} exceeds the {ceiling:.0%} ceiling; "
                        f"book scaled by {shrink:.3f}",
                        predicted,
                        ceiling,
                    )
                )
            # --- correlation concentration warning ---
            self._check_correlation(ts, w, covariance, events)

        # --- turnover ------------------------------------------------------------
        turnover = float((w - prev).abs().sum())
        if turnover > self.portfolio_cfg.maximum_daily_turnover + 1e-9:
            lam = self.portfolio_cfg.maximum_daily_turnover / turnover
            w = prev + lam * (w - prev)
            events.append(
                RiskEvent(
                    ts,
                    "maximum_daily_turnover",
                    RiskSeverity.WARNING,
                    RiskAction.SCALED,
                    f"turnover {turnover:.2%} exceeds the "
                    f"{self.portfolio_cfg.maximum_daily_turnover:.0%} cap; trade blended at {lam:.3f}",
                    turnover,
                    self.portfolio_cfg.maximum_daily_turnover,
                )
            )

        return self._finish(w, multiplier, events, approved=True, reason=None)

    # -- order-level checks ----------------------------------------------------

    def check_order(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        quantity: float,
        price: float,
        *,
        buying_power: float | None = None,
        now: datetime | None = None,
        enforce_trading_window: bool = False,
    ) -> tuple[bool, str]:
        """Validate a single order. Returns ``(allowed, reason)``.

        Checks, in order: kill switch, halt state, price validity, notional
        bounds, buying power, trading window, and duplicate suppression.
        """
        ts = pd.Timestamp(timestamp)

        if self.risk.kill_switch.enabled:
            return False, f"kill switch active: {self.halt_reason}"
        if self._halted:
            return False, f"trading halted: {self._halt_reason}"
        if not np.isfinite(price) or price <= 0:
            return False, f"invalid price for {symbol}: {price!r}"
        if not np.isfinite(quantity) or abs(quantity) < 1e-12:
            return False, f"zero or invalid quantity for {symbol}"

        notional = abs(float(quantity) * float(price))
        if notional < self.limits.min_order_notional:
            return (
                False,
                f"order notional ${notional:,.2f} is below the ${self.limits.min_order_notional:,.2f} minimum",
            )
        if notional > self.limits.max_order_notional:
            return (
                False,
                f"order notional ${notional:,.2f} exceeds the ${self.limits.max_order_notional:,.2f} maximum",
            )
        if buying_power is not None and quantity > 0 and notional > buying_power:
            return (
                False,
                f"order notional ${notional:,.2f} exceeds available buying power ${buying_power:,.2f}",
            )

        if enforce_trading_window and not self.within_trading_window(now):
            window = self.config.execution.trading_window
            return False, f"outside the configured trading window {window.start}-{window.end} {window.timezone}"

        fingerprint = f"{symbol}|{np.sign(quantity):.0f}|{abs(quantity):.4f}|{ts.date()}"
        moment = now or datetime.now(UTC)
        window_seconds = self.config.execution.order_defaults.duplicate_window_seconds
        for seen_at, seen in self._recent_orders:
            if seen == fingerprint and (moment - seen_at).total_seconds() < window_seconds:
                return False, f"duplicate order for {symbol} within {window_seconds}s"
        self._recent_orders.append((moment, fingerprint))
        return True, ""

    def within_trading_window(self, now: datetime | None = None) -> bool:
        """True when ``now`` falls inside the configured trading window."""
        window = self.config.execution.trading_window
        moment = now or datetime.now(UTC)
        try:
            from zoneinfo import ZoneInfo

            local = moment.astimezone(ZoneInfo(window.timezone))
        except Exception:
            local = moment
        if local.weekday() not in window.trading_days:
            return False
        start_h, start_m = (int(x) for x in window.start.split(":"))
        end_h, end_m = (int(x) for x in window.end.split(":"))
        return time(start_h, start_m) <= local.time() <= time(end_h, end_m)

    # -- operational counters ---------------------------------------------------

    def record_api_failure(self, context: str = "") -> bool:
        """Record a broker API failure; halt after too many consecutive ones.

        Returns True when the failure triggered a halt.
        """
        self._api_failures += 1
        if self._api_failures >= self.limits.max_consecutive_api_failures:
            self._halt(
                f"{self._api_failures} consecutive broker API failures"
                + (f" ({context})" if context else "")
            )
            return True
        return False

    def record_api_success(self) -> None:
        """Reset the consecutive-failure counter."""
        self._api_failures = 0

    def record_reconciliation_failure(self, detail: str) -> None:
        """Halt trading because positions could not be reconciled."""
        self._halt(f"unresolved position mismatch: {detail}")

    # -- internals --------------------------------------------------------------

    def _halt(self, reason: str, *, until: pd.Timestamp | None = None) -> None:
        """Halt trading and log it loudly.

        ``until`` sets an automatic resume time, used for loss-limit circuit
        breakers. A halt with no ``until`` latches until a human resumes it.
        """
        self._halted = True
        self._halt_reason = reason
        self._halt_until = until
        log.error(
            "TRADING HALTED",
            extra={"context": {"reason": reason, "until": str(until) if until else "manual resume"}},
        )

    def _maybe_lift_halt(self, timestamp: pd.Timestamp) -> None:
        """Lift a timed halt once its cool-off period has elapsed."""
        if not self._halted or self._halt_until is None:
            return
        if pd.Timestamp(timestamp) >= self._halt_until:
            log.warning(
                "loss-limit halt lifted after the cool-off period",
                extra={"context": {"previous_reason": self._halt_reason}},
            )
            self._halted = False
            self._halt_reason = ""
            self._halt_until = None

    def _blocked(
        self,
        ts: pd.Timestamp,
        weights: pd.Series,
        control: str,
        message: str,
        events: list[RiskEvent],
    ) -> RiskAssessment:
        """Build a blocked assessment that leaves the book untouched."""
        events.append(
            RiskEvent(ts, control, RiskSeverity.CRITICAL, RiskAction.HALTED, message)
        )
        return self._finish(weights * 0.0, 0.0, events, approved=False, reason=message)

    def _finish(
        self,
        weights: pd.Series,
        multiplier: float,
        events: list[RiskEvent],
        *,
        approved: bool,
        reason: str | None,
    ) -> RiskAssessment:
        """Record the events and build the assessment."""
        self.events.extend(events)
        assessment = RiskAssessment(
            approved=approved,
            weights=weights,
            risk_multiplier=multiplier,
            events=events,
            blocked_reason=reason,
        )
        if not approved:
            log.warning("risk assessment blocked trading", extra={"context": assessment.summary()})
        return assessment

    def _cap_to_risk_reduction(self, target: pd.Series, previous: pd.Series) -> pd.Series:
        """Allow only trades that reduce exposure, per symbol."""
        out = target.copy()
        for symbol in out.index:
            new, old = float(out[symbol]), float(previous.get(symbol, 0.0))
            if abs(new) > abs(old):
                out[symbol] = old
        return out

    def _enforce_position_limits(
        self, ts: pd.Timestamp, weights: pd.Series, events: list[RiskEvent]
    ) -> pd.Series:
        """Enforce per-symbol, class, count and leverage limits."""
        cfg = self.portfolio_cfg
        w = weights.copy()

        # per symbol
        breaches = w[w.abs() > cfg.max_asset_weight + 1e-9]
        if not breaches.empty:
            w = w.clip(-cfg.max_asset_weight, cfg.max_asset_weight)
            events.append(
                RiskEvent(
                    ts,
                    "max_asset_weight",
                    RiskSeverity.WARNING,
                    RiskAction.SCALED,
                    f"{len(breaches)} position(s) capped at {cfg.max_asset_weight:.0%}",
                    float(breaches.abs().max()),
                    cfg.max_asset_weight,
                    details={"symbols": [str(s) for s in breaches.index]},
                )
            )

        # asset class
        class_positions = self._class_positions(w.index)
        values = w.to_numpy(dtype="float64").copy()
        class_scaled = False
        for name in self._class_groups(w.index):
            positions = class_positions[name]
            exposure = float(np.abs(values[positions]).sum())
            if exposure > cfg.max_asset_class_weight + 1e-9:
                values[positions] *= cfg.max_asset_class_weight / exposure
                class_scaled = True
                events.append(
                    RiskEvent(
                        ts,
                        "max_asset_class_weight",
                        RiskSeverity.WARNING,
                        RiskAction.SCALED,
                        f"asset class {name!r} exposure {exposure:.1%} scaled to "
                        f"{cfg.max_asset_class_weight:.0%}",
                        exposure,
                        cfg.max_asset_class_weight,
                    )
                )
        if class_scaled:
            w = pd.Series(values, index=w.index)

        # position count
        active = w[w.abs() > 1e-9]
        if len(active) > cfg.max_active_positions:
            keep = set(active.abs().nlargest(cfg.max_active_positions).index)
            dropped = {str(s) for s in active.index if s not in keep}
            w = pd.Series(
                np.where([str(s) in dropped for s in w.index], 0.0, w.to_numpy()), index=w.index
            )
            events.append(
                RiskEvent(
                    ts,
                    "max_active_positions",
                    RiskSeverity.INFO,
                    RiskAction.SCALED,
                    f"reduced to the {cfg.max_active_positions} largest positions",
                    float(len(active)),
                    float(cfg.max_active_positions),
                    details={"dropped": sorted(dropped)},
                )
            )

        # gross / net leverage
        gross = float(w.abs().sum())
        gross_cap = min(cfg.max_gross_exposure, cfg.max_leverage)
        if gross > gross_cap + 1e-9:
            w = w * (gross_cap / gross)
            events.append(
                RiskEvent(
                    ts,
                    "max_gross_exposure",
                    RiskSeverity.WARNING,
                    RiskAction.SCALED,
                    f"gross exposure {gross:.2f}x scaled to {gross_cap:.2f}x",
                    gross,
                    gross_cap,
                )
            )
        net = float(w.sum())
        if abs(net) > cfg.max_net_exposure + 1e-9:
            w = w * (cfg.max_net_exposure / abs(net))
            events.append(
                RiskEvent(
                    ts,
                    "max_net_exposure",
                    RiskSeverity.WARNING,
                    RiskAction.SCALED,
                    f"net exposure {net:.2f}x scaled to {cfg.max_net_exposure:.2f}x",
                    net,
                    cfg.max_net_exposure,
                )
            )
        return w

    def _check_correlation(
        self,
        ts: pd.Timestamp,
        weights: pd.Series,
        covariance: pd.DataFrame,
        events: list[RiskEvent],
    ) -> None:
        """Warn when the largest holdings are highly correlated with each other."""
        held = weights[weights.abs() > 1e-9]
        if len(held) < 2:
            return
        symbols = [s for s in held.index if s in covariance.columns]
        if len(symbols) < 2:
            return
        corr = correlation_from_covariance(covariance.loc[symbols, symbols])
        values = corr.to_numpy()
        upper = values[np.triu_indices_from(values, k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size == 0:
            return
        worst = float(np.max(upper))
        if worst > self.limits.correlation_warning_threshold:
            i, j = np.unravel_index(np.argmax(np.triu(values, k=1)), values.shape)
            events.append(
                RiskEvent(
                    ts,
                    "correlation_concentration",
                    RiskSeverity.WARNING,
                    RiskAction.ALLOWED,
                    f"{symbols[i]} and {symbols[j]} have a correlation of {worst:.2f}; "
                    "diversification across held positions is limited",
                    worst,
                    self.limits.correlation_warning_threshold,
                    details={"pair": f"{symbols[i]}/{symbols[j]}"},
                )
            )

    # -- reporting -------------------------------------------------------------

    def events_frame(self) -> pd.DataFrame:
        """All recorded risk events as a DataFrame."""
        if not self.events:
            return pd.DataFrame(
                columns=["timestamp", "control", "severity", "action", "message", "observed", "limit"]
            )
        return pd.DataFrame([e.as_dict() for e in self.events])

    def status(self) -> dict[str, Any]:
        """Current risk state, for the dashboard and CLI."""
        equity = self._equity_history[-1][1] if self._equity_history else None
        dd = self.current_drawdown() if equity is not None else 0.0
        multiplier, breached = self.drawdown_multiplier()
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "kill_switch": self.risk.kill_switch.enabled,
            "equity": equity,
            "peak_equity": self._peak_equity,
            "current_drawdown": dd,
            "reference_peak": self.reference_peak(),
            "all_time_drawdown": self.all_time_drawdown() if equity is not None else 0.0,
            "drawdown_multiplier": multiplier,
            "drawdown_threshold_breached": breached,
            "consecutive_api_failures": self._api_failures,
            "halt_until": str(self._halt_until) if self._halt_until else None,
            "n_events": len(self.events),
        }

    def raise_if_halted(self) -> None:
        """Raise :class:`RiskLimitBreached` when trading is halted."""
        if self.risk.kill_switch.enabled:
            raise KillSwitchActive(f"kill switch active: {self.halt_reason}")
        if self._halted:
            raise RiskLimitBreached(
                f"trading is halted: {self._halt_reason}", limit_name="trading_halt"
            )


def business_days_between(start: date, end: date) -> int:
    """Number of business days between two dates (inclusive of neither end)."""
    return max(0, len(pd.bdate_range(start + timedelta(days=1), end)))
