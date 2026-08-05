"""Pre-trade safety gates.

Everything in this module exists to make it hard to do something dangerous by
accident. The rules are deliberately blunt:

* **Live trading is not implemented.** ``EXECUTION_MODE=live`` is rejected at
  configuration load *and* again here, so removing one check is not enough to
  enable it.
* **Paper accounts only.** Interactive Brokers paper accounts are prefixed
  ``DU``. When ``require_paper_account`` is set, any other account identifier is
  refused regardless of what the allowlist says.
* **Account allowlist.** Even a paper account must appear on the configured
  allowlist before an order may be sent, so connecting to the wrong TWS session
  cannot silently trade the wrong account.
* **Kill switch.** One flag stops everything.

:class:`SafetyGate` is checked before every execution cycle and again before
every order. A failure raises rather than returning a status code, because a
safety check that can be ignored by forgetting an ``if`` is not a safety check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from atlas.config import AtlasConfig, ExecutionMode
from atlas.exceptions import ExecutionBlocked, KillSwitchActive
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["PAPER_ACCOUNT_PREFIXES", "SafetyCheck", "SafetyGate", "SafetyReport"]

#: Account-identifier prefixes that Interactive Brokers uses for paper accounts.
PAPER_ACCOUNT_PREFIXES: tuple[str, ...] = ("DU", "DF")


@dataclass(frozen=True)
class SafetyCheck:
    """The outcome of a single safety check."""

    name: str
    passed: bool
    message: str
    fatal: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "check": self.name,
            "passed": self.passed,
            "fatal": self.fatal,
            "message": self.message,
        }


@dataclass
class SafetyReport:
    """The result of running every applicable safety check."""

    checks: list[SafetyCheck] = field(default_factory=list)
    mode: ExecutionMode = ExecutionMode.DRY_RUN
    account: str | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        """True when no fatal check failed."""
        return all(c.passed for c in self.checks if c.fatal)

    @property
    def failures(self) -> list[SafetyCheck]:
        """Failed fatal checks."""
        return [c for c in self.checks if c.fatal and not c.passed]

    @property
    def warnings(self) -> list[SafetyCheck]:
        """Failed non-fatal checks."""
        return [c for c in self.checks if not c.fatal and not c.passed]

    def raise_if_failed(self) -> None:
        """Raise :class:`ExecutionBlocked` when any fatal check failed."""
        if not self.passed:
            reasons = "; ".join(f"{c.name}: {c.message}" for c in self.failures)
            raise ExecutionBlocked(f"pre-trade safety checks failed - {reasons}")

    def render(self) -> str:
        """Human-readable report."""
        lines = [
            "Atlas pre-trade safety report",
            "=" * 60,
            f"  mode:    {self.mode.value}",
            f"  account: {self.account or '(not connected)'}",
            f"  time:    {self.checked_at.isoformat()}",
            "-" * 60,
        ]
        for check in self.checks:
            status = "PASS" if check.passed else ("FAIL" if check.fatal else "WARN")
            lines.append(f"  [{status}] {check.name}: {check.message}")
        lines.append("-" * 60)
        lines.append(f"  overall: {'PASSED' if self.passed else 'BLOCKED'}")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs."""
        return {
            "mode": self.mode.value,
            "account": self.account,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
        }


class SafetyGate:
    """Run the pre-trade safety checks for a session."""

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.execution = config.execution
        self.broker = config.execution.broker

    # -- mode ------------------------------------------------------------------

    @property
    def mode(self) -> ExecutionMode:
        """The configured execution mode."""
        return self.execution.mode

    @property
    def is_dry_run(self) -> bool:
        """True when orders must be previewed, never submitted."""
        return self.mode in {ExecutionMode.BACKTEST, ExecutionMode.DRY_RUN}

    @property
    def may_submit_orders(self) -> bool:
        """True only in paper mode - Atlas never submits live orders."""
        return self.mode is ExecutionMode.PAPER

    def assert_not_live(self) -> None:
        """Raise if anything has managed to select live mode.

        This is the second of two independent barriers; the first is in
        :class:`~atlas.config.ExecutionConfig`.
        """
        if self.mode is ExecutionMode.LIVE:
            raise ExecutionBlocked(
                "live trading is not implemented in Atlas. The mode exists only as a disabled "
                "placeholder. Use EXECUTION_MODE=paper against an Interactive Brokers paper "
                "account."
            )

    # -- account ---------------------------------------------------------------

    def check_account(self, account: str | None) -> list[SafetyCheck]:
        """Validate a broker account identifier."""
        checks: list[SafetyCheck] = []

        if not account:
            checks.append(
                SafetyCheck(
                    "account_identified",
                    False,
                    "the broker reported no account identifier; cannot verify it is a paper account",
                )
            )
            return checks
        checks.append(SafetyCheck("account_identified", True, f"connected account {account}"))

        is_paper = account.upper().startswith(PAPER_ACCOUNT_PREFIXES)
        if self.broker.require_paper_account:
            checks.append(
                SafetyCheck(
                    "paper_account",
                    is_paper,
                    (
                        f"{account} is an Interactive Brokers paper account"
                        if is_paper
                        else f"{account} does not look like a paper account (expected a "
                        f"{'/'.join(PAPER_ACCOUNT_PREFIXES)} prefix). Atlas refuses to trade it."
                    ),
                )
            )
        else:
            checks.append(
                SafetyCheck(
                    "paper_account",
                    True,
                    "paper-account enforcement is disabled in configuration - this is not "
                    "recommended",
                    fatal=False,
                )
            )

        allowlist = [a.strip().upper() for a in self.broker.account_allowlist if a.strip()]
        if allowlist:
            allowed = account.upper() in allowlist
            checks.append(
                SafetyCheck(
                    "account_allowlist",
                    allowed,
                    (
                        f"{account} is on the configured allowlist"
                        if allowed
                        else f"{account} is not on the allowlist {allowlist}. Set "
                        "ATLAS_IBKR_ACCOUNT_ALLOWLIST to authorise it."
                    ),
                )
            )
        else:
            checks.append(
                SafetyCheck(
                    "account_allowlist",
                    False,
                    "no account allowlist is configured. Set ATLAS_IBKR_ACCOUNT_ALLOWLIST to the "
                    "paper account you intend to trade before submitting orders.",
                    fatal=self.may_submit_orders,
                )
            )
        return checks

    # -- full gate -------------------------------------------------------------

    def run(
        self,
        *,
        account: str | None = None,
        connected: bool | None = None,
        data_is_fresh: bool | None = None,
        positions_known: bool | None = None,
        reconciled: bool | None = None,
        risk_halted: bool = False,
        risk_halt_reason: str = "",
    ) -> SafetyReport:
        """Run every applicable check and return the report.

        Parameters that are ``None`` are treated as not-applicable and produce no
        check, which lets the same gate serve a dry run (no broker) and a paper
        session (full checks).
        """
        self.assert_not_live()
        checks: list[SafetyCheck] = []

        # --- kill switch ---
        kill = self.config.risk.kill_switch
        checks.append(
            SafetyCheck(
                "kill_switch",
                not kill.enabled,
                (
                    "kill switch is off"
                    if not kill.enabled
                    else f"kill switch is ENGAGED: {kill.reason or 'no reason recorded'}"
                ),
            )
        )

        # --- mode ---
        checks.append(
            SafetyCheck(
                "execution_mode",
                self.mode in {ExecutionMode.BACKTEST, ExecutionMode.DRY_RUN, ExecutionMode.PAPER},
                f"execution mode is {self.mode.value}"
                + (" - orders will be previewed only" if self.is_dry_run else ""),
            )
        )

        # --- risk manager state ---
        checks.append(
            SafetyCheck(
                "risk_manager",
                not risk_halted,
                "risk manager is not halted"
                if not risk_halted
                else f"risk manager has halted trading: {risk_halt_reason}",
            )
        )

        if connected is not None:
            checks.append(
                SafetyCheck(
                    "broker_connection",
                    connected,
                    "connected to the broker" if connected else "not connected to the broker",
                    fatal=not self.is_dry_run,
                )
            )

        if account is not None or self.may_submit_orders:
            checks.extend(self.check_account(account))

        if data_is_fresh is not None:
            checks.append(
                SafetyCheck(
                    "market_data_fresh",
                    data_is_fresh,
                    "market data is within the configured staleness limit"
                    if data_is_fresh
                    else f"market data is older than "
                    f"{self.config.risk.limits.max_data_staleness_days} day(s)",
                )
            )

        if positions_known is not None:
            checks.append(
                SafetyCheck(
                    "positions_known",
                    positions_known,
                    "current positions retrieved"
                    if positions_known
                    else "current positions could not be retrieved",
                )
            )

        if reconciled is not None:
            checks.append(
                SafetyCheck(
                    "position_reconciliation",
                    reconciled,
                    "broker and local positions agree"
                    if reconciled
                    else "broker and local positions do not reconcile; trading is blocked until "
                    "the mismatch is resolved",
                )
            )

        report = SafetyReport(checks=checks, mode=self.mode, account=account)
        log.info("safety gate evaluated", extra={"context": report.summary()})
        for failure in report.failures:
            log.error(
                "safety check failed",
                extra={"context": {"check": failure.name, "message": failure.message}},
            )
        return report

    def guard(self, **kwargs: Any) -> SafetyReport:
        """Run :meth:`run` and raise unless every fatal check passed."""
        if self.config.risk.kill_switch.enabled:
            raise KillSwitchActive(
                f"kill switch active: {self.config.risk.kill_switch.reason or 'no reason recorded'}"
            )
        report = self.run(**kwargs)
        report.raise_if_failed()
        return report
