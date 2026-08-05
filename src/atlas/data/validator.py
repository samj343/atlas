"""Market-data validation.

Atlas refuses to trade on data it has not checked. :class:`DataValidator`
inspects a :class:`~atlas.data.loader.PricePanel` and produces a
:class:`ValidationReport` listing every issue found, classified by severity:

``ERROR``
    The data is unusable - e.g. non-positive prices, ``high < low``. The symbol
    is excluded from trading.
``WARNING``
    The data is suspicious but usable - e.g. an abnormal single-day gap, a short
    run of missing days, a stale series.
``INFO``
    Informational - e.g. an asset whose history starts later than the rest.

The validator never repairs data silently. Short internal gaps can be filled
*explicitly* via :meth:`DataValidator.clean`, which caps the fill length and
records exactly what it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from atlas.data.loader import PricePanel
from atlas.exceptions import DataValidationError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["DataValidator", "Severity", "ValidationIssue", "ValidationReport"]


class Severity(str, Enum):
    """Severity of a validation finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    severity: Severity
    code: str
    symbol: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable representation."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "symbol": self.symbol or "",
            "message": self.message,
            **{f"detail_{k}": v for k, v in self.details.items()},
        }


@dataclass
class ValidationReport:
    """Aggregated result of a validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)
    checked_symbols: list[str] = field(default_factory=list)
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # -- queries --------------------------------------------------------------

    @property
    def errors(self) -> list[ValidationIssue]:
        """Findings with ``ERROR`` severity."""
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Findings with ``WARNING`` severity."""
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """True when no ``ERROR``-severity issue was found."""
        return not self.errors

    @property
    def failed_symbols(self) -> set[str]:
        """Symbols that produced at least one ``ERROR``."""
        return {i.symbol for i in self.errors if i.symbol}

    def usable_symbols(self) -> list[str]:
        """Checked symbols excluding those with errors."""
        bad = self.failed_symbols
        return [s for s in self.checked_symbols if s not in bad]

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        symbol: str | None = None,
        **details: Any,
    ) -> None:
        """Record a new finding."""
        self.issues.append(ValidationIssue(severity, code, symbol, message, details))

    def to_frame(self) -> pd.DataFrame:
        """Return the findings as a DataFrame (empty with correct columns if none)."""
        if not self.issues:
            return pd.DataFrame(columns=["severity", "code", "symbol", "message"])
        return pd.DataFrame([i.as_dict() for i in self.issues])

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dictionary."""
        counts = {s.value: 0 for s in Severity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return {
            "generated_at": self.generated_at.isoformat(),
            "symbols_checked": len(self.checked_symbols),
            "symbols_usable": len(self.usable_symbols()),
            "start_date": str(self.start_date.date()) if self.start_date is not None else None,
            "end_date": str(self.end_date.date()) if self.end_date is not None else None,
            "is_valid": self.is_valid,
            **{f"n_{k}": v for k, v in counts.items()},
        }

    def render(self, max_rows: int = 40) -> str:
        """Render a human-readable text report."""
        lines = ["Atlas data validation report", "=" * 60]
        for key, value in self.summary().items():
            lines.append(f"  {key:<20} {value}")
        lines.append("-" * 60)
        if not self.issues:
            lines.append("  no issues found")
            return "\n".join(lines)
        ordered = sorted(
            self.issues,
            key=lambda i: ({"error": 0, "warning": 1, "info": 2}[i.severity.value], i.symbol or ""),
        )
        for issue in ordered[:max_rows]:
            sym = f"[{issue.symbol}]" if issue.symbol else "[portfolio]"
            lines.append(f"  {issue.severity.value.upper():<8} {sym:<8} {issue.code}: {issue.message}")
        if len(ordered) > max_rows:
            lines.append(f"  ... {len(ordered) - max_rows} more")
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        """Raise :class:`DataValidationError` when any ``ERROR`` was recorded."""
        if not self.is_valid:
            details = "; ".join(f"{i.symbol}: {i.message}" for i in self.errors[:5])
            raise DataValidationError(
                f"{len(self.errors)} data validation error(s): {details}"
            )


class DataValidator:
    """Validate a :class:`~atlas.data.loader.PricePanel`.

    Parameters
    ----------
    max_staleness_days:
        A series whose last observation is older than this many calendar days
        (relative to the panel's last date) is flagged stale.
    min_history_days:
        Series with fewer observations than this are flagged as too short.
    max_daily_move:
        Absolute one-day return above this threshold is flagged as a suspicious
        gap (default 0.35 == 35%).
    max_gap_days:
        A run of consecutive missing trading days longer than this is an error
        rather than a warning.
    max_fill_days:
        Longest gap that :meth:`clean` is permitted to forward-fill.
    """

    def __init__(
        self,
        *,
        max_staleness_days: int = 5,
        min_history_days: int = 260,
        max_daily_move: float = 0.35,
        max_gap_days: int = 10,
        max_fill_days: int = 3,
    ) -> None:
        self.max_staleness_days = int(max_staleness_days)
        self.min_history_days = int(min_history_days)
        self.max_daily_move = float(max_daily_move)
        self.max_gap_days = int(max_gap_days)
        self.max_fill_days = int(max_fill_days)

    # -- main entry point -----------------------------------------------------

    def validate(
        self,
        panel: PricePanel,
        *,
        expected_symbols: list[str] | None = None,
        as_of: date | None = None,
    ) -> ValidationReport:
        """Run every check against ``panel`` and return the report."""
        report = ValidationReport(
            checked_symbols=list(panel.symbols),
            start_date=panel.dates.min() if len(panel.dates) else None,
            end_date=panel.dates.max() if len(panel.dates) else None,
        )

        if len(panel.dates) == 0:
            report.add(Severity.ERROR, "empty_panel", "the price panel contains no observations")
            return report

        if expected_symbols:
            self._check_missing_symbols(panel, expected_symbols, report)

        self._check_calendar(panel, report)

        # Per-symbol checks slice by label, which raises on a duplicated or
        # unsorted index. Report the structural problem and stop rather than
        # letting a pandas KeyError escape from a validation routine.
        structural = {"duplicate_dates", "unsorted_calendar"}
        if any(i.code in structural for i in report.errors):
            report.add(
                Severity.ERROR,
                "validation_incomplete",
                "per-symbol checks were skipped because the calendar is duplicated or unsorted; "
                "fix the index and re-run validation",
            )
            log.error("data validation stopped early", extra={"context": report.summary()})
            return report

        for symbol in panel.symbols:
            self._check_symbol(panel, symbol, report, as_of=as_of)

        self._check_cross_sectional(panel, report)

        log.info("data validation complete", extra={"context": report.summary()})
        return report

    # -- individual checks ----------------------------------------------------

    @staticmethod
    def _check_missing_symbols(
        panel: PricePanel, expected: list[str], report: ValidationReport
    ) -> None:
        """Flag requested symbols that the provider did not return."""
        present = set(panel.symbols)
        for symbol in expected:
            if symbol.upper() not in present:
                report.add(
                    Severity.ERROR,
                    "unsupported_symbol",
                    "symbol was requested but no data was returned by the provider",
                    symbol=symbol.upper(),
                )

    def _check_calendar(self, panel: PricePanel, report: ValidationReport) -> None:
        """Flag duplicate index entries and unexpected calendar gaps."""
        idx = panel.dates
        if idx.has_duplicates:
            dupes = idx[idx.duplicated()].unique()
            report.add(
                Severity.ERROR,
                "duplicate_dates",
                f"{len(dupes)} duplicated date(s) in the shared calendar",
                n_duplicates=len(dupes),
                first=str(pd.Timestamp(dupes[0]).date()),
            )
        if not idx.is_monotonic_increasing:
            report.add(Severity.ERROR, "unsorted_calendar", "the calendar is not sorted ascending")

        # Expected business days that are entirely absent from the panel. A
        # handful are market holidays; a long run indicates a real download gap.
        expected = pd.bdate_range(idx.min(), idx.max())
        missing = expected.difference(idx)
        if len(missing) > 0:
            runs = _consecutive_runs(missing)
            longest = max((len(r) for r in runs), default=0)
            severity = Severity.ERROR if longest > self.max_gap_days else Severity.INFO
            report.add(
                severity,
                "calendar_gaps",
                f"{len(missing)} business day(s) absent from the calendar; "
                f"longest consecutive run {longest} day(s) "
                "(short runs are normally exchange holidays)",
                n_missing=len(missing),
                longest_run=int(longest),
            )

    def _check_symbol(
        self,
        panel: PricePanel,
        symbol: str,
        report: ValidationReport,
        *,
        as_of: date | None,
    ) -> None:
        """Run all per-symbol checks."""
        close = panel.close[symbol]
        adj = panel.adj_close[symbol]
        high = panel.high[symbol]
        low = panel.low[symbol]
        open_ = panel.open[symbol]
        volume = panel.volume[symbol]

        valid = adj.dropna()
        if valid.empty:
            report.add(Severity.ERROR, "no_observations", "series contains no prices", symbol=symbol)
            return

        first, last = valid.index.min(), valid.index.max()

        # --- history length ---
        if len(valid) < self.min_history_days:
            report.add(
                Severity.WARNING,
                "short_history",
                f"only {len(valid)} observations (< {self.min_history_days}); the asset will be "
                "ineligible until it has enough history",
                symbol=symbol,
                observations=len(valid),
            )

        # --- late inception ---
        if first > panel.dates.min():
            report.add(
                Severity.INFO,
                "late_inception",
                f"history starts {first.date()}, after the panel start {panel.dates.min().date()}",
                symbol=symbol,
                first_date=str(first.date()),
            )

        # --- staleness ---
        reference = pd.Timestamp(as_of) if as_of is not None else panel.dates.max()
        staleness = (reference - last).days
        if staleness > self.max_staleness_days:
            report.add(
                Severity.WARNING,
                "stale_series",
                f"last observation {last.date()} is {staleness} calendar day(s) old",
                symbol=symbol,
                staleness_days=int(staleness),
            )

        # --- non-positive prices ---
        window = slice(first, last)
        for name, series in (("close", close), ("adj_close", adj), ("open", open_)):
            sub = series.loc[window].dropna()
            bad = sub[sub <= 0.0]
            if len(bad) > 0:
                report.add(
                    Severity.ERROR,
                    "non_positive_price",
                    f"{len(bad)} non-positive value(s) in {name}, first on {bad.index[0].date()}",
                    symbol=symbol,
                    column=name,
                    count=len(bad),
                )

        # --- OHLC coherence ---
        ohlc = pd.concat(
            {"open": open_, "high": high, "low": low, "close": close}, axis=1
        ).loc[window].dropna()
        if len(ohlc) > 0:
            inverted = ohlc[ohlc["high"] < ohlc["low"]]
            if len(inverted) > 0:
                report.add(
                    Severity.ERROR,
                    "inverted_high_low",
                    f"{len(inverted)} bar(s) where high < low",
                    symbol=symbol,
                    count=len(inverted),
                )
            outside = ohlc[
                (ohlc["close"] > ohlc["high"] * 1.0001)
                | (ohlc["close"] < ohlc["low"] * 0.9999)
                | (ohlc["open"] > ohlc["high"] * 1.0001)
                | (ohlc["open"] < ohlc["low"] * 0.9999)
            ]
            if len(outside) > 0:
                report.add(
                    Severity.WARNING,
                    "ohlc_out_of_range",
                    f"{len(outside)} bar(s) where open/close sits outside the high-low range",
                    symbol=symbol,
                    count=len(outside),
                )

        # --- internal missing observations ---
        span = panel.adj_close.loc[window, symbol]
        internal_missing = span[span.isna()]
        if len(internal_missing) > 0:
            runs = _consecutive_runs(pd.DatetimeIndex(internal_missing.index))
            longest = max((len(r) for r in runs), default=0)
            severity = Severity.ERROR if longest > self.max_gap_days else Severity.WARNING
            report.add(
                severity,
                "missing_observations",
                f"{len(internal_missing)} missing observation(s) inside the series; "
                f"longest run {longest} day(s)",
                symbol=symbol,
                count=len(internal_missing),
                longest_run=int(longest),
            )

        # --- suspicious returns ---
        returns = adj.loc[window].pct_change(fill_method=None).dropna()
        extreme = returns[returns.abs() > self.max_daily_move]
        if len(extreme) > 0:
            worst = extreme.abs().idxmax()
            report.add(
                Severity.WARNING,
                "abnormal_return",
                f"{len(extreme)} day(s) with |return| > {self.max_daily_move:.0%}; "
                f"largest {extreme.loc[worst]:+.1%} on {pd.Timestamp(worst).date()} "
                "(often an unadjusted split or distribution)",
                symbol=symbol,
                count=len(extreme),
                worst_date=str(pd.Timestamp(worst).date()),
                worst_return=float(extreme.loc[worst]),
            )

        # --- flat series (possible stale feed) ---
        if len(returns) >= 20:
            zero_run = _longest_run_of(returns.to_numpy() == 0.0)
            if zero_run >= 10:
                report.add(
                    Severity.WARNING,
                    "flat_series",
                    f"{zero_run} consecutive days with zero return - the feed may be stale",
                    symbol=symbol,
                    longest_zero_run=int(zero_run),
                )

        # --- volume ---
        vol = volume.loc[window].dropna()
        if len(vol) == 0:
            report.add(
                Severity.INFO,
                "no_volume",
                "no volume data; market-impact costs will fall back to a flat estimate",
                symbol=symbol,
            )
        else:
            zero_vol = vol[vol <= 0.0]
            if len(zero_vol) > len(vol) * 0.05:
                report.add(
                    Severity.WARNING,
                    "zero_volume",
                    f"{len(zero_vol)} bar(s) with non-positive volume ({len(zero_vol)/len(vol):.1%})",
                    symbol=symbol,
                    count=len(zero_vol),
                )

        # --- adjusted vs raw coherence ---
        ratio = (adj.loc[window] / close.loc[window]).dropna()
        if len(ratio) > 1 and (ratio.diff().dropna() < -1e-6).any():
            n_down = int((ratio.diff().dropna() < -1e-6).sum())
            report.add(
                Severity.INFO,
                "adjustment_factor_decrease",
                f"the adj_close/close factor decreases on {n_down} day(s); implied-dividend "
                "estimates on those days are set to zero",
                symbol=symbol,
                count=n_down,
            )

    def _check_cross_sectional(self, panel: PricePanel, report: ValidationReport) -> None:
        """Checks that involve more than one symbol."""
        coverage = panel.adj_close.notna().mean(axis=1)
        thin = coverage[coverage < 0.5]
        if len(thin) > 0:
            report.add(
                Severity.WARNING,
                "thin_cross_section",
                f"{len(thin)} day(s) where fewer than half the symbols have prices",
                n_days=len(thin),
                first=str(pd.Timestamp(thin.index[0]).date()),
            )
        if len(panel.symbols) < 2:
            report.add(
                Severity.WARNING,
                "single_asset_universe",
                "cross-sectional strategies require at least two assets",
            )

    # -- optional, explicit cleaning ------------------------------------------

    def clean(self, panel: PricePanel, report: ValidationReport | None = None) -> PricePanel:
        """Return a panel with short internal gaps forward-filled.

        Only gaps of at most ``max_fill_days`` consecutive days *inside* a
        series are filled; leading (pre-inception) and long gaps are left as
        ``NaN``. Every fill is counted and reported - Atlas never fills silently.
        """
        filled_fields = {}
        total_filled = 0
        for name in ("open", "high", "low", "close", "adj_close"):
            frame = panel.field(name)
            filled = frame.ffill(limit=self.max_fill_days)
            # Never fill before a series has started.
            filled = filled.where(frame.ffill().notna())
            total_filled += int((filled.notna() & frame.isna()).to_numpy().sum())
            filled_fields[name] = filled
        # Volume is not forward-filled: an unknown volume is zero traded, not a
        # repeat of yesterday's.
        filled_fields["volume"] = panel.volume

        if report is not None and total_filled:
            report.add(
                Severity.INFO,
                "gaps_filled",
                f"forward-filled {total_filled} price cell(s) across gaps of at most "
                f"{self.max_fill_days} day(s)",
                cells_filled=int(total_filled),
            )
        log.info("panel cleaned", extra={"context": {"cells_filled": total_filled}})
        return PricePanel(
            **filled_fields, source=dict(panel.source), downloaded_at=dict(panel.downloaded_at)
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _consecutive_runs(index: pd.DatetimeIndex) -> list[list[pd.Timestamp]]:
    """Group a sorted DatetimeIndex into runs of consecutive business days."""
    if len(index) == 0:
        return []
    runs: list[list[pd.Timestamp]] = [[index[0]]]
    for prev, cur in pairwise(index):
        gap = len(pd.bdate_range(prev, cur)) - 1
        if gap <= 1:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return runs


def _longest_run_of(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array."""
    if mask.size == 0:
        return 0
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best
