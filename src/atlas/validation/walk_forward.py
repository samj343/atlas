"""Walk-forward out-of-sample validation.

Time-series data must never be shuffled. Atlas therefore validates with rolling
windows that always move forward in time:

.. code-block:: text

    |<-- train 5y -->|<- val 1y ->|<- test 1y ->|
                     ^ embargo    ^ embargo
         |<-- train 5y -->|<- val 1y ->|<- test 1y ->|      (+1 year)
              |<-- train 5y -->|<- val 1y ->|<- test 1y ->| (+2 years)

For every window:

1. Each candidate parameter set is run over ``train + validation`` in one
   backtest, and metrics are computed separately for the two slices.
2. Candidates are scored on the **validation** slice with a multi-objective
   function that penalises drawdown, turnover and train/validation instability -
   not on return alone.
3. The winner's parameters are **frozen** and re-run over
   ``train + validation + test``; only the **test** slice is reported.
4. The window advances.

The concatenated test slices form the out-of-sample record, which is the only
result worth taking seriously. In-sample and validation performance are reported
alongside it precisely so the gap between them is visible.

Embargo
-------
An ``embargo_days`` gap is inserted between adjacent slices. A feature with a
252-day lookback computed on the first day of the test window would otherwise
overlap the validation period; the embargo removes that overlap from the
reported test returns.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from atlas.backtest.engine import BacktestEngine, BacktestResult
from atlas.backtest.metrics import PerformanceMetrics, compute_metrics
from atlas.config import AtlasConfig, WalkForwardConfig
from atlas.data.features import FeatureEngineer, FeatureSet
from atlas.data.loader import PricePanel
from atlas.exceptions import ConfigurationError, InsufficientDataError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "WalkForwardResult",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "apply_overrides",
    "expand_grid",
]


# ---------------------------------------------------------------------------
# Parameter grid helpers
# ---------------------------------------------------------------------------


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of candidate dictionaries.

    Keys use dotted paths such as ``"trend.slow_ma"`` or
    ``"portfolio.target_volatility"``.
    """
    if not grid:
        return [{}]
    keys = sorted(grid)
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, values, strict=True)) for values in combos]


def apply_overrides(config: AtlasConfig, overrides: dict[str, Any]) -> AtlasConfig:
    """Return a copy of ``config`` with dotted-path ``overrides`` applied.

    Supported prefixes:

    * ``trend.*``, ``mean_reversion.*``, ``cross_sectional_momentum.*`` -
      strategy parameters;
    * ``portfolio.*`` - portfolio limits and the volatility target;
    * ``ensemble.*``, ``backtest.*``, ``covariance.*``, ``volatility.*``,
      ``regime.*`` (detector).

    A deep copy is made, so the caller's configuration is never mutated - which
    matters because a walk-forward run evaluates dozens of candidates.
    """
    new = config.model_copy(deep=True)
    strategy_blocks = {"trend", "mean_reversion", "cross_sectional_momentum"}

    for path, value in overrides.items():
        head, _, tail = path.partition(".")
        if not tail:
            raise ValueError(f"override {path!r} must use a dotted path, e.g. 'trend.slow_ma'")
        if head in strategy_blocks:
            target = getattr(new.strategies.strategies, head)
        elif head == "portfolio":
            target = new.risk.portfolio
        elif head == "covariance":
            target = new.risk.covariance
        elif head == "volatility":
            target = new.risk.volatility
        elif head == "ensemble":
            target = new.strategies.ensemble
        elif head == "backtest":
            target = new.strategies.backtest
        elif head == "regime":
            target = new.regimes.detector
        elif head == "costs":
            target = new.execution.costs
        else:
            raise ValueError(f"unknown override prefix {head!r} in {path!r}")
        if not hasattr(target, tail):
            raise ValueError(f"{head} has no parameter {tail!r}")
        setattr(target, tail, value)
    return new


# ---------------------------------------------------------------------------
# Window and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardWindow:
    """One train / validation / test split."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "window": self.index,
            "train_start": self.train_start.date(),
            "train_end": self.train_end.date(),
            "validation_start": self.validation_start.date(),
            "validation_end": self.validation_end.date(),
            "test_start": self.test_start.date(),
            "test_end": self.test_end.date(),
        }


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output.

    Attributes
    ----------
    windows:
        The window definitions.
    selections:
        One row per window: chosen parameters, selection score, and the train,
        validation and test metrics.
    test_returns:
        Concatenated out-of-sample daily returns across all test slices.
    candidate_scores:
        Every candidate's validation score in every window, for inspection.
    """

    windows: list[WalkForwardWindow] = field(default_factory=list)
    selections: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    candidate_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_metrics: PerformanceMetrics | None = None
    in_sample_metrics: PerformanceMetrics | None = None
    validation_metrics: PerformanceMetrics | None = None
    regime_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    @property
    def oos_equity_curve(self) -> pd.Series:
        """Equity curve compounded from the concatenated test returns."""
        if self.test_returns.empty:
            return pd.Series(dtype="float64")
        return (1.0 + self.test_returns).cumprod()

    def parameter_stability(self) -> pd.DataFrame:
        """How often each parameter value was selected across windows."""
        if self.selections.empty:
            return pd.DataFrame()
        param_cols = [c for c in self.selections.columns if c.startswith("param_")]
        rows = []
        for col in param_cols:
            counts = self.selections[col].value_counts()
            for value, count in counts.items():
                rows.append(
                    {
                        "parameter": col.removeprefix("param_"),
                        "value": value,
                        "times_selected": int(count),
                        "share": float(count / len(self.selections)),
                    }
                )
        return pd.DataFrame(rows).sort_values(["parameter", "times_selected"], ascending=[True, False])

    def degradation(self) -> dict[str, float]:
        """In-sample vs out-of-sample gap - the number that matters most.

        A large positive gap means the selection process was fitting noise.
        """
        if self.selections.empty:
            return {}
        return {
            "mean_train_sharpe": float(self.selections["train_sharpe"].mean()),
            "mean_validation_sharpe": float(self.selections["validation_sharpe"].mean()),
            "mean_test_sharpe": float(self.selections["test_sharpe"].mean()),
            "train_minus_test_sharpe": float(
                self.selections["train_sharpe"].mean() - self.selections["test_sharpe"].mean()
            ),
            "validation_minus_test_sharpe": float(
                self.selections["validation_sharpe"].mean() - self.selections["test_sharpe"].mean()
            ),
            "pct_windows_positive_test": float((self.selections["test_return"] > 0).mean()),
        }

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        out: dict[str, Any] = {
            "n_windows": len(self.windows),
            "oos_observations": len(self.test_returns),
        }
        if self.test_metrics:
            out.update(
                {
                    "oos_total_return": round(self.test_metrics.values.get("total_return", 0.0), 4),
                    "oos_cagr": round(self.test_metrics.cagr, 4),
                    "oos_sharpe": round(self.test_metrics.sharpe, 3),
                    "oos_max_drawdown": round(self.test_metrics.max_drawdown, 4),
                }
            )
        out.update({k: round(v, 4) for k, v in self.degradation().items()})
        return out


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Run walk-forward parameter selection and out-of-sample evaluation."""

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.wf: WalkForwardConfig = config.validation.walk_forward

    # -- window construction --------------------------------------------------

    def build_windows(self, dates: pd.DatetimeIndex) -> list[WalkForwardWindow]:
        """Construct the walk-forward windows spanning ``dates``.

        Raises
        ------
        InsufficientDataError
            If the history is too short for even one complete window.
        """
        wf = self.wf
        if len(dates) == 0:
            raise InsufficientDataError("cannot build walk-forward windows from an empty calendar")

        start, end = dates.min(), dates.max()
        train = (
            pd.DateOffset(years=int(wf.train_years))
            if float(wf.train_years).is_integer()
            else pd.Timedelta(days=int(wf.train_years * 365.25))
        )
        validation = pd.Timedelta(days=int(wf.validation_years * 365.25))
        test = pd.Timedelta(days=int(wf.test_years * 365.25))
        step = pd.Timedelta(days=int(wf.step_years * 365.25))
        embargo = pd.Timedelta(days=wf.embargo_days)

        windows: list[WalkForwardWindow] = []
        train_start = start
        index = 0
        while True:
            train_end = train_start + train
            validation_start = train_end + embargo
            validation_end = validation_start + validation
            test_start = validation_end + embargo
            test_end = test_start + test
            if test_start > end:
                break
            # A truncated final test window is still usable, provided it holds a
            # meaningful number of observations.
            effective_test_end = min(test_end, end)
            n_test = len(dates[(dates >= test_start) & (dates <= effective_test_end)])
            n_train = len(dates[(dates >= train_start) & (dates <= train_end)])
            if n_train >= wf.min_train_observations and n_test >= 20:
                windows.append(
                    WalkForwardWindow(
                        index=index,
                        train_start=train_start,
                        train_end=train_end,
                        validation_start=validation_start,
                        validation_end=validation_end,
                        test_start=test_start,
                        test_end=effective_test_end,
                    )
                )
                index += 1
            train_start = train_start + step

        if not windows:
            span_years = (end - start).days / 365.25
            required = wf.train_years + wf.validation_years + wf.test_years
            raise InsufficientDataError(
                f"history spans {span_years:.1f} years but a walk-forward window needs at least "
                f"{required:.1f} years; shorten train/validation/test in configs/validation.yaml"
            )
        return windows

    # -- scoring --------------------------------------------------------------

    def score(
        self, validation: PerformanceMetrics, train: PerformanceMetrics
    ) -> tuple[float, bool, str]:
        """Score a candidate. Returns ``(score, accepted, reason_if_rejected)``.

        The score deliberately never rewards raw return: a candidate wins by
        producing risk-adjusted performance that survives from training into
        validation, at acceptable turnover.
        """
        objective = self.wf.objective
        drawdown = validation.values.get("max_drawdown", 0.0)
        turnover = validation.values.get("annual_turnover", 0.0)

        if drawdown > objective.max_acceptable_drawdown:
            return (
                float("-inf"),
                False,
                f"validation drawdown {drawdown:.1%} exceeds the "
                f"{objective.max_acceptable_drawdown:.0%} limit",
            )
        if turnover > objective.max_acceptable_annual_turnover:
            return (
                float("-inf"),
                False,
                f"annual turnover {turnover:.1f}x exceeds the "
                f"{objective.max_acceptable_annual_turnover:.1f}x limit",
            )

        instability = abs(train.sharpe - validation.sharpe)
        value = (
            objective.w_sharpe * validation.sharpe
            - objective.w_drawdown * drawdown
            - objective.w_turnover * turnover
            - objective.w_stability * instability
        )
        return float(value), True, ""

    # -- main entry point -----------------------------------------------------

    def validate_grid(self) -> list[dict[str, Any]]:
        """Expand the parameter grid and check every candidate is constructible.

        A grid value that cannot produce a valid configuration - for example a
        ``trend.slow_ma`` at or below ``trend.fast_ma`` - would otherwise be
        discovered only after the run had already spent minutes on the other
        candidates, and would silently shrink the search.

        Raises
        ------
        ConfigurationError
            Listing every invalid combination and why it failed.
        """
        candidates = expand_grid(self.wf.parameter_grid)
        invalid: list[str] = []
        for candidate in candidates:
            try:
                apply_overrides(self.config, candidate)
            except (ValueError, TypeError) as exc:
                reason = str(exc).splitlines()[0]
                invalid.append(f"{candidate}: {reason}")
        if invalid:
            raise ConfigurationError(
                "the walk-forward parameter grid contains combinations that cannot produce a "
                "valid configuration:\n  - " + "\n  - ".join(invalid)
            )
        return candidates

    def run(
        self,
        panel: PricePanel,
        *,
        features: FeatureSet | None = None,
        progress: bool = True,
    ) -> WalkForwardResult:
        """Run the full walk-forward procedure over ``panel``."""
        candidates = self.validate_grid()
        windows = self.build_windows(panel.dates)
        log.info(
            "walk-forward starting",
            extra={
                "context": {
                    "n_windows": len(windows),
                    "n_candidates": len(candidates),
                    "backtests": len(windows) * (len(candidates) + 1),
                }
            },
        )

        selections: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        test_return_pieces: list[pd.Series] = []
        train_return_pieces: list[pd.Series] = []
        validation_return_pieces: list[pd.Series] = []
        regime_rows: list[dict[str, Any]] = []
        warnings: list[str] = []

        for window in windows:
            best: dict[str, Any] | None = None

            for candidate in candidates:
                try:
                    result = self._backtest(
                        panel, candidate, window.train_start, window.validation_end, features
                    )
                except (InsufficientDataError, ValueError) as exc:
                    log.warning(
                        "candidate failed; skipping",
                        extra={"context": {"window": window.index, "params": candidate, "error": str(exc)}},
                    )
                    continue

                train_metrics = self._slice_metrics(
                    result, window.train_start, window.train_end, "train"
                )
                validation_metrics = self._slice_metrics(
                    result, window.validation_start, window.validation_end, "validation"
                )
                value, accepted, reason = self.score(validation_metrics, train_metrics)

                candidate_rows.append(
                    {
                        "window": window.index,
                        **{f"param_{k}": v for k, v in candidate.items()},
                        "score": value,
                        "accepted": accepted,
                        "rejection_reason": reason,
                        "train_sharpe": train_metrics.sharpe,
                        "validation_sharpe": validation_metrics.sharpe,
                        "validation_max_drawdown": validation_metrics.max_drawdown,
                        "validation_annual_turnover": validation_metrics.values.get(
                            "annual_turnover", float("nan")
                        ),
                    }
                )
                if accepted and (best is None or value > best["score"]):
                    best = {
                        "score": value,
                        "params": candidate,
                        "train": train_metrics,
                        "validation": validation_metrics,
                    }

            if best is None:
                message = (
                    f"window {window.index} ({window.test_start.date()}-{window.test_end.date()}): "
                    "no candidate passed the drawdown/turnover filters; the window is skipped and "
                    "contributes no out-of-sample return"
                )
                log.warning(message)
                warnings.append(message)
                continue

            # --- freeze the parameters and evaluate the unseen test slice ---
            frozen = best["params"]
            test_result = self._backtest(
                panel, frozen, window.train_start, window.test_end, features
            )
            test_metrics = self._slice_metrics(
                test_result, window.test_start, window.test_end, "test"
            )
            test_slice = self._returns_slice(test_result, window.test_start, window.test_end)
            test_return_pieces.append(test_slice)
            train_return_pieces.append(
                self._returns_slice(test_result, window.train_start, window.train_end)
            )
            validation_return_pieces.append(
                self._returns_slice(test_result, window.validation_start, window.validation_end)
            )

            if test_result.regimes is not None:
                distribution = test_result.regimes.slice(
                    window.test_start, window.test_end
                ).distribution()
                regime_rows.append({"window": window.index, **distribution.to_dict()})

            selections.append(
                {
                    **window.as_dict(),
                    **{f"param_{k}": v for k, v in frozen.items()},
                    "selection_score": best["score"],
                    "train_sharpe": best["train"].sharpe,
                    "train_return": best["train"].values.get("total_return", 0.0),
                    "validation_sharpe": best["validation"].sharpe,
                    "validation_return": best["validation"].values.get("total_return", 0.0),
                    "test_sharpe": test_metrics.sharpe,
                    "test_return": test_metrics.values.get("total_return", 0.0),
                    "test_max_drawdown": test_metrics.max_drawdown,
                    "test_annual_turnover": test_metrics.values.get("annual_turnover", float("nan")),
                    "test_transaction_costs": test_metrics.values.get(
                        "total_transaction_costs", float("nan")
                    ),
                    "test_observations": len(test_slice),
                }
            )
            if progress:
                log.info(
                    "walk-forward window complete",
                    extra={
                        "context": {
                            "window": f"{window.index + 1}/{len(windows)}",
                            "test": f"{window.test_start.date()}..{window.test_end.date()}",
                            "params": frozen,
                            "test_sharpe": round(test_metrics.sharpe, 3),
                        }
                    },
                )

        risk_free = self.config.validation.benchmarks.risk_free_rate
        oos = (
            pd.concat(test_return_pieces).sort_index()
            if test_return_pieces
            else pd.Series(dtype="float64")
        )
        # Overlapping training slices would double-count observations.
        oos = oos[~oos.index.duplicated(keep="first")]

        walk_forward_result = WalkForwardResult(
            windows=windows,
            selections=pd.DataFrame(selections),
            test_returns=oos,
            candidate_scores=pd.DataFrame(candidate_rows),
            test_metrics=compute_metrics(
                oos, risk_free_rate=risk_free, label="Atlas (out-of-sample)"
            )
            if len(oos)
            else None,
            in_sample_metrics=self._concat_metrics(
                train_return_pieces, risk_free, "Atlas (in-sample)"
            ),
            validation_metrics=self._concat_metrics(
                validation_return_pieces, risk_free, "Atlas (validation)"
            ),
            regime_distribution=pd.DataFrame(regime_rows).fillna(0.0) if regime_rows else pd.DataFrame(),
            warnings=warnings,
        )
        log.info("walk-forward complete", extra={"context": walk_forward_result.summary()})
        return walk_forward_result

    # -- helpers --------------------------------------------------------------

    def _backtest(
        self,
        panel: PricePanel,
        overrides: dict[str, Any],
        start: pd.Timestamp,
        end: pd.Timestamp,
        features: FeatureSet | None,
    ) -> BacktestResult:
        """Run one backtest with ``overrides`` applied, over ``[start, end]``.

        The panel is *not* truncated at ``start``: history before the window is
        needed to warm up the lookbacks, and using it is not look-ahead because
        it lies strictly in the past. Only the traded window is restricted.
        """
        candidate_config = apply_overrides(self.config, overrides)
        engine = BacktestEngine(candidate_config)
        window_features = features
        if window_features is None or self._features_stale(overrides):
            window_features = FeatureEngineer.from_config(candidate_config).build(panel)
        return engine.run(panel, features=window_features, start=start, end=end)

    @staticmethod
    def _features_stale(overrides: dict[str, Any]) -> bool:
        """True when an override changes a window a precomputed feature set lacks."""
        feature_affecting = {"slow_ma", "fast_ma", "momentum_lookback", "zscore_lookback", "lookback"}
        return any(path.split(".")[-1] in feature_affecting for path in overrides)

    @staticmethod
    def _returns_slice(
        result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Daily returns restricted to ``[start, end]``."""
        returns = result.returns
        if returns.empty:
            return returns
        mask = (returns.index >= start) & (returns.index <= end)
        return returns.loc[mask]

    def _slice_metrics(
        self, result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp, label: str
    ) -> PerformanceMetrics:
        """Metrics for one slice of a backtest, including its turnover and costs."""
        returns = self._returns_slice(result, start, end)
        snapshots = result.snapshots
        turnover = None
        if not snapshots.empty and "turnover" in snapshots.columns:
            mask = (snapshots.index >= start) & (snapshots.index <= end)
            turnover = snapshots.loc[mask, "turnover"]
        costs = None
        if not result.fills.empty:
            stamps = pd.DatetimeIndex(result.fills["timestamp"])
            in_window = (stamps >= start) & (stamps <= end)
            costs = float(result.fills.loc[in_window, "total_cost"].sum())
        return compute_metrics(
            returns,
            risk_free_rate=self.config.validation.benchmarks.risk_free_rate,
            label=label,
            turnover=turnover,
            costs=costs,
        )

    def _concat_metrics(
        self, pieces: list[pd.Series], risk_free: float, label: str
    ) -> PerformanceMetrics | None:
        """Metrics for concatenated, de-duplicated return slices."""
        if not pieces:
            return None
        series = pd.concat(pieces).sort_index()
        series = series[~series.index.duplicated(keep="first")]
        if series.empty:
            return None
        return compute_metrics(series, risk_free_rate=risk_free, label=label)
