"""Parameter-stability analysis.

A backtest result that depends on one exact parameter value is a curve fit. A
result that persists across a *neighbourhood* of values is at least consistent
with a real effect. This module sweeps one parameter at a time around the
baseline configuration and reports whether performance forms a broad plateau or
a lone spike.

Two flags are produced per parameter:

``fragile_gap``
    The best setting beats the median setting by more than
    ``fragility_sharpe_gap`` Sharpe units - the result depends on picking the
    single right value.
``fragile_dispersion``
    The coefficient of variation of Sharpe across the sweep exceeds
    ``cv_threshold`` - performance is unstable across the neighbourhood.

The sweep is also run gross of costs when ``include_gross`` is set, so a strategy
whose edge exists only before trading costs is exposed rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestEngine
from atlas.config import AtlasConfig
from atlas.data.features import FeatureEngineer, FeatureSet
from atlas.data.loader import PricePanel
from atlas.exceptions import AtlasError, ConfigurationError
from atlas.logging_utils import get_logger
from atlas.validation.walk_forward import apply_overrides

log = get_logger(__name__)

__all__ = ["ParameterStabilityAnalyzer", "StabilityResult"]


@dataclass
class StabilityResult:
    """Sweep results and fragility verdicts."""

    sweeps: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: list[str] = field(default_factory=list)

    @property
    def fragile_parameters(self) -> list[str]:
        """Parameters flagged fragile by either test."""
        if self.verdicts.empty:
            return []
        flagged = self.verdicts[
            self.verdicts["fragile_gap"] | self.verdicts["fragile_dispersion"]
        ]
        return [str(p) for p in flagged["parameter"]]

    def curve(self, parameter: str) -> pd.DataFrame:
        """Sweep results for one parameter, sorted by value."""
        if self.sweeps.empty:
            return pd.DataFrame()
        return (
            self.sweeps[self.sweeps["parameter"] == parameter]
            .sort_values("value")
            .reset_index(drop=True)
        )

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        return {
            "n_parameters": int(self.sweeps["parameter"].nunique()) if not self.sweeps.empty else 0,
            "n_runs": len(self.sweeps),
            "n_fragile": len(self.fragile_parameters),
            "fragile": self.fragile_parameters,
            "n_failures": len(self.failures),
        }


class ParameterStabilityAnalyzer:
    """Sweep individual parameters around the baseline configuration."""

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.settings = config.validation.parameter_stability

    def validate_sweep(self, grid: dict[str, list[Any]]) -> None:
        """Check every swept value can produce a valid configuration.

        A value that never constructs - ``trend.slow_ma`` at or below
        ``trend.fast_ma``, say - would otherwise be dropped mid-run with only a
        log warning, and the verdict would then be computed over fewer points
        than the sweep claims to cover. A stability verdict measured on a
        silently smaller neighbourhood is worse than no verdict at all.

        Raises
        ------
        ConfigurationError
            Listing every value that cannot be applied, and why.
        """
        invalid: list[str] = []
        for path, values in grid.items():
            for value in values:
                try:
                    apply_overrides(self.config, {path: value})
                except (ValueError, TypeError) as exc:
                    invalid.append(f"{path}={value}: {str(exc).splitlines()[0]}")
        if invalid:
            raise ConfigurationError(
                "the parameter-stability sweep contains values that cannot produce a valid "
                "configuration:\n  - " + "\n  - ".join(invalid)
            )

    def run(
        self,
        panel: PricePanel,
        *,
        parameters: dict[str, list[Any]] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        include_gross: bool | None = None,
    ) -> StabilityResult:
        """Run the one-at-a-time sweep.

        Parameters
        ----------
        panel:
            Price history.
        parameters:
            Override the configured sweep, as ``{dotted_path: [values]}``.
        start, end:
            Restrict the evaluation window.
        include_gross:
            Also run each setting with transaction costs zeroed, to measure
            cost sensitivity. Defaults to the configured value.
        """
        grid = parameters if parameters is not None else self.settings.parameters
        if not grid:
            raise AtlasError("no parameters configured for the stability sweep")
        self.validate_sweep(grid)
        gross_too = self.settings.include_gross if include_gross is None else include_gross

        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        total = sum(len(v) for v in grid.values())
        done = 0

        for path, values in grid.items():
            for value in values:
                done += 1
                try:
                    net = self._evaluate(panel, {path: value}, start, end, gross=False)
                except (AtlasError, ValueError) as exc:
                    message = f"{path}={value}: {exc}"
                    failures.append(message)
                    log.warning("stability run failed", extra={"context": {"detail": message}})
                    continue

                row = {"parameter": path, "value": value, **net}
                if gross_too:
                    try:
                        gross = self._evaluate(panel, {path: value}, start, end, gross=True)
                        row["gross_sharpe"] = gross["sharpe"]
                        row["gross_cagr"] = gross["cagr"]
                        row["cost_sharpe_impact"] = gross["sharpe"] - net["sharpe"]
                    except (AtlasError, ValueError) as exc:  # pragma: no cover - defensive
                        log.warning(
                            "gross-of-cost run failed",
                            extra={"context": {"parameter": path, "value": value, "error": str(exc)}},
                        )
                rows.append(row)
                log.info(
                    "stability sweep progress",
                    extra={
                        "context": {
                            "run": f"{done}/{total}",
                            "parameter": path,
                            "value": value,
                            "sharpe": round(row.get("sharpe", float("nan")), 3),
                        }
                    },
                )

        sweeps = pd.DataFrame(rows)
        result = StabilityResult(
            sweeps=sweeps, verdicts=self._verdicts(sweeps), failures=failures
        )
        log.info("parameter stability complete", extra={"context": result.summary()})
        return result

    # -- internals ------------------------------------------------------------

    def _evaluate(
        self,
        panel: PricePanel,
        overrides: dict[str, Any],
        start: Any,
        end: Any,
        *,
        gross: bool,
    ) -> dict[str, float]:
        """Run one backtest and extract the headline metrics."""
        config = apply_overrides(self.config, overrides)
        if gross:
            # Zero every cost component to isolate the strategy's gross edge.
            costs = config.execution.costs
            costs.commission_per_order = 0.0
            costs.commission_per_share = 0.0
            costs.min_commission = 0.0
            costs.half_spread_bps = 0.0
            costs.slippage_bps = 0.0
            costs.market_impact_enabled = False

        features: FeatureSet = FeatureEngineer.from_config(config).build(panel)
        result = BacktestEngine(config).run(panel, features=features, start=start, end=end)
        metrics = result.metrics
        if metrics is None:  # pragma: no cover - engine always computes metrics
            raise AtlasError("backtest produced no metrics")
        return {
            "sharpe": metrics.sharpe,
            "cagr": metrics.cagr,
            "volatility": metrics.values.get("annualized_volatility", float("nan")),
            "max_drawdown": metrics.max_drawdown,
            "calmar": metrics.values.get("calmar_ratio", float("nan")),
            "annual_turnover": metrics.values.get("annual_turnover", float("nan")),
            "total_costs": metrics.values.get("total_transaction_costs", float("nan")),
            "n_trades": metrics.values.get("n_trades", float("nan")),
        }

    def _verdicts(self, sweeps: pd.DataFrame) -> pd.DataFrame:
        """Judge each parameter's stability from its sweep."""
        if sweeps.empty:
            return pd.DataFrame()

        rows = []
        for parameter, group in sweeps.groupby("parameter"):
            sharpe = group["sharpe"].astype("float64")
            best, median = float(sharpe.max()), float(sharpe.median())
            spread = best - median
            mean = float(sharpe.mean())
            std = float(sharpe.std(ddof=1)) if len(sharpe) > 1 else 0.0
            cv = float(std / abs(mean)) if abs(mean) > 1e-9 else float("inf") if std > 0 else 0.0
            best_value = group.loc[sharpe.idxmax(), "value"]

            rows.append(
                {
                    "parameter": parameter,
                    "n_settings": len(group),
                    "best_value": best_value,
                    "best_sharpe": best,
                    "median_sharpe": median,
                    "worst_sharpe": float(sharpe.min()),
                    "sharpe_spread": spread,
                    "sharpe_cv": cv,
                    "mean_max_drawdown": float(group["max_drawdown"].mean()),
                    "fragile_gap": bool(spread > self.settings.fragility_sharpe_gap),
                    "fragile_dispersion": bool(cv > self.settings.cv_threshold),
                    "mean_cost_sharpe_impact": (
                        float(group["cost_sharpe_impact"].mean())
                        if "cost_sharpe_impact" in group.columns
                        else float("nan")
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        frame["verdict"] = np.where(
            frame["fragile_gap"] | frame["fragile_dispersion"],
            "fragile - performance depends on the specific value",
            "stable - performance persists across neighbouring values",
        )
        return frame.sort_values("sharpe_spread", ascending=False).reset_index(drop=True)
