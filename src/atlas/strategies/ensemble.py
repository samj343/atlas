"""Signal combination across strategies.

For asset :math:`i` on date :math:`t` the combined signal is

.. math::

    s_{i,t} = \\sum_{k=1}^{K} a_k(r_t)\\, c_{i,k,t}\\, s_{i,k,t}

where :math:`s_{i,k,t}` is strategy :math:`k`'s signal, :math:`a_k(r_t)` is the
strategy weight in the regime :math:`r_t` prevailing at *t*, and
:math:`c_{i,k,t}` is an optional per-observation confidence.

Design choices
--------------
* Strategy weights **sum to one before any risk scaling**, so the ensemble
  changes the *mix* of strategies, never the total risk. Risk scaling is a
  separate, explicit step in :mod:`atlas.portfolio`.
* Weights are read from configuration per regime. There is no in-sample
  optimisation of strategy weights - that is the fastest way to overfit an
  ensemble, and the walk-forward results would not survive it.
* Three modes are supported: ``regime`` (config weights per regime), ``fixed``
  (static config weights) and ``equal`` (the baseline every ensemble should be
  compared against).
* Turnover is damped by exponentially smoothing the combined signal and by a
  minimum-signal threshold.
* Every strategy's contribution to the final signal is recorded so attribution
  is exact rather than inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import EnsembleConfig, RegimeConfig, RegimeLabel
from atlas.exceptions import StrategyError
from atlas.logging_utils import get_logger
from atlas.regimes.base import RegimeResult
from atlas.strategies.base import SignalResult, Strategy

log = get_logger(__name__)

__all__ = ["CombinedSignal", "SignalEnsemble"]


@dataclass
class CombinedSignal:
    """Output of the ensemble.

    Attributes
    ----------
    signal:
        Combined, smoothed signal per asset (date x symbol), in ``[-clip, clip]``.
    raw:
        The combined signal before smoothing and thresholding.
    contributions:
        Mapping strategy name -> that strategy's weighted contribution to the
        combined signal (date x symbol). ``sum(contributions) == raw``.
    strategy_weights:
        Applied strategy weights per date (date x strategy).
    regime_labels:
        The regime in force on each date, when regime weighting is active.
    """

    signal: pd.DataFrame
    raw: pd.DataFrame
    contributions: dict[str, pd.DataFrame] = field(default_factory=dict)
    strategy_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_labels: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))

    @property
    def index(self) -> pd.DatetimeIndex:
        """Date index of the combined signal."""
        return pd.DatetimeIndex(self.signal.index)

    def last(self) -> pd.Series:
        """The combined signal on the most recent date."""
        if self.signal.empty:
            return pd.Series(dtype="float64")
        return self.signal.iloc[-1]

    def contribution_frame(self) -> pd.DataFrame:
        """Total absolute contribution per strategy per date (date x strategy)."""
        if not self.contributions:
            return pd.DataFrame()
        return pd.DataFrame(
            {name: frame.abs().sum(axis=1) for name, frame in self.contributions.items()}
        )

    def contribution_share(self) -> pd.DataFrame:
        """Each strategy's share of the total absolute signal, per date."""
        totals = self.contribution_frame()
        if totals.empty:
            return totals
        denom = totals.sum(axis=1).replace(0.0, np.nan)
        return totals.div(denom, axis=0)

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        share = self.contribution_share()
        out: dict[str, Any] = {
            "n_dates": len(self.signal),
            "n_symbols": int(self.signal.shape[1]),
            "mean_abs_signal": float(self.signal.abs().to_numpy().mean()) if not self.signal.empty else 0.0,
            "mean_active": float((self.signal.abs() > 1e-9).sum(axis=1).mean())
            if not self.signal.empty
            else 0.0,
        }
        if not share.empty:
            out.update(
                {f"share_{k}": round(float(v), 4) for k, v in share.mean(skipna=True).items()}
            )
        return out


class SignalEnsemble:
    """Combine multiple strategies into one signal per asset.

    Parameters
    ----------
    strategies:
        The strategy instances to combine.
    config:
        Ensemble settings (mode, smoothing, thresholds).
    regime_config:
        Per-regime strategy weights. Required when ``config.mode == 'regime'``.
    base_weights:
        Static weights used in ``fixed`` mode; defaults to each strategy's own
        configured weight.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        config: EnsembleConfig,
        *,
        regime_config: RegimeConfig | None = None,
        base_weights: dict[str, float] | None = None,
    ) -> None:
        if not strategies:
            raise StrategyError("an ensemble needs at least one strategy")
        self.strategies = list(strategies)
        self.config = config
        self.regime_config = regime_config
        if config.mode == "regime" and regime_config is None:
            raise StrategyError("ensemble mode 'regime' requires a regime configuration")

        self.base_weights = self._normalize(
            base_weights or {s.name: s.base_weight for s in strategies}
        )
        self._log = log.bind(mode=config.mode, n_strategies=len(strategies))

    # -- weights --------------------------------------------------------------

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        """Scale non-negative weights to sum to one; fall back to equal weight."""
        clean = {k: max(0.0, float(v)) for k, v in weights.items()}
        total = sum(clean.values())
        if total <= 0.0:
            n = len(clean)
            return dict.fromkeys(clean, 1.0 / n) if n else {}
        return {k: v / total for k, v in clean.items()}

    def weights_for_regime(self, regime: RegimeLabel | str | None) -> dict[str, float]:
        """Return the strategy weights that apply in ``regime``.

        The returned weights always sum to one over the strategies present in the
        ensemble, whatever the mode.
        """
        names = [s.name for s in self.strategies]

        if self.config.mode == "equal":
            return {n: 1.0 / len(names) for n in names}

        if self.config.mode == "fixed" or regime is None or self.regime_config is None:
            return self._normalize({n: self.base_weights.get(n, 0.0) for n in names})

        block = self.regime_config.weights_for(regime)
        regime_weights = block.strategy_weights()
        return self._normalize({n: regime_weights.get(n, self.base_weights.get(n, 0.0)) for n in names})

    def weight_frame(self, index: pd.DatetimeIndex, regimes: RegimeResult | None) -> pd.DataFrame:
        """Build the (date x strategy) weight frame for the whole sample."""
        names = [s.name for s in self.strategies]

        if self.config.mode != "regime" or regimes is None:
            weights = self.weights_for_regime(None)
            return pd.DataFrame(
                np.tile([weights[n] for n in names], (len(index), 1)), index=index, columns=names
            )

        labels = regimes.labels.reindex(index).ffill()
        # Cache one weight vector per distinct regime rather than recomputing per row.
        unique = set(labels.dropna().unique())
        table = {lab: self.weights_for_regime(lab) for lab in unique}
        default = self.weights_for_regime(RegimeLabel.UNKNOWN)
        rows = [
            [table.get(lab, default)[n] for n in names]
            for lab in labels.tolist()
        ]
        return pd.DataFrame(rows, index=index, columns=names)

    # -- combination ----------------------------------------------------------

    def combine(
        self,
        signals: dict[str, SignalResult],
        *,
        regimes: RegimeResult | None = None,
    ) -> CombinedSignal:
        """Combine precomputed per-strategy signals.

        Parameters
        ----------
        signals:
            Mapping of strategy name -> :class:`~atlas.strategies.base.SignalResult`.
        regimes:
            Regime classification, required for ``regime`` mode.
        """
        present = [s.name for s in self.strategies if s.name in signals]
        if not present:
            raise StrategyError("no strategy signals were supplied to the ensemble")

        template = signals[present[0]].signal
        index = pd.DatetimeIndex(template.index)
        columns = list(template.columns)

        weights = self.weight_frame(index, regimes)

        contributions: dict[str, pd.DataFrame] = {}
        for name in present:
            result = signals[name]
            sig = result.signal.reindex(index=index, columns=columns).fillna(0.0)
            if self.config.use_confidence:
                conf = result.confidence.reindex(index=index, columns=columns).fillna(0.0)
                # Confidence dampens but never fully mutes a strategy: a floor of
                # 0.25 keeps a low-confidence view from silently disappearing.
                conf = 0.25 + 0.75 * conf.clip(0.0, 1.0)
                sig = sig * conf
            contributions[name] = sig.mul(weights[name], axis=0)

        raw = sum(contributions.values())
        raw = pd.DataFrame(raw, index=index, columns=columns)

        smoothed = self._smooth(raw)
        smoothed = smoothed.clip(-self.config.clip, self.config.clip)
        if self.config.min_signal > 0.0:
            smoothed = smoothed.where(smoothed.abs() >= self.config.min_signal, other=0.0)

        combined = CombinedSignal(
            signal=smoothed,
            raw=raw,
            contributions=contributions,
            strategy_weights=weights,
            regime_labels=(
                regimes.labels.reindex(index).ffill() if regimes is not None else pd.Series(dtype=object)
            ),
        )
        self._log.info("signals combined", extra={"context": combined.summary()})
        return combined

    def run(
        self,
        prices: pd.DataFrame,
        features: Any = None,
        *,
        regimes: RegimeResult | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> CombinedSignal:
        """Compute every strategy's signals and combine them."""
        signals = {
            s.name: s.compute(prices, features, as_of_date=as_of_date) for s in self.strategies
        }
        return self.combine(signals, regimes=regimes)

    # -- helpers --------------------------------------------------------------

    def _smooth(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Exponentially smooth the combined signal to damp turnover.

        ``alpha == 1`` disables smoothing. ``ewm`` is causal, so smoothing cannot
        introduce look-ahead.
        """
        alpha = self.config.signal_smoothing_alpha
        if alpha >= 1.0:
            return frame
        return frame.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    def describe(self) -> dict[str, Any]:
        """Describe the ensemble configuration and its members."""
        return {
            "mode": self.config.mode,
            "use_confidence": self.config.use_confidence,
            "smoothing_alpha": self.config.signal_smoothing_alpha,
            "base_weights": self.base_weights,
            "strategies": [s.describe() for s in self.strategies],
        }
