"""Common strategy interface.

Every Atlas strategy is a pure function of *point-in-time* prices and features:
given data known at the close of day *t*, it emits a continuous conviction
signal in ``[-1, 1]`` per asset. Signals express **desired direction and
conviction, not portfolio weights** - turning them into weights is the job of
:mod:`atlas.portfolio`.

Two views of the same computation are exposed:

* :meth:`Strategy.compute` returns a :class:`SignalResult` holding wide frames
  (index = date, columns = symbol). This is the fast path used by the ensemble
  and the backtest engine.
* :meth:`Strategy.generate_signals` returns the tidy long record form required
  by the specification - one row per ``(date, symbol)`` carrying the raw signal,
  the normalised signal, a confidence score and strategy diagnostics.

Sub-classes implement :meth:`Strategy._compute`, which must respect the
point-in-time contract: a value at index *t* may only depend on data dated
``<= t``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.data.features import FeatureSet
from atlas.exceptions import StrategyError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "STRATEGY_REGISTRY",
    "SignalResult",
    "Strategy",
    "build_strategies",
    "coerce_features",
    "normalize_signal",
    "register_strategy",
    "soft_threshold",
]


# ---------------------------------------------------------------------------
# Signal container
# ---------------------------------------------------------------------------


@dataclass
class SignalResult:
    """Wide-format output of a single strategy.

    Attributes
    ----------
    strategy:
        Strategy name.
    signal:
        Normalised signal in ``[-1, 1]`` (date x symbol).
    raw:
        The pre-normalisation score, kept for diagnostics and attribution.
    confidence:
        Per-observation confidence in ``[0, 1]``. The ensemble may use this to
        down-weight a strategy where its own inputs are weak or unreliable.
    diagnostics:
        Named intermediate frames (component scores, filters, gates) that make
        the signal explainable rather than a black box.
    """

    strategy: str
    signal: pd.DataFrame
    raw: pd.DataFrame
    confidence: pd.DataFrame
    diagnostics: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal.empty and (self.signal.abs() > 1.0 + 1e-9).to_numpy().any():
            raise StrategyError(
                f"strategy {self.strategy!r} emitted a signal outside [-1, 1]; "
                "normalise before returning"
            )

    @property
    def index(self) -> pd.DatetimeIndex:
        """Date index of the signal frame."""
        return pd.DatetimeIndex(self.signal.index)

    @property
    def symbols(self) -> list[str]:
        """Symbols covered by the signal frame."""
        return list(self.signal.columns)

    def last(self) -> pd.Series:
        """Signal on the most recent date (empty Series when there is none)."""
        if self.signal.empty:
            return pd.Series(dtype="float64")
        return self.signal.iloc[-1]

    def slice(self, start: Any = None, end: Any = None) -> SignalResult:
        """Return a date-restricted copy."""

        def _cut(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame
            if start is not None:
                out = out.loc[out.index >= pd.Timestamp(start)]
            if end is not None:
                out = out.loc[out.index <= pd.Timestamp(end)]
            return out

        return SignalResult(
            strategy=self.strategy,
            signal=_cut(self.signal),
            raw=_cut(self.raw),
            confidence=_cut(self.confidence),
            diagnostics={k: _cut(v) for k, v in self.diagnostics.items()},
        )

    def to_records(self, *, dropna: bool = True) -> pd.DataFrame:
        """Return the tidy long record form.

        Columns: ``date``, ``symbol``, ``strategy``, ``raw_signal``,
        ``normalized_signal``, ``confidence`` plus one column per diagnostic.
        """
        pieces = {
            "raw_signal": self.raw,
            "normalized_signal": self.signal,
            "confidence": self.confidence,
            **{f"diag_{k}": v for k, v in self.diagnostics.items()},
        }
        frames = []
        for name, wide in pieces.items():
            if wide is None or wide.empty:
                continue
            stacked = wide.stack(future_stack=True).rename(name)
            frames.append(stacked)
        if not frames:
            return pd.DataFrame(
                columns=["date", "symbol", "strategy", "raw_signal", "normalized_signal", "confidence"]
            )
        out = pd.concat(frames, axis=1).reset_index()
        out.columns = ["date", "symbol", *out.columns[2:]]
        out.insert(2, "strategy", self.strategy)
        if dropna:
            out = out.dropna(subset=["normalized_signal"])
        return out.reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        active = self.signal.abs() > 1e-9
        return {
            "strategy": self.strategy,
            "n_dates": len(self.signal),
            "n_symbols": int(self.signal.shape[1]),
            "mean_abs_signal": float(self.signal.abs().stack(future_stack=True).mean(skipna=True))
            if not self.signal.empty
            else 0.0,
            "mean_active_positions": float(active.sum(axis=1).mean()) if not self.signal.empty else 0.0,
            "mean_confidence": float(self.confidence.stack(future_stack=True).mean(skipna=True))
            if not self.confidence.empty
            else 0.0,
        }


# ---------------------------------------------------------------------------
# Helpers shared by strategies
# ---------------------------------------------------------------------------


def normalize_signal(
    frame: pd.DataFrame, *, method: str = "tanh", scale: float = 1.0
) -> pd.DataFrame:
    """Map an unbounded score into ``[-1, 1]``.

    Parameters
    ----------
    frame:
        Raw scores.
    method:
        ``"tanh"`` squashes smoothly and never saturates abruptly;
        ``"clip"`` simply truncates at ``+/- scale``;
        ``"sign"`` returns the sign only.
    scale:
        Divisor applied before squashing. Larger values make the signal gentler.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    if method == "tanh":
        return np.tanh(frame / scale)
    if method == "clip":
        return (frame / scale).clip(-1.0, 1.0)
    if method == "sign":
        return np.sign(frame)
    raise ValueError(f"unknown normalisation method {method!r}")


def soft_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Zero out signals whose magnitude is below ``threshold``.

    Unlike a hard cut this rescales the surviving signal so the mapping stays
    continuous at the boundary, which avoids a jump in position size (and thus
    turnover) as a signal drifts across the threshold::

        f(s) = sign(s) * (|s| - k) / (1 - k)   for |s| > k, else 0
    """
    if threshold <= 0.0:
        return frame
    if threshold >= 1.0:
        raise ValueError("threshold must be below 1.0")
    magnitude = frame.abs()
    adjusted = (magnitude - threshold) / (1.0 - threshold)
    return np.sign(frame) * adjusted.clip(lower=0.0)


def coerce_features(features: FeatureSet | pd.DataFrame | None) -> FeatureSet:
    """Accept either a :class:`FeatureSet` or a MultiIndex-column frame.

    The specification types the ``features`` argument as ``pd.DataFrame``; Atlas
    uses a richer :class:`FeatureSet` internally. This helper accepts both so the
    public signature stays as specified without losing structure internally.
    """
    if features is None:
        return FeatureSet()
    if isinstance(features, FeatureSet):
        return features
    if isinstance(features, pd.DataFrame):
        if not isinstance(features.columns, pd.MultiIndex):
            raise StrategyError(
                "a plain features DataFrame must have MultiIndex (feature, symbol) columns; "
                "pass a FeatureSet or use FeatureSet.to_wide()"
            )
        names = list(dict.fromkeys(features.columns.get_level_values(0)))
        return FeatureSet(features={n: features[n] for n in names})
    raise StrategyError(f"unsupported features type: {type(features)!r}")


def _percentile_rank_expanding(frame: pd.DataFrame, min_periods: int = 252) -> pd.DataFrame:
    """Expanding-window percentile rank of each value within its own history.

    Strictly point-in-time: the rank at *t* uses only observations ``<= t``.
    Implemented with a rank-of-the-last-value trick that is O(n log n) per column
    rather than the O(n^2) of a naive expanding ``apply``.
    """
    out = {}
    for col in frame.columns:
        series = frame[col]
        values = series.to_numpy(dtype="float64")
        ranks = np.full(values.shape, np.nan)
        seen: list[float] = []
        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            # Insert while keeping `seen` sorted, then read off the position.
            lo, hi = 0, len(seen)
            while lo < hi:
                mid = (lo + hi) // 2
                if seen[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            seen.insert(lo, v)
            if len(seen) >= min_periods:
                ranks[i] = lo / max(len(seen) - 1, 1)
        out[col] = pd.Series(ranks, index=series.index)
    return pd.DataFrame(out, index=frame.index)


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------


class Strategy(abc.ABC):
    """Abstract base class for signal-generating strategies.

    Parameters
    ----------
    config:
        The strategy's own pydantic configuration model.
    asset_classes:
        Optional mapping of symbol -> asset class, used by strategies that
        neutralise exposure across classes.
    """

    #: Machine-readable strategy name; must be unique.
    name: str = "strategy"

    def __init__(self, config: Any, *, asset_classes: dict[str, str] | None = None) -> None:
        self.config = config
        self.asset_classes = dict(asset_classes or {})
        self._log = log.bind(strategy=self.name)

    # -- required interface ---------------------------------------------------

    @property
    @abc.abstractmethod
    def required_lookback(self) -> int:
        """Trading days of history required before the strategy can emit a signal."""

    @abc.abstractmethod
    def _compute(self, prices: pd.DataFrame, features: FeatureSet) -> SignalResult:
        """Compute the strategy's signals. Implemented by subclasses."""

    # -- public API -----------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        features: FeatureSet | pd.DataFrame | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> SignalResult:
        """Return the strategy's wide-format :class:`SignalResult`.

        Parameters
        ----------
        prices:
            Wide price frame (index = date, columns = symbol). Should be the
            adjusted-close series so that signals reflect total return.
        features:
            A :class:`~atlas.data.features.FeatureSet` (preferred) or a
            MultiIndex-column DataFrame.
        as_of_date:
            When given, prices and features are truncated at this date first, so
            the result is exactly what the strategy would have produced in real
            time on that day.
        """
        if prices is None or prices.empty:
            raise StrategyError(f"{self.name}: cannot generate signals from empty prices")

        fs = coerce_features(features)
        if as_of_date is not None:
            ts = pd.Timestamp(as_of_date)
            prices = prices.loc[prices.index <= ts]
            fs = fs.as_of(ts)
            if prices.empty:
                raise StrategyError(f"{self.name}: no price history at or before {ts.date()}")

        if len(prices) < self.required_lookback:
            self._log.debug(
                "insufficient history; emitting neutral signals",
                extra={"context": {"have": len(prices), "need": self.required_lookback}},
            )
            zeros = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            return SignalResult(
                strategy=self.name,
                signal=zeros,
                raw=zeros.copy(),
                confidence=zeros.copy(),
            )

        result = self._compute(prices, fs)
        # Defensive: never emit a signal for an asset with no price that day.
        valid = prices.notna()
        result.signal = result.signal.where(valid, other=0.0).fillna(0.0)
        result.confidence = result.confidence.where(valid, other=0.0).fillna(0.0).clip(0.0, 1.0)
        return result

    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: FeatureSet | pd.DataFrame | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return signals in the tidy long record form.

        One row per ``(date, symbol)`` with ``strategy``, ``raw_signal``,
        ``normalized_signal``, ``confidence`` and diagnostic columns.
        """
        return self.compute(prices, features, as_of_date).to_records()

    def signal_matrix(
        self,
        prices: pd.DataFrame,
        features: FeatureSet | pd.DataFrame | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return only the normalised wide signal frame."""
        return self.compute(prices, features, as_of_date).signal

    def latest_signal(
        self,
        prices: pd.DataFrame,
        features: FeatureSet | pd.DataFrame | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Return the signal for the most recent available date."""
        return self.compute(prices, features, as_of_date).last()

    # -- introspection --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the strategy is switched on in configuration."""
        return bool(getattr(self.config, "enabled", True))

    @property
    def base_weight(self) -> float:
        """The strategy's static ensemble weight before regime adjustment."""
        return float(getattr(self.config, "weight", 0.0))

    def parameters(self) -> dict[str, Any]:
        """Return the strategy's parameters as a plain dictionary."""
        if hasattr(self.config, "model_dump"):
            return self.config.model_dump(mode="json")
        return dict(vars(self.config))

    def describe(self) -> dict[str, Any]:
        """Return name, docstring summary and parameters."""
        doc = (self.__class__.__doc__ or "").strip().split("\n\n")[0]
        return {"name": self.name, "description": " ".join(doc.split()), "parameters": self.parameters()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"

    # -- shared utilities available to subclasses -----------------------------

    @staticmethod
    def _volatility_gate(
        volatility: pd.DataFrame,
        percentile_cap: float,
        *,
        min_periods: int = 252,
        floor: float = 0.25,
    ) -> pd.DataFrame:
        """Return a multiplicative gate in ``[floor, 1]`` that shrinks the signal
        when trailing volatility sits in an extreme percentile of its own history.

        The percentile is computed on an expanding window, so it never uses
        future volatility observations.
        """
        pct = _percentile_rank_expanding(volatility, min_periods=min_periods)
        # Below the cap the gate is 1; above it the gate decays linearly to `floor`.
        excess = ((pct - percentile_cap) / max(1.0 - percentile_cap, 1e-6)).clip(0.0, 1.0)
        gate = 1.0 - (1.0 - floor) * excess
        return gate.fillna(1.0)

    def _asset_class_frame(self, symbols: list[str]) -> pd.Series:
        """Series mapping symbol -> asset class for the given symbols."""
        return pd.Series(
            {s: self.asset_classes.get(s, "unclassified") for s in symbols}, name="asset_class"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator registering a strategy under its ``name``."""
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def build_strategies(config: Any, *, only_enabled: bool = True) -> list[Strategy]:
    """Instantiate every configured strategy.

    Parameters
    ----------
    config:
        An :class:`~atlas.config.AtlasConfig`.
    only_enabled:
        Skip strategies whose ``enabled`` flag is False.
    """
    # Imported here so the registry is populated before it is read.
    from atlas.strategies import (  # noqa: F401
        cross_sectional_momentum,
        mean_reversion,
        trend,
    )

    asset_classes = config.asset_class_map
    strategies: list[Strategy] = []
    for name, strategy_cfg in config.strategies.strategies.as_dict().items():
        if only_enabled and not getattr(strategy_cfg, "enabled", False):
            continue
        cls = STRATEGY_REGISTRY.get(name)
        if cls is None:
            raise StrategyError(f"no strategy registered under the name {name!r}")
        strategies.append(cls(strategy_cfg, asset_classes=asset_classes))
    if not strategies:
        raise StrategyError("no strategies are enabled; check configs/strategies.yaml")
    return strategies
