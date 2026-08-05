"""Regime-detection interface and result container.

A regime detector maps market history onto a small set of interpretable states.
Atlas uses regimes for two things only:

1. tilting the ensemble's strategy weights, and
2. scaling the portfolio's risk budget.

It deliberately does *not* use regimes as a return forecast.

Like every other signal in Atlas, a regime label at date *t* may only depend on
data dated ``<= t``. The rules-based detector satisfies this by construction; the
clustering detector satisfies it by fitting on a training slice and then only
*predicting* on later data.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import pandas as pd

from atlas.config import RegimeLabel
from atlas.data.features import FeatureSet
from atlas.data.loader import PricePanel

__all__ = ["RegimeDetector", "RegimeResult"]


@dataclass
class RegimeResult:
    """Output of a regime detector.

    Attributes
    ----------
    labels:
        Series of :class:`~atlas.config.RegimeLabel` values indexed by date.
    features:
        The continuous indicators behind the classification (trend score,
        volatility, volatility percentile, breadth, average correlation,
        drawdown), stored so the label is explainable and auditable.
    detector:
        Name of the detector that produced the result.
    """

    labels: pd.Series
    features: pd.DataFrame
    detector: str = "unknown"

    @property
    def index(self) -> pd.DatetimeIndex:
        """Date index of the labels."""
        return pd.DatetimeIndex(self.labels.index)

    def label_at(self, timestamp: pd.Timestamp | str) -> RegimeLabel:
        """Return the regime in force on ``timestamp``.

        Falls back to the most recent earlier label, and to
        :attr:`RegimeLabel.UNKNOWN` when there is none.
        """
        ts = pd.Timestamp(timestamp)
        if ts in self.labels.index:
            value = self.labels.loc[ts]
        else:
            earlier = self.labels.loc[self.labels.index <= ts]
            if earlier.empty:
                return RegimeLabel.UNKNOWN
            value = earlier.iloc[-1]
        return value if isinstance(value, RegimeLabel) else RegimeLabel(str(value))

    def as_strings(self) -> pd.Series:
        """Labels as plain strings, convenient for grouping and plotting."""
        return self.labels.map(lambda x: x.value if isinstance(x, RegimeLabel) else str(x))

    def distribution(self) -> pd.Series:
        """Fraction of observations spent in each regime."""
        return self.as_strings().value_counts(normalize=True).sort_index()

    def transitions(self) -> pd.DataFrame:
        """Count of transitions from one regime to another."""
        s = self.as_strings()
        pairs = pd.DataFrame({"from": s.shift(1), "to": s}).dropna()
        pairs = pairs[pairs["from"] != pairs["to"]]
        if pairs.empty:
            return pd.DataFrame()
        return (
            pairs.groupby(["from", "to"]).size().rename("count").reset_index().sort_values(
                "count", ascending=False
            )
        )

    def episodes(self) -> pd.DataFrame:
        """Contiguous regime episodes with their start, end and length."""
        s = self.as_strings()
        if s.empty:
            return pd.DataFrame(columns=["regime", "start", "end", "days"])
        block = (s != s.shift(1)).cumsum()
        rows = []
        for _, group in s.groupby(block):
            rows.append(
                {
                    "regime": group.iloc[0],
                    "start": group.index[0],
                    "end": group.index[-1],
                    "days": len(group),
                }
            )
        return pd.DataFrame(rows)

    def slice(self, start: Any = None, end: Any = None) -> RegimeResult:
        """Return a date-restricted copy."""
        labels, features = self.labels, self.features
        if start is not None:
            labels = labels.loc[labels.index >= pd.Timestamp(start)]
            features = features.loc[features.index >= pd.Timestamp(start)]
        if end is not None:
            labels = labels.loc[labels.index <= pd.Timestamp(end)]
            features = features.loc[features.index <= pd.Timestamp(end)]
        return RegimeResult(labels=labels, features=features, detector=self.detector)

    def to_records(self) -> pd.DataFrame:
        """Tidy frame with the label and every underlying indicator."""
        out = self.features.copy()
        out.insert(0, "regime", self.as_strings())
        out.insert(0, "detector", self.detector)
        return out.reset_index().rename(columns={"index": "date"})

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        dist = self.distribution()
        return {
            "detector": self.detector,
            "n_observations": len(self.labels),
            "n_transitions": len(self.transitions()) if len(self.labels) > 1 else 0,
            **{f"pct_{k}": round(float(v), 4) for k, v in dist.items()},
        }


class RegimeDetector(abc.ABC):
    """Abstract base class for regime detectors."""

    #: Machine-readable detector name.
    name: str = "detector"

    @abc.abstractmethod
    def detect(
        self,
        panel: PricePanel,
        features: FeatureSet | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> RegimeResult:
        """Classify every date in ``panel`` into a regime.

        Parameters
        ----------
        panel:
            Price history.
        features:
            Optional precomputed features; detectors fall back to computing what
            they need.
        as_of_date:
            Truncate the history at this date before classifying.
        """

    def current_regime(
        self,
        panel: PricePanel,
        features: FeatureSet | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> RegimeLabel:
        """Return the regime in force on the most recent available date."""
        result = self.detect(panel, features, as_of_date)
        if result.labels.empty:
            return RegimeLabel.UNKNOWN
        last = result.labels.iloc[-1]
        return last if isinstance(last, RegimeLabel) else RegimeLabel(str(last))
