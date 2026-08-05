"""Optional clustering-based regime detection.

This is a **comparison model**, not a replacement. The rules-based detector is
the baseline because it is interpretable and its inputs are economically
meaningful; a clustering model is harder to reason about and easier to overfit.

Two rules make the comparison honest:

1. The model is **fitted on a training slice only** and then used to *predict*
   labels on later data. Fitting on the whole sample would leak future
   information into every historical label.
2. It is **disabled by default** (``clustering.enabled: false``) and never
   silently substitutes for the rules-based labels. Enabling it is an explicit
   choice, and :meth:`ClusteringRegimeDetector.compare_with` exists to report how
   the two disagree.

Cluster indices carry no inherent meaning, so each cluster is mapped onto the
nearest interpretable regime label using the sign of its mean trend score and
whether its mean volatility is above the training median.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atlas.config import ClusteringRegimeConfig, RegimeConfig, RegimeLabel
from atlas.data.features import FeatureSet
from atlas.data.loader import PricePanel
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger
from atlas.regimes.base import RegimeDetector, RegimeResult
from atlas.regimes.rules_based import RulesBasedRegimeDetector

log = get_logger(__name__)

__all__ = ["ClusteringRegimeDetector"]


class ClusteringRegimeDetector(RegimeDetector):
    """K-means regime detector fitted on a training slice.

    Parameters
    ----------
    config:
        The full :class:`~atlas.config.RegimeConfig`.
    train_end:
        Last date used for fitting. Everything after it is predicted only. When
        omitted, the first ``min_train_obs`` observations are used.
    """

    name = "clustering"

    def __init__(self, config: RegimeConfig, *, train_end: pd.Timestamp | None = None) -> None:
        self.config = config
        self.settings: ClusteringRegimeConfig = config.clustering
        self.train_end = train_end
        self._model: Any | None = None
        self._scaler: Any | None = None
        self._cluster_labels: dict[int, RegimeLabel] = {}
        self._baseline = RulesBasedRegimeDetector(config)

    # -- detection ------------------------------------------------------------

    def detect(
        self,
        panel: PricePanel,
        features: FeatureSet | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> RegimeResult:
        """Fit on the training slice, then label the whole sample.

        Raises
        ------
        InsufficientDataError
            If there are fewer than ``min_train_obs`` usable training rows.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise InsufficientDataError(
                "scikit-learn is required for the clustering regime detector"
            ) from exc

        # Reuse the rules-based indicators so both detectors see identical inputs
        # and any difference is attributable to the model, not the features.
        baseline = self._baseline.detect(panel, features, as_of_date)
        indicators = baseline.features

        columns = [c for c in self.settings.features if c in indicators.columns]
        if not columns:
            raise InsufficientDataError(
                f"none of the configured clustering features {self.settings.features} "
                f"are available; got {list(indicators.columns)}"
            )

        usable = indicators[columns].dropna()
        if len(usable) < self.settings.min_train_obs:
            raise InsufficientDataError(
                f"clustering needs {self.settings.min_train_obs} complete observations, "
                f"got {len(usable)}"
            )

        train_end = self.train_end or usable.index[self.settings.min_train_obs - 1]
        train = usable.loc[usable.index <= train_end]
        if len(train) < self.settings.min_train_obs:
            raise InsufficientDataError(
                f"training slice ending {pd.Timestamp(train_end).date()} has {len(train)} rows, "
                f"below the {self.settings.min_train_obs} required"
            )

        values = train.to_numpy(dtype="float64")
        if self.settings.standardize:
            self._scaler = StandardScaler().fit(values)
            values = self._scaler.transform(values)

        self._model = KMeans(
            n_clusters=self.settings.n_clusters,
            random_state=self.settings.random_state,
            n_init=10,
        ).fit(values)

        self._cluster_labels = self._map_clusters(train, self._model.labels_)

        # Predict across the whole sample. Only the fit used training data.
        all_values = usable.to_numpy(dtype="float64")
        if self._scaler is not None:
            all_values = self._scaler.transform(all_values)
        assignments = self._model.predict(all_values)

        # Build the full label series in Python rather than via
        # `reindex().ffill().fillna(RegimeLabel.UNKNOWN)`. RegimeLabel subclasses
        # `str`, and pandas' fillna treats a str-like fill value as a character
        # sequence, silently writing a truncated string such as 'RegimeL' into
        # the gaps.
        predicted = dict(
            zip(
                usable.index,
                (self._cluster_labels.get(int(c), RegimeLabel.UNKNOWN) for c in assignments),
                strict=True,
            )
        )
        label_values: list[RegimeLabel] = []
        carried = RegimeLabel.UNKNOWN
        for timestamp in indicators.index:
            carried = predicted.get(timestamp, carried)
            label_values.append(carried)
        labels = pd.Series(label_values, index=indicators.index, dtype=object)

        enriched = indicators.copy()
        enriched["cluster"] = pd.Series(assignments, index=usable.index).reindex(indicators.index)

        result = RegimeResult(labels=labels, features=enriched, detector=self.name)
        log.info(
            "clustering regimes detected",
            extra={
                "context": {
                    **result.summary(),
                    "train_end": str(pd.Timestamp(train_end).date()),
                    "train_observations": len(train),
                }
            },
        )
        return result

    # -- helpers --------------------------------------------------------------

    def _map_clusters(self, train: pd.DataFrame, assignments: np.ndarray) -> dict[int, RegimeLabel]:
        """Map cluster indices onto interpretable regime labels.

        A cluster index is arbitrary; the mapping uses the cluster's mean trend
        score (direction) and whether its mean volatility exceeds the training
        median (turbulence), so labels mean the same thing as in the rules-based
        detector.
        """
        frame = train.copy()
        frame["_cluster"] = assignments
        trend_column = "trend_score" if "trend_score" in frame.columns else frame.columns[0]
        vol_column = "volatility" if "volatility" in frame.columns else frame.columns[-1]
        median_volatility = float(frame[vol_column].median())

        mapping: dict[int, RegimeLabel] = {}
        for cluster, group in frame.groupby("_cluster"):
            bullish = float(group[trend_column].mean()) >= 0.0
            turbulent = float(group[vol_column].mean()) > median_volatility
            if bullish:
                mapping[int(cluster)] = (
                    RegimeLabel.BULL_HIGH_VOL if turbulent else RegimeLabel.BULL_LOW_VOL
                )
            else:
                mapping[int(cluster)] = (
                    RegimeLabel.BEAR_HIGH_VOL if turbulent else RegimeLabel.BEAR_LOW_VOL
                )
        return mapping

    def compare_with(
        self,
        panel: PricePanel,
        features: FeatureSet | None = None,
    ) -> pd.DataFrame:
        """Compare clustering labels against the rules-based baseline.

        Returns a frame with both label series, an agreement flag and a summary
        of where they disagree. This is the evidence a clustering model must
        provide before anyone considers preferring it.
        """
        baseline = self._baseline.detect(panel, features).as_strings()
        clustered = self.detect(panel, features).as_strings()
        common = baseline.index.intersection(clustered.index)
        frame = pd.DataFrame(
            {"rules_based": baseline.loc[common], "clustering": clustered.loc[common]}
        )
        frame["agree"] = frame["rules_based"] == frame["clustering"]
        log.info(
            "regime detector comparison",
            extra={
                "context": {
                    "n_observations": len(frame),
                    "agreement_rate": round(float(frame["agree"].mean()), 4),
                }
            },
        )
        return frame
