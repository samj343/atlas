"""Cross-sectional (relative-strength) momentum.

Economic rationale
------------------
Assets that have outperformed their peers over the past 6-12 months tend to keep
outperforming over the following months (Jegadeesh & Titman 1993; Asness,
Moskowitz & Pedersen, *Value and Momentum Everywhere*, JF 2013). Unlike
time-series momentum, this strategy is a *relative* bet: it ranks the universe
and buys the leaders, so it can stay invested even when every asset is falling
unless it is explicitly combined with a directional overlay.

Scoring
-------
Assets are ranked by risk-adjusted trailing return:

.. math::

    \\text{score}_i = \\frac{r_i(t-L, t-S)}{\\sigma_i}

where :math:`L` is ``lookback``, :math:`S` is ``skip_recent_days`` and
:math:`\\sigma_i` is trailing annualised volatility. Skipping the most recent
month is the standard correction for the short-term reversal effect, which
otherwise contaminates a 12-month momentum measure.

Asset-class neutralisation
--------------------------
A universe with four US equity ETFs and one REIT ETF will, in an equity bull
market, rank all four equity funds at the top - the portfolio then holds one
economic bet in four disguises. With ``neutralize_asset_class`` on, scores are
demeaned within each asset class before ranking, so an asset competes against
its own peers first. This deliberately trades some raw momentum capture for
diversification.

Ranking and tie-breaking
------------------------
Ranks use the *average* method, so tied assets share the average of the ranks
they span; this keeps the cross-sectional mean at zero and avoids an arbitrary
ordering. Assets without sufficient history, without a current price, or without
a defined volatility estimate are dropped from the cross-section entirely rather
than being ranked last, so their absence does not distort the surviving ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import CrossSectionalMomentumConfig
from atlas.data.features import FeatureSet, realized_volatility
from atlas.strategies.base import SignalResult, Strategy, register_strategy

__all__ = ["CrossSectionalMomentumStrategy"]


@register_strategy
class CrossSectionalMomentumStrategy(Strategy):
    """Rank-based relative-strength momentum with optional class neutralisation."""

    name = "cross_sectional_momentum"

    def __init__(
        self,
        config: CrossSectionalMomentumConfig,
        *,
        asset_classes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(config, asset_classes=asset_classes)
        self.config: CrossSectionalMomentumConfig = config

    @property
    def required_lookback(self) -> int:
        """History needed before the momentum measure is defined."""
        return max(self.config.lookback, self.config.volatility_lookback) + 1

    # -- computation ----------------------------------------------------------

    def _compute(self, prices: pd.DataFrame, features: FeatureSet) -> SignalResult:
        cfg = self.config

        # --- momentum measured from t-lookback to t-skip ----------------------
        # Both endpoints are in the past, so the measure is point-in-time.
        start = prices.shift(cfg.lookback)
        end = prices.shift(cfg.skip_recent_days)
        momentum = (end / start) - 1.0

        vol = self._lookup(
            features,
            f"volatility_{cfg.volatility_lookback}d",
            lambda: realized_volatility(prices.pct_change(fill_method=None), cfg.volatility_lookback),
        )
        # Use the volatility as of the same skip point so the numerator and
        # denominator describe the same window.
        vol_at_skip = vol.shift(cfg.skip_recent_days)
        vol_floor = vol_at_skip.clip(lower=0.01)
        score = momentum / vol_floor

        # --- eligibility -------------------------------------------------------
        # An asset must have a live price, a defined score, and enough history.
        history_count = prices.notna().cumsum()
        eligible = (
            prices.notna()
            & score.notna()
            & vol_at_skip.notna()
            & (history_count >= self.required_lookback)
        )
        score = score.where(eligible)

        # --- optional asset-class neutralisation -------------------------------
        if cfg.neutralize_asset_class and self.asset_classes:
            score = self._neutralize_by_class(score)

        # --- cross-sectional ranking -------------------------------------------
        n_valid = score.notna().sum(axis=1)
        ranks = score.rank(axis=1, method="average", na_option="keep", ascending=True)

        if cfg.scoring == "rank":
            signal = self._rank_score(ranks, n_valid, cfg)
        else:
            signal = self._top_k_score(ranks, n_valid, cfg)

        # Rows without enough assets to form a meaningful cross-section emit
        # nothing at all rather than a degenerate one-asset bet.
        sufficient = (n_valid >= cfg.min_assets).astype(float)
        signal = signal.mul(sufficient, axis=0)

        if cfg.long_only:
            signal = signal.clip(lower=0.0)

        # Confidence rises with cross-sectional dispersion: when every asset has
        # a similar score, the ranking carries little information.
        dispersion = score.std(axis=1, ddof=1)
        scaled_dispersion = (dispersion / dispersion.expanding(min_periods=60).median()).clip(0.0, 2.0)
        confidence = pd.DataFrame(
            np.repeat(
                (scaled_dispersion / 2.0).fillna(0.5).to_numpy()[:, None], len(prices.columns), axis=1
            ),
            index=prices.index,
            columns=prices.columns,
        )
        confidence = confidence.where(signal.abs() > 0.0, other=0.0)

        return SignalResult(
            strategy=self.name,
            signal=signal.fillna(0.0).clip(-1.0, 1.0),
            raw=score.fillna(0.0),
            confidence=confidence,
            diagnostics={
                "momentum": momentum,
                "risk_adjusted_score": score,
                "rank": ranks,
                "eligible": eligible.astype(float),
            },
        )

    # -- scoring variants -----------------------------------------------------

    @staticmethod
    def _rank_score(
        ranks: pd.DataFrame, n_valid: pd.Series, cfg: CrossSectionalMomentumConfig
    ) -> pd.DataFrame:
        """Continuous rank score in ``[-1, 1]``, zero-mean across the cross-section.

        The top ``top_k`` assets keep a positive score; the rest are set to zero
        (long-only) or scaled negatively when shorts are enabled.
        """
        denominator = (n_valid - 1).replace(0, np.nan)
        centred = 2.0 * (ranks.sub(1.0)).div(denominator, axis=0) - 1.0

        # Restrict to the top-k (and optionally bottom-k) names so the portfolio
        # stays concentrated in the strongest relative performers.
        top_cut = n_valid - cfg.top_k
        is_top = ranks.ge(top_cut + 1, axis=0)
        selected = centred.where(is_top, other=0.0)

        if not cfg.long_only and cfg.bottom_k > 0:
            is_bottom = ranks.le(cfg.bottom_k)
            selected = selected.where(~is_bottom, other=centred)

        # Rescale so the strongest selected name reaches +1 on each date.
        peak = selected.abs().max(axis=1).replace(0.0, np.nan)
        return selected.div(peak, axis=0).fillna(0.0)

    @staticmethod
    def _top_k_score(
        ranks: pd.DataFrame, n_valid: pd.Series, cfg: CrossSectionalMomentumConfig
    ) -> pd.DataFrame:
        """Equal-weight signal: +1 for the top ``k``, -1 for the bottom ``k``."""
        top_cut = n_valid - cfg.top_k
        signal = ranks.ge(top_cut + 1, axis=0).astype(float)
        if not cfg.long_only and cfg.bottom_k > 0:
            signal = signal - ranks.le(cfg.bottom_k).astype(float)
        return signal.where(ranks.notna(), other=0.0)

    # -- helpers --------------------------------------------------------------

    def _neutralize_by_class(self, score: pd.DataFrame) -> pd.DataFrame:
        """Demean scores within each asset class.

        Classes containing a single symbol are left unchanged - demeaning a
        one-member group would zero it out and permanently exclude the asset.
        """
        groups: dict[str, list[str]] = {}
        for symbol in score.columns:
            groups.setdefault(self.asset_classes.get(symbol, "unclassified"), []).append(symbol)

        out = score.copy()
        for members in groups.values():
            if len(members) < 2:
                continue
            block = score[members]
            out[members] = block.sub(block.mean(axis=1), axis=0)
        return out

    @staticmethod
    def _lookup(features: FeatureSet, name: str, fallback) -> pd.DataFrame:
        """Return a precomputed feature, or compute it if the set lacks it."""
        if name in features:
            return features.get(name)
        return fallback()
