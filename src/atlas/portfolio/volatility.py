"""Per-asset volatility estimation.

Volatility enters Atlas in three places: inverse-volatility position scaling,
the portfolio volatility target, and the strategies' own risk gates. All of them
use this module so the estimate is consistent and configurable in one place.

Two estimators are provided:

``rolling``
    Equal-weighted sample standard deviation over a fixed window. Simple and
    unbiased, but reacts slowly and drops observations abruptly when they leave
    the window.
``ewma``
    Exponentially weighted standard deviation. Reacts faster to a volatility
    regime change and decays smoothly, which is usually preferable for sizing.

Both are strictly backward-looking, and both floor the result at
``min_annual_volatility`` so a near-constant series (a T-bill ETF, a halted
instrument) cannot produce an unbounded inverse-volatility weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import VolatilityConfig
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["VolatilityEstimator", "annualize_volatility", "inverse_volatility_weights"]


def annualize_volatility(daily_vol: float | pd.Series, periods_per_year: int = 252):
    """Scale a daily volatility to an annual figure by ``sqrt(periods_per_year)``."""
    return daily_vol * np.sqrt(periods_per_year)


class VolatilityEstimator:
    """Estimate trailing annualised volatility per asset.

    Parameters
    ----------
    config:
        A :class:`~atlas.config.VolatilityConfig`.
    """

    def __init__(self, config: VolatilityConfig) -> None:
        self.config = config

    # -- estimation -----------------------------------------------------------

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return a frame of trailing annualised volatilities (date x symbol).

        Parameters
        ----------
        returns:
            Daily simple returns.
        """
        if returns.empty:
            return returns.copy()

        cfg = self.config
        if cfg.method == "ewma":
            vol = returns.ewm(
                halflife=cfg.halflife, min_periods=cfg.min_periods, adjust=True
            ).std(bias=False)
        elif cfg.method == "rolling":
            vol = returns.rolling(window=cfg.lookback, min_periods=cfg.min_periods).std(ddof=1)
        else:  # pragma: no cover - guarded by the config model
            raise ValueError(f"unknown volatility method {cfg.method!r}")

        annual = vol * np.sqrt(cfg.annualization_factor)
        return annual.clip(lower=cfg.min_annual_volatility)

    def estimate_latest(self, returns: pd.DataFrame) -> pd.Series:
        """Return the most recent volatility estimate per asset.

        Raises
        ------
        InsufficientDataError
            If no asset has enough observations for an estimate.
        """
        estimates = self.estimate(returns)
        if estimates.empty:
            raise InsufficientDataError("cannot estimate volatility from an empty return frame")
        latest = estimates.iloc[-1]
        if latest.isna().all():
            raise InsufficientDataError(
                f"no asset has the {self.config.min_periods} observations required for a "
                "volatility estimate"
            )
        return latest

    def realized_portfolio_volatility(
        self, portfolio_returns: pd.Series, window: int | None = None
    ) -> pd.Series:
        """Trailing annualised volatility of a portfolio return series."""
        w = window or self.config.lookback
        vol = portfolio_returns.rolling(window=w, min_periods=max(2, w // 4)).std(ddof=1)
        return vol * np.sqrt(self.config.annualization_factor)


def inverse_volatility_weights(
    signals: pd.Series,
    volatilities: pd.Series,
    *,
    min_volatility: float = 0.005,
) -> pd.Series:
    """Scale signals by the inverse of each asset's volatility.

    .. math:: \\tilde w_i = s_i / \\sigma_i

    This equalises the *risk* each signal contributes rather than the capital,
    so a 1.0 signal on a 30%-volatility equity ETF does not dominate a 1.0
    signal on a 5%-volatility bond ETF.

    Assets with a missing volatility estimate receive a zero weight - Atlas
    prefers not to hold an asset over sizing it with a guessed risk number.
    """
    aligned_vol = volatilities.reindex(signals.index)
    safe_vol = aligned_vol.clip(lower=min_volatility)
    weights = signals / safe_vol
    return weights.where(aligned_vol.notna(), other=0.0).fillna(0.0)
