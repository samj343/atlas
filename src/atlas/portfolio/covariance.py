"""Covariance estimation for portfolio risk.

Estimating a covariance matrix from daily returns is the least reliable step in
portfolio construction: with :math:`N` assets there are :math:`N(N+1)/2`
parameters and rarely enough independent observations to pin them down. The
sample estimator is unbiased but noisy, and its smallest eigenvalues are biased
toward zero, which is exactly the direction that an optimiser exploits.

Three estimators are provided:

``sample``
    Plain sample covariance over a trailing window. Included as a baseline.
``shrinkage``
    Ledoit-Wolf shrinkage toward a scaled identity target. The analytic optimal
    intensity is used unless one is configured. This is the default because it
    is well conditioned by construction and needs no tuning.
``ewma``
    Exponentially weighted covariance, which tracks a changing correlation
    structure faster than an equal-weighted window.

Every estimate passes through :func:`nearest_positive_definite`, so downstream
code can assume the matrix is usable. Volatilities are floored, and a ridge is
added when the matrix is still not positive definite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas.config import CovarianceConfig
from atlas.exceptions import InsufficientDataError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "CovarianceEstimator",
    "CovarianceResult",
    "correlation_from_covariance",
    "nearest_positive_definite",
    "portfolio_volatility",
    "risk_contributions",
]


@dataclass(frozen=True)
class CovarianceResult:
    """An estimated covariance matrix and its provenance.

    Attributes
    ----------
    matrix:
        Annualised covariance matrix (symbols x symbols).
    method:
        Estimator actually used - may differ from the requested one if a
        fallback was triggered.
    n_observations:
        Number of return observations behind the estimate.
    shrinkage:
        Applied shrinkage intensity, when relevant.
    repaired:
        True when the matrix had to be repaired to become positive definite.
    """

    matrix: pd.DataFrame
    method: str
    n_observations: int
    shrinkage: float | None = None
    repaired: bool = False

    @property
    def symbols(self) -> list[str]:
        """Symbols covered by the matrix."""
        return list(self.matrix.columns)

    @property
    def volatilities(self) -> pd.Series:
        """Implied annualised volatilities (the square-rooted diagonal)."""
        return pd.Series(np.sqrt(np.diag(self.matrix.to_numpy())), index=self.matrix.index)

    def correlation(self) -> pd.DataFrame:
        """The implied correlation matrix."""
        return correlation_from_covariance(self.matrix)

    def condition_number(self) -> float:
        """Condition number of the matrix; large values indicate instability."""
        if self.matrix.empty:
            return float("nan")
        eigenvalues = np.linalg.eigvalsh(self.matrix.to_numpy())
        smallest = float(np.min(eigenvalues))
        if smallest <= 0:
            return float("inf")
        return float(np.max(eigenvalues) / smallest)


class CovarianceEstimator:
    """Estimate an annualised covariance matrix from daily returns."""

    def __init__(self, config: CovarianceConfig) -> None:
        self.config = config

    # -- public API -----------------------------------------------------------

    def estimate(
        self, returns: pd.DataFrame, *, symbols: list[str] | None = None
    ) -> CovarianceResult:
        """Estimate the covariance matrix from the trailing window of ``returns``.

        Parameters
        ----------
        returns:
            Daily simple returns (date x symbol). Only the trailing
            ``config.lookback`` rows are used.
        symbols:
            Restrict the estimate to these symbols; defaults to every column
            with enough data.

        Raises
        ------
        InsufficientDataError
            If fewer than ``config.min_observations`` usable rows remain.
        """
        cfg = self.config
        frame = returns if symbols is None else returns.reindex(columns=symbols)
        window = frame.tail(cfg.lookback)

        # Drop assets that are mostly missing over the window; they would
        # otherwise force the pairwise-complete estimate to be near-degenerate.
        coverage = window.notna().mean()
        usable = [str(c) for c in window.columns if coverage.get(c, 0.0) >= 0.8]
        dropped = [str(c) for c in window.columns if c not in usable]
        if dropped:
            log.debug("dropping thin columns from covariance", extra={"context": {"symbols": dropped}})
        window = window[usable].dropna(how="any")

        if len(window) < cfg.min_observations:
            raise InsufficientDataError(
                f"covariance needs {cfg.min_observations} complete observations, got {len(window)}"
            )
        if window.shape[1] == 0:
            raise InsufficientDataError("no asset has sufficient coverage for a covariance estimate")

        method = cfg.method
        if method == "sample":
            cov, intensity = self._sample(window), None
        elif method == "ewma":
            cov, intensity = self._ewma(window), None
        elif method == "shrinkage":
            cov, intensity = self._shrinkage(window)
        else:  # pragma: no cover - guarded by the config model
            raise ValueError(f"unknown covariance method {method!r}")

        annual = cov * 252.0
        annual, repaired = self._condition(annual)

        result = CovarianceResult(
            matrix=annual,
            method=method,
            n_observations=len(window),
            shrinkage=intensity,
            repaired=repaired,
        )
        if repaired:
            log.warning(
                "covariance matrix repaired to positive definite",
                extra={"context": {"method": method, "n_obs": result.n_observations}},
            )
        return result

    def estimate_or_fallback(
        self, returns: pd.DataFrame, *, symbols: list[str] | None = None
    ) -> CovarianceResult:
        """Estimate a covariance matrix, degrading gracefully rather than failing.

        Falls back, in order, to shrinkage estimation and then to a diagonal
        matrix built from whatever volatility estimates are available. A diagonal
        fallback is conservative: it ignores diversification benefits, so the
        portfolio volatility target binds sooner rather than later.
        """
        try:
            return self.estimate(returns, symbols=symbols)
        except InsufficientDataError as exc:
            log.warning(
                "covariance estimation failed; falling back to a diagonal matrix",
                extra={"context": {"error": str(exc)}},
            )

        frame = returns if symbols is None else returns.reindex(columns=symbols)
        window = frame.tail(self.config.lookback)
        vol = window.std(ddof=1) * np.sqrt(252.0)
        vol = vol.fillna(self.config.min_annual_volatility).clip(
            lower=self.config.min_annual_volatility
        )
        matrix = pd.DataFrame(np.diag(vol.to_numpy() ** 2), index=vol.index, columns=vol.index)
        return CovarianceResult(
            matrix=matrix,
            method="diagonal_fallback",
            n_observations=int(window.notna().any(axis=1).sum()),
            repaired=True,
        )

    # -- estimators -----------------------------------------------------------

    @staticmethod
    def _sample(window: pd.DataFrame) -> pd.DataFrame:
        """Plain sample covariance."""
        return window.cov(ddof=1)

    def _ewma(self, window: pd.DataFrame) -> pd.DataFrame:
        """Exponentially weighted covariance with the configured half-life."""
        halflife = self.config.ewma_halflife
        decay = 0.5 ** (1.0 / halflife)
        n = len(window)
        # Weight the most recent observation most heavily.
        weights = decay ** np.arange(n - 1, -1, -1)
        weights = weights / weights.sum()

        values = window.to_numpy(dtype="float64")
        mean = np.average(values, axis=0, weights=weights)
        centred = values - mean
        cov = (centred * weights[:, None]).T @ centred
        # Bias correction for weighted samples.
        cov = cov / max(1.0 - np.sum(weights**2), 1e-8)
        return pd.DataFrame(cov, index=window.columns, columns=window.columns)

    def _shrinkage(self, window: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Ledoit-Wolf shrinkage toward a scaled-identity target."""
        values = window.to_numpy(dtype="float64")
        n, p = values.shape

        configured = self.config.shrinkage_intensity
        if configured is not None:
            sample = np.cov(values, rowvar=False, ddof=1)
            target = np.eye(p) * (np.trace(sample) / p)
            shrunk = configured * target + (1.0 - configured) * sample
            return (
                pd.DataFrame(shrunk, index=window.columns, columns=window.columns),
                float(configured),
            )

        try:
            from sklearn.covariance import LedoitWolf

            estimator = LedoitWolf(assume_centered=False).fit(values)
            return (
                pd.DataFrame(
                    estimator.covariance_, index=window.columns, columns=window.columns
                ),
                float(estimator.shrinkage_),
            )
        except Exception as exc:
            log.warning(
                "Ledoit-Wolf unavailable; using the analytic shrinkage formula",
                extra={"context": {"error": str(exc)}},
            )

        # Analytic Ledoit-Wolf toward a scaled identity, implemented directly.
        sample = np.cov(values, rowvar=False, ddof=1)
        mu = np.trace(sample) / p
        target = mu * np.eye(p)
        centred = values - values.mean(axis=0)
        # Expected squared deviation of the sample estimator from its own mean.
        phi = sum(
            np.sum((np.outer(centred[i], centred[i]) - sample) ** 2) for i in range(n)
        ) / (n**2)
        gamma = np.sum((sample - target) ** 2)
        intensity = float(np.clip(phi / gamma if gamma > 0 else 0.0, 0.0, 1.0))
        shrunk = intensity * target + (1.0 - intensity) * sample
        return pd.DataFrame(shrunk, index=window.columns, columns=window.columns), intensity

    # -- conditioning ---------------------------------------------------------

    def _condition(self, matrix: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Floor volatilities and repair the matrix if it is not positive definite."""
        cfg = self.config
        values = matrix.to_numpy(dtype="float64").copy()
        repaired = False

        # Floor the diagonal, preserving correlations.
        variances = np.diag(values).copy()
        floor = cfg.min_annual_volatility**2
        if np.any(variances < floor) or np.any(~np.isfinite(variances)):
            repaired = True
            scale = np.sqrt(np.maximum(variances, floor) / np.maximum(variances, 1e-16))
            scale = np.where(np.isfinite(scale), scale, 1.0)
            values = values * np.outer(scale, scale)
            np.fill_diagonal(values, np.maximum(np.diag(values), floor))

        values = 0.5 * (values + values.T)  # enforce exact symmetry

        eigenvalues = np.linalg.eigvalsh(values)
        if eigenvalues.min() <= 0:
            repaired = True
            values = nearest_positive_definite(values, ridge=cfg.ridge)

        return pd.DataFrame(values, index=matrix.index, columns=matrix.columns), repaired


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def nearest_positive_definite(matrix: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    """Return the nearest positive-definite matrix by eigenvalue clipping.

    Negative and zero eigenvalues are clipped to a small positive floor derived
    from the largest eigenvalue, then a ridge is added for numerical headroom.
    Correlations are broadly preserved; only the unstable directions change.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(np.max(eigenvalues)) if eigenvalues.size else 1.0
    floor = max(largest * 1e-10, 1e-14)
    clipped = np.clip(eigenvalues, floor, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    if ridge > 0:
        repaired = repaired + ridge * np.eye(repaired.shape[0])
    return repaired


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix into a correlation matrix."""
    std = np.sqrt(np.diag(cov.to_numpy()))
    std = np.where(std > 1e-12, std, np.nan)
    corr = cov.to_numpy() / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Predicted annualised portfolio volatility :math:`\\sqrt{w^\\top \\Sigma w}`.

    Symbols absent from the covariance matrix are treated as having no estimate
    and are excluded; the caller should ensure the weight vector is aligned.
    """
    # Fast path: identical, identically-ordered labels need no realignment. This
    # is the common case inside the backtest loop and avoids a costly reindex.
    if len(weights) == len(cov.columns) and weights.index.equals(cov.columns):
        w = np.nan_to_num(weights.to_numpy(dtype="float64"))
        sigma = cov.to_numpy(dtype="float64")
        return float(np.sqrt(max(float(w @ sigma @ w), 0.0)))

    common = [s for s in weights.index if s in cov.columns]
    if not common:
        return 0.0
    w = weights.reindex(common).fillna(0.0).to_numpy(dtype="float64")
    sigma = cov.loc[common, common].to_numpy(dtype="float64")
    variance = float(w @ sigma @ w)
    return float(np.sqrt(max(variance, 0.0)))


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Decompose portfolio variance into each asset's contribution.

    The marginal contribution of asset *i* is :math:`(\\Sigma w)_i` and its total
    contribution is :math:`w_i (\\Sigma w)_i`. Contributions sum to the portfolio
    variance; this function returns them normalised to sum to one, so they read
    as "share of portfolio risk".
    """
    common = [s for s in weights.index if s in cov.columns]
    if not common:
        return pd.Series(dtype="float64")
    w = weights.reindex(common).fillna(0.0)
    sigma = cov.loc[common, common]
    marginal = sigma.to_numpy() @ w.to_numpy()
    contribution = w.to_numpy() * marginal
    total = contribution.sum()
    if abs(total) < 1e-16:
        return pd.Series(0.0, index=common)
    return pd.Series(contribution / total, index=common)
