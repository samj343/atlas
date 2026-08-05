"""Turning combined signals into constrained target portfolio weights.

The pipeline, in order:

1. **Inverse-volatility scaling** - :math:`\\tilde w_i = s_i / \\sigma_i`, so a
   unit of conviction buys the same amount of *risk* in every asset rather than
   the same amount of capital.
2. **Gross normalisation** - :math:`w_i = \\tilde w_i / \\sum_j |\\tilde w_j|`,
   producing a fully invested book of unit gross exposure.
3. **Volatility targeting** - scale the whole book by
   :math:`\\sigma_{target} / \\hat\\sigma_p` where
   :math:`\\hat\\sigma_p = \\sqrt{w^\\top \\Sigma w}` is the predicted portfolio
   volatility. This is what makes risk, rather than capital, the constant.
4. **Regime risk multiplier and drawdown multiplier** - external scalars that
   scale the risk budget down (never up beyond the configured maximum).
5. **Defensive allocation** - a floor allocation to the cash-like asset in
   defensive regimes or when breadth deteriorates.
6. **Constraints** - see :mod:`atlas.portfolio.constraints`.

Everything is deterministic and every intermediate is retained on the result, so
the path from signal to weight can be inspected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import AtlasConfig, PortfolioConfig, RegimeLabel
from atlas.logging_utils import get_logger
from atlas.portfolio.constraints import ConstraintResult, PortfolioConstraints
from atlas.portfolio.covariance import (
    CovarianceEstimator,
    CovarianceResult,
    portfolio_volatility,
    risk_contributions,
)
from atlas.portfolio.volatility import VolatilityEstimator, inverse_volatility_weights

log = get_logger(__name__)

__all__ = ["AllocationResult", "PortfolioAllocator"]


@dataclass
class AllocationResult:
    """Target weights plus every intermediate quantity that produced them."""

    weights: pd.Series
    pre_constraint_weights: pd.Series
    raw_weights: pd.Series
    signals: pd.Series
    volatilities: pd.Series
    predicted_volatility: float
    target_volatility: float
    vol_scaler: float
    risk_multiplier: float
    defensive_weight: float = 0.0
    covariance: CovarianceResult | None = None
    constraints: ConstraintResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute target weights."""
        return float(self.weights.abs().sum())

    @property
    def net_exposure(self) -> float:
        """Sum of signed target weights."""
        return float(self.weights.sum())

    @property
    def cash_weight(self) -> float:
        """Fraction of equity left in cash."""
        return 1.0 - self.net_exposure

    def realized_predicted_volatility(self) -> float:
        """Predicted volatility of the *final* (constrained) weights."""
        if self.covariance is None:
            return float("nan")
        return portfolio_volatility(self.weights, self.covariance.matrix)

    def risk_contributions(self) -> pd.Series:
        """Share of portfolio variance contributed by each asset."""
        if self.covariance is None:
            return pd.Series(dtype="float64")
        return risk_contributions(self.weights, self.covariance.matrix)

    def summary(self) -> dict[str, Any]:
        """Compact description used in logs and reports."""
        return {
            "gross_exposure": round(self.gross_exposure, 4),
            "net_exposure": round(self.net_exposure, 4),
            "cash_weight": round(self.cash_weight, 4),
            "n_positions": int((self.weights.abs() > 1e-9).sum()),
            "predicted_volatility": round(self.predicted_volatility, 4),
            "final_predicted_volatility": round(self.realized_predicted_volatility(), 4),
            "target_volatility": round(self.target_volatility, 4),
            "vol_scaler": round(self.vol_scaler, 4),
            "risk_multiplier": round(self.risk_multiplier, 4),
            "defensive_weight": round(self.defensive_weight, 4),
        }


class PortfolioAllocator:
    """Convert combined signals into constrained, volatility-targeted weights.

    Parameters
    ----------
    config:
        The full :class:`~atlas.config.AtlasConfig`.
    """

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.portfolio_cfg: PortfolioConfig = config.portfolio
        self.volatility = VolatilityEstimator(config.risk.volatility)
        self.covariance = CovarianceEstimator(config.risk.covariance)
        self.constraints = PortfolioConstraints(config.portfolio, config.asset_class_map)
        self.defensive_asset = (config.universe.defensive_asset or "").upper() or None

    # -- main entry point -----------------------------------------------------

    def allocate(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        *,
        previous_weights: pd.Series | None = None,
        risk_multiplier: float = 1.0,
        defensive_weight: float = 0.0,
        tradable: list[str] | None = None,
    ) -> AllocationResult:
        """Build target weights for a single rebalance date.

        Parameters
        ----------
        signals:
            Combined signal per symbol, in ``[-1, 1]``.
        returns:
            Trailing daily returns up to (and including) the rebalance date.
            Only history is used - the caller is responsible for not passing
            future rows.
        previous_weights:
            Current portfolio weights, for turnover control.
        risk_multiplier:
            Multiplier applied to the volatility target, typically the product of
            the regime multiplier and the drawdown multiplier.
        defensive_weight:
            Minimum allocation routed to the defensive asset.
        tradable:
            Restrict allocation to these symbols (e.g. those with fresh data and
            enough history).

        Returns
        -------
        AllocationResult
        """
        cfg = self.portfolio_cfg
        universe = list(signals.index) if tradable is None else [s for s in signals.index if s in set(tradable)]
        sig = signals.reindex(universe).fillna(0.0).astype("float64")

        # --- volatility estimates --------------------------------------------
        vol_frame = self.volatility.estimate(returns.reindex(columns=universe))
        volatilities = (
            vol_frame.iloc[-1]
            if not vol_frame.empty
            else pd.Series(np.nan, index=universe)
        )
        volatilities = volatilities.reindex(universe)

        # --- step 1: inverse-volatility scaling ------------------------------
        raw = inverse_volatility_weights(
            sig, volatilities, min_volatility=self.config.risk.volatility.min_annual_volatility
        )

        # --- step 2: gross normalisation --------------------------------------
        gross = float(raw.abs().sum())
        if gross <= 1e-12:
            # No conviction anywhere: hold cash. This is a legitimate outcome,
            # not an error.
            empty = pd.Series(0.0, index=universe)
            return AllocationResult(
                weights=empty,
                pre_constraint_weights=empty.copy(),
                raw_weights=raw,
                signals=sig,
                volatilities=volatilities,
                predicted_volatility=0.0,
                target_volatility=cfg.target_volatility * risk_multiplier,
                vol_scaler=0.0,
                risk_multiplier=risk_multiplier,
                diagnostics={"reason": "no active signals; portfolio held in cash"},
            )
        normalized = raw / gross

        # --- step 3: volatility targeting -------------------------------------
        cov = self.covariance.estimate_or_fallback(returns, symbols=universe)
        predicted = portfolio_volatility(normalized, cov.matrix)
        target = cfg.target_volatility * float(np.clip(risk_multiplier, 0.0, 1.0))

        scaler = cfg.min_vol_target_scaler if predicted <= 1e-8 else target / predicted
        scaler = float(np.clip(scaler, cfg.min_vol_target_scaler, cfg.max_vol_target_scaler))
        scaled = normalized * scaler

        # --- hard ceiling on predicted volatility -----------------------------
        scaled_predicted = portfolio_volatility(scaled, cov.matrix)
        if scaled_predicted > cfg.max_portfolio_volatility > 0:
            shrink = cfg.max_portfolio_volatility / scaled_predicted
            scaled = scaled * shrink
            log.info(
                "predicted volatility ceiling binding",
                extra={
                    "context": {
                        "predicted": round(scaled_predicted, 4),
                        "ceiling": cfg.max_portfolio_volatility,
                        "shrink": round(shrink, 4),
                    }
                },
            )

        # --- step 5: defensive sleeve ------------------------------------------
        pre_constraint = self._apply_defensive(scaled, defensive_weight)

        # --- step 6: constraints ------------------------------------------------
        constrained = self.constraints.apply(pre_constraint, previous_weights=previous_weights)

        result = AllocationResult(
            weights=constrained.weights,
            pre_constraint_weights=pre_constraint,
            raw_weights=raw,
            signals=sig,
            volatilities=volatilities,
            predicted_volatility=predicted,
            target_volatility=target,
            vol_scaler=scaler,
            risk_multiplier=risk_multiplier,
            defensive_weight=defensive_weight,
            covariance=cov,
            constraints=constrained,
            diagnostics={
                "covariance_method": cov.method,
                "covariance_observations": cov.n_observations,
                "covariance_condition": cov.condition_number(),
                "gross_before_normalisation": gross,
            },
        )
        log.debug("allocation complete", extra={"context": result.summary()})
        return result

    # -- helpers --------------------------------------------------------------

    def _apply_defensive(self, weights: pd.Series, defensive_weight: float) -> pd.Series:
        """Route a floor allocation to the defensive asset.

        The remaining book is scaled down to make room, so the defensive sleeve
        genuinely de-risks the portfolio rather than adding leverage on top.
        """
        if defensive_weight <= 0.0 or not self.defensive_asset:
            return weights
        if self.defensive_asset not in weights.index:
            log.warning(
                "defensive asset is not in the tradable universe today",
                extra={"context": {"symbol": self.defensive_asset}},
            )
            return weights

        out = weights.copy()
        others = [s for s in out.index if s != self.defensive_asset]
        other_gross = float(out[others].abs().sum())
        room = max(0.0, 1.0 - defensive_weight)
        if other_gross > room > 0.0:
            out.loc[others] = out.loc[others] * (room / other_gross)
        out.loc[self.defensive_asset] = max(float(out.get(self.defensive_asset, 0.0)), defensive_weight)
        return out

    def risk_posture(
        self,
        regime: RegimeLabel | str | None,
        *,
        drawdown_multiplier: float = 1.0,
        weak_breadth: bool = False,
    ) -> tuple[float, float]:
        """Return ``(risk_multiplier, defensive_weight)`` for a regime.

        Combines the regime block from configuration with the drawdown ladder
        and the breadth overlay. Multipliers compose multiplicatively, so two
        independent de-risking signals do not cancel each other out.
        """
        regimes_cfg = self.config.regimes
        block = regimes_cfg.weights_for(regime if regime is not None else RegimeLabel.UNKNOWN)
        risk_multiplier = float(block.risk_multiplier) * float(drawdown_multiplier)
        defensive = float(block.defensive_weight)
        overlay = regimes_cfg.breadth_overlay
        if weak_breadth and overlay.enabled:
            defensive = min(1.0, defensive + overlay.additional_defensive_weight)
        return float(np.clip(risk_multiplier, 0.0, 2.0)), float(np.clip(defensive, 0.0, 1.0))
