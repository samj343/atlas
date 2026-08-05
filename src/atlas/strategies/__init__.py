"""Signal-generating strategies and the ensemble that combines them."""

from atlas.strategies.base import (
    STRATEGY_REGISTRY,
    SignalResult,
    Strategy,
    build_strategies,
    register_strategy,
)
from atlas.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from atlas.strategies.ensemble import CombinedSignal, SignalEnsemble
from atlas.strategies.mean_reversion import MeanReversionStrategy
from atlas.strategies.trend import TrendStrategy

__all__ = [
    "STRATEGY_REGISTRY",
    "CombinedSignal",
    "CrossSectionalMomentumStrategy",
    "MeanReversionStrategy",
    "SignalEnsemble",
    "SignalResult",
    "Strategy",
    "TrendStrategy",
    "build_strategies",
    "register_strategy",
]
