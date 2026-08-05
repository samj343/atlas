"""Research validation: walk-forward, stability, stress tests and benchmarks."""

from atlas.validation.benchmarks import BenchmarkResult, BenchmarkSuite
from atlas.validation.parameter_stability import ParameterStabilityAnalyzer, StabilityResult
from atlas.validation.stress_tests import (
    StressTester,
    StressTestResult,
    block_bootstrap,
    bootstrap_statistics,
)
from atlas.validation.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
    apply_overrides,
    expand_grid,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "ParameterStabilityAnalyzer",
    "StabilityResult",
    "StressTestResult",
    "StressTester",
    "WalkForwardResult",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "apply_overrides",
    "block_bootstrap",
    "bootstrap_statistics",
    "expand_grid",
]
