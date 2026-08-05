#!/usr/bin/env python3
"""Run a backtest with benchmarks, stress tests and a research report.

    python scripts/run_backtest.py --provider synthetic --start 2008-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.backtest.engine import BacktestEngine
from atlas.config import load_config
from atlas.data.features import FeatureEngineer
from atlas.data.loader import MarketDataLoader
from atlas.data.validator import DataValidator
from atlas.logging_utils import configure_logging
from atlas.reporting.export import export_backtest
from atlas.reporting.performance_report import ReportBundle, ResearchReport
from atlas.validation.benchmarks import BenchmarkSuite
from atlas.validation.stress_tests import StressTester, block_bootstrap


def main() -> int:
    """Run one backtest end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["yfinance", "csv", "synthetic"], default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-benchmarks", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, force=True)
    config = load_config()
    if args.provider:
        config.assets.data.provider = args.provider

    start = dt.date.fromisoformat(args.start) if args.start else config.data.start_date
    end = dt.date.fromisoformat(args.end) if args.end else config.data.end_date

    loader = MarketDataLoader.from_config(config)
    panel = loader.load(config.universe.symbols, start, end)
    report = DataValidator(min_history_days=config.data.min_history_days).validate(panel)
    if not report.is_valid:
        print(report.render())
        return 1
    panel = DataValidator().clean(panel, report)

    features = FeatureEngineer.from_config(config).build(panel)
    result = BacktestEngine(config).run(panel, features=features, progress=True)

    print()
    print(result.metrics.render())
    for warning in result.warnings:
        print(f"\nWARNING: {warning}")

    benchmarks = None
    if not args.no_benchmarks:
        benchmarks = BenchmarkSuite(config).run(panel, atlas_result=result, features=features)
        print("\nBenchmarks:")
        print(benchmarks.table().round(4).to_string())

    stress = StressTester(config.validation).run(result, panel=panel)
    bootstrap = None
    if config.validation.monte_carlo.enabled and len(result.returns) > 300:
        bootstrap = block_bootstrap(
            result.returns,
            n_simulations=config.validation.monte_carlo.n_simulations,
            block_size=config.validation.monte_carlo.block_size_days,
            horizon=config.validation.monte_carlo.horizon_days,
            seed=config.validation.monte_carlo.random_seed,
        )

    destination = Path(args.output_dir) if args.output_dir else (
        config.project_root / "reports" / "backtests" / result.run_id
    )
    export_backtest(result, destination, config=config)

    if not args.no_report:
        bundle = ReportBundle(
            result=result, config=config, panel=panel, benchmarks=benchmarks,
            stress=stress, bootstrap=bootstrap, label=f"Atlas backtest {result.run_id}",
        )
        path = ResearchReport(bundle).to_html(destination / "report.html")
        print(f"\nreport: {path}")

    print(f"artefacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
