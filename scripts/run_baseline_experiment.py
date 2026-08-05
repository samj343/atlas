#!/usr/bin/env python3
"""Run the Atlas baseline experiment end to end.

This is the reference research run described in the README and in
``docs/methodology.md``:

* daily data over the full available history;
* weekly rebalancing;
* 10% annualised volatility target;
* long-only ETF portfolio;
* trend 40% / mean reversion 25% / cross-sectional momentum 35%;
* realistic transaction costs;
* walk-forward out-of-sample evaluation;
* comparison against all eight benchmarks.

Usage
-----
    python scripts/run_baseline_experiment.py --provider yfinance
    python scripts/run_baseline_experiment.py --provider synthetic --quick

Nothing here fabricates a number. Every figure is computed from the data the
provider actually returned, and a run on synthetic data is labelled as such in
the console output, in the report and in the stored manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atlas import DISCLAIMER
from atlas.backtest.engine import BacktestEngine
from atlas.config import load_config
from atlas.data.features import FeatureEngineer
from atlas.data.loader import MarketDataLoader
from atlas.data.validator import DataValidator
from atlas.exceptions import AtlasError, DataError
from atlas.logging_utils import configure_logging, get_logger
from atlas.reporting.export import export_backtest, write_json
from atlas.reporting.performance_report import ReportBundle, ResearchReport
from atlas.validation.benchmarks import BenchmarkSuite
from atlas.validation.parameter_stability import ParameterStabilityAnalyzer
from atlas.validation.stress_tests import StressTester, block_bootstrap
from atlas.validation.walk_forward import WalkForwardValidator, expand_grid

log = get_logger(__name__)

SETUP_HELP = """
Market data could not be retrieved.

Atlas ships a provider-agnostic data layer; the default provider is Yahoo
Finance via the optional `yfinance` package, which needs network access.

To reproduce this experiment on real market data:

  1. Install the data extra:
         pip install -e ".[data]"
  2. Check that outbound HTTPS to Yahoo Finance is permitted from this machine.
  3. Re-run:
         python scripts/run_baseline_experiment.py --provider yfinance

To use a vendor export instead of Yahoo:

  1. Put one CSV per symbol in data/raw/ named <SYMBOL>.csv, with columns
     date, open, high, low, close, adj_close, volume.
  2. Set `provider: csv` in configs/assets.yaml, or pass --provider csv.

To exercise the full pipeline with no network at all:

         python scripts/run_baseline_experiment.py --provider synthetic

  The synthetic provider generates deterministic *simulated* price paths. They
  demonstrate that every component runs and that the risk controls behave as
  specified. They say NOTHING about real market performance, and every artefact
  produced from them is labelled accordingly.
"""


def _banner(title: str) -> None:
    """Print a section banner."""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    """Run the baseline experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["yfinance", "csv", "synthetic"], default=None,
        help="Data provider (default: whatever configs/assets.yaml specifies).",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--quick", action="store_true",
        help="Shorten the walk-forward windows and skip the stability sweep.",
    )
    parser.add_argument(
        "--skip-walk-forward", action="store_true",
        help="Skip the walk-forward stage (much faster, but no out-of-sample result).",
    )
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, force=True)
    started = time.time()

    config = load_config()
    if args.provider:
        config.assets.data.provider = args.provider

    _banner("Atlas baseline experiment")
    print(f"  provider          {config.data.provider}")
    print(f"  universe          {config.universe.name} ({len(config.symbols)} symbols)")
    print(f"  rebalance         {config.backtest.rebalance_frequency.value}")
    print(f"  execution timing  {config.backtest.execution_timing.value}")
    print(f"  volatility target {config.portfolio.target_volatility:.1%}")
    print(f"  long only         {config.portfolio.long_only}")
    weights = config.strategies.strategies.base_weights()
    print(f"  strategy weights  {', '.join(f'{k} {v:.0%}' for k, v in weights.items())}")
    print(f"  config hash       {config.config_hash}")

    # ---- 1. data ----------------------------------------------------------
    _banner("1. Data")
    start = dt.date.fromisoformat(args.start) if args.start else config.data.start_date
    end = dt.date.fromisoformat(args.end) if args.end else config.data.end_date
    try:
        loader = MarketDataLoader.from_config(config)
        panel = loader.load(config.universe.symbols, start, end)
    except (DataError, OSError) as exc:
        print(f"\n  FAILED: {exc}")
        print(SETUP_HELP)
        return 1

    print(panel.describe().to_string())
    synthetic = panel.is_synthetic
    if synthetic:
        print(
            "\n  *** SYNTHETIC DATA ***\n"
            "  These are simulated prices, not market history. Every result below\n"
            "  demonstrates that the machinery works and says nothing about real\n"
            "  market performance."
        )

    # ---- 2. validation ----------------------------------------------------
    _banner("2. Data validation")
    validator = DataValidator(
        max_staleness_days=config.data.max_staleness_days,
        min_history_days=config.data.min_history_days,
    )
    report = validator.validate(panel, expected_symbols=config.universe.symbols)
    print(report.render(max_rows=30))
    if not report.is_valid:
        print("\n  data validation failed; stopping")
        return 1
    panel = validator.clean(panel, report)

    # ---- 3. in-sample backtest -------------------------------------------
    _banner("3. In-sample backtest")
    features = FeatureEngineer.from_config(config).build(panel)
    result = BacktestEngine(config).run(panel, features=features, progress=True)
    print(result.metrics.render())
    drag = result.cost_drag()
    print("\n  Cost impact:")
    for key in ("gross_cagr", "net_cagr", "cost_drag_annual", "cost_share_of_gross_return"):
        value = drag.get(key)
        if value is not None and pd.notna(value):
            print(f"    {key:<32} {value:>10.4f}")

    # ---- 4. benchmarks ----------------------------------------------------
    _banner("4. Benchmark comparison")
    benchmarks = BenchmarkSuite(config).run(panel, atlas_result=result, features=features)
    print(benchmarks.table().round(4).to_string())
    for note in benchmarks.notes:
        print(f"\n  note: {note}")

    # ---- 5. stress periods ------------------------------------------------
    _banner("5. Stress periods")
    stress = StressTester(config.validation).run(result, panel=panel)
    if stress.periods.empty:
        print("  no configured stress window overlapped the backtest range")
    else:
        columns = [
            "name", "total_return", "max_drawdown", "annualized_volatility",
            "worst_day", "avg_gross_exposure",
        ]
        print(stress.periods[columns].round(4).to_string(index=False))
    for note in stress.skipped:
        print(f"  skipped: {note}")

    # ---- 6. bootstrap -----------------------------------------------------
    bootstrap = None
    if config.validation.monte_carlo.enabled and len(result.returns) > 300:
        _banner("6. Block-bootstrap resampling (statistical stress test, not a forecast)")
        monte = config.validation.monte_carlo
        bootstrap = block_bootstrap(
            result.returns,
            n_simulations=200 if args.quick else monte.n_simulations,
            block_size=monte.block_size_days,
            horizon=monte.horizon_days,
            seed=monte.random_seed,
            stationary=monte.method == "stationary_bootstrap",
        )
        from atlas.validation.stress_tests import bootstrap_statistics

        print(bootstrap_statistics(bootstrap, monte.percentiles).round(4).to_string())

    # ---- 7. walk-forward --------------------------------------------------
    walk_forward = None
    if not args.skip_walk_forward:
        _banner("7. Walk-forward out-of-sample validation")
        if args.quick:
            config.validation.walk_forward.train_years = 3
            config.validation.walk_forward.validation_years = 1
            config.validation.walk_forward.test_years = 1
            config.validation.walk_forward.step_years = 2
            config.validation.walk_forward.parameter_grid = {
                "trend.slow_ma": [150, 200],
                "portfolio.target_volatility": [0.08, 0.12],
            }
        validator_wf = WalkForwardValidator(config)
        try:
            windows = validator_wf.build_windows(panel.dates)
            candidates = expand_grid(config.validation.walk_forward.parameter_grid)
            print(
                f"  {len(windows)} window(s) x {len(candidates)} candidate(s) "
                f"= {len(windows) * (len(candidates) + 1)} backtests\n"
            )
            walk_forward = validator_wf.run(panel, features=features, progress=True)
            if walk_forward.test_metrics is not None:
                print(walk_forward.test_metrics.render())
            print("\n  In-sample versus out-of-sample:")
            for key, value in walk_forward.degradation().items():
                print(f"    {key:<34} {value:>10.4f}")
            if not walk_forward.selections.empty:
                print("\n  Selected parameters by window:")
                columns = [c for c in walk_forward.selections.columns if c.startswith("param_")]
                print(
                    walk_forward.selections[
                        ["window", "test_start", "test_end", *columns, "test_sharpe"]
                    ].to_string(index=False)
                )
            for warning in walk_forward.warnings:
                print(f"  warning: {warning}")
        except AtlasError as exc:
            print(f"  walk-forward could not run: {exc}")

    # ---- 8. parameter stability -------------------------------------------
    stability = None
    if not (args.skip_stability or args.quick):
        _banner("8. Parameter stability")
        try:
            stability = ParameterStabilityAnalyzer(config).run(panel)
            print(stability.verdicts.round(4).to_string(index=False))
            if stability.fragile_parameters:
                print(f"\n  FRAGILE: {', '.join(stability.fragile_parameters)}")
            else:
                print("\n  no parameter was flagged fragile")
        except AtlasError as exc:
            print(f"  stability sweep could not run: {exc}")

    # ---- 9. artefacts ------------------------------------------------------
    _banner("9. Artefacts")
    destination = Path(args.output_dir) if args.output_dir else (
        config.project_root / "reports" / "backtests" / f"baseline_{result.run_id}"
    )
    export_backtest(result, destination, config=config)

    bundle = ReportBundle(
        result=result,
        config=config,
        panel=panel,
        benchmarks=benchmarks,
        walk_forward=walk_forward,
        stability=stability,
        stress=stress,
        bootstrap=bootstrap,
        label="Atlas baseline experiment"
        + (" (SYNTHETIC DATA)" if synthetic else ""),
    )
    report_path = ResearchReport(bundle).to_html(destination / "report.html")
    ResearchReport(bundle).to_markdown(destination / "report.md")

    if walk_forward is not None:
        walk_forward.selections.to_csv(destination / "walk_forward_selections.csv", index=False)
        walk_forward.test_returns.to_frame("return").to_csv(destination / "oos_returns.csv")
    if stability is not None:
        stability.sweeps.to_csv(destination / "parameter_sweeps.csv", index=False)
        stability.verdicts.to_csv(destination / "parameter_verdicts.csv", index=False)
    if not benchmarks.table().empty:
        benchmarks.table().to_csv(destination / "benchmark_comparison.csv")
    if not stress.periods.empty:
        stress.periods.to_csv(destination / "stress_periods.csv", index=False)

    write_json(
        {
            "experiment": "baseline",
            "provider": config.data.provider,
            "synthetic_data": synthetic,
            "config_hash": config.config_hash,
            "run_id": result.run_id,
            "in_sample": result.summary(),
            "out_of_sample": walk_forward.summary() if walk_forward else None,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        destination / "experiment_summary.json",
    )

    # ---- 10. persist -------------------------------------------------------
    try:
        from atlas.database import AtlasRepository, DatabaseManager

        repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
        repository.save_configuration(config.config_hash, config.snapshot(), "baseline experiment")
        repository.save_backtest(result, label="baseline experiment")
        print(f"  stored in the database as run {result.run_id}")
    except Exception as exc:
        print(f"  could not persist to the database: {exc}")

    print(f"  artefacts:  {destination}")
    print(f"  report:     {report_path}")
    print(f"  elapsed:    {time.time() - started:.0f}s")

    _banner("Disclaimer")
    print(f"  {DISCLAIMER}")
    if synthetic:
        print(
            "\n  This run used SYNTHETIC data. No performance conclusion of any kind\n"
            "  can be drawn from it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
