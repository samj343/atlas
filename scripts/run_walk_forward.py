#!/usr/bin/env python3
"""Run walk-forward out-of-sample validation.

    python scripts/run_walk_forward.py --provider synthetic

This is the slowest command in Atlas: it runs one backtest per candidate per
window. The default grid is 12 candidates; widening it costs proportionally more.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.config import load_config
from atlas.data.features import FeatureEngineer
from atlas.data.loader import MarketDataLoader
from atlas.logging_utils import configure_logging
from atlas.reporting.export import write_json
from atlas.validation.walk_forward import WalkForwardValidator, expand_grid


def main() -> int:
    """Run the walk-forward procedure and write its results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["yfinance", "csv", "synthetic"], default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-years", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, force=True)
    config = load_config()
    if args.provider:
        config.assets.data.provider = args.provider
    if args.train_years:
        config.validation.walk_forward.train_years = args.train_years

    start = dt.date.fromisoformat(args.start) if args.start else config.data.start_date
    end = dt.date.fromisoformat(args.end) if args.end else config.data.end_date

    panel = MarketDataLoader.from_config(config).load(config.universe.symbols, start, end)
    features = FeatureEngineer.from_config(config).build(panel)

    validator = WalkForwardValidator(config)
    windows = validator.build_windows(panel.dates)
    candidates = expand_grid(config.validation.walk_forward.parameter_grid)
    print(
        f"{len(windows)} window(s) x {len(candidates)} candidate(s) "
        f"= {len(windows) * (len(candidates) + 1)} backtests"
    )

    result = validator.run(panel, features=features, progress=True)

    print("\nOut-of-sample (concatenated test periods):")
    if result.test_metrics is not None:
        print(result.test_metrics.render())
    print("\nIn-sample versus out-of-sample:")
    for key, value in result.degradation().items():
        print(f"  {key:<34} {value:>10.4f}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = Path(args.output_dir) if args.output_dir else (
        config.project_root / "reports" / "backtests" / f"walk_forward_{stamp}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    result.selections.to_csv(destination / "selections.csv", index=False)
    result.candidate_scores.to_csv(destination / "candidate_scores.csv", index=False)
    result.test_returns.to_frame("return").to_csv(destination / "oos_returns.csv")
    write_json(result.summary(), destination / "summary.json")
    print(f"\nresults: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
