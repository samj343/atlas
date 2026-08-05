#!/usr/bin/env python3
"""Download and cache historical market data.

    python scripts/download_data.py --provider yfinance
    python scripts/download_data.py --provider synthetic --start 2010-01-01

Thin wrapper over ``atlas download-data`` for anyone who prefers scripts to a CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.config import load_config
from atlas.data.loader import MarketDataLoader, panel_to_long
from atlas.data.validator import DataValidator
from atlas.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    """Download, validate and optionally persist market data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["yfinance", "csv", "synthetic"], default=None)
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cache.")
    parser.add_argument("--persist", action="store_true", help="Write prices to the database.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, force=True)
    config = load_config()
    if args.provider:
        config.assets.data.provider = args.provider

    import datetime as dt

    start = dt.date.fromisoformat(args.start) if args.start else config.data.start_date
    end = dt.date.fromisoformat(args.end) if args.end else config.data.end_date

    loader = MarketDataLoader.from_config(config)
    panel = loader.load(config.universe.symbols, start, end, force_refresh=args.refresh)

    print(panel.describe().to_string())
    if panel.is_synthetic:
        print("\nWARNING: this is SYNTHETIC simulated data, not real market history.")

    report = DataValidator(
        max_staleness_days=config.data.max_staleness_days,
        min_history_days=config.data.min_history_days,
    ).validate(panel, expected_symbols=config.universe.symbols)
    print()
    print(report.render())

    if args.persist:
        from atlas.database import AtlasRepository, DatabaseManager

        repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
        repository.upsert_assets(
            [
                {
                    "symbol": a.symbol,
                    "name": a.name,
                    "asset_class": a.asset_class,
                    "tradable": a.tradable,
                }
                for a in config.universe.assets
            ]
        )
        rows = repository.save_prices(panel_to_long(panel))
        print(f"\npersisted {rows:,} rows")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
