#!/usr/bin/env python3
"""Regenerate a research report from stored artefacts.

    python scripts/generate_report.py --run-dir reports/backtests/bt-...

Reads the CSV artefacts written by `export_backtest` and re-renders the report,
so a report can be refreshed without re-running the backtest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.logging_utils import configure_logging
from atlas.reporting.export import load_manifest


def main() -> int:
    """Summarise a stored run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory written by export_backtest.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, force=True)
    directory = Path(args.run_dir)
    if not directory.is_dir():
        print(f"no such directory: {directory}")
        return 1

    manifest = load_manifest(directory)
    print(f"Run:          {manifest['run_id']}")
    print(f"Started:      {manifest['started_at']}")
    print(f"Git commit:   {manifest.get('git_commit')}")
    print(f"Config hash:  {manifest.get('config_hash')}")
    print(f"Data source:  {manifest['data_source']}")
    print(f"Synthetic:    {manifest['is_synthetic_data']}")
    print("\nSummary:")
    for key, value in manifest["summary"].items():
        print(f"  {key:<24} {value}")
    print("\nCost breakdown:")
    for key, value in manifest["cost_breakdown"].items():
        print(f"  {key:<24} {value:,.2f}")
    print("\nFiles:")
    for name in manifest["files"]:
        print(f"  {name}")

    report = directory / "report.html"
    if report.is_file():
        print(f"\nreport: {report}")
    else:
        print("\nNo report.html found. Re-run `atlas backtest` to regenerate it with figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
