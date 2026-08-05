#!/usr/bin/env python3
"""Run one paper-trading execution cycle.

    python scripts/run_paper_trader.py --mock            # no TWS needed
    python scripts/run_paper_trader.py                   # connects to TWS/Gateway

Orders are only submitted when EXECUTION_MODE=paper *and* the connected account
is a paper account on the configured allowlist. In every other case this
previews the trades and stops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.config import ExecutionMode, load_config
from atlas.data.loader import MarketDataLoader
from atlas.exceptions import AtlasError
from atlas.logging_utils import configure_logging


def main() -> int:
    """Run a single execution cycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["yfinance", "csv", "synthetic"], default=None)
    parser.add_argument("--mock", action="store_true", help="Use the in-memory mock broker.")
    parser.add_argument("--submit", action="store_true", help="Submit (paper mode only).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, force=True)
    config = load_config()
    if args.provider:
        config.assets.data.provider = args.provider

    panel = MarketDataLoader.from_config(config).load(
        config.universe.symbols, config.data.start_date, config.data.end_date
    )

    from atlas.database import AtlasRepository, DatabaseManager
    from atlas.execution.ibkr_client import IBKRClient, MockBrokerClient
    from atlas.execution.order_manager import OrderManager

    broker = (
        MockBrokerClient(
            prices=panel.adj_close.iloc[-1], initial_cash=config.portfolio.initial_capital
        )
        if args.mock
        else IBKRClient(config.execution)
    )
    repository = AtlasRepository(DatabaseManager(project_root=config.project_root))

    try:
        manager = OrderManager(config, broker=broker, repository=repository)
        submit = args.submit and config.execution.mode is ExecutionMode.PAPER
        if args.submit and not submit:
            print(
                f"--submit ignored: EXECUTION_MODE is '{config.execution.mode.value}', not 'paper'."
            )
        result = manager.run_cycle(panel, submit=submit)
    except AtlasError as exc:
        print(f"blocked: {exc}")
        return 3
    finally:
        broker.disconnect()

    print()
    print(result.render())
    if result.plan is not None:
        print()
        print(result.plan.render())
    return 3 if result.blocked_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
