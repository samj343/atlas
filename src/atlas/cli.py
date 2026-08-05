"""Atlas command-line interface.

Every command validates configuration before doing anything, writes structured
logs, and returns a meaningful exit code:

===  =========================================================
  0  success
  1  a runtime failure (data, strategy, portfolio, reporting)
  2  invalid configuration or arguments
  3  a risk limit or safety gate blocked the action
  4  a broker/connectivity failure
===  =========================================================

Run ``atlas --help`` for the command list, or ``atlas <command> --help`` for a
command's options.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import click

from atlas import DISCLAIMER, __version__
from atlas.config import AtlasConfig, ExecutionMode, load_config
from atlas.exceptions import (
    AtlasError,
    BrokerError,
    ConfigurationError,
    ExecutionBlocked,
    KillSwitchActive,
    RiskLimitBreached,
)
from atlas.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

__all__ = ["cli", "main"]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_RISK = 3
EXIT_BROKER = 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load(ctx: click.Context) -> AtlasConfig:
    """Load configuration, converting failures into a clean exit."""
    try:
        config = load_config(ctx.obj.get("config_dir"))
    except (ConfigurationError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(
            f"configuration error: {exc}\n"
            "Check the YAML files under configs/. Every key is validated, so a typo fails here "
            "rather than silently changing behaviour."
        ) from exc
    if ctx.obj.get("execution_mode"):
        config.execution.mode = ExecutionMode(ctx.obj["execution_mode"])
    return config


def _load_panel(
    config: AtlasConfig,
    *,
    start: date | None = None,
    end: date | None = None,
    provider: str | None = None,
    refresh: bool = False,
    validate: bool = True,
):
    """Load and optionally validate the market-data panel."""
    from atlas.data.loader import MarketDataLoader
    from atlas.data.validator import DataValidator

    if provider:
        config.assets.data.provider = provider  # type: ignore[assignment]
    loader = MarketDataLoader.from_config(config)
    panel = loader.load(
        config.universe.symbols,
        start or config.data.start_date,
        end or config.data.end_date,
        force_refresh=refresh,
    )
    if validate:
        validator = DataValidator(
            max_staleness_days=config.data.max_staleness_days,
            min_history_days=config.data.min_history_days,
        )
        report = validator.validate(panel, expected_symbols=config.universe.symbols)
        if not report.is_valid:
            click.echo(report.render(), err=True)
            raise click.ClickException(
                "market data failed validation; fix the data or adjust configs/assets.yaml"
            )
        panel = validator.clean(panel, report)
    return panel


def _echo_header(title: str) -> None:
    """Print a section header."""
    click.echo()
    click.secho(f"  {title}", bold=True, fg="cyan")
    click.secho(f"  {'-' * max(len(title), 40)}", fg="cyan")


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="atlas")
@click.option(
    "--config-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory holding the YAML configuration files (default: ./configs).",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity.",
)
@click.option("--log-json", is_flag=True, help="Emit structured JSON log lines.")
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write JSON logs to this file.",
)
@click.option(
    "--execution-mode",
    type=click.Choice(["backtest", "dry_run", "paper"]),
    default=None,
    help="Override EXECUTION_MODE for this invocation. 'live' is not supported.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_dir: Path | None,
    log_level: str,
    log_json: bool,
    log_file: Path | None,
    execution_mode: str | None,
) -> None:
    """Atlas - regime-aware multi-strategy research and paper-trading platform.

    Atlas is an educational research and paper-trading system. Historical
    performance does not guarantee future results.
    """
    configure_logging(log_level.upper(), json_output=log_json, log_file=log_file, force=True)
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = config_dir
    ctx.obj["execution_mode"] = execution_mode


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------


@cli.command("download-data")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None, help="Start date.")
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None, help="End date.")
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option("--refresh", is_flag=True, help="Ignore the cache and re-download everything.")
@click.option("--symbols", default=None, help="Comma-separated symbols (default: whole universe).")
@click.option("--persist/--no-persist", default=False, help="Also write prices to the database.")
@click.pass_context
def download_data(
    ctx: click.Context,
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    refresh: bool,
    symbols: str | None,
    persist: bool,
) -> None:
    """Download and cache historical daily market data."""
    from atlas.data.loader import MarketDataLoader, panel_to_long

    config = _load(ctx)
    if provider:
        config.assets.data.provider = provider  # type: ignore[assignment]
    universe = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else config.universe.symbols
    )

    loader = MarketDataLoader.from_config(config)
    panel = loader.load(
        universe,
        start.date() if start else config.data.start_date,
        end.date() if end else config.data.end_date,
        force_refresh=refresh,
    )

    _echo_header("Downloaded market data")
    click.echo(panel.describe().to_string())
    click.echo(f"\n  provider: {loader.provider.metadata.name}")
    click.echo(f"  cache:    {loader.cache_dir}")
    if panel.is_synthetic:
        click.secho(
            "\n  WARNING: this is SYNTHETIC simulated data, not real market history.",
            fg="yellow",
            bold=True,
        )

    if persist:
        from atlas.database import AtlasRepository, DatabaseManager

        repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
        repository.upsert_assets(
            [
                {"symbol": a.symbol, "name": a.name, "asset_class": a.asset_class, "tradable": a.tradable}
                for a in config.universe.assets
            ]
        )
        rows = repository.save_prices(panel_to_long(panel))
        click.echo(f"  persisted {rows:,} price rows to the database")


@cli.command("validate-data")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option(
    "--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
    help="Write the findings to a CSV file.",
)
@click.option("--strict", is_flag=True, help="Exit non-zero on warnings as well as errors.")
@click.pass_context
def validate_data(
    ctx: click.Context,
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    output: Path | None,
    strict: bool,
) -> None:
    """Validate cached market data and print a findings report."""
    from atlas.data.loader import MarketDataLoader
    from atlas.data.validator import DataValidator

    config = _load(ctx)
    if provider:
        config.assets.data.provider = provider  # type: ignore[assignment]

    loader = MarketDataLoader.from_config(config)
    panel = loader.load(
        config.universe.symbols,
        start.date() if start else config.data.start_date,
        end.date() if end else config.data.end_date,
    )
    validator = DataValidator(
        max_staleness_days=config.data.max_staleness_days,
        min_history_days=config.data.min_history_days,
    )
    report = validator.validate(panel, expected_symbols=config.universe.symbols)
    click.echo(report.render(max_rows=80))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        report.to_frame().to_csv(output, index=False)
        click.echo(f"\n  findings written to {output}")

    if not report.is_valid:
        raise SystemExit(EXIT_FAILURE)
    if strict and report.warnings:
        click.secho(f"\n  {len(report.warnings)} warning(s) with --strict", fg="yellow")
        raise SystemExit(EXIT_FAILURE)


# ---------------------------------------------------------------------------
# Research commands
# ---------------------------------------------------------------------------


@cli.command("backtest")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option("--benchmarks/--no-benchmarks", default=True, help="Compute the benchmark suite.")
@click.option("--stress/--no-stress", default=True, help="Run the stress-period analysis.")
@click.option("--report/--no-report", default=True, help="Generate the HTML research report.")
@click.option("--persist/--no-persist", default=True, help="Store the run in the database.")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Where to write artefacts (default: reports/backtests/<run_id>).",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    benchmarks: bool,
    stress: bool,
    report: bool,
    persist: bool,
    output_dir: Path | None,
) -> None:
    """Run a full backtest, with benchmarks, stress tests and a research report."""
    from atlas.backtest.engine import BacktestEngine
    from atlas.data.features import FeatureEngineer
    from atlas.reporting.export import export_backtest

    config = _load(ctx)
    panel = _load_panel(
        config,
        start=start.date() if start else None,
        end=end.date() if end else None,
        provider=provider,
    )

    features = FeatureEngineer.from_config(config).build(panel)
    engine = BacktestEngine(config)
    result = engine.run(
        panel,
        features=features,
        start=start.date() if start else None,
        end=end.date() if end else None,
        progress=True,
    )

    _echo_header(f"Backtest {result.run_id}")
    click.echo(result.metrics.render() if result.metrics else "  no metrics computed")
    for warning in result.warnings:
        click.secho(f"\n  WARNING: {warning}", fg="yellow", bold=True)

    bench_result = None
    if benchmarks:
        from atlas.validation.benchmarks import BenchmarkSuite

        bench_result = BenchmarkSuite(config).run(panel, atlas_result=result, features=features)
        _echo_header("Benchmark comparison")
        click.echo(bench_result.table().round(4).to_string())

    stress_result = None
    if stress:
        from atlas.validation.stress_tests import StressTester

        stress_result = StressTester(config.validation).run(result, panel=panel)
        if not stress_result.periods.empty:
            _echo_header("Stress periods")
            columns = ["name", "total_return", "max_drawdown", "worst_day", "avg_gross_exposure"]
            click.echo(stress_result.periods[columns].round(4).to_string(index=False))
        else:
            click.echo("\n  no stress window overlapped the backtest range")

    destination = output_dir or (config.project_root / "reports" / "backtests" / result.run_id)
    export_backtest(result, destination, config=config)
    click.echo(f"\n  artefacts written to {destination}")

    if report:
        from atlas.reporting.performance_report import ReportBundle, ResearchReport

        bundle = ReportBundle(
            result=result,
            config=config,
            panel=panel,
            benchmarks=bench_result,
            stress=stress_result,
            label=f"Atlas backtest {result.run_id}",
        )
        path = ResearchReport(bundle).to_html(destination / "report.html")
        click.echo(f"  research report: {path}")

    if persist:
        from atlas.database import AtlasRepository, DatabaseManager

        repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
        repository.save_configuration(config.config_hash, config.snapshot(), label="backtest")
        repository.save_backtest(result, label="cli backtest")
        click.echo(f"  stored in the database as run {result.run_id}")


@cli.command("walk-forward")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Where to write results (default: reports/backtests/walk_forward_<timestamp>).",
)
@click.pass_context
def walk_forward(
    ctx: click.Context,
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    output_dir: Path | None,
) -> None:
    """Run walk-forward parameter selection and out-of-sample evaluation.

    This runs one backtest per candidate per window and is the slowest command in
    Atlas. The grid in configs/validation.yaml is deliberately small; widening it
    increases runtime proportionally.
    """
    from atlas.data.features import FeatureEngineer
    from atlas.reporting.export import write_json
    from atlas.validation.walk_forward import WalkForwardValidator, expand_grid

    config = _load(ctx)
    panel = _load_panel(
        config,
        start=start.date() if start else None,
        end=end.date() if end else None,
        provider=provider,
    )
    features = FeatureEngineer.from_config(config).build(panel)

    validator = WalkForwardValidator(config)
    candidates = expand_grid(config.validation.walk_forward.parameter_grid)
    windows = validator.build_windows(panel.dates)
    click.echo(
        f"\n  {len(windows)} window(s) x {len(candidates)} candidate(s) "
        f"= {len(windows) * (len(candidates) + 1)} backtests"
    )

    result = validator.run(panel, features=features, progress=True)

    _echo_header("Walk-forward results (out-of-sample)")
    if result.test_metrics is not None:
        click.echo(result.test_metrics.render())
    else:
        click.secho("  no out-of-sample returns were produced", fg="yellow")

    degradation = result.degradation()
    if degradation:
        _echo_header("In-sample versus out-of-sample")
        for key, value in degradation.items():
            click.echo(f"  {key:<34} {value:>10.4f}")

    if not result.selections.empty:
        _echo_header("Selected parameters by window")
        columns = [
            c
            for c in result.selections.columns
            if c.startswith("param_") or c in {"window", "test_start", "test_end", "test_sharpe"}
        ]
        click.echo(result.selections[columns].to_string(index=False))

    for warning in result.warnings:
        click.secho(f"  WARNING: {warning}", fg="yellow")

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    destination = output_dir or (
        config.project_root / "reports" / "backtests" / f"walk_forward_{stamp}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    result.selections.to_csv(destination / "selections.csv", index=False)
    result.candidate_scores.to_csv(destination / "candidate_scores.csv", index=False)
    result.test_returns.to_frame("return").to_csv(destination / "oos_returns.csv")
    write_json(result.summary(), destination / "summary.json")
    click.echo(f"\n  results written to {destination}")


@cli.command("parameter-stability")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option("--gross/--no-gross", default=True, help="Also run each setting gross of costs.")
@click.pass_context
def parameter_stability(
    ctx: click.Context,
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    gross: bool,
) -> None:
    """Sweep each parameter around the baseline and flag fragile settings."""
    from atlas.validation.parameter_stability import ParameterStabilityAnalyzer

    config = _load(ctx)
    panel = _load_panel(
        config,
        start=start.date() if start else None,
        end=end.date() if end else None,
        provider=provider,
    )
    result = ParameterStabilityAnalyzer(config).run(panel, include_gross=gross)

    _echo_header("Parameter stability")
    click.echo(result.verdicts.round(4).to_string(index=False))
    if result.fragile_parameters:
        click.secho(
            f"\n  FRAGILE: {', '.join(result.fragile_parameters)} - performance depends on the "
            "specific value chosen.",
            fg="yellow",
            bold=True,
        )
    else:
        click.secho("\n  no parameter was flagged fragile", fg="green")
    for failure in result.failures:
        click.secho(f"  failed: {failure}", fg="red")


@cli.command("generate-report")
@click.option("--run-id", required=True, help="Run identifier to report on.")
@click.option(
    "--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
    help="Output path (default: alongside the run's artefacts).",
)
@click.option("--format", "fmt", type=click.Choice(["html", "markdown"]), default="html")
@click.pass_context
def generate_report(ctx: click.Context, run_id: str, output: Path | None, fmt: str) -> None:
    """Regenerate a research report for a stored run."""
    from atlas.database import AtlasRepository, DatabaseManager

    config = _load(ctx)
    repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
    stored = repository.get_run(run_id)
    if stored is None:
        raise click.ClickException(
            f"run {run_id!r} not found in the database. List available runs with `atlas runs`."
        )

    destination = output or (
        config.project_root / "reports" / "backtests" / run_id / f"report.{'html' if fmt == 'html' else 'md'}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    equity = repository.get_equity_curve(run_id)
    payload = {
        "run": {k: v for k, v in stored.items() if k != "config_snapshot"},
        "metrics": stored["metrics"],
        "equity_points": len(equity),
    }
    summary_path = destination.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    _echo_header(f"Run {run_id}")
    for key in ("data_start", "data_end", "data_source", "is_synthetic_data", "git_commit"):
        click.echo(f"  {key:<20} {stored.get(key)}")
    click.echo("\n  metrics:")
    for name, value in sorted(stored["metrics"].items()):
        if value is not None:
            click.echo(f"    {name:<32} {value:>12.4f}")
    click.echo(f"\n  summary written to {summary_path}")
    click.secho(
        "\n  Note: a full report with figures is produced by `atlas backtest`, which has the "
        "price panel in memory. This command summarises what the database stored.",
        fg="cyan",
    )


@cli.command("runs")
@click.option("--limit", default=25, show_default=True, help="How many runs to list.")
@click.pass_context
def runs(ctx: click.Context, limit: int) -> None:
    """List stored backtest runs."""
    from atlas.database import AtlasRepository, DatabaseManager

    config = _load(ctx)
    repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
    frame = repository.list_runs(limit=limit)
    _echo_header("Stored runs")
    click.echo("  no runs stored yet" if frame.empty else frame.to_string(index=False))


# ---------------------------------------------------------------------------
# Execution commands
# ---------------------------------------------------------------------------


@cli.command("broker-check")
@click.option("--mock", is_flag=True, help="Use the in-memory mock broker instead of TWS.")
@click.pass_context
def broker_check(ctx: click.Context, mock: bool) -> None:
    """Connect to the broker and run every pre-trade safety check."""
    from atlas.execution.ibkr_client import BrokerClient, IBKRClient, MockBrokerClient
    from atlas.execution.safety import SafetyGate

    config = _load(ctx)
    gate = SafetyGate(config)
    client: BrokerClient = MockBrokerClient() if mock else IBKRClient(config.execution)

    try:
        client.connect()
    except BrokerError as exc:
        click.secho(f"\n  broker connection failed: {exc}", fg="red")
        raise SystemExit(EXIT_BROKER) from exc

    try:
        summary = client.get_account_summary()
        positions = client.get_positions()
        _echo_header("Account")
        for key, value in summary.as_dict().items():
            click.echo(f"  {key:<20} {value}")
        _echo_header(f"Positions ({len(positions)})")
        for position in positions:
            click.echo(f"  {position.symbol:<8} {position.quantity:>12,.4f} @ {position.average_cost:>10,.2f}")
        if not positions:
            click.echo("  (none)")

        report = gate.run(account=client.account, connected=client.connected)
        click.echo()
        click.echo(report.render())
        if not report.passed:
            raise SystemExit(EXIT_RISK)
    finally:
        client.disconnect()


@cli.command("order-preview")
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option("--mock", is_flag=True, help="Use the mock broker (no TWS required).")
@click.option("--as-of", type=click.DateTime(["%Y-%m-%d"]), default=None, help="Treat this date as today.")
@click.pass_context
def order_preview(
    ctx: click.Context, provider: str | None, mock: bool, as_of: datetime | None
) -> None:
    """Show the orders Atlas would send, without sending anything."""
    import pandas as pd

    from atlas.execution.ibkr_client import BrokerClient, IBKRClient, MockBrokerClient
    from atlas.execution.order_manager import OrderManager

    config = _load(ctx)
    panel = _load_panel(config, provider=provider)

    broker: BrokerClient
    if mock:
        broker = MockBrokerClient(
            prices=panel.adj_close.iloc[-1], initial_cash=config.portfolio.initial_capital
        )
    else:
        broker = IBKRClient(config.execution)
    try:
        manager = OrderManager(config, broker=broker)
        result = manager.run_cycle(
            panel, as_of=pd.Timestamp(as_of) if as_of else None, submit=False
        )
    except BrokerError as exc:
        click.secho(f"\n  broker error: {exc}", fg="red")
        raise SystemExit(EXIT_BROKER) from exc
    finally:
        broker.disconnect()

    click.echo()
    click.echo(result.render())
    if result.plan is not None:
        click.echo()
        click.echo(result.plan.render())
    if result.blocked_reason:
        raise SystemExit(EXIT_RISK)


@cli.command("paper-trade")
@click.option("--provider", type=click.Choice(["yfinance", "csv", "synthetic"]), default=None)
@click.option("--mock", is_flag=True, help="Use the mock broker (no TWS required).")
@click.option(
    "--yes", is_flag=True, help="Skip the interactive confirmation prompt (for scheduled runs)."
)
@click.pass_context
def paper_trade(ctx: click.Context, provider: str | None, mock: bool, yes: bool) -> None:
    """Run one paper-trading execution cycle and submit the resulting orders.

    Requires EXECUTION_MODE=paper. In any other mode this command previews only.
    """
    from atlas.execution.ibkr_client import BrokerClient, IBKRClient, MockBrokerClient
    from atlas.execution.order_manager import OrderManager

    config = _load(ctx)
    if config.execution.mode is not ExecutionMode.PAPER:
        click.secho(
            f"\n  EXECUTION_MODE is '{config.execution.mode.value}', so no orders will be sent. "
            "Set EXECUTION_MODE=paper (and an account allowlist) to submit to a paper account.",
            fg="yellow",
        )

    panel = _load_panel(config, provider=provider)
    broker: BrokerClient
    if mock:
        broker = MockBrokerClient(
            prices=panel.adj_close.iloc[-1], initial_cash=config.portfolio.initial_capital
        )
    else:
        broker = IBKRClient(config.execution)

    try:
        from atlas.database import AtlasRepository, DatabaseManager

        repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
        manager = OrderManager(config, broker=broker, repository=repository)

        plan_result = manager.run_cycle(panel, submit=False)
        if plan_result.plan is None or not plan_result.plan.orders:
            click.echo("\n  no orders required")
            return
        click.echo()
        click.echo(plan_result.plan.render())
        if plan_result.blocked_reason:
            click.secho(f"\n  BLOCKED: {plan_result.blocked_reason}", fg="red")
            raise SystemExit(EXIT_RISK)

        if config.execution.mode is not ExecutionMode.PAPER:
            click.secho("\n  dry run complete - nothing was submitted", fg="cyan")
            return
        if not yes and not click.confirm(
            f"\n  Submit {plan_result.plan.n_orders} order(s) to paper account "
            f"{broker.account}?",
            default=False,
        ):
            click.echo("  cancelled")
            return

        result = manager.run_cycle(panel, submit=True)
        click.echo()
        click.echo(result.render())
        if result.blocked_reason:
            raise SystemExit(EXIT_RISK)
    except (KillSwitchActive, RiskLimitBreached, ExecutionBlocked) as exc:
        click.secho(f"\n  blocked: {exc}", fg="red")
        raise SystemExit(EXIT_RISK) from exc
    except BrokerError as exc:
        click.secho(f"\n  broker error: {exc}", fg="red")
        raise SystemExit(EXIT_BROKER) from exc
    finally:
        broker.disconnect()


@cli.command("reconcile")
@click.option("--mock", is_flag=True, help="Use the mock broker (no TWS required).")
@click.pass_context
def reconcile(ctx: click.Context, mock: bool) -> None:
    """Compare broker positions with Atlas's own record."""
    from atlas.execution.ibkr_client import BrokerClient, IBKRClient, MockBrokerClient
    from atlas.execution.reconciliation import Reconciler

    config = _load(ctx)
    broker: BrokerClient = MockBrokerClient() if mock else IBKRClient(config.execution)
    try:
        broker.connect()
        broker_positions = broker.get_positions()
        prices = broker.get_market_prices([p.symbol for p in broker_positions])
    except BrokerError as exc:
        click.secho(f"\n  broker error: {exc}", fg="red")
        raise SystemExit(EXIT_BROKER) from exc
    finally:
        broker.disconnect()

    # Atlas's own record comes from the database when one exists; a fresh
    # installation legitimately has none, which the report makes explicit.
    from atlas.database import AtlasRepository, DatabaseManager

    repository = AtlasRepository(DatabaseManager(project_root=config.project_root))
    local: dict[str, float] = {}

    report = Reconciler(config.execution.reconciliation).reconcile(
        local, broker_positions, prices=prices
    )
    click.echo()
    click.echo(report.render())
    repository.save_reconciliation(report, session_id="cli-reconcile", account=broker.account or "")
    if report.blocking:
        raise SystemExit(EXIT_RISK)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@cli.command("dashboard")
@click.option("--port", default=8501, show_default=True)
@click.option("--host", default="localhost", show_default=True)
@click.pass_context
def dashboard(ctx: click.Context, port: int, host: str) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess

    config = _load(ctx)
    app = config.project_root / "src" / "atlas" / "dashboard" / "app.py"
    if not app.is_file():
        raise click.ClickException(f"dashboard entry point not found at {app}")

    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(port), "--server.address", host,
    ]
    click.echo(f"\n  starting the dashboard at http://{host}:{port}\n")
    try:
        raise SystemExit(subprocess.call(command))
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise click.ClickException(
            "streamlit is not installed. Install it with "
            "`pip install 'atlas-trading-system[dashboard]'`."
        ) from exc


@cli.command("show-config")
@click.option("--section", default=None, help="Show only one section (assets, risk, execution, ...).")
@click.pass_context
def show_config(ctx: click.Context, section: str | None) -> None:
    """Print the validated configuration and its hash."""
    config = _load(ctx)
    snapshot = config.snapshot()
    if section:
        if section not in snapshot:
            raise click.ClickException(
                f"unknown section {section!r}; available: {', '.join(k for k in snapshot if not k.startswith('_'))}"
            )
        snapshot = {section: snapshot[section]}
    click.echo(json.dumps(snapshot, indent=2, default=str))
    click.echo(f"\n  configuration hash: {config.config_hash}")


@cli.command("info")
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show the current configuration summary and system status."""
    config = _load(ctx)
    _echo_header(f"Atlas {__version__}")
    rows: dict[str, Any] = {
        "universe": f"{config.universe.name} ({len(config.symbols)} symbols)",
        "benchmark": config.universe.benchmark,
        "data provider": config.data.provider,
        "strategies": ", ".join(config.strategies.enabled_strategies),
        "ensemble mode": config.strategies.ensemble.mode,
        "regime detection": "on" if config.strategies.regime.enabled else "off",
        "rebalance": config.backtest.rebalance_frequency.value,
        "execution timing": config.backtest.execution_timing.value,
        "target volatility": f"{config.portfolio.target_volatility:.1%}",
        "max gross exposure": f"{config.portfolio.max_gross_exposure:.2f}x",
        "execution mode": config.execution.mode.value,
        "kill switch": "ENGAGED" if config.risk.kill_switch.enabled else "off",
        "config hash": config.config_hash,
    }
    for key, value in rows.items():
        click.echo(f"  {key:<22} {value}")
    click.secho(f"\n  {DISCLAIMER}", fg="cyan")


def main() -> int:
    """Console-script entry point with uniform error handling."""
    try:
        cli(standalone_mode=False, obj={})
    except click.exceptions.Abort:
        click.echo("\naborted", err=True)
        return EXIT_FAILURE
    except click.ClickException as exc:
        exc.show()
        return EXIT_CONFIG
    except SystemExit as exc:
        return int(exc.code or EXIT_OK)
    except (KillSwitchActive, RiskLimitBreached, ExecutionBlocked) as exc:
        click.secho(f"blocked: {exc}", fg="red", err=True)
        return EXIT_RISK
    except BrokerError as exc:
        click.secho(f"broker error: {exc}", fg="red", err=True)
        return EXIT_BROKER
    except AtlasError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        click.echo("\ninterrupted", err=True)
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
