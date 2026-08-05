"""Automated research reports.

Every backtest can produce a self-contained HTML (or Markdown) report with the
structure a research note would have: what was tested, on what data, under what
assumptions, what happened in and out of sample, and - most importantly - what
the limitations are.

The report never states a number the run did not compute. Where a section has no
data (no walk-forward was run, no stress window overlapped the sample), it says
so explicitly rather than omitting the section silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from atlas import DISCLAIMER, __version__
from atlas.backtest.engine import BacktestResult
from atlas.config import AtlasConfig
from atlas.logging_utils import get_logger
from atlas.reporting import charts

log = get_logger(__name__)

__all__ = ["ReportBundle", "ResearchReport"]


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown pipe table.

    ``DataFrame.to_markdown`` needs the optional ``tabulate`` package. Rendering
    a pipe table directly keeps the Markdown report working with no extra
    dependency.
    """
    if frame.empty:
        return "*(no data)*"
    display = frame.reset_index() if frame.index.name or not isinstance(
        frame.index, pd.RangeIndex
    ) else frame
    headers = [str(c) for c in display.columns]

    def _cell(value: object) -> str:
        if isinstance(value, float):
            return "" if not np.isfinite(value) else f"{value:,.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _strip_html(text: str) -> str:
    """Convert the small set of HTML fragments used here into plain Markdown."""
    import re

    text = re.sub(r"<li>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h4>(.*?)</h4>", r"### \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<caption>(.*?)</caption>", r"**\1**\n\n", text, flags=re.DOTALL)
    text = re.sub(r"<tr><th>(.*?)</th><td>(.*?)</td></tr>", r"- **\1**: \2\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


@dataclass
class ReportBundle:
    """The inputs a report can draw on. Everything except ``result`` is optional."""

    result: BacktestResult
    config: AtlasConfig
    panel: Any | None = None
    benchmarks: Any | None = None
    walk_forward: Any | None = None
    stability: Any | None = None
    stress: Any | None = None
    bootstrap: pd.DataFrame | None = None
    label: str = "Atlas baseline experiment"
    extra_notes: list[str] = field(default_factory=list)


class ResearchReport:
    """Render a :class:`ReportBundle` as HTML or Markdown."""

    def __init__(self, bundle: ReportBundle) -> None:
        self.bundle = bundle
        self.result = bundle.result
        self.config = bundle.config
        self.generated_at = datetime.now(UTC)

    # -- public API -----------------------------------------------------------

    def to_html(self, path: str | Path) -> Path:
        """Write the full HTML report and return its path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sections = self._sections()
        html = self._html_document(sections)
        destination.write_text(html, encoding="utf-8")
        log.info(
            "report written",
            extra={"context": {"path": str(destination), "size_kb": round(len(html) / 1024, 1)}},
        )
        return destination

    def to_markdown(self, path: str | Path) -> Path:
        """Write a Markdown report (tables only, no embedded figures)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        parts = [f"# {self.bundle.label}", "", f"*Generated {self.generated_at:%Y-%m-%d %H:%M UTC} "
                 f"by Atlas {__version__}*", ""]
        if self.result.warnings:
            parts.append("> **Warnings**")
            for warning in self.result.warnings:
                parts.append(f"> - {warning}")
            parts.append("")
        for title, blocks in self._sections():
            parts.append(f"## {title}")
            parts.append("")
            for block in blocks:
                if isinstance(block, go.Figure):
                    parts.append(f"*(figure: {block.layout.title.text or 'chart'})*")
                elif isinstance(block, pd.DataFrame):
                    parts.append(_markdown_table(block))
                elif isinstance(block, pd.Series):
                    parts.append(_markdown_table(block.to_frame()))
                else:
                    parts.append(_strip_html(str(block)))
                parts.append("")
        parts.extend(["---", "", f"**Disclaimer.** {DISCLAIMER}"])
        destination.write_text("\n".join(parts), encoding="utf-8")
        log.info("markdown report written", extra={"context": {"path": str(destination)}})
        return destination

    # -- sections -------------------------------------------------------------

    def _sections(self) -> list[tuple[str, list[Any]]]:
        """Build the ordered list of report sections."""
        return [
            ("1. Executive summary", self._executive_summary()),
            ("2. Dataset and time period", self._dataset()),
            ("3. Asset universe", self._universe()),
            ("4. Strategy definitions", self._strategies()),
            ("5. Portfolio construction", self._portfolio_construction()),
            ("6. Risk controls", self._risk_controls()),
            ("7. Cost assumptions", self._costs()),
            ("8. In-sample results", self._in_sample()),
            ("9. Out-of-sample results", self._out_of_sample()),
            ("10. Benchmark comparison", self._benchmarks()),
            ("11. Drawdown analysis", self._drawdown()),
            ("12. Stress-period analysis", self._stress()),
            ("13. Regime analysis", self._regimes()),
            ("14. Strategy contribution", self._contributions()),
            ("15. Parameter sensitivity", self._sensitivity()),
            ("16. Statistical resampling", self._bootstrap()),
            ("17. Limitations", self._limitations()),
            ("18. Conclusion", self._conclusion()),
        ]

    def _executive_summary(self) -> list[Any]:
        metrics = self.result.metrics
        if metrics is None:
            return ["<p>No metrics were computed for this run.</p>"]

        drag = self.result.cost_drag()
        rows = {
            "Total return": f"{metrics.values.get('total_return', 0):.2%}",
            "CAGR": f"{metrics.cagr:.2%}",
            "Annualised volatility": f"{metrics.values.get('annualized_volatility', 0):.2%}",
            "Sharpe ratio": f"{metrics.sharpe:.2f}",
            "Sortino ratio": f"{metrics.values.get('sortino_ratio', 0):.2f}",
            "Calmar ratio": f"{metrics.values.get('calmar_ratio', 0):.2f}",
            "Maximum drawdown": f"{metrics.max_drawdown:.2%}",
            "Trades": f"{self.result.n_trades:,}",
            "Transaction costs": f"${self.result.total_costs:,.0f}",
            "Cost drag (annual)": f"{drag.get('cost_drag_annual', 0):.2%}",
        }
        blocks: list[Any] = [
            self._kv_table(rows, "Headline results (net of all modelled costs)"),
            (
                "<p>The Sharpe and Sortino ratios assume an annualised risk-free rate of "
                f"<b>{self.config.validation.benchmarks.risk_free_rate:.2%}</b>, de-annualised "
                "geometrically and subtracted from each day's return, and are annualised by "
                "&radic;252.</p>"
            ),
        ]
        if self.result.warnings:
            blocks.insert(0, self._callout(self.result.warnings, kind="warning"))
        return blocks

    def _dataset(self) -> list[Any]:
        equity = self.result.equity_curve
        rows = {
            "Data source": self.result.data_source,
            "Synthetic data": "YES - not real market data" if self.result.is_synthetic_data else "no",
            "Backtest start": str(equity.index.min().date()) if len(equity) else "-",
            "Backtest end": str(equity.index.max().date()) if len(equity) else "-",
            "Trading days": f"{len(equity):,}",
            "Rebalance frequency": self.config.backtest.rebalance_frequency.value,
            "Execution timing": self.config.backtest.execution_timing.value,
            "Price series": self.config.backtest.price_field,
            "Run id": self.result.run_id,
            "Git commit": (self.result.git_commit or "unknown")[:12],
            "Configuration hash": self.config.config_hash,
        }
        note = (
            "<p>Signals are computed from the close of day <i>t</i> and executed at the "
            "<b>open of day t+1</b>. Same-close execution is available only as an explicitly "
            "labelled unrealistic research mode.</p>"
        )
        return [self._kv_table(rows, "Run parameters"), note]

    def _universe(self) -> list[Any]:
        universe = self.config.universe
        frame = pd.DataFrame(
            [
                {"Symbol": a.symbol, "Name": a.name, "Asset class": a.asset_class, "Tradable": a.tradable}
                for a in universe.assets
            ]
        )
        blocks: list[Any] = [frame]
        if self.bundle.panel is not None:
            blocks.append(self.bundle.panel.describe().reset_index())
        return blocks

    def _strategies(self) -> list[Any]:
        block = self.config.strategies.strategies
        blocks: list[Any] = []
        descriptions = {
            "trend": (
                "Time-series trend following. Blends price relative to a fast and a slow moving "
                "average with volatility-normalised 63-day and 252-day returns. Rationale: the "
                "documented time-series momentum premium (Moskowitz, Ooi & Pedersen 2012)."
            ),
            "mean_reversion": (
                "Short-horizon mean reversion. Fades large short-term moves measured by a z-score "
                "and RSI, with trend and volatility filters and a maximum holding period. "
                "Rationale: compensation for short-term liquidity provision (Nagel 2012)."
            ),
            "cross_sectional_momentum": (
                "Cross-sectional momentum. Ranks assets by risk-adjusted trailing return, skipping "
                "the most recent month, optionally neutralised within asset class. Rationale: "
                "relative-strength persistence (Jegadeesh & Titman 1993)."
            ),
        }
        for name, cfg in block.as_dict().items():
            status = "enabled" if cfg.enabled else "disabled"
            blocks.append(f"<h4>{name} ({status})</h4><p>{descriptions.get(name, '')}</p>")
            blocks.append(self._kv_table(cfg.model_dump(mode="json"), f"{name} parameters"))
        ensemble = self.config.strategies.ensemble
        blocks.append(
            f"<p>Signals are combined in <b>{ensemble.mode}</b> mode with exponential smoothing "
            f"(alpha = {ensemble.signal_smoothing_alpha}); strategy weights always sum to one "
            "before any risk scaling, so the ensemble changes the mix, never the total risk.</p>"
        )
        return blocks

    def _portfolio_construction(self) -> list[Any]:
        portfolio = self.config.portfolio
        method = (
            "<p>Combined signals are scaled by the inverse of each asset's volatility, normalised "
            "to unit gross exposure, then scaled by "
            "<code>target_volatility / sqrt(w' &Sigma; w)</code> to hit the volatility target. "
            f"Covariance is estimated with the <b>{self.config.risk.covariance.method}</b> "
            f"estimator over {self.config.risk.covariance.lookback} days, repaired to positive "
            "definite when necessary.</p>"
        )
        return [method, self._kv_table(portfolio.model_dump(mode="json"), "Portfolio limits")]

    def _risk_controls(self) -> list[Any]:
        blocks: list[Any] = [
            self._kv_table(self.config.risk.limits.model_dump(mode="json"), "Hard risk limits"),
        ]
        ladder = self.config.risk.drawdown_controls
        if ladder.enabled and ladder.thresholds:
            frame = pd.DataFrame(
                [
                    {"Drawdown at least": f"{t.drawdown:.0%}", "Risk multiplier": f"{t.multiplier:.2f}"}
                    for t in ladder.thresholds
                ]
            )
            blocks.append("<h4>Drawdown-based risk reduction</h4>")
            blocks.append(frame)
            blocks.append(
                "<p>The ladder scales the portfolio's risk budget only. Atlas never liquidates "
                "automatically on drawdown unless <code>liquidate_on_breach</code> is explicitly "
                "enabled.</p>"
            )
        events = self.result.risk_events
        if not events.empty and "control" in events.columns:
            counts = events["control"].value_counts().rename("times fired").to_frame()
            blocks.append("<h4>Risk controls that fired during the run</h4>")
            blocks.append(counts.reset_index().rename(columns={"index": "control"}))
        else:
            blocks.append("<p>No risk control fired during this run.</p>")
        return blocks

    def _costs(self) -> list[Any]:
        from atlas.backtest.costs import TransactionCostModel

        model = TransactionCostModel(self.config.costs)
        breakdown = self.result.cost_breakdown()
        drag = self.result.cost_drag()
        blocks: list[Any] = [
            self._kv_table(model.describe(), "Cost assumptions"),
            self._kv_table(
                {k: f"${v:,.2f}" for k, v in breakdown.items()}, "Realised costs by category"
            ),
            self._kv_table(
                {
                    "Gross CAGR (costs added back)": f"{drag.get('gross_cagr', 0):.2%}",
                    "Net CAGR": f"{drag.get('net_cagr', 0):.2%}",
                    "Annual cost drag": f"{drag.get('cost_drag_annual', 0):.2%}",
                    "Share of gross return lost to costs": (
                        f"{drag.get('cost_share_of_gross_return', float('nan')):.1%}"
                        if np.isfinite(drag.get("cost_share_of_gross_return", float("nan")))
                        else "n/a"
                    ),
                },
                "Cost impact on performance",
            ),
        ]
        if not self.result.snapshots.empty:
            blocks.append(charts.turnover_cost_chart(self.result.snapshots, self.result.fills))
        return blocks

    def _in_sample(self) -> list[Any]:
        metrics = self.result.metrics
        blocks: list[Any] = [
            self._callout(
                [
                    "The results in this section come from a single pass over the full sample. "
                    "They are IN-SAMPLE and should be read as a description of the system's "
                    "behaviour, not as evidence of predictive skill. See section 9."
                ],
                kind="note",
            )
        ]
        if metrics is not None:
            blocks.append(metrics.to_series().to_frame("value"))
        curves = {"Atlas": self.result.equity_curve}
        blocks.append(charts.equity_curve_chart(curves, title="Atlas equity curve"))
        if not self.result.returns.empty:
            blocks.append(charts.monthly_heatmap(self.result.returns))
            blocks.append(charts.rolling_sharpe_chart(
                self.result.returns, risk_free_rate=self.config.validation.benchmarks.risk_free_rate
            ))
            blocks.append(charts.rolling_volatility_chart(
                self.result.returns, target=self.config.portfolio.target_volatility
            ))
            blocks.append(charts.return_distribution_chart(self.result.returns))
        return blocks

    def _out_of_sample(self) -> list[Any]:
        wf = self.bundle.walk_forward
        if wf is None:
            return [
                self._callout(
                    [
                        "No walk-forward validation was run for this report, so there are no "
                        "out-of-sample results. Run <code>atlas walk-forward</code> and regenerate "
                        "the report before drawing any conclusion about predictive performance."
                    ],
                    kind="warning",
                )
            ]
        blocks: list[Any] = []
        if wf.test_metrics is not None:
            blocks.append(
                self._kv_table(
                    {
                        "Out-of-sample total return": f"{wf.test_metrics.values.get('total_return', 0):.2%}",
                        "Out-of-sample CAGR": f"{wf.test_metrics.cagr:.2%}",
                        "Out-of-sample Sharpe": f"{wf.test_metrics.sharpe:.2f}",
                        "Out-of-sample max drawdown": f"{wf.test_metrics.max_drawdown:.2%}",
                        "Windows": f"{len(wf.windows)}",
                        "Out-of-sample observations": f"{len(wf.test_returns):,}",
                    },
                    "Concatenated out-of-sample test periods",
                )
            )
        degradation = wf.degradation()
        if degradation:
            blocks.append(
                self._kv_table(
                    {k: f"{v:.3f}" for k, v in degradation.items()},
                    "In-sample versus out-of-sample degradation",
                )
            )
            blocks.append(
                "<p>A large positive <code>train_minus_test_sharpe</code> means the parameter "
                "selection was fitting noise. This number matters more than the headline "
                "out-of-sample Sharpe.</p>"
            )
        if not wf.selections.empty:
            blocks.append(charts.walk_forward_chart(wf.selections))
            blocks.append(wf.selections)
        stability = wf.parameter_stability()
        if not stability.empty:
            blocks.append("<h4>How often each parameter value was selected</h4>")
            blocks.append(stability)
        for warning in getattr(wf, "warnings", []):
            blocks.append(self._callout([warning], kind="warning"))
        return blocks

    def _benchmarks(self) -> list[Any]:
        bench = self.bundle.benchmarks
        if bench is None:
            return ["<p>No benchmark suite was computed for this report.</p>"]
        blocks: list[Any] = [bench.table().round(4)]
        curves = bench.equity_curves()
        if not curves.empty:
            blocks.append(
                charts.equity_curve_chart(
                    {c: curves[c] for c in curves.columns},
                    title="Atlas versus benchmarks",
                )
            )
        correlation = bench.correlation()
        if not correlation.empty:
            blocks.append(charts._correlation_figure(correlation, "Benchmark return correlation"))
        if bench.notes:
            blocks.append(self._callout(bench.notes, kind="note"))
        return blocks

    def _drawdown(self) -> list[Any]:
        blocks: list[Any] = [charts.drawdown_chart(self.result.equity_curve)]
        metrics = self.result.metrics
        if metrics is not None:
            keys = [
                "max_drawdown", "average_drawdown", "max_drawdown_duration_days",
                "average_drawdown_duration_days", "max_recovery_duration_days",
                "n_drawdown_episodes", "time_in_drawdown_pct",
            ]
            blocks.append(
                self._kv_table(
                    {k: f"{metrics.values.get(k, 0):.4f}" for k in keys}, "Drawdown statistics"
                )
            )
        return blocks

    def _stress(self) -> list[Any]:
        stress = self.bundle.stress
        if stress is None or stress.periods.empty:
            return [
                "<p>No stress window overlapped this backtest's date range, so no stress results "
                "are reported.</p>"
            ]
        blocks: list[Any] = [
            stress.periods.round(4),
            self._callout(
                [
                    "A system that survived one crisis has not been shown to survive the next: "
                    "crises differ in mechanism. Read this table as a panel, looking for "
                    "consistency across windows rather than a good result in any one."
                ],
                kind="note",
            ),
        ]
        if stress.skipped:
            blocks.append(self._callout(stress.skipped, kind="note"))
        return blocks

    def _regimes(self) -> list[Any]:
        regimes = self.result.regimes
        if regimes is None:
            return ["<p>Regime detection was disabled for this run.</p>"]
        blocks: list[Any] = [
            charts.regime_timeline_chart(regimes.labels, equity=self.result.equity_curve),
            regimes.distribution().rename("share of days").to_frame().reset_index(),
        ]
        transitions = regimes.transitions()
        if not transitions.empty:
            blocks.append("<h4>Regime transitions</h4>")
            blocks.append(transitions)

        # Performance by regime, computed from the realised return series.
        if not self.result.returns.empty:
            labels = regimes.as_strings().reindex(self.result.returns.index).ffill()
            grouped = self.result.returns.groupby(labels)
            rows = []
            for name, group in grouped:
                if len(group) < 5:
                    continue
                rows.append(
                    {
                        "regime": name,
                        "days": len(group),
                        "annualised return": float((1 + group).prod() ** (252 / len(group)) - 1),
                        "annualised volatility": float(group.std(ddof=1) * np.sqrt(252)),
                        "worst day": float(group.min()),
                        "share of days": float(len(group) / len(self.result.returns)),
                    }
                )
            if rows:
                blocks.append("<h4>Portfolio performance by regime</h4>")
                blocks.append(pd.DataFrame(rows).round(4))
        return blocks

    def _contributions(self) -> list[Any]:
        blocks: list[Any] = []
        share = self.result.strategy_contributions
        if not share.empty:
            blocks.append(charts.strategy_contribution_chart(share))
            blocks.append(
                share.mean().rename("average share of combined signal").to_frame().round(4).reset_index()
            )
        if self.bundle.panel is not None and not self.result.weights.empty:
            from atlas.backtest.metrics import return_contribution

            contribution = return_contribution(
                self.result.weights, self.bundle.panel.returns("adj_close")
            )
            blocks.append("<h4>Return contribution by asset</h4>")
            blocks.append(contribution.rename("cumulative contribution").to_frame().round(4).reset_index())
        if not self.result.weights.empty:
            blocks.append(charts.weights_chart(self.result.weights))
        if not self.result.snapshots.empty:
            blocks.append(charts.exposure_chart(self.result.snapshots))
        return blocks

    def _sensitivity(self) -> list[Any]:
        stability = self.bundle.stability
        if stability is None or stability.sweeps.empty:
            return [
                "<p>No parameter-stability sweep was run for this report. Run "
                "<code>atlas parameter-stability</code> to test whether the results depend on "
                "specific parameter values.</p>"
            ]
        blocks: list[Any] = [stability.verdicts.round(4)]
        for parameter in stability.sweeps["parameter"].unique():
            blocks.append(charts.parameter_sensitivity_chart(stability.sweeps, str(parameter)))
        if stability.fragile_parameters:
            blocks.append(
                self._callout(
                    [
                        "The following parameters were flagged FRAGILE - performance depends "
                        f"materially on the specific value chosen: {', '.join(stability.fragile_parameters)}. "
                        "Treat any result that relies on them with scepticism."
                    ],
                    kind="warning",
                )
            )
        return blocks

    def _bootstrap(self) -> list[Any]:
        paths = self.bundle.bootstrap
        if paths is None or paths.empty:
            return ["<p>No bootstrap analysis was run for this report.</p>"]
        from atlas.validation.stress_tests import bootstrap_statistics

        blocks: list[Any] = [
            self._callout(
                [
                    "Block bootstrap resamples contiguous blocks of historical returns to preserve "
                    "serial dependence and volatility clustering. This is a STATISTICAL STRESS "
                    "TEST that describes the range of outcomes consistent with the historical "
                    "return distribution. It is not a forecast."
                ],
                kind="note",
            ),
            bootstrap_statistics(paths, self.config.validation.monte_carlo.percentiles).round(4),
            charts.bootstrap_distribution_chart(paths, "annualized_return"),
            charts.bootstrap_distribution_chart(paths, "max_drawdown"),
        ]
        return blocks

    def _limitations(self) -> list[Any]:
        items = [
            "<b>Backtests are not out-of-sample by construction.</b> Only the walk-forward test "
            "slices in section 9 are genuinely unseen; every other number in this report was "
            "computed on data that was available when the parameters were chosen.",
            "<b>Survivorship and selection bias.</b> The universe is a fixed list of ETFs that "
            "exist today and are liquid today. Funds that closed are absent, and the choice of "
            "these fourteen instruments is itself a decision made with hindsight.",
            "<b>Cost model is an approximation.</b> Commission, half-spread, slippage and a "
            "square-root impact term are modelled, but the real spread varies through the day and "
            "widens exactly when the strategy most wants to trade.",
            "<b>No intraday or latency modelling.</b> Orders fill at the next open regardless of "
            "what happens between the signal and the fill; a gap through the open is not modelled.",
            "<b>Adjusted prices are revised.</b> Total-return series from a free vendor can change "
            "retroactively, so results computed today may not reproduce exactly next year.",
            "<b>Paper fills are not real fills.</b> Paper trading assumes fills that a real "
            "market may not provide, especially in size or in stress.",
            "<b>Limited sample of crises.</b> The dataset contains a handful of stress episodes. "
            "Statistical confidence about tail behaviour from that sample is low.",
            "<b>Parameter selection still touches the data.</b> Even a small grid, selected on "
            "validation slices, uses information from the sample; the out-of-sample degradation "
            "figures in section 9 are the honest measure of how much that cost.",
        ]
        if self.result.is_synthetic_data:
            items.insert(
                0,
                "<b>THIS RUN USED SYNTHETIC DATA.</b> The prices behind every number in this "
                "report were simulated, not observed. The results demonstrate that the system "
                "runs; they say nothing whatsoever about performance on real markets.",
            )
        return ["<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"]

    def _conclusion(self) -> list[Any]:
        metrics = self.result.metrics
        wf = self.bundle.walk_forward
        parts: list[str] = []

        if self.result.is_synthetic_data:
            parts.append(
                "<p>This run used <b>simulated data</b>. The correct conclusion is only that the "
                "pipeline executes end to end and that its risk controls behave as specified. No "
                "conclusion about market performance can be drawn.</p>"
            )
        elif metrics is not None:
            parts.append(
                f"<p>Over the full sample the system produced a CAGR of {metrics.cagr:.2%} at "
                f"{metrics.values.get('annualized_volatility', 0):.2%} annualised volatility "
                f"(Sharpe {metrics.sharpe:.2f}), with a maximum drawdown of "
                f"{metrics.max_drawdown:.2%}.</p>"
            )
        if wf is not None and wf.test_metrics is not None:
            degradation = wf.degradation()
            parts.append(
                f"<p>Out of sample, across {len(wf.windows)} walk-forward windows, the Sharpe "
                f"ratio was {wf.test_metrics.sharpe:.2f} against an in-sample "
                f"{degradation.get('mean_train_sharpe', float('nan')):.2f} - a degradation of "
                f"{degradation.get('train_minus_test_sharpe', float('nan')):.2f} Sharpe units. "
                "That gap, not the in-sample figure, is the honest estimate of what this "
                "process delivers on data it has not seen.</p>"
            )
        else:
            parts.append(
                "<p>No walk-forward evaluation accompanies this run, so no claim about "
                "out-of-sample performance is supported by it.</p>"
            )
        parts.append(
            "<p>Atlas is a research and paper-trading platform. Its purpose is to make "
            "assumptions explicit, keep risk controls independent of strategy logic, and report "
            "what actually happened - including the costs and the degradation.</p>"
        )
        return parts

    # -- rendering helpers ----------------------------------------------------

    @staticmethod
    def _kv_table(mapping: dict[str, Any], caption: str = "") -> str:
        """Render a mapping as a two-column HTML table."""
        rows = "".join(
            f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in mapping.items()
        )
        caption_html = f"<caption>{caption}</caption>" if caption else ""
        return f'<table class="kv">{caption_html}{rows}</table>'

    @staticmethod
    def _callout(messages: list[str], *, kind: str = "note") -> str:
        """Render a highlighted callout block."""
        items = "".join(f"<li>{m}</li>" for m in messages)
        return f'<div class="callout {kind}"><ul>{items}</ul></div>'

    def _html_document(self, sections: list[tuple[str, list[Any]]]) -> str:
        """Assemble the complete HTML document."""
        body: list[str] = []
        toc: list[str] = []

        for title, blocks in sections:
            anchor = title.lower().replace(" ", "-").replace(".", "")
            toc.append(f'<li><a href="#{anchor}">{title}</a></li>')
            body.append(f'<section id="{anchor}"><h2>{title}</h2>')
            for block in blocks:
                body.append(self._render_block(block))
            body.append("</section>")

        equity = self.result.equity_curve
        subtitle = (
            f"{equity.index.min().date()} to {equity.index.max().date()}"
            if len(equity)
            else "no data"
        )
        warning_banner = ""
        if self.result.is_synthetic_data:
            warning_banner = (
                '<div class="banner">SYNTHETIC DATA - these results were computed from simulated '
                "prices and carry no information about real market performance.</div>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self.bundle.label}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{ --ink:#1a1a1a; --muted:#5a6672; --line:#e2e6ea; --accent:#0072B2; --warn:#D55E00; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--ink); margin: 0; padding: 0 0 4rem; line-height: 1.55; background:#fff; }}
header {{ background: linear-gradient(135deg,#0b3d62,#0072B2); color:#fff; padding: 2.5rem 2rem 2rem; }}
header h1 {{ margin:0 0 .35rem; font-size: 1.9rem; }}
header p {{ margin:.15rem 0; opacity:.92; font-size:.95rem; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 2rem; }}
.banner {{ background: var(--warn); color:#fff; padding:.9rem 2rem; font-weight:600; letter-spacing:.02em; }}
nav {{ background:#f6f8fa; border-bottom:1px solid var(--line); padding:1rem 0; }}
nav ol {{ columns: 3; margin:0; padding-left:1.25rem; font-size:.9rem; }}
nav a {{ color: var(--accent); text-decoration:none; }}
nav a:hover {{ text-decoration:underline; }}
section {{ padding: 2rem 0 1rem; border-bottom:1px solid var(--line); }}
h2 {{ font-size:1.35rem; margin:0 0 1rem; color:#0b3d62; }}
h4 {{ margin:1.4rem 0 .5rem; font-size:1.02rem; color:#334; }}
table {{ border-collapse: collapse; margin:1rem 0; font-size:.9rem; width:100%; }}
table caption {{ text-align:left; font-weight:600; padding-bottom:.4rem; color:var(--muted); }}
th, td {{ border:1px solid var(--line); padding:.45rem .7rem; text-align:left; }}
th {{ background:#f6f8fa; font-weight:600; }}
table.kv th {{ width: 42%; }}
td {{ font-variant-numeric: tabular-nums; }}
.table-scroll {{ overflow-x:auto; }}
.callout {{ border-left:4px solid var(--accent); background:#f2f8fc; padding:.8rem 1rem; margin:1rem 0; border-radius:0 4px 4px 0; }}
.callout.warning {{ border-left-color: var(--warn); background:#fdf3ee; }}
.callout ul {{ margin:0; padding-left:1.1rem; }}
.chart {{ margin:1.2rem 0; }}
footer {{ padding:2rem; color:var(--muted); font-size:.87rem; max-width:1180px; margin:0 auto; }}
code {{ background:#f2f4f6; padding:.1rem .3rem; border-radius:3px; font-size:.88em; }}
</style>
</head>
<body>
{warning_banner}
<header>
  <div class="wrap">
    <h1>{self.bundle.label}</h1>
    <p>Atlas {__version__} &middot; regime-aware multi-strategy research platform</p>
    <p>Period: {subtitle} &middot; Run: <code>{self.result.run_id}</code></p>
    <p>Generated {self.generated_at:%Y-%m-%d %H:%M UTC}</p>
  </div>
</header>
<nav><div class="wrap"><ol>{''.join(toc)}</ol></div></nav>
<main class="wrap">
{''.join(body)}
</main>
<footer>
  <p><b>Disclaimer.</b> {DISCLAIMER}</p>
  <p>Configuration hash <code>{self.config.config_hash}</code> &middot;
     git commit <code>{(self.result.git_commit or 'unknown')[:12]}</code></p>
</footer>
</body>
</html>"""

    def _render_block(self, block: Any) -> str:
        """Render one content block as an HTML fragment."""
        if isinstance(block, go.Figure):
            html = pio.to_html(
                block, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
            )
            return f'<div class="chart">{html}</div>'
        if isinstance(block, pd.DataFrame):
            return f'<div class="table-scroll">{block.to_html(index=False, border=0, escape=False)}</div>'
        if isinstance(block, pd.Series):
            return f'<div class="table-scroll">{block.to_frame().to_html(border=0)}</div>'
        return str(block)
