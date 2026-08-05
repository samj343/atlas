"""Exporting results to disk.

Everything a run produced can be written to a single directory: the equity curve,
daily snapshots, weights, fills, orders, risk events, regimes and metrics, plus a
``manifest.json`` recording the run id, git commit and configuration hash. That
directory is the reproducible artefact of a research run.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["export_backtest", "export_figures", "load_manifest", "write_json"]


def _json_default(value: Any) -> Any:
    """Serialise types the JSON encoder does not handle natively."""
    if isinstance(value, datetime | date | pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    return str(value)


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` as formatted JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, default=_json_default, sort_keys=True), encoding="utf-8"
    )
    return destination


def export_backtest(
    result: Any,
    directory: str | Path,
    *,
    config: Any | None = None,
    include_signals: bool = False,
) -> Path:
    """Write every artefact from a backtest into ``directory``.

    Returns the directory. Existing files with the same names are overwritten,
    so re-running an export refreshes rather than accumulates.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame | pd.Series] = {
        "equity_curve": result.equity_curve,
        "returns": result.returns,
        "snapshots": result.snapshots,
        "weights": result.weights,
        "target_weights": result.target_weights,
        "fills": result.fills,
        "orders": result.orders,
        "rejected_orders": result.rejected_orders,
        "risk_events": result.risk_events,
        "strategy_contributions": result.strategy_contributions,
    }
    if result.regimes is not None:
        frames["regimes"] = result.regimes.to_records()
    if include_signals:
        for name, frame in result.signals.items():
            frames[f"signal_{name}"] = frame

    written: list[str] = []
    for name, frame in frames.items():
        if frame is None or len(frame) == 0:
            continue
        path = destination / f"{name}.csv"
        (frame.to_frame() if isinstance(frame, pd.Series) else frame).to_csv(path)
        written.append(path.name)

    if result.metrics is not None:
        write_json(result.metrics.values, destination / "metrics.json")
        written.append("metrics.json")
        write_json(result.metrics.describe_conventions(), destination / "metric_conventions.json")
        written.append("metric_conventions.json")

    manifest = {
        "run_id": result.run_id,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "git_commit": result.git_commit,
        "data_source": result.data_source,
        "is_synthetic_data": result.is_synthetic_data,
        "warnings": result.warnings,
        "summary": result.summary(),
        "cost_breakdown": result.cost_breakdown().to_dict(),
        "cost_drag": result.cost_drag(),
        "files": sorted(written),
    }
    if config is not None:
        manifest["config_hash"] = config.config_hash
        write_json(config.snapshot(), destination / "config_snapshot.json")
        manifest["files"] = sorted([*written, "config_snapshot.json"])
    write_json(manifest, destination / "manifest.json")

    log.info(
        "backtest exported",
        extra={"context": {"directory": str(destination), "files": len(manifest["files"]) + 1}},
    )
    return destination


def export_figures(figures: dict[str, Any], directory: str | Path, *, fmt: str = "html") -> list[Path]:
    """Write Plotly figures to ``directory``.

    HTML always works. Static image formats need the optional ``kaleido``
    package; when it is missing the figure falls back to HTML and a warning is
    logged rather than the export failing.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, figure in figures.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        if fmt == "html":
            path = destination / f"{safe}.html"
            figure.write_html(path, include_plotlyjs="cdn", config={"displaylogo": False})
            written.append(path)
            continue
        path = destination / f"{safe}.{fmt}"
        try:
            figure.write_image(path, width=1200, height=600, scale=2)
            written.append(path)
        except Exception as exc:
            log.warning(
                "static image export failed; writing HTML instead",
                extra={"context": {"figure": name, "error": str(exc)}},
            )
            fallback = destination / f"{safe}.html"
            figure.write_html(fallback, include_plotlyjs="cdn")
            written.append(fallback)

    log.info(
        "figures exported",
        extra={"context": {"directory": str(destination), "count": len(written)}},
    )
    return written


def load_manifest(directory: str | Path) -> dict[str, Any]:
    """Read a previously written ``manifest.json``."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest.json in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))
