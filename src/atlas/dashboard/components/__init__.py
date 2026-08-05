"""Reusable Streamlit display components."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

__all__ = [
    "chart",
    "error_banner",
    "metric_row",
    "render_dataframe",
    "section",
    "status_pill",
    "warning_banner",
]


def section(title: str, description: str = "") -> None:
    """Render a section heading with an optional description."""
    st.subheader(title)
    if description:
        st.caption(description)


def metric_row(metrics: dict[str, Any], columns: int = 4) -> None:
    """Render a row of metric tiles.

    Each value may be a plain value or a ``(value, delta)`` tuple.
    """
    items = list(metrics.items())
    for start in range(0, len(items), columns):
        row = st.columns(columns)
        for column, (label, value) in zip(row, items[start : start + columns], strict=False):
            if isinstance(value, tuple) and len(value) == 2:
                column.metric(label, value[0], value[1])
            else:
                column.metric(label, value)


def render_dataframe(
    frame: pd.DataFrame,
    *,
    height: int | None = None,
    hide_index: bool = True,
    empty_message: str = "No data available.",
) -> None:
    """Render a DataFrame, or an explanatory message when it is empty."""
    if frame is None or frame.empty:
        st.info(empty_message)
        return
    # Streamlit rejects height=None, so the argument is only passed when set.
    kwargs = {"height": height} if height is not None else {}
    st.dataframe(frame, width="stretch", hide_index=hide_index, **kwargs)


def chart(figure: Any, *, key: str | None = None) -> None:
    """Render a Plotly figure at full container width."""
    st.plotly_chart(figure, width="stretch", key=key)


def warning_banner(message: str) -> None:
    """Render a prominent warning."""
    st.warning(message, icon="⚠️")


def error_banner(message: str) -> None:
    """Render a prominent error."""
    st.error(message, icon="🚫")


def status_pill(label: str, ok: bool, ok_text: str = "OK", bad_text: str = "ATTENTION") -> None:
    """Render a pass/fail status line."""
    if ok:
        st.success(f"**{label}** — {ok_text}")
    else:
        st.error(f"**{label}** — {bad_text}")
