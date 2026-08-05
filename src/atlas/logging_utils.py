"""Structured logging for Atlas.

Atlas uses the standard library ``logging`` module with a structured formatter so
that log records carry machine-readable context (run id, symbol, order id, ...)
rather than being embedded in free-form strings.

Use :func:`get_logger` everywhere; never ``print``.

Example
-------
>>> from atlas.logging_utils import configure_logging, get_logger
>>> configure_logging(level="INFO")
>>> log = get_logger(__name__)
>>> log.info("order accepted", extra={"context": {"symbol": "SPY", "qty": 10}})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "AtlasLoggerAdapter",
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "log_context",
]

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s%(context_suffix)s"

# Record attributes present on every LogRecord; anything else is user context.
_STANDARD_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName", "context_suffix",
}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON string."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = _jsonable(context)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in {"context"}:
                payload.setdefault(key, _jsonable(value))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ContextTextFormatter(logging.Formatter):
    """Human-readable formatter that appends structured context as ``key=value``."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` with a trailing context suffix."""
        context = getattr(record, "context", None)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and k != "context"
        }
        merged: dict[str, Any] = {}
        if isinstance(context, dict):
            merged.update(context)
        merged.update(extras)
        record.context_suffix = (
            "  [" + " ".join(f"{k}={v}" for k, v in merged.items()) + "]" if merged else ""
        )
        return super().format(record)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of ``value`` into something JSON-serialisable."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def configure_logging(
    level: str | int = "INFO",
    *,
    json_output: bool | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configure root logging for the process.

    Parameters
    ----------
    level:
        Logging level name or numeric level.
    json_output:
        Emit JSON lines instead of human-readable text. Defaults to the
        ``ATLAS_LOG_JSON`` environment variable, else ``False``.
    log_file:
        Optional path for a rotating file handler (5 MB x 3 backups).
    force:
        Reconfigure even if logging was already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if json_output is None:
        json_output = os.environ.get("ATLAS_LOG_JSON", "").lower() in {"1", "true", "yes"}

    level_value = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(level_value, int):
        level_value = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level_value)

    formatter: logging.Formatter = (
        JsonFormatter() if json_output else _ContextTextFormatter(_DEFAULT_FORMAT)
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO.
    for noisy in ("urllib3", "matplotlib", "yfinance", "peewee", "ib_insync"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


class AtlasLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges a persistent context into every record."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Merge adapter context with any per-call ``extra['context']``."""
        extra = dict(kwargs.get("extra") or {})
        context = dict(self.extra or {})
        context.update(extra.pop("context", {}) or {})
        extra["context"] = context
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **context: Any) -> AtlasLoggerAdapter:
        """Return a new adapter with additional persistent context."""
        merged = dict(self.extra or {})
        merged.update(context)
        return AtlasLoggerAdapter(self.logger, merged)


def get_logger(name: str, **context: Any) -> AtlasLoggerAdapter:
    """Return a structured logger for ``name`` with optional bound ``context``."""
    if not _CONFIGURED:
        configure_logging(os.environ.get("ATLAS_LOG_LEVEL", "INFO"))
    return AtlasLoggerAdapter(logging.getLogger(name), dict(context))


@contextmanager
def log_context(logger: AtlasLoggerAdapter, **context: Any) -> Iterator[AtlasLoggerAdapter]:
    """Temporarily bind extra context to ``logger`` inside a ``with`` block."""
    yield logger.bind(**context)
