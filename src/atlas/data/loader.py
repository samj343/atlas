"""Caching market-data loader and the wide price panel used across Atlas.

:class:`MarketDataLoader` sits between a :class:`~atlas.data.provider.DataProvider`
and the rest of the system. It caches per-symbol history on disk, only downloads
the missing tail of a series, and hands back a :class:`PricePanel` - a small
container of aligned wide frames (index = trading day, columns = symbol).

Design notes
------------
* Cache files are keyed by symbol and provider so two providers never collide.
* Incremental updates re-download an overlap window (default 5 days) so that
  late vendor restatements of recent bars are picked up.
* Assets legitimately have different inception dates. The panel aligns them on a
  shared calendar and leaves pre-inception observations as ``NaN`` rather than
  back-filling, so no strategy can trade an asset before it existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.config import AtlasConfig, DataConfig
from atlas.data.provider import (
    CANONICAL_COLUMNS,
    DataProvider,
    build_provider,
    empty_price_frame,
)
from atlas.exceptions import DataError, InsufficientDataError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["MarketDataLoader", "PricePanel", "long_to_panel", "panel_to_long"]

_PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class PricePanel:
    """Aligned wide price frames for a set of symbols.

    Attributes
    ----------
    open, high, low, close, adj_close, volume:
        ``DataFrame`` objects indexed by trading day with one column per symbol.
    source:
        Mapping of symbol -> data source label (e.g. ``"yfinance"``).
    downloaded_at:
        Mapping of symbol -> UTC timestamp of the most recent download.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj_close: pd.DataFrame
    volume: pd.DataFrame
    source: dict[str, str] = field(default_factory=dict)
    downloaded_at: dict[str, pd.Timestamp] = field(default_factory=dict)

    # -- basic accessors ------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        """Symbols present in the panel."""
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The shared trading calendar."""
        return pd.DatetimeIndex(self.close.index)

    @property
    def is_synthetic(self) -> bool:
        """True when any series in the panel came from the synthetic provider."""
        return any(str(s).lower() == "synthetic" for s in self.source.values())

    def field(self, name: str) -> pd.DataFrame:
        """Return the wide frame for a price field by name."""
        if name not in _PRICE_FIELDS:
            raise DataError(f"unknown price field {name!r}; expected one of {_PRICE_FIELDS}")
        return getattr(self, name)

    def returns(self, field_name: str = "adj_close") -> pd.DataFrame:
        """Simple daily returns computed from ``field_name``.

        The first observation of each series is ``NaN`` by construction; no
        forward filling is applied, so a gap in prices produces a ``NaN`` return
        rather than a fabricated zero.
        """
        prices = self.field(field_name)
        return prices.pct_change(fill_method=None)

    def log_returns(self, field_name: str = "adj_close") -> pd.DataFrame:
        """Continuously compounded daily returns."""
        prices = self.field(field_name)
        return np.log(prices).diff()

    # -- slicing --------------------------------------------------------------

    def slice(
        self,
        start: pd.Timestamp | date | str | None = None,
        end: pd.Timestamp | date | str | None = None,
    ) -> PricePanel:
        """Return a new panel restricted to ``[start, end]`` (inclusive)."""
        idx = self.dates
        mask = pd.Series(True, index=idx)
        if start is not None:
            mask &= idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end)
        keep = idx[mask.to_numpy()]
        return PricePanel(
            **{f: getattr(self, f).loc[keep] for f in _PRICE_FIELDS},
            source=dict(self.source),
            downloaded_at=dict(self.downloaded_at),
        )

    def select(self, symbols: list[str]) -> PricePanel:
        """Return a new panel restricted to ``symbols`` (order preserved)."""
        keep = [s for s in symbols if s in self.close.columns]
        missing = [s for s in symbols if s not in self.close.columns]
        if missing:
            log.warning("symbols absent from panel", extra={"context": {"symbols": missing}})
        return PricePanel(
            **{f: getattr(self, f)[keep] for f in _PRICE_FIELDS},
            source={k: v for k, v in self.source.items() if k in keep},
            downloaded_at={k: v for k, v in self.downloaded_at.items() if k in keep},
        )

    def head_dates(self, n: int) -> PricePanel:
        """Return a panel containing only the first ``n`` trading days."""
        return self.slice(end=self.dates[min(n, len(self.dates)) - 1]) if len(self.dates) else self

    def as_of(self, timestamp: pd.Timestamp | date | str) -> PricePanel:
        """Return the panel truncated at ``timestamp`` (inclusive).

        This is the primary guard against look-ahead bias: every component that
        needs "the data as it was known on day *t*" calls this.
        """
        return self.slice(end=timestamp)

    def tradable_mask(self, min_history: int = 1) -> pd.DataFrame:
        """Boolean frame: True where a symbol has at least ``min_history`` prior
        observations *and* a price on that day.

        Used to prevent a strategy from taking a position in an asset before it
        has enough history, or on a day where it has no price.
        """
        available = self.adj_close.notna()
        counted = available.cumsum()
        return available & (counted >= min_history)

    def last_valid_date(self) -> pd.Series:
        """Most recent date with a valid adjusted close, per symbol."""
        return self.adj_close.apply(lambda col: col.last_valid_index())

    def implied_dividends(self) -> pd.DataFrame:
        """Estimate per-share cash distributions from the adjustment factor.

        The adjustment factor ``f_t = adj_close_t / close_t`` rises on ex-dividend
        dates. Assuming splits are already reflected in both series, the implied
        distribution on day *t* is::

            div_t = close_t * (f_t / f_{t-1} - 1)

        This lets the backtester run on *raw* prices while still crediting cash
        dividends. The estimate is approximate and is only used when
        ``price_field='close'``.
        """
        factor = (self.adj_close / self.close).replace([np.inf, -np.inf], np.nan)
        ratio = factor / factor.shift(1)
        dividends = self.close * (ratio - 1.0)
        return dividends.where(dividends > 0.0, 0.0).fillna(0.0)

    def describe(self) -> pd.DataFrame:
        """Per-symbol summary: first/last date, observation count and source."""
        rows = []
        for sym in self.symbols:
            series = self.adj_close[sym].dropna()
            rows.append(
                {
                    "symbol": sym,
                    "first_date": series.index.min() if len(series) else pd.NaT,
                    "last_date": series.index.max() if len(series) else pd.NaT,
                    "observations": len(series),
                    "source": self.source.get(sym, "unknown"),
                    "downloaded_at": self.downloaded_at.get(sym, pd.NaT),
                }
            )
        return pd.DataFrame(rows).set_index("symbol")


def long_to_panel(frame: pd.DataFrame) -> PricePanel:
    """Convert a canonical long price frame into a :class:`PricePanel`."""
    if frame.empty:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return PricePanel(**{f: empty.copy() for f in _PRICE_FIELDS})

    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise DataError(f"long frame is missing canonical columns: {sorted(missing)}")

    frame = frame.sort_values(["date", "symbol"])
    wide: dict[str, pd.DataFrame] = {}
    for f in _PRICE_FIELDS:
        pivot = frame.pivot_table(index="date", columns="symbol", values=f, aggfunc="last")
        pivot.index = pd.DatetimeIndex(pivot.index, name="date")
        pivot.columns.name = None
        wide[f] = pivot.astype("float64")

    # Union calendar across every field so all frames share one index.
    index = wide["close"].index
    # Symbol columns are held as an object-dtype Index rather than the arrow-backed
    # string dtype pandas now infers. The backtest performs millions of small
    # per-symbol lookups and reindexes, and arrow string indexes make each one
    # materially slower - this single choice roughly halves total run time.
    columns = pd.Index([str(c) for c in wide["close"].columns], dtype=object)
    for f in _PRICE_FIELDS:
        wide[f] = wide[f].set_axis(
            pd.Index([str(c) for c in wide[f].columns], dtype=object), axis=1
        ).reindex(index=index, columns=columns)

    meta = frame.groupby("symbol").agg(source=("source", "last"), downloaded_at=("downloaded_at", "max"))
    return PricePanel(
        **wide,
        source={str(k): str(v) for k, v in meta["source"].items()},
        downloaded_at={str(k): pd.Timestamp(v) for k, v in meta["downloaded_at"].items()},
    )


def panel_to_long(panel: PricePanel) -> pd.DataFrame:
    """Convert a :class:`PricePanel` back into the canonical long frame."""
    frames = []
    for f in _PRICE_FIELDS:
        wide = panel.field(f)
        melted = wide.stack(future_stack=True).rename(f).reset_index()
        melted.columns = ["date", "symbol", f]
        frames.append(melted.set_index(["date", "symbol"]))
    out = pd.concat(frames, axis=1).reset_index()
    out["source"] = out["symbol"].map(panel.source).fillna("unknown")
    out["downloaded_at"] = out["symbol"].map(panel.downloaded_at)
    out = out.dropna(subset=["close"])
    return out[list(CANONICAL_COLUMNS)].sort_values(["symbol", "date"]).reset_index(drop=True)


class MarketDataLoader:
    """Load market data through a provider with a local on-disk cache.

    Parameters
    ----------
    provider:
        The data provider to use.
    cache_dir:
        Directory for cached parquet/CSV files. Created on demand.
    overlap_days:
        When updating a cached series, re-download this many days before the last
        cached observation so that vendor restatements are captured.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache_dir: str | Path = "data/raw",
        *,
        overlap_days: int = 5,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.overlap_days = max(0, int(overlap_days))
        self._cache_format = "parquet" if _parquet_available() else "csv"

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_config(cls, config: AtlasConfig, **provider_kwargs) -> MarketDataLoader:
        """Build a loader from an :class:`~atlas.config.AtlasConfig`."""
        data_cfg: DataConfig = config.data
        kwargs = dict(provider_kwargs)
        if data_cfg.provider == "csv" and "directory" not in kwargs:
            kwargs["directory"] = config.resolve_path(data_cfg.cache_dir)
        provider = build_provider(data_cfg.provider, **kwargs)
        return cls(provider, config.resolve_path(data_cfg.cache_dir))

    # -- public API -----------------------------------------------------------

    def load(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None = None,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> PricePanel:
        """Load prices for ``symbols``, using and updating the local cache.

        Parameters
        ----------
        symbols:
            Ticker symbols to load.
        start_date, end_date:
            Inclusive bounds. ``end_date=None`` means today.
        use_cache:
            Read from and write to the on-disk cache.
        force_refresh:
            Ignore any cached data and re-download the full range.

        Returns
        -------
        PricePanel
            Aligned wide frames for the requested symbols.
        """
        end = end_date or date.today()
        symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        if not symbols:
            return long_to_panel(empty_price_frame())

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            frames.append(
                self._load_symbol(
                    symbol, start_date, end, use_cache=use_cache, force_refresh=force_refresh
                )
            )
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise DataError(
                f"no data could be loaded for any of {symbols}. Check the provider "
                "configuration in configs/assets.yaml and network access."
            )
        combined = pd.concat(frames, ignore_index=True)
        combined = combined[
            (combined["date"] >= pd.Timestamp(start_date)) & (combined["date"] <= pd.Timestamp(end))
        ]
        panel = long_to_panel(combined)
        log.info(
            "market data loaded",
            extra={
                "context": {
                    "symbols": len(panel.symbols),
                    "rows": len(combined),
                    "first_date": str(panel.dates.min().date()) if len(panel.dates) else None,
                    "last_date": str(panel.dates.max().date()) if len(panel.dates) else None,
                    "provider": self.provider.metadata.name,
                }
            },
        )
        return panel

    def load_universe(
        self,
        config: AtlasConfig,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> PricePanel:
        """Load the entire configured universe."""
        return self.load(
            config.universe.symbols,
            config.data.start_date,
            config.data.end_date,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    # -- caching --------------------------------------------------------------

    def cache_path(self, symbol: str) -> Path:
        """Cache file path for ``symbol`` under the configured provider."""
        return self.cache_dir / self.provider.metadata.name / f"{symbol}.{self._cache_format}"

    def _load_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        use_cache: bool,
        force_refresh: bool,
    ) -> pd.DataFrame:
        """Load one symbol, downloading only the portion not already cached."""
        cached = pd.DataFrame()
        if use_cache and not force_refresh:
            cached = self._read_cache(symbol)

        need_start = start_date
        if not cached.empty:
            cached_start = cached["date"].min().date()
            cached_end = cached["date"].max().date()
            covers_start = cached_start <= start_date
            if covers_start and cached_end >= end_date:
                log.debug("cache hit", extra={"context": {"symbol": symbol}})
                return cached
            if covers_start:
                # Only the tail is missing; re-download with an overlap window.
                need_start = cached_end - timedelta(days=self.overlap_days)
            else:
                need_start = start_date

        try:
            fresh = self.provider.get_historical_prices([symbol], need_start, end_date)
        except DataError:
            if not cached.empty:
                log.warning(
                    "download failed; serving cached data",
                    extra={"context": {"symbol": symbol}},
                )
                return cached
            raise

        if fresh.empty and cached.empty:
            log.warning("no data returned", extra={"context": {"symbol": symbol}})
            return empty_price_frame()

        merged = self._merge(cached, fresh)
        if use_cache and not merged.empty:
            self._write_cache(symbol, merged)
        return merged

    @staticmethod
    def _merge(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        """Combine cached and freshly downloaded rows, preferring fresh data."""
        if cached.empty:
            return fresh.reset_index(drop=True)
        if fresh.empty:
            return cached.reset_index(drop=True)
        combined = pd.concat([cached, fresh], ignore_index=True)
        combined = combined.sort_values(["symbol", "date", "downloaded_at"])
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        return combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _read_cache(self, symbol: str) -> pd.DataFrame:
        """Read a cached symbol file, returning an empty frame when absent."""
        path = self.cache_path(symbol)
        if not path.is_file():
            return empty_price_frame()
        try:
            frame = (
                pd.read_parquet(path)
                if self._cache_format == "parquet"
                else pd.read_csv(path, parse_dates=["date", "downloaded_at"])
            )
        except Exception as exc:
            log.warning(
                "unreadable cache file; ignoring",
                extra={"context": {"symbol": symbol, "path": str(path), "error": str(exc)}},
            )
            return empty_price_frame()
        missing = set(CANONICAL_COLUMNS) - set(frame.columns)
        if missing:
            log.warning(
                "cache schema mismatch; ignoring",
                extra={"context": {"symbol": symbol, "missing": sorted(missing)}},
            )
            return empty_price_frame()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["downloaded_at"] = pd.to_datetime(frame["downloaded_at"])
        return frame[list(CANONICAL_COLUMNS)]

    def _write_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        """Persist a symbol's history to the cache."""
        path = self.cache_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if self._cache_format == "parquet":
            frame.to_parquet(tmp, index=False)
        else:
            frame.to_csv(tmp, index=False)
        tmp.replace(path)

    def clear_cache(self, symbols: list[str] | None = None) -> int:
        """Delete cached files. Returns the number of files removed."""
        target_dir = self.cache_dir / self.provider.metadata.name
        if not target_dir.is_dir():
            return 0
        removed = 0
        for path in sorted(target_dir.glob(f"*.{self._cache_format}")):
            if symbols is not None and path.stem.upper() not in {s.upper() for s in symbols}:
                continue
            path.unlink()
            removed += 1
        log.info("cache cleared", extra={"context": {"files_removed": removed}})
        return removed


def _parquet_available() -> bool:
    """True when a parquet engine is importable."""
    try:  # pragma: no cover - environment dependent
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:  # pragma: no cover - environment dependent
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def require_min_history(panel: PricePanel, min_days: int) -> list[str]:
    """Return symbols with at least ``min_days`` valid observations.

    Raises
    ------
    InsufficientDataError
        If no symbol satisfies the requirement.
    """
    counts = panel.adj_close.notna().sum()
    eligible = [str(s) for s, n in counts.items() if int(n) >= min_days]
    if not eligible:
        raise InsufficientDataError(
            f"no symbol has the required {min_days} observations "
            f"(best available: {int(counts.max()) if len(counts) else 0})"
        )
    return eligible
