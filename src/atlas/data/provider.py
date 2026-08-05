"""Provider-agnostic market-data interface.

All historical data enters Atlas through :class:`DataProvider`. Concrete
implementations return a *canonical long frame* with exactly these columns:

``date`` (datetime64[ns]), ``symbol`` (str), ``open``, ``high``, ``low``,
``close``, ``adj_close``, ``volume``, plus the provider metadata columns
``source`` and ``downloaded_at``.

Three implementations ship with Atlas:

* :class:`YFinanceProvider` - free daily data (requires the optional ``yfinance``
  dependency and network access).
* :class:`CSVProvider` - reads a directory of ``<SYMBOL>.csv`` files; useful for
  vendor exports and for fully offline reproduction.
* :class:`SyntheticProvider` - deterministic simulated price paths used by the
  test suite and by anyone who cannot reach a data vendor. Data produced by this
  provider is *always* labelled ``synthetic`` so it can never be mistaken for
  real market history.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.exceptions import DataError
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "CANONICAL_COLUMNS",
    "CSVProvider",
    "DataProvider",
    "ProviderMetadata",
    "SyntheticProvider",
    "YFinanceProvider",
    "build_provider",
    "empty_price_frame",
]

CANONICAL_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "downloaded_at",
)

_NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class ProviderMetadata:
    """Descriptive metadata about a data provider."""

    name: str
    description: str
    supports_adjusted: bool
    supports_volume: bool
    is_synthetic: bool = False


def empty_price_frame() -> pd.DataFrame:
    """Return an empty frame with the canonical schema and correct dtypes."""
    frame = pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "symbol": pd.Series(dtype="object"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "adj_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="object"),
            "downloaded_at": pd.Series(dtype="datetime64[ns]"),
        }
    )
    return frame[list(CANONICAL_COLUMNS)]


class DataProvider(abc.ABC):
    """Abstract base class for historical market-data providers.

    Subclasses implement :meth:`_fetch`; the public :meth:`get_historical_prices`
    normalises whatever the vendor returns into the canonical schema, so callers
    never have to care which provider produced the data.
    """

    #: Provider metadata, overridden by subclasses.
    metadata: ProviderMetadata = ProviderMetadata(
        name="abstract", description="abstract provider", supports_adjusted=False,
        supports_volume=False,
    )

    # -- public API -----------------------------------------------------------

    def get_historical_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Return daily OHLCV history for ``symbols`` between two dates.

        Parameters
        ----------
        symbols:
            Ticker symbols to retrieve. Order is not significant.
        start_date, end_date:
            Inclusive date bounds.

        Returns
        -------
        pandas.DataFrame
            A long frame with the canonical columns, sorted by ``(symbol, date)``
            and free of duplicate ``(symbol, date)`` pairs.

        Raises
        ------
        DataError
            If the provider fails or returns an unusable payload.
        """
        if not symbols:
            return empty_price_frame()
        if end_date < start_date:
            raise DataError(f"end_date {end_date} precedes start_date {start_date}")

        clean_symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        log.debug(
            "fetching historical prices",
            extra={
                "context": {
                    "provider": self.metadata.name,
                    "n_symbols": len(clean_symbols),
                    "start": str(start_date),
                    "end": str(end_date),
                }
            },
        )
        raw = self._fetch(clean_symbols, start_date, end_date)
        frame = self._normalize(raw)
        frame = frame[(frame["date"].dt.date >= start_date) & (frame["date"].dt.date <= end_date)]
        return frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    def get_latest_prices(self, symbols: list[str]) -> pd.Series:
        """Return the most recent adjusted close per symbol.

        The default implementation pulls a short recent window through
        :meth:`get_historical_prices`; providers with a real-time endpoint should
        override this.
        """
        today = datetime.now(UTC).date()
        start = today - pd.Timedelta(days=14).to_pytimedelta()
        frame = self.get_historical_prices(symbols, start, today)
        if frame.empty:
            return pd.Series(dtype="float64")
        latest = frame.sort_values("date").groupby("symbol").tail(1)
        return latest.set_index("symbol")["adj_close"].astype(float)

    # -- subclass hook --------------------------------------------------------

    @abc.abstractmethod
    def _fetch(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> pd.DataFrame:  # pragma: no cover - abstract
        """Fetch raw vendor data. Must return a long frame with at least
        ``date``, ``symbol``, ``open``, ``high``, ``low``, ``close``."""
        raise NotImplementedError

    # -- normalisation --------------------------------------------------------

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce a vendor frame into the canonical schema."""
        if frame is None or len(frame) == 0:
            return empty_price_frame()

        out = frame.copy()
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

        aliases = {
            "adjclose": "adj_close",
            "adjusted_close": "adj_close",
            "adj._close": "adj_close",
            "datetime": "date",
            "timestamp": "date",
            "ticker": "symbol",
            "vol": "volume",
        }
        out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})

        missing = {"date", "symbol", "close"} - set(out.columns)
        if missing:
            raise DataError(f"provider {self.metadata.name} omitted required columns: {missing}")

        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()

        if "adj_close" not in out.columns:
            out["adj_close"] = out["close"]
        for col in ("open", "high", "low"):
            if col not in out.columns:
                out[col] = out["close"]
        if "volume" not in out.columns:
            out["volume"] = np.nan

        for col in _NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["source"] = out.get("source", self.metadata.name)
        if "downloaded_at" in out.columns:
            out["downloaded_at"] = pd.to_datetime(out["downloaded_at"], errors="coerce")
        else:
            out["downloaded_at"] = pd.Timestamp.utcnow().tz_localize(None)

        out = out.dropna(subset=["date", "symbol"])
        # Keep the last observation for a duplicated (symbol, date) - vendors
        # occasionally emit a provisional and then a final bar for the same day.
        out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
        return out[list(CANONICAL_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------


class YFinanceProvider(DataProvider):
    """Daily history from Yahoo Finance via the optional ``yfinance`` package.

    Yahoo data is free and convenient but is *not* survivorship-bias free and its
    adjustments can change retroactively. It is adequate for research on large
    liquid ETFs; swap in a vendor provider for anything that matters.
    """

    metadata = ProviderMetadata(
        name="yfinance",
        description="Yahoo Finance daily OHLCV via the yfinance package",
        supports_adjusted=True,
        supports_volume=True,
    )

    def __init__(self, *, batch_size: int = 20, retries: int = 2) -> None:
        self.batch_size = max(1, int(batch_size))
        self.retries = max(0, int(retries))

    def _fetch(self, symbols: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DataError(
                "yfinance is not installed. Install it with `pip install 'atlas-trading-system[data]'` "
                "or choose a different provider in configs/assets.yaml."
            ) from exc

        frames: list[pd.DataFrame] = []
        # yfinance's end bound is exclusive for daily bars.
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i : i + self.batch_size]
            raw = self._download_with_retry(yfinance, batch, start_date, end_exclusive)
            if raw is None or raw.empty:
                log.warning("provider returned no rows", extra={"context": {"symbols": batch}})
                continue
            frames.append(self._reshape(raw, batch))

        if not frames:
            return empty_price_frame()
        return pd.concat(frames, ignore_index=True)

    def _download_with_retry(
        self, yfinance_module, batch: list[str], start: date, end_exclusive: pd.Timestamp
    ) -> pd.DataFrame | None:
        """Download a batch, retrying read failures a bounded number of times."""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return yfinance_module.download(
                    tickers=batch,
                    start=str(start),
                    end=str(end_exclusive.date()),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
            except Exception as exc:
                last_error = exc
                log.warning(
                    "download attempt failed",
                    extra={"context": {"attempt": attempt + 1, "error": str(exc)}},
                )
        raise DataError(f"yfinance download failed after {self.retries + 1} attempts: {last_error}")

    @staticmethod
    def _reshape(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
        """Flatten yfinance's wide/multi-index output into the long form."""
        records: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            available = {str(c) for c in raw.columns.get_level_values(0)}
            for symbol in batch:
                if symbol not in available:
                    continue
                sub = raw[symbol].copy()
                sub = sub.dropna(how="all")
                if sub.empty:
                    continue
                sub = sub.reset_index()
                sub["symbol"] = symbol
                records.append(sub)
        else:
            sub = raw.dropna(how="all").reset_index()
            sub["symbol"] = batch[0]
            records.append(sub)

        if not records:
            return empty_price_frame()
        out = pd.concat(records, ignore_index=True)
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
        return out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class CSVProvider(DataProvider):
    """Read daily history from ``<directory>/<SYMBOL>.csv`` files.

    Each file must contain a date column plus ``open``/``high``/``low``/``close``
    (case-insensitive); ``adj_close`` and ``volume`` are optional.
    """

    metadata = ProviderMetadata(
        name="csv",
        description="Local CSV files, one per symbol",
        supports_adjusted=True,
        supports_volume=True,
    )

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _fetch(
        self,
        symbols: list[str],
        start_date: date,  # noqa: ARG002 - the base class applies the date filter
        end_date: date,  # noqa: ARG002
    ) -> pd.DataFrame:
        if not self.directory.is_dir():
            raise DataError(f"CSV directory does not exist: {self.directory}")
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            path = self.directory / f"{symbol}.csv"
            if not path.is_file():
                log.warning("no CSV file for symbol", extra={"context": {"symbol": symbol}})
                continue
            sub = pd.read_csv(path)
            sub.columns = [str(c).strip().lower().replace(" ", "_") for c in sub.columns]
            sub["symbol"] = symbol
            frames.append(sub)
        if not frames:
            return empty_price_frame()
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


class SyntheticProvider(DataProvider):
    """Deterministic simulated price history for tests and offline runs.

    The generator produces a correlated multi-asset geometric random walk with a
    slowly varying volatility state, so that regime detection, drawdown controls
    and cross-sectional ranking all have something meaningful to act on. Output
    is fully reproducible from ``seed``.

    Warning
    -------
    These are **simulated prices, not market data**. Every row is tagged with
    ``source='synthetic'`` and downstream reports label results accordingly.
    Never present metrics computed from this provider as historical performance.
    """

    metadata = ProviderMetadata(
        name="synthetic",
        description="Deterministic simulated prices - NOT real market data",
        supports_adjusted=True,
        supports_volume=True,
        is_synthetic=True,
    )

    def __init__(
        self,
        *,
        seed: int = 7,
        annual_drift: float = 0.06,
        annual_volatility: float = 0.16,
        market_beta_range: tuple[float, float] = (0.2, 1.2),
        start_price: float = 100.0,
    ) -> None:
        self.seed = int(seed)
        self.annual_drift = float(annual_drift)
        self.annual_volatility = float(annual_volatility)
        self.market_beta_range = market_beta_range
        self.start_price = float(start_price)

    @staticmethod
    def _symbol_seed(base_seed: int, symbol: str) -> int:
        """Derive a stable per-symbol seed.

        Each symbol's idiosyncratic path is drawn from its *own* generator, keyed
        by the symbol name. That makes the output independent of how symbols are
        batched - fetching ``["SPY", "TLT"]`` in one call produces exactly the
        same series as two separate calls, which matters because the caching
        loader fetches one symbol at a time.
        """
        digest = int.from_bytes(
            hashlib.sha256(f"{base_seed}:{symbol}".encode()).digest()[:8], "big"
        )
        return digest % (2**32)

    #: Every simulated path is generated from this fixed anchor date.
    ANCHOR_DATE = date(1995, 1, 2)

    def _fetch(self, symbols: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        # The path is always generated from a fixed anchor and then sliced to the
        # requested window. Generating from `start_date` instead would make each
        # series restart at `start_price` on whatever day happened to be asked
        # for, so an incremental cache refresh would splice two unrelated paths
        # together at the seam. Anchoring makes any window a slice of one global
        # path, which is what "reproducible" has to mean here.
        anchor = min(self.ANCHOR_DATE, start_date)
        dates = pd.bdate_range(start=anchor, end=end_date, freq="C")
        if len(dates) == 0:
            return empty_price_frame()
        keep = dates >= pd.Timestamp(start_date)

        n_days = len(dates)
        dt = 1.0 / 252.0

        # The market factor depends only on the base seed and the calendar, so
        # every symbol in every call shares the same common component.
        market_rng = np.random.default_rng(self.seed)
        vol_state = np.ones(n_days)
        state = 0
        for i in range(1, n_days):
            switch = market_rng.random() < (0.004 if state == 0 else 0.020)
            if switch:
                state = 1 - state
            vol_state[i] = 1.0 if state == 0 else 2.4
        market_shock = market_rng.standard_normal(n_days) * vol_state
        market_returns = self.annual_drift * dt + self.annual_volatility * np.sqrt(dt) * market_shock

        low_beta, high_beta = self.market_beta_range
        frames: list[pd.DataFrame] = []
        downloaded_at = pd.Timestamp("2000-01-01")  # fixed for reproducible caching

        for symbol in symbols:
            rng = np.random.default_rng(self._symbol_seed(self.seed, symbol))
            beta = low_beta + (high_beta - low_beta) * rng.random()
            idio_vol = self.annual_volatility * (0.4 + 0.6 * rng.random())
            idio = idio_vol * np.sqrt(dt) * rng.standard_normal(n_days) * vol_state
            # A slow mean-reverting component gives short-term reversion signals
            # something real to detect.
            ou = np.zeros(n_days)
            ou_shocks = rng.standard_normal(n_days)
            for i in range(1, n_days):
                ou[i] = 0.90 * ou[i - 1] + 0.002 * ou_shocks[i]
            returns = beta * market_returns + idio - 0.5 * ou
            returns[0] = 0.0

            close = self.start_price * np.exp(np.cumsum(returns))
            noise = rng.random(n_days)
            high = close * (1.0 + 0.004 * noise)
            low = close * (1.0 - 0.004 * (1.0 - noise))
            open_ = np.concatenate([[close[0]], close[:-1] * (1.0 + 0.001 * rng.standard_normal(n_days - 1))])
            open_ = np.clip(open_, low, high)
            volume = np.round(1e6 * (1.0 + 0.5 * rng.random(n_days)) * (1.0 + beta))

            frame = pd.DataFrame(
                    {
                        "date": dates,
                        "symbol": symbol,
                        "open": open_,
                        "high": np.maximum.reduce([high, close, open_]),
                        "low": np.minimum.reduce([low, close, open_]),
                        "close": close,
                        # Simulated dividends: a small, steady accrual so that
                        # adj_close and close legitimately differ.
                        "adj_close": close * np.exp(np.linspace(0.0, 0.015 * n_days / 252.0, n_days)),
                        "volume": volume,
                        "source": "synthetic",
                        "downloaded_at": downloaded_at,
                    }
                )
            # Slice to the requested window only after the full path is built.
            frames.append(frame.loc[keep].reset_index(drop=True))
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_provider(name: str, **kwargs) -> DataProvider:
    """Construct a provider by name.

    Parameters
    ----------
    name:
        One of ``"yfinance"``, ``"csv"`` or ``"synthetic"``.
    **kwargs:
        Forwarded to the provider constructor.
    """
    key = name.strip().lower()
    if key == "yfinance":
        return YFinanceProvider(**kwargs)
    if key == "csv":
        return CSVProvider(**kwargs)
    if key == "synthetic":
        return SyntheticProvider(**kwargs)
    raise DataError(f"unknown data provider {name!r}; expected yfinance, csv or synthetic")
