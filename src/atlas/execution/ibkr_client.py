"""Interactive Brokers connectivity.

Atlas talks to TWS or IB Gateway through the optional ``ib_insync`` package.
The broker interface is defined abstractly as :class:`BrokerClient` and has two
implementations:

* :class:`IBKRClient` - the real connection.
* :class:`MockBrokerClient` - an in-memory broker with the same interface, used
  by the test suite and by ``atlas order-preview`` when no TWS is running. It
  simulates fills, rejections and partial fills so the execution path can be
  exercised end to end without a broker.

Two rules shape the design:

**Read operations are retried; order submissions are not.** A failed read is
safe to repeat. A failed *submission* may or may not have reached the exchange,
and repeating it risks doubling a position - so a submission failure surfaces
for a human to resolve.

**Every order carries a unique client identifier.** The identifier is recorded
before submission, so a submission whose response is lost can still be matched
against the broker's open orders rather than re-sent.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.orders import Fill, Order, OrderSide, OrderStatus, OrderType
from atlas.config import BrokerConfig, ExecutionConfig
from atlas.exceptions import BrokerError, OrderRejected
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "AccountSummary",
    "BrokerClient",
    "BrokerPosition",
    "IBKRClient",
    "MockBrokerClient",
]


@dataclass(frozen=True)
class AccountSummary:
    """Key account values reported by the broker."""

    account: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    available_funds: float = 0.0
    maintenance_margin: float = 0.0
    currency: str = "USD"
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "account": self.account,
            "net_liquidation": self.net_liquidation,
            "total_cash": self.total_cash,
            "buying_power": self.buying_power,
            "available_funds": self.available_funds,
            "maintenance_margin": self.maintenance_margin,
            "currency": self.currency,
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class BrokerPosition:
    """A position as reported by the broker."""

    symbol: str
    quantity: float
    average_cost: float
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat, serialisable representation."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_cost": self.average_cost,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
        }


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class BrokerClient(abc.ABC):
    """Abstract broker interface used by the execution layer."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """True when a session is established."""

    @property
    @abc.abstractmethod
    def account(self) -> str | None:
        """The connected account identifier, when known."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the connection."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    @abc.abstractmethod
    def get_account_summary(self) -> AccountSummary:
        """Retrieve account values."""

    @abc.abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Retrieve current positions."""

    @abc.abstractmethod
    def get_market_prices(self, symbols: list[str]) -> pd.Series:
        """Retrieve current prices for ``symbols``."""

    @abc.abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns the order with its broker identifier set."""

    @abc.abstractmethod
    def cancel_order(self, order: Order) -> bool:
        """Cancel an open order. Returns True when the cancel was accepted."""

    @abc.abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Retrieve open orders."""

    @abc.abstractmethod
    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        """Retrieve fills, optionally only those after ``since``."""

    # -- shared helpers -------------------------------------------------------

    def __enter__(self) -> BrokerClient:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()

    def positions_frame(self) -> pd.DataFrame:
        """Broker positions as a DataFrame indexed by symbol."""
        positions = self.get_positions()
        if not positions:
            return pd.DataFrame(
                columns=["symbol", "quantity", "average_cost", "market_price", "market_value"]
            ).set_index("symbol")
        return pd.DataFrame([p.as_dict() for p in positions]).set_index("symbol")


# ---------------------------------------------------------------------------
# Interactive Brokers
# ---------------------------------------------------------------------------


class IBKRClient(BrokerClient):
    """Interactive Brokers client backed by ``ib_insync``.

    Parameters
    ----------
    config:
        The execution configuration block.
    risk_manager:
        Optional risk manager; consecutive API failures are reported to it so it
        can halt trading.
    """

    def __init__(self, config: ExecutionConfig, *, risk_manager: Any | None = None) -> None:
        self.config = config
        self.broker_config: BrokerConfig = config.broker
        self.risk_manager = risk_manager
        self._ib: Any | None = None
        self._account: str | None = None
        self._contracts: dict[str, Any] = {}

    # -- connection -----------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True when the ``ib_insync`` session is live."""
        return bool(self._ib is not None and self._ib.isConnected())

    @property
    def account(self) -> str | None:
        """The connected account identifier."""
        return self._account

    def connect(self) -> None:
        """Connect to TWS or IB Gateway.

        Raises
        ------
        BrokerError
            If ``ib_insync`` is not installed or the connection fails. The error
            message names the usual causes, because "connection refused" alone is
            not actionable.
        """
        try:
            from ib_insync import IB
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BrokerError(
                "ib_insync is not installed. Install it with "
                "`pip install 'atlas-trading-system[broker]'`, or run with EXECUTION_MODE=dry_run "
                "to preview orders without a broker."
            ) from exc

        cfg = self.broker_config
        self._ib = IB()
        try:
            self._ib.connect(
                cfg.host,
                cfg.port,
                clientId=cfg.client_id,
                timeout=cfg.connect_timeout_seconds,
                readonly=False,
            )
        except Exception as exc:
            raise BrokerError(
                f"could not connect to Interactive Brokers at {cfg.host}:{cfg.port} "
                f"(client id {cfg.client_id}): {exc}. Check that TWS or IB Gateway is running, "
                "that the API is enabled under Global Configuration > API > Settings, that "
                "'Read-Only API' is unchecked, that the port matches (7497 TWS paper, "
                "4002 Gateway paper), and that this client id is not already in use."
            ) from exc

        # Delayed data by default: a paper account often lacks live subscriptions,
        # and silently receiving nothing is worse than receiving delayed prices.
        self._ib.reqMarketDataType(cfg.market_data_type)
        accounts = self._ib.managedAccounts()
        self._account = accounts[0] if accounts else None
        log.info(
            "connected to Interactive Brokers",
            extra={
                "context": {
                    "host": cfg.host,
                    "port": cfg.port,
                    "client_id": cfg.client_id,
                    "account": self._account,
                    "market_data_type": cfg.market_data_type,
                }
            },
        )

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
            log.info("disconnected from Interactive Brokers")
        self._ib = None

    # -- retry helper ---------------------------------------------------------

    def _retry_read(self, operation: str, func, *args, **kwargs):
        """Run a *read* operation with bounded retries and backoff.

        Only reads are retried. See the module docstring for why writes are not.
        """
        cfg = self.broker_config
        last_error: Exception | None = None
        for attempt in range(cfg.read_retry_attempts + 1):
            try:
                value = func(*args, **kwargs)
                if self.risk_manager is not None:
                    self.risk_manager.record_api_success()
                return value
            except Exception as exc:
                last_error = exc
                log.warning(
                    "broker read failed",
                    extra={
                        "context": {
                            "operation": operation,
                            "attempt": attempt + 1,
                            "error": str(exc),
                        }
                    },
                )
                if attempt < cfg.read_retry_attempts:
                    time.sleep(cfg.read_retry_backoff_seconds * (2**attempt))
        if self.risk_manager is not None:
            self.risk_manager.record_api_failure(operation)
        raise BrokerError(
            f"broker read {operation!r} failed after {cfg.read_retry_attempts + 1} attempt(s): "
            f"{last_error}"
        )

    def _require_connection(self) -> Any:
        """Return the live IB handle or raise."""
        if not self.connected:
            raise BrokerError("not connected to Interactive Brokers; call connect() first")
        return self._ib

    def _contract(self, symbol: str) -> Any:
        """Return (and cache) a qualified contract for ``symbol``.

        US-listed ETFs and equities are requested as SMART-routed USD stocks.
        """
        key = symbol.upper()
        if key in self._contracts:
            return self._contracts[key]
        from ib_insync import Stock

        ib = self._require_connection()
        contract = Stock(key, "SMART", "USD")
        qualified = self._retry_read(f"qualify:{key}", ib.qualifyContracts, contract)
        if not qualified:
            raise BrokerError(
                f"Interactive Brokers could not qualify a contract for {key}. Check the symbol and "
                "that the account has permissions for US stocks/ETFs."
            )
        self._contracts[key] = qualified[0]
        return qualified[0]

    # -- reads ----------------------------------------------------------------

    def get_account_summary(self) -> AccountSummary:
        """Retrieve the account summary."""
        ib = self._require_connection()
        rows = self._retry_read("accountSummary", ib.accountSummary)
        values = {row.tag: row.value for row in rows}
        account = self._account or (rows[0].account if rows else "")

        def _value(tag: str) -> float:
            try:
                return float(values.get(tag, "0") or 0.0)
            except ValueError:
                return 0.0

        return AccountSummary(
            account=account,
            net_liquidation=_value("NetLiquidation"),
            total_cash=_value("TotalCashValue"),
            buying_power=_value("BuyingPower"),
            available_funds=_value("AvailableFunds"),
            maintenance_margin=_value("MaintMarginReq"),
            currency=values.get("Currency", "USD"),
        )

    def get_positions(self) -> list[BrokerPosition]:
        """Retrieve current positions."""
        ib = self._require_connection()
        items = self._retry_read("positions", ib.positions)
        out: list[BrokerPosition] = []
        for item in items:
            symbol = getattr(item.contract, "symbol", "")
            out.append(
                BrokerPosition(
                    symbol=str(symbol).upper(),
                    quantity=float(item.position),
                    average_cost=float(item.avgCost),
                )
            )
        return out

    def get_market_prices(self, symbols: list[str]) -> pd.Series:
        """Retrieve current prices.

        Prefers the last traded price, then the mid, then the previous close -
        and reports ``NaN`` rather than guessing when none is available. A
        missing price blocks the order for that symbol downstream.
        """
        ib = self._require_connection()
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                contract = self._contract(symbol)
                ticker = self._retry_read(f"ticker:{symbol}", ib.reqMktData, contract, "", False, False)
                ib.sleep(1.0)
                price = self._extract_price(ticker)
                prices[symbol.upper()] = price
            except BrokerError as exc:
                log.warning(
                    "no market price",
                    extra={"context": {"symbol": symbol, "error": str(exc)}},
                )
                prices[symbol.upper()] = float("nan")
        return pd.Series(prices, dtype="float64")

    @staticmethod
    def _extract_price(ticker: Any) -> float:
        """Best available price from a ticker, or NaN."""
        for attribute in ("last", "close", "marketPrice"):
            value = getattr(ticker, attribute, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if value is not None and np.isfinite(value) and value > 0:
                return float(value)
        bid = getattr(ticker, "bid", None)
        ask = getattr(ticker, "ask", None)
        if bid and ask and np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        return float("nan")

    def get_open_orders(self) -> list[Order]:
        """Retrieve open orders, mapped onto Atlas order objects."""
        ib = self._require_connection()
        trades = self._retry_read("openTrades", ib.openTrades)
        out: list[Order] = []
        for trade in trades:
            order = Order(
                symbol=str(getattr(trade.contract, "symbol", "")).upper(),
                quantity=float(trade.order.totalQuantity),
                side=OrderSide.BUY if trade.order.action.upper() == "BUY" else OrderSide.SELL,
                order_type=OrderType.LIMIT if trade.order.orderType == "LMT" else OrderType.MARKET,
                order_id=str(trade.order.orderRef or trade.order.permId or trade.order.orderId),
            )
            order.broker_order_id = str(trade.order.orderId)
            order.status = OrderStatus.SUBMITTED
            order.filled_quantity = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
            average = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
            order.average_fill_price = average
            out.append(order)
        return out

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        """Retrieve fills reported by the broker."""
        ib = self._require_connection()
        executions = self._retry_read("fills", ib.fills)
        out: list[Fill] = []
        for item in executions:
            execution = item.execution
            stamp = pd.Timestamp(execution.time)
            if since is not None and stamp.to_pydatetime() < since:
                continue
            side = OrderSide.BUY if execution.side.upper().startswith("B") else OrderSide.SELL
            commission = 0.0
            report = getattr(item, "commissionReport", None)
            if report is not None and getattr(report, "commission", None) is not None:
                commission = float(report.commission)
            out.append(
                Fill(
                    order_id=str(getattr(execution, "orderRef", "") or execution.orderId),
                    symbol=str(getattr(item.contract, "symbol", "")).upper(),
                    quantity=abs(float(execution.shares)),
                    price=float(execution.price),
                    side=side,
                    timestamp=stamp,
                    commission=commission,
                    fill_id=str(execution.execId),
                )
            )
        return out

    # -- writes ---------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker.

        Never retried: see the module docstring. A failure marks the order
        rejected and raises, so the caller must decide what to do rather than
        the client quietly trying again.
        """
        from ib_insync import LimitOrder, MarketOrder

        ib = self._require_connection()
        contract = self._contract(order.symbol)
        action = "BUY" if order.side is OrderSide.BUY else "SELL"

        if order.order_type is OrderType.LIMIT:
            if order.limit_price is None:
                raise OrderRejected(
                    f"limit order for {order.symbol} has no limit price",
                    order_id=order.order_id,
                    reason="missing_limit_price",
                )
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        else:
            ib_order = MarketOrder(action, order.quantity)

        # The Atlas order id travels with the order so a lost response can still
        # be reconciled against the broker's records.
        ib_order.orderRef = order.order_id
        ib_order.tif = order.time_in_force.value
        ib_order.transmit = True

        try:
            trade = ib.placeOrder(contract, ib_order)
        except Exception as exc:
            order.reject(f"submission failed: {exc}")
            if self.risk_manager is not None:
                self.risk_manager.record_api_failure("submit_order")
            raise OrderRejected(
                f"order submission for {order.symbol} failed and was NOT retried: {exc}. "
                "Check the broker's open orders before resubmitting - the order may have reached "
                "the exchange.",
                order_id=order.order_id,
                reason="submission_error",
            ) from exc

        order.broker_order_id = str(trade.order.orderId)
        order.status = OrderStatus.SUBMITTED
        log.info(
            "order submitted",
            extra={
                "context": {
                    "order_id": order.order_id,
                    "broker_order_id": order.broker_order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                }
            },
        )
        return order

    def cancel_order(self, order: Order) -> bool:
        """Cancel an open order."""
        ib = self._require_connection()
        for trade in ib.openTrades():
            if str(trade.order.orderId) == str(order.broker_order_id):
                ib.cancelOrder(trade.order)
                order.cancel("cancelled via Atlas")
                log.info("order cancelled", extra={"context": {"order_id": order.order_id}})
                return True
        log.warning(
            "cancel requested for an order the broker does not report as open",
            extra={"context": {"order_id": order.order_id}},
        )
        return False


# ---------------------------------------------------------------------------
# Mock broker
# ---------------------------------------------------------------------------


class MockBrokerClient(BrokerClient):
    """In-memory broker with the :class:`BrokerClient` interface.

    Used by tests and by ``atlas order-preview`` when no TWS is available. It
    deliberately reproduces the awkward parts of a real broker - partial fills,
    rejections, and positions that drift from what the caller expects - so the
    execution and reconciliation logic is genuinely exercised.

    Parameters
    ----------
    account:
        Account identifier. Defaults to a ``DU`` paper-style identifier.
    prices:
        Market prices to report.
    fill_ratio:
        Fraction of each order that fills; ``1.0`` fills completely.
    reject_symbols:
        Symbols whose orders are always rejected.
    position_drift:
        Extra shares added to the broker's view of a position after a fill, used
        to exercise reconciliation mismatch handling.
    """

    def __init__(
        self,
        *,
        account: str = "DU1234567",
        initial_cash: float = 100_000.0,
        prices: pd.Series | None = None,
        fill_ratio: float = 1.0,
        reject_symbols: set[str] | None = None,
        position_drift: dict[str, float] | None = None,
        fail_reads: int = 0,
    ) -> None:
        self._account = account
        self._connected = False
        self.cash = float(initial_cash)
        self.prices = prices if prices is not None else pd.Series(dtype="float64")
        self.fill_ratio = float(fill_ratio)
        self.reject_symbols = {s.upper() for s in (reject_symbols or set())}
        self.position_drift = dict(position_drift or {})
        self.fail_reads = int(fail_reads)
        self._positions: dict[str, BrokerPosition] = {}
        self._open_orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self.submitted_orders: list[Order] = []

    # -- connection -----------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True when :meth:`connect` has been called."""
        return self._connected

    @property
    def account(self) -> str | None:
        """The mock account identifier."""
        return self._account if self._connected else None

    def connect(self) -> None:
        """Mark the mock session connected."""
        self._connected = True
        log.info("mock broker connected", extra={"context": {"account": self._account}})

    def disconnect(self) -> None:
        """Mark the mock session disconnected."""
        self._connected = False

    def _check(self) -> None:
        """Raise when a read is attempted while disconnected or while failing."""
        if not self._connected:
            raise BrokerError("mock broker is not connected")
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise BrokerError("simulated broker API failure")

    # -- reads ----------------------------------------------------------------

    def get_account_summary(self) -> AccountSummary:
        """Return simulated account values."""
        self._check()
        positions_value = sum(
            p.quantity * float(self.prices.get(p.symbol, p.average_cost))
            for p in self._positions.values()
        )
        equity = self.cash + positions_value
        return AccountSummary(
            account=self._account,
            net_liquidation=equity,
            total_cash=self.cash,
            buying_power=max(0.0, self.cash),
            available_funds=max(0.0, self.cash),
        )

    def get_positions(self) -> list[BrokerPosition]:
        """Return simulated positions, marked at the current prices."""
        self._check()
        out = []
        for position in self._positions.values():
            if abs(position.quantity) < 1e-9:
                continue
            price = float(self.prices.get(position.symbol, position.average_cost))
            out.append(
                BrokerPosition(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    average_cost=position.average_cost,
                    market_price=price,
                    market_value=position.quantity * price,
                    unrealized_pnl=(price - position.average_cost) * position.quantity,
                )
            )
        return out

    def get_market_prices(self, symbols: list[str]) -> pd.Series:
        """Return the configured prices for ``symbols``."""
        self._check()
        return pd.Series(
            {s.upper(): float(self.prices.get(s.upper(), float("nan"))) for s in symbols},
            dtype="float64",
        )

    def get_open_orders(self) -> list[Order]:
        """Return orders that are still open."""
        self._check()
        return [o for o in self._open_orders.values() if o.is_open]

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        """Return simulated fills."""
        self._check()
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f.timestamp.to_pydatetime() >= since]

    # -- writes ---------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Simulate submitting an order, producing a (possibly partial) fill."""
        if not self._connected:
            raise BrokerError("mock broker is not connected")
        if order.symbol in self.reject_symbols:
            order.reject(f"mock broker rejects orders for {order.symbol}")
            self.submitted_orders.append(order)
            raise OrderRejected(
                f"order for {order.symbol} rejected by the broker",
                order_id=order.order_id,
                reason="symbol_rejected",
            )

        price = float(self.prices.get(order.symbol, order.reference_price or float("nan")))
        if not np.isfinite(price) or price <= 0:
            order.reject(f"no market price for {order.symbol}")
            self.submitted_orders.append(order)
            raise OrderRejected(
                f"no market price for {order.symbol}", order_id=order.order_id, reason="no_price"
            )

        order.broker_order_id = f"MOCK-{len(self.submitted_orders) + 1:06d}"
        order.status = OrderStatus.SUBMITTED
        self._open_orders[order.order_id] = order
        self.submitted_orders.append(order)

        quantity = order.quantity * max(0.0, min(1.0, self.fill_ratio))
        if quantity <= 0:
            return order

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=quantity,
            price=price,
            side=order.side,
            timestamp=pd.Timestamp.now(),
            commission=max(1.0, 0.005 * quantity),
            reference_price=price,
        )
        order.apply_fill(fill)
        self._fills.append(fill)
        self._apply_position(fill)
        return order

    def cancel_order(self, order: Order) -> bool:
        """Cancel a simulated open order."""
        self._check()
        existing = self._open_orders.get(order.order_id)
        if existing is None or not existing.is_open:
            return False
        existing.cancel("cancelled in the mock broker")
        return True

    def _apply_position(self, fill: Fill) -> None:
        """Update the simulated position and cash from a fill."""
        symbol = fill.symbol
        signed = fill.signed_quantity
        drift = self.position_drift.get(symbol, 0.0)
        existing = self._positions.get(symbol)

        if existing is None:
            quantity = signed + drift
            self._positions[symbol] = BrokerPosition(
                symbol=symbol, quantity=quantity, average_cost=fill.price
            )
        else:
            quantity = existing.quantity + signed + drift
            if abs(quantity) < 1e-9:
                self._positions.pop(symbol, None)
            else:
                if np.sign(quantity) == np.sign(existing.quantity) and abs(quantity) > abs(
                    existing.quantity
                ):
                    total = existing.quantity + signed
                    cost = (
                        (existing.average_cost * existing.quantity + fill.price * signed) / total
                        if abs(total) > 1e-9
                        else fill.price
                    )
                else:
                    cost = existing.average_cost
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol, quantity=quantity, average_cost=cost
                )
        self.cash += fill.cash_flow

    # -- test helpers ---------------------------------------------------------

    def set_position(self, symbol: str, quantity: float, average_cost: float = 100.0) -> None:
        """Seed a position directly, bypassing order flow."""
        self._positions[symbol.upper()] = BrokerPosition(
            symbol=symbol.upper(), quantity=float(quantity), average_cost=float(average_cost)
        )

    def set_prices(self, prices: pd.Series | dict[str, float]) -> None:
        """Replace the price series."""
        self.prices = pd.Series(prices, dtype="float64")
