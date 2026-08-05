# Architecture

## Layering

```
                        configs/*.yaml
                              |
                    pydantic validation
                              v
                    atlas.config.AtlasConfig
                              |
   +----------+---------------+---------------+-------------+
   v          v               v               v             v
 data     strategies      regimes        portfolio      execution
   |          |               |               |             |
   |          +-------+-------+               |             |
   |                  v                       |             |
   |             ensemble ------------------->|             |
   |                                          v             v
   +----------------> backtest.engine <--- risk_manager --> broker
                              |
              +---------------+---------------+
              v               v               v
          metrics       validation        database
                              |               |
                              v               v
                          reporting  <---- dashboard
```

Dependencies point downward and inward. `strategies` does not import
`portfolio`; `portfolio` does not import `execution`; nothing imports
`dashboard`.

## Modules

| Module | Responsibility | Key types |
|---|---|---|
| `atlas.config` | Typed, validated configuration | `AtlasConfig` |
| `atlas.data` | Providers, caching loader, validation, features | `DataProvider`, `PricePanel`, `FeatureSet` |
| `atlas.strategies` | Signal generation and combination | `Strategy`, `SignalResult`, `SignalEnsemble` |
| `atlas.regimes` | Market-state classification | `RegimeDetector`, `RegimeResult` |
| `atlas.portfolio` | Sizing, constraints, risk | `PortfolioAllocator`, `RiskManager` |
| `atlas.backtest` | Event loop, broker simulation, costs, metrics | `BacktestEngine`, `Portfolio` |
| `atlas.validation` | Walk-forward, stability, stress, benchmarks | `WalkForwardValidator` |
| `atlas.execution` | Broker connectivity and safety | `BrokerClient`, `OrderManager`, `SafetyGate` |
| `atlas.database` | Persistence | `AtlasRepository` |
| `atlas.reporting` | Charts and research reports | `ResearchReport` |
| `atlas.dashboard` | Streamlit UI | — |

## Key interfaces

```python
class DataProvider(ABC):
    def get_historical_prices(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> pd.DataFrame: ...

class Strategy(ABC):
    def generate_signals(
        self, prices: pd.DataFrame, features, as_of_date=None
    ) -> pd.DataFrame: ...          # tidy long records

    def compute(self, prices, features, as_of_date=None) -> SignalResult: ...
                                     # wide frames, the fast path

class RegimeDetector(ABC):
    def detect(self, panel, features=None, as_of_date=None) -> RegimeResult: ...

class BrokerClient(ABC):
    def get_account_summary(self) -> AccountSummary: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def submit_order(self, order: Order) -> Order: ...
```

`BrokerClient` has two implementations — `IBKRClient` and `MockBrokerClient` —
which is what lets the entire execution path be tested without a broker.

## Design decisions

**No global mutable state.** Configuration is loaded once and passed explicitly.
Two backtests with different settings can run in the same process without
interfering.

**Research and execution are separate.** `atlas.backtest` never imports
`atlas.execution`; they share only the order and fill *types*. A change to
broker handling cannot alter a backtest.

**Risk is independent of strategy.** The risk manager knows nothing about
signals. A strategy bug cannot disable a risk control.

**Point-in-time by construction.** Features are backward-looking, execution is
lagged a bar, and `tests/unit/test_lookahead.py` perturbs the future to prove it.

**Everything is auditable.** Constraint adjustments, risk events, orders, fills
and reconciliation outcomes all carry a reason and are persisted.

**Fail loudly.** Unknown YAML keys are rejected. Missing data raises rather than
being filled in. A specific exception type exists for each failure mode.

## Data flow through one rebalance

```
prices (t-N..t)                       -> features (t-N..t)
features + prices                     -> per-strategy signals (t)
signals + regime(t) + config weights  -> combined signal (t)
combined signal / volatility(t)       -> raw weights
normalise + volatility target         -> scaled weights
regime & drawdown risk multipliers    -> risk-adjusted weights
constraint projection                 -> feasible weights
risk-manager assessment               -> approved weights
weights x equity / price(t)           -> target shares
target - current                      -> orders, queued for t+1
```

Every arrow is a pure function of quantities dated `<= t`.

## Extending Atlas

* **A new data provider** — subclass `DataProvider`, implement `_fetch`,
  register it in `build_provider`.
* **A new strategy** — subclass `Strategy`, implement `_compute` and
  `required_lookback`, decorate with `@register_strategy`, add a config model
  and a YAML block.
* **A new regime detector** — subclass `RegimeDetector`, return a `RegimeResult`.
* **A new broker** — implement `BrokerClient`. `MockBrokerClient` is the
  reference for the required behaviour.

## Persistence schema

`assets`, `price_bars`, `feature_values`, `configuration_versions`,
`backtest_runs`, `backtest_metrics`, `strategy_signals`, `regime_observations`,
`target_weights`, `orders`, `fills`, `positions`, `portfolio_snapshots`,
`risk_events`, `reconciliations`.

Money and prices use `Numeric`, not `Float`, so repeated arithmetic does not
accumulate binary rounding error in stored records. Every research artefact
hangs off a `backtest_runs` row carrying the configuration snapshot, git commit
and data range.
