# Configuration

All configuration lives in version-controlled YAML under `configs/` and is
validated by pydantic models in `atlas/config.py`. **Unknown keys are rejected**,
so a typo fails at load time rather than silently changing behaviour.

```
configs/
├── assets.yaml       universe, asset classes, data provider
├── strategies.yaml   strategy parameters, ensemble, backtest settings
├── regimes.yaml      regime detector and per-regime weights
├── risk.yaml         portfolio limits, drawdown ladder, hard limits, kill switch
├── execution.yaml    transaction costs, broker, trading window, reconciliation
└── validation.yaml   walk-forward, stability, stress periods, benchmarks
```

Inspect the merged, validated result:

```bash
atlas show-config
atlas show-config --section risk
```

Every configuration has a **hash** (`atlas info`) stored with each run, so a
result can always be traced back to the settings that produced it. Changing a
setting creates a new version; previous versions are never overwritten.

---

## What belongs where

**YAML** — anything that affects research results. Keeping it in git is what
makes a stored result reproducible.

**Environment** (`.env`, see `.env.example`) — deployment-specific settings only:
execution mode, broker host/port/client id, the account allowlist, the kill
switch, the database URL and logging. Atlas never reads a broker username,
password or API key: the IB API authenticates through a TWS/Gateway session you
log into yourself.

---

## Common changes

### Change the universe

```yaml
# configs/assets.yaml
universe:
  benchmark: SPY          # must be in the universe
  defensive_asset: BIL    # must be in the universe
  assets:
    - symbol: SPY
      asset_class: us_equity   # drives the per-class exposure cap
      tradable: true
```

### Switch data provider

```yaml
data:
  provider: yfinance   # yfinance | csv | synthetic
```

Or per command: `atlas backtest --provider synthetic`.

### Turn a strategy off

```yaml
# configs/strategies.yaml
strategies:
  mean_reversion:
    enabled: false
```

### Change the risk target

```yaml
# configs/risk.yaml
portfolio:
  target_volatility: 0.08
  max_portfolio_volatility: 0.12   # must be >= target_volatility
```

### Use fixed rather than regime-conditional weights

```yaml
# configs/strategies.yaml
ensemble:
  mode: fixed     # regime | fixed | equal
regime:
  enabled: false
```

`equal` is the baseline every regime-weighted result should be compared against.

### Adjust transaction costs

```yaml
# configs/execution.yaml
costs:
  commission_per_share: 0.005
  min_commission: 1.0
  half_spread_bps: 1.0
  slippage_bps: 2.0
  market_impact_coefficient: 0.10
```

Raise these for anything less liquid than a large ETF.

---

## Cross-field rules

The configuration models enforce internal consistency, so an impossible
combination fails at load:

* `trend.fast_ma < trend.slow_ma`
* `mean_reversion.exit_threshold < entry_threshold`
* `cross_sectional_momentum.skip_recent_days < lookback`
* `portfolio.target_volatility <= max_portfolio_volatility`
* `portfolio.max_net_exposure <= max_gross_exposure`
* drawdown thresholds have non-increasing multipliers
* the benchmark and defensive asset are in the universe
* `sixty_forty` benchmark weights sum to 1.0
* stress periods have `end > start`
* the walk-forward grid expands to at most 500 combinations
* every walk-forward grid value produces a valid configuration
* `execution.mode` is never `live`

---

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `EXECUTION_MODE` | `backtest` / `dry_run` / `paper` | `dry_run` |
| `ATLAS_IBKR_HOST` | TWS/Gateway host | `127.0.0.1` |
| `ATLAS_IBKR_PORT` | 7497 TWS paper, 4002 Gateway paper | `7497` |
| `ATLAS_IBKR_CLIENT_ID` | API client id | `17` |
| `ATLAS_IBKR_ACCOUNT_ALLOWLIST` | Comma-separated accounts Atlas may trade | empty |
| `ATLAS_KILL_SWITCH` | `1` blocks all trading | `0` |
| `ATLAS_DATABASE_URL` | SQLAlchemy URL | local SQLite |
| `ATLAS_LOG_LEVEL` | Log verbosity | `INFO` |
| `ATLAS_LOG_JSON` | JSON log lines | `0` |
| `ATLAS_PROJECT_ROOT` | Override root detection | auto |

Research parameters are deliberately **not** overridable from the environment —
that would break the link between a stored result and its configuration.

---

## Programmatic overrides

```python
from atlas.config import load_config

config = load_config(overrides={"risk": {"portfolio": {"target_volatility": 0.06}}})
```

Or, for walk-forward-style dotted paths:

```python
from atlas.validation.walk_forward import apply_overrides

variant = apply_overrides(config, {"trend.slow_ma": 150, "portfolio.target_volatility": 0.08})
```

`apply_overrides` deep-copies, so the original is never mutated.
