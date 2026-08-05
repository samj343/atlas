# Risk management

The risk manager (`atlas/portfolio/risk_manager.py`) is **independent of
strategy logic**. It receives proposed portfolio weights and either passes,
scales or blocks them, recording a `RiskEvent` for every decision. Nothing in it
knows what a moving average is — which is the point: a bug or an overfit in a
strategy cannot disable a risk control.

## Layers

Risk is enforced in three places, deliberately overlapping:

1. **Strategy level** — volatility gates inside trend and mean reversion.
2. **Portfolio level** — volatility targeting and the constraint projection.
3. **Risk manager** — hard limits applied *after* portfolio construction, plus
   per-order checks applied *after* orders are sized.

An error at one layer is caught by the next.

## Position controls

| Control | Default | Action |
|---|---|---|
| Max weight per symbol | 20% | cap |
| Max asset-class exposure | 50% | scale the class |
| Max gross exposure | 1.25× | scale the book |
| Max net exposure | 1.00× | scale the book |
| Max leverage | 1.25× | scale the book |
| Max active positions | 12 | keep the largest |
| Min order notional | $100 | skip |
| Max order notional | $50,000 | trim |

## Portfolio controls

| Control | Default | Action |
|---|---|---|
| Target volatility | 10% | scale to target |
| Max predicted volatility | 15% | hard shrink |
| Max drawdown | 25% | risk-reducing trades only |
| Daily loss limit | 4% | halt (5-day cool-off) |
| Weekly loss limit | 8% | halt (5-day cool-off) |
| Max daily turnover | 30% | blend toward current weights |
| Correlation warning | 0.80 | warn |

### Drawdown ladder

| Drawdown | Risk multiplier |
|---|---|
| < 5% | 1.00 |
| 5–10% | 0.75 |
| 10–15% | 0.50 |
| > 15% | 0.25 |

Two design decisions matter here:

**The ladder scales the risk *budget*, not the positions directly.** Atlas never
liquidates automatically on drawdown unless `liquidate_on_breach` is explicitly
enabled.

**The reference peak is the highest equity of the trailing two years, not the
all-time high.** Against an all-time peak the ladder is a one-way ratchet: a
portfolio cut to a quarter of its risk budget cannot earn its way back to the old
high, so the reduction never lifts and the strategy is permanently disabled by a
single bad year. A rolling peak lets the reference decay. Set
`peak_lookback_days: 0` for all-time-peak behaviour if you want it.

### Loss limits are circuit breakers

A daily or weekly loss-limit breach halts trading for
`loss_limit_cooldown_days` trading days and then lifts automatically — provided
the limit is no longer breached. A latched halt would freeze the remainder of a
multi-year backtest and make the result meaningless.

Halts that require human judgement — unreconciled positions, repeated API
failures, the kill switch — **never** auto-resume.

## Operational controls

* **Stale data** — reject the cycle when the newest observation is older than
  the limit.
* **Missing prices** — drop the affected position rather than blocking
  everything.
* **Duplicate orders** — an identical `(symbol, side, quantity, date)` within
  the duplicate window is refused.
* **Trading window** — orders outside 09:35–15:55 New York on a weekday are
  refused.
* **Buying power** — a buy exceeding available funds is refused.
* **Reconciliation** — a material broker/local mismatch blocks the next cycle.
* **API failures** — three consecutive failures halt trading.
* **Kill switch** — `ATLAS_KILL_SWITCH=1` blocks everything, everywhere.

## Pre-trade safety gate

Before any order can be submitted (`atlas/execution/safety.py`):

1. kill switch off;
2. execution mode is `paper` (never `live` — rejected twice, independently);
3. broker connected;
4. account identified;
5. account is an Interactive Brokers **paper** account (`DU`/`DF` prefix);
6. account is on the configured allowlist;
7. market data is fresh;
8. positions retrieved;
9. positions reconcile;
10. risk manager not halted.

A failure raises rather than returning a status code — a safety check that can
be ignored by forgetting an `if` is not a safety check.

## Reconciliation

A difference must exceed **all three** tolerances (shares, value, percentage) to
count as a mismatch. A position present on only one side is always material,
regardless of size, because that indicates a structural problem rather than
rounding.

## Audit trail

Every decision produces a `RiskEvent` with a timestamp, control name, severity,
action, message and the observed-versus-limit values. Events are persisted with
the run and surfaced on the dashboard's Risk Monitor page.
