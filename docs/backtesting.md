# Backtesting

## The event loop

`atlas/backtest/engine.py` advances one trading day at a time. Nothing is
vectorised across time, so a bug that reaches forward has nowhere to hide.

For each bar *t*:

1. **Market event** — bar *t*'s data becomes visible.
2. **Fills** — orders queued at *t-1* execute at *t*'s **open**. Cash and
   positions update; costs are charged.
3. **Dividends** — credited when running on raw prices.
4. **Mark to market** — positions valued at *t*'s close; a portfolio snapshot is
   recorded.
5. **Rebalance** (on schedule) — signals from data through *t*'s close produce
   target weights, the risk manager vets them, and orders are queued for *t+1*.

Step 5 happens *after* step 4 and produces orders for the *next* bar. That
ordering is the whole point.

## Execution timing

| Mode | Fill price | Use |
|---|---|---|
| `next_open` | open of *t+1* | **default**, realistic |
| `next_close` | close of *t+1* | research comparison |
| `same_close_unrealistic` | close of *t* | **not achievable**; warns on every run and is flagged in the result |

## Order lifecycle

`created → pending → submitted → partially_filled → filled`
(or `rejected` / `cancelled`)

Orders carry a positive quantity plus a side, matching broker APIs and removing
a class of sign errors. Each has a unique id so a lost response can still be
reconciled.

## Simulated broker

`atlas/backtest/broker.py` models what actually goes wrong at daily frequency:

* **Rejection** when no price exists on the execution bar.
* **Partial fills** when an order exceeds `max_participation × ADV`.
* **Whole-share rounding** when fractional shares are off — including rejecting
  an order that rounds to zero shares.
* **Limit orders** that do not cross are cancelled, not filled.
* **Sells execute before buys**, so proceeds fund the purchases on the same bar.

It is deterministic: identical inputs give identical fills.

## Cash accounting

`equity = cash + Σ(quantity × price)` — always, checked by an integration test
to within 1e-6 on every day of the run. Every fill moves cash; every dividend
credits cash. Nothing is inferred from returns.

## Rebalance schedules

* `daily` — every bar.
* `weekly` — the configured weekday, or that week's last trading day when the
  market was shut on it.
* `monthly` — each month's last trading day.

## Warm-up

History *before* the trading window is used to warm up lookbacks. This is not
look-ahead: it only ever looks backwards. The first tradable date is
`max(warmup_days, longest_strategy_lookback)` bars into the panel.

## Reproducibility

Every run records a unique run id, the git commit hash, the full configuration
snapshot and its hash, the data range, the data source, and whether the data was
synthetic. `export_backtest` writes all of it plus every artefact to one
directory with a `manifest.json`.

## Performance

A 19-year, 14-asset, weekly-rebalanced backtest takes roughly 25 seconds. The
loop is deliberately not vectorised; the optimisations that matter are:

* symbol columns held as an object-dtype index (arrow-backed string indexes make
  the millions of small per-symbol lookups materially slower — this alone roughly
  halves total runtime);
* constraint and risk layers operating on NumPy arrays with memoised
  asset-class groupings;
* the allocator receiving only the trailing window each estimator needs, rather
  than the whole history so far.

## What a backtest cannot tell you

See [limitations.md](limitations.md). In short: only the walk-forward test
slices are out of sample, the cost model is an approximation, and paper fills
are not real fills.
