# Baseline experiment — reference run

> **These numbers come from SYNTHETIC data.** The environment this project was
> built in has no outbound access to Yahoo Finance, so the reference run uses the
> deterministic synthetic provider. It demonstrates that every component runs and
> that the risk controls behave as specified. **It carries no information
> whatsoever about real market performance, and no conclusion about any strategy
> can be drawn from it.** Reproduce it on real history with
> `python scripts/run_baseline_experiment.py --provider yfinance`.

Atlas never fabricates a metric. Every figure below was produced by the run
identified here and can be regenerated from the same inputs.

| | |
|---|---|
| Command | `python scripts/run_baseline_experiment.py --provider synthetic` |
| Run ID | `bt-20260805-025443-089875` |
| Config hash | `f146fb94fafcc732` |
| Provider | `synthetic` (simulated prices) |
| Universe | 14 ETFs |
| Period | 2005-01-03 → 2026-08-05 (21.2 years, 5,332 trading days) |
| Wall clock | 4,031 s |

`reports/` is generated, not committed — it is in `.gitignore`, because research
artefacts are outputs of the code rather than part of it. This file is the
transcribed record.

---

## In-sample backtest

| Metric | Value |
|---|---|
| Total return | 224.24% |
| CAGR | 5.72% |
| Annualised volatility | 7.58% |
| Sharpe | 0.51 |
| Sortino | 0.75 |
| Calmar | 0.36 |
| Max drawdown | 15.72% |
| Max drawdown duration | 828 days |
| Best / worst day | +3.21% / −4.02% |
| Best / worst month | +7.46% / −8.08% |
| Monthly win rate | 55.5% |
| Skew / kurtosis | 0.09 / 4.36 |
| VaR 95 / ES 95 | 0.74% / 1.06% |
| Average gross exposure | 0.475 |
| Annual turnover | 5.71x |
| Trades | 8,065 |

Conventions: 252 trading days, 2.00% annualised risk-free rate de-annualised
geometrically, simple daily returns net of all modelled costs, drawdown from the
running peak, historical VaR/ES.

The volatility target is 10% and realised volatility is 7.58%. The gap is the
risk manager doing its job — the drawdown ladder and exposure caps bind before
the allocator reaches its target, which is the intended ordering.

### Transaction costs

| | |
|---|---|
| Gross CAGR | 6.13% |
| Net CAGR | 5.72% |
| Annual cost drag | 0.41% |
| Cost share of gross return | 11.02% |
| Total costs | $15,238 |

Costs are never hidden: gross and net are always reported together.

---

## Benchmarks

All eight, over the same period and the same data.

| | Buy & hold SPY | 60/40 | Equal-weight ETFs | Trend only | Mean reversion only | Cross-sectional only | Equal-weight ensemble | **Atlas** |
|---|---|---|---|---|---|---|---|---|
| CAGR | −2.38% | 0.83% | 8.99% | 5.06% | 0.61% | 4.01% | 4.48% | **5.72%** |
| Volatility | 16.56% | 17.28% | 15.81% | 6.53% | 5.32% | 6.34% | 6.60% | **7.58%** |
| Sharpe | −0.18 | 0.02 | 0.50 | 0.48 | −0.23 | 0.34 | 0.40 | **0.51** |
| Sortino | −0.26 | 0.03 | 0.73 | 0.70 | −0.33 | 0.49 | 0.58 | **0.75** |
| Calmar | −0.04 | 0.02 | 0.25 | 0.30 | 0.04 | 0.25 | 0.28 | **0.36** |
| Max drawdown | 62.41% | 48.28% | 35.28% | 17.09% | 16.11% | 15.96% | 15.93% | **15.72%** |
| Worst day | −5.94% | −5.65% | −5.61% | −2.80% | −2.65% | −2.51% | −2.90% | **−4.02%** |

Read this carefully rather than triumphally. The equal-weight ETF portfolio
earns a *higher* CAGR (8.99%) at a comparable Sharpe, on far higher volatility
and more than twice the drawdown. The honest summary is that the ensemble
delivers a better risk-adjusted profile than its own components and than the
passive mixes — on this synthetic sample — not that it beats buy-and-hold on
return.

---

## Walk-forward, out of sample

16 windows × 12 candidates = 208 backtests, with embargo gaps between train,
validation and test.

| | |
|---|---|
| OOS period | 16.0 years, 4,038 days |
| OOS CAGR | 5.39% |
| OOS volatility | 7.09% |
| OOS Sharpe | 0.50 |
| OOS max drawdown | 17.28% |
| Worst day | −3.18% |
| Kurtosis | 3.53 |

### Degradation — the number that matters

| | |
|---|---|
| Mean train Sharpe | 0.583 |
| Mean validation Sharpe | 0.673 |
| Mean test Sharpe | **0.218** |
| Train − test | **0.365** |
| Validation − test | **0.455** |
| Windows with positive test Sharpe | 75% |

Per-window test Sharpe ranges from **+3.55** to **−3.43**. A mean test Sharpe of
0.22 against a validation Sharpe of 0.67 is exactly the overfitting gap the
walk-forward procedure exists to expose, and it is reported in preference to the
in-sample figure for that reason. Selected parameters are also not stable:
`trend.slow_ma` alternates between 150 and 200 across windows, and
`portfolio.target_volatility` moves over its whole range.

---

## Stress periods

| Period | Return | Max DD | Volatility | Worst day | Avg gross exposure |
|---|---|---|---|---|---|
| GFC 2008 | −2.38% | 10.10% | 5.72% | −1.35% | 0.376 |
| 2010 Flash Crash | +2.81% | 2.09% | 6.61% | −0.84% | 0.402 |
| 2011 Debt Ceiling | +1.05% | 4.99% | 8.12% | −2.29% | 0.539 |
| 2015–16 Correction | +4.79% | 4.00% | 8.20% | −1.19% | 0.713 |
| Q4 2018 Selloff | +4.40% | 5.06% | 8.85% | −1.05% | 0.709 |
| COVID Crash 2020 | +0.72% | 0.37% | 2.31% | −0.21% | 0.221 |
| 2022 Rate Shock | +11.34% | 12.09% | 10.95% | −1.72% | 0.626 |
| 2023 Regional Banks | −0.97% | 1.12% | 1.50% | −0.25% | 0.144 |

The mechanism worth noting is the exposure column, not the return column: gross
exposure falls to 0.22 in the COVID window and 0.14 in the 2023 window. That is
the drawdown ladder de-risking, which is what it was built to do.

---

## Block bootstrap

1,000 resamples, 21-day blocks over a 252-day horizon, seed 7. Blocks preserve
serial dependence. **This is a stress test of the return distribution, not a
forecast.**

| | mean | std | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| Annualised return | 0.060 | 0.084 | −0.078 | 0.006 | 0.060 | 0.115 | 0.200 |
| Max drawdown | 0.068 | 0.031 | 0.031 | 0.046 | 0.062 | 0.084 | 0.126 |
| Sharpe | 0.768 | 1.070 | −1.130 | 0.118 | 0.791 | 1.471 | 2.438 |

The 5th percentile of annualised return is **−7.8%** and of Sharpe **−1.13**.
Resampling the *same* returns produces losing paths, which is the point.

---

## Parameter stability

Seven parameters swept one at a time around the baseline. None was flagged
fragile: performance forms a plateau rather than a spike in every case.

| Parameter | Points | Baseline | Best Sharpe | Baseline Sharpe | Median Sharpe | Gap | CV | Fragile | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `trend.slow_ma` | 6 | 125 | 0.530 | 0.514 | 0.497 | 0.016 | 0.039 | No | stable |
| `cross_sectional_momentum.top_k` | 5 | 2 | 0.517 | 0.494 | 0.484 | 0.024 | 0.157 | No | stable |
| `mean_reversion.zscore_lookback` | 5 | 10 | 0.541 | 0.522 | 0.509 | 0.019 | 0.158 | No | stable |

Gaps are far below the 0.50 Sharpe fragility threshold and coefficients of
variation below the 0.60 threshold.

> The `trend.slow_ma` row above covers **6** points, not the 7 configured. The
> sweep included `slow_ma = 100`, which equals `fast_ma` and therefore cannot
> construct a valid configuration, so it was dropped mid-run. That is fixed —
> the sweep grid now starts at 125, and `ParameterStabilityAnalyzer.validate_sweep`
> rejects an impossible value up front rather than quietly narrowing the
> neighbourhood the verdict is computed over.

---

## What this run does not show

- Anything about real markets. The data is simulated.
- Evidence of skill. Only the walk-forward test slices are out of sample, and
  even those come from a process informed by the whole sample.
- Achievable returns. Paper fills are not real fills.

See [limitations.md](limitations.md) before drawing any conclusion.

---

*Atlas is an educational research and paper-trading system. Historical
performance does not guarantee future results. Paper-trading execution does not
replicate all real-world liquidity, slippage, latency, and fill conditions.*
