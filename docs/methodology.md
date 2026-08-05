# Methodology

This document states what Atlas computes and why. Every formula here is
implemented in the module named beside it, and every claim is testable.

---

## 1. Point-in-time discipline

The single rule that everything else depends on:

> A value indexed at date *t* may depend only on observations dated **≤ t**.

Three mechanisms enforce it:

1. **Backward-looking features only.** `atlas/data/features.py` uses trailing
   rolling windows exclusively. No centred windows, no full-sample
   normalisation, no `bfill`.
2. **Execution lag.** A signal formed from the close of day *t* becomes an order
   that fills at the **open of day t+1** (`atlas/backtest/engine.py`). Same-close
   execution exists only as `execution_timing: same_close_unrealistic`, which
   emits a warning on every run and is labelled in the result object.
3. **Tests that try to break it.** `tests/unit/test_lookahead.py` multiplies
   every price after a chosen date by 1.5 and asserts that no historical
   feature, signal or regime label changes. If any did, it would fail.

Warm-up history *before* a backtest's start date is used, and that is not
look-ahead — it only ever looks backwards.

### Price scales

A related discipline, and one that is easier to get wrong: execution and
valuation prices must share a single scale. Vendors report OHLC on the **raw**
scale and supply a separate **adjusted** close, so a run that values positions
at the adjusted close converts its execution open through the same day's
`adj_close / close` factor.

Skipping that conversion books the whole accumulated dividend adjustment as
instant profit on every purchase, and gives it back on every sale. It is not a
small effect — over twenty years at a 1.5% yield the factor exceeds 1.3, and a
30% discontinuity injected on each trade is enough to make the return series
look wildly profitable and wildly fat-tailed at the same time.

---

## 2. Features

`atlas/data/features.py`

| Feature | Definition |
|---|---|
| Trailing return, horizon *h* | $r_t^{(h)} = P_t / P_{t-h} - 1$ |
| Realised volatility, window *w* | $\sigma_t = \text{sd}(r_{t-w+1..t}) \times \sqrt{252}$ |
| EWM volatility | exponentially weighted sd, half-life configurable |
| Moving average | trailing simple mean over *w* closes |
| Price-to-MA | $P_t / MA_t^{(w)} - 1$ |
| Rolling z-score | $(x_t - \mu_{t}^{(w)}) / \sigma_{t}^{(w)}$, clipped at ±5 |
| Drawdown | $P_t / \max(P_{\le t}) - 1$ |
| ATR | mean true range over *w*, divided by price |
| RSI | Wilder smoothing, $\alpha = 1/w$ |
| Cross-sectional rank | percentile rank within each date, scaled to [-1, 1] |
| Breadth | fraction of assets above their own *w*-day MA |
| Average correlation | mean pairwise correlation over a trailing window |

Rolling standard deviations use `ddof=1`. A zero trailing standard deviation
yields `NaN` rather than an infinity.

---

## 3. Strategies

Each strategy returns a continuous signal in $[-1, 1]$ expressing **direction
and conviction**, never a portfolio weight.

### 3.1 Time-series trend following — `strategies/trend.py`

$$s_i = w_1 \tanh\!\left(\frac{P/MA_{fast} - 1}{\sigma_d\sqrt{L_{fast}/2}}\right)
      + w_2 \tanh\!\left(\frac{P/MA_{slow} - 1}{\sigma_d\sqrt{L_{slow}/2}}\right)
      + w_3 \tanh\!\left(\frac{r^{(63)}}{\sigma_d\sqrt{63}}\right)
      + w_4 \tanh\!\left(\frac{r^{(252)}}{\sigma_d\sqrt{252}}\right)$$

The specification suggests `sign(P/MA - 1)` for the first two terms. Atlas uses a
volatility-scaled `tanh` instead, for one reason: `sign` flips discontinuously
whenever price grazes the moving average, and each flip costs a full round trip.
The `tanh` form keeps the same economic meaning — positive above the average,
negative below — while making position size a continuous function of distance,
so turnover falls sharply without changing the signal's direction.

**Rationale.** Time-series momentum is documented across asset classes and
decades (Moskowitz, Ooi & Pedersen 2012). Candidate mechanisms: slow diffusion
of information, under-reaction, and forced trading by leveraged holders.

**Turnover control.** A soft neutral zone zeroes signals below
`neutral_threshold` and rescales the remainder continuously, so position size
does not jump at the boundary.

**Risk control.** When `volatility_scaling` is on, the signal is multiplied by a
gate that decays toward 0.25 as trailing volatility enters the top percentile of
its own *expanding* history.

### 3.2 Short-term mean reversion — `strategies/mean_reversion.py`

Blends the negative z-score of the trailing 5-day return, the negative z-score
of the deviation from a 20-day average, and a rescaled RSI. A position opens at
$|z| \ge$ `entry_threshold`, is held while $|z| >$ `exit_threshold`, and is
force-closed after `max_holding_days` bars — and may not reopen on the bar it was
force-closed, or the holding rule would be unobservable.

**Rationale.** Short-horizon reversal compensates liquidity provision (Nagel
2012). The effect is strongest exactly when providing liquidity is most
dangerous, which is why both a trend filter and a volatility filter are on by
default.

### 3.3 Cross-sectional momentum — `strategies/cross_sectional_momentum.py`

$$\text{score}_i = \frac{P_{t-S}/P_{t-L} - 1}{\sigma_{i, t-S}}$$

with $L =$ `lookback` and $S =$ `skip_recent_days`. Skipping the most recent
month removes contamination from short-term reversal.

**Asset-class neutralisation.** With four US equity ETFs in a fourteen-asset
universe, an unmodified ranking puts all four at the top in any equity bull
market — one bet in four disguises. With `neutralize_asset_class` on, scores are
demeaned within each class before ranking. This deliberately gives up some raw
momentum capture in exchange for diversification. Classes with one member are
left alone, since demeaning a singleton zeroes it permanently.

**Tie-breaking.** Ranks use the *average* method, so tied assets share the
average of the ranks they span and the cross-sectional mean stays at zero.
Ineligible assets are dropped from the cross-section entirely rather than ranked
last, so their absence does not distort the surviving ranks.

---

## 4. Regime detection — `regimes/rules_based.py`

Two observable quantities, crossed:

* **Direction** — is the benchmark above its 200-day moving average?
* **Turbulence** — is 20-day realised volatility above the 70th percentile of its
  own *expanding* history?

|  | Low volatility | High volatility |
|---|---|---|
| **Above trend** | `bull_low_vol` | `bull_high_vol` |
| **Below trend** | `bear_low_vol` | `bear_high_vol` |

A percentile rather than an absolute threshold: a fixed "VIX > 20" rule
misclassifies both 2017 and 2008 at different points in a long sample.

Breadth, average pairwise correlation and drawdown are recorded alongside the
label. They do not drive the discrete classification but do feed the defensive
overlay.

A **persistence filter** requires a new classification to hold `min_regime_days`
consecutive days before it is accepted. The filter is causal — it can only ever
delay a change, never anticipate one, which `test_lookahead.py` asserts.

An optional clustering detector is available for comparison. It is fitted on
training data only, is off by default, and never replaces the rules-based
baseline.

---

## 5. Signal combination — `strategies/ensemble.py`

$$s_{i,t} = \sum_{k=1}^{K} a_k(r_t)\, c_{i,k,t}\, s_{i,k,t}$$

* $a_k(r_t)$ — strategy weight in the regime prevailing at *t*, read from
  `configs/regimes.yaml`. **Weights sum to one before any risk scaling**, so the
  ensemble changes the *mix* of strategies, never the total risk.
* $c_{i,k,t}$ — optional confidence in $[0,1]$, floored at 0.25 so a
  low-confidence view is damped rather than silently deleted.

Strategy weights are **never optimised on returns**. In-sample optimisation of
ensemble weights is the fastest route to a result that does not survive
walk-forward, so Atlas does not offer it. `equal` mode is the baseline every
regime-weighted result should be compared against.

The combined signal is exponentially smoothed (`signal_smoothing_alpha`) and
thresholded (`min_signal`) to damp turnover. Each strategy's contribution is
retained, so attribution is exact rather than inferred.

---

## 6. Portfolio construction — `portfolio/allocator.py`

1. **Inverse-volatility scaling:** $\tilde w_i = s_i / \sigma_i$ — a unit of
   conviction buys the same *risk* in a 30%-volatility equity ETF as in a
   5%-volatility bond ETF.
2. **Gross normalisation:** $w_i = \tilde w_i / \sum_j |\tilde w_j|$.
3. **Volatility targeting:**
   $w^*_i = w_i \cdot \sigma_{\text{target}} / \hat\sigma_p$ where
   $\hat\sigma_p = \sqrt{w^\top \Sigma w}$. The scaler is clipped to
   $[\text{min\_vol\_target\_scaler}, \text{max\_vol\_target\_scaler}]$ so a low
   predicted volatility cannot manufacture extreme leverage.
4. **Risk multipliers** — regime multiplier × drawdown multiplier, applied to
   the target, never above 1.
5. **Defensive sleeve** — a floor allocation to the cash-like asset, funded by
   scaling the rest of the book down rather than adding leverage on top.
6. **Constraints** — see below.

### Covariance — `portfolio/covariance.py`

| Estimator | Notes |
|---|---|
| `sample` | Unbiased, noisy; smallest eigenvalues biased toward zero. |
| `shrinkage` | Ledoit-Wolf toward a scaled identity. **Default.** Well conditioned by construction. |
| `ewma` | Exponentially weighted; tracks a changing correlation structure faster. |

Every estimate is floored on the diagonal, symmetrised, and repaired by
eigenvalue clipping if it is not positive definite. When estimation is impossible
the fallback is a **diagonal** matrix — conservative, because ignoring
diversification makes the volatility target bind sooner rather than later.

### Constraints — `portfolio/constraints.py`

Applied by projection in a fixed order, each adjustment recorded with a reason:
long-only → per-asset cap (with proportional redistribution) → asset-class cap →
position count → gross/net/leverage → cash reserve → turnover → small-trade
threshold. Steps 5–7 are pure scalings of an already-feasible vector, so they
cannot reintroduce an earlier violation; the hard caps are re-applied once at the
end because the turnover blend can pull a weight back toward a previous value
that itself sat at a cap.

A full exit is never blocked by the small-trade rule — leaving a residual sliver
would defeat the intent of closing a position.

---

## 7. Risk management — `portfolio/risk_manager.py`

The risk manager is independent of strategy logic. It receives proposed weights
and passes, scales or blocks them, recording a `RiskEvent` for every decision.
Nothing in it knows what a moving average is: a bug or an overfit in a strategy
cannot disable a risk control.

**Drawdown ladder.** Risk budget multiplier by current drawdown: <5% → 1.00,
5–10% → 0.75, 10–15% → 0.50, >15% → 0.25. The ladder scales the *risk budget*.
Atlas never liquidates automatically on drawdown unless `liquidate_on_breach` is
explicitly set. Beyond `max_drawdown` only risk-*reducing* trades are permitted.

Loss limits (daily, weekly), a predicted-volatility ceiling, a turnover cap, a
correlation-concentration warning, order notional bounds, buying-power checks,
duplicate-order suppression, a trading-window check, a consecutive-API-failure
halt, and a global kill switch complete the set.

---

## 8. Transaction costs — `backtest/costs.py`

$$\text{Cost} = \underbrace{\max(c_{fix} + c_{sh} q,\ c_{min})}_{\text{commission}}
+ \underbrace{\tfrac{1}{2}s \cdot Pq}_{\text{spread}}
+ \underbrace{\kappa \cdot Pq}_{\text{slippage}}
+ \underbrace{\lambda \sigma_d \sqrt{q/V} \cdot Pq}_{\text{impact}}$$

Impact follows a square-root law in participation rate — the standard functional
form from the microstructure literature (Almgren et al. 2005). Spread, slippage
and impact move the **fill price** against the trader; commission is billed
separately, matching how a broker actually charges. Costs are tracked by category
and the report shows gross performance, net performance and the share of gross
return consumed by trading.

When volume data is unavailable, impact is **zero rather than guessed**, and the
validation report flags the affected symbols.

---

## 9. Performance metrics — `backtest/metrics.py`

Stated conventions, because most disagreements about a Sharpe ratio are really
disagreements about conventions:

* Simple daily returns, annualised by **252** trading days.
* Sharpe and Sortino use **excess** returns; the annual risk-free rate is
  de-annualised **geometrically** ($(1+r_f)^{1/252} - 1$) before subtraction, and
  the assumed rate is printed with the numbers.
* Drawdowns are measured on the **equity curve**, so they survive gaps.
* VaR and expected shortfall are **historical** (empirical quantiles) at 95% and
  99%, reported as positive loss magnitudes.
* Sortino returns `NaN`, not 0, when no observation falls below the target —
  returning zero would rank a never-losing series worst.

---

## 10. Walk-forward validation — `validation/walk_forward.py`

```
|<-- train 5y -->|~|<- val 1y ->|~|<- test 1y ->|
     |<-- train 5y -->|~|<- val 1y ->|~|<- test 1y ->|   (+1 year)
```
(`~` = embargo gap)

Per window: run each candidate over train+validation in one backtest; score
candidates on the **validation** slice; freeze the winner; re-run over
train+validation+test and report only the **test** slice.

**Selection score** — deliberately not return:

$$\text{score} = w_1 \text{Sharpe}_{val} - w_2 \text{MDD}_{val}
- w_3 \text{Turnover}_{val} - w_4 |\text{Sharpe}_{train} - \text{Sharpe}_{val}|$$

Candidates breaching a maximum drawdown or turnover are discarded before scoring.

**Embargo.** A gap of `embargo_days` between adjacent slices, because a feature
with a 252-day lookback computed on the first day of a test window would
otherwise overlap validation.

**The number that matters** is `train_minus_test_sharpe`. A large positive value
means the selection process was fitting noise, and it is reported prominently
rather than buried.

The parameter grid ships at **12 combinations** and the configuration model
rejects anything above 500. Cost scales as
`n_candidates × n_windows × (train_years + validation_years)`.

---

## 11. Parameter stability — `validation/parameter_stability.py`

Each parameter is swept one at a time around the baseline. Two fragility flags:

* **gap** — the best setting beats the median by more than
  `fragility_sharpe_gap` Sharpe units;
* **dispersion** — the coefficient of variation of Sharpe across the sweep
  exceeds `cv_threshold`.

The sweep is repeated gross of costs, so a strategy whose edge exists only before
trading costs is exposed rather than hidden. A broad plateau is evidence; a lone
spike is a curve fit.

---

## 12. Stress testing and resampling — `validation/stress_tests.py`

Named historical windows (2008, 2011, 2015–16, Q4 2018, 2020, 2022, 2023) are
evaluated as a **panel**. Consistency across windows is the signal; one good
crisis is not. Windows outside the sample are reported as skipped, never dropped
silently.

**Block bootstrap.** Resampling individual days independently destroys the serial
dependence that produces drawdowns, and would make almost any strategy look
safer than it is. Atlas samples contiguous blocks (fixed or Politis-Romano
geometric), preserving autocorrelation and volatility clustering, and wraps
around the sample end so every observation is equally likely to appear.

This is labelled throughout as a **statistical stress test, not a forecast**.

---

## 13. Benchmarks — `validation/benchmarks.py`

SPY buy-and-hold, 60/40 SPY/IEF, equal-weight universe, each strategy standalone,
the equal-weight ensemble, and the regime-aware ensemble.

The strategy variants run through *identical* portfolio, cost and risk machinery,
so any difference between them is attributable to the signal rather than the
plumbing. Passive benchmarks let weights drift between rebalances and pay
round-trip costs when they rebalance — continuous free rebalancing would flatter
them. All benchmarks are evaluated over the window Atlas actually traded.

---

## 14. What this methodology cannot tell you

* The universe is a fixed list of ETFs that exist and are liquid *today*.
  Survivorship and selection bias are present and not corrected.
* A handful of crises is a small sample for any claim about tail behaviour.
* Free adjusted-price data is revised retroactively; a result computed today may
  not reproduce exactly next year.
* Even a 12-point grid selected on validation slices touches the data. The
  out-of-sample degradation figure is the honest measure of what that cost.

See [limitations.md](limitations.md) for the full list.
