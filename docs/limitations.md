# Limitations

Atlas is a research and paper-trading platform. This page lists what it does
*not* establish, because a system that only advertises its strengths is not a
research tool.

---

## 1. Backtests are not evidence of skill

A backtest is a description of what a rule would have done on data that already
exists. Only the **walk-forward test slices** are genuinely out of sample, and
even those are produced by a process that saw the rest of the sample when the
universe, the strategy families and the constraint structure were chosen.

The number to look at is `train_minus_test_sharpe` in the walk-forward output.
Atlas reports it prominently, not in a footnote.

## 2. Survivorship and selection bias

The universe is a fixed list of fourteen ETFs that exist today and are liquid
today. Funds that closed or never gathered assets are absent. The choice of
these fourteen instruments is itself a decision made with hindsight — nobody
assembled this list in 2005.

Atlas does **not** correct for this. Nothing in the codebase can.

## 3. Data quality

* Free adjusted-price series are **revised retroactively**. A total-return
  series downloaded today may differ from the same series next year, so a result
  is not exactly reproducible from the vendor alone. The configuration snapshot
  and git hash pin the *code*; they cannot pin the *vendor's* data.
* Corporate actions, splits and distributions are taken as the vendor reports
  them. The validator flags suspicious moves but cannot repair a bad adjustment.
* Volume is used for market-impact estimation. Where it is missing, impact is
  set to **zero rather than guessed**, which understates the cost of large trades
  in those names.

## 4. The cost model is an approximation

Commission, half-spread, slippage and a square-root impact term are modelled.
Real trading costs differ in ways that systematically hurt:

* the spread **widens exactly when the strategy most wants to trade** — during a
  volatility spike — and Atlas models it as a constant;
* the impact coefficient is an assumption, not a calibration;
* no borrow cost, financing cost, or short availability is modelled (the default
  configuration is long-only, which sidesteps rather than solves this);
* no fund expense ratio is deducted from the ETFs themselves.

## 5. Execution modelling

* Orders fill at the **next open**, whatever happened overnight. A gap through
  the open is not modelled.
* No intraday path, no latency, no queue position, no partial-day liquidity.
* Partial fills are modelled only through a participation cap; a real
  unfillable order looks different.
* **Paper fills are not real fills.** A paper account will fill orders that a
  real market would not, particularly in size or in stress. This is the single
  largest gap between paper results and reality.

## 6. Statistical power

The sample contains a handful of genuine stress episodes — 2008, 2011, 2015-16,
Q4 2018, 2020, 2022. That is a very small sample for any claim about tail
behaviour. The block bootstrap widens the distribution of outcomes consistent
with the historical *return distribution*, but it cannot invent a crisis the
sample never contained, and it is labelled throughout as a statistical stress
test rather than a forecast.

## 7. Parameter selection still touches the data

The grid is deliberately small (12 combinations) and selection uses a
multi-objective score on validation slices rather than raw return. This reduces
overfitting; it does not eliminate it. Every reported out-of-sample figure is
downstream of choices informed by the sample.

## 8. Regime detection is a description, not a prediction

The rules-based detector labels the state the market is *in*, with a deliberate
lag from the persistence filter. It does not forecast the next regime, and Atlas
does not use it as a return forecast — only to tilt strategy weights and scale
the risk budget.

## 9. Modelling choices that could be wrong

* Volatility targeting assumes volatility is more forecastable than return. It
  usually is, but it fails when correlations jump.
* Covariance shrinkage improves conditioning at the cost of some bias.
* Inverse-volatility scaling implicitly assumes similar risk-adjusted returns
  across assets.
* The drawdown ladder measures against a **rolling two-year peak**. Against an
  all-time peak it becomes a one-way ratchet that permanently disables the
  strategy; against a shorter window it de-risks too late. Two years is a
  judgement call.

## 10. Operational scope

* Daily frequency only. No intraday data, signals or execution.
* One account, one currency, one broker.
* No tax modelling, no wash-sale rules, no accounting for the difference between
  a taxable and a tax-deferred account.
* The system is designed to be **supervised**. It has a kill switch, halts and
  reconciliation blocks precisely because it is not intended to run unattended.

## 11. Live trading is not implemented

`EXECUTION_MODE=live` is rejected at configuration load *and* by the pre-trade
safety gate. This is deliberate and is not an oversight to be "fixed" by
deleting the checks: the system has never been validated against real fills, and
the gap between paper and real execution (section 5) is exactly the thing that
would need validating first.

---

## What Atlas *does* establish

* That a multi-strategy, regime-aware, volatility-targeted portfolio can be
  implemented with no look-ahead bias — and that the absence of look-ahead is
  **testable**, not merely asserted (`tests/unit/test_lookahead.py`).
* What the strategy's turnover and transaction costs actually are, itemised.
* How the risk controls behave, including when they bind.
* How much performance degrades from in-sample to out-of-sample under a
  disciplined selection process.
* That the whole pipeline is reproducible from a configuration hash and a git
  commit.

Those are engineering and methodology claims, and they are the claims this
project is making.
