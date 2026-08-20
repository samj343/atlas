"""Per-version sanity: import, quote validity, pricing accuracy, solvency."""
import math
import random

from common import load_all

MODS = load_all()
T = MODS["v4"]  # canonical template classes for building market objects

FED, AJR, THR = T.FED_FUNDS_RATE_UNDERLYING_ID, T.AJARAI_UNDERLYING_ID, T.THERIODIC_UNDERLYING_ID

TRUE_PARAMS = T.MarketParameters(
    ajarai_drift=0.002, ajarai_idio_std_dev=0.025, ajarai_rate_beta=-0.15, ajarai_sector_beta=1.2,
    rate_down_probability=0.10, rate_reversion_strength=0.08, rate_up_probability=0.15,
    sector_std_dev=0.015, theriodic_drift=0.001, theriodic_idio_std_dev=0.02,
    theriodic_rate_beta=-0.25, theriodic_sector_beta=0.9,
)
START = {FED: 2.25, AJR: 850.0, THR: 910.0}


def underlyings(v):
    return [T.Underlying("FED", FED, v[FED]), T.Underlying("AJR", AJR, v[AJR]), T.Underlying("THR", THR, v[THR])]


def gen_history(params, values, num_days, seed):
    random.seed(seed)
    series = {k: [values[k]] for k in values}
    v = dict(values)
    for _ in range(num_days - 1):
        v = params.advance_step(v)
        for k in series:
            series[k].append(v[k])
    return T.MarketHistory(values_by_underlying_id={k: tuple(x) for k, x in series.items()}), v


def mc_price(params, values, option, n_sims, seed=7):
    random.seed(seed)
    hits = 0
    for _ in range(n_sims):
        v = dict(values)
        for _ in range(option.steps_until_expiry):
            v = params.advance_step(v)
        hits += option.expiry_valuation(v)
    return hits / n_sims


OPTIONS = [
    T.BinaryOption(legs=(T.OptionLeg(FED, 1.0),), option_id=1, steps_until_expiry=3, strike=2.5),
    T.BinaryOption(legs=(T.OptionLeg(AJR, 1.0),), option_id=2, steps_until_expiry=5, strike=900.0),
    T.BinaryOption(legs=(T.OptionLeg(THR, 1.0),), option_id=3, steps_until_expiry=4, strike=880.0),
    T.BinaryOption(legs=(T.OptionLeg(AJR, 1.0), T.OptionLeg(THR, -1.0)), option_id=4, steps_until_expiry=4, strike=0.0),
    T.BinaryOption(legs=(T.OptionLeg(FED, -1.0),), option_id=5, steps_until_expiry=2, strike=-2.25),
]


def test_pricing():
    mc = {o.option_id: mc_price(TRUE_PARAMS, START, o, 60_000) for o in OPTIONS}
    for tag, mod in MODS.items():
        mm = mod.MarketMaker(underlyings(START), [], 1000.0)
        worst = 0.0
        for opt in OPTIONS:
            theo = mm.price_option_from_parameters(TRUE_PARAMS, opt)
            assert 0.0 <= theo <= 1.0
            worst = max(worst, abs(theo - mc[opt.option_id]))
        limit = 0.40 if tag == "v1" else 0.01  # v1 is deliberately crude (independent legs)
        status = "OK" if worst < limit else "FAIL"
        print(f"  {status} {tag}: worst THEO error vs MC = {worst:.4f} (limit {limit})")
        assert worst < limit, tag


def test_quotes_and_estimation():
    history, final_values = gen_history(TRUE_PARAMS, START, 400, seed=3)
    for tag, mod in MODS.items():
        mm = mod.MarketMaker(underlyings(final_values), [], 500.0)
        mm.warm_up(history)
        for opt in OPTIONS:
            p = mm.price_option(opt)
            assert 0.0 <= p <= 1.0, (tag, p)
            for cpty in (1, 2):
                q = mm.quote(opt, cpty)
                assert 0.0 <= q.bid_price < q.offer_price <= 1.0
                assert q.bid_quantity > 0 and q.offer_quantity > 0
            fok = T.FokOrder(counterparty_id=1, option_id=opt.option_id, order_type=T.OrderType.BUY, price=0.99, quantity=5)
            assert isinstance(mm.respond_to_fok(opt, fok), bool)
        print(f"  OK {tag}: warm-up, pricing, quotes, fok all valid")


def test_solvency_session(tag, mod, days=50, cash0=100.0, seed=42):
    """Single-MM session with grader-style accounting; assert never bankrupt."""
    random.seed(seed)
    history, values = gen_history(TRUE_PARAMS, START, 350, seed=seed)
    mm = mod.MarketMaker(underlyings(values), [], cash0)
    mm.warm_up(history)
    grader_cash = cash0
    positions = {}
    active = []
    next_id = [1]

    def new_option(v):
        next_id[0] += 1
        r = random.random()
        days_left = random.randint(1, 6)
        if r < 0.4:
            k = max(round(v[FED] + random.choice([-0.5, -0.25, 0, 0.25, 0.5]), 2), 0.0)
            return T.BinaryOption(legs=(T.OptionLeg(FED, 1.0),), option_id=next_id[0], steps_until_expiry=days_left, strike=k)
        if r < 0.8:
            uid = random.choice([AJR, THR])
            return T.BinaryOption(legs=(T.OptionLeg(uid, 1.0),), option_id=next_id[0], steps_until_expiry=days_left, strike=round(v[uid] * random.uniform(0.96, 1.04), 2))
        legs = (T.OptionLeg(AJR, 1.0), T.OptionLeg(THR, -1.0)) if random.random() < 0.5 else (T.OptionLeg(THR, 1.0), T.OptionLeg(AJR, -1.0))
        return T.BinaryOption(legs=legs, option_id=next_id[0], steps_until_expiry=days_left, strike=0.0)

    def fill(opt, px, signed_qty):
        nonlocal grader_cash
        mm.on_trade(opt, px, signed_qty, 1)
        grader_cash -= signed_qty * px if signed_qty > 0 else (-signed_qty) * (1.0 - px)
        entry = positions.setdefault(opt.option_id, [0, 0])
        if signed_qty > 0:
            entry[0] += signed_qty
        else:
            entry[1] += -signed_qty

    for day in range(days):
        for _ in range(random.randint(1, 3)):
            active.append(new_option(values))
        for _ in range(random.randint(2, 6)):
            opt = random.choice(active)
            q = mm.quote(opt, random.randint(1, 4))
            if random.random() < 0.6:
                if random.random() < 0.5 and q.offer_price <= 0.97:
                    fill(opt, q.offer_price, -random.randint(1, q.offer_quantity))
                elif q.bid_price >= 0.03:
                    fill(opt, q.bid_price, random.randint(1, q.bid_quantity))
        for _ in range(random.randint(0, 3)):
            opt = random.choice(active)
            fok = T.FokOrder(counterparty_id=9, option_id=opt.option_id, order_type=random.choice(list(T.OrderType)), price=round(random.uniform(0.01, 0.99), 2), quantity=random.randint(1, 30))
            if mm.respond_to_fok(opt, fok):
                fill(opt, fok.price, fok.quantity if fok.order_type == T.OrderType.SELL else -fok.quantity)
        values = TRUE_PARAMS.advance_step(values)
        active = [o.advance_step() for o in active]
        expired = [o for o in active if o.steps_until_expiry == 0]
        active = [o for o in active if o.steps_until_expiry > 0]
        for o in expired:
            payoff = o.expiry_valuation(values)
            long_qty, short_qty = positions.pop(o.option_id, (0, 0))
            grader_cash += long_qty * payoff + short_qty * (1.0 - payoff)
        assert grader_cash >= -1e-9, f"{tag} BANKRUPT day {day}: {grader_cash}"
        mm.on_step_advance(underlyings(values), expired + active)
        if tag != "v1":
            assert abs(mm._tracked_cash - grader_cash) < 1e-6, (tag, day, mm._tracked_cash, grader_cash)
    print(f"  OK {tag}: 50-day session solvent, grader cash {grader_cash:.2f} (start {cash0})")


if __name__ == "__main__":
    print("pricing accuracy (THEO vs Monte Carlo):")
    test_pricing()
    print("quotes / estimation:")
    test_quotes_and_estimation()
    print("solvency sessions:")
    for tag, mod in MODS.items():
        test_solvency_session(tag, mod)
    print("ALL VERSION TESTS PASSED")
