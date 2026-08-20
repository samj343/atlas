"""Head-to-head tournament: all four versions in one simulated exchange.

Mechanics mirror the task description: RFQs are quoted by every MM and routed
to the best price (splitting down the book), FOKs go to every MM and are split
among acceptors, the grader deducts max loss per trade and credits expiry
payoffs at day end, and a negative balance at day end is bankruptcy.
Counterparties: noise traders (random side, wide reservation prices) and
informed traders (know the true parameters, trade only with edge, stable ids).
"""
import random
import statistics
import sys

SEED_BASE = 2000

from common import load_all

MODS = load_all()
T = MODS["v4"]
FED, AJR, THR = T.FED_FUNDS_RATE_UNDERLYING_ID, T.AJARAI_UNDERLYING_ID, T.THERIODIC_UNDERLYING_ID

NOISE_IDS = [1, 2, 3]
INFORMED_IDS = [11, 12]


def underlyings(v):
    return [T.Underlying("FED", FED, v[FED]), T.Underlying("AJR", AJR, v[AJR]), T.Underlying("THR", THR, v[THR])]


def random_true_params(rng):
    up = rng.uniform(0.05, 0.30)
    down = rng.uniform(0.05, min(0.30, 0.9 - up))
    return T.MarketParameters(
        ajarai_drift=rng.gauss(0.0, 0.002),
        ajarai_idio_std_dev=rng.uniform(0.01, 0.04),
        ajarai_rate_beta=rng.uniform(-0.4, 0.1),
        ajarai_sector_beta=rng.uniform(0.5, 1.5),
        rate_down_probability=down,
        rate_reversion_strength=rng.uniform(0.0, 0.15),
        rate_up_probability=up,
        sector_std_dev=rng.uniform(0.005, 0.02),
        theriodic_drift=rng.gauss(0.0, 0.002),
        theriodic_idio_std_dev=rng.uniform(0.01, 0.04),
        theriodic_rate_beta=rng.uniform(-0.4, 0.1),
        theriodic_sector_beta=rng.uniform(0.5, 1.5),
    )


class Sim:
    def __init__(self, seed, days=50, cash0=300.0, warmup_days=350, informed_share=0.25, tags=None):
        self.tags = list(MODS) if tags is None else tags
        self.rng = random.Random(seed)
        self.days = days
        self.cash0 = cash0
        self.informed_share = informed_share
        self.params = random_true_params(self.rng)
        fed0 = self.rng.choice([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
        self.values = {FED: fed0, AJR: round(self.rng.uniform(500, 1500), 2), THR: round(self.rng.uniform(500, 1500), 2)}
        # burn-in history (uses global `random`, seeded for reproducibility)
        random.seed(seed * 7 + 1)
        series = {k: [v] for k, v in self.values.items()}
        for _ in range(warmup_days - 1):
            self.values = self.params.advance_step(self.values)
            for k in series:
                series[k].append(self.values[k])
        self.history = T.MarketHistory(values_by_underlying_id={k: tuple(v) for k, v in series.items()})

        self.mms = {}
        self.grader_cash = {}
        self.positions = {}  # tag -> {option_id: qty}
        self.bankrupt = {}
        for tag in self.tags:
            mm = MODS[tag].MarketMaker(underlyings(self.values), [], cash0)
            mm.warm_up(self.history)
            self.mms[tag] = mm
            self.grader_cash[tag] = cash0
            self.positions[tag] = {}
            self.bankrupt[tag] = None
        self.active = []
        self.next_id = 100
        self.pricer_mm = MODS["v4"].MarketMaker(underlyings(self.values), [], cash0)  # reference for true fair

    def solvent_tags(self):
        return [t for t in self.mms if self.bankrupt[t] is None]

    def true_fair(self, opt):
        self.pricer_mm.underlying_state = underlyings(self.values)
        return self.pricer_mm.price_option_from_parameters(self.params, opt)

    def new_option(self):
        self.next_id += 1
        r = self.rng.random()
        days_left = self.rng.randint(1, 7)
        if r < 0.4:
            k = max(round(self.values[FED] + self.rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]), 2), 0.0)
            legs, strike = (T.OptionLeg(FED, 1.0),), k
        elif r < 0.8:
            uid = self.rng.choice([AJR, THR])
            legs, strike = (T.OptionLeg(uid, 1.0),), round(self.values[uid] * self.rng.uniform(0.95, 1.05), 2)
        else:
            if self.rng.random() < 0.5:
                legs = (T.OptionLeg(AJR, 1.0), T.OptionLeg(THR, -1.0))
            else:
                legs = (T.OptionLeg(THR, 1.0), T.OptionLeg(AJR, -1.0))
            strike = 0.0
        return T.BinaryOption(legs=legs, option_id=self.next_id, steps_until_expiry=days_left, strike=strike)

    def fill(self, tag, opt, px, signed_qty, cpty):
        self.mms[tag].on_trade(opt, px, signed_qty, cpty)
        loss = signed_qty * px if signed_qty > 0 else (-signed_qty) * (1.0 - px)
        self.grader_cash[tag] -= loss
        entry = self.positions[tag].setdefault(opt.option_id, [0, 0])  # [gross long, gross short]
        if signed_qty > 0:
            entry[0] += signed_qty
        else:
            entry[1] += -signed_qty

    def rfq(self):
        if not self.active:
            return
        opt = self.rng.choice(self.active)
        informed = self.rng.random() < self.informed_share
        cpty = self.rng.choice(INFORMED_IDS if informed else NOISE_IDS)
        fair = self.true_fair(opt)
        quotes = []
        for tag in self.solvent_tags():
            try:
                quotes.append((tag, self.mms[tag].quote(opt, cpty)))
            except Exception:
                pass
        if not quotes:
            return
        if informed:
            best_offer = min(q.offer_price for _, q in quotes)
            best_bid = max(q.bid_price for _, q in quotes)
            if fair - best_offer >= 0.02:
                side, qty, limit = "buy", self.rng.randint(10, 40), fair - 0.02
            elif best_bid - fair >= 0.02:
                side, qty, limit = "sell", self.rng.randint(10, 40), fair + 0.02
            else:
                return
        else:
            side = self.rng.choice(["buy", "sell"])
            qty = self.rng.randint(5, 35)
            # noise reservation: usually a few cents past fair, occasionally more
            slack = min(0.03 + abs(self.rng.gauss(0.0, 0.04)), 0.15)
            limit = min(fair + slack, 0.98) if side == "buy" else max(fair - slack, 0.02)
        remaining = qty
        if side == "buy":
            book = sorted(quotes, key=lambda t: (t[1].offer_price, self.rng.random()))
            for tag, q in book:
                if remaining <= 0 or q.offer_price > limit:
                    break
                take = min(remaining, q.offer_quantity)
                self.fill(tag, opt, q.offer_price, -take, cpty)
                remaining -= take
        else:
            book = sorted(quotes, key=lambda t: (-t[1].bid_price, self.rng.random()))
            for tag, q in book:
                if remaining <= 0 or q.bid_price < limit:
                    break
                take = min(remaining, q.bid_quantity)
                self.fill(tag, opt, q.bid_price, take, cpty)
                remaining -= take

    def fok(self):
        if not self.active:
            return
        opt = self.rng.choice(self.active)
        informed = self.rng.random() < min(self.informed_share + 0.2, 0.6)
        cpty = self.rng.choice(INFORMED_IDS if informed else NOISE_IDS)
        fair = self.true_fair(opt)
        order_type = self.rng.choice([T.OrderType.BUY, T.OrderType.SELL])
        if informed:
            # priced so that any MM accepting is trading at negative true edge
            slip = self.rng.uniform(0.0, 0.02)
            px = fair - slip if order_type == T.OrderType.BUY else fair + slip
        else:
            px = fair + self.rng.uniform(-0.08, 0.08)
        px = min(max(round(px, 2), 0.01), 0.99)
        qty = self.rng.randint(5, 30)
        fok = T.FokOrder(counterparty_id=cpty, option_id=opt.option_id, order_type=order_type, price=px, quantity=qty)
        acceptors = []
        for tag in self.solvent_tags():
            try:
                if self.mms[tag].respond_to_fok(opt, fok):
                    acceptors.append(tag)
            except Exception:
                pass
        if not acceptors:
            return
        share, rem = divmod(qty, len(acceptors))
        for i, tag in enumerate(acceptors):
            take = share + (1 if i < rem else 0)
            if take == 0:
                continue
            signed = take if order_type == T.OrderType.SELL else -take
            self.fill(tag, opt, px, signed, cpty)

    def advance_day(self, with_flow=True):
        if with_flow:
            for _ in range(self.rng.randint(2, 4)):
                self.active.append(self.new_option())
            for _ in range(self.rng.randint(6, 10)):
                self.rfq()
            for _ in range(self.rng.randint(1, 3)):
                self.fok()
        self.values = self.params.advance_step(self.values)
        self.active = [o.advance_step() for o in self.active]
        expired = [o for o in self.active if o.steps_until_expiry == 0]
        self.active = [o for o in self.active if o.steps_until_expiry > 0]
        for tag in self.mms:
            for o in expired:
                long_qty, short_qty = self.positions[tag].pop(o.option_id, (0, 0))
                if long_qty or short_qty:
                    payoff = o.expiry_valuation(self.values)
                    self.grader_cash[tag] += long_qty * payoff + short_qty * (1.0 - payoff)
        for tag in self.solvent_tags():
            if self.grader_cash[tag] < -1e-9:
                self.bankrupt[tag] = True
        state = underlyings(self.values)
        for tag in self.solvent_tags():
            try:
                self.mms[tag].on_step_advance(state, expired + self.active)
            except Exception:
                pass

    def run(self):
        for _ in range(self.days):
            self.advance_day(with_flow=True)
        guard = 0
        while self.active and guard < 20:
            self.advance_day(with_flow=False)
            guard += 1
        return {tag: (None if self.bankrupt[tag] else self.grader_cash[tag] - self.cash0) for tag in self.mms}


def main(num_seeds_per_regime=10):
    regimes = {"benign(10% informed)": 0.10, "mixed(25%)": 0.25, "toxic(45%)": 0.45}
    overall = {tag: [] for tag in MODS}
    for regime_name, share in regimes.items():
        results = {tag: [] for tag in MODS}
        wins = {tag: 0 for tag in MODS}
        bankrupt = {tag: 0 for tag in MODS}
        for seed in range(num_seeds_per_regime):
            outcome = Sim(seed=SEED_BASE + seed, informed_share=share).run()
            best = max((p for p in outcome.values() if p is not None), default=None)
            for tag, pnl in outcome.items():
                if pnl is None:
                    bankrupt[tag] += 1
                else:
                    results[tag].append(pnl)
                    overall[tag].append(pnl)
                    if pnl == best:
                        wins[tag] += 1
        print(f"--- {regime_name} ---")
        for tag in MODS:
            r = results[tag]
            print(f"  {tag}: mean {statistics.mean(r):+8.2f}  median {statistics.median(r):+8.2f}  min {min(r):+8.2f}  wins {wins[tag]}  bankrupt {bankrupt[tag]}")
    print("=== overall (all regimes) ===")
    for tag in MODS:
        r = overall[tag]
        print(f"  {tag}: mean {statistics.mean(r):+8.2f}  median {statistics.median(r):+8.2f}  min {min(r):+8.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        SEED_BASE = int(sys.argv[2])
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
