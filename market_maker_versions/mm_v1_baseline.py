"""Market maker evolution 1/4 -- "baseline".

Crude normal-approximation pricing (independent legs, no reversion), fixed
5-cent spread, fixed size, never trades FOKs, one-way budget that ignores
expiry credits. Safe but weak: the starting point.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

# ============================================================================
# YOUR MARKET MAKER -- v1 "baseline"
#
# First working cut. Prices every option with a one-shot normal approximation:
# legs are treated as independent (spreads ignore the sector correlation), the
# rate walk ignores mean reversion and the zero floor, and company values are
# moment-matched lognormals. Quoting is a fixed 5-cent half spread at fixed
# size, FOKs are never traded, and the cash check is a one-way ratchet (max
# loss subtracted, expiry credits ignored) -- solvent but self-strangling.
# ============================================================================


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class MarketMaker:
    HALF_SPREAD: Final[float] = 0.05
    QUOTE_SIZE: Final[int] = 10

    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.position: Position = Position()

        # One-way budget: max loss is subtracted on every trade but expiry
        # payoffs are never credited back, so the budget only shrinks.
        self._budget_left: float = cash_balance
        self._rate_up_freq: float = 0.10
        self._rate_down_freq: float = 0.10
        self._rate_step: float = 0.25
        self._drift_by_id: dict[int, float] = {AJARAI_UNDERLYING_ID: 0.0, THERIODIC_UNDERLYING_ID: 0.0}
        self._vol_by_id: dict[int, float] = {AJARAI_UNDERLYING_ID: 0.03, THERIODIC_UNDERLYING_ID: 0.03}

    def on_step_advance(self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]) -> None:
        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state

    def on_trade(self, option: BinaryOption, price: float, quantity: int, counterparty_id: int) -> None:
        self.position.add_option_quantity(option.option_id, quantity)
        max_loss = quantity * price if quantity >= 0 else (-quantity) * (1.0 - price)
        self._budget_left -= max_loss

    @property
    def name(self) -> str:
        return "AtlasMM-v1"

    def _normal_approx_price(
        self,
        option: BinaryOption,
        up_freq: float,
        down_freq: float,
        rate_step: float,
        drift_by_id: dict[int, float],
        vol_by_id: dict[int, float],
    ) -> float:
        values = {u.underlying_id: u.value for u in self.underlying_state}
        num_steps = option.steps_until_expiry
        if num_steps == 0:
            return option.expiry_valuation(values)
        mean = 0.0
        var = 0.0
        for leg in option.legs:
            spot = values[leg.underlying_id]
            if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                step_mean = rate_step * (up_freq - down_freq)
                leg_mean = spot + num_steps * step_mean
                leg_var = num_steps * (rate_step**2 * (up_freq + down_freq) - step_mean**2)
            else:
                mu = num_steps * drift_by_id[leg.underlying_id]
                s2 = num_steps * vol_by_id[leg.underlying_id] ** 2
                leg_mean = spot * math.exp(mu + 0.5 * s2)
                leg_var = spot**2 * math.exp(2.0 * mu + s2) * (math.exp(s2) - 1.0)
            mean += leg.weight * leg_mean
            var += leg.weight**2 * leg_var  # pretends the legs are independent
        if var <= 1e-18:
            return 1.0 if mean >= option.strike - 1e-9 else 0.0
        return 1.0 - _phi((option.strike - mean) / math.sqrt(var))

    def price_option(self, option: BinaryOption) -> float:
        return self._normal_approx_price(
            option, self._rate_up_freq, self._rate_down_freq, self._rate_step, self._drift_by_id, self._vol_by_id
        )

    def price_option_from_parameters(self, market_parameters: MarketParameters, option: BinaryOption) -> float:
        vol_by_id = {
            AJARAI_UNDERLYING_ID: math.hypot(
                market_parameters.ajarai_sector_beta * market_parameters.sector_std_dev,
                market_parameters.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: math.hypot(
                market_parameters.theriodic_sector_beta * market_parameters.sector_std_dev,
                market_parameters.theriodic_idio_std_dev,
            ),
        }
        drift_by_id = {
            AJARAI_UNDERLYING_ID: market_parameters.ajarai_drift,
            THERIODIC_UNDERLYING_ID: market_parameters.theriodic_drift,
        }
        return self._normal_approx_price(
            option,
            market_parameters.rate_up_probability,  # ignores the reversion tilt
            market_parameters.rate_down_probability,
            market_parameters.rate_step,
            drift_by_id,
            vol_by_id,
        )

    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:
        try:
            return self._make_quote(option)
        except Exception:
            return Quote(bid_price=0.0, bid_quantity=1, offer_price=1.0, offer_quantity=1)

    def _make_quote(self, option: BinaryOption) -> Quote:
        fair = min(max(self.price_option(option), 0.0), 1.0)
        bid = math.floor((fair - self.HALF_SPREAD) * 100 + 1e-9) / 100.0
        offer = math.ceil((fair + self.HALF_SPREAD) * 100 - 1e-9) / 100.0
        bid = min(max(bid, 0.0), 0.99)
        offer = min(max(offer, 0.01), 1.0)
        if bid >= offer:
            bid = max(0.0, round(offer - 0.01, 2))
        budget = 0.5 * self._budget_left
        bid_quantity = self.QUOTE_SIZE
        offer_quantity = self.QUOTE_SIZE
        if bid > 0 and bid * bid_quantity > budget:
            bid_quantity = int(budget / bid)
            if bid_quantity < 1:
                bid, bid_quantity = 0.0, 1  # riskless bid
        offer_loss = 1.0 - offer
        if offer_loss * offer_quantity > budget:
            offer_quantity = int(budget / offer_loss) if offer_loss > 0 else self.QUOTE_SIZE
            if offer_quantity < 1:
                offer, offer_quantity = 1.0, 1  # riskless offer
        if bid >= offer:
            bid = max(0.0, round(offer - 0.01, 2))
        return Quote(bid_price=bid, bid_quantity=bid_quantity, offer_price=offer, offer_quantity=offer_quantity)

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:
        return False  # v1 never trades FOKs

    def warm_up(self, market_history: MarketHistory) -> None:
        try:
            values = market_history.values_by_underlying_id
            rates = values[FED_FUNDS_RATE_UNDERLYING_ID]
            num_transitions = len(rates) - 1
            if num_transitions <= 0:
                return
            changes = [round(rates[i + 1] - rates[i], 2) for i in range(num_transitions)]
            self._rate_up_freq = max(sum(1 for c in changes if c > 1e-9) / num_transitions, 1e-3)
            self._rate_down_freq = max(sum(1 for c in changes if c < -1e-9) / num_transitions, 1e-3)
            nonzero = sorted({abs(c) for c in changes if abs(c) > 1e-9})
            self._rate_step = nonzero[0] if nonzero else 0.25
            for underlying_id in (AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID):
                series = values[underlying_id]
                returns = [math.log(series[i + 1] / series[i]) for i in range(num_transitions)]
                mean_return = sum(returns) / num_transitions
                var_return = sum((r - mean_return) ** 2 for r in returns) / max(num_transitions - 1, 1)
                self._drift_by_id[underlying_id] = mean_return
                self._vol_by_id[underlying_id] = math.sqrt(var_return)
        except Exception:
            pass
