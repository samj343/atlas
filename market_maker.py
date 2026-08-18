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
# Pricing helpers
# ============================================================================


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _build_std_nodes(num_points: int = 61, spread: float = 6.0) -> tuple[tuple[float, float], ...]:
    """Discretized standard normal: (z, weight) nodes with weights summing to 1."""
    nodes: list[tuple[float, float]] = []
    for i in range(num_points):
        z = -spread + 2.0 * spread * i / (num_points - 1)
        nodes.append((z, math.exp(-0.5 * z * z)))
    total = sum(w for _, w in nodes)
    return tuple((z, w / total) for z, w in nodes)


_STD_NODES: Final[tuple[tuple[float, float], ...]] = _build_std_nodes()

_EPS: Final[float] = 1e-9


def _terminal_rate_distribution(params: MarketParameters, rate0: float, num_steps: int) -> dict[float, float]:
    """Exact distribution of the FED rate after num_steps daily moves."""
    dist: dict[float, float] = {round(rate0, 2): 1.0}
    for _ in range(num_steps):
        nxt: dict[float, float] = defaultdict(float)
        for rate, prob in dist.items():
            up, down = params.tilted_rate_probabilities(rate)
            stay = max(1.0 - up - down, 0.0)
            if up > 0.0:
                nxt[params.next_rate_value(rate, 1)] += prob * up
            if down > 0.0:
                nxt[params.next_rate_value(rate, -1)] += prob * down
            if stay > 0.0:
                nxt[rate] += prob * stay
        dist = dict(nxt)
    return dist


def _company_leg_params(params: MarketParameters, underlying_id: int) -> tuple[float, float, float, float]:
    """(drift, rate_beta, sector_beta, idio_std_dev) for a company underlying."""
    if underlying_id == AJARAI_UNDERLYING_ID:
        return (params.ajarai_drift, params.ajarai_rate_beta, params.ajarai_sector_beta, params.ajarai_idio_std_dev)
    if underlying_id == THERIODIC_UNDERLYING_ID:
        return (
            params.theriodic_drift,
            params.theriodic_rate_beta,
            params.theriodic_sector_beta,
            params.theriodic_idio_std_dev,
        )
    raise ValueError(f"Unknown company underlying id: {underlying_id}")


def _lognormal_ge_prob(weight: float, spot: float, mu: float, var: float, threshold: float) -> float:
    """P(weight * spot * exp(X) >= threshold) for X ~ N(mu, var), spot > 0."""
    if var <= 1e-18:
        return 1.0 if weight * spot * math.exp(mu) >= threshold - _EPS else 0.0
    sd = math.sqrt(var)
    if weight > 0.0:
        if threshold <= 0.0:
            return 1.0
        z = (math.log(threshold / (weight * spot)) - mu) / sd
        return 1.0 - _phi(z)
    # weight < 0: event is exp(X) <= threshold / (weight * spot)
    if threshold >= 0.0:
        return 0.0
    bound = threshold / (weight * spot)
    z = (math.log(bound) - mu) / sd
    return _phi(z)


def _two_company_prob(
    params: MarketParameters,
    values: dict[int, float],
    legs: "tuple[OptionLeg, ...]",
    num_steps: int,
    rate_change: float,
    strike: float,
) -> float:
    """P(w_a*A_n + w_b*B_n >= strike) for two company legs, conditional on the total rate change."""
    leg_a, leg_b = legs
    drift_a, rbeta_a, sbeta_a, istd_a = _company_leg_params(params, leg_a.underlying_id)
    drift_b, rbeta_b, sbeta_b, istd_b = _company_leg_params(params, leg_b.underlying_id)
    spot_a = values[leg_a.underlying_id]
    spot_b = values[leg_b.underlying_id]
    mu_a = num_steps * drift_a + rbeta_a * rate_change
    mu_b = num_steps * drift_b + rbeta_b * rate_change
    sector_var = num_steps * params.sector_std_dev**2
    var_a = sbeta_a**2 * sector_var + num_steps * istd_a**2
    var_b = sbeta_b**2 * sector_var + num_steps * istd_b**2
    cov = sbeta_a * sbeta_b * sector_var
    w_a, w_b = leg_a.weight, leg_b.weight

    if abs(strike) <= _EPS and w_a * w_b < 0.0:
        # Reduces to a ratio of lognormals: P(w_pos * S_pos >= |w_neg| * S_neg).
        if w_a > 0.0:
            w_pos, spot_pos, mu_pos, var_pos = w_a, spot_a, mu_a, var_a
            w_neg, spot_neg, mu_neg, var_neg = w_b, spot_b, mu_b, var_b
        else:
            w_pos, spot_pos, mu_pos, var_pos = w_b, spot_b, mu_b, var_b
            w_neg, spot_neg, mu_neg, var_neg = w_a, spot_a, mu_a, var_a
        mu_diff = math.log((w_pos * spot_pos) / ((-w_neg) * spot_neg)) + mu_pos - mu_neg
        var_diff = var_pos + var_neg - 2.0 * cov
        if var_diff <= 1e-18:
            return 1.0 if mu_diff >= -_EPS else 0.0
        return 1.0 - _phi(-mu_diff / math.sqrt(var_diff))

    # General case: integrate over the shared sector shock (making the legs
    # independent), then over leg b, with the leg a tail in closed form.
    sector_sd = math.sqrt(sector_var)
    if sector_sd <= 1e-12 or (abs(sbeta_a) <= 1e-12 and abs(sbeta_b) <= 1e-12):
        outer: tuple[tuple[float, float], ...] = ((0.0, 1.0),)
    else:
        outer = tuple((z * sector_sd, w) for z, w in _STD_NODES)
    inner_sd = math.sqrt(num_steps) * istd_b
    inner = ((0.0, 1.0),) if inner_sd <= 1e-12 else _STD_NODES
    ivar_a = num_steps * istd_a**2
    total = 0.0
    for sector_val, sector_wt in outer:
        mu_a2 = mu_a + sbeta_a * sector_val
        mu_b2 = mu_b + sbeta_b * sector_val
        for z, zw in inner:
            log_ret_b = mu_b2 + z * inner_sd
            residual_strike = strike - w_b * spot_b * math.exp(log_ret_b)
            total += sector_wt * zw * _lognormal_ge_prob(w_a, spot_a, mu_a2, ivar_a, residual_strike)
    return total


def _price_binary_option(params: MarketParameters, values: dict[int, float], option: BinaryOption) -> float:
    """Probability the option expires in the money under `params` from current `values`."""
    num_steps = option.steps_until_expiry
    if num_steps == 0:
        return option.expiry_valuation(values)

    fed_weight = 0.0
    has_fed = False
    company_legs: list[OptionLeg] = []
    for leg in option.legs:
        if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
            has_fed = True
            fed_weight = leg.weight
        else:
            company_legs.append(leg)

    rate0 = values[FED_FUNDS_RATE_UNDERLYING_ID]
    rate_dist = _terminal_rate_distribution(params, rate0, num_steps)

    total = 0.0
    for rate, prob in rate_dist.items():
        if prob <= 0.0:
            continue
        rate_change = rate - rate0
        strike = option.strike - (fed_weight * rate if has_fed else 0.0)
        if not company_legs:
            total += prob * (1.0 if strike <= _EPS else 0.0)
        elif len(company_legs) == 1:
            leg = company_legs[0]
            drift, rbeta, sbeta, istd = _company_leg_params(params, leg.underlying_id)
            spot = values[leg.underlying_id]
            mu = num_steps * drift + rbeta * rate_change
            var = num_steps * ((sbeta * params.sector_std_dev) ** 2 + istd**2)
            total += prob * _lognormal_ge_prob(leg.weight, spot, mu, var, strike)
        else:
            total += prob * _two_company_prob(params, values, tuple(company_legs), num_steps, rate_change, strike)
    return min(max(total, 0.0), 1.0)


# ============================================================================
# Parameter estimation helpers
# ============================================================================


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, list[float]]:
    """Least-squares fit y ~ intercept + slope * x; returns (intercept, slope, residuals)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        slope = 0.0
    else:
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    return intercept, slope, residuals


def _estimate_market_parameters(market_history: MarketHistory) -> MarketParameters:
    values = market_history.values_by_underlying_id
    rates = list(values[FED_FUNDS_RATE_UNDERLYING_ID])
    ajr = list(values[AJARAI_UNDERLYING_ID])
    thr = list(values[THERIODIC_UNDERLYING_ID])
    num_transitions = len(rates) - 1
    if num_transitions < 3:
        raise ValueError("Not enough history to estimate parameters")

    rate_changes = [round(rates[i + 1] - rates[i], 2) for i in range(num_transitions)]

    # --- Rate chain: p_up(r) = a - kappa*r, p_down(r) = b + kappa*r (unclamped model).
    step_sizes = sorted({round(abs(chg), 2) for chg in rate_changes if abs(chg) > _EPS})
    rate_step = step_sizes[0] if step_sizes else 0.25

    counts_by_rate: dict[float, list[int]] = {}
    for i in range(num_transitions):
        rate_key = round(rates[i], 2)
        counts = counts_by_rate.setdefault(rate_key, [0, 0, 0])
        counts[0] += 1
        if rate_changes[i] > _EPS:
            counts[1] += 1
        elif rate_changes[i] < -_EPS:
            counts[2] += 1

    total_n = sum(c[0] for c in counts_by_rate.values())
    mean_rate = sum(r * c[0] for r, c in counts_by_rate.items()) / total_n
    mean_up = sum(c[1] for c in counts_by_rate.values()) / total_n
    mean_down = sum(c[2] for c in counts_by_rate.values()) / total_n
    sxx = sum(c[0] * (r - mean_rate) ** 2 for r, c in counts_by_rate.items())
    if sxx > 1e-12:
        # Weighted least squares for the common reversion slope kappa.
        sxy = sum(
            c[0] * (r - mean_rate) * ((c[2] / c[0]) - (c[1] / c[0])) for r, c in counts_by_rate.items()
        )
        kappa = sxy / (2.0 * sxx)
    else:
        kappa = 0.0
    kappa = min(max(kappa, 0.0), 1.0)
    rate_target = 2.0
    up_prob = (mean_up + kappa * mean_rate) - kappa * rate_target
    down_prob = (mean_down - kappa * mean_rate) + kappa * rate_target
    up_prob = min(max(up_prob, 1e-3), 0.95)
    down_prob = min(max(down_prob, 1e-3), 0.95)
    if up_prob + down_prob > 0.995:
        scale = 0.995 / (up_prob + down_prob)
        up_prob *= scale
        down_prob *= scale

    # --- Company log-returns regressed on rate changes.
    tiny = 1e-12
    ajr_returns = [math.log(max(ajr[i + 1], tiny) / max(ajr[i], tiny)) for i in range(num_transitions)]
    thr_returns = [math.log(max(thr[i + 1], tiny) / max(thr[i], tiny)) for i in range(num_transitions)]
    ajr_drift, ajr_rate_beta, ajr_resid = _ols(rate_changes, ajr_returns)
    thr_drift, thr_rate_beta, thr_resid = _ols(rate_changes, thr_returns)
    dof = max(num_transitions - 2, 1)
    ajr_var = max(sum(e * e for e in ajr_resid) / dof, 1e-10)
    thr_var = max(sum(e * e for e in thr_resid) / dof, 1e-10)
    resid_cov = sum(ea * et for ea, et in zip(ajr_resid, thr_resid)) / dof

    # Only (variance_a, variance_b, covariance) are identified, so pick the
    # decomposition sector_std_dev = 1, betas = sqrt(|cov|).
    common_var = min(abs(resid_cov), 0.999 * ajr_var, 0.999 * thr_var)
    sector_beta = math.sqrt(common_var)
    ajr_sector_beta = sector_beta
    thr_sector_beta = math.copysign(sector_beta, resid_cov) if resid_cov != 0.0 else 0.0
    ajr_idio = math.sqrt(max(ajr_var - common_var, 1e-12))
    thr_idio = math.sqrt(max(thr_var - common_var, 1e-12))

    return MarketParameters(
        ajarai_drift=ajr_drift,
        ajarai_idio_std_dev=ajr_idio,
        ajarai_rate_beta=ajr_rate_beta,
        ajarai_sector_beta=ajr_sector_beta,
        rate_down_probability=down_prob,
        rate_reversion_strength=kappa,
        rate_up_probability=up_prob,
        sector_std_dev=1.0,
        theriodic_drift=thr_drift,
        theriodic_idio_std_dev=thr_idio,
        theriodic_rate_beta=thr_rate_beta,
        theriodic_sector_beta=thr_sector_beta,
        rate_step=rate_step,
        rate_target=rate_target,
    )


# ============================================================================
# YOUR MARKET MAKER
# ============================================================================


class MarketMaker:
    RISK_FRACTION: Final[float] = 0.08
    FOK_RISK_FRACTION: Final[float] = 0.30
    MAX_QUOTE_QUANTITY: Final[int] = 200

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

        # Worst-case cash tracking mirrors the autograder: every trade reserves
        # its maximum loss immediately; expiries can only credit cash back.
        self._tracked_cash: float = cash_balance
        self._estimated_parameters: MarketParameters | None = None
        self._fallback_params: MarketParameters | None = None
        self._price_cache: dict[Any, float] = {}
        self._known_options: dict[int, BinaryOption] = {o.option_id: o for o in option_initial_state}
        self._settled_option_ids: set[int] = set()

    def on_step_advance(self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]) -> None:
        new_values = {u.underlying_id: u.value for u in new_underlying_state}
        new_ids = {o.option_id for o in new_option_state}
        for option in new_option_state:
            if option.steps_until_expiry == 0:
                self._settle_option(option, new_values)
        for option in list(self._known_options.values()):
            if option.option_id not in new_ids:
                self._settle_option(option, new_values)

        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state
        self._known_options = {o.option_id: o for o in new_option_state}
        self._price_cache.clear()

    def on_trade(self, option: BinaryOption, price: float, quantity: int, counterparty_id: int) -> None:
        self.position.add_option_quantity(option.option_id, quantity)
        self._known_options[option.option_id] = option
        if quantity >= 0:
            max_loss = quantity * price
        else:
            max_loss = (-quantity) * (1.0 - price)
        self._tracked_cash -= max_loss
        self.cash_balance = self._tracked_cash

    def _settle_option(self, option: BinaryOption, value_by_underlying_id: dict[int, float]) -> None:
        option_id = option.option_id
        if option_id in self._settled_option_ids:
            return
        self._settled_option_ids.add(option_id)
        quantity = self.position.option_quantity_by_option_id.get(option_id, 0)
        if quantity == 0:
            return
        payoff = option.expiry_valuation(value_by_underlying_id)
        if quantity > 0:
            credit = quantity * payoff
        else:
            credit = (-quantity) * (1.0 - payoff)
        self._tracked_cash += credit
        self.cash_balance = self._tracked_cash
        self.position.option_quantity_by_option_id.pop(option_id, None)

    def _fallback_parameters(self) -> MarketParameters:
        if self._fallback_params is None:
            self._fallback_params = MarketParameters(
                ajarai_drift=0.0,
                ajarai_idio_std_dev=0.02,
                ajarai_rate_beta=0.0,
                ajarai_sector_beta=0.0,
                rate_down_probability=0.1,
                rate_reversion_strength=0.1,
                rate_up_probability=0.1,
                sector_std_dev=1.0,
                theriodic_drift=0.0,
                theriodic_idio_std_dev=0.02,
                theriodic_rate_beta=0.0,
                theriodic_sector_beta=0.0,
            )
        return self._fallback_params

    @property
    def name(self) -> str:
        return "AtlasMM"

    def price_option(self, option: BinaryOption) -> float:
        params = self._estimated_parameters
        if params is None:
            params = self._fallback_parameters()
        return self.price_option_from_parameters(params, option)

    def price_option_from_parameters(self, market_parameters: MarketParameters, option: BinaryOption) -> float:
        values = {u.underlying_id: u.value for u in self.underlying_state}
        cache_key = (
            market_parameters,
            option.legs,
            option.steps_until_expiry,
            option.strike,
            tuple(sorted(values.items())),
        )
        cached = self._price_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            price = _price_binary_option(market_parameters, values, option)
        except Exception:
            return 0.5
        if len(self._price_cache) > 20000:
            self._price_cache.clear()
        self._price_cache[cache_key] = price
        return price

    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:
        try:
            return self._make_quote(option)
        except Exception:
            return Quote(bid_price=0.0, bid_quantity=1, offer_price=1.0, offer_quantity=1)

    def _make_quote(self, option: BinaryOption) -> Quote:
        cash = self._tracked_cash
        if cash < 0.25:
            # Too poor to take on any risk: quote only riskless prices.
            return Quote(bid_price=0.0, bid_quantity=1, offer_price=1.0, offer_quantity=1)

        fair = self.price_option(option)
        net_position = self.position.option_quantity_by_option_id.get(option.option_id, 0)
        skew = min(max(-0.002 * net_position, -0.04), 0.04)
        center = min(max(fair + skew, 0.005), 0.995)
        half_spread = 0.01 + 0.04 * fair * (1.0 - fair)

        bid = math.floor((center - half_spread) * 100 + 1e-9) / 100.0
        offer = math.ceil((center + half_spread) * 100 - 1e-9) / 100.0
        bid = min(max(bid, 0.0), 0.99)
        offer = min(max(offer, 0.01), 1.0)
        if bid >= offer:
            bid = max(0.0, round(offer - 0.01, 2))

        # Keep either side's worst-case loss within a fraction of remaining cash,
        # so no sequence of fills can ever take the tracked balance negative.
        budget = self.RISK_FRACTION * cash
        if bid > budget:
            bid = max(math.floor(budget * 100) / 100.0, 0.0)
            if bid >= offer:
                bid = max(0.0, round(offer - 0.01, 2))
        offer_loss = 1.0 - offer
        if offer_loss > budget:
            offer = min(math.ceil((1.0 - budget) * 100) / 100.0, 1.0)
            if offer <= bid:
                offer = min(1.0, round(bid + 0.01, 2))
            offer_loss = 1.0 - offer

        bid_quantity = int(budget / bid) if bid > 0.004 else self.MAX_QUOTE_QUANTITY
        offer_quantity = int(budget / offer_loss) if offer_loss > 0.004 else self.MAX_QUOTE_QUANTITY
        bid_quantity = max(1, min(bid_quantity, self.MAX_QUOTE_QUANTITY))
        offer_quantity = max(1, min(offer_quantity, self.MAX_QUOTE_QUANTITY))
        return Quote(
            bid_price=bid,
            bid_quantity=bid_quantity,
            offer_price=offer,
            offer_quantity=offer_quantity,
        )

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:
        try:
            fair = self.price_option(option)
            required_edge = 0.02 + 0.02 * fair * (1.0 - fair)
            if fok_order.order_type == OrderType.BUY:
                # Counterparty buys: we would sell at the order price.
                max_loss = fok_order.quantity * (1.0 - fok_order.price)
                if max_loss > self.FOK_RISK_FRACTION * self._tracked_cash:
                    return False
                return fok_order.price >= fair + required_edge
            # Counterparty sells: we would buy at the order price.
            max_loss = fok_order.quantity * fok_order.price
            if max_loss > self.FOK_RISK_FRACTION * self._tracked_cash:
                return False
            return fok_order.price <= fair - required_edge
        except Exception:
            return False

    def warm_up(self, market_history: MarketHistory) -> None:
        try:
            self._estimated_parameters = _estimate_market_parameters(market_history)
        except Exception:
            self._estimated_parameters = self._fallback_parameters()
        self._price_cache.clear()
