"""The heuristic bidding policies of the provider agent, and the decline rule.

Called by ``BlockchainHandler._heuristic_reasoning``. Everything here is a pure
function of

    (AuctionView, AgentEconomics, PolicyState, PolicyParams, rng)

except that ``PolicyState`` is mutated in place: it is the controller's memory
across auctions (margin, prev_wins, bids_placed). No blockchain access, no I/O,
and every random draw goes through the injected ``rng``.

The cost estimators (Regression, LLM-Estimate, Oracle) plug in through
``AgentEconomics.task_cost_multiplier``: they change the cost belief, and the
same pricing policy (Adaptive) turns it into a bid.

``zi_random``       Random: uniform between execution cost and the maximum price
``h2_strategy_1``   Random-in-band: uniform in the margin band over the cost basis
``h2_strategy_3``   Adaptive: margin raised after a win, lowered after a loss
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

#: Fallback for ad-hoc callers only; the agent passes its own seeded generator. A
#: private instance, so a forgotten ``rng=`` never draws from the global generator.
_DEFAULT_RNG = _random.Random()

EFFORT_TIERS: Tuple[str, str, str] = ("low", "medium", "high")


# --- Inputs ---
@dataclass(frozen=True)
class AuctionView:
    """What the agent can observe about one auction and its own standing."""

    max_price: int                                  # micro-USDC
    reputation_weight: int = 50                     # 0-100, as the contract stores it
    our_reputation: float = 50.0                    # 0-100 rating
    our_feedback_count: int = 0
    competitor_reputations: Tuple[float, ...] = ()  # visible competitors, self excluded


@dataclass(frozen=True)
class AgentEconomics:
    """The agent's beliefs about its own costs and delivered quality.

    They come from the agent's ``CostTracker`` history, adjusted by its cost
    estimator if it has one.
    """

    execution_cost_micro: int  # running average or prior
    execution_cost_by_tier_micro: Mapping[str, int] = field(default_factory=dict)
    quality_by_tier: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    overhead_per_participation_micro: int = 0
    unrecovered_overhead_micro: int = 0
    wins: int = 0  # jobs delivered so far

    #: Per-tier, job-specific deviation from the history level, or None for "no
    #: estimate — bid the history" (Adaptive). The estimator supplies the shape;
    #: history supplies the level:
    #:
    #:     believed_cost[tier] = history_tier_level[tier] * multiplier[tier]
    #:
    #: A value of 1.0 recovers the pure heuristic exactly.
    task_cost_multiplier: Optional[Mapping[str, float]] = None

    #: Every price this agent has actually won at, in the order won: the one input
    #: the decline rule needs beyond the cost belief. Deliberately not the
    #: competitors' revealed bids. Empty means "never won"; the rule then has no
    #: opinion.
    own_winning_bids_micro: Tuple[int, ...] = ()

@dataclass(frozen=True)
class PolicyParams:
    """Configured constants. Margins are normalised the same way for every caller."""

    min_margin: float = 0.10
    max_margin: float = 0.30
    initial_margin_mode: str = "max"          # "max" | "mid" | "random"
    initial_reputation_default: float = 50.0

    #: "price_only": the agent prices every job it is eligible for and never
    #: declines. "price_or_decline": the decline rule (:func:`abstention_reason`).
    action_space: str = "price_only"       # "price_only" | "price_or_decline"

    #: Which point of the agent's own winning-bid history the decline rule
    #: compares against: ``None`` is the mean, a quantile in (0, 1] a point of the
    #: distribution. 1.0, the highest price the agent has won at, is the rule of
    #: the experiments; the mean would decline jobs still winnable at a profit.
    #: A pinned constant, not a config option.
    decline_price_quantile: Optional[float] = 1.0

    # Adaptive's margin steps after a win and after a loss.
    step_up: float = 0.05
    step_down: float = 0.05

    def __post_init__(self) -> None:
        # Mirrors the clamping _heuristic_reasoning applies up front.
        object.__setattr__(self, "min_margin", max(0.0, float(self.min_margin)))
        object.__setattr__(self, "max_margin", max(self.min_margin, float(self.max_margin)))
        object.__setattr__(
            self, "initial_margin_mode",
            str(self.initial_margin_mode or "max").strip().lower(),
        )
        action_space = str(self.action_space or "price_only").strip().lower()
        if action_space not in ("price_only", "price_or_decline"):
            raise ValueError(
                f"unknown action_space {self.action_space!r}; expected "
                f"'price_only' or 'price_or_decline'"
            )
        object.__setattr__(self, "action_space", action_space)

    @property
    def can_decline(self) -> bool:
        return self.action_space == "price_or_decline"


@dataclass
class PolicyState:
    """Controller memory. Mutable, persists across auctions, owned by the caller."""

    margin: Optional[float] = None      # None = not yet initialised
    prev_wins: int = 0
    bids_placed: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    """What the policy decided, plus everything the caller needs to act and log."""

    bid_micro: Optional[int]                        # None = abstain
    effort_tier: Optional[str] = None               # None = caller's default tier
    reason: str = ""  # logged with the bid
    profit_cost_basis_micro: int = 0                # true cost of what would be delivered
    bid_interval: Optional[Tuple[int, int]] = None  # support of the draw, for stochastic policies

    @property
    def abstained(self) -> bool:
        return self.bid_micro is None


def _abstain(reason: str, profit_cost_basis_micro: int = 0) -> PolicyDecision:
    return PolicyDecision(
        bid_micro=None, effort_tier=None, reason=reason,
        profit_cost_basis_micro=profit_cost_basis_micro,
    )


# --- Shared controller pieces ---
def expected_clearing_price_micro(
    econ: AgentEconomics, params: PolicyParams
) -> Optional[int]:
    """Where this agent believes the market clears, from its own wins alone.

    An agent that has never won has no estimate, so the decline rule is inert
    until the first win, which also keeps a new agent from declining its way out
    of ever winning.
    """
    bids = econ.own_winning_bids_micro
    if not bids:
        return None
    quantile = params.decline_price_quantile
    if quantile is None:
        return int(sum(bids) / len(bids))
    ordered = sorted(bids)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return int(ordered[index])


def abstention_reason(
    econ: AgentEconomics, params: PolicyParams
) -> Optional[str]:
    """Whether the decline rule declines this auction, and why. ``None`` means bid.

    Decline when the job's believed cost cannot be covered at the going rate:
    ``cost + overhead > expected clearing price``, the clearing price being the
    highest price the agent has won at. The threshold is a price the market has
    actually paid this agent, not a multiple of its own cost.

    An agent without a job-specific estimate (``task_cost_multiplier is None``,
    Adaptive) never declines: its believed cost does not vary between jobs, so
    the rule would give the same answer for every auction.

    The gate sits before pricing, so a declined auction costs neither gas nor
    overhead. The threshold only rises over a run, since the agent's winning
    bids only accumulate.
    """
    if not params.can_decline:
        return None
    # Without a job-specific estimate the rule cannot tell auctions apart.
    if econ.task_cost_multiplier is None:
        return None
    expected = expected_clearing_price_micro(econ, params)
    if expected is None:
        return None

    total_cost = econ.execution_cost_micro + econ.overhead_per_participation_micro
    if total_cost <= expected:
        return None
    return f"declined: cost {total_cost} above expected clearing price {expected}"


def initial_derivative_follower_margin(params: PolicyParams, rng=_DEFAULT_RNG) -> float:
    """Adaptive's starting margin.

    "max" (default) starts at max_margin and concedes down; "mid" starts at the
    midpoint; "random" draws it uniformly, which desynchronises agents of one group.
    """
    lo, hi = params.min_margin, params.max_margin
    if params.initial_margin_mode == "mid":
        return (lo + hi) / 2.0
    if params.initial_margin_mode == "random":
        return rng.uniform(lo, hi)
    return hi


def _derivative_follower_update(state: PolicyState, wins: int, params: PolicyParams,
                                rng) -> None:
    """Raise the margin after a win, lower it after a loss.

    Steady-state P(win) ≈ step_down / (step_up + step_down).
    """
    if state.margin is None:
        state.margin = initial_derivative_follower_margin(params, rng)
        state.prev_wins = wins
        state.bids_placed = 0
    if state.bids_placed > 0:
        if wins > state.prev_wins:
            state.margin += params.step_up
        else:
            state.margin -= params.step_down
    state.prev_wins = wins


# --- Policies ---
def decide_zi_random(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                     params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Random: Uniform[cost, max_price].

    Carries no controller state and reads no win/loss, competitor or reputation
    signal. The floor is the execution-cost estimate only: participation overhead
    (gas per auction entered) depends on outcomes, and putting it in the bid would
    reintroduce the feedback this strategy is meant to lack. It is still charged
    against profit.
    """
    if view.max_price <= 0:
        return _abstain(f"invalid max_price={view.max_price}")

    profit_cost_basis = econ.execution_cost_micro + econ.overhead_per_participation_micro
    floor = max(1, int(econ.execution_cost_micro))
    if floor > view.max_price:
        return _abstain(
            f"cost floor above max_price (floor={floor}, max_price={view.max_price})",
            profit_cost_basis,
        )

    bid = rng.randint(floor, view.max_price)
    return PolicyDecision(
        bid_micro=bid,
        reason=f"bid={bid} [{floor}, {view.max_price}] cost={floor}",
        profit_cost_basis_micro=profit_cost_basis,
        bid_interval=(floor, view.max_price),
    )


def decide_h2_strategy_1(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                         params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Random-in-band: Uniform[cost_basis×(1+min), cost_basis×(1+max)].

    The cost basis includes participation overhead (gas, reasoning) as well as
    the execution cost.
    """
    if view.max_price <= 0:
        return _abstain(f"invalid max_price={view.max_price}")

    cost_basis = econ.execution_cost_micro + econ.overhead_per_participation_micro

    min_bid = int(cost_basis * (1.0 + params.min_margin))
    max_bid = min(int(cost_basis * (1.0 + params.max_margin)), view.max_price)
    effective_ceiling = max_bid

    if effective_ceiling < min_bid:
        return _abstain(
            f"no profitable region (min_bid={min_bid}, max_bid={max_bid})", cost_basis
        )

    bid = rng.randint(min_bid, effective_ceiling)
    return PolicyDecision(
        bid_micro=bid,
        reason=(
            f"bid={bid} [{min_bid}, {effective_ceiling}] cost_basis={cost_basis} "
            f"overhead={econ.overhead_per_participation_micro}"
        ),
        profit_cost_basis_micro=cost_basis,
        bid_interval=(min_bid, effective_ceiling),
    )


def decide_h2_strategy_3(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                         params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Adaptive: bid = cost_basis × (1 + margin), the margin adjusted after each auction.

    The margin rises by ``step_up`` after a win and falls by ``step_down`` after a
    loss, within [min_margin, the margin that reaches max_price].
    """
    if view.max_price <= 0:
        return _abstain(f"invalid max_price={view.max_price}")

    cost_basis = econ.execution_cost_micro + econ.overhead_per_participation_micro

    min_bid = int(cost_basis * (1.0 + params.min_margin))
    if min_bid > view.max_price:
        return _abstain(
            f"max_price below profit floor (max_price={view.max_price}, min_bid={min_bid})",
            cost_basis,
        )

    _derivative_follower_update(state, econ.wins, params, rng)

    margin_cap = float(view.max_price) / max(cost_basis, 1) - 1.0
    state.margin = max(params.min_margin, min(state.margin, margin_cap))

    bid = int(cost_basis * (1.0 + state.margin))
    bid = max(1, min(bid, view.max_price))
    state.bids_placed += 1

    return PolicyDecision(
        bid_micro=bid,
        reason=(
            f"bid={bid} margin={state.margin:.3f} cost_basis={cost_basis} "
            f"wins={econ.wins} bids={state.bids_placed}"
        ),
        profit_cost_basis_micro=cost_basis,
    )


POLICIES = {
    "zi_random": decide_zi_random,
    "h2_strategy_1": decide_h2_strategy_1,
    "h2_strategy_3": decide_h2_strategy_3,
}


def decide(strategy: str, view: AuctionView, econ: AgentEconomics, state: PolicyState,
           params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Dispatch to a named policy. Raises KeyError for an unknown strategy.

    The decline rule is checked first, so a declined auction leaves the
    controller's state untouched, and the rule applies whatever the policy.
    """
    declined = abstention_reason(econ, params)
    if declined:
        return PolicyDecision(
            bid_micro=None,
            reason=declined,
            profit_cost_basis_micro=(
                econ.execution_cost_micro + econ.overhead_per_participation_micro
            ),
        )
    return POLICIES[strategy](view, econ, state, params, rng)
