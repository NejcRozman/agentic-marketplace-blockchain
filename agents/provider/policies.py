"""Pure bidding policies for the provider agent's heuristic ladder.

Both the live agent (``BlockchainHandler._heuristic_reasoning``) and the offline
market simulator (S1 §2–§3) call these functions, so a mismatch between
simulated and observed markets can only be a *mechanism* difference — never a
policy re-implementation drift.  See ``docs/s1_implementation.md`` §1.

Everything here is a pure function of

    (AuctionView, AgentEconomics, PolicyState, PolicyParams, rng)

with the single deliberate exception that ``PolicyState`` is *mutated in place*:
it is the controller's memory across auctions (margin, prev_wins, bids_placed).
No blockchain access, no I/O, no module-level ``random`` use — every stochastic
draw goes through the injected ``rng`` so the simulator can seed it explicitly.

The module boundary is also the S7 hybrid arm: feeding an ``AgentEconomics``
built from LLM estimates into the same ``decide_*`` function *is* the
estimator/controller split, with no other code change.

Extracted policies
------------------
``zi_random``       zero-intelligence uniform bidder — the β null (new in S1)
``h2_strategy_1``   overhead-aware cost-plus, uniform draw in the margin band
``h2_strategy_2``   win-rate-adaptive cost-plus
``h2_strategy_3``   derivative-follower pricing, fixed effort
``h3_strategy_2``   derivative follower + career-concerns effort gate

The H1 family (``cost_plus``, ``feasible_random``, ``h1_strategy_3``) and
``h3_strategy_1`` are intentionally *not* extracted: they are outside the S1 §4
policy set and are not simulated.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

#: Fallback for ad-hoc callers only. Both real callers — the live agent
#: (``blockchain_handler``) and the simulator (``experiments/simulator/run.py``)
#: — pass their own seeded generator. This is a private *instance* rather than
#: the ``random`` module so that a forgotten ``rng=`` argument can never draw
#: from the process-global generator (S7 §A.1).
_DEFAULT_RNG = _random.Random()

EFFORT_TIERS: Tuple[str, str, str] = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

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

    In the live agent these come from ``CostTracker`` history; in the simulator
    from its bookkeeping; in the S7 hybrid arm from the LLM estimator; in the
    oracle arms from the dataset's ground truth.
    """

    execution_cost_micro: int                                    # EMA or prior
    execution_cost_by_tier_micro: Mapping[str, int] = field(default_factory=dict)
    quality_by_tier: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    overhead_per_participation_micro: int = 0
    unrecovered_overhead_micro: int = 0
    wins: int = 0                                                # deliveries observed so far

    #: Per-tier task-conditional deviation from the history level, or None for
    #: "no estimate — bid the history" (hybrid_agent_design §2.1). The estimator
    #: supplies the *shape*; history supplies the *level*:
    #:
    #:     believed_cost[tier] = history_tier_level[tier] * multiplier[tier]
    #:
    #: Arm R writes a regression's output here; arm X writes an LLM's. A value of
    #: 1.0 recovers the pure heuristic exactly, which is what makes the cold-start
    #: degradation and the identity test (gate G-B) checkable rather than assumed.
    task_cost_multiplier: Optional[Mapping[str, float]] = None

    #: Every price this agent has actually **won** at, in the order won (S7 §C).
    #: The one input O2c's decline rule needs beyond the cost belief above, and
    #: deliberately *not* the competitors' revealed bids: that is a different
    #: information axis (S1b §4, the bidding oracle), and no cost estimator could
    #: ever supply it. Empty means "never won" — the rule then has no opinion.
    own_winning_bids_micro: Tuple[int, ...] = ()

    def tier_cost_micro(self, tier: str) -> int:
        """Per-tier execution cost, falling back to the scalar estimate."""
        raw = self.execution_cost_by_tier_micro.get(tier)
        return self.execution_cost_micro if raw is None else int(raw)


@dataclass(frozen=True)
class PolicyParams:
    """Configured constants. Margins are normalised the same way for every caller."""

    min_margin: float = 0.10
    max_margin: float = 0.30
    initial_margin_mode: str = "max"          # "max" | "mid" | "random"
    initial_reputation_default: float = 50.0
    quality_prior_by_tier: Mapping[str, float] = field(default_factory=dict)

    # S7 §C — the abstention gate. This **is** S6's rung O2c, field for field:
    # ``action_space`` and ``decline_price_quantile`` carry the same names, the
    # same defaults and the same meanings as ``SimulatedProvider``'s spec fields
    # in ``experiments/simulator/run.py``, and :func:`abstention_reason` below is
    # that provider's ``_should_decline`` with the object graph removed.
    #
    # Shared vocabulary on purpose. S6 is the result S7 exists to reproduce in a
    # live market, so the two cannot be allowed to mean different things by the
    # same word — the same reason ``cost_estimators.requirement_payload``
    # delegates to S5's own projection instead of rebuilding it.

    #: "price_only" — the live agent's original action space: it can price a job
    #: but never decline one it is eligible for. "price_or_decline" adds
    #: abstention, which is what makes cost information actionable (S1b §3,
    #: rung O2). Off by default, so every existing caller and the S1 equivalence
    #: fixture are untouched.
    action_space: str = "price_only"       # "price_only" | "price_or_decline"

    #: Which point of the agent's own winning-bid history the decline rule
    #: compares against. ``None`` is S1b §3's rule as written — the mean — which
    #: is rung **O2**. A quantile in (0, 1] makes the rule less aggressive; 1.0
    #: is rung **O2c**, the highest price the agent has actually won at.
    #:
    #: 1.0 by default because O2c is the rung S6 reported and the only one S7
    #: runs. S6's own note on why the mean is wrong: it "declines any job costing
    #: more than the *mean* clearing price, but half of all auctions clear above
    #: the mean, so it discards jobs that were still winnable at a profit."
    #: Deliberately not exposed in the run configs — a pinned constant of the
    #: rung, not a knob to sweep.
    decline_price_quantile: Optional[float] = 1.0

    # Controller constants. step_up/step_down are shared by h2_strategy_3 and
    # h3_strategy_2 so the pair stays a pure effort-controller ablation.
    step_up: float = 0.05
    step_down: float = 0.05
    margin_step: float = 0.05                 # h2_strategy_2
    rw_gate: float = 0.30                     # h3_strategy_2
    invest_payback_wins: float = 10.0         # h3_strategy_2
    quality_pseudo_count: float = 3.0         # h3_strategy_2

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
        """S6's ``SimulatedProvider.spec.can_decline``, verbatim."""
        return self.action_space == "price_or_decline"

    def quality_prior(self, tier: str) -> float:
        raw = self.quality_prior_by_tier.get(tier)
        if raw is None:
            return float(self.initial_reputation_default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(self.initial_reputation_default)


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
    reason: str = ""                                # audit parity with the old log lines
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


# ---------------------------------------------------------------------------
# Shared controller pieces
# ---------------------------------------------------------------------------

def expected_clearing_price_micro(
    econ: AgentEconomics, params: PolicyParams
) -> Optional[int]:
    """Where this agent believes the market clears, **from its own wins alone**.

    S6's ``SimulatedProvider.expected_clearing_price_micro``, moved into
    ``policies`` so the live agent and the simulator run the identical
    arithmetic (S1 §1) and the handler only has to supply the observations.

    Deliberately not the competitors' revealed bids: that is a different
    information axis (S1b §4, the bidding oracle), and no task-text estimator
    could ever supply it. **An agent that has never won has no estimate**, so the
    rule is inert until the first win — which is also what keeps a cold-start
    agent from declining its way out of ever getting one.
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
    """Whether O2c declines this auction, and why. ``None`` means bid.

    **S6's rung O2c, unchanged**: abstain when the job's believed cost cannot be
    covered at the going rate — ``cost + overhead > expected clearing price``.
    This is ``SimulatedProvider._should_decline`` with the object graph removed;
    S6 measured the hybrid beating R only at this rung, so the live system has to
    run this rule and not a cousin of it.

    **Revenue-anchored**, which the rule it replaced was not: the threshold is a
    price the market has actually paid this agent, not a multiple of its own
    cost. A systematically biased cost estimator therefore shifts *which* jobs
    are declined, but cannot by itself drive the decline rate to the ceiling the
    way a cost-anchored threshold sitting at the estimator's modal output does.

    **One deliberate departure from S6: an arm with no task-conditional estimate
    never declines** (``task_cost_multiplier is None`` — arm H). S6 handed the
    action space to every provider including its history baseline; here H is held
    to price-only, because for H the rule is degenerate rather than informative.
    Its believed cost is its EMA, which does not vary with the requirement, and
    the threshold is its own win maximum, which does not vary within a polling
    cycle — so the comparison returns the *same answer for every auction*. It is
    a market-exit switch ("my average job no longer clears"), not a decision
    about this job, and reporting it as abstention would put a task-blind arm's
    all-or-nothing flip in the same column as X's per-requirement declining.

    The gate sits before pricing on purpose: the point is to skip the
    participation entirely — each pointless bid costs gas plus per-participation
    overhead, which is the transaction cost S6 found abstention recovers.

    Note the threshold ratchets: ``own_winning_bids_micro`` only grows, and at
    ``decline_price_quantile = 1.0`` the comparison is against its maximum, so
    the bar is monotonically non-decreasing over a run. That is S6's behaviour
    and it is inherited deliberately, not by oversight.
    """
    if not params.can_decline:
        return None
    # Arm H. See the docstring: with no task-conditional estimate the rule cannot
    # discriminate between auctions, so letting it fire would record a market-exit
    # flip as if it were per-job abstention.
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
    """Starting margin for the derivative-follower controllers.

    "max" (default): start greedy at max_margin and concede down — classic
    ZIP/Cliff, which spends the first several auctions probing down from the
    ceiling.  "mid": competitive from the first bid.  "random": a uniform draw,
    which desynchronises agents in the same group.
    """
    lo, hi = params.min_margin, params.max_margin
    if params.initial_margin_mode == "mid":
        return (lo + hi) / 2.0
    if params.initial_margin_mode == "random":
        return rng.uniform(lo, hi)
    return hi


def _derivative_follower_update(state: PolicyState, wins: int, params: PolicyParams,
                                rng) -> None:
    """Margin creeps up after a win, concedes after a loss.

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


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def decide_zi_random(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                     params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Zero-intelligence bidder: Uniform[cost, max_price]. The ladder floor.

    Carries no controller state and reads no win/loss, competitor or reputation
    signal, which is what makes it the β null.  S0 showed h2_strategy_1 is not a
    valid zero-β control, because its floor moves with observed overhead.

    The floor is the *execution*-cost estimate only.  Participation overhead is
    excluded deliberately: it is outcome-coupled (gas spent per auction
    participated in), and routing it into the bid would reintroduce the very
    feedback channel this arm exists to switch off.  It is still charged against
    profit, so the arm's economics stay honest.

    Caveat for the live agent: ``econ.execution_cost_micro`` is an EMA over
    completed jobs there, a weak residual feedback channel.  Feed a fixed cost
    prior (as the simulator does) if β must be exactly zero by construction.
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
    """Overhead-aware cost-plus: Uniform[cost_basis×(1+min), cost_basis×(1+max)].

    Differs from H1 ``cost_plus`` in that the cost basis includes participation
    overhead (gas, commitment costs) rather than execution cost alone.
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


def decide_h2_strategy_2(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                         params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Win-rate-adaptive cost-plus.

        bid = cost_basis × (1 + margin)

    The margin starts at the midpoint of the band and adjusts after each
    auction against a target win rate of 1/(n_competitors+1) — equal market
    share.  No competitor bid assumptions; no abstention beyond the profit
    floor.  Overhead is recovered implicitly: losses lower the win rate → the
    margin drops → more wins.
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

    if state.margin is None:
        state.margin = (params.min_margin + params.max_margin) / 2
        state.bids_placed = 0

    n_competitors = len(view.competitor_reputations)
    target_win_rate = 1.0 / (n_competitors + 1) if n_competitors > 0 else 0.5

    if state.bids_placed > 0:
        win_rate = econ.wins / state.bids_placed
        if win_rate > target_win_rate:
            state.margin = min(state.margin + params.margin_step, params.max_margin)
        elif win_rate < target_win_rate:
            state.margin = max(state.margin - params.margin_step, params.min_margin)
    else:
        win_rate = 0.0

    bid = min(int(cost_basis * (1.0 + state.margin)), view.max_price)
    bid = max(bid, min_bid)
    state.bids_placed += 1

    return PolicyDecision(
        bid_micro=bid,
        reason=(
            f"bid={bid} margin={state.margin:.3f} "
            f"wins={econ.wins}/{state.bids_placed} win_rate={win_rate:.2f} "
            f"target={target_win_rate:.2f} cost_basis={cost_basis}"
        ),
        profit_cost_basis_micro=cost_basis,
    )


def decide_h2_strategy_3(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                         params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Derivative-follower pricing, fixed default effort (the H3-S2 ablation twin).

    The exact price controller of :func:`decide_h3_strategy_2` with the effort
    controller removed, so running the two against each other isolates the
    causal contribution of the effort controller.
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


def decide_h3_strategy_2(view: AuctionView, econ: AgentEconomics, state: PolicyState,
                         params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Derivative-follower price controller + career-concerns effort gate.

    Price controller — identical to :func:`decide_h2_strategy_3`.

    Effort controller — a mobility gate (Holmström 1982/1999; empirically Cabral
    & Hortaçsu 2010): effort is front-loaded while the market's record of you is
    still malleable and stops once it hardens.  With an all-time-average rating,
    one delivery moves the rating by (q_high − rating)/(n+1), and the current
    rating is itself shrunk toward the platform prior by ``quality_pseudo_count``
    so a single lucky draw at n=1 cannot whipsaw the gate.

        rw below gate                     → low tier (reputation not priced; shirk)
        (q_high − rep_shrunk)/(n+1) ≥ τ   → high tier (a delivery still buys ≥ τ points)
        else                              → medium (maintain)

    τ(rw) = cost_premium / (point_value(rw) × invest_payback_wins) scales the
    payback bar with what a rating point is worth at *this* auction's rw.

    The pricing basis is min(medium, chosen-tier) cost, so it matches H2-S3
    exactly whenever the tier is medium/high (a pure effort ablation) and is
    deliberately cheaper when shirking.  No competitor reference enters either
    controller: the only market signal is winning or losing.
    """
    if view.max_price <= 0:
        return _abstain(f"invalid max_price={view.max_price}")

    overhead = econ.overhead_per_participation_micro
    rw = max(0.0, min(1.0, view.reputation_weight / 100.0))
    feedback_count = max(0, int(view.our_feedback_count))

    # Achievable quality ceiling: observed high-tier average blended with the
    # config prior by sample count. An internal belief about future delivery
    # quality, estimated from very few samples — the prior keeps a single 84 or
    # 90 draw from opening/closing the gate.
    q_high_prior = params.quality_prior("high")
    high_stats = econ.quality_by_tier.get("high") or {}
    q_high_obs = high_stats.get("average_rating")
    q_high_n = max(0.0, float(high_stats.get("feedback_count", 0) or 0))
    if q_high_obs is not None and q_high_n > 0:
        q_high_obs = max(0.0, min(100.0, float(q_high_obs)))
        q_high = (
            q_high_n * q_high_obs + params.quality_pseudo_count * q_high_prior
        ) / (q_high_n + params.quality_pseudo_count)
    else:
        q_high = q_high_prior

    # Confidence discount on the rating itself, distinct from the (n+1)
    # dilution below; fades to ~raw rating by n≈30, exact prior at n=0.
    rep_shrunk = (
        feedback_count * view.our_reputation
        + params.quality_pseudo_count * float(params.initial_reputation_default)
    ) / (feedback_count + params.quality_pseudo_count)

    # Rating points one more delivery can buy (shrinks as 1/n as the record hardens).
    rating_mobility = (q_high - rep_shrunk) / (feedback_count + 1.0)

    tier_cost_micros = {t: econ.tier_cost_micro(t) for t in EFFORT_TIERS}

    # Investment payback threshold τ(rw): $ of bid room per rating point right
    # now (straight from the score formula) times invest_payback_wins future
    # wins must cover the high-vs-medium execution cost premium.
    cost_premium_micro = max(0, tier_cost_micros["high"] - tier_cost_micros["medium"])
    rw_capped = min(rw, 0.99)
    point_value_micro = (rw_capped / (1.0 - rw_capped)) * view.max_price / 100.0
    if cost_premium_micro <= 0:
        mobility_threshold = 0.0
    elif point_value_micro <= 0:
        mobility_threshold = float("inf")
    else:
        mobility_threshold = cost_premium_micro / (point_value_micro * params.invest_payback_wins)

    # Effort thermostat.
    if rw < params.rw_gate:
        desired_tier = "low"
    elif rating_mobility >= mobility_threshold:
        desired_tier = "high"
    else:
        desired_tier = "medium"

    # Degrade to the best affordable tier; abstain when none is affordable.
    fallback_order = {
        "high": ("high", "medium", "low"),
        "medium": ("medium", "low"),
        "low": ("low",),
    }
    chosen_tier = None
    for tier in fallback_order[desired_tier]:
        if tier_cost_micros[tier] + overhead <= view.max_price:
            chosen_tier = tier
            break
    if chosen_tier is None:
        return _abstain(
            f"no affordable effort tier (max_price={view.max_price}, "
            f"tier_costs={tier_cost_micros})"
        )

    _derivative_follower_update(state, econ.wins, params, rng)

    # Bid off min(chosen tier cost, medium cost). The medium anchor protects
    # competitiveness when INVESTING; taking the min stops it from inflating the
    # bid above what a cheaper delivery actually costs when shirking.
    pricing_cost_basis = min(
        tier_cost_micros["medium"], tier_cost_micros[chosen_tier]
    ) + overhead
    margin_cap = float(view.max_price) / max(pricing_cost_basis, 1) - 1.0
    state.margin = max(params.min_margin, min(state.margin, margin_cap))

    bid = int(pricing_cost_basis * (1.0 + state.margin))
    bid = max(1, min(bid, view.max_price))
    state.bids_placed += 1

    cost_basis = tier_cost_micros[chosen_tier] + overhead
    return PolicyDecision(
        bid_micro=bid,
        effort_tier=chosen_tier,
        reason=(
            f"bid={bid} tier={chosen_tier} margin={state.margin:.3f} "
            f"mobility={rating_mobility:.2f} threshold={mobility_threshold:.2f} "
            f"our_rep={view.our_reputation:.1f} (shrunk={rep_shrunk:.1f}) "
            f"q_high={q_high:.1f} n={feedback_count} rw={rw:.2f} "
            f"wins={econ.wins} bids={state.bids_placed}"
        ),
        profit_cost_basis_micro=cost_basis,
    )


POLICIES = {
    "zi_random": decide_zi_random,
    "h2_strategy_1": decide_h2_strategy_1,
    "h2_strategy_2": decide_h2_strategy_2,
    "h2_strategy_3": decide_h2_strategy_3,
    "h3_strategy_2": decide_h3_strategy_2,
}


def decide(strategy: str, view: AuctionView, econ: AgentEconomics, state: PolicyState,
           params: PolicyParams, rng=_DEFAULT_RNG) -> PolicyDecision:
    """Dispatch to a named policy. Raises KeyError for an unknown strategy.

    Rung O2 sits **here**, ahead of the dispatch, exactly where S6 puts it
    (``SimulatedProvider.decide`` checks ``_should_decline`` before calling this
    function). Two consequences, both S6's and both wanted:

    * An abstention leaves the controller's state untouched — no margin update,
      no ``bids_placed`` increment — because the policy never runs.
    * The gate is a property of the *agent*, not of one strategy, so it applies
      to whichever policy the arm carries.
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
