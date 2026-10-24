"""The heuristic bidding policies decide exactly as in the experiments."""
import random

from agents.provider import policies as P
from conftest import GOLDEN, digest


def play():
    """3000 random auction sequences of 40 decisions each, over every policy setting."""
    out = []
    for trial in range(3000):
        g = random.Random(trial)
        strategy = g.choice(["zi_random", "h2_strategy_1", "h2_strategy_3"])
        params = P.PolicyParams(min_margin=g.choice([0.2, 0.0, 0.1, -0.1]), max_margin=g.choice([0.5, 0.3, 0.2, 1.0]),
                                initial_margin_mode=g.choice(["max", "mid", "random", None]),
                                initial_reputation_default=g.choice([80.0, 50.0]),
                                action_space=g.choice(["price_only", "price_or_decline"]))
        state = P.PolicyState()
        rng = random.Random(1000 + trial)
        wins, won_bids, trace = 0, [], []
        for step in range(40):
            view = P.AuctionView(max_price=g.choice([1_000_000, 1_000_000, 50_000, 0, 999_999]),
                                 reputation_weight=g.choice([0, 20, 50, 80]), our_reputation=g.uniform(60, 95),
                                 our_feedback_count=step,
                                 competitor_reputations=tuple(g.uniform(60, 95) for _ in range(g.randint(0, 3))))
            econ = P.AgentEconomics(execution_cost_micro=g.randint(5_000, 200_000),
                                    execution_cost_by_tier_micro={"medium": g.randint(5_000, 200_000)},
                                    overhead_per_participation_micro=g.randint(0, 20_000), wins=wins,
                                    task_cost_multiplier=g.choice([None, {"medium": g.uniform(0.3, 3)}]),
                                    own_winning_bids_micro=tuple(won_bids))
            d = P.decide(strategy, view, econ, state, params, rng=rng)
            trace.append([d.bid_micro, d.effort_tier, d.reason, d.profit_cost_basis_micro, d.bid_interval,
                          state.margin, state.prev_wins, state.bids_placed, params.can_decline])
            if d.bid_micro is not None and g.random() < 0.45:
                wins += 1
                won_bids.append(d.bid_micro)
        out.append([strategy, trace])
    return out


def test_decisions_identical_to_the_experiments():
    assert digest(play()) == GOLDEN["policies"]


def test_unknown_strategy_is_rejected():
    import pytest
    with pytest.raises(Exception):
        P.decide("no_such_strategy", P.AuctionView(max_price=1_000_000, reputation_weight=20, our_reputation=80,
                                                   our_feedback_count=0, competitor_reputations=()),
                 P.AgentEconomics(execution_cost_micro=60_000, execution_cost_by_tier_micro={"medium": 60_000},
                                  overhead_per_participation_micro=0, wins=0),
                 P.PolicyState(), P.PolicyParams(), rng=random.Random(0))
