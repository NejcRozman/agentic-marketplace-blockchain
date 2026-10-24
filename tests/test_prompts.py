"""The LLM pricing prompt and the LLM tool descriptions are those of the experiments."""
import ast
import hashlib
import random
from pathlib import Path

from agents.common import prompts
from conftest import GOLDEN, digest

ROOT = Path(__file__).resolve().parents[1]


def maybe(rng, value, p=0.15):
    return None if rng.random() < p else value


def state(rng):
    s = {}
    s["agent_id"] = rng.choice([9694, 9695, "9694", None])
    if rng.random() > 0.05:
        s["agent_reputation"] = {"rating": maybe(rng, rng.choice([80, 77.5, 85.333, "82"])), "feedback_count": rng.randint(0, 30)}
    s["competitors_reputation"] = [{"agent_id": rng.choice([9695, "9696", None]), "rating": rng.choice([80, 88.25, None, "79"]),
                                    "feedback_count": rng.randint(0, 30)} for _ in range(rng.randint(0, 3))]
    s["estimated_execution_cost"] = maybe(rng, round(rng.uniform(0.005, 0.15), rng.choice([3, 6, 9])))
    if rng.random() > 0.1:
        s["non_execution_cost_per_participation"] = maybe(rng, rng.uniform(0, 0.01))
    for k, f in (("bids_placed_count", lambda: rng.randint(0, 30)), ("wins_count", lambda: rng.randint(0, 15)),
                 ("win_rate", lambda: round(rng.random(), 3)), ("avg_gross_margin_per_win_usd", lambda: rng.uniform(-0.02, 0.05)),
                 ("current_balance", lambda: rng.uniform(-0.3, 0.5)), ("past_execution_costs", lambda: [round(rng.uniform(0.01, 0.1), 4) for _ in range(rng.randint(0, 5))])):
        if rng.random() > 0.1:
            s[k] = maybe(rng, f())
    if rng.random() < 0.6:
        s["llm_last_margin_pct"] = maybe(rng, round(rng.uniform(-10, 150), 1), 0.3)
    if rng.random() < 0.5:
        s["llm_outcome_history"] = [{"auction_id": rng.randint(1, 30), "tier": rng.choice(["medium", None]),
                                     "margin_pct": maybe(rng, rng.uniform(-5, 120)), "bid_micro": rng.randint(10000, 1000000),
                                     "outcome": rng.choice(["won", "lost", "pending"]), "reputation_before": maybe(rng, rng.uniform(70, 95)),
                                     "reputation_after": maybe(rng, rng.uniform(70, 95), 0.3), "reputation_delta": maybe(rng, rng.uniform(-3, 3)),
                                     "balance_change_usd": maybe(rng, rng.uniform(-0.01, 0.05))} for _ in range(rng.randint(0, 6))]
    if rng.random() < 0.3:
        s["llm_strategy_memo"] = rng.choice(["", "  ", "Keep margin near 30%.\nRaise after wins.", None])
    if rng.random() < 0.5:
        s["llm_margin_bound_min"] = maybe(rng, rng.choice([0.2, 0.0, 0.15]))
        s["llm_margin_bound_max"] = maybe(rng, rng.choice([0.5, 1.0]))
    s["eligible_active_auctions"] = [{"auction_id": rng.randint(1, 30), "max_price": rng.choice([1000000, 999999, "1000000", None, 0]),
                                      "reputation_weight": rng.choice([0, 20, 80]), "winning_agent_id": rng.choice([0, 9694, 9695, "9695", None]),
                                      "current_winner_reputation": maybe(rng, rng.choice([80, "83", 77.25]), 0.6),
                                      "time_remaining": rng.uniform(1, 40)} for _ in range(rng.randint(0, 2))]
    # Keys of other architectures, which the pricing prompt ignores.
    if rng.random() < 0.2:
        s["estimated_service_cost"] = 0.05
        s["effort_level_execution_costs"] = {"low": 0.03, "medium": 0.06}
    return s


def test_pricing_prompt_identical_to_the_experiments():
    """20,000 random agent states, including missing, None and string-typed fields."""
    rendered = [prompts.get_prompt("2", state(random.Random(i))) for i in range(20000)]
    assert digest(rendered) == GOLDEN["pricing_prompts"]


def tool_docstring(path, name):
    for node in ast.walk(ast.parse((ROOT / path).read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_docstring(node, clean=False) or ""
    raise AssertionError(f"{name} not found in {path}")


def test_tool_descriptions_identical_to_the_experiments():
    """LangChain sends a tool's docstring to the LLM as its description."""
    for path, name in [("agents/provider/blockchain_handler.py", "place_bid"),
                       ("agents/provider/service_executor.py", "search_literature")]:
        assert hashlib.sha256(tool_docstring(path, name).encode()).hexdigest() == GOLDEN["tools"][name], name
