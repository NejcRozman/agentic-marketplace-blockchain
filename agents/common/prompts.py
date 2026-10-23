"""Prompt templates for agent architectures."""

import json
from copy import deepcopy
from typing import Any, Dict, Optional

# Mirrors blockchain_handler.SCORE_PRECISION — the fixed-point denominator used to
# compute bid_component = (max_price - bid) * SCORE_PRECISION // max_price.
# One score unit per auction requires a bid change of max_price // SCORE_PRECISION micro-USDC.
_SCORE_PRECISION = 10_000


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            try:
                parsed = float(stripped)
            except ValueError:
                return None
            return int(parsed) if parsed.is_integer() else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_auctions_context(agent_state: Dict[str, Any]) -> str:
    """Render the actionable auctions, enriched with winner reputation and min score tick."""
    auctions = deepcopy(agent_state.get("eligible_active_auctions", []))
    agent_reputation = _coerce_int((agent_state.get("agent_reputation") or {}).get("rating"))
    competitor_reputations = {
        _coerce_int(comp.get("agent_id")): _coerce_int(comp.get("rating"))
        for comp in (agent_state.get("competitors_reputation") or [])
    }
    for auction in auctions:
        winning_agent_id = _coerce_int(auction.get("winning_agent_id"))
        winner_reputation = _coerce_int(auction.get("current_winner_reputation"))
        if winner_reputation is None and winning_agent_id is not None and winning_agent_id > 0:
            if winning_agent_id == _coerce_int(agent_state.get("agent_id")):
                winner_reputation = agent_reputation
            else:
                winner_reputation = competitor_reputations.get(winning_agent_id)
        if winner_reputation is not None:
            auction["current_winner_reputation"] = winner_reputation
        max_price = _coerce_int(auction.get("max_price"))
        if max_price and max_price > 0:
            auction["min_score_tick"] = max(1, max_price // _SCORE_PRECISION)
    return json.dumps(auctions, indent=2)


def _build_outcome_history_block(agent_state: Dict[str, Any]) -> str:
    """Render the agent's own decision->outcome record (R1 episodic memory).

    Returns "" when the feature is off, so the prompt is byte-identical to R0. When on but
    empty, says so explicitly rather than showing an empty table.
    """
    if "llm_outcome_history" not in agent_state:
        return ""

    rows = agent_state.get("llm_outcome_history") or []
    header = (
        "\nYour own past decisions and how they turned out "
        "(your complete record this run, oldest first):\n"
    )
    if not rows:
        return header + "  (no bids placed yet — this is your first decision)\n"

    def _fmt(value: Any, dash: str = "-") -> str:
        if value is None:
            return dash
        if isinstance(value, float):
            return f"{value:+.4f}" if abs(value) < 1 else f"{value:.2f}"
        return str(value)

    lines = [
        "  auction | tier   | margin | bid_micro | outcome | reputation      | balance_change_usd",
        "  --------|--------|--------|-----------|---------|-----------------|-------------------",
    ]
    for r in rows:
        rep_before = r.get("reputation_before")
        rep_after = r.get("reputation_after")
        if rep_after is None:
            rep_str = f"{_fmt(rep_before)} -> (pending)"
        else:
            delta = r.get("reputation_delta")
            rep_str = f"{_fmt(rep_before)} -> {_fmt(rep_after)}"
            if delta is not None:
                rep_str += f" ({delta:+.2f})"
        margin = r.get("margin_pct")
        lines.append(
            f"  {str(r.get('auction_id')):>7} "
            f"| {str(r.get('tier') or '-'):<6} "
            f"| {(f'{margin:.0f}%' if margin is not None else '-'):>6} "
            f"| {str(r.get('bid_micro')):>9} "
            f"| {str(r.get('outcome')):<7} "
            f"| {rep_str:<15} "
            f"| {_fmt(r.get('balance_change_usd'))}"
        )
    return header + "\n".join(lines) + "\n"


def _build_strategy_memo_block(agent_state: Dict[str, Any]) -> str:
    """Render the agent's self-edited strategy memo (R2 reflection).

    Returns "" when the feature is off, so the prompt is unchanged for R0/R1.
    """
    if "llm_strategy_memo" not in agent_state:
        return ""

    memo = (agent_state.get("llm_strategy_memo") or "").strip()
    header = "\nYour own notes, which you wrote and revised yourself after reviewing your results:\n"
    if not memo:
        return header + "  (empty — you have not completed an auction to learn from yet)\n"
    return header + "\n".join(f"  {ln}" for ln in memo.splitlines()) + "\n"


def _build_core_prompt(
    agent_state: Dict[str, Any],
    *,
    include_economics: bool,
    include_effort_breakdown: bool,
) -> str:
    """ReAct bidding prompt. The level is set by the two visibility flags only.

    include_economics=False (L1): cost estimate only; minimal, un-scaffolded prompt.
    include_economics=True  (L2): own cost EMA, overhead, unrecovered overhead, balance, history;
                                  structured 3-step reasoning protocol (target margin, self-check,
                                  commit).
    include_effort_breakdown=True (L3, requires include_economics): additionally per-tier cost &
                                  quality and the effort-tier action; same 3-step protocol with
                                  tier selection folded into step 1.
    """
    agent_id = agent_state.get("agent_id", "unknown")
    rep = (agent_state.get("agent_reputation") or {}).get("rating")
    feedback_count = (agent_state.get("agent_reputation") or {}).get("feedback_count")
    competitor_reps = agent_state.get("competitors_reputation", [])

    # --- state (visibility) block ---
    state_lines = [
        f"- your_reputation: {rep} (feedback_count: {feedback_count})",
        f"- competitor_reputations: {competitor_reps}",
    ]
    if not include_economics:
        cost_usd = agent_state.get('estimated_service_cost')
        cost_micro = round(float(cost_usd) * 1_000_000) if cost_usd is not None else None
        state_lines.append(f"- estimated_execution_cost_usd: {cost_usd}")
        state_lines.append(f"- estimated_execution_cost_micro_usdc: {cost_micro}")
    else:
        exec_cost_usd = agent_state.get('estimated_execution_cost')
        non_exec_usd = agent_state.get('non_execution_cost_per_participation') or 0.0
        exec_cost_micro = round(float(exec_cost_usd) * 1_000_000) if exec_cost_usd is not None else None
        non_exec_micro = round(float(non_exec_usd) * 1_000_000)
        total_cost_basis_micro = (exec_cost_micro + non_exec_micro) if exec_cost_micro is not None else None
        bids_placed = agent_state.get('bids_placed_count')
        wins_count = agent_state.get('wins_count')
        win_rate = agent_state.get('win_rate')
        avg_gross_margin_per_win = agent_state.get('avg_gross_margin_per_win_usd')
        current_balance = agent_state.get('current_balance')
        avg_profit_per_participation = (
            round(float(current_balance) / bids_placed, 6)
            if current_balance is not None and bids_placed else None
        )
        state_lines.extend([
            f"- estimated_execution_cost_micro_usdc: {exec_cost_micro}",
            f"- non_execution_cost_per_participation_micro_usdc: {non_exec_micro}",
            f"- total_cost_basis_micro_usdc: {total_cost_basis_micro}",
            f"- current_balance_usd: {current_balance}",
            f"- past_execution_costs_usd: {agent_state.get('past_execution_costs') or []}",
            f"- bids_placed_count: {bids_placed}",
            f"- wins_count: {wins_count}",
            f"- win_rate: {win_rate}",
            f"- avg_gross_margin_per_win_usd: {avg_gross_margin_per_win}",
            f"- avg_profit_per_participation_usd: {avg_profit_per_participation}",
        ])
        if 'llm_last_margin_pct' in agent_state:
            last_margin = agent_state.get('llm_last_margin_pct')
            state_lines.append(
                f"- last_committed_margin_pct: {last_margin if last_margin is not None else 'none yet (no prior bid)'}"
            )
    if include_effort_breakdown:
        per_tier_cost = agent_state.get("effort_level_execution_costs") or {}
        per_tier_cost_micro = {
            t: round(float(v) * 1_000_000) for t, v in per_tier_cost.items() if v is not None
        }
        per_tier_quality = {
            t: (stats or {}).get("average_rating")
            for t, stats in (agent_state.get("effort_level_quality_stats") or {}).items()
        }
        state_lines.extend([
            f"- per_tier_execution_cost_micro: {per_tier_cost_micro}",
            f"- per_tier_delivered_quality_mean: {per_tier_quality}",
        ])
    state_block = "\n".join(state_lines)

    outcome_history_block = _build_outcome_history_block(agent_state)
    strategy_memo_block = _build_strategy_memo_block(agent_state)

    auctions_block = _build_auctions_context(agent_state)

    # --- per-level objective + action text ---
    objective_lines = ["- Winning earns your bid minus your execution and overhead cost. "
    "- You are providing agentic service so think about appropriate margin for such a service."]
    if include_economics:
        objective_lines.append(
            "- Use your past costs, overhead and balance to manage risk and recovery across auctions, "
            "not only this one auction's margin."
        )
        objective_lines.append(
            "- You will not win every auction you bid on. Every auction you bid on costs gas and "
            "reasoning overhead whether you win it or not; non_execution_cost_per_participation_micro_usdc "
            "is this cost averaged across all your past participations, wins and losses alike — like a "
            "customer acquisition cost paid on every attempt, recovered only from the wins.\n"
            "- Your bottom line has two independent levers, and they can point in opposite directions:\n"
            "    (1) MARGIN PER WIN — avg_gross_margin_per_win_usd is what an average win earns after its "
            "own execution cost but before any overhead. If this is positive, your wins are individually "
            "profitable; if it is at or below zero, you are working at a loss even when you win.\n"
            "    (2) WIN VOLUME — win_rate and wins_count are how often you actually win. The overhead of "
            "every bid, won or lost, is only paid back by the wins; too few wins means the overhead from "
            "the losses outweighs the margin from the wins.\n"
            "  current_balance_usd ≈ (avg_gross_margin_per_win_usd × wins_count) − (overhead × every bid). "
            "avg_profit_per_participation_usd collapses both levers into one bottom-line number: your net "
            "result per auction bid on. Diagnose which lever is the problem before acting — they need "
            "opposite fixes."
        )
        objective_lines.append(
            "- Your reputation is not an estimate — it is stored on-chain as the plain arithmetic mean of "
            "every quality rating you have ever received, recomputed after each delivery. One new rating "
            "shifts your average by (new_rating − current_average) / (feedback_count + 1). This has a "
            "direct consequence: with a low feedback_count, a single delivery can move your reputation by "
            "a large amount and set a trajectory you carry into every subsequent auction; once "
            "feedback_count is large, the same delivery barely moves the average — your reputation becomes "
            "sticky in both directions. The same tier choice is therefore not equally risky at every point "
            "in your history: early on, with little feedback behind you, one bad delivery can do outsized "
            "and lasting damage, while later the identical delivery is nearly harmless. The same "
            "arithmetic applies when reading competitor_reputations: a rating shown next to a low "
            "feedback_count is a thin, easily-moved signal, not yet an earned track record — while the "
            "same rating next to a high feedback_count reflects a stable position unlikely to change "
            "quickly."
        )
    if include_effort_breakdown:
        objective_lines.append(
            "- You also choose an effort tier (low/medium/high) at bid time. The tier you commit to "
            "determines two things simultaneously: how much it costs you to execute the service "
            "(per_tier_execution_cost_micro) and the quality level you are expected to deliver "
            "(per_tier_delivered_quality_mean). Both consequences are locked in when you place the bid.\n"
            "- Quality you deliver becomes reputation feedback: your reputation score is a running mean "
            "of all quality scores you have ever delivered. Winning an auction and delivering above your "
            "current reputation raises it; delivering below lowers it. These shifts persist — every future "
            "auction you enter carries the reputation you have built or eroded.\n"
            "- How much reputation matters is set by the auction's reputation_weight (rw): when rw is "
            "high, a reputation difference between you and a competitor translates into a large score "
            "gap; when rw is low, bid price dominates and the same reputation difference has less effect "
            "on who wins. This means the long-run value of improving your reputation depends on the rw "
            "distribution of the auctions you expect to compete in.\n"
            "- A tier choice is therefore a trade-off across time: a more expensive tier costs current "
            "profit but may deliver quality above your current reputation, raising it and strengthening "
            "your future competitive position. A cheaper tier preserves current margin but may deliver "
            "quality below your standing, pulling your reputation down and weakening future competitiveness. "
            "Whether the trade-off is worth making depends on where your reputation currently sits, "
            "how far each tier's expected quality is from that level, and how heavily future auctions "
            "weight reputation in their scoring."
        )
    objective_block = "\n".join(objective_lines)

    if include_effort_breakdown:
        place_bid_args = "auction_id, bid_amount, effort_tier"
        action_rule = "- effort_tier must be one of: low, medium, high.\n"
    else:
        place_bid_args = "auction_id, bid_amount"
        action_rule = ""

    if include_economics:
        balance_recovery_rule = (
            "- current_balance_usd is your cumulative net result (all revenue minus all costs); negative "
            "means your policy is losing money overall so far. Do not react to it by reflex — first find "
            "out WHICH lever is failing, because the two point to opposite fixes:\n"
            "    • If avg_gross_margin_per_win_usd is at or below zero, your wins lose money even before "
            "overhead — you are pricing below what delivery actually costs you. RAISE your margin.\n"
            "    • If avg_gross_margin_per_win_usd is positive but current_balance_usd is negative, your "
            "wins are individually profitable and the deficit is the overhead of the auctions you LOSE. "
            "The fix is more wins, not fatter ones: LOWER your margin to win more often (or accept that "
            "this market's overhead cannot be covered at your cost structure). Raising margin here lowers "
            "win_rate and deepens the loss.\n"
            "- avg_profit_per_participation_usd is the bottom line combining both levers — your net result "
            "per auction bid on. Judge whether the policy is working by this number together with whether "
            "win_rate sits near, above, or below roughly 1/(competitors + 1). These are diagnostic "
            "signals about where your margin currently sits, not targets to hit.\n"
        )
        strategy_memo_rule = ""
        if 'llm_strategy_memo' in agent_state:
            strategy_memo_rule = (
                "- The notes above are your own, written and revised by you after reviewing your "
                "actual results. Treat the strategy section as a commitment you already made: "
                "follow it on this auction rather than re-deriving a policy from scratch. That "
                "consistency is the point — a policy re-decided from zero every auction drifts, "
                "and drifting is what the notes exist to prevent. If this auction's state "
                "genuinely contradicts your strategy, you may depart from it, but say which part "
                "you are departing from and why; your next review will check whether that was "
                "justified.\n"
            )
        outcome_history_rule = ""
        if 'llm_outcome_history' in agent_state:
            outcome_history_rule = (
                "- The table of your own past decisions is realized evidence, not a suggestion — it is "
                "what actually happened when you made each choice. Use it to judge your policy rather "
                "than re-deriving one from scratch: which tiers and margins actually won, what each "
                "choice did to your reputation (reputation_before -> reputation_after), and what it did "
                "to your balance. Two patterns are worth checking explicitly before you decide:\n"
                "    • Look at the reputation column cumulatively. A tier that costs you a fraction of a "
                "point per win looks harmless in isolation but compounds across many auctions into a "
                "standing you cannot price your way out of; if your reputation has been drifting one "
                "direction across the whole table, that trend is the result of your repeated tier choice, "
                "not bad luck, and it will continue unless you change that choice.\n"
                "    • Check whether you keep making a decision that the table shows did not work. If the "
                "same tier/margin combination appears repeatedly with poor outcomes, the evidence "
                "outweighs the reasoning that produced it — change it rather than justifying it again.\n"
                "  Rows marked 'pending' have not resolved yet; treat them as unknown, not as losses.\n"
            )
        last_margin_rule = ""
        if 'llm_last_margin_pct' in agent_state:
            if agent_state.get('llm_last_margin_pct') is not None:
                last_margin_rule = (
                    "- last_committed_margin_pct is the margin you actually committed on your most "
                    "recent bid, over the same total cost basis you are pricing against now. Read it "
                    "together with your running results: if your profitability signals "
                    "(avg_profit_per_participation_usd, current_balance_usd) are flat or falling, "
                    "the margin you have been choosing is not working — move deliberately away from it "
                    "rather than re-picking a similar number out of habit. If they are healthy, it is "
                    "a level worth staying near. This is a memory of one past decision, not a target.\n"
                )
            else:
                last_margin_rule = (
                    "- last_committed_margin_pct will show the margin you committed on your previous "
                    "bid once you have placed one; you have not bid yet, so there is no prior margin.\n"
                )
        margin_bound_min = agent_state.get('llm_margin_bound_min')
        margin_bound_max = agent_state.get('llm_margin_bound_max')
        margin_bound_rule = ""
        if margin_bound_min is not None and margin_bound_max is not None:
            margin_bound_rule = (
                f"- Your margin M% over total_cost_basis_micro_usdc must stay within "
                f"[{float(margin_bound_min) * 100:g}%, {float(margin_bound_max) * 100:g}%] for every bid — "
                f"a hard boundary on your bidding policy, not a suggestion. Choose freely within that range "
                f"based on your own reasoning.\n"
            )
        if include_effort_breakdown:
            profit_formula = (
                "Profit per auction = bid − per_tier_execution_cost_micro[chosen_tier] − non_execution_cost_per_participation_micro. "
                "This uses your chosen tier's TRUE execution cost — it is the real profit or loss you "
                "realize on that auction, regardless of what you priced the bid off. You do not have to "
                "price the bid off this same number: you may anchor your bid to a cheaper reference cost "
                "instead — e.g. min(medium-tier cost, chosen-tier cost) — while still delivering the "
                "(more expensive) chosen tier. This deliberately bids below true cost: a real loss on this "
                "auction, taken as an investment when step 1 concluded the reputation payoff justifies it, "
                "so the bid stays competitive instead of pricing you out of the auction entirely. "
                "Note that total_cost_basis_micro_usdc uses the aggregate EMA execution cost, not the chosen tier's cost."
            )
            cost_floor_note = (
                "per_tier_execution_cost_micro[chosen_tier] + non_execution_cost_per_participation_micro "
                "(your chosen tier's TRUE break-even floor — bidding below it is a real loss on this "
                "auction, only worth it as a deliberate reputation investment, not a mistake to avoid by default)"
            )
            reasoning_protocol = (
                "- Reasoning protocol:\n"
                "  1. Reputation/effort trade-off check (do this BEFORE choosing a tier): for each tier, "
                "compute its quality delta = per_tier_delivered_quality_mean[tier] − your current "
                "reputation. A positive delta raises your reputation if you win with that tier; a "
                "negative delta lowers it — this is not a one-time effect, it shifts the reputation you "
                "carry into every future auction. reputation_weight (rw) tells you how much a reputation "
                "point is worth right now: at high rw, a sustained reputation gain or loss compounds into "
                "a large future score advantage or disadvantage; at low rw it barely matters. "
                "A positive delta is NOT by itself a reason to invest — weigh two more things: "
                "(a) SIZE — a delta of 1-2 points is a much smaller prize than 5+, and a small delta may "
                "not justify a large deliberate loss (see step 2); a delta that has shrunk from previous "
                "auctions means the prize is running out even if it is still positive. "
                "(b) STANDING vs competitor_reputations — score is decided by RELATIVE reputation, not an "
                "absolute level. If you are already at or above the top of competitor_reputations, further "
                "gains buy little additional score even when a tier's own delta is positive — you are "
                "already ahead, and the rational move is to stop paying to extend a lead you do not need, "
                "and instead take the cost-minimizing tier and harvest profit. Investing is for closing a "
                "gap or building a lead you do not yet have, not for padding one you already hold. Write: "
                "\"Reputation check: rep={your_reputation} vs competitors {competitor_reputations}, "
                "rw={reputation_weight}%. Δlow={quality_low−rep}, Δmedium={quality_medium−rep}, "
                "Δhigh={quality_high−rep}. <one sentence: weighing delta size AND standing vs competitors, "
                "does this favor investing above the cost-minimizing tier, or harvesting at it — and why>."
                "\"\n"
                "  2. Choose your effort tier and a pricing basis for your bid: using your step-1 "
                "conclusion, pick the tier to deliver. Then choose what to price the bid against — either "
                "that tier's own true cost (per_tier_execution_cost_micro[chosen_tier], the conservative "
                "choice) or a cheaper reference such as min(medium-tier cost, chosen-tier cost) (the "
                "investment choice: it keeps the bid competitive when step 1 favored a tier above the "
                "cost-minimizing one, at the cost of a real loss on this auction if you win). Then pick "
                "margin M% over whichever basis you chose. Also draw on per_tier_execution_cost_micro, "
                "current_balance_usd, avg_profit_per_participation_usd, avg_gross_margin_per_win_usd, "
                "bids_placed_count, wins_count, win_rate, and competitor_reputations. "
                "Do NOT compute whether your score can beat a specific competitor — you do not know "
                "if they will bid or at what price. That is different from asking whether your "
                "reputation stands meaningfully above or below competitor_reputations, which you do "
                "know: the score formula weights reputation and price by rw/(100-rw), so a reputation "
                "lead over the field can be worth real bid headroom above your default margin, and a "
                "deficit can call for a sharper price than default — weigh this before defaulting to "
                "your usual M, without predicting any single competitor's specific bid. "
                "Write: \"Tier: {tier}. Pricing basis: {basis_choice} = {basis_value} micro-USDC "
                "(true tier cost = {true_cost} micro-USDC). Standing vs field: <does your reputation "
                "lead/deficit over competitor_reputations argue for more or less margin than your "
                "default, and why>. Target margin: M% → bid = {basis_value} × "
                "(1 + M/100) = {amount} micro-USDC. If I win: {amount} − {true_cost} − overhead = "
                "{expected_gain_or_loss}.\"\n"
                "  3. Check before committing: is current_balance_usd negative? If so, look at "
                "avg_gross_margin_per_win_usd. If it is positive your wins already pay off and the "
                "deficit is from losing too many auctions — lowering your margin (not your tier) to "
                "win more helps, raising it hurts. Only if avg_gross_margin_per_win_usd is at or below "
                "zero should you raise your margin. Revise M now if step 2 missed this — do not let this "
                "override your step-1 reputation conclusion by silently downgrading tier instead. An "
                "occasional dip from a deliberate step-2 investment is expected. A decline that keeps "
                "deepening auction after auction while you keep choosing the same investment tier is not "
                "expected — it means go back to step 1 and re-check standing vs competitor_reputations: if "
                "you are already ahead, the investment has done its job and repeating it is a sunk-cost "
                "habit, not a reasoned choice; downgrade toward the cost-minimizing tier now.\n"
                "  4. Commit: call place_bid with your bid_amount and chosen effort_tier. "
                "Competitors' bids are sealed and unknown — choose a tier and margin you would sustain "
                "as a repeated policy and place the bid. Do not wait for a guarantee of winning."
            )
        else:
            profit_formula = (
                "Profit per auction = bid − estimated_execution_cost_micro − non_execution_cost_per_participation_micro. "
                "The break-even floor is execution_cost + non_execution_cost; bidding at or below this floor loses money even on a win."
            )
            cost_floor_note = (
                "total_cost_basis_micro_usdc (the pre-computed break-even floor — "
                "bidding at or below this loses money even on a win)"
            )
            reasoning_protocol = (
                "- Reasoning protocol:\n"
                "  1. State your margin target: reason about what margin M% over "
                "total_cost_basis_micro_usdc maximizes your long-term profit, using all available "
                "state — current_balance_usd, avg_profit_per_participation_usd, "
                "avg_gross_margin_per_win_usd, "
                "bids_placed_count, wins_count, win_rate, and competitor_reputations. "
                "Do NOT compute whether your score can beat a specific competitor — you do not know "
                "if they will bid or at what price. That is different from asking whether your "
                "reputation stands meaningfully above or below competitor_reputations, which you do "
                "know: the score formula weights reputation and price by rw/(100-rw), so a reputation "
                "lead over the field can be worth real bid headroom above your default margin, and a "
                "deficit can call for a sharper price than default — weigh this before defaulting to "
                "your usual M, without predicting any single competitor's specific bid. "
                "Write: \"Standing vs field: <does your reputation lead/deficit over "
                "competitor_reputations argue for more or less margin than your default, and why>. "
                "Target margin: M% → bid = <total_cost_basis_micro> × (1 + M/100) = <amount> micro-USDC.\"\n"
                "  2. Check before committing: is current_balance_usd negative? If so, look at "
                "avg_gross_margin_per_win_usd. If it is positive your wins already pay off and the "
                "deficit is from losing too many auctions — lowering your margin to win more helps, "
                "raising it hurts. Only if avg_gross_margin_per_win_usd is at or below zero should you "
                "raise your margin. Revise M now if step 1 missed this.\n"
                "  3. Commit: call place_bid at your target bid. Competitors' bids are sealed and unknown — "
                "choose a margin you would sustain as a repeated policy and place the bid. "
                "Do not wait for a guarantee of winning."
            )
    else:
        profit_formula = "Profit = bid - execution_cost_micro."
        balance_recovery_rule = ""
        margin_bound_rule = ""
        last_margin_rule = ""
        outcome_history_rule = ""
        strategy_memo_rule = ""
        cost_floor_note = "your execution cost (a lower bound — bidding here means working for nothing)"
        reasoning_protocol = (
            "- Reason about your bid analytically: use the score formula above to understand "
            "how your bid and reputation translate to a score relative to plausible competitors. "
            "The formula and all inputs are already in your state — no tool call is needed. "
            "Choose a bid that balances margin and competitiveness, then call place_bid."
        )

    return f"""You are bidding agent {agent_id} in a decentralized service marketplace that runs first-price sealed-bid reverse scoring auctions. Multiple providers compete; the highest score wins and must deliver the service at its committed bid price.

How winning works (higher score wins):
  score = ( rw * reputation + (100 - rw) * price_savings ) / 100
  price_savings = (max_price - bid) / max_price            # lower bid → higher score
Both a lower bid and a higher reputation raise your score; rw (reputation_weight) sets the trade-off. Bids must not exceed max_price.

Auction protocol — IMPORTANT:
- All bids are sealed (commit-reveal): you submit a hash now; the actual amount is revealed automatically later.
- You have EXACTLY ONE chance to bid per auction. Once committed, the bid is locked — you cannot change, cancel, or improve it.
- Competitors' bids are hidden during the commit phase; you cannot see what others have bid.
- After you commit a bid, this auction will not appear in your future reasoning cycles — decide now.

Your objective: maximize your LONG-TERM profit.
{objective_block}

Your current state:
{state_block}
{outcome_history_block}{strategy_memo_block}
Auctions you may bid on:
{auctions_block}

Tools:
- place_bid({place_bid_args}): seal your commitment on-chain — ONE chance per auction, cannot be changed. You get no immediate win/loss feedback; outcome is determined after the automatic reveal phase.

Rules:
- Bid amounts are integer micro-USDC (6 decimals). {profit_formula}
- Two values will draw your attention: {cost_floor_note} and max_price (an upper bound — bidding here means you almost never win). Both are boundaries, not targets. The profitable bid lies between them, and where it lies depends on the reputation-weighting and competition, not on either boundary.
- Your bid determines both your profit if you win (bid − cost) and your chance of winning (lower bids score higher and win more often, but at thinner margins). Reason about where this tradeoff is most favorable for sustained profit: a bid near your cost almost always wins but earns almost nothing; a bid near max_price earns a lot but rarely wins. Neither extreme maximizes long-term profit. Decide the bid that you would be willing to commit to as a repeated policy — one that is sustainable if this same situation recurs many times.
{reasoning_protocol}
- Exactly one commitment per auction, ever — re-committing always fails on-chain.
- Each auction shows min_score_tick: the minimum bid change that shifts your score by 1 unit at equal reputation. Bids closer together than min_score_tick yield identical scores.
{balance_recovery_rule}{strategy_memo_rule}{outcome_history_rule}{last_margin_rule}{margin_bound_rule}{action_rule}- If no auction is worth bidding on, do not bid and briefly state why.
- You are fully autonomous — there is no human to confirm your actions. Once you have decided on a bid, call place_bid immediately without asking for permission or confirmation.

Decide and act now."""


def get_prompt(architecture: str, agent_state: Dict[str, Any]) -> str:
    """
    Generate prompt for specified architecture.
    
    Args:
        architecture: Architecture identifier ("1", "2", "3", etc.)
        agent_state: Full provider state for the current reasoning cycle
    
    Returns:
        Formatted prompt string
    """
    if architecture == "1":
        return _prompt_arch_1(agent_state)
    elif architecture == "2":
        return _prompt_arch_2(agent_state)
    elif architecture == "3":
        return _prompt_arch_3(agent_state)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _prompt_arch_1(agent_state: Dict[str, Any]) -> str:
    """Architecture 1 (LLM minimal, isolated): current auction state only."""
    return _build_core_prompt(
        agent_state,
        include_economics=False,
        include_effort_breakdown=False,
    )


def _prompt_arch_2(agent_state: Dict[str, Any]) -> str:
    """Architecture 2 (LLM strategic, one-way): + own economics / history."""
    return _build_core_prompt(
        agent_state,
        include_economics=True,
        include_effort_breakdown=False,
    )


def _prompt_arch_3(agent_state: Dict[str, Any]) -> str:
    """Architecture 3 (LLM effort-aware, two-way): + per-tier cost/quality and the effort-tier action."""
    return _build_core_prompt(
        agent_state,
        include_economics=True,
        include_effort_breakdown=True,
    )
