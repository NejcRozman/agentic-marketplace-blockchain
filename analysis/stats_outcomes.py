#!/usr/bin/env python3
"""Outcome statistics quoted in the Results and Discussion, printed by section number.

One definition throughout, the one fig_rq2_mechanism uses: a per-job
quantity is first summarised within a seed (mean, or median for margins), and
the reported value is the mean of these per-seed values over the 40 seeds;
seeds in which the group won nothing are skipped. Win share is the share of the
30 auctions the group won. Realised margin = price / execution cost − 1.
Decline rate = declines / (declines + bids attempted), from the agent's log
(data.decline_rate); conditions not reported in the paper are left out.
Gas share = mean gas / (mean net balance + mean gas).

    python -m analysis.stats_outcomes
"""
import json
import statistics as st

from agents.common import paths
from analysis.data import (ADAPTIVE_BID_LINE, EXEC_COST_LINE, LEDGER, NOT_IN_FAMILY, decline_rate, final_net,
                           ok_runs, outcomes, provider_groups, run_metrics)
from analysis.dataset import requirement_means


def per_seed(cell, group, field, summary=st.mean):
    vals = [summary(getattr(o, field)) for o in outcomes(cell, group) if getattr(o, field)]
    return st.mean(vals)


def gas_share(cell, group):
    """Gas as a share of earnings before gas (net + gas), means over seeds."""
    gas = st.mean(st.mean(s["financials"]["gas_costs"] for s in agent_status(row, group))
                  for row in ok_runs(cell))
    return gas / (gas + st.mean(final_net(cell, group)))


def agent_status(row, group):
    return [json.loads((paths.resolve(row["run_dir"]) / f"provider_{pid}_status.json").read_text())
            for pid in provider_groups(row)[group]]


def underestimated(cell, group):
    """Share of Adaptive's bids, and of its won jobs, whose true cost exceeded its
    execution-cost estimate (the running average in force when it bid)."""
    truth = requirement_means()
    bids = bids_under = won = won_under = 0
    for row in ok_runs(cell):
        pid = provider_groups(row)[group][0]
        auctions = {int(a["auction_id"]): a for a in run_metrics(row)["auctions"]}
        estimate = None
        for line in (paths.resolve(row["run_dir"]) / f"provider_agent_{pid}.log").open(errors="ignore"):
            m = EXEC_COST_LINE.search(line)
            if m:
                estimate = float(m.group(1))
            m = ADAPTIVE_BID_LINE.search(line)
            if m and estimate:
                a = auctions[int(m.group(1))]
                under = truth[a["sampled_requirement_id"]] > estimate
                bids += 1
                bids_under += under
                if a.get("winner_id") == pid:
                    won += 1
                    won_under += under
    return bids_under / bids, won_under / won


def win(cell, group):
    return st.mean(o.win_share for o in outcomes(cell, group))


def rep(cell, group):
    return st.mean(o.final_reputation for o in outcomes(cell, group))


def line(label, value, fmt):
    print(f"  {label:58s} {format(value, fmt)}")


def main():
    print("6.1 Heuristic baselines")
    line("Random's win share vs Adaptive (w = 0.2)", win("S91_ladder_zi_vs_adaptive", "X_zi"), ".1%")
    line("Adaptive's median realised margin vs Random",
         per_seed("S91_ladder_zi_vs_adaptive", "B_s3", "job_margin", st.median), ".2f")
    c = "S91_ladder_band_vs_adaptive"
    line("Random-in-band's win share vs Adaptive (w = 0.2)", win(c, "X_band"), ".1%")
    line("  mean price: Random-in-band", per_seed(c, "X_band", "job_price"), ".3f")
    line("  mean price: Adaptive", per_seed(c, "B_s3", "job_price"), ".3f")
    c = "S91_ladder_band_vs_adaptive_rw80"
    line("w = 0.8: win share Adaptive", win(c, "B_s3"), ".1%")
    line("w = 0.8: win share Random-in-band", win(c, "X_band"), ".1%")
    line("w = 0.8: mean price Adaptive", per_seed(c, "B_s3", "job_price"), ".3f")
    line("w = 0.8: mean price Random-in-band", per_seed(c, "X_band", "job_price"), ".3f")
    comps = ["S91_comp_1band_vs_3adaptive", "S91_comp_2v2_band_vs_adaptive", "S91_comp_3band_vs_1adaptive"]
    line("four providers: largest per-provider mean net balance",
         max(st.mean(final_net(c, g)) for c in comps for g in ("X_band", "B_s3")), ".3f")

    print("6.2 Price-setting")
    line("Adaptive's win share vs LLM-Base (m_min = 0.20)", win("S91_var_R0_vs_s3", "B_s3"), ".1%")
    line("Adaptive's win share vs LLM-Base (m_min = 0)", win("S91_var_R0_vs_s3_floor0", "B_s3"), ".1%")
    c = "S91_disc_R1c_rw80"
    line("w = 0.8: win share LLM-History", win(c, "X_llm_R1c"), ".1%")
    line("w = 0.8: win share Adaptive", win(c, "B_s3"), ".1%")
    line("w = 0.8: mean price Adaptive", per_seed(c, "B_s3", "job_price"), ".3f")
    line("w = 0.8: mean price LLM-History", per_seed(c, "X_llm_R1c", "job_price"), ".3f")

    print("6.4 Sources of the advantage (gpt-oss-20b)")
    for cell, opp, name in (("A_gptoss20b_xd_vs_h", "B_none", "Adaptive"),
                            ("A_gptoss20b_xd_vs_rguarded", "B_r_guarded", "Regression")):
        line(f"w = 0.2 vs {name}: cost of jobs won, LLM-Estimate", per_seed(cell, "X_decomposed", "job_cost"), ".3f")
        line(f"w = 0.2 vs {name}: cost of jobs won, {name}", per_seed(cell, opp, "job_cost"), ".3f")
        line(f"w = 0.2 vs {name}: price, LLM-Estimate", per_seed(cell, "X_decomposed", "job_price"), ".3f")
        line(f"w = 0.2 vs {name}: price, {name}", per_seed(cell, opp, "job_price"), ".3f")
        line(f"w = 0.2 vs {name}: median margin, LLM-Estimate",
             per_seed(cell, "X_decomposed", "job_margin", st.median), ".0%")
        line(f"w = 0.2 vs {name}: median margin, {name}", per_seed(cell, opp, "job_margin", st.median), ".0%")
        line(f"w = 0.2 vs {name}: final reputation, LLM-Estimate", rep(cell, "X_decomposed"), ".2f")
        line(f"w = 0.2 vs {name}: final reputation, {name}", rep(cell, opp), ".2f")
    line("win share of LLM-Estimate vs Adaptive, w = 0", win("B_gptoss20b_rw0_xd_vs_h", "X_decomposed"), ".1%")
    line("win share of LLM-Estimate vs Adaptive, w = 0.8", win("B_gptoss20b_rw80_xd_vs_h", "X_decomposed"), ".1%")
    line("Adaptive's mean price, w = 0", per_seed("B_gptoss20b_rw0_xd_vs_h", "B_none", "job_price"), ".3f")
    line("Adaptive's mean price, w = 0.8", per_seed("B_gptoss20b_rw80_xd_vs_h", "B_none", "job_price"), ".3f")
    bid_share, won_share = underestimated("A_gptoss20b_xd_vs_h", "B_none")
    line("Adaptive vs LLM-Estimate, w = 0.2: bids on underestimated jobs", bid_share, ".0%")
    line("Adaptive vs LLM-Estimate, w = 0.2: won jobs it underestimated", won_share, ".0%")

    print("6.5 Decline rates of LLM-Estimate (declines / auctions entered or declined)")
    cells = sorted({json.loads(l)["cell"] for l in LEDGER.read_text().splitlines() if l.strip()}
                   - NOT_IN_FAMILY)
    for c in cells:
        groups = provider_groups(ok_runs(c)[0])
        if "X_decomposed" in groups:
            line(c, decline_rate(c, "X_decomposed"), ".1%")

    print("7.4 Gas as a share of earnings before gas")
    for cell, groups in (("A_gptoss20b_xd_vs_h", ("X_decomposed", "B_none")),
                         ("A_gptoss20b_xd_vs_rguarded", ("X_decomposed", "B_r_guarded"))):
        for g in groups:
            line(f"{cell} {g}", gas_share(cell, g), ".1%")
    shares = []
    for c in cells:
        for g in provider_groups(ok_runs(c)[0]):
            gas = st.mean(st.mean(s["financials"]["gas_costs"] for s in agent_status(r, g)) for r in ok_runs(c))
            before = gas + st.mean(final_net(c, g))
            shares.append((gas / before if before > 0 else float("inf"), c, g))
    lo, hi = min(shares), max(shares)
    line(f"lowest over all strategies and conditions ({lo[1]} {lo[2]})", lo[0], ".1%")
    line(f"highest ({hi[1]} {hi[2]}; >100% = gas exceeds earnings)", hi[0], ".1%")


if __name__ == "__main__":
    main()
