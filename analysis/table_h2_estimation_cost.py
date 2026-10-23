#!/usr/bin/env python3
"""H2 — what LLM-Estimate's estimation calls cost, per model, and whether paying more helped (table).

Same columns as the H1 reasoning-cost table (table_h1_reasoning_cost.py), so
the two can be read side by side. rw = 20, 1 vs 1, 40 seeds per cell.

Cost is what OpenRouter billed for the estimation calls (`usage.cost`), per run
of 30 auctions; cost, calls and tokens are pooled over the cells against both
opponents. "Net" is LLM-Estimate's final net balance in the cell against
Adaptive (it already has the call cost subtracted); "before estimation cost"
adds that cost back, separating what the estimates earned from what they cost.
Δ columns are paired differences against each opponent, with 95% t intervals.

    python -m analysis.table_h2_estimation_cost
"""
import json
import statistics as st

from agents.common import paths
from analysis.data import _groups, final_net, ok_runs, paired_difference

XD = "X_decomposed"
# (model, (cell vs Adaptive, Adaptive group), (cell vs Regression, Regression group))
ROWS = [
    ("gpt-oss-20b",  ("A_gptoss20b_xd_vs_h", "B_none"), ("A_gptoss20b_xd_vs_rguarded", "B_r_guarded")),
    ("gpt-4o-mini",  ("E_xd_vs_h_v3",        "B_none"), ("E_xd_vs_r_v3",               "R_guarded")),
    ("gpt-5.4-mini", ("A_gpt54mini_xd_vs_h", "B_none"), ("A_gpt54mini_xd_vs_rguarded", "B_r_guarded")),
]


def calls_cost(cells):
    cost, calls, tokens = [], [], []
    for cell in cells:
        for row in ok_runs(cell):
            pid = _groups(row)[XD][0]
            f = json.loads((paths.resolve(row["run_dir"]) / f"provider_{pid}_status.json").read_text())["financials"]
            cost.append(f.get("reasoning_llm_costs") or 0.0)
            calls.append(f.get("reasoning_llm_calls") or 0)
            tokens.append((f.get("reasoning_input_tokens") or 0) + (f.get("reasoning_output_tokens") or 0))
    return st.mean(cost), st.mean(calls), st.mean(tokens)


def fmt(d):
    return f"{d.mean:+.3f} [{d.lo:+.2f}, {d.hi:+.2f}]{'*' if d.excludes_zero else ''}"


def main():
    head = ["Model", "Cost / run (USD)", "Calls / run", "Tokens / run", "Net (USD)",
            "Before estimation cost", "Δ vs Adaptive [95% CI]", "Δ vs Regression [95% CI]"]
    body = []
    for model, (ch, gh), (cr, gr) in ROWS:
        cost, calls, tokens = calls_cost([ch, cr])
        net = st.mean(final_net(ch, XD))
        body.append([model, f"{cost:.4f}", f"{calls:.0f}", f"{tokens / 1000:.0f}k",
                     f"{net:+.3f}", f"{net + cost:+.3f}",
                     fmt(paired_difference(ch, XD, gh)), fmt(paired_difference(cr, XD, gr))])

    print("Markdown:\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")
    print("\n\\* 95% CI excludes 0. Net is from the cell against Adaptive.")

    print("\nLaTeX:\n")
    print("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule")
    print(" & ".join(h.replace("Δ", "$\\Delta$") for h in head) + " \\\\\n\\midrule")
    for r in body:
        print(" & ".join(r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
