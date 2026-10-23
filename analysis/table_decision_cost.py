#!/usr/bin/env python3
"""Decision cost of the LLM components, for both roles (merged table, Section 6.5).

Cost is what OpenRouter billed for the market module's LLM calls, per run of 30
auctions (w = 0.2, 1 vs 1, 40 seeds per cell). "Net" has that cost subtracted;
"before cost" adds it back. Price-setting rows are against Adaptive; cost-
estimation rows pool cost, calls and tokens over the cells against Adaptive and
Regression, with net and Δ from the cell against Adaptive. Δ = LLM strategy −
Adaptive, 95% t interval; * = significant after Holm (data.holm_table).

    .venv/bin/python -m analysis.table_decision_cost
"""
import json
import statistics as st

from agents.common import paths
from analysis.data import _groups, final_net, ok_runs, paired_difference, significant

XD = "X_decomposed"
PRICING = [  # (strategy, model, cell, LLM group)
    ("LLM-Base",                  "gpt-oss-20b",  "S91_var_R0_vs_s3",         "X_llm_R0"),
    ("LLM-Base, vs $m_{\\min}=0$", "gpt-oss-20b",  "S91_var_R0_vs_s3_floor0",  "X_llm_R0"),
    ("LLM-Band",                  "gpt-oss-20b",  "S91_var_R1a_vs_s3",        "X_llm_R1a"),
    ("LLM-History",               "gpt-oss-20b",  "S91_var_R1c_vs_s3",        "X_llm_R1c"),
    ("LLM-Reflect",               "gpt-oss-20b",  "S91_var_R2_vs_s3",         "X_llm_R2"),
    ("LLM-History",               "gpt-4o-mini",  "S91_stage3_R1c_gpt4omini", "X_llm_R1c"),
    ("LLM-History",               "gpt-5.4-mini", "S91_stage3_R1c_gpt54mini", "X_llm_R1c"),
]
ESTIMATION = [  # (model, cell vs Adaptive, Adaptive group, cell vs Regression)
    ("gpt-oss-20b",  "A_gptoss20b_xd_vs_h", "B_none", "A_gptoss20b_xd_vs_rguarded"),
    ("gpt-4o-mini",  "E_xd_vs_h_v3",        "B_none", "E_xd_vs_r_v3"),
    ("gpt-5.4-mini", "A_gpt54mini_xd_vs_h", "B_none", "A_gpt54mini_xd_vs_rguarded"),
]


def llm_cost(cells, group):
    """Mean billed LLM cost, calls and tokens per run of the group's agent, pooled over the cells."""
    cost, calls, tokens = [], [], []
    for cell in cells:
        for run in ok_runs(cell):
            pid = _groups(run)[group][0]
            f = json.loads((paths.resolve(run["run_dir"]) / f"provider_{pid}_status.json").read_text())["financials"]
            cost.append(f.get("reasoning_llm_costs") or 0.0)
            calls.append(f.get("reasoning_llm_calls") or 0)
            tokens.append((f.get("reasoning_input_tokens") or 0) + (f.get("reasoning_output_tokens") or 0))
    return st.mean(cost), st.mean(calls), st.mean(tokens)


def row(name, model, cost, calls, tokens, cell, group, opp):
    net = st.mean(final_net(cell, group))
    d = paired_difference(cell, group, opp)
    return [name, model, f"{cost:.4f}", f"{calls:.0f}", f"{tokens / 1000:.0f}k", f"{net:+.3f}",
            f"{net + cost:+.3f}", f"{d.mean:+.3f}{'*' if significant(cell) else ''}"]


def main():
    sections = [("Price-setting", [row(n, m, *llm_cost([c], g), c, g, "B_s3") for n, m, c, g in PRICING]),
                ("Cost estimation", [row("LLM-Estimate", m, *llm_cost([ch, cr], XD), ch, XD, gh)
                                     for m, ch, gh, cr in ESTIMATION])]
    head = ["Strategy", "Model", "Cost/run", "Calls/run", "Tokens/run", "Net", "Before cost", "Δ vs Adaptive"]
    print("| " + " | ".join(head) + " |")
    for title, rows in sections:
        print(f"| *{title}* |")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    print("\n* significant (Holm). USD per run of 30 auctions.\n\nLaTeX rows:")
    for title, rows in sections:
        print(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for r in rows:
            r = [c.replace("*", "$^{*}$") for c in r]
            print(" & ".join(r) + " \\\\")


if __name__ == "__main__":
    main()
