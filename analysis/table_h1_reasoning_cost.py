#!/usr/bin/env python3
"""H1 — what the LLM's reasoning cost, and whether spending more helped (table).

rw = 20, 1 vs 1, against adaptive, 40 seeds per cell. Cost is what OpenRouter
billed for the pricing calls (`usage.cost`), per run of 30 auctions. "LLM net"
already has that cost subtracted; "before reasoning cost" adds it back, so the
two columns separate what the reasoning earned from what it cost.

    python -m analysis.table_h1_reasoning_cost
"""
import json
import statistics as st

from agents.common import paths
from analysis.data import _groups, final_net, ok_runs, paired_difference

# (variant, model, cell, LLM group)
ROWS = [
    ("LLM-Base",    "gpt-oss-20b",  "S91_var_R0_vs_s3",         "X_llm_R0"),
    ("LLM-Band",    "gpt-oss-20b",  "S91_var_R1a_vs_s3",        "X_llm_R1a"),
    ("LLM-History", "gpt-oss-20b",  "S91_var_R1c_vs_s3",        "X_llm_R1c"),
    ("LLM-Reflect", "gpt-oss-20b",  "S91_var_R2_vs_s3",         "X_llm_R2"),
    ("LLM-History", "gpt-4o-mini",  "S91_stage3_R1c_gpt4omini", "X_llm_R1c"),
    ("LLM-History", "gpt-5.4-mini", "S91_stage3_R1c_gpt54mini", "X_llm_R1c"),
]


def reasoning(cell, group):
    cost, calls, tokens = [], [], []
    for row in ok_runs(cell):
        pid = _groups(row)[group][0]
        f = json.loads((paths.resolve(row["run_dir"]) / f"provider_{pid}_status.json").read_text())["financials"]
        cost.append(f.get("reasoning_llm_costs") or 0.0)
        calls.append(f.get("reasoning_llm_calls") or 0)
        tokens.append((f.get("reasoning_input_tokens") or 0) + (f.get("reasoning_output_tokens") or 0))
    return st.mean(cost), st.mean(calls), st.mean(tokens)


def main():
    head = ["Variant", "Model", "Cost / run (USD)", "Calls / run", "Tokens / run",
            "LLM net (USD)", "Before reasoning cost", "Δ vs adaptive [95% CI]"]
    body = []
    for name, model, cell, group in ROWS:
        cost, calls, tokens = reasoning(cell, group)
        net = st.mean(final_net(cell, group))
        d = paired_difference(cell, group, "B_s3")
        star = "*" if d.excludes_zero else ""
        body.append([name, model, f"{cost:.4f}", f"{calls:.0f}", f"{tokens / 1000:.0f}k",
                     f"{net:+.3f}", f"{net + cost:+.3f}",
                     f"{d.mean:+.3f} [{d.lo:+.2f}, {d.hi:+.2f}]{star}"])

    print("Markdown:\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---", "---"] + ["---:"] * (len(head) - 2)) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")
    print("\n\\* 95% CI excludes 0.")

    print("\nLaTeX:\n")
    print("\\begin{tabular}{ll" + "r" * (len(head) - 2) + "}\n\\toprule")
    print(" & ".join(h.replace("Δ", "$\\Delta$") for h in head) + " \\\\\n\\midrule")
    for r in body:
        print(" & ".join(c.replace("%", "\\%") for c in r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
