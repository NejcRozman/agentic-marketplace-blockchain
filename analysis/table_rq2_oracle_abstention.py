#!/usr/bin/env python3
"""RQ2: the value of perfect cost knowledge, and the effect of the decline rule (table).

Oracle knows each job's true mean cost from the dataset: an upper bound on any
cost estimator. The decline rule lets an agent decline an auction when its
believed cost plus overhead exceeds the highest price it has ever won at.

w = 0.2 unless stated, 1 vs 1, 40 seed-paired runs per cell, LLM-Estimate on
gpt-oss-20b. Δ = first agent − second agent, final net balance, 95% t interval.
Declined = share of eligible auctions the first agent declined, counted from
its own log.

    python -m analysis.table_rq2_oracle_abstention
"""
import statistics as st

from analysis import style
from analysis.data import decline_rate, final_net, paired_difference, significant

ORACLE, XD = "X_oracle_bound", "X_decomposed"

# (section, comparison label, decline rule, cell, first group, second group)
ROWS = [
    ("Perfect cost knowledge", "Oracle vs Adaptive",     "on",  "A_oracle_vs_h",              ORACLE, "B_none"),
    ("Perfect cost knowledge", "Oracle vs Adaptive",     "off", "A_oracle_noO2c_vs_h",        ORACLE, "B_none"),
    ("Perfect cost knowledge", "Oracle vs Regression",   "off", "A_oracle_noO2c_vs_rguarded", ORACLE, "B_r_guarded"),
    ("Perfect cost knowledge", "Oracle vs LLM-Estimate", "on",  "A_oracle_vs_xd",             ORACLE, "B_decomposed"),
    ("Perfect cost knowledge", "Oracle vs LLM-Estimate", "off", "A_oracle_noO2c_vs_xd",       ORACLE, "B_decomposed"),
    ("Decline rule",        "LLM-Estimate vs Regression, w = 0.2", "on",  "A_gptoss20b_xd_vs_rguarded", XD, "B_r_guarded"),
    ("Decline rule",        "LLM-Estimate vs Regression, w = 0.2", "off", "C_o2c_off_rw20",             XD, "B_r_guarded"),
    ("Decline rule",        "LLM-Estimate vs Regression, w = 0.8", "on",  "C_o2c_on_rw80",              XD, "B_r_guarded"),
    ("Decline rule",        "LLM-Estimate vs Regression, w = 0.8", "off", "C_o2c_off_rw80",             XD, "B_r_guarded"),
]


def main():
    head = ["Comparison", "Decline rule", "First agent net", "Second agent net",
            "Δ first − second [95% CI]", "First agent declined"]
    body, sections = [], []
    for section, label, rule, cell, a, b in ROWS:
        d = paired_difference(cell, a, b)
        star = "*" if significant(cell) else ""
        body.append([label, rule, f"{st.mean(final_net(cell, a)):+.3f}", f"{st.mean(final_net(cell, b)):+.3f}",
                     f"{d.mean:+.3f} [{d.lo:+.2f}, {d.hi:+.2f}]{star}",
                     f"{100 * decline_rate(cell, a):.1f}%"])
        sections.append(section)

    print("Markdown:\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---", "---"] + ["---:"] * (len(head) - 2)) + "|")
    last = None
    for s, r in zip(sections, body):
        if s != last:
            print("| **" + s + "** |" + " |" * (len(head) - 1))
            last = s
        print("| " + " | ".join(r) + " |")
    print("\n\\* significant (Holm-adjusted, RQ2 family). Net balances in USD per run. Oracle vs Adaptive and vs Regression "
          "used a 10 s commit window (no LLM in either agent); all other cells 20 s.")

    print("\nLaTeX:\n")
    print("\\begin{tabular}{ll" + "r" * (len(head) - 2) + "}\n\\toprule")
    print(" & ".join(h.replace("Δ", "$\\Delta$").replace("−", "$-$") for h in head) + " \\\\")
    last = None
    for s, r in zip(sections, body):
        if s != last:
            print("\\midrule\n\\multicolumn{" + str(len(head)) + "}{l}{\\textit{" + s + "}} \\\\")
            last = s
        print(" & ".join(c.replace("%", "\\%") for c in r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    style.apply()
    main()
