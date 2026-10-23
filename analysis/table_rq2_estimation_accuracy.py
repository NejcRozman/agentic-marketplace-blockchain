#!/usr/bin/env python3
"""H2 — how accurately each agent estimated the execution cost of a job (table).

In-market estimates at rw = 20, 1 vs 1, 40 seeds per cell. Scoring follows the
S8 convention (experiments/analysis/in_market_sigma.py): true cost = the
requirement's mean over its three recorded executions; residual =
log(estimate / true cost).

  σ             spread of the residuals after removing their mean (a constant
                bias is learnable from the agent's own history, so it is not
                charged) — lower is better
  σ excl. cold  the same, without fallback estimates, where the estimator had
                no fit to apply and used the moving average: LLM-Estimate before
                its first calibration or when its call failed or missed the
                deadline; Regression while its guard withholds the fit or it has
                no usable observation (see data.estimates)
  correlation   Pearson correlation of log estimate with log true cost — does
                the estimate tell cheap jobs from expensive ones? Adaptive gives
                every job its current running average, so ~0 by construction
  bias          geometric-mean ratio estimate / true cost (1 = unbiased)

Each agent is pooled over its rw = 20 cells against both opponents. Oracle knows
the true cost, so σ = 0 and correlation = 1 by definition (shown as reference).

    python -m analysis.table_rq2_estimation_accuracy
"""
import math
import statistics as st

from analysis.data import estimates
from analysis.dataset import corr, debiased_sigma, requirement_means


# (agent, model, [(cell, group), ...])
ROWS = [
    ("LLM-Estimate", "gpt-oss-20b",  [("A_gptoss20b_xd_vs_h", "X_decomposed"), ("A_gptoss20b_xd_vs_rguarded", "X_decomposed")]),
    ("LLM-Estimate", "gpt-4o-mini",  [("E_xd_vs_h_v3", "X_decomposed"), ("E_xd_vs_r_v3", "X_decomposed")]),
    ("LLM-Estimate", "gpt-5.4-mini", [("A_gpt54mini_xd_vs_h", "X_decomposed"), ("A_gpt54mini_xd_vs_rguarded", "X_decomposed")]),
    ("Regression",   "—",            [("A_gptoss20b_xd_vs_rguarded", "B_r_guarded"), ("A_gpt54mini_xd_vs_rguarded", "B_r_guarded"),
                                      ("E_xd_vs_r_v3", "R_guarded")]),
    ("Adaptive",     "—",            [("A_gptoss20b_xd_vs_h", "B_none"), ("A_gpt54mini_xd_vs_h", "B_none"),
                                      ("E_xd_vs_h_v3", "B_none")]),
]


def score(ests, truth):
    pairs = [(e.predicted, truth[e.requirement], e.cold_start) for e in ests
             if e.requirement in truth and e.predicted > 0]
    res = [math.log(p / t) for p, t, _ in pairs]
    warm = [math.log(p / t) for p, t, c in pairs if not c]
    return {
        "n": len(res), "cold": sum(c for *_, c in pairs),
        "sigma": debiased_sigma(res), "sigma_warm": debiased_sigma(warm),
        "corr": corr([math.log(p) for p, _, _ in pairs], [math.log(t) for _, t, _ in pairs]),
        "corr_warm": corr([math.log(p) for p, _, c in pairs if not c], [math.log(t) for _, t, c in pairs if not c]),
        "bias": math.exp(st.mean(res)),
    }


def main():
    """Paper table: fallback share, then σ, correlation and bias without the
    fallback (cold) estimates, as Section 5.6 defines the measure."""
    truth = requirement_means()
    head = ["Estimator", "Model", "Fallback", "σ", "Correlation", "Bias"]
    body = [["Oracle", "—", "—", "0", "1", "1"]]
    for agent, model, cells in ROWS:
        ests = [e for cell, group in cells for e in estimates(cell, group)]
        s = score(ests, truth)
        warm = [e for e in ests if not e.cold_start and e.requirement in truth and e.predicted > 0]
        bias_warm = math.exp(st.mean(math.log(e.predicted / truth[e.requirement]) for e in warm))
        body.append([agent, model, f"{100 * s['cold'] / s['n']:.0f}%", f"{s['sigma_warm']:.2f}",
                     f"{s['corr_warm']:+.2f}", f"{bias_warm:.2f}"])

    print("Markdown:\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---", "---"] + ["---:"] * (len(head) - 2)) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")
    print("\nFallback: share of estimates that fell back to the moving average; σ, correlation and "
          "bias exclude them. σ: SD of log(estimate / true cost); true cost = mean of the request's "
          "three recorded executions.")

    print("\nLaTeX:\n")
    print("\\begin{tabular}{@{}ll" + "r" * (len(head) - 2) + "@{}}\n\\toprule")
    print(" & ".join(h.replace("σ", "$\\sigma$") for h in head) + " \\\\\n\\midrule")
    for r in body:
        print(" & ".join(c.replace("%", "\\%").replace("—", "--") for c in r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
