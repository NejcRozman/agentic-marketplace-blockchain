#!/usr/bin/env python3
"""Dataset v3 summary (table), plus the statistics quoted in the text.

One row per designed synthesis depth and one for the whole dataset. Cost is the
executor's LLM cost of one recorded execution (USD); quality is the evaluator's
rating (0-100); retrieval rounds are summed over a job's prompts.

    python -m analysis.table_dataset
"""
import statistics as st

from analysis.dataset import (DEPTHS, between_within, corr, runs,
                                                share_between)


def row(label, rs):
    c = [r["execution_cost_usd"] for r in rs]
    q = [r["evaluator_rating"] for r in rs]
    k = [r["retrieval_rounds_total"] for r in rs]
    return [label, f"{len({r['requirement_id'] for r in rs})}", f"{len(rs)}",
            f"{st.mean(c):.3f} ({st.stdev(c):.3f})", f"{min(c):.3f}–{max(c):.3f}",
            f"{st.mean(q):.1f} ({st.stdev(q):.1f})", f"{st.mean(k):.1f}"]


def main():
    rs = runs()
    head = ["Depth", "Requests", "Recordings", "Cost, mean (SD)", "Cost range",
            "Rating, mean (SD)", "Searches per job"]
    body = [row(d, [r for r in rs if r["demanded_synthesis_depth"] == d]) for d in DEPTHS]
    body.append(row("All", rs))

    print("Markdown:\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")
    print("\nCost in USD per execution; quality is the evaluator rating (0–100).")

    print("\nLaTeX:\n")
    print("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule")
    print(" & ".join(head) + " \\\\\n\\midrule")
    for i, r in enumerate(body):
        if i == len(body) - 1:
            print("\\midrule")
        print(" & ".join(c.replace("–", "--") for c in r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")

    cost = [r["execution_cost_usd"] for r in rs]
    qual = [r["evaluator_rating"] for r in rs]
    rounds = [r["retrieval_rounds_total"] for r in rs]
    b, w = between_within("execution_cost_usd", "evaluator_rating")
    print("\nStatistics quoted in the text:")
    print(f"  cost spread: {min(cost):.4f}–{max(cost):.4f} USD ({max(cost)/min(cost):.0f}x), "
          f"median {st.median(cost):.4f}, mean {st.mean(cost):.4f}, CV {st.stdev(cost)/st.mean(cost):.2f}")
    print(f"  variance explained by which requirement: cost {100*share_between('execution_cost_usd'):.0f}%, "
          f"retrieval rounds {100*share_between('retrieval_rounds_total'):.0f}%, "
          f"quality {100*share_between('evaluator_rating'):.0f}%")
    print(f"  corr(retrieval rounds, cost) {corr(rounds, cost):+.2f}; corr(rounds, quality) {corr(rounds, qual):+.2f}")
    print(f"  corr(cost, quality): all runs {corr(cost, qual):+.2f}, between requirements {b:+.2f}, "
          f"within requirement {w:+.2f}")
    for s in ("depth", "citations", "clarity", "completeness"):
        print(f"  corr(cost, quality sub-score '{s}') "
              f"{corr(cost, [r['evaluator_quality_scores'][s] for r in rs]):+.2f}")
    models = {(r.get("evaluator_model"), r["service_output"].get("agent_id")) for r in rs}
    print(f"  evaluator model(s): {sorted({m[0] for m in models})}")


if __name__ == "__main__":
    main()
