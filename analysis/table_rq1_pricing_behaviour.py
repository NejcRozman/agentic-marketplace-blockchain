#!/usr/bin/env python3
"""RQ1: pricing behaviour of each strategy (table).

w = 0.2, 1 vs 1, gpt-oss-20b, 40 seeds per cell, each LLM variant against
adaptive. Margin = bid / cost basis − 1, the quantity each strategy chooses.

  median margin       median over all bids
  below 20% floor     share of bids under adaptive's minimum margin (0.20)
  raises after win    share of next bids with a higher margin, after a win
  cuts after loss     share of next bids with a lower margin, after a loss;
                      for adaptive, only losses where it was above its floor
                      (at the floor it cannot cut): 20%, or 0 in the matched-
                      floor condition (rows "m_min = 0")
  round margins       share of bids within ±0.5 points of a multiple of 5%
                      (5%, 10%, …); not reported for adaptive, whose margin
                      moves in 5-point steps by construction

Changes of at most 0.5 percentage points count as no change. Thresholds are
applied with a tolerance of 0.05 percentage points: bids are whole micro-USDC,
so a bid placed at exactly 20% over its cost basis can sit a hair below 20% and
must not count as below the floor. Only bids committed on chain count (see
data.bid_sequences). Adaptive is pooled over its four LLM opponents.

Also printed (quoted in the text): share of LLM bids below the agent's own cost
basis, and the 90th-percentile margin of LLM-History and Adaptive at w = 0.8.

    python -m analysis.table_rq1_pricing_behaviour
"""
import statistics as st

from analysis import style
from analysis.data import bid_sequences

EPS = 0.005
FLOOR = 0.20
TOL = 0.0005
MATCHED = "S91_var_R0_vs_s3_floor0"


def percentile(sorted_values, q):
    """Linear-interpolation percentile (numpy's default) of a sorted list."""
    k = (len(sorted_values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def behaviour(seqs, floor=None):
    margins = [b.margin for s in seqs for b in s]
    rw = nw = cl = nl = 0
    for s in seqs:
        for prev, nxt in zip(s, s[1:]):
            d = nxt.margin - prev.margin
            if prev.won:
                nw += 1
                rw += d > EPS
            elif floor is None or prev.margin > floor + 1e-9:
                nl += 1
                cl += d < -EPS
    rounded = sum(abs(m * 100 - 5 * round(m * 20)) <= 0.5 for m in margins)
    return {
        "median": st.median(margins),
        "below": sum(m < FLOOR - TOL for m in margins) / len(margins),
        "below_basis": sum(m < -TOL for m in margins) / len(margins),
        "raise": rw / nw,
        "cut": cl / nl,
        "round": rounded / len(margins),
        "n": len(margins),
    }


def rows():
    out = []
    adaptive = []
    for _, (_, suffix) in style.LLM_VARIANTS.items():
        adaptive += bid_sequences(f"S91_var_{suffix}_vs_s3", "B_s3")
    out.append(("Adaptive", behaviour(adaptive, floor=FLOOR), True))
    # Matched floor: Adaptive with m_min = 0 against LLM-Base (S91_var_R0_vs_s3_floor0).
    out.append(("Adaptive, m_min = 0", behaviour(bid_sequences(MATCHED, "B_s3"), floor=0.0), True))
    for _, (name, suffix) in style.LLM_VARIANTS.items():
        seqs = bid_sequences(f"S91_var_{suffix}_vs_s3", f"X_llm_{suffix}")
        out.append((name, behaviour(seqs), False))
        if suffix == "R0":
            out.append((f"{name}, vs m_min = 0", behaviour(bid_sequences(MATCHED, "X_llm_R0")), False))
    return out


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    data = rows()
    head = ["Strategy", "Median margin", "Bids below 20% floor",
            "Raises after win", "Cuts after loss", "Bids at round margins"]
    body = []
    for name, b, is_adaptive in data:
        body.append([name, pct(b["median"]), pct(b["below"]), pct(b["raise"]),
                     pct(b["cut"]) + ("*" if is_adaptive else ""),
                     "—" if is_adaptive else pct(b["round"])])

    print("Markdown:\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")
    print("\n\\* only losses where adaptive was above its minimum margin; at the floor it cannot cut.")

    print("\nLaTeX:\n")
    print("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule")
    print(" & ".join(head) + " \\\\\n\\midrule")
    for r in body:
        print(" & ".join(c.replace("%", "\\%") for c in r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")

    print("\nbids per strategy: " + ", ".join(f"{n}={b['n']}" for n, b, _ in data))
    print("bids below own cost basis: " + ", ".join(f"{n}={100 * b['below_basis']:.1f}%" for n, b, a in data if not a))
    for group, label in (("X_llm_R1c", "LLM-History"), ("B_s3", "Adaptive")):
        ms = sorted(b.margin for s in bid_sequences("S91_disc_R1c_rw80", group) for b in s)
        print(f"90th-percentile margin at rw = 80, {label}: {pct(percentile(ms, 0.9))} (n={len(ms)})")


if __name__ == "__main__":
    main()
