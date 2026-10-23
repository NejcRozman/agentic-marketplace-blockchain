#!/usr/bin/env python3
"""RQ1: LLM minus heuristic, final net balance, across conditions.

LLM-History (the best variant at w = 0.2) is shown in every condition, and
LLM-Base is added as a contrast where it was run. Each point is the mean
per-seed difference in final net balance over 40 seed-paired runs; whiskers are
95% t intervals; a filled marker means the difference is significant
(Holm-adjusted within the RQ1 family).

  a) reputation weight (and Adaptive with m_min = 0), vs Adaptive, 1:1, gpt-oss-20b
  b) heuristic opponent, w = 0.2, 1:1 and 1:3, gpt-oss-20b
  c) model, vs Adaptive, w = 0.2, 1:1

    python -m analysis.fig_rq1_llm_conditions
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import paired_difference, significant

# Marker shape distinguishes the two LLM variants (both use the LLM colour).
MARKERS = {"history": "o", "base": "s"}
OFFSET = 0.16          # vertical separation when both variants share a row

# Each row: label, {variant: (cell, llm group, opponent group)}
PANEL_A = [
    ("w = 0.2", {"history": ("S91_var_R1c_vs_s3", "X_llm_R1c", "B_s3"),
                 "base":    ("S91_var_R0_vs_s3",  "X_llm_R0",  "B_s3")}),
    # Adaptive with m_min = 0: the same pricing range as the unconstrained LLM-Base
    ("w = 0.2, $m_{\\min}=0$", {"base": ("S91_var_R0_vs_s3_floor0", "X_llm_R0", "B_s3")}),
    ("w = 0.5", {"history": ("S91_stage4_R1c_rw50", "X_llm_R1c", "B_s3")}),
    ("w = 0.8", {"history": ("S91_disc_R1c_rw80", "X_llm_R1c", "B_s3"),
                 "base":    ("S91_disc_R0_rw80",  "X_llm_R0",  "B_s3")}),
]
PANEL_B = [
    ("vs Random",         {"history": ("S91_stage2_R1c_vs_zi",   "X_llm_R1c", "B_zi")}),
    ("vs Random-in-band", {"history": ("S91_stage2_R1c_vs_band", "X_llm_R1c", "B_band")}),
    ("vs Adaptive",       {"history": ("S91_var_R1c_vs_s3",      "X_llm_R1c", "B_s3")}),
    ("vs 3 Adaptive",     {"history": ("S91_disc_R1c_1v3",       "X_llm_R1c", "B_s3"),
                           "base":    ("S91_disc_R0_1v3",        "X_llm_R0",  "B_s3")}),
]
PANEL_C = [
    ("gpt-oss-20b",  {"history": ("S91_var_R1c_vs_s3",        "X_llm_R1c", "B_s3")}),
    ("gpt-4o-mini",  {"history": ("S91_stage3_R1c_gpt4omini", "X_llm_R1c", "B_s3")}),
    ("gpt-5.4-mini", {"history": ("S91_stage3_R1c_gpt54mini", "X_llm_R1c", "B_s3")}),
]


def forest(ax, rows):
    c = style.color("llm")
    out = []
    ys = list(range(len(rows)))[::-1]
    for y, (label, variants) in zip(ys, rows):
        both = len(variants) > 1
        for variant, (cell, llm, opp) in variants.items():
            d = paired_difference(cell, llm, opp)
            yy = y + (OFFSET if variant == "history" else -OFFSET) if both else y
            ax.errorbar(d.mean, yy, xerr=[[d.mean - d.lo], [d.hi - d.mean]], fmt="none",
                        ecolor=c, elinewidth=style.ERR_LW, capsize=style.CAP, capthick=style.CAP_LW, zorder=2)
            ax.plot(d.mean, yy, MARKERS[variant], ms=style.MS, mec=c, mew=style.MEW,
                    mfc=c if significant(cell) else "white", zorder=3)
            out.append((label, style.LLM_VARIANTS[variant][0], d, significant(cell)))
    ax.axvline(0, color=style.ZERO, lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=style.INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for x, ha, text in ((0.0, "left", "← heuristic"), (1.0, "right", "LLM →")):
        ax.text(x, -0.22, text, transform=ax.transAxes, ha=ha, va="top",
                fontsize=6.5, color=style.INK_2)
    ax.text(0.5, -0.35, "Δ net balance (USD)", transform=ax.transAxes, ha="center",
            va="top", fontsize=8, color=style.INK_2)
    return out


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 2.3), sharex=True)
    results = []
    for letter, ax, rows in zip("abc", axes, (PANEL_A, PANEL_B, PANEL_C)):
        results += [(letter, *r) for r in forest(ax, rows)]
        ax.text(0.5, -0.52, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=style.INK)

    c = style.color("llm")
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc=c, mec=c, mew=style.MEW,
                   label=style.LLM_VARIANTS["history"][0]),
        plt.Line2D([], [], marker="s", ls="", ms=style.MS, mfc=c, mec=c, mew=style.MEW,
                   label=style.LLM_VARIANTS["base"][0]),
        # Neutral grey so the fill key doesn't read as another variant.
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc=style.INK_2, mec=style.INK_2, mew=style.MEW,
                   label="significant (Holm)"),
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc="white", mec=style.INK_2, mew=style.MEW,
                   label="not significant"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    style.save(fig, "rq1_llm_conditions")

    print("LLM − heuristic, final net balance (USD), mean [95% CI], n")
    for letter, label, variant, d, sig in results:
        print(f"  {letter}) {label:<22} {variant:<12} {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"
              f"  n={d.n}{'  * (Holm)' if sig else ''}")


if __name__ == "__main__":
    main()
