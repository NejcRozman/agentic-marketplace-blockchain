#!/usr/bin/env python3
"""RQ1: net balance over the 30 auctions for the LLM pricing variants against Adaptive.

w = 0.2, 1 vs 1, 40 seed-paired runs per panel, gpt-oss-20b. Row 1: LLM-Base
against Adaptive with its usual 20% floor (a) and with the floor at 0 (b,
matched pricing range); row 2: LLM-Band, LLM-History, LLM-Reflect (c–e). Lines
are the mean across seeds; shading is the 95% CI of that mean. All panels share
one y-scale, because a variant can beat Adaptive by starving it of jobs rather
than by earning more itself.

    python -m analysis.fig_rq1_llm_variant_trajectories
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import N_AUCTIONS, mean_ci, paired_difference, significant, trajectory

ADAPTIVE = "B_s3"

#: (title, cell, LLM group, grid slot). Slots are column spans in a 6-column grid,
#: so every panel has the same width.
PANELS = [
    ("LLM-Base",               "S91_var_R0_vs_s3",        "X_llm_R0",  (0, slice(1, 3))),
    ("LLM-Base, $m_{\\min}=0$", "S91_var_R0_vs_s3_floor0", "X_llm_R0",  (0, slice(3, 5))),
    ("LLM-Band",               "S91_var_R1a_vs_s3",       "X_llm_R1a", (1, slice(0, 2))),
    ("LLM-History",            "S91_var_R1c_vs_s3",       "X_llm_R1c", (1, slice(2, 4))),
    ("LLM-Reflect",            "S91_var_R2_vs_s3",        "X_llm_R2",  (1, slice(4, 6))),
]


def main():
    style.apply(column=True)
    fig = plt.figure(figsize=(style.COL_WIDTH, 3.3))
    gs = fig.add_gridspec(2, 6, hspace=0.95, wspace=1.3)
    xs = list(range(N_AUCTIONS + 1))
    print("Final net balance, mean [95% CI], USD — and paired difference LLM − adaptive")

    axes = []
    for letter, (name, cell, llm_group, (row, cols)) in zip("abcde", PANELS):
        ax = fig.add_subplot(gs[row, cols], sharey=axes[0] if axes else None)
        axes.append(ax)
        for skey, group in (("adaptive", ADAPTIVE), ("llm", llm_group)):
            seeds = trajectory(cell, group)
            per_x = [mean_ci([s[i] for s in seeds]) for i in xs]
            c = style.color(skey)
            ax.fill_between(xs, [p.lo for p in per_x], [p.hi for p in per_x],
                            color=c, alpha=0.18, lw=0)
            ax.plot(xs, [p.mean for p in per_x], color=c, lw=style.LINE_LW)
            end = per_x[-1]
            who = "LLM" if skey == "llm" else "Adaptive"
            print(f"  {cell:<26} {who:<9} {end.mean:+.4f} [{end.lo:+.4f}, {end.hi:+.4f}]  n={end.n}")
        d = paired_difference(cell, llm_group, ADAPTIVE)
        print(f"  {cell:<26} LLM − adaptive: {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"
              f"{'  * (Holm)' if significant(cell) else ''}")

        ax.axhline(0, color=style.ZERO, lw=0.8, zorder=1)
        ax.set_xlim(0, N_AUCTIONS)
        ax.set_xticks([0, 15, 30])
        ax.set_xlabel("Auction")
        ax.set_title(name, loc="center", color=style.INK, fontweight="bold", fontsize=6.5)
        ax.text(0.5, -0.45, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=style.LETTER_FS, color=style.INK)
        if cols.start in (0, 1):
            ax.set_ylabel("Net balance (USD)")

    handles = [plt.Line2D([], [], color=style.color(k), lw=style.LINE_LW, label=style.name(k))
               for k in ("llm", "adaptive")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.12)
    style.save(fig, "rq1_llm_variant_trajectories")


if __name__ == "__main__":
    main()
