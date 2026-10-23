#!/usr/bin/env python3
"""H2 — LLM-Estimate vs the two non-LLM estimators: net balance over the 30 auctions.

rw = 20, 1 vs 1, gpt-oss-20b, 40 seed-paired runs per pairing. All arms price
with the adaptive controller; they differ in the cost estimate it is given:
Adaptive a running average of past job costs (the same agent as H1's Adaptive),
Regression a ridge regression on three surface features of the requirement
(prompt length and two counts) fitted to the agent's own completed jobs, LLM-Estimate an LLM
estimate of the job from its requirement text.
Abstention (O2c) on and a 20 s commit window, as in all of block A.

Lines are the mean across seeds; shading is the 95% CI of that mean. Both
panels share one y-scale: LLM-Estimate is the same agent in both, so its level is
directly comparable across opponents.

    python -m analysis.fig_rq2_trajectories
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import N_AUCTIONS, mean_ci, paired_difference, trajectory

XD = "X_decomposed"
# (cell, [(strategy key, group name), ...])
PANELS = [
    ("A_gptoss20b_xd_vs_h",        [("h", "B_none"), ("xd", XD)]),
    ("A_gptoss20b_xd_vs_rguarded", [("rguarded", "B_r_guarded"), ("xd", XD)]),
]


def main():
    style.apply(column=True)
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_WIDTH, 1.8),
                             sharex=True, sharey=True)
    xs = list(range(N_AUCTIONS + 1))
    print("Final net balance, mean [95% CI], USD — and paired difference LLM-Estimate − opponent")

    for letter, ax, (cell, arms) in zip("ab", axes, PANELS):
        for key, group in arms:
            seeds = trajectory(cell, group)
            per_x = [mean_ci([s[i] for s in seeds]) for i in xs]
            c = style.color(key)
            ax.fill_between(xs, [p.lo for p in per_x], [p.hi for p in per_x],
                            color=c, alpha=0.18, lw=0)
            ax.plot(xs, [p.mean for p in per_x], color=c, lw=style.LINE_LW)
            end = per_x[-1]
            print(f"  {cell:<28} {style.name(key):<10} {end.mean:+.4f} [{end.lo:+.4f}, {end.hi:+.4f}]  n={end.n}")
        opp = arms[0][1]
        d = paired_difference(cell, XD, opp)
        print(f"  {cell:<28} LLM-Estimate − {style.name(arms[0][0])}: {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"
              f"{'  (CI excludes 0)' if d.excludes_zero else ''}")

        ax.axhline(0, color=style.ZERO, lw=0.8, zorder=1)
        ax.set_xlim(0, N_AUCTIONS)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("Auction")
        ax.text(0.5, -0.40, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=style.LETTER_FS, color=style.INK)
    axes[0].set_ylabel("Net balance (USD)")

    handles = [plt.Line2D([], [], color=style.color(k), lw=style.LINE_LW, label=style.name(k))
               for k in ("xd", "h", "rguarded")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.10), handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    style.save(fig, "rq2_trajectories")


if __name__ == "__main__":
    main()
