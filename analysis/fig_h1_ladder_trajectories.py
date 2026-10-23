#!/usr/bin/env python3
"""H1 — heuristic ladder: net balance over the 30 auctions, three pairings.

rw = 20, 1 vs 1, 40 seed-paired runs per pairing. Lines are the mean across
seeds; shading is the 95% CI of that mean at each auction. Each panel has its
own y-scale: the pairings with random bidding span ~±2 USD, band vs adaptive
~0.3, so a shared scale would flatten the third panel.

    python -m analysis.fig_h1_ladder_trajectories
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import N_AUCTIONS, mean_ci, paired_difference, trajectory

# (title, cell, [(strategy key, group name), ...])
PANELS = [
    ("Random vs band",     "S91_ladder_zi_vs_band",       [("band", "B_band"), ("random", "X_zi")]),
    ("Random vs adaptive", "S91_ladder_zi_vs_adaptive",   [("adaptive", "B_s3"), ("random", "X_zi")]),
    ("Band vs adaptive",   "S91_ladder_band_vs_adaptive", [("adaptive", "B_s3"), ("band", "X_band")]),
]


def main():
    style.apply(column=True)
    fig, axes = plt.subplots(1, 3, figsize=(style.COL_WIDTH, 1.75), sharex=True)
    xs = list(range(N_AUCTIONS + 1))
    print("Final net balance, mean [95% CI], USD — and paired difference")

    for letter, ax, (title, cell, arms) in zip("abc", axes, PANELS):
        # Panels are identified by letter only; the caption says what each shows.
        ax.text(0.5, -0.42, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=style.LETTER_FS, color=style.INK)
        for key, group in arms:
            seeds = trajectory(cell, group)
            per_x = [mean_ci([s[i] for s in seeds]) for i in xs]
            c = style.color(key)
            ax.fill_between(xs, [p.lo for p in per_x], [p.hi for p in per_x],
                            color=c, alpha=0.18, lw=0)
            ax.plot(xs, [p.mean for p in per_x], color=c, lw=style.LINE_LW)
            end = per_x[-1]
            print(f"  {title:<20} {key:<9} {end.mean:+.4f} [{end.lo:+.4f}, {end.hi:+.4f}]  n={end.n}")
        (k1, g1), (k2, g2) = arms
        d = paired_difference(cell, g1, g2)
        print(f"  {title:<20} {k1} − {k2}: {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"
              f"{'  (CI excludes 0)' if d.excludes_zero else ''}")

        ax.axhline(0, color=style.ZERO, lw=0.8, zorder=1)

        ax.set_xlim(0, N_AUCTIONS)
        ax.set_xticks([0, 15, 30])
        ax.set_xlabel("Auction")
    axes[0].set_ylabel("Net balance (USD)")

    handles = [plt.Line2D([], [], color=style.color(k), lw=style.LINE_LW, label=style.name(k))
               for k in ("adaptive", "band", "random")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.10), handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    style.save(fig, "h1_ladder_trajectories")


if __name__ == "__main__":
    main()
