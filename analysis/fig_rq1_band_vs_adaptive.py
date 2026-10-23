#!/usr/bin/env python3
"""H1 — adaptive minus band, final net balance, by reputation weight and composition.

Each point is the mean per-seed difference in final net balance (adaptive −
band, USD) over 40 seed-paired runs; whiskers are 95% t intervals. A filled
marker means the difference is significant: paired t-test, Holm-adjusted within
the RQ1 family (data.holm_table, Section 5.7). In 2v2/1v3/3v1 each arm is the
per-agent mean. Both panels share one x-scale because they show the same
measure.

    python -m analysis.fig_rq1_band_vs_adaptive
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import paired_difference, significant

ADAPTIVE, BAND = "B_s3", "X_band"

RW = [
    ("w = 0.2", "S91_ladder_band_vs_adaptive"),
    ("w = 0.5", "S91_ladder_band_vs_adaptive_rw50"),
    ("w = 0.8", "S91_ladder_band_vs_adaptive_rw80"),
]
COMPOSITION = [
    # B = random-in-band agents, A = adaptive agents
    ("1B:1A", "S91_ladder_band_vs_adaptive"),
    ("2B:2A", "S91_comp_2v2_band_vs_adaptive"),
    ("1B:3A", "S91_comp_1band_vs_3adaptive"),
    ("3B:1A", "S91_comp_3band_vs_1adaptive"),
]


def forest(ax, rows, title, subtitle):
    c = style.color("adaptive")
    results = [(label, paired_difference(cell, ADAPTIVE, BAND), significant(cell)) for label, cell in rows]
    ys = list(range(len(results)))[::-1]
    for y, (_, d, sig) in zip(ys, results):
        ax.errorbar(d.mean, y, xerr=[[d.mean - d.lo], [d.hi - d.mean]], fmt="none",
                    ecolor=c, elinewidth=style.ERR_LW, capsize=style.CAP, capthick=style.CAP_LW, zorder=2)
        ax.plot(d.mean, y, "o", ms=style.MS, mec=c, mew=style.MEW,
                mfc=c if sig else "white", zorder=3)
    ax.axvline(0, color=style.ZERO, lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in results], color=style.INK)
    ax.set_ylim(-0.6, len(results) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Which side wins, so the reader need not work out the sign of the difference.
    # Δ = adaptive − band; the exact definition is in the caption.
    # The axis label sits between the two hints on one line.
    # Direction hints on their own line under the ticks, the axis label beneath.
    for x, ha, text in ((0.0, "left", "← Random-in-band"), (1.0, "right", "Adaptive →")):
        ax.text(x, -0.25, text, transform=ax.transAxes, ha=ha, va="top",
                fontsize=style.HINT_FS, color=style.INK_2)
    ax.text(0.5, -0.40, "Δ net balance (USD)", transform=ax.transAxes, ha="center",
            va="top", fontsize=style.AXNOTE_FS, color=style.INK_2)
    return results


def main():
    style.apply(column=True)
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_WIDTH, 1.9), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    a = forest(axes[0], RW, "By reputation weight", "1 vs 1")
    b = forest(axes[1], COMPOSITION, "By market composition", "rw = 20, per-agent means")
    # Panels are identified by letter only; the caption says what each shows.
    for letter, ax in zip("ab", axes):
        ax.text(0.5, -0.60, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=style.LETTER_FS, color=style.INK)

    c = style.color("adaptive")
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc=c, mec=c, mew=style.MEW, label="significant (Holm)"),
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc="white", mec=c, mew=style.MEW, label="not significant"),
    ], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.5)
    style.save(fig, "rq1_band_vs_adaptive")

    print("adaptive − band, final net balance (USD), mean [95% CI], n")
    for label, d, sig in a + b:
        print(f"  {label:<20} {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]  n={d.n}"
              f"{'  * (Holm)' if sig else ''}")


if __name__ == "__main__":
    main()
