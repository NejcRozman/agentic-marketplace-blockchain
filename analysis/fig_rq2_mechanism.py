#!/usr/bin/env python3
"""RQ2: why LLM-Estimate wins at low reputation weight and loses at high.

LLM-Estimate (red) vs Adaptive (blue), 1 vs 1, gpt-oss-20b, by reputation
weight. Each point is the mean over 40 seeds of a per-seed value; whiskers are
95% t intervals across seeds. Per-job quantities are averaged within a seed
first (seeds where the agent won nothing are skipped), so every seed counts
once.

  a) share of the 30 auctions won by LLM-Estimate
  b) execution cost of the jobs each agent won (USD)
  c) consumer quality rating of the jobs each agent won. An agent's reputation
     is the running mean of these ratings — final reputation matches this to
     within 0.004 at every w — so it is not plotted separately.
  d) price received per job won (winning bid, USD)
  e) realized margin per job won: price / execution cost − 1 (median within seed)

The same quantities against Regression are printed (not plotted).

    python -m analysis.fig_rq2_mechanism
"""
import math
import statistics as st

import matplotlib.pyplot as plt

from analysis import style
from analysis.data import mean_ci, outcomes

XD = "X_decomposed"
RW = [0, 20, 35, 50, 65, 80]
CELLS_H = {0: "B_gptoss20b_rw0_xd_vs_h", 20: "A_gptoss20b_xd_vs_h",
           35: "B_gptoss20b_rw35_xd_vs_h", 50: "B_gptoss20b_rw50_xd_vs_h",
           65: "B_gptoss20b_rw65_xd_vs_h", 80: "B_gptoss20b_rw80_xd_vs_h"}
CELLS_R = {0: "B_gptoss20b_rw0_xd_vs_rguarded", 20: "A_gptoss20b_xd_vs_rguarded",
           35: "B_gptoss20b_rw35_xd_vs_rguarded", 50: "B_gptoss20b_rw50_xd_vs_rguarded",
           65: "B_gptoss20b_rw65_xd_vs_rguarded", 80: "C_o2c_on_rw80"}

# (label, function seed-outcome -> value or None, scale for display)
METRICS = [
    ("Auctions won (%)",       lambda o: o.win_share, 100),
    ("Cost per job (USD)",     lambda o: st.mean(o.job_cost) if o.job_cost else None, 1),
    ("Quality (= reputation)", lambda o: st.mean(o.job_quality) if o.job_quality else None, 1),
    ("Price per job (USD)",    lambda o: st.mean(o.job_price) if o.job_price else None, 1),
    ("Margin per job (%)",     lambda o: st.median(o.job_margin) if o.job_margin else None, 100),
]


def series(cells, group, fn, scale):
    pts = []
    for rw in RW:
        vals = [v for v in (fn(o) for o in outcomes(cells[rw], group))
                if v is not None and not math.isnan(v)]
        ci = mean_ci(vals)
        pts.append((rw, ci.mean * scale, ci.lo * scale, ci.hi * scale, ci.n))
    return pts


def draw(ax, pts, key):
    c = style.color(key)
    xs = [p[0] for p in pts]
    ax.plot(xs, [p[1] for p in pts], color=c, lw=1.4, zorder=2)
    ax.errorbar(xs, [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                fmt="o", ms=style.MS, mfc=c, mec=c, mew=style.MEW, ecolor=c,
                elinewidth=style.ERR_LW, capsize=style.CAP, capthick=style.CAP_LW, zorder=3)


def main():
    style.apply()
    # Five panels in one row, page-wide: what work each agent takes (a-c) and
    # what it earns per job (d-e). Panel a) is LLM-Estimate's share of auctions.
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 2.0))
    gs = fig.add_gridspec(1, 5)
    slots = [gs[0, i] for i in range(5)]
    axes = []
    for s in slots:
        axes.append(fig.add_subplot(s, sharex=axes[0] if axes else None))
    for letter, ax, (label, fn, scale) in zip("abcde", axes, METRICS):
        xd = series(CELLS_H, XD, fn, scale)
        draw(ax, xd, "xd")
        if letter == "a":
            ax.axhline(50, color=style.ZERO, lw=0.8, ls="--", zorder=1)
            other = None
        else:
            other = series(CELLS_H, "B_none", fn, scale)
            draw(ax, other, "h")
        ax.set_title(label, fontsize=8, color=style.INK, loc="left")
        ax.set_xticks([0, 20, 50, 80])
        ax.set_xticklabels(["0", "0.2", "0.5", "0.8"])
        ax.text(0.5, -0.38, f"{letter})", transform=ax.transAxes,
                ha="center", va="top", fontsize=style.LETTER_FS, color=style.INK)

        # Table view: vs Adaptive (plotted) and vs Regression (printed only).
        rx = series(CELLS_R, XD, fn, scale)
        rr = None if letter == "a" else series(CELLS_R, "B_r_guarded", fn, scale)
        print(f"\n{letter}) {label}")
        print(f"  {'rw':>3}  {'LLM-Est|Adapt':>14} {'Adaptive':>9}   {'LLM-Est|Regr':>13} {'Regression':>10}")
        for i, rw in enumerate(RW):
            fmt = lambda p: f"{p[1]:9.3f}" if p else f"{'—':>9}"
            print(f"  {rw:>3}  {xd[i][1]:14.3f} {fmt(other[i] if other else None)}   "
                  f"{rx[i][1]:13.3f} {fmt(rr[i] if rr else None):>10}")
    for ax in axes:
        ax.set_xlabel("$w$")

    handles = [plt.Line2D([], [], color=style.color(k), lw=1.4, marker="o", ms=style.MS,
                          label=style.name(k)) for k in ("xd", "h")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(w_pad=1.2)
    style.save(fig, "rq2_mechanism")


if __name__ == "__main__":
    main()
