#!/usr/bin/env python3
"""H2 — LLM-Estimate across conditions: Δ final net balance, LLM-Estimate − baseline.

Circles = against Adaptive, squares = against Regression. Each point is the mean
per-seed difference in final net balance over 40 seed-paired runs; whiskers are
95% t intervals; a filled marker means the difference is significant (Holm-adjusted, RQ2 family). Panels share
one x-scale because they show the same measure. All cells: abstention (O2c) on,
20 s commit window, prior 80.

  a) reputation weight, 1 vs 1, gpt-oss-20b; dashed line: mixed-model fit vs
     Adaptive (stats_rw_gradient, Eq. 12), diamond: its zero crossing w*
  b) model, rw = 20, 1 vs 1
  c) market composition, rw = 20, gpt-oss-20b: one LLM-Estimate agent against
     one or three baseline agents (baseline arm = per-agent mean)

rw = 80 vs Regression is cell C_o2c_on_rw80: config-identical to the block B
cell that would have been generated, and run instead of it.

    python -m analysis.fig_h2_conditions
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import paired_difference, significant
from analysis import stats_rw_gradient as grad

XD = "X_decomposed"
MARKERS = {"h": "o", "rguarded": "s"}
OFFSET = 0.16

# Each row: label, {baseline key: (cell, baseline group)}
PANEL_A = [
    ("w = 0",    {"h": ("B_gptoss20b_rw0_xd_vs_h",  "B_none"), "rguarded": ("B_gptoss20b_rw0_xd_vs_rguarded",  "B_r_guarded")}),
    ("w = 0.2",  {"h": ("A_gptoss20b_xd_vs_h",      "B_none"), "rguarded": ("A_gptoss20b_xd_vs_rguarded",      "B_r_guarded")}),
    ("w = 0.35", {"h": ("B_gptoss20b_rw35_xd_vs_h", "B_none"), "rguarded": ("B_gptoss20b_rw35_xd_vs_rguarded", "B_r_guarded")}),
    ("w = 0.5",  {"h": ("B_gptoss20b_rw50_xd_vs_h", "B_none"), "rguarded": ("B_gptoss20b_rw50_xd_vs_rguarded", "B_r_guarded")}),
    ("w = 0.65", {"h": ("B_gptoss20b_rw65_xd_vs_h", "B_none"), "rguarded": ("B_gptoss20b_rw65_xd_vs_rguarded", "B_r_guarded")}),
    ("w = 0.8",  {"h": ("B_gptoss20b_rw80_xd_vs_h", "B_none"), "rguarded": ("C_o2c_on_rw80",                   "B_r_guarded")}),
]
PANEL_B = [
    ("gpt-oss-20b",  {"h": ("A_gptoss20b_xd_vs_h", "B_none"), "rguarded": ("A_gptoss20b_xd_vs_rguarded", "B_r_guarded")}),
    ("gpt-4o-mini",  {"h": ("E_xd_vs_h_v3",        "B_none"), "rguarded": ("E_xd_vs_r_v3",               "R_guarded")}),
    ("gpt-5.4-mini", {"h": ("A_gpt54mini_xd_vs_h", "B_none"), "rguarded": ("A_gpt54mini_xd_vs_rguarded", "B_r_guarded")}),
]
PANEL_C = [
    ("1:1", {"h": ("A_gptoss20b_xd_vs_h", "B_none"), "rguarded": ("A_gptoss20b_xd_vs_rguarded", "B_r_guarded")}),
    ("1:3", {"h": ("D_invasion_1v3",      "B_none"), "rguarded": ("D_invasion_1v3_rguarded",    "B_r_guarded")}),
]


def forest(ax, rows):
    c = style.color("xd")
    out = []
    ys = list(range(len(rows)))[::-1]
    for y, (label, baselines) in zip(ys, rows):
        for key, (cell, group) in baselines.items():
            d = paired_difference(cell, XD, group)
            yy = y + (OFFSET if key == "h" else -OFFSET)
            ax.errorbar(d.mean, yy, xerr=[[d.mean - d.lo], [d.hi - d.mean]], fmt="none",
                        ecolor=c, elinewidth=style.ERR_LW, capsize=style.CAP,
                        capthick=style.CAP_LW, zorder=2)
            ax.plot(d.mean, yy, MARKERS[key], ms=style.MS, mec=c, mew=style.MEW,
                    mfc=c if significant(cell) else "white", zorder=3)
            out.append((label, style.name(key), d, significant(cell)))
    ax.axvline(0, color=style.ZERO, lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=style.INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for x, ha, text in ((0.0, "left", "← baseline"), (1.0, "right", "LLM-Estimate →")):
        ax.text(x, -0.22, text, transform=ax.transAxes, ha=ha, va="top",
                fontsize=6.5, color=style.INK_2)
    ax.text(0.5, -0.35, "Δ net balance (USD)", transform=ax.transAxes, ha="center",
            va="top", fontsize=8, color=style.INK_2)
    return out


#: w of each panel-a row, top to bottom (rows are evenly spaced, w is not).
W_ROWS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]


def fitted_gradient(ax):
    """Overlay the mixed model of Eq. (12) on panel a: fitted Δ against Adaptive
    at each row, and the crossing point w* where it reaches zero."""
    res = grad.fit(grad.frame())
    b0, b1 = res.params["Intercept"], res.params["w"]
    ys = list(range(len(W_ROWS)))[::-1]
    ax.plot([b0 + b1 * w for w in W_ROWS], ys, ls="--", lw=1.0, color=style.INK_2, zorder=1)
    w_star = -b0 / b1
    # row position of w* by linear interpolation between neighbouring rows
    for (wa, ya), (wb, yb) in zip(zip(W_ROWS, ys), zip(W_ROWS[1:], ys[1:])):
        if wa <= w_star <= wb:
            y_star = ya + (yb - ya) * (w_star - wa) / (wb - wa)
    ax.plot(0, y_star, marker="D", ms=style.MS - 0.5, color=style.INK_2, zorder=4)
    ax.annotate(f"$w^*$ = {w_star:.2f}", (0, y_star), xytext=(-7, 7), textcoords="offset points",
                ha="right", va="bottom", fontsize=7, color=style.INK_2)
    print(f"fitted line: Δ = {b0:+.4f} {b1:+.4f}·w (vs Adaptive); w* = {w_star:.3f}")


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 2.4), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    results = []
    for letter, ax, rows in zip("abc", axes, (PANEL_A, PANEL_B, PANEL_C)):
        results += [(letter, *r) for r in forest(ax, rows)]
        ax.text(0.5, -0.50, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=style.INK)
    fitted_gradient(axes[0])

    c = style.color("xd")
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc=c, mec=c, mew=style.MEW,
                   label="vs " + style.name("h")),
        plt.Line2D([], [], marker="s", ls="", ms=style.MS, mfc=c, mec=c, mew=style.MEW,
                   label="vs " + style.name("rguarded")),
        # Neutral grey so the fill key doesn't read as another agent.
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc=style.INK_2, mec=style.INK_2,
                   mew=style.MEW, label="significant (Holm)"),
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc="white", mec=style.INK_2,
                   mew=style.MEW, label="not significant"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2)
    style.save(fig, "h2_conditions")

    print("LLM-Estimate − baseline, final net balance (USD), mean [95% CI], n")
    for letter, label, base, d, sig in results:
        print(f"  {letter}) {label:<13} vs {base:<11} {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"
              f"  n={d.n}{'  * (Holm)' if sig else ''}")


if __name__ == "__main__":
    main()
