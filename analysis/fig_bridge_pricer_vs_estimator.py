#!/usr/bin/env python3
"""H1 <-> H2 bridge — the same LLM used as a pricer and as a cost estimator, against Adaptive.

Both on gpt-oss-20b, both against the same opponent (Adaptive: the adaptive
controller with a historical-average cost estimate), 1 vs 1, 40 seed-paired
runs per cell. Each approach is run as designed:
  - LLM as pricer (H1, LLM-History): sets the price itself; cannot decline;
    40 s commit window (its ReAct call is slow).
  - LLM as estimator (H2, LLM-Estimate): estimates the job's cost; the adaptive
    controller sets the price; abstention (O2c) on; 20 s commit window.
The commit window does not constrain the estimator (0 late commits, <= 3
deadline-exceeded estimates per ~1,000 bids). Abstention is part of the H2
design — it acts on the per-job estimate — and the estimator still beats
Regression at rw = 20 with abstention off (see the abstention table).

  a) LLM as pricer vs Adaptive, rw = 20, net balance over auctions
  b) LLM as estimator vs Adaptive, rw = 20, same y-scale as a)
  c) Δ final net balance vs Adaptive at rw = 20, 50, 80, both approaches

    python -m analysis.fig_bridge_pricer_vs_estimator
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import N_AUCTIONS, mean_ci, paired_difference, significant, trajectory

PRICER = ("llm", "LLM as pricer", "X_llm_R1c", "B_s3")
ESTIMATOR = ("xd", "LLM as estimator", "X_decomposed", "B_none")
TRAJ = [(PRICER, "S91_var_R1c_vs_s3"), (ESTIMATOR, "A_gptoss20b_xd_vs_h")]
FOREST = [
    ("w = 0.2", "S91_var_R1c_vs_s3",   "A_gptoss20b_xd_vs_h"),
    ("w = 0.5", "S91_stage4_R1c_rw50", "B_gptoss20b_rw50_xd_vs_h"),
    ("w = 0.8", "S91_disc_R1c_rw80",   "B_gptoss20b_rw80_xd_vs_h"),
]
OFFSET = 0.16


def main():
    style.apply()
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])
    xs = list(range(N_AUCTIONS + 1))

    print("w = 0.2 final net balance, mean [95% CI]")
    for ax, ((key, label, llm, opp), cell) in zip((ax_a, ax_b), TRAJ):
        for k, g in (("h", opp), (key, llm)):
            seeds = trajectory(cell, g)
            per_x = [mean_ci([s[i] for s in seeds]) for i in xs]
            c = style.color(k)
            ax.fill_between(xs, [p.lo for p in per_x], [p.hi for p in per_x], color=c, alpha=0.18, lw=0)
            ax.plot(xs, [p.mean for p in per_x], color=c, lw=style.LINE_LW)
            e = per_x[-1]
            print(f"  {label:<17} {'Adaptive' if k == 'h' else 'LLM':<9} {e.mean:+.4f} [{e.lo:+.4f}, {e.hi:+.4f}]")
        ax.axhline(0, color=style.ZERO, lw=0.8, zorder=1)
        ax.set_xlim(0, N_AUCTIONS)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("Auction")
    ax_a.set_ylabel("Net balance (USD)")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    print("\nΔ final net balance vs Adaptive, mean [95% CI]")
    ys = list(range(len(FOREST)))[::-1]
    for y, (rw, pcell, ecell) in zip(ys, FOREST):
        for (key, label, llm, opp), cell, dy in ((PRICER, pcell, OFFSET), (ESTIMATOR, ecell, -OFFSET)):
            d = paired_difference(cell, llm, opp)
            c = style.color(key)
            ax_c.errorbar(d.mean, y + dy, xerr=[[d.mean - d.lo], [d.hi - d.mean]], fmt="none", ecolor=c,
                          elinewidth=style.ERR_LW, capsize=style.CAP, capthick=style.CAP_LW, zorder=2)
            ax_c.plot(d.mean, y + dy, "o", ms=style.MS, mec=c, mew=style.MEW,
                      mfc=c if significant(cell) else "white", zorder=3)
            print(f"  {rw:<8} {label:<17} {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]{'  *' if significant(cell) else ''}")
    ax_c.axvline(0, color=style.ZERO, lw=0.8, zorder=1)
    ax_c.set_yticks(ys)
    ax_c.set_yticklabels([f[0] for f in FOREST], color=style.INK)
    ax_c.set_ylim(-0.6, len(FOREST) - 0.4)
    ax_c.grid(axis="y", visible=False)
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    for x, ha, text in ((0.0, "left", "← Adaptive"), (1.0, "right", "LLM →")):
        ax_c.text(x, -0.15, text, transform=ax_c.transAxes, ha=ha, va="top", fontsize=6.5, color=style.INK_2)
    ax_c.text(0.5, -0.26, "Δ net balance (USD)", transform=ax_c.transAxes, ha="center", va="top",
              fontsize=8, color=style.INK_2)

    for letter, ax, y in (("a", ax_a, -0.36), ("b", ax_b, -0.36), ("c", ax_c, -0.40)):
        ax.text(0.5, y, f"{letter})", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=style.INK)

    handles = [plt.Line2D([], [], color=style.color("llm"), lw=style.LINE_LW, marker="o", ms=style.MS,
                          label="LLM as pricer (LLM-History)"),
               plt.Line2D([], [], color=style.color("xd"), lw=style.LINE_LW, marker="o", ms=style.MS,
                          label="LLM as estimator (LLM-Estimate)"),
               plt.Line2D([], [], color=style.color("h"), lw=style.LINE_LW, label="Adaptive")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.86)
    style.save(fig, "bridge_pricer_vs_estimator")


if __name__ == "__main__":
    main()
