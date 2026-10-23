#!/usr/bin/env python3
"""Execution cost against quality rating of the recorded service dataset (not in the paper).

Each point is one of the 90 recorded executions, coloured by the requirement's
designed depth (locate < enumerate < contrast < reconcile). The correlations
between and within requirements are printed in the corner.

    python -m analysis.fig_dataset_cost_quality
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.dataset import DEPTHS, between_within, by_requirement

# Four distinct hues, light to dark, so the order of the depths still reads.
RAMP = ["#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # yellow, pink, green, violet


def main():
    style.apply()
    colour = dict(zip(DEPTHS, RAMP))
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH * 0.55, 3.0))
    for runs in by_requirement().values():
        c = colour[runs[0]["demanded_synthesis_depth"]]
        pts = [(r["execution_cost_usd"], r["evaluator_rating"]) for r in runs]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", ls="", ms=style.MS,
                mfc=c, mec="white", mew=0.6, zorder=3)

    b, w = between_within("execution_cost_usd", "evaluator_rating")
    ax.text(0.98, 0.04, f"r between requirements = {b:+.2f}\nr within requirement = {w:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=style.INK_2)
    ax.set_xlabel("Execution cost (USD)")
    ax.set_ylabel("Quality rating")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=style.MS + 1, mfc=colour[d],
                          mec="white", label=d) for d in DEPTHS]
    leg = ax.legend(handles=handles, title="Designed depth", loc="upper center",
                    bbox_to_anchor=(0.5, 1.27), ncol=4, frameon=False, fontsize=7,
                    title_fontsize=7, handletextpad=0.2, columnspacing=1.0)
    leg.get_title().set_color(style.INK_2)
    fig.tight_layout()
    style.save(fig, "dataset_cost_quality")


if __name__ == "__main__":
    main()
