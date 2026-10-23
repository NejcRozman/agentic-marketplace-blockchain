#!/usr/bin/env python3
"""Dataset v3 — execution cost vs quality rating, by designed depth.

Each point is one of the 90 recorded executions, coloured by the requirement's
designed depth. The between/within-requirement split is given as correlations
in the corner rather than drawn: joining each requirement's three runs was
tried, but runs of one requirement also differ in cost, so the lines tangled.

Depth is ordered (locate < enumerate < contrast < reconcile). It uses four
distinct hues, assigned light to dark: a one-hue ramp was tried first but its
middle steps were too close to tell apart. This chart describes the dataset,
not strategies, so the strategy colours do not apply.

    python -m analysis.fig_dataset_cost_quality
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.dataset import DEPTHS, between_within, by_requirement

# Four distinct palette hues rather than one-hue steps: four shades of blue
# could not be told apart at this marker size. This set is the best-separated
# four of the palette, all pairs (worst-pair CVD ΔE 16.2, normal-vision 19.6),
# assigned light to dark so the order locate -> reconcile still reads.
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
