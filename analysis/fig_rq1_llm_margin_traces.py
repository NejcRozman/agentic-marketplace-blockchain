#!/usr/bin/env python3
"""H1 (appendix) — one run per LLM variant: the margin each agent chose, auction by auction.

Same seed in every panel, so every variant faces the same sequence of jobs.
Row 1: LLM-Base against Adaptive with its 20% floor and with m_min = 0;
row 2: LLM-Band, LLM-History, LLM-Reflect.
Filled marker = that agent won the auction, hollow = it lost. Adaptive's rule
is visible as a staircase (up after a win, down after a loss, held at its 0.20
floor); the LLM's margins show whether it reacts to outcomes at all. Margins
are bid / cost basis − 1; points outside the y-range are drawn at the edge
with an arrow marker and listed in the printed table.

    python -m analysis.fig_rq1_llm_margin_traces
"""
import matplotlib.pyplot as plt

from analysis import style
from analysis.data import N_AUCTIONS, bid_sequences, ok_runs

SEED = 1007
YLIM = (-20, 100)          # percent; bids outside are drawn as arrows at the edge


def seed_index(cell):
    seeds = [r["seed"] for r in ok_runs(cell)]
    return seeds.index(SEED)


def draw(ax, seq, key):
    c = style.color(key)
    xs = [b.auction for b in seq]
    ys = [100 * b.margin for b in seq]
    clipped = ys if YLIM is None else [min(max(y, YLIM[0]), YLIM[1]) for y in ys]
    ax.plot(xs, clipped, color=c, lw=1.2, alpha=0.7, zorder=2)
    for b, x, y, yc in zip(seq, xs, ys, clipped):
        if y != yc:
            ax.plot(x, yc, "^" if y > yc else "v", ms=style.MS + 1, color=c, zorder=3)
        else:
            ax.plot(x, y, "o", ms=style.MS, mec=c, mew=style.MEW, mfc=c if b.won else "white", zorder=3)
    if YLIM is None:
        return []
    return [(b.auction, round(100 * b.margin, 1), b.won) for b in seq if not YLIM[0] <= 100 * b.margin <= YLIM[1]]


#: (title, cell, LLM group, grid slot, Adaptive's floor in %). Same layout as the
#: variant-trajectory figure: the two LLM-Base panels side by side in row 1.
PANELS = [
    ("LLM-Base",                 "S91_var_R0_vs_s3",        "X_llm_R0",  (0, slice(1, 3)), 20),
    ("LLM-Base, $m_{\\min}=0$",   "S91_var_R0_vs_s3_floor0", "X_llm_R0",  (0, slice(3, 5)), 0),
    ("LLM-Band",                 "S91_var_R1a_vs_s3",       "X_llm_R1a", (1, slice(0, 2)), 20),
    ("LLM-History",              "S91_var_R1c_vs_s3",       "X_llm_R1c", (1, slice(2, 4)), 20),
    ("LLM-Reflect",              "S91_var_R2_vs_s3",        "X_llm_R2",  (1, slice(4, 6)), 20),
]


def main():
    style.apply(column=True)
    fig = plt.figure(figsize=(style.COL_WIDTH, 3.3))
    gs = fig.add_gridspec(2, 6, hspace=0.95, wspace=1.3)
    first = None
    for letter, (name, cell, group, (row, cols), floor) in zip("abcde", PANELS):
        ax = fig.add_subplot(gs[row, cols], sharey=first)
        first = first or ax
        i = seed_index(cell)
        llm = bid_sequences(cell, group)[i]
        adaptive = bid_sequences(cell, "B_s3")[i]
        draw(ax, adaptive, "adaptive")
        out = draw(ax, llm, "llm")
        if out:
            print(f"  {cell}: margins outside {YLIM}% (auction, %, won): {out}")
        print(f"  {cell:<26} seed {SEED}: LLM won {sum(b.won for b in llm)}/{len(llm)}, adaptive won "
              f"{sum(b.won for b in adaptive)}/{len(adaptive)}")

        ax.axhline(floor, color=style.ZERO, lw=0.8, ls="--", zorder=1)
        ax.set_title(name, color=style.INK, fontweight="bold", fontsize=6.5)
        ax.set_xlim(0, N_AUCTIONS + 1)
        ax.set_xticks([0, 15, 30])
        if YLIM is not None:
            ax.set_ylim(YLIM[0] - 4, YLIM[1] + 4)
        ax.set_xlabel("Auction")
        ax.text(0.5, -0.45, f"{letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=style.LETTER_FS, color=style.INK)
        if cols.start in (0, 1):
            ax.set_ylabel("Margin (%)")

    handles = [
        plt.Line2D([], [], color=style.color("llm"), lw=1.5, label="LLM"),
        plt.Line2D([], [], color=style.color("adaptive"), lw=1.5, label="Adaptive"),
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc=style.INK_2, mec=style.INK_2, mew=style.MEW, label="won"),
        plt.Line2D([], [], marker="o", ls="", ms=style.MS, mfc="white", mec=style.INK_2, mew=style.MEW, label="lost"),
        plt.Line2D([], [], color=style.ZERO, lw=0.8, ls="--", label="floor $m_{\\min}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.55, 1.0), handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.12)
    style.save(fig, "rq1_llm_margin_traces")


if __name__ == "__main__":
    main()
