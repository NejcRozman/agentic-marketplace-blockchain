"""Shared figure style. One colour per strategy, fixed across the whole paper.

The colours are distinguishable under colour-vision deficiency for every pair
of strategies that appear in the same figure.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures"

#: strategy key -> (display name, colour)
STRATEGIES = {
    "adaptive": ("Adaptive",                 "#2a78d6"),   # slot 1 blue
    "band":     ("Random-in-band",           "#eb6834"),   # slot 2 orange
    "random":   ("Random",                   "#1baf7a"),   # slot 3 aqua
    # One colour for every LLM pricing variant; the variant is identified by panel
    # or marker shape.
    "llm":      ("LLM",                      "#b5179e"),   # magenta

    # RQ2: every agent prices with the adaptive margin and differs in where its
    # cost estimate comes from. Group names in the cells: B_none, B_r_guarded,
    # X_decomposed, X_oracle_bound. "h" is the same agent as RQ1's Adaptive.
    "h":        ("Adaptive",                 "#2a78d6"),   # blue, as RQ1's Adaptive
    "xd":       ("LLM-Estimate",             "#e34948"),   # red
    "rguarded": ("Regression",               "#eda100"),   # yellow
    "oracle":   ("Oracle",                   "#9b2f2f"),   # dark red: LLM-Estimate made perfect
}

#: LLM pricing variants. Each adds to the one before it (cumulative).
#: key -> (display name, suffix of the group name in the RQ1 cells)
LLM_VARIANTS = {
    "base":    ("LLM-Base",    "R0"),    # auction + own economics/history
    "band":    ("LLM-Band",    "R1a"),   # + keep margin within the heuristic band
    "history": ("LLM-History", "R1c"),   # + full record of past bids and outcomes
    "reflect": ("LLM-Reflect", "R2"),    # + per-round reflection rewriting a strategy memo
}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
ZERO = "#9a9993"

TEXT_WIDTH = 7.0    # page width of the two-column final layout (figure*), inches
COL_WIDTH = 3.46    # one column of the final layout (figure), inches

#: Text sizes for annotations drawn by the scripts. apply(column=True) shrinks
#: them, so a column-wide figure prints at the same type size as a page-wide one.
LETTER_FS = 9       # panel letters a), b), ...
HINT_FS = 6.5       # direction hints under forest axes
AXNOTE_FS = 8       # axis notes drawn as text (e.g. "Δ net balance (USD)")

# Mark sizes, shared by every chart so markers look the same throughout the paper.
MS = 4.0            # marker size (points), plot and legend alike
MEW = 1.2           # marker edge width
ERR_LW = 1.4        # confidence-interval line width
CAP = 2.5           # confidence-interval end-cap length
CAP_LW = 1.4        # end-cap line width
LINE_LW = 2.0       # trajectory lines


def color(key: str) -> str:
    return STRATEGIES[key][1]


def name(key: str) -> str:
    return STRATEGIES[key][0]


def apply(column: bool = False):
    """Set the shared style.

    column=True draws the figure at COL_WIDTH, one column of the two-column layout,
    with type and marks scaled to match page-wide figures.
    """
    global MS, MEW, ERR_LW, CAP, CAP_LW, LINE_LW, LETTER_FS, HINT_FS, AXNOTE_FS
    if column:
        MS, MEW, ERR_LW, CAP, CAP_LW, LINE_LW = 3.0, 1.0, 1.1, 2.0, 1.1, 1.5
        LETTER_FS, HINT_FS, AXNOTE_FS = 7.5, 5.5, 6.5
        sizes = dict(base=7, title=7, label=6.5, tick=6, legend=6.5)
    else:
        sizes = dict(base=8, title=9, label=8, tick=7, legend=8)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": sizes["base"],
        "axes.titlesize": sizes["title"],
        "axes.labelsize": sizes["label"],
        "xtick.labelsize": sizes["tick"],
        "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5 if column else 0.6,
        "axes.axisbelow": True,
        "savefig.facecolor": "white",
    })


def save(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        # No creation date in the PDF, so that regenerated figures are byte-identical.
        metadata = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=220, bbox_inches="tight", metadata=metadata)
    print(f"wrote {OUT_DIR / stem}.(pdf|png)")
