#!/usr/bin/env python3
"""S8 §A — the pre-registered primary endpoint, and the ledger-backed loader it uses.

    python -m analysis.pooled_model --cells E_xd_vs_h_v3 E_xd_vs_r_v3 \
        E_xd_vs_rd_v3 E_xd_vs_rnaive_v3

**The endpoint.** One pooled model over every cell in the confirmatory family,
fitted on the paired per-seed difference ``X_d - opponent``::

    difference ~ 1 + (1 | cell) + (1 | seed)

The test of the primary hypothesis is the **intercept**, two-sided, alpha = 0.05.

Per-cell significance is deliberately *not* the endpoint. With 22 cells in S9's
blocks A and B, testing each would either be underpowered on the expensive model
tiers or need a multiplicity correction that discards real signal. The pooled
intercept is the one number carrying the alpha = 0.05 claim.

**Seed is crossed with cell, not nested.** Every cell runs the same seeds
1007-1046, and the same seed means the same requirement arrival order, so a
nested fit would throw away the pairing that makes the design efficient.
statsmodels reaches crossed effects through variance components on a single
dummy group; that is what ``_CROSSED`` below is doing.

**Run membership comes from the ledger, never from a directory listing.**
``experiments/data/metrics/`` holds runs that belong to no confirmatory set —
earlier smoke, superseded attempts. On block E that is 100 extra directories, and
globbing them moves the headline X-vs-H difference from +0.2406 to +0.2053. See
S8 §B; :func:`load_ledger_runs` is the shared loader that fix is built on.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agents.common import paths  # noqa: E402

REPO_ROOT = paths.REPO_ROOT
LEDGER = paths.LEDGER
LOGS = paths.RUN_LOGS_DIR
METRICS = paths.RUN_METRICS_DIR


def load_ledger_runs(cells: Iterable[str], ledger: Optional[Path] = None) -> List[Dict]:
    """Every ``status: ok`` ledger row for these cells. The single source of truth.

    Returns one dict per run with ``cell``, ``seed``, ``repeat`` and ``run_id``.
    A run that is on disk but not in the ledger is *not* a member of the
    confirmatory set and is never returned.
    """
    path = ledger or LEDGER
    if not path.exists():
        raise SystemExit(f"no ledger at {path}")
    wanted = set(cells)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("cell") in wanted:
            rows.append({
                "cell": row["cell"], "seed": int(row["seed"]),
                "repeat": int(row.get("repeat", 1)), "run_id": row.get("run_id"),
            })
    return rows


def paired_difference(cell: str, run_id: str) -> Optional[float]:
    """``X_d`` net balance minus its opponent's, for one run.

    ``None`` when the run directory is missing or an arm produced no status file,
    which the caller reports rather than silently dropping.

    Reads the per-agent status files under ``data/logs/``; ``data/metrics/``
    holds only ``experiment_metrics.json``. The two trees are keyed by the same
    run id, and the ledger row is what joins them.
    """
    run_dir = LOGS / f"s9_{cell}" / str(run_id)
    if not run_dir.is_dir():
        return None
    nets = {}
    for path in sorted(run_dir.glob("provider_*_status.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        group = (data.get("profile") or {}).get("group_name") or ""
        financials = data.get("financials") or {}
        if group and "net_balance" in financials:
            nets[group] = float(financials["net_balance"])
    x = next((v for k, v in nets.items() if k.startswith("X_")), None)
    other = next((v for k, v in nets.items() if not k.startswith("X_")), None)
    if x is None or other is None:
        return None
    return x - other


def _count_dirs(root: Path) -> int:
    return len([d for d in root.glob("*/") if d.is_dir()]) if root.is_dir() else 0


def build_frame(cells: Iterable[str], ledger: Optional[Path] = None):
    """One row per (cell, seed): the paired difference. Requires pandas.

    ``ledger`` overrides which ledger defines membership. Analysis always uses
    the active one; tests point it at an archived ledger so they pin the analysis
    code against a *frozen* input rather than against whatever ran last.
    """
    import pandas as pd

    records, missing = [], []
    for run in load_ledger_runs(cells, ledger):
        diff = paired_difference(run["cell"], run["run_id"])
        if diff is None:
            missing.append(run)
            continue
        records.append({"cell": run["cell"], "seed": run["seed"], "diff": diff})
    return pd.DataFrame.from_records(records), missing


#: statsmodels has one native grouping factor. Crossed random intercepts come
#: from variance components over a single constant group — the standard idiom.
_CROSSED = {"cell": "0 + C(cell)", "seed": "0 + C(seed)"}


def fit_pooled(frame):
    """``diff ~ 1 + (1|cell) + (1|seed)``, REML. Returns the fitted result.

    Raises on an empty or single-row frame rather than letting statsmodels fail
    inside patsy: that failure cannot be formatted by pytest (patsy's
    ``EvalEnvironment.__getitem__`` raises ``KeyError`` when the traceback
    machinery enumerates frame locals), so the whole test session dies with an
    INTERNALERROR and no indication of which assertion failed.
    """
    import statsmodels.formula.api as smf

    if len(frame) < 2:
        raise ValueError(
            f"pooled model needs at least 2 paired runs, got {len(frame)}. "
            f"Usually this means the ledger has no `ok` rows for the requested "
            f"cells — check the cell names, and that the campaign has run."
        )

    frame = frame.copy()
    frame["_grp"] = 1
    model = smf.mixedlm("diff ~ 1", frame, groups=frame["_grp"], vc_formula=_CROSSED)
    return model.fit(reml=True)


def sign_test_across_cells(frame):
    """Exact two-sided binomial on the sign of each cell's mean difference.

    Pre-specified as a secondary in its own right: under H0 each cell's sign is a
    fair coin, so a consistent sign across many cells is strong evidence — and
    legitimate only because it is named in advance.
    """
    from scipy import stats

    means = frame.groupby("cell")["diff"].mean()
    positive = int((means > 0).sum())
    n = int(len(means))
    p = stats.binomtest(positive, n, 0.5, alternative="two-sided").pvalue
    return positive, n, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args()

    frame, missing = build_frame(args.cells)
    if frame.empty:
        raise SystemExit("no runs matched — check the cell names against the ledger")

    print("=" * 74)
    print("RUN SELECTION — from the ledger, not the metrics directory")
    print("=" * 74)
    for cell, group in frame.groupby("cell"):
        n_logs = _count_dirs(LOGS / f"s9_{cell}")
        n_metrics = _count_dirs(METRICS / f"s9_{cell}")
        print(f"  {cell:<24} used {len(group):>3}   logs/ {n_logs:>3}"
              f"   metrics/ {n_metrics:>3}   ignored {max(n_logs, n_metrics) - len(group):>3}")
    if missing:
        print(f"  ⚠️  {len(missing)} ledger rows had no readable run directory")
    print(f"\n  total runs in the model: {len(frame)}")

    print("\n" + "=" * 74)
    print("PER-CELL MEANS (descriptive — not the endpoint)")
    print("=" * 74)
    for cell, group in frame.groupby("cell"):
        d = group["diff"]
        print(f"  {cell:<24} n={len(d):<4} mean {d.mean():+.4f}   sd {d.std(ddof=1):.4f}")

    print("\n" + "=" * 74)
    print("PRIMARY ENDPOINT — pooled mixed model, diff ~ 1 + (1|cell) + (1|seed)")
    print("=" * 74)
    result = fit_pooled(frame)
    intercept = result.params["Intercept"]
    se = result.bse["Intercept"]
    pvalue = result.pvalues["Intercept"]
    low, high = intercept - 1.959964 * se, intercept + 1.959964 * se
    print(f"  intercept (X_d - opponent) : {intercept:+.4f}")
    print(f"  standard error             : {se:.4f}")
    print(f"  95% CI                     : [{low:+.4f}, {high:+.4f}]")
    print(f"  p (two-sided)              : {pvalue:.3g}")
    names = list(getattr(result.model, "exog_vc", {}).names) if hasattr(result.model, "exog_vc") else []
    comps = [float(v) for v in result.vcomp]
    labels = names if len(names) == len(comps) else [f"vc{i}" for i in range(len(comps))]
    print("  variance components        : " +
          ", ".join(f"{k} sd={math.sqrt(max(v, 0.0)):.4f}" for k, v in zip(labels, comps)))
    print(f"  residual sd                : {math.sqrt(max(result.scale, 0.0)):.4f}")
    print(f"\n  verdict at alpha=0.05      : "
          f"{'REJECT H0 — advantage is nonzero' if pvalue < 0.05 else 'fail to reject H0'}")

    if len({c.rsplit("_", 1)[0] for c in frame["cell"].unique()}) == 1 and len(frame["cell"].unique()) > 1:
        pass
    print("\n  ⚠️  The intercept is the mean of the CELL means, so pooling cells that\n"
          "      differ in OPPONENT answers no research question — a weak opponent\n"
          "      simply drags it up. S9 pools cells that vary MODEL and RW over a\n"
          "      fixed opponent set, with opponent as a moderator. On block E this\n"
          "      run is a technical demonstration that the endpoint fits, nothing more.")

    positive, n, p_sign = sign_test_across_cells(frame)
    print("\n" + "=" * 74)
    print("SECONDARY — sign test across cells")
    print("=" * 74)
    print(f"  cells with a positive mean : {positive}/{n}")
    print(f"  exact binomial p           : {p_sign:.3g}")
    if n < 6:
        floor = 2 * 0.5 ** n
        print(f"  ⚠️  with {n} cells the smallest attainable p is {floor:.3g} — the sign test\n"
              f"      cannot reach alpha=0.05 here. It needs S9's 22 cells to have power.")


if __name__ == "__main__":
    main()
