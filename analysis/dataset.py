"""Dataset v3 — the recorded service executions every market run replays.

Path is the one the market uses (`sampled_dataset_dir` in s9_base_v3.yaml):
`.../executor_tier_dataset_v3/final`, 90 runs = 30 requirements x 3 recorded
executions. The folder above it holds 94 runs / 34 requirements; the extra four
were not part of the market and are excluded here.
"""
import collections
import json
import math
import statistics as st
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from agents.common import paths

RUNS = paths.DATASET_FINAL_DIR / "runs.jsonl"

#: Designed synthesis depth, from least to most demanding.
DEPTHS = ["locate", "enumerate", "contrast", "reconcile"]


@lru_cache(maxsize=None)
def runs() -> tuple:
    rows = tuple(json.loads(l) for l in RUNS.read_text().splitlines() if l.strip())
    if len(rows) != 90 or len({r["requirement_id"] for r in rows}) != 30:
        raise ValueError(f"{RUNS}: expected 90 runs / 30 requirements")
    return rows


def by_requirement() -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in runs():
        out[r["requirement_id"]].append(r)
    return dict(out)


def corr(x, y) -> float:
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def between_within(field_x: str, field_y: str):
    """Correlation of requirement means, and of run deviations from them."""
    groups = by_requirement().values()
    mx = [st.mean(r[field_x] for r in g) for g in groups]
    my = [st.mean(r[field_y] for r in g) for g in groups]
    dx, dy = [], []
    for g in groups:
        ax, ay = st.mean(r[field_x] for r in g), st.mean(r[field_y] for r in g)
        dx += [r[field_x] - ax for r in g]
        dy += [r[field_y] - ay for r in g]
    return corr(mx, my), corr(dx, dy)


def share_between(field: str) -> float:
    """Share of the field's variance explained by which requirement it is."""
    allv = [r[field] for r in runs()]
    means = {k: st.mean(r[field] for r in g) for k, g in by_requirement().items()}
    return st.pvariance([means[r["requirement_id"]] for r in runs()]) / st.pvariance(allv)
