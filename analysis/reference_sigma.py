#!/usr/bin/env python3
"""S6's reference σ, recomputed from the committed probe artifacts (S6 §A/§B).

    python -m analysis.reference_sigma

Free — it reads CSVs already in the repo and makes no calls. Every number S6
prices its arms against is produced here rather than quoted from a report, so
the *convention* travels with the *value*: `test_s6_constants.py` recomputes
each literal in `decision_value.py` and `estimators.py` from these functions and
fails if the two drift apart.

Why this module exists at all
-----------------------------

The re-run's headline σ is quoted three different ways in three documents, and
the three differ by more than the margin G2 turns on:

===========================================  ========  ===========================
convention                                   σ         where it comes from
===========================================  ========  ===========================
mean of 3 calls vs the 3-run mean, per draw   0.2913    `research_plan.md` G1 row
**one call vs the 3-run mean, per draw**      **0.3116**  **what S6 is fed**
one call vs the 3-run mean, all draws pooled  0.3283    a single global offset
===========================================  ========  ===========================

They are all the same predictions. What changes is how many LLM calls the agent
is assumed to make and whether the per-history offset is fitted per draw.

**S6 uses the single-call figure**, for one reason that decides it: the ensemble
costs 3× inference. At `gpt-5.4-mini`'s $0.00206/call that is $0.0062 per auction
against a P6 value-per-auction on the order of $0.002, so the 0.0203 σ the
ensemble buys cannot repay what it costs. Pricing the ensemble would price an
estimator dominated by the cheaper one it is being compared against.

The target stays the 3-run mean, and that is not a convenience
--------------------------------------------------------------

`s6_implementation.md` §B labels 0.3116 "single call vs a single realized draw".
**It is not** — it is a single call against the mean of the requirement's runs
(this module reproduces all three rows above from the same file, so the label,
not the number, is what was wrong). Scoring against individual runs is a fourth
number, ~5.5% higher (`s5_open_problems.md` P7).

Keeping the 3-run mean is what matches the instrument. The simulator's
:class:`~experiments.simulator.estimators.NoisyOracleEstimator` perturbs
``DatasetOutcomes.ground_truth()``, which *is* the mean over recorded runs, while
settlement draws an individual run via ``sample()``. Run-to-run execution noise
is therefore already modelled inside the market. A σ measured against individual
runs would charge the estimator for that dispersion a second time.

The regression opponent
-----------------------

Arm **R** is not a σ — it is the real fitted table (`decision_value.regression_
cost_table`). Its σ here is a *label on the curve*, and two are needed:

* :func:`regression_sigma` on held-out (**0.4212**) — the like-for-like pairing
  with the hybrid, same 10 requirements, same target, same debiasing;
* :func:`regression_sigma` over all 30 (**0.4133**) — the accuracy the opponent
  actually operates at, because the market draws all 30 requirements and the
  table is fitted on the 20 dev ones. B1 barely overfits (dev 0.4093), so the two
  are close; the gap is stated rather than averaged away.

S6 §A proposes reconciling 0.4212 against an independent refit's 0.4339. That
refit is not reproduced here: it is a different fit, and adopting it would break
the like-for-like pairing with 0.3116 to close a 0.013 gap in a label.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agents.provider.estimation import probe_hybrid as ph

DATA_DIR = Path(__file__).resolve().parent

#: The single held-out run G1 was issued on — 10 requirements × 3 draws × 3
#: repeats, `X_k_random`, stage 5.
HYBRID_HELDOUT_CSV = DATA_DIR / "results_hybrid_heldout.csv"

#: S5's probe under protocol `s2-v5`, carrying the B0/B1 baselines.
PROBE_V2_CSV = DATA_DIR / "results_v2.csv"

#: L2 delivers medium and prices off it alone, so every reference is medium-tier.
#: A pooled σ overstates the error by including tiers the controller never uses —
#: which is exactly the defect S6 §A corrects in P6's reference lines.
TIER = "medium"

#: The best mechanical control, and the arm whose predictions become the R
#: opponent's lookup table.
REGRESSION_ARM = "B1_surface_pooled"

#: The task-blind control: guess the tier mean.
TASK_BLIND_ARM = "B0_tier_mean"


@dataclass(frozen=True)
class SigmaReading:
    """A σ and the spread across history draws it must never be quoted without.

    G1's caveat is structural, not decorative: at 10 requirements and 3 draws one
    draw lands at 0.4349 and sits near break-even against a free regression. A
    bare mean hides that, so the per-draw values are carried in the type.
    """

    sigma: float
    spread: float
    per_draw: Sequence[float]

    def __str__(self) -> str:
        draws = " / ".join(f"{s:.4f}" for s in self.per_draw)
        return f"{self.sigma:.4f} +/- {self.spread:.4f}  [{draws}]"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_hybrid_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Held-out hybrid predictions, typed the way the scoring helpers expect."""
    rows = []
    with (path or HYBRID_HELDOUT_CSV).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["parse_status"] != "ok":
                continue
            rows.append({
                **row,
                "draw": int(row["draw"]),
                "repeat": int(row["repeat"]),
                "predicted_cost_usd": float(row["predicted_cost_usd"]),
                "realized_cost_usd": float(row["realized_cost_usd"]),
            })
    if not rows:
        raise ValueError(f"no usable rows in {path or HYBRID_HELDOUT_CSV}")
    return rows


def load_probe_v2_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """S5 probe predictions for the mechanical baselines."""
    with (path or PROBE_V2_CSV).open(encoding="utf-8") as handle:
        return [
            {
                **row,
                "predicted_cost_usd": float(row["predicted_cost_usd"]),
                "realized_cost_usd": float(row["realized_cost_usd"]),
            }
            for row in csv.DictReader(handle)
        ]


#: S5 repeated on dataset v3 (`probe_v3.py`, 5 trajectories x 30 requirements,
#: gpt-4o-mini, $0.011). The first measurement in which arms X and R-D differ in
#: *one* thing only — where the retrieval depth comes from — because dataset v3
#: is the first to record realised depth, and R-D shares arm X's stage 2 byte for
#: byte.
PROBE_V3_CSV = DATA_DIR / "results_v3_probe.csv"

#: Prequential, not cross-validated: requirements are walked in a shuffled order
#: and each arm may use only jobs it has already completed. That is the live
#: agent's constraint, and it is what makes these figures realistic rather than
#: an in-sample fit — the distinction that invalidated S5's original readings.
S5_V3_ARMS = ("H", "R", "R-D", "X")


def load_probe_v3_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Per-prediction rows from the v3 probe, one per (trajectory, step, arm)."""
    with (path or PROBE_V3_CSV).open(encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({
                **row,
                "priced": row["priced"] == "1",
                "realized_cost_usd": float(row["realized_cost_usd"]),
                "predicted_cost_usd": (
                    float(row["predicted_cost_usd"]) if row["predicted_cost_usd"] else None
                ),
            })
        return rows


def probe_v3_sigma(arm: str, rows: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, float]:
    """Debiased σ, bias and coverage for one arm of the v3 probe.

    ``priced`` matters as much as ``sigma``: an arm that degrades to the
    heuristic is not scored on those auctions, so a low σ over few auctions is
    not the same result as the same σ over all of them. R-D prices 125 of 150
    because it must learn features -> depth from its own completed jobs before it
    can price at all; X prices 145, the same coverage as H, because its stage 1
    needs no history.
    """
    rows = rows if rows is not None else load_probe_v3_rows()
    mine = [r for r in rows if r["arm"] == arm]
    if not mine:
        raise KeyError(f"no rows for arm {arm!r}; have {sorted({r['arm'] for r in rows})}")
    residuals = [
        math.log(r["predicted_cost_usd"] / r["realized_cost_usd"])
        for r in mine if r["priced"] and r["predicted_cost_usd"]
    ]
    bias = sum(residuals) / len(residuals)
    return {
        "sigma": debiased_sigma(residuals),
        "bias": bias,
        "priced": float(len(residuals)),
        "total": float(len(mine)),
    }


# ---------------------------------------------------------------------------
# The three conventions
# ---------------------------------------------------------------------------

def debiased_sigma(residuals: Sequence[float]) -> float:
    """RMS of ``log(predicted/realized)`` about its own mean.

    Debiased because a constant multiplicative offset is learnable from the
    agent's own history without reading a single task — charging the estimator
    for it would credit it with information it does not need.
    """
    if not residuals:
        raise ValueError("no residuals")
    offset = statistics.mean(residuals)
    return math.sqrt(sum((r - offset) ** 2 for r in residuals) / len(residuals))


def ensemble_sigma(rows: Sequence[Dict[str, Any]]) -> SigmaReading:
    """**0.2913** — the mean of 3 calls, scored per draw. The published headline.

    Delegates to :func:`probe_hybrid.score_draws` so there is exactly one
    implementation of the convention G1 was issued under. Reported by S6 as a
    stated variant; it is **not** the G2 headline, because the 3× inference it
    assumes costs more than the accuracy it buys.
    """
    per_draw = [s.sigma for s in ph.score_draws(rows)]
    return SigmaReading(statistics.mean(per_draw), statistics.pstdev(per_draw), tuple(per_draw))


def single_call_sigma(rows: Sequence[Dict[str, Any]]) -> SigmaReading:
    """**0.3116** — what one call achieves. The figure S6 prices.

    Every repeat is its own residual rather than being averaged into one, so the
    reading is "an agent that calls once", which is the deployed action. The
    offset is still fitted per draw: a deployed agent debiases against the
    history it actually has, not against three histories it does not.
    """
    per_draw = []
    for draw in sorted({row["draw"] for row in rows}):
        residuals = [
            math.log(row["predicted_cost_usd"] / row["realized_cost_usd"])
            for row in rows
            if row["draw"] == draw
            and row["predicted_cost_usd"] > 0
            and row["realized_cost_usd"] > 0
        ]
        per_draw.append(debiased_sigma(residuals))
    return SigmaReading(statistics.mean(per_draw), statistics.pstdev(per_draw), tuple(per_draw))


def single_call_pooled_sigma(rows: Sequence[Dict[str, Any]]) -> float:
    """**0.3283** — one call, one global offset across all draws.

    The pessimistic reading: an agent that cannot recalibrate when its history
    changes. Carried to bracket the convention, not used.

    Not to be confused with :func:`probe_hybrid.pooled_sigma`, which pools in the
    other direction — it averages *predictions* across draws, an ensemble over
    histories no single agent has.
    """
    residuals = [
        math.log(row["predicted_cost_usd"] / row["realized_cost_usd"])
        for row in rows
        if row["predicted_cost_usd"] > 0 and row["realized_cost_usd"] > 0
    ]
    return debiased_sigma(residuals)


# ---------------------------------------------------------------------------
# The opponents
# ---------------------------------------------------------------------------

def baseline_sigma(
    rows: Sequence[Dict[str, Any]],
    arm: str,
    *,
    split: Optional[str] = "heldout",
    tier: str = TIER,
) -> float:
    """Debiased σ for one S5 baseline arm. ``split=None`` scores all 30.

    Same target and same debiasing as the hybrid readings above, which is the
    whole point: a σ is only comparable to another σ measured the same way.
    """
    selected = [
        row for row in rows
        if row["arm"] == arm
        and row["tier"] == tier
        and (split is None or row["split"] == split)
    ]
    if not selected:
        raise ValueError(f"no rows for arm={arm!r} split={split!r} tier={tier!r}")
    return debiased_sigma([
        math.log(row["predicted_cost_usd"] / row["realized_cost_usd"]) for row in selected
    ])


def regression_sigma(
    rows: Optional[Sequence[Dict[str, Any]]] = None, *, split: Optional[str] = "heldout"
) -> float:
    """Arm **R**'s σ: 0.4212 on held-out, 0.4133 over all 30 (``split=None``).

    The second is what the R arm operates at inside the simulator — the market
    draws every requirement, and the table is in-sample on the 20 it was fitted
    on. That it is *lower* than held-out is the direction to state plainly: the
    opponent S6 runs against is slightly stronger than a fresh regression, so the
    hybrid's margin is measured against the harder of the two.
    """
    return baseline_sigma(rows or load_probe_v2_rows(), REGRESSION_ARM, split=split)


def task_blind_sigma(
    rows: Optional[Sequence[Dict[str, Any]]] = None, *, split: Optional[str] = "heldout"
) -> float:
    """Arm **H**'s σ: 0.6491 on held-out. The floor both informed arms beat."""
    return baseline_sigma(rows or load_probe_v2_rows(), TASK_BLIND_ARM, split=split)


# ---------------------------------------------------------------------------
# Why L3 is closed (S6 §D)
# ---------------------------------------------------------------------------

def tier_deviation_correlations(
    runs_path: Optional[Path] = None,
) -> Dict[str, float]:
    """The mechanism behind §D, recomputed from `final/runs.jsonl`.

    L3's gate reads only the **premium** ``high − medium``. `hybrid_agent_design.md`
    §3.2a argued that made it *easier* to predict than per-tier σ suggests, because
    correlated errors cancel in a difference. The premise is right and the
    conclusion is backwards: the correlation is **+0.97**, so what cancels is the
    **signal**. Difficulty is a common factor — a requirement expensive at medium
    is expensive at high in the same proportion — and only ~9% of high-tier
    deviation variance survives into the premium.

    Here rather than in prose because it is a paper-bound claim about a mechanism,
    and the first thing a reviewer will do is recompute it. ``surviving_variance``
    is the load-bearing number: it is what is left for *any* L3 estimator to
    predict, not just the one that was tried.
    """
    import json
    from collections import defaultdict

    path = runs_path or (
        DATA_DIR.parents[1]
        / "experiments/executor_analysis/data/datasets/executor_tier_dataset_v2/final/runs.jsonl"
    )
    costs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("execution_cost_usd") is None:
                continue
            costs[(row["requirement_id"], row["tier"])].append(
                float(row["execution_cost_usd"])
            )
    means = {key: statistics.mean(values) for key, values in costs.items()}

    def deviations(tier: str) -> Dict[str, float]:
        logs = {
            requirement: math.log(cost)
            for (requirement, t), cost in means.items()
            if t == tier and cost > 0
        }
        level = statistics.mean(logs.values())
        return {requirement: value - level for requirement, value in logs.items()}

    by_tier = {tier: deviations(tier) for tier in ("low", "medium", "high")}

    def correlation(first: str, second: str) -> float:
        keys = sorted(set(by_tier[first]) & set(by_tier[second]))
        xs = [by_tier[first][k] for k in keys]
        ys = [by_tier[second][k] for k in keys]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denominator = math.sqrt(
            sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
        )
        return numerator / denominator

    keys = sorted(set(by_tier["medium"]) & set(by_tier["high"]))
    premium = [by_tier["high"][k] - by_tier["medium"][k] for k in keys]
    high = [by_tier["high"][k] for k in keys]
    sd_high, sd_premium = statistics.pstdev(high), statistics.pstdev(premium)

    return {
        "n_requirements": float(len(keys)),
        "medium_high": correlation("medium", "high"),
        "low_medium": correlation("low", "medium"),
        "low_high": correlation("low", "high"),
        "sd_high_deviation": sd_high,
        "sd_premium": sd_premium,
        "surviving_variance": (sd_premium / sd_high) ** 2 if sd_high else 0.0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def reference_table() -> Dict[str, float]:
    """Every σ S6 pins, in one dict. The test asserts the literals against this."""
    hybrid = load_hybrid_rows()
    baselines = load_probe_v2_rows()
    return {
        "hybrid_single_call": single_call_sigma(hybrid).sigma,
        "hybrid_ensemble": ensemble_sigma(hybrid).sigma,
        "hybrid_single_call_pooled": single_call_pooled_sigma(hybrid),
        "regression_heldout": regression_sigma(baselines, split="heldout"),
        "regression_all": regression_sigma(baselines, split=None),
        "task_blind_heldout": task_blind_sigma(baselines, split="heldout"),
    }


def format_report() -> str:
    hybrid = load_hybrid_rows()
    baselines = load_probe_v2_rows()
    lines = [
        "# S6 reference sigma (medium tier, debiased)",
        "",
        f"  source: {HYBRID_HELDOUT_CSV.name} ({len(hybrid)} calls), "
        f"{PROBE_V2_CSV.name}",
        "",
        "## The hybrid, by convention",
        "",
        f"  one call vs 3-run mean, per draw   {single_call_sigma(hybrid)}   <- S6 headline",
        f"  mean of 3 calls, per draw          {ensemble_sigma(hybrid)}   3x inference",
        f"  one call, one global offset        {single_call_pooled_sigma(hybrid):.4f}",
        "",
        "## The opponents",
        "",
        f"  B1 regression, held-out (10)       {regression_sigma(baselines):.4f}"
        "   <- like-for-like with the hybrid",
        f"  B1 regression, all 30              {regression_sigma(baselines, split=None):.4f}"
        "   <- what arm R runs at in-market",
        f"  B1 regression, dev (20, in-sample) "
        f"{regression_sigma(baselines, split='dev'):.4f}",
        f"  B0 task-blind, held-out            {task_blind_sigma(baselines):.4f}",
        "",
        "## Why L3 is closed (S6 §D) — from final/runs.jsonl",
        "",
    ]
    tiers = tier_deviation_correlations()
    lines += [
        f"  corr(medium, high) deviations      {tiers['medium_high']:+.4f}"
        f"   (low/medium {tiers['low_medium']:+.4f}, low/high {tiers['low_high']:+.4f})",
        f"  sd(high dev) -> sd(premium)        {tiers['sd_high_deviation']:.4f}"
        f" -> {tiers['sd_premium']:.4f}",
        f"  variance surviving into the premium {tiers['surviving_variance']:.1%}"
        "   <- all any L3 estimator could predict",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    print(format_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
