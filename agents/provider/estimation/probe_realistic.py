#!/usr/bin/env python3
"""S5-R — the S5 probe re-run under the live system's conditions.

S5 measured sigma with a fixed population level, 3-repeat-mean history costs,
3 histories x 3 repeats per requirement, and only the 10 held-out targets. The
live market gives an agent none of that. This probe keeps S5's harness — same
prompt renderer, same schema, parser, transport, cache — and replaces only the
*conditions* with the live agent's, so that if sigma lands near the pilot's
0.53 the gap is proven to be probe design, and if it lands near 0.39 something
in the market itself is still unexplained.

The live conditions replicated, each traced to the code that creates it:

==============  ================================================================
condition       source
==============  ================================================================
trajectory      the agent estimates requirement i with history = jobs it already
                completed in *this same sequence*, then completes i and moves on
history depth   caller's staging; default 5 targets at k=1, 5 at k=2, 10 at k=3
history costs   ONE realization per job (a seeded repeat of dataset v2), not the
                3-repeat mean -- a live job runs once
level           the EMA `cost_tracker.record_execution_cost` maintains:
                alpha 0.25, seeded from the s7 config prior $0.06, updated with
                each completed job's single realized cost. low/high never
                update under L2 and stay at their priors 0.03 / 0.10
quality         running mean of the completed jobs' evaluator ratings, starting
                from `initial_reputation_default` = 50
calls           ONE per target. No repeats, no draw averaging.
==============  ================================================================

Scoring stays S5's: debiased sigma of log(predicted / 3-repeat-mean realized),
medium tier, so the number is read against 0.3653 / 0.6491 directly.

    python -m agents.provider.estimation.probe_realistic --dry-run     # free
    python -m agents.provider.estimation.probe_realistic --trajectories 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics as st
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agents.provider.estimation import dataset_v2 as d2
from agents.provider.estimation import llm_client
from agents.provider.estimation import prompts_decomposed as pdc
from agents.provider.estimation import prompts_hybrid as ph
from agents.provider.estimation.probe_hybrid import _api_key, parse_hybrid


def parse_decomposed(content: str):
    """Validate the decomposed schema; carry the per-prompt depths.

    Returns an ``Estimate`` shaped like the point protocol's — with the model's
    own ``cost_multiplier``, which is the paired control — plus ``prompt_rounds``
    attached. The decomposed *estimate* is not computed here: it needs a
    calibration fitted on jobs completed later in the trajectory, so it is
    replayed offline from the recorded depths.
    """
    import json as _json
    try:
        payload = _json.loads(content)
        rounds = [float(r) for r in payload["prompt_rounds"]]
        # Absent in the production variants; the decomposed estimate never uses
        # it, and 1.0 is the heuristic, so a missing field degrades to H.
        multiplier = float((payload.get("cost_multiplier") or {}).get("medium", 1.0))
    except Exception:
        return parse_hybrid(content)  # its repair ladder, then a failed Estimate
    estimate = parse_hybrid(_json.dumps({
        "reasoning": payload.get("reasoning", ""),
        "cost_multiplier": {"medium": multiplier},
    }))
    estimate.prompt_rounds = rounds
    return estimate

TIER = "medium"

#: The live campaign cell's priors and EMA constant (s9_cells/*.yaml,
#: cost_tracker.__init__). The medium prior matters twice: it seeds the EMA and
#: it is what the level would be with no jobs at all.
PRIORS = {"low": 0.03, "medium": 0.06, "high": 0.10}
EMA_ALPHA = 0.25
INITIAL_QUALITY = 50.0

#: k shown to each target, in order. 20 targets: the user's staging.
DEFAULT_STAGING = (1,) * 5 + (2,) * 5 + (3,) * 10

DEFAULT_CSV = Path(__file__).resolve().parent / "results_realistic.csv"

CSV_COLUMNS = (
    "trajectory", "position", "repeat", "requirement_id", "k_shown", "level_stated",
    "cost_multiplier", "predicted_cost_usd", "realized_mean_usd",
    "realized_single_usd", "parse_status", "n_prompts", "prompt_rounds", "prompt_tokens",
    "completion_tokens", "call_cost_usd", "cached", "reasoning",
)


def _single_runs() -> Dict[str, List[Tuple[float, Optional[float]]]]:
    """requirement_id -> [(execution_cost, evaluator_rating)] by repeat, medium tier."""
    runs: Dict[str, Dict[int, Tuple[float, Optional[float]]]] = {}
    for line in (d2.DATASET_V2_DIR / "runs.jsonl").open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["tier"] != TIER or not row.get("execution_cost_usd"):
            continue
        runs.setdefault(row["requirement_id"], {})[int(row["repeat_idx"])] = (
            float(row["execution_cost_usd"]),
            float(row["evaluator_rating"]) if row.get("evaluator_rating") is not None else None,
        )
    return {rid: [by_repeat[i] for i in sorted(by_repeat)] for rid, by_repeat in runs.items()}


@dataclass
class AgentState:
    """The slice of live provider state the prompt can see."""

    level_medium: float = PRIORS[TIER]
    quality_sum: float = 0.0
    quality_n: int = 0
    #: (requirement_id, single realized cost) in completion order.
    completed: List[Tuple[str, float]] = field(default_factory=list)

    def complete(self, requirement_id: str, cost: float, rating: Optional[float]) -> None:
        # cost_tracker._update_ema, seeded from the prior on the first job.
        self.level_medium = EMA_ALPHA * cost + (1.0 - EMA_ALPHA) * self.level_medium
        if rating is not None:
            self.quality_sum += rating
            self.quality_n += 1
        self.completed.append((requirement_id, cost))

    def levels(self) -> Dict[str, Dict[str, float]]:
        quality = self.quality_sum / self.quality_n if self.quality_n else INITIAL_QUALITY
        return {
            tier: {
                "cost": self.level_medium if tier == TIER else PRIORS[tier],
                "quality": quality if tier == TIER else INITIAL_QUALITY,
            }
            for tier in ("low", "medium", "high")
        }


@dataclass
class PlannedCall:
    trajectory: int
    position: int
    requirement: d2.ProbeRequirement
    k_shown: int
    level_stated: float
    user_message: str
    realized_single: float
    #: Prompts in the target requirement — the length the depth array must match.
    n_prompts: int = 0


def _knn_select(
    target: d2.ProbeRequirement,
    completed: Sequence[Tuple[str, float]],
    requirements_by_id: Dict[str, d2.ProbeRequirement],
    k: int,
) -> List[Tuple[str, float]]:
    """The k most similar completed jobs, by z-scored mechanical features.

    Analogy-based estimation: distance over the same countable features B1
    regresses on, standardized over the agent's own pool so no population
    statistics leak in. Ties broken by recency. Most similar rendered first.
    """
    from agents.provider.estimation import dataset as ds

    pool = [(rid, cost, ds.mechanical_features(requirements_by_id[rid].visible))
            for rid, cost in completed]
    names = sorted(pool[0][2])
    stats = {}
    for name in names:
        values = [feats[name] for _, _, feats in pool]
        mean = st.fmean(values)
        sd = st.pstdev(values) or 1.0
        stats[name] = (mean, sd)
    target_feats = ds.mechanical_features(target.visible)

    def distance(feats: Dict[str, float]) -> float:
        return math.sqrt(sum(
            ((feats[n] - target_feats[n]) / stats[n][1]) ** 2 for n in names
        ))

    ranked = sorted(
        enumerate(pool), key=lambda iv: (distance(iv[1][2]), -iv[0])
    )
    return [(rid, cost) for _, (rid, cost, _) in ranked[:k]]


def plan_trajectory(
    index: int,
    requirements: Sequence[d2.ProbeRequirement],
    singles: Dict[str, List[Tuple[float, Optional[float]]]],
    *,
    seed: int,
    staging: Sequence[int] = DEFAULT_STAGING,
    warmup: int = 1,
    selection: str = "recency",
    protocol: str = "point",
    history_mode: str = "as_measured",
) -> List[PlannedCall]:
    """One agent lifetime: warmup job, then len(staging) estimated-and-completed jobs.

    The shuffle, the repeat drawn for each job, everything derives from
    (seed, trajectory index) so the plan the dry run prints is the plan the
    paid run executes.
    """
    rng = random.Random(f"s5r|{seed}|{index}")
    order = list(requirements)
    rng.shuffle(order)
    order = order[: len(staging) + warmup]  # warmup jobs + targets

    def draw(requirement: d2.ProbeRequirement) -> Tuple[float, Optional[float]]:
        repeats = singles[requirement.requirement_id]
        return repeats[rng.randrange(len(repeats))]

    state = AgentState()
    for warm in order[:warmup]:
        state.complete(warm.requirement_id, *draw(warm))

    by_id = {r.requirement_id: r for r in order}
    payloads = {r.requirement_id: r.visible.as_payload() for r in order}
    calls: List[PlannedCall] = []
    for position, (target, k) in enumerate(zip(order[warmup:], staging), start=1):
        levels = state.levels()
        chosen = (
            _knn_select(target, state.completed, by_id, k)
            if selection == "knn"
            else state.completed[-k:]
        )
        if protocol == "decomposed" and history_mode == "none":
            history = []
        elif protocol == "decomposed" and history_mode == "depth":
            history = [
                pdc.render_depth_example(
                    slot, payloads[rid],
                    realized_cost=cost, tier_level_cost=levels[TIER]["cost"],
                )
                for slot, (rid, cost) in enumerate(chosen, start=1)
            ]
        else:
            history = [
                ph.render_history_example(
                    slot,
                    payloads[rid],
                    tier=TIER,
                    realized_cost=cost,
                    tier_level_cost=levels[TIER]["cost"],
                )
                for slot, (rid, cost) in enumerate(chosen, start=1)
            ]
        cost, rating = draw(target)
        calls.append(
            PlannedCall(
                trajectory=index,
                position=position,
                requirement=target,
                k_shown=len(history),
                level_stated=levels[TIER]["cost"],
                user_message=(
                    pdc.render_user_message(
                        payloads[target.requirement_id], levels, history=history,
                        with_multiplier=(history_mode == "as_measured"),
                    )
                    if protocol == "decomposed"
                    else ph.render_user_message(
                        payloads[target.requirement_id], levels, history=history, level="L2"
                    )
                ),
                n_prompts=len(payloads[target.requirement_id].get("prompts") or []),
                realized_single=cost,
            )
        )
        state.complete(target.requirement_id, cost, rating)
    return calls


def debiased_sigma(log_ratios: Sequence[float]) -> Tuple[float, float]:
    bias = st.fmean(log_ratios)
    return math.sqrt(st.fmean((x - bias) ** 2 for x in log_ratios)), bias


def score(rows: List[Dict], realized_mean: Dict[str, float]) -> None:
    ok = [r for r in rows if r["parse_status"] == "ok"]
    print(f"\n## S5-R — {len(ok)}/{len(rows)} calls parsed, "
          f"{len({r['trajectory'] for r in ok})} trajectories")
    residuals = [
        math.log(float(r["predicted_cost_usd"]) / realized_mean[r["requirement_id"]])
        for r in ok
    ]
    sigma, bias = debiased_sigma(residuals)
    print(f"\n  sigma_cost, SINGLE call (debiased, medium)        {sigma:.4f}   bias {bias:+.4f}")
    grouped: Dict[Tuple, List[Dict]] = {}
    for r in ok:
        grouped.setdefault((r["trajectory"], r["position"]), []).append(r)
    if any(len(v) > 1 for v in grouped.values()):
        averaged = [
            math.log(
                st.fmean(float(r["predicted_cost_usd"]) for r in v)
                / realized_mean[v[0]["requirement_id"]]
            )
            for v in grouped.values()
        ]
        s_avg, b_avg = debiased_sigma(averaged)
        print(f"  sigma_cost, {max(len(v) for v in grouped.values())} repeats AVERAGED (S5 convention)   "
              f"{s_avg:.4f}   bias {b_avg:+.4f}")
    print("\n  reference points:")
    for label, value in (
        ("S5 idealised probe, same model, held-out", 0.3653),
        ("live pilot, all 30, as bid", 0.5309),
        ("B0 / arm H", 0.6491),
    ):
        print(f"    {label:44s} {value:.4f}")
    print("\n  by history stage:")
    for k in sorted({r["k_shown"] for r in ok}):
        subset = [
            math.log(float(r["predicted_cost_usd"]) / realized_mean[r["requirement_id"]])
            for r in ok if r["k_shown"] == k
        ]
        if len(subset) > 2:
            s, b = debiased_sigma(subset)
            print(f"    k={k}: sigma {s:.4f}  bias {b:+.4f}  (n={len(subset)})")
    print("\n  by trajectory:")
    for t in sorted({r["trajectory"] for r in ok}):
        subset = [
            math.log(float(r["predicted_cost_usd"]) / realized_mean[r["requirement_id"]])
            for r in ok if r["trajectory"] == t
        ]
        s, b = debiased_sigma(subset)
        print(f"    trajectory {t}: sigma {s:.4f}  bias {b:+.4f}  (n={len(subset)})")
    spend = sum(float(r["call_cost_usd"] or 0) for r in rows)
    print(f"\n  spend ${spend:.4f}")


async def run(args: argparse.Namespace) -> None:
    split = d2.load_split_dataset()
    requirements = list(split.dev.requirements) + list(split.heldout.requirements)
    singles = _single_runs()
    realized_mean = {r.requirement_id: r.realized_cost(TIER) for r in requirements}

    staging = (args.k,) * len(DEFAULT_STAGING) if args.k else DEFAULT_STAGING
    plans = [
        call
        for t in range(args.trajectories)
        for call in plan_trajectory(
            t, requirements, singles, seed=args.seed,
            staging=staging, warmup=args.warmup, selection=args.selection,
            protocol=args.protocol, history_mode=args.history_mode,
        )
    ]

    print(f"planned: {args.trajectories} trajectories x {len(staging)} targets x {args.repeats} repeats "
          f"= {len(plans) * args.repeats} calls, model {args.model}")
    mean_chars = st.fmean(len(p.user_message) for p in plans)
    est_cost = len(plans) * args.repeats * (mean_chars / 3.7 * 0.15 + 120 * 0.60) / 1e6
    print(f"mean prompt {mean_chars:.0f} chars — estimated spend ${est_cost:.3f}\n")
    print(f"{'traj':>4} {'pos':>3} {'requirement':30s} {'k':>2} {'level':>8} {'true_mean':>9}")
    for p in plans:
        print(f"{p.trajectory:>4} {p.position:>3} {p.requirement.requirement_id:30s} "
              f"{p.k_shown:>2} {p.level_stated:8.4f} "
              f"{realized_mean[p.requirement.requirement_id]:9.4f}")

    if args.dry_run:
        if args.show_prompt is not None:
            print("\n" + "=" * 72)
            print(plans[args.show_prompt].user_message)
        print("\ndry run — nothing sent. Re-run without --dry-run to spend.")
        return

    model = llm_client.model_by_key(args.model, llm_client.ALL_MODELS)
    probe = llm_client.OpenRouterProbe(
        _api_key(), max_spend_usd=args.max_spend
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(plan: PlannedCall, repeat: int = 0) -> Dict:
        async with semaphore:
            result = await probe.call_live(
                model,
                user_message=plan.user_message,
                system_message=(pdc.SYSTEM_MESSAGE if args.protocol == "decomposed"
                                else ph.SYSTEM_MESSAGES["L2"]),
                schema=(pdc.response_schema(
                            plan.n_prompts,
                            with_multiplier=(args.history_mode == "as_measured"))
                        if args.protocol == "decomposed" else ph.L2_RESPONSE_SCHEMA),
                parser=(parse_decomposed if args.protocol == "decomposed" else parse_hybrid),
                run_seed=args.seed,
                # repeat 0 reproduces the single-repeat run's cache key exactly,
                # so those 60 calls replay free; repeats 1, 2 are new samples.
                agent_id=plan.trajectory + 1000 * repeat,
            )
        estimate = result.estimate
        multiplier = float(estimate.cost_multiplier.get(TIER, float("nan"))) \
            if not estimate.failed else float("nan")
        return {
            "trajectory": plan.trajectory,
            "position": plan.position,
            "repeat": repeat,
            "requirement_id": plan.requirement.requirement_id,
            "k_shown": plan.k_shown,
            "level_stated": plan.level_stated,
            "cost_multiplier": multiplier,
            "predicted_cost_usd": plan.level_stated * multiplier,
            "realized_mean_usd": realized_mean[plan.requirement.requirement_id],
            "realized_single_usd": plan.realized_single,
            "parse_status": "ok" if not estimate.failed else estimate.status,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "call_cost_usd": result.cost_usd,
            "cached": result.cached,
            # Raw depths, so every calibration variant is replayable offline
            # from one paid run. Empty for the point protocol.
            "n_prompts": plan.n_prompts,
            "prompt_rounds": json.dumps(
                pdc.clean_rounds(getattr(estimate, "prompt_rounds", []) or [], plan.n_prompts)
                or []
            ),
            "reasoning": (estimate.reasoning or "")[:400],
        }

    jobs = [(p, rep) for p in plans for rep in range(args.repeats)]
    rows = [await one(p, rep) for p, rep in jobs] if args.concurrency == 1 else \
        list(await asyncio.gather(*(one(p, rep) for p, rep in jobs)))
    rows.sort(key=lambda r: (r["trajectory"], r["position"], r["repeat"]))

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
    score(rows, realized_mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1,
                        help="jobs completed before the first estimate")
    parser.add_argument("--k", type=int, default=None,
                        help="uniform history depth for every target (overrides the default staging)")
    parser.add_argument("--selection", default="recency", choices=("recency", "knn"),
                        help="which completed jobs the prompt shows")
    parser.add_argument("--protocol", default="point", choices=("point", "decomposed"),
                        help="point = the live S5/S7 protocol; decomposed = the S5-D branch")
    parser.add_argument("--history-mode", default="as_measured",
                        choices=("as_measured", "none", "depth"),
                        help="decomposed only. as_measured = 3 past jobs by cost + the "
                             "cost_multiplier field (the validated protocol); none = neither; "
                             "depth = 3 past jobs described per prompt, no cost_multiplier")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-spend", type=float, default=0.25)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-prompt", type=int, default=None,
                        help="dry run: print this planned call's full prompt")
    parser.add_argument("--report-only", action="store_true",
                        help="rescore an existing --out CSV, no calls")
    args = parser.parse_args()

    if args.report_only:
        rows = list(csv.DictReader(args.out.open()))
        for r in rows:
            r["trajectory"], r["k_shown"] = int(r["trajectory"]), int(r["k_shown"])
        split = d2.load_split_dataset()
        realized = {r.requirement_id: r.realized_cost(TIER)
                    for part in (split.dev, split.heldout) for r in part.requirements}
        score(rows, realized)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
