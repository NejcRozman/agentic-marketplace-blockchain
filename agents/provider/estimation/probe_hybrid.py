#!/usr/bin/env python3
"""S5 re-run — the hybrid estimator's offline probe (step 5), Level 2.

```
python -m agents.provider.estimation.probe_hybrid --stage 0    # 3 calls,  ~$0.01
python -m agents.provider.estimation.probe_hybrid --stage 1    # 20 calls, ~$0.07
python -m agents.provider.estimation.probe_hybrid --stage 2    # 180 calls, ~$0.63
python -m agents.provider.estimation.probe_hybrid --report-only  # rescore, free
```

**Staged deliberately.** The redesign changed four things at once — corrected
tier mechanics, the agent's own cost level stated, k-shot history, ``confidence``
dropped — so the first question is not "what is σ" but "does any of this work,
and if not, which change failed". Stage 1 answers that for **$0.07**; stage 2 is
only paid once there is a reason to.

**Only the X_k arm runs here.** H is B0 by construction, R@k and R@20 are pure
computation, and X₀ under the old prompt is already measured (σ_medium 0.4937,
450 cached calls). X_kf needs a completed X_k pass and is an ablation, not the
hypothesis. None of them justifies a call at this stage.

Attribution without extra spend
-------------------------------

σ alone cannot say *which* change failed. These five, all computed from the same
predictions, separate the failure modes:

============  ==================================================================
diagnostic    a bad value means
============  ==================================================================
bias          ≠ 0 → the stated cost level is not being used
slope         ≫ 1 → ranks but compresses; it has not learned the *range*
spread        narrow vs realized → not discriminating between requirements
rho           ≈ 0 → it cannot read the task; the prompt is wrong, not miscalibrated
draw sd       large → *which* examples were shown dominates the answer
============  ==================================================================

The ``reasoning`` strings are the sixth, and the only qualitative one: a model
talking about retrieval rounds and cross-document synthesis has read the
mechanics block; one talking about prompt length has not.

**Targets are medium-tier**, because L2 delivers medium and ``h2_strategy_3``
prices off ``execution_cost_micro`` alone. Reading a pooled σ against them
overstates the error by including a tier the controller never uses.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.provider.estimation import (
    dataset as ds,
    dataset_v2 as d2,
    history as H,
    llm_client,
    metrics,
    prompts_hybrid as ph,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = OUTPUT_DIR / "results_hybrid.csv"

#: L2 delivers medium and prices off it alone.
TIER = "medium"

#: Held-out, debiased, **medium-tier** σ_cost. Everything the run is read
#: against. Recomputed from ``results_v2.csv``; see the module docstring for why
#: these are not the pooled figures quoted in ``s5_report.md``.
TARGETS: Dict[str, float] = {
    "B0 / arm H (guess the tier mean)": 0.6491,
    "B1 regression on 20 dev": 0.4212,
    "economic target (clears G2)": 0.3750,
    "gpt-5.4-mini, old prompt, zero-shot": 0.4937,
    "gpt-5.4, old prompt, zero-shot": 0.3500,
}

STAGES: Dict[int, Dict[str, Any]] = {
    0: {"requirements": 3, "draws": 1, "repeats": 1, "selection": "random"},
    1: {"requirements": 20, "draws": 1, "repeats": 1, "selection": "random"},
    2: {"requirements": 20, "draws": 3, "repeats": 3, "selection": "random"},
    # The mature-agent endpoint. ``span`` selection is deterministic, so one
    # "draw" is the whole arm — 60 calls rather than 180.
    3: {"requirements": 20, "draws": 1, "repeats": 3, "selection": "span"},
    # Same idea, anchored at the 10th/50th/90th percentile instead of the
    # literal extremes — the min and max are single observations and the most
    # extreme points in the history.
    4: {"requirements": 20, "draws": 1, "repeats": 3, "selection": "span90"},
    # ---- HELD-OUT. One run only (§B). Configuration frozen before it fires. ----
    5: {"requirements": 10, "draws": 3, "repeats": 3, "selection": "random"},
    6: {"requirements": 10, "draws": 1, "repeats": 3, "selection": "span"},
}

CSV_COLUMNS = (
    "protocol_version", "fingerprint", "stage", "arm", "model", "split",
    "probe_id", "requirement_id", "tier", "draw", "repeat",
    "history_ids", "history_n",
    "cost_multiplier", "predicted_cost_usd", "realized_cost_usd",
    "parse_status", "prompt_tokens", "completion_tokens", "call_cost_usd",
    "cached", "reasoning",
)


# ---------------------------------------------------------------------------
# Parsing — the hybrid schema, not S2's
# ---------------------------------------------------------------------------

def parse_hybrid(content: str, tier: str = TIER) -> llm_client.Estimate:
    """Parse ``{"cost_multiplier": {"medium": x}, "reasoning": "..."}``.

    Reuses S2's repair ladder (whole-body JSON → first ``{...}`` span → literal
    eval) and its pre-specified failure rule: anything unparseable, refused, or
    out of bounds becomes multiplier 1.0 flagged as failed, which scores exactly
    as arm H. An arm must not be able to improve its σ by declining the
    requirements it finds hard.
    """
    base = llm_client.parse_estimate(content)
    if base.failed and not (content or "").strip():
        return base

    payload = llm_client._try_json((content or "").strip())
    if payload is None:
        match = llm_client._JSON_BLOCK.search(content or "")
        if match:
            payload = llm_client._try_json(match.group()) or llm_client._try_literal(match.group())
    if not isinstance(payload, dict):
        return llm_client.Estimate.fallback(base.status if base.failed else "malformed")

    try:
        raw = float(payload["cost_multiplier"][tier])
    except (KeyError, TypeError, ValueError):
        return llm_client.Estimate.fallback("malformed")
    if raw != raw or raw in (float("inf"), float("-inf")):
        return llm_client.Estimate.fallback("malformed")

    low, high = ph.COST_MULTIPLIER_BOUNDS
    clamped = () if low <= raw <= high else (f"cost_multiplier.{tier}",)
    return llm_client.Estimate(
        cost_multiplier={tier: max(low, min(high, raw))},
        quality_delta={},
        confidence=0.0,
        reasoning=str(payload.get("reasoning", "")),
        status="ok",
        clamped=clamped,
    )


def parse_hybrid_l3(content: str) -> llm_client.Estimate:
    """Parse L3's four fields.

    ``high_to_medium_ratio`` is carried in ``cost_multiplier["premium"]`` rather
    than as a new field on :class:`Estimate` — the dataclass is S2's and shared
    with the frozen path. The controller reconstructs the high-tier cost as
    ``level_medium x multiplier_medium x premium``.
    """
    payload = llm_client._try_json((content or "").strip())
    if payload is None:
        match = llm_client._JSON_BLOCK.search(content or "")
        if match:
            payload = llm_client._try_json(match.group()) or llm_client._try_literal(match.group())
    if not isinstance(payload, dict):
        return llm_client.Estimate.fallback("malformed")

    def bounded(value, low, high, label, clamped):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(label)
        if not low <= number <= high:
            clamped.append(label)
        return max(low, min(high, number))

    clamped: List[str] = []
    try:
        medium = bounded(payload["cost_multiplier"]["medium"],
                         *ph.COST_MULTIPLIER_BOUNDS, "cost_multiplier.medium", clamped)
        premium = bounded(payload["high_to_medium_ratio"],
                          *ph.PREMIUM_RATIO_BOUNDS, "high_to_medium_ratio", clamped)
    except (KeyError, TypeError, ValueError):
        return llm_client.Estimate.fallback("malformed")

    return llm_client.Estimate(
        cost_multiplier={"medium": medium, "premium": premium},
        quality_delta={},
        confidence=0.0,
        reasoning=str(payload.get("reasoning", "")),
        status="ok",
        clamped=tuple(clamped),
    )


# ---------------------------------------------------------------------------
# Building the run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One (requirement, draw) pair and the prompt it produces."""

    target: ds.ProbeRequirement
    draw: H.HistoryDraw
    user_message: str


def build_cells(
    data: d2.SplitDataset,
    *,
    n_requirements: int,
    n_draws: int,
    seed: int,
    selection: str = "random",
    level: str = "L2",
    split: str = "dev",
) -> List[Cell]:
    """Render every prompt the run will send, before any call is made.

    The leakage guard runs here, on the exact string that goes out — a guard
    that only runs under ``unittest`` guards the tests.

    ``split`` selects the **targets**. The history **pool is always dev**: those
    are the jobs the agent has completed, and drawing a held-out requirement's
    history from other held-out requirements would let the test set inform
    itself. So on ``split="heldout"`` the 10 test requirements are scored using
    histories built from the 20 dev jobs — which is also what a deployed agent
    would have.
    """
    pool = data.dev.requirements
    targets = (data.heldout if split == "heldout" else data.dev).requirements
    levels = {
        t: {"cost": data.dev.tier_cost_level(t), "quality": data.dev.tier_quality_level(t)}
        for t in ds.EFFORT_TIERS
    }
    profile = data.dev.history_profile()

    cells: List[Cell] = []
    for target in targets[:n_requirements]:
        for draw in H.draw_histories(
            target, pool, level=level, k=H.DEFAULT_K_L2, n_draws=n_draws,
            seed=seed, selection=selection,
        ):
            payload = target.visible.as_payload()
            d2.assert_payload_allowed(payload)
            message = ph.render_user_message(
                payload, levels, profile,
                H.render(draw, levels, level=level), level=level,
            )
            d2.assert_no_v2_leakage(message, data.raw_record(target))
            cells.append(Cell(target=target, draw=draw, user_message=message))
    return cells


def build_requests(
    cells: Sequence[Cell], model: llm_client.PinnedModel, repeats: int,
    selection: str = "random", level: str = "L2",
) -> List[Dict[str, Any]]:
    """One request per (cell, repeat).

    The draw index rides in the ``arm`` string so ``llm_client``'s existing cache
    key separates draws without the client needing to know what a draw is. The
    hybrid fingerprint rides in ``cache_salt``, so editing the prompt misses the
    cache and re-spends — which is the whole point of caching a probe.
    """
    salt = ph.protocol_fingerprint(level)
    schema = ph.L2_RESPONSE_SCHEMA if level == "L2" else ph.L3_RESPONSE_SCHEMA
    parser = parse_hybrid if level == "L2" else parse_hybrid_l3
    return [
        {
            "model": model,
            "arm": (f"{level}_X_k_{selection}" if selection != "random"
                    else f"{level}_X_k_d{cell.draw.draw_index}"),
            # probe_id already namespaces held-out requirements, so no salt needed
            "probe_id": cell.target.probe_id,
            "repeat": repeat,
            "user_message": cell.user_message,
            "system_message": ph.SYSTEM_MESSAGES[level],
            "schema": schema,
            "cache_salt": salt,
            "parser": parser,
        }
        for cell in cells
        for repeat in range(1, repeats + 1)
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class DrawScore:
    """σ and its decomposition for one history draw."""

    draw_index: int
    n: int
    sigma: float
    bias: float
    slope: float
    rho: float
    predicted_spread: float
    realized_spread: float


def _collapse(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], Tuple[float, float]]:
    """Mean predicted cost per (draw, requirement), repeats averaged."""
    grouped: Dict[Tuple[int, str], List[Tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault((row["draw"], row["requirement_id"]), []).append(
            (row["predicted_cost_usd"], row["realized_cost_usd"])
        )
    return {
        key: (statistics.mean([p for p, _ in v]), v[0][1]) for key, v in grouped.items()
    }


def score_draws(rows: Sequence[Dict[str, Any]]) -> List[DrawScore]:
    """Per-draw σ. **Not** averaged across draws first.

    A deployed agent has *one* history, not three. Averaging predictions across
    draws before scoring would remove variance the agent actually eats and
    flatter the estimator. The headline is the mean of these σ with their
    spread; the draw-averaged figure is reported separately and labelled.
    """
    collapsed = _collapse(rows)
    scores: List[DrawScore] = []
    for draw in sorted({k[0] for k in collapsed}):
        pairs = [v for k, v in collapsed.items() if k[0] == draw]
        predicted = [p for p, _ in pairs]
        realized = [r for _, r in pairs]
        residuals = [math.log(p / r) for p, r in pairs if p > 0 and r > 0]
        offset = statistics.mean(residuals)
        fit = metrics.calibration_fit(
            [math.log(p) for p in predicted], [math.log(r) for r in realized]
        )
        scores.append(
            DrawScore(
                draw_index=draw,
                n=len(pairs),
                sigma=math.sqrt(sum((x - offset) ** 2 for x in residuals) / len(residuals)),
                bias=offset,
                slope=fit.slope,
                rho=metrics.spearman(predicted, realized),
                predicted_spread=statistics.pstdev([math.log(p) for p in predicted]),
                realized_spread=statistics.pstdev([math.log(r) for r in realized]),
            )
        )
    return scores


def pooled_sigma(rows: Sequence[Dict[str, Any]]) -> float:
    """σ of the draw-averaged prediction — the "if you could average" number."""
    collapsed = _collapse(rows)
    by_requirement: Dict[str, List[float]] = {}
    realized: Dict[str, float] = {}
    for (_, requirement_id), (predicted, actual) in collapsed.items():
        by_requirement.setdefault(requirement_id, []).append(predicted)
        realized[requirement_id] = actual
    residuals = [
        math.log(statistics.mean(v) / realized[r])
        for r, v in by_requirement.items()
        if realized[r] > 0
    ]
    offset = statistics.mean(residuals)
    return math.sqrt(sum((x - offset) ** 2 for x in residuals) / len(residuals))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(rows: Sequence[Dict[str, Any]], stage: int, spend: float) -> str:
    lines: List[str] = []
    add = lines.append
    scores = score_draws(rows)
    sigmas = [s.sigma for s in scores]

    add(f"S5 re-run — hybrid X_k, Level 2 (medium tier), stage {stage}")
    add(f"  protocol {ph.PROTOCOL_VERSION} / {ph.protocol_fingerprint('L2')}")
    add(f"  {len(rows)} rows, {len(scores)} draw(s) x {rows[0]['n_requirements']} requirements")
    add(f"  spend ${spend:.4f}")

    add("\n## The number\n")
    add(f"  sigma_cost (debiased, medium)   {statistics.mean(sigmas):.4f}"
        + (f"  +/- {statistics.pstdev(sigmas):.4f} across draws" if len(sigmas) > 1 else ""))
    if len(scores) > 1:
        add(f"  draw-averaged prediction        {pooled_sigma(rows):.4f}   "
            "(optimistic: a real agent has one history)")

    add("\n## Against the targets (all medium-tier, debiased, held-out)\n")
    measured = statistics.mean(sigmas)
    add(f"  {'reference':<40}{'sigma':>8}   verdict")
    for label, value in sorted(TARGETS.items(), key=lambda kv: -kv[1]):
        verdict = "BEATEN" if measured < value else "not beaten"
        add(f"  {label:<40}{value:>8.4f}   {verdict}")
    add(f"  {'>>> this run':<40}{measured:>8.4f}")

    add("\n## Attribution — which part of the design is working\n")
    add(f"  {'draw':>5}{'sigma':>9}{'bias':>9}{'slope':>8}{'rho':>7}"
        f"{'pred sd':>9}{'real sd':>9}")
    for s in scores:
        add(f"  {s.draw_index:>5}{s.sigma:>9.4f}{s.bias:>+9.4f}{s.slope:>8.2f}"
            f"{s.rho:>7.2f}{s.predicted_spread:>9.4f}{s.realized_spread:>9.4f}")
    add("")
    add("  bias  != 0     -> the stated cost level is not being used")
    add("  slope >> 1     -> ranks but compresses; has not learned the range")
    add("  pred sd << real sd -> not discriminating between requirements")
    add("  rho   ~ 0      -> cannot read the task at all")

    failures = [r for r in rows if r["parse_status"] not in ("ok", "repaired")]
    if failures:
        add(f"\n  ** {len(failures)} of {len(rows)} calls failed to parse "
            "(each scores as arm H by the pre-specified rule)")

    add("\n## Reasoning — did the mechanics block land?\n")
    seen = set()
    for row in rows:
        if row["requirement_id"] in seen or not row["reasoning"]:
            continue
        seen.add(row["requirement_id"])
        add(f"  [{row['cost_multiplier']:.2f}x] {row['reasoning'][:150]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(ds.REPO_ROOT) / "agents" / ".env", override=False)
    except ImportError:
        pass
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S5 re-run: hybrid X_k probe, L2")
    parser.add_argument("--stage", type=int, default=0, choices=sorted(STAGES))
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-spend", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--level", default="L2", choices=("L2", "L3"))
    parser.add_argument("--split", default="dev", choices=("dev", "heldout"))
    parser.add_argument("--dry-run", action="store_true", help="render prompts, issue nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plan = STAGES[args.stage]
    data = d2.load_split_dataset()
    model = llm_client.model_by_key(args.model, llm_client.ALL_MODELS)

    cells = build_cells(
        data, n_requirements=plan["requirements"], n_draws=plan["draws"],
        seed=args.seed, selection=plan["selection"], level=args.level,
        split=args.split,
    )
    requests = build_requests(
        cells, model, plan['repeats'], plan['selection'], args.level
    )
    chars = statistics.mean(len(c.user_message) for c in cells)
    print(f"stage {args.stage}: {plan['requirements']} requirements x "
          f"{plan['draws']} draw(s) x {plan['repeats']} repeat(s) = {len(requests)} calls"
          f"  [selection={plan['selection']}]")
    print(f"model {model.key} | mean prompt {chars:.0f} chars (~{chars/3.7:.0f} tokens)")
    print(f"estimated spend ~${len(requests) * (chars/3.7/1e6*0.75 + 150/1e6*4.5):.3f}")

    if args.dry_run:
        print("\n--dry-run: nothing issued.")
        return 0

    probe = llm_client.OpenRouterProbe(
        _api_key(), concurrency=args.concurrency,
        max_spend_usd=0.0 if args.report_only else args.max_spend,
    )
    try:
        results = asyncio.run(probe.run_batch(requests))
    except RuntimeError as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 2
    print(f"calls: {probe.calls_made} issued, {probe.calls_cached} cached, "
          f"spend ${probe.spend_usd:.4f}")

    by_probe = {c.target.probe_id: c for c in cells}
    levels_cost = data.dev.tier_cost_level(TIER)
    rows: List[Dict[str, Any]] = []
    for result in results:
        cell = by_probe[result.probe_id]
        draw_index = (
            0 if "span" in result.arm else int(result.arm.rsplit("_d", 1)[1])
        )
        multiplier = result.estimate.cost_multiplier.get(TIER, 1.0)
        rows.append({
            "protocol_version": ph.PROTOCOL_VERSION,
            "fingerprint": ph.protocol_fingerprint("L2"),
            "stage": args.stage, "arm": f"X_k_{plan['selection']}", "model": result.model_key,
            "split": cell.target.split, "probe_id": result.probe_id,
            "requirement_id": cell.target.requirement_id, "tier": TIER,
            "draw": draw_index, "repeat": result.repeat,
            "history_ids": ";".join(cell.draw.requirement_ids()),
            "history_n": cell.draw.n,
            "cost_multiplier": round(multiplier, 6),
            "predicted_cost_usd": round(levels_cost * multiplier, 8),
            "realized_cost_usd": round(cell.target.realized_cost(TIER), 8),
            "parse_status": result.estimate.status,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "call_cost_usd": round(result.cost_usd, 8),
            "cached": int(result.cached),
            "reasoning": result.estimate.reasoning.replace("\n", " ")[:400],
            "n_requirements": plan["requirements"],
        })

    print("\n" + format_report(rows, args.stage, probe.spend_usd))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
