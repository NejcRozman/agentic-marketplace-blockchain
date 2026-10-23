#!/usr/bin/env python3
"""The in-market estimator error, under one fixed scoring convention (S8 §C.2).

    python -m analysis.in_market_sigma --cells E_xd_vs_h_v3

**The convention, decided 2026-09-07.** σ is the debiased RMS of
``log(predicted / truth)``, and *truth is the requirement's mean over its three
recorded executions* — convention **A**. The alternative, scoring against the one
recording the market happened to bill for, is reported as a secondary.

Why A is primary, in order of weight:

1. **B charges the estimator for a coin flip it cannot see.** Each requirement
   was executed three times and the same job does not cost the same twice
   (σ = 0.0958 across dataset v3). The market draws one recording at random
   *after* the estimate is made. Measured on block E, A gives 0.4058 and B gives
   0.4164, and ``sqrt(0.4058² + 0.0958²) = 0.4170`` — the entire gap between the
   two conventions is that executor noise, added in quadrature.
2. **B can only be measured on auctions the arm won**, because a provider is
   billed only when it wins: 686 estimates against A's 1,200. Those 686 are not
   a random subset — they are the jobs the arm bid hardest on, which are
   systematically cheaper than the population (block E: won jobs cost
   $0.036–0.048 against a $0.054 population mean). B therefore measures estimator
   quality on a selected sample.
3. **Only A is comparable to S5's 0.3404**, which was measured against the
   3-run mean. §F.2 asks for exactly that comparison, and
   :mod:`analysis.reference_sigma` exists because this project has
   already been burned by comparing σs measured different ways.

Nothing economic is lost: σ is a secondary DV, and the primary — profit — already
uses the amount actually charged, so the draw is fully accounted for in the money.

This module exists so the convention is *executable* rather than a sentence in a
spec that the next script can quietly contradict.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from analysis.reference_sigma import debiased_sigma

from agents.common import paths  # noqa: E402

REPO_ROOT = paths.REPO_ROOT
LOGS = paths.RUN_LOGS_DIR
DATASET_V3 = paths.DATASET_FINAL_DIR / "runs.jsonl"

_LEVEL = re.compile(r"Effort estimates this loop: costs=(\{[^}]*\})")
_ESTIMATE = re.compile(r"(?:decomposed|hybrid): requirement (\S+) tier=(\w+) multiplier=([0-9.]+) (\{.*\})")
_REPLAY = re.compile(r"replayed for job (\d+): tier=(\w+), repeat=(\d+), cost=\$([0-9.]+)")
_AUCTION = re.compile(r"(?:auction (\d+)|Skipping auction (\d+))")


def requirement_means(dataset: Optional[Path] = None) -> Dict[str, float]:
    """Truth for convention A: each requirement's mean over its recorded runs."""
    by_requirement: Dict[str, List[float]] = collections.defaultdict(list)
    for line in (dataset or DATASET_V3).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cost = row.get("execution_cost_usd")
        if row.get("requirement_id") and cost:
            by_requirement[row["requirement_id"]].append(float(cost))
    return {k: statistics.mean(v) for k, v in by_requirement.items()}


def execution_noise(dataset: Optional[Path] = None) -> float:
    """σ of one recorded execution about its own requirement's mean.

    The floor no estimator can predict, and exactly the gap between the two
    conventions.
    """
    by_requirement: Dict[str, List[float]] = collections.defaultdict(list)
    for line in (dataset or DATASET_V3).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cost = row.get("execution_cost_usd")
        if row.get("requirement_id") and cost:
            by_requirement[row["requirement_id"]].append(float(cost))
    residuals = []
    for costs in by_requirement.values():
        if len(costs) < 2:
            continue
        mean = statistics.mean(costs)
        residuals += [math.log(c / mean) for c in costs if c > 0 and mean > 0]
    return debiased_sigma(residuals)


def estimates_from_run(run_dir: Path) -> List[dict]:
    """Every estimate the LLM arm made in one run, with what it was later billed.

    ``charged`` is None for an auction the arm did not win — which is precisely
    why convention B cannot be measured on every estimate.
    """
    out: List[dict] = []
    for log in sorted(run_dir.glob("provider_agent_*.log")):
        status = run_dir / log.name.replace("provider_agent_", "provider_").replace(".log", "_status.json")
        if not status.exists():
            continue
        group = (json.loads(status.read_text()).get("profile") or {}).get("group_name") or ""
        if not group.startswith("X_"):
            continue

        lines = log.read_text(errors="ignore").splitlines()
        level: Optional[float] = None
        estimates: Dict[int, Tuple[str, float, bool]] = {}
        billed: Dict[int, float] = {}
        for i, line in enumerate(lines):
            match = _LEVEL.search(line)
            if match:
                try:
                    level = json.loads(match.group(1)).get("medium")
                except (ValueError, AttributeError):
                    pass
                continue
            match = _REPLAY.search(line)
            if match:
                billed[int(match.group(1))] = float(match.group(4))
                continue
            match = _ESTIMATE.search(line)
            if match and level:
                auction_id = None
                for follow in lines[i + 1:i + 4]:
                    found = _AUCTION.search(follow)
                    if found:
                        auction_id = int(found.group(1) or found.group(2))
                        break
                if auction_id is not None:
                    degraded = "cold_start" in match.group(4) or "degraded" in match.group(4)
                    estimates[auction_id] = (match.group(1), level * float(match.group(3)), degraded)
        for auction_id, (requirement, predicted, degraded) in estimates.items():
            out.append({
                "auction_id": auction_id, "requirement_id": requirement,
                "predicted": predicted, "degraded": degraded,
                "charged": billed.get(auction_id),
            })
    return out


def score(estimates: Iterable[dict], truth: Dict[str, float]) -> Dict[str, float]:
    """σ and multiplicative bias under convention A (truth = requirement mean)."""
    residuals = [
        math.log(e["predicted"] / truth[e["requirement_id"]])
        for e in estimates
        if e["requirement_id"] in truth and e["predicted"] > 0 and truth[e["requirement_id"]] > 0
    ]
    if not residuals:
        return {"n": 0, "sigma": float("nan"), "bias": float("nan")}
    return {"n": len(residuals), "sigma": debiased_sigma(residuals),
            "bias": math.exp(statistics.mean(residuals))}


def score_against_charge(estimates: Iterable[dict]) -> Dict[str, float]:
    """σ under convention B (truth = the single recording actually billed).

    Secondary. Only defined on auctions the arm won.
    """
    residuals = [
        math.log(e["predicted"] / e["charged"])
        for e in estimates
        if e.get("charged") and e["predicted"] > 0
    ]
    if not residuals:
        return {"n": 0, "sigma": float("nan"), "bias": float("nan")}
    return {"n": len(residuals), "sigma": debiased_sigma(residuals),
            "bias": math.exp(statistics.mean(residuals))}


def main() -> None:
    from analysis.pooled_model import load_ledger_runs

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", nargs="+", required=True)
    args = parser.parse_args()

    truth = requirement_means()
    noise = execution_noise()
    print(f"executor noise (one recording about its job's mean): σ = {noise:.4f}\n")
    print(f"{'cell':<24}{'n(A)':>7}{'σ A':>9}{'bias A':>9}{'n(B)':>7}{'σ B':>9}{'√(A²+noise²)':>15}")

    for cell in args.cells:
        estimates: List[dict] = []
        for run in load_ledger_runs([cell]):
            estimates += estimates_from_run(LOGS / f"s9_{cell}" / str(run["run_id"]))
        a = score(estimates, truth)
        b = score_against_charge(estimates)
        predicted_b = math.sqrt(a["sigma"] ** 2 + noise ** 2)
        print(f"{cell:<24}{a['n']:>7}{a['sigma']:>9.4f}{a['bias']:>9.3f}"
              f"{b['n']:>7}{b['sigma']:>9.4f}{predicted_b:>15.4f}")

    print("\n  A (primary)   truth = the requirement's mean over its 3 recorded executions")
    print("  B (secondary) truth = the one recording the market billed for; won auctions only")
    print("  The gap between them is executor noise: B ≈ √(A² + noise²).")
    print("  S5's held-out reference, 0.3404, was measured under A.")


if __name__ == "__main__":
    main()
