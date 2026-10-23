"""Cost estimators: where a provider's cost estimate for the job in an auction comes from.

Every estimator returns a multiplier on the agent's running-average cost for the
tier, so ``believed cost = level x multiplier``, and 1.0 reproduces Adaptive
exactly. ``none`` (Adaptive) constructs no estimator.

``r_guarded``  Regression: a ridge regression of log cost on countable features
               of the request, fitted on the agent's own completed jobs; it
               returns 1.0 until it has enough observations.
``decomposed`` LLM-Estimate: an LLM predicts the retrieval depth of each prompt,
               and a cost model fitted on the agent's own completed jobs turns
               the depths into dollars.
``oracle``     Oracle: the request's true mean cost from the dataset. A bound,
               not a competitor.

Every fit sees only jobs this agent has already completed, never the request it
is currently pricing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Config values that select an estimator.
ESTIMATOR_MODES = ("none", "r_guarded", "decomposed", "oracle")


@dataclass(frozen=True)
class JobOutcome:
    """One completed job, as the agent remembers it."""

    requirement_id: str
    effort_tier: str
    realized_cost_usd: float


def surface_features(requirement: Mapping) -> Optional[Dict[str, float]]:
    """The three countable features of a request: total prompt length, PDF count, prompt count.

    None when the record cannot supply them; the estimator then bids its history.
    """
    if not requirement:
        return None

    inner = requirement.get("requirements") or {}
    prompts = inner.get("prompts") or requirement.get("prompts") or []
    if not prompts:
        return None

    pdfs = requirement.get("source_pdfs") or inner.get("source_pdfs") or []
    pdf_count = requirement.get("source_pdf_count")
    if pdf_count is None:
        pdf_count = len(pdfs)

    return {
        "total_prompt_chars": float(sum(len(str(p)) for p in prompts)),
        "source_pdf_count": float(pdf_count),
        "prompt_count": float(len(prompts)),
    }


@dataclass
class OracleCostEstimator:
    """Oracle: a provider that already knows what each job will cost. A bound, not a competitor.

    It reads the request's true mean execution cost from the dataset and bids off
    that: the mean over its three recorded executions, not the single recording
    the market will draw and bill, which is drawn after every bid is committed.
    No LLM call; one dictionary lookup.
    """

    mode: str = "oracle"
    #: requirement_id -> the mean cost across its recorded executions.
    truth: Dict[str, float] = field(default_factory=dict)
    outcomes: List[JobOutcome] = field(default_factory=list)
    degraded: Dict[str, int] = field(default_factory=dict)
    #: Unused (the oracle needs no features), but the bidding path reads it on
    #: every estimator.
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Where the truth is read from: the run's own dataset, set by the handler.
    dataset_dir: Optional[str] = None

    def _degrade(self, reason: str) -> None:
        self.degraded[reason] = self.degraded.get(reason, 0) + 1

    def remember_features(self, requirement_id: str, requirement: Mapping) -> None:
        """No-op: the oracle needs no features."""

    def record_outcome(self, requirement_id: str, effort_tier: str, realized_cost_usd: float) -> None:
        """Recorded for reporting only; the oracle never learns."""
        self.outcomes.append(JobOutcome(str(requirement_id), str(effort_tier), float(realized_cost_usd)))

    def _load_truth(self) -> None:
        import json
        from pathlib import Path

        if self.truth or not self.dataset_dir:
            return
        runs = Path(self.dataset_dir) / "runs.jsonl"
        if not runs.exists():
            return
        totals: Dict[str, List[float]] = {}
        for line in runs.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cost = row.get("execution_cost_usd")
            requirement = row.get("requirement_id")
            if requirement and cost:
                totals.setdefault(str(requirement), []).append(float(cost))
        self.truth = {k: sum(v) / len(v) for k, v in totals.items() if v}

    def multiplier_for(
        self,
        requirement_id: str,
        tier: str,
        tier_level_usd: float,
        feature_names: Sequence[str] = (),
        deadline_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, object]]:
        """``true_mean / level``, so ``level x multiplier`` is the true mean.

        Returns 1.0 (the history, as Adaptive bids) if the request is not in the dataset,
        counted in ``degraded``.
        """
        self._load_truth()
        if tier_level_usd <= 0:
            self._degrade("no_level")
            return 1.0, {"reason": "no_level"}
        true_cost = self.truth.get(str(requirement_id))
        if not true_cost:
            self._degrade("requirement_not_in_dataset")
            return 1.0, {"reason": "requirement_not_in_dataset"}
        return true_cost / tier_level_usd, {
            "reason": "oracle",
            "true_cost_usd": true_cost,
            "n_truth": len(self.truth),
        }


@dataclass
class RegressionCostEstimator:
    """Regression, refitted on every bid from the jobs the agent has completed so far."""

    mode: str = "r_guarded"
    #: requirement_id -> features, for requirements this agent has seen.
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Completed jobs, oldest first.
    outcomes: List[JobOutcome] = field(default_factory=list)
    #: Why the estimator fell back to the heuristic, by reason.
    degraded: Dict[str, int] = field(default_factory=dict)

    def _degrade(self, reason: str) -> None:
        self.degraded[reason] = self.degraded.get(reason, 0) + 1

    @property
    def guarded(self) -> bool:
        return self.mode == "r_guarded"

    def remember_features(self, requirement_id: str, requirement: Mapping) -> None:
        computed = surface_features(requirement)
        if computed is not None:
            self.features[str(requirement_id)] = computed

    def record_outcome(self, requirement_id: str, effort_tier: str, realized_cost_usd: float) -> None:
        self.outcomes.append(JobOutcome(str(requirement_id), str(effort_tier), float(realized_cost_usd)))

    def multiplier_for(
        self,
        requirement_id: str,
        tier: str,
        tier_level_usd: float,
        feature_names: Sequence[str] = (),
        deadline_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, object]]:
        """Return ``(multiplier, diagnostics)`` for one tier of one auction.

        ``deadline_seconds`` is part of the shared estimator interface and is ignored:
        the regression makes no call that could time out. A multiplier of exactly 1.0
        means "no adjustment" (the agent bids its history, as Adaptive does), which is
        the behaviour while the guard withholds the fit and when no feature is usable.
        """
        from agents.provider.estimation import regression
        from agents.provider.estimation.requirement import SURFACE_FEATURES

        names = tuple(feature_names) or SURFACE_FEATURES
        target = self.features.get(str(requirement_id))
        if target is None or tier_level_usd <= 0:
            self._degrade("no_features")
            return 1.0, {"reason": "no_features"}

        # Strictly the agent's own completed jobs at this tier, and never the
        # request being priced right now.
        observed = [
            (self.features[outcome.requirement_id], outcome.realized_cost_usd)
            for outcome in self.outcomes
            if outcome.effort_tier == tier
            and outcome.requirement_id != str(requirement_id)
            and outcome.requirement_id in self.features
            and outcome.realized_cost_usd > 0
        ]

        estimate = regression.fit_cost_multiplier(
            observed, dict(target), float(tier_level_usd),
            feature_names=names, guard=self.guarded,
        )
        return estimate.multiplier, {
            "n_observations": estimate.n_observations,
            "guard_withheld": estimate.guard_withheld,
            "degenerate": estimate.degenerate,
            "usable_features": estimate.usable_features,
        }


async def _within_deadline(awaitable, deadline_seconds: Optional[float]):
    """Await ``awaitable``, cancelling it after ``deadline_seconds`` (None: no limit).

    An estimate that arrives after the commit window has closed cannot be bid on.
    """
    if deadline_seconds is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=max(0.1, float(deadline_seconds)))


def requirement_payload(requirement: Mapping) -> Dict:
    """The fields of a request that the LLM-Estimate prompt renders. No identifiers.

    Source PDFs appear as bare file stems (their names carry topic information a
    provider would see); their directory is excluded.
    """
    from agents.provider.estimation import requirement as estimation_requirement

    # ``probe_id`` is excluded from ``as_payload()``, so any value serves here.
    return estimation_requirement.visible_view(requirement, probe_id="live").as_payload()


#: Completed jobs needed before LLM-Estimate prices with its own fit.
DECOMPOSED_MIN_CALIBRATION = 1

#: The tier whose retrieval budget the prompt describes. Other tiers return 1.0.
DECOMPOSED_TIER = "medium"

#: Seconds of the commit window reserved for everything after the estimator
#: returns: the freshness re-check, pricing, signing and committing the bid.
ESTIMATOR_DEADLINE_RESERVE_SECONDS = 3.0


@dataclass
class DecomposedCostEstimator:
    """LLM-Estimate: the LLM predicts retrieval depth; the arithmetic happens here.

    The model is asked how many of the medium tier's three retrieval rounds each
    prompt will consume. Dollars are recovered with
    ``cost = a * n_prompts + b * sum(rounds)``, with ``a`` and ``b`` fitted on this
    agent's own completed jobs: one observation per job it won and completed,
    recorded when it bid and closed when it learned what it paid.

    Every failure (timeout, transport error, unparseable or misaligned answer)
    returns 1.0, the pure heuristic, and is counted in ``degraded``. The call
    happens inside the commit window, on the agent's own event loop.
    """

    mode: str = "decomposed"
    model_key: str = "openai/gpt-4o-mini"
    run_seed: Optional[int] = None
    agent_id: int = 0
    api_key: str = ""
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outcomes: List[JobOutcome] = field(default_factory=list)
    #: requirement_id -> the payload the prompt renders.
    payloads: Dict[str, Dict] = field(default_factory=dict)
    #: The agent's own per-tier levels, refreshed each auction by the caller.
    levels: Dict[str, Dict[str, float]] = field(default_factory=dict)

    calls_made: int = 0
    calls_failed: int = 0
    calls_cached: int = 0
    reasoning_cost_usd: float = 0.0
    #: Why the estimator fell back to the heuristic, by reason.
    degraded: Dict[str, int] = field(default_factory=dict)

    #: (requirement_id, tier) -> (n_prompts, sum_rounds) for auctions bid on but
    #: not yet settled. An entry becomes a calibration observation only if the
    #: agent wins and completes the job.
    pending_rounds: Dict[Tuple[str, str], Tuple[int, float]] = field(default_factory=dict)

    #: (n_prompts, sum_rounds, realized_cost_usd), strictly causal.
    calibration: List[Tuple[int, float, float]] = field(default_factory=list)

    def _degrade(self, reason: str) -> None:
        self.degraded[reason] = self.degraded.get(reason, 0) + 1

    def remember_features(self, requirement_id: str, requirement: Mapping) -> None:
        computed = surface_features(requirement)
        if computed is not None:
            self.features[str(requirement_id)] = computed
        self.payloads[str(requirement_id)] = requirement_payload(requirement)

    def record_outcome(
        self, requirement_id: str, effort_tier: str, realized_cost_usd: float
    ) -> None:
        """Pair a completed job's predicted depths with what it actually cost."""
        self.outcomes.append(
            JobOutcome(str(requirement_id), str(effort_tier), float(realized_cost_usd))
        )
        key = (str(requirement_id), str(effort_tier))
        predicted = self.pending_rounds.pop(key, None)
        if predicted is not None and realized_cost_usd > 0:
            n_prompts, sum_rounds = predicted
            self.calibration.append((n_prompts, sum_rounds, float(realized_cost_usd)))

    def n_prompts_for(self, requirement_id: str) -> int:
        payload = self.payloads.get(str(requirement_id)) or {}
        return len(payload.get("prompts") or [])

    def build_prompt(self, requirement_id: str, tier: str) -> Optional[str]:
        """Render the prompt for this auction, or None if it cannot be built."""
        from agents.provider.estimation import prompt as estimation_prompt

        payload = self.payloads.get(str(requirement_id))
        if payload is None or not self.levels or not (payload.get("prompts") or []):
            return None
        return estimation_prompt.render_user_message(payload, self.levels)

    async def multiplier_for(
        self, requirement_id: str, tier: str, tier_level_usd: float,
        deadline_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, object]]:
        """One estimator call; returns 1.0 (the heuristic) on any failure.

        ``deadline_seconds`` bounds the call to what is left of the commit window (see
        :data:`ESTIMATOR_DEADLINE_RESERVE_SECONDS`).
        """
        from agents.provider.estimation import llm_client
        from agents.provider.estimation import prompt as estimation_prompt

        if tier != DECOMPOSED_TIER:
            # Not counted as a degradation: the handler asks every tier on every
            # auction, and the agent prices off medium alone.
            return 1.0, {"reason": "tier_not_supported", "tier": tier}
        if tier_level_usd <= 0:
            self._degrade("no_level")
            return 1.0, {"reason": "no_level"}

        n_prompts = self.n_prompts_for(requirement_id)
        prompt = self.build_prompt(requirement_id, tier)
        if prompt is None or n_prompts <= 0:
            self._degrade("no_payload")
            return 1.0, {"reason": "no_payload"}

        # The call is not gated on calibration: the depths are always elicited and
        # recorded, because calibration only grows from jobs whose depths were
        # predicted. The gate controls only whether the estimate is bid on; below
        # it the estimator returns 1.0 (the heuristic) while observations accrue.
        fit = (
            estimation_prompt.calibrate(self.calibration, two_param=True)
            if len(self.calibration) >= DECOMPOSED_MIN_CALIBRATION else None
        )

        probe = llm_client.OpenRouterProbe(self.api_key, cache_dir=llm_client.LIVE_CACHE_DIR)
        call = probe.call_live(
            llm_client.model_by_key(self.model_key),
            user_message=prompt,
            system_message=estimation_prompt.SYSTEM_MESSAGE,
            schema=estimation_prompt.response_schema(n_prompts),
            parser=estimation_prompt.parse_response,
            run_seed=self.run_seed,
            agent_id=self.agent_id,
        )
        try:
            result = await _within_deadline(call, deadline_seconds)
        except asyncio.TimeoutError:
            self.calls_failed += 1
            logger.warning(
                "🤖 decomposed estimator exceeded its %.1fs window; bidding history",
                deadline_seconds,
            )
            self._degrade("deadline_exceeded")
            return 1.0, {"reason": "deadline_exceeded", "deadline_s": deadline_seconds}
        except Exception as e:  # transport, parse or other error
            self.calls_failed += 1
            logger.warning("🤖 decomposed call failed (%s); bidding history", e)
            self._degrade("call_failed")
            return 1.0, {"reason": "call_failed", "error": str(e)}

        self.calls_made += 1
        self.reasoning_cost_usd += float(result.cost_usd)
        if result.cached:
            self.calls_cached += 1

        diagnostics: Dict[str, object] = {
            "n_calibration": len(self.calibration),
            "a": round(fit[0], 6) if fit else None,
            "b": round(fit[1], 6) if fit else None,
            "n_prompts": n_prompts,
            "cost_usd": result.cost_usd,
            "cached": result.cached,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

        if result.estimate.failed:
            self.calls_failed += 1
            self._degrade("parse_failed")
            return 1.0, {**diagnostics, "reason": "parse_failed",
                         "status": result.estimate.status}

        rounds = estimation_prompt.clean_rounds(
            getattr(result.estimate, "prompt_rounds", []) or [], n_prompts
        )
        if rounds is None:
            # Wrong length or out of range: the alignment between depths and prompts
            # is the protocol, so a mismatch is not repaired by guessing.
            self.calls_failed += 1
            self._degrade("bad_rounds")
            return 1.0, {**diagnostics, "reason": "bad_rounds"}

        sum_rounds = float(sum(rounds))
        # Recorded before the bidding decision; it pairs with the realised cost if the
        # agent wins and completes the job.
        self.pending_rounds[(str(requirement_id), str(tier))] = (n_prompts, sum_rounds)
        diagnostics["sum_rounds"] = round(sum_rounds, 2)
        # The per-prompt vector, not just its sum: the dataset records how many
        # rounds each prompt actually took, so the depths can be checked prompt by prompt.
        diagnostics["prompt_rounds"] = [round(float(r), 2) for r in rounds]

        if fit is None:
            # Depths in hand but no fit yet: bid the heuristic.
            self._degrade("cold_start")
            return 1.0, {**diagnostics, "reason": "cold_start"}

        predicted_cost = estimation_prompt.predict_cost(n_prompts, sum_rounds, fit)
        multiplier = predicted_cost / tier_level_usd if tier_level_usd > 0 else 1.0
        # No band on the output multiplier: the depths are already validated and the
        # fit is non-negative, and a band would reject correct estimates for the
        # cheapest jobs (a fifth of the agent's average). Only arithmetic sanity
        # remains: a prediction must be a positive finite number.
        if (multiplier != multiplier or multiplier in (float("inf"), float("-inf"))
                or multiplier <= 0.0):
            self.calls_failed += 1
            self._degrade("not_a_positive_number")
            return 1.0, {**diagnostics, "reason": "not_a_positive_number", "raw": multiplier}
        diagnostics["predicted_cost_usd"] = round(predicted_cost, 6)

        return multiplier, diagnostics
