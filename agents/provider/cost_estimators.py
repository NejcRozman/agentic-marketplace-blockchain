"""S7 §B.2 — arm R on the live agent path: a regression on its own history.

H bids off its per-tier EMA and nothing else. R asks the honest question behind
the whole paper: **can an agent do this with ordinary statistics instead of an
LLM call?** It fits a regression on the jobs it has already completed and uses it
to adjust its cost belief for the job in front of it — no model, no API, $0.

Two arms, differing *only* in whether the fit is guarded (§B.2):

``r_naive``
    Always fits, however little history it has. With 3 features plus an
    intercept and n ~ 4 wins the system is exactly determined and extrapolates
    wildly — S6 measured sigma 0.9327 at 3 jobs, *worse than guessing the mean*.
    It runs because it reproduces S6's published curve.

``r_guarded``
    Ridge-penalised, and refuses to fit below 2 observations per parameter,
    returning the tier level instead. It therefore **is** the pure heuristic at
    cold start, by construction. This is the opponent that matters: beating it
    means beating the best thing a careful engineer could deploy at n = 0,
    rather than beating a regression nobody would ship.

**Everything here is strictly causal.** The fit sees only jobs this agent has
already completed, in the order it completed them, and never the requirement it
is currently pricing. Nothing is fitted on the run as a whole and nothing
consults the dataset's dev/held-out labels — those are an offline probe
construct and mean nothing to an agent inside a market.

The fit itself is delegated to ``agents.provider.estimation.history`` — the same
function the S5 probe and the S6 simulator call — so "the regression an agent
could build" cannot come to mean two different things in the three places the
project compares it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Config values that select an arm. `none` leaves H untouched.
ESTIMATOR_MODES = ("none", "r_naive", "r_guarded", "hybrid", "decomposed", "r_rounds", "oracle")


@dataclass(frozen=True)
class JobOutcome:
    """One completed job, as the agent remembers it."""

    requirement_id: str
    effort_tier: str
    realized_cost_usd: float


def surface_features(requirement: Mapping) -> Optional[Dict[str, float]]:
    """The three countable features, from a dataset requirement record.

    Deliberately mechanical — no reading, no model. ``total_prompt_chars`` is
    the summed length of the prompts, and the other two are counts. An agent in
    a deployed system would compute exactly these from the service description
    it was sent.

    Returns ``None`` when the record cannot supply them, which makes the arm
    fall through to its plain history beliefs: an agent cannot regress on
    features it cannot compute.
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
    """A provider that already knows what each job will cost. **A bound, not a competitor.**

    It reads the requirement's true mean execution cost straight out of the
    dataset and bids off that. Nothing about it is realisable — it is the
    denominator for `research_plan.md` §S8's *fraction of oracle value
    recovered*, `(X - H) / (Oracle - H)`, which has no other way of being
    measured: no S9 cell could otherwise supply the `Oracle` term.

    **What it knows, precisely.** The requirement's mean over its three recorded
    executions — not the single recording the market will draw and bill it for.
    That draw is a coin flip made after every bid is committed (σ = 0.0958 across
    dataset v3), so an arm that knew it would be a different and impossible
    object: a bidder with knowledge of the future rather than of the task. The
    3-run mean is exactly the truth convention S8 §C.2 fixed for σ, so this arm
    scores σ = 0 under it, and the residual 0.0958 under the secondary
    convention. It is the same quantity S6's oracle policy was fed.

    **It reads the dataset, so it must never be in a confirmatory contrast.**
    Every other arm sees only the requirement text and its own completed jobs.
    This one is given the answer. It belongs in block A as a reference line and
    nowhere else, and results involving it are labelled a bound.

    Free in money and nearly free in time: no LLM call, one dictionary lookup.
    Its only cost to the campaign is the wall clock of the runs themselves.
    """

    mode: str = "oracle"
    #: requirement_id -> the mean cost across its recorded executions.
    truth: Dict[str, float] = field(default_factory=dict)
    outcomes: List[JobOutcome] = field(default_factory=list)
    degraded: Dict[str, int] = field(default_factory=dict)
    #: Unused by this arm — it is handed the answer and needs no features. It
    #: exists because the bidding path gates on `requirement_id not in
    #: estimator.features` before calling `remember_features`, and every other
    #: estimator carries it. Omitting it raised `AttributeError` on the first
    #: auction of every run, which the handler caught as "Error in deterministic
    #: reasoning" — so the arm silently never bid, rather than crashing.
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Where the truth is read from. Set by the handler from the run's own
    #: sampled-dataset config, so an oracle can never be pointed at a dataset
    #: the market is not drawing from.
    dataset_dir: Optional[str] = None

    def _degrade(self, reason: str) -> None:
        self.degraded[reason] = self.degraded.get(reason, 0) + 1

    def remember_features(self, requirement_id: str, requirement: Mapping) -> None:
        """No-op. The oracle needs no features; it is handed the answer."""

    def record_outcome(self, requirement_id: str, effort_tier: str, realized_cost_usd: float) -> None:
        """Recorded for reporting parity only — the oracle never learns."""
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

        Returns 1.0 — bid the history, exactly as H does — if the requirement is
        absent from the dataset, rather than inventing a number. That should
        never happen when the arm is pointed at the dataset the market draws
        from, and it is counted in ``degraded`` if it does.
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
    """Arm R, refitted on every bid from whatever the agent has won so far."""

    mode: str = "r_naive"
    #: requirement_id -> features, for requirements this agent has seen.
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Completed jobs, oldest first.
    outcomes: List[JobOutcome] = field(default_factory=list)
    #: Why this arm fell back to the heuristic, by reason. Arm R degrades for
    #: its own reasons (no features, too little history), and an analysis that
    #: cannot tell those apart cannot say why a run behaved like H.
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

        ``deadline_seconds`` is accepted and ignored. The bidding path passes it
        to whatever estimator is configured, because arm X runs an LLM call
        inside the commit window and must be cut off before the window closes.
        Arm R makes no call and has nothing to time out — but the parameter is
        part of the shared interface, and omitting it here raised
        ``TypeError: unexpected keyword argument 'deadline_seconds'`` on the
        first bid of every R arm. Seventeen of the S7 matrix's thirty-two cells
        are R contrasts; none had been run between the deadline landing and this
        being caught.

        A multiplier of exactly 1.0 means "no adjustment" — the agent bids its
        history, which is what H does. That is the cold-start behaviour of
        ``r_guarded`` and the no-usable-feature behaviour of both arms.
        """
        from agents.provider.estimation import dataset as estimation_dataset
        from agents.provider.estimation import history as estimation_history

        names = tuple(feature_names) or estimation_dataset.SURFACE_FEATURES
        target = self.features.get(str(requirement_id))
        if target is None or tier_level_usd <= 0:
            self._degrade("no_features")
            return 1.0, {"reason": "no_features"}

        # Strictly the agent's own completed jobs at this tier, and never the
        # requirement being priced right now (§B.3's rule, which applies to R
        # for the same reason it applies to the hybrid).
        observed = [
            (self.features[outcome.requirement_id], outcome.realized_cost_usd)
            for outcome in self.outcomes
            if outcome.effort_tier == tier
            and outcome.requirement_id != str(requirement_id)
            and outcome.requirement_id in self.features
            and outcome.realized_cost_usd > 0
        ]

        estimate = estimation_history.fit_cost_multiplier(
            observed, dict(target), float(tier_level_usd),
            feature_names=names, guard=self.guarded,
        )
        return estimate.multiplier, {
            "n_observations": estimate.n_observations,
            "guard_withheld": estimate.guard_withheld,
            "degenerate": estimate.degenerate,
            "usable_features": estimate.usable_features,
        }


# ---------------------------------------------------------------------------
# S7 §B.3 — arm X, the hybrid
# ---------------------------------------------------------------------------

#: Most recent completed jobs shown to the estimator (§B.3, k = 3).
HYBRID_HISTORY_K = 3

#: Completed jobs required before the arm calls at all. **One, not k.**
#:
#: S5's recorded calls are *all* at `history_n = 3` — 0, 1 and 2 were never
#: measured — and S6 assumed a sigma rather than simulating an empty-history
#: call, so "the hybrid at cold start" has no data behind it either way. At k = 0
#: the model would have the requirement text and the agent's own averages but no
#: anchor connecting them: the regime most likely to produce a confident wrong
#: number.
#:
#: So the arm is H until it has completed one job, then estimates with whatever
#: it has — "is this cheaper or dearer than that one?" is answerable from a
#: single example — and the history grows to k with each win. That keeps the R
#: contrast sharp (X estimates at k = 1 while `R_guarded` still withholds)
#: without bidding off an untested prompt regime.
HYBRID_MIN_HISTORY = 1


async def _within_deadline(awaitable, deadline_seconds: Optional[float]):
    """Await `awaitable`, cancelling it if `deadline_seconds` elapses.

    ``None`` means unbounded, which is what the offline probes want. A live
    caller always passes a budget: an estimate that arrives after the commit
    window has closed cannot be bid on, so waiting for it only costs the agent
    the auctions it would otherwise have entered.
    """
    if deadline_seconds is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=max(0.1, float(deadline_seconds)))


def requirement_payload(requirement: Mapping) -> Dict:
    """The fields S5 renders into the prompt. No identifiers, by construction.

    **Delegates to S5's own projection** rather than rebuilding it. The two used
    to be parallel implementations and drifted: this one passed
    ``source_pdf_names`` as stored — ``utils/files/v2/Luu_Scaling.pdf`` — while
    S5 renders the stem, ``Luu_Scaling`` (`dataset._pdf_stem`, admitted because
    "the filenames carry topic information a real provider would see", with the
    directory deliberately excluded). Every one of dataset v2's 30 requirements
    differed, in the current job *and* in every history example, so the live arm
    X was never sent the prompt S5 measured.

    Delegating is the same rule this module already applies to the fit itself:
    the estimator S5 measured and the estimator the agent runs cannot be allowed
    to mean two different things. Pinned by
    ``test_s7_prompt_parity.py::TestPayloadParity``.
    """
    from agents.provider.estimation import dataset as estimation_dataset

    # ``probe_id`` is the probe harness's join key back to its own results and
    # is excluded from ``as_payload()`` by construction, so it never reaches the
    # model and any value serves here.
    return estimation_dataset.visible_view(requirement, probe_id="live").as_payload()


@dataclass
class HybridCostEstimator:
    """Arm X: one structured LLM call per auction, then the unchanged controller.

    Differs from :class:`RegressionCostEstimator` in exactly one place — where
    the multiplier comes from. Same seam, same strictly-causal history, same
    ``believed_cost = level x multiplier``.

    **The protocol is S5's, unmodified** (§J): prompt, schema, parser, repair
    ladder, transport and cache all come from ``agents.provider.estimation``, so the
    estimator running in the market is the one whose sigma S5 measured. The live
    L2 fingerprint still matches the recorded runs byte-for-byte.

    **Every failure degrades to H.** Parse failure, timeout, API error, or a
    multiplier outside ``[0.2, 5.0]`` all return 1.0, which reproduces the pure
    heuristic exactly, and is logged and counted.

    The estimating method is ``async`` because the call happens inside the
    commit window, on the agent's own event loop. Arm R stays synchronous: it
    has nothing to await.
    """

    mode: str = "hybrid"
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
    #: Why the arm fell back to the heuristic, by reason. `calls_failed` counts
    #: the same events in aggregate; the breakdown is what tells a later analysis
    #: whether a run bid H because the model was slow, unparseable, or absent —
    #: three very different findings that a single counter cannot separate.
    degraded: Dict[str, int] = field(default_factory=dict)

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
        self.outcomes.append(
            JobOutcome(str(requirement_id), str(effort_tier), float(realized_cost_usd))
        )

    def history_for(self, requirement_id: str, tier: str) -> List[JobOutcome]:
        """The last k completed jobs at this tier, excluding the one being priced.

        §B.3 requires that the test requirement never appear in its own history,
        "enforced in code, not convention" — this is that enforcement.
        """
        return [
            outcome for outcome in self.outcomes
            if outcome.effort_tier == tier
            and outcome.requirement_id != str(requirement_id)
            and outcome.requirement_id in self.payloads
            and outcome.realized_cost_usd > 0
        ][-HYBRID_HISTORY_K:]

    def build_prompt(self, requirement_id: str, tier: str) -> Optional[str]:
        """Render S5's L2 prompt for this auction, or None if it cannot be built."""
        from agents.provider.estimation import prompts_hybrid as ph

        payload = self.payloads.get(str(requirement_id))
        if payload is None or not self.levels:
            return None

        level_cost = self.levels[tier]["cost"]
        history = [
            ph.render_history_example(
                position, self.payloads[outcome.requirement_id],
                tier=outcome.effort_tier,
                realized_cost=outcome.realized_cost_usd,
                tier_level_cost=level_cost,
            )
            for position, outcome in enumerate(self.history_for(requirement_id, tier), start=1)
        ]
        return ph.render_user_message(payload, self.levels, history=history, level="L2")

    async def multiplier_for(
        self, requirement_id: str, tier: str, tier_level_usd: float,
        deadline_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, object]]:
        """One estimator call. Returns 1.0 — pure heuristic — on any degradation.

        ``deadline_seconds`` bounds the call to what is left of the commit
        window; see :data:`ESTIMATOR_DEADLINE_RESERVE_SECONDS`.
        """
        from agents.provider.estimation import llm_client
        from agents.provider.estimation import prompts_hybrid as ph
        from agents.provider.estimation.probe_hybrid import parse_hybrid

        if tier_level_usd <= 0:
            self._degrade("no_level")
            return 1.0, {"reason": "no_level"}

        past = self.history_for(requirement_id, tier)
        if len(past) < HYBRID_MIN_HISTORY:
            self._degrade("cold_start")
            return 1.0, {"reason": "cold_start", "n_history": len(past)}

        prompt = self.build_prompt(requirement_id, tier)
        if prompt is None:
            self._degrade("no_payload")
            return 1.0, {"reason": "no_payload"}

        probe = llm_client.OpenRouterProbe(self.api_key, cache_dir=llm_client.LIVE_CACHE_DIR)
        call = probe.call_live(
            llm_client.model_by_key(self.model_key, llm_client.ALL_MODELS),
            user_message=prompt,
            system_message=ph.SYSTEM_MESSAGES["L2"],
            schema=ph.L2_RESPONSE_SCHEMA,
            parser=parse_hybrid,
            run_seed=self.run_seed,
            agent_id=self.agent_id,
        )
        try:
            result = await _within_deadline(call, deadline_seconds)
        except asyncio.TimeoutError:
            self.calls_failed += 1
            logger.warning(
                "🤖 estimator exceeded its %.1fs window; bidding history", deadline_seconds
            )
            self._degrade("deadline_exceeded")
            return 1.0, {"reason": "deadline_exceeded", "deadline_s": deadline_seconds}
        except Exception as e:                              # transport, parse, other
            self.calls_failed += 1
            logger.warning("🤖 estimator call failed (%s); bidding history", e)
            self._degrade("call_failed")
            return 1.0, {"reason": "call_failed", "error": str(e)}

        self.calls_made += 1
        self.reasoning_cost_usd += float(result.cost_usd)
        if result.cached:
            self.calls_cached += 1

        diagnostics = {
            "n_history": len(past),
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

        multiplier = float(result.estimate.cost_multiplier.get(tier, 1.0))
        low, high = ph.COST_MULTIPLIER_BOUNDS
        if multiplier != multiplier or not (low <= multiplier <= high):
            self.calls_failed += 1
            self._degrade("out_of_bounds")
            return 1.0, {**diagnostics, "reason": "out_of_bounds", "raw": multiplier}

        return multiplier, diagnostics


# ---------------------------------------------------------------------------
# Arm X-D — decomposed estimation (S5-D)
# ---------------------------------------------------------------------------

#: Completed jobs required before the decomposed arm **prices off** its estimate.
#:
#: Not a gate on the call. Depths are elicited on every auction from the first
#: one, and a calibration observation is recorded whenever the agent wins and
#: completes a job, whatever it bid — gating the *call* deadlocked the arm, since
#: calibration could then never grow (see :meth:`DecomposedCostEstimator
#: .multiplier_for`). Below the gate the arm bids 1.0, the pure heuristic.
#:
#: **Five, not two.** Two was enough to *fit* ``b`` but not to fit it well, and
#: the E_xd_vs_h smoke run showed why. The arm's own accuracy starves its
#: calibration: bidding off a sharp estimate wins only the cheap tail, so ``b``
#: is fitted on a narrow, error-selected slice and comes out 13% high, which
#: biases every later estimate upward. Measured on that run, with the same depth
#: predictions throughout:
#:
#: ===========================================  ==========  ==============
#: ``b`` fitted on                              value       error vs truth
#: ===========================================  ==========  ==============
#: all 30 auctions (unreachable)                0.004361    --
#: the 14 jobs won while bidding off the        0.004974    +0.132
#: estimator (gate = 2)
#: the 16 jobs won while bidding off an EMA     0.004271    -0.021
#: ===========================================  ==========  ==============
#:
#: Bidding the heuristic for longer wins a *wider* sample — $0.0395-$0.1264
#: against $0.0115-$0.0652 — because those wins are not selected on the
#: estimator's own error. The warm-up is close to free: the arm bids H, which is
#: what the baseline does anyway, and the calls it makes meanwhile are what fill
#: the calibration set.
#:
#: This delays the winner's-curse loop rather than closing it; keeping some
#: unselected sampling alive after the gate opens is the fuller fix and is not
#: implemented here.
#: Completed jobs needed before the decomposed arm prices with its own fit.
#:
#: Was 5, on the belief that a deeper fit reduces bias. Tested against dataset
#: v3 and the belief does not hold: min=5 showed the *worst* bias of any setting
#: (+0.105 vs +0.054 at min=2) and bought only ~6% in the estimator's own sigma
#: (0.373 -> 0.352). What it cost was coverage. The agent wins about half the
#: auctions it bids on, so five completed jobs takes roughly eleven auctions —
#: 39% of a 30-auction run spent bidding the plain heuristic, which lifted the
#: *blended* sigma the market actually sees from 0.402 to 0.565 and carried the
#: arm past the 0.428 break-even against a free regression (hybrid_agent_design
#: §2). Coverage dominates fit depth here.
DECOMPOSED_MIN_CALIBRATION = 1

#: The tier whose retrieval budget the V1 prompt describes. L2 (architecture 2)
#: delivers medium only, and the depth scale in
#: ``prompts_decomposed.TASK_BLOCK_NO_MULTIPLIER`` is written against medium's
#: three rounds. Other tiers return the heuristic rather than silently scoring a
#: prompt whose scale does not match their budget.
DECOMPOSED_TIER = "medium"

#: Seconds of the commit window reserved for everything after the estimator:
#: the freshness re-check, pricing, signing and committing the bid.
#:
#: The transport is built for the offline probe — ``REQUEST_TIMEOUT_SECONDS =
#: 180`` with ``MAX_ATTEMPTS = 3``, so a worst case of nine minutes. That is
#: correct when a probe is scoring a dataset and catastrophic inside a 20-second
#: auction: one call was measured at **187.5s**, which cost its own auction to
#: the freshness gate and then four more that expired while the agent, which is
#: single-threaded through the call, was still waiting.
#:
#: So the caller passes what is left of the window and the estimator treats it as
#: a hard deadline. Exceeding it degrades to the heuristic like any other
#: failure — a late estimate is worth nothing, because the bid it would have
#: priced can no longer be committed.
ESTIMATOR_DEADLINE_RESERVE_SECONDS = 3.0


@dataclass
class DecomposedCostEstimator(HybridCostEstimator):
    """Arm X-D: the LLM predicts retrieval depth; the arithmetic happens here.

    Same seam as :class:`HybridCostEstimator` — ``believed_cost = level x
    multiplier``, same controller, same degradation contract — and it inherits
    that class's feature/payload/outcome bookkeeping and call accounting
    unchanged. Two things differ, and only two.

    **What is asked.** Not "what multiple of your average will this cost" but
    "how many of the medium tier's three retrieval rounds will each prompt
    consume". S5-R diagnosed why this matters: against ``log(prompt_count)`` the
    truth scales at +1.19 while the point protocol's answers scale at +0.51 —
    the model reads each prompt's depth well (residual corr +0.53) and then
    fails to *add up*. Summation is exactly what a program does exactly.

    **How dollars are recovered.** ``cost = a * n_prompts + b * sum(rounds)``,
    with ``a`` and ``b`` fitted on this agent's own finished jobs — one
    observation per job it has actually completed, recorded when it bid and
    closed when it learned what it paid. No population constant and no oracle.

    Measured (S5-R, gpt-4o-mini, realistic trajectories, V1 prompt): sigma
    **0.3387** [0.240, 0.409] at **$0.00038/call**, against the point protocol's
    0.4515 at $0.00064 and H's 0.6491. The controls that matter both passed —
    replacing the model's depths with a constant collapses it to a prompt-count
    regression (0.4611), and random depths give 0.4478, so the fitting machinery
    contributes nothing on its own.
    """

    mode: str = "decomposed"

    #: (requirement_id, tier) -> (n_prompts, sum_rounds) for auctions bid on but
    #: not yet settled. An entry becomes a calibration observation only if the
    #: agent wins and completes the job, so the fit is trained on won work alone
    #: — which is the real constraint a deployed agent faces.
    pending_rounds: Dict[Tuple[str, str], Tuple[int, float]] = field(default_factory=dict)

    #: (n_prompts, sum_rounds, realized_cost_usd), strictly causal.
    calibration: List[Tuple[int, float, float]] = field(default_factory=list)

    def record_outcome(
        self, requirement_id: str, effort_tier: str, realized_cost_usd: float
    ) -> None:
        """Close the loop: pair what was predicted with what it actually cost."""
        super().record_outcome(requirement_id, effort_tier, realized_cost_usd)
        key = (str(requirement_id), str(effort_tier))
        predicted = self.pending_rounds.pop(key, None)
        if predicted is not None and realized_cost_usd > 0:
            n_prompts, sum_rounds = predicted
            self.calibration.append((n_prompts, sum_rounds, float(realized_cost_usd)))

    def n_prompts_for(self, requirement_id: str) -> int:
        payload = self.payloads.get(str(requirement_id)) or {}
        return len(payload.get("prompts") or [])

    def build_prompt(self, requirement_id: str, tier: str) -> Optional[str]:
        """Render the V1 decomposed prompt — no history block, no multiplier field."""
        from agents.provider.estimation import prompts_decomposed as pdc

        payload = self.payloads.get(str(requirement_id))
        if payload is None or not self.levels or not (payload.get("prompts") or []):
            return None
        return pdc.render_user_message(
            payload, self.levels, history=(), with_multiplier=False
        )

    async def multiplier_for(
        self, requirement_id: str, tier: str, tier_level_usd: float,
        deadline_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, object]]:
        """One estimator call. Returns 1.0 — pure heuristic — on any degradation.

        ``deadline_seconds`` bounds the call to what is left of the commit
        window; see :data:`ESTIMATOR_DEADLINE_RESERVE_SECONDS`.
        """
        from agents.provider.estimation import llm_client
        from agents.provider.estimation import prompts_decomposed as pdc
        # Same implementation the S5-D measurement used, imported rather than
        # reimplemented, for the reason `requirement_payload` documents.
        from agents.provider.estimation.probe_realistic import parse_decomposed

        if tier != DECOMPOSED_TIER:
            # **Not counted as a degradation.** The handler asks every tier on
            # every auction, so this fires twice per auction by construction —
            # 594 times across a ten-seed pilot, against 103 real cold starts and
            # one out-of-bounds estimate. Counting it buried the two entries the
            # counter exists to surface. Nothing is lost: no call is made, the
            # multiplier is the heuristic's 1.0, architecture 2 prices off medium
            # alone so the value is never used, and the reason is still on the
            # per-auction log line.
            return 1.0, {"reason": "tier_not_supported", "tier": tier}
        if tier_level_usd <= 0:
            self._degrade("no_level")
            return 1.0, {"reason": "no_level"}

        n_prompts = self.n_prompts_for(requirement_id)
        prompt = self.build_prompt(requirement_id, tier)
        if prompt is None or n_prompts <= 0:
            self._degrade("no_payload")
            return 1.0, {"reason": "no_payload"}

        # **The call is not gated.** An earlier version returned here when the
        # calibration set was short, to avoid paying for an estimate it could not
        # convert — and deadlocked: `calibration` only grows from auctions where
        # rounds were predicted, rounds are only predicted when the estimator
        # runs, and the estimator only ran once `calibration` was long enough. The
        # arm could never leave cold start, and every gate test passed because the
        # fixtures injected `pending_rounds` directly, reaching a state the live
        # loop cannot.
        #
        # So the depths are always elicited and always recorded. What the gate
        # controls is only whether they are *bid on*: below it the arm returns 1.0
        # (pure heuristic, the caller's chosen cold-start behaviour) while the
        # observation still accrues. Two warm-up calls at ~$0.0004 buy the
        # calibration the arm runs on for the rest of the market.
        fit = (
            pdc.calibrate(self.calibration, two_param=True)
            if len(self.calibration) >= DECOMPOSED_MIN_CALIBRATION else None
        )

        probe = llm_client.OpenRouterProbe(self.api_key, cache_dir=llm_client.LIVE_CACHE_DIR)
        call = probe.call_live(
            llm_client.model_by_key(self.model_key, llm_client.ALL_MODELS),
            user_message=prompt,
            system_message=pdc.SYSTEM_MESSAGE,
            schema=pdc.response_schema(n_prompts, with_multiplier=False),
            parser=parse_decomposed,
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
        except Exception as e:                              # transport, parse, other
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

        rounds = pdc.clean_rounds(
            getattr(result.estimate, "prompt_rounds", []) or [], n_prompts
        )
        if rounds is None:
            # Wrong length or out of range: the alignment between depths and
            # prompts *is* the protocol, so a mismatch is not repaired by
            # guessing at it.
            self.calls_failed += 1
            self._degrade("bad_rounds")
            return 1.0, {**diagnostics, "reason": "bad_rounds"}

        sum_rounds = float(sum(rounds))
        # Recorded before any decision about bidding: the realised cost will pair
        # with it at completion, and that pairing is what bootstraps the fit. An
        # auction the agent loses simply never closes the pair.
        self.pending_rounds[(str(requirement_id), str(tier))] = (n_prompts, sum_rounds)
        diagnostics["sum_rounds"] = round(sum_rounds, 2)
        # The per-prompt vector, not just its sum. Pricing needs only the sum,
        # but validating the *mechanism* needs the alignment: dataset v3 records
        # how many rounds each prompt actually took, and the sharp question —
        # does the model judge difficulty, or recognise the template the
        # requirement was written from? — is answered by comparing prompt to
        # prompt within a job, not job to job. Logging only the sum would leave
        # v3's per-prompt column with nothing to pair against.
        diagnostics["prompt_rounds"] = [round(float(r), 2) for r in rounds]

        if fit is None:
            # Depths in hand, no way to price them yet. Bid the heuristic.
            self._degrade("cold_start")
            return 1.0, {**diagnostics, "reason": "cold_start"}

        predicted_cost = pdc.predict_cost(n_prompts, sum_rounds, fit)
        multiplier = predicted_cost / tier_level_usd if tier_level_usd > 0 else 1.0
        # No output band. Under decomposition the model never emits a
        # multiplier: it emits one depth per prompt, `clean_rounds` has already
        # rejected anything outside the protocol's range or misaligned with the
        # prompt count, and `calibrate` refuses negative coefficients. A cost
        # built from validated inputs and a non-negative fit is a legitimate
        # prediction, so a band on the *output* adds nothing.
        #
        # It also actively harmed: the `[0.2, 5.0]` band inherited from the
        # point protocol — where the model DOES emit the multiplier and the
        # guard is real — fired six times in 1156 estimates, every one of them
        # on a 3-prompt `locate` task the model had correctly priced at roughly
        # a fifth of the agent's average. Rejecting to 1.0 then bid the tier
        # average, a 5x overestimate on the cheapest work in the market.
        #
        # What remains is arithmetic sanity, which needs no calibration: a
        # prediction must be a positive finite number.
        if (multiplier != multiplier or multiplier in (float("inf"), float("-inf"))
                or multiplier <= 0.0):
            self.calls_failed += 1
            self._degrade("not_a_positive_number")
            return 1.0, {**diagnostics, "reason": "not_a_positive_number", "raw": multiplier}
        diagnostics["predicted_cost_usd"] = round(predicted_cost, 6)

        return multiplier, diagnostics


def ph_bounds() -> Tuple[float, float]:
    """The multiplier band the controller accepts, shared with arm X.

    Read from ``prompts_hybrid`` rather than restated, so the two arms cannot
    disagree about what counts as an absurd estimate.
    """
    from agents.provider.estimation import prompts_hybrid as ph
    return ph.COST_MULTIPLIER_BOUNDS


#: Arm R-D needs its own completed jobs before it can fit features -> depth.
#: Four regressors plus an intercept cannot be identified from fewer, and a
#: degenerate fit is worse than the heuristic it would replace.
ROUNDS_REGRESSION_MIN_FIT = 5


def _solve_normal_equations(
    rows: Sequence[Sequence[float]], targets: Sequence[float]
) -> Optional[List[float]]:
    """Least squares by Gaussian elimination on the normal equations.

    Pure Python on purpose: nothing under ``agents/`` imports numpy, and arm R-D
    is not a good enough reason to put a numeric stack on the bidding path. The
    systems here are 4x4 at most.

    Returns ``None`` when the system is singular — collinear features, or fewer
    distinct observations than parameters — rather than a coefficient vector
    built from a near-zero pivot.
    """
    if not rows:
        return None
    k = len(rows[0])
    if len(rows) < k:
        return None
    # X'X and X'y
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    atb = [sum(r[i] * t for r, t in zip(rows, targets)) for i in range(k)]
    # A whisper of ridge on the diagonal. Early in a run the agent has barely
    # more completed jobs than parameters, and any feature that has not varied
    # yet — one PDF count across its first five jobs — is exactly collinear with
    # the intercept. Without this the solve is singular and the arm sits in cold
    # start until the market happens to hand it some variety. Scaled to the
    # matrix so it is negligible once real spread arrives.
    scale = sum(ata[i][i] for i in range(k)) / k
    if scale > 0:
        for i in range(k):
            ata[i][i] += 1e-6 * scale
    for i in range(k):
        pivot_row = max(range(i, k), key=lambda r: abs(ata[r][i]))
        if abs(ata[pivot_row][i]) < 1e-12:
            return None
        ata[i], ata[pivot_row] = ata[pivot_row], ata[i]
        atb[i], atb[pivot_row] = atb[pivot_row], atb[i]
        for r in range(i + 1, k):
            factor = ata[r][i] / ata[i][i]
            if factor == 0.0:
                continue
            for c in range(i, k):
                ata[r][c] -= factor * ata[i][c]
            atb[r] -= factor * atb[i]
    out = [0.0] * k
    for i in range(k - 1, -1, -1):
        total = atb[i] - sum(ata[i][j] * out[j] for j in range(i + 1, k))
        out[i] = total / ata[i][i]
    return out


@dataclass
class RoundsRegressionCostEstimator(RegressionCostEstimator):
    """Arm R-D: counting predicts the depth, then the decomposed cost model.

    The fair-comparison arm. Arm X elicits a retrieval depth per prompt from the
    model and turns it into dollars with ``cost = a*n_prompts + b*sum(rounds)``;
    arm R has always predicted *cost* directly from countable features. Those are
    two different architectures, so a difference between them cannot be
    attributed to reading versus counting.

    This arm closes that gap. Stage 2 is byte-identical to
    :class:`DecomposedCostEstimator` — the same ``calibrate`` and
    ``predict_cost``, fitted on the agent's own completed jobs — and stage 1 is a
    regression on the same countable features arm R already uses. The only
    remaining difference between R-D and X is **where the depth number comes
    from**, which is the claim under test.

    Stage 1 is fitted on the agent's *own* completed jobs, whose realised depth
    it observes at completion. That is the same standing arm R already has with
    realised cost: an agent may learn from what it has finished. The depth of the
    job currently being priced is never read — that is the answer key.

    Requires dataset v3 or later. v2 never recorded realised rounds, so on a v2
    dataset this arm degrades to the heuristic for want of a fit, by design
    rather than by silently treating unmeasured depth as zero.
    """

    mode: str = "r_rounds"
    #: (feature vector, realised mean rounds per prompt) from completed jobs.
    rounds_history: List[Tuple[Tuple[float, ...], float]] = field(default_factory=list)
    #: (requirement_id, tier) -> (n_prompts, predicted sum_rounds), awaiting a cost.
    pending_rounds: Dict[Tuple[str, str], Tuple[int, float]] = field(default_factory=dict)
    #: (n_prompts, predicted sum_rounds, realised cost), strictly causal.
    calibration: List[Tuple[int, float, float]] = field(default_factory=list)

    def _feature_row(self, requirement_id: str, names: Sequence[str]) -> Optional[Tuple[float, ...]]:
        feats = self.features.get(str(requirement_id))
        if not feats or any(n not in feats for n in names):
            return None
        return tuple(float(feats[n]) for n in names) + (1.0,)

    def record_outcome(
        self,
        requirement_id: str,
        effort_tier: str,
        realized_cost_usd: float,
        realized_mean_rounds: Optional[float] = None,
    ) -> None:
        """Close both loops: depth for stage 1, cost for stage 2."""
        from agents.provider.estimation import dataset as estimation_dataset

        super().record_outcome(requirement_id, effort_tier, realized_cost_usd)

        names = tuple(estimation_dataset.SURFACE_FEATURES)
        row = self._feature_row(requirement_id, names)
        if row is not None and realized_mean_rounds is not None and realized_mean_rounds > 0:
            self.rounds_history.append((row, float(realized_mean_rounds)))

        key = (str(requirement_id), str(effort_tier))
        predicted = self.pending_rounds.pop(key, None)
        if predicted is not None and realized_cost_usd > 0:
            n_prompts, sum_rounds = predicted
            self.calibration.append((n_prompts, sum_rounds, float(realized_cost_usd)))

    def predict_mean_rounds(self, requirement_id: str) -> Optional[float]:
        """Stage 1: countable features -> retrieval depth for one requirement."""
        from agents.provider.estimation import dataset as estimation_dataset
        from agents.provider.estimation import prompts_decomposed as pdc

        if len(self.rounds_history) < ROUNDS_REGRESSION_MIN_FIT:
            return None
        names = tuple(estimation_dataset.SURFACE_FEATURES)
        target_row = self._feature_row(requirement_id, names)
        if target_row is None:
            return None
        beta = _solve_normal_equations(
            [row for row, _ in self.rounds_history],
            [value for _, value in self.rounds_history],
        )
        if beta is None:
            return None
        predicted = sum(b * x for b, x in zip(beta, target_row))
        low, high = pdc.ROUND_BOUNDS
        # Clamped to the range the protocol defines. An unclamped regression
        # happily predicts negative depth on an unusual feature vector, and a
        # negative sum_rounds would make `calibrate` reject the whole fit.
        return min(high, max(low, predicted))

    async def multiplier_for(
        self,
        requirement_id: str,
        tier: str,
        tier_level_usd: float,
        feature_names: Sequence[str] = (),
        deadline_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, object]]:
        """Return ``(multiplier, diagnostics)``. Never calls a model."""
        from agents.provider.estimation import prompts_decomposed as pdc

        diagnostics: Dict[str, object] = {"mode": self.mode}
        if tier != DECOMPOSED_TIER:
            # Same scope as arm X: the decomposed cost model is defined for the
            # medium tier only. Not a degradation — there is nothing to degrade.
            return 1.0, {**diagnostics, "reason": "tier_not_supported"}

        feats = self.features.get(str(requirement_id))
        if not feats:
            self._degrade("no_features")
            return 1.0, {**diagnostics, "reason": "no_features"}

        n_prompts = int(feats.get("prompt_count") or 0)
        if n_prompts <= 0:
            self._degrade("no_features")
            return 1.0, {**diagnostics, "reason": "no_features"}

        mean_rounds = self.predict_mean_rounds(requirement_id)
        if mean_rounds is None:
            self._degrade("cold_start_rounds")
            return 1.0, {**diagnostics, "reason": "cold_start_rounds",
                         "n_rounds_history": len(self.rounds_history)}

        sum_rounds = mean_rounds * n_prompts
        self.pending_rounds[(str(requirement_id), str(tier))] = (n_prompts, sum_rounds)
        diagnostics["n_prompts"] = n_prompts
        diagnostics["mean_rounds"] = round(mean_rounds, 3)
        diagnostics["sum_rounds"] = round(sum_rounds, 2)
        diagnostics["n_rounds_history"] = len(self.rounds_history)

        fit = pdc.calibrate(self.calibration, two_param=True)
        if fit is None:
            self._degrade("cold_start")
            return 1.0, {**diagnostics, "reason": "cold_start"}

        predicted_cost = pdc.predict_cost(n_prompts, sum_rounds, fit)
        multiplier = predicted_cost / tier_level_usd if tier_level_usd > 0 else 1.0
        # Same reasoning as the decomposed arm: depths are clamped to the
        # protocol range before use and the fit is non-negative, so only
        # arithmetic sanity is left to check.
        if (multiplier != multiplier or multiplier in (float("inf"), float("-inf"))
                or multiplier <= 0.0):
            self._degrade("not_a_positive_number")
            return 1.0, {**diagnostics, "reason": "not_a_positive_number", "raw": multiplier}
        diagnostics["predicted_cost_usd"] = round(predicted_cost, 6)
        return multiplier, diagnostics
