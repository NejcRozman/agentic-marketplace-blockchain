"""Probe-side history construction (step 2ʹ) and the per-agent regression (step 4).

`hybrid_agent_design.md` §3.2 says the estimator receives *k past jobs* as
``(requirement text → realized outcome)``, and §5 says arm **R** is "B1 fitted on
the agent's **own** history at the **same n** the LLM sees". Both read from the
same drawn set, so they live in one module: a draw is made once and handed to
every arm, which is what makes the comparison paired rather than merely similar.

Why the draw is random, and why that is not a shortcut
------------------------------------------------------

The design says "most recent". **Dataset v2 cannot supply that.** Its 30
requirements carry **7 distinct ``created_at`` values** — generation-batch
timestamps correlated with ``collection_id``, not job-completion times. An
agent's "last three jobs" has no counterpart here, and imposing one would order
the history by which generation chunk a requirement landed in.

So history is **sampled uniformly without replacement**, and — because *which*
three jobs are shown changes the prompt and therefore the estimate — several
independent draws are taken per test requirement and averaged
(:data:`DEFAULT_N_DRAWS`). That was already anticipated by `s5_implementation.md`
§C.2's "average over several history draws so the result is not an artifact of
which k were drawn"; the timestamp finding makes it mandatory rather than
prudent.

**The spread across draws is a measurement, not just error bars.** It says how
much exemplar *choice* matters, which is exactly what decides whether
similarity-based selection (`hybrid_agent_design.md` §9, deferred) is worth
building. :func:`draw_sensitivity` reports it.

Two invariants the probe depends on
-----------------------------------

* **A test requirement never appears in its own history** (§C.1), enforced here
  rather than left to the caller.
* **Draws are deterministic given a seed**, so arm X_k and arm R see byte-identical
  history and any difference between them is the estimator, not the sample.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agents.provider.estimation import baselines, dataset as ds, prompts_hybrid as ph
from agents.provider.estimation.dataset import ProbeRequirement

#: Independent history draws per test requirement. Each is a different set of k
#: past jobs, so N > 1 separates "the estimator is this accurate" from "that
#: particular draw was helpful". Costs N × repeats calls per requirement.
DEFAULT_N_DRAWS = 3

#: L2 shows k past jobs, all at the tier the controller will deliver at.
DEFAULT_K_L2 = 3

#: The tier L2 estimates. `h2_strategy_3` returns ``effort_tier=None`` = the
#: caller's default tier, which is **medium** by config — see
#: `hybrid_agent_design.md` §3.2a's "L2's shape, not L2's tier" note.
L2_TIER = "medium"

#: L3 above the gate shows the two tiers whose costs the effort gate compares
#: (§3.2a).
L3_TIERS: Tuple[str, str] = ("medium", "high")

#: Examples per tier at L3. **3, not 1.** The original design showed one job per
#: tier; L2 established that the model works by *bracketing* against a range, and
#: one observation offers nothing to bracket within — against a pool spanning
#: 0.21x-2.27x, a single example says nothing about scale. Six examples, three
#: per tier.
L3_K_PER_TIER = 3

#: What each tier's examples are chosen to span.
#:
#: Medium spans **cost**: it is the tier the multiplier is asked about.
#: High spans **quality**: ``quality_delta.high`` is the hardest prediction in
#: the project (G1 found no arm beat B0 on it) and the medium examples already
#: demonstrate how much cost varies between jobs, so quality is the dimension
#: with no other source of scale. Three examples can span one dimension, not two.
L3_SPAN_KEY: Dict[str, str] = {"medium": "cost", "high": "quality"}


@dataclass(frozen=True)
class HistoryExample:
    """One past job as the agent observed it.

    ``realized_cost`` and ``realized_quality`` are the **3-repeat cell means**,
    matching how S5 measured the σ targets. A real agent saw one draw; scoring
    and history are kept on the mean so the comparison against 0.4684 / 0.375 is
    like-for-like. The single-draw view is P7 and belongs in the report.
    """

    requirement: ProbeRequirement
    tier: str
    realized_cost: float
    realized_quality: float
    #: What the estimator itself predicted for this job, if known. Populated only
    #: on arm **X_kf**'s second pass — the arm needs X_k's output, which does not
    #: exist until X_k has run.
    predicted_cost: Optional[float] = None


@dataclass(frozen=True)
class HistoryDraw:
    """One sampled history for one test requirement."""

    target: ProbeRequirement
    draw_index: int
    examples: Tuple[HistoryExample, ...]

    @property
    def n(self) -> int:
        return len(self.examples)

    def requirement_ids(self) -> Tuple[str, ...]:
        return tuple(e.requirement.requirement_id for e in self.examples)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _eligible(target: ProbeRequirement, pool: Sequence[ProbeRequirement],
              tier: str) -> List[ProbeRequirement]:
    """Pool members that could have been past jobs for ``target``.

    Excludes the target itself — the §C.1 invariant — and anything missing an
    outcome at ``tier``, since the agent could not have observed a cost it never
    incurred.
    """
    return [
        r for r in pool
        if r.requirement_id != target.requirement_id and r.has(tier)
    ]


#: Which points of the agent's own cost distribution the ``span`` modes show.
#:
#: ``span``
#:     The literal extremes. Shows the widest range the agent has seen, but the
#:     endpoints are single observations and therefore the noisiest points in
#:     the history — on v2 the minimum is 0.21x, far from the bulk.
#: ``span90``
#:     The 10th / 50th / 90th percentile. Still spans the range, but anchors on
#:     points with mass either side of them rather than on the two most extreme
#:     jobs the agent happens to have run.
SPAN_QUANTILES: Dict[str, Tuple[float, ...]] = {
    "span": (0.0, 0.5, 1.0),
    "span90": (0.1, 0.5, 0.9),
}


def _span_select(
    candidates: Sequence[ProbeRequirement],
    k: int,
    tier: str,
    quantiles: Optional[Sequence[float]] = None,
    key: str = "cost",
) -> List[ProbeRequirement]:
    """The k jobs that span the agent's observed cost range, evenly spaced.

    For k = 3: cheapest, median, dearest. The agent knows what its own jobs
    cost, so choosing by realised cost uses nothing privileged — but it does
    require *having* jobs to choose from, which makes the arm a statement about
    a **mature** agent rather than a cold-starting one (see
    :func:`draw_histories`).

    Motivated by measurement, not taste: stage 2 found σ varying 0.339–0.396
    purely with *which* three jobs were drawn — a spread comparable to the whole
    margin over the economic target. A draw of three middling jobs leaves the
    model nothing to bracket the extremes against, and stage 2's error
    decomposition showed exactly that failure (cheap jobs over-predicted +29%,
    dear jobs under-predicted −12%, ``corr(error, true multiplier) = −0.76``).
    """
    rank = (
        (lambda r: r.realized_cost(tier)) if key == "cost"
        else (lambda r: r.realized_quality(tier))
    )
    ordered = sorted(candidates, key=rank)
    if k >= len(ordered):
        return list(ordered)
    if quantiles is None:
        quantiles = (
            (0.5,) if k == 1
            else tuple(i / (k - 1) for i in range(k))
        )
    last = len(ordered) - 1
    picked: List[ProbeRequirement] = []
    used: set = set()
    for q in quantiles[:k]:
        index = round(q * last)
        # Two quantiles can land on the same job in a small pool; step outward
        # rather than silently returning fewer than k examples.
        while index in used:
            index = index + 1 if index < last else index - 1
        used.add(index)
        picked.append(ordered[index])
    return picked


def draw_histories(
    target: ProbeRequirement,
    pool: Sequence[ProbeRequirement],
    *,
    level: str = "L2",
    k: int = DEFAULT_K_L2,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = 0,
    selection: str = "random",
) -> List[HistoryDraw]:
    """``n_draws`` independent histories for one test requirement.

    ``level`` selects the shape, not merely the size:

    * ``L2`` — ``k`` jobs, all delivered at medium.
    * ``L3`` — one medium job and one high job, the two tiers the effort gate
      compares. ``k`` is ignored; §3.2a fixes it.

    The seed is derived from the target's id as well as the caller's seed, so
    two different requirements do not receive correlated draws while the whole
    set stays reproducible.

    ``selection`` picks *which* past jobs are shown, and the two modes describe
    **different points on an agent's trajectory**:

    ``random``
        Three jobs sampled uniformly. Statistically this is *"the three jobs an
        agent happens to have completed"*, so it is the **early** regime — an
        agent with exactly k jobs has no choice to make. Its draw-to-draw spread
        is the variance across agent trajectories.
    ``span``
        The cheapest, median and dearest of the pool. This is the **mature**
        regime: it needs a history large enough to choose from, and to choose it
        the agent must know all its past costs. **Deterministic**, so ``n_draws``
        is forced to 1 — there is nothing to average over.

    Neither is "the" answer; a real agent moves from the first toward the second
    as jobs accumulate, and the two bracket that trajectory.
    """
    if level not in ("L2", "L3"):
        raise ValueError(f"level must be L2 or L3, got {level!r}")
    if selection not in ("random",) + tuple(SPAN_QUANTILES):
        raise ValueError(
            f"selection must be 'random' or one of {sorted(SPAN_QUANTILES)}, "
            f"got {selection!r}"
        )
    if selection != "random":
        # Span selection is deterministic given the pool, so there is nothing to
        # average over. Defined for both levels: L2 spans cost, L3 spans cost at
        # medium and quality at high (:data:`L3_SPAN_KEY`).
        n_draws = 1

    draws: List[HistoryDraw] = []
    for index in range(n_draws):
        rng = random.Random(f"{seed}|{target.requirement_id}|{index}")
        if level == "L2":
            candidates = _eligible(target, pool, L2_TIER)
            if len(candidates) < k:
                raise ValueError(
                    f"pool has {len(candidates)} eligible jobs for "
                    f"{target.requirement_id}, need k={k}"
                )
            chosen = (
                rng.sample(candidates, k)
                if selection == "random"
                else _span_select(candidates, k, L2_TIER, SPAN_QUANTILES[selection])
            )
            examples = tuple(_example(r, L2_TIER) for r in chosen)
        else:
            examples_list: List[HistoryExample] = []
            used: set = set()
            for tier in L3_TIERS:
                candidates = [
                    r for r in _eligible(target, pool, tier)
                    if r.requirement_id not in used
                ]
                if len(candidates) < L3_K_PER_TIER:
                    raise ValueError(
                        f"pool has {len(candidates)} eligible {tier}-tier jobs for "
                        f"{target.requirement_id}, need {L3_K_PER_TIER}"
                    )
                if selection == "random":
                    chosen = rng.sample(candidates, L3_K_PER_TIER)
                else:
                    # An agent delivers ONE tier per job, so medium and high
                    # examples are necessarily different requirements — the
                    # premium high/medium is never directly observable and must
                    # be inferred across jobs. Hence ``used``.
                    chosen = _span_select(
                        candidates, L3_K_PER_TIER, tier,
                        SPAN_QUANTILES[selection], key=L3_SPAN_KEY[tier],
                    )
                for pick in chosen:
                    used.add(pick.requirement_id)
                    examples_list.append(_example(pick, tier))
            examples = tuple(examples_list)
        draws.append(HistoryDraw(target=target, draw_index=index, examples=examples))
    return draws


def _example(requirement: ProbeRequirement, tier: str) -> HistoryExample:
    return HistoryExample(
        requirement=requirement,
        tier=tier,
        realized_cost=requirement.realized_cost(tier),
        realized_quality=requirement.realized_quality(tier),
    )


def with_predictions(
    draw: HistoryDraw, predicted: Dict[str, float]
) -> HistoryDraw:
    """Attach the estimator's own past predictions — arm **X_kf**'s second pass.

    ``predicted`` maps ``requirement_id`` to what the estimator predicted for
    that job. An example with no entry keeps ``predicted_cost=None`` and renders
    without the feedback line, so a partial first pass degrades to X_k for those
    examples rather than failing.
    """
    return HistoryDraw(
        target=draw.target,
        draw_index=draw.draw_index,
        examples=tuple(
            HistoryExample(
                requirement=e.requirement, tier=e.tier,
                realized_cost=e.realized_cost, realized_quality=e.realized_quality,
                predicted_cost=predicted.get(e.requirement.requirement_id),
            )
            for e in draw.examples
        ),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(
    draw: HistoryDraw,
    levels: Dict[str, Dict[str, float]],
    *,
    level: str = "L2",
) -> List[str]:
    """The draw as prompt blocks, ready for ``prompts_hybrid.render_user_message``.

    Quality is shown **only on high-tier examples, and only at L3**.

    L2's controller never reads quality at all. At L3 the only quality the
    controller reads is ``q_high`` (`policies.py` ``rating_mobility``), and the
    only quality the estimator is asked for is ``quality_delta.high`` — so a
    medium-tier example's rating is a number nothing consumes, shown next to the
    number the model is being asked to produce. It invites the model to reason
    about medium quality and to carry that reasoning into a high-tier answer.
    Cost only on the medium examples.
    """
    blocks: List[str] = []
    for position, example in enumerate(draw.examples, start=1):
        tier_levels = levels[example.tier]
        show_quality = level == "L3" and example.tier == "high"
        blocks.append(
            ph.render_history_example(
                position,
                example.requirement.visible.as_payload(),
                tier=example.tier,
                realized_cost=example.realized_cost,
                tier_level_cost=tier_levels["cost"],
                realized_quality=example.realized_quality if show_quality else None,
                tier_level_quality=tier_levels["quality"] if show_quality else None,
                predicted_cost=example.predicted_cost,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Step 4 — the regression, fitted on the same history
# ---------------------------------------------------------------------------

#: S7 §B.2 — `R_guarded`'s two pinned constants. **Pinned, not fitted.** They are
#: what a careful engineer would choose before seeing outcomes; tuning them
#: against the market until the hybrid wins would make the comparison circular,
#: exactly as §C warns about kappa.

#: Observations required per fitted parameter before the regression is trusted
#: at all. With 3 surface features the fit has 4 parameters, so the arm needs
#: n >= 8 and is the historic mean below that. That threshold is the whole
#: difference between the two R arms: it is why `R_guarded` degrades to H at
#: cold start by construction rather than extrapolating off 3 points.
GUARD_OBSERVATIONS_PER_PARAMETER = 2

#: Ridge shrinkage on standardised columns. 1.0 is a conventional default (it is
#: sklearn's), chosen for being conventional rather than favourable.
GUARD_RIDGE_LAMBDA = 1.0


@dataclass(frozen=True)
class RegressionEstimate:
    """Arm R's prediction, with the diagnostics that say whether to believe it."""

    predicted_cost: float
    multiplier: float
    n_observations: int
    usable_features: Tuple[str, ...]
    dropped_features: Tuple[str, ...]
    #: True when `R_guarded` declined to fit and returned the tier level. Not an
    #: error — the arm behaving as designed at small n (S7 §B.2).
    guard_withheld: bool = False

    @property
    def degenerate(self) -> bool:
        """True when the fit had no usable feature, so R collapsed to B0.

        Not a failure — it is P2's prediction made concrete. With k = 3 an agent
        cannot fit three coefficients on three points in any meaningful sense,
        and the honest baseline in the cold-start regime *is* the tier mean.
        """
        return not self.usable_features

    @property
    def exactly_determined(self) -> bool:
        """As many usable features as observations — interpolates, generalises poorly.

        Distinct from :attr:`degenerate`: the fit succeeds and returns a
        confident number built on no residual degrees of freedom.
        """
        return bool(self.usable_features) and len(self.usable_features) >= self.n_observations


def fit_cost_multiplier(
    observed: Sequence[Tuple[Dict[str, float], float]],
    target_features: Dict[str, float],
    tier_level_cost: float,
    *,
    feature_names: Sequence[str] = ds.SURFACE_FEATURES,
    guard: bool = False,
) -> RegressionEstimate:
    """Arm **R**'s fit, on features alone. The one implementation of it.

    ``guard`` selects between S7 §B.2's two arms, which differ *only* here:

    * ``guard=False`` — **`R_naive`**. Always fits, however little it has. This
      is the arm S6 measured at sigma 0.9327 on 3 jobs, worse than guessing, and
      reproducing that curve is why it still runs.
    * ``guard=True`` — **`R_guarded`**. Ridge-penalised, and it declines to fit
      at all below :data:`GUARD_OBSERVATIONS_PER_PARAMETER` observations per
      parameter, returning the tier level instead. It therefore *is* the pure
      heuristic at cold start, by construction.

    The second is the honest opponent: if the hybrid only beat `R_naive`, the
    cold-start result would be an artifact of a strawman nobody would deploy.

    ``observed`` is ``[(features, realized_cost), ...]`` for the agent's own past
    jobs at one tier. Callers differ only in where those come from: the probe
    draws them (:func:`regression_estimate`), and the simulator's cold-start arm
    accumulates them from the jobs it actually won. **Both must fit identically**
    or "the regression an agent can construct" means two different things in the
    two places S6 compares.

    Target is ``log(realized / tier level)``, the same as S5's B1, so the
    multiplier is comparable to the LLM's; and :func:`baselines.fit_linear` is the
    same, so zero-variance columns are screened rather than producing arbitrary
    coefficients. With no usable observation the arm *is* the tier mean, which is
    P2's cold-start prediction rather than a failure — see
    :attr:`RegressionEstimate.degenerate`.
    """
    usable = [(row, cost) for row, cost in observed if cost > 0]
    if not usable or tier_level_cost <= 0:
        return RegressionEstimate(
            predicted_cost=tier_level_cost, multiplier=1.0, n_observations=0,
            usable_features=(), dropped_features=tuple(feature_names),
        )

    rows = [row for row, _ in usable]
    targets = [math.log(cost / tier_level_cost) for _, cost in usable]

    if guard:
        # Count the parameters this fit would actually estimate — features that
        # survive the zero-variance screen, plus the intercept — and refuse if
        # the history cannot support them. Screening first matters: on a history
        # where two columns are constant, the fit is univariate and needs 4
        # observations, not 8.
        informative = [
            name for name in feature_names
            if len({row.get(name) for row in rows}) > 1
        ]
        required = GUARD_OBSERVATIONS_PER_PARAMETER * (len(informative) + 1)
        if len(usable) < required:
            return RegressionEstimate(
                predicted_cost=tier_level_cost,
                multiplier=1.0,
                n_observations=len(usable),
                usable_features=(),
                dropped_features=tuple(feature_names),
                guard_withheld=True,
            )

    model = baselines.fit_linear(
        rows, targets, feature_names,
        ridge_lambda=GUARD_RIDGE_LAMBDA if guard else 0.0,
    )

    low, high = ph.COST_MULTIPLIER_BOUNDS
    # Same bounds the LLM's output is held to. An exactly-determined fit
    # extrapolates hard, and an unbounded R would lose to the LLM on outliers
    # the LLM is protected from.
    #
    # **Clamped in log space, before exponentiating.** Identical arithmetic
    # wherever ``exp`` is finite, and the only version that survives where it is
    # not: three near-collinear observations can put ``predict`` in the hundreds,
    # and ``exp(800)`` raises ``OverflowError`` before any bound can be applied.
    # Found by S6 §E's cold-start arm, which fits on whatever an agent happened to
    # win rather than on curated draws — but the defect was always here, and the
    # offline probe simply never fed it an extreme enough history.
    log_multiplier = model.predict(target_features)
    if log_multiplier != log_multiplier:            # NaN from a degenerate solve
        log_multiplier = 0.0
    multiplier = math.exp(
        max(math.log(low), min(math.log(high), log_multiplier))
    )

    return RegressionEstimate(
        predicted_cost=tier_level_cost * multiplier,
        multiplier=multiplier,
        n_observations=len(usable),
        usable_features=tuple(model.coefficients),
        dropped_features=tuple(model.dropped),
    )


def regression_estimate(
    draw: HistoryDraw,
    tier: str,
    tier_level_cost: float,
    *,
    feature_names: Sequence[str] = ds.SURFACE_FEATURES,
) -> RegressionEstimate:
    """Arm **R**: B1 fitted on exactly the jobs the LLM was shown (§5, P2).

    The fit is on ``draw.n`` points, which at L2's k = 3 is three. That is the
    point of the arm, not a defect in it: `hybrid_agent_design.md` §5 rejects
    B1-on-20 as "an opponent no agent in a 50-auction market can construct".
    Read :attr:`RegressionEstimate.degenerate` and
    :attr:`~RegressionEstimate.exactly_determined` before quoting the number.
    """
    return fit_cost_multiplier(
        [
            (ds.mechanical_features(e.requirement.visible), e.realized_cost)
            for e in draw.examples
            if e.tier == tier
        ],
        ds.mechanical_features(draw.target.visible),
        tier_level_cost,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def draw_sensitivity(values_by_draw: Sequence[float]) -> Optional[float]:
    """Spread of an estimate across independent history draws.

    Reported because it answers a question the gate does not: how much does
    *which* examples were shown change the answer? A large spread says
    similarity-based exemplar selection (§9, deferred) is worth building; a
    negligible one says random-most-recent is fine and that ablation can be
    dropped. Returned in log space, so it is comparable to σ_cost.
    """
    logs = [math.log(v) for v in values_by_draw if v > 0]
    return statistics.pstdev(logs) if len(logs) > 1 else None


def coverage(draw: HistoryDraw) -> Dict[str, int]:
    """Examples per tier — `hybrid_agent_design.md` §3.2a's logging requirement.

    ``tier_cost_micro`` falls back to a tier-agnostic scalar when a tier has no
    history, so ``multiplier × level`` still returns a number while the premium
    ``high − medium`` rests on a wrong base. The premium metric must be split by
    whether high-tier history existed, which needs this recorded per row.
    """
    counts: Dict[str, int] = {tier: 0 for tier in ds.EFFORT_TIERS}
    for example in draw.examples:
        counts[example.tier] = counts.get(example.tier, 0) + 1
    return counts
