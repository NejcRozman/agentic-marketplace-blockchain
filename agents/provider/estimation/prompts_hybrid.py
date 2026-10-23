"""The hybrid estimator's prompts — S5 re-run (`docs/hybrid_agent_design.md`).

**This module does not replace :mod:`prompts`.** That module holds the frozen
``s2-v5`` protocol, which stays byte-identical so S2 and the 2026-08-11 S5 run
remain reproducible. This is a *new* protocol for the re-run, and the two are
scored against the same targets only because both resolve their multiplier
against the same dev tier levels.

What changed, and which open problem each change closes
------------------------------------------------------

**P5 — the tier mechanics were wrong, and one was missing entirely.** ``s2-v5``
states ``high: 10 tool calls, 10 retrieval rounds``. Dataset v2 was generated at
**8 and 8**. Worse, ``retrieval_k`` — chunks fetched *per round*, **2 / 4 / 8**
by tier — was never mentioned at all, while the same prompt told the model
"execution cost is dominated by retrieved-context tokens". The parameter that
most directly scales retrieved tokens was invisible. Both fixed, from the
dataset's own ``config_fingerprint``.

**P1 — the estimator was never given its own cost level.** ``s2-v5`` says "you
know your own mean cost at each tier" and then never states the number, while
the regression it is compared against receives the tier level for free (its
target is a deviation *from* that level). :data:`AGENT_STATE_BLOCK` now states
the levels explicitly.

**P4/§3.2 — no history.** The estimator saw no past job. It now receives k past
jobs as ``(requirement text → realized outcome)``, with each example's realized
cost also expressed as a multiplier so the output scale is demonstrated rather
than described.

**§3.4 — ``confidence`` is dropped.** Measured on S5 held-out data it correlated
with the model's own error at +0.06 (``gpt-5.4``) and **+0.48**
(``gpt-5.4-mini`` — anti-calibrated, more confident when more wrong). Nothing
consumed it and the regression has no counterpart.

Two prompts, not one
--------------------

``hybrid_agent_design.md`` §3.2a derives from ``decide_h3_strategy_2`` that the
estimator is **rw-conditional**, because the effort gate compares a quality
quantity against a *cost* quantity and therefore needs both tiers' costs before
the tier is chosen:

======  =====================================  ==============================
rw      history shown                          fields returned
======  =====================================  ==============================
L2      k = 3 most recent jobs                 ``cost_multiplier.medium``
≥ 30    most recent medium **and** high job    ``cost_multiplier.{medium,high}``
                                               ``quality_delta.high``
======  =====================================  ==============================

The rw < 30 branch is L2's *shape* with low substituted for medium and is not
separately templated — L3 runs at rw = 80.

**Only high-tier quality is estimated.** ``q_high`` is the sole quality the
controller reads (`policies.py` ``rating_mobility``). Medium quality is not
emitted; a reported metric nothing consumes invites misreading.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

#: Bumped from ``s2-v5``. **Not frozen during the dev loop** — `s5_implementation.md`
#: §B allows free iteration on the 20 dev requirements. Pin the fingerprint in
#: ``test_estimation_probe_v2.py`` at the moment the design is frozen, immediately
#: before the single held-out confirmation run.
PROTOCOL_VERSION = "s5-hybrid-v1"

#: The executor's real per-tier search budget, read from dataset v2's
#: ``final/manifest.json`` ``config_fingerprint.effort_budget`` rather than from
#: ``agents/config.py`` — the dataset is what the estimator must predict, and
#: config has drifted since it was generated (that drift *is* P5).
TIER_BUDGETS: Dict[str, Dict[str, int]] = {
    "low": {"tool_calls": 1, "retrieval_rounds": 1, "retrieval_k": 2},
    "medium": {"tool_calls": 3, "retrieval_rounds": 3, "retrieval_k": 4},
    "high": {"tool_calls": 8, "retrieval_rounds": 8, "retrieval_k": 8},
}

#: L2's, unchanged since the L2 runs — the wording is accurate for L2 and
#: altering it would move L2's fingerprint and discard measurements already paid
#: for. The defect was that **L3 shared it**: "predict what it will cost you to
#: run" is true of L2 and false of L3, and it was the first line the model read.
SYSTEM_MESSAGE = (
    "You are the cost-estimation component of an autonomous service provider "
    "that bids in a reverse auction for retrieval-augmented literature-review "
    "work. Before bidding, you read the requirement and predict what it will "
    "cost you to run, relative to your own historical average.\n\n"
    "You do not set prices and you do not decide whether to bid. You produce "
    "one estimate, as strict JSON matching the requested schema, and nothing "
    "else."
)

#: L3 predicts two costs, not one.
SYSTEM_MESSAGE_L3 = (
    "You are the cost-estimation component of an autonomous service provider "
    "that bids in a reverse auction for retrieval-augmented literature-review "
    "work. Before bidding, you read the requirement and predict what it will "
    "cost you to run at each of two effort budgets, relative to your own "
    "historical averages.\n\n"
    "You do not set prices and you do not decide whether to bid. You produce "
    "one estimate, as strict JSON matching the requested schema, and nothing "
    "else."
)

SYSTEM_MESSAGES: Dict[str, str] = {"L2": SYSTEM_MESSAGE, "L3": SYSTEM_MESSAGE_L3}

#: The execution mechanics. Every number is the dataset's own.
MECHANICS_BLOCK = """\
## How the job runs

You answer every prompt in the requirement by retrieval-augmented generation \
over the requirement's source PDFs. Each effort tier is a fixed search budget:

| tier | tool calls | retrieval rounds | chunks per round |
|---|---|---|---|
| low | {low_calls} | {low_rounds} | {low_k} |
| medium | {medium_calls} | {medium_rounds} | {medium_k} |
| high | {high_calls} | {high_rounds} | {high_k} |

Two execution rules always apply. **Grounding:** every claim must be supported \
by retrieved text, so an answer that cannot be grounded forces further \
retrieval. **Persistence:** you keep retrieving until the prompt is answered or \
the budget is exhausted.

**What actually drives cost.** Cost is dominated by retrieved-context tokens, \
and retrieved tokens are `rounds actually used x chunks per round`. The budget \
is a *ceiling*, not a target:

- A prompt answerable from one passage of one paper terminates after round 1 \
and costs far below the tier's ceiling, whatever the tier.
- A prompt that must reconcile or contrast findings across several papers keeps \
failing the grounding check and consumes rounds until the cap. It costs near \
the ceiling, and the ceiling is what differs between tiers.

So **the spread between requirements at the same tier comes from how many \
rounds each prompt forces**, not from the tier alone. Two requirements over the \
same documents can differ several-fold at the same tier if one asks for lookups \
and the other asks for synthesis.

Quality is rated 0-100 by an independent evaluator.\
"""

#: What the agent knows about itself. This is the P1 fix: the levels are stated,
#: not alluded to.
AGENT_STATE_BLOCK = """\
## What you already know about yourself

Your realised averages over your completed jobs — **this is the level your \
answer is relative to**:

| tier | your mean cost | your mean quality |
|---|---|---|
| low | ${low_cost:.4f} | {low_quality:.1f} |
| medium | ${medium_cost:.4f} | {medium_quality:.1f} |
| high | ${high_cost:.4f} | {high_quality:.1f} |\
"""

L2_TASK_BLOCK = """\
## What to predict

Answer in two steps, in this order.

**1. `reasoning`** — first place this job against the past jobs above: for each \
one, is this job cheaper, similar, or more expensive, and roughly by how much? \
Then say which features of the work drove that placement.

Judge from the content of the work: how many distinct source papers each prompt \
forces you to pull from, whether each answer is a single-passage lookup or a \
cross-document synthesis, and therefore how many retrieval rounds each prompt \
will consume before it can be grounded. **Do not simply count characters** — \
prompt length is a weak proxy for retrieval depth.

**2. `cost_multiplier.medium`** — what this job will cost at the **medium** tier, \
as a ratio to your ${medium_cost:.4f} medium average. Let the placement you just \
made decide the number: if the job sits between two of the past jobs, its \
multiplier sits between theirs.\
"""

L3_TASK_BLOCK = """\
## What to predict

You will deliver this job at either the **medium** or the **high** effort \
budget. Predict what each would cost. Answer in this order.

**1. `reasoning`** — first place this job against the past jobs above: for each \
one, is this job cheaper, similar, or more expensive, and roughly by how much? \
Then say which features of the work drove that placement.

Judge from the content of the work: how many distinct source papers each prompt \
forces you to pull from, whether each answer is a single-passage lookup or a \
cross-document synthesis, and how many retrieval rounds each prompt will consume \
before it can be grounded. **Do not simply count characters.**

**2. `cost_multiplier.medium`** — cost at medium, as a ratio to your \
${medium_cost:.4f} medium average. Let the placement you just made decide it.

**3. `high_to_medium_ratio`** — how many times more **this job** costs at high \
than at medium. Across your completed jobs the high tier costs about \
**{typical_ratio:.2f}x** what the medium tier costs, so that is your reference \
point: `{typical_ratio:.2f}` means this job behaves typically, higher means the \
extra budget gets used more than usual, lower means less.

A job whose prompts already ground in one or two rounds gains little from the \
larger budget and its ratio is low; a job that exhausts the medium budget will \
use the whole high budget and its ratio is high.\
"""

HISTORY_HEADER = """\
## Your last {k} completed jobs

Same format as the job below, with what it actually cost you. Use these to \
calibrate the scale of your answer.\
"""

NO_HISTORY_BLOCK = """\
## Your completed jobs

You have not yet completed a job of this kind, so you have no examples to \
calibrate against. Judge the requirement against the averages above.\
"""

CURRENT_JOB_HEADER = """\
## The job you are bidding on now\
"""

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(properties: Dict[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": "task_estimate",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": properties,
        },
    }


def _tier_object(tiers: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(tiers),
        "properties": {t: {"type": "number"} for t in tiers},
    }


L2_RESPONSE_SCHEMA: Dict[str, Any] = _schema(
    {
        "reasoning": {"type": "string"},
        "cost_multiplier": _tier_object(["medium"]),
    },
    ["reasoning", "cost_multiplier"],
)

#: L3's output. **Quality was dropped 2026-08-17** — measured, not assumed:
#: high-tier quality has a true between-requirement sd of 4.13 points against
#: 2.78 of irreducible noise, so there *is* signal, but **no feature set finds
#: it**. Leave-one-out σ_quality: guessing the tier mean 4.97; synthesis_rank
#: 5.13; rank + reconciliation + papers-per-prompt 5.19; countable features
#: 6.50. Even the realized *medium-tier cost* — which requires delivering the
#: job and is unavailable at bid time — scores 5.11. Everything is worse than
#: guessing. The gate keeps its history-based ``q_high``, which is that guess.
#:
#: ``high_to_medium_ratio`` replaces a second cost multiplier
#: because the effort gate consumes ``high − medium``, not the two levels: asking
#: for two multipliers and subtracting compounds two errors into the one quantity
#: that matters, whereas a directly estimated ratio lets correlated error cancel
#: by construction. L2 established that this model is far better at relative
#: judgement than absolute, which is the same reason.
L3_RESPONSE_SCHEMA: Dict[str, Any] = _schema(
    {
        "reasoning": {"type": "string"},
        "cost_multiplier": _tier_object(["medium"]),
        "high_to_medium_ratio": {"type": "number"},
    },
    ["reasoning", "cost_multiplier", "high_to_medium_ratio"],
)

#: A job costing less at high than at medium is a unit error, not an estimate;
#: 5x the typical premium is far outside anything v2 produced (max 2.24x).
PREMIUM_RATIO_BOUNDS = (0.5, 5.0)

#: Unchanged from ``s2-v5``: a multiplier outside this range is a unit error,
#: not an estimate. Out-of-bounds ⇒ fall back to the pure heuristic.
COST_MULTIPLIER_BOUNDS = (0.2, 5.0)
QUALITY_DELTA_BOUNDS = (-100.0, 100.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_job(payload: Mapping[str, Any]) -> str:
    """One requirement as the model sees it — raw text plus extracted features.

    §3.1: the estimator gets the **same** JSON payload the regression's features
    are derived from, so neither side is shown less.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_history_example(
    index: int,
    payload: Mapping[str, Any],
    *,
    tier: str,
    realized_cost: float,
    tier_level_cost: float,
    realized_quality: Optional[float] = None,
    tier_level_quality: Optional[float] = None,
    predicted_cost: Optional[float] = None,
) -> str:
    """A past job and how it turned out.

    The realized cost is shown **both** in dollars and as the multiplier it
    implies. Stating the multiplier is what makes the examples teach the output
    scale rather than merely illustrate the format — the model can read
    "$0.0212, which is 0.39x your medium average" and calibrate directly.

    ``predicted_cost`` is arm **X_kf** only: what this estimator itself predicted
    for that job, shown beside what happened. It tests something the regression
    structurally cannot do — notice it was wrong last time and correct. It
    requires a completed X_k pass, so it is never available on the first call.
    """
    lines = [f"### Past job {index}", render_job(payload), ""]
    multiplier = realized_cost / tier_level_cost if tier_level_cost > 0 else 1.0
    outcome = (
        f"**Delivered at {tier.upper()} tier. "
        f"Actual cost ${realized_cost:.4f} = {multiplier:.2f}x your {tier} average."
    )
    if realized_quality is not None and tier_level_quality is not None:
        delta = realized_quality - tier_level_quality
        outcome += (
            f" Evaluator rating {realized_quality:.0f} = "
            f"{delta:+.1f} vs your {tier} average."
        )
    if predicted_cost is not None and tier_level_cost > 0:
        predicted_multiplier = predicted_cost / tier_level_cost
        error = predicted_multiplier / multiplier if multiplier > 0 else 1.0
        direction = "over" if error > 1.0 else "under"
        outcome += (
            f" YOU PREDICTED {predicted_multiplier:.2f}x — "
            f"{direction} by {abs(error - 1.0):.0%}."
        )
    outcome += "**"
    lines.append(outcome)
    return "\n".join(lines)


def _state_block(levels: Mapping[str, Mapping[str, float]]) -> str:
    """Only the agent's own realised averages.

    The four-number job-shape profile that used to sit here was **removed**
    (2026-08-17). Two reasons, either sufficient. It described the *distribution
    requirements are drawn from* — population knowledge a provider acquires only
    by seeing jobs, and this agent is shown k = 3. And it was computed from all
    20 dev requirements, so it leaked aggregate structure from 17 jobs the agent
    never saw, contradicting `hybrid_agent_design.md` §3.3's "computed from the
    agent's own history, **not from the dataset**".

    What remains is knowable by construction: the agent incurred these costs.
    """
    return AGENT_STATE_BLOCK.format(
        low_cost=levels["low"]["cost"], low_quality=levels["low"]["quality"],
        medium_cost=levels["medium"]["cost"], medium_quality=levels["medium"]["quality"],
        high_cost=levels["high"]["cost"], high_quality=levels["high"]["quality"],
    )


def _mechanics_block() -> str:
    b = TIER_BUDGETS
    return MECHANICS_BLOCK.format(
        low_calls=b["low"]["tool_calls"], low_rounds=b["low"]["retrieval_rounds"],
        low_k=b["low"]["retrieval_k"],
        medium_calls=b["medium"]["tool_calls"], medium_rounds=b["medium"]["retrieval_rounds"],
        medium_k=b["medium"]["retrieval_k"],
        high_calls=b["high"]["tool_calls"], high_rounds=b["high"]["retrieval_rounds"],
        high_k=b["high"]["retrieval_k"],
    )


def render_user_message(
    payload: Mapping[str, Any],
    levels: Mapping[str, Mapping[str, float]],
    profile: Optional[Mapping[str, Any]] = None,
    history: Sequence[str] = (),
    *,
    level: str = "L2",
) -> str:
    """Assemble one estimator prompt.

    ``levels`` is ``{tier: {"cost": float, "quality": float}}`` — the agent's own
    per-tier averages, which the multiplier is relative to. ``history`` is
    pre-rendered by :func:`render_history_example`.

    ``profile`` is accepted and **ignored**; see :func:`_state_block` for why the
    job-shape profile was removed. The parameter is kept so existing callers do
    not break, and will be dropped once they are updated.
    """
    if level not in ("L2", "L3"):
        raise ValueError(f"level must be L2 or L3, got {level!r}")

    medium_cost = levels["medium"]["cost"]
    task = (L2_TASK_BLOCK if level == "L2" else L3_TASK_BLOCK).format(
        medium_cost=medium_cost,
        high_cost=levels["high"]["cost"],
        # The agent's own average premium, derived from the two levels it
        # already knows — the reference point the ratio is stated against.
        typical_ratio=(levels["high"]["cost"] / medium_cost if medium_cost else 1.0),
    )

    parts = [_mechanics_block(), _state_block(levels)]
    if history:
        parts.append(HISTORY_HEADER.format(k=len(history)))
        parts.extend(history)
    else:
        parts.append(NO_HISTORY_BLOCK)
    parts.extend([task, CURRENT_JOB_HEADER, render_job(payload), "Respond with JSON only."])
    return "\n\n".join(parts)


def protocol_fingerprint(level: str = "L2") -> str:
    """SHA-256 over the frozen strings, so a silent edit is detectable.

    Per-level: the two prompts are different protocols and are pinned
    separately.
    """
    schema = L2_RESPONSE_SCHEMA if level == "L2" else L3_RESPONSE_SCHEMA
    task = L2_TASK_BLOCK if level == "L2" else L3_TASK_BLOCK
    digest = hashlib.sha256()
    for part in (
        PROTOCOL_VERSION, level, SYSTEM_MESSAGES[level], MECHANICS_BLOCK,
        AGENT_STATE_BLOCK, task, HISTORY_HEADER, NO_HISTORY_BLOCK,
        json.dumps(schema, sort_keys=True), json.dumps(TIER_BUDGETS, sort_keys=True),
        repr(COST_MULTIPLIER_BOUNDS), repr(QUALITY_DELTA_BOUNDS),
    ):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:16]
