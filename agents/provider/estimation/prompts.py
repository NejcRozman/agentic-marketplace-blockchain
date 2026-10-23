"""The frozen prompt, schema, and repair rules (S2 §1, §5).

**This file is the protocol.** ``docs/estimation_protocol.md`` documents it;
this module is what actually runs. After S2 closes, editing anything here
invalidates S5's held-out result and makes it exploratory — S5 is the test set
and this is the thing that must not be tuned on it. :data:`PROTOCOL_VERSION` is
stamped into every row of ``results_v1.csv`` so a later reader can tell whether
a result predates a change, and :func:`protocol_fingerprint` hashes the actual
strings so an *undocumented* change is detectable rather than merely
discouraged.

Why the output is a **multiplier** and not a dollar amount (S2 §1): the agent
already has a well-calibrated level in its per-tier EMA. What history cannot
supply is the task-conditional *deviation* from it. Asking for absolute dollars
would mix arithmetic the model is bad at into the semantic judgement it may be
good at, and would make the answer depend on model pricing the agent knows
anyway. The hybrid then composes as
``econ.execution_cost_by_tier_micro × multiplier`` — a one-line change to the
``AgentEconomics`` the S1 policies already consume, which is why the estimator/
controller split stays a literal module boundary rather than a diagram.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

#: Bumped only when the protocol deliberately changes. S2 froze v1.
PROTOCOL_VERSION = "s2-v5"

#: The provider's own service-execution budgets, from ``agents/config.py``.
#: This is configuration the agent knows about itself, not information about
#: the requirement, so including it is not leakage — and excluding it would
#: make "predict the cost at the high tier" unanswerable rather than hard.
TIER_BUDGETS = {
    "low": {"max_tool_calls": 1, "max_retrieval_rounds": 1},
    "medium": {"max_tool_calls": 3, "max_retrieval_rounds": 3},
    "high": {"max_tool_calls": 10, "max_retrieval_rounds": 10},
}

SYSTEM_MESSAGE = (
    "You are the cost-estimation component of an autonomous service provider "
    "that bids in a reverse auction for literature-review work. Before bidding, "
    "you read the requirement and judge how much work it will take relative to "
    "a typical job.\n\n"
    "You do not set prices and you do not decide whether to bid. You produce "
    "one estimate, as strict JSON matching the requested schema, and nothing "
    "else."
)

USER_TEMPLATE = """\
You are estimating a single literature-review job before deciding what to bid on it.

## How the job runs

The provider answers every prompt in the requirement using retrieval-augmented \
generation over the source PDFs. Each effort tier is a fixed search budget:

- low:    at most {low_calls} tool call, {low_rounds} retrieval round per prompt
- medium: at most {medium_calls} tool calls, {medium_rounds} retrieval rounds per prompt
- high:   at most {high_calls} tool calls, {high_rounds} retrieval rounds per prompt

Execution cost is dominated by retrieved-context tokens: prompts that force the \
model to pull material from several different papers consume more of the budget \
than prompts answerable from one passage. Quality is rated 0-100 by an \
independent evaluator on completeness, depth, citations, clarity, and synthesis.

## What to predict

You have already run many jobs of this kind, so you know your own **mean cost \
and mean quality rating at each tier**. What you do not know is how *this* job \
differs from that mean. Predict the difference, not the level:

Calibrate against the right population. Your history is not a general sample of \
literature reviews — it is **{n_jobs} jobs of the same kind as the one below**. \
A typical one has about {mean_pdfs} source papers and about {mean_prompts} \
prompts, and its prompts each draw on about {mean_papers_per_prompt} of those \
papers, so **cross-document synthesis is your normal workload, not a hard case**. \
Work that looks demanding in absolute terms may be entirely ordinary for you. A \
multiplier of 1.0 is the correct answer for a typical job from that population, \
and across your history your multipliers are about evenly split above and below \
1.0. Use values below 1.0 as readily as values above it, and reserve large \
deviations for jobs that are genuinely unusual **relative to that population** \
rather than merely large in absolute terms.

- `cost_multiplier[tier]`: this job's execution cost at that tier, as a ratio to \
your mean cost at that tier. 1.0 means exactly typical; 1.3 means 30% more \
expensive than your average job; 0.8 means 20% cheaper.
- `quality_delta[tier]`: this job's evaluator rating at that tier, minus your \
mean rating at that tier, in rating points. 0.0 means exactly typical; -8.0 \
means this job scores 8 points below your average.
- `confidence`: 0.0-1.0, how much you would trust this estimate.
- `reasoning`: one or two sentences naming the specific features of this \
requirement that drove your answer.

Judge from the content of the work: how many distinct source papers each prompt \
requires, whether answers demand cross-document synthesis or single-passage \
extraction, and how much material each prompt asks the provider to bring \
together. Do not simply count characters.

## The requirement

{payload}

Respond with JSON only.
"""

#: The structured-output schema. ``additionalProperties: false`` throughout and
#: every field required, so a provider honouring ``strict`` cannot return a
#: partially-filled object that parses but means something else.
RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "cost_quality_estimate",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["cost_multiplier", "quality_delta", "confidence", "reasoning"],
        "properties": {
            "cost_multiplier": {
                "type": "object",
                "additionalProperties": False,
                "required": ["low", "medium", "high"],
                "properties": {
                    "low": {"type": "number"},
                    "medium": {"type": "number"},
                    "high": {"type": "number"},
                },
            },
            "quality_delta": {
                "type": "object",
                "additionalProperties": False,
                "required": ["low", "medium", "high"],
                "properties": {
                    "low": {"type": "number"},
                    "medium": {"type": "number"},
                    "high": {"type": "number"},
                },
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
    },
}

# ---------------------------------------------------------------------------
# Bounds — the pre-specified handling of implausible output (S2 §5)
# ---------------------------------------------------------------------------

#: A multiplier outside this range is not a cost estimate, it is a unit error.
#: The range spans a 5× cost swing in each direction, which is roughly 25×
#: dataset v1's entire observed per-requirement spread — wide enough that
#: clamping never touches a good-faith answer, narrow enough that a model
#: replying in dollars (≈0.09) or basis points is caught.
COST_MULTIPLIER_BOUNDS = (0.2, 5.0)

#: Rating deltas are bounded by the 0-100 scale itself.
QUALITY_DELTA_BOUNDS = (-100.0, 100.0)

CONFIDENCE_BOUNDS = (0.0, 1.0)


#: Fallback profile for a caller with no history to describe. Deliberately
#: vague — an agent that has run no jobs has no population to calibrate
#: against, and inventing specific numbers would be worse than admitting it.
DEFAULT_HISTORY_PROFILE: Dict[str, Any] = {
    "n_jobs": "many",
    "mean_pdfs": "several",
    "mean_prompts": "several",
    "mean_papers_per_prompt": "several",
}


def render_user_message(
    payload: Mapping[str, Any],
    history_profile: Optional[Mapping[str, Any]] = None,
) -> str:
    """Fill the frozen template with one requirement's visible payload.

    ``payload`` is whatever ``VisibleRequirement.as_payload()`` or
    ``redacted_payload()`` produced. It is serialised as indented JSON rather
    than prose so that the full-text and redacted arms differ in exactly one
    place — the ``prompts`` value — and in nothing else.

    ``history_profile`` describes the *population* the multiplier is relative
    to: how many jobs the agent has run and their typical shape. Three things
    about it matter.

    **It is necessary.** Without it the models are badly miscalibrated. S2
    measured this directly: the unanchored protocol produced a cost bias of
    +0.33 to +0.49 in log space, with **every one of 600 predictions above
    1.0** — the models read v1's five-paper synthesis jobs as unusually hard
    because their implicit reference class is literature reviews in general,
    not this agent's workload. Supplying the profile cut the bias by an order
    of magnitude.

    **It is not leakage.** The profile is aggregate and identical for every
    requirement in a dataset, so it cannot help the model tell one requirement
    from another — it can only fix the level. It is also information a deployed
    agent genuinely has: these are its own completed jobs. The allow-list
    governs what may be said about *this* requirement, and the profile says
    nothing about it.

    **It is parameterised rather than written into the template**, because the
    template is frozen for S5 and v2 is designed to vary document count from 1
    to 8. A hard-coded "about five papers" would be true on v1 and actively
    false on v2 — the frozen protocol would then mislead the model on precisely
    the dataset the gate depends on. S2 measured this trade-off too: a generic
    anchor with no numbers recovered only part of the calibration benefit
    (bias +0.23 vs +0.02), so the numbers earn their place and must therefore
    come from the caller.
    """
    profile = dict(DEFAULT_HISTORY_PROFILE)
    profile.update(history_profile or {})
    return USER_TEMPLATE.format(
        low_calls=TIER_BUDGETS["low"]["max_tool_calls"],
        low_rounds=TIER_BUDGETS["low"]["max_retrieval_rounds"],
        medium_calls=TIER_BUDGETS["medium"]["max_tool_calls"],
        medium_rounds=TIER_BUDGETS["medium"]["max_retrieval_rounds"],
        high_calls=TIER_BUDGETS["high"]["max_tool_calls"],
        high_rounds=TIER_BUDGETS["high"]["max_retrieval_rounds"],
        n_jobs=profile["n_jobs"],
        mean_pdfs=profile["mean_pdfs"],
        mean_prompts=profile["mean_prompts"],
        mean_papers_per_prompt=profile["mean_papers_per_prompt"],
        payload=json.dumps(payload, indent=2, ensure_ascii=False),
    )


def protocol_fingerprint() -> str:
    """SHA-256 over the frozen strings, so a silent edit is detectable.

    ``PROTOCOL_VERSION`` records what someone *said* changed; this records what
    actually did. Both go into ``results_v1.csv``.
    """
    digest = hashlib.sha256()
    for part in (
        PROTOCOL_VERSION,
        SYSTEM_MESSAGE,
        USER_TEMPLATE,
        json.dumps(RESPONSE_SCHEMA, sort_keys=True),
        json.dumps(TIER_BUDGETS, sort_keys=True),
        repr(COST_MULTIPLIER_BOUNDS),
        repr(QUALITY_DELTA_BOUNDS),
    ):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:16]
