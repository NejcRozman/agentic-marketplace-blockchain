"""S5-D — decomposed elicitation. A SIDE BRANCH; the live protocol is untouched.

``prompts_hybrid.py`` is what S7's agents import and what its fingerprints pin.
This module changes none of it: everything shared — mechanics, state, history and
job rendering — is *imported* from there, so the two cannot drift; everything
that differs lives here under its own ``PROTOCOL_VERSION``.

Why decompose
-------------

Cost is generated additively: each prompt burns retrieval rounds until it is
grounded or the budget caps it, and cost tracks retrieved tokens. So

    cost ~ a * n_prompts + b * sum_j(rounds_j)

``n_prompts`` is *countable* — it needs no model. ``rounds_j`` is not: whether a
prompt is a one-passage lookup or a cross-document reconciliation is a reading
judgement, and the countable proxies for it fail (all mechanical features
together reach corr +0.33 with the depth channel **fitted in-sample**, against
the LLM's +0.53 on unseen calls).

The S5-R diagnostics say the point protocol splits those two badly. Against
``log(prompt_count)`` the truth scales at **+1.19**; the model's answers scale at
**+0.51** — it reads each prompt's depth well (residual corr +0.531) and then
fails to *add up*, compressing toward the middle. Summation is exactly what a
program does exactly.

So this protocol elicits **only the per-prompt depth** and computes the money
outside:

    predicted_cost = a * n_prompts + b * sum(rounds)

The model keeps the judgement it is good at; counting, summation and the dollar
conversion move into arithmetic.

**Not the comparative protocol's mistake.** S5-C also moved arithmetic outside
and lost badly, because the quantity it elicited — pairwise ratios — was itself
compressed toward 1.0. Here the elicited quantity is a bounded physical count on
a scale the prompt defines, and the model's competence at it is already measured.

Calibration
-----------

``a`` and ``b`` are fitted from the agent's **own** completed jobs: it recorded
``sum(rounds)`` when it bid and learned the realised cost afterwards, so every
finished job is one observation. No population constant, no oracle, and no round
predictions for history jobs — which keeps the response short. Before the fit is
estimable the caller degrades to the heuristic, which is the ``k >= 2`` gate the
live agent wants anyway. See :func:`calibrate` and :func:`predict_cost`.

The response also carries ``cost_multiplier`` — the *old* protocol's answer — so
one paid run yields a within-call paired comparison instead of a cross-run one.
**It is experiment scaffolding and would be dropped in production**; the
decomposed estimate never reads it.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from agents.provider.estimation.prompts_hybrid import (  # shared, unchanged, imported
    CURRENT_JOB_HEADER,
    HISTORY_HEADER,
    MECHANICS_BLOCK,
    NO_HISTORY_BLOCK,
    _mechanics_block,
    _schema,
    _state_block,
    render_history_example,  # noqa: F401  — re-exported for probe convenience
    render_job,
)

PROTOCOL_VERSION = "s5-decomposed-v1"

#: Retrieval rounds available at the medium tier — the scale the answer lives on.
#: Taken from the same ``TIER_BUDGETS`` the mechanics block renders, so the two
#: cannot disagree.
MEDIUM_ROUNDS = 3

#: A depth outside this band is a formatting failure, not a judgement.
ROUND_BOUNDS = (1.0, float(MEDIUM_ROUNDS))

SYSTEM_MESSAGE = (
    "You are the cost-estimation component of an autonomous service provider "
    "that bids in a reverse auction for retrieval-augmented literature-review "
    "work. Before bidding, you read each question in the requirement and judge "
    "how much retrieval it will take to answer.\n\n"
    "You do not set prices and you do not decide whether to bid. You produce "
    "one estimate, as strict JSON matching the requested schema, and nothing "
    "else."
)

TASK_BLOCK = """\
## What to predict

Answer in three steps, in this order.

**1. `reasoning`** — go through the prompts of the job below one at a time. For \
each, say whether it can be answered from a single passage of a single paper, or \
whether it forces you to pull from several papers and reconcile them before the \
answer can be grounded.

**2. `prompt_rounds`** — one number per prompt in the job below, **in the same \
order they appear**, and exactly {n_prompts} of them.

`prompt_rounds[j]` is how many of the medium tier's {rounds} retrieval rounds \
prompt *j* will actually consume before it is answered. Use the scale:

- **1.0** — answered by the first retrieval; a named fact in one paper.
- **2.0** — needs a second pass, typically a second source or a check.
- **3.0** — runs to the cap: cross-document synthesis that keeps failing the \
grounding check.

Decimals are allowed for prompts that sit between these. Judge each prompt on \
its own — **do not** average them, and **do not** adjust them so the job as a \
whole looks typical. Long wording does not mean deep retrieval: a long prompt \
can be a single lookup, and a short one can demand synthesis.

**3. `cost_multiplier.medium`** — as before, this job's total cost at the medium \
tier as a ratio to your ${medium_cost:.4f} medium average.\
"""

#: The same task without the point protocol's answer. `cost_multiplier` is
#: experiment scaffolding — the decomposed estimate never reads it — so the
#: production protocol drops it and the two steps stand alone.
TASK_BLOCK_NO_MULTIPLIER = """\
## What to predict

Answer in two steps, in this order.

**1. `reasoning`** — go through the prompts of the job below one at a time. For \
each, say whether it can be answered from a single passage of a single paper, or \
whether it forces you to pull from several papers and reconcile them before the \
answer can be grounded.

**2. `prompt_rounds`** — one number per prompt in the job below, **in the same \
order they appear**, and exactly {n_prompts} of them.

`prompt_rounds[j]` is how many of the medium tier's {rounds} retrieval rounds \
prompt *j* will actually consume before it is answered. Use the scale:

- **1.0** — answered by the first retrieval; a named fact in one paper.
- **2.0** — needs a second pass, typically a second source or a check.
- **3.0** — runs to the cap: cross-document synthesis that keeps failing the \
grounding check.

Decimals are allowed for prompts that sit between these. Judge each prompt on \
its own — **do not** average them, and **do not** adjust them so the job as a \
whole looks typical. Long wording does not mean deep retrieval: a long prompt \
can be a single lookup, and a short one can demand synthesis.\
"""

#: History rendered for the *depth* task rather than the cost task.
#:
#: The point protocol shows each past job's total cost and its multiple of the
#: agent's average — the right anchor when the answer is a cost multiple, the
#: wrong one when the answer is retrieval rounds. This renders the observable
#: shadow of depth instead: cost **per prompt**, relative to the agent's own
#: typical prompt. Deeper jobs cost more per prompt, which is the quantity the
#: model is being asked to predict.
#:
#: **Only observable quantities appear here.** Rounds actually consumed are not
#: recorded by the executor, and ``demanded_retrieval_rounds`` is a hidden design
#: variable of the requirement, so neither may be shown.
HISTORY_HEADER_DEPTH = """\
## Your last {k} completed jobs

Same format as the job below. For each, the number of prompts it contained and \
what it cost **per prompt** compared with your own typical prompt — a job whose \
prompts averaged well above your typical one was a job whose prompts kept \
forcing further retrieval.\
"""


def response_schema(n_prompts: int, *, with_multiplier: bool = True) -> Dict[str, Any]:
    """One depth per prompt, length pinned so a short answer cannot misalign."""
    properties: Dict[str, Any] = {
        "reasoning": {"type": "string"},
        "prompt_rounds": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": n_prompts,
            "maxItems": n_prompts,
        },
    }
    required = ["reasoning", "prompt_rounds"]
    if with_multiplier:
        properties["cost_multiplier"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["medium"],
            "properties": {"medium": {"type": "number"}},
        }
        required.append("cost_multiplier")
    return _schema(properties, required)


def clean_rounds(rounds: Sequence[float], n_prompts: int) -> Optional[Tuple[float, ...]]:
    """Depths, bounds-checked and length-checked. ``None`` if unusable.

    A wrong-length array is rejected rather than padded: the alignment between
    depths and prompts is the protocol, and a caller that guesses at it would
    silently score a different quantity.
    """
    if len(rounds) != n_prompts:
        return None
    try:
        values = tuple(float(r) for r in rounds)
    except (TypeError, ValueError):
        return None
    if any(r != r or not (ROUND_BOUNDS[0] <= r <= ROUND_BOUNDS[1]) for r in values):
        return None
    return values


def calibrate(
    observations: Sequence[Tuple[int, float, float]], *, two_param: bool = True
) -> Optional[Tuple[float, float]]:
    """Fit ``(a, b)`` in ``cost = a*n_prompts + b*sum(rounds)`` on own history.

    ``observations`` are ``(n_prompts, sum_rounds, realised_cost)`` from the
    agent's completed jobs — one per finished job, all of them strictly causal.

    Two parameters need two observations and an actual spread in the regressors;
    with less, or when the fit degenerates, this falls back to the one-parameter
    form ``cost = b*sum(rounds)`` and finally returns ``None``, which the caller
    degrades to the heuristic. A negative fitted coefficient is rejected for the
    same reason: cost cannot fall as work rises, so a negative fit is noise, not
    a finding.
    """
    usable = [(float(n), float(s), float(c))
              for n, s, c in observations if n > 0 and s > 0 and c > 0]
    if not usable:
        return None
    if two_param and len(usable) >= 2:
        # Ordinary least squares through the origin on two regressors.
        s_nn = sum(n * n for n, _, _ in usable)
        s_ns = sum(n * s for n, s, _ in usable)
        s_ss = sum(s * s for _, s, _ in usable)
        s_nc = sum(n * c for n, _, c in usable)
        s_sc = sum(s * c for _, s, c in usable)
        det = s_nn * s_ss - s_ns * s_ns
        if abs(det) > 1e-12:
            a = (s_ss * s_nc - s_ns * s_sc) / det
            b = (s_nn * s_sc - s_ns * s_nc) / det
            if a >= 0.0 and b >= 0.0 and (a + b) > 0.0:
                return a, b
    s_ss = sum(s * s for _, s, _ in usable)
    if s_ss <= 0:
        return None
    b = sum(s * c for _, s, c in usable) / s_ss
    return (0.0, b) if b > 0 else None


def predict_cost(n_prompts: int, sum_rounds: float, fit: Tuple[float, float]) -> float:
    a, b = fit
    return a * float(n_prompts) + b * float(sum_rounds)


def render_depth_example(
    index: int,
    payload: Mapping[str, Any],
    *,
    realized_cost: float,
    tier_level_cost: float,
) -> str:
    """A past job, described in the units the depth question is asked in.

    ``tier_level_cost`` is the agent's average cost per *job*; dividing by each
    job's own prompt count on both sides turns it into a per-prompt comparison
    that does not require the agent to know a dollars-per-round constant.
    """
    prompts = payload.get("prompts") or []
    n = len(prompts)
    lines = [f"### Past job {index}", render_job(payload), ""]
    if n > 0 and tier_level_cost > 0:
        per_prompt = realized_cost / n
        # The agent's typical prompt: its average job cost over the average job
        # size it has seen. `tier_level_cost` already carries the former; using
        # this job's own prompt count for the latter keeps the statement local
        # and needs no extra state.
        relative = realized_cost / tier_level_cost
        lines.append(
            f"**Delivered at MEDIUM tier. {n} prompts, "
            f"${per_prompt:.4f} per prompt. "
            f"The job as a whole cost {relative:.2f}x your medium average.**"
        )
    return "\n".join(lines)


def render_user_message(
    payload: Mapping[str, Any],
    levels: Mapping[str, Mapping[str, float]],
    history: Sequence[str] = (),
    *,
    with_multiplier: bool = True,
) -> str:
    """Assemble one decomposed prompt.

    Mirrors the hybrid L2 assembly exactly — same blocks, same order — with only
    the task block swapped, so the A/B isolates the elicitation format alone.
    The history is still shown: it anchors the model's ``cost_multiplier`` answer
    (the paired control) and costs nothing extra, since it is already rendered.
    """
    prompts = payload.get("prompts") or []
    block = TASK_BLOCK if with_multiplier else TASK_BLOCK_NO_MULTIPLIER
    task = block.format(
        n_prompts=len(prompts),
        rounds=MEDIUM_ROUNDS,
        medium_cost=levels["medium"]["cost"],
    )
    parts = [_mechanics_block(), _state_block(levels)]
    if history:
        header = HISTORY_HEADER if with_multiplier else HISTORY_HEADER_DEPTH
        parts.append(header.format(k=len(history)))
        parts.extend(history)
    parts.extend([task, CURRENT_JOB_HEADER, render_job(payload), "Respond with JSON only."])
    return "\n\n".join(parts)


def protocol_fingerprint(n_prompts: int = 6) -> str:
    """Pins this variant separately; the hybrid fingerprints are untouched."""
    digest = hashlib.sha256()
    for part in (
        PROTOCOL_VERSION,
        SYSTEM_MESSAGE,
        MECHANICS_BLOCK,
        TASK_BLOCK.format(n_prompts=n_prompts, rounds=MEDIUM_ROUNDS, medium_cost=0.05),
        json.dumps(response_schema(n_prompts), sort_keys=True),
    ):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:16]
