"""LLM-Estimate: the prompt, the response schema and parser, and the cost model.

The model is asked only how many retrieval rounds each prompt of the request will
consume. The agent turns those depths into a cost with
``cost = a * n_prompts + b * sum(rounds)``, where ``a`` and ``b`` are fitted on
its own completed jobs (:func:`calibrate`).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from agents.provider.estimation import llm_client
from agents.provider.estimation.llm_client import COST_MULTIPLIER_BOUNDS

#: The tier the prompt describes; the market delivers medium only.
TIER = "medium"

#: The executor's per-tier search budget, as stated in the prompt.
TIER_BUDGETS: Dict[str, Dict[str, int]] = {
    "low": {"tool_calls": 1, "retrieval_rounds": 1, "retrieval_k": 2},
    "medium": {"tool_calls": 3, "retrieval_rounds": 3, "retrieval_k": 4},
    "high": {"tool_calls": 8, "retrieval_rounds": 8, "retrieval_k": 8},
}

#: Retrieval rounds available at the medium tier: the scale the answer lives on.
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

TASK_BLOCK = """\
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

CURRENT_JOB_HEADER = """\
## The job you are bidding on now\
"""


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


def response_schema(n_prompts: int) -> Dict[str, Any]:
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
    return _schema(properties, ["reasoning", "prompt_rounds"])


def render_job(payload: Mapping[str, Any]) -> str:
    """One request as the model sees it: its text and countable fields, as JSON."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _state_block(levels: Mapping[str, Mapping[str, float]]) -> str:
    """The agent's own realised per-tier averages, which the answer is relative to."""
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
    payload: Mapping[str, Any], levels: Mapping[str, Mapping[str, float]]
) -> str:
    """Assemble the prompt for one request, given the agent's own per-tier levels."""
    prompts = payload.get("prompts") or []
    task = TASK_BLOCK.format(
        n_prompts=len(prompts),
        rounds=MEDIUM_ROUNDS,
        medium_cost=levels["medium"]["cost"],
    )
    parts = [_mechanics_block(), _state_block(levels), task, CURRENT_JOB_HEADER,
             render_job(payload), "Respond with JSON only."]
    return "\n\n".join(parts)


def parse_multiplier(content: str, tier: str = TIER) -> llm_client.Estimate:
    """Parse ``{"cost_multiplier": {"medium": x}, "reasoning": "..."}``.

    Uses the repair ladder of :func:`llm_client.parse_estimate` (whole-body JSON,
    then the first ``{...}`` span, then a literal eval). Anything unparseable or
    non-finite becomes a flagged fallback with multiplier 1.0, which bids exactly as
    the heuristic would.
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

    low, high = COST_MULTIPLIER_BOUNDS
    clamped = () if low <= raw <= high else (f"cost_multiplier.{tier}",)
    return llm_client.Estimate(
        cost_multiplier={tier: max(low, min(high, raw))},
        quality_delta={},
        confidence=0.0,
        reasoning=str(payload.get("reasoning", "")),
        status="ok",
        clamped=clamped,
    )


def parse_response(content: str):
    """Parse LLM-Estimate's answer and attach its per-prompt depths (``prompt_rounds``).

    The estimate itself is computed by the caller from the depths and the agent's
    own calibration.
    """
    import json as _json
    try:
        payload = _json.loads(content)
        rounds = [float(r) for r in payload["prompt_rounds"]]
        # Absent from LLM-Estimate's schema, so it defaults to 1.0 (the heuristic).
        multiplier = float((payload.get("cost_multiplier") or {}).get("medium", 1.0))
    except Exception:
        return parse_multiplier(content)  # its repair ladder, then a failed Estimate
    estimate = parse_multiplier(_json.dumps({
        "reasoning": payload.get("reasoning", ""),
        "cost_multiplier": {"medium": multiplier},
    }))
    estimate.prompt_rounds = rounds
    return estimate


def clean_rounds(rounds: Sequence[float], n_prompts: int) -> Optional[Tuple[float, ...]]:
    """The depths, checked for length and range; None if unusable.

    A wrong-length answer is rejected rather than padded: the alignment of depths with
    prompts is the protocol.
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
    """Fit ``(a, b)`` in ``cost = a * n_prompts + b * sum(rounds)`` on the agent's own jobs.

    ``observations`` are ``(n_prompts, sum_rounds, realised_cost)``, one per completed
    job. Two parameters need at least two observations with spread in the regressors;
    otherwise, or if the fit degenerates, the one-parameter form ``cost = b * sum(rounds)``
    is used, and None if that fails too. Negative coefficients are rejected: cost cannot
    fall as work rises.
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
