"""OpenRouter transport, repair rules, and the cache that stops double-spending.

Three things live here, and only the first is interesting.

**The repair rules are pre-specified** (S2 §5). What happens to a malformed or
refused response is decided here, in code, before any data exists — not
improvised later when a particular model turns out to fail on a particular
requirement. The rule is: retry the transport up to
:data:`MAX_ATTEMPTS` times, then give up and hand the harness a *B0 prediction*
(multiplier 1.0, delta 0.0) flagged as a failure. That choice is deliberate. An
arm that cannot answer should score exactly as well as not asking, because that
is what a deployed agent would fall back to; dropping the cell instead would
quietly let a model improve its σ by refusing the requirements it finds hard.

**The cache is keyed on the protocol fingerprint.** Re-running the probe after
editing the prompt correctly misses the cache and re-spends; re-running it
after editing this docstring does not. Cache entries are plain JSON so a
reviewer can read exactly what came back.

**Temperature.** S2 §5 freezes temperature at 0, but the pinned ``gpt-5``-family
models do not accept a ``temperature`` parameter at all — OpenRouter's model
metadata lists ``seed`` and ``reasoning`` for them and no ``temperature``.
Determinism is therefore pursued with a fixed :data:`SAMPLING_SEED` and a fixed
reasoning effort, and the 5 repeats measure whatever non-determinism remains
rather than assuming it away. :data:`PinnedModel.sampling` records exactly what
was sent, and that is what ``docs/estimation_protocol.md`` freezes.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import aiohttp

from agents.provider.estimation import prompts

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"

#: Live market runs cache separately from the probe (S7 §A.4). Two reasons:
#: S5's 2,000-odd entries stay untouched by anything S7 does, and a live cache
#: can be archived, inspected or deleted per experiment without disturbing the
#: probe's frozen record.
from agents.common.paths import LLM_CACHE_DIR as LIVE_CACHE_DIR  # noqa: E402

#: Transport/parse attempts before a call is recorded as failed. Three matches
#: the retry budget the live evaluator uses, so a probe failure rate and a
#: market-run failure rate mean the same thing.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

#: Concurrent in-flight requests. Low deliberately: the whole probe is a few
#: hundred calls, there is no deadline, and a rate-limit error retried three
#: times costs more than waiting.
DEFAULT_CONCURRENCY = 4

REQUEST_TIMEOUT_SECONDS = 180

#: Fixed so the probe reproduces. Not a claim that the models honour it.
SAMPLING_SEED = 20260804


@dataclass(frozen=True)
class PinnedModel:
    """One of the two models S2 freezes for S5 (§5).

    ``sampling`` is sent verbatim. It is a dict rather than named fields
    because the two pinned models are from a family whose accepted parameters
    have changed across versions, and recording what was *sent* is the thing
    that makes the run reproducible.
    """

    key: str
    model_id: str
    role: str
    sampling: Mapping[str, Any] = field(default_factory=dict)


#: The pinned pair. Both are small models; see ``docs/estimation_protocol.md``
#: §2 for why, and for what that costs the S5 model comparison.
PINNED_MODELS: Tuple[PinnedModel, ...] = (
    PinnedModel(
        key="gpt-5.4-mini",
        model_id="openai/gpt-5.4-mini",
        role="primary",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
    PinnedModel(
        key="gpt-5-mini",
        model_id="openai/gpt-5-mini",
        role="secondary",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
)


#: **S5's pair, chosen and recorded before the run** (S5 §2). ``docs/s2_report.md``
#: §8 flagged that S2 ran two *small* models and so could say nothing about
#: whether its null was a capability ceiling; S5 §2 requires one frontier and one
#: small to fix that.
#:
#: **Small: ``gpt-5.4-mini``, unchanged from S2.** Holding one model constant
#: across v1 and v2 is what makes "the S2 null was about v1, not about the model"
#: a testable statement rather than an assertion. Swapping both models would
#: confound dataset with capability and forfeit that comparison.
#:
#: **Frontier: ``gpt-5.4``, the same family and generation.** §2 wants the second
#: arm to answer "is any effect model-dependent?", which is a question about
#: *capability*. A different vendor would confound capability with training
#: lineage, tokenizer, and instruction-following idiom; same-family is the
#: controlled version of the comparison. It is also the family that generated and
#: executed dataset v2, which matches the deployed case the hybrid would face —
#: an agent estimating the cost of its own execution.
#:
#: Sampling is S2's verbatim: same seed, same ``reasoning.effort``. The protocol
#: fingerprint does not cover the model roster, but reasoning effort is a lever
#: that would change what the protocol *is*, so it stays fixed.
S5_MODELS: Tuple[PinnedModel, ...] = (
    PinnedModel(
        key="gpt-5.4",
        model_id="openai/gpt-5.4",
        role="frontier",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
    PinnedModel(
        key="gpt-5.4-mini",
        model_id="openai/gpt-5.4-mini",
        role="small",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
)


#: Exploratory arms — **not** part of S5's pre-registered pair. Used to probe the
#: capability gradient below ``gpt-5.4-mini``: does the redesigned prompt still
#: work on an older, cheaper model, and what does that cost?
#:
#: ``gpt-4o-mini`` is a different family and does not accept ``reasoning.effort``,
#: so sampling is ``seed`` + ``temperature: 0`` instead. That is a protocol
#: difference and must be stated wherever its σ is quoted — it is not a
#: like-for-like sampling comparison with the gpt-5.4 family.
EXPLORATORY_MODELS: Tuple[PinnedModel, ...] = (
    PinnedModel(
        key="gpt-4o-mini",
        model_id="openai/gpt-4o-mini",
        role="exploratory",
        sampling={"seed": SAMPLING_SEED, "temperature": 0},
    ),
    # Open-weights, 20B. A reasoning model, so it takes ``reasoning.effort``
    # like the gpt-5.4 family — which makes it the more like-for-like of the two
    # exploratory arms.
    PinnedModel(
        key="gpt-oss-20b",
        model_id="openai/gpt-oss-20b",
        role="exploratory",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
)

ALL_MODELS: Tuple[PinnedModel, ...] = PINNED_MODELS + S5_MODELS + EXPLORATORY_MODELS


def model_by_key(key: str, roster: Sequence[PinnedModel] = PINNED_MODELS) -> PinnedModel:
    for model in roster:
        if model.key == key or model.model_id == key:
            return model
    known = ", ".join(m.key for m in roster)
    raise KeyError(f"unknown model {key!r}; pinned models are: {known}")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

@dataclass
class Estimate:
    """A parsed, bounded estimate — or a flagged failure standing in for one."""

    cost_multiplier: Dict[str, float]
    quality_delta: Dict[str, float]
    confidence: float
    reasoning: str
    status: str = "ok"
    clamped: Tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status not in ("ok", "repaired")

    @classmethod
    def fallback(cls, status: str) -> "Estimate":
        """The pre-specified stand-in: B0's answer, flagged as a failure.

        Costs the arm exactly what not asking would have cost it — no more, so
        a failure is not punished beyond its information content; no less, so a
        model cannot improve its score by declining hard requirements.
        """
        return cls(
            cost_multiplier={t: 1.0 for t in ("low", "medium", "high")},
            quality_delta={t: 0.0 for t in ("low", "medium", "high")},
            confidence=0.0,
            reasoning="",
            status=status,
        )


@dataclass
class CallResult:
    """One completed call: the estimate plus what it cost to get it."""

    probe_id: str
    model_key: str
    arm: str
    repeat: int
    estimate: Estimate
    raw_content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int
    cached: bool


# ---------------------------------------------------------------------------
# Parsing and repair
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")

_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable",
    "as an ai", "i won't", "i will not",
)


def parse_estimate(content: str) -> Estimate:
    """Turn raw model output into an :class:`Estimate`, repairing what is fixable.

    The repair ladder, in order, each step attempted only if the previous
    failed:

    1. ``json.loads`` on the whole response — what a compliant structured
       output gives.
    2. ``json.loads`` on the first ``{...}`` span, which recovers responses
       wrapped in prose or a markdown fence.
    3. ``ast.literal_eval`` on that span, which recovers Python-style output
       (single quotes, ``True``) that is otherwise well-formed.

    Anything surviving that is malformed and returns a flagged fallback. Note
    that a *refusal* is classified separately from a *malformed* response: they
    have the same consequence for scoring but very different meanings for the
    report, and lumping them together would hide a model that systematically
    declines to estimate.
    """
    text = (content or "").strip()
    if not text:
        return Estimate.fallback("empty")

    payload = _try_json(text)
    status = "ok"
    if payload is None:
        match = _JSON_BLOCK.search(text)
        if match:
            payload = _try_json(match.group())
            if payload is None:
                payload = _try_literal(match.group())
            if payload is not None:
                status = "repaired"

    if payload is None:
        lowered = text.lower()
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return Estimate.fallback("refused")
        return Estimate.fallback("malformed")

    try:
        return _coerce(payload, status)
    except (KeyError, TypeError, ValueError):
        return Estimate.fallback("malformed")


def _try_json(text: str) -> Optional[Any]:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _try_literal(text: str) -> Optional[Any]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _coerce(payload: Mapping[str, Any], status: str) -> Estimate:
    """Validate structure and apply the pre-specified bounds."""
    clamped: List[str] = []

    def bounded(value: Any, low: float, high: float, label: str) -> float:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite value for {label}")
        if number < low or number > high:
            clamped.append(label)
            return max(low, min(high, number))
        return number

    multipliers: Dict[str, float] = {}
    deltas: Dict[str, float] = {}
    for tier in ("low", "medium", "high"):
        multipliers[tier] = bounded(
            payload["cost_multiplier"][tier],
            *prompts.COST_MULTIPLIER_BOUNDS,
            label=f"cost_multiplier.{tier}",
        )
        deltas[tier] = bounded(
            payload["quality_delta"][tier],
            *prompts.QUALITY_DELTA_BOUNDS,
            label=f"quality_delta.{tier}",
        )

    confidence = bounded(
        payload.get("confidence", 0.0), *prompts.CONFIDENCE_BOUNDS, label="confidence"
    )
    return Estimate(
        cost_multiplier=multipliers,
        quality_delta=deltas,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")),
        status=status,
        clamped=tuple(clamped),
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _cache_key(
    model: PinnedModel,
    arm: str,
    probe_id: str,
    repeat: int,
    salt: str = "",
) -> str:
    """Cache identity for one call.

    ``salt`` lets a caller running a *different* protocol — the S5 re-run's
    hybrid prompts — key its cache on its own fingerprint instead of S2's, so
    editing the hybrid prompt correctly misses the cache and re-spends. Empty by
    default, and the empty case appends nothing, so every key S2 has already
    written stays byte-identical.
    """
    parts = [
        prompts.protocol_fingerprint(),
        model.model_id,
        json.dumps(model.sampling, sort_keys=True),
        arm,
        probe_id,
        str(repeat),
    ]
    if salt:
        parts.append(salt)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def live_cache_key(
    model: PinnedModel,
    *,
    user_message: str,
    run_seed: Optional[int],
    agent_id: int,
    auction_id: Optional[int] = None,
    system_message: str = "",
) -> str:
    """Cache identity for one *live* call (S7 §A.4).

    The probe's key is built from grid coordinates (arm, probe_id, repeat),
    which a market auction does not have. This one is built from what actually
    determined the answer: the exact prompt, the model, the sampling block, and
    the identity of the agent and run that asked.

    ``run_seed`` and ``agent_id`` are in the key deliberately, and they are what
    stop this from being a spend-reduction tool. Two seeds never share an entry,
    and the two hybrid agents in a 2 v 2 never share one either — they are meant
    to be independent economic actors, and one cached answer serving both would
    make them one agent with two names. The budget already assumes every call is
    paid once; what the cache buys is exact re-analysis, restart, and audit.
    """
    parts = [
        "live",
        model.model_id,
        json.dumps(model.sampling, sort_keys=True),
        hashlib.sha256(system_message.encode("utf-8")).hexdigest(),
        hashlib.sha256(user_message.encode("utf-8")).hexdigest(),
        "unseeded" if run_seed is None else str(int(run_seed)),
        str(int(agent_id)),
        "" if auction_id is None else str(int(auction_id)),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


class OpenRouterProbe:
    """Issues probe calls, caches them, and tracks spend.

    Spend is tracked from OpenRouter's own ``usage`` block (requested with
    ``include: true``) rather than reconstructed from token counts and a price
    table, because the price table would go stale silently and the number is
    going into a report.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        cache_dir: Path = CACHE_DIR,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_spend_usd: Optional[float] = None,
    ) -> None:
        if not api_key:
            raise ValueError("no OpenRouter API key; set OPENROUTER_API_KEY")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._max_spend_usd = max_spend_usd
        self.spend_usd = 0.0
        self.calls_made = 0
        self.calls_cached = 0

    # -- public ------------------------------------------------------------

    async def run_batch(self, requests: Sequence[Dict[str, Any]]) -> List[CallResult]:
        """Run every request, respecting concurrency and the spend cap."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._one(session, **request) for request in requests]
            return list(await asyncio.gather(*tasks))

    async def call_live(
        self,
        model: PinnedModel,
        *,
        user_message: str,
        run_seed: Optional[int],
        agent_id: int,
        auction_id: Optional[int] = None,
        system_message: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        parser: Optional[Any] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> CallResult:
        """One call from a live agent, cached by S7 §A.4's key.

        Everything else — payload shape, JSON-schema enforcement, the retry
        ladder, and the B0 fallback on give-up — is the probe's, unchanged. That
        is the point: an estimator that differs from S5 in its transport or its
        failure rule is not the estimator whose sigma S5 measured.

        The returned ``CallResult`` carries ``prompt_tokens``, ``completion_tokens``
        and ``cost_usd`` whether it came from the network or from disk, so the
        caller charges a replayed call exactly what the live one cost (§B.3).
        """
        key = live_cache_key(
            model,
            user_message=user_message,
            run_seed=run_seed,
            agent_id=agent_id,
            auction_id=auction_id,
            system_message=system_message or prompts.SYSTEM_MESSAGE,
        )
        request = dict(
            model=model,
            arm="live",
            probe_id=f"agent{agent_id}"
                     + ("" if auction_id is None else f":auction{auction_id}"),
            repeat=0,
            user_message=user_message,
            system_message=system_message,
            schema=schema,
            parser=parser,
            cache_key=key,
        )

        if session is not None:
            return await self._one(session, **request)
        async with aiohttp.ClientSession() as owned:
            return await self._one(owned, **request)

    def cache_stats(self) -> Dict[str, Any]:
        """Live/replayed split for this client, for the run record.

        A run served from cache and a run made live must not look the same in
        the metrics: the paper's claim is that the calls were real, and that is
        checkable only if the split is recorded (S7 §A.4).
        """
        return {
            "calls_made": self.calls_made,
            "calls_cached": self.calls_cached,
            "spend_usd": round(self.spend_usd, 6),
            "cache_dir": str(self._cache_dir),
        }

    # -- internals ---------------------------------------------------------

    async def _one(
        self,
        session: aiohttp.ClientSession,
        *,
        model: PinnedModel,
        arm: str,
        probe_id: str,
        repeat: int,
        user_message: str,
        system_message: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        cache_salt: str = "",
        parser: Optional[Any] = None,
        cache_key: Optional[str] = None,
    ) -> CallResult:
        # The S5 re-run sends a different system message and schema. Defaults
        # keep every S2 caller on the frozen protocol untouched.
        #
        # `cache_key` is the live path's override (S7 §A.4). Absent — which is
        # every S2 and S5 caller — the probe key is computed exactly as before,
        # so entries already written stay addressable.
        key = cache_key or _cache_key(model, arm, probe_id, repeat, cache_salt)
        cached = self._read_cache(key)
        if cached is not None:
            self.calls_cached += 1
            return self._result_from(
                cached, model, arm, probe_id, repeat, cached=True, parser=parser
            )

        if self._max_spend_usd is not None and self.spend_usd >= self._max_spend_usd:
            raise RuntimeError(
                f"spend cap ${self._max_spend_usd:.2f} reached after "
                f"{self.calls_made} calls (${self.spend_usd:.4f}); aborting"
            )

        payload = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system_message or prompts.SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": dict(schema) if schema else prompts.RESPONSE_SCHEMA,
            },
            "usage": {"include": True},
            **dict(model.sampling),
        }

        record: Dict[str, Any] = {"content": "", "usage": {}, "attempts": 0}
        async with self._semaphore:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                record["attempts"] = attempt
                try:
                    async with session.post(
                        self._url,
                        json=payload,
                        headers=self._headers,
                        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                    record["content"] = (
                        data["choices"][0]["message"].get("content") or ""
                    )
                    record["usage"] = data.get("usage") or {}
                    self.calls_made += 1
                    self.spend_usd += float(record["usage"].get("cost") or 0.0)
                    # A response that arrived but does not parse is retried too:
                    # the failure is as likely to be a truncated stream as a
                    # model that genuinely cannot answer.
                    check = (parser or parse_estimate)(record["content"])
                    if not check.failed:
                        break
                    if attempt == MAX_ATTEMPTS:
                        break
                    logger.warning(
                        "unparseable response (%s, %s, repeat %d), attempt %d/%d",
                        model.key, probe_id, repeat, attempt, MAX_ATTEMPTS,
                    )
                except Exception as exc:  # transport, HTTP, or shape error
                    if attempt == MAX_ATTEMPTS:
                        logger.error(
                            "call failed after %d attempts (%s, %s): %s",
                            MAX_ATTEMPTS, model.key, probe_id, exc,
                        )
                        record["content"] = ""
                        record["error"] = str(exc)
                        break
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        self._write_cache(key, record)
        return self._result_from(
            record, model, arm, probe_id, repeat, cached=False, parser=parser
        )

    def _result_from(
        self,
        record: Mapping[str, Any],
        model: PinnedModel,
        arm: str,
        probe_id: str,
        repeat: int,
        *,
        cached: bool,
        parser: Optional[Any] = None,
    ) -> CallResult:
        content = str(record.get("content") or "")
        parse = parser or parse_estimate
        estimate = parse(content) if content else Estimate.fallback("no_response")
        usage = record.get("usage") or {}
        return CallResult(
            probe_id=probe_id,
            model_key=model.key,
            arm=arm,
            repeat=repeat,
            estimate=estimate,
            raw_content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(usage.get("cost") or 0.0),
            attempts=int(record.get("attempts") or 0),
            cached=cached,
        )

    # -- cache -------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A truncated cache file is not worth a crash; re-fetching costs
            # one call and removes the ambiguity.
            return None

    def _write_cache(self, key: str, record: Mapping[str, Any]) -> None:
        """Write atomically (S7 §A.6 P6).

        Under 8-way parallelism two processes can write the same key at once; a
        bare ``write_text`` lets them interleave and leave a truncated entry.
        Writing to a unique temp file in the same directory and then
        ``os.replace`` makes the swap atomic on POSIX — a reader sees either the
        old entry or the new one, never half of either.
        """
        path = self._cache_path(key)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
