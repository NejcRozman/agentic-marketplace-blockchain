"""OpenRouter transport for the estimator, its repair rules, and the on-disk cache.

A malformed or refused response is retried up to :data:`MAX_ATTEMPTS` times and
then replaced by a flagged fallback (multiplier 1.0), so a failed call costs the
agent exactly what not asking would have: it bids its running average. Each call
is cached under a key built from the exact prompt, the model, the sampling block
and the run and agent that asked (:func:`live_cache_key`), so a re-run replays
the recorded answers.
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

from agents.common.paths import LLM_CACHE_DIR as LIVE_CACHE_DIR

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_CONCURRENCY = 4
REQUEST_TIMEOUT_SECONDS = 180
SAMPLING_SEED = 20260804

#: A multiplier outside this range is a unit error, not an estimate.
COST_MULTIPLIER_BOUNDS = (0.2, 5.0)
QUALITY_DELTA_BOUNDS = (-100.0, 100.0)
CONFIDENCE_BOUNDS = (0.0, 1.0)


@dataclass(frozen=True)
class PinnedModel:
    """A model of the experiments and the sampling block sent with it.

    ``sampling`` is sent verbatim and is part of the cache key.
    """

    key: str
    model_id: str
    role: str
    sampling: Mapping[str, Any] = field(default_factory=dict)


#: The models of the experiments and the sampling block sent with each. The
#: block is also part of the cache key.
MODELS: Tuple[PinnedModel, ...] = (
    PinnedModel(
        key="gpt-5.4-mini",
        model_id="openai/gpt-5.4-mini",
        role="primary",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
    PinnedModel(
        key="gpt-4o-mini",
        model_id="openai/gpt-4o-mini",
        role="exploratory",
        sampling={"seed": SAMPLING_SEED, "temperature": 0},
    ),
    PinnedModel(
        key="gpt-oss-20b",
        model_id="openai/gpt-oss-20b",
        role="exploratory",
        sampling={"seed": SAMPLING_SEED, "reasoning": {"effort": "low"}},
    ),
)


def model_by_key(key: str, roster: Sequence[PinnedModel] = MODELS) -> PinnedModel:
    for model in roster:
        if model.key == key or model.model_id == key:
            return model
    known = ", ".join(m.key for m in roster)
    raise KeyError(f"unknown model {key!r}; pinned models are: {known}")


@dataclass
class Estimate:
    """A parsed, bounded estimate, or a flagged failure standing in for one."""

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
        """The stand-in for a failed call: multiplier 1.0, flagged as a failure.

        It costs the agent exactly what not asking would have.
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
    """One completed call: the estimate and what it cost."""

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


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")

_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable",
    "as an ai", "i won't", "i will not",
)


def parse_estimate(content: str) -> Estimate:
    """Turn raw model output into an :class:`Estimate`, repairing what is fixable.

    Tried in order: ``json.loads`` on the whole response; on the first ``{...}`` span
    (prose or a markdown fence around it); ``ast.literal_eval`` on that span
    (Python-style output). Anything else is a flagged fallback; a refusal is
    classified separately from a malformed answer.
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
    """Validate the structure and apply the bounds."""
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
            *COST_MULTIPLIER_BOUNDS,
            label=f"cost_multiplier.{tier}",
        )
        deltas[tier] = bounded(
            payload["quality_delta"][tier],
            *QUALITY_DELTA_BOUNDS,
            label=f"quality_delta.{tier}",
        )

    confidence = bounded(
        payload.get("confidence", 0.0), *CONFIDENCE_BOUNDS, label="confidence"
    )
    return Estimate(
        cost_multiplier=multipliers,
        quality_delta=deltas,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")),
        status=status,
        clamped=tuple(clamped),
    )


def live_cache_key(
    model: PinnedModel,
    *,
    user_message: str,
    run_seed: Optional[int],
    agent_id: int,
    auction_id: Optional[int] = None,
    system_message: str = "",
) -> str:
    """Cache identity for one call, built from what determined the answer.

    The exact prompt, the model, the sampling block, and the run seed and agent that
    asked. Seed and agent are in the key so that two runs or two agents never share
    an answer; the cache is for exact replay, not for saving calls.
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
    """Issues estimator calls and caches them.

    The cost is OpenRouter's own ``usage`` block (requested with ``include: true``),
    not a reconstruction from tokens and a price table.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        cache_dir: Path = LIVE_CACHE_DIR,
        concurrency: int = DEFAULT_CONCURRENCY,
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
        self.spend_usd = 0.0
        self.calls_made = 0
        self.calls_cached = 0

    # --- Public ---
    async def call_live(
        self,
        model: PinnedModel,
        *,
        user_message: str,
        run_seed: Optional[int],
        agent_id: int,
        auction_id: Optional[int] = None,
        system_message: str,
        schema: Mapping[str, Any],
        parser: Any,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> CallResult:
        """One estimator call, served from the cache when the same call was made before.

        The returned ``CallResult`` carries ``prompt_tokens``, ``completion_tokens`` and
        ``cost_usd`` whether it came from the network or from disk, so the caller
        charges a replayed call exactly what the live one cost.
        """
        key = live_cache_key(
            model,
            user_message=user_message,
            run_seed=run_seed,
            agent_id=agent_id,
            auction_id=auction_id,
            system_message=system_message,
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

    # --- Internals ---
    async def _one(
        self,
        session: aiohttp.ClientSession,
        *,
        model: PinnedModel,
        arm: str,
        probe_id: str,
        repeat: int,
        user_message: str,
        system_message: str,
        schema: Mapping[str, Any],
        parser: Any,
        cache_key: str,
    ) -> CallResult:
        key = cache_key
        cached = self._read_cache(key)
        if cached is not None:
            self.calls_cached += 1
            return self._result_from(
                cached, model, arm, probe_id, repeat, cached=True, parser=parser
            )

        payload = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": dict(schema),
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
                    # A response that does not parse is retried too: a truncated stream is as likely
                    # as a model that cannot answer.
                    check = (parser or parse_estimate)(record["content"])
                    if not check.failed:
                        break
                    if attempt == MAX_ATTEMPTS:
                        break
                    logger.warning(
                        "unparseable response (%s, %s, repeat %d), attempt %d/%d",
                        model.key, probe_id, repeat, attempt, MAX_ATTEMPTS,
                    )
                except Exception as exc:  # transport, HTTP or shape error
                    if attempt == MAX_ATTEMPTS:
                        logger.error(
                            "call failed after %d attempts (%s, %s): %s",
                            MAX_ATTEMPTS, model.key, probe_id, exc,
                        )
                        record["content"] = ""
                        record["error"] = str(exc)
                        break
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        # A call that never got a response is not cached, so a later run asks
        # again instead of replaying the failure (e.g. a run without a valid key).
        if "error" not in record:
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

    # --- Cache ---
    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A truncated cache file is fetched again rather than raising.
            return None

    def _write_cache(self, key: str, record: Mapping[str, Any]) -> None:
        """Write atomically.

        Parallel runs can write the same key at once; writing to a unique temporary
        file and then ``os.replace`` means a reader sees either the old entry or the
        new one, never half of either.
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
