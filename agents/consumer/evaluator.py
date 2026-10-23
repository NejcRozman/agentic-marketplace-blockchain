"""The consumer's LLM judge: rates a delivered result from 0 to 100.

Per-answer criteria (depth, clarity, ...) are scored in parallel, one call per
answer; holistic criteria (completeness) are scored in one call over all answers.
The scores are averaged into the rating.
"""

import asyncio
import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

try:
    from ..config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

# Criteria whose value can only be judged by looking at the full response set.
HOLISTIC_CRITERIA = {"completeness"}

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds, doubled on each attempt


class _UsageAccumulator:
    """Token usage and cost of one ``evaluate()`` call (``len(responses) + 1`` LLM calls).

    Passed down the call chain rather than kept on the evaluator, so concurrent
    evaluations on one instance cannot mix their totals.
    """

    __slots__ = ("llm_calls", "input_tokens", "output_tokens")

    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, usage: Optional[Dict[str, Any]]) -> None:
        self.llm_calls += 1
        if not usage:
            return

        def _non_negative(value: Any) -> int:
            try:
                return max(0, int(value or 0))
            except (TypeError, ValueError):
                return 0

        self.input_tokens += _non_negative(usage.get("prompt_tokens") or usage.get("input_tokens"))
        self.output_tokens += _non_negative(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )

    def cost_usd(self, input_price_per_1k: float, output_price_per_1k: float) -> float:
        return (
            (self.input_tokens / 1000.0) * input_price_per_1k
            + (self.output_tokens / 1000.0) * output_price_per_1k
        )


class ServiceEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self._api_url = config.openrouter_base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        # Read directly: a missing `openrouter_model` must fail rather than fall back to
        # a model nobody chose.
        self._model = config.openrouter_model
        self._input_price_per_1k = float(getattr(config, "consumer_llm_input_price_per_1k", 0.0))
        self._output_price_per_1k = float(getattr(config, "consumer_llm_output_price_per_1k", 0.0))
        logger.info("ServiceEvaluator initialized with model %s", self._model)

    @property
    def model(self) -> str:
        """The judge model, recorded with its ratings."""
        return self._model

    # --- Public API ---
    async def evaluate(
        self,
        service_requirements: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rate a delivered result.

        Returns the rating (0-100), the per-criterion quality_scores, explanations, an
        error if any, and what the judge cost.
        """
        logger.info("Starting evaluation...")

        quality_criteria: Dict[str, str] = service_requirements.get("quality_criteria", {})
        required_prompts: List[str] = service_requirements.get("prompts", [])
        responses: List[Dict[str, str]] = result.get("responses", [])

        if not responses or not quality_criteria:
            logger.warning("No responses or quality criteria found; returning neutral scores.")
            return {
                "rating": 0,
                "quality_scores": {},
                "explanations": [],
                "error": "No responses or quality criteria found.",
                **self._usage_payload(_UsageAccumulator()),
            }

        per_pair_criteria = {k: v for k, v in quality_criteria.items() if k not in HOLISTIC_CRITERIA}
        holistic_criteria = {k: v for k, v in quality_criteria.items() if k in HOLISTIC_CRITERIA}
        usage = _UsageAccumulator()

        async with aiohttp.ClientSession() as session:
            # Per-answer criteria, in parallel
            pair_tasks = [
                self._score_pair(session, pair, per_pair_criteria, usage)
                for pair in responses
            ]
            pair_results: List[Optional[Dict]] = await asyncio.gather(*pair_tasks)

            # Holistic criteria, one call over all answers
            holistic_scores: Dict[str, int] = {}
            holistic_explanation: str = ""
            if holistic_criteria:
                holistic_scores, holistic_explanation = await self._score_holistic(
                    session, responses, holistic_criteria, usage
                )

        # Completeness cannot exceed the share of required prompts that were answered.
        if "completeness" in holistic_scores:
            prompt_coverage_cap = self._completeness_cap(required_prompts, responses)
            holistic_scores["completeness"] = min(holistic_scores["completeness"], prompt_coverage_cap)

        # Average the per-answer scores, skipping failed ones.
        per_pair_dim_scores: Dict[str, List[int]] = {dim: [] for dim in per_pair_criteria}
        explanations: List[str] = []
        for r in pair_results:
            if r is None:
                continue
            for dim in per_pair_criteria:
                if dim in r.get("scores", {}):
                    per_pair_dim_scores[dim].append(r["scores"][dim])
            if r.get("explanation"):
                explanations.append(r["explanation"])

        agg_scores: Dict[str, int] = {}
        for dim, vals in per_pair_dim_scores.items():
            agg_scores[dim] = round(sum(vals) / len(vals)) if vals else 0

        # Holistic scores override the per-answer ones for their criteria.
        agg_scores.update(holistic_scores)
        if holistic_explanation:
            explanations.append(holistic_explanation)

        overall = max(0, min(100, round(sum(agg_scores.values()) / len(agg_scores)))) if agg_scores else 0

        logger.info(f"Aggregated scores: {agg_scores}, overall: {overall}")
        return {
            "rating": overall,
            "quality_scores": agg_scores,
            "explanations": explanations,
            "error": None,
            **self._usage_payload(usage),
        }

    def _usage_payload(self, usage: _UsageAccumulator) -> Dict[str, Any]:
        """What the judge cost, reported with every rating."""
        return {
            "evaluator_model": self._model,
            "evaluator_llm_calls": usage.llm_calls,
            "evaluator_input_tokens": usage.input_tokens,
            "evaluator_output_tokens": usage.output_tokens,
            "evaluator_cost_usd": round(
                usage.cost_usd(self._input_price_per_1k, self._output_price_per_1k), 8
            ),
        }

    # --- Internals ---
    async def _score_pair(
        self,
        session: aiohttp.ClientSession,
        pair: Dict[str, str],
        criteria: Dict[str, str],
        usage: _UsageAccumulator,
    ) -> Optional[Dict]:
        """Score one prompt and its answer; None on unrecoverable failure."""
        if not criteria:
            return None

        prompt_text = pair.get("prompt", "")
        response_text = pair.get("response", "")
        criteria_str = "\n".join(f"- {dim}: {desc}" for dim, desc in criteria.items())
        dim_keys = list(criteria.keys())
        score_schema = ", ".join(f'"{d}": <int>' for d in dim_keys)

        eval_prompt = (
            "You are an expert evaluator of literature-analysis responses.\n\n"

            "IMPORTANT:\n"
            "- You do NOT have access to the source papers.\n"
            "- Evaluate ONLY the quality of the response itself.\n"
            "- Do NOT assume facts that are not present.\n"
            "- Do NOT speculate about missing findings or inaccuracies that cannot be verified from the response alone.\n"
            "- However, you SHOULD evaluate whether the response addresses all entities, papers, concepts, and comparison requests explicitly mentioned in the prompt.\n"
            "- If a prompt requests comparison of multiple papers, responses that analyze only a subset of those papers should receive substantially lower completeness and synthesis scores, even when the omission is explicitly acknowledged.\n"
            "- Do NOT reward verbosity, repetition, hedging, or disclaimers by themselves.\n"
            "- Evaluate information content rather than response length.\n\n"

            "Scoring Guidelines:\n"
            "- Use the full 0-100 range.\n\n"

            "Global Interpretation:\n"
            "90-100: Exceptional\n"
            "80-89: Strong\n"
            "70-79: Acceptable\n"
            "60-69: Weak\n"
            "40-59: Poor\n"
            "0-39: Unacceptable\n\n"

            "Dimension Definitions:\n\n"

            "Depth:\n"
            "Measures the level of detail, explanation, reasoning, evidence usage, and analytical rigor present in the response.\n\n"

            "Depth Anchors:\n"
            "90-100: Extensive explanation, multiple supporting arguments, limitations discussed, evidence used throughout.\n"
            "80-89: Detailed explanation with supporting evidence and reasoning.\n"
            "70-79: Adequate explanation but limited detail, evidence, or reasoning.\n"
            "60-69: Mostly descriptive with limited analysis or support.\n"
            "Below 60: Superficial, minimal reasoning, little supporting evidence.\n\n"

            "Citations:\n"
            "Measures how effectively citations support substantive claims.\n\n"

            "Citation Anchors:\n"
            "90-100: Most substantive claims are directly linked to citations.\n"
            "80-89: Many important claims are supported by citations.\n"
            "70-79: Citations present but inconsistently connected to claims.\n"
            "60-69: Sparse citations or weak linkage between citations and claims.\n"
            "Below 60: Few or no meaningful citations.\n\n"

            "Clarity:\n"
            "Measures organization, readability, structure, coherence, and ease of understanding.\n\n"

            "Clarity Anchors:\n"
            "90-100: Exceptionally clear, well-structured, easy to follow.\n"
            "80-89: Clear organization with only minor readability issues.\n"
            "70-79: Generally understandable but unevenly organized.\n"
            "60-69: Difficult to follow in places.\n"
            "Below 60: Poorly organized or confusing.\n\n"

            "Synthesis:\n"
            "Measures how effectively the response integrates information across the papers named in the prompt.\n\n"
            "High synthesis requires:\n"
            "- explicit cross-paper reasoning,\n"
            "- comparison of multiple papers,\n"
            "- identification of agreements,\n"
            "  disagreements,\n"
            "  tensions,\n"
            "  tradeoffs,\n"
            "  or emergent insights.\n\n"
            "Responses that discuss papers independently, or that analyze only a subset of the papers requested in the prompt, cannot receive a good synthesis score.\n\n"

            "Synthesis Anchors:\n"
            "90-100: Integrates nearly all relevant papers, develops non-obvious cross-paper insights, tradeoffs, agreements, and disagreements.\n"
            "80-89: Strong comparison across most papers with meaningful integration.\n"
            "70-79: Some comparison but limited integration.\n"
            "60-69: Mostly summarizes papers individually.\n"
            "Below 60: Little or no cross-paper synthesis.\n\n"

            "Evaluation Procedure:\n"
            "1. Identify all papers, concepts, entities, and comparison requests explicitly mentioned in the prompt.\n"
            "2. Determine which of those are actually discussed in the response.\n"
            "3. Assess depth, citations, clarity, and synthesis only after coverage is established.\n"
            "4. Do NOT reward verbosity, hedging, disclaimers, or uncertainty statements by themselves.\n"
            "5. Score only what is present in the response.\n\n"

            f"Prompt:\n{prompt_text}\n\n"
            f"Response:\n{response_text}\n\n"
            f"Quality Criteria:\n{criteria_str}\n\n"

            f"Return JSON only: {{\"scores\": {{{score_schema}}}, \"explanation\": \"<str>\"}}"
        )

        parsed = await self._llm_call_with_retry(
            session, eval_prompt, label=f"pair '{prompt_text[:40]}'", usage=usage
        )
        if parsed is None or "scores" not in parsed:
            logger.warning(f"No valid scores returned for pair: {prompt_text[:60]}")
            return None

        normalized_scores: Dict[str, int] = {}
        for dim in criteria:
            if dim in parsed.get("scores", {}):
                coerced = self._coerce_score(parsed["scores"][dim])
                if coerced is not None:
                    normalized_scores[dim] = coerced

        if not normalized_scores:
            logger.warning(f"No valid numeric scores returned for pair: {prompt_text[:60]}")
            return None

        return {"scores": normalized_scores, "explanation": parsed.get("explanation", "")}

    async def _score_holistic(
        self,
        session: aiohttp.ClientSession,
        responses: List[Dict[str, str]],
        criteria: Dict[str, str],
        usage: _UsageAccumulator,
    ) -> Tuple[Dict[str, int], str]:
        """Score the holistic criteria (completeness) over all answers."""
        responses_str = "\n\n".join(
            f"[{i+1}] Prompt: {r.get('prompt', '')}\n    Response: {r.get('response', '')}"
            for i, r in enumerate(responses)
        )
        criteria_str = "\n".join(f"- {dim}: {desc}" for dim, desc in criteria.items())
        dim_keys = list(criteria.keys())
        score_schema = ", ".join(f'"{d}": <int>' for d in dim_keys)

        eval_prompt = (
            "You are an expert evaluator.\n\n"

            "IMPORTANT:\n"
            "- You do NOT have access to the source papers.\n"
            "- Evaluate ONLY the provided responses.\n"
            "- Do NOT assume facts that are not present.\n"
            "- Do NOT speculate about missing findings.\n"
            "- Ignore factual correctness because it cannot be verified.\n"
            "- Coverage is measured against the prompt itself.\n\n"

            "Dimension Definition:\n"
            "- Completeness measures whether all major information requests in the prompt are substantively addressed.\n\n"

            "Key Rule:\n"
            "If a prompt requests comparison, alignment, conflict, tradeoffs, synthesis, or integration among multiple papers, all named papers must be substantively addressed for a high completeness score.\n\n"
            "Acknowledging missing evidence does NOT count as addressing that paper and it should reduce score.\n\n"

            "Scoring Guidelines:\n"
            "90-100: Every major component of the prompt is substantively addressed. All requested papers, concepts, entities, and comparisons are covered.\n"
            "80-89: Minor omissions only. Most requested components are substantively addressed.\n"
            "70-79: Noticeable omissions. Some requested papers, concepts, or comparisons receive limited treatment.\n"
            "60-69: Multiple important components missing or only briefly mentioned.\n"
            "40-59: Major gaps across several requested components.\n"
            "0-39: Many requested components unanswered or substantially incomplete.\n\n"

            "Evaluation Procedure:\n"
            "1. Identify all papers, concepts, entities, and comparison requests explicitly mentioned in the prompt.\n"
            "2. Determine which of those are substantively addressed in the response.\n"
            "3. Evaluate coverage against the prompt itself.\n"
            "4. If multiple papers are requested, verify that each paper receives meaningful discussion.\n"
            "5. Acknowledging unavailable evidence does not count as coverage.\n"
            "6. Base the score only on coverage.\n"
            "7. Ignore factual correctness.\n\n"

            f"Responses:\n{responses_str}\n\n"
            f"Quality Criteria:\n{criteria_str}\n\n"

            f"Return JSON only: {{\"scores\": {{{score_schema}}}, \"explanation\": \"<str>\"}}"
        )

        parsed = await self._llm_call_with_retry(session, eval_prompt, label="holistic", usage=usage)
        if parsed is None or "scores" not in parsed:
            logger.warning("No valid holistic scores returned; defaulting to 0 for holistic dimensions.")
            return {dim: 0 for dim in criteria}, ""

        scores: Dict[str, int] = {}
        for dim in criteria:
            if dim in parsed.get("scores", {}):
                coerced = self._coerce_score(parsed["scores"][dim])
                if coerced is not None:
                    scores[dim] = coerced

        return scores, parsed.get("explanation", "")

    async def _llm_call_with_retry(
        self,
        session: aiohttp.ClientSession,
        user_prompt: str,
        label: str = "",
        usage: Optional[_UsageAccumulator] = None,
    ) -> Optional[Dict]:
        """Call the judge with up to _MAX_RETRIES attempts; the parsed JSON, or None.

        Every attempt that reaches the API is charged to ``usage``, retries included.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a strict, fair, and concise evaluator. Respond with JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with session.post(self._api_url, json=payload, headers=self._headers, timeout=60) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if usage is not None:
                        usage.record(data.get("usage"))
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_llm_json(content)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(f"LLM call failed for {label} (attempt {attempt}/{_MAX_RETRIES}): {exc}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)

        logger.error(f"LLM call for {label} failed after {_MAX_RETRIES} attempts: {last_exc}")
        return None

    @staticmethod
    def _parse_llm_json(content: str) -> Optional[Dict]:
        """Extract and parse the first JSON object from LLM output."""
        match = re.search(r'\{[\s\S]*\}', content)
        if not match:
            return None
        block = match.group()
        try:
            return json.loads(block)
        except Exception:
            try:
                return ast.literal_eval(block)
            except Exception:
                return None

    @staticmethod
    def _completeness_cap(required_prompts: List[str], responses: List[Dict[str, str]]) -> int:
        """A 0-100 cap from the share of required prompts with a non-empty answer."""
        if not required_prompts:
            return 100

        required_set = {str(p).strip() for p in required_prompts if str(p).strip()}
        if not required_set:
            return 100

        answered_prompts = {
            str(item.get("prompt", "")).strip()
            for item in responses
            if str(item.get("response", "")).strip()
        }
        covered = len(required_set.intersection(answered_prompts))
        return round((covered / len(required_set)) * 100)

    @staticmethod
    def _coerce_score(value: Any) -> Optional[int]:
        """Convert a score value to an integer in [0, 100]."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        return max(0, min(100, round(numeric)))
