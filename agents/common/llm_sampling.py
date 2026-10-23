"""LLM sampling settings for the agents' LLM calls.

The models and their sampling blocks come from
:mod:`agents.provider.estimation.llm_client`, so price-setters and cost
estimators sample every model the same way. Sampling is not uniform across
models: ``gpt-5.4-mini`` and ``gpt-oss-20b`` are reasoning models, which take
``reasoning: {effort}`` and no ``temperature``; ``gpt-4o-mini`` takes
``temperature: 0``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.provider.estimation.llm_client import MODELS, SAMPLING_SEED

__all__ = ["SAMPLING_SEED", "resolve_sampling", "chat_model_kwargs", "is_pinned_model"]


def _find(model_id: str):
    for model in MODELS:
        if model.model_id == model_id or model.key == model_id:
            return model
    return None


def is_pinned_model(model_id: str) -> bool:
    """Whether this model has a pinned sampling block."""
    return _find(model_id) is not None


def resolve_sampling(model_id: str) -> Dict[str, Any]:
    """Return the sampling block pinned for ``model_id``, verbatim.

    Raises for an unpinned model, so a config naming a model without a pinned
    block fails at startup rather than inventing a setting.
    """
    model = _find(model_id)
    if model is None:
        known = ", ".join(sorted({m.model_id for m in MODELS}))
        raise KeyError(
            f"no pinned sampling for model {model_id!r}; the estimation protocol "
            f"covers: {known}"
        )
    return dict(model.sampling)


def chat_model_kwargs(
    model_id: str,
    temperature_override: Optional[float] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Translate the sampling block into ``ChatOpenAI`` constructor kwargs.

    ``temperature`` is passed only for models that take one — sending
    ``temperature=0`` to a reasoning model is at best ignored and at worst an
    error. Anything else in the block (``reasoning``) rides in ``extra_body``,
    which is where OpenRouter reads it. The overrides are for unpinned models.
    """
    sampling = resolve_sampling(model_id) if is_pinned_model(model_id) else {}

    kwargs: Dict[str, Any] = {}
    seed = seed_override if seed_override is not None else sampling.get("seed", SAMPLING_SEED)
    kwargs["seed"] = int(seed)

    if "temperature" in sampling:
        kwargs["temperature"] = float(sampling["temperature"])
    elif not sampling:
        # Unpinned model: fall back to the caller's configured temperature.
        kwargs["temperature"] = float(temperature_override or 0.0)

    extra_body = {k: v for k, v in sampling.items() if k not in {"seed", "temperature"}}
    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs
