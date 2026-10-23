"""LLM sampling settings for the live agent (S7 §A.2, defect D2).

**One source of truth, deliberately imported rather than restated.** The S5
probe measured the estimator's σ through :mod:`agents.provider.estimation.llm_client`
under a fixed seed and a per-model sampling block. If the live agent sampled
differently, it would not be running the estimator S5 validated — and the §F.2
comparison of in-market σ against S5's held-out 0.3116 would be measuring the
sampling difference as well as the market. So the roster and the seed are
imported from that module; a copy here would be free to drift.

The import direction (``agents`` → ``experiments``) is backwards for production
code and right for this one case: the live S7 agent *is* an experiment, and
fidelity to the recorded protocol outranks package hygiene. Both packages are
installed together, so the import resolves in a spawned agent process.

Sampling is **not** uniform across models, which is why a flat
``temperature = 0`` would be wrong:

- the ``gpt-5.4`` family and ``gpt-oss-20b`` are reasoning models. They do not
  accept ``temperature`` at all and take ``reasoning: {effort}`` instead.
- ``gpt-4o-mini`` takes ``temperature: 0``.

Block A spans both kinds, so the per-model block is the only correct answer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.provider.estimation.llm_client import ALL_MODELS, SAMPLING_SEED

__all__ = ["SAMPLING_SEED", "resolve_sampling", "chat_model_kwargs", "is_pinned_model"]


def _find(model_id: str):
    for model in ALL_MODELS:
        if model.model_id == model_id or model.key == model_id:
            return model
    return None


def is_pinned_model(model_id: str) -> bool:
    """Whether this model's sampling was pinned by the estimation protocol."""
    return _find(model_id) is not None


def resolve_sampling(model_id: str) -> Dict[str, Any]:
    """Return the sampling block S5 sends for ``model_id``, verbatim.

    Raises for an unpinned model. That is the point: an S7 config naming a model
    whose sampling was never recorded must fail at startup rather than invent a
    setting and quietly break the comparison to S5.
    """
    model = _find(model_id)
    if model is None:
        known = ", ".join(sorted({m.model_id for m in ALL_MODELS}))
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
    error, and either way it is not what S5 sent. Anything else in the block
    (``reasoning``) rides in ``extra_body``, which is where OpenRouter reads it.

    The overrides exist for unpinned models on non-S7 paths (the A-series react
    arms), which have no recorded sampling to reproduce.
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
