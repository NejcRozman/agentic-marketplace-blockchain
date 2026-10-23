"""S2 — the estimation probe (`docs/s2_implementation.md`).

This package answers one question and deliberately not the next one: **can a
language model read task difficulty out of a requirement's text?** It never
touches the market, never runs the simulator, and produces no economic claim.
Dataset v1 is far too small (10 requirements, 6 distinct PDF sets) for an
economic conclusion, and S2's own spec withdrew the original ρ ≥ 0.4 gate for
that reason. What S2 *does* produce is the frozen protocol S5 will execute
verbatim, and a measured σ the simulator can be parameterised with.

Module map, in the order the harness uses them:

``dataset``
    Loads dataset v1 and enforces the leakage allow-list. The model sees the
    requirement text and nothing that identifies it.
``baselines``
    B0 (tier mean), B1 (surface features), B1b (breadth). B1b is the control
    that decides whether "semantic inference" means anything.
``metrics``
    σ in the units ``experiments.simulator.estimators`` consumes, plus the
    descriptive statistics the report needs.
``prompts``
    The frozen template and response schema. Changing this file after S2 ends
    invalidates S5 — see ``docs/estimation_protocol.md``.
``llm_client``
    OpenRouter transport, the pre-specified repair rules, and the on-disk cache
    that stops a re-run from spending money twice.
``probe_v1``
    The CLI harness that ties them together.
"""
