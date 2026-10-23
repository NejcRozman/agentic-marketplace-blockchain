"""Per-agent RNG derivation (S7 §A.1, defects D1/D3/D4/D5/D7/D9).

Every controllable draw on the live agent path must come from an RNG instance
owned by the drawing agent, never from the ``random`` module's global generator.

A global ``random.seed()`` is **not** an acceptable substitute. The agents are
async and interleave at ``await`` points, so a global seed fixes the *sequence*
of draws but not which agent receives which draw — the appearance of control
without the substance.

One seed per run, derived per agent::

    run_seed = 1007
    provider 4427 -> random.Random(1007 + 4427)
    provider 4428 -> random.Random(1007 + 4428)
    consumer 1    -> random.Random(1007 + 1)

The derivation mirrors the one already used for the dataset sampler in
``provider_agent/orchestrator.py`` so an agent's policy draws and its dataset
draws move together under the same run seed.
"""

from __future__ import annotations

import random
from typing import Optional


def derive_agent_seed(run_seed: Optional[int], agent_id: Optional[int]) -> Optional[int]:
    """Return the per-agent seed, or ``None`` when the run is unseeded.

    Malformed values degrade to the run seed alone rather than raising: a bad
    agent id should not take down a run, and the run stays reproducible.
    """
    if run_seed is None:
        return None
    try:
        base = int(run_seed)
    except (TypeError, ValueError):
        return None
    try:
        return base + int(agent_id or 0)
    except (TypeError, ValueError):
        return base


def agent_rng(run_seed: Optional[int], agent_id: Optional[int]) -> random.Random:
    """Return this agent's private RNG.

    Unseeded runs still get a private instance, not the global module, so the
    ``test_s7_no_global_random`` scan holds regardless of configuration.
    """
    seed = derive_agent_seed(run_seed, agent_id)
    return random.Random(seed) if seed is not None else random.Random()
