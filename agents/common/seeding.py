"""Per-agent random number generators, derived from the run seed.

Every random draw of an agent comes from an RNG instance it owns, never from the
``random`` module's global generator: the agents are async and interleave at
``await`` points, so a global seed would fix the sequence of draws but not which
agent receives which draw.

One seed per run, derived per agent::

    run_seed = 1007
    provider 4427 -> random.Random(1007 + 4427)
    consumer 1    -> random.Random(1007 + 1)

The dataset sampler in ``provider/orchestrator.py`` uses the same derivation.
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
    """Return this agent's private RNG; unseeded runs still get a private instance."""
    seed = derive_agent_seed(run_seed, agent_id)
    return random.Random(seed) if seed is not None else random.Random()
