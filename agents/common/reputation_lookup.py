"""Read an agent's reputation from the ERC-8004 Reputation Registry, with its decimal precision.

The registry stores feedback as fixed-point (value, valueDecimals) pairs and
reverts getSummary() when given an empty list of clients, so the clients are
read with getClients() first and getSummary() is called only when there are any.
"""

from typing import Tuple


def fetch_reputation_summary(contract, agent_id: int) -> Tuple[int, int, int]:
    """Return ``(feedback_count, reputation_value, reputation_decimals)`` for an agent.

    ``reputation_value`` is the raw fixed-point integer (see :func:`reputation_to_float`).
    All three are 0 when the agent has no feedback yet.
    """
    clients = contract.functions.getClients(agent_id).call()
    if not clients:
        return 0, 0, 0

    count, value, decimals = contract.functions.getSummary(agent_id, clients, "", "").call()
    count = int(count)
    if count == 0 or value < 0:
        return 0, 0, 0
    return count, int(value), int(decimals)


def reputation_to_float(value: int, decimals: int) -> float:
    """Convert a fixed-point (value, decimals) reputation pair into a plain float."""
    return float(value) / (10 ** int(decimals))
