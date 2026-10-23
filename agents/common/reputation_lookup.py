"""Helpers for querying the ERC-8004 Reputation Registry's decimal-precision getSummary.

The v2 registry (erc-8004/erc-8004-contracts) represents feedback as fixed-point
(value, valueDecimals) pairs rather than a plain 0-100 integer, and reverts
getSummary() outright when given an empty clientAddresses array. Every caller
needs the same two-step dance: enumerate clients via getClients(), then only call
getSummary() when that list is non-empty. This module centralizes that logic so
it isn't re-implemented (and re-risked) at each call site.
"""

from typing import Tuple


def fetch_reputation_summary(contract, agent_id: int) -> Tuple[int, int, int]:
    """Fetch an agent's aggregated reputation, preserving decimal precision.

    Args:
        contract: A web3 Contract instance for the ReputationRegistry (sync .call()).
        agent_id: The agent token ID to look up.

    Returns:
        (feedback_count, reputation_value, reputation_decimals). reputation_value is
        the raw fixed-point integer; divide by 10**reputation_decimals for the real
        number (see reputation_to_float). All three are 0 when the agent has no
        feedback yet.
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
