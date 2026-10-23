// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IReputationRegistry
 * @dev Interface for the ERC-8004 Reputation Registry (v2 — decimal-precision feedback)
 * @notice Mirrors the deployed ReputationRegistryUpgradeable contract from
 *         erc-8004/erc-8004-contracts (verified against contracts/ReputationRegistryUpgradeable.sol
 *         on the `master` branch). Feedback values are arbitrary-precision fixed-point
 *         (value, valueDecimals) pairs rather than plain 0-100 integers.
 */
interface IReputationRegistry {
    /**
     * @notice Submit feedback for an agent
     * @param agentId The agent receiving feedback
     * @param value Fixed-point feedback value; combine with valueDecimals to get the real number
     * @param valueDecimals Number of decimal places represented by `value` (0-18)
     * @param tag1 First categorization tag (free-form string; empty string = no tag)
     * @param tag2 Second categorization tag (free-form string; empty string = no tag)
     * @param endpoint Optional endpoint identifier the feedback pertains to
     * @param feedbackURI Optional URI with detailed feedback content
     * @param feedbackHash Optional hash of off-chain feedback content
     */
    function giveFeedback(
        uint256 agentId,
        int128 value,
        uint8 valueDecimals,
        string calldata tag1,
        string calldata tag2,
        string calldata endpoint,
        string calldata feedbackURI,
        bytes32 feedbackHash
    ) external;

    /**
     * @notice Get the aggregated feedback summary for an agent
     * @param agentId The agent to summarize
     * @param clientAddresses Clients whose feedback to include. MUST be non-empty —
     *        the registry reverts on an empty array, so callers must enumerate via
     *        getClients() first and skip calling getSummary entirely when that list is empty.
     * @param tag1 Filter by first tag (empty string = no filter)
     * @param tag2 Filter by second tag (empty string = no filter)
     * @return count Number of feedback entries included in the average
     * @return summaryValue Average feedback value, fixed-point (combine with summaryValueDecimals)
     * @return summaryValueDecimals Number of decimal places represented by summaryValue
     */
    function getSummary(
        uint256 agentId,
        address[] calldata clientAddresses,
        string calldata tag1,
        string calldata tag2
    ) external view returns (uint64 count, int128 summaryValue, uint8 summaryValueDecimals);

    /**
     * @notice List all clients who have ever given feedback for an agent
     * @dev Required because getSummary() reverts when given an empty clientAddresses array.
     */
    function getClients(uint256 agentId) external view returns (address[] memory);

    function getIdentityRegistry() external view returns (address identityRegistry);
}
