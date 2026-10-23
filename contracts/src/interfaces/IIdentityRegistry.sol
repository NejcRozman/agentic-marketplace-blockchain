// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC721} from "@openzeppelin/contracts/token/ERC721/IERC721.sol";

/**
 * @title IIdentityRegistry
 * @dev Interface for ERC-8004 Identity Registry
 * @notice This is the core registry that manages agent identities as ERC-721 NFTs
 */
interface IIdentityRegistry is IERC721 {
    /// @notice Metadata entry for on-chain agent data
    struct MetadataEntry {
        string key;
        bytes value;
    }

    /// @notice Emitted when a new agent is registered
    event Registered(uint256 indexed agentId, string tokenURI, address indexed owner);

    /// @notice Emitted when metadata is set for an agent
    event MetadataSet(uint256 indexed agentId, string indexed indexedKey, string key, bytes value);

    /**
     * @notice Register a new agent with tokenURI and optional metadata
     * @param tokenURI The URI pointing to the agent registration file (e.g., ipfs:// or https://)
     * @param metadata Array of metadata entries to set on-chain
     * @return agentId The newly minted agent token ID
     */
    function register(string calldata tokenURI, MetadataEntry[] calldata metadata) external returns (uint256 agentId);

    /**
     * @notice Register a new agent with tokenURI only
     * @param tokenURI The URI pointing to the agent registration file
     * @return agentId The newly minted agent token ID
     */
    function register(string calldata tokenURI) external returns (uint256 agentId);

    /**
     * @notice Register a new agent without tokenURI (can be set later)
     * @return agentId The newly minted agent token ID
     */
    function register() external returns (uint256 agentId);

    /**
     * @dev Returns the owner of the agent NFT
     * @param agentId The agent token ID
     * @return owner The address that owns this agent
     * @notice Inherited from IERC721, but explicitly documented for clarity
     */
    function ownerOf(uint256 agentId) external view returns (address owner);
    
    /**
     * @dev Checks if an agent exists in the registry
     * @param agentId The agent token ID to check
     * @return exists True if the agent has been minted
     * @notice Can be implemented by checking if ownerOf doesn't revert or returns non-zero
     */
    function exists(uint256 agentId) external view returns (bool exists);

    /**
     * @notice Get on-chain metadata for an agent
     * @param agentId The agent token ID
     * @param key The metadata key to retrieve
     * @return value The metadata value as bytes
     */
    function getMetadata(uint256 agentId, string calldata key) external view returns (bytes memory value);

    /**
     * @notice Set on-chain metadata for an agent (only callable by owner or operator)
     * @param agentId The agent token ID
     * @param key The metadata key
     * @param value The metadata value as bytes
     */
    function setMetadata(uint256 agentId, string calldata key, bytes calldata value) external;
}
