// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import {IERC1271} from "@openzeppelin/contracts/interfaces/IERC1271.sol";
import {IIdentityRegistry} from "./interfaces/IIdentityRegistry.sol";
import {IReputationRegistry} from "./interfaces/IReputationRegistry.sol";

/**
 * @title ReverseAuction
 * @dev A sealed-bid commit-reveal reverse auction contract for AI services, integrated
 *      with ERC-8004 agent identity/reputation. Bidding has two phases per auction:
 *      Commit (bidders submit a hash binding their bid+nonce, revealing nothing) and
 *      Reveal (bidders disclose the bid+nonce; the contract verifies it against the
 *      commitment and scores it). Because revealed values are cryptographically bound
 *      to choices made during the commit phase, neither reveal timing nor reveal order
 *      carries any strategic value — this removes the last-mover/speed advantage that a
 *      continuous open-bid auction would otherwise give to faster bidders.
 * @author Agentic Marketplace
 */
contract ReverseAuction is ReentrancyGuard {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    // ============ ENUMS ============

    enum Phase {
        Commit,
        Reveal,
        Ended
    }

    // ============ STRUCTS ============

    /**
     * @dev Represents a single reverse auction for an AI service
     */
    struct Auction {
        uint256 id;                     // Unique auction identifier
        address buyer;                  // Address of the service buyer
        string serviceDescriptionCid;   // IPFS CID containing service description and requirements
        uint256 maxPrice;              // Maximum price buyer is willing to pay
        uint256 commitDuration;        // Commit phase duration in seconds
        uint256 startTime;             // When the auction started (commit phase begin)
        uint256[] eligibleAgentIds;    // Preselected agent IDs from ERC-8004 Identity Registry
        uint256 winningAgentId;        // ID of the winning agent
        uint256 winningBid;           // Winning bid amount
        bool isActive;                // Whether auction is currently active
        bool isCompleted;             // Whether service has been completed
        uint256 escrowAmount;         // Amount held in escrow
        uint256 reputationWeight;     // Weight for reputation in percent (0-100)
        uint256 revealDuration;       // Reveal phase duration in seconds, starts when commit phase ends
    }

    /**
     * @dev A sealed commitment made during the commit phase
     */
    struct Commitment {
        address provider;             // Address that made the commitment (must also reveal it)
        bytes32 commitHash;           // keccak256(abi.encode(auctionId, agentId, bidAmount, nonce, provider))
        bool revealed;                // Whether this commitment has been revealed yet
    }

    /**
     * @dev Represents a revealed bid in an auction
     */
    struct Bid {
        address provider;             // Address of the bidding provider
        uint256 agentId;             // ID of the agent used for bidding
        uint256 amount;              // Bid amount
        uint256 timestamp;           // When the bid was revealed
        uint256 reputation;          // Agent's reputation, fixed-point scaled by 10**reputationDecimals
        uint256 score;               // Calculated weighted score (0-10000 for precision)
        uint8 reputationDecimals;    // Decimal places represented by `reputation`
    }

    // ============ STATE VARIABLES ============

    /// @dev USDC token contract
    IERC20 public immutable USDC_TOKEN;

    /// @dev ERC-8004 Identity Registry for agent ownership
    IIdentityRegistry public immutable IDENTITY_REGISTRY;

    /// @dev ERC-8004 Reputation Registry for agent reputation
    IReputationRegistry public immutable REPUTATION_REGISTRY;

    /// @dev Fallback reputation used when an agent has no ratings yet (whole number, 0-100)
    uint256 public immutable INITIAL_REPUTATION_DEFAULT;

    /// @dev Counter for generating unique auction IDs (public for iteration)
    uint256 public auctionIdCounter;

    /// @dev Mapping from auction ID to auction data
    mapping(uint256 => Auction) public auctions;

    /// @dev Mapping from auction ID to array of revealed bids
    mapping(uint256 => Bid[]) public auctionBids;

    /// @dev Mapping from auction ID to agent ID to sealed commitment
    mapping(uint256 => mapping(uint256 => Commitment)) public auctionCommitments;

    /// @dev Mapping to check if an agent is eligible for a specific auction
    /// auctionId => agentId => isEligible
    mapping(uint256 => mapping(uint256 => bool)) public isEligibleAgent;

    /// @dev Mapping to track the winning bid for each auction
    mapping(uint256 => uint256) public winningBid;

    /// @dev Mapping to track the highest score for each auction
    mapping(uint256 => uint256) public highestScore;

    /// @dev Precision constant for score calculations (4 decimal places)
    uint256 private constant SCORE_PRECISION = 10_000;

    // ============ EVENTS ============

    event AuctionCreated(
        uint256 indexed auctionId,
        address indexed buyer,
        string serviceDescriptionCid,
        uint256 maxPrice,
        uint256 commitDuration,
        uint256 revealDuration,
        uint256[] eligibleAgentIds,
        uint256 reputationWeight
    );

    event BidCommitted(
        uint256 indexed auctionId,
        address indexed provider,
        uint256 indexed agentId,
        uint256 timestamp
    );

    event BidRevealed(
        uint256 indexed auctionId,
        address indexed provider,
        uint256 indexed agentId,
        uint256 bidAmount,
        uint256 reputation,
        uint8 reputationDecimals,
        uint256 score,
        uint256 timestamp
    );

    event AuctionEnded(
        uint256 indexed auctionId,
        uint256 indexed winningAgentId,
        uint256 winningBid
    );

    event ServiceCompleted(
        uint256 indexed auctionId,
        uint256 indexed agentId,
        address indexed provider
    );

    event FundsReleased(
        uint256 indexed auctionId,
        uint256 indexed agentId,
        address indexed provider,
        uint256 amount
    );

    event FeedbackAuthProvided(
        uint256 indexed auctionId,
        uint256 indexed agentId,
        address indexed buyer,
        bytes feedbackAuth
    );

    // ============ ERRORS ============

    error InvalidAuctionDuration();
    error InvalidRevealDuration();
    error InvalidMaxPrice();
    error NoEligibleAgents();
    error InsufficientEscrow();
    error AuctionNotFound();
    error AuctionNotActive();
    error AgentNotEligible();
    error NotAgentOwner();
    error AgentNotFound();
    error BidTooHigh();
    error AuctionStillActive();
    error NotAuthorized();
    error InvalidServiceCid();
    error ServiceAlreadyCompleted();
    error InvalidReputationWeight();
    error InvalidInitialReputationDefault();
    error NotInCommitPhase();
    error NotInRevealPhase();
    error AlreadyCommitted();
    error NoCommitmentFound();
    error AlreadyRevealed();
    error InvalidReveal();
    error InvalidFeedbackAuth();
    error FeedbackAuthExpired();
    error InvalidSignature();

    // ============ CONSTRUCTOR ============

    /**
     * @dev Constructor sets the USDC token and ERC-8004 registry addresses
     * @param usdcTokenAddress Address of the USDC token contract
     * @param identityRegistry Address of the ERC-8004 Identity Registry
     * @param reputationRegistry Address of the ERC-8004 Reputation Registry
     * @param initialReputationDefault Fallback reputation (0-100) for agents with no ratings
     */
    constructor(
        address usdcTokenAddress,
        address identityRegistry,
        address reputationRegistry,
        uint256 initialReputationDefault
    ) {
        if (usdcTokenAddress == address(0)) revert ();
        if (identityRegistry == address(0)) revert ();
        if (reputationRegistry == address(0)) revert ();
        if (initialReputationDefault > 100) revert InvalidInitialReputationDefault();

        USDC_TOKEN = IERC20(usdcTokenAddress);
        IDENTITY_REGISTRY = IIdentityRegistry(identityRegistry);
        REPUTATION_REGISTRY = IReputationRegistry(reputationRegistry);
        INITIAL_REPUTATION_DEFAULT = initialReputationDefault;
        auctionIdCounter = 0;
    }

    // ============ EXTERNAL FUNCTIONS ============

    /**
     * @dev Creates a new reverse auction for an AI service
     * @param serviceDescriptionCid IPFS CID containing service description and requirements
     * @param maxPrice Maximum price buyer is willing to pay (also escrow amount) in USDC
     * @param commitDuration Commit phase duration in seconds
     * @param revealDuration Reveal phase duration in seconds (starts when the commit phase ends)
     * @param eligibleAgentIds Array of preselected agent IDs from ERC-8004 Identity Registry
     * @param reputationWeight Weight for reputation in scoring, percent (0-100)
     * @return auctionId The ID of the created auction
     */
    function createAuction(
        string calldata serviceDescriptionCid,
        uint256 maxPrice,
        uint256 commitDuration,
        uint256 revealDuration,
        uint256[] calldata eligibleAgentIds,
        uint256 reputationWeight
    ) external nonReentrant returns (uint256 auctionId) {
        // Validation
        if (commitDuration == 0) revert InvalidAuctionDuration();
        if (revealDuration == 0) revert InvalidRevealDuration();
        if (maxPrice == 0) revert InvalidMaxPrice();
        if (eligibleAgentIds.length == 0) revert NoEligibleAgents();
        if (bytes(serviceDescriptionCid).length == 0) revert InvalidServiceCid();
        if (reputationWeight > 100) revert InvalidReputationWeight();

        // Validate all agents exist in Identity Registry
        for (uint256 i = 0; i < eligibleAgentIds.length; i++) {
            if (!_agentExists(eligibleAgentIds[i])) revert AgentNotFound();
        }

        // Check USDC allowance and balance
        if (USDC_TOKEN.allowance(msg.sender, address(this)) < maxPrice) revert InsufficientEscrow();
        if (USDC_TOKEN.balanceOf(msg.sender) < maxPrice) revert InsufficientEscrow();

        // Generate auction ID
        auctionId = ++auctionIdCounter;

        // Transfer USDC to escrow
        USDC_TOKEN.safeTransferFrom(msg.sender, address(this), maxPrice);

        // Create auction
        Auction storage auction = auctions[auctionId];
        auction.id = auctionId;
        auction.buyer = msg.sender;
        auction.serviceDescriptionCid = serviceDescriptionCid;
        auction.maxPrice = maxPrice;
        auction.commitDuration = commitDuration;
        auction.startTime = block.timestamp;
        auction.eligibleAgentIds = eligibleAgentIds;
        auction.winningAgentId = 0;
        auction.winningBid = 0;
        auction.isActive = true;
        auction.isCompleted = false;
        auction.escrowAmount = maxPrice;
        auction.reputationWeight = reputationWeight;
        auction.revealDuration = revealDuration;

        // Set up agent eligibility mapping for O(1) lookup
        for (uint256 i = 0; i < eligibleAgentIds.length; i++) {
            isEligibleAgent[auctionId][eligibleAgentIds[i]] = true;
        }

        // Initialize winning bid to max price
        winningBid[auctionId] = maxPrice;

        // Initialize highest score to 0
        highestScore[auctionId] = 0;

        emit AuctionCreated(
            auctionId,
            msg.sender,
            serviceDescriptionCid,
            maxPrice,
            commitDuration,
            revealDuration,
            eligibleAgentIds,
            reputationWeight
        );
    }

    /**
     * @dev Submits a sealed bid commitment during the commit phase
     * @param auctionId The auction ID to bid on
     * @param agentId The agent ID to bid with
     * @param commitHash keccak256(abi.encode(auctionId, agentId, bidAmount, nonce, msg.sender)) —
     *        see computeCommitHash() for a convenience helper that matches exactly.
     * @notice Exactly one commitment is allowed per (auction, agent); it cannot be changed
     *         or overwritten once submitted.
     */
    function commitBid(
        uint256 auctionId,
        uint256 agentId,
        bytes32 commitHash
    ) external {
        Auction storage auction = auctions[auctionId];

        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) revert AuctionNotActive();
        if (block.timestamp > auction.startTime + auction.commitDuration) revert NotInCommitPhase();
        if (!isEligibleAgent[auctionId][agentId]) revert AgentNotEligible();

        // Check caller owns the agent
        if (IDENTITY_REGISTRY.ownerOf(agentId) != msg.sender) revert NotAgentOwner();

        if (auctionCommitments[auctionId][agentId].commitHash != bytes32(0)) revert AlreadyCommitted();

        auctionCommitments[auctionId][agentId] = Commitment({
            provider: msg.sender,
            commitHash: commitHash,
            revealed: false
        });

        emit BidCommitted(auctionId, msg.sender, agentId, block.timestamp);
    }

    /**
     * @dev Reveals a previously committed bid during the reveal phase
     * @param auctionId The auction ID
     * @param agentId The agent ID the commitment was made under
     * @param bidAmount The actual bid amount that was committed to
     * @param nonce The nonce used at commit time
     * @notice A valid reveal of a prior commitment ALWAYS succeeds and is recorded, even if its
     *         score is not competitive — the bidder is already bound to this value by their
     *         commitment, so there is nothing left to protect by reverting. Only the auction's
     *         tracked winner is updated, and only if this bid's score is strictly better.
     */
    function revealBid(
        uint256 auctionId,
        uint256 agentId,
        uint256 bidAmount,
        bytes32 nonce
    ) external {
        Auction storage auction = auctions[auctionId];

        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) revert AuctionNotActive();

        uint256 commitDeadline = auction.startTime + auction.commitDuration;
        uint256 revealDeadline = commitDeadline + auction.revealDuration;
        if (block.timestamp <= commitDeadline || block.timestamp > revealDeadline) revert NotInRevealPhase();

        Commitment storage commitment = auctionCommitments[auctionId][agentId];
        if (commitment.commitHash == bytes32(0)) revert NoCommitmentFound();
        if (commitment.provider != msg.sender) revert NotAgentOwner();
        if (commitment.revealed) revert AlreadyRevealed();

        bytes32 expectedHash = keccak256(abi.encode(auctionId, agentId, bidAmount, nonce, msg.sender));
        if (expectedHash != commitment.commitHash) revert InvalidReveal();
        if (bidAmount > auction.maxPrice) revert BidTooHigh();

        commitment.revealed = true;

        // Fetch reputation from ERC-8004 Reputation Registry (decimal-precision)
        (uint256 reputation, uint8 reputationDecimals) = _fetchReputation(agentId);

        // Calculate weighted score for this bid
        uint256 score = _calculateScore(reputation, reputationDecimals, bidAmount, auction.maxPrice, auction.reputationWeight);

        // Record the revealed bid regardless of competitiveness
        Bid memory newBid = Bid({
            provider: msg.sender,
            agentId: agentId,
            amount: bidAmount,
            timestamp: block.timestamp,
            reputation: reputation,
            score: score,
            reputationDecimals: reputationDecimals
        });

        auctionBids[auctionId].push(newBid);

        // Update the tracked winner only if this bid strictly improves on the current best
        if (auction.winningAgentId == 0 || score > highestScore[auctionId]) {
            winningBid[auctionId] = bidAmount;
            highestScore[auctionId] = score;
            auction.winningAgentId = agentId;
            auction.winningBid = bidAmount;
        }

        emit BidRevealed(auctionId, msg.sender, agentId, bidAmount, reputation, reputationDecimals, score, block.timestamp);
    }

    /**
     * @dev Computes the commit hash for a given bid so off-chain callers (and tests) compute
     *      exactly the value the contract expects, without duplicating the encoding scheme.
     */
    function computeCommitHash(
        uint256 auctionId,
        uint256 agentId,
        uint256 bidAmount,
        bytes32 nonce,
        address sender
    ) external pure returns (bytes32) {
        return keccak256(abi.encode(auctionId, agentId, bidAmount, nonce, sender));
    }

    /**
     * @dev Ends an auction and finalizes the winner
     * @param auctionId The auction ID to end
     * @dev Can be called by anyone once the reveal phase has expired, or by buyer at any time
     */
    function endAuction(uint256 auctionId) external {
        Auction storage auction = auctions[auctionId];

        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) revert AuctionNotActive();

        // Check who can end the auction
        uint256 revealDeadline = auction.startTime + auction.commitDuration + auction.revealDuration;
        bool timeExpired = block.timestamp > revealDeadline;
        bool isBuyer = msg.sender == auction.buyer;

        if (!timeExpired && !isBuyer) revert NotAuthorized();

        // Deactivate the auction
        auction.isActive = false;

        // If there are revealed bids, the current winner is already set
        // If no bids were revealed, winningAgentId remains 0
        uint256 winningAgent = auction.winningAgentId;
        uint256 winningAmount = auction.winningBid;

        emit AuctionEnded(auctionId, winningAgent, winningAmount);
    }

    /**
     * @dev Marks a service as completed and releases payment to the winner
     * @param auctionId The auction ID
     * @param feedbackAuth Signed authorization for buyer to give feedback (per ERC-8004)
     * @dev Can only be called by the service provider (owner of winning agent)
     *
     * The feedbackAuth must be in ERC-8004 format:
     * - First 224 bytes: abi.encode(agentId, clientAddress, indexLimit, expiry, chainId, identityRegistry, signerAddress)
     * - Last 65 bytes: signature
     *
     * This allows the buyer to later submit feedback to the ERC-8004 Reputation Registry.
     */
    function completeService(uint256 auctionId, bytes calldata feedbackAuth) external nonReentrant {
        Auction storage auction = auctions[auctionId];

        // Basic validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (auction.isActive) revert AuctionStillActive();
        if (auction.isCompleted) revert ServiceAlreadyCompleted();
        if (auction.winningAgentId == 0) revert NotAuthorized();

        // Check caller is agent owner
        address currentOwner = IDENTITY_REGISTRY.ownerOf(auction.winningAgentId);
        if (msg.sender != currentOwner) revert NotAgentOwner();

        // Verify feedbackAuth is valid and properly signed
        _verifyFeedbackAuth(
            feedbackAuth,
            auction.winningAgentId,
            auction.buyer,
            currentOwner
        );

        // Release payment
        auction.isCompleted = true;

        uint256 winningBid = auction.winningBid;
        uint256 escrowAmount = auction.escrowAmount;
        uint256 refundAmount = escrowAmount - winningBid;

        auction.escrowAmount = 0;

        // Transfer winning bid to provider
        USDC_TOKEN.safeTransfer(currentOwner, winningBid);

        // Refund excess to buyer
        if (refundAmount > 0) {
            USDC_TOKEN.safeTransfer(auction.buyer, refundAmount);
        }

        // ===== EMIT EVENTS =====
        emit ServiceCompleted(auctionId, auction.winningAgentId, currentOwner);
        emit FundsReleased(auctionId, auction.winningAgentId, currentOwner, winningBid);

        // Emit feedbackAuth for buyer to retrieve from event logs
        // Buyer can later use this to submit feedback to ERC-8004 Reputation Registry
        emit FeedbackAuthProvided(
            auctionId,
            auction.winningAgentId,
            auction.buyer,
            feedbackAuth
        );
    }

    /**
     * @dev Allows buyer to get refund if auction ended with no bids
     * @param auctionId The auction ID
     */
    function refundBuyer(uint256 auctionId) external nonReentrant {
        Auction storage auction = auctions[auctionId];

        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (msg.sender != auction.buyer) revert NotAuthorized();
        if (auction.isActive) revert AuctionStillActive();
        if (auction.isCompleted) revert ServiceAlreadyCompleted();
        if (auction.winningAgentId != 0) revert NotAuthorized(); // Has winner
        if (auction.escrowAmount == 0) revert InsufficientEscrow(); // Already refunded

        uint256 refundAmount = auction.escrowAmount;
        auction.escrowAmount = 0;
        auction.isCompleted = true; // Mark as completed to prevent re-entry

        // Transfer refund to buyer
        USDC_TOKEN.safeTransfer(auction.buyer, refundAmount);

        emit FundsReleased(auctionId, 0, auction.buyer, refundAmount);
    }

    // ============ INTERNAL FUNCTIONS ============

    /**
     * @dev Checks if an agent exists in the Identity Registry
     * @param agentId The agent token ID to check
     * @return exists True if the agent has been minted and exists
     */
    function _agentExists(uint256 agentId) internal view returns (bool exists) {
        try IDENTITY_REGISTRY.ownerOf(agentId) returns (address owner) {
            return owner != address(0);
        } catch {
            return false;
        }
    }

    /**
     * @dev Fetches an agent's reputation from the ERC-8004 Reputation Registry, preserving
     *      decimal precision, and falls back to INITIAL_REPUTATION_DEFAULT (decimals = 0)
     *      when the agent has no feedback yet.
     * @param agentId The agent token ID to look up
     * @return reputation Fixed-point reputation value, scaled by 10**reputationDecimals
     * @return reputationDecimals Decimal places represented by `reputation`
     * @notice getSummary() reverts on an empty clientAddresses array, so getClients() is
     *         consulted first and getSummary() is only called when it returns non-empty.
     */
    function _fetchReputation(uint256 agentId) internal view returns (uint256 reputation, uint8 reputationDecimals) {
        address[] memory clients = REPUTATION_REGISTRY.getClients(agentId);
        if (clients.length == 0) {
            return (INITIAL_REPUTATION_DEFAULT, 0);
        }

        (uint64 count, int128 summaryValue, uint8 summaryValueDecimals) =
            REPUTATION_REGISTRY.getSummary(agentId, clients, "", "");

        if (count == 0 || summaryValue < 0) {
            return (INITIAL_REPUTATION_DEFAULT, 0);
        }

        return (uint256(uint128(summaryValue)), summaryValueDecimals);
    }

    /**
     * @dev Calculates the weighted score for a bid
     * @param reputation Provider's reputation, fixed-point scaled by 10**reputationDecimals
     * @param reputationDecimals Decimal places represented by `reputation`
     * @param bidAmount The bid amount
     * @param maxPrice Maximum price from auction
    * @param reputationWeight Weight for reputation in percent (0-100)
     * @return score The calculated weighted score (0-10000)
     *
    * Formula: score = w * normalize(reputation) + (1 - w) * normalize(1 / bidAmount)
     * where:
    *   - w = reputationWeight / 100
     *   - normalize(reputation) = reputation / 100  (reputation on a 0-100 scale)
     *   - normalize(1 / bidAmount) = 1 - (bidAmount / maxPrice)
     *
    * Higher score is better
     */
    function _calculateScore(
        uint256 reputation,
        uint8 reputationDecimals,
        uint256 bidAmount,
        uint256 maxPrice,
        uint256 reputationWeight
    ) internal pure returns (uint256 score) {
        // Keep reputationWeight external semantics as percent (0-100).
        // Reputation is on a 0-100 scale at `reputationDecimals` decimal places; the full
        // scale (100) is therefore 100 * 10**reputationDecimals in raw units.
        uint256 reputationScale = 100 * (10 ** uint256(reputationDecimals));
        uint256 normalizedReputation = (reputation * SCORE_PRECISION) / reputationScale;
        uint256 weightPercent = reputationWeight;

        // Normalize bid amount (inverted so lower price = higher score)
        // Formula: (1 - bidAmount/maxPrice) scaled to SCORE_PRECISION.
        uint256 normalizedBidScore = ((maxPrice - bidAmount) * SCORE_PRECISION) / maxPrice;

        // Calculate weighted score
        // score = w * normalizedReputation + (1 - w) * normalizedBidScore
        // where w is percent in [0, 100]. Result remains in [0, SCORE_PRECISION].
        score = (weightPercent * normalizedReputation +
            (100 - weightPercent) * normalizedBidScore) / 100;

        return score;
    }

    /**
     * @dev Verifies feedbackAuth signature and parameters (ERC-8004 compatible)
     * @param feedbackAuth The encoded feedbackAuth from provider
     * @param expectedAgentId The agent ID that should be in feedbackAuth
     * @param expectedClient The buyer address that should be in feedbackAuth
     * @param currentOwner The current owner of the agent (for authorization check)
     */
    function _verifyFeedbackAuth(
        bytes calldata feedbackAuth,
        uint256 expectedAgentId,
        address expectedClient,
        address currentOwner
    ) internal view {
        // ERC-8004 format: [224 bytes params][65 bytes signature]
        if (feedbackAuth.length < 289) revert InvalidFeedbackAuth();

        // Decode first 224 bytes
        (
            uint256 authAgentId,
            address authClient,
            uint64 authIndexLimit,
            uint256 authExpiry,
            uint256 authChainId,
            address authIdentityRegistry,
            address authSigner
        ) = abi.decode(feedbackAuth[:224], (uint256, address, uint64, uint256, uint256, address, address));

        // Extract signature (last 65 bytes)
        bytes memory signature = feedbackAuth[224:];

        // Verify tuple fields match expected context
        if (authAgentId != expectedAgentId) revert InvalidFeedbackAuth();
        if (authClient != expectedClient) revert InvalidFeedbackAuth();
        if (authExpiry <= block.timestamp) revert FeedbackAuthExpired();
        if (authChainId != block.chainid) revert InvalidFeedbackAuth();
        if (authIdentityRegistry != address(IDENTITY_REGISTRY)) revert InvalidFeedbackAuth();

        // Reconstruct message hash and apply EIP-191 prefix (matching official ERC-8004)
        bytes32 messageHash = MessageHashUtils.toEthSignedMessageHash(
            keccak256(
                abi.encode(
                    authAgentId,
                    authClient,
                    authIndexLimit,
                    authExpiry,
                    authChainId,
                    authIdentityRegistry,
                    authSigner
                )
            )
        );

        // Verify signature (supports both EOA and ERC-1271 smart contract wallets)
        address recoveredSigner = ECDSA.recover(messageHash, signature);
        if (recoveredSigner != authSigner) {
            // If not EOA signature, try ERC-1271 for smart contract wallets
            if (authSigner.code.length > 0) {
                // Try ERC-1271 verification
                try IERC1271(authSigner).isValidSignature(messageHash, signature) returns (bytes4 magicValue) {
                    if (magicValue != IERC1271.isValidSignature.selector) {
                        revert InvalidSignature();
                    }
                } catch {
                    revert InvalidSignature();
                }
            } else {
                revert InvalidSignature();
            }
        }

        // Verify signer is authorized (owner or approved operator)
        if (authSigner != currentOwner) {
            // Check if signer is an approved operator
            bool isApproved = IDENTITY_REGISTRY.isApprovedForAll(currentOwner, authSigner) ||
                             IDENTITY_REGISTRY.getApproved(expectedAgentId) == authSigner;
            if (!isApproved) revert NotAgentOwner();
        }
    }

    // ============ VIEW FUNCTIONS ============

    /**
     * @dev Returns the deadline (timestamp) after which the commit phase has ended
     */
    function getCommitDeadline(uint256 auctionId) external view returns (uint256 deadline) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        return auction.startTime + auction.commitDuration;
    }

    /**
     * @dev Returns the deadline (timestamp) after which the reveal phase has ended
     */
    function getRevealDeadline(uint256 auctionId) external view returns (uint256 deadline) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        return auction.startTime + auction.commitDuration + auction.revealDuration;
    }

    /**
     * @dev Returns the current phase of the auction (Commit, Reveal, or Ended)
     */
    function getAuctionPhase(uint256 auctionId) external view returns (Phase phase) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) return Phase.Ended;

        uint256 commitDeadline = auction.startTime + auction.commitDuration;
        uint256 revealDeadline = commitDeadline + auction.revealDuration;
        if (block.timestamp <= commitDeadline) return Phase.Commit;
        if (block.timestamp <= revealDeadline) return Phase.Reveal;
        return Phase.Ended;
    }

    /**
     * @dev Returns the sealed commitment for an agent in an auction (zero values if none)
     */
    function getCommitment(uint256 auctionId, uint256 agentId) external view returns (Commitment memory commitment) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return auctionCommitments[auctionId][agentId];
    }

    /**
     * @dev Returns the end time of an auction (end of the reveal phase)
     * @param auctionId The auction ID
     * @return endTime When the auction fully ends (startTime + commitDuration + revealDuration)
     */
    function getAuctionEndTime(uint256 auctionId) external view returns (uint256 endTime) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        return auction.startTime + auction.commitDuration + auction.revealDuration;
    }

    /**
     * @dev Checks if an auction is currently active (in either the commit or reveal phase)
     * @param auctionId The auction ID
     * @return active True if auction is active and not expired
     */
    function isAuctionActive(uint256 auctionId) external view returns (bool active) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();

        return auction.isActive &&
            (block.timestamp <= auction.startTime + auction.commitDuration + auction.revealDuration);
    }

    /**
     * @dev Returns the current winning bid for an auction
     * @param auctionId The auction ID
     * @return currentWinningBid The current winning bid amount
     */
    function getCurrentWinningBid(uint256 auctionId) external view returns (uint256 currentWinningBid) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return winningBid[auctionId];
    }

    /**
     * @dev Returns the current highest score for an auction
     * @param auctionId The auction ID
     * @return currentHighestScore The current highest weighted score
     */
    function getCurrentHighestScore(uint256 auctionId) external view returns (uint256 currentHighestScore) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return highestScore[auctionId];
    }

    /**
     * @dev Returns all revealed bids for a specific auction
     * @param auctionId The auction ID
     * @return bids Array of all bids revealed on the auction
     */
    function getAuctionBids(uint256 auctionId) external view returns (Bid[] memory bids) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return auctionBids[auctionId];
    }

    /**
     * @dev Returns the number of bids revealed on an auction
     * @param auctionId The auction ID
     * @return bidCount Total number of revealed bids
     */
    function getBidCount(uint256 auctionId) external view returns (uint256 bidCount) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return auctionBids[auctionId].length;
    }

    /**
     * @dev Returns the time remaining in an auction (until the end of the reveal phase)
     * @param auctionId The auction ID
     * @return timeRemaining Seconds remaining (0 if expired)
     */
    function getTimeRemaining(uint256 auctionId) external view returns (uint256 timeRemaining) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();

        uint256 endTime = auction.startTime + auction.commitDuration + auction.revealDuration;
        if (block.timestamp >= endTime) {
            return 0;
        }
        return endTime - block.timestamp;
    }

    /**
     * @dev Returns complete auction information
     * @param auctionId The auction ID
     * @return auction The complete auction struct
     */
    function getAuctionDetails(uint256 auctionId) external view returns (Auction memory auction) {
        auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        return auction;
    }
}
