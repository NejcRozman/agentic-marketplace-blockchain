// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console2 as console} from "forge-std/Test.sol";
import {ReverseAuction} from "../src/ReverseAuction.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {IIdentityRegistry} from "../src/interfaces/IIdentityRegistry.sol";
import {IReputationRegistry} from "../src/interfaces/IReputationRegistry.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

contract MockIdentityRegistry is IIdentityRegistry {
    mapping(uint256 => address) private _owners;
    mapping(uint256 => mapping(string => bytes)) private _metadata;
    uint256 private _tokenIdCounter;

    function register() external returns (uint256) {
        uint256 tokenId = ++_tokenIdCounter;
        _owners[tokenId] = msg.sender;
        return tokenId;
    }

    function register(string calldata) external returns (uint256) {
        uint256 tokenId = ++_tokenIdCounter;
        _owners[tokenId] = msg.sender;
        return tokenId;
    }

    function register(string calldata, MetadataEntry[] calldata) external returns (uint256) {
        uint256 tokenId = ++_tokenIdCounter;
        _owners[tokenId] = msg.sender;
        return tokenId;
    }

    function ownerOf(uint256 tokenId) external view override returns (address) {
        address owner = _owners[tokenId];
        require(owner != address(0), "Token does not exist");
        return owner;
    }

    function exists(uint256 tokenId) external view override returns (bool) {
        return _owners[tokenId] != address(0);
    }

    function setOwner(uint256 tokenId, address newOwner) external {
        _owners[tokenId] = newOwner;
    }

    function getMetadata(uint256, string calldata) external pure returns (bytes memory) {
        return "";
    }

    function setMetadata(uint256 agentId, string calldata key, bytes calldata value) external {
        _metadata[agentId][key] = value;
    }

    // ERC721 required functions (minimal implementation)
    function balanceOf(address) external pure returns (uint256) {
        return 0;
    }

    function safeTransferFrom(address, address, uint256) external pure {
        revert("Not implemented");
    }

    function safeTransferFrom(address, address, uint256, bytes memory) external pure {
        revert("Not implemented");
    }

    function transferFrom(address, address, uint256) external pure {
        revert("Not implemented");
    }

    function approve(address, uint256) external pure {
        revert("Not implemented");
    }

    function setApprovalForAll(address, bool) external pure {
        revert("Not implemented");
    }

    function getApproved(uint256) external pure returns (address) {
        return address(0);
    }

    function isApprovedForAll(address, address) external pure returns (bool) {
        return false;
    }

    function supportsInterface(bytes4) external pure returns (bool) {
        return true;
    }
}

/// @dev Mirrors the real ERC-8004 v2 registry's two key behaviors that
///      ReverseAuction._fetchReputation() depends on: getSummary() reverts on an empty
///      clientAddresses array, and feedback is a decimal-precision (value, decimals) pair.
contract MockReputationRegistry is IReputationRegistry {
    mapping(uint256 => address[]) private _clients;
    mapping(uint256 => uint64) private _counts;
    mapping(uint256 => int128) private _values;
    mapping(uint256 => uint8) private _decimals;
    address private _identityRegistry;

    constructor(address identityRegistry_) {
        _identityRegistry = identityRegistry_;
    }

    /// @dev Configures an agent's aggregated reputation. A synthetic client is recorded so
    ///      getClients() returns non-empty, exactly as the real registry would once feedback exists.
    function setReputation(uint256 agentId, uint64 count, int128 value, uint8 decimals) external {
        if (_clients[agentId].length == 0) {
            _clients[agentId].push(address(0xCAFE));
        }
        _counts[agentId] = count;
        _values[agentId] = value;
        _decimals[agentId] = decimals;
    }

    function giveFeedback(
        uint256,
        int128,
        uint8,
        string calldata,
        string calldata,
        string calldata,
        string calldata,
        bytes32
    ) external override {
        // Mock implementation - does nothing
    }

    function getSummary(
        uint256 agentId,
        address[] calldata clientAddresses,
        string calldata,
        string calldata
    ) external view override returns (uint64 count, int128 summaryValue, uint8 summaryValueDecimals) {
        require(clientAddresses.length > 0, "MockReputationRegistry: empty clientAddresses");
        return (_counts[agentId], _values[agentId], _decimals[agentId]);
    }

    function getClients(uint256 agentId) external view override returns (address[] memory) {
        return _clients[agentId];
    }

    function getIdentityRegistry() external view override returns (address) {
        return _identityRegistry;
    }
}

contract ReverseAuctionTest is Test {
    ReverseAuction public auction;
    ERC20Mock public usdc;
    MockIdentityRegistry public identityRegistry;
    MockReputationRegistry public reputationRegistry;

    address public buyer;
    address public provider1;
    address public provider2;
    address public provider3;

    uint256 public buyerPk = 1;
    uint256 public provider1Pk = 2;
    uint256 public provider2Pk = 3;
    uint256 public provider3Pk = 4;

    uint256 public agentId1;
    uint256 public agentId2;
    uint256 public agentId3;

    uint256 constant MAX_PRICE = 1000e6; // 1000 USDC
    uint256 constant COMMIT_DURATION = 6 days;
    uint256 constant REVEAL_DURATION = 1 days;
    uint256 constant TOTAL_DURATION = COMMIT_DURATION + REVEAL_DURATION; // 7 days, matches old single-phase DURATION
    uint256 constant REPUTATION_WEIGHT = 50; // 0.50

    bytes32 constant NONCE_1 = bytes32(uint256(1));
    bytes32 constant NONCE_2 = bytes32(uint256(2));
    bytes32 constant NONCE_3 = bytes32(uint256(3));

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

    event AuctionEnded(uint256 indexed auctionId, uint256 indexed winningAgentId, uint256 winningBid);

    event ServiceCompleted(uint256 indexed auctionId, uint256 indexed agentId, address indexed provider);

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

    function setUp() public {
        // Setup addresses from private keys
        buyer = vm.addr(buyerPk);
        provider1 = vm.addr(provider1Pk);
        provider2 = vm.addr(provider2Pk);
        provider3 = vm.addr(provider3Pk);

        // Deploy mocks
        usdc = new ERC20Mock();
        identityRegistry = new MockIdentityRegistry();
        reputationRegistry = new MockReputationRegistry(address(identityRegistry));

        // Deploy auction contract
        auction = new ReverseAuction(address(usdc), address(identityRegistry), address(reputationRegistry), 50);

        // Setup agents
        vm.startPrank(provider1);
        agentId1 = identityRegistry.register();
        vm.stopPrank();

        vm.startPrank(provider2);
        agentId2 = identityRegistry.register();
        vm.stopPrank();

        vm.startPrank(provider3);
        agentId3 = identityRegistry.register();
        vm.stopPrank();

        // Setup reputations (whole numbers, 0 decimals) - agentId3 is left unset (no feedback yet)
        reputationRegistry.setReputation(agentId1, 10, 80, 0); // 80/100
        reputationRegistry.setReputation(agentId2, 5, 60, 0); // 60/100

        // Give buyer USDC
        usdc.mint(buyer, 10000e6);
    }

    function testDebugSignatureRecovery() public view {
        // Test that our signing mechanism works correctly
        uint256 testPk = provider1Pk;
        address expectedAddr = provider1;

        console.log("Testing signature recovery:");
        console.log("Private key:", testPk);
        console.log("Expected address:", expectedAddr);
        console.log("vm.addr(pk):", vm.addr(testPk));

        // Create a simple message
        bytes32 messageHash = keccak256(abi.encodePacked("test message"));
        console.log("Message hash:");
        console.logBytes32(messageHash);

        // Apply EIP-191 prefix
        bytes32 ethSignedMessageHash = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", messageHash)
        );
        console.log("ETH signed message hash:");
        console.logBytes32(ethSignedMessageHash);

        // Sign it
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(testPk, ethSignedMessageHash);
        bytes memory signature = abi.encodePacked(r, s, v);
        console.log("Signature v:", v);
        console.logBytes32(r);
        console.logBytes32(s);

        // Try to recover using ECDSA
        address recoveredAddr = ECDSA.recover(ethSignedMessageHash, signature);
        console.log("Recovered address:", recoveredAddr);
        console.log("Match:", recoveredAddr == expectedAddr);

        require(recoveredAddr == expectedAddr, "Signature recovery failed");
    }

    function testCreateAuction() public {
        uint256[] memory eligibleAgents = new uint256[](2);
        eligibleAgents[0] = agentId1;
        eligibleAgents[1] = agentId2;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectEmit(true, true, false, true);
        emit AuctionCreated(1, buyer, "QmTest", MAX_PRICE, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);

        uint256 auctionId =
            auction.createAuction("QmTest", MAX_PRICE, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();

        assertEq(auctionId, 1);
        assertEq(usdc.balanceOf(address(auction)), MAX_PRICE);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.buyer, buyer);
        assertEq(auc.maxPrice, MAX_PRICE);
        assertEq(auc.commitDuration, COMMIT_DURATION);
        assertEq(auc.revealDuration, REVEAL_DURATION);
        assertEq(auc.isActive, true);
        assertEq(auc.isCompleted, false);
    }

    function testCreateAuctionRevertsWithInvalidCommitDuration() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId1;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.InvalidAuctionDuration.selector);
        auction.createAuction("QmTest", MAX_PRICE, 0, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithInvalidRevealDuration() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId1;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.InvalidRevealDuration.selector);
        auction.createAuction("QmTest", MAX_PRICE, COMMIT_DURATION, 0, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithInvalidMaxPrice() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId1;

        vm.startPrank(buyer);
        vm.expectRevert(ReverseAuction.InvalidMaxPrice.selector);
        auction.createAuction("QmTest", 0, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithNoEligibleAgents() public {
        uint256[] memory eligibleAgents = new uint256[](0);

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.NoEligibleAgents.selector);
        auction.createAuction("QmTest", MAX_PRICE, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithNonExistentAgent() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = 999; // Non-existent agent

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.AgentNotFound.selector);
        auction.createAuction("QmTest", MAX_PRICE, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testComputeCommitHashMatchesManualEncoding() public view {
        bytes32 expected = keccak256(abi.encode(uint256(1), agentId1, uint256(800e6), NONCE_1, provider1));
        bytes32 actual = auction.computeCommitHash(1, agentId1, 800e6, NONCE_1, provider1);
        assertEq(actual, expected);
    }

    function testCommitBid() public {
        uint256 auctionId = _createTestAuction();
        bytes32 commitHash = auction.computeCommitHash(auctionId, agentId1, 800e6, NONCE_1, provider1);

        vm.expectEmit(true, true, true, true);
        emit BidCommitted(auctionId, provider1, agentId1, block.timestamp);

        vm.prank(provider1);
        auction.commitBid(auctionId, agentId1, commitHash);

        ReverseAuction.Commitment memory commitment = auction.getCommitment(auctionId, agentId1);
        assertEq(commitment.provider, provider1);
        assertEq(commitment.commitHash, commitHash);
        assertEq(commitment.revealed, false);

        // A commitment alone is not yet a revealed bid.
        assertEq(auction.getBidCount(auctionId), 0);
    }

    function testCommitBidRevertsWhenNotAgentOwner() public {
        uint256 auctionId = _createTestAuction();
        bytes32 commitHash = auction.computeCommitHash(auctionId, agentId1, 800e6, NONCE_1, provider2);

        vm.prank(provider2); // Not owner of agentId1
        vm.expectRevert(ReverseAuction.NotAgentOwner.selector);
        auction.commitBid(auctionId, agentId1, commitHash);
    }

    function testCommitBidRevertsWhenAgentNotEligible() public {
        uint256 auctionId = _createTestAuction();
        bytes32 commitHash = auction.computeCommitHash(auctionId, agentId3, 800e6, NONCE_1, provider3);

        vm.prank(provider3);
        vm.expectRevert(ReverseAuction.AgentNotEligible.selector);
        auction.commitBid(auctionId, agentId3, commitHash); // agentId3 not in eligible list
    }

    function testCommitBidRevertsWhenAlreadyCommitted() public {
        uint256 auctionId = _createTestAuction();
        bytes32 firstHash = auction.computeCommitHash(auctionId, agentId1, 800e6, NONCE_1, provider1);
        bytes32 secondHash = auction.computeCommitHash(auctionId, agentId1, 600e6, NONCE_2, provider1);

        vm.prank(provider1);
        auction.commitBid(auctionId, agentId1, firstHash);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.AlreadyCommitted.selector);
        auction.commitBid(auctionId, agentId1, secondHash);
    }

    function testCommitBidRevertsWhenNotInCommitPhase() public {
        uint256 auctionId = _createTestAuction();
        bytes32 commitHash = auction.computeCommitHash(auctionId, agentId1, 800e6, NONCE_1, provider1);

        _warpPastCommitDeadline(auctionId);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.NotInCommitPhase.selector);
        auction.commitBid(auctionId, agentId1, commitHash);
    }

    function testRevealBid() public {
        uint256 auctionId = _createTestAuction();
        uint256 bidAmount = 800e6;

        _commit(auctionId, agentId1, provider1, bidAmount, NONCE_1);
        _warpPastCommitDeadline(auctionId);

        // score = w*normalize(reputation) + (1-w)*normalize(1 - bidAmount/maxPrice)
        //        = 0.5 * (80/100) + 0.5 * (1 - 800e6/1000e6) = 0.5*0.8 + 0.5*0.2 = 0.5 -> 5000
        vm.expectEmit(true, true, true, true);
        emit BidRevealed(auctionId, provider1, agentId1, bidAmount, 80, 0, 5000, block.timestamp);

        _reveal(auctionId, agentId1, provider1, bidAmount, NONCE_1);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.winningAgentId, agentId1);
        assertEq(auc.winningBid, bidAmount);

        ReverseAuction.Commitment memory commitment = auction.getCommitment(auctionId, agentId1);
        assertEq(commitment.revealed, true);
    }

    function testRevealBidWithBetterScore() public {
        uint256 auctionId = _createTestAuction();

        // provider2 has lower reputation (60) but a lower price; provider1 has higher
        // reputation (80) and a slightly higher price. Both commit before either reveals.
        _commit(auctionId, agentId2, provider2, 700e6, NONCE_1);
        _commit(auctionId, agentId1, provider1, 750e6, NONCE_2);
        _warpPastCommitDeadline(auctionId);

        _reveal(auctionId, agentId2, provider2, 700e6, NONCE_1);
        _reveal(auctionId, agentId1, provider1, 750e6, NONCE_2);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.winningAgentId, agentId1); // Should win with better overall score
    }

    function testRevealBidSucceedsEvenWhenNotCompetitive() public {
        uint256 auctionId = _createTestAuction();

        // provider1 reveals an excellent score first.
        _commit(auctionId, agentId1, provider1, 600e6, NONCE_1);
        _commit(auctionId, agentId2, provider2, 900e6, NONCE_2);
        _warpPastCommitDeadline(auctionId);

        _reveal(auctionId, agentId1, provider1, 600e6, NONCE_1);

        // provider2's reveal is uncompetitive, but a valid reveal of a bound commitment must
        // always succeed and be recorded - there is nothing left to protect by reverting.
        _reveal(auctionId, agentId2, provider2, 900e6, NONCE_2);

        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        assertEq(bids.length, 2);

        // The uncompetitive bid does not become the winner.
        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.winningAgentId, agentId1);
        assertEq(auc.winningBid, 600e6);
    }

    function testRevealBidRevertsWhenNotInRevealPhaseBeforeCommitDeadline() public {
        uint256 auctionId = _createTestAuction();
        uint256 bidAmount = 800e6;

        _commit(auctionId, agentId1, provider1, bidAmount, NONCE_1);

        // Still within the commit phase - reveal is not allowed yet.
        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.NotInRevealPhase.selector);
        auction.revealBid(auctionId, agentId1, bidAmount, NONCE_1);
    }

    function testRevealBidRevertsWhenNotInRevealPhaseAfterRevealDeadline() public {
        uint256 auctionId = _createTestAuction();
        uint256 bidAmount = 800e6;

        _commit(auctionId, agentId1, provider1, bidAmount, NONCE_1);
        _warpPastRevealDeadline(auctionId);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.NotInRevealPhase.selector);
        auction.revealBid(auctionId, agentId1, bidAmount, NONCE_1);
    }

    function testRevealBidRevertsWithNoCommitmentFound() public {
        uint256 auctionId = _createTestAuction();
        _warpPastCommitDeadline(auctionId);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.NoCommitmentFound.selector);
        auction.revealBid(auctionId, agentId1, 800e6, NONCE_1);
    }

    function testRevealBidRevertsWhenNotCommitter() public {
        uint256 auctionId = _createTestAuction();
        uint256 bidAmount = 800e6;

        _commit(auctionId, agentId1, provider1, bidAmount, NONCE_1);
        _warpPastCommitDeadline(auctionId);

        // provider2 did not make this commitment, even with the right (amount, nonce).
        vm.prank(provider2);
        vm.expectRevert(ReverseAuction.NotAgentOwner.selector);
        auction.revealBid(auctionId, agentId1, bidAmount, NONCE_1);
    }

    function testRevealBidRevertsWhenAlreadyRevealed() public {
        uint256 auctionId = _createTestAuction();
        uint256 bidAmount = 800e6;

        _commitAndReveal(auctionId, agentId1, provider1, bidAmount, NONCE_1);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.AlreadyRevealed.selector);
        auction.revealBid(auctionId, agentId1, bidAmount, NONCE_1);
    }

    function testRevealBidRevertsWithInvalidRevealOnWrongAmount() public {
        uint256 auctionId = _createTestAuction();

        _commit(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastCommitDeadline(auctionId);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.InvalidReveal.selector);
        auction.revealBid(auctionId, agentId1, 750e6, NONCE_1); // Different amount than committed
    }

    function testRevealBidRevertsWithInvalidRevealOnWrongNonce() public {
        uint256 auctionId = _createTestAuction();

        _commit(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastCommitDeadline(auctionId);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.InvalidReveal.selector);
        auction.revealBid(auctionId, agentId1, 800e6, NONCE_2); // Different nonce than committed
    }

    function testRevealBidRevertsWhenBidTooHigh() public {
        uint256 auctionId = _createTestAuction();
        uint256 tooHigh = MAX_PRICE + 1;

        // Commit does not see the plaintext amount, so it succeeds even though the
        // amount will turn out to exceed maxPrice; the check only happens at reveal time.
        _commit(auctionId, agentId1, provider1, tooHigh, NONCE_1);
        _warpPastCommitDeadline(auctionId);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.BidTooHigh.selector);
        auction.revealBid(auctionId, agentId1, tooHigh, NONCE_1);
    }

    function testRevealBidWithDefaultReputation() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId3; // Agent with no reputation feedback yet

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);
        uint256 auctionId =
            auction.createAuction("QmTest", MAX_PRICE, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();

        _commitAndReveal(auctionId, agentId3, provider3, 800e6, NONCE_1);

        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        assertEq(bids[0].reputation, 50); // Default reputation
        assertEq(bids[0].reputationDecimals, 0);
    }

    function testScoreCalculationWithDecimalReputation() public {
        // Isolate the effect of decimal precision: same price, only the reputation's
        // fractional digit differs (80.5 vs 80.0). Without decimal precision these would
        // be indistinguishable (both round to 80) and tie; with it, 80.5 must score higher.
        reputationRegistry.setReputation(agentId1, 10, 805, 1); // 80.5
        reputationRegistry.setReputation(agentId2, 10, 80, 0); // 80.0

        uint256 auctionId1 = _createTestAuction();
        _commitAndReveal(auctionId1, agentId1, provider1, 700e6, NONCE_1);
        ReverseAuction.Bid[] memory bids1 = auction.getAuctionBids(auctionId1);

        uint256 auctionId2 = _createTestAuction();
        _commitAndReveal(auctionId2, agentId2, provider2, 700e6, NONCE_1);
        ReverseAuction.Bid[] memory bids2 = auction.getAuctionBids(auctionId2);

        assertEq(bids1[0].reputation, 805);
        assertEq(bids1[0].reputationDecimals, 1);
        assertGt(bids1[0].score, bids2[0].score);
    }

    function testEndAuction() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);

        vm.expectEmit(true, true, false, true);
        emit AuctionEnded(auctionId, agentId1, 800e6);

        auction.endAuction(auctionId);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isActive, false);
    }

    function testEndAuctionByBuyerEarly() public {
        uint256 auctionId = _createTestAuction();

        // Reveal phase has started but not yet expired.
        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);

        // Buyer can end early
        vm.prank(buyer);
        auction.endAuction(auctionId);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isActive, false);
    }

    function testEndAuctionRevertsWhenNotAuthorized() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);

        // Random user cannot end auction before the reveal deadline expires
        vm.prank(address(999));
        vm.expectRevert(ReverseAuction.NotAuthorized.selector);
        auction.endAuction(auctionId);
    }

    function testCompleteService() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        // Create feedbackAuth with proper private key
        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk);

        uint256 providerBalanceBefore = usdc.balanceOf(provider1);
        uint256 buyerBalanceBefore = usdc.balanceOf(buyer);

        vm.prank(provider1);
        auction.completeService(auctionId, feedbackAuth);

        assertEq(usdc.balanceOf(provider1), providerBalanceBefore + 800e6);
        assertEq(usdc.balanceOf(buyer), buyerBalanceBefore + 200e6); // Refund of excess

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isCompleted, true);
        assertEq(auc.escrowAmount, 0);
    }

    function testCompleteServiceRevertsWhenNotAgentOwner() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk);

        vm.prank(provider2); // Not the owner
        vm.expectRevert(ReverseAuction.NotAgentOwner.selector);
        auction.completeService(auctionId, feedbackAuth);
    }

    function testCompleteServiceRevertsWithInvalidFeedbackAuth() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        // Create feedbackAuth with wrong agentId
        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId2, buyer, provider1, provider1Pk);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.InvalidFeedbackAuth.selector);
        auction.completeService(auctionId, feedbackAuth);
    }

    function testCompleteServiceRevertsWithExpiredFeedbackAuth() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        // Create feedbackAuth that's already expired
        uint256 expiry = block.timestamp - 1;
        bytes memory feedbackAuth =
            _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk, expiry);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.FeedbackAuthExpired.selector);
        auction.completeService(auctionId, feedbackAuth);
    }

    function testCompleteServiceHandlesAgentTransfer() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        // Transfer agent to new owner
        address newOwner = address(999);
        usdc.mint(newOwner, 1000e6); // Give new owner some tokens for test
        identityRegistry.setOwner(agentId1, newOwner);

        // Use vm.addr to get the correct address for private key 999
        uint256 newOwnerPk = 999;
        address newOwnerAddr = vm.addr(newOwnerPk);
        identityRegistry.setOwner(agentId1, newOwnerAddr);

        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, newOwnerAddr, newOwnerPk);

        uint256 newOwnerBalanceBefore = usdc.balanceOf(newOwnerAddr);

        vm.prank(newOwnerAddr);
        auction.completeService(auctionId, feedbackAuth);

        // Payment goes to current owner, not original bidder
        assertEq(usdc.balanceOf(newOwnerAddr), newOwnerBalanceBefore + 800e6);
    }

    function testRefundBuyer() public {
        uint256 auctionId = _createTestAuction();

        // No bids committed or revealed
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        uint256 buyerBalanceBefore = usdc.balanceOf(buyer);

        vm.prank(buyer);
        auction.refundBuyer(auctionId);

        assertEq(usdc.balanceOf(buyer), buyerBalanceBefore + MAX_PRICE);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isCompleted, true);
        assertEq(auc.escrowAmount, 0);
    }

    function testRefundBuyerRevertsWhenHasWinner() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        vm.prank(buyer);
        vm.expectRevert(ReverseAuction.NotAuthorized.selector);
        auction.refundBuyer(auctionId);
    }

    function testGetAuctionBids() public {
        uint256 auctionId = _createTestAuction();

        _commit(auctionId, agentId2, provider2, 900e6, NONCE_1);
        _commit(auctionId, agentId1, provider1, 800e6, NONCE_2);
        _warpPastCommitDeadline(auctionId);

        _reveal(auctionId, agentId2, provider2, 900e6, NONCE_1);
        _reveal(auctionId, agentId1, provider1, 800e6, NONCE_2);

        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        assertEq(bids.length, 2);
        assertEq(bids[0].provider, provider2);
        assertEq(bids[1].provider, provider1);
    }

    function testGetBidCount() public {
        uint256 auctionId = _createTestAuction();

        assertEq(auction.getBidCount(auctionId), 0);

        _commit(auctionId, agentId1, provider1, 800e6, NONCE_1);
        assertEq(auction.getBidCount(auctionId), 0); // Commitment alone doesn't count as a bid

        _warpPastCommitDeadline(auctionId);
        _reveal(auctionId, agentId1, provider1, 800e6, NONCE_1);

        assertEq(auction.getBidCount(auctionId), 1);
    }

    function testGetCommitDeadlineAndRevealDeadline() public {
        uint256 auctionId = _createTestAuction();
        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);

        assertEq(auction.getCommitDeadline(auctionId), auc.startTime + COMMIT_DURATION);
        assertEq(auction.getRevealDeadline(auctionId), auc.startTime + COMMIT_DURATION + REVEAL_DURATION);
    }

    function testGetAuctionPhase() public {
        uint256 auctionId = _createTestAuction();

        assertEq(uint8(auction.getAuctionPhase(auctionId)), uint8(ReverseAuction.Phase.Commit));

        _warpPastCommitDeadline(auctionId);
        assertEq(uint8(auction.getAuctionPhase(auctionId)), uint8(ReverseAuction.Phase.Reveal));

        _warpPastRevealDeadline(auctionId);
        assertEq(uint8(auction.getAuctionPhase(auctionId)), uint8(ReverseAuction.Phase.Ended));
    }

    function testGetTimeRemaining() public {
        uint256 auctionId = _createTestAuction();

        uint256 timeRemaining = auction.getTimeRemaining(auctionId);
        assertEq(timeRemaining, TOTAL_DURATION);

        vm.warp(block.timestamp + 3 days);
        timeRemaining = auction.getTimeRemaining(auctionId);
        assertEq(timeRemaining, 4 days);

        vm.warp(block.timestamp + 5 days);
        timeRemaining = auction.getTimeRemaining(auctionId);
        assertEq(timeRemaining, 0);
    }

    function testIsAuctionActive() public {
        uint256 auctionId = _createTestAuction();

        assertTrue(auction.isAuctionActive(auctionId));

        vm.warp(block.timestamp + TOTAL_DURATION + 1);
        assertFalse(auction.isAuctionActive(auctionId));
    }

    function testGetCurrentWinningBid() public {
        uint256 auctionId = _createTestAuction();

        assertEq(auction.getCurrentWinningBid(auctionId), MAX_PRICE);

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);

        assertEq(auction.getCurrentWinningBid(auctionId), 800e6);
    }

    function testScoreCalculation() public {
        // Test that higher reputation + lower price = higher score
        uint256 auctionId = _createTestAuction();
        _commitAndReveal(auctionId, agentId1, provider1, 700e6, NONCE_1); // rep: 80, price: 700
        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        uint256 score1 = bids[0].score;

        // Create new auction for comparison
        uint256 auctionId2 = _createTestAuction();
        _commitAndReveal(auctionId2, agentId2, provider2, 800e6, NONCE_1); // rep: 60, price: 800
        bids = auction.getAuctionBids(auctionId2);
        uint256 score2 = bids[0].score;

        // Higher reputation and lower price should give higher score
        assertGt(score1, score2);
    }

    function testFeedbackAuthEvent() public {
        uint256 auctionId = _createTestAuction();

        _commitAndReveal(auctionId, agentId1, provider1, 800e6, NONCE_1);
        _warpPastRevealDeadline(auctionId);
        auction.endAuction(auctionId);

        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk);

        vm.prank(provider1);
        // Just check the indexed event parameters, skip the feedbackAuth bytes data
        vm.expectEmit(true, true, true, false);
        emit FeedbackAuthProvided(auctionId, agentId1, buyer, feedbackAuth);

        auction.completeService(auctionId, feedbackAuth);
    }

    // Helper functions

    function _createTestAuction() internal returns (uint256) {
        uint256[] memory eligibleAgents = new uint256[](2);
        eligibleAgents[0] = agentId1;
        eligibleAgents[1] = agentId2;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);
        uint256 auctionId =
            auction.createAuction("QmTest", MAX_PRICE, COMMIT_DURATION, REVEAL_DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();

        return auctionId;
    }

    function _commit(uint256 auctionId, uint256 agentId, address provider, uint256 bidAmount, bytes32 nonce)
        internal
    {
        bytes32 commitHash = auction.computeCommitHash(auctionId, agentId, bidAmount, nonce, provider);
        vm.prank(provider);
        auction.commitBid(auctionId, agentId, commitHash);
    }

    function _reveal(uint256 auctionId, uint256 agentId, address provider, uint256 bidAmount, bytes32 nonce)
        internal
    {
        vm.prank(provider);
        auction.revealBid(auctionId, agentId, bidAmount, nonce);
    }

    function _commitAndReveal(
        uint256 auctionId,
        uint256 agentId,
        address provider,
        uint256 bidAmount,
        bytes32 nonce
    ) internal {
        _commit(auctionId, agentId, provider, bidAmount, nonce);
        _warpPastCommitDeadline(auctionId);
        _reveal(auctionId, agentId, provider, bidAmount, nonce);
    }

    function _warpPastCommitDeadline(uint256 auctionId) internal {
        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        uint256 commitDeadline = auc.startTime + auc.commitDuration;
        if (block.timestamp <= commitDeadline) {
            vm.warp(commitDeadline + 1);
        }
    }

    function _warpPastRevealDeadline(uint256 auctionId) internal {
        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        uint256 revealDeadline = auc.startTime + auc.commitDuration + auc.revealDuration;
        if (block.timestamp <= revealDeadline) {
            vm.warp(revealDeadline + 1);
        }
    }

    function _createFeedbackAuth(uint256 auctionId, uint256 agentId, address client, address signer)
        internal
        view
        returns (bytes memory)
    {
        return _createFeedbackAuthWithExpiry(auctionId, agentId, client, signer, block.timestamp + 30 days);
    }

    function _createFeedbackAuthWithExpiry(
        uint256, /* auctionId */
        uint256 agentId,
        address client,
        address signer,
        uint256 expiry
    ) internal view returns (bytes memory) {
        // Get private key for signer (test helper)
        uint256 signerPrivateKey = _getPrivateKey(signer);
        return _createFeedbackAuthSigned(0, agentId, client, signer, signerPrivateKey, expiry);
    }

    function _createFeedbackAuthSigned(
        uint256, /* auctionId */
        uint256 agentId,
        address client,
        address signer,
        uint256 signerPrivateKey
    ) internal view returns (bytes memory) {
        return _createFeedbackAuthSigned(0, agentId, client, signer, signerPrivateKey, block.timestamp + 30 days);
    }

    function _createFeedbackAuthSigned(
        uint256, /* auctionId */
        uint256 agentId,
        address client,
        address signer,
        uint256 signerPrivateKey,
        uint256 expiry
    ) internal view returns (bytes memory) {
        uint64 indexLimit = 1;
        uint256 chainId = block.chainid;
        address identityReg = address(identityRegistry);

        // Create message hash using abi.encode (matching ERC-8004)
        bytes32 rawHash = keccak256(
            abi.encode(agentId, client, indexLimit, expiry, chainId, identityReg, signer)
        );

        // Apply EIP-191 prefix using MessageHashUtils (matching official implementation)
        bytes32 messageHash = MessageHashUtils.toEthSignedMessageHash(rawHash);

        // Sign the prefixed hash
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(signerPrivateKey, messageHash);
        bytes memory signature = abi.encodePacked(r, s, v);

        // Encode in ERC-8004 format: [224 bytes params][65 bytes signature]
        bytes memory params = abi.encode(agentId, client, indexLimit, expiry, chainId, identityReg, signer);
        return abi.encodePacked(params, signature);
    }

    function _getPrivateKey(address addr) internal view returns (uint256) {
        // Map test addresses to private keys
        if (addr == buyer) return buyerPk;
        if (addr == provider1) return provider1Pk;
        if (addr == provider2) return provider2Pk;
        if (addr == provider3) return provider3Pk;
        return 1;
    }
}
