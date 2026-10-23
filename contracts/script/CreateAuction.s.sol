// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {ReverseAuction} from "../src/ReverseAuction.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title CreateAuction Script
 * @notice Creates a test auction for the provider agent to discover and bid on
 * 
 * Usage:
 *   1. Make sure ReverseAuction is deployed (run Deploy.s.sol first)
 *   2. Make sure AGENT_ID is set in .env (run RegisterAgent.s.sol first)
 *   3. Run this script:
 *      forge script script/CreateAuction.s.sol --rpc-url sepolia --broadcast
 * 
 * Required .env variables:
 *   - PRIVATE_KEY: Private key of the buyer account (must have USDC)
 *   - REVERSE_AUCTION_ADDRESS: Deployed ReverseAuction contract
 *   - MOCK_USDC_ADDRESS: Mock USDC token contract (from Deploy.s.sol)
 *   - AGENT_ID: The agent ID to include as eligible bidder
 *   - SERVICE_DESCRIPTION_CID: (optional) IPFS CID for service description
 */
contract CreateAuctionScript is Script {
    function run() external {
        // Load from environment
        uint256 privateKey = vm.envUint("PRIVATE_KEY");
        address buyer = vm.addr(privateKey);
        address reverseAuctionAddr = vm.envAddress("REVERSE_AUCTION_ADDRESS");
        address usdcAddr = vm.envAddress("MOCK_USDC_ADDRESS");
        uint256 agentId = vm.envUint("AGENT_ID");
        
        console.log("=== Create Auction ===");
        console.log("Buyer:", buyer);
        console.log("ReverseAuction:", reverseAuctionAddr);
        console.log("USDC:", usdcAddr);
        console.log("Eligible Agent ID:", agentId);
        console.log("");
        
        ReverseAuction reverseAuction = ReverseAuction(reverseAuctionAddr);
        IERC20 usdc = IERC20(usdcAddr);
        
        // Load service description CID from environment (or use default for testing)
        string memory serviceDescriptionCID = vm.envOr(
            "SERVICE_DESCRIPTION_CID",
            string("QmTestServiceDescriptionCID123456789")
        );
        
        // Auction parameters
        uint256 maxBudget = 100 * 1e6; // 100 USDC (6 decimals)
        uint256 commitDuration = 1 hours;
        uint256 revealDuration = 15 minutes;
        uint256 reputationWeight = 50; // 50% reputation, 50% price

        // Create eligible agents array with the provider's agent
        uint256[] memory eligibleAgentIds = new uint256[](1);
        eligibleAgentIds[0] = agentId;

        vm.startBroadcast(privateKey);

        // Check buyer's USDC balance
        uint256 balance = usdc.balanceOf(buyer);
        console.log("Buyer USDC balance:", balance / 1e6, "USDC");
        require(balance >= maxBudget, "Insufficient USDC balance");

        // Approve ReverseAuction to spend USDC
        usdc.approve(reverseAuctionAddr, maxBudget);
        console.log("Approved ReverseAuction to spend", maxBudget / 1e6, "USDC");

        // Create the auction
        uint256 auctionId = reverseAuction.createAuction(
            serviceDescriptionCID,
            maxBudget,
            commitDuration,
            revealDuration,
            eligibleAgentIds,
            reputationWeight
        );
        
        vm.stopBroadcast();
        
        // Get auction details
        ReverseAuction.Auction memory auction = reverseAuction.getAuctionDetails(auctionId);
        
        console.log("");
        console.log("=== Auction Created ===");
        console.log("Auction ID:", auctionId);
        console.log("Buyer:", auction.buyer);
        console.log("Max Budget:", auction.maxPrice / 1e6, "USDC");
        console.log("Commit deadline:", auction.startTime + auction.commitDuration);
        console.log("Reveal deadline:", auction.startTime + auction.commitDuration + auction.revealDuration);
        console.log("Service CID:", auction.serviceDescriptionCid);
        console.log("Is Active:", auction.isActive);
        console.log("");
        console.log("The provider agent can now discover this auction via monitor_auctions()");
    }
}
