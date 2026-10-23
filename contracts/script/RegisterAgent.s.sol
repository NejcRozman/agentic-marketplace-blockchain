// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {IIdentityRegistry} from "../src/interfaces/IIdentityRegistry.sol";

/**
 * @title RegisterAgent Script
 * @notice Registers an agent on ERC-8004 IdentityRegistry
 * 
 * Usage:
 *   1. Add PRIVATE_KEY to .env (funded Sepolia account)
 *   2. Run this script:
 *      forge script script/RegisterAgent.s.sol --rpc-url sepolia --broadcast
 * 
 * The script will:
 *   - Register a new agent NFT for the caller
 *   - Output the agent ID to use in tests
 */
contract RegisterAgentScript is Script {
    function run() external {
        // Load from environment
        address identityRegistryAddr = vm.envAddress("IDENTITY_REGISTRY_ADDRESS");
        uint256 privateKey = vm.envUint("PRIVATE_KEY");
        address caller = vm.addr(privateKey);
        
        console.log("=== Register Agent ===");
        console.log("Caller:", caller);
        console.log("IdentityRegistry:", identityRegistryAddr);
        console.log("");
        
        IIdentityRegistry identityRegistry = IIdentityRegistry(identityRegistryAddr);
        
        // Check current balance
        uint256 balanceBefore = identityRegistry.balanceOf(caller);
        console.log("Current agent count:", balanceBefore);
        
        vm.startBroadcast(privateKey);
        
        // Register agent using the interface function
        uint256 agentId = identityRegistry.register();
        
        vm.stopBroadcast();
        
        // Verify registration (view calls AFTER stopBroadcast to avoid nonce issues)
        uint256 balanceAfter = identityRegistry.balanceOf(caller);
        address agentOwner = identityRegistry.ownerOf(agentId);
        
        console.log("");
        console.log("=== Registration Successful ===");
        console.log("Agent ID:", agentId);
        console.log("Agent owner:", agentOwner);
        console.log("New agent count:", balanceAfter);
        console.log("");
        console.log("Add this to your .env:");
        console.log("  AGENT_ID=", agentId);
    }
}
