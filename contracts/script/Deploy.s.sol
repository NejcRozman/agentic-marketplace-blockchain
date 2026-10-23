// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {ReverseAuction} from "../src/ReverseAuction.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

/**
 * @title Deploy Script
 * @notice Deploys ReverseAuction contract with mock USDC to a forked Sepolia network
 * 
 * Usage:
 *   1. Start Anvil with Sepolia fork:
 *      anvil --fork-url https://sepolia.infura.io/v3/${INFURA_API_KEY}
 * 
 *   2. Run this script:
 *      forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
 * 
 * The script will:
 *   - Deploy a mock USDC token
 *   - Deploy ReverseAuction with the ERC-8004 registries from Sepolia
 *   - Mint USDC to test accounts for auction creation
 */
contract DeployScript is Script {
    // Anvil's first account for deployment (has funds by default)
    uint256 constant ANVIL_DEFAULT_PK = 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;
    
    function run() external {
        // Load private key from environment (the account that will use the contracts)
        uint256 userPrivateKey = vm.envUint("PRIVATE_KEY");
        address userAccount = vm.addr(userPrivateKey);
        
        // Load ERC-8004 addresses from environment
        address identityRegistry = vm.envAddress("IDENTITY_REGISTRY_ADDRESS");
        address reputationRegistry = vm.envAddress("REPUTATION_REGISTRY_ADDRESS");
        uint256 initialReputationDefault = vm.envOr("INITIAL_REPUTATION_DEFAULT", uint256(50));
        
        console.log("=== Deploy Configuration ===");
        console.log("Deployer (Anvil default):", vm.addr(ANVIL_DEFAULT_PK));
        console.log("User Account (from PRIVATE_KEY):", userAccount);
        console.log("IdentityRegistry:", identityRegistry);
        console.log("ReputationRegistry:", reputationRegistry);
        console.log("Initial reputation default:", initialReputationDefault);
        console.log("");
        
        // Use Anvil's default account for deployment (has funds)
        vm.startBroadcast(ANVIL_DEFAULT_PK);
        
        // 1. Deploy Mock USDC
        ERC20Mock usdc = new ERC20Mock();
        console.log("Mock USDC deployed at:", address(usdc));
        
        // 2. Deploy ReverseAuction
        ReverseAuction auction = new ReverseAuction(
            address(usdc),
            identityRegistry,
            reputationRegistry,
            initialReputationDefault
        );
        console.log("ReverseAuction deployed at:", address(auction));
        
        // 3. Mint USDC to the user account (from PRIVATE_KEY in .env)
        uint256 mintAmount = 100_000 * 1e6; // 100,000 USDC (6 decimals like real USDC)
        usdc.mint(userAccount, mintAmount);
        console.log("Minted", mintAmount / 1e6, "USDC to user account:", userAccount);
        
        vm.stopBroadcast();
        
        console.log("");
        console.log("=== Deployment Complete ===");
        console.log("Add these to your agents/.env:");
        console.log("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS=", address(auction));
        console.log("MOCK_USDC_ADDRESS=", address(usdc));
    }
}
