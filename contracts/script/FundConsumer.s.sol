// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

/**
 * @title Fund Consumer Script
 * @notice Funds a consumer account with ETH and USDC
 * 
 * Usage:
 *   forge script script/FundConsumer.s.sol --rpc-url http://localhost:8545 --broadcast
 * 
 * Environment variables:
 *   DEPLOYER_PRIVATE_KEY - Account that has ETH and USDC to distribute
 *   CONSUMER_ADDRESS - Consumer address to fund
 *   USDC_ADDRESS - Address of Mock USDC contract
 *   FUND_AMOUNT_ETH - Amount of ETH to send (in ether, e.g., "1.0")
 *   FUND_AMOUNT_USDC - Amount of USDC to send (in USDC units, e.g., "10000")
 */
contract FundConsumerScript is Script {
    function run() external {
        // Load configuration from environment
        uint256 deployerPrivateKey = vm.envUint("DEPLOYER_PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);
        address consumer = vm.envAddress("CONSUMER_ADDRESS");
        address usdcAddress = vm.envAddress("USDC_ADDRESS");
        uint256 ethAmount = vm.envUint("FUND_AMOUNT_ETH"); // In wei
        uint256 usdcAmount = vm.envUint("FUND_AMOUNT_USDC"); // In USDC micro units (6 decimals)
        
        console.log("=== Funding Configuration ===");
        console.log("Deployer:", deployer);
        console.log("Consumer:", consumer);
        console.log("USDC Contract:", usdcAddress);
        console.log("ETH to send:", ethAmount, "wei");
        console.log("USDC to send:", usdcAmount / 1e6, "USDC");
        console.log("");
        
        vm.startBroadcast(deployerPrivateKey);
        
        // 1. Transfer ETH for gas fees
        (bool success,) = consumer.call{value: ethAmount}("");
        require(success, "ETH transfer failed");
        console.log("Transferred", ethAmount / 1e18, "ETH to consumer");
        
        // 2. Transfer USDC
        ERC20Mock usdc = ERC20Mock(usdcAddress);
        
        // Check if deployer has enough USDC, if not mint it
        uint256 deployerBalance = usdc.balanceOf(deployer);
        if (deployerBalance < usdcAmount) {
            uint256 mintAmount = usdcAmount - deployerBalance;
            usdc.mint(deployer, mintAmount);
            console.log("Minted", mintAmount / 1e6, "USDC to deployer");
        }
        
        usdc.transfer(consumer, usdcAmount);
        console.log("Transferred", usdcAmount / 1e6, "USDC to consumer");
        
        vm.stopBroadcast();
        
        console.log("");
        console.log("=== Funding Complete ===");
        console.log("Consumer", consumer, "is ready to use");
        console.log("ETH balance:", consumer.balance / 1e18, "ETH");
        console.log("USDC balance:", usdc.balanceOf(consumer) / 1e6, "USDC");
    }
}
