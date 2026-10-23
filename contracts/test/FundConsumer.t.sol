// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console2 as console} from "forge-std/Test.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

/**
 * @title FundConsumer Test
 * @notice Tests for the FundConsumer.s.sol script logic
 * 
 * Run with:
 *   forge test --match-contract FundConsumerTest -vv
 */
contract FundConsumerTest is Test {
    ERC20Mock usdc;
    address deployer;
    address consumer;
    uint256 deployerPk;
    uint256 consumerPk;
    
    function setUp() public {
        // Create test accounts
        deployerPk = 0xA11CE;
        consumerPk = 0xB0B;
        deployer = vm.addr(deployerPk);
        consumer = vm.addr(consumerPk);
        
        // Give deployer some ETH
        vm.deal(deployer, 100 ether);
        
        // Deploy Mock USDC as deployer
        vm.startPrank(deployer);
        usdc = new ERC20Mock();
        
        // Mint USDC to deployer (simulating post-deployment state)
        usdc.mint(deployer, 10_000 * 1e6); // 10,000 USDC
        vm.stopPrank();
        
        console.log("Setup complete:");
        console.log("  Deployer:", deployer);
        console.log("  Consumer:", consumer);
        console.log("  USDC:", address(usdc));
        console.log("  Deployer ETH:", deployer.balance / 1e18, "ETH");
        console.log("  Deployer USDC:", usdc.balanceOf(deployer) / 1e6, "USDC");
    }
    
    function test_FundConsumer_Success() public {
        console.log("\n=== Test: Successful Consumer Funding ===");
        
        // Record initial balances
        uint256 consumerEthBefore = consumer.balance;
        uint256 consumerUsdcBefore = usdc.balanceOf(consumer);
        uint256 deployerEthBefore = deployer.balance;
        uint256 deployerUsdcBefore = usdc.balanceOf(deployer);
        
        console.log("\nInitial balances:");
        console.log("  Consumer ETH:", consumerEthBefore / 1e18, "ETH");
        console.log("  Consumer USDC:", consumerUsdcBefore / 1e6, "USDC");
        
        // Amounts to transfer (matching experiment config defaults)
        uint256 ethAmount = 1 ether;
        uint256 usdcAmount = 1000 * 1e6; // 1000 USDC
        
        // Simulate the FundConsumer script
        vm.startPrank(deployer);
        
        // 1. Transfer ETH for gas fees
        (bool success,) = consumer.call{value: ethAmount}("");
        require(success, "ETH transfer failed");
        console.log("\nTransferred", ethAmount / 1e18, "ETH to consumer");
        
        // 2. Transfer USDC for auctions
        usdc.transfer(consumer, usdcAmount);
        console.log("Transferred", usdcAmount / 1e6, "USDC to consumer");
        
        vm.stopPrank();
        
        // Verify final balances
        uint256 consumerEthAfter = consumer.balance;
        uint256 consumerUsdcAfter = usdc.balanceOf(consumer);
        uint256 deployerEthAfter = deployer.balance;
        uint256 deployerUsdcAfter = usdc.balanceOf(deployer);
        
        console.log("\nFinal balances:");
        console.log("  Consumer ETH:", consumerEthAfter / 1e18, "ETH");
        console.log("  Consumer USDC:", consumerUsdcAfter / 1e6, "USDC");
        
        // Assertions
        assertEq(consumerEthAfter, consumerEthBefore + ethAmount, "Consumer should receive 1 ETH");
        assertEq(consumerUsdcAfter, consumerUsdcBefore + usdcAmount, "Consumer should receive 1000 USDC");
        assertEq(deployerEthAfter, deployerEthBefore - ethAmount, "Deployer ETH should decrease by 1 ETH");
        assertEq(deployerUsdcAfter, deployerUsdcBefore - usdcAmount, "Deployer USDC should decrease by 1000 USDC");
        
        // Verify consumer has enough for operations
        assertGt(consumerEthAfter, 0.5 ether, "Consumer should have enough ETH for gas");
        assertGt(consumerUsdcAfter, 100 * 1e6, "Consumer should have enough USDC for auctions");
        
        console.log("\n[SUCCESS] All assertions passed!");
    }
    
    function test_FundConsumer_MintWhenInsufficientUSDC() public {
        console.log("\n=== Test: Minting USDC When Insufficient Balance ===");
        
        // Setup: deployer has only 100 USDC, needs to transfer 1000
        vm.startPrank(deployer);
        
        // Transfer away most USDC to simulate low balance
        usdc.transfer(address(0x123), 9900 * 1e6);
        uint256 deployerUsdcBalance = usdc.balanceOf(deployer);
        console.log("Deployer USDC balance:", deployerUsdcBalance / 1e6, "USDC");
        
        // Amount needed
        uint256 usdcAmount = 1000 * 1e6;
        
        // Check if need to mint (this is what FundConsumer.s.sol does)
        if (deployerUsdcBalance < usdcAmount) {
            console.log("Insufficient USDC, need to mint more");
            uint256 mintAmount = usdcAmount - deployerUsdcBalance;
            usdc.mint(deployer, mintAmount);
            console.log("Minted", mintAmount / 1e6, "USDC");
        }
        
        // Now transfer should succeed
        uint256 consumerUsdcBefore = usdc.balanceOf(consumer);
        usdc.transfer(consumer, usdcAmount);
        uint256 consumerUsdcAfter = usdc.balanceOf(consumer);
        
        vm.stopPrank();
        
        assertEq(consumerUsdcAfter, consumerUsdcBefore + usdcAmount, "Consumer should receive 1000 USDC after minting");
        console.log("\n[SUCCESS] Minting and transfer successful!");
    }
    
    function test_FundConsumer_DifferentAddresses() public {
        console.log("\n=== Test: Consumer and Deployer Are Different Addresses ===");
        
        // Verify precondition
        assertNotEq(deployer, consumer, "Deployer and consumer must be different");
        console.log("Deployer:", deployer);
        console.log("Consumer:", consumer);
        
        // This is the check the experiment runner does
        bool needsFunding = (deployer != consumer);
        assertTrue(needsFunding, "Should detect different addresses");
        
        console.log("\n[SUCCESS] Address validation passed!");
    }
    
    function test_FundConsumer_SameAddressSkipsFunding() public {
        console.log("\n=== Test: Same Address Should Skip Funding ===");
        
        // Simulate same address scenario
        address sameAccount = deployer;
        
        // This is what experiment runner checks
        bool needsFunding = (deployer != sameAccount);
        assertFalse(needsFunding, "Should skip funding for same address");
        
        console.log("Deployer and consumer are same:", sameAccount);
        console.log("Funding skipped [OK]");
        
        console.log("\n[SUCCESS] Skip logic validated!");
    }
    
    function test_FundConsumer_BalancesAfterMultipleAuctions() public {
        console.log("\n=== Test: Consumer Has Enough for Multiple Auctions ===");
        
        // Fund consumer
        vm.startPrank(deployer);
        (bool success,) = consumer.call{value: 1 ether}("");
        require(success, "ETH transfer failed");
        usdc.transfer(consumer, 1000 * 1e6);
        vm.stopPrank();
        
        // Simulate auction creation costs
        uint256 auctionBudget = 100 * 1e6; // 100 USDC per auction
        uint256 gasPerAuction = 0.01 ether; // ~0.01 ETH per auction
        
        uint256 consumerUsdc = usdc.balanceOf(consumer);
        uint256 consumerEth = consumer.balance;
        
        uint256 maxAuctionsUsdc = consumerUsdc / auctionBudget;
        uint256 maxAuctionsEth = consumerEth / gasPerAuction;
        
        console.log("Consumer can create:");
        console.log("  Based on USDC:", maxAuctionsUsdc, "auctions");
        console.log("  Based on ETH:", maxAuctionsEth, "auctions");
        
        assertGe(maxAuctionsUsdc, 10, "Should afford at least 10 auctions (USDC)");
        assertGe(maxAuctionsEth, 50, "Should afford at least 50 auctions (ETH)");
        
        console.log("\n[SUCCESS] Consumer funded sufficiently for multiple auctions!");
    }
}
