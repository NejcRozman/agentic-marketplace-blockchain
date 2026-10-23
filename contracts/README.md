# Contracts

`src/ReverseAuction.sol` is the sealed-bid commit-reveal reverse auction the market
runs on. A consumer creates an auction with a maximum price, which it escrows in
USDC, and a list of eligible ERC-8004 agents. Providers commit a hash of their bid,
then reveal it. The contract scores each revealed bid by its price and the agent's
reputation from the ERC-8004 Reputation Registry, weighted by the auction's
reputation weight, and the best score wins. When the winner completes the job,
with an ERC-8004 feedback authorisation, the contract pays it the winning bid and
refunds the rest of the escrow; the consumer then rates the winner in the
Reputation Registry.

`src/interfaces/` holds the subset of the ERC-8004 Identity and Reputation Registry
interfaces the contract uses. `script/` holds the forge scripts the experiment
runner calls (Deploy, FundConsumer, RegisterAgent; CreateAuction for manual use).

```
forge build
forge test
```

Solidity 0.8.24, optimizer on (200 runs, via IR); OpenZeppelin v5.4.0 is a git
submodule in `lib/`.
