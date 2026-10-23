#!/usr/bin/env python3
"""Gas used by each marketplace transaction, against the real ERC-8004 registries.

Starts an offline Anvil from the experiment's chain snapshot (the Sepolia
ERC-8004 Identity and Reputation registries with their real code and storage),
deploys a fresh mock USDC and ReverseAuction from `contracts/out`, and drives
complete auctions: register -> approve -> createAuction -> commitBid ->
revealBid -> endAuction -> completeService -> giveFeedback, plus refundBuyer
for an auction with no bids. Gas is `gasUsed` from the transaction receipts
(intrinsic and calldata cost included).

revealBid reads the provider's reputation summary, and the registry's
getSummary loops over every feedback record, so its gas is measured as a
function of the number of prior ratings the provider has received (all from
one client, as in the experiments).

No LLM or external RPC calls. Requires `forge build` in contracts/ first.

    .venv/bin/python -m experiments.scripts.measure_gas [--port 8611] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import warnings
from pathlib import Path

from eth_abi import encode
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

# event decoding sees unrelated logs (e.g. token transfers) in the same receipt
warnings.filterwarnings("ignore", message=".*MismatchedABI.*")

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "experiments/chain/sepolia_registries.json"
MANIFEST = json.loads(SNAPSHOT.with_suffix(".manifest.json").read_text())
OUT = REPO / "contracts/out"
HISTORY = [0, 1, 10, 30, 100]          # prior ratings of the revealing provider
MAX_PRICE = 1_000_000                  # 1 USDC (6 decimals)
REP_DEFAULT = 80
# fixed throwaway keys, funded with anvil_setBalance
KEYS = [Web3.keccak(text=f"measure_gas account {i}") for i in range(4)]
REP_ABI = [{"type": "function", "name": "giveFeedback", "stateMutability": "nonpayable", "outputs": [],
            "inputs": [{"name": "agentId", "type": "uint256"}, {"name": "value", "type": "int128"},
                       {"name": "valueDecimals", "type": "uint8"}, {"name": "tag1", "type": "string"},
                       {"name": "tag2", "type": "string"}, {"name": "endpoint", "type": "string"},
                       {"name": "feedbackURI", "type": "string"}, {"name": "feedbackHash", "type": "bytes32"}]}]
ID_ABI = [{"type": "function", "name": "register", "stateMutability": "nonpayable",
           "inputs": [], "outputs": [{"name": "agentId", "type": "uint256"}]},
          {"type": "event", "name": "Transfer", "anonymous": False,
           "inputs": [{"name": "from", "type": "address", "indexed": True},
                      {"name": "to", "type": "address", "indexed": True},
                      {"name": "tokenId", "type": "uint256", "indexed": True}]}]


def artifact(name):
    a = json.loads((OUT / f"{name}.sol" / f"{name}.json").read_text())
    return a["abi"], a["bytecode"]["object"]


class Chain:
    def __init__(self, port):
        self.proc = subprocess.Popen(
            ["anvil", "--port", str(port), "--chain-id", "8000", "--load-state", str(SNAPSHOT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        self.w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}"))
        for _ in range(100):
            if self.w3.is_connected():
                break
            time.sleep(0.1)
        self.accts = [Account.from_key(k) for k in KEYS]
        for a in self.accts:
            self.w3.provider.make_request("anvil_setBalance", [a.address, hex(10**21)])

    def send(self, acct, fn=None, deploy=None):
        tx = (deploy or fn).build_transaction({"from": acct.address, "nonce": self.w3.eth.get_transaction_count(acct.address),
                                               "gas": 8_000_000, "gasPrice": self.w3.eth.gas_price})
        h = self.w3.eth.send_raw_transaction(acct.sign_transaction(tx).raw_transaction)
        r = self.w3.eth.wait_for_transaction_receipt(h)
        assert r.status == 1, f"reverted: {fn or deploy}"
        return r

    def warp(self, seconds):
        self.w3.provider.make_request("evm_increaseTime", [seconds])
        self.w3.provider.make_request("evm_mine", [])

    def close(self):
        os.killpg(self.proc.pid, signal.SIGTERM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    c = Chain(args.port)
    w3 = c.w3
    buyer, p1, p2, deployer = c.accts[0], c.accts[1], c.accts[2], c.accts[3]
    try:
        abi, code = artifact("ERC20Mock")
        usdc = w3.eth.contract(address=c.send(deployer, deploy=w3.eth.contract(abi=abi, bytecode=code).constructor()).contractAddress, abi=abi)
        abi, code = artifact("ReverseAuction")
        ident = w3.eth.contract(address=MANIFEST["identity_registry"], abi=ID_ABI)
        rep = w3.eth.contract(address=MANIFEST["reputation_registry"], abi=REP_ABI)
        auc = w3.eth.contract(address=c.send(deployer, deploy=w3.eth.contract(abi=abi, bytecode=code).constructor(
            usdc.address, ident.address, rep.address, REP_DEFAULT)).contractAddress, abi=abi)
        c.send(deployer, usdc.functions.mint(buyer.address, 10**15))
        chain_id = w3.eth.chain_id

        gas = {}

        def record(op, r):
            gas.setdefault(op, []).append(r.gasUsed)

        def register(owner):
            r = c.send(owner, ident.functions.register())
            record("register (Identity Registry)", r)
            return ident.events.Transfer().process_receipt(r)[0]["args"]["tokenId"]

        a1, a2 = register(p1), register(p2)
        record("approve USDC (buyer)", c.send(buyer, usdc.functions.approve(auc.address, 2**255)))

        def give_feedback(agent, rating):
            return c.send(buyer, rep.functions.giveFeedback(agent, rating * 100, 2, "", "", "", "", b"\0" * 32))

        def auction(bids):
            """One auction; bids = {(owner, agentId): amount}. Returns (auctionId, receipts by op)."""
            r = c.send(buyer, auc.functions.createAuction("bafy" + "x" * 55, MAX_PRICE, 60, 60, [a1, a2], 20))
            aid = auc.events.AuctionCreated().process_receipt(r)[0]["args"]["auctionId"]
            ops = {"createAuction": r, "commitBid": [], "revealBid": []}
            nonces = {}
            for (owner, agent), amt in bids.items():
                nonce = os.urandom(32)
                nonces[agent] = nonce
                h = Web3.keccak(encode(["uint256", "uint256", "uint256", "bytes32", "address"],
                                       [aid, agent, amt, nonce, owner.address]))
                ops["commitBid"].append(c.send(owner, auc.functions.commitBid(aid, agent, h)))
            c.warp(61)
            for (owner, agent), amt in bids.items():
                ops["revealBid"].append(c.send(owner, auc.functions.revealBid(aid, agent, amt, nonces[agent])))
            c.warp(61)
            ops["endAuction"] = c.send(buyer, auc.functions.endAuction(aid))
            return aid, ops

        def complete(aid, owner, agent):
            exp = w3.eth.get_block("latest")["timestamp"] + 3600
            params = encode(["uint256", "address", "uint64", "uint256", "uint256", "address", "address"],
                            [agent, buyer.address, 1000, exp, chain_id, ident.address, owner.address])
            sig = Account.sign_message(encode_defunct(Web3.keccak(params)), owner.key).signature
            return c.send(owner, auc.functions.completeService(aid, params + bytes(sig)))

        # history-dependent reveal: p1 accumulates ratings between measurements
        reveal_by_history = {}
        given = 0
        for n in HISTORY:
            while given < n:
                give_feedback(a1, 85)
                given += 1
            aid, ops = auction({(p1, a1): 400_000, (p2, a2): 500_000})
            reveal_by_history[n] = ops["revealBid"][0].gasUsed
            for op in ("createAuction", "endAuction"):
                record(op, ops[op])
            for r in ops["commitBid"]:
                record("commitBid", r)
            record("completeService", complete(aid, p1, a1))
            record("giveFeedback (Reputation Registry)", give_feedback(a1, 85))
            given += 1

        r = c.send(buyer, auc.functions.createAuction("bafy" + "x" * 55, MAX_PRICE, 60, 60, [a1, a2], 20))
        aid = auc.events.AuctionCreated().process_receipt(r)[0]["args"]["auctionId"]
        c.warp(130)
        c.send(buyer, auc.functions.endAuction(aid))
        record("refundBuyer (no bids)", c.send(buyer, auc.functions.refundBuyer(aid)))
    finally:
        c.close()

    print(f"{'operation':<38}{'caller':<10}{'gas (median)':>14}{'min':>10}{'max':>10}")
    callers = {"register": "provider", "approve": "consumer", "createAuction": "consumer", "commitBid": "provider",
               "endAuction": "consumer", "completeService": "provider", "giveFeedback": "consumer", "refundBuyer": "consumer"}
    for op, xs in gas.items():
        xs = sorted(xs)
        who = next(v for k, v in callers.items() if op.startswith(k))
        print(f"{op:<38}{who:<10}{xs[len(xs) // 2]:>14,}{xs[0]:>10,}{xs[-1]:>10,}")
    print("\nrevealBid (provider) by prior ratings of the revealing agent:")
    for n, g in reveal_by_history.items():
        print(f"  {n:>4} ratings: {g:>9,} gas")
    if args.json:
        args.json.write_text(json.dumps({"ops": gas, "reveal_by_history": reveal_by_history}, indent=2))


if __name__ == "__main__":
    main()
