#!/usr/bin/env python3
"""Fork Sepolia once, then never again (S7 §A.6 P5 / §A.6.2).

Every Anvil in the campaign currently forks Sepolia over Infura. Eight
concurrent runs is eight times the fetch rate against a rate limit the code
already works around, and the evidence that nothing was being reused is on
disk: ``~/.foundry/cache/rpc/sepolia/`` accumulated **4,062 block directories**,
one per unpinned fork head, none of them ever hit twice.

This script forks **once**, drives the ordinary Phase-1 setup so that every
account and contract the experiment touches is pulled into local state — above
all the two ERC-8004 registries, which are the only reason the fork exists at
all (§A.6.2) — and dumps the result. Runs then start with ``--load-state`` and
no ``--fork-url``, which makes them:

* **offline** — zero external RPC calls on the run path, so Infura cannot
  throttle, rate-limit or fail a confirmatory run;
* **identical** — every run begins from the same bytes, which is stronger than
  pinning a block number: the snapshot *is* the state rather than a recipe for
  fetching it;
* **faster to start** — no fork handshake, and Anvil serves from memory.

The ERC-8004 claim is unaffected. The registries in the snapshot are the real
Sepolia contracts, captured with their real code and storage; nothing is
reimplemented or stubbed.

**Where it stops matters.** Setup runs through deploy, funding and agent
registration, and then stops. If it ran auctions, their reputation and cost
history would be baked into the starting state of all 720 runs.

Usage::

    python experiments/scripts/build_chain_snapshot.py experiments/configs/<cfg>.yaml \\
        --output experiments/chain/sepolia_registries.json

Then in the config::

    blockchain:
      state_snapshot: "experiments/chain/sepolia_registries.json"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.runner.experiment_runner import ExperimentRunner  # noqa: E402

logger = logging.getLogger("build_chain_snapshot")

DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "chain" / "sepolia_registries.json"

#: ERC-1967 implementation slot. Both ERC-8004 registries are proxies: the
#: address in config holds ~130 bytes of forwarding code and the actual logic
#: lives in a separate implementation contract. Fetching only the proxy gives a
#: chain where every registry call reverts, so the implementations have to be
#: pulled in too.
ERC1967_IMPLEMENTATION_SLOT = (
    "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
)


def rpc(url: str, method: str, params: list) -> Any:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"{method} failed: {payload['error']}")
    return payload["result"]


def code_size(url: str, address: str) -> int:
    return len(rpc(url, "eth_getCode", [address, "latest"])) // 2 - 1


def selector(signature: str) -> str:
    from eth_utils import keccak

    return "0x" + keccak(text=signature).hex()[:8]


def encoded_uint(value: int) -> str:
    return f"{value:064x}"


#: A read-only entry point per registry, used both to pull the contract into
#: local state and to prove offline that it still works. Both were chosen
#: because their return value is checkable, and because the deployed
#: implementations actually have them: ``exists()`` and ``totalSupply()`` are in
#: this repo's ``IIdentityRegistry`` but revert on the contract deployed at the
#: configured address, so the interface here is not a safe guide to the ABI.
PROBE_CALLS = {
    "identity_registry": "ownerOf(uint256)",
    "reputation_registry": "getIdentityRegistry()",
}


def prefetch_registries(
    url: str, registries: Mapping[str, str], known_agent_id: int
) -> Dict[str, str]:
    """Pull the registries *and what they delegate to* into local state.

    Anvil fetches fork state lazily, so an account nobody touched is simply
    absent from the dump. Setup registers agents, which exercises the identity
    registry — but nothing touches the reputation registry until the first
    feedback submission, and that happens during auctions, after this script
    deliberately stops.

    **The prefetch has to be a transaction, not a call.** Measured on this
    fork: ``eth_getCode`` and ``eth_getStorageAt`` leave nothing in the dump,
    and ``anvil_setBalance`` captures the code but *not* the storage — which is
    worse than useless for a proxy, because the ERC-1967 implementation pointer
    lives in storage. A transaction that executes a view function captures the
    code, the pointer, and the implementation account together.

    Returns proxy -> implementation for the manifest.
    """
    accounts = rpc(url, "eth_accounts", [])
    sender = accounts[0]
    implementations: Dict[str, str] = {}
    pending: List[str] = []

    for name, address in registries.items():
        size = code_size(url, address)
        logger.info("  prefetch %s (%s): %d code bytes", name, address, size)
        if size == 0:
            raise RuntimeError(f"{address} has no code on the fork; wrong address or block?")

        signature = PROBE_CALLS[name]
        argument = encoded_uint(known_agent_id) if "uint256" in signature else ""
        pending.append(rpc(url, "eth_sendTransaction", [{
            "from": sender,
            "to": address,
            "data": selector(signature) + argument,
            "gas": "0x100000",
        }]))
        logger.info("    sent %s to capture code, storage and delegate", signature)

        slot = rpc(url, "eth_getStorageAt", [address, ERC1967_IMPLEMENTATION_SLOT, "latest"])
        if int(slot, 16) == 0:
            continue
        implementation = "0x" + slot[-40:]
        logger.info(
            "    -> ERC-1967 implementation %s: %d bytes",
            implementation, code_size(url, implementation),
        )
        implementations[address] = implementation

    # The node mines on a timer, so a sent transaction has not executed yet and
    # a dump taken now would contain none of what it touches. This is exactly
    # how the first build came out empty.
    wait_for_receipts(url, pending)
    return implementations


def wait_for_receipts(url: str, tx_hashes: Sequence[str], timeout_seconds: int = 60) -> None:
    """Block until every transaction is mined."""
    import time

    deadline = time.time() + timeout_seconds
    for tx_hash in tx_hashes:
        while True:
            receipt = rpc(url, "eth_getTransactionReceipt", [tx_hash])
            if receipt is not None:
                break
            if time.time() > deadline:
                raise RuntimeError(f"prefetch transaction {tx_hash} was never mined")
            time.sleep(0.5)
    logger.info("  %d prefetch transaction(s) mined", len(tx_hashes))


def verify_snapshot(
    snapshot: Path,
    required: Sequence[str],
    registries: Mapping[str, str],
    known_agent_id: int,
    port: int = 8599,
) -> None:
    """Load the snapshot offline and prove the registries actually work.

    Code presence alone is not enough: a proxy whose code was captured without
    its storage delegates to address zero, so every call reverts while
    ``eth_getCode`` looks healthy. The functional check below is the one that
    would have caught that — ``getIdentityRegistry()`` must come back with the
    identity registry's address, which means the proxy resolved its
    implementation and read its own storage, entirely offline.

    Without this the failure surfaces mid-run, hours in. The check costs seconds
    and turns a landmine into a build error.
    """
    logger.info("=== Verifying snapshot offline (no fork) ===")
    process = subprocess.Popen(
        ["anvil", "--port", str(port), "--chain-id", "9999", "--accounts", "1",
         "--load-state", str(snapshot)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
    )
    url = f"http://localhost:{port}"
    try:
        for _ in range(30):
            try:
                rpc(url, "eth_blockNumber", [])
                break
            except Exception:
                import time
                time.sleep(1)
        else:
            raise RuntimeError("verification Anvil did not start")

        missing = []
        for address in required:
            size = code_size(url, address)
            logger.info("  code at %s: %d bytes", address, size)
            if size == 0:
                missing.append(address)
        if missing:
            raise RuntimeError(
                "snapshot is incomplete — no code at " + ", ".join(missing) + ". "
                "Every call to these addresses would revert. Extend the warm-up "
                "so they are touched before the dump."
            )

        # Functional check: the proxies must resolve their implementations and
        # read their own storage with no fork behind them.
        identity = registries["identity_registry"]
        reputation = registries["reputation_registry"]

        returned = rpc(url, "eth_call", [
            {"to": reputation, "data": selector("getIdentityRegistry()")}, "latest",
        ])
        resolved = "0x" + returned[-40:]
        logger.info("  reputation.getIdentityRegistry() -> %s", resolved)
        if resolved.lower() != identity.lower():
            raise RuntimeError(
                f"offline call through the reputation proxy returned {resolved}, "
                f"expected {identity}. The proxy's ERC-1967 storage did not make it "
                f"into the snapshot, so every registry call would delegate to "
                f"address zero."
            )

        owner = rpc(url, "eth_call", [
            {"to": identity,
             "data": selector("ownerOf(uint256)") + encoded_uint(known_agent_id)}, "latest",
        ])
        logger.info("  identity.ownerOf(%d) -> 0x%s", known_agent_id, owner[-40:])
        if int(owner, 16) == 0:
            raise RuntimeError(
                f"agent {known_agent_id} has no owner offline; the identity "
                f"registry's storage did not survive into the snapshot."
            )
    finally:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
    logger.info("✓ Snapshot verified: every required contract is present offline")


async def build(config_path: Path, output: Path, run_index: int) -> Path:
    """Fork, set up, dump. Returns the snapshot path."""
    runner = ExperimentRunner(config_path, run_index=run_index)
    runner.load_config()

    if runner.state_snapshot is not None:
        raise SystemExit(
            f"{config_path.name} already sets blockchain.state_snapshot "
            f"({runner.state_snapshot}); building a snapshot from a snapshot would "
            f"bake one run's setup into the next. Point this at a forking config."
        )
    if runner.fork_block_number is None:
        raise SystemExit(
            "blockchain.fork_block_number must be pinned before snapshotting, or "
            "the snapshot records an arbitrary Sepolia head (D8)."
        )

    # Anvil writes the dump on SIGTERM, so the destination is fixed at launch.
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_path = output.with_suffix(".raw.json")

    original_build = runner.build_anvil_command

    def command_with_dump():
        return original_build() + ["--dump-state", str(dump_path)]

    runner.build_anvil_command = command_with_dump

    try:
        logger.info("=== Forking Sepolia at block %s (once) ===", runner.fork_block_number)
        runner.cleanup_orphaned_processes()
        await runner.start_anvil()

        logger.info("=== Running setup so the touched state is captured ===")
        await runner.deploy_contracts()
        await runner.fund_consumer_account()
        await runner.register_agents()

        logger.info("=== Prefetching the ERC-8004 registries and their implementations ===")
        registries = {
            "identity_registry": runner.config["blockchain"]["contracts"]["identity_registry"],
            "reputation_registry": runner.config["blockchain"]["contracts"]["reputation_registry"],
        }
        implementations = prefetch_registries(
            runner.rpc_url, registries, int(runner.consumer_agent_id)
        )
        known_agent_id = int(runner.consumer_agent_id)

        logger.info("=== Setup complete; stopping before any auction runs ===")

        # SIGTERM rather than SIGKILL: the dump happens on graceful shutdown.
        pid = runner.anvil_process.pid
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        runner.anvil_process.process.wait(timeout=60)
        runner.anvil_process = None
    finally:
        runner.cleanup()

    if not dump_path.exists():
        raise SystemExit(f"Anvil exited without writing {dump_path}")

    shutil.move(str(dump_path), str(output))

    # Everything a run will call must be present offline, or the snapshot is a
    # chain where those calls revert.
    verify_snapshot(
        output,
        list(registries.values()) + list(implementations.values()),
        registries,
        known_agent_id,
    )

    digest = hashlib.sha256(output.read_bytes()).hexdigest()

    manifest = {
        "snapshot": output.name,
        "sha256": digest,
        "size_bytes": output.stat().st_size,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_config": str(config_path),
        "fork_network": runner.config["blockchain"]["fork_network"],
        "fork_block_number": runner.fork_block_number,
        "dumped_at_chain_id": runner.chain_id,
        # Recorded because these are what the setup produced; a run loading the
        # snapshot redeploys its own, so these are provenance, not addresses to
        # reuse.
        "identity_registry": registries["identity_registry"],
        "reputation_registry": registries["reputation_registry"],
        # Both registries are ERC-1967 proxies; these are the contracts their
        # calls actually execute, and the snapshot is useless without them.
        "erc1967_implementations": implementations,
        "verification_agent_id": known_agent_id,
        "verified_offline": True,
    }
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("✓ Snapshot written: %s (%.1f MB)", output, output.stat().st_size / 1e6)
    logger.info("  sha256: %s", digest)
    logger.info("  manifest: %s", manifest_path)
    logger.info("")
    logger.info("Add to every S7 config:")
    logger.info("  blockchain:")
    try:
        shown = output.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = output
    logger.info("    state_snapshot: \"%s\"", shown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="A config that still forks (not one using a snapshot)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-index", type=int, default=0,
        help="Port/chain-id index to build on; only matters if a run is already live.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(build(args.config, args.output, args.run_index))


if __name__ == "__main__":
    main()
