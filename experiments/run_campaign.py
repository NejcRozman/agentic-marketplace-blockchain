#!/usr/bin/env python3
"""Run (condition, seed) pairs, resumably, and record each in the ledger.

The unit of work is one (condition, seed) pair. Every finished pair appends one
line to the ledger (`data/ledger.jsonl`): condition, seed, run id, git SHA and
whether the run passed the acceptance checks. Pairs already recorded as `ok` are
skipped unless `--force`, so a campaign can be stopped and resumed at any time.

A run enters the ledger as `ok` only if it actually produced a market (agents
started, bids committed, no replay failures or late commits); a crashed agent
can otherwise produce a complete-looking results table.

    python experiments/run_campaign.py --seeds 1007 1008 --dry-run
    python experiments/run_campaign.py --conditions A_gptoss20b_xd_vs_h --seeds 1007 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from agents.common import paths  # noqa: E402

CONDITIONS_DIR = paths.CONDITIONS_DIR
LEDGER = paths.LEDGER
RUNNER = REPO_ROOT / "experiments" / "runner" / "experiment_runner.py"

#: Seconds between the startups of concurrent runs. Several runners loading the
#: chain snapshot at the same instant failed at startup; two did not.
STARTUP_STAGGER_SECONDS = 6.0

RUN_ID_PATTERN = re.compile(r"Run ID:\s*(\S+)")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def completed_pairs() -> set:
    if not LEDGER.exists():
        return set()
    done = set()
    for line in LEDGER.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok":
            done.add((row["cell"], int(row["seed"]), int(row.get("repeat", 1))))
    return done


def record(entry: Dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def check_run(log_dir: Path, run_id: Optional[str]) -> Dict[str, object]:
    """Did this run produce a market? Counted from its own agent logs.

    The run directory is found by the run id the runner printed, not by "the
    newest directory": under parallelism that belongs to whichever run finished
    last.
    """
    if run_id:
        latest = log_dir / run_id
        if not latest.exists():
            return {"ok": False, "reason": f"run directory {run_id} missing"}
    else:
        runs = sorted(log_dir.glob("*/"), key=lambda p: p.stat().st_mtime) if log_dir.exists() else []
        if not runs:
            return {"ok": False, "reason": "no run directory"}
        latest = runs[-1]

    provider_logs = list(latest.glob("provider_agent_*.log"))
    text = "\n".join(p.read_text(errors="ignore") for p in provider_logs)
    consumer = "\n".join(p.read_text(errors="ignore") for p in latest.glob("consumer_*.log"))

    checks = {
        "providers_started": text.count("Orchestrator started"),
        "auctions_created": consumer.count("✅ Created auction"),
        "bids": text.count("Bid committed successfully"),
        "reveals": text.count("Bid revealed successfully"),
        "late_commits": text.count("LATE COMMIT"),
        "missed_reveals": text.count("Skipping reveal"),
        "replay_failures": text.count("replay data missing"),
    }
    failures = []
    if checks["providers_started"] < len(provider_logs) or not provider_logs:
        failures.append("an arm never started")
    if checks["bids"] == 0:
        failures.append("no bids were committed")
    if checks["replay_failures"]:
        failures.append(f"{checks['replay_failures']} replay failures")
    if checks["late_commits"]:
        failures.append(f"{checks['late_commits']} late commits")

    checks["ok"] = not failures
    checks["reason"] = "; ".join(failures) if failures else "ok"
    # Relative to the data directory, which paths.resolve maps back ("data/...").
    checks["run_dir"] = "data/" + str(latest.relative_to(paths.DATA_DIR))
    return checks


def _pids_on_port(port: int) -> List[int]:
    """PIDs listening on `port`, via `ss`. Empty when nothing holds it."""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = set()
    for line in out.splitlines():
        if f":{port} " not in line and not line.split()[3:4] == [f"0.0.0.0:{port}"]:
            if f":{port}" not in line:
                continue
        for token in re.findall(r"pid=(\d+)", line):
            pids.add(int(token))
    return sorted(pids)


def _listening_pids() -> Set[int]:
    """Every PID holding a listening TCP socket."""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(t) for t in re.findall(r"pid=(\d+)", out)}


def _pids_bound_to_rpc(port: int) -> List[int]:
    """Agent processes attached to this slot's chain, found via their environment.

    Agents are RPC clients and never listen, so a port sweep misses them; an
    orphaned consumer would keep creating auctions on the next run's chain. The
    runner exports `BLOCKCHAIN_RPC_URL` to each agent, which identifies the chain
    it is attached to. Listeners are excluded: Anvil inherits the default
    `BLOCKCHAIN_RPC_URL` of every slot and is identified by its port instead.
    """
    needle = f"BLOCKCHAIN_RPC_URL=http://localhost:{port}".encode()
    listeners = _listening_pids()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in listeners:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                if needle in fh.read():
                    found.append(pid)
        except OSError:
            continue
    return found


def free_slot(run_index: int) -> None:
    """Stop whatever still holds this slot's chain before a run starts on it.

    A runner killed by the timeout cannot clean up, and its Anvil and agents run
    in their own process groups and outlive it. Processes are found by port and
    by the chain they are attached to, never by name, so runs in other slots are
    left alone.
    """
    port = 8545 + run_index

    def occupants() -> List[int]:
        return sorted(set(_pids_on_port(port)) | set(_pids_bound_to_rpc(port)))

    for pid in occupants():
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.time() + 10
    while occupants() and time.time() < deadline:
        time.sleep(0.5)
    for pid in occupants():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.5)


class SlotPool:
    """Hands out isolation slots (port and chain), each held for exactly one run."""

    def __init__(self, size: int) -> None:
        self._free = list(range(size))
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(size)

    def acquire(self) -> int:
        self._sem.acquire()
        with self._lock:
            return self._free.pop(0)

    def release(self, slot: int) -> None:
        with self._lock:
            self._free.append(slot)
        self._sem.release()


def _run_with_slot(slots: SlotPool, cell: str, seed: int, timeout: int, repeat: int) -> Dict:
    slot = slots.acquire()
    try:
        return run_one(cell, seed, slot, timeout, repeat)
    finally:
        slots.release(slot)


def run_one(cell: str, seed: int, run_index: int, timeout: int, repeat: int = 1) -> Dict:
    config = CONDITIONS_DIR / f"{cell}.yaml"
    log_dir = paths.RUN_LOGS_DIR / cell
    if run_index:
        time.sleep(run_index * STARTUP_STAGGER_SECONDS)
    # Only before the run: freeing the slot afterwards could kill the next run
    # that has already claimed it.
    free_slot(run_index)
    started = datetime.now(timezone.utc)

    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER), str(config),
             "--run-seed", str(seed), "--run-index", str(run_index)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        # Keep whatever the runner printed before it was killed.
        def _text(blob) -> str:
            if blob is None:
                return ""
            return blob.decode("utf-8", "replace") if isinstance(blob, bytes) else str(blob)
        crash = log_dir / f"_timeout_seed{seed}_r{repeat}_slot{run_index}.log"
        try:
            crash.parent.mkdir(parents=True, exist_ok=True)
            crash.write_text((_text(expired.stdout) + _text(expired.stderr))[-20000:])
        except OSError:
            pass
        raise
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        crash = log_dir / f"_failed_seed{seed}_r{repeat}_slot{run_index}.log"
        try:
            crash.parent.mkdir(parents=True, exist_ok=True)
            crash.write_text(combined[-20000:])
        except OSError:
            pass
    # Which branch ended the runner's monitoring loop: target reached, timeout,
    # or the consumer exiting.
    exit_match = re.search(r"Monitoring loop ended: (\S+)", combined)
    drain_match = re.search(r"Drained (\d+) in-flight auction", combined)
    match = RUN_ID_PATTERN.search(combined)
    run_id = match.group(1) if match else None
    checks = check_run(log_dir, run_id)
    entry = {
        "cell": cell, "seed": seed, "repeat": repeat, "run_index": run_index,
        "status": "ok" if (proc.returncode == 0 and checks["ok"]) else "failed",
        "exit_code": proc.returncode,
        "run_id": run_id,
        "git_sha": git_sha(),
        "monitor_exit": exit_match.group(1) if exit_match else None,
        "drained": int(drain_match.group(1)) if drain_match else 0,
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        **{k: v for k, v in checks.items() if k != "ok"},
    }
    record(entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", help="condition ids (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1,
                        help="run each (condition, seed) this many times, to measure run-to-run noise")
    parser.add_argument("--parallel", type=int, default=1,
                        help="concurrent runs; each gets its own port and chain id")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--force", action="store_true", help="re-run pairs already marked ok")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cells = sorted(p.stem for p in CONDITIONS_DIR.glob("*.yaml") if p.stem != "base")
    if not cells:
        raise SystemExit("no conditions; run python -m experiments.generate_conditions first")
    if args.conditions:
        unknown = sorted(set(args.conditions) - set(cells))
        if unknown:
            raise SystemExit(f"unknown condition(s): {', '.join(unknown)}")
        cells = [c for c in cells if c in args.conditions]

    done = set() if args.force else completed_pairs()
    wanted = [
        (cell, seed, repeat)
        for cell in cells for seed in args.seeds
        for repeat in range(1, args.repeats + 1)
        if (cell, seed, repeat) not in done
    ]

    total = len(cells) * len(args.seeds) * args.repeats
    print(f"conditions: {len(cells)}  seeds: {len(args.seeds)}  repeats: {args.repeats}  "
          f"runs wanted: {total}  already done: {total - len(wanted)}  "
          f"to run: {len(wanted)}")
    if args.dry_run or not wanted:
        for cell, seed, repeat in wanted[:20]:
            print(f"  would run {cell} seed={seed} repeat={repeat}")
        if len(wanted) > 20:
            print(f"  ... and {len(wanted) - 20} more")
        return

    results: List[Dict] = []
    slots = SlotPool(args.parallel)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(_run_with_slot, slots, cell, seed, args.timeout, repeat)
            for cell, seed, repeat in wanted
        ]
        for future, (cell, seed, repeat) in zip(futures, wanted):
            # One failed run is recorded and the campaign continues; a recorded
            # failure is retried on the next invocation.
            try:
                entry = future.result()
            except Exception as exc:
                entry = {
                    "cell": cell, "seed": seed, "repeat": repeat,
                    "status": "failed", "seconds": 0,
                    "reason": f"{type(exc).__name__}: {exc}"[:200],
                }
                record(entry)
            results.append(entry)
            flag = "✓" if entry["status"] == "ok" else "✗"
            print(f"  {flag} {entry['cell']} seed={entry['seed']}"
                  f"{'' if args.repeats == 1 else ' r' + str(entry['repeat'])} "
                  f"{entry['seconds']}s — {entry['reason']}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} ok · ledger: {LEDGER}")


if __name__ == "__main__":
    main()
