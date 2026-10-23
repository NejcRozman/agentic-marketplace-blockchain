#!/usr/bin/env python3
"""S7 §D — run the matrix, resumably.

The campaign will not be one long session. It runs block by block, pauses
between seeds, and sometimes a single `(cell, seed)` has to be repeated. So the
unit of work is one `(cell, seed)` pair and there is a durable ledger of what
finished:

    data/ledger.jsonl

Every completed pair appends one line — cell, seed, run id, **git SHA**, and
whether it passed the acceptance checks. The launcher runs `wanted - completed`,
so stopping and resuming needs no bookkeeping from you, and re-running a pair
that already succeeded is skipped unless `--force`.

The git SHA matters more than it looks: a pause of days means code can move
underneath the campaign, and "which commit produced this cell" is unanswerable
afterwards without it.

**Acceptance is checked per run, not at the end.** A run that produced 30
auctions with a crashed arm looks exactly like a healthy one in the summary —
that failure has happened three times in this project — so a run only enters the
ledger as `ok` if it actually bid.

    python experiments/scripts/run_s9_campaign.py --block A --seeds 1007 1008 --dry-run
    python experiments/scripts/run_s9_campaign.py --block A --seeds 1007 1008 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from agents.common import paths  # noqa: E402

CELL_DIR = paths.CONDITIONS_DIR
LEDGER = paths.LEDGER
RUNNER = REPO_ROOT / "experiments" / "runner" / "experiment_runner.py"


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


RUN_ID_PATTERN = re.compile(r"Run ID:\s*(\S+)")


def check_run(cell: str, log_dir: Path, run_id: Optional[str]) -> Dict[str, object]:
    """Positive assertions: did *this* run produce a market?

    Absence of an exception is not evidence — a crashed arm, a pattern kill and
    a missing dataset each produced a full-looking results table in this project.

    ``run_id`` is parsed from the runner's own stdout rather than inferred from
    "the newest directory". Under parallelism the newest directory belongs to
    whichever run finished last, so inferring it attributed three different
    seeds to one run and produced three identical differences in the pilot.
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
    checks["run_dir"] = str(latest.relative_to(REPO_ROOT))
    return checks


#: Seconds of separation between concurrent run startups.
#:
#: Launching five runs at once failed half of them in under two seconds, with an
#: empty run directory and not even an ``anvil.log`` -- five ``anvil
#: --load-state`` processes opening the same snapshot while five runners raced
#: their readiness checks. Two concurrent runs were unaffected. Staggering by
#: slot removes the simultaneity without giving up the concurrency.
STARTUP_STAGGER_SECONDS = float(__import__('os').environ.get('S7_STAGGER', 6.0))



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
    """Every PID holding a listening TCP socket.

    Used to keep the environment sweep off servers. Anvil is a *listener*; the
    agents the sweep is looking for are pure clients.
    """
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(t) for t in re.findall(r"pid=(\d+)", out)}


def _pids_bound_to_rpc(port: int) -> List[int]:
    """Agent processes pointed at this slot's chain, found via their environment.

    Agents are RPC *clients*: they never listen, so a port-based sweep misses
    them entirely. An orphaned consumer left over from an earlier batch will
    happily reconnect to the fresh Anvil on the same port and keep creating
    auctions — which is exactly how seeds 1009 and 1013 came to bid on 58
    auctions when their own consumer created 30.

    `/proc/<pid>/environ` carries the `BLOCKCHAIN_RPC_URL` the runner exported,
    so it identifies the chain a process is attached to without matching on
    command-line patterns (the S7 §A.6 P1 defect).

    Listeners are excluded, and that exclusion is load-bearing. Anvil is spawned
    without a custom `env` — it takes its port on the *command line* — so it
    inherits the ambient `BLOCKCHAIN_RPC_URL`, which is the default 8545 for
    every slot. Without this filter, `free_slot(0)` matched the Anvil of every
    other slot and SIGTERMed the whole campaign: on 2026-08-31 seeds 1008-1011
    lost their chains at the instant seed 1007 finished and freed slot 0, and
    each then polled a dead RPC at 29/30 until the timeout. Identify Anvil by
    the port it listens on (`_pids_on_port`), never by its environment.
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
    """Make sure this slot's Anvil port is actually free before starting.

    `subprocess.run(timeout=...)` SIGKILLs the runner, so the runner's `finally`
    cleanup never runs — and its children are `setsid`-ed into their own process
    groups (S7 §A.6 P1), so they outlive it. The orphaned Anvil keeps listening,
    and every later run on that slot hits the runner's own port guard and dies in
    about a second with "an arm never started". That guard is right; nothing was
    reclaiming the port.

    Killing by port rather than by name is deliberate: a pattern kill is exactly
    the P1 defect that destroys *other* runs in flight.
    """
    port = 8545 + run_index

    def occupants() -> List[int]:
        # Both halves matter: the listener (Anvil) holds the port, and the
        # clients (agents) hold the *market*. Reclaiming only the port produced
        # a fresh chain with a stale consumer still writing to it.
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


def _run_with_slot(slots: "SlotPool", cell: str, seed: int, timeout: int, repeat: int) -> Dict:
    """Hold an isolation slot for exactly this run, then hand it back."""
    slot = slots.acquire()
    try:
        return run_one(cell, seed, slot, timeout, repeat)
    finally:
        slots.release(slot)


class SlotPool:
    """Hands out isolation slots so two live runs can never share one.

    `index % parallel` looked equivalent and is not: the pool hands task *n* to
    whichever worker frees up, not to the worker that ran task *n - parallel*.
    A worker released by task 5 would pick up task 7 and claim slot 2 while task
    2 was still running on it — then `free_slot` SIGTERMed the incumbent. Three
    of ten runs were killed roughly twenty minutes in, each losing an auction,
    and surfaced only as `consumer_exited(code=-2)` with a doubled runtime.

    A slot is held for exactly the lifetime of the run that owns it.
    """

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


def run_one(cell: str, seed: int, run_index: int, timeout: int, repeat: int = 1) -> Dict:
    config = CELL_DIR / f"s9_{cell}.yaml"
    log_dir = paths.RUN_LOGS_DIR / f"s9_{cell}"
    if run_index:
        time.sleep(run_index * STARTUP_STAGGER_SECONDS)
    free_slot(run_index)
    started = datetime.now(timezone.utc)

    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER), str(config),
             "--run-seed", str(seed), "--run-index", str(run_index)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        # A timed-out run is the one most worth reading and the one that used to
        # leave nothing behind: `TimeoutExpired` is raised before `proc` exists,
        # so the non-zero-exit capture below never sees it. The exception carries
        # whatever the runner managed to emit.
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
    # **No post-run `free_slot`.** Reclaiming the slot on the way out races the
    # next run that has already claimed it: the pool hands slot `i` to whichever
    # worker frees up, so an outgoing run's cleanup can kill the *incoming*
    # run's freshly started Anvil — observed as "Anvil process terminated
    # unexpectedly" one second after "Port 8545 is free". The pre-run
    # reclamation above is sufficient and cannot race, because it runs before
    # this slot's chain exists.
    # The runner logs through `logging`, which writes to stderr by default.
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        # The runner's own diagnosis was being discarded: `capture_output=True`
        # collects it, `run_one` only mined it for a run id, and a failing run
        # left an empty directory with nothing to read. Keep it.
        crash = log_dir / f"_failed_seed{seed}_r{repeat}_slot{run_index}.log"
        try:
            crash.parent.mkdir(parents=True, exist_ok=True)
            crash.write_text(combined[-20000:])
        except OSError:
            pass
    # The runner's own stdout was discarded on success, so which branch ended
    # the monitoring loop — target reached, timeout, or the consumer exiting —
    # was unknowable after the fact. It is the difference between "the run
    # finished" and "the run was cut short", and every run finished at 29 of 30
    # auctions without anyone noticing.
    exit_match = re.search(r"Monitoring loop ended: (\S+)", combined)
    drain_match = re.search(r"Drained (\d+) in-flight auction", combined)
    match = RUN_ID_PATTERN.search(combined)
    run_id = match.group(1) if match else None
    checks = check_run(cell, log_dir, run_id)
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
    parser.add_argument("--block", help="only cells from this block (A/B/C/D)")
    parser.add_argument("--cell", help="a single cell id")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Run each (cell, seed) this many times. >1 is the determinism-audit "
             "cell (§E.1 set b): identical seed, repeated, to measure run-to-run "
             "noise. Different seeds measure market variance instead — the two "
             "are separate experiments and one cannot substitute for the other.",
    )
    parser.add_argument("--parallel", type=int, default=1,
                        help="concurrent runs; each gets its own port and chain id (§A.6)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--force", action="store_true", help="re-run pairs already marked ok")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cells = sorted(p.stem[len("s9_"):] for p in CELL_DIR.glob("s9_*.yaml"))
    if not cells:
        raise SystemExit("no cells; run generate_s9_cells.py --write first")
    if args.block:
        cells = [c for c in cells if c.startswith(args.block.upper() + "_")]
    if args.cell:
        cells = [c for c in cells if c == args.cell]

    done = set() if args.force else completed_pairs()
    wanted = [
        (cell, seed, repeat)
        for cell in cells for seed in args.seeds
        for repeat in range(1, args.repeats + 1)
        if (cell, seed, repeat) not in done
    ]

    total = len(cells) * len(args.seeds) * args.repeats
    print(f"cells: {len(cells)}  seeds: {len(args.seeds)}  repeats: {args.repeats}  "
          f"runs wanted: {total}  already done: {total - len(wanted)}  "
          f"to run: {len(wanted)}")
    if args.repeats > 1:
        print("  (determinism audit: identical seed repeated — measures run-to-run "
              "noise, not market variance)")
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
            # One run must not take the campaign down with it. `run_one` raising
            # — a subprocess timeout is the observed case — used to propagate out
            # of `future.result()`, abandoning every queued run *and* leaving the
            # in-flight siblings' anvil and agent processes alive to hold their
            # ports, so everything afterwards failed instantly. The failure is
            # recorded and the sweep continues.
            try:
                entry = future.result()
            except Exception as exc:
                entry = {
                    "cell": cell, "seed": seed, "repeat": repeat,
                    "status": "failed", "seconds": 0,
                    "reason": f"{type(exc).__name__}: {exc}"[:200],
                }
                # Record the failure too. `run_one` writes its own row on the
                # paths it survives, but a timeout or a crash raises straight
                # past it, and the ledger was left with no trace at all: on
                # 2026-08-31 four seeds died and the ledger showed only the
                # three that lived, so the batch looked like a short clean run
                # rather than a damaged one. `completed_runs` counts only
                # `status == "ok"`, so a recorded failure still gets retried.
                record(entry)
            results.append(entry)
            flag = "✓" if entry["status"] == "ok" else "✗"
            print(f"  {flag} {entry['cell']} seed={entry['seed']}"
                  f"{'' if args.repeats == 1 else ' r' + str(entry['repeat'])} "
                  f"{entry['seconds']}s — {entry['reason']}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} ok · ledger: {LEDGER.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
