"""Ledger-backed data access for the paper figures.

The ledger (`experiments/data/s9_ledger.jsonl`) is the sole authority on which
runs belong to a cell: only `status == "ok"` rows count, and each row names its
own run directory. Globbing run directories instead would also pick up runs
that crashed or were superseded — e.g. the eight half-finished gpt-4o-mini runs
killed by the 2026-09-13 reboot.

Arms are identified by provider group name. Cells with several agents per arm
(2v2, 1v3, 3v1) are reduced to the per-agent mean so they stay comparable with
1v1.
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

from agents.common import paths

REPO = paths.REPO_ROOT
LEDGER = paths.LEDGER
N_AUCTIONS = 30

#: Balance snapshots are written as local wall-clock strings; auction end times
#: are Unix epoch. All runs were recorded on a machine in this zone.
SNAPSHOT_TZ = ZoneInfo("Europe/Ljubljana")


def t95(n: int) -> float:
    """Two-sided 95% Student-t quantile for n observations."""
    table = {39: 2.0227, 38: 2.0244, 29: 2.0452, 19: 2.0930}
    return table.get(n - 1, 1.96)


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def mean_ci(values: List[float]) -> Interval:
    n = len(values)
    m = st.mean(values)
    half = t95(n) * st.stdev(values) / math.sqrt(n) if n > 1 else float("nan")
    return Interval(m, m - half, m + half, n)


@lru_cache(maxsize=None)
def ok_runs(cell: str) -> Tuple[dict, ...]:
    rows = []
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("cell") == cell and row.get("status") == "ok":
            rows.append(row)
    seeds = [r["seed"] for r in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{cell}: a seed has more than one ok row")
    if not rows:
        raise ValueError(f"{cell}: no ok runs in the ledger")
    return tuple(sorted(rows, key=lambda r: r["seed"]))


def _metrics(row: dict) -> dict:
    path = paths.resolve(row["run_dir"].replace("/logs/", "/metrics/")) / "experiment_metrics.json"
    return json.loads(path.read_text())


def _groups(row: dict) -> Dict[str, List[int]]:
    """group name -> provider ids, from the run's status files."""
    out: Dict[str, List[int]] = {}
    for f in (paths.resolve(row["run_dir"])).glob("provider_*_status.json"):
        s = json.loads(f.read_text())
        out.setdefault(s["profile"]["group_name"], []).append(int(f.name.split("_")[1]))
    return out


def final_net(cell: str, group: str) -> List[float]:
    """Per-seed final net balance of `group` (per-agent mean), seed-ordered."""
    out = []
    for row in ok_runs(cell):
        ids = _groups(row).get(group)
        if not ids:
            raise ValueError(f"{row['run_dir']}: no group {group!r}")
        nets = []
        for pid in ids:
            s = json.loads((paths.resolve(row["run_dir"]) / f"provider_{pid}_status.json").read_text())
            nets.append(s["financials"]["net_balance"])
        out.append(st.mean(nets))
    return out


def paired_difference(cell: str, a: str, b: str) -> Interval:
    """Mean of per-seed (a − b) final net balance, with a 95% t interval."""
    xa, xb = final_net(cell, a), final_net(cell, b)
    return mean_ci([p - q for p, q in zip(xa, xb)])


# ---------------------------------------------------------------------------
# Significance (Section 5.7): two-sided paired t-test per condition, p-values
# Holm-adjusted within each research question. A condition is one cell; its
# contrast is the X_ group against the other group. The self-play check and the
# two unreported regression variants are in no family. Cells that do not yet
# have all 40 runs are left out until they do.
# ---------------------------------------------------------------------------

NOT_IN_FAMILY = {"S91_selfplay_adaptive", "E_xd_vs_rd_v3", "E_xd_vs_rnaive_v3"}
RUNS_PER_CELL = 40


def family(cell: str) -> str:
    return "RQ1" if cell.startswith("S91") else "RQ2"


def contrast(cell: str) -> Tuple[str, str]:
    groups = sorted(_groups(ok_runs(cell)[0]))
    x = next(g for g in groups if g.startswith("X_"))
    return x, next(g for g in groups if g != x)


def _holm(ps: List[float]) -> List[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


@lru_cache(maxsize=None)
def holm_table() -> Dict[str, Tuple[float, float]]:
    """cell -> (unadjusted p, Holm-adjusted p within its research question)."""
    from scipy import stats

    cells = sorted({json.loads(l)["cell"] for l in LEDGER.read_text().splitlines()
                    if l.strip() and json.loads(l).get("status") == "ok"} - NOT_IN_FAMILY)
    cells = [c for c in cells if len(ok_runs(c)) == RUNS_PER_CELL]
    raw: Dict[str, float] = {}
    for c in cells:
        x, b = contrast(c)
        d = [p - q for p, q in zip(final_net(c, x), final_net(c, b))]
        raw[c] = float(stats.ttest_1samp(d, 0.0).pvalue)
    out: Dict[str, Tuple[float, float]] = {}
    for fam in ("RQ1", "RQ2"):
        members = [c for c in cells if family(c) == fam]
        for c, pa in zip(members, _holm([raw[c] for c in members])):
            out[c] = (raw[c], pa)
    return out


def significant(cell: str) -> bool:
    """Holm-adjusted p < 0.05 for the cell's contrast (direction-free)."""
    return holm_table()[cell][1] < 0.05


_HEUR_BID = re.compile(r"auction (\d+) bid=(\d+) margin=([0-9.]+) cost_basis=")
_TOOL_AUDIT = re.compile(r"TOOL_AUDIT (\{.*\})\s*$")


@dataclass
class Bid:
    auction: int
    margin: float        # bid / cost basis − 1, the quantity each strategy actually chooses
    won: bool


def bid_sequences(cell: str, group: str) -> List[List[Bid]]:
    """Per seed, the agent's committed bids in auction order with the margin it chose.

    Heuristic agents log their margin directly. LLM agents write one TOOL_AUDIT
    line per place_bid call, holding the bid and the cost estimates in force for
    that decision; the cost basis is execution estimate + per-participation
    overhead, the value shown as total_cost_basis_micro_usdc in that call's
    prompt. (The prompt itself is printed in the reasoning trace *after* the
    bid line, so pairing a bid with the nearest preceding prompt would use the
    previous auction's cost basis.)

    Only bids that entered the auction count: a bid is kept if the agent
    appears in the auction's bids_on_chain, which lists revealed bids. Bids
    rejected because the commit phase had closed or the price exceeded the
    maximum never entered the auction; a committed bid whose reveal was missed
    did not compete either. 1v1 cells only (the agent is the group's single
    provider).
    """
    out = []
    for row in ok_runs(cell):
        ids = _groups(row).get(group)
        if not ids or len(ids) != 1:
            raise ValueError(f"{row['run_dir']}: expected one agent in {group!r}")
        pid = ids[0]
        auctions = _metrics(row)["auctions"]
        winner = {int(a["auction_id"]): a.get("winner_id") for a in auctions}
        on_chain = {int(a["auction_id"]): {b["agent_id"]: b["amount"] for b in a.get("bids_on_chain") or []}
                    for a in auctions}
        bids: Dict[int, Tuple[int, float]] = {}
        for line in (paths.resolve(row["run_dir"]) / f"provider_agent_{pid}.log").open(errors="ignore"):
            m = _HEUR_BID.search(line)
            if m:
                bids[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
                continue
            m = _TOOL_AUDIT.search(line)
            if m:
                audit = json.loads(m.group(1))
                if audit.get("tool") != "place_bid":
                    continue
                args, state = audit["llm_args"], audit["current_state"]
                # Rounded per component, exactly as the prompt computes
                # total_cost_basis_micro_usdc (the *_micro fields are truncated).
                basis = (round(state["estimated_execution_cost_usd"] * 1e6)
                         + round(state["non_execution_cost_per_participation_usd"] * 1e6))
                amount = int(args["bid_amount"])
                bids[int(args["auction_id"])] = (amount, amount / basis - 1.0)
        seq = []
        for a, (amount, margin) in sorted(bids.items()):
            committed = on_chain.get(a, {})
            if pid not in committed:
                continue
            if committed[pid] != amount:
                raise ValueError(f"{row['run_dir']}: auction {a} logged bid {amount} "
                                 f"but {committed[pid]} on chain")
            seq.append(Bid(a, margin, winner.get(a) == pid))
        out.append(seq)
    return out


@dataclass
class SeedOutcome:
    """What one group won in one run (1v1 or per-agent where noted)."""
    win_share: float              # auctions won by the group / auctions held
    job_cost: List[float]         # execution cost (USD) of each job the group won
    job_quality: List[float]      # consumer quality rating of each job the group won
    job_price: List[float]        # price received (winning bid, USD) for each job won
    job_margin: List[float]       # price / execution cost − 1 for each job won (realized)
    final_reputation: float       # reputation score after the last auction (per-agent mean)


def outcomes(cell: str, group: str) -> List[SeedOutcome]:
    """Per seed: win share, cost and quality of jobs won, final reputation."""
    out = []
    for row in ok_runs(cell):
        ids = set(_groups(row).get(group) or [])
        if not ids:
            raise ValueError(f"{row['run_dir']}: no group {group!r}")
        m = _metrics(row)
        auctions = m["auctions"]
        won = [a for a in auctions if a.get("winner_id") in ids]
        cost = [float(a["quality_details"]["winner_llm_cost"]) for a in won
                if (a.get("quality_details") or {}).get("winner_llm_cost") is not None]
        qual = [float(a["quality_rating"]) for a in won if a.get("quality_rating") is not None]
        price, margin = [], []
        for a in won:
            bid = a.get("winning_bid_amount")
            c = (a.get("quality_details") or {}).get("winner_llm_cost")
            if bid is None:
                continue
            price.append(float(bid) / 1e6)
            if c:
                margin.append(float(bid) / 1e6 / float(c) - 1.0)
        reps = (m.get("reputation_evolution") or [{}])[-1].get("reputations", {})
        scores = [float(reps[str(i)]["score"]) for i in ids if str(i) in reps]
        out.append(SeedOutcome(len(won) / len(auctions), cost, qual, price, margin,
                               st.mean(scores) if scores else float("nan")))
    return out


def decline_rate(cell: str, group: str) -> float:
    """Share of eligible auctions the group's agent declined (abstention, O2c).

    Counted from the agent's own log ("declined:" is written once per decline),
    so an opponent's declines are never mixed in. Denominator = declines + bids
    attempted.
    """
    declines = attempted = 0
    for row in ok_runs(cell):
        for pid in _groups(row).get(group) or []:
            log = paths.resolve(row["run_dir"]) / f"provider_agent_{pid}.log"
            declines += log.read_text(errors="ignore").count("declined:")
            s = json.loads((paths.resolve(row["run_dir"]) / f"provider_{pid}_status.json").read_text())
            attempted += s["bidding"]["attempted"]
    return declines / (declines + attempted) if declines + attempted else 0.0


_LEVEL = re.compile(r"Effort estimates this loop: costs=(\{[^}]*\})")
_EST = re.compile(r"📈 (?:decomposed|r_guarded|hybrid): requirement (\S+) tier=\w+ multiplier=([0-9.]+) (\{.*\})")
_EXEC = re.compile(r"Cost estimates this loop: execution=\$([0-9.]+)")
_HEUR_AUCTION = re.compile(r"h2_strategy_3: auction (\d+) bid=")
_AUCTION_REF = re.compile(r"(?:auction (\d+)|Skipping auction (\d+))")
_REASON = re.compile(r"'reason': '(\w+)'")


@dataclass
class Estimate:
    requirement: str
    predicted: float      # USD, execution cost of the job
    cold_start: bool      # fallback: no fit to apply, the moving average was used


def estimates(cell: str, group: str) -> List[Estimate]:
    """Every per-job execution-cost estimate the group's agent made, all seeds.

    Estimator agents (LLM-Estimate, Regression) log one line per job:
    predicted = the medium-tier cost level × the job's multiplier — the same
    rule as experiments/analysis/in_market_sigma.py. Adaptive has no per-job
    estimate: it applies its current running-average execution cost to every
    job, read from the last cost line before each of its bids; the job is
    identified through the auction's sampled requirement.
    """
    out: List[Estimate] = []
    for row in ok_runs(cell):
        pid = (_groups(row).get(group) or [None])[0]
        if pid is None:
            raise ValueError(f"{row['run_dir']}: no group {group!r}")
        req_of = {int(a["auction_id"]): a.get("sampled_requirement_id")
                  for a in _metrics(row)["auctions"]}
        lines = (paths.resolve(row["run_dir"]) / f"provider_agent_{pid}.log").read_text(errors="ignore").splitlines()
        level = execution = None
        seen = set()
        for i, line in enumerate(lines):
            m = _LEVEL.search(line)
            if m:
                try:
                    level = json.loads(m.group(1)).get("medium")
                except ValueError:
                    pass
            m = _EXEC.search(line)
            if m:
                execution = float(m.group(1))
            m = _EST.search(line)
            if m and level:
                aid = None
                for follow in lines[i + 1:i + 4]:
                    f = _AUCTION_REF.search(follow)
                    if f:
                        aid = int(f.group(1) or f.group(2))
                        break
                if aid is not None and aid not in seen:
                    seen.add(aid)
                    # Fallback = the estimator had no fit to apply and returned
                    # multiplier 1.0, i.e. the moving average. LLM-Estimate logs a
                    # 'reason' for every such case (cold_start before its first
                    # calibration, deadline_exceeded, call_failed, parse_failed,
                    # bad_rounds, ...). Regression either withholds its fit
                    # (guard_withheld) or has no usable feature, which includes
                    # having no observations yet (degenerate).
                    diag = m.group(3)
                    reason = _REASON.search(diag)
                    fallback = ((reason is not None and reason.group(1) != "oracle")
                                or "'guard_withheld': True" in diag
                                or "'degenerate': True" in diag)
                    out.append(Estimate(m.group(1), level * float(m.group(2)), fallback))
                continue
            m = _HEUR_AUCTION.search(line)
            if m and execution and group.startswith("B_none"):
                aid = int(m.group(1))
                if aid not in seen and req_of.get(aid):
                    seen.add(aid)
                    out.append(Estimate(req_of[aid], execution, False))
    return out


def _snapshot_epoch(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=SNAPSHOT_TZ).timestamp()


def trajectory(cell: str, group: str) -> List[List[float]]:
    """Per-seed net balance after each auction, index 0..N_AUCTIONS.

    Point 0 is the start (0.0). Point k (1..29) is the last balance snapshot
    taken before auction k+1 closed, so it includes auction k's settlement
    (execution cost and payment land after the auction itself ends). Point 30 is
    the run's final balance — the endpoint metric — taken from the status file.
    """
    out = []
    for row in ok_runs(cell):
        ids = _groups(row).get(group)
        if not ids:
            raise ValueError(f"{row['run_dir']}: no group {group!r}")
        m = _metrics(row)
        ends = sorted(float(a["timestamp_ended"]) for a in m["auctions"])
        if len(ends) != N_AUCTIONS:
            raise ValueError(f"{row['run_dir']}: {len(ends)} auctions, expected {N_AUCTIONS}")
        evo = m["provider_financials"]["evolution"]
        per_agent = []
        for pid in ids:
            snaps = sorted((_snapshot_epoch(e["timestamp"]), e["net_balance"])
                           for e in evo if e["provider_id"] == pid)
            series = [0.0]
            for k in range(1, N_AUCTIONS):
                cutoff = ends[k]                      # auction k+1 closes
                before = [v for t, v in snaps if t <= cutoff]
                series.append(before[-1] if before else 0.0)
            s = json.loads((paths.resolve(row["run_dir"]) / f"provider_{pid}_status.json").read_text())
            series.append(s["financials"]["net_balance"])
            per_agent.append(series)
        out.append([st.mean(col) for col in zip(*per_agent)])
    return out
