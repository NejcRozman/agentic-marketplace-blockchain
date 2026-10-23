#!/usr/bin/env python3
"""S9.1 — H1 reruns on the v3 apparatus (docs/s9_1_implementation.md).

Cells are deltas over `s9_base_v3.yaml`, exactly as S9's generator does, so the
v3 apparatus (dataset, priors, seeds, executor model) is inherited rather than
restated. Emitted into the same directory the launcher reads, under the `s9_`
file prefix it requires:

    experiments/configs/s9_cells/s9_<id>.yaml   ->  --cell <id>
    experiments/data/logs/s9_<id>/              ->  check_run's log dir

Those three must agree or runs complete and are then marked `run directory
missing` — the S9 config-path defect. `--check` asserts it.

    python experiments/scripts/generate_s91_cells.py --list
    python experiments/scripts/generate_s91_cells.py --write --stage 1
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "experiments" / "conditions" / "templates"
BASE = CONFIG_DIR / "s9_base_v3.yaml"
OUT_DIR = REPO_ROOT / "experiments" / "conditions"

RW = 20
ACTION_SPACE = "price_only"          # §5 Q4 — H1 has no O2c gate
VARIANT_MODEL = "openai/gpt-oss-20b"  # §5 Q5 — the model X_d uses, so the bridge is exact

#: Short group labels, so a run directory names what it ran.
SHORT = {"zi_random": "zi", "h2_strategy_1": "band", "h2_strategy_3": "s3"}

#: composition -> (total providers, (a-arm agents, b-arm agents)).
#: The random arm wins only on a low draw, so its winning margin is the low
#: tail of its own band; beating k rivals needs a draw below all of them, so
#: that penalty should grow with k while the controller's does not.
COMPOSITIONS = {"1v1": (2, (1, 1)), "2v2": (4, (2, 2)),
                "1v3": (4, (1, 3)), "3v1": (4, (3, 1))}

# id, stage, a-arm, b-arm.  An arm is a dict of behavior overrides.
def heur(strategy: str) -> Dict[str, Any]:
    return {"reasoning_mode": "heuristic", "heuristic_strategy": strategy}

def llm(**flags: Any) -> Dict[str, Any]:
    # `llm_structured_bid_enabled`: the model returns schema-enforced JSON that
    # the system parses and executes, exactly as H2's estimator does. Under
    # native tool-calling gpt-oss-20b wrote a well-formed bid and failed to
    # emit the call in 28-33% of auctions, which confounds pricing reasoning
    # with call formatting and biases the comparison against the LLM.
    model = flags.pop("model", VARIANT_MODEL)
    base = {"reasoning_mode": "llm_react", "blockchain_llm_model": model,
            "llm_structured_bid_enabled": True}
    base.update(flags)
    return base

#: The Stage 0 variants, by name, so follow-up stages reuse the SAME flag sets
#: rather than restating them (a restatement is how a variant silently drifts).
VARIANTS: Dict[str, Dict[str, Any]] = {
    "R0":  {},
    "R1a": {"llm_margin_bounds_enabled": True},
    "R1c": {"llm_margin_bounds_enabled": True, "llm_outcome_history_enabled": True},
    "R2":  {"llm_margin_bounds_enabled": True, "llm_outcome_history_enabled": True,
            "llm_reflection_enabled": True},
}


def followup_cells(winner: str) -> List[Dict[str, Any]]:
    """Every cell that depends on which Stage 0 variant won.

    Four discriminating cells (rw=80 and 1-vs-3, for the minimal variant R0 and
    for the winner) then spec stages 2, 3 and 4. rw=20 1v1 is the one condition
    Stage 1 showed flatters the less sophisticated strategy, so the winner is
    re-tested where the heuristic ladder actually separated.
    """
    w = VARIANTS[winner]
    cells: List[Dict[str, Any]] = []

    def add(cid, a, b, **kw):
        cells.append({"id": cid, "stage": 2, "a": a, "b": b, **kw})

    # -- discriminating: does the rw=20 1v1 ordering survive? --
    for name, flags in (("R0", VARIANTS["R0"]), (winner, w)):
        if name == "R0" and winner == "R0" and any(c["id"].endswith("R0_rw80") for c in cells):
            continue
        add(f"S91_disc_{name}_rw80", llm(**flags), heur("h2_strategy_3"),
            rw=80, note=f"{name} vs s3 at rw=80")
        add(f"S91_disc_{name}_1v3", llm(**flags), heur("h2_strategy_3"),
            composition="1v3", note=f"{name} outnumbered by 3 s3")
        if winner == "R0":
            break

    # -- stage 2: the winner on the rest of the ladder --
    add(f"S91_stage2_{winner}_vs_zi", llm(**w), heur("zi_random"),
        note=f"{winner} vs the beta null")
    add(f"S91_stage2_{winner}_vs_band", llm(**w), heur("h2_strategy_1"),
        note=f"{winner} vs random-in-band")

    # -- stage 3: model robustness --
    for model in ("openai/gpt-4o-mini", "openai/gpt-5.4-mini"):
        short = model.split("/")[-1].replace(".", "").replace("-", "")
        add(f"S91_stage3_{winner}_{short}", llm(model=model, **w), heur("h2_strategy_3"),
            note=f"{winner} on {model}")

    # -- stage 4: the one extra rw point --
    add(f"S91_stage4_{winner}_rw50", llm(**w), heur("h2_strategy_3"),
        rw=50, note=f"{winner} vs s3 at rw=50")
    return cells


CELLS: List[Dict[str, Any]] = [
    # ---- Stage 1: the heuristic ladder. No LLM calls. ----
    {"id": "S91_ladder_zi_vs_band", "stage": 1,
     "a": heur("zi_random"), "b": heur("h2_strategy_1"),
     "note": "what is the margin band worth?"},
    {"id": "S91_ladder_band_vs_adaptive", "stage": 1,
     "a": heur("h2_strategy_1"), "b": heur("h2_strategy_3"),
     "note": "what is the feedback controller worth?"},
    {"id": "S91_ladder_zi_vs_adaptive", "stage": 1,
     "a": heur("zi_random"), "b": heur("h2_strategy_3"),
     "note": "end-to-end spread; transitivity check"},
    {"id": "S91_selfplay_adaptive", "stage": 1,
     "a": heur("h2_strategy_3"), "b": heur("h2_strategy_3"),
     "note": "SEAT-SYMMETRY CONTROL — paired difference must be ~0"},

    # ---- Stage 1b: does the ladder order change with reputation weight? ----
    # rw=20 is S91_ladder_band_vs_adaptive above. commit_duration stays at the
    # heuristic value (10) across all three so the sweep varies rw ALONE.
    {"id": "S91_ladder_band_vs_adaptive_rw50", "stage": 1, "rw": 50,
     "a": heur("h2_strategy_1"), "b": heur("h2_strategy_3"),
     "note": "band vs adaptive at rw=50"},
    {"id": "S91_ladder_band_vs_adaptive_rw80", "stage": 1, "rw": 80,
     "a": heur("h2_strategy_1"), "b": heur("h2_strategy_3"),
     "note": "band vs adaptive at rw=80"},

    # ---- Stage 1c: composition. Does random-in-band degrade with rivals? ----
    {"id": "S91_comp_2v2_band_vs_adaptive", "stage": 1, "composition": "2v2",
     "a": heur("h2_strategy_1"), "b": heur("h2_strategy_3"),
     "note": "2 band vs 2 adaptive"},

    # The sharpest test of the winner's-curse mechanism: ONE random-in-band
    # agent against THREE adapting rivals. To win it must draw below all three,
    # pushing its winning draws deep into its own low tail, while each
    # controller simply wins less often at the price it set. Named for what it
    # contains, not "3v1", which is ambiguous about which arm is which.
    {"id": "S91_comp_1band_vs_3adaptive", "stage": 1, "composition": "1v3",
     "a": heur("h2_strategy_1"), "b": heur("h2_strategy_3"),
     "note": "1 random-in-band outnumbered by 3 adaptive"},

    # The mirror: ONE adaptive controller invading three random-in-band
    # incumbents. Tests whether the controller's advantage survives as a
    # minority — the composition analogue of H2's invasion probe.
    {"id": "S91_comp_3band_vs_1adaptive", "stage": 1, "composition": "3v1",
     "a": heur("h2_strategy_1"), "b": heur("h2_strategy_3"),
     "note": "3 random-in-band vs 1 adaptive (adaptive outnumbered)"},

    # ---- Stage 0: LLM variant selection, each against the top rung. ----
    {"id": "S91_var_R0_vs_s3", "stage": 0,
     "a": llm(), "b": heur("h2_strategy_3"), "note": "R0 bare A2"},
    {"id": "S91_var_R1a_vs_s3", "stage": 0,
     "a": llm(llm_margin_bounds_enabled=True), "b": heur("h2_strategy_3"),
     "note": "R1a + margin band"},
    {"id": "S91_var_R1c_vs_s3", "stage": 0,
     "a": llm(llm_margin_bounds_enabled=True, llm_outcome_history_enabled=True),
     "b": heur("h2_strategy_3"), "note": "R1c + full outcome history"},
    {"id": "S91_var_R2_vs_s3", "stage": 0,
     "a": llm(llm_margin_bounds_enabled=True, llm_outcome_history_enabled=True,
              llm_reflection_enabled=True),
     "b": heur("h2_strategy_3"), "note": "R2 + reflection memo (2 LLM calls/decision)"},

    # ---- Matched floor (paper review, 2026-10-19). LLM-Base beat Adaptive at
    # rw=20 only by pricing below Adaptive's 20% floor (median margin 10%). Here
    # Adaptive may go down to its cost basis (m_min = 0), so neither arm has a
    # pricing region the other is denied. Stage 3 so `--stage 3` regenerates it
    # alone without rewriting the cells that already ran.
    {"id": "S91_var_R0_vs_s3_floor0", "stage": 3,
     "a": llm(), "b": {**heur("h2_strategy_3"), "heuristic_min_margin": 0.0},
     "note": "R0 vs adaptive with floor 0 (matched action space)"},
]


def label(arm: Dict[str, Any], prefix: str) -> str:
    if arm["reasoning_mode"] == "llm_react":
        rung = "R0"
        if arm.get("llm_reflection_enabled"):
            rung = "R2"
        elif arm.get("llm_outcome_history_enabled"):
            rung = "R1c"
        elif arm.get("llm_margin_bounds_enabled"):
            rung = "R1a"
        return f"{prefix}_llm_{rung}"
    return f"{prefix}_{SHORT[arm['heuristic_strategy']]}"


def render(cell: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(base)
    cid = cell["id"]

    config["experiment"]["id"] = f"s9_{cid}"
    config["experiment"]["name"] = f"S9.1 stage {cell['stage']} — {cid}"
    uses_llm = any(a["reasoning_mode"] == "llm_react" for a in (cell["a"], cell["b"]))
    config["experiment"]["description"] = (
        f"{label(cell['a'],'a')} vs {label(cell['b'],'b')} · rw={cell.get('rw', RW)} · "
        f"O2c={ACTION_SPACE} · {cell.get('composition','1v1')} · {cell['note']}"
    )
    config["experiment"].pop("run_seed", None)

    consumer = config["agents"]["consumer"]["config"]
    consumer["reputation_weight"] = cell.get("rw", RW)
    # As S9: the runner reads sampled-dataset settings from the consumer only.
    config["agents"]["consumer"]["behavior"]["sampled_dataset_run_selection_mode"] = "random"
    # An LLM arm needs the wider commit window to get its call in.
    #
    # 40s, not H2's 20s: a full ReAct turn has a long tail (median 7.3s, p90
    # 26.1s, max 46.1s on gpt-oss-20b), and at 20s about 4/30 bids arrived after
    # the commit phase closed (NotInCommitPhase). That penalty scales with how
    # slow a variant is — R2 issues TWO calls per decision — so a fixed 20s
    # window would make the variant comparison partly a latency race.
    consumer["commit_duration"] = 40 if uses_llm else 10

    providers = config["agents"]["providers"]
    count, ratios = COMPOSITIONS[cell.get("composition", "1v1")]
    providers["count"] = count
    providers["agent_ids"] = [None] * count
    providers["config"]["action_space"] = ACTION_SPACE

    template = providers["groups"][0]
    groups = []
    for (side, prefix), ratio in zip((("a", "X"), ("b", "B")), ratios):
        arm = cell[side]
        g = copy.deepcopy(template)
        g["name"] = label(arm, prefix)
        g["ratio"] = ratio
        beh = g["behavior"]
        # STRING, not int: the runner forwards architecture into a subprocess
        # env var, and an int raises `TypeError: expected str ... not int` in
        # subprocess before any agent starts. S9's generator emits '2' likewise.
        beh["architecture"] = "2"
        # §5 Q3 — S9.1 tests pricing only. Set explicitly rather than removing
        # the key, matching how S9 spells the no-estimator arm.
        beh["cost_estimator"] = "none"
        beh.pop("blockchain_llm_model", None)
        # An llm_react arm must not carry a heuristic strategy: it is unused, but
        # leaving it in a config makes a silent fallback to heuristic pricing
        # indistinguishable from the intended arm in the status JSON.
        if arm["reasoning_mode"] == "llm_react":
            beh.pop("heuristic_strategy", None)
        for k in ("llm_margin_bounds_enabled", "llm_last_margin_enabled",
                  "llm_outcome_history_enabled", "llm_reflection_enabled",
                  "llm_structured_bid_enabled"):
            beh.pop(k, None)
        beh.update(arm)
        groups.append(g)
    providers["groups"] = groups

    for section, sub in (("logs", "logs"), ("output", "metrics")):
        config["monitoring"][section]["directory"] = f"data/runs/{sub}/s9_{cid}"
    return config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="assert id/path/launcher agreement")
    ap.add_argument("--stage", type=int, choices=(0, 1, 2, 3))
    ap.add_argument("--emit-followups", metavar="WINNER",
                    help="generate the cells that depend on the Stage 0 winner")
    args = ap.parse_args()

    base = yaml.safe_load(BASE.read_text())
    if args.emit_followups:
        cells = followup_cells(args.emit_followups)
    else:
        cells = [c for c in CELLS if args.stage is None or c["stage"] == args.stage]

    if args.list:
        for c in cells:
            print(f"  stage {c['stage']}  {c['id']:<32} "
                  f"{label(c['a'],'X'):<14} vs {label(c['b'],'B'):<10} {c['note']}")
        return

    failures = 0
    for c in cells:
        cfg = render(c, base)
        path = OUT_DIR / f"s9_{c['id']}.yaml"
        if args.check:
            want = f"data/runs/logs/s9_{c['id']}"
            got = cfg["monitoring"]["logs"]["directory"]
            ok = got == want and cfg["experiment"]["id"] == f"s9_{c['id']}"
            print(f"  {'OK  ' if ok else 'FAIL'} {c['id']}  logs={got}")
            failures += 0 if ok else 1
            continue
        if args.write:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            # width high enough that safe_dump never wraps a ${VAR} placeholder.
            text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=4096)
            # Re-quote the ${VAR} placeholders. safe_dump emits them bare, and a
            # bare ${BLOCKCHAIN_PRIVATE_KEY} resolving to `0x52cc...` is parsed by
            # YAML as a HEX INTEGER, not a string; the runner then hands an int to
            # subprocess and every run dies before deploying a contract.
            # generate_s9_cells.py carries the same fix for the same reason.
            text = re.sub(r'(?<![\'"])\$\{(\w+)\}(?![\'"])', r'"${\1}"', text)
            header = ("# GENERATED by experiments/scripts/generate_s91_cells.py — do not edit.\n"
                      f"# S9.1 stage {c['stage']} cell {c['id']} (docs/s9_1_implementation.md).\n"
                      "# Seed is supplied per run: --run-seed <n>.\n\n")
            path.write_text(header + text)
            # Verify the placeholder survives a round-trip as a string.
            back = yaml.safe_load(path.read_text())
            pk = back["blockchain"]["accounts"]["deployer"]
            assert isinstance(pk, str), f"{path.name}: deployer key parsed as {type(pk).__name__}"
    if args.check:
        raise SystemExit(1 if failures else 0)
    if args.write:
        print(f"written {len(cells)} configs to {OUT_DIR}")


if __name__ == "__main__":
    main()
