#!/usr/bin/env python3
"""Write one runner config per experimental condition, from the table below.

Every condition shares experiments/conditions/base.yaml and differs only in the
reputation weight w, the commit window, whether the decline rule is on, the LLM
model, and its providers: (group name, strategy, number of agents). Group names
are the keys the recorded data and the analysis use; X_ is the strategy under
test, B_ its opponent.

    python -m experiments.generate_conditions           # write experiments/conditions/<id>.yaml
    python -m experiments.generate_conditions --check   # verify the committed configs
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

from agents.common import paths

BASE = paths.CONDITIONS_DIR / "base.yaml"

#: Strategy name -> (display name, provider behaviour).
_HEURISTIC = {"architecture": "2", "reasoning_mode": "heuristic", "cost_estimator": "none",
              "heuristic_min_margin": 0.2, "heuristic_max_margin": 0.5}
_LLM_PRICER = {"architecture": "2", "reasoning_mode": "llm_react", "cost_estimator": "none",
               "heuristic_min_margin": 0.2, "heuristic_max_margin": 0.5, "llm_structured_bid_enabled": True}
STRATEGIES = {
    "adaptive":        ("Adaptive", {**_HEURISTIC, "heuristic_strategy": "h2_strategy_3"}),
    "adaptive_floor0": ("Adaptive (m_min = 0)", {**_HEURISTIC, "heuristic_strategy": "h2_strategy_3",
                                                 "heuristic_min_margin": 0.0}),
    "band":            ("Random-in-band", {**_HEURISTIC, "heuristic_strategy": "h2_strategy_1"}),
    "random":          ("Random", {**_HEURISTIC, "heuristic_strategy": "zi_random"}),
    "regression":      ("Regression", {**_HEURISTIC, "heuristic_strategy": "h2_strategy_3",
                                       "cost_estimator": "r_guarded"}),
    "oracle":          ("Oracle", {**_HEURISTIC, "heuristic_strategy": "h2_strategy_3", "cost_estimator": "oracle"}),
    "llm_estimate":    ("LLM-Estimate", {**_HEURISTIC, "heuristic_strategy": "h2_strategy_3",
                                         "cost_estimator": "decomposed"}),
    "llm_base":        ("LLM-Base", _LLM_PRICER),
    "llm_band":        ("LLM-Band", {**_LLM_PRICER, "llm_margin_bounds_enabled": True}),
    "llm_history":     ("LLM-History", {**_LLM_PRICER, "llm_margin_bounds_enabled": True,
                                        "llm_outcome_history_enabled": True}),
    "llm_reflect":     ("LLM-Reflect", {**_LLM_PRICER, "llm_margin_bounds_enabled": True,
                                        "llm_outcome_history_enabled": True, "llm_reflection_enabled": True}),
}
LLM_STRATEGIES = {"llm_estimate", "llm_base", "llm_band", "llm_history", "llm_reflect"}

OSS, MINI4O, MINI54 = "openai/gpt-oss-20b", "openai/gpt-4o-mini", "openai/gpt-5.4-mini"

# id, w, commit window (s), decline rule, LLM model, providers [(group, strategy, agents)]
CONDITIONS = [
    # RQ1: heuristic baselines
    ("S91_ladder_zi_vs_band",            20, 10, False, None,   [("X_zi", "random", 1), ("B_band", "band", 1)]),
    ("S91_ladder_zi_vs_adaptive",        20, 10, False, None,   [("X_zi", "random", 1), ("B_s3", "adaptive", 1)]),
    ("S91_ladder_band_vs_adaptive",      20, 10, False, None,   [("X_band", "band", 1), ("B_s3", "adaptive", 1)]),
    ("S91_ladder_band_vs_adaptive_rw50", 50, 10, False, None,   [("X_band", "band", 1), ("B_s3", "adaptive", 1)]),
    ("S91_ladder_band_vs_adaptive_rw80", 80, 10, False, None,   [("X_band", "band", 1), ("B_s3", "adaptive", 1)]),
    ("S91_comp_2v2_band_vs_adaptive",    20, 10, False, None,   [("X_band", "band", 2), ("B_s3", "adaptive", 2)]),
    ("S91_comp_1band_vs_3adaptive",      20, 10, False, None,   [("X_band", "band", 1), ("B_s3", "adaptive", 3)]),
    ("S91_comp_3band_vs_1adaptive",      20, 10, False, None,   [("X_band", "band", 3), ("B_s3", "adaptive", 1)]),
    ("S91_selfplay_adaptive",            20, 10, False, None,   [("X_s3", "adaptive", 1), ("B_s3", "adaptive", 1)]),
    # RQ1: LLM price-setters against Adaptive
    ("S91_var_R0_vs_s3",                 20, 40, False, OSS,    [("X_llm_R0", "llm_base", 1), ("B_s3", "adaptive", 1)]),
    ("S91_var_R0_vs_s3_floor0",          20, 40, False, OSS,    [("X_llm_R0", "llm_base", 1), ("B_s3", "adaptive_floor0", 1)]),
    ("S91_var_R1a_vs_s3",                20, 40, False, OSS,    [("X_llm_R1a", "llm_band", 1), ("B_s3", "adaptive", 1)]),
    ("S91_var_R1c_vs_s3",                20, 40, False, OSS,    [("X_llm_R1c", "llm_history", 1), ("B_s3", "adaptive", 1)]),
    ("S91_var_R2_vs_s3",                 20, 40, False, OSS,    [("X_llm_R2", "llm_reflect", 1), ("B_s3", "adaptive", 1)]),
    ("S91_stage4_R1c_rw50",              50, 40, False, OSS,    [("X_llm_R1c", "llm_history", 1), ("B_s3", "adaptive", 1)]),
    ("S91_disc_R1c_rw80",                80, 40, False, OSS,    [("X_llm_R1c", "llm_history", 1), ("B_s3", "adaptive", 1)]),
    ("S91_disc_R0_rw80",                 80, 40, False, OSS,    [("X_llm_R0", "llm_base", 1), ("B_s3", "adaptive", 1)]),
    ("S91_disc_R1c_1v3",                 20, 40, False, OSS,    [("X_llm_R1c", "llm_history", 1), ("B_s3", "adaptive", 3)]),
    ("S91_disc_R0_1v3",                  20, 40, False, OSS,    [("X_llm_R0", "llm_base", 1), ("B_s3", "adaptive", 3)]),
    ("S91_stage2_R1c_vs_zi",             20, 40, False, OSS,    [("X_llm_R1c", "llm_history", 1), ("B_zi", "random", 1)]),
    ("S91_stage2_R1c_vs_band",           20, 40, False, OSS,    [("X_llm_R1c", "llm_history", 1), ("B_band", "band", 1)]),
    ("S91_stage3_R1c_gpt4omini",         20, 40, False, MINI4O, [("X_llm_R1c", "llm_history", 1), ("B_s3", "adaptive", 1)]),
    ("S91_stage3_R1c_gpt54mini",         20, 40, False, MINI54, [("X_llm_R1c", "llm_history", 1), ("B_s3", "adaptive", 1)]),
    # RQ2: LLM-Estimate against Adaptive and Regression
    ("A_gptoss20b_xd_vs_h",              20, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("A_gptoss20b_xd_vs_rguarded",       20, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("A_gpt54mini_xd_vs_h",              20, 20, True,  MINI54, [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("A_gpt54mini_xd_vs_rguarded",       20, 20, True,  MINI54, [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("E_xd_vs_h_v3",                     20, 20, True,  MINI4O, [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("E_xd_vs_r_v3",                     20, 20, True,  MINI4O, [("X_decomposed", "llm_estimate", 1), ("R_guarded", "regression", 1)]),
    ("B_gptoss20b_rw0_xd_vs_h",           0, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("B_gptoss20b_rw35_xd_vs_h",         35, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("B_gptoss20b_rw50_xd_vs_h",         50, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("B_gptoss20b_rw65_xd_vs_h",         65, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("B_gptoss20b_rw80_xd_vs_h",         80, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 1)]),
    ("B_gptoss20b_rw0_xd_vs_rguarded",    0, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("B_gptoss20b_rw35_xd_vs_rguarded",  35, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("B_gptoss20b_rw50_xd_vs_rguarded",  50, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("B_gptoss20b_rw65_xd_vs_rguarded",  65, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("C_o2c_on_rw80",                    80, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("D_invasion_1v3",                   20, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_none", "adaptive", 3)]),
    ("D_invasion_1v3_rguarded",          20, 20, True,  OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 3)]),
    # RQ2: perfect cost knowledge and the decline rule
    ("A_oracle_vs_h",                    20, 10, True,  None,   [("X_oracle_bound", "oracle", 1), ("B_none", "adaptive", 1)]),
    ("A_oracle_noO2c_vs_h",              20, 10, False, None,   [("X_oracle_bound", "oracle", 1), ("B_none", "adaptive", 1)]),
    ("A_oracle_noO2c_vs_rguarded",       20, 10, False, None,   [("X_oracle_bound", "oracle", 1), ("B_r_guarded", "regression", 1)]),
    ("A_oracle_vs_xd",                   20, 20, True,  OSS,    [("X_oracle_bound", "oracle", 1), ("B_decomposed", "llm_estimate", 1)]),
    ("A_oracle_noO2c_vs_xd",             20, 20, False, OSS,    [("X_oracle_bound", "oracle", 1), ("B_decomposed", "llm_estimate", 1)]),
    ("C_o2c_off_rw20",                   20, 20, False, OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
    ("C_o2c_off_rw80",                   80, 20, False, OSS,    [("X_decomposed", "llm_estimate", 1), ("B_r_guarded", "regression", 1)]),
]


def describe(w, commit, decline, model, providers):
    def side(strategy, n):
        name = STRATEGIES[strategy][0]
        if strategy in LLM_STRATEGIES:
            name += f" ({model.split('/')[-1]})"
        return name if n == 1 else f"{n} x {name}"
    (_, s1, n1), (_, s2, n2) = providers
    return (f"{side(s1, n1)} vs {side(s2, n2)}; w = {w / 100:g}, {n1}:{n2}, "
            f"decline rule {'on' if decline else 'off'}, {commit} s commit window")


def render(base, row):
    cid, w, commit, decline, model, providers = row
    cfg = copy.deepcopy(base)
    cfg["experiment"] = {"id": cid, "description": describe(w, commit, decline, model, providers),
                         **cfg["experiment"]}
    consumer = cfg["agents"]["consumer"]["config"]
    consumer["commit_duration"] = commit
    consumer["reputation_weight"] = w
    prov = cfg["agents"]["providers"]
    count = sum(n for _, _, n in providers)
    prov["count"] = count
    prov["agent_ids"] = [None] * count
    prov["config"]["action_space"] = "price_or_decline" if decline else "price_only"
    prov["groups"] = []
    for group, strategy, n in providers:
        behavior = dict(STRATEGIES[strategy][1])
        if strategy in LLM_STRATEGIES:
            behavior["blockchain_llm_model"] = model
        prov["groups"].append({"name": group, "ratio": n, "behavior": behavior})
    return cfg


def to_yaml(cfg) -> str:
    text = yaml.safe_dump(cfg, sort_keys=False, width=100, allow_unicode=True)
    # Quote the ${VAR} placeholders: unquoted, a private key 0x... would be read as an integer.
    text = re.sub(r"(:|-) \$ \{(\w+)\}", r'\1 "${\2}"', text).replace("$ {", "${")
    return ("# Generated by experiments/generate_conditions.py; do not edit.\n"
            "# The seed is supplied per run: --run-seed <n>.\n\n" + text)


def load(path: Path):
    return yaml.safe_load(path.read_text().replace("${", "$ {"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare with the committed configs")
    args = parser.parse_args()
    base = load(BASE)
    ids = [row[0] for row in CONDITIONS]
    assert len(ids) == len(set(ids)), "duplicate condition id"
    failed = 0
    for row in CONDITIONS:
        path = paths.CONDITIONS_DIR / f"{row[0]}.yaml"
        text = to_yaml(render(base, row))
        if args.check:
            ok = path.exists() and path.read_text() == text
            failed += not ok
            if not ok:
                print(f"differs: {path.name}")
        else:
            path.write_text(text)
    stale = sorted(p.name for p in paths.CONDITIONS_DIR.glob("*.yaml") if p.stem not in ids and p != BASE)
    for name in stale:
        print(f"not in the table: {name}")
    print(f"{len(CONDITIONS)} conditions {'checked' if args.check else 'written'}"
          f"{f', {failed} differ' if failed else ''}")
    sys.exit(1 if args.check and (failed or stale) else 0)


if __name__ == "__main__":
    main()
