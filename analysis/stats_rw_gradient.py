#!/usr/bin/env python3
"""RQ2 — how LLM-Estimate's advantage depends on the reputation weight w (Section 5.7, S3).

Linear mixed model on the per-seed paired differences Δ = N(LLM-Estimate) − N(opponent)
of the reputation-weight sweep (gpt-oss-20b, 1 vs 1, decline rule on):

    Δ_{s,c} = β0 + β1·w_c + β2·[opponent = Regression] + u_s + ε_{s,c}

with a random intercept u_s per seed (the same 40 seeds recur in every condition).
w is on the 0–1 scale. The crossing point — the w at which the expected advantage is
zero — is w* = −β0/β1 against Adaptive and −(β0 + β2)/β1 against Regression, with
95% percentile intervals from a bootstrap that resamples seeds (all conditions of a
seed move together, preserving the pairing).

Must be run with the project virtualenv (statsmodels):

    .venv/bin/python -m analysis.stats_rw_gradient [--boot 2000]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis.data import final_net, ok_runs

XD = "X_decomposed"
# (w, opponent, cell, opponent group)
CELLS = [
    (0.00, "Adaptive",   "B_gptoss20b_rw0_xd_vs_h",          "B_none"),
    (0.20, "Adaptive",   "A_gptoss20b_xd_vs_h",              "B_none"),
    (0.35, "Adaptive",   "B_gptoss20b_rw35_xd_vs_h",         "B_none"),
    (0.50, "Adaptive",   "B_gptoss20b_rw50_xd_vs_h",         "B_none"),
    (0.65, "Adaptive",   "B_gptoss20b_rw65_xd_vs_h",         "B_none"),
    (0.80, "Adaptive",   "B_gptoss20b_rw80_xd_vs_h",         "B_none"),
    (0.00, "Regression", "B_gptoss20b_rw0_xd_vs_rguarded",   "B_r_guarded"),
    (0.20, "Regression", "A_gptoss20b_xd_vs_rguarded",       "B_r_guarded"),
    (0.35, "Regression", "B_gptoss20b_rw35_xd_vs_rguarded",  "B_r_guarded"),
    (0.50, "Regression", "B_gptoss20b_rw50_xd_vs_rguarded",  "B_r_guarded"),
    (0.65, "Regression", "B_gptoss20b_rw65_xd_vs_rguarded",  "B_r_guarded"),
    # config-identical to the sweep cell that would have been generated at w = 0.8
    (0.80, "Regression", "C_o2c_on_rw80",                    "B_r_guarded"),
]


def frame() -> pd.DataFrame:
    rows = []
    for w, opp, cell, group in CELLS:
        seeds = [int(r["seed"]) for r in ok_runs(cell)]
        diffs = [a - b for a, b in zip(final_net(cell, XD), final_net(cell, group))]
        rows += [{"w": w, "regression": int(opp == "Regression"), "seed": s, "diff": d}
                 for s, d in zip(seeds, diffs)]
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm("diff ~ w + regression", df, groups=df["seed"]).fit(reml=True)


def crossings(params) -> tuple:
    b0, b1, b2 = params["Intercept"], params["w"], params["regression"]
    return -b0 / b1, -(b0 + b2) / b1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    df = frame()
    res = fit(df)
    ci = res.conf_int()
    print(f"{len(df)} paired differences, {df['seed'].nunique()} seeds, {len(CELLS)} conditions\n")
    for name, label in (("Intercept", "β0 (w = 0, vs Adaptive)"), ("w", "β1 per unit w"),
                        ("regression", "β2 (opponent = Regression)")):
        print(f"  {label:<28} {res.params[name]:+.4f} [{ci.loc[name, 0]:+.4f}, {ci.loc[name, 1]:+.4f}]"
              f"  p = {res.pvalues[name]:.2g}")
    print(f"  {'β1 per 0.01 of w':<28} {res.params['w'] / 100:+.5f}")

    wa, wr = crossings(res.params)
    rng = np.random.default_rng(args.seed)
    seeds = df["seed"].unique()
    by_seed = {s: g for s, g in df.groupby("seed")}
    boot = []
    for _ in range(args.boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        # relabel resampled seeds so duplicates are separate groups
        bdf = pd.concat([by_seed[s].assign(seed=i) for i, s in enumerate(pick)], ignore_index=True)
        boot.append(crossings(fit(bdf).params))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"\n  crossing w* vs Adaptive    {wa:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]")
    print(f"  crossing w* vs Regression  {wr:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")
    print(f"  ({args.boot} seed-bootstrap replicates)")


if __name__ == "__main__":
    main()
