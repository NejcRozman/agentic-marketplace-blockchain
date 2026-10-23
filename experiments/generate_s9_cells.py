#!/usr/bin/env python3
"""S7 §D — materialise the run matrix from one declaration.

Thirty hand-written YAMLs would drift: settings have already moved four times
(priors, phase durations, dataset path, abstention gate), and each move would
have meant thirty edits with no way to check they landed. Here the cells are
parameter deltas over one base config, so the spec table and
`experiments/configs/s9_matrix.yaml` can be diffed by eye.

Seeds are deliberately absent. One config serves all its seeds via the runner's
`--run-seed`, which is what makes the pairing work: X-vs-H and X-vs-R_guarded at
seed 1007 read the same number and therefore draw the same requirement order.

    python experiments/scripts/generate_s9_cells.py --list
    python experiments/scripts/generate_s9_cells.py --write            # all cells
    python experiments/scripts/generate_s9_cells.py --write --block A  # one block
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
MATRIX = CONFIG_DIR / "s9_matrix.yaml"
OUT_DIR = REPO_ROOT / "experiments" / "conditions"

#: Arms that issue an LLM call, and therefore need the wider commit window.
LLM_ARMS = {"hybrid", "decomposed"}


def short_model(model: str) -> str:
    return model.split("/")[-1].replace(".", "").replace("-", "")


def cells_from_matrix(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the declaration into one dict per cell."""
    defaults = matrix["defaults"]
    contrasts = matrix["contrasts"]
    cells: List[Dict[str, Any]] = []

    def add(block: str, cell_id: str, contrast: str, **overrides) -> None:
        spec = {
            "block": block,
            "id": cell_id,
            "contrast": contrast,
            "arms": contrasts[contrast],
            "rw": overrides.pop("rw", defaults["rw"]),
            "action_space": overrides.pop("action_space", defaults["action_space"]),
            "model": overrides.pop("model", matrix["models"][0]),
            "composition": overrides.pop("composition", "1v1"),
        }
        spec.update(overrides)
        cells.append(spec)

    # Block A — 4 tiers x 2 contrasts at the carrier rw.
    a = matrix["blocks"]["A"]["vary"]
    for model in matrix[a["model"]]:
        for contrast in a["contrast"]:
            add("A", f"A_{short_model(model)}_{contrast}", contrast, model=model)
    # Block A also carries explicitly-named cells — the oracle reference, which
    # is not a model tier and so cannot come out of the cross-product above.
    for cell in matrix["blocks"]["A"].get("cells", []):
        cell = dict(cell)
        add("A", cell.pop("id"), cell.pop("contrast"), **cell)

    # Block B — the rw sweep, excluding what block A already ran.
    b = matrix["blocks"]["B"]
    excluded = set(b.get("exclude_rw", []))
    for model in matrix[b["vary"]["model"]]:
        for rw in matrix[b["vary"]["rw"]]:
            if rw in excluded:
                continue
            for contrast in b["vary"]["contrast"]:
                add("B", f"B_{short_model(model)}_rw{rw}_{contrast}", contrast,
                    model=model, rw=rw)

    # Blocks C and D are enumerated explicitly: they are ablations and probes,
    # not a cross-product, and writing them out keeps them readable.
    #
    # Block E is deliberately absent. Its four cells are complete (§E.5, 160 runs
    # in the ledger) and hand-authored; generating over them would rewrite their
    # provider group names — `E_xd_vs_rd_v3` names its opponent `R_rounds`, not
    # the `B_<estimator>` this file emits — and orphan the finished analysis.
    for block in ("C", "D"):
        for cell in matrix["blocks"][block].get("cells", []):
            cell = dict(cell)
            add(block, cell.pop("id"), cell.pop("contrast"), **cell)

    return cells


def render(cell: Dict[str, Any], base: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """One cell as a complete experiment config."""
    config = copy.deepcopy(base)
    arms = cell["arms"]

    config["experiment"]["id"] = f"s9_{cell['id']}"
    config["experiment"]["name"] = f"S9 {cell['block']} — {cell['id']}"
    config["experiment"]["description"] = (
        f"{cell['contrast']} · model={cell['model']} · rw={cell['rw']} · "
        f"O2c={cell['action_space']} · {cell['composition']}"
    )
    # The seed is supplied per run by `--run-seed`; leaving a value here would
    # invite a whole cell being run at one seed by accident.
    config["experiment"].pop("run_seed", None)

    consumer = config["agents"]["consumer"]["config"]
    consumer["reputation_weight"] = cell["rw"]

    # Which of a requirement's three recordings a job replays. This must be set
    # on the *consumer's behavior* block: the runner reads the sampled-dataset
    # settings from there only (`shared_sampled_dataset_args`) and copies them
    # to every provider, so the same key on a provider group is silently
    # ignored.
    #
    # Without it the mode defaults to `round_robin`, whose cursors all start at
    # 0; combined with `shuffled_pass` drawing each requirement exactly once per
    # 30-auction run, every job replayed **repeat 1**. Two thirds of dataset v2
    # were unreachable, execution cost was identical in every run (zero
    # across-seed variance), and a perfect predictor of the 3-repeat mean still
    # scored sigma 0.107 against what the agent actually paid, as a fixed
    # per-requirement bias.
    #
    # This was previously only hand-edited into the generated cell files, with a
    # comment here claiming it was set below when nothing set it. The cell
    # directory is untracked, so a regeneration silently reverted every cell to
    # repeat-1 replay. Generate it instead of patching it.
    config["agents"]["consumer"]["behavior"]["sampled_dataset_run_selection_mode"] = "random"

    needs_llm = any(arm in LLM_ARMS for arm in arms.values())
    consumer["commit_duration"] = (
        defaults["commit_duration_with_llm"] if needs_llm
        else defaults["commit_duration_no_llm"]
    )

    providers = config["agents"]["providers"]
    providers["config"]["action_space"] = cell["action_space"]

    # 1v1 unless the cell is the invasion probe.
    if cell["composition"] == "1v3":
        providers["count"] = 4
        ratios = (1, 3)
    else:
        providers["count"] = 2
        ratios = (1, 1)
    providers["agent_ids"] = [None] * providers["count"]

    template = providers["groups"][0]
    groups = []
    for (label, estimator), ratio in zip(arms.items(), ratios):
        group = copy.deepcopy(template)
        # The a-arm is named for what it *is*, so a run directory cannot be
        # mistaken for the other X protocol when both appear in the matrix.
        # The a-arm is named for what it *is*, so a run directory cannot be
        # mistaken for a different protocol when several appear in the matrix.
        # The analysis keys the treated arm off the `X_` prefix, so the oracle
        # keeps it while saying plainly that it is a bound, not a competitor.
        a_names = {
            "decomposed": "X_decomposed",
            "hybrid": "X_hybrid",
            "oracle": "X_oracle_bound",
        }
        group["name"] = (
            a_names.get(estimator, f"X_{estimator}") if label == "a"
            else f"B_{estimator}"
        )
        group["ratio"] = ratio
        behavior = group["behavior"]
        behavior["cost_estimator"] = estimator
        # NOTE: `sampled_dataset_run_selection_mode` is deliberately NOT set
        # here. The runner reads the sampled-dataset settings from the
        # *consumer's* behavior block only (`shared_sampled_dataset_args`) and
        # copies them to every provider; a value on a provider group is silently
        # ignored. Setting it here looked correct and changed nothing — every
        # job still replayed repeat 1. It is set on the consumer above.
        # Which of a requirement's three recordings a job replays.
        #
        # `round_robin` starts every cursor at 0, and `shuffled_pass` draws each
        # requirement exactly once per 30-auction run, so every job replayed
        # **repeat 1** — two thirds of dataset v2 were unreachable, execution
        # cost was identical in every run (zero across-seed variance), and a
        # perfect predictor of the 3-repeat mean still scored sigma 0.107
        # against what the agent actually paid, as a fixed per-requirement bias.
        #
        # Seeded from the run seed plus the agent id, so a run stays
        # reproducible while different seeds now differ in realised cost rather
        # than only in arrival order.

        if estimator in LLM_ARMS:
            behavior["blockchain_llm_model"] = cell["model"]
        else:
            behavior.pop("blockchain_llm_model", None)
        groups.append(group)
    providers["groups"] = groups

    for section in ("logs", "output"):
        config["monitoring"][section]["directory"] = (
            f"data/runs/{'logs' if section == 'logs' else 'metrics'}/s9_{cell['id']}"
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the configs")
    parser.add_argument("--block", help="only this block (A/B/C/D/E)")
    parser.add_argument("--list", action="store_true", help="list cells and exit")
    args = parser.parse_args()

    matrix = yaml.safe_load(MATRIX.read_text())
    base_text = (CONFIG_DIR / matrix["base"]).read_text()
    base = yaml.safe_load(base_text.replace("${", "$ {"))

    cells = cells_from_matrix(matrix)
    if args.block:
        cells = [c for c in cells if c["block"] == args.block.upper()]

    from collections import Counter
    counts = Counter(c["block"] for c in cells)
    print(f"cells: {len(cells)}  " + "  ".join(f"{b}={n}" for b, n in sorted(counts.items())))
    if args.list or not args.write:
        for cell in cells:
            print(f"  {cell['block']}  {cell['id']:<42} rw={cell['rw']:<3} "
                  f"O2c={cell['action_space']:<16} {short_model(cell['model'])}")
        if not args.write:
            return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        rendered = render(cell, base, matrix["defaults"])
        text = yaml.safe_dump(rendered, sort_keys=False, width=100)
        # Restore the ${VAR} placeholders the loader needs — **quoted**.
        #
        # safe_dump re-emits them as bare scalars, and a bare `${...}` that
        # resolves to a private key `0x52cc...` is parsed by YAML as a *hex
        # integer*, not a string. The runner then hands an int to subprocess and
        # every run dies before deploying. The hand-written configs quote these
        # for exactly this reason.
        # Both forms safe_dump can emit: `key: ${VAR}` and the sequence item
        # `- ${VAR}` (the providers list). Missing the second left one unquoted
        # per file, which is enough to kill every run.
        text = re.sub(r"(:|-) \$ \{(\w+)\}", r'\1 "${\2}"', text)
        text = text.replace("$ {", "${")
        header = (
            f"# GENERATED by experiments/scripts/generate_s9_cells.py — do not edit.\n"
            f"# Cell {cell['id']} (block {cell['block']}, S7 §D).\n"
            f"# Seed is supplied per run: --run-seed <n>.\n\n"
        )
        (OUT_DIR / f"s9_{cell['id']}.yaml").write_text(header + text)
    print(f"\nwritten {len(cells)} configs to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
