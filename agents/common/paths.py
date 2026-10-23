"""Locations of code assets and data.

All experiment data lives under one data directory, ``data/`` in the repository
by default, or the directory named by the ``MARKETPLACE_DATA_DIR`` environment
variable:

    data/ledger.jsonl            one line per completed (condition, seed) run
    data/runs/logs/<id>/<run>/   agent logs and status files of each run
    data/runs/metrics/<id>/<run>/experiment_metrics.json
    data/dataset/final/          recorded service dataset (requirements, runs)
    data/dataset/artifacts/      per-execution artifacts of the dataset
    data/llm_cache/              cache of estimator LLM calls

Records written during the experiments store paths relative to the original
repository layout (e.g. ``experiments/data/logs/...``); :func:`resolve` maps
them to their location under the data directory.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"
CONDITIONS_DIR = REPO_ROOT / "experiments" / "conditions"
CHAIN_SNAPSHOT = REPO_ROOT / "experiments" / "chain" / "s7_base.json"

DATA_DIR = Path(os.environ.get("MARKETPLACE_DATA_DIR", REPO_ROOT / "data")).resolve()
LEDGER = DATA_DIR / "ledger.jsonl"
RUN_LOGS_DIR = DATA_DIR / "runs" / "logs"
RUN_METRICS_DIR = DATA_DIR / "runs" / "metrics"
DATASET_DIR = DATA_DIR / "dataset"
DATASET_FINAL_DIR = DATASET_DIR / "final"
DATASET_ARTIFACTS_DIR = DATASET_DIR / "artifacts"
LLM_CACHE_DIR = DATA_DIR / "llm_cache"

_LEGACY_PREFIXES = (
    ("experiments/data/logs/", RUN_LOGS_DIR),
    ("experiments/data/metrics/", RUN_METRICS_DIR),
    ("experiments/data/llm_cache", LLM_CACHE_DIR),
    ("experiments/data/chain_snapshots/s7_base.json", CHAIN_SNAPSHOT),
    ("experiments/executor_analysis/data/datasets/executor_tier_dataset_v3/final", DATASET_FINAL_DIR),
    ("experiments/executor_analysis/data/artifacts/executor_tier_dataset_v3/", DATASET_ARTIFACTS_DIR),
    ("data/", DATA_DIR),
)


def resolve(path: str | Path) -> Path:
    """Absolute location of a path from a config or a recorded data file."""
    text = str(path)
    if os.path.isabs(text):
        return Path(text)
    for prefix, target in _LEGACY_PREFIXES:
        if text == prefix.rstrip("/") or text.startswith(prefix):
            rest = text[len(prefix):].lstrip("/")
            return target / rest if rest else target
    return REPO_ROOT / text
