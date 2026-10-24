"""Offline tests: no chain, no network, no LLM call.

`golden.json` holds SHA-256 digests of what the code that ran the experiments
produced on the inputs these tests generate: the bidding policies' decisions, the
LLM pricing prompt, the LLM tool descriptions and the cost estimators' outputs.
A test fails if the current code produces anything else.

Tests marked `data` read the recorded data (see paths.DATA_DIR) and are skipped
when it is absent.
"""
import hashlib
import json
from pathlib import Path

import pytest

from agents.common import paths

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())
HAS_DATA = (paths.DATASET_FINAL_DIR / "runs.jsonl").exists() and paths.LEDGER.exists()


def digest(obj, **dumps_kwargs) -> str:
    return hashlib.sha256(json.dumps(obj, **dumps_kwargs).encode()).hexdigest()


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the recorded data under the data directory")


def pytest_collection_modifyitems(config, items):
    if HAS_DATA:
        return
    skip = pytest.mark.skip(reason=f"no recorded data under {paths.DATA_DIR}")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
