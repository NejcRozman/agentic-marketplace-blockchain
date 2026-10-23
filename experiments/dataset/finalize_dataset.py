"""Select one collection into data/dataset/final/, the layout the market reads.

The sampler reads only `requirements.jsonl` and `runs.jsonl`; the manifest is
written for provenance.

    python -m experiments.dataset.finalize_dataset --collection full
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from agents.common import paths

DATASET_DIR = paths.DATASET_DIR
#: The collection the experiments' dataset was finalized from.
COLLECTION = "v3_full_001"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize(dataset_dir: Path, collection: str) -> Dict[str, Any]:
    reqs = [r for r in _read_jsonl(dataset_dir / "requirements.jsonl")
            if r.get("collection_id") == collection]
    runs = [r for r in _read_jsonl(dataset_dir / "runs.jsonl")
            if r.get("collection_id") == collection]

    req_ids = {r["requirement_id"] for r in reqs}
    orphans = [r["requirement_id"] for r in runs if r["requirement_id"] not in req_ids]
    if orphans:
        raise ValueError(
            f"{len(orphans)} run(s) reference requirements not in the collection, "
            f"e.g. {sorted(set(orphans))[:3]}"
        )

    measured = sum(1 for r in runs if r.get("retrieval_rounds_measured"))
    final_dir = dataset_dir / "final"
    _write_jsonl(final_dir / "requirements.jsonl", reqs)
    _write_jsonl(final_dir / "runs.jsonl", runs)

    manifest = {
        "dataset_id": reqs[0]["dataset_id"] if reqs else None,
        "built_by": "finalize_dataset",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": collection,
        "requirements": len(reqs),
        "runs_total": len(runs),
        "runs_with_measured_rounds": measured,
        "tiers": sorted({r.get("tier") for r in runs}),
        "repeats": sorted({r.get("repeat_idx") for r in runs}),
        "splits": dict(Counter(r.get("split") for r in reqs)),
        "reused_requirements_from": sorted(
            {r.get("reused_requirements_from") for r in reqs if r.get("reused_requirements_from")}
        ),
        "spend_usd": {
            "service_execution": round(sum(r.get("execution_cost_usd") or 0.0 for r in runs), 4),
            "evaluation": round(sum(r.get("evaluation_cost_usd") or 0.0 for r in runs), 4),
        },
    }
    manifest["spend_usd"]["total"] = round(sum(manifest["spend_usd"].values()), 4)
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()

    manifest = finalize(args.dataset_dir, args.collection)
    print(json.dumps(manifest, indent=2))

    ok = True
    if manifest["requirements"] != 30:
        print(f"\n⚠️  expected 30 requirements, got {manifest['requirements']}")
        ok = False
    if manifest["runs_total"] != 90:
        print(f"⚠️  expected 90 runs, got {manifest['runs_total']}")
        ok = False
    if manifest["runs_with_measured_rounds"] != manifest["runs_total"]:
        print(f"⚠️  rounds measured on only "
              f"{manifest['runs_with_measured_rounds']}/{manifest['runs_total']} runs")
        ok = False
    if manifest["tiers"] != ["medium"]:
        print(f"⚠️  expected medium tier only, got {manifest['tiers']}")
        ok = False
    print("\n✓ finalized and complete" if ok else "\n✗ finalized WITH WARNINGS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
