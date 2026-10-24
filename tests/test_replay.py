"""Replay of the recorded dataset: the job order of a run follows from its seed alone."""
import json

import pytest

from agents.common import paths
from agents.common.dataset_sampler import DatasetSampler
from experiments.runner.experiment_runner import EXPERIMENT_CONSUMER_SEED_OFFSET


def sampler(seed, root=paths.DATASET_FINAL_DIR):
    return DatasetSampler(str(root), None, requirement_selection_mode="shuffled_pass",
                          random_seed=seed + EXPERIMENT_CONSUMER_SEED_OFFSET, requirement_pass_limit=30)


@pytest.fixture
def tiny_dataset(tmp_path):
    reqs = [{"requirement_id": f"r{i}", "prompts": [f"p{i}"]} for i in range(30)]
    runs = [{"requirement_id": f"r{i}", "effort_tier": "medium", "run_index": k, "execution_cost_usd": 0.01 * (i + 1)}
            for i in range(30) for k in range(3)]
    (tmp_path / "requirements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    return tmp_path


def test_shuffled_pass_is_a_seeded_permutation_without_replacement(tiny_dataset):
    a, b, c = sampler(1007, tiny_dataset), sampler(1007, tiny_dataset), sampler(1008, tiny_dataset)
    order = [a.next_requirement_id() for _ in range(30)]
    assert sorted(order) == sorted(a.list_requirement_ids())
    assert order == b.requirement_pass_order()
    assert order != c.requirement_pass_order()
    with pytest.raises(RuntimeError):
        a.next_requirement_id()


def test_runs_of_a_request_are_served_round_robin(tiny_dataset):
    s = sampler(1007, tiny_dataset)
    assert [s.sample_run("r0", "medium")["run_index"] for _ in range(4)] == [0, 1, 2, 0]


def test_service_cid_round_trip():
    cid = DatasetSampler.build_dataset_service_cid("executor_tier_dataset_v3", "slot/7 a")
    assert DatasetSampler.parse_dataset_service_cid(cid) == {"dataset_id": "executor_tier_dataset_v3",
                                                            "requirement_id": "slot/7 a"}
    assert DatasetSampler.parse_dataset_service_cid("Qm123") is None


@pytest.mark.data
def test_every_recorded_run_drew_the_jobs_its_seed_gives():
    """All runs in the ledger: auction k sold the k-th request of the seed's pass."""
    rows = [json.loads(line) for line in paths.LEDGER.read_text().splitlines() if line.strip()]
    checked = 0
    for row in (r for r in rows if r["status"] == "ok"):
        metrics = paths.resolve(row["run_dir"].replace("/logs/", "/metrics/", 1)) / "experiment_metrics.json"
        auctions = sorted(json.loads(metrics.read_text())["auctions"], key=lambda a: a["auction_id"])
        drawn = [DatasetSampler.parse_dataset_service_cid(a["service_cid"])["requirement_id"] for a in auctions]
        assert drawn == sampler(row["seed"]).requirement_pass_order()[:len(drawn)], row["run_id"]
        checked += 1
    assert checked > 0
