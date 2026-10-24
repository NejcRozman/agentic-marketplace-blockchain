"""The cost estimators produce exactly what they produced in the experiments.

LLM-Estimate runs end to end with its transport replaced by a fake that answers
with cached responses, synthetic answers, timeouts and errors; the fake records
every request it receives (model, sampling, messages, schema, cache key).
"""
import asyncio
import json
import random
import tempfile
from functools import lru_cache

import pytest

from agents.common import llm_sampling, paths
from agents.common.dataset_sampler import DatasetSampler
from agents.provider import cost_estimators as ce
from agents.provider.estimation import llm_client
from agents.provider.estimation import prompt as P
from agents.provider.estimation import requirement as R
from agents.provider.estimation.prompt import parse_response
from conftest import GOLDEN, digest

MODELS_USED = ["openai/gpt-oss-20b", "openai/gpt-4o-mini", "openai/gpt-5.4-mini"]
SYNTHETIC = ["", "   ", "not json", "I cannot do that", "{'prompt_rounds': [1, 2]}",
             '{"prompt_rounds": [1, "x"]}', '{"prompt_rounds": [1.5, 2.5], "reasoning": "r"}',
             '```json\n{"prompt_rounds": [3, 1], "reasoning": "r"}\n```', '{"cost_multiplier": {"medium": 9}}',
             '{"prompt_rounds": [], "reasoning": ""}', '[1, 2, 3]', '{"prompt_rounds": [NaN]}']


def canon(x):
    return json.loads(json.dumps(x, sort_keys=True, default=repr))


def same(section, value):
    return digest(value, sort_keys=True) == GOLDEN["estimators"][section]


def est_dict(e):
    return {k: getattr(e, k) for k in ("cost_multiplier", "quality_delta", "confidence",
                                       "reasoning", "status", "clamped")} | {
        "failed": e.failed, "prompt_rounds": getattr(e, "prompt_rounds", None)}


@lru_cache(maxsize=None)
def dataset():
    sampler = DatasetSampler(str(paths.DATASET_FINAL_DIR), None)
    runs = [json.loads(line) for line in (paths.DATASET_FINAL_DIR / "runs.jsonl").read_text().splitlines()
            if line.strip()]
    rids = sorted({r["requirement_id"] for r in runs})
    cost_of = {}
    for r in runs:
        if r.get("execution_cost_usd"):
            cost_of.setdefault(r["requirement_id"], []).append(float(r["execution_cost_usd"]))
    return rids, {rid: sampler.get_requirement(rid) for rid in rids}, cost_of


@lru_cache(maxsize=None)
def cached_responses():
    """Every response in the LLM cache, in a fixed order."""
    out = []
    for f in sorted(paths.LLM_CACHE_DIR.glob("*.json")):
        try:
            out.append(str(json.loads(f.read_text()).get("content") or ""))
        except Exception:
            pass
    return out


def test_calibration():
    value = [canon(P.calibrate([(rng.randint(1, 8), rng.uniform(0, 20), rng.uniform(-0.01, 0.2))
                                for _ in range(rng.randint(0, 12))], two_param=True))
             for rng in [random.Random(1000 + i) for i in range(2000)]]
    assert same("calibrate", value)


def test_sampling_and_constants():
    value = [canon([llm_sampling.resolve_sampling(m), llm_sampling.chat_model_kwargs(m),
                    llm_sampling.is_pinned_model(m), llm_client.model_by_key(m, llm_client.MODELS).__dict__])
             for m in MODELS_USED]
    assert same("sampling", value)
    assert same("surface_features_const", list(R.SURFACE_FEATURES))
    assert same("deadline_reserve", ce.ESTIMATOR_DEADLINE_RESERVE_SECONDS)


@pytest.mark.data
def test_request_view_and_features():
    rids, records, _ = dataset()
    value = [canon([R.visible_view(records[r], "live").as_payload(), R.mechanical_features(R.visible_view(records[r], "x")),
                    ce.surface_features(records[r]), ce.requirement_payload(records[r])]) for r in rids]
    assert same("view", value)


@pytest.mark.data
def test_parser_on_every_cached_response():
    assert same("parser", [canon(est_dict(parse_response(c))) for c in cached_responses() + SYNTHETIC])


@pytest.mark.data
def test_llm_estimate_end_to_end(monkeypatch):
    rids, records, cost_of = dataset()
    decomposed = [c for c in cached_responses() if "prompt_rounds" in c]
    real_probe = llm_client.OpenRouterProbe
    fake_cache = tempfile.mkdtemp()

    class FakeProbe:
        script, calls = [], []

        def __init__(self, api_key, cache_dir=None, **_):
            self.real = real_probe("fake-key", cache_dir=fake_cache)

        async def call_live(self, model, **kw):
            FakeProbe.calls.append(canon({
                "model": model.model_id, "sampling": dict(model.sampling),
                "user": kw["user_message"], "system": kw["system_message"], "schema": kw["schema"],
                "seed": kw["run_seed"], "agent": kw["agent_id"],
                "key": llm_client.live_cache_key(model, user_message=kw["user_message"], run_seed=kw["run_seed"],
                                                 agent_id=kw["agent_id"], system_message=kw["system_message"]),
            }))
            action, content = FakeProbe.script.pop(0)
            if action == "timeout":
                raise asyncio.TimeoutError()
            if action == "error":
                raise RuntimeError("transport down")
            record = {"content": content, "usage": {"prompt_tokens": 1500, "completion_tokens": 250,
                                                     "cost": 0.0000831}, "attempts": 1}
            return self.real._result_from(record, model, "live", "p", 0, cached=action == "cached",
                                          parser=kw["parser"])

    monkeypatch.setattr(llm_client, "OpenRouterProbe", FakeProbe)

    def levels_for(rng):
        return {t: {"cost": round(rng.uniform(0.005, 0.2), rng.choice([3, 4, 6, 9])),
                    "quality": rng.uniform(40, 100)} for t in ("low", "medium", "high")}

    def response_for(rng, n):
        u = rng.random()
        if u < 0.05:
            return ("timeout", "")
        if u < 0.10:
            return ("error", "")
        if u < 0.30 and decomposed:
            return (rng.choice(["live", "cached"]), rng.choice(decomposed))
        if u < 0.35:
            return ("live", rng.choice(SYNTHETIC))
        k = n if rng.random() > 0.05 else max(0, n + rng.choice([-1, 1]))
        rounds = [round(rng.uniform(0.5 if rng.random() < 0.03 else 1.0, 3.0), rng.choice([0, 1, 2]))
                  for _ in range(k)]
        return (rng.choice(["live", "cached"]), json.dumps({"reasoning": "r" * rng.randint(0, 5), "prompt_rounds": rounds}))

    traces = []
    for trial in range(300):
        rng = random.Random(trial)
        est = ce.DecomposedCostEstimator(model_key=rng.choice(MODELS_USED), run_seed=1007 + trial, agent_id=9694,
                                         api_key="x")
        trace = []
        for step in range(30):
            rid = rng.choice(rids)
            if rid not in est.features:
                est.remember_features(rid, records[rid])
            est.levels = levels_for(rng)
            n = len(est.payloads[rid].get("prompts") or [])
            for tier in ("low", "medium", "high"):
                if tier == "medium":
                    FakeProbe.script.append(response_for(rng, n))
                level = est.levels[tier]["cost"] if rng.random() > 0.02 else 0.0
                m, diag = asyncio.run(est.multiplier_for(rid, tier, level, deadline_seconds=rng.choice([None, 12.5])))
                trace.append(canon([tier, m, diag]))
            FakeProbe.script.clear()
            if rng.random() < 0.5:
                est.record_outcome(rid, "medium", rng.choice(cost_of[rid]) if rng.random() > 0.03 else 0.0)
        trace.append(canon([est.calls_made, est.calls_failed, est.calls_cached, est.reasoning_cost_usd,
                            est.degraded, est.calibration, sorted(map(list, est.pending_rounds)),
                            [o.__dict__ for o in est.outcomes]]))
        traces.append(trace)
    assert same("decomposed_traces", traces)
    assert same("decomposed_calls", FakeProbe.calls)


@pytest.mark.data
def test_regression():
    rids, records, cost_of = dataset()
    traces = []
    for trial in range(1500):
        rng = random.Random(10_000 + trial)
        est = ce.RegressionCostEstimator(mode="r_guarded")
        trace = []
        for step in range(30):
            rid = rng.choice(rids)
            if rid not in est.features:
                est.remember_features(rid, records[rid])
            for tier in ("low", "medium", "high"):
                level = rng.uniform(0.005, 0.2) if rng.random() > 0.02 else 0.0
                m, diag = est.multiplier_for(rid, tier, level, deadline_seconds=5.0)
                trace.append(canon([m, diag]))
            if rng.random() < 0.55:
                est.record_outcome(rid, rng.choice(["medium", "medium", "low"]), rng.choice(cost_of[rid]))
        trace.append(canon([est.degraded, [o.__dict__ for o in est.outcomes]]))
        traces.append(trace)
    assert same("regression_traces", traces)


@pytest.mark.data
def test_oracle():
    rids, _, _ = dataset()
    oracle = ce.OracleCostEstimator(dataset_dir=str(paths.DATASET_FINAL_DIR))
    rng = random.Random(7)
    value = [canon(oracle.multiplier_for(rid, "medium", rng.choice([0.0, 0.03, 0.06, 0.1])))
             for rid in rids * 5 + ["unknown_requirement"]]
    assert same("oracle", value)
