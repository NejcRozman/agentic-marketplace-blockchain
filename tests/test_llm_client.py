"""The estimator's OpenRouter transport caches every answer, and never a call that got none."""
import asyncio

import pytest

from agents.provider.estimation import llm_client
from agents.provider.estimation import prompt as P

GOOD = {"choices": [{"message": {"content": '{"reasoning": "r", "prompt_rounds": [1, 2, 3]}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001}}
BAD = {"choices": [{"message": {"content": "nonsense"}}], "usage": {"cost": 0.00002}}


class FakeResponse:
    def __init__(self, item):
        self.item = item

    async def __aenter__(self):
        if isinstance(self.item, Exception):
            raise self.item
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.item


class FakeSession:
    def __init__(self, items):
        self.items, self.sent = list(items), 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.sent += 1
        return FakeResponse(self.items.pop(0))


@pytest.mark.parametrize("answers, parsed, answered", [
    ([GOOD], True, True),
    ([BAD, GOOD], True, True),
    ([RuntimeError("401"), GOOD], True, True),
    ([BAD, BAD, BAD], False, True),                # an answer, if unusable: cached, as in the experiments
    ([RuntimeError("401")] * 3, False, False),     # no answer (e.g. no valid key): not cached
])
def test_calls_without_an_answer_are_not_cached(tmp_path, monkeypatch, answers, parsed, answered):
    async def no_sleep(*_):
        return None
    monkeypatch.setattr(llm_client.asyncio, "sleep", no_sleep)
    model = llm_client.model_by_key("openai/gpt-oss-20b", llm_client.MODELS)
    kw = dict(user_message="user", run_seed=1007, agent_id=9694, system_message=P.SYSTEM_MESSAGE,
              schema=P.response_schema(3), parser=P.parse_response)
    probe = llm_client.OpenRouterProbe("sk-test", cache_dir=tmp_path)
    first = asyncio.run(probe.call_live(model, session=FakeSession(answers), **kw))
    assert first.estimate.failed is (not parsed)
    assert len(list(tmp_path.glob("*.json"))) == (1 if answered else 0)

    again = FakeSession([GOOD])
    second = asyncio.run(probe.call_live(model, session=again, **kw))
    assert second.cached is answered
    assert again.sent == (0 if answered else 1)
    assert second.estimate.failed is (answered and not parsed)


def test_cache_key_covers_prompt_model_seed_and_agent():
    model = llm_client.model_by_key("openai/gpt-oss-20b", llm_client.MODELS)
    other = llm_client.model_by_key("openai/gpt-4o-mini", llm_client.MODELS)
    base = dict(user_message="u", run_seed=1007, agent_id=9694, system_message="s")
    key = llm_client.live_cache_key(model, **base)
    assert key == llm_client.live_cache_key(model, **base)
    for change in ({"user_message": "u2"}, {"run_seed": 1008}, {"agent_id": 9695}, {"system_message": "s2"}):
        assert llm_client.live_cache_key(model, **(base | change)) != key
    assert llm_client.live_cache_key(other, **base) != key
