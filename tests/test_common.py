"""Shared infrastructure: data paths, secret redaction, feedback authorisation, IPFS, commitments."""
import asyncio

import pytest
from eth_account import Account

from agents.common import ipfs_client, paths
from agents.common.commitment_store import CommitmentStore
from agents.common.feedback_auth import generate_feedback_auth, parse_feedback_auth, verify_feedback_auth_format
from experiments.runner.experiment_runner import redact_secrets


def test_recorded_paths_map_into_the_data_directory():
    assert paths.resolve("experiments/data/logs/s9_X/run1") == paths.RUN_LOGS_DIR / "s9_X" / "run1"
    assert paths.resolve("experiments/data/metrics/s9_X/run1") == paths.RUN_METRICS_DIR / "s9_X" / "run1"
    assert paths.resolve("experiments/data/llm_cache") == paths.LLM_CACHE_DIR
    assert paths.resolve("experiments/chain/s7_base.json") == paths.CHAIN_SNAPSHOT
    assert paths.resolve("data/dataset/final") == paths.DATASET_FINAL_DIR
    assert paths.resolve("experiments/conditions/base.yaml") == paths.CONDITIONS_DIR / "base.yaml"
    assert paths.resolve("/abs/path") == paths.Path("/abs/path")


def test_secrets_are_redacted_to_stable_fingerprints():
    key = "0x" + "ab" * 32
    config = {"blockchain": {"accounts": {"main_account": key, "providers": [key, "ab" * 32]}},
              "llm": {"api_key": "anything", "model": "openai/gpt-oss-20b"}, "token": "sk-or-v1-xyz",
              "experiment": {"id": "A", "run_seed": 1007}}
    out = redact_secrets(config)
    text = repr(out)
    assert "ab" * 32 not in text and "anything" not in text and "sk-or" not in text
    assert out["blockchain"]["accounts"]["main_account"] == out["blockchain"]["accounts"]["providers"][0]
    assert out["llm"]["model"] == "openai/gpt-oss-20b" and out["experiment"] == {"id": "A", "run_seed": 1007}


def test_feedback_auth_round_trip():
    signer = Account.from_key("0x" + "11" * 32)
    client = Account.from_key("0x" + "22" * 32).address
    registry = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
    auth = generate_feedback_auth(9694, client, 5, 1_900_000_000, 8000, registry, signer.address, signer.key.hex())
    assert len(auth) == 289 and verify_feedback_auth_format(auth)
    params, signature = parse_feedback_auth(auth)
    assert params["agent_id"] == 9694 and params["chain_id"] == 8000 and params["index_limit"] == 5
    assert params["client_address"] == client.lower() and params["signer_address"] == signer.address.lower()
    assert len(signature) == 65
    with pytest.raises(ValueError):
        generate_feedback_auth(9694, client, 5, 1, 8000, registry, client, signer.key.hex())


class _Response:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self, content_type=None):
        return {"ok": True}


def _fake_session(log, statuses):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, url, headers=None, timeout=None):
            log.append((url, headers))
            return _Response(statuses.pop(0))
    return Session


def test_ipfs_reads_the_dedicated_gateway_first_with_its_token(monkeypatch):
    monkeypatch.setenv("IPFS_GATEWAY", "https://example.mypinata.cloud/ipfs")
    monkeypatch.setenv("PINATA_GATEWAY_TOKEN", "tok")
    log = []
    monkeypatch.setattr(ipfs_client.aiohttp, "ClientSession", _fake_session(log, [404, 200]))
    assert asyncio.run(ipfs_client.IPFSClient(jwt="j").fetch_json("Qm1")) == {"ok": True}
    assert log == [("https://example.mypinata.cloud/ipfs/Qm1", {"x-pinata-gateway-token": "tok"}),
                   (ipfs_client.IPFS_GATEWAYS[0] + "Qm1", None)]


def test_ipfs_without_a_dedicated_gateway_uses_the_public_ones(monkeypatch):
    monkeypatch.delenv("IPFS_GATEWAY", raising=False)
    monkeypatch.delenv("PINATA_GATEWAY_TOKEN", raising=False)
    log = []
    monkeypatch.setattr(ipfs_client.aiohttp, "ClientSession", _fake_session(log, [500] * len(ipfs_client.IPFS_GATEWAYS)))
    assert asyncio.run(ipfs_client.IPFSClient(jwt="j").fetch_json("Qm1")) is None
    assert [url for url, _ in log] == [g + "Qm1" for g in ipfs_client.IPFS_GATEWAYS]


def test_commitments_survive_a_restart(tmp_path):
    store = CommitmentStore(9694, storage_dir=tmp_path)
    store.save_commitment(3, bid_amount=72_000, nonce_hex="0x" + "00" * 32, sender="0xabc", effort_tier="medium")
    store.save_commitment(4, bid_amount=80_000, nonce_hex="0x" + "01" * 32, sender="0xabc")
    store.mark_revealed(3)
    restarted = CommitmentStore(9694, storage_dir=tmp_path)
    assert restarted.pending_auction_ids() == [4]
    assert restarted.get_commitment(3)["bid_amount"] == 72_000 and restarted.get_commitment(3)["revealed"]
    assert CommitmentStore(9695, storage_dir=tmp_path).pending_auction_ids() == []
