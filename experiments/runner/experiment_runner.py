"""Runs one experiment: a local chain, the contracts, the agents and the metrics.

Starts Anvil from the chain snapshot (or a pinned fork), deploys the ReverseAuction
and a mock USDC, registers the consumer and the providers, spawns them as processes,
monitors the auctions until the target is reached, collects the metrics and shuts
everything down.
"""

import asyncio
import fcntl
import socket
import shutil
import subprocess
import hashlib
import logging
import platform
import yaml
import json
import os
import sys
import re
import signal
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.runner.metrics_collector import MetricsCollector
from agents.provider.estimation.llm_client import LIVE_CACHE_DIR

# API keys and endpoints come from agents/.env.
AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"
ENV_FILE = AGENTS_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

logger = logging.getLogger(__name__)

#: Offset added to the run seed for the consumer agent; 7430 in all experiments.
EXPERIMENT_CONSUMER_SEED_OFFSET = 7430

#: Sepolia, the network the ERC-8004 registries live on.
SEPOLIA_CHAIN_ID = 11155111

from agents.common import paths  # noqa: E402
REPO_ROOT = Path(__file__).parent.parent.parent

#: Config keys whose values are secret regardless of what they look like.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(private_?key|secret|password|passphrase|mnemonic|api_?key|access_?key|auth_?token)",
    re.IGNORECASE,
)
#: A raw ECDSA private key, with or without the 0x prefix. This is what catches the
#: substituted `blockchain.accounts.*` values, whose key names look innocuous.
PRIVATE_KEY_PATTERN = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
#: Common provider API-key prefixes.
API_KEY_PREFIXES = ("sk-", "sk-ant-", "ghp_", "gho_", "xoxb-", "xoxp-", "AIza")


def _fingerprint(value: str) -> str:
    """Stable, non-reversible tag for a redacted value.

    Unlike a bare '***', it still shows whether two runs used the same account or key.
    """
    return f"<redacted:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}>"


def _is_secret_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(PRIVATE_KEY_PATTERN.match(value)) or value.startswith(API_KEY_PREFIXES)


def redact_secrets(value: Any, key_hint: str = "") -> Any:
    """Recursively replace secrets in a resolved config with fingerprints.

    The resolved config, with environment variables already substituted, is written
    into `experiment_metrics.json`.
    """
    if isinstance(value, dict):
        return {k: redact_secrets(v, key_hint=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v, key_hint=key_hint) for v in value]
    if isinstance(value, str) and (SENSITIVE_KEY_PATTERN.search(key_hint) or _is_secret_value(value)):
        return _fingerprint(value)
    return value


def _git_output(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        logger.warning(f"Provenance: git {' '.join(args)} failed ({e})")
        return None


def build_provenance(
    config_path: Path,
    resolved_config: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record what produced this run: code version, config identity and environment.

    `git_dirty` marks a run made from an uncommitted working tree, which its `git_sha`
    alone cannot reproduce.
    """
    sha = _git_output(["rev-parse", "HEAD"])
    porcelain = _git_output(["status", "--porcelain"])
    config_path = Path(config_path)

    try:
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as e:
        logger.warning(f"Provenance: could not hash config {config_path} ({e})")
        config_sha256 = None

    experiment_section = resolved_config.get("experiment", {}) or {}

    return {
        "git_sha": sha,
        "git_dirty": None if porcelain is None else bool(porcelain),
        # Surfaced out of resolved_config so analysis can group runs by seed
        # without parsing the whole config.
        "run_seed": experiment_section.get("run_seed"),
        "llm_temperature": experiment_section.get("llm_temperature"),
        "llm_sampling_seed": experiment_section.get("llm_sampling_seed"),
        # Which cache replayed answers could come from; the live/replayed split is
        # reported per run by the estimator.
        "llm_cache_dir": str(LIVE_CACHE_DIR),
        # Run isolation, resolved rather than configured: which port
        # and chain this run actually used, and the block its fork started from.
        **(runtime or {}),
        "git_branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "resolved_config": redact_secrets(resolved_config),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


@dataclass
class ProcessHandle:
    """Handle for a managed subprocess."""
    name: str
    process: subprocess.Popen
    log_file: Path
    pid: int


class ExperimentRunner:
    """Orchestrates execution of simulation experiments."""

    #: Concurrent runs are isolated by index. Port and chain id are both
    #: derived from it, so there is one input to get wrong instead of two.
    BASE_RPC_PORT = 8545
    BASE_CHAIN_ID = 8000

    def __init__(self, config_path: Path, run_index: int = 0, run_seed: Optional[int] = None):
        """``run_index`` isolates concurrent runs: index 0 uses port 8545, index 1 port 8546, and so on."""
        self.config_path = config_path
        self.run_index = int(run_index)
        # One config serves every seed of its condition. Supplying the seed per run
        # is also what makes the pairing work: every condition at seed 1007 draws
        # the same request order.
        self._run_seed_override = None if run_seed is None else int(run_seed)
        self.config: Dict[str, Any] = {}
        self.processes: List[ProcessHandle] = []

        self.anvil_process: Optional[ProcessHandle] = None
        self.agent_processes: List[ProcessHandle] = []

        self.reverse_auction_address: Optional[str] = None
        self.payment_token_address: Optional[str] = None

        self.consumer_agent_id: Optional[int] = None
        self.provider_agent_ids: List[int] = []
        self.provider_metadata: List[Dict[str, Any]] = []

        self.run_seed: Optional[int] = None
        self._contracts_built = False
        self.llm_temperature: Optional[float] = None
        self.llm_sampling_seed: Optional[int] = None

        self.experiment_id: str = ""
        self.run_id: str = ""
        self.log_dir: Path = Path()
        self.metrics_dir: Path = Path()

        self.start_time: Optional[datetime] = None
        self.consumer_status_file: Optional[Path] = None
        self.provider_status_files: Dict[int, Path] = {}

        self.metrics_collector: Optional[MetricsCollector] = None
        self.phase_timings: Dict[str, float] = {}

    @property
    def jobs_dir(self) -> Path:
        """This run's provider->consumer result directory."""
        return self.log_dir / "jobs"

    @property
    def commitments_dir(self) -> Path:
        """This run's sealed-bid commitment store."""
        return self.log_dir / "commitments"

    @property
    def rpc_port(self) -> int:
        """This run's Anvil port."""
        return self.BASE_RPC_PORT + self.run_index

    @property
    def rpc_url(self) -> str:
        """This run's RPC endpoint.

        At index 0 the config's ``rpc_url`` is used; any other index must not read the
        ``localhost:8545`` written in the YAML.
        """
        if self.run_index == 0:
            configured = (self.config.get('blockchain') or {}).get('rpc_url')
            if configured:
                return str(configured)
        return f"http://localhost:{self.rpc_port}"

    @property
    def chain_id(self) -> int:
        """This run's chain id.

        `forge script --broadcast` writes to
        ``contracts/broadcast/<script>/<chainId>/run-latest.json``; with one chain id,
        concurrent runs would race on that file and could read each other's contract
        addresses.
        """
        return self.BASE_CHAIN_ID + self.run_index

    @property
    def fork_block_number(self) -> Optional[int]:
        """Pinned fork height, so every forked run starts from the same state, served from Anvil's cache."""
        raw = (self.config.get('blockchain') or {}).get('fork_block_number')
        return None if raw is None else int(raw)

    @staticmethod
    def _generate_run_id() -> str:
        """Unique, sortable identifier for one run of a config."""
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{timestamp}-{suffix}"

    def _allocate_provider_group_counts(self, total_count: int, groups: List[Dict[str, Any]]) -> List[int]:
        """Allocate integer provider counts from ratio weights."""
        if total_count < 0:
            raise ValueError("Provider count must be non-negative")
        if not groups:
            raise ValueError("Provider groups must be a non-empty list")

        ratios: List[float] = []
        for idx, group in enumerate(groups):
            ratio = group.get('ratio')
            if ratio is None:
                raise ValueError(f"Provider group at index {idx} is missing required field 'ratio'")

            ratio_value = float(ratio)
            if ratio_value < 0:
                raise ValueError(f"Provider group at index {idx} has invalid negative ratio: {ratio}")
            ratios.append(ratio_value)

        ratio_sum = sum(ratios)
        if ratio_sum <= 0:
            raise ValueError("Provider group ratios must sum to a positive value")

        raw_counts = [(total_count * ratio) / ratio_sum for ratio in ratios]
        allocated = [int(raw_count) for raw_count in raw_counts]
        remainder = total_count - sum(allocated)

        ranked_indexes = sorted(
            range(len(groups)),
            key=lambda index: (raw_counts[index] - allocated[index], ratios[index]),
            reverse=True,
        )
        for index in ranked_indexes[:remainder]:
            allocated[index] += 1

        return allocated

    def _expand_provider_group_configs(self, provider_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand provider config into a per-provider runtime plan."""
        provider_count = int(provider_config['count'])
        groups = provider_config.get('groups') or []
        if not groups:
            return [
                {
                    'group_name': 'default',
                    'group_config': provider_config,
                    'group_index': 0,
                    'group_member_index': index,
                }
                for index in range(provider_count)
            ]

        shared_config = dict(provider_config.get('config', {}))
        shared_behavior = dict(provider_config.get('behavior', {}))
        allocated_counts = self._allocate_provider_group_counts(provider_count, groups)

        provider_plan: List[Dict[str, Any]] = []
        for group_index, (group, group_count) in enumerate(zip(groups, allocated_counts)):
            group_name = group.get('name', f'group_{group_index + 1}')
            group_config = {
                'config': {**shared_config, **group.get('config', {})},
                'behavior': {**shared_behavior, **group.get('behavior', {})},
            }

            for group_member_index in range(group_count):
                provider_plan.append({
                    'group_name': group_name,
                    'group_config': group_config,
                    'group_index': group_index,
                    'group_member_index': group_member_index,
                })

        if len(provider_plan) != provider_count:
            raise RuntimeError(
                f"Provider plan expansion mismatch: expected {provider_count}, got {len(provider_plan)}"
            )

        return provider_plan

    @staticmethod
    def _profile_from_group_config(provider_id: int, group_name: str, group_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build normalized provider profile metadata for metrics and analysis."""
        provider_behavior = group_config.get('behavior', {})
        provider_cfg = group_config.get('config', {})
        architecture = provider_behavior.get('architecture', provider_cfg.get('architecture'))
        reasoning_mode = provider_behavior.get('reasoning_mode', provider_cfg.get('reasoning_mode'))
        # Which cost estimator this group bids with: per group, because the
        # strategies in one market differ in it.
        cost_estimator = provider_behavior.get('cost_estimator', provider_cfg.get('cost_estimator'))
        # The decline rule: per provider config, not per group, so every agent in a
        # market runs the same rule.
        action_space = provider_cfg.get('action_space')
        heuristic_strategy = provider_behavior.get('heuristic_strategy', provider_cfg.get('heuristic_strategy'))
        heuristic_min_margin = provider_behavior.get('heuristic_min_margin', provider_cfg.get('heuristic_min_margin'))
        heuristic_max_margin = provider_behavior.get('heuristic_max_margin', provider_cfg.get('heuristic_max_margin'))
        heuristic_recovery_pressure_max = provider_behavior.get(
            'heuristic_recovery_pressure_max', provider_cfg.get('heuristic_recovery_pressure_max')
        )
        heuristic_initial_margin_mode = provider_behavior.get(
            'heuristic_initial_margin_mode', provider_cfg.get('heuristic_initial_margin_mode')
        )
        heuristic_concession_exponent = provider_behavior.get(
            'heuristic_concession_exponent', provider_cfg.get('heuristic_concession_exponent')
        )
        heuristic_bid_tick_usd = provider_behavior.get('heuristic_bid_tick_usd', provider_cfg.get('heuristic_bid_tick_usd'))
        heuristic_min_rebid_delta_usd = provider_behavior.get(
            'heuristic_min_rebid_delta_usd', provider_cfg.get('heuristic_min_rebid_delta_usd')
        )
        llm_margin_bounds_enabled = provider_behavior.get(
            'llm_margin_bounds_enabled', provider_cfg.get('llm_margin_bounds_enabled')
        )
        llm_last_margin_enabled = provider_behavior.get(
            'llm_last_margin_enabled', provider_cfg.get('llm_last_margin_enabled')
        )
        llm_outcome_history_enabled = provider_behavior.get(
            'llm_outcome_history_enabled', provider_cfg.get('llm_outcome_history_enabled')
        )
        llm_outcome_history_limit = provider_behavior.get(
            'llm_outcome_history_limit', provider_cfg.get('llm_outcome_history_limit')
        )
        llm_reflection_enabled = provider_behavior.get(
            'llm_reflection_enabled', provider_cfg.get('llm_reflection_enabled')
        )
        llm_structured_bid_enabled = provider_behavior.get(
            'llm_structured_bid_enabled', provider_cfg.get('llm_structured_bid_enabled')
        )
        blockchain_llm_model = provider_behavior.get('blockchain_llm_model', provider_cfg.get('blockchain_llm_model'))
        service_llm_model = provider_behavior.get('service_llm_model', provider_cfg.get('service_llm_model'))
        initial_service_execution_cost = provider_cfg.get('initial_service_execution_cost')
        # The commit-window freshness gate and how often the provider polls, both
        # per run.
        bid_freshness_min_seconds = provider_cfg.get('bid_freshness_min_seconds')
        provider_monitor_interval = provider_cfg.get('provider_monitor_interval')
        initial_service_execution_cost_low = provider_cfg.get('initial_service_execution_cost_low')
        initial_service_execution_cost_medium = provider_cfg.get('initial_service_execution_cost_medium')
        initial_service_execution_cost_high = provider_cfg.get('initial_service_execution_cost_high')
        initial_quality_rating = provider_cfg.get('initial_quality_rating')
        initial_quality_rating_low = provider_cfg.get('initial_quality_rating_low')
        initial_quality_rating_medium = provider_cfg.get('initial_quality_rating_medium')
        initial_quality_rating_high = provider_cfg.get('initial_quality_rating_high')
        initial_non_execution_cost_per_participation = provider_cfg.get('initial_non_execution_cost_per_participation')

        return {
            'provider_id': int(provider_id),
            'group_name': group_name or 'default',
            'architecture': str(architecture) if architecture is not None else None,
            'reasoning_mode': str(reasoning_mode) if reasoning_mode is not None else None,
            'heuristic_strategy': str(heuristic_strategy) if heuristic_strategy is not None else None,
            'cost_estimator': str(cost_estimator) if cost_estimator is not None else None,
            'action_space': action_space,
            'heuristic_min_margin': heuristic_min_margin,
            'heuristic_max_margin': heuristic_max_margin,
            'heuristic_recovery_pressure_max': heuristic_recovery_pressure_max,
            'heuristic_initial_margin_mode': heuristic_initial_margin_mode,
            'heuristic_concession_exponent': heuristic_concession_exponent,
            'heuristic_bid_tick_usd': heuristic_bid_tick_usd,
            'heuristic_min_rebid_delta_usd': heuristic_min_rebid_delta_usd,
            'llm_margin_bounds_enabled': llm_margin_bounds_enabled,
            'llm_last_margin_enabled': llm_last_margin_enabled,
            'llm_outcome_history_enabled': llm_outcome_history_enabled,
            'llm_outcome_history_limit': llm_outcome_history_limit,
            'llm_reflection_enabled': llm_reflection_enabled,
            'llm_structured_bid_enabled': llm_structured_bid_enabled,
            'blockchain_llm': blockchain_llm_model,
            'service_llm': service_llm_model,
            'initial_service_execution_cost': initial_service_execution_cost,
            'bid_freshness_min_seconds': bid_freshness_min_seconds,
            'provider_monitor_interval': provider_monitor_interval,
            'initial_service_execution_cost_low': initial_service_execution_cost_low,
            'initial_service_execution_cost_medium': initial_service_execution_cost_medium,
            'initial_service_execution_cost_high': initial_service_execution_cost_high,
            'initial_quality_rating': initial_quality_rating,
            'initial_quality_rating_low': initial_quality_rating_low,
            'initial_quality_rating_medium': initial_quality_rating_medium,
            'initial_quality_rating_high': initial_quality_rating_high,
            'initial_non_execution_cost_per_participation': initial_non_execution_cost_per_participation,
        }

    def _provider_profiles_for_metrics(self) -> List[Dict[str, Any]]:
        """Build provider profile metadata keyed by registered provider ids."""
        profiles: List[Dict[str, Any]] = []
        for provider_meta in self.provider_metadata:
            profiles.append(
                self._profile_from_group_config(
                    provider_id=provider_meta['agent_id'],
                    group_name=provider_meta.get('group_name', 'default'),
                    group_config=provider_meta['group_config'],
                )
            )
        return profiles

    def load_config(self):
        """Load and validate experiment configuration from YAML."""
        logger.info(f"Loading configuration from {self.config_path}")

        with open(self.config_path, 'r') as f:
            config_text = f.read()

        # Substitute ${VAR_NAME} with the environment variable.
        def replace_env_var(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable {var_name} not found")
            return value

        config_text = re.sub(r'\$\{(\w+)\}', replace_env_var, config_text)

        self.config = yaml.safe_load(config_text)

        self.experiment_id = self.config['experiment']['id']

        # The run seed. One number per run; every agent derives its own
        # stream from it as `run_seed + agent_id`. Paired arms share it, so the
        # market draw is common-mode and differences out of the comparison.
        if self._run_seed_override is not None:
            self.run_seed = self._run_seed_override
            # Keep the resolved value in the config so provenance records what
            # actually ran, not what the file happened to say.
            self.config['experiment']['run_seed'] = self.run_seed
        else:
            run_seed_raw = self.config['experiment'].get('run_seed')
            self.run_seed = None if run_seed_raw is None else int(run_seed_raw)

        # LLM sampling, run-level rather than per group, so every agent in a market
        # samples the same way. Left unset, each model uses its pinned block.
        llm_temperature_raw = self.config['experiment'].get('llm_temperature')
        self.llm_temperature = None if llm_temperature_raw is None else float(llm_temperature_raw)
        llm_sampling_seed_raw = self.config['experiment'].get('llm_sampling_seed')
        self.llm_sampling_seed = None if llm_sampling_seed_raw is None else int(llm_sampling_seed_raw)

        # Each run gets its own timestamped directory under the experiment id, so
        # repeated runs of a config do not overwrite each other.
        self.run_id = self._generate_run_id()
        self.log_dir = paths.RUN_LOGS_DIR / self.experiment_id / self.run_id
        # Start from an empty result directory: the consumer reads
        # `auction_<id>/result.json` by id alone, so any file surviving from a
        # previous run would be picked up as this run's own outcome.
        for sub in ("jobs", "commitments"):
            d = self.log_dir / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = paths.RUN_METRICS_DIR / self.experiment_id / self.run_id

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_collector = MetricsCollector(
            experiment_id=self.experiment_id,
            metrics_dir=self.metrics_dir,
            logs_dir=self.log_dir
        )

        provenance = build_provenance(
            self.config_path,
            self.config,
            runtime={
                "run_index": self.run_index,
                "rpc_url": self.rpc_url,
                "chain_id": self.chain_id,
                "fork_block_number": self.fork_block_number,
                # The snapshot is the starting chain state; its hash identifies it.
                "state_snapshot": None if self.state_snapshot is None else str(self.state_snapshot),
                "state_snapshot_sha256": self._state_snapshot_sha256(),
            },
        )
        self.metrics_collector.record_provenance(provenance)
        if provenance.get("git_dirty"):
            logger.warning(
                "Provenance: working tree is dirty at run start (git_sha=%s). "
                "This run is excluded from confirmatory analysis.",
                provenance.get("git_sha"),
            )

        logger.info(f"✓ Configuration loaded for experiment: {self.experiment_id}")
        logger.info(f"  Run ID: {self.run_id}")
        logger.info(f"  Run seed: {self.run_seed if self.run_seed is not None else 'UNSEEDED'}")
        logger.info(f"  Run index: {self.run_index} (port {self.rpc_port}, chain id {self.chain_id})")
        logger.info(f"  Git SHA: {provenance.get('git_sha')} (dirty={provenance.get('git_dirty')})")
        logger.info(f"  Log directory: {self.log_dir}")
        logger.info(f"  Metrics directory: {self.metrics_dir}")

    def cleanup_orphaned_processes(self):
        """Refuse to start if this run's port is already in use.

        Processes are not killed by name, which would also hit concurrent runs; `cleanup()`
        kills this run's own process groups by pid.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", self.rpc_port)) == 0:
                raise RuntimeError(
                    f"port {self.rpc_port} is already in use — another run is live "
                    f"at run_index {self.run_index}, or a previous Anvil was left "
                    f"behind. Refusing to start rather than sharing a chain."
                )
        logger.info(f"✓ Port {self.rpc_port} is free")

    @property
    def state_snapshot(self) -> Optional[Path]:
        """Local chain snapshot to start from instead of forking.

        A snapshot is a dump of a chain forked once: a run makes no external RPC calls and
        starts from byte-identical state. Built by ``experiments/chain/build_chain_snapshot.py``.
        """
        raw = (self.config.get('blockchain') or {}).get('state_snapshot')
        if not raw:
            return None
        return paths.resolve(raw)

    def _state_snapshot_sha256(self) -> Optional[str]:
        """Hash of the snapshot this run started from, for provenance."""
        snapshot = self.state_snapshot
        if snapshot is None or not snapshot.exists():
            return None
        return hashlib.sha256(snapshot.read_bytes()).hexdigest()

    def build_anvil_command(self) -> List[str]:
        """Assemble Anvil's command line."""
        cmd = [
            "anvil",
            "--host", "0.0.0.0",
            # One port and one chain id per run, so concurrent runs share neither a chain
            # nor a forge broadcast record.
            "--port", str(self.rpc_port),
            "--chain-id", str(self.chain_id),
            "--accounts", "1",  # One default account, to limit fork RPC requests
            "--block-time", "1",  # Mine every second so block.timestamp advances
        ]

        snapshot = self.state_snapshot
        if snapshot is not None:
            if not snapshot.exists():
                raise FileNotFoundError(
                    f"chain snapshot {snapshot} not found; build it with "
                    f"experiments/chain/build_chain_snapshot.py"
                )
            # No --fork-url at all: this run makes zero external RPC calls. The
            # snapshot loads under whichever chain id this run was given, so one
            # snapshot serves every concurrent run.
            cmd += ["--load-state", str(snapshot)]
            logger.info(f"  Chain from snapshot {snapshot.name} (offline, no fork)")
            return cmd

        fork_network = self.config['blockchain']['fork_network']
        infura_key_env = self.config['blockchain']['infura_key_env']
        infura_key = os.getenv(infura_key_env)
        if not infura_key:
            raise ValueError(f"Environment variable {infura_key_env} not found")

        cmd += ["--fork-url", f"https://{fork_network}.infura.io/v3/{infura_key}"]

        # Pin the fork height, so the starting state is reproducible; Foundry caches
        # fork state per block under ~/.foundry/cache/rpc/.
        if self.fork_block_number is not None:
            cmd += [
                "--fork-block-number", str(self.fork_block_number),
                # Anvil's documented offline-start mode: skips the startup RPC
                # that asks the remote for its chain id.
                "--fork-chain-id", str(SEPOLIA_CHAIN_ID),
            ]
            logger.info(f"  Fork pinned at block {self.fork_block_number}")
        else:
            logger.warning(
                "Fork block is NOT pinned; this run starts from Sepolia's current "
                "head and is not reproducible. Set blockchain.fork_block_number "
                "or, better, blockchain.state_snapshot."
            )

        return cmd

    async def start_anvil(self):
        """Start Anvil: from a local state snapshot if one is configured, else a fork."""
        logger.info("Starting Anvil blockchain...")

        log_file = self.log_dir / "anvil.log"
        anvil_cmd = self.build_anvil_command()

        logger.info(f"  RPC: {self.rpc_url} (chain id {self.chain_id})")

        # Anvil reads none of these variables but inherits them, and the campaign
        # identifies a process's chain from `/proc/<pid>/environ`, so set them to this
        # run's values.
        anvil_env = {
            **os.environ,
            "BLOCKCHAIN_RPC_URL": self.rpc_url,
            "BLOCKCHAIN_CHAIN_ID": str(self.chain_id),
        }
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                anvil_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=anvil_env,
                preexec_fn=os.setsid  # Own process group, for a clean shutdown
            )

        self.anvil_process = ProcessHandle(
            name="anvil",
            process=process,
            log_file=log_file,
            pid=process.pid
        )
        self.processes.append(self.anvil_process)

        # Wait until the RPC answers.
        logger.info("Waiting for Anvil to initialize...")
        max_wait = 30
        for i in range(max_wait):
            await asyncio.sleep(1)

            if self.anvil_process.process.poll() is not None:
                raise RuntimeError("Anvil process terminated unexpectedly")

            try:
                result = subprocess.run(
                    ["curl", "-s", "-X", "POST",
                     "-H", "Content-Type: application/json",
                     "--data", '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
                     self.rpc_url],
                    capture_output=True,
                    timeout=1
                )
                if result.returncode == 0 and b"result" in result.stdout:
                    logger.info(f"✓ Anvil ready after {i+1} seconds")
                    break
            except:
                pass
        else:
            raise RuntimeError("Anvil failed to start within 30 seconds")

        logger.info(f"✓ Anvil started (PID: {self.anvil_process.pid})")

        # Reset the deployer's nonce, which a forked account carries over from Sepolia.
        logger.info("Resetting account nonce...")
        personal_account = self.config['blockchain']['accounts']['main_account']
        from eth_account import Account
        account = Account.from_key(personal_account)
        address = account.address

        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "-H", "Content-Type: application/json",
             "--data", f'{{"jsonrpc":"2.0","method":"anvil_setNonce","params":["{address}","0x0"],"id":1}}',
             self.rpc_url],
            capture_output=True
        )

        if result.returncode == 0:
            logger.info(f"✓ Reset nonce for {address}")
        else:
            logger.warning(f"Failed to reset nonce: {result.stderr.decode()}")

    def run_forge_script(self, script_name: str, env_vars: Dict[str, str]) -> str:
        """Run a forge script with extra environment variables and return its stdout."""
        logger.info(f"Running forge script: {script_name}")

        contracts_dir = Path(__file__).parent.parent.parent / "contracts"
        script_path = contracts_dir / "script" / script_name

        self.build_contracts_once()

        env = os.environ.copy()
        env.update(env_vars)

        result = subprocess.run(
            [
                "forge", "script", str(script_path),
                "--rpc-url", self.rpc_url,
                "--broadcast"
            ],
            cwd=contracts_dir,
            env=env,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Forge script failed:\n{result.stderr}")
            raise RuntimeError(f"Forge script {script_name} failed")

        logger.info(f"✓ {script_name} executed successfully")
        return result.stdout

    def build_contracts_once(self) -> None:
        """Compile the contracts, at most one process at a time.

        `contracts/cache/` and `contracts/out/` are shared, and concurrent compilations
        corrupt each other's artifacts. Compiling up front under an exclusive file lock
        means no `forge script` ever compiles, also for a run started by hand.
        """
        if self._contracts_built:
            return

        contracts_dir = Path(__file__).parent.parent.parent / "contracts"
        lock_path = contracts_dir / ".forge-build.lock"

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                logger.info("Building contracts (exclusive)...")
                result = subprocess.run(
                    ["forge", "build"],
                    cwd=contracts_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"forge build failed:\n{result.stderr}")
                    raise RuntimeError("forge build failed")
                logger.info("✓ Contracts built")
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        self._contracts_built = True

    async def deploy_contracts(self):
        """Deploy the ReverseAuction and the mock USDC."""
        logger.info("Deploying contracts...")

        # The deployer is the config's main account.
        private_key = self.config['blockchain']['accounts']['deployer']
        await self._ensure_deployer_funded(private_key)

        env_vars = {
            "PRIVATE_KEY": private_key,
            "IDENTITY_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['identity_registry'],
            "REPUTATION_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['reputation_registry'],
            "INITIAL_REPUTATION_DEFAULT": str(self.config['blockchain'].get('initial_reputation_default', 50))
        }

        output = self.run_forge_script("Deploy.s.sol", env_vars)

        # Parse the addresses from lines such as "Mock USDC deployed at: 0x...".
        usdc_match = re.search(r'Mock USDC deployed at:\s*(0x[a-fA-F0-9]+)', output)
        auction_match = re.search(r'ReverseAuction deployed at:\s*(0x[a-fA-F0-9]+)', output)

        if not usdc_match or not auction_match:
            raise RuntimeError("Failed to parse contract addresses from deploy script output")

        self.payment_token_address = usdc_match.group(1)
        self.reverse_auction_address = auction_match.group(1)

        logger.info("✓ Contracts deployed:")
        logger.info(f"  Mock USDC: {self.payment_token_address}")
        logger.info(f"  ReverseAuction: {self.reverse_auction_address}")

    async def fund_consumer_account(self):
        """Fund the consumer with ETH and USDC when it is a separate account."""
        logger.info("Checking if consumer needs funding...")

        consumer_pk = self.config['blockchain']['accounts']['consumer']
        deployer_pk = self.config['blockchain']['accounts']['deployer']

        from eth_account import Account
        consumer_address = Account.from_key(consumer_pk).address
        deployer_address = Account.from_key(deployer_pk).address

        # Only when the consumer and the deployer are different accounts.
        if consumer_address.lower() == deployer_address.lower():
            logger.info("Consumer and deployer are the same account - no funding needed")
            return

        logger.info(f"Funding consumer account: {consumer_address}")

        funding_config = self.config['blockchain']['accounts'].get('consumer_funding', {})
        eth_amount = funding_config.get('eth_amount', 1000000000000000000)  # 1 ETH
        usdc_amount = funding_config.get('usdc_amount', 1000000000)  # 1000 USDC

        env_vars = {
            "DEPLOYER_PRIVATE_KEY": deployer_pk,
            "CONSUMER_ADDRESS": consumer_address,
            "USDC_ADDRESS": self.payment_token_address,
            "FUND_AMOUNT_ETH": str(eth_amount),
            "FUND_AMOUNT_USDC": str(usdc_amount)
        }

        self.run_forge_script("FundConsumer.s.sol", env_vars)

        logger.info("✓ Consumer account funded")
        logger.info(f"  ETH: {eth_amount / 1e18} ETH")
        logger.info(f"  USDC: {usdc_amount / 1e6} USDC")

    @staticmethod
    def _derive_provider_key(base_pk: str, index: int) -> str:
        """Return a deterministic unique private key derived from base_pk and provider index."""
        import hashlib
        parent_bytes = bytes.fromhex(base_pk.lstrip('0x'))
        seed = parent_bytes + index.to_bytes(32, 'big')
        return '0x' + hashlib.sha256(seed).hexdigest()

    async def _ensure_deployer_funded(self, private_key: str, minimum_wei: int = 10 ** 18) -> None:
        """Give a fresh deployer key enough ETH on the local chain to deploy and fund.

        The experiments' deployer already has a balance (3.95 ETH) in the chain snapshot;
        a new key has none.
        """
        from eth_account import Account
        from web3 import Web3

        address = Account.from_key(private_key).address
        balance = Web3(Web3.HTTPProvider(self.rpc_url)).eth.get_balance(address)
        if balance < minimum_wei:
            await self._fund_address_eth(address, 10 * 10 ** 18)
            logger.info(f"✓ Funded deployer {address} with 10 ETH for gas")

    async def _fund_address_eth(self, target_address: str, amount_wei: int = 10 ** 18) -> None:
        """Set an address's ETH balance with Anvil's `anvil_setBalance`.

        Retries when Anvil answers more slowly than the client's read timeout, instead of
        aborting the run.
        """
        from web3 import Web3

        rpc_url = self.rpc_url
        w3 = Web3(Web3.HTTPProvider(rpc_url))

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                w3.provider.make_request("anvil_setBalance", [target_address, hex(amount_wei)])
                return
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                backoff_seconds = 5 * (2 ** (attempt - 1))
                logger.warning(
                    f"anvil_setBalance failed for {target_address} "
                    f"(attempt {attempt}/{max_attempts}): {e}. Retrying in {backoff_seconds}s..."
                )
                await asyncio.sleep(backoff_seconds)

    def register_agent(self, private_key: str) -> int:
        """Register an agent with the given private key and return its agent id."""
        env_vars = {
            "PRIVATE_KEY": private_key,
            "IDENTITY_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['identity_registry']
        }

        output = self.run_forge_script("RegisterAgent.s.sol", env_vars)

        # Parse the id from a line such as "Agent ID: 1234".
        match = re.search(r'Agent ID:\s*(\d+)', output)

        if not match:
            raise RuntimeError("Failed to parse agent ID from registration output")

        agent_id = int(match.group(1))
        logger.info(f"✓ Agent registered with ID: {agent_id}")

        return agent_id

    async def register_agents(self):
        """Register the consumer and the providers."""
        logger.info("Registering agents...")

        consumer_pk = self.config['blockchain']['accounts']['consumer']
        self.consumer_agent_id = self.register_agent(consumer_pk)
        logger.info(f"Consumer agent ID: {self.consumer_agent_id}")

        provider_pks = self.config['blockchain']['accounts']['providers']
        provider_config = self.config['agents']['providers']
        provider_plan = self._expand_provider_group_configs(provider_config)

        provider_count = len(provider_plan)
        self.provider_metadata = []

        group_counts: Dict[str, int] = {}
        for plan_entry in provider_plan:
            group_name = plan_entry['group_name']
            group_counts[group_name] = group_counts.get(group_name, 0) + 1

        logger.info(
            "Resolved provider mix: %s",
            ", ".join(f"{name}={count}" for name, count in group_counts.items())
        )

        for i in range(provider_count):
            plan_entry = provider_plan[i]
            if i < len(provider_pks):
                pk = provider_pks[i]
            else:
                # Derive a unique key, so this provider has its own nonce.
                pk = self._derive_provider_key(provider_pks[0], i)
                from eth_account import Account as EthAccount
                derived_addr = EthAccount.from_key(pk).address
                logger.info(f"Derived unique key for provider {i + 1}: {derived_addr}")
                await self._fund_address_eth(derived_addr)
                logger.info(f"✓ Funded provider {i + 1} ({derived_addr}) with 1 ETH for gas")
            agent_id = self.register_agent(pk)
            self.provider_agent_ids.append(agent_id)

            self.provider_metadata.append({
                'agent_id': agent_id,
                'private_key': pk,
                'group_name': plan_entry['group_name'],
                'group_config': plan_entry['group_config'],
                'index': i,
            })

            logger.info(f"Provider {i+1} agent ID: {agent_id} ({plan_entry['group_name']})")

        logger.info(f"✓ All agents registered ({len(self.provider_agent_ids)} providers)")

    def spawn_agent_process(
        self,
        agent_type: str,
        agent_id: int,
        private_key: str,
        additional_args: Dict[str, Any] = None
    ) -> ProcessHandle:
        """Spawn a consumer or provider process with the given agent id, key and CLI arguments."""
        logger.info(f"Spawning {agent_type} agent (ID: {agent_id})...")

        agents_dir = Path(__file__).parent.parent.parent / "agents"

        if agent_type == "consumer":
            script = agents_dir / "consumer" / "consumer_orchestrator.py"
            status_file = self.log_dir / f"consumer_{agent_id}_status.json"
            self.consumer_status_file = status_file
        else:  # provider_agent
            script = agents_dir / "provider" / "orchestrator.py"
            status_file = self.log_dir / f"provider_{agent_id}_status.json"
            self.provider_status_files[agent_id] = status_file

        # The runner's own interpreter, so the agents run in the same environment.
        cmd = [
            sys.executable, str(script),
            "--agent-id", str(agent_id),
            "--private-key", private_key,
            "--status-file", str(status_file),
            "--log-level", "INFO"
        ]

        if agent_type == "consumer" and additional_args:
            if 'eligible_providers' in additional_args:
                cmd.extend(["--eligible-providers"] + [str(pid) for pid in additional_args['eligible_providers']])
            if 'provider_pool' in additional_args:
                cmd.extend(["--provider-pool"] + [str(pid) for pid in additional_args['provider_pool']])
            if 'eligible_per_auction' in additional_args:
                cmd.extend(["--eligible-per-auction", str(additional_args['eligible_per_auction'])])
            if 'max_budget' in additional_args:
                cmd.extend(["--max-budget", str(additional_args['max_budget'])])
            if 'commit_duration' in additional_args:
                cmd.extend(["--commit-duration", str(additional_args['commit_duration'])])
            if 'reveal_duration' in additional_args:
                cmd.extend(["--reveal-duration", str(additional_args['reveal_duration'])])
            if 'reputation_weight' in additional_args:
                cmd.extend(["--reputation-weight", str(additional_args['reputation_weight'])])
            if 'check_interval' in additional_args:
                cmd.extend(["--check-interval", str(additional_args['check_interval'])])
            if 'check_interval_jitter' in additional_args:
                cmd.extend(["--check-interval-jitter", str(additional_args['check_interval_jitter'])])
            if 'auto_create_auction' in additional_args and additional_args['auto_create_auction']:
                cmd.append("--auto-create-auction")
            if 'num_auctions' in additional_args:
                cmd.extend(["--num-auctions", str(additional_args['num_auctions'])])
            if 'pdf_directory' in additional_args:
                cmd.extend(["--pdf-directory", str(additional_args['pdf_directory'])])
            if 'pdfs_per_service' in additional_args:
                cmd.extend(["--pdfs-per-service", str(additional_args['pdfs_per_service'])])
            if 'service_generation_mode' in additional_args:
                cmd.extend(["--service-generation-mode", str(additional_args['service_generation_mode'])])
            if 'service_generation_iterations' in additional_args:
                cmd.extend(["--service-generation-iterations", str(additional_args['service_generation_iterations'])])
            if 'service_generation_seed' in additional_args:
                cmd.extend(["--service-generation-seed", str(additional_args['service_generation_seed'])])
            if 'service_complexity' in additional_args:
                cmd.extend(["--service-complexity", str(additional_args['service_complexity'])])
            if 'auction_creation_delay' in additional_args:
                cmd.extend(["--auction-creation-delay", str(additional_args['auction_creation_delay'])])
            if 'inter_auction_delay' in additional_args:
                cmd.extend(["--inter-auction-delay", str(additional_args['inter_auction_delay'])])
        elif agent_type == "provider_agent" and additional_args:
            if 'group_name' in additional_args:
                cmd.extend(["--group-name", str(additional_args['group_name'])])
            if 'architecture' in additional_args:
                cmd.extend(["--architecture", str(additional_args['architecture'])])
            if 'reasoning_mode' in additional_args:
                cmd.extend(["--reasoning-mode", str(additional_args['reasoning_mode'])])
            if 'heuristic_strategy' in additional_args:
                cmd.extend(["--heuristic-strategy", str(additional_args['heuristic_strategy'])])
            if 'heuristic_min_margin' in additional_args:
                cmd.extend(["--heuristic-min-margin", str(additional_args['heuristic_min_margin'])])
            if 'heuristic_max_margin' in additional_args:
                cmd.extend(["--heuristic-max-margin", str(additional_args['heuristic_max_margin'])])
            if 'heuristic_concession_exponent' in additional_args:
                cmd.extend(["--heuristic-concession-exponent", str(additional_args['heuristic_concession_exponent'])])
            if 'heuristic_bid_tick_usd' in additional_args:
                cmd.extend(["--heuristic-bid-tick-usd", str(additional_args['heuristic_bid_tick_usd'])])
            if 'heuristic_min_rebid_delta_usd' in additional_args:
                cmd.extend(["--heuristic-min-rebid-delta-usd", str(additional_args['heuristic_min_rebid_delta_usd'])])

        env = os.environ.copy()

        # Every agent, consumer and provider alike, seeds its own RNG from this.
        # Unset only when the config omits `experiment.run_seed`.
        if self.run_seed is not None:
            env["RUN_SEED"] = str(self.run_seed)
        if self.llm_temperature is not None:
            env["LLM_TEMPERATURE"] = str(self.llm_temperature)
        if self.llm_sampling_seed is not None:
            env["LLM_SAMPLING_SEED"] = str(self.llm_sampling_seed)

        if additional_args:
            if 'service_selection' in additional_args:
                env["SERVICE_SELECTION"] = str(additional_args['service_selection'])
            if 'service_source_mode' in additional_args:
                env["SERVICE_SOURCE_MODE"] = str(additional_args['service_source_mode'])
            if 'sampled_dataset_dir' in additional_args:
                env["SAMPLED_DATASET_DIR"] = str(paths.resolve(additional_args['sampled_dataset_dir']))
            if 'sampled_dataset_id' in additional_args:
                env["SAMPLED_DATASET_ID"] = str(additional_args['sampled_dataset_id'])
            if 'sampled_dataset_run_selection_mode' in additional_args:
                env["SAMPLED_DATASET_RUN_SELECTION_MODE"] = str(additional_args['sampled_dataset_run_selection_mode'])
            if 'sampled_dataset_selection_seed' in additional_args:
                env["SAMPLED_DATASET_SELECTION_SEED"] = str(additional_args['sampled_dataset_selection_seed'])
            if 'sampled_dataset_pass_limit' in additional_args:
                env["SAMPLED_DATASET_PASS_LIMIT"] = str(additional_args['sampled_dataset_pass_limit'])

        if agent_type == "provider_agent" and additional_args and 'quality_profile' in additional_args:
            env["QUALITY_PROFILE"] = additional_args['quality_profile']
            logger.info(f"Setting QUALITY_PROFILE={additional_args['quality_profile']} for provider {agent_id}")

        # Provider reasoning configuration, also set in the environment.
        if agent_type == "provider_agent" and additional_args:
            if 'group_name' in additional_args:
                env["PROVIDER_GROUP_NAME"] = str(additional_args['group_name'])
            if 'architecture' in additional_args:
                env["ARCHITECTURE"] = str(additional_args['architecture'])
            if 'reasoning_mode' in additional_args:
                env["REASONING_MODE"] = str(additional_args['reasoning_mode'])
            if 'heuristic_strategy' in additional_args:
                env["HEURISTIC_STRATEGY"] = str(additional_args['heuristic_strategy'])
            if additional_args.get('cost_estimator') is not None:
                env["COST_ESTIMATOR"] = str(additional_args['cost_estimator'])
            if additional_args.get('action_space') is not None:
                env["ACTION_SPACE"] = str(additional_args['action_space'])
            if 'heuristic_min_margin' in additional_args:
                env["HEURISTIC_MIN_MARGIN"] = str(additional_args['heuristic_min_margin'])
            if 'heuristic_max_margin' in additional_args:
                env["HEURISTIC_MAX_MARGIN"] = str(additional_args['heuristic_max_margin'])
            if 'heuristic_recovery_pressure_max' in additional_args:
                env["HEURISTIC_RECOVERY_PRESSURE_MAX"] = str(additional_args['heuristic_recovery_pressure_max'])
            if 'heuristic_initial_margin_mode' in additional_args:
                env["HEURISTIC_INITIAL_MARGIN_MODE"] = str(additional_args['heuristic_initial_margin_mode'])
            if 'llm_margin_bounds_enabled' in additional_args:
                env["LLM_MARGIN_BOUNDS_ENABLED"] = str(additional_args['llm_margin_bounds_enabled'])
            if 'llm_last_margin_enabled' in additional_args:
                env["LLM_LAST_MARGIN_ENABLED"] = str(additional_args['llm_last_margin_enabled'])
            if 'llm_outcome_history_enabled' in additional_args:
                env["LLM_OUTCOME_HISTORY_ENABLED"] = str(additional_args['llm_outcome_history_enabled'])
            if 'llm_outcome_history_limit' in additional_args:
                env["LLM_OUTCOME_HISTORY_LIMIT"] = str(additional_args['llm_outcome_history_limit'])
            if 'llm_reflection_enabled' in additional_args:
                env["LLM_REFLECTION_ENABLED"] = str(additional_args['llm_reflection_enabled'])
            if 'llm_structured_bid_enabled' in additional_args:
                env["LLM_STRUCTURED_BID_ENABLED"] = str(additional_args['llm_structured_bid_enabled'])
            if 'blockchain_llm_model' in additional_args:
                env["BLOCKCHAIN_LLM_MODEL"] = str(additional_args['blockchain_llm_model'])
            if 'service_llm_model' in additional_args:
                env["SERVICE_LLM_MODEL"] = str(additional_args['service_llm_model'])
            if 'heuristic_concession_exponent' in additional_args:
                env["HEURISTIC_CONCESSION_EXPONENT"] = str(additional_args['heuristic_concession_exponent'])
            if 'heuristic_bid_tick_usd' in additional_args:
                env["HEURISTIC_BID_TICK_USD"] = str(additional_args['heuristic_bid_tick_usd'])
            if 'heuristic_min_rebid_delta_usd' in additional_args:
                env["HEURISTIC_MIN_REBID_DELTA_USD"] = str(additional_args['heuristic_min_rebid_delta_usd'])
            if 'initial_service_execution_cost' in additional_args:
                env["INITIAL_SERVICE_EXECUTION_COST"] = str(additional_args['initial_service_execution_cost'])
            if additional_args.get('bid_freshness_min_seconds') is not None:
                env["BID_FRESHNESS_MIN_SECONDS"] = str(additional_args['bid_freshness_min_seconds'])
            if additional_args.get('provider_monitor_interval') is not None:
                env["PROVIDER_MONITOR_INTERVAL"] = str(additional_args['provider_monitor_interval'])
            if 'initial_quality_rating' in additional_args:
                env["INITIAL_QUALITY_RATING"] = str(additional_args['initial_quality_rating'])
            if 'initial_non_execution_cost_per_participation' in additional_args:
                env["INITIAL_NON_EXECUTION_COST_PER_PARTICIPATION"] = str(
                    additional_args['initial_non_execution_cost_per_participation']
                )
            if 'gas_cost_mode' in additional_args:
                env["GAS_COST_MODE"] = str(additional_args['gas_cost_mode'])
            if 'gas_price_gwei' in additional_args:
                env["GAS_PRICE_GWEI"] = str(additional_args['gas_price_gwei'])
            if 'eth_price_usd' in additional_args:
                env["ETH_PRICE_USD"] = str(additional_args['eth_price_usd'])
            if 'embedding_input_price_per_1k' in additional_args:
                env["EMBEDDING_INPUT_PRICE_PER_1K"] = str(additional_args['embedding_input_price_per_1k'])
            if 'initial_service_execution_cost_low' in additional_args:
                env["INITIAL_SERVICE_EXECUTION_COST_LOW"] = str(additional_args['initial_service_execution_cost_low'])
            if 'initial_service_execution_cost_medium' in additional_args:
                env["INITIAL_SERVICE_EXECUTION_COST_MEDIUM"] = str(additional_args['initial_service_execution_cost_medium'])
            if 'initial_service_execution_cost_high' in additional_args:
                env["INITIAL_SERVICE_EXECUTION_COST_HIGH"] = str(additional_args['initial_service_execution_cost_high'])
            if 'initial_quality_rating_low' in additional_args:
                env["INITIAL_QUALITY_RATING_LOW"] = str(additional_args['initial_quality_rating_low'])
            if 'initial_quality_rating_medium' in additional_args:
                env["INITIAL_QUALITY_RATING_MEDIUM"] = str(additional_args['initial_quality_rating_medium'])
            if 'initial_quality_rating_high' in additional_args:
                env["INITIAL_QUALITY_RATING_HIGH"] = str(additional_args['initial_quality_rating_high'])

        # Make the repository root importable for the agents.
        project_root = str(Path(__file__).parent.parent.parent)
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = project_root

        env.update({
            # Results pass from provider to consumer as `<jobs>/auction_<id>/result.json`,
            # keyed by auction id alone; a shared directory lets a run evaluate another
            # run's job.
            "AGENT_JOBS_DIR": str(self.jobs_dir),
            # Commitments are keyed by agent id, which repeats across runs; a shared
            # directory lets runs read each other's sealed bids and delete each other's
            # nonces.
            "AGENT_COMMITMENTS_DIR": str(self.commitments_dir),
            "BLOCKCHAIN_RPC_URL": self.rpc_url,
            # Agents sign with `config.chain_id`, which otherwise comes from `agents/.env`
            # (Sepolia, 11155111); without it every transaction is signed for the wrong chain.
            "BLOCKCHAIN_CHAIN_ID": str(self.chain_id),
            "BLOCKCHAIN_REVERSE_AUCTION_ADDRESS": self.reverse_auction_address,
            "BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['identity_registry'],
            "BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['reputation_registry'],
            "BLOCKCHAIN_INITIAL_REPUTATION_DEFAULT": str(self.config['blockchain'].get('initial_reputation_default', 50)),
            "BLOCKCHAIN_PAYMENT_TOKEN_ADDRESS": self.payment_token_address,
            "BLOCKCHAIN_PRIVATE_KEY": private_key,
            "BLOCKCHAIN_AGENT_ID": str(agent_id),
            # The consumer's sampling seed is run_seed + CONSUMER_AGENT_ID (see
            # Consumer.__init__). In all experiments this was 7430, read from the local
            # agents/.env; it is set here so that runs reproduce without that file.
            **({"CONSUMER_AGENT_ID": str(EXPERIMENT_CONSUMER_SEED_OFFSET)} if agent_type == "consumer" else {}),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            "PINATA_API_KEY": os.getenv("PINATA_API_KEY", ""),
            "PINATA_API_SECRET": os.getenv("PINATA_API_SECRET", ""),
        })

        log_file = self.log_dir / f"{agent_type}_{agent_id}.log"
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )

        handle = ProcessHandle(
            name=f"{agent_type}_{agent_id}",
            process=process,
            log_file=log_file,
            pid=process.pid
        )

        self.agent_processes.append(handle)
        self.processes.append(handle)

        logger.info(f"✓ {agent_type.capitalize()} agent spawned (PID: {handle.pid})")
        return handle

    async def spawn_agents(self):
        """Spawn the consumer and provider processes."""
        logger.info("Spawning agent processes...")

        consumer_config = self.config['agents']['consumer']
        consumer_pk = self.config['blockchain']['accounts']['consumer']

        consumer_args = {
            'eligible_providers': self.provider_agent_ids,
            'max_budget': consumer_config['config']['max_budget'],
            'commit_duration': consumer_config['config']['commit_duration'],
            'reveal_duration': consumer_config['config']['reveal_duration'],
            'reputation_weight': consumer_config['config'].get('reputation_weight', 30),
            'check_interval': consumer_config['config']['check_interval'],
            'check_interval_jitter': consumer_config['config'].get('check_interval_jitter', 0.0),
        }

        shared_sampled_dataset_args = {}
        if 'behavior' in consumer_config:
            behavior = consumer_config['behavior']
            consumer_args.update({
                'auto_create_auction': behavior.get('auto_create_auction', False),
                'num_auctions': behavior.get('num_auctions', 1),
                'pdf_directory': behavior.get('pdf_directory', ''),
                'service_complexity': behavior.get('service_complexity', 'medium'),
                'auction_creation_delay': behavior.get('auction_creation_delay', 0),
                'inter_auction_delay': behavior.get('inter_auction_delay', 30),
            })

            if 'service_selection' in behavior:
                consumer_args['service_selection'] = behavior['service_selection']
            for key in (
                'service_source_mode',
                'sampled_dataset_dir',
                'sampled_dataset_id',
                'sampled_dataset_run_selection_mode',
                'sampled_dataset_selection_seed',
                'sampled_dataset_pass_limit',
            ):
                if key in behavior:
                    consumer_args[key] = behavior[key]
                    shared_sampled_dataset_args[key] = behavior[key]

            # Pass pdfs_per_service only when the YAML sets it; otherwise the consumer uses
            # its own default.
            if 'pdfs_per_service' in behavior:
                consumer_args['pdfs_per_service'] = behavior['pdfs_per_service']
            if 'service_generation_mode' in behavior:
                consumer_args['service_generation_mode'] = behavior['service_generation_mode']
            if 'service_generation_iterations' in behavior:
                consumer_args['service_generation_iterations'] = behavior['service_generation_iterations']
            if 'service_generation_seed' in behavior:
                consumer_args['service_generation_seed'] = behavior['service_generation_seed']

            # Each auction draws eligible_per_auction providers from the registered ones.
            if 'eligible_per_auction' in behavior:
                consumer_args['provider_pool'] = self.provider_agent_ids
                consumer_args['eligible_per_auction'] = behavior['eligible_per_auction']
                logger.info(f"Random provider selection enabled: {behavior['eligible_per_auction']} from pool of {len(self.provider_agent_ids)}")
            elif 'provider_pool' in behavior:
                # Otherwise an explicit provider_pool from the config.
                consumer_args['provider_pool'] = behavior['provider_pool']
                if 'eligible_per_auction' in behavior:
                    consumer_args['eligible_per_auction'] = behavior['eligible_per_auction']

        self.spawn_agent_process(
            "consumer",
            self.consumer_agent_id,
            consumer_pk,
            consumer_args
        )

        for provider_meta in self.provider_metadata:
            agent_id = provider_meta['agent_id']
            pk = provider_meta['private_key']
            group_name = provider_meta.get('group_name', 'default')
            group_config = provider_meta['group_config']

            provider_args = {'group_name': group_name}
            if 'config' in group_config:
                if 'quality_profile' in group_config['config']:
                    provider_args['quality_profile'] = group_config['config']['quality_profile']

            provider_behavior = group_config.get('behavior', {})
            provider_cfg = group_config.get('config', {})

            # Strategy selection, from the YAML.
            architecture = provider_behavior.get('architecture', provider_cfg.get('architecture'))
            reasoning_mode = provider_behavior.get('reasoning_mode', provider_cfg.get('reasoning_mode'))
            heuristic_strategy = provider_behavior.get('heuristic_strategy', provider_cfg.get('heuristic_strategy'))
            cost_estimator = provider_behavior.get('cost_estimator', provider_cfg.get('cost_estimator'))
            if cost_estimator is not None:
                provider_args['cost_estimator'] = cost_estimator

            # The decline rule, per provider config rather than per group, so every
            # agent in a market runs the same rule.
            if provider_cfg.get('action_space') is not None:
                provider_args['action_space'] = provider_cfg['action_space']
            heuristic_min_margin = provider_behavior.get('heuristic_min_margin', provider_cfg.get('heuristic_min_margin'))
            heuristic_max_margin = provider_behavior.get('heuristic_max_margin', provider_cfg.get('heuristic_max_margin'))
            heuristic_recovery_pressure_max = provider_behavior.get(
                'heuristic_recovery_pressure_max', provider_cfg.get('heuristic_recovery_pressure_max')
            )
            heuristic_initial_margin_mode = provider_behavior.get(
                'heuristic_initial_margin_mode', provider_cfg.get('heuristic_initial_margin_mode')
            )
            heuristic_concession_exponent = provider_behavior.get(
                'heuristic_concession_exponent', provider_cfg.get('heuristic_concession_exponent')
            )
            heuristic_bid_tick_usd = provider_behavior.get('heuristic_bid_tick_usd', provider_cfg.get('heuristic_bid_tick_usd'))
            heuristic_min_rebid_delta_usd = provider_behavior.get(
                'heuristic_min_rebid_delta_usd', provider_cfg.get('heuristic_min_rebid_delta_usd')
            )
            llm_margin_bounds_enabled = provider_behavior.get(
                'llm_margin_bounds_enabled', provider_cfg.get('llm_margin_bounds_enabled')
            )
            llm_last_margin_enabled = provider_behavior.get(
                'llm_last_margin_enabled', provider_cfg.get('llm_last_margin_enabled')
            )
            llm_outcome_history_enabled = provider_behavior.get(
                'llm_outcome_history_enabled', provider_cfg.get('llm_outcome_history_enabled')
            )
            llm_outcome_history_limit = provider_behavior.get(
                'llm_outcome_history_limit', provider_cfg.get('llm_outcome_history_limit')
            )
            llm_reflection_enabled = provider_behavior.get(
                'llm_reflection_enabled', provider_cfg.get('llm_reflection_enabled')
            )
            llm_structured_bid_enabled = provider_behavior.get(
                'llm_structured_bid_enabled', provider_cfg.get('llm_structured_bid_enabled')
            )
            blockchain_llm_model = provider_behavior.get('blockchain_llm_model', provider_cfg.get('blockchain_llm_model'))
            service_llm_model = provider_behavior.get('service_llm_model', provider_cfg.get('service_llm_model'))

            if architecture is not None:
                provider_args['architecture'] = architecture
            if reasoning_mode is not None:
                provider_args['reasoning_mode'] = reasoning_mode
            if heuristic_strategy is not None:
                provider_args['heuristic_strategy'] = heuristic_strategy
            if heuristic_min_margin is not None:
                provider_args['heuristic_min_margin'] = heuristic_min_margin
            if heuristic_max_margin is not None:
                provider_args['heuristic_max_margin'] = heuristic_max_margin
            if heuristic_recovery_pressure_max is not None:
                provider_args['heuristic_recovery_pressure_max'] = heuristic_recovery_pressure_max
            if heuristic_initial_margin_mode is not None:
                provider_args['heuristic_initial_margin_mode'] = heuristic_initial_margin_mode
            if heuristic_concession_exponent is not None:
                provider_args['heuristic_concession_exponent'] = heuristic_concession_exponent
            if heuristic_bid_tick_usd is not None:
                provider_args['heuristic_bid_tick_usd'] = heuristic_bid_tick_usd
            if heuristic_min_rebid_delta_usd is not None:
                provider_args['heuristic_min_rebid_delta_usd'] = heuristic_min_rebid_delta_usd
            if llm_margin_bounds_enabled is not None:
                provider_args['llm_margin_bounds_enabled'] = llm_margin_bounds_enabled
            if llm_last_margin_enabled is not None:
                provider_args['llm_last_margin_enabled'] = llm_last_margin_enabled
            if llm_outcome_history_enabled is not None:
                provider_args['llm_outcome_history_enabled'] = llm_outcome_history_enabled
            if llm_outcome_history_limit is not None:
                provider_args['llm_outcome_history_limit'] = llm_outcome_history_limit
            if llm_reflection_enabled is not None:
                provider_args['llm_reflection_enabled'] = llm_reflection_enabled
            if llm_structured_bid_enabled is not None:
                provider_args['llm_structured_bid_enabled'] = llm_structured_bid_enabled
            if blockchain_llm_model is not None:
                provider_args['blockchain_llm_model'] = blockchain_llm_model
            if service_llm_model is not None:
                provider_args['service_llm_model'] = service_llm_model

            embedding_input_price_per_1k = provider_cfg.get('embedding_input_price_per_1k')
            if embedding_input_price_per_1k is not None:
                provider_args['embedding_input_price_per_1k'] = embedding_input_price_per_1k

            # The commit-window freshness gate, per run.
            bid_freshness_min_seconds = provider_cfg.get('bid_freshness_min_seconds')
            if bid_freshness_min_seconds is not None:
                provider_args['bid_freshness_min_seconds'] = bid_freshness_min_seconds

            provider_monitor_interval = provider_cfg.get('provider_monitor_interval')
            if provider_monitor_interval is not None:
                provider_args['provider_monitor_interval'] = provider_monitor_interval

            initial_service_execution_cost = provider_cfg.get('initial_service_execution_cost')
            initial_service_execution_cost_low = provider_cfg.get('initial_service_execution_cost_low')
            initial_service_execution_cost_medium = provider_cfg.get('initial_service_execution_cost_medium')
            initial_service_execution_cost_high = provider_cfg.get('initial_service_execution_cost_high')
            initial_quality_rating = provider_cfg.get('initial_quality_rating')
            initial_quality_rating_low = provider_cfg.get('initial_quality_rating_low')
            initial_quality_rating_medium = provider_cfg.get('initial_quality_rating_medium')
            initial_quality_rating_high = provider_cfg.get('initial_quality_rating_high')
            initial_non_execution_cost_per_participation = provider_cfg.get('initial_non_execution_cost_per_participation')
            gas_cost_mode = provider_cfg.get('gas_cost_mode')
            gas_price_gwei = provider_cfg.get('gas_price_gwei')
            eth_price_usd = provider_cfg.get('eth_price_usd')

            if initial_service_execution_cost is not None:
                provider_args['initial_service_execution_cost'] = initial_service_execution_cost
            if initial_service_execution_cost_low is not None:
                provider_args['initial_service_execution_cost_low'] = initial_service_execution_cost_low
            if initial_service_execution_cost_medium is not None:
                provider_args['initial_service_execution_cost_medium'] = initial_service_execution_cost_medium
            if initial_service_execution_cost_high is not None:
                provider_args['initial_service_execution_cost_high'] = initial_service_execution_cost_high
            if initial_quality_rating is not None:
                provider_args['initial_quality_rating'] = initial_quality_rating
            if initial_quality_rating_low is not None:
                provider_args['initial_quality_rating_low'] = initial_quality_rating_low
            if initial_quality_rating_medium is not None:
                provider_args['initial_quality_rating_medium'] = initial_quality_rating_medium
            if initial_quality_rating_high is not None:
                provider_args['initial_quality_rating_high'] = initial_quality_rating_high
            if initial_non_execution_cost_per_participation is not None:
                provider_args['initial_non_execution_cost_per_participation'] = initial_non_execution_cost_per_participation
            if gas_cost_mode is not None:
                provider_args['gas_cost_mode'] = gas_cost_mode
            if gas_price_gwei is not None:
                provider_args['gas_price_gwei'] = gas_price_gwei
            if eth_price_usd is not None:
                provider_args['eth_price_usd'] = eth_price_usd

            provider_args.update(shared_sampled_dataset_args)

            logger.info(
                f"Provider {agent_id} runtime config: "
                f"group={group_name}, "
                f"architecture={provider_args.get('architecture', 'default')}, "
                f"reasoning_mode={provider_args.get('reasoning_mode', 'default')}, "
                f"heuristic_strategy={provider_args.get('heuristic_strategy', 'default')}, "
                f"blockchain_llm_model={provider_args.get('blockchain_llm_model', 'default')}, "
                f"service_llm_model={provider_args.get('service_llm_model', 'default')}"
            )

            self.spawn_agent_process(
                "provider_agent",
                agent_id,
                pk,
                provider_args
            )

        logger.info("Waiting for agents to initialize...")
        await asyncio.sleep(3)

        logger.info("✓ All agents spawned")

    async def snapshot_reputations(self, after_auction_id: int):
        """Record every provider's reputation after auction ``after_auction_id`` completes."""
        try:
            from web3 import Web3
            from agents.common.contract_abis import get_reputation_registry_abi
            from agents.common.reputation_lookup import fetch_reputation_summary, reputation_to_float

            w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            reputation_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.config['blockchain']['contracts']['reputation_registry']),
                abi=get_reputation_registry_abi()
            )

            snapshot = {
                "after_auction": after_auction_id,
                "timestamp": datetime.now().isoformat(),
                "reputations": {}
            }
            initial_reputation_default = int(self.config['blockchain'].get('initial_reputation_default', 50))

            for provider_id in self.provider_agent_ids:
                try:
                    feedback_count, rep_value, rep_decimals = fetch_reputation_summary(reputation_contract, provider_id)

                    # If no feedback yet, preserve configured initial reputation.
                    average_score = (
                        reputation_to_float(rep_value, rep_decimals)
                        if feedback_count > 0
                        else float(initial_reputation_default)
                    )

                    snapshot["reputations"][provider_id] = {
                        "score": average_score,
                        "feedback_count": int(feedback_count)
                    }
                    logger.info(f"   Provider {provider_id}: score={average_score}, feedback_count={feedback_count}")
                except Exception as e:
                    logger.warning(f"   Failed to get reputation for provider {provider_id}: {e}")

            if self.metrics_collector:
                self.metrics_collector.metrics["reputation_evolution"].append(snapshot)

        except Exception as e:
            logger.error(f"Failed to snapshot reputations: {e}", exc_info=True)

    async def monitor_experiment(self):
        """Monitor the run until its stopping criteria are met."""
        logger.info("Monitoring experiment progress...")

        flow = self.config['experiment_flow']
        stopping_criteria = flow['phases'][2]['stopping_criteria']

        criteria_type = stopping_criteria['type']
        # The consumer's num_auctions is the target; stopping_criteria['target'] is the
        # fallback.
        target = self.config['agents']['consumer']['behavior'].get('num_auctions', stopping_criteria.get('target', 1))
        poll_interval = stopping_criteria['poll_interval']

        max_timeout = self.config['experiment'].get('duration_timeout', stopping_criteria.get('max_timeout', 7200))

        logger.info(f"Stopping criteria: {criteria_type}, target={target} auctions, max_timeout={max_timeout}s ({max_timeout/3600:.1f}h)")

        reputation_snapshots_enabled = self.config['experiment_flow'].get('reputation_snapshots', {}).get('enabled', True)
        last_completed_count = 0

        start_time = time.time()

        consumer_process = next(
            (p for p in self.agent_processes if p.name.startswith("consumer_")),
            None
        )

        while True:
            # Stop at once if the consumer exits.
            if consumer_process and consumer_process.process.poll() is not None:
                exit_code = consumer_process.process.returncode
                logger.error(
                    f"Consumer process exited unexpectedly with code {exit_code}. "
                    f"See log: {consumer_process.log_file}"
                )
                self.monitor_exit_reason = f"consumer_exited(code={exit_code})"
                break

            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                logger.info(f"Maximum timeout reached ({max_timeout}s = {max_timeout/3600:.1f}h)")
                self.monitor_exit_reason = f"max_timeout({max_timeout}s)"
                break

            # Persist the agents' runtime status snapshots.
            self._collect_runtime_status_snapshots()

            if criteria_type == "completed_auctions":
                try:
                    if self.consumer_status_file and self.consumer_status_file.exists():
                        with open(self.consumer_status_file, 'r') as f:
                            status = json.load(f)

                        if self.metrics_collector:
                            self.metrics_collector.record_consumer_status_snapshot(
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                status=status,
                                source="runtime_status",
                            )

                        completed = status.get('completed_auctions', 0)
                        logger.info(f"Completed auctions: {completed}/{target} (elapsed: {int(elapsed)}s)")

                        # Snapshot reputations when a new auction completes.
                        if reputation_snapshots_enabled and completed > last_completed_count:
                            logger.info(f"📸 Taking reputation snapshot after auction {completed}...")
                            await self.snapshot_reputations(completed)
                            last_completed_count = completed

                        if completed >= target:
                            logger.info(f"✓ Target reached: {completed} auctions completed")
                            self.monitor_exit_reason = f"target_reached({completed}/{target})"
                            break
                except Exception as e:
                    logger.debug(f"Error reading consumer status: {e}")

            await asyncio.sleep(poll_interval)

        logger.info("Monitoring loop ended: %s", getattr(self, "monitor_exit_reason", "unknown"))
        await self._drain_inflight_auctions()

        # One final snapshot at loop exit.
        self._collect_runtime_status_snapshots()

        logger.info("✓ Experiment monitoring complete")

    async def _drain_inflight_auctions(self, max_wait: float = 120.0) -> None:
        """Let auctions still in flight settle before the chain is torn down.

        Without it, `cleanup()` killed Anvil while the last auction was still in Commit or
        Reveal, so that auction was never evaluated or paid. The wait is bounded and
        returns at once when nothing is in flight.
        """
        deadline = time.time() + max_wait
        last_active = None
        while time.time() < deadline:
            if not (self.consumer_status_file and self.consumer_status_file.exists()):
                return
            try:
                with open(self.consumer_status_file) as fh:
                    status = json.load(fh)
            except (OSError, json.JSONDecodeError):
                await asyncio.sleep(1.0)
                continue
            active = int(status.get("active_auctions", 0) or 0)
            if active == 0:
                if last_active:
                    logger.info("✓ Drained %d in-flight auction(s) before teardown", last_active)
                return
            if active != last_active:
                logger.info("⏳ Waiting for %d in-flight auction(s) to settle...", active)
                last_active = active
            await asyncio.sleep(2.0)
        logger.warning(
            "Drain window of %.0fs expired with %s auction(s) still in flight; "
            "their outcome is not recorded", max_wait, last_active,
        )

    def _collect_runtime_status_snapshots(self):
        """Collect structured runtime snapshots from consumer/provider status files."""
        if not self.metrics_collector:
            return

        snapshot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        provider_profiles = {
            int(profile['provider_id']): profile
            for profile in self._provider_profiles_for_metrics()
        }
        for provider_id, status_file in self.provider_status_files.items():
            try:
                if not status_file.exists():
                    continue

                with open(status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)

                self.metrics_collector.record_provider_status_snapshot(
                    provider_id=provider_id,
                    timestamp=snapshot_ts,
                    status=status,
                    provider_profile=provider_profiles.get(int(provider_id)),
                    source="runtime_status",
                )

                financials = status.get("financials", {})
                if not isinstance(financials, dict):
                    continue

                self.metrics_collector.record_provider_financial_snapshot(
                    provider_id=provider_id,
                    timestamp=snapshot_ts,
                    revenue=float(financials.get("revenue", 0.0)),
                    llm_costs=float(financials.get("llm_costs", 0.0)),
                    # None when the provider runtime did not report the split.
                    reasoning_llm_costs=financials.get("reasoning_llm_costs"),
                    service_llm_costs=financials.get("service_llm_costs"),
                    cost_breakdown={
                        k: financials[k]
                        for k in (
                            "reasoning_llm_calls", "service_llm_calls",
                            "reasoning_input_tokens", "reasoning_output_tokens",
                            "service_input_tokens", "service_output_tokens",
                        )
                        if financials.get(k) is not None
                    },
                    gas_costs=float(financials.get("gas_costs", 0.0)),
                    total_costs=float(financials.get("total_costs", 0.0)),
                    net_balance=float(financials.get("net_balance", 0.0)),
                    provider_profile=provider_profiles.get(int(provider_id)),
                    source="runtime_status",
                )
            except Exception as e:
                logger.debug(f"Failed to collect runtime financial snapshot for provider {provider_id}: {e}")

        if self.consumer_status_file and self.consumer_status_file.exists():
            try:
                with open(self.consumer_status_file, 'r', encoding='utf-8') as f:
                    consumer_status = json.load(f)
                self.metrics_collector.record_consumer_status_snapshot(
                    timestamp=snapshot_ts,
                    status=consumer_status,
                    source="runtime_status",
                )
            except Exception as e:
                logger.debug(f"Failed to collect runtime consumer status snapshot: {e}")

    def collect_metrics(self):
        """Collect the run's metrics from the chain and the agents' status files, and save them."""
        logger.info("="*80)
        logger.info("Collecting comprehensive experiment metrics...")
        logger.info("="*80)

        if not self.metrics_collector:
            logger.error("MetricsCollector not initialized!")
            return

        self.metrics_collector.set_end_time(datetime.now())
        self.metrics_collector.set_contracts(
            reverse_auction=self.reverse_auction_address,
            payment_token=self.payment_token_address,
            identity_registry=self.config['blockchain']['contracts']['identity_registry'],
            reputation_registry=self.config['blockchain']['contracts']['reputation_registry']
        )
        self.metrics_collector.set_agents(
            consumer_id=self.consumer_agent_id,
            provider_ids=self.provider_agent_ids,
            provider_profiles=self._provider_profiles_for_metrics(),
        )

        # Capture a final runtime snapshot right before collection.
        self._collect_runtime_status_snapshots()

        # Auctions, from the chain.
        logger.info("📊 Collecting auction data from blockchain...")
        try:
            from web3 import Web3
            from agents.common.contract_abis import get_reverse_auction_abi, get_reputation_registry_abi

            w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            auction_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.reverse_auction_address),
                abi=get_reverse_auction_abi()
            )

            auction_count = auction_contract.functions.auctionIdCounter().call()
            logger.info(f"   Found {auction_count} auction(s) on-chain")

            summary = self.metrics_collector.metrics["summary"]

            reputation_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.config['blockchain']['contracts']['reputation_registry']),
                abi=get_reputation_registry_abi()
            )

            for auction_id in range(1, auction_count + 1):  # auction ids start at 1
                logger.info(f"   Collecting auction {auction_id}...")
                auction_data = self.metrics_collector.collect_auction_from_blockchain(
                    w3, auction_contract, auction_id, reputation_contract
                )
                self.metrics_collector.metrics["auctions"].append(auction_data)

                summary["total_auctions"] += 1
                if auction_data["status"] == "completed":
                    summary["completed_auctions"] += 1
                elif auction_data["status"] == "failed":
                    summary["failed_auctions"] += 1
                summary["total_bids_attempted"] += len(auction_data["bids_attempted"])
                summary["total_bids_on_chain"] += len(auction_data["bids_on_chain"])

                logger.info(f"      Status: {auction_data['status']}")
                logger.info(f"      Bids attempted: {len(auction_data['bids_attempted'])}")
                logger.info(f"      Bids on-chain: {len(auction_data['bids_on_chain'])}")
                logger.info(f"      Winner: {auction_data['winner_id']}")
                logger.info(f"      Quality rating: {auction_data['quality_rating']}/100")

            logger.info("\n📈 Summary:")
            logger.info(f"   Completed auctions: {summary['completed_auctions']}/{summary['total_auctions']}")
            logger.info(f"   Total bids attempted: {summary['total_bids_attempted']}")
            logger.info(f"   Total bids on-chain: {summary['total_bids_on_chain']}")
            if summary['total_bids_attempted'] > 0:
                success_rate = summary['total_bids_on_chain'] / summary['total_bids_attempted']
                logger.info(f"   Bid success rate: {success_rate:.1%}")

        except Exception as e:
            logger.error(f"   Failed to collect auction data: {e}", exc_info=True)

        # Reputation evolution.
        logger.info("\n🏆 Collecting reputation data...")
        try:
            from agents.common.contract_abis import get_reputation_registry_abi
            from agents.common.reputation_lookup import fetch_reputation_summary, reputation_to_float

            reputation_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.config['blockchain']['contracts']['reputation_registry']),
                abi=get_reputation_registry_abi()
            )

            final_snapshot = {
                "after_auction": "final",
                "timestamp": datetime.now().isoformat(),
                "source": "final_snapshot",
                "reputations": {}
            }

            for provider_id in self.provider_agent_ids:
                try:
                    feedback_count, rep_value, rep_decimals = fetch_reputation_summary(reputation_contract, provider_id)

                    initial_reputation_default = int(self.config['blockchain'].get('initial_reputation_default', 50))
                    reputation = (
                        reputation_to_float(rep_value, rep_decimals)
                        if feedback_count > 0
                        else float(initial_reputation_default)
                    )
                    final_snapshot["reputations"][str(provider_id)] = {
                        "score": reputation,
                        "feedback_count": int(feedback_count)
                    }
                    logger.info(f"   Provider {provider_id}: {reputation} (feedback_count={int(feedback_count)})")
                except Exception as e:
                    logger.debug(f"   Could not get reputation for provider {provider_id}: {e}")

            if final_snapshot["reputations"]:
                self.metrics_collector.metrics["reputation_evolution"].append(final_snapshot)

        except Exception as e:
            logger.error(f"   Failed to collect reputation: {e}")

        # Timing.
        logger.info("\n⏱️  Collecting timing metrics...")
        timing = self.metrics_collector.metrics["timing"]
        timing["phase_durations"] = self.phase_timings

        total_time = sum(self.phase_timings.values())
        logger.info(f"   Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")
        for phase, duration in self.phase_timings.items():
            logger.info(f"   {phase}: {duration:.1f}s")

        auctions = self.metrics_collector.metrics["auctions"]
        if auctions:
            exec_times = [a["execution_duration_seconds"] for a in auctions if a["execution_duration_seconds"] > 0]
            eval_times = [a["evaluation_duration_seconds"] for a in auctions if a["evaluation_duration_seconds"] > 0]

            timing["execution_times"] = exec_times
            timing["evaluation_times"] = eval_times

        # Errors.
        logger.info("\n⚠️  Collecting errors...")
        errors = self.metrics_collector.collect_errors()
        logger.info(f"   Total errors: {errors.get('total_errors')}")
        logger.info(f"   Total warnings: {errors.get('total_warnings')}")

        if errors.get('critical_issues'):
            logger.warning(f"   ⚠️  {len(errors['critical_issues'])} critical issue(s) detected")

        # Determine overall success from runtime-complete criteria.
        completeness = self.metrics_collector.collect_system_completeness()
        success = self.metrics_collector.determine_success()
        logger.info(f"   Runtime completeness: {completeness}")
        logger.info(f"\n{'✅' if success else '❌'} Experiment success: {success}")

        metrics_file = self.metrics_collector.save_metrics()
        logger.info("="*80)
        logger.info(f"📁 Comprehensive metrics saved to: {metrics_file}")
        logger.info("="*80)

    def shutdown_agents(self):
        """SIGTERM every agent and wait, so each writes its final status file.

        Called before `collect_metrics`. Agents write their status every 10 s and once more
        on shutdown, after every job has settled and every cost is booked; reading an
        earlier write paired complete chain revenue with incomplete costs, so `net_balance`
        in `experiment_metrics.json` came out too high. (The analysis reads the final
        status files and was not affected.) Idempotent: `cleanup()` calls it again.
        """
        if getattr(self, "_agents_shut_down", False):
            return
        self._agents_shut_down = True
        for handle in self.agent_processes:
            try:
                logger.info(f"Stopping {handle.name} (PID: {handle.pid})...")
                os.killpg(os.getpgid(handle.pid), signal.SIGTERM)
                # Long enough for the `finally` block to write the final status.
                handle.process.wait(timeout=15)
            except Exception as e:
                logger.warning(f"Error stopping {handle.name}: {e}")
                try:
                    os.killpg(os.getpgid(handle.pid), signal.SIGKILL)
                except:
                    pass

    def cleanup(self):
        """Shut down every process."""
        logger.info("Shutting down processes...")

        # Agents first. Normally `shutdown_agents()` has already stopped them; this
        # covers the failure paths.
        self.shutdown_agents()

        if self.anvil_process:
            try:
                logger.info(f"Stopping Anvil (PID: {self.anvil_process.pid})...")
                os.killpg(os.getpgid(self.anvil_process.pid), signal.SIGTERM)
                self.anvil_process.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping Anvil: {e}")
                try:
                    os.killpg(os.getpgid(self.anvil_process.pid), signal.SIGKILL)
                except:
                    pass

        logger.info("✓ All processes stopped")

    async def run(self):
        """Run the complete experiment."""
        self.start_time = datetime.now()

        try:
            # Load configuration (initializes metrics_collector)
            self.load_config()

            # Refuse to start if this run's port is taken.
            self.cleanup_orphaned_processes()

            if self.metrics_collector:
                self.metrics_collector.set_start_time(self.start_time)

            # Phase 1: Setup blockchain
            logger.info("=== Phase 1: Setup Blockchain ===")
            phase_start = datetime.now()
            await self.start_anvil()
            await self.deploy_contracts()
            await self.fund_consumer_account()
            await self.register_agents()
            self.phase_timings["setup_blockchain"] = (datetime.now() - phase_start).total_seconds()
            logger.info(f"✓ Phase 1 completed in {self.phase_timings['setup_blockchain']:.1f}s")

            # Phase 2: Start agents
            logger.info("=== Phase 2: Start Agents ===")
            phase_start = datetime.now()
            await self.spawn_agents()
            self.phase_timings["start_agents"] = (datetime.now() - phase_start).total_seconds()
            logger.info(f"✓ Phase 2 completed in {self.phase_timings['start_agents']:.1f}s")

            # Phase 3: Run experiment
            logger.info("=== Phase 3: Run Experiment ===")
            phase_start = datetime.now()
            await self.monitor_experiment()
            self.phase_timings["experiment_run"] = (datetime.now() - phase_start).total_seconds()
            logger.info(f"✓ Phase 3 completed in {self.phase_timings['experiment_run']:.1f}s")

            # Stop the agents before collecting metrics, so each has written its final
            # status file (see `shutdown_agents`).
            self.shutdown_agents()

            self.collect_metrics()

            logger.info("=== Experiment Complete ===")

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment simulation")
    parser.add_argument("config", type=Path, help="Path to experiment YAML configuration")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--run-seed", type=int, default=None,
        help="Seed for this run, overriding the config. One config serves "
             "every seed of its cell; paired arms at the same seed see the same market.",
    )
    parser.add_argument(
        "--run-index", type=int, default=0,
        help="Isolation index for concurrent runs: port 8545+i, chain id 8000+i. "
             "Index 0 keeps the historical port and the config's rpc_url.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    runner = ExperimentRunner(args.config, run_index=args.run_index, run_seed=args.run_seed)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
