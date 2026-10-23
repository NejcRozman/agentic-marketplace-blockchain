"""Agent configuration, read from environment variables (and ``agents/.env``).

The experiment runner sets the variables per agent from the condition config;
``agents/.env`` only supplies defaults, such as API keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

AGENTS_DIR = Path(__file__).parent
ENV_FILE_PATH = AGENTS_DIR / ".env"

# Values already in the environment (set by the experiment runner) take precedence.
load_dotenv(ENV_FILE_PATH, override=False)

#: The consumer side's model, for request generation and evaluation: the same
#: model the architectures give the executor.
CONSUMER_LLM_DEFAULT = "openai/gpt-5.4-mini"


class Config:
    """Settings of one agent process."""

    def __init__(self):
        # OpenRouter
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Pinata (IPFS)
        self.pinata_api_key = os.getenv("PINATA_API_KEY")
        self.pinata_api_secret = os.getenv("PINATA_API_SECRET")
        self.pinata_jwt = os.getenv("PINATA_JWT")

        # Chain
        self.rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
        self.chain_id = int(os.getenv("BLOCKCHAIN_CHAIN_ID", "31337"))
        self.private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
        self.gas_limit = int(os.getenv("BLOCKCHAIN_GAS_LIMIT", "3000000"))

        # Contract addresses, set by the runner after deployment
        self.reverse_auction_address = os.getenv("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS")
        self.identity_registry_address = os.getenv("BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS")
        self.reputation_registry_address = os.getenv("BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS")
        self.payment_token_address = os.getenv("BLOCKCHAIN_PAYMENT_TOKEN_ADDRESS")  # mock USDC
        self.initial_reputation_default = int(os.getenv("BLOCKCHAIN_INITIAL_REPUTATION_DEFAULT", "50"))

        self.agent_id = int(os.getenv("BLOCKCHAIN_AGENT_ID", "4427"))

        # The decline rule.
        #   price_only       - bid on every eligible auction, never decline.
        #   price_or_decline - decline when believed cost + overhead exceeds the
        #                      highest price this agent has itself won at.
        self.action_space = os.getenv("ACTION_SPACE", "price_only").strip().lower() or "price_only"
        if self.action_space not in ("price_only", "price_or_decline"):
            raise ValueError(
                f"ACTION_SPACE={self.action_space!r}; expected 'price_only' or "
                f"'price_or_decline'"
            )

        # Where the provider's cost estimate comes from (see provider/cost_estimators.py):
        #   none (Adaptive's running average), r_guarded (Regression),
        #   decomposed (LLM-Estimate), oracle.
        self.cost_estimator = os.getenv("COST_ESTIMATOR", "none").strip().lower() or "none"

        # How often the provider loop polls the chain. The default 5 s cannot
        # reliably catch a short reveal window.
        monitor_interval_raw = os.getenv("PROVIDER_MONITOR_INTERVAL", "").strip()
        if monitor_interval_raw:
            self.provider_monitor_interval = float(monitor_interval_raw)

        # Minimum seconds left in the commit phase for a bid to be worth sending.
        # Unset keeps the default of 10 s.
        bid_freshness_raw = os.getenv("BID_FRESHNESS_MIN_SECONDS", "").strip()
        if bid_freshness_raw:
            self.bid_freshness_min_seconds = float(bid_freshness_raw)

        # One seed per run, from which every agent derives its own stream as
        # `run_seed + agent_id`. Unset means unseeded (not reproducible).
        run_seed_raw = os.getenv("RUN_SEED", "").strip()
        self.run_seed = int(run_seed_raw) if run_seed_raw else None

        # Consumer
        self.consumer_agent_id = int(os.getenv("CONSUMER_AGENT_ID", "0"))  # 0 if not a consumer
        eligible_str = os.getenv("ELIGIBLE_PROVIDERS", "")
        self.eligible_providers = [int(id.strip()) for id in eligible_str.split(",") if id.strip()]
        self.consumer_check_interval = float(os.getenv("CONSUMER_CHECK_INTERVAL", "5"))
        self.consumer_check_interval_jitter = float(os.getenv("CONSUMER_CHECK_INTERVAL_JITTER", "0.0"))

        # Optional random selection of the eligible providers
        pool_str = os.getenv("PROVIDER_POOL", "")
        self.provider_pool = [int(id.strip()) for id in pool_str.split(",") if id.strip()]
        eligible_per_auction_str = os.getenv("ELIGIBLE_PER_AUCTION", "")
        self.eligible_per_auction = int(eligible_per_auction_str) if eligible_per_auction_str else None

        # Automatic auction creation
        self.auto_create_auction = os.getenv("AUTO_CREATE_AUCTION", "false").lower() == "true"
        self.num_auctions = int(os.getenv("NUM_AUCTIONS", "1"))
        self.auction_creation_delay = int(os.getenv("AUCTION_CREATION_DELAY", "0"))  # seconds before the first auction
        self.inter_auction_delay = int(os.getenv("INTER_AUCTION_DELAY", "30"))  # seconds between auctions
        self.pdf_directory = os.getenv("PDF_DIRECTORY", "")

        # Where a provider writes a finished job and the consumer reads it back. Must be
        # per run: the path is keyed by auction id alone, so runs sharing it would
        # evaluate each other's results.
        self.jobs_directory = os.getenv(
            "AGENT_JOBS_DIR",
            str(Path(__file__).resolve().parent / "data" / "jobs"),
        )

        # Where a provider persists the nonces of its sealed bids. Must be per run: the
        # file is keyed by agent id, agent ids repeat across runs, and the runner
        # deletes it at startup, which would wipe the bids of parallel runs.
        self.commitments_directory = os.getenv(
            "AGENT_COMMITMENTS_DIR",
            str(Path(__file__).resolve().parent / "data" / "commitments"),
        )
        self.pdfs_per_service = max(1, int(os.getenv("PDFS_PER_SERVICE", "5")))
        self.service_selection = os.getenv("SERVICE_SELECTION", "sequential").strip().lower() or "sequential"
        self.service_generation_mode = os.getenv("SERVICE_GENERATION_MODE", "sequential_groups")
        self.service_generation_iterations = max(
            1,
            int(os.getenv("SERVICE_GENERATION_ITERATIONS", "1")),
        )
        service_generation_seed_raw = os.getenv("SERVICE_GENERATION_SEED", "").strip()
        self.service_generation_seed = (
            int(service_generation_seed_raw)
            if service_generation_seed_raw
            else None
        )
        self.service_source_mode = os.getenv("SERVICE_SOURCE_MODE", "live_generation").strip().lower() or "live_generation"
        self.sampled_dataset_dir = os.getenv("SAMPLED_DATASET_DIR", "")
        sampled_dataset_id_raw = os.getenv("SAMPLED_DATASET_ID", "").strip()
        self.sampled_dataset_id = sampled_dataset_id_raw or None
        self.sampled_dataset_run_selection_mode = (
            os.getenv("SAMPLED_DATASET_RUN_SELECTION_MODE", "round_robin").strip().lower()
            or "round_robin"
        )
        # How many requests one run consumes.
        sampled_dataset_pass_limit_raw = os.getenv("SAMPLED_DATASET_PASS_LIMIT", "").strip()
        self.sampled_dataset_pass_limit = (
            int(sampled_dataset_pass_limit_raw) if sampled_dataset_pass_limit_raw else None
        )
        sampled_dataset_selection_seed_raw = os.getenv("SAMPLED_DATASET_SELECTION_SEED", "").strip()
        self.sampled_dataset_selection_seed = (
            int(sampled_dataset_selection_seed_raw)
            if sampled_dataset_selection_seed_raw
            else None
        )
        self.service_generation_timeout = int(os.getenv("SERVICE_GENERATION_TIMEOUT", "180"))
        # Consumer-side models: request generation and evaluation both run the
        # executor's model by default. The dataset collection records them in every
        # collection manifest.
        self.service_generation_model = (
            os.getenv("SERVICE_GENERATION_MODEL", "").strip() or CONSUMER_LLM_DEFAULT
        )
        self.openrouter_model = (
            os.getenv("EVALUATOR_MODEL", "").strip()
            or os.getenv("OPENROUTER_MODEL", "").strip()
            or CONSUMER_LLM_DEFAULT
        )
        self.max_budget = int(os.getenv("MAX_BUDGET", "10000000"))  # micro-USDC (10 USDC)
        # Sealed-bid phases: providers commit a hash during commit_duration, then
        # disclose (bid, nonce) during reveal_duration.
        self.commit_duration = int(os.getenv("COMMIT_DURATION", "1800"))  # seconds
        self.reveal_duration = int(os.getenv("REVEAL_DURATION", "300"))  # seconds
        self.reputation_weight = int(os.getenv("REPUTATION_WEIGHT", "30"))  # 0-100: weight of reputation in the score

        self.workspace_dir = os.getenv("WORKSPACE_DIR", "./workspaces")

        # RAG parameters
        self.rag_temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))
        self.rag_retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "3"))
        self.rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        self.rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.service_recursion_limit = int(os.getenv("SERVICE_RECURSION_LIMIT", "25"))
        self.react_recursion_limit = int(os.getenv("REACT_RECURSION_LIMIT", "20"))

        # Provider bidding
        self.initial_service_execution_cost = float(os.getenv("INITIAL_SERVICE_EXECUTION_COST", "5"))
        self.initial_service_execution_cost_by_effort = {
            "low": float(
                os.getenv(
                    "INITIAL_SERVICE_EXECUTION_COST_LOW",
                    str(self.initial_service_execution_cost),
                )
            ),
            "medium": float(
                os.getenv(
                    "INITIAL_SERVICE_EXECUTION_COST_MEDIUM",
                    str(self.initial_service_execution_cost),
                )
            ),
            "high": float(
                os.getenv(
                    "INITIAL_SERVICE_EXECUTION_COST_HIGH",
                    str(self.initial_service_execution_cost),
                )
            ),
        }
        self.initial_quality_rating = float(
            os.getenv(
                "INITIAL_QUALITY_RATING",
                str(self.initial_reputation_default),
            )
        )
        self.initial_quality_rating_by_effort = {
            "low": float(
                os.getenv(
                    "INITIAL_QUALITY_RATING_LOW",
                    str(self.initial_quality_rating),
                )
            ),
            "medium": float(
                os.getenv(
                    "INITIAL_QUALITY_RATING_MEDIUM",
                    str(self.initial_quality_rating),
                )
            ),
            "high": float(
                os.getenv(
                    "INITIAL_QUALITY_RATING_HIGH",
                    str(self.initial_quality_rating),
                )
            ),
        }
        self.initial_non_execution_cost_per_participation = float(
            os.getenv("INITIAL_NON_EXECUTION_COST_PER_PARTICIPATION", "0")
        )
        # Heuristic bidding
        self.heuristic_strategy = os.getenv("HEURISTIC_STRATEGY", "cost_plus")
        self.heuristic_min_margin = float(os.getenv("HEURISTIC_MIN_MARGIN", "0.10"))
        self.heuristic_max_margin = float(os.getenv("HEURISTIC_MAX_MARGIN", "0.30"))
        self.heuristic_bid_tick_usd = float(os.getenv("HEURISTIC_BID_TICK_USD", "0.000001"))
        self.heuristic_recovery_pressure_max = float(os.getenv("HEURISTIC_RECOVERY_PRESSURE_MAX", "1.0"))
        # Adaptive's starting margin: "max" (default) starts at max_margin and concedes
        # down; "mid" starts at the midpoint; "random" draws it uniformly.
        self.heuristic_initial_margin_mode = os.getenv("HEURISTIC_INITIAL_MARGIN_MODE", "max").strip().lower()

        # LLM-Band and later variants: the prompt asks the LLM to keep its margin within
        # [heuristic_min_margin, heuristic_max_margin], the heuristics' band. An
        # instruction, not an enforced clamp.
        self.llm_margin_bounds_enabled = os.getenv("LLM_MARGIN_BOUNDS_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # Show the LLM the margin it committed on its most recent bid.
        self.llm_last_margin_enabled = os.getenv("LLM_LAST_MARGIN_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # The LLM returns schema-enforced JSON that the agent executes as a bid, instead
        # of a native tool call: with tool calling, gpt-oss-20b often reasoned out a bid
        # but never emitted the call.
        self.llm_structured_bid_enabled = os.getenv("LLM_STRUCTURED_BID_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # LLM-History and LLM-Reflect: show the LLM the full record of its own past
        # bids and their outcomes (margin, win or loss, reputation, balance change).
        self.llm_outcome_history_enabled = os.getenv("LLM_OUTCOME_HISTORY_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )
        # How many past auctions are shown, most recent first; 0 = all.
        self.llm_outcome_history_limit = int(os.getenv("LLM_OUTCOME_HISTORY_LIMIT", "0"))

        # LLM-Reflect: before each decision, a separate LLM call reviews what happened and
        # revises a persistent strategy memo (beliefs about the market, current strategy,
        # open questions), which the bidding prompt then shows. Its cost is charged to
        # the agent like any other reasoning cost.
        self.llm_reflection_enabled = os.getenv("LLM_REFLECTION_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # Per-run overrides of the architecture's default models; unset uses the default.
        self.blockchain_llm_override = os.getenv("BLOCKCHAIN_LLM_MODEL", "").strip() or None
        self.service_llm_override = os.getenv("SERVICE_LLM_MODEL", "").strip() or None

        # Sampling for live LLM calls. Both default to the block pinned for the
        # model in agents/common/llm_sampling.py (seed, and reasoning effort or
        # temperature); the overrides are for models without a pinned block.
        llm_temperature_raw = os.getenv("LLM_TEMPERATURE", "").strip()
        self.llm_temperature_override = float(llm_temperature_raw) if llm_temperature_raw else None
        llm_sampling_seed_raw = os.getenv("LLM_SAMPLING_SEED", "").strip()
        self.llm_sampling_seed = int(llm_sampling_seed_raw) if llm_sampling_seed_raw else None

        # Execution-cost estimate (EMA)
        self.execution_cost_ema_alpha = min(
            1.0,
            max(0.0, float(os.getenv("EXECUTION_COST_EMA_ALPHA", "0.25"))),
        )
        self.non_execution_cost_ema_alpha = min(
            1.0,
            max(0.0, float(os.getenv("NON_EXECUTION_COST_EMA_ALPHA", "0.20"))),
        )

        # Gas
        self.gas_cost_mode = os.getenv("GAS_COST_MODE", "receipt")  # "receipt" or "fixed"
        self.gas_price_gwei = float(os.getenv("GAS_PRICE_GWEI", "20.0"))
        self.eth_price_usd = float(os.getenv("ETH_PRICE_USD", "3000.0"))

        # Token prices per role
        self.reasoning_llm_input_price_per_1k = float(
            os.getenv("REASONING_LLM_INPUT_PRICE_PER_1K", "0.00075")
        )
        self.reasoning_llm_output_price_per_1k = float(
            os.getenv("REASONING_LLM_OUTPUT_PRICE_PER_1K", "0.0045")
        )
        self.service_llm_input_price_per_1k = float(
            os.getenv("SERVICE_LLM_INPUT_PRICE_PER_1K", "0.00075")
        )
        self.service_llm_output_price_per_1k = float(
            os.getenv("SERVICE_LLM_OUTPUT_PRICE_PER_1K", "0.0045")
        )
        self.embedding_input_price_per_1k = float(
            os.getenv("EMBEDDING_INPUT_PRICE_PER_1K", "0.00002")
        )
        # Consumer-side prices (request generation, evaluation); default to the
        # service-LLM prices, since the consumer runs the same model.
        self.consumer_llm_input_price_per_1k = float(
            os.getenv("CONSUMER_LLM_INPUT_PRICE_PER_1K", str(self.service_llm_input_price_per_1k))
        )
        self.consumer_llm_output_price_per_1k = float(
            os.getenv("CONSUMER_LLM_OUTPUT_PRICE_PER_1K", str(self.service_llm_output_price_per_1k))
        )

        self.coupling_mode = os.getenv("COUPLING_MODE", "isolated")  # "isolated", "one_way" or "two_way"

        self.architecture = os.getenv("ARCHITECTURE", "1")

        # Search budget per effort tier
        self.service_max_tool_calls_by_effort = {
            "low": int(os.getenv("SERVICE_MAX_TOOL_CALLS_LOW", "1")),
            "medium": int(os.getenv("SERVICE_MAX_TOOL_CALLS_MEDIUM", "3")),
            "high": int(os.getenv("SERVICE_MAX_TOOL_CALLS_HIGH", "10")),
        }
        self.service_max_retrieval_rounds_by_effort = {
            "low": int(os.getenv("SERVICE_MAX_RETRIEVAL_ROUNDS_LOW", "1")),
            "medium": int(os.getenv("SERVICE_MAX_RETRIEVAL_ROUNDS_MEDIUM", "3")),
            "high": int(os.getenv("SERVICE_MAX_RETRIEVAL_ROUNDS_HIGH", "10")),
        }
        # Executor model per effort tier. Empty means every tier uses the
        # architecture's service_llm.
        self.service_llm_by_effort = {
            "low": os.getenv("SERVICE_LLM_LOW", "").strip() or None,
            "medium": os.getenv("SERVICE_LLM_MEDIUM", "").strip() or None,
            "high": os.getenv("SERVICE_LLM_HIGH", "").strip() or None,
        }
        # Token prices per tier, so a cheaper model actually costs less in the record.
        # Falls back to the single service-LLM price when a tier has no override.
        self.service_llm_prices_by_effort = {
            "low": {}, "medium": {}, "high": {},
        }
        # Passages returned per search, per tier: the evidence a tier buys is
        # rounds x passages.
        self.service_retrieval_k_by_effort = {
            "low": int(os.getenv("SERVICE_RETRIEVAL_K_LOW", str(self.rag_retrieval_k))),
            "medium": int(os.getenv("SERVICE_RETRIEVAL_K_MEDIUM", str(self.rag_retrieval_k))),
            "high": int(os.getenv("SERVICE_RETRIEVAL_K_HIGH", str(self.rag_retrieval_k))),
        }
        self.default_effort_tier = "medium"

        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

    def get_initial_service_execution_cost_for_effort(self, effort_tier: str = None) -> float:
        """The execution-cost prior for an effort tier."""
        normalized = str(effort_tier or self.default_effort_tier).strip().lower()
        if normalized not in {"low", "medium", "high"}:
            normalized = self.default_effort_tier

        return float(
            self.initial_service_execution_cost_by_effort.get(
                normalized,
                self.initial_service_execution_cost,
            )
        )

    def get_initial_quality_rating_for_effort(self, effort_tier: str = None) -> float:
        """The quality-rating prior for an effort tier."""
        normalized = str(effort_tier or self.default_effort_tier).strip().lower()
        if normalized not in {"low", "medium", "high"}:
            normalized = self.default_effort_tier

        return max(
            0.0,
            min(
                100.0,
                float(
                    self.initial_quality_rating_by_effort.get(
                        normalized,
                        self.initial_quality_rating,
                    )
                ),
            ),
        )

    def get_service_execution_budget_for_effort(self, effort_tier: str = None) -> dict:
        """The tool-call and retrieval budget of an effort tier."""
        normalized = str(effort_tier or self.default_effort_tier).strip().lower()
        if normalized not in {"low", "medium", "high"}:
            normalized = self.default_effort_tier

        return {
            "effort_tier": normalized,
            "max_tool_calls": max(
                0,
                int(self.service_max_tool_calls_by_effort.get(normalized, 0) or 0),
            ),
            "max_retrieval_rounds": max(
                0,
                int(self.service_max_retrieval_rounds_by_effort.get(normalized, 0) or 0),
            ),
            "retrieval_k": max(
                1,
                int(
                    self.service_retrieval_k_by_effort.get(normalized, self.rag_retrieval_k)
                    or self.rag_retrieval_k
                ),
            ),
            "model": (self.service_llm_by_effort or {}).get(normalized) or None,
            "input_price_per_1k": float(
                (self.service_llm_prices_by_effort.get(normalized) or {}).get(
                    "input", self.service_llm_input_price_per_1k
                )
            ),
            "output_price_per_1k": float(
                (self.service_llm_prices_by_effort.get(normalized) or {}).get(
                    "output", self.service_llm_output_price_per_1k
                )
            ),
        }

    def validate(self) -> bool:
        """Whether the required settings are present; logs what is missing."""
        errors = []

        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required")

        if not self.private_key:
            errors.append("BLOCKCHAIN_PRIVATE_KEY is required for transactions")

        if not self.reverse_auction_address:
            errors.append("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS is required")

        if errors:
            print("⚠️  Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False

        return True

    def display(self):
        """Log the configuration, hiding secrets."""
        print("Configuration:")
        print(f"  Google API Key: {'✓ Set' if self.google_api_key else '✗ Not set'}")
        print(f"  Pinata JWT: {'✓ Set' if self.pinata_jwt else '✗ Not set'}")
        print(f"  Pinata API Key: {'✓ Set' if self.pinata_api_key else '✗ Not set'}")
        print(f"  RPC URL: {self.rpc_url}")
        print(f"  Chain ID: {self.chain_id}")
        print(f"  Private Key: {'✓ Set' if self.private_key else '✗ Not set'}")
        print(f"  Gas Limit: {self.gas_limit}")
        print(f"  ReverseAuction: {self.reverse_auction_address or '✗ Not set'}")
        print(f"  IdentityRegistry: {self.identity_registry_address or '✗ Not set'}")
        print(f"  ReputationRegistry: {self.reputation_registry_address or '✗ Not set'}")
        print(f"  Agent ID: {self.agent_id}")
        print(f"  Workspace: {self.workspace_dir}")
        print(f"  Environment: {self.environment}")
        print(f"  Debug: {self.debug}")


# --- Agent architectures ---
from dataclasses import dataclass
from typing import List


@dataclass
class ArchitectureConfig:
    """An agent architecture: its reasoning mode, state level and default models."""
    name: str
    description: str
    state_level: int
    reasoning_mode: str  # "heuristic" or "llm_react"
    coupling_mode: str  # "isolated", "one_way" or "two_way"
    prompt_template: str
    enabled_tools: List[str]
    blockchain_llm: str = "openai/gpt-4o-mini"
    service_llm: str = "openai/gpt-4o-mini"
    # Models with a pinned sampling block (agents/common/llm_sampling.py) ignore
    # this and use their block instead.
    llm_temperature: float = 0.0


ARCHITECTURES = {
    "1": ArchitectureConfig(
        name="Architecture 1",
        description="LLM Minimal - Current state only, reactive reasoning, isolated coupling",
        state_level=0,
        reasoning_mode="llm_react",
        coupling_mode="isolated",
        prompt_template="1",
        enabled_tools=["place_bid"],
        blockchain_llm="openai/gpt-oss-120b:free",
        service_llm="openai/gpt-5.4-mini",
        llm_temperature=0.0,
    ),
    "2": ArchitectureConfig(
        name="Architecture 2",
        description="LLM Strategic - Performance-aware state, reactive reasoning, one-way coupling",
        state_level=1,
        reasoning_mode="llm_react",
        coupling_mode="one_way",
        prompt_template="2",
        enabled_tools=["place_bid"],
        blockchain_llm="openai/gpt-4o-mini",
        service_llm="openai/gpt-5.4-mini",
        llm_temperature=0.0,
    ),
}


def get_architecture(name: str) -> ArchitectureConfig:
    if name not in ARCHITECTURES:
        available = ", ".join(ARCHITECTURES.keys())
        raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
    return ARCHITECTURES[name]


# Shared instance.
config = Config()
