"""Configuration management for agentic marketplace agents."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the path to the agents directory where .env file is located
AGENTS_DIR = Path(__file__).parent
ENV_FILE_PATH = AGENTS_DIR / ".env"

# Load environment variables from .env file (use as defaults, don't override existing)
# This allows experiment runner to override contract addresses for each run
load_dotenv(ENV_FILE_PATH, override=False)

#: The consumer side's model — requirement generation and evaluation (S4 §2.1).
#: The same model the architectures give the executor, so "consumer side" is one
#: deliberate choice rather than two accidental ones.
CONSUMER_LLM_DEFAULT = "openai/gpt-5.4-mini"


class Config:
    """Simple configuration class for marketplace agents."""
    
    def __init__(self):
        """Load configuration from environment variables."""
        
        # LLM API (OpenRouter)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Pinata IPFS
        self.pinata_api_key = os.getenv("PINATA_API_KEY")
        self.pinata_api_secret = os.getenv("PINATA_API_SECRET")
        self.pinata_jwt = os.getenv("PINATA_JWT")
        
        # Blockchain connection
        self.rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
        self.chain_id = int(os.getenv("BLOCKCHAIN_CHAIN_ID", "31337"))
        self.private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
        self.gas_limit = int(os.getenv("BLOCKCHAIN_GAS_LIMIT", "3000000"))
        
        # Contract addresses (set these after deployment)
        self.reverse_auction_address = os.getenv("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS")
        self.identity_registry_address = os.getenv("BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS")
        self.reputation_registry_address = os.getenv("BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS")
        self.payment_token_address = os.getenv("BLOCKCHAIN_PAYMENT_TOKEN_ADDRESS")  # USDC or mock USDC
        self.initial_reputation_default = int(os.getenv("BLOCKCHAIN_INITIAL_REPUTATION_DEFAULT", "50"))
        
        # Provider Agent ID
        self.agent_id = int(os.getenv("BLOCKCHAIN_AGENT_ID", "4427"))

        # S7 §C — the abstention gate, which is S6's rung O2c. Same name and same
        # meaning as the simulator's provider spec field.
        #   price_only       - the original behaviour: bid on every eligible
        #                      auction, never decline (rung O1).
        #   price_or_decline - abstain when believed cost + overhead exceeds the
        #                      highest price this agent has itself won at (O2c).
        self.action_space = os.getenv("ACTION_SPACE", "price_only").strip().lower() or "price_only"
        if self.action_space not in ("price_only", "price_or_decline"):
            raise ValueError(
                f"ACTION_SPACE={self.action_space!r}; expected 'price_only' or "
                f"'price_or_decline'"
            )

        # S7 §B.2: which cost estimator the provider bids with.
        #   none      - H, the pure heuristic (default; nothing changes)
        #   r_naive   - regression on own history, unguarded
        #   r_guarded - ridge + refuses to fit below 2 obs/parameter
        self.cost_estimator = os.getenv("COST_ESTIMATOR", "none").strip().lower() or "none"

        # S7 §A.6.5: how often the provider loop polls the chain. The default
        # 5s cannot reliably catch a 5s reveal window.
        monitor_interval_raw = os.getenv("PROVIDER_MONITOR_INTERVAL", "").strip()
        if monitor_interval_raw:
            self.provider_monitor_interval = float(monitor_interval_raw)

        # S7 §A.6.5 / gate G-E: minimum seconds left in the commit phase for a
        # bid to be worth sending. Unset keeps the historic 10s.
        bid_freshness_raw = os.getenv("BID_FRESHNESS_MIN_SECONDS", "").strip()
        if bid_freshness_raw:
            self.bid_freshness_min_seconds = float(bid_freshness_raw)

        # Reproducibility (S7 §A.1): one seed per run, from which every agent
        # derives its own stream as `run_seed + agent_id`. Unset means unseeded,
        # which is legacy behaviour and must not be used for S7 runs.
        run_seed_raw = os.getenv("RUN_SEED", "").strip()
        self.run_seed = int(run_seed_raw) if run_seed_raw else None

        # Consumer Agent Configuration
        self.consumer_agent_id = int(os.getenv("CONSUMER_AGENT_ID", "0"))  # Set to 0 if not a consumer
        eligible_str = os.getenv("ELIGIBLE_PROVIDERS", "")
        self.eligible_providers = [int(id.strip()) for id in eligible_str.split(",") if id.strip()]
        self.consumer_check_interval = float(os.getenv("CONSUMER_CHECK_INTERVAL", "5"))
        self.consumer_check_interval_jitter = float(os.getenv("CONSUMER_CHECK_INTERVAL_JITTER", "0.0"))
        
        # Random provider selection configuration
        pool_str = os.getenv("PROVIDER_POOL", "")
        self.provider_pool = [int(id.strip()) for id in pool_str.split(",") if id.strip()]
        eligible_per_auction_str = os.getenv("ELIGIBLE_PER_AUCTION", "")
        self.eligible_per_auction = int(eligible_per_auction_str) if eligible_per_auction_str else None
        
        # Consumer Auto-Auction Configuration
        self.auto_create_auction = os.getenv("AUTO_CREATE_AUCTION", "false").lower() == "true"
        self.num_auctions = int(os.getenv("NUM_AUCTIONS", "1"))
        self.auction_creation_delay = int(os.getenv("AUCTION_CREATION_DELAY", "0"))  # Delay before first auction
        self.inter_auction_delay = int(os.getenv("INTER_AUCTION_DELAY", "30"))  # Delay between auctions
        self.pdf_directory = os.getenv("PDF_DIRECTORY", "")

        # Where a provider writes a finished job and the consumer reads it back.
        #
        # **Must be per-run.** The path is keyed by auction id alone
        # (`<jobs>/auction_<id>/result.json`), so two runs sharing it overwrite
        # each other's results and each consumer evaluates whichever job was
        # written last. That is not hypothetical: it gave one agent a rating of
        # 42 belonging to a different run's execution of the same requirement,
        # collapsing its reputation and costing it nineteen consecutive
        # auctions. Ports and chain ids were correctly isolated; the leak was
        # through the filesystem.
        self.jobs_directory = os.getenv(
            "AGENT_JOBS_DIR",
            str(Path(__file__).resolve().parent / "data" / "jobs"),
        )

        # Where a provider persists its sealed-bid commit-reveal nonces.
        #
        # **Must be per-run**, for a sharper reason than the jobs directory: the
        # file is keyed by agent id alone, agent ids repeat across runs, and the
        # runner *deletes* it at startup ("stale commitments from a previous
        # run"). That is right serially and destructive in parallel — a run
        # starting six seconds after another wipes the commitments of every run
        # already in flight. Observed: an agent reads a concurrent run's file,
        # treats auction 1 as already committed and never bids on it, and an
        # agent whose file was wiped cannot reveal a bid it did place. Eight of
        # twenty arm-runs lost auction 1 this way, and only under parallelism —
        # which is why no test and no earlier sequential experiment caught it.
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
        # S7 §A.3: how many requirements one run consumes. Dataset v2 holds 116
        # and a run is 30 auctions, so the pass is a horizon, not the whole file.
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
        # Consumer-side models (S4 §2.1), chosen deliberately rather than inherited:
        # requirement generation and evaluation both run the executor's model, so the
        # consumer side is one model. Requirement generation was `openai/gpt-oss-20b`
        # (hardcoded in service_generator.py); evaluation resolved through an
        # `openrouter_model` attribute config.py never defined, so its `gpt-4o-mini`
        # fallback always fired and nobody chose it. Both are explicit and overridable
        # now, and the collector records them into every collection manifest.
        self.service_generation_model = (
            os.getenv("SERVICE_GENERATION_MODEL", "").strip() or CONSUMER_LLM_DEFAULT
        )
        self.openrouter_model = (
            os.getenv("EVALUATOR_MODEL", "").strip()
            or os.getenv("OPENROUTER_MODEL", "").strip()
            or CONSUMER_LLM_DEFAULT
        )
        self.max_budget = int(os.getenv("MAX_BUDGET", "10000000"))  # 10 USDC with 6 decimals
        # Sealed-bid commit-reveal auction phases: providers commit a hash during
        # commit_duration, then disclose (bid, nonce) during reveal_duration.
        self.commit_duration = int(os.getenv("COMMIT_DURATION", "1800"))  # 30 minutes
        self.reveal_duration = int(os.getenv("REVEAL_DURATION", "300"))  # 5 minutes
        self.reputation_weight = int(os.getenv("REPUTATION_WEIGHT", "30"))  # 0-100, weight of reputation in bid scoring
        
        # Agent workspace
        self.workspace_dir = os.getenv("WORKSPACE_DIR", "./workspaces")
        
        # RAG parameters (fixed configuration)
        self.rag_temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))
        self.rag_retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "3"))
        self.rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        self.rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.service_recursion_limit = int(os.getenv("SERVICE_RECURSION_LIMIT", "25"))
        self.react_recursion_limit = int(os.getenv("REACT_RECURSION_LIMIT", "20"))
        
        # Provider bidding configuration
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
        # Heuristic bidding configuration
        self.heuristic_strategy = os.getenv("HEURISTIC_STRATEGY", "cost_plus")
        self.heuristic_min_margin = float(os.getenv("HEURISTIC_MIN_MARGIN", "0.10"))
        self.heuristic_max_margin = float(os.getenv("HEURISTIC_MAX_MARGIN", "0.30"))
        self.heuristic_bid_tick_usd = float(os.getenv("HEURISTIC_BID_TICK_USD", "0.000001"))
        self.heuristic_recovery_pressure_max = float(os.getenv("HEURISTIC_RECOVERY_PRESSURE_MAX", "1.0"))
        # Starting margin for the derivative-follower controllers (h2_strategy_3, h3_strategy_2):
        #   "max" (default) start greedy at max_margin and concede down (classic ZIP; spends the
        #          first auctions probing down from the ceiling, gifting them to non-conceders);
        #   "mid"  start at the midpoint of [min, max] — competitive from the first bid;
        #   "random" start at a uniform draw in [min, max] — desynchronises same-group agents.
        self.heuristic_initial_margin_mode = os.getenv("HEURISTIC_INITIAL_MARGIN_MODE", "max").strip().lower()

        # When enabled, LLM-based providers (architecture 2/3) are told to keep their bid
        # margin within [heuristic_min_margin, heuristic_max_margin] — the same band the
        # heuristic providers sample from. This is a prompt-level instruction only (not an
        # enforced clamp), used to give the LLM and heuristic agents comparable action spaces.
        self.llm_margin_bounds_enabled = os.getenv("LLM_MARGIN_BOUNDS_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # Reasoning-sophistication axis (R1): when enabled, LLM-based providers are shown the
        # margin they committed on their most recent bid, and the prompt explains it. This is a
        # single-scalar episodic memory of the last decision (distinct from the aggregate stats
        # that R0 already exposes), toggled independently of architecture level so R0 vs R1 can
        # be compared in the same environment.
        self.llm_last_margin_enabled = os.getenv("LLM_LAST_MARGIN_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # Reasoning-sophistication axis (R1, full episodic memory): when enabled, LLM-based
        # providers are shown the complete decision->outcome record of their own past bids —
        # tier chosen, margin, win/loss, reputation before/after, realized balance change.
        # This is the superset of llm_last_margin_enabled: it supplies the outcome feedback a
        # single-shot (R0) agent structurally lacks, so a committed policy can be learned from
        # realized consequences rather than re-derived each cycle.
        # Actuation mode for LLM providers. Native tool-calling asks the model to
        # emit a `place_bid` function call; when enabled, the model instead returns
        # schema-enforced JSON that the *system* parses and executes.
        #
        # Why this exists: on gpt-oss-20b the tool path silently lost 28-33% of
        # auctions — the model reasoned correctly, wrote a well-formed bid in its
        # text, and never emitted the call, which logged as "0 bids placed". H2's
        # estimator (experiments/estimation/llm_client.py) has always used
        # `response_format: json_schema` and never had to call a tool, so leaving
        # the react arm on tool-calling handicaps it in a dimension H2's X_d was
        # never tested on and confounds pricing reasoning with call formatting.
        self.llm_structured_bid_enabled = os.getenv("LLM_STRUCTURED_BID_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        self.llm_outcome_history_enabled = os.getenv("LLM_OUTCOME_HISTORY_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )
        # Cap on how many past auctions are rendered (most recent first). 0 = no limit.
        self.llm_outcome_history_limit = int(os.getenv("LLM_OUTCOME_HISTORY_LIMIT", "0"))

        # Reasoning-sophistication axis (R2, reflection with a self-edited strategy memo).
        # Before each bidding decision the agent makes a separate LLM call that reviews what just
        # happened and revises a single persistent memo — keeping what still holds, rewriting what
        # the evidence contradicts, adding what is new (self-editing memory, MemGPT/Letta-style;
        # verbal self-critique from Reflexion, Shinn et al. 2023; insight abstraction from ExpeL,
        # Zhao et al. 2024). The memo is sectioned into beliefs about the market, current strategy,
        # and open questions, so the world model and the policy can be revised independently.
        # Toggled independently of llm_outcome_history_enabled so R0 / R1 / R2 / R1+R2 are all
        # runnable: reflection reads the outcome record internally either way. The reflection
        # call's LLM cost is charged to the agent like any other reasoning cost.
        self.llm_reflection_enabled = os.getenv("LLM_REFLECTION_ENABLED", "false").strip().lower() in (
            "1", "true", "yes",
        )

        # Optional per-run overrides of the architecture's default LLM model, so the model
        # can be an experimental variable (set via YAML/CLI) instead of a code edit to
        # ARCHITECTURES. Empty/unset means "use the architecture's hardcoded default".
        self.blockchain_llm_override = os.getenv("BLOCKCHAIN_LLM_MODEL", "").strip() or None
        self.service_llm_override = os.getenv("SERVICE_LLM_MODEL", "").strip() or None

        # S7 §A.2 (D2). Sampling for live LLM calls. Both default to "use what the
        # estimation protocol pinned for this model" — the seed and reasoning
        # effort S5 sent — because an estimator sampled differently is not the
        # estimator whose sigma S5 measured. Overrides exist for models the
        # protocol never pinned, which is the A-series react arms only.
        llm_temperature_raw = os.getenv("LLM_TEMPERATURE", "").strip()
        self.llm_temperature_override = float(llm_temperature_raw) if llm_temperature_raw else None
        llm_sampling_seed_raw = os.getenv("LLM_SAMPLING_SEED", "").strip()
        self.llm_sampling_seed = int(llm_sampling_seed_raw) if llm_sampling_seed_raw else None

        # Cost estimate smoothing configuration
        self.execution_cost_ema_alpha = min(
            1.0,
            max(0.0, float(os.getenv("EXECUTION_COST_EMA_ALPHA", "0.25"))),
        )
        self.non_execution_cost_ema_alpha = min(
            1.0,
            max(0.0, float(os.getenv("NON_EXECUTION_COST_EMA_ALPHA", "0.20"))),
        )
        
        # Cost tracking constants (for economic simulation)
        self.gas_cost_mode = os.getenv("GAS_COST_MODE", "receipt")  # "receipt" or "fixed"
        self.gas_price_gwei = float(os.getenv("GAS_PRICE_GWEI", "20.0"))
        self.eth_price_usd = float(os.getenv("ETH_PRICE_USD", "3000.0"))

        # Token pricing by execution domain.
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
        # Consumer-side token pricing (requirement generation + evaluation). Defaults
        # follow the service-LLM prices because the consumer side runs the same model;
        # override if the two are split. Without these, S4 §3.6 cannot price the
        # consumer-side model choice and §2.1's cost question stays unanswerable.
        self.consumer_llm_input_price_per_1k = float(
            os.getenv("CONSUMER_LLM_INPUT_PRICE_PER_1K", str(self.service_llm_input_price_per_1k))
        )
        self.consumer_llm_output_price_per_1k = float(
            os.getenv("CONSUMER_LLM_OUTPUT_PRICE_PER_1K", str(self.service_llm_output_price_per_1k))
        )
        
        # Coupling mode configuration
        self.coupling_mode = os.getenv("COUPLING_MODE", "isolated")  # "isolated", "one_way", "two_way"
        
        # Experimental architecture selection
        self.architecture = os.getenv("ARCHITECTURE", "1")  # Default to architecture 1
        
        # Effort tier configuration for service-execution search budgets.
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
        # Passages returned per search, per tier. S4 §3's cap sweep showed the round
        # count alone is a near-binary knob — two rounds saturate every depth, because
        # each search already returns rag_retrieval_k passages. The evidence budget a
        # tier buys is rounds x passages, so the second factor has to move too or the
        # middle tier cannot sit between the other two.
        # Executor model per effort tier (S4 §3c). Empty means "every tier uses the
        # architecture's service_llm", which is the pre-§3c behaviour. Model capability
        # is a far larger lever than retrieval budget, and it is a realistic effort
        # decision — a provider choosing a cheaper subcontractor.
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
        self.service_retrieval_k_by_effort = {
            "low": int(os.getenv("SERVICE_RETRIEVAL_K_LOW", str(self.rag_retrieval_k))),
            "medium": int(os.getenv("SERVICE_RETRIEVAL_K_MEDIUM", str(self.rag_retrieval_k))),
            "high": int(os.getenv("SERVICE_RETRIEVAL_K_HIGH", str(self.rag_retrieval_k))),
        }
        self.default_effort_tier = "medium"
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def get_initial_service_execution_cost_for_effort(self, effort_tier: str = None) -> float:
        """Return prior execution-cost estimate for the requested effort tier."""
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
        """Return prior quality-rating estimate for the requested effort tier."""
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
        """Return the configured tool/retrieval budget for a service effort tier."""
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
        """
        Validate that required configuration is present.
        
        Returns:
            True if valid, False otherwise
        """
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
        """Display current configuration (hiding sensitive data)."""
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


# ============================================================================
# Experimental Architecture Configurations
# ============================================================================

from dataclasses import dataclass
from typing import List


@dataclass
class ArchitectureConfig:
    """Configuration for a specific agent architecture variant."""
    name: str
    description: str
    state_level: int 
    reasoning_mode: str  # "deterministic", "llm_react", "llm_strategic"
    coupling_mode: str  # "isolated", "one_way", "two_way"
    prompt_template: str  # "minimal", "guided", "rich"
    enabled_tools: List[str]
    blockchain_llm: str = "openai/gpt-4o-mini"
    service_llm: str = "openai/gpt-4o-mini"
    # S7 §A.2 (D2): 0, not 0.3. An architecture that forgets to set this must
    # not sample stochastically by default. Models whose sampling is pinned by
    # the estimation protocol ignore this and use their recorded block instead
    # (agents/infrastructure/llm_sampling.py).
    llm_temperature: float = 0.0


# Architecture registry - add more as experiments evolve
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
    "3": ArchitectureConfig(
        name="Architecture 3",
        description="LLM Effort-Aware - Performance + effort-history state, LLM reasoning, two-way coupling",
        state_level=1,
        reasoning_mode="llm_react",
        coupling_mode="two_way",
        prompt_template="3",
        enabled_tools=["place_bid"],
        blockchain_llm="openai/gpt-4o-mini",
        service_llm="openai/gpt-5.4-mini",
        llm_temperature=0.0,
    ),
    # Future architectures will be added here:
    # "4": ArchitectureConfig(...),
    # "5": ArchitectureConfig(...),
}


def get_architecture(name: str) -> ArchitectureConfig:
    """Get architecture configuration by name."""
    if name not in ARCHITECTURES:
        available = ", ".join(ARCHITECTURES.keys())
        raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
    return ARCHITECTURES[name]


# Global configuration instance
config = Config()