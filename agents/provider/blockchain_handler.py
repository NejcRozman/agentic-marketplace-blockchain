"""
BlockchainHandler - Agentic handler for blockchain operations.

This agent uses LangGraph to handle blockchain operations:
- Complete service path: Deterministic workflow for completing auctions
- Monitor path: Heuristic or ReAct agent with tools for intelligent bidding decisions

Built with LangGraph for agentic reasoning and decision-making.
"""

import asyncio
import inspect
import logging
import pathlib
import json
import time
import re
import math
import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from ..common.blockchain_client import BlockchainClient
    from ..common.contract_abis import (
        get_reverse_auction_abi,
        get_identity_registry_abi,
        get_reputation_registry_abi
    )
    from ..common.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
    from ..common.ipfs_client import IPFSClient
    from ..common.cost_tracker import CostTracker, LLMCostCallback
    from ..common.dataset_sampler import DatasetSampler
    from ..common.commitment_store import CommitmentStore
    from ..common.reputation_lookup import reputation_to_float
    from ..common.seeding import agent_rng
    from ..common.llm_sampling import chat_model_kwargs
    from . import cost_estimators
    from ..common.prompts import get_prompt
    from ..config import config, get_architecture
    from . import policies
except ImportError:
    from common.blockchain_client import BlockchainClient
    from common.contract_abis import (
        get_reverse_auction_abi,
        get_identity_registry_abi,
        get_reputation_registry_abi
    )
    from common.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
    from common.ipfs_client import IPFSClient
    from common.cost_tracker import CostTracker, LLMCostCallback
    from common.dataset_sampler import DatasetSampler
    from common.commitment_store import CommitmentStore
    from common.reputation_lookup import reputation_to_float
    from common.seeding import agent_rng
    from common.llm_sampling import chat_model_kwargs
    import cost_estimators
    from common.prompts import get_prompt
    from config import Config, get_architecture
    from provider import policies
    config = Config()

logger = logging.getLogger(__name__)


def _auction_rw(auction: Dict[str, Any], default: int = 50) -> int:
    """Reputation weight from an auction dict, preserving a legitimate 0.

    `auction.get("reputation_weight", 50) or 50` silently turned rw=0 into
    rw=50, because 0 is falsy. rw=0 is a real setting (award on price alone),
    so absent/None must be distinguished from zero.
    """
    value = auction.get("reputation_weight")
    if value is None:
        return default
    return int(value)


class AgentState(TypedDict):
    """
    State for blockchain agent workflow.
    
    Organized by temporal dimension (current/history/future) and parameter type
    (performance/market/decision). Not all fields need to be populated - simpler
    architectures can leave history/future fields empty without affecting reasoning.
    """
    
    # ============================================================================
    # WORKFLOW CONTROL (always populated)
    # ============================================================================
    agent_id: int
    action: str  # "complete_service" or "monitor"
    
    # Transaction results
    tx_hash: Optional[str]
    error: Optional[str]
    messages: List[Any]
    
    # ============================================================================
    # CURRENT STATE - Present moment snapshot
    # ============================================================================

    # Active auctions and market conditions
    eligible_active_auctions: List[Dict[str, Any]]  # Auctions we can bid on (BC + IPFS data)
    won_auctions: List[Dict[str, Any]]  # Auctions we won (for service completion) 

    # Current reputation context
    competitors_reputation: List[Dict[str, Any]]  # All other agents bidding in auctions with their reputation
    agent_reputation: Dict[str, Any]  # Agent's current reputation score and feedback count

    # Agent's current performance context
    estimated_service_cost: Optional[float]  # Basic cost estimate exposed to isolated architectures (USD)
    estimated_execution_cost: Optional[float]  # Estimated direct execution cost per completed service (USD)
    non_execution_cost_per_participation: Optional[float]  # Historical non-execution cost per auction participation (USD)
    unrecovered_overhead_balance: Optional[float]  # Historical non-execution overhead not yet repaid by service margin (USD)
    current_balance: float  # Current net balance (revenue - costs, USD)
    
    # Results of current decision cycle
    bids_placed: List[Dict[str, Any]]  # Bids submitted this invocation
    reveals: List[Dict[str, Any]]  # Sealed-bid reveals attempted this invocation

    # Service completion workflow (only for complete_service action)
    auction_id: Optional[int]
    client_address: Optional[str]
    feedback_auth: Optional[bytes]
    
    # ============================================================================
    # HISTORY STATE - Past observations and outcomes
    # ============================================================================
    
    # --- History Performance Parameters ---
    past_execution_costs: List[float]  # Historical service execution costs (USD, chronological)
    effort_level_execution_costs: Dict[str, float]  # Per-effort execution cost estimates (USD)
    effort_level_execution_counts: Dict[str, int]  # Completed service counts per effort level
    effort_level_quality_stats: Dict[str, Dict[str, float]]  # Per-effort quality stats (avg/variance/count)
    bids_placed_count: Optional[int]  # On-chain commit attempts that succeeded (for win-rate tracking)
    wins_count: Optional[int]  # Auctions won and services completed
    win_rate: Optional[float]  # wins_count / bids_placed_count; None until first bid
    avg_gross_margin_per_win_usd: Optional[float]  # (revenue - execution) / wins on won auctions; None until first win
    llm_margin_bound_min: Optional[float]  # Prompt-only bid-margin floor shared with heuristic agents (fraction)
    llm_margin_bound_max: Optional[float]  # Prompt-only bid-margin ceiling shared with heuristic agents (fraction)
    llm_last_margin_pct: Optional[float]  # R1: margin % committed on the most recent bid; None until first commit
    llm_outcome_history: Optional[List[Dict[str, Any]]]  # R1: full decision->outcome record of own past bids
    llm_strategy_memo: Optional[str]  # R2: self-edited strategy memo, revised before each decision
    # past_auctions: Optional[List[Dict[str, Any]]]  # Last N auctions participated
    # past_reputation_trajectory: Optional[List[float]]  # Reputation over time
    
    # --- History Market Parameters ---
    # past_market_prices: Optional[List[float]]  # Historical winning bids
    # past_competition_levels: Optional[List[float]]  # Competition over time
    
    # --- History Decision Parameters ---
    # past_bidding_strategy: Optional[List[str]]  # What strategies were used
    # past_bid_outcomes: Optional[List[Dict[str, Any]]]  # Win/loss with details
    
    # ============================================================================
    # FUTURE STATE - Predictions and forward-looking parameters
    # ============================================================================
    
    # --- Future Market Parameters ---
    # predicted_competition: Optional[float]  # Expected future competition
    # predicted_price_trend: Optional[str]  # "increasing", "stable", "decreasing"
    
    # --- Future Performance Parameters ---
    # expected_reputation_change: Optional[float]  # If we win/lose this auction
    # expected_capacity_utilization: Optional[float]  # Future workload
    
    # --- Future Decision Parameters ---
    # recommended_bid_range: Optional[Dict[str, float]]  # Min/max profitable bids
    # risk_assessment: Optional[str]  # "low", "medium", "high"


class BlockchainHandler:
    """
    Agentic blockchain handler using LangGraph.
    
    - complete_service: Deterministic path for service completion
    - monitor: Heuristic or ReAct agent for intelligent bidding decisions
    """

    # Keep weight semantics in percent (0-100) while increasing score precision.
    SCORE_PRECISION = 10_000
    WEIGHT_SCALE = 100
    # Max random pre-reasoning delay (seconds) to desync providers detecting the same new auction.
    _BID_JITTER_MAX_SECONDS = 3.0

    # The jitter above must never eat the commit window. A fixed 0-3s delay is
    # 30% of a 10s phase (S7 §A.6.5), which is enough on its own to push every
    # bid past the freshness gate below and produce a market with no bids at all.
    _JITTER_WINDOW_FRACTION = 0.15

    # Max random delay before revealing, mirroring the commit jitter above.
    #
    # The commit jitter desynchronises the agents for the phase where ordering
    # does not matter, and left untouched the one phase where it does. Two
    # identical strategies compute an identical first bid, both wait on the same
    # on-chain deadline (`getAuctionPhase` flipping to Reveal), and fire within a
    # millisecond of each other into the same one-second Anvil block — so
    # `ReverseAuction.revealBid`'s strict `score >` tie-break is decided by
    # mempool arrival order, i.e. OS scheduling. Measured over six identical
    # re-runs of seed 1007, that coin flip selected between two attractors whose
    # profit outcomes differed by $0.44.
    #
    # Drawn from the agent's own seeded RNG, so reveal order becomes a
    # reproducible function of (run_seed, agent_id) rather than a race.
    _REVEAL_JITTER_MAX_SECONDS = 1.5

    #: Keeps the reveal stream from colliding with any agent's bidding stream.
    _REVEAL_RNG_STREAM_OFFSET = 1_000_000

    # A commit sent with less than this left is likely to be mined after the
    # phase closes. Measured on 160 real commits against a *forked* chain:
    # median 0.59s, p95 1.17s, max 5.56s from send to confirmation.
    #
    # The default stays at the historic 10s, which is what the S1 equivalence
    # fixture captured — changing it silently would invalidate that proof. S7's
    # shortened phases (§A.6.5) opt in to a smaller value via config, because a
    # 10s gate on a 10s commit window can never pass: `time_remaining` is at most
    # the window itself, so every auction is skipped and the market records no
    # bids at all rather than an error.
    _DEFAULT_BID_FRESHNESS_MIN_SECONDS = 10.0
    
    def __init__(
        self, 
        agent_id: int, 
        blockchain_client: Optional[BlockchainClient] = None,
        architecture: Optional[str] = None,
        system_prompt: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        enabled_tools: Optional[List[str]] = None,
        reasoning_mode: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize the blockchain handler agent.
        
        Args:
            agent_id: The agent's ID
            blockchain_client: Optional blockchain client instance
            architecture: Architecture name. If None, uses config.architecture
            system_prompt: System prompt for the ReAct agent (if None, uses architecture's template)
            llm_temperature: LLM temperature (if None, uses architecture's default)
            llm_base_url: LLM API base URL (default: from config)
            llm_api_key: LLM API key (default: from config)
            enabled_tools: List of tool names to enable (if None, uses architecture's tools)
            reasoning_mode: "heuristic" or "llm_react" (if None, uses architecture's mode)
            cost_tracker: Optional CostTracker instance (creates new if None)
        """
        self.agent_id = agent_id
        self.client = blockchain_client or BlockchainClient()
        self.config = config
        self.initial_reputation_default = self.config.initial_reputation_default

        # S7 §A.1 (D1/D3/D7): this agent's private RNG. Every controllable draw
        # below goes through it — never through the `random` module, whose global
        # generator is shared across agents and unstable under async interleaving.
        self._rng = agent_rng(getattr(self.config, "run_seed", None), agent_id)
        # Separate stream for the reveal jitter: drawing it from `_rng` would
        # interleave with the bidding draws and change bids that have nothing to
        # do with revealing. The offset keeps the two streams distinct while
        # staying a pure function of the run seed.
        self._reveal_rng = agent_rng(
            getattr(self.config, "run_seed", None),
            (agent_id or 0) + self._REVEAL_RNG_STREAM_OFFSET,
        )

        # S7 §A.6.5 / gate G-E.
        self._bid_freshness_min_seconds = float(getattr(
            self.config, "bid_freshness_min_seconds", self._DEFAULT_BID_FRESHNESS_MIN_SECONDS
        ))
        #: Commits that were mined after their commit phase closed. Must be zero
        #: across the pilot before the phase durations are accepted (G-E).
        self.late_commits = 0
        #: Commitments abandoned because the reveal window would have closed
        #: first. Each one is a forfeited auction, so this must also be zero.
        self.missed_reveals = 0
        self._forfeited_commitments: set = set()

        #: Auctions known to be finished. Monotonic: an auction never returns
        #: from settled to active, so this only ever grows and is safe to trust.
        self._settled_auctions: set = set()

        
        # Load architecture configuration
        arch_name = architecture or self.config.architecture
        self.arch_config = get_architecture(arch_name)
        
        # Apply architecture settings (can be overridden by explicit kwargs)
        self.reasoning_mode = reasoning_mode or self.arch_config.reasoning_mode
        if self.reasoning_mode not in ["deterministic", "heuristic", "llm_react", "llm_strategic"]:
            raise ValueError(f"Invalid reasoning_mode: {self.reasoning_mode}")
        
        # System prompt configuration
        self.system_prompt = system_prompt
        
        # LLM configuration (with architecture defaults, overridable per-run)
        self.llm_model = getattr(self.config, "blockchain_llm_override", None) or self.arch_config.blockchain_llm
        if llm_temperature is not None:
            self.llm_temperature = llm_temperature
        elif getattr(self.config, "llm_temperature_override", None) is not None:
            self.llm_temperature = self.config.llm_temperature_override
        else:
            self.llm_temperature = self.arch_config.llm_temperature
        self.llm_base_url = llm_base_url or self.config.openrouter_base_url
        self.llm_api_key = llm_api_key or self.config.openrouter_api_key

        # Constructed here, *after* the LLM configuration above: the hybrid arm
        # reads `self.llm_model` and `self.llm_api_key`, and placing this earlier
        # crashed every X provider at startup with an AttributeError.
        # S7 §B.2: arm R. `none` leaves H exactly as it was — no estimator is
        # constructed and `_policy_economics` returns the history beliefs
        # untouched.
        estimator_mode = str(getattr(self.config, "cost_estimator", "none") or "none").lower()
        if estimator_mode not in cost_estimators.ESTIMATOR_MODES:
            raise ValueError(
                f"unknown cost_estimator {estimator_mode!r}; "
                f"expected one of {cost_estimators.ESTIMATOR_MODES}"
            )
        self.cost_estimator_mode = estimator_mode
        #: Auctions whose winning bid has already been booked. Completion is
        #: retried on failure and may be re-entered after the chain has already
        #: settled, so revenue is keyed by auction rather than by attempt.
        self._revenue_recorded: set = set()
        if estimator_mode == "none":
            self._cost_estimator = None
        elif estimator_mode == "decomposed":
            self._cost_estimator = cost_estimators.DecomposedCostEstimator(
                model_key=self.llm_model,
                run_seed=getattr(self.config, "run_seed", None),
                agent_id=agent_id,
                api_key=self.llm_api_key or "",
            )
        elif estimator_mode == "oracle":
            # A bound, not a competitor: it reads the dataset's own ground truth.
            # Pointed at the run's sampled dataset so an oracle can never be fed
            # truth from a dataset the market is not drawing from.
            self._cost_estimator = cost_estimators.OracleCostEstimator(
                dataset_dir=str(getattr(self.config, "sampled_dataset_dir", "") or "") or None
            )
        else:
            self._cost_estimator = cost_estimators.RegressionCostEstimator(mode=estimator_mode)
        self._estimator_sampler: Optional[DatasetSampler] = None

        # S7 §A.2 (D2): sampling comes from the estimation protocol's pinned block
        # for this model — seed plus reasoning effort, or seed plus temperature,
        # depending on the family. Resolved once so both call sites below cannot
        # diverge from each other.
        self.llm_sampling_kwargs = chat_model_kwargs(
            self.llm_model,
            temperature_override=self.llm_temperature,
            seed_override=getattr(self.config, "llm_sampling_seed", None),
        )
        
        # Tools configuration (from architecture)
        self.enabled_tools = enabled_tools or self.arch_config.enabled_tools
        
        # Store coupling mode from architecture (used in state population)
        self.coupling_mode = self.arch_config.coupling_mode
        
        self.contracts_loaded = False
        self.reverse_auction_contract = None
        self.identity_registry_contract = None
        self.reputation_registry_contract = None
        
        # Cost tracking (create new if not provided)
        self.cost_tracker = cost_tracker or CostTracker(agent_id=agent_id, config=self.config)
        
        # Block tracking - persists across invocations
        self._last_processed_block: int = 0
        
        # Bids placed tracker (for ReAct tool results)
        self._bids_placed: List[Dict[str, Any]] = []
        self._react_stop_after_bid_attempt: bool = False
        self._bid_attempt_stats: Dict[str, int] = {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
        }
        self._bid_failures: List[Dict[str, Any]] = []
        self._own_current_bid_by_auction: Dict[int, int] = {}

        # S7 §C / S6 rung O2c: every price this agent has actually won at, keyed
        # by auction so a re-scan of the same AuctionEnded event cannot count a
        # win twice. S6 keeps a plain list (``SimulatedProvider._winning_bids_micro``)
        # because its market resolves each auction exactly once; the live agent
        # reads events from the chain and needs the idempotency. Same multiset,
        # so ``expected_clearing_price_micro`` sees identical input either way.
        self._own_winning_bids_micro: Dict[int, int] = {}

        # R1 episodic memory: margin % (over total cost basis) committed on the most recent
        # successful bid; surfaced next cycle only when llm_last_margin_enabled. None until first commit.
        self._llm_last_margin_pct: Optional[float] = None

        # R1 full episodic memory: one decision->outcome record per committed bid, surfaced only
        # when llm_outcome_history_enabled. Entries start as "pending" and are resolved to
        # won/lost as AuctionEnded events arrive; reputation-after is filled in once the win's
        # feedback lands on-chain. _llm_outcome_by_auction indexes into the same dicts.
        self._llm_outcome_history: List[Dict[str, Any]] = []
        self._llm_outcome_by_auction: Dict[int, Dict[str, Any]] = {}
        self._recently_ended_auction_ids: List[int] = []

        # R2 reflection: a single self-edited strategy memo, revised before every bidding
        # decision. Independent of the R1 flag — reflection reads the outcome record internally
        # even when that record is not shown verbatim in the bidding prompt. Every revision is
        # logged (🧠 STRATEGY MEMO v..) so the trajectory survives the overwrite.
        self._llm_strategy_memo: Optional[str] = None
        self._llm_memo_version: int = 0
        # Compact rationale the model wrote for its own recent decisions, used for credit
        # assignment during reflection (Reflexion feeds thoughts, not just outcomes).
        self._llm_decision_rationales: List[Dict[str, Any]] = []
        self._RATIONALES_IN_REFLECTION = 3

        # Per-cycle state snapshot used by tool-call logging
        self._tool_audit_context: Dict[str, Any] = {}

        # Controller memory for the extracted policies (policies.py), keyed by
        # strategy name. Created lazily by _policy_state_for.
        self._policy_states: Dict[str, "policies.PolicyState"] = {}

        self._default_two_way_effort_tier = "medium"
        self._auction_effort_tier_by_auction: Dict[int, str] = {}
        self._pending_feedback_effort_queue: List[Dict[str, Any]] = []
        self._last_reputation_feedback_count: Optional[int] = None
        self._last_reputation_average_score: Optional[float] = None

        # Durable local storage for sealed-bid commit-reveal nonces (survives process restarts)
        # Per-run storage via `AGENT_COMMITMENTS_DIR`; see
        # `Config.commitments_directory` for why sharing it breaks parallel runs.
        _commit_dir = getattr(self.config, "commitments_directory", None)
        self._commitment_store = CommitmentStore(
            agent_id=agent_id,
            storage_dir=pathlib.Path(_commit_dir) if _commit_dir else None,
        )
        # In-memory set of auction IDs already reasoned about this session (includes abstentions).
        self._decided_auctions: set = set()
        # Per-cycle set: reset before each LLM reasoning run; blocks duplicate place_bid
        # calls within the same LangGraph ReAct execution for the same auction_id.
        self._bid_attempted_auctions_this_cycle: set = set()

        # Build tools for ReAct agent
        self._tools = self._build_tools()
        
        # Build the graph
        self.graph = self._build_graph()
        
        prompt_info = "custom" if system_prompt else self.arch_config.prompt_template
        logger.info(
            f"BlockchainHandler initialized for agent {agent_id} "
            f"(architecture={self.arch_config.name}, reasoning_mode={self.reasoning_mode}, "
            f"coupling_mode={self.coupling_mode}, prompt={prompt_info}, "
            f"model={self.llm_model}, temp={self.llm_temperature}, tools={len(self._tools)})"
        )
        logger.info("Tool-call logging enabled (state snapshot + raw tool args)")
    
    #: Schema for the structured (non-tool-calling) bid decision.
    BID_RESPONSE_SCHEMA = {
        "name": "bid_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "auction_id": {"type": "integer"},
                "bid_amount": {"type": "integer"},
            },
            "required": ["reasoning", "auction_id", "bid_amount"],
            "additionalProperties": False,
        },
    }

    #: Appended to the prompt in structured mode. Supersedes the prompt's
    #: "call place_bid" step — the system places the bid from this JSON.
    STRUCTURED_BID_INSTRUCTION = (
        "\n\nOUTPUT FORMAT — this supersedes any instruction above to call a tool.\n"
        "Do NOT call any function. Return a single JSON object and nothing else:\n"
        '  {"reasoning": "<your margin reasoning>", "auction_id": <int>, '
        '"bid_amount": <int micro-USDC>}\n'
        "The system places the bid from this object. Choose exactly one auction "
        "from the list above and commit to it."
    )

    @staticmethod
    def _parse_structured_bid(content: str) -> Optional[Dict[str, Any]]:
        """The bid object from a model response, or None.

        Tolerates a fenced code block or surrounding prose, because
        `strict: true` is honoured by the endpoint but not by every model.
        """
        if not content:
            return None
        text = content.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fence:
            text = fence.group(1)
        else:
            brace = re.search(r"\{.*\}", text, re.S)
            if brace:
                text = brace.group(0)
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        try:
            auction_id = int(obj["auction_id"])
            bid_amount = int(obj["bid_amount"])
        except (KeyError, TypeError, ValueError):
            return None
        return {"auction_id": auction_id, "bid_amount": bid_amount,
                "reasoning": str(obj.get("reasoning") or "")}

    async def _structured_bid_invoke(self, llm, prompt: str, tool_lookup: Dict[str, Any]) -> Dict[str, Any]:
        """One schema-enforced call, parsed by us, executed by us.

        Mirrors how H2's cost estimator works (`response_format: json_schema`,
        then `parse_estimate`), so the react arm and X_d face the same actuation
        path and a comparison between them isolates pricing reasoning rather
        than function-call formatting.

        Returns a message dict shaped like the react agent's, so the caller's
        tracing, rationale capture and outcome history are unchanged.
        """
        structured = llm.bind(
            response_format={"type": "json_schema",
                             "json_schema": self.BID_RESPONSE_SCHEMA}
        )
        response = await structured.ainvoke(
            [HumanMessage(content=prompt + self.STRUCTURED_BID_INSTRUCTION)]
        )
        content = str(getattr(response, "content", "") or "")
        messages = [HumanMessage(content=prompt), response]

        decision = self._parse_structured_bid(content)
        if decision is None:
            # Unparseable output is a genuine no-bid, and is counted as one.
            self._structured_parse_failures = getattr(self, "_structured_parse_failures", 0) + 1
            logger.warning("🤖 structured bid unparseable; no bid this cycle: %.200s", content)
            return {"messages": messages}

        place_bid = tool_lookup.get("place_bid")
        if place_bid is None:
            logger.error("🤖 structured mode: place_bid tool unavailable")
            return {"messages": messages}

        args = {"auction_id": decision["auction_id"], "bid_amount": decision["bid_amount"]}
        logger.info("🔧 structured place_bid(%s)", args)
        try:
            # `place_bid` is a SYNC tool that calls `asyncio.run()` internally.
            # LangChain runs sync tools in a worker thread; invoking it directly
            # from this coroutine raises "asyncio.run() cannot be called from a
            # running event loop" and every bid silently fails. Hop to a thread
            # so it sees no running loop, exactly as the react path does.
            outcome = await asyncio.to_thread(place_bid.invoke, args)
        except Exception as exc:                       # noqa: BLE001 - surfaced, not swallowed
            logger.warning("🤖 structured place_bid failed: %s", exc)
            return {"messages": messages}
        messages.append(ToolMessage(content=json.dumps(outcome, default=str),
                                    tool_call_id="structured_bid"))
        return {"messages": messages}

    def _llm_kwargs_with_usage(self) -> Dict[str, Any]:
        """Sampling kwargs plus OpenRouter's usage accounting, merged.

        `extra_body` already carries per-model sampling settings (e.g.
        `reasoning.effort` for gpt-oss-20b), so the usage flag must be merged
        into it. Passing a second `extra_body=` raises "got multiple values for
        keyword argument 'extra_body'", which surfaced only as
        "Error in ReAct reasoning" and left the arm bidding zero times.
        """
        kwargs = dict(self.llm_sampling_kwargs)
        extra = dict(kwargs.get("extra_body") or {})
        usage = dict(extra.get("usage") or {})
        usage["include"] = True
        extra["usage"] = usage
        kwargs["extra_body"] = extra
        return kwargs

    def _build_tools(self) -> List:
        """Build tools for the ReAct agent."""
        handler = self

        # ============================================================================
        # COMPUTATION TOOLS - perform operation LLM is bad at
        # ============================================================================

        def _get_cycle_auction_context(auction_id: int) -> Dict[str, Any]:
            auction_key = str(int(auction_id))
            auctions = (handler._tool_audit_context or {}).get("auctions") or {}
            auction_context = auctions.get(auction_key)
            if auction_context is None:
                raise ValueError(
                    f"auction_id {auction_id} is not present in the current reasoning-cycle snapshot"
                )
            return auction_context

        def _get_cycle_agent_reputation() -> float:
            rating = (handler._tool_audit_context or {}).get("agent_reputation")
            if isinstance(rating, (int, float)):
                return rating
            return handler.initial_reputation_default

        def _get_cycle_micro_value(field_name: str) -> int:
            value = (handler._tool_audit_context or {}).get(field_name)
            if isinstance(value, int):
                return value
            raise ValueError(f"{field_name} is not available in the current reasoning-cycle snapshot")

        # ============================================================================
        # ACTUATION TOOLS - perform actions that affect the world (e.g., placing bids)
        # ============================================================================
                
        @tool(return_direct=True)
        def place_bid(auction_id: int, bid_amount: int, effort_tier: Optional[str] = None) -> Dict[str, Any]:
            """Submit a sealed bid commitment for an auction on the blockchain.

            This seals your bid (auction_id, bid_amount) behind a hash on-chain; the actual
            amount stays hidden until the automatic reveal step later, so there is no
            immediate win/loss feedback. You can commit to a given auction_id only ONCE,
            ever - a second call for the same auction_id always fails, even with a different
            bid_amount. Revealing and any eventual win/loss determination happen
            automatically outside your control once the auction's reveal phase begins.

            effort_tier (optional): "low" | "medium" | "high" — the quality tier to deliver if
            you win. Only used in effort-aware mode (architecture 3); ignored otherwise.
            """
            raw_args = {"auction_id": auction_id, "bid_amount": bid_amount, "effort_tier": effort_tier}
            handler._react_stop_after_bid_attempt = True
            auction_id_int = int(auction_id) if auction_id is not None else 0

            # One bid attempt per auction per reasoning cycle — prevents the LLM from
            # retrying after NotInCommitPhase or any other failure within the same graph
            # execution, which was causing 10+ wasted blockchain calls per auction.
            if auction_id_int in handler._bid_attempted_auctions_this_cycle:
                result = {
                    "success": False,
                    "error": "AlreadyAttempted",
                    "explanation": (
                        "A bid attempt was already made for this auction in the current "
                        "reasoning cycle. Each auction allows exactly one commit attempt — "
                        "this is enforced by the smart contract. Do not retry."
                    ),
                    "retry_recommended": False,
                }
                handler._audit_tool_call("place_bid", raw_args, raw_args, result, None)
                return result
            handler._bid_attempted_auctions_this_cycle.add(auction_id_int)

            try:
                logger.info(
                    f"📤 Committing bid: auction={auction_id}, amount={bid_amount} ({bid_amount/1e6:.6f} USDC)"
                )

                fresh_auction = asyncio.run(
                    handler.client.call_contract_method("ReverseAuction", "getAuctionDetails", auction_id)
                )
                max_price = int(fresh_auction[3])
                commit_duration = int(fresh_auction[4])
                start_time = int(fresh_auction[5])
                is_active = bool(fresh_auction[9])

                latest_block = asyncio.run(handler.client.get_block("latest"))
                current_time = int(latest_block["timestamp"])
                commit_deadline = start_time + commit_duration

                if (not is_active) or current_time >= commit_deadline:
                    result = {
                        "success": False,
                        "error": "NotInCommitPhase",
                        "error_code": "0xabfb7dc3",
                        "explanation": "Commit phase ended (or auction inactive) before submission. State refreshed and tx skipped.",
                        "retry_recommended": False,
                    }
                    handler._record_bid_attempt(
                        auction_id=auction_id,
                        bid_amount=bid_amount,
                        success=False,
                        error=result.get("error"),
                        error_code=result.get("error_code"),
                        mode="react",
                    )
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result

                if bid_amount > max_price:
                    result = {
                        "success": False,
                        "error": "BidTooHigh",
                        "error_code": "0xc9b80cd4",
                        "explanation": "Bid exceeds current auction max_price. State refreshed and tx skipped.",
                        "retry_recommended": True,
                    }
                    handler._record_bid_attempt(
                        auction_id=auction_id,
                        bid_amount=bid_amount,
                        success=False,
                        error=result.get("error"),
                        error_code=result.get("error_code"),
                        mode="react",
                    )
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result

                if asyncio.run(handler._has_committed(auction_id)):
                    result = {
                        "success": False,
                        "error": "AlreadyCommitted",
                        "error_code": "0xbfec5558",
                        "explanation": "You already have a sealed commitment for this auction. No tx sent.",
                        "retry_recommended": False,
                    }
                    handler._record_bid_attempt(
                        auction_id=auction_id,
                        bid_amount=bid_amount,
                        success=False,
                        error=result.get("error"),
                        error_code=result.get("error_code"),
                        mode="react",
                    )
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result

                resolved_bid_effort_tier = handler._resolve_bid_effort_tier(effort_tier)
                result = asyncio.run(
                    handler._commit_bid(
                        auction_id, bid_amount, effort_tier=resolved_bid_effort_tier, mode="react"
                    )
                )
                if result.get("success"):
                    handler._record_llm_last_margin(bid_amount)
                    handler._record_llm_decision(
                        auction_id, bid_amount, resolved_bid_effort_tier
                    )
                handler._audit_tool_call(
                    "place_bid", raw_args, raw_args, result, result.get("error_code") or result.get("error")
                )
                return result

            except Exception as e:
                logger.error(f"Error committing bid: {e}")
                error_msg = str(e)
                result = handler._classify_commit_error(error_msg)
                handler._record_bid_attempt(
                    auction_id=auction_id,
                    bid_amount=bid_amount,
                    success=False,
                    error=result.get("error"),
                    error_code=result.get("error_code"),
                    mode="react",
                )
                handler._audit_tool_call(
                    "place_bid", raw_args, raw_args, result, result.get("error_code") or result.get("error")
                )
                return result

        # Build tool list based on enabled_tools configuration
        all_tools = {
            "place_bid": place_bid,
        }
        
        enabled = []
        for name in self.enabled_tools:
            if name not in all_tools:
                continue
            enabled.append(all_tools[name])
        
        if len(enabled) < len(self.enabled_tools):
            missing = set(self.enabled_tools) - set(all_tools.keys())
            logger.warning(f"Some requested tools not found: {missing}")
        
        logger.info(f"Enabled tools: {[t.name for t in enabled]}")
        return enabled

    @staticmethod
    def _normalize_tool_name(raw_name: str) -> str:
        """Normalize malformed tool names emitted by the model.

        Keeps canonical names intact while stripping known transport artifacts.
        """
        if not raw_name:
            return raw_name

        name = str(raw_name).strip().strip("`").strip('"').strip("'")
        name = re.sub(r"\s+", "", name)

        # Common malformed suffix pattern: tool_name<|channel|>commentary
        if "<|" in name:
            name = name.split("<|", 1)[0]

        return name

    class _ToolNameCanonicalizationMiddleware(AgentMiddleware):
        """Canonicalize malformed tool names before ToolNode dispatch."""

        def __init__(self, tool_lookup: Dict[str, Any], normalize_fn):
            self._tool_lookup = tool_lookup
            self._normalize_fn = normalize_fn

        def _rewrite_request_if_needed(self, request):
            call = request.tool_call or {}
            raw_name = call.get("name", "")
            canonical_name = self._normalize_fn(raw_name)

            if not canonical_name or canonical_name == raw_name:
                return request

            tool_obj = self._tool_lookup.get(canonical_name)
            if tool_obj is None:
                return request

            updated_call = dict(call)
            updated_call["name"] = canonical_name
            updated_request = request.override(tool_call=updated_call)

            # ToolCallRequest.override does not include `tool`; set it explicitly
            # so ToolNode can execute the resolved canonical tool.
            object.__setattr__(updated_request, "tool", tool_obj)

            logger.warning(f"Normalized malformed tool name '{raw_name}' -> '{canonical_name}'")
            return updated_request

        def wrap_tool_call(self, request, handler):
            return handler(self._rewrite_request_if_needed(request))

        async def awrap_tool_call(self, request, handler):
            return await handler(self._rewrite_request_if_needed(request))

    def _record_llm_last_margin(self, bid_amount: int) -> None:
        """Store the margin % this bid represents over the total cost basis shown to the LLM.

        Uses the same basis (execution EMA + participation overhead) the prompt surfaced this
        cycle, so the recorded margin matches the "M% over total_cost_basis_micro_usdc" the model
        reasoned about. Silently skips if the basis is unavailable/non-positive. Only meaningful
        for LLM providers; harmless to compute otherwise.
        """
        ctx = self._tool_audit_context or {}
        exec_micro = ctx.get("estimated_execution_cost_micro")
        non_exec_micro = ctx.get("non_execution_cost_per_participation_micro") or 0
        if not isinstance(exec_micro, (int, float)):
            return
        basis = float(exec_micro) + float(non_exec_micro)
        if basis <= 0:
            return
        self._llm_last_margin_pct = round((float(bid_amount) / basis - 1.0) * 100.0, 1)

    def _record_llm_decision(
        self,
        auction_id: int,
        bid_amount: int,
        effort_tier: Optional[str],
    ) -> None:
        """Open a decision->outcome record for a freshly committed bid (R1 episodic memory).

        Captures only what is known at decision time: the action taken plus the reputation and
        balance it was taken from. Outcome fields stay None until AuctionEnded resolves them in
        _resolve_llm_outcome_history(). Recorded unconditionally (cheap) so toggling the prompt
        flag mid-run cannot produce a half-empty history.
        """
        ctx = self._tool_audit_context or {}
        exec_micro = ctx.get("estimated_execution_cost_micro")
        non_exec_micro = ctx.get("non_execution_cost_per_participation_micro") or 0
        margin_pct = None
        if isinstance(exec_micro, (int, float)):
            basis = float(exec_micro) + float(non_exec_micro)
            if basis > 0:
                margin_pct = round((float(bid_amount) / basis - 1.0) * 100.0, 1)

        try:
            auction_id = int(auction_id)
        except (TypeError, ValueError):
            return

        entry: Dict[str, Any] = {
            "auction_id": auction_id,
            "effort_tier": effort_tier,
            "bid_micro": int(bid_amount),
            "margin_pct": margin_pct,
            "reputation_before": ctx.get("agent_reputation"),
            "feedback_count_before": ctx.get("agent_feedback_count"),
            "balance_before_usd": ctx.get("current_balance_usd"),
            "outcome": "pending",
            "reputation_after": None,
            "balance_after_usd": None,
        }
        self._llm_outcome_history.append(entry)
        self._llm_outcome_by_auction[auction_id] = entry

    def _resolve_llm_outcome_history(self, state: AgentState) -> None:
        """Fill in outcomes for past decisions once the chain reveals them.

        Two-stage, because the two facts arrive at different times:
          1. AuctionEnded tells us won/lost (and, for a win, the realized revenue).
          2. The reputation effect of a win only appears after its feedback is posted, so
             reputation_after is left None until feedback_count actually increases.
        """
        current_rep = (state.get("agent_reputation") or {}).get("rating")
        current_feedback = (state.get("agent_reputation") or {}).get("feedback_count")
        balance_now = self.cost_tracker.get_net_balance()
        won_ids = {
            int(a.get("auction_id"))
            for a in (state.get("won_auctions") or [])
            if a.get("auction_id") is not None
        }

        # Stage 1: settle win/loss for auctions that ended this cycle.
        for auction_id in self._recently_ended_auction_ids:
            entry = self._llm_outcome_by_auction.get(int(auction_id))
            if entry is None or entry["outcome"] != "pending":
                continue
            if int(auction_id) in won_ids:
                entry["outcome"] = "won"
                # A loss has no feedback, so its reputation is settled immediately; a win's
                # reputation effect waits for feedback (stage 2).
            else:
                entry["outcome"] = "lost"
                entry["reputation_after"] = current_rep
                entry["balance_after_usd"] = balance_now
        self._recently_ended_auction_ids = []

        # Stage 2: a win's reputation effect lands only once its feedback is counted.
        if isinstance(current_feedback, int):
            for entry in self._llm_outcome_history:
                if entry["outcome"] != "won" or entry["reputation_after"] is not None:
                    continue
                before = entry.get("feedback_count_before")
                if isinstance(before, int) and current_feedback > before:
                    entry["reputation_after"] = current_rep
                    entry["balance_after_usd"] = balance_now

    def _build_llm_outcome_history_view(self) -> List[Dict[str, Any]]:
        """Render the history as compact prompt-ready rows (oldest first, optionally capped)."""
        limit = int(getattr(self.config, "llm_outcome_history_limit", 0) or 0)
        entries = self._llm_outcome_history
        if limit > 0:
            entries = entries[-limit:]

        rows: List[Dict[str, Any]] = []
        for e in entries:
            rep_before = e.get("reputation_before")
            rep_after = e.get("reputation_after")
            rep_delta = None
            if isinstance(rep_before, (int, float)) and isinstance(rep_after, (int, float)):
                rep_delta = round(float(rep_after) - float(rep_before), 2)
            bal_before = e.get("balance_before_usd")
            bal_after = e.get("balance_after_usd")
            bal_delta = None
            if isinstance(bal_before, (int, float)) and isinstance(bal_after, (int, float)):
                bal_delta = round(float(bal_after) - float(bal_before), 6)
            rows.append(
                {
                    "auction_id": e.get("auction_id"),
                    "tier": e.get("effort_tier"),
                    "margin_pct": e.get("margin_pct"),
                    "bid_micro": e.get("bid_micro"),
                    "outcome": e.get("outcome"),
                    "reputation_before": rep_before,
                    "reputation_after": rep_after,
                    "reputation_delta": rep_delta,
                    "balance_change_usd": bal_delta,
                }
            )
        return rows

    def _record_decision_rationale(self, text: str, auction_ids: List[int]) -> None:
        """Keep the model's own stated reasoning for its most recent decisions.

        Reflexion feeds the self-reflection model the trajectory *including thoughts*, not just
        actions and rewards, because credit assignment needs to know why an action was chosen —
        a bad outcome from sound reasoning warrants a different revision than one from a broken
        belief. The reasoning protocol already forces compact structured write-outs, so this
        stores those lines rather than whole prompts or states.
        """
        text = (text or "").strip()
        if not text or not auction_ids:
            return
        keep = [
            ln.strip() for ln in text.splitlines()
            if ln.strip().startswith(("Reputation check:", "Tier:", "Target margin:", "Check", "Standing vs field:"))
        ]
        summary = " ".join(keep) if keep else text
        self._llm_decision_rationales.append(
            {"auction_ids": list(auction_ids), "text": summary[:1200]}
        )
        # Only the newest few are ever rendered; drop the rest so memory cannot grow unbounded.
        excess = len(self._llm_decision_rationales) - self._RATIONALES_IN_REFLECTION
        if excess > 0:
            del self._llm_decision_rationales[:excess]

    def _render_outcome_table(self, rows: List[Dict[str, Any]]) -> str:
        """Compact numeric record of every past decision — cheap enough to show in full."""
        if not rows:
            return "  (no completed decisions yet)"
        lines = [
            "  auction | tier   | margin | bid_micro | outcome | reputation             | balance_change_usd",
            "  --------|--------|--------|-----------|---------|------------------------|-------------------",
        ]
        for r in rows:
            rep_b, rep_a = r.get("reputation_before"), r.get("reputation_after")
            rep_str = f"{rep_b} -> {rep_a if rep_a is not None else '(pending)'}"
            if r.get("reputation_delta") is not None:
                rep_str += f" ({r['reputation_delta']:+.2f})"
            margin = r.get("margin_pct")
            lines.append(
                f"  {str(r.get('auction_id')):>7} | {str(r.get('tier') or '-'):<6} "
                f"| {(f'{margin:.0f}%' if margin is not None else '-'):>6} "
                f"| {str(r.get('bid_micro')):>9} | {str(r.get('outcome')):<7} "
                f"| {rep_str:<22} | {r.get('balance_change_usd')}"
            )
        return "\n".join(lines)

    _MEMO_TEMPLATE = (
        "## What I believe about this market\n"
        "## My current strategy\n"
        "## What I am still unsure about"
    )

    def _build_reflection_prompt(self, state: AgentState) -> str:
        """Prompt for the R2 reflection call: revise the memo from the newest evidence."""
        record = self._render_outcome_table(self._build_llm_outcome_history_view())
        rationales = "\n".join(
            f"  [auctions {r['auction_ids']}] {r['text']}" for r in self._llm_decision_rationales
        ) or "  (none recorded yet)"

        rep = (state.get("agent_reputation") or {}).get("rating")
        feedback = (state.get("agent_reputation") or {}).get("feedback_count")
        current_memo = self._llm_strategy_memo or (
            "  (empty — this is your first revision; create it using the section headings below)"
        )

        return f"""You are provider agent {self.agent_id} competing in a repeated sealed-bid reverse auction market. Each auction is scored as ( rw * reputation + (100 - rw) * price_savings ) / 100, so both your bid price and your reputation decide who wins. Your reputation is the running mean of every quality rating you have received, so the effort tier you deliver moves it, and early ratings move it far more than later ones.

This is a review step, not a bidding step. You are revising the notes you carry into every future decision.

YOUR CURRENT NOTES:
{current_memo}

YOUR COMPLETE DECISION RECORD:
{record}

WHY YOU CHOSE YOUR MOST RECENT ACTIONS (your own words at the time):
{rationales}

WHERE YOU STAND NOW:
- your reputation: {rep} (feedback_count: {feedback})
- competitor reputations: {state.get('competitors_reputation')}
- wins: {state.get('wins_count')} of {state.get('bids_placed_count')} bids placed
- current net balance (USD): {state.get('current_balance')}

Revise your notes and output the full revised version. Work through this before writing:
1. Compare what you EXPECTED when you made those recent decisions against what actually happened. Name anything that surprised you — a surprise means your understanding of the market is wrong, and that is the most valuable thing you can learn here.
2. Decide what to KEEP (still supported by the record), what to REWRITE (contradicted by the record), and what to ADD (new). Do not rewrite things the evidence still supports — churning your notes every round is itself a failure. Do not delete a hard-won conclusion just because it is inconvenient this round.
3. Distinguish a belief that was wrong from a choice that was merely unlucky. If a sound decision produced a bad outcome once, that is noise, not a lesson.

Output EXACTLY these three sections, nothing else:

## What I believe about this market
How this market actually works, learned from your record — the mechanics you have verified, not restatements of the rules you were given. Include quantities where the record supports them.

## My current strategy
The policy you will follow on the next auction, specific enough to act on: which effort tier, what margin, and under what conditions you deviate. This is a commitment — you should follow it next round unless the state genuinely contradicts it.

## What I am still unsure about
What you do not yet know, and what you could try next to find out."""

    async def _maybe_generate_reflection(self, state: AgentState) -> None:
        """Revise the strategy memo before deciding. No-op unless llm_reflection_enabled.

        Runs once per bidding cycle, but only after at least one decision exists to learn from.
        Cost is charged to the agent as ordinary (non-execution) reasoning overhead. Failures are
        swallowed: losing a revision must never block the bidding cycle, and the previous memo
        stays in force.
        """
        if not getattr(self.config, "llm_reflection_enabled", False):
            return
        if not self._llm_outcome_history:
            return

        try:
            callback = LLMCostCallback(
                cost_tracker=self.cost_tracker,
                model=self.llm_model,
                input_price_per_1k=self.config.reasoning_llm_input_price_per_1k,
                output_price_per_1k=self.config.reasoning_llm_output_price_per_1k,
                scope="reflection",
            )
            llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                callbacks=[callback],
                # Sampling kwargs merged with OpenRouter's usage accounting, so
                # the cost callback can use what was actually billed.
                **self._llm_kwargs_with_usage(),
            )
            result = await llm.ainvoke(self._build_reflection_prompt(state))
            text = (getattr(result, "content", "") or "").strip()
            if not text:
                logger.warning("🧠 Reflection returned empty memo; keeping previous version")
                return

            self._llm_strategy_memo = text
            self._llm_memo_version += 1
            logger.info(
                "🧠 STRATEGY MEMO v%d (after %d decisions, cost $%.5f):\n%s",
                self._llm_memo_version,
                len(self._llm_outcome_history),
                callback.total_cost,
                text,
            )
        except Exception as e:
            logger.warning(f"Reflection failed (keeping previous memo): {e}")

    def _build_tool_audit_context(self, state: AgentState) -> Dict[str, Any]:
        """Build per-cycle state snapshot to include in each tool-call audit event."""
        auctions = state.get("eligible_active_auctions", []) or []
        agent_reputation = (state.get("agent_reputation") or {}).get("rating")
        competitor_reps = {
            int(comp.get("agent_id")): comp.get("rating")
            for comp in (state.get("competitors_reputation") or [])
            if comp.get("agent_id") is not None
        }
        auction_map = {}
        for a in auctions:
            auction_id = a.get("auction_id")
            if auction_id is None:
                continue
            winning_agent_id = a.get("winning_agent_id")
            current_winner_reputation = None
            if isinstance(winning_agent_id, int) and winning_agent_id > 0:
                if winning_agent_id == self.agent_id:
                    current_winner_reputation = agent_reputation
                else:
                    current_winner_reputation = competitor_reps.get(winning_agent_id)
            auction_map[str(auction_id)] = {
                "max_price": a.get("max_price"),
                "winning_bid": a.get("winning_bid"),
                "winning_agent_id": winning_agent_id,
                "current_winner_reputation": current_winner_reputation,
                "reputation_weight": a.get("reputation_weight"),
                "time_remaining": a.get("time_remaining"),
            }

        estimated_cost_usd = state.get("estimated_service_cost")
        estimated_cost_micro = None
        if isinstance(estimated_cost_usd, (int, float)):
            estimated_cost_micro = int(float(estimated_cost_usd) * 1e6)

        execution_cost_usd = state.get("estimated_execution_cost")
        execution_cost_micro = None
        if isinstance(execution_cost_usd, (int, float)):
            execution_cost_micro = int(float(execution_cost_usd) * 1e6)

        non_execution_cost_usd = state.get("non_execution_cost_per_participation")
        non_execution_cost_micro = None
        if isinstance(non_execution_cost_usd, (int, float)):
            non_execution_cost_micro = int(float(non_execution_cost_usd) * 1e6)

        unrecovered_overhead_usd = state.get("unrecovered_overhead_balance")
        unrecovered_overhead_micro = None
        if isinstance(unrecovered_overhead_usd, (int, float)):
            unrecovered_overhead_micro = int(float(unrecovered_overhead_usd) * 1e6)

        execution_history = state.get("past_execution_costs") or []
        current_balance = state.get("current_balance")

        return {
            "agent_id": self.agent_id,
            "agent_reputation": (state.get("agent_reputation") or {}).get("rating"),
            "agent_feedback_count": (state.get("agent_reputation") or {}).get("feedback_count"),
            "estimated_service_cost_usd": estimated_cost_usd,
            "estimated_service_cost_micro": estimated_cost_micro,
            "estimated_execution_cost_usd": execution_cost_usd,
            "estimated_execution_cost_micro": execution_cost_micro,
            "non_execution_cost_per_participation_usd": non_execution_cost_usd,
            "non_execution_cost_per_participation_micro": non_execution_cost_micro,
            "unrecovered_overhead_balance_usd": unrecovered_overhead_usd,
            "unrecovered_overhead_balance_micro": unrecovered_overhead_micro,
            "past_execution_costs": execution_history,
            "past_execution_costs_count": len(execution_history),
            "current_balance_usd": current_balance,
            "auctions": auction_map,
            "auction_count": len(auctions),
        }

    def _audit_tool_call(
        self,
        tool_name: str,
        llm_args: Dict[str, Any],
        normalized_args: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a structured tool-call event with state snapshot and raw LLM args."""
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent_id": self.agent_id,
            "tool": tool_name,
            "llm_args": llm_args,
            "current_state": self._tool_audit_context or {},
        }
        logger.info(f"TOOL_AUDIT {json.dumps(event, default=str)}")
    
    def _get_default_system_prompt(self, auctions_context: str) -> str:
        """
        Generate default system prompt for the ReAct agent.
        
        This is used as fallback when no custom system_prompt is provided.
        In experiments, custom prompts should be injected via config.
        
        Args:
            auctions_context: JSON string of eligible auctions
            
        Returns:
            Default system prompt string
        """
        # Dynamically generate tool descriptions based on enabled tools
        tool_descriptions = {
            "place_bid": "place_bid(auction_id, bid_amount): Seal your bid commitment on-chain. ONE chance per auction ever — cannot be changed or cancelled. Hidden until automatic reveal; no immediate win/loss feedback.",
        }

        enabled_tool_descs = [f"- {tool_descriptions[name]}" for name in self.enabled_tools if name in tool_descriptions]
        tools_section = "\n".join(enabled_tool_descs)

        return f"""You are a bidding agent (ID: {self.agent_id}) for a decentralized AI service marketplace.

Available tools:
{tools_section}

BIDDING GUIDELINES:
1. Your estimated execution cost is provided in the auction context; profit = bid − cost.
2. Service requirements are already embedded in auction details.
3. Bids are sealed (commit-reveal): place_bid only commits a hash now; reveal and any win/loss determination happen automatically later, outside your control.
4. This contract ranks revealed bids by weighted score where HIGHER score wins (lower bids improve score via the bid component, reputation also improves score).
5. Consider time remaining in the commit phase - it is your only window to submit a bid for this auction.
6. You can commit to multiple auctions if profitable; if none are profitable, don't bid. You can commit to a given auction_id only once, ever.
7. If you end a cycle without placing a bid, explain the blocker clearly. The runtime will handle the separate continuation decision.

Current eligible auctions:
{auctions_context}

Analyze these auctions and decide which to bid on."""
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for blockchain operations."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self._router_node)
        workflow.add_node("generate_feedback_auth", self._generate_feedback_auth_node)
        workflow.add_node("call_complete_service", self._call_complete_service_node)
        workflow.add_node("reveal_commitments", self._reveal_commitments_node)
        workflow.add_node("gather_state", self._gather_state_node)
        workflow.add_node("reasoning", self._reasoning_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_action,
            {"complete_service": "generate_feedback_auth", "monitor": "reveal_commitments"}
        )

        workflow.add_edge("generate_feedback_auth", "call_complete_service")
        workflow.add_edge("call_complete_service", END)
        # Minimal baseline: the bidding path ends after reasoning. Abstention is implicit
        # (a cycle that ends without place_bid); the separate continuation node is removed.
        # Any locally pending sealed-bid commitments are revealed (if their auction has
        # entered the Reveal phase) before gathering state for this cycle's bidding decision.
        workflow.add_edge("reveal_commitments", "gather_state")
        workflow.add_edge("gather_state", "reasoning")
        workflow.add_edge("reasoning", END)
        
        return workflow.compile()
    
    async def initialize(self) -> bool:
        """Initialize the blockchain handler by loading contracts."""
        try:
            await self.client._initialize()
            
            if not self.config.reverse_auction_address:
                logger.warning("ReverseAuction address not configured")
                return False
            
            self.reverse_auction_contract = await self.client.load_contract(
                name="ReverseAuction",
                address=self.config.reverse_auction_address,
                abi=get_reverse_auction_abi()
            )
            
            if self.config.identity_registry_address:
                self.identity_registry_contract = await self.client.load_contract(
                    name="IdentityRegistry",
                    address=self.config.identity_registry_address,
                    abi=get_identity_registry_abi()
                )
            
            if self.config.reputation_registry_address:
                self.reputation_registry_contract = await self.client.load_contract(
                    name="ReputationRegistry",
                    address=self.config.reputation_registry_address,
                    abi=get_reputation_registry_abi()
                )
            
            self.contracts_loaded = True
            logger.info("✅ BlockchainHandler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BlockchainHandler: {e}", exc_info=True)
            return False
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router node - entry point that examines action field."""
        logger.info(f"🔀 Router: action={state['action']}")
        return state
    
    async def _fetch_reputation(self, agent_id: int) -> Dict[str, Any]:
        """Fetch reputation for an agent from the ERC-8004 ReputationRegistry.

        getSummary() reverts outright when given an empty clientAddresses array, so
        clients must be enumerated via getClients() first. tag1/tag2 are string filters
        (empty string = no filter), not bytes32. The returned average is a fixed-point
        (value, decimals) pair, preserved here as a float rather than truncated to int.

        Args:
            agent_id: Agent ID to fetch reputation for

        Returns:
            Dictionary with rating (float) and feedback_count
        """
        try:
            if not self.reputation_registry_contract:
                return {"rating": self.initial_reputation_default, "feedback_count": 0}

            clients = await self.client.call_contract_method(
                "ReputationRegistry", "getClients", agent_id
            )
            if not clients:
                return {"rating": self.initial_reputation_default, "feedback_count": 0}

            count, value, decimals = await self.client.call_contract_method(
                "ReputationRegistry", "getSummary", agent_id, clients, "", ""
            )
            feedback_count = int(count)
            if feedback_count == 0 or value < 0:
                return {"rating": self.initial_reputation_default, "feedback_count": 0}

            average_score = reputation_to_float(int(value), int(decimals))
            return {"rating": average_score, "feedback_count": feedback_count}
        except Exception as e:
            logger.warning(f"Error fetching reputation for agent {agent_id}: {e}")
            return {"rating": self.initial_reputation_default, "feedback_count": 0}

    def _has_local_commitment(self, auction_id: int) -> bool:
        """Cheap, local-only check for an existing sealed-bid commitment. Use for
        early/cheap pre-filtering; use _has_committed() as the authoritative gate
        immediately before sending a transaction."""
        return self._commitment_store.get_commitment(int(auction_id)) is not None

    async def _has_committed(self, auction_id: int) -> bool:
        """Authoritative check for whether we've already committed to this auction
        (local store first, then an on-chain fallback in case the local record was
        lost but a commitment still exists from a prior process)."""
        if self._has_local_commitment(auction_id):
            return True
        try:
            on_chain_commitment = await self.client.call_contract_method(
                "ReverseAuction", "getCommitment", auction_id, self.agent_id
            )
            return on_chain_commitment[1] != b"\x00" * 32
        except Exception:
            return False

    def _classify_commit_error(self, error_msg: str) -> Dict[str, Any]:
        """Map a commitBid revert reason/selector to a structured result."""
        if "0xbfec5558" in error_msg or "AlreadyCommitted" in error_msg:
            return {
                "success": False, "error": "AlreadyCommitted", "error_code": "0xbfec5558",
                "explanation": "You already have a sealed commitment for this auction.",
                "retry_recommended": False,
            }
        if "0xabfb7dc3" in error_msg or "NotInCommitPhase" in error_msg:
            return {
                "success": False, "error": "NotInCommitPhase", "error_code": "0xabfb7dc3",
                "explanation": "The commit phase for this auction has ended.",
                "retry_recommended": False,
            }
        if "0xce0ca137" in error_msg or "AgentNotEligible" in error_msg:
            return {
                "success": False, "error": "AgentNotEligible", "error_code": "0xce0ca137",
                "explanation": "Agent is not eligible for this auction.",
                "retry_recommended": False,
            }
        if "0x69b8d0fe" in error_msg or "AuctionNotActive" in error_msg:
            return {
                "success": False, "error": "AuctionNotActive", "error_code": "0x69b8d0fe",
                "explanation": "Auction is no longer active.",
                "retry_recommended": False,
            }
        if "0x0e28b8be" in error_msg or "AuctionNotFound" in error_msg:
            return {
                "success": False, "error": "AuctionNotFound", "error_code": "0x0e28b8be",
                "explanation": "Auction does not exist.",
                "retry_recommended": False,
            }
        if "0x390772fc" in error_msg or "NotAgentOwner" in error_msg:
            return {
                "success": False, "error": "NotAgentOwner", "error_code": "0x390772fc",
                "explanation": "This account does not own the agent ID used to commit.",
                "retry_recommended": False,
            }
        if "0xc9b80cd4" in error_msg or "BidTooHigh" in error_msg:
            return {
                "success": False, "error": "BidTooHigh", "error_code": "0xc9b80cd4",
                "explanation": "Your bid amount exceeds auction max price.",
                "retry_recommended": True,
            }
        return {"success": False, "error": error_msg, "retry_recommended": False}

    def _bid_jitter_seconds(self, auctions: List[Dict[str, Any]]) -> float:
        """Pre-reasoning delay, capped to a fraction of the shortest commit window.

        On the long phases this repo used to run (60-80s) the cap is the usual
        3s. On S7's shortened phases it shrinks with the window, so the
        desynchronisation stays useful without consuming the time the bid needs.
        """
        cap = self._BID_JITTER_MAX_SECONDS
        windows = [int(a.get("duration") or 0) for a in auctions if a.get("duration")]
        if windows:
            cap = min(cap, max(0.0, self._JITTER_WINDOW_FRACTION * min(windows)))
        return self._rng.uniform(0, cap)

    def _reveal_jitter_seconds(self, reveal_windows: List[int]) -> float:
        """Pre-reveal delay, capped to a fraction of the shortest reveal window.

        Mirrors :meth:`_bid_jitter_seconds`. The cap matters more here: the
        reveal window is short (6s in S7) and a commitment not revealed inside it
        is dropped, so the delay must stay a small fraction of it.

        Drawn from :attr:`_reveal_rng` rather than ``self._rng`` so that adding
        this does not shift the bidding draw sequence — the commit jitter for a
        given (run_seed, agent_id) is unchanged by this feature existing.
        """
        cap = self._REVEAL_JITTER_MAX_SECONDS
        windows = [int(w) for w in reveal_windows if w]
        if windows:
            cap = min(cap, max(0.0, self._JITTER_WINDOW_FRACTION * min(windows)))
        return self._reveal_rng.uniform(0, cap)

    async def _chain_timestamp(self) -> Optional[int]:
        """Current block timestamp, or None if the chain cannot be read.

        Block time, not wall time: on a forked or snapshot-loaded Anvil the two
        are unrelated.
        """
        try:
            block = await self.client.get_block("latest")
            return int(block["timestamp"])
        except Exception as e:  # a stale value is better than no decision
            logger.debug(f"Could not read chain timestamp: {e}")
            return None

    def _fresh_time_remaining(self, auction: Dict[str, Any], decision_time: Optional[int]) -> int:
        """Seconds left in this auction's commit phase, as of *now*.

        Takes the **more pessimistic** of the value captured during state
        gathering and the one recomputed against the current block clock. On a
        real chain the recomputed value always wins, because time only moves
        forward and the captured value predates the jitter sleep — which is the
        staleness this exists to fix. Taking the minimum rather than replacing
        outright means a caller that supplies a deliberately short window (the
        equivalence harness does) still gets the window it asked for: for a
        deadline gate, the safe reading of two numbers is the smaller one.
        """
        captured = int(auction.get("time_remaining", 0) or 0)
        end_time = auction.get("end_time")
        if decision_time is None or not end_time:
            return captured
        recomputed = max(0, int(end_time) - int(decision_time))
        return min(captured, recomputed) if captured else recomputed

    async def _commit_bid(
        self,
        auction_id: int,
        bid_amount: int,
        *,
        effort_tier: Optional[str] = None,
        mode: str = "heuristic",
        commit_deadline: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Seal and submit a bid commitment for an auction.

        Generates a fresh nonce, persists it locally BEFORE broadcasting the
        transaction (so a crash right after broadcast doesn't forfeit the ability
        to reveal later), then sends commitBid. The nonce/bid amount are only
        recoverable from local storage - if this process loses them with no
        successful save, the commitment becomes permanently unrevealable.
        """
        auction_id = int(auction_id)
        bid_amount = int(bid_amount)
        self._record_bid_attempt(
            auction_id=auction_id, bid_amount=bid_amount, success=False,
            error="attempt_started", mode=mode,
        )

        nonce_bytes = secrets.token_bytes(32)
        nonce_hex = nonce_bytes.hex()
        sender = self.client.account.address

        tx_broadcast = False
        try:
            commit_hash = await self.client.call_contract_method(
                "ReverseAuction", "computeCommitHash",
                auction_id, self.agent_id, bid_amount, nonce_bytes, sender,
            )

            # Persist BEFORE sending the tx: an orphaned local record (tx never lands)
            # self-heals via the on-chain getCommitment check; a lost nonce after a
            # successful commit does not.
            self._commitment_store.save_commitment(
                auction_id,
                bid_amount=bid_amount,
                nonce_hex=nonce_hex,
                sender=sender,
                effort_tier=effort_tier,
            )

            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction", "commitBid", auction_id, self.agent_id, commit_hash
            )
            tx_broadcast = True
            tx_hash = await self.client.send_transaction(
                "ReverseAuction", "commitBid", auction_id, self.agent_id, commit_hash,
                gas_limit=estimated_gas + 50000,
            )
            receipt = await self.client.wait_for_transaction(tx_hash)

            if receipt["status"] == 1:
                self.cost_tracker.add_gas_cost(
                    gas_used=receipt.get("gasUsed", 0),
                    gas_price_wei=receipt.get("effectiveGasPrice", 0),
                    context="commit_bid",
                )
                logger.info(f"✅ Bid committed successfully for auction {auction_id}: {tx_hash}")
                await self._check_commit_landed_in_time(auction_id, receipt, commit_deadline)
                result = {
                    "success": True,
                    "tx_hash": tx_hash,
                    "auction_id": auction_id,
                    "bid_amount": bid_amount,
                    "block_number": receipt["blockNumber"],
                }
                if effort_tier is not None:
                    result["effort_tier"] = effort_tier
                self._bids_placed.append(result)
                self._own_current_bid_by_auction[auction_id] = bid_amount
                if effort_tier is not None:
                    self._auction_effort_tier_by_auction[auction_id] = effort_tier
                self._record_bid_attempt(
                    auction_id=auction_id, bid_amount=bid_amount, success=True, mode=mode,
                )
                return result

            logger.warning(f"❌ Commit transaction failed for auction {auction_id}")
            self._commitment_store.remove_commitment(auction_id)
            result = {"success": False, "error": "TransactionFailed", "auction_id": auction_id}
            self._record_bid_attempt(
                auction_id=auction_id, bid_amount=bid_amount, success=False,
                error=result["error"], mode=mode,
            )
            return result

        except Exception as e:
            logger.error(f"Error committing bid for auction {auction_id}: {e}")
            error_msg = str(e)

            # Only safe to drop the local record if we know the tx was never broadcast;
            # if broadcast happened and only the wait/confirmation step threw, the tx may
            # still land - keep the record and let reveal-time getCommitment self-heal.
            if not tx_broadcast:
                self._commitment_store.remove_commitment(auction_id)

            result = self._classify_commit_error(error_msg)
            result["auction_id"] = auction_id
            self._record_bid_attempt(
                auction_id=auction_id, bid_amount=bid_amount, success=False,
                error=result.get("error"), error_code=result.get("error_code"), mode=mode,
            )
            return result

    async def _check_commit_landed_in_time(
        self, auction_id: int, receipt: Dict[str, Any], commit_deadline: Optional[int]
    ) -> None:
        """Record whether this commit was mined before its phase closed (gate G-E).

        §A.6.5 shortens the phases and requires them "verified timeout-free". A
        commit that lands late does not raise — it is simply a bid that never
        counted — so without counting them a run with shortened phases looks
        like a market where nobody wanted to bid. This is the measurement that
        distinguishes the two.
        """
        if not commit_deadline:
            return
        try:
            end_time = int(commit_deadline)
            block = await self.client.get_block(receipt["blockNumber"])
            mined_at = int(block["timestamp"])
            if mined_at > end_time:
                self.late_commits += 1
                logger.warning(
                    "⏰ LATE COMMIT: auction %s commit mined %ss after the phase closed "
                    "(gate G-E; %d late so far). Phase durations are too short for this "
                    "chain, or bid_freshness_min_seconds is too low.",
                    auction_id, mined_at - end_time, self.late_commits,
                )
        except Exception as e:
            logger.debug(f"Could not verify commit timing for auction {auction_id}: {e}")

    async def _reveal_bid(self, auction_id: int) -> Dict[str, Any]:
        """Reveal a previously committed bid using the locally persisted bid amount + nonce."""
        auction_id = int(auction_id)
        commitment = self._commitment_store.get_commitment(auction_id)
        if commitment is None:
            return {"success": False, "auction_id": auction_id, "error": "NoLocalCommitment"}

        bid_amount = int(commitment["bid_amount"])
        nonce_bytes = bytes.fromhex(commitment["nonce_hex"])

        try:
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction", "revealBid", auction_id, self.agent_id, bid_amount, nonce_bytes
            )
            tx_hash = await self.client.send_transaction(
                "ReverseAuction", "revealBid", auction_id, self.agent_id, bid_amount, nonce_bytes,
                gas_limit=estimated_gas + 50000,
            )
            receipt = await self.client.wait_for_transaction(tx_hash)

            if receipt["status"] == 1:
                self.cost_tracker.add_gas_cost(
                    gas_used=receipt.get("gasUsed", 0),
                    gas_price_wei=receipt.get("effectiveGasPrice", 0),
                    context="reveal_bid",
                )
                logger.info(f"✅ Bid revealed successfully for auction {auction_id}: {tx_hash}")
                self._commitment_store.mark_revealed(auction_id)
                return {
                    "success": True,
                    "tx_hash": tx_hash,
                    "auction_id": auction_id,
                    "bid_amount": bid_amount,
                    "block_number": receipt["blockNumber"],
                }

            logger.warning(f"❌ Reveal transaction failed for auction {auction_id}")
            return {"success": False, "auction_id": auction_id, "error": "TransactionFailed"}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error revealing bid for auction {auction_id}: {e}")
            if "0xa89ac151" in error_msg or "AlreadyRevealed" in error_msg:
                self._commitment_store.mark_revealed(auction_id)
                return {"success": True, "auction_id": auction_id, "already_revealed": True}
            if "0x74683d71" in error_msg or "NotInRevealPhase" in error_msg:
                return {"success": False, "auction_id": auction_id, "error": "NotInRevealPhase"}
            if "0xc6c155d8" in error_msg or "NoCommitmentFound" in error_msg:
                self._commitment_store.remove_commitment(auction_id)
                return {"success": False, "auction_id": auction_id, "error": "NoCommitmentFound"}
            return {"success": False, "auction_id": auction_id, "error": error_msg}

    async def _reveal_commitments_node(self, state: AgentState) -> AgentState:
        """Reveal any locally pending commitments whose auction has entered the Reveal phase.

        A seeded delay precedes the first reveal of each cycle. Unlike the
        commit, which each agent paces by its own jittered clock, the reveal is
        triggered by a *shared* on-chain deadline: every agent polling for the
        phase flip fires the instant it sees it, so the reveals land in the same
        block and their order is decided by mempool arrival. See
        :attr:`_REVEAL_JITTER_MAX_SECONDS`.
        """
        pending_ids = self._commitment_store.pending_auction_ids()
        reveals: List[Dict[str, Any]] = []
        jittered = False

        for auction_id in pending_ids:
            try:
                phase = await self.client.call_contract_method(
                    "ReverseAuction", "getAuctionPhase", auction_id
                )
                phase = int(phase)
            except Exception as e:
                logger.warning(f"Could not fetch auction phase for {auction_id}: {e}")
                continue

            if phase == 0:  # Commit - not yet time to reveal
                continue

            # Once per cycle, and only when something is actually revealable, so
            # an agent with nothing to reveal never pays the delay and the cap is
            # measured against this auction's own reveal window.
            if phase == 1 and not jittered:
                jittered = True
                window = 0
                try:
                    details = await self.client.call_contract_method(
                        "ReverseAuction", "getAuctionDetails", auction_id
                    )
                    window = int(details[self._AUCTION_REVEAL_DURATION])
                except Exception as e:
                    logger.debug(f"Could not read reveal duration for {auction_id}: {e}")
                delay = self._reveal_jitter_seconds([window] if window else [])
                if delay > 0:
                    logger.info(
                        "⏳ Reveal jitter: sleeping %.2fs before revealing %d commitment(s)",
                        delay, len(pending_ids),
                    )
                    await asyncio.sleep(delay)

            if phase == 2:  # Ended - reveal window passed without us revealing
                logger.warning(
                    f"⏰ Reveal window for auction {auction_id} ended before we revealed; "
                    f"dropping local commitment"
                )
                self._commitment_store.remove_commitment(auction_id)
                continue

            # phase == 1 (Reveal)
            try:
                on_chain_commitment = await self.client.call_contract_method(
                    "ReverseAuction", "getCommitment", auction_id, self.agent_id
                )
                commit_hash = on_chain_commitment[1]
                already_revealed = bool(on_chain_commitment[2])
            except Exception as e:
                logger.warning(f"Could not fetch on-chain commitment for auction {auction_id}: {e}")
                continue

            if commit_hash == b"\x00" * 32:
                logger.warning(
                    f"No on-chain commitment found for auction {auction_id}; "
                    f"local record orphaned, dropping"
                )
                self._commitment_store.remove_commitment(auction_id)
                continue

            if already_revealed:
                self._commitment_store.mark_revealed(auction_id)
                continue

            # S7 §A.6.5: the reveal window is short, and a reveal that lands
            # after it closes is worse than one never attempted — the bid is
            # already committed and the escrow already locked, so a late reveal
            # forfeits the auction *and* pays gas for the privilege. Skip when
            # there is not enough of the window left to be mined in it.
            reveal_remaining = await self._reveal_seconds_remaining(auction_id)
            if reveal_remaining is not None and reveal_remaining <= self._bid_freshness_min_seconds:
                # Count the *commitment*, not the poll. The loop revisits a
                # closing window every cycle, so incrementing per poll reported
                # 73 for 24 actually-forfeited commitments in an earlier run.
                if auction_id not in self._forfeited_commitments:
                    self._forfeited_commitments.add(auction_id)
                    self.missed_reveals += 1
                logger.warning(
                    "⏰ Skipping reveal for auction %s: only %ss of the reveal window left "
                    "(gate %.1fs). The commitment is forfeit; reveal_duration is too short "
                    "for this poll interval (gate G-E).",
                    auction_id, reveal_remaining, self._bid_freshness_min_seconds,
                )
                continue

            result = await self._reveal_bid(auction_id)
            reveals.append(result)

        state["reveals"] = reveals
        return state

    async def _reveal_seconds_remaining(self, auction_id: int) -> Optional[int]:
        """Seconds left in this auction's reveal window, or None if unreadable."""
        try:
            auction = await self.client.call_contract_method(
                "ReverseAuction", "getAuctionDetails", auction_id
            )
            # Auction struct order (ReverseAuction.sol): 4 commitDuration,
            # 5 startTime, 13 revealDuration. The reveal window opens when the
            # commit phase ends.
            duration, start_time, reveal_duration = int(auction[4]), int(auction[5]), int(auction[13])
            block = await self.client.get_block("latest")
            return max(0, start_time + duration + reveal_duration - int(block["timestamp"]))
        except Exception as e:
            logger.debug(f"Could not read reveal window for auction {auction_id}: {e}")
            return None

    def _resolve_bid_effort_tier(self, selected_effort_tier: Optional[str]) -> Optional[str]:
        """Resolve bid effort tier when two-way coupling is active."""
        if self.coupling_mode != "two_way":
            return None

        return self.cost_tracker.normalize_effort_tier(
            selected_effort_tier,
            default=self._default_two_way_effort_tier,
        )

    # ------------------------------------------------------------------
    # Adapters onto the pure policies in policies.py (S1 §1). These translate
    # the agent's state dict / config into the dataclasses the policies take;
    # the pricing arithmetic itself lives in policies.py so the offline
    # simulator runs the identical code.
    # ------------------------------------------------------------------

    def _observed_execution_cost_usd(self, state: AgentState) -> float:
        """Observable execution-cost estimate (USD), falling back to the configured prior."""
        observed_cost_usd = state.get("estimated_execution_cost")
        if observed_cost_usd is None:
            observed_cost_usd = state.get("estimated_service_cost")
        if observed_cost_usd is None:
            observed_cost_usd = self.cost_tracker.get_initial_execution_cost_for_effort(
                getattr(self.config, "default_effort_tier", "medium")
            )
        return max(0.0, float(observed_cost_usd))

    def _policy_state_for(self, strategy: str) -> "policies.PolicyState":
        """Controller memory (margin, prev_wins, bids_placed) for one strategy.

        Persists for the agent's lifetime; only the active strategy is ever used.
        """
        return self._policy_states.setdefault(strategy, policies.PolicyState())

    def _policy_params(self, min_margin: float, max_margin: float) -> "policies.PolicyParams":
        return policies.PolicyParams(
            min_margin=min_margin,
            max_margin=max_margin,
            initial_margin_mode=getattr(self.config, "heuristic_initial_margin_mode", "max"),
            initial_reputation_default=self.initial_reputation_default,
            quality_prior_by_tier=getattr(self.config, "initial_quality_rating_by_effort", {}) or {},
            # S7 §C = S6 rung O2c. Every arm in a cell gets the identical rule,
            # judged against the prices it has itself won at, so the factor
            # measures *declining* and not who happens to be able to decline.
            # `decline_price_quantile` is left at PolicyParams' pinned 1.0 (O2c);
            # it is not a config knob.
            action_space=str(getattr(self.config, "action_space", "price_only") or "price_only"),
        )

    def _policy_auction_view(self, auction: Dict[str, Any], state: AgentState) -> "policies.AuctionView":
        rep_info = state.get("agent_reputation") or {}
        return policies.AuctionView(
            max_price=int(auction.get("max_price", 0) or 0),
            reputation_weight=_auction_rw(auction),
            our_reputation=float(rep_info.get("rating", self.initial_reputation_default)),
            our_feedback_count=max(0, int(rep_info.get("feedback_count", 0) or 0)),
            competitor_reputations=tuple(
                float(rep.get("rating") or self.initial_reputation_default)
                for rep in (state.get("competitors_reputation") or [])
                if int(rep.get("agent_id", 0) or 0) != int(self.agent_id)
            ),
        )

    def _policy_economics(
        self, state: AgentState, observed_cost_usd: float, estimated_cost: int
    ) -> "policies.AgentEconomics":
        participation_overhead_usd = max(
            0.0, float(state.get("non_execution_cost_per_participation") or 0.0)
        )
        unrecovered_usd = max(0.0, float(state.get("unrecovered_overhead_balance") or 0.0))

        effort_execution_costs = state.get("effort_level_execution_costs") or {}
        tier_cost_micros: Dict[str, int] = {}
        for tier in policies.EFFORT_TIERS:
            raw_cost = effort_execution_costs.get(tier, observed_cost_usd)
            try:
                tier_cost_micros[tier] = int(round(max(0.0, float(raw_cost)) * 1e6))
            except (TypeError, ValueError):
                tier_cost_micros[tier] = int(round(observed_cost_usd * 1e6))

        # S7 §B.2/§B.3: the estimator supplies the shape, history the level.
        # `multipliers` is None for H, so its economics are byte-identical to
        # what they were before any estimator existed.
        multipliers = self._task_cost_multipliers(state, tier_cost_micros)
        if multipliers:
            tier_cost_micros = {
                tier: max(1, int(round(tier_cost_micros[tier] * multipliers.get(tier, 1.0))))
                for tier in tier_cost_micros
            }
            estimated_cost = tier_cost_micros.get(
                self.cost_tracker.normalize_effort_tier(
                    getattr(self.config, "default_effort_tier", "medium")
                ),
                estimated_cost,
            )

        return policies.AgentEconomics(
            execution_cost_micro=estimated_cost,
            execution_cost_by_tier_micro=tier_cost_micros,
            task_cost_multiplier=multipliers,
            quality_by_tier=state.get("effort_level_quality_stats") or {},
            overhead_per_participation_micro=int(round(participation_overhead_usd * 1e6)),
            unrecovered_overhead_micro=int(round(unrecovered_usd * 1e6)),
            # The win signal every outcome-driven controller uses: one entry per
            # completed delivery observed so far.
            wins=len(state.get("past_execution_costs") or []),
            # S7 §C: the only extra input O2c's decline rule needs. Insertion
            # order is win order, which is what S6's list preserves.
            own_winning_bids_micro=tuple(self._own_winning_bids_micro.values()),
        )

    async def _prepare_task_multipliers(self, state: AgentState, auction: Dict[str, Any]) -> None:
        """Run the configured estimator for this auction, before the policy decides.

        Both arms land here so the economics adapter stays synchronous and pure
        (the simulator builds it with no I/O). Arm R computes in microseconds;
        arm X awaits an LLM call, which is why this is the async site.
        """
        state.pop("_task_cost_multipliers", None)
        estimator = getattr(self, "_cost_estimator", None)
        if estimator is None:
            return

        state["_estimator_auction"] = auction
        requirement_id = self._current_requirement_id(state)
        if requirement_id is None:
            return

        levels = state.get("effort_level_execution_costs") or {}
        quality = state.get("effort_level_quality_stats") or {}
        if isinstance(estimator, cost_estimators.DecomposedCostEstimator):
            # The prompt states the agent's own averages as the level the
            # multiplier is relative to. Quality comes from its own feedback.
            estimator.levels = {
                tier: {
                    "cost": float(levels.get(tier, 0.0) or 0.0),
                    "quality": float(
                        (quality.get(tier) or {}).get("average_rating")
                        or self.initial_reputation_default
                    ),
                }
                for tier in policies.EFFORT_TIERS
            }

        default_tier = self.cost_tracker.normalize_effort_tier(
            getattr(self.config, "default_effort_tier", "medium")
        )
        multipliers: Dict[str, float] = {}
        for tier in policies.EFFORT_TIERS:
            level_usd = float(levels.get(tier, 0.0) or 0.0)
            # What is left of the commit window, minus what the bid still needs
            # after the estimate returns. Without this the transport's own
            # 180s x 3 retry budget applies and a single slow call costs the
            # agent its auction *and* every auction that opens while it waits.
            outcome = estimator.multiplier_for(
                requirement_id, tier, level_usd,
                deadline_seconds=self._estimator_deadline_seconds(auction),
            )
            multiplier, diagnostics = (
                await outcome if inspect.isawaitable(outcome) else outcome
            )
            multipliers[tier] = multiplier

            # §B.3: the estimator's tokens are charged to the agent, so the
            # hybrid bids off a *higher* basis and has to earn the call back
            # inside the mechanism. `add_llm_cost` routes to the reasoning
            # bucket while no service is executing, and that feeds
            # `_get_total_non_execution_costs()` — the overhead the controller
            # already prices in. No post-hoc accounting.
            #
            # A cached call is charged the same as a live one: A.4 stores the
            # usage block precisely so a replayed estimate does not become a
            # free estimate.
            call_cost = float(diagnostics.get("cost_usd") or 0.0)
            if call_cost > 0:
                self.cost_tracker.add_llm_cost(
                    cost_usd=call_cost,
                    context=f"estimator:{self.cost_estimator_mode}:{requirement_id}:{tier}",
                    input_tokens=int(diagnostics.get("prompt_tokens") or 0),
                    output_tokens=int(diagnostics.get("completion_tokens") or 0),
                )
            if tier == default_tier:
                logger.info(
                    "📈 %s: requirement %s tier=%s multiplier=%.4f %s",
                    self.cost_estimator_mode, requirement_id, tier, multiplier,
                    {k: v for k, v in diagnostics.items() if k != "usable_features"},
                )
        state["_task_cost_multipliers"] = multipliers

    def _estimator_deadline_seconds(self, auction: Dict[str, Any]) -> Optional[float]:
        """Seconds the estimator may spend, from this auction's own window.

        Uses the captured `time_remaining` rather than a re-read chain clock:
        this runs *before* the call, the value is already fresh, and an extra
        RPC round-trip here would eat the budget it is measuring. The freshness
        gate re-checks against the chain after the call returns, which is where
        an over-run is actually caught.
        """
        try:
            remaining = float(auction.get("time_remaining") or 0.0)
        except (TypeError, ValueError):
            return None
        if remaining <= 0:
            return None
        budget = remaining - float(self._bid_freshness_min_seconds) - \
            cost_estimators.ESTIMATOR_DEADLINE_RESERVE_SECONDS
        # A window too short to finish in is not a reason to wait longer; the
        # estimator returns the heuristic almost immediately at this floor.
        return max(1.0, budget)

    def _task_cost_multipliers(
        self, state: AgentState, tier_cost_micros: Dict[str, int]
    ) -> Optional[Dict[str, float]]:
        """Per-tier multipliers from the configured estimator, or None for H.

        Returns None — not a dict of 1.0s — when no estimator is configured, so
        that H's `AgentEconomics` is exactly what it was before arm R existed.
        """
        # A pure read of what `_prepare_task_multipliers` already computed. The
        # economics adapter must stay free of I/O: the simulator's equivalence
        # test builds a handler with `__new__` and no client at all, and arm X's
        # estimator is a network call that belongs on the async path.
        return state.get("_task_cost_multipliers")

    def _current_requirement_id(self, state: AgentState) -> Optional[str]:
        """Requirement behind the auction being priced, and its features.

        The service CID *is* the dataset reference, so the agent reads the
        requirement the same way it would read a service description from IPFS
        in a live market — and the estimator gets features for it.
        """
        auction = state.get("_estimator_auction") or {}
        cid = auction.get("service_description_cid")
        reference = DatasetSampler.parse_dataset_service_cid(cid or "")
        if reference is None:
            return None

        requirement_id = reference.get("requirement_id")
        estimator = getattr(self, "_cost_estimator", None)
        if estimator is not None and requirement_id and requirement_id not in estimator.features:
            try:
                sampler = self._get_estimator_sampler()
                estimator.remember_features(
                    requirement_id, sampler.get_requirement(requirement_id)
                )
            except Exception as e:
                logger.debug(f"Could not load features for {requirement_id}: {e}")
        return requirement_id

    def _get_estimator_sampler(self) -> DatasetSampler:
        """Dataset access for feature extraction, cached."""
        if getattr(self, "_estimator_sampler", None) is None:
            self._estimator_sampler = DatasetSampler(
                str(getattr(self.config, "sampled_dataset_dir", "")),
                getattr(self.config, "sampled_dataset_id", None),
            )
        return self._estimator_sampler

    def get_estimator_metrics(self) -> Dict[str, Any]:
        """What this agent's estimator did, for the run record (S7 §A.4/§B.3).

        The live/replayed split is the part that matters: the paper claims the
        calls were real, and a run served from cache must be distinguishable
        from a live one *in its own record*, not inferable from wall clock.
        """
        estimator = getattr(self, "_cost_estimator", None)
        if estimator is None:
            return {"mode": "none"}

        metrics: Dict[str, Any] = {"mode": self.cost_estimator_mode}

        # Every arm reports why it fell back to the heuristic, not just the LLM
        # ones. Coverage is part of a result: an arm that priced 18 of 30
        # auctions and one that priced all 30 are different findings even at the
        # same sigma, and "R-D was cold for 12 auctions" was previously
        # recoverable only by scraping the agent log — which an analysis reading
        # run records would silently report as zero.
        metrics["degraded"] = dict(getattr(estimator, "degraded", {}) or {})
        metrics["outcomes_len"] = len(getattr(estimator, "outcomes", []) or [])
        if isinstance(estimator, cost_estimators.DecomposedCostEstimator):
            metrics.update({
                "model": estimator.model_key,
                "calls_made": estimator.calls_made,
                "calls_cached": estimator.calls_cached,
                "calls_failed": estimator.calls_failed,
                # Per-reason, so an analysis can tell "the model was too slow"
                # from "the model could not be parsed" from "no history yet".
                "degraded": dict(getattr(estimator, "degraded", {}) or {}),
                "reasoning_cost_usd": round(estimator.reasoning_cost_usd, 6),
                "history_len": len(estimator.outcomes),
            })
            if isinstance(estimator, cost_estimators.DecomposedCostEstimator):
                # The decomposed arm is gated on calibration, not on anchors, so
                # a run where it never fired is diagnosable from the status file.
                fit = None
                try:
                    from agents.provider.estimation import prompt as estimation_prompt
                    fit = estimation_prompt.calibrate(estimator.calibration, two_param=True)
                except Exception:
                    pass
                metrics.update({
                    "n_calibration": len(estimator.calibration),
                    "calibration_a": round(fit[0], 6) if fit else None,
                    "calibration_b": round(fit[1], 6) if fit else None,
                })
        else:
            metrics["history_len"] = len(estimator.outcomes)
        return metrics

    def record_job_outcome(
        self,
        *,
        requirement_id: str,
        effort_tier: str,
        realized_cost_usd: float,
    ) -> None:
        """Add a completed job to arm R's fit history (S7 §B.2).

        Called by the orchestrator at completion, so the history only ever
        contains jobs this agent has actually finished.
        """
        if self._cost_estimator is None:
            return
        try:
            self._cost_estimator.remember_features(
                requirement_id, self._get_estimator_sampler().get_requirement(requirement_id)
            )
        except Exception as e:
            logger.debug(f"Could not load features for completed job {requirement_id}: {e}")
        self._cost_estimator.record_outcome(requirement_id, effort_tier, realized_cost_usd)
        logger.info(
            "📒 Recorded job outcome for arm R: requirement=%s tier=%s cost=$%.6f (history n=%d)",
            requirement_id, effort_tier, realized_cost_usd, len(self._cost_estimator.outcomes),
        )

    def _sync_effort_quality_from_reputation(self, agent_reputation: Dict[str, Any]) -> None:
        """Infer newly posted on-chain feedback and attribute it to completed effort tiers."""
        feedback_count = int(agent_reputation.get("feedback_count", 0) or 0)
        average_score = float(
            agent_reputation.get("rating", self.initial_reputation_default)
            or self.initial_reputation_default
        )

        if self._last_reputation_feedback_count is None:
            self._last_reputation_feedback_count = feedback_count
            self._last_reputation_average_score = average_score
            return

        previous_count = int(self._last_reputation_feedback_count or 0)
        previous_average = float(
            self._last_reputation_average_score
            if self._last_reputation_average_score is not None
            else self.initial_reputation_default
        )

        self._last_reputation_feedback_count = feedback_count
        self._last_reputation_average_score = average_score

        delta_count = feedback_count - previous_count
        if delta_count <= 0:
            return

        previous_total = previous_average * previous_count
        current_total = average_score * feedback_count
        inferred_score = (current_total - previous_total) / max(1, delta_count)
        inferred_score = max(0.0, min(100.0, inferred_score))

        applied = 0
        fallback_used = 0
        for _ in range(delta_count):
            if self._pending_feedback_effort_queue:
                pending = self._pending_feedback_effort_queue.pop(0)
                effort_tier = pending.get("effort_tier")
            else:
                effort_tier = self._default_two_way_effort_tier
                fallback_used += 1

            normalized_effort = self.cost_tracker.normalize_effort_tier(
                effort_tier,
                default=self._default_two_way_effort_tier,
            )
            self.cost_tracker.record_effort_quality_feedback(normalized_effort, inferred_score)
            applied += 1

        logger.info(
            "⭐ H3 feedback sync: delta_count=%d, inferred_score=%.3f, applied=%d, fallback=%d",
            delta_count,
            inferred_score,
            applied,
            fallback_used,
        )
    
    def _route_action(self, state: AgentState) -> str:
        """Routing function for action field."""
        if state.get("action") == "complete_service":
            return "complete_service"
        return "monitor"
    
    async def _generate_feedback_auth_node(self, state: AgentState) -> AgentState:
        """Generate ERC-8004 feedbackAuth for service completion."""
        logger.info("🔐 Generating feedbackAuth...")
        
        try:
            client_address = state.get("client_address")
            if not client_address:
                raise ValueError("client_address is required")
            
            # Use blockchain time for expiry to handle time-travel scenarios (e.g., in tests)
            current_block = await self.client.w3.eth.get_block('latest')
            block_timestamp = current_block['timestamp']
            expiry = block_timestamp + 3600  # 1 hour from current block time
            
            feedback_auth = generate_feedback_auth(
                agent_id=self.agent_id,
                client_address=client_address,
                index_limit=1000,
                expiry=expiry,
                chain_id=self.config.chain_id,
                identity_registry_address=self.config.identity_registry_address,
                signer_address=self.client.account.address,
                private_key=self.config.private_key
            )
            
            if not verify_feedback_auth_format(feedback_auth):
                raise ValueError("Generated feedbackAuth has invalid format")
            
            state["feedback_auth"] = feedback_auth
            logger.info("✅ FeedbackAuth generated")
            
        except Exception as e:
            logger.error(f"Error generating feedbackAuth: {e}")
            state["error"] = str(e)
        
        return state
    
    #: getAuctionDetails struct index 13 — `revealDuration`.
    _AUCTION_REVEAL_DURATION = 13
    #: getAuctionDetails struct index 10 — `isCompleted`.
    _AUCTION_IS_COMPLETED = 10
    #: getAuctionDetails struct index 8 — `winningBid`.
    _AUCTION_WINNING_BID = 8

    async def _auction_is_completed(self, auction_id: int) -> bool:
        """Read completion straight from the chain.

        The chain, not the receipt, is the authority on whether the work settled.
        A send that raises locally may still have landed — the observed failure
        mode is a receipt the client reads as failed followed by retries that
        revert with `ServiceAlreadyCompleted()`, which is the contract saying the
        first attempt worked.
        """
        try:
            auction = await self.client.call_contract_method(
                "ReverseAuction", "getAuctionDetails", auction_id
            )
            return bool(auction[self._AUCTION_IS_COMPLETED])
        except Exception as e:
            logger.debug(f"Could not read completion state for auction {auction_id}: {e}")
            return False

    def revenue_booked(self, auction_id: int) -> bool:
        """Whether this auction's winning bid is already in the ledger.

        The orchestrator's status report needs this to avoid double-counting a
        job's execution cost while it is still ``DELIVERING`` — see
        ``Orchestrator.get_status``. Exposed as a method rather than leaving
        callers to read ``_revenue_recorded``, so the reporting layer does not
        depend on a private attribute of this one.
        """
        try:
            return int(auction_id) in self._revenue_recorded
        except (TypeError, ValueError):
            return False

    async def _record_completion_revenue(self, auction_id: int) -> None:
        """Book the winning bid once, however many attempts it took.

        Idempotent by auction id: a retry that discovers the work already
        completed must still record the revenue the failed attempt never booked,
        but must not book it twice.
        """
        if auction_id in self._revenue_recorded:
            return
        auction = await self.client.call_contract_method(
            "ReverseAuction", "getAuctionDetails", auction_id
        )
        revenue_usd = auction[self._AUCTION_WINNING_BID] / 1e6
        self.cost_tracker.add_revenue(
            revenue_usd=revenue_usd, context=f"auction_{auction_id}"
        )
        self._revenue_recorded.add(int(auction_id))
        logger.info(f"💰 Revenue from auction {auction_id}: ${revenue_usd:.2f} USD")

    async def _call_complete_service_node(self, state: AgentState) -> AgentState:
        """Call completeService on ReverseAuction contract.

        **Idempotent.** Both failure paths — a receipt read as failed, and a send
        that reverts because the service is already completed — re-read the chain
        before reporting an error, so a job that actually settled is not retried
        into a loop and is not lost to the agent's own bookkeeping.
        """
        logger.info("📝 Calling completeService...")
        
        try:
            auction_id = state.get("auction_id")
            feedback_auth = state.get("feedback_auth")
            
            if auction_id is None:
                raise ValueError("auction_id is required")
            if not feedback_auth:
                raise ValueError("feedback_auth is required")
            
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "completeService",
                auction_id,
                feedback_auth
            )
            
            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "completeService",
                auction_id,
                feedback_auth,
                gas_limit=estimated_gas + 50000
            )
            
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                # Track gas cost
                gas_used = receipt.get('gasUsed', 0)
                gas_price_wei = receipt.get('effectiveGasPrice', 0)
                self.cost_tracker.add_gas_cost(
                    gas_used=gas_used,
                    gas_price_wei=gas_price_wei,
                    context="complete_service"
                )
                
                await self._record_completion_revenue(auction_id)

                state["tx_hash"] = tx_hash
                logger.info(f"✅ Service completed: {tx_hash}")
            elif await self._auction_is_completed(auction_id):
                # The receipt said failure; the chain says the work settled. The
                # chain wins. Gas is not booked here — the failed receipt's gas
                # is not this transaction's to claim.
                await self._record_completion_revenue(auction_id)
                state["tx_hash"] = tx_hash
                logger.warning(
                    "✅ Service completed for auction %s despite a failed receipt "
                    "(tx=%s); chain state confirms it.", auction_id, tx_hash
                )
            else:
                raise Exception("Transaction failed")

        except Exception as e:
            # `ServiceAlreadyCompleted()` (0xdb700e9d) surfaces here as a gas
            # estimation revert on a retry. Treating it as an error would fail a
            # job that is, in fact, done — and would keep the agent from ever
            # recording the outcome.
            auction_id = state.get("auction_id")
            if auction_id is not None and await self._auction_is_completed(auction_id):
                await self._record_completion_revenue(auction_id)
                state["error"] = None
                logger.warning(
                    "✅ Auction %s is already completed on-chain; treating "
                    "completeService as succeeded (%s)", auction_id, e
                )
                return state
            logger.error(f"Error calling completeService: {e}")
            state["error"] = str(e)

        return state
    
    async def _gather_state_node(self, state: AgentState) -> AgentState:
        """Gather blockchain state: won auctions and active eligible auctions."""
        logger.info("📡 Gathering blockchain state...")
        
        try:
            current_block = await self.client.get_block_number()
            from_block = self._last_processed_block + 1 if self._last_processed_block > 0 else max(0, current_block - 100)
            
            # 1. Fetch AuctionEnded events to find won auctions
            won_auctions = []
            try:
                ended_events = await self.client.get_contract_events(
                    contract_name="ReverseAuction",
                    event_name="AuctionEnded",
                    from_block=from_block,
                    to_block=current_block
                )

                ended_auction_ids: List[int] = []

                for event in ended_events:
                    args = event.get("args", {})
                    self.cost_tracker.finalize_auction_participation(args.get("auctionId"))
                    auction_id = args.get("auctionId")
                    if auction_id is not None:
                        auction_id = int(auction_id)
                        ended_auction_ids.append(auction_id)
                        self._own_current_bid_by_auction.pop(auction_id, None)
                
                for event in ended_events:
                    args = event.get("args", {})
                    if args.get("winningAgentId") == self.agent_id:
                        auction_id = args.get("auctionId")
                        if auction_id is None:
                            continue

                        auction_id = int(auction_id)
                        effort_tier = self._resolve_bid_effort_tier(
                            self._auction_effort_tier_by_auction.get(auction_id)
                        )
                        
                        # Fetch full auction details
                        try:
                            auction = await self.client.call_contract_method(
                                "ReverseAuction",
                                "getAuctionDetails",
                                auction_id
                            )
                            
                            auction_info = {
                                "auction_id": auction_id,
                                "buyer_address": auction[1],
                                "service_cid": auction[2],
                                "max_price": auction[3],
                                "duration": auction[4],
                                "start_time": auction[5],
                                "winning_agent_id": auction[7],
                                "winning_bid": auction[8],
                                "is_active": auction[9],
                                "is_completed": auction[10],
                                "escrow_amount": auction[11],
                                "reputation_weight": auction[12],
                                "effort_tier": effort_tier,
                            }

                            if effort_tier is not None:
                                self._pending_feedback_effort_queue.append(
                                    {
                                        "auction_id": auction_id,
                                        "effort_tier": effort_tier,
                                    }
                                )
                            
                            won_auctions.append(auction_info)
                            # S6's ``observe_win``: the clearing-price belief is
                            # updated when the auction *resolves*, not when the
                            # job later completes, because that is when the agent
                            # learns the price. Index 8 = winningBid, micro-USDC.
                            try:
                                self._own_winning_bids_micro[auction_id] = int(auction[8])
                            except (TypeError, ValueError):
                                logger.warning(
                                    f"Unreadable winning bid for auction {auction_id}; "
                                    f"O2c will not count it"
                                )
                            logger.info(f"🎉 Won auction {auction_id}! Buyer: {auction[1]}, Service CID: {auction[2]}")
                        except Exception as e:
                            logger.error(f"Error fetching details for won auction {auction_id}: {e}")

                # Hand the ended ids to _resolve_llm_outcome_history(), which runs after
                # reputation is fetched below (it needs the post-auction rating).
                self._recently_ended_auction_ids.extend(ended_auction_ids)

                for auction_id in ended_auction_ids:
                    self._auction_effort_tier_by_auction.pop(int(auction_id), None)
            except Exception as e:
                logger.warning(f"Error fetching AuctionEnded events: {e}")
            
            state["won_auctions"] = won_auctions
            
            # 2. Fetch all active auctions we're eligible for
            eligible_active_auctions = []
            try:
                auction_count = await self.client.call_contract_method(
                    "ReverseAuction",
                    "auctionIdCounter"
                )
                
                # One block read for the whole scan, not one per auction. The
                # timestamp is identical for every auction in a single pass, so
                # asking N times was N-1 wasted round trips (S7 §A.6.5).
                block = await self.client.get_block('latest')
                current_time = block['timestamp']

                # An auction that has settled can never become active again, so
                # re-reading it every cycle is pure cost. Without this the scan
                # is O(every auction ever created) and grows through the run:
                # measured 0.03s at auction 1, 1.56s at auction 30, which is what
                # stretched the loop cycle to an 8s tail and made agents miss
                # reveal windows late in the run.
                scannable = [
                    auction_id for auction_id in range(1, auction_count + 1)
                    if auction_id not in self._settled_auctions
                ]
                logger.info(
                    "Checking %d of %d auctions for eligibility (%d settled, skipped)...",
                    len(scannable), auction_count, len(self._settled_auctions),
                )

                for auction_id in scannable:
                    try:
                        is_eligible = await self.client.call_contract_method(
                            "ReverseAuction",
                            "isEligibleAgent",
                            auction_id,
                            self.agent_id
                        )
                        
                        if not is_eligible:
                            continue
                        
                        auction = await self.client.call_contract_method(
                            "ReverseAuction",
                            "getAuctionDetails",
                            auction_id
                        )
                        
                        # Auction struct indices (based on contract):
                        # 0: id, 1: buyer, 2: serviceDescriptionCid, 3: maxPrice
                        # 4: duration, 5: startTime, 6: eligibleAgentIds (array)
                        # 7: winningAgentId, 8: winningBid, 9: isActive
                        # 10: isCompleted, 11: escrowAmount, 12: reputationWeight
                        
                        is_active = auction[9]
                        is_completed = auction[10]
                        start_time = auction[5]
                        duration = auction[4]
                        # Index 13 is `revealDuration`. Read defensively: an
                        # IndexError here lands in the enclosing `except` and
                        # would silently drop the auction from the scan
                        # altogether, which is a far worse failure than not
                        # knowing the reveal window. Absent, the auction simply
                        # is not classified as settled yet.
                        reveal_duration = auction[13] if len(auction) > 13 else 0

                        # `current_time` comes from the single block read above:
                        # blockchain time, not wall clock, because on a forked or
                        # snapshot-loaded Anvil the two are unrelated.
                        end_time = start_time + duration
                        time_remaining = max(0, end_time - current_time)

                        # Settled for good: no phase remains and nothing further
                        # can be committed or revealed. Remember it so later
                        # cycles skip it entirely.
                        if is_completed or (
                            not is_active
                            and current_time > start_time + duration + reveal_duration
                        ):
                            self._settled_auctions.add(auction_id)
                            continue

                        if not is_active or time_remaining == 0:
                            continue
                        
                        auction_info = {
                            "auction_id": auction_id,
                            "buyer": auction[1],
                            "service_description_cid": auction[2],
                            "max_price": auction[3],
                            "duration": duration,
                            "start_time": start_time,
                            "end_time": end_time,
                            "time_remaining": time_remaining,
                            "winning_agent_id": auction[7],
                            "winning_bid": auction[8],
                            "is_active": is_active,
                            "reputation_weight": auction[12],
                            "eligible_agent_count": len(auction[6]) if isinstance(auction[6], list) else 0,
                            "eligible_agents": [int(a) for a in auction[6]] if isinstance(auction[6], list) else [],
                        }
                        
                        eligible_active_auctions.append(auction_info)
                        logger.info(f"📋 Eligible auction {auction_id}: max_price={auction[3]}, time_remaining={time_remaining}s")
                        
                    except Exception as e:
                        logger.debug(f"Error checking auction {auction_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error fetching active auctions: {e}")
            
            state["eligible_active_auctions"] = eligible_active_auctions
            self._last_processed_block = current_block

            for auction in eligible_active_auctions:
                self.cost_tracker.record_auction_participation(auction.get("auction_id"))
            
            # Pre-fetch agent's own reputation
            agent_reputation = await self._fetch_reputation(self.agent_id)
            self._sync_effort_quality_from_reputation(agent_reputation)
            state["agent_reputation"] = agent_reputation
            logger.info(f"📊 Agent reputation: rating={agent_reputation['rating']}, feedback_count={agent_reputation['feedback_count']}")

            # Settle outcomes of past bids now that the post-auction rating is known (R1).
            self._resolve_llm_outcome_history(state)
            
            # Pre-fetch IPFS service requirements and embed in auction data
            ipfs_client = IPFSClient()
            for auction in eligible_active_auctions:
                cid = auction.get("service_description_cid")
                if cid:
                    sampled_reference = DatasetSampler.parse_dataset_service_cid(cid)
                    if sampled_reference is not None:
                        auction["service_requirements"] = {
                            "execution_mode": "sampled_dataset",
                            **sampled_reference,
                        }
                        logger.debug(
                            "Skipping IPFS fetch for sampled dataset auction %s",
                            auction.get("auction_id"),
                        )
                        continue
                    try:
                        service_reqs = await ipfs_client.fetch_json(cid)
                        auction["service_requirements"] = service_reqs if service_reqs else {}
                        logger.debug(f"📥 Fetched service requirements for auction {auction['auction_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch IPFS data for auction {auction['auction_id']}: {e}")
                        auction["service_requirements"] = {}
                else:
                    auction["service_requirements"] = {}
            
            # Pre-fetch competitors' reputation (all eligible agents across active auctions)
            competitors_reputation = []
            seen_agents = set()
            for auction in eligible_active_auctions:
                agents_to_check = list(auction.get("eligible_agents", []))
                winning_agent_id = auction.get("winning_agent_id")
                if winning_agent_id and int(winning_agent_id) not in agents_to_check:
                    agents_to_check.append(int(winning_agent_id))
                for agent_id in agents_to_check:
                    agent_id_int = int(agent_id) if agent_id else 0
                    if agent_id_int and agent_id_int != int(self.agent_id) and agent_id_int not in seen_agents:
                        seen_agents.add(agent_id_int)
                        comp_rep = await self._fetch_reputation(agent_id_int)
                        competitors_reputation.append({
                            "agent_id": agent_id_int,
                            "rating": comp_rep["rating"],
                            "feedback_count": comp_rep["feedback_count"]
                        })
            
            state["competitors_reputation"] = competitors_reputation
            logger.info(f"👥 Fetched reputation for {len(competitors_reputation)} competitors")
            
            # Track financial context in state for downstream reasoning/prompting.
            execution_history = self.cost_tracker.get_execution_cost_history()
            current_balance = self.cost_tracker.get_net_balance()
            state["past_execution_costs"] = execution_history
            state["current_balance"] = current_balance
            default_effort = getattr(self.config, "default_effort_tier", "medium")
            default_execution_cost = float(
                self.cost_tracker.get_initial_execution_cost_for_effort(default_effort)
            )
            initial_execution_costs_by_effort = {
                effort: float(self.cost_tracker.get_initial_execution_cost_for_effort(effort))
                for effort in self.cost_tracker.get_effort_levels()
            }

            # Populate state based on architecture's state_level and coupling_mode
            if self.arch_config.state_level == 0:
                # State Level 0: Current only (no history) - Architecture 1
                state["estimated_service_cost"] = default_execution_cost
                state["estimated_execution_cost"] = default_execution_cost
                state["non_execution_cost_per_participation"] = 0.0
                state["unrecovered_overhead_balance"] = 0.0
                state["effort_level_execution_costs"] = initial_execution_costs_by_effort
                state["effort_level_execution_counts"] = {
                    effort: 0 for effort in self.cost_tracker.get_effort_levels()
                }
                state["effort_level_quality_stats"] = {
                    effort: {
                        "average_rating": float(self.initial_reputation_default),
                        "variance": 0.0,
                        "feedback_count": 0,
                    }
                    for effort in self.cost_tracker.get_effort_levels()
                }
            elif self.arch_config.state_level >= 1:
                # State Level 1+: Performance history (if coupling allows) - Future architectures
                if self.coupling_mode in ["one_way", "two_way"]:
                    average_execution_cost = self.cost_tracker.get_average_execution_cost(
                        default=default_execution_cost
                    )
                    average_non_execution_overhead = self.cost_tracker.get_average_non_execution_cost_per_auction(
                        default=self.config.initial_non_execution_cost_per_participation
                    )
                    unrecovered_overhead_balance = self.cost_tracker.get_unrecovered_overhead_balance()

                    state["estimated_execution_cost"] = average_execution_cost
                    state["non_execution_cost_per_participation"] = average_non_execution_overhead
                    state["unrecovered_overhead_balance"] = unrecovered_overhead_balance
                    state["estimated_service_cost"] = None
                    state["effort_level_execution_costs"] = self.cost_tracker.get_average_execution_cost_by_effort()
                    state["effort_level_execution_counts"] = self.cost_tracker.get_execution_counts_by_effort()
                    state["effort_level_quality_stats"] = self.cost_tracker.get_quality_stats_by_effort()
                    # Win-rate stats: bids placed on-chain and auctions won (= completed jobs).
                    bids_placed = self._bid_attempt_stats.get("succeeded", 0)
                    wins = self.cost_tracker.get_completed_service_count()
                    state["bids_placed_count"] = bids_placed
                    state["wins_count"] = wins
                    state["win_rate"] = round(wins / bids_placed, 3) if bids_placed > 0 else None
                    gross_margin_per_win = self.cost_tracker.get_avg_gross_margin_per_win()
                    state["avg_gross_margin_per_win_usd"] = (
                        round(gross_margin_per_win, 6) if gross_margin_per_win is not None else None
                    )
                    if getattr(self.config, "llm_margin_bounds_enabled", False):
                        bound_min = max(0.0, self.config.heuristic_min_margin)
                        bound_max = max(bound_min, self.config.heuristic_max_margin)
                        state["llm_margin_bound_min"] = bound_min
                        state["llm_margin_bound_max"] = bound_max
                    if getattr(self.config, "llm_last_margin_enabled", False):
                        state["llm_last_margin_pct"] = self._llm_last_margin_pct
                    if getattr(self.config, "llm_outcome_history_enabled", False):
                        state["llm_outcome_history"] = self._build_llm_outcome_history_view()
                    if getattr(self.config, "llm_reflection_enabled", False):
                        state["llm_strategy_memo"] = self._llm_strategy_memo
                else:
                    # Isolated coupling: no history available
                    state["estimated_service_cost"] = default_execution_cost
                    state["estimated_execution_cost"] = default_execution_cost
                    state["non_execution_cost_per_participation"] = 0.0
                    state["unrecovered_overhead_balance"] = 0.0
                    state["effort_level_execution_costs"] = initial_execution_costs_by_effort
                    state["effort_level_execution_counts"] = {
                        effort: 0 for effort in self.cost_tracker.get_effort_levels()
                    }
                    state["effort_level_quality_stats"] = self.cost_tracker.get_quality_stats_by_effort()

            # State Level 2+ (market history) - to be implemented for future architectures
            # if self.arch_config.state_level >= 2:
            #     state["past_winning_bids"] = await self._fetch_market_history()

            execution_cost_estimate = float(state.get("estimated_execution_cost") or 0.0)
            non_execution_cost_estimate = float(state.get("non_execution_cost_per_participation") or 0.0)
            unrecovered_overhead_estimate = float(state.get("unrecovered_overhead_balance") or 0.0)
            effort_execution_costs = state.get("effort_level_execution_costs") or {}
            effort_execution_counts = state.get("effort_level_execution_counts") or {}
            effort_quality_stats = state.get("effort_level_quality_stats") or {}
            logger.info(
                "💹 Cost estimates this loop: execution=$%.6f, non_execution_per_participation=$%.6f, unrecovered_overhead=$%.6f, participated=%d, active_participations=%d",
                execution_cost_estimate,
                non_execution_cost_estimate,
                unrecovered_overhead_estimate,
                self.cost_tracker.get_participated_auction_count(),
                self.cost_tracker.get_active_participated_auction_count(),
            )
            logger.info(
                "💹 Effort estimates this loop: costs=%s, counts=%s, quality=%s",
                json.dumps(effort_execution_costs, sort_keys=True),
                json.dumps(effort_execution_counts, sort_keys=True),
                json.dumps(effort_quality_stats, sort_keys=True),
            )
            
            logger.info(f"✅ State gathered: {len(won_auctions)} won, {len(eligible_active_auctions)} active eligible")
            
        except Exception as e:
            logger.error(f"Error gathering state: {e}")
            state["error"] = str(e)
            state["won_auctions"] = []
            state["eligible_active_auctions"] = []
        
        return state
    
    async def _reasoning_node(self, state: AgentState) -> AgentState:
        """Route to appropriate reasoning strategy based on reasoning_mode."""
        if self.reasoning_mode in ["deterministic", "heuristic"]:
            return await self._heuristic_reasoning(state)
        else:
            return await self._llm_react_reasoning(state)

    def _get_llm_react_actionable_auctions(
        self,
        eligible_auctions: List[Dict[str, Any]],
        *,
        log_skips: bool = True,
    ) -> List[Dict[str, Any]]:
        """Filter auctions that still require LLM reasoning in this cycle."""
        actionable_auctions: List[Dict[str, Any]] = []

        for auction in eligible_auctions:
            auction_id = auction.get("auction_id")
            if auction_id is None:
                continue

            auction_id = int(auction_id)
            if self._has_local_commitment(auction_id):
                if log_skips:
                    logger.info(f"⏭️ Skipping auction {auction_id} before ReAct reasoning: already committed")
                continue

            if auction_id in self._decided_auctions:
                if log_skips:
                    logger.info(f"⏭️ Skipping auction {auction_id} before ReAct reasoning: already decided this session")
                continue

            winning_agent_id = auction.get("winning_agent_id", 0) or 0
            if winning_agent_id == self.agent_id:
                if log_skips:
                    logger.info(f"⏭️ Skipping auction {auction_id} before ReAct reasoning: already winning")
                continue

            actionable_auctions.append(auction)

        return actionable_auctions

    async def _heuristic_reasoning(self, state: AgentState) -> AgentState:
        """Heuristic bidding strategy with selectable approaches."""
        strategy = (self.config.heuristic_strategy or "random_markup").lower()
        min_margin = max(0.0, self.config.heuristic_min_margin)
        max_margin = max(min_margin, self.config.heuristic_max_margin)

        logger.info(
            f"🎯 Using heuristic reasoning (strategy={strategy}, "
            f"min_margin={min_margin:.2f}, max_margin={max_margin:.2f})"
        )
        
        eligible_auctions = state.get("eligible_active_auctions", [])

        if not eligible_auctions:
            logger.info("No eligible auctions to consider")
            state["bids_placed"] = []
            return state

        # One-shot policy: only reason about auctions not yet decided this session.
        new_auctions = [
            a for a in eligible_auctions
            if int(a.get("auction_id", 0) or 0) not in self._decided_auctions
            and not self._has_local_commitment(int(a.get("auction_id", 0) or 0))
        ]
        if not new_auctions:
            logger.info("⏭️ Skipping heuristic reasoning: all eligible auctions already decided")
            state["bids_placed"] = []
            return state

        # Desynchronize providers detecting the same new auction simultaneously,
        # without spending a meaningful fraction of the commit window doing it.
        jitter = self._bid_jitter_seconds(new_auctions)
        logger.info("⏳ Pre-reasoning jitter: sleeping %.2fs before bidding on %d new auction(s)", jitter, len(new_auctions))
        await asyncio.sleep(jitter)

        # The state above was gathered *before* the sleep, so its `time_remaining`
        # is already stale by the jitter plus however long state gathering took.
        # Re-reading the chain clock here is what makes the freshness gate below
        # measure the thing it claims to measure (S7 §A.6.5).
        decision_time = await self._chain_timestamp()

        # Reset bids tracker
        self._bids_placed = []

        try:
            for auction in new_auctions:
                auction_id = auction["auction_id"]
                max_price = auction["max_price"]
                winning_agent_id = auction.get("winning_agent_id", 0) or 0
                time_remaining = self._fresh_time_remaining(auction, decision_time)
                # Mark decided upfront so every exit path (continue, exception) is covered.
                self._decided_auctions.add(auction_id)

                # Skip last-second bids that are likely to expire before inclusion
                if time_remaining <= self._bid_freshness_min_seconds:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: only {time_remaining}s remaining "
                        f"(gate {self._bid_freshness_min_seconds:.1f}s)"
                    )
                    continue
                
                # Use observable execution-cost state when available.
                observed_cost_usd = self._observed_execution_cost_usd(state)
                estimated_cost = max(1, int(round(observed_cost_usd * 1e6)))
                profit_cost_basis = estimated_cost
                selected_effort_tier = self._resolve_bid_effort_tier(None)

                # H1: cost_plus – sealed-bid rational bidding.
                # Samples uniformly from [profitable floor, score-competitive ceiling].
                # The ceiling is the highest bid that still beats a reference competitor
                # at the platform default reputation bidding at their own cost floor.
                # This strategy is commit-reveal aware: it never references winning_bid
                # (which is sealed during the commit phase).
                if strategy == "cost_plus":
                    if max_price <= 0:
                        logger.info(f"⏭️ Skipping auction {auction_id}: invalid max_price={max_price}")
                        continue

                    min_bid = int(estimated_cost * (1.0 + min_margin))
                    max_bid = int(estimated_cost * (1.0 + max_margin))

                    rw = max(0.0, min(1.0, _auction_rw(auction) / 100.0))
                    our_rep = float(
                        (state.get("agent_reputation") or {}).get("rating", self.initial_reputation_default)
                    )
                    ref_rep = float(self.initial_reputation_default)

                    # Score-competitive ceiling: the highest bid b where our score still
                    # matches or exceeds a reference competitor at default reputation bidding
                    # at max_price (their last-resort bid). This is the tightest bound a
                    # rational seller must satisfy; using max_price as the reference bid
                    # ensures equal-rep providers always have a feasible region.
                    #
                    #   score(b) = rw*(our_rep/100) + (1-rw)*(1 - b/max_price)
                    #   reference score = rw*(ref_rep/100)  [bid at max_price → price term = 0]
                    #   ceiling: rw*(our_rep/100) + (1-rw)*(1-b/max_price) >= rw*(ref_rep/100)
                    #   → b <= max_price * (1 - penalty)
                    #   where penalty = rw*(ref_rep - our_rep) / (100*(1-rw))   [when our_rep < ref_rep]
                    if rw >= 1.0:
                        score_ceiling = max_price if our_rep >= ref_rep else -1
                    elif our_rep >= ref_rep:
                        score_ceiling = max_price  # always win at any bid
                    else:
                        rep_gap = ref_rep - our_rep
                        penalty = rw * rep_gap / (100.0 * (1.0 - rw))
                        score_ceiling = -1 if penalty >= 1.0 else int(max_price * (1.0 - penalty))

                    effective_ceiling = min(score_ceiling, max_bid, max_price)

                    if effective_ceiling < min_bid:
                        logger.info(
                            f"⏭️ Skipping auction {auction_id}: no profitable region "
                            f"(min_bid={min_bid}, score_ceiling={score_ceiling}, max_bid={max_bid})"
                        )
                        continue

                    bid_amount = self._rng.randint(min_bid, effective_ceiling)
                    logger.info(
                        f"🎯 cost_plus: auction {auction_id} bid={bid_amount} "
                        f"[{min_bid}, {effective_ceiling}] our_rep={our_rep:.0f} ref_rep={ref_rep:.0f}"
                    )
                # H1 - Strategy 2 (feasible_random) — commit-reveal incompatible, kept for reference.
                elif strategy == "feasible_random":
                    min_profitable_bid = int(estimated_cost * (1.0 + min_margin))

                    winning_bid = auction.get("winning_bid", 0) or 0
                    reputation_weight = _auction_rw(auction)
                    our_reputation = int(
                        (state.get("agent_reputation") or {}).get(
                            "rating", self.initial_reputation_default
                        )
                    )
                    winner_reputation = self.initial_reputation_default
                    if winning_agent_id and winning_agent_id != self.agent_id:
                        for rep in state.get("competitors_reputation", []):
                            if int(rep.get("agent_id", 0) or 0) == int(winning_agent_id):
                                winner_reputation = int(rep.get("rating", self.initial_reputation_default))
                                break
                        else:
                            rep_data = await self._fetch_reputation(int(winning_agent_id))
                            winner_reputation = int(rep_data.get("rating", self.initial_reputation_default))

                    if winning_bid > 0:
                        rw = max(0, min(self.WEIGHT_SCALE, reputation_weight))
                        our_reputation_scaled = our_reputation * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                        winner_reputation_scaled = winner_reputation * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                        current_bid_component = ((max_price - winning_bid) * self.SCORE_PRECISION) // max_price
                        current_winner_score = (
                            rw * winner_reputation_scaled + (self.WEIGHT_SCALE - rw) * current_bid_component
                        ) // self.WEIGHT_SCALE

                        if rw >= 100:
                            # Bid component has zero effect when reputation is fully weighted.
                            max_competitive_bid = max_price if our_reputation > winner_reputation else -1
                        else:
                            required_numerator = self.WEIGHT_SCALE * (current_winner_score + 1) - (rw * our_reputation_scaled)
                            if required_numerator <= 0:
                                min_required_bid_component = 0
                            else:
                                min_required_bid_component = (
                                    required_numerator + (self.WEIGHT_SCALE - rw) - 1
                                ) // (self.WEIGHT_SCALE - rw)

                            if min_required_bid_component > self.SCORE_PRECISION:
                                max_competitive_bid = -1
                            else:
                                # Enforce ((max_price - bid) * SCORE_PRECISION // max_price) >= min_required_bid_component
                                min_delta = (
                                    min_required_bid_component * max_price + self.SCORE_PRECISION - 1
                                ) // self.SCORE_PRECISION
                                max_competitive_bid = max_price - min_delta
                    else:
                        max_competitive_bid = max_price

                    # Cap to auction max price
                    max_competitive_bid = min(max_competitive_bid, max_price)

                    if max_competitive_bid < min_profitable_bid:
                        logger.info(
                            f"⏭️ Skipping auction {auction_id}: no profitable competitive region "
                            f"(min_profitable={min_profitable_bid}, max_competitive={max_competitive_bid})"
                        )
                        continue

                    bid_amount = self._rng.randint(min_profitable_bid, max_competitive_bid)
                # H1 - Strategy 3: time-based concession from competitive ceiling to min profitable bid.
                elif strategy in {"h1_strategy_3", "h1_concession"}:
                    if time_remaining <= 0:
                        logger.info(f"⏭️ Skipping auction {auction_id}: auction closed")
                        continue

                    prior_total_cost_usd = state.get("estimated_service_cost")
                    if prior_total_cost_usd is None:
                        prior_total_cost_usd = self.cost_tracker.get_initial_execution_cost_for_effort(
                            getattr(self.config, "default_effort_tier", "medium")
                        )
                    prior_total_cost_usd = max(0.0, float(prior_total_cost_usd))
                    estimated_cost = int(round(prior_total_cost_usd * 1e6))
                    min_profitable_bid = int(estimated_cost * (1.0 + min_margin))
                    tick_usd = max(0.0, float(self.config.heuristic_bid_tick_usd))
                    tick_micro = max(1, int(round(tick_usd * 1e6)))

                    competitive_ceiling_raw = auction.get("winning_bid")
                    if competitive_ceiling_raw is None or int(competitive_ceiling_raw) <= 0:
                        opening_bid = int(estimated_cost * (1.0 + max_margin))
                        opening_bid = (opening_bid // tick_micro) * tick_micro
                        bid_amount = max(min_profitable_bid, opening_bid)
                        logger.info(
                            f"📌 Opening concession bid for auction {auction_id}: no current winning bid "
                            f"(bid_amount={bid_amount}, estimated_cost={estimated_cost})"
                        )
                    else:
                        winning_bid = int(competitive_ceiling_raw)
                        reputation_weight = _auction_rw(auction)
                        our_reputation = int(
                            (state.get("agent_reputation") or {}).get(
                                "rating", self.initial_reputation_default
                            )
                        )
                        winner_reputation = self.initial_reputation_default
                        if winning_agent_id and winning_agent_id != self.agent_id:
                            for rep in state.get("competitors_reputation", []):
                                if int(rep.get("agent_id", 0) or 0) == int(winning_agent_id):
                                    winner_reputation = int(rep.get("rating", self.initial_reputation_default))
                                    break
                            else:
                                rep_data = await self._fetch_reputation(int(winning_agent_id))
                                winner_reputation = int(rep_data.get("rating", self.initial_reputation_default))

                        rw = max(0, min(self.WEIGHT_SCALE, reputation_weight))
                        our_reputation_scaled = our_reputation * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                        winner_reputation_scaled = winner_reputation * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                        current_bid_component = ((max_price - winning_bid) * self.SCORE_PRECISION) // max_price
                        current_winner_score = (
                            rw * winner_reputation_scaled + (self.WEIGHT_SCALE - rw) * current_bid_component
                        ) // self.WEIGHT_SCALE

                        if rw >= 100:
                            competitive_ceiling = max_price if our_reputation > winner_reputation else -1
                        else:
                            required_numerator = self.WEIGHT_SCALE * (current_winner_score + 1) - (rw * our_reputation_scaled)
                            if required_numerator <= 0:
                                min_required_bid_component = 0
                            else:
                                min_required_bid_component = (
                                    required_numerator + (self.WEIGHT_SCALE - rw) - 1
                                ) // (self.WEIGHT_SCALE - rw)

                            if min_required_bid_component > self.SCORE_PRECISION:
                                competitive_ceiling = -1
                            else:
                                min_delta = (
                                    min_required_bid_component * max_price + self.SCORE_PRECISION - 1
                                ) // self.SCORE_PRECISION
                                competitive_ceiling = max_price - min_delta

                        competitive_ceiling = min(competitive_ceiling, max_price)

                        if competitive_ceiling < min_profitable_bid:
                            logger.info(
                                f"⏭️ Skipping auction {auction_id}: competitive ceiling below min profitable bid "
                                f"(min_profitable={min_profitable_bid}, competitive_ceiling={competitive_ceiling}, "
                                f"winning_bid={winning_bid}, our_reputation={our_reputation}, "
                                f"winner_reputation={winner_reputation}, reputation_weight={rw})"
                            )
                            continue

                        auction_start_raw = auction.get("start_time")
                        auction_end_raw = auction.get("end_time")
                        if auction_end_raw is None and auction_start_raw is not None:
                            duration_raw = auction.get("duration")
                            if duration_raw is not None:
                                auction_end_raw = int(auction_start_raw) + int(duration_raw)

                        try:
                            auction_start = int(auction_start_raw)
                            auction_end = int(auction_end_raw)
                        except (TypeError, ValueError):
                            logger.info(
                                f"⏭️ Skipping auction {auction_id}: invalid auction timing data "
                                f"(start={auction_start_raw}, end={auction_end_raw})"
                            )
                            continue

                        total_window = auction_end - auction_start
                        if total_window <= 0:
                            logger.info(
                                f"⏭️ Skipping auction {auction_id}: invalid auction window "
                                f"(start={auction_start}, end={auction_end})"
                            )
                            continue

                        now_timestamp = auction_end - int(time_remaining)
                        progress = (now_timestamp - auction_start) / float(total_window)
                        progress = max(0.0, min(1.0, progress))

                        concession_exponent = max(0.0, float(self.config.heuristic_concession_exponent))
                        if concession_exponent == 0.0:
                            position = 0.0
                        else:
                            position = 1.0 - (progress ** concession_exponent)

                        target_bid_float = min_profitable_bid + position * (competitive_ceiling - min_profitable_bid)
                        target_bid = int(target_bid_float)
                        target_bid = (target_bid // tick_micro) * tick_micro
                        target_bid = max(min_profitable_bid, min(target_bid, competitive_ceiling))

                        own_current_bid = self._own_current_bid_by_auction.get(int(auction_id))
                        if own_current_bid is None:
                            bid_amount = target_bid
                        else:
                            if winning_agent_id == self.agent_id:
                                logger.info(
                                    f"⏭️ Skipping auction {auction_id}: currently winning with own bid"
                                )
                                continue

                            min_rebid_delta_usd = max(0.0, float(self.config.heuristic_min_rebid_delta_usd))
                            min_rebid_delta = max(1, int(round(min_rebid_delta_usd * 1e6)))
                            if target_bid > (int(own_current_bid) - min_rebid_delta):
                                logger.info(
                                    f"⏭️ Skipping auction {auction_id}: concession step below min rebid delta "
                                    f"(target_bid={target_bid}, own_current_bid={own_current_bid}, "
                                    f"min_rebid_delta={min_rebid_delta})"
                                )
                                continue

                            bid_amount = target_bid
                
                # H2/H3 policy ladder: the pricing arithmetic lives in
                # policies.py so the live agent and the offline simulator run
                # identical code (S1 §1). Everything blockchain-facing — the
                # freshness gate, commit, audit trail — stays here.
                #   zi_random     : zero-intelligence uniform bidder (β null)
                #   h2_strategy_1 : overhead-aware cost-plus band
                #   h2_strategy_2 : win-rate-adaptive cost-plus
                #   h2_strategy_3 : derivative-follower pricing, fixed effort
                #   h3_strategy_2 : derivative follower + career-concerns effort gate
                elif strategy in policies.POLICIES:
                    # Which auction is being priced — arm R and arm X both need
                    # the requirement behind it, and `_policy_economics` takes
                    # only the shared state dict.
                    state["_estimator_auction"] = auction

                    # Arm X's estimator call happens HERE, inside the commit
                    # window and on the running loop, not inside the synchronous
                    # economics adapter. Measured latency by tier: gpt-4o-mini
                    # 2.8s, gpt-oss-20b 5.0s, gpt-5.4 8.1s median — so the call
                    # consumes a real fraction of the window and the freshness
                    # gate has to be re-checked once it returns.
                    await self._prepare_task_multipliers(state, auction)
                    time_remaining = self._fresh_time_remaining(
                        auction, await self._chain_timestamp()
                    )
                    if time_remaining <= self._bid_freshness_min_seconds:
                        logger.info(
                            "⏭️ Skipping auction %s: estimator returned with only %ss "
                            "left (gate %.1fs)",
                            auction_id, time_remaining, self._bid_freshness_min_seconds,
                        )
                        continue
                    decision = policies.decide(
                        strategy,
                        self._policy_auction_view(auction, state),
                        self._policy_economics(state, observed_cost_usd, estimated_cost),
                        self._policy_state_for(strategy),
                        self._policy_params(min_margin, max_margin),
                        rng=self._rng,
                    )

                    if decision.abstained:
                        logger.info(f"⏭️ Skipping auction {auction_id}: {decision.reason}")
                        continue

                    bid_amount = decision.bid_micro
                    profit_cost_basis = decision.profit_cost_basis_micro
                    selected_effort_tier = self._resolve_bid_effort_tier(decision.effort_tier)
                    logger.info(f"📐 {strategy}: auction {auction_id} {decision.reason}")

                # H3 - Strategy 1: effort-adaptive cost-plus, priced off medium-tier cost.
                # Effort tier (competitive-rank work-shirk) picks what gets delivered; the
                # bid itself is always priced as if executing at medium, so choosing "high"
                # doesn't also make the bid uncompetitive.
                elif strategy == "h3_strategy_1":
                    if max_price <= 0:
                        logger.info(f"⏭️ Skipping auction {auction_id}: invalid max_price={max_price}")
                        continue

                    participation_overhead_usd = max(
                        0.0,
                        float(state.get("non_execution_cost_per_participation") or 0.0),
                    )
                    participation_cost_micro = int(round(participation_overhead_usd * 1e6))

                    # Effort tier: competitive-rank work-shirk policy
                    #   strictly highest reputation → coast (low effort)
                    #   strictly lowest reputation  → invest (high effort)
                    #   otherwise                   → maintain (medium)
                    competitor_ratings = [
                        float(rep.get("rating") or self.initial_reputation_default)
                        for rep in state.get("competitors_reputation", [])
                        if int(rep.get("agent_id", 0) or 0) != int(self.agent_id)
                    ]
                    our_rep = float(
                        (state.get("agent_reputation") or {}).get("rating", self.initial_reputation_default)
                    )

                    if competitor_ratings and our_rep > max(competitor_ratings):
                        chosen_tier = "low"
                    elif competitor_ratings and our_rep < min(competitor_ratings):
                        chosen_tier = "high"
                    else:
                        chosen_tier = "medium"

                    # Per-tier execution costs from A3 state; fall back to EMA scalar.
                    effort_execution_costs = state.get("effort_level_execution_costs") or {}

                    def _h3s1_cost(key: str) -> int:
                        _raw = effort_execution_costs.get(key, observed_cost_usd)
                        try:
                            return int(round(max(0.0, float(_raw)) * 1e6))
                        except (TypeError, ValueError):
                            return int(round(observed_cost_usd * 1e6))

                    tier_cost_micro = _h3s1_cost(chosen_tier)
                    cost_basis      = tier_cost_micro + participation_cost_micro

                    # Bid is always priced off medium-tier cost, regardless of chosen_tier.
                    # Anchoring to a fixed medium reference keeps the bid uniformly competitive
                    # regardless of effort choice; the tier decision instead shows up as a real
                    # profit/loss difference via cost_basis/profit_cost_basis (the true cost of
                    # what was delivered).
                    medium_tier_cost_micro = _h3s1_cost("medium")
                    pricing_cost_basis     = medium_tier_cost_micro + participation_cost_micro

                    # Overhead recovery: the min-margin floor rises proportionally to
                    # unrecovered overhead, bounded by the margin gap and by
                    # heuristic_recovery_pressure_max.
                    unrecovered_usd = max(0.0, float(state.get("unrecovered_overhead_balance") or 0.0))
                    max_price_usd = max_price / 1e6
                    margin_gap = max(0.0, max_margin - min_margin)
                    pressure_max = max(0.0, min(1.0, float(getattr(self.config, 'heuristic_recovery_pressure_max', 1.0))))
                    recovery_fraction = min(unrecovered_usd / max(max_price_usd, 1e-6), margin_gap * pressure_max)
                    eff_min_margin = min_margin + recovery_fraction

                    # Bid interval anchored to medium-tier pricing cost - identical
                    # competitiveness regardless of which tier was actually chosen.
                    min_bid = int(pricing_cost_basis * (1.0 + eff_min_margin))
                    max_bid = min(int(pricing_cost_basis * (1.0 + max_margin)), max_price)
                    effective_ceiling = max_bid

                    if effective_ceiling < min_bid:
                        logger.info(
                            f"⏭️ Skipping auction {auction_id}: no profitable region "
                            f"(min_bid={min_bid}, max_bid={max_bid}, "
                            f"chosen_tier={chosen_tier}, recovery={recovery_fraction:.4f})"
                        )
                        continue

                    bid_amount = self._rng.randint(min_bid, effective_ceiling)
                    profit_cost_basis = cost_basis
                    selected_effort_tier = self._resolve_bid_effort_tier(chosen_tier)
                    logger.info(
                        f"📐 h3_strategy_1: auction {auction_id} bid={bid_amount} "
                        f"[{min_bid}, {effective_ceiling}] tier={chosen_tier} "
                        f"tier_cost={tier_cost_micro} pricing_cost={medium_tier_cost_micro} "
                        f"overhead={participation_cost_micro} "
                        f"recovery={recovery_fraction:.4f} our_rep={our_rep:.0f}"
                    )
                
                else:
                    logger.warning(f"Unknown heuristic strategy '{strategy}', defaulting to cost_plus")
                    min_bid = int(estimated_cost * (1.0 + min_margin))
                    max_bid = int(estimated_cost * (1.0 + max_margin))
                    bid_amount = self._rng.randint(min_bid, min(max_bid, max_price))
                
                # Check if bid is within max_price
                if bid_amount > max_price:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: bid {bid_amount} exceeds max_price {max_price}"
                    )
                    continue
                
                # Check profitability (should always be profitable with fixed markup)
                profit = bid_amount - profit_cost_basis
                profit_margin = (profit / profit_cost_basis) * 100
                
                logger.info(
                    f"📊 Auction {auction_id}: cost={profit_cost_basis/1e6:.6f} USDC, "
                    f"bid={bid_amount/1e6:.6f} USDC, margin={profit_margin:.3f}%"
                )

                # Strict pre-bid refresh gate: if refresh fails, skip bid.
                try:
                    fresh_auction = await self.client.call_contract_method(
                        "ReverseAuction",
                        "getAuctionDetails",
                        auction_id,
                    )
                    if not isinstance(fresh_auction, (list, tuple)) or len(fresh_auction) < 13:
                        raise ValueError("invalid getAuctionDetails payload")

                    fresh_max_price = int(fresh_auction[3])
                    fresh_commit_duration = int(fresh_auction[4])
                    fresh_start_time = int(fresh_auction[5])
                    fresh_is_active = bool(fresh_auction[9])

                    latest_block = await self.client.get_block("latest")
                    current_time = int(latest_block["timestamp"])
                    commit_deadline = fresh_start_time + fresh_commit_duration
                except Exception as e:
                    logger.warning(
                        f"Pre-bid refresh check failed for auction {auction_id}: {e}. Skipping bid."
                    )
                    continue

                if (not fresh_is_active) or (current_time >= commit_deadline):
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: commit phase no longer open"
                    )
                    continue

                if bid_amount > fresh_max_price:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: bid {bid_amount} exceeds max_price {fresh_max_price}"
                    )
                    continue

                # Sealed commitments are one-time-ever per auction; this is the final
                # authoritative gate immediately before sending the transaction.
                if await self._has_committed(auction_id):
                    logger.info(f"⏭️ Skipping auction {auction_id}: already committed")
                    continue

                # Commit the sealed bid
                try:
                    resolved_bid_effort_tier = self._resolve_bid_effort_tier(selected_effort_tier)
                    result = await self._commit_bid(
                        auction_id, bid_amount, effort_tier=resolved_bid_effort_tier,
                        mode="heuristic", commit_deadline=auction.get("end_time"),
                    )
                    if not result.get("success"):
                        logger.warning(
                            f"❌ Commit failed for auction {auction_id}: {result.get('error')}"
                        )
                except Exception as e:
                    logger.error(f"Error committing bid on auction {auction_id}: {e}")
                    self._record_bid_attempt(
                        auction_id=auction_id,
                        bid_amount=bid_amount,
                        success=False,
                        error=str(e),
                        mode="heuristic",
                    )
                    continue
            
            state["bids_placed"] = self._bids_placed
            logger.info(f"✅ Heuristic reasoning completed: {len(self._bids_placed)} bids placed")
            
        except Exception as e:
            logger.error(f"Error in deterministic reasoning: {e}")
            state["error"] = str(e)
            state["bids_placed"] = []
        
        return state
    
    async def _llm_react_reasoning(self, state: AgentState) -> AgentState:
        """Use ReAct agent to reason about bidding decisions."""
        logger.info("🤔 ReAct reasoning about auctions...")
        
        eligible_auctions = state.get("eligible_active_auctions", [])
        actionable_auctions = self._get_llm_react_actionable_auctions(eligible_auctions)
        
        if not eligible_auctions:
            logger.info("No eligible auctions to consider")
            state["bids_placed"] = []
            return state

        if not actionable_auctions:
            already_committed = [
                a["auction_id"] for a in eligible_auctions
                if self._has_local_commitment(int(a.get("auction_id", 0) or 0))
            ]
            if already_committed:
                logger.info(
                    "⏭️ Skipping LLM reasoning: already committed to all eligible auctions %s",
                    already_committed,
                )
            else:
                logger.info("No actionable auctions to consider after winner filter")
            state["bids_placed"] = []
            return state

        # Mark all actionable auctions decided before LLM reasoning (one-shot policy).
        for a in actionable_auctions:
            self._decided_auctions.add(int(a.get("auction_id", 0) or 0))

        # Desynchronize providers detecting the same new auction simultaneously.
        jitter = self._rng.uniform(0, self._BID_JITTER_MAX_SECONDS)
        logger.info("⏳ Pre-reasoning jitter: sleeping %.2fs before LLM reasoning on %d auction(s)", jitter, len(actionable_auctions))
        await asyncio.sleep(jitter)

        # R2: self-critique before deciding, so any new lesson informs this bid. No-op unless
        # llm_reflection_enabled and `interval` bids have accrued since the last reflection.
        await self._maybe_generate_reflection(state)
        if getattr(self.config, "llm_reflection_enabled", False):
            state["llm_strategy_memo"] = self._llm_strategy_memo

        reasoning_state = state
        if len(actionable_auctions) != len(eligible_auctions):
            logger.info(
                "Skipping %s auction(s) before ReAct reasoning",
                len(eligible_auctions) - len(actionable_auctions),
            )
            reasoning_state = {
                **state,
                "eligible_active_auctions": actionable_auctions,
            }

        # Reset per-cycle trackers
        self._bids_placed = []
        self._react_stop_after_bid_attempt = False
        self._bid_attempted_auctions_this_cycle = set()
        self._tool_audit_context = self._build_tool_audit_context(reasoning_state)
        
        try:
            auctions_context = json.dumps(actionable_auctions, indent=2)
            
            # Use injected system_prompt or generate from architecture's template
            if self.system_prompt is not None:
                # Use custom prompt from config
                prompt = self.system_prompt.format(
                    agent_id=self.agent_id,
                    auctions_context=auctions_context
                )
                logger.info("Using custom system prompt from config")
            else:
                # Use architecture's prompt template
                prompt = get_prompt(
                    architecture=self.arch_config.prompt_template,
                    agent_state=reasoning_state,
                )
                logger.info(f"Using {self.arch_config.name} prompt")
            
            # Create ReAct agent with configured LLM and cost tracking
            callback = LLMCostCallback(
                cost_tracker=self.cost_tracker,
                model=self.llm_model,
                input_price_per_1k=self.config.reasoning_llm_input_price_per_1k,
                output_price_per_1k=self.config.reasoning_llm_output_price_per_1k,
                scope="reasoning",
            )
            
            llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                callbacks=[callback],
                # Sampling kwargs merged with OpenRouter's usage accounting, so
                # the cost callback can use what was actually billed.
                **self._llm_kwargs_with_usage(),
            )

            tool_lookup = {tool.name: tool for tool in self._tools}
            middleware = [
                self._ToolNameCanonicalizationMiddleware(
                    tool_lookup=tool_lookup,
                    normalize_fn=self._normalize_tool_name,
                )
            ]
            recursion_limit = int(getattr(self.config, "react_recursion_limit", 20))
            decisions_before = len(self._llm_outcome_history)

            if getattr(self.config, "llm_structured_bid_enabled", False):
                result = await self._structured_bid_invoke(llm, prompt, tool_lookup)
            else:
                react_agent = create_agent(llm, self._tools, middleware=middleware)
                result = await react_agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": recursion_limit}
                )

            # Keep the model's stated reasoning for whatever it just committed to, so the next
            # reflection can judge the reasoning and not only the outcome.
            new_auction_ids = [
                e["auction_id"] for e in self._llm_outcome_history[decisions_before:]
            ]
            if new_auction_ids:
                ai_text = "\n".join(
                    str(m.content) for m in result.get("messages", [])
                    if m.__class__.__name__ == "AIMessage" and getattr(m, "content", None)
                )
                self._record_decision_rationale(ai_text, new_auction_ids)
            
            # Log the agent's reasoning trace
            logger.info("\n" + "=" * 80)
            logger.info("🤖 AGENT REASONING TRACE")
            logger.info("=" * 80)
            for i, msg in enumerate(result["messages"], 1):
                msg_class = msg.__class__.__name__
                
                # Human/System messages
                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    logger.info(f"\n[{i}] {msg_class}:\n{content}")
                
                # AI messages with tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"\n[{i}] {msg_class} - Tool Calls:")
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        logger.info(f"  🔧 {tool_name}({tool_args})")
                
                # Tool messages (results)
                if msg_class == "ToolMessage":
                    tool_output = str(msg.content)[:300]
                    logger.info(f"\n[{i}] ToolMessage:\n  ✅ {tool_output}")

            logger.info("=" * 80 + "\n")

            if self._react_stop_after_bid_attempt:
                logger.info("ReAct cycle ended after place_bid attempt; next monitor cycle will use refreshed blockchain state.")

            state["bids_placed"] = self._bids_placed
            logger.info(f"✅ ReAct completed: {len(self._bids_placed)} bids placed")
            
        except Exception as e:
            err_text = str(e)
            if "GRAPH_RECURSION_LIMIT" in err_text or "Recursion limit" in err_text:
                logger.warning(f"ReAct recursion limit reached; returning partial bidding decisions: {e}")
                state["error"] = None
                state["bids_placed"] = self._bids_placed
                return state

            logger.error(f"Error in ReAct reasoning: {e}")
            state["error"] = err_text
            state["bids_placed"] = []
        finally:
            # Avoid stale cross-cycle context
            self._tool_audit_context = {}
        
        return state
    
    async def monitor_auctions(self) -> Dict[str, Any]:
        """
        Monitor blockchain for auctions and make bidding decisions.
        
        Returns:
            Dictionary with won_auctions, bids_placed, and any errors
        """
        initial_state: AgentState = {
            "agent_id": self.agent_id,
            "action": "monitor",
            "auction_id": None,
            "client_address": None,
            "feedback_auth": None,
            "won_auctions": [],
            "eligible_active_auctions": [],
            "competitors_reputation": [],
            "agent_reputation": {"rating": self.initial_reputation_default, "feedback_count": 0},
            "estimated_service_cost": None,
            "estimated_execution_cost": None,
            "non_execution_cost_per_participation": None,
            "unrecovered_overhead_balance": None,
            "past_execution_costs": [],
            "effort_level_execution_costs": {},
            "effort_level_execution_counts": {},
            "effort_level_quality_stats": {},
            "current_balance": 0.0,
            "bids_placed": [],
            "reveals": [],
            "tx_hash": None,
            "error": None,
            "messages": []
        }

        result = await self.graph.ainvoke(initial_state)

        # Log cost summary after monitoring cycle
        self.cost_tracker.log_summary()

        return {
            "won_auctions": result.get("won_auctions", []),
            "eligible_active_auctions_count": len(result.get("eligible_active_auctions", [])),
            "bids_placed": result.get("bids_placed", []),
            "reveals": result.get("reveals", []),
            "error": result.get("error"),
            "bidding": self.get_bidding_metrics(),
        }

    def _record_bid_attempt(
        self,
        auction_id: int,
        bid_amount: int,
        success: bool,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        """Track bid attempt outcomes for runtime status reporting."""
        if error == "attempt_started":
            self.cost_tracker.record_auction_participation(auction_id)
            self._bid_attempt_stats["attempted"] += 1
            return

        if success:
            self._bid_attempt_stats["succeeded"] += 1
            return

        self._bid_attempt_stats["failed"] += 1
        self._bid_failures.append(
            {
                "timestamp": datetime.now().isoformat(),
                "auction_id": int(auction_id),
                "bid_amount": int(bid_amount),
                "error": (error or "bid_failed")[:300],
                "error_code": error_code,
                "mode": mode,
            }
        )
        if len(self._bid_failures) > 200:
            self._bid_failures = self._bid_failures[-200:]

    def get_bidding_metrics(self) -> Dict[str, Any]:
        """Return cumulative runtime bidding metrics for status snapshots."""
        return {
            "attempted": int(self._bid_attempt_stats.get("attempted", 0)),
            "succeeded": int(self._bid_attempt_stats.get("succeeded", 0)),
            "failed": int(self._bid_attempt_stats.get("failed", 0)),
            "recent_failures": self._bid_failures[-50:],
        }
    
    async def complete_service(self, auction_id: int, client_address: str) -> Dict[str, Any]:
        """
        Complete service for an auction.
        
        Args:
            auction_id: The auction ID to complete
            client_address: The buyer's address
            
        Returns:
            Dictionary with tx_hash, feedback_auth, and any errors
        """
        initial_state: AgentState = {
            "agent_id": self.agent_id,
            "action": "complete_service",
            "auction_id": auction_id,
            "client_address": client_address,
            "feedback_auth": None,
            "won_auctions": [],
            "eligible_active_auctions": [],
            "competitors_reputation": [],
            "agent_reputation": {"rating": self.initial_reputation_default, "feedback_count": 0},
            "estimated_service_cost": None,
            "estimated_execution_cost": None,
            "non_execution_cost_per_participation": None,
            "unrecovered_overhead_balance": None,
            "past_execution_costs": [],
            "effort_level_execution_costs": {},
            "effort_level_execution_counts": {},
            "effort_level_quality_stats": {},
            "current_balance": 0.0,
            "bids_placed": [],
            "reveals": [],
            "tx_hash": None,
            "error": None,
            "messages": []
        }

        result = await self.graph.ainvoke(initial_state)

        return {
            "tx_hash": result.get("tx_hash"),
            "feedback_auth": result.get("feedback_auth"),
            "error": result.get("error")
        }
    