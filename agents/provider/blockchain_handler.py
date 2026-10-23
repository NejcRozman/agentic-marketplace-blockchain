"""The provider's chain-facing agent: gathers market state, decides bids, commits and reveals them, completes won jobs.

A LangGraph workflow with two paths: ``monitor`` (gather state, reveal pending
bids, decide new bids with a heuristic policy or an LLM) and ``complete_service``
(sign the feedbackAuth and complete the job on chain).
"""

import asyncio
import inspect
import logging
import pathlib
import json
import re
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
    """An auction's reputation weight, keeping a legitimate 0.

    ``get(..., 50) or 50`` would turn w = 0 (award on price alone) into 50.
    """
    value = auction.get("reputation_weight")
    if value is None:
        return default
    return int(value)


class AgentState(TypedDict):
    """State of one workflow run: control fields, the current market, and the agent's history.

    Not every field is populated; the heuristic path, for instance, leaves the LLM fields empty.
    """

    # --- Workflow control ---
    agent_id: int
    action: str  # "complete_service" or "monitor"

    tx_hash: Optional[str]
    error: Optional[str]
    messages: List[Any]

    # --- Current market ---
    eligible_active_auctions: List[Dict[str, Any]]  # chain data plus the job description
    won_auctions: List[Dict[str, Any]]

    competitors_reputation: List[Dict[str, Any]]  # competitors' reputations
    agent_reputation: Dict[str, Any]  # own rating and feedback count

    estimated_service_cost: Optional[float]  # USD; architecture 1 only
    estimated_execution_cost: Optional[float]  # USD per completed job
    non_execution_cost_per_participation: Optional[float]  # USD per auction entered
    unrecovered_overhead_balance: Optional[float]  # USD, not yet repaid by margin
    current_balance: float  # USD, revenue minus costs

    bids_placed: List[Dict[str, Any]]  # this cycle
    reveals: List[Dict[str, Any]]  # this cycle

    # complete_service only
    auction_id: Optional[int]
    client_address: Optional[str]
    feedback_auth: Optional[bytes]

    # --- History ---
    past_execution_costs: List[float]  # USD, oldest first
    effort_level_execution_costs: Dict[str, float]  # USD per tier
    effort_level_execution_counts: Dict[str, int]  # per tier
    effort_level_quality_stats: Dict[str, Dict[str, float]]  # mean, variance and count per tier
    bids_placed_count: Optional[int]  # bids committed on chain
    wins_count: Optional[int]  # jobs won and completed
    win_rate: Optional[float]  # None until the first bid
    avg_gross_margin_per_win_usd: Optional[float]  # None until the first win
    llm_margin_bound_min: Optional[float]  # LLM-Band and later: margin band (fraction)
    llm_margin_bound_max: Optional[float]
    llm_last_margin_pct: Optional[float]  # margin of the last committed bid
    llm_outcome_history: Optional[List[Dict[str, Any]]]  # LLM-History
    llm_strategy_memo: Optional[str]  # LLM-Reflect


class BlockchainHandler:
    """The provider's chain-facing agent (LangGraph): bidding on the ``monitor`` path, delivery on ``complete_service``."""

    # Scores are computed in percent (0-100) times this, for precision.
    SCORE_PRECISION = 10_000
    WEIGHT_SCALE = 100
    # Maximum random pre-reasoning delay (seconds), to desynchronise providers.
    _BID_JITTER_MAX_SECONDS = 3.0

    # The jitter must never eat the commit window: a fixed 0-3 s delay is 30% of
    # a 10 s phase, enough to push every bid past the freshness gate below.
    _JITTER_WINDOW_FRACTION = 0.15

    # Maximum random delay before revealing. Two identical strategies compute the same
    # bid, wait on the same on-chain deadline and would reveal in the same block, so
    # the contract's strict tie-break would be decided by mempool arrival order, that
    # is by OS scheduling. Drawn from the agent's seeded RNG, the reveal order is a
    # reproducible function of (run_seed, agent_id).
    _REVEAL_JITTER_MAX_SECONDS = 1.5

    #: Keeps the reveal stream from colliding with any agent's bidding stream.
    _REVEAL_RNG_STREAM_OFFSET = 1_000_000

    # A commit sent with less than this left is likely to be mined after the phase
    # closes. Short commit windows need a smaller value via config: a 10 s gate on a
    # 10 s window never passes.
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
        """Defaults come from the architecture (``architecture`` or ``config.architecture``).

        ``reasoning_mode`` is "heuristic" or "llm_react"; ``system_prompt``, ``enabled_tools``
        and the LLM settings override the architecture's.
        """
        self.agent_id = agent_id
        self.client = blockchain_client or BlockchainClient()
        self.config = config
        self.initial_reputation_default = self.config.initial_reputation_default

        # This agent's private RNG, derived from the run seed; every random draw below
        # goes through it.
        self._rng = agent_rng(getattr(self.config, "run_seed", None), agent_id)
        # A separate stream for the reveal jitter, so it does not shift the bidding draws.
        self._reveal_rng = agent_rng(
            getattr(self.config, "run_seed", None),
            (agent_id or 0) + self._REVEAL_RNG_STREAM_OFFSET,
        )

        # Minimum seconds left in a phase for a commit or reveal to be sent.
        self._bid_freshness_min_seconds = float(getattr(
            self.config, "bid_freshness_min_seconds", self._DEFAULT_BID_FRESHNESS_MIN_SECONDS
        ))
        #: Commits that were mined after their commit phase closed; the campaign
        #: check rejects a run with any.
        self.late_commits = 0
        #: Commitments abandoned because the reveal window would close first; each is a
        #: forfeited auction.
        self.missed_reveals = 0
        self._forfeited_commitments: set = set()

        #: Auctions known to be finished; an auction never becomes active again.
        self._settled_auctions: set = set()

        arch_name = architecture or self.config.architecture
        self.arch_config = get_architecture(arch_name)

        # Architecture defaults, overridable by explicit arguments
        self.reasoning_mode = reasoning_mode or self.arch_config.reasoning_mode
        if self.reasoning_mode not in ["deterministic", "heuristic", "llm_react", "llm_strategic"]:
            raise ValueError(f"Invalid reasoning_mode: {self.reasoning_mode}")

        self.system_prompt = system_prompt

        self.llm_model = getattr(self.config, "blockchain_llm_override", None) or self.arch_config.blockchain_llm
        if llm_temperature is not None:
            self.llm_temperature = llm_temperature
        elif getattr(self.config, "llm_temperature_override", None) is not None:
            self.llm_temperature = self.config.llm_temperature_override
        else:
            self.llm_temperature = self.arch_config.llm_temperature
        self.llm_base_url = llm_base_url or self.config.openrouter_base_url
        self.llm_api_key = llm_api_key or self.config.openrouter_api_key

        # Constructed after the LLM configuration above: LLM-Estimate reads
        # `self.llm_model` and `self.llm_api_key`.
        # `none` (Adaptive) constructs no estimator, and `_policy_economics` returns
        # the history beliefs untouched.
        estimator_mode = str(getattr(self.config, "cost_estimator", "none") or "none").lower()
        if estimator_mode not in cost_estimators.ESTIMATOR_MODES:
            raise ValueError(
                f"unknown cost_estimator {estimator_mode!r}; "
                f"expected one of {cost_estimators.ESTIMATOR_MODES}"
            )
        self.cost_estimator_mode = estimator_mode
        #: Auctions whose winning bid has been booked. Completion may be retried after
        #: the chain has settled, so revenue is keyed by auction, not by attempt.
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
            # A bound, not a competitor: it reads the true costs from the run's own dataset.
            self._cost_estimator = cost_estimators.OracleCostEstimator(
                dataset_dir=str(getattr(self.config, "sampled_dataset_dir", "") or "") or None
            )
        else:
            self._cost_estimator = cost_estimators.RegressionCostEstimator(mode=estimator_mode)
        self._estimator_sampler: Optional[DatasetSampler] = None

        # Sampling comes from the block pinned for this model (seed plus reasoning
        # effort, or seed plus temperature). Resolved once so both call sites below
        # agree.
        self.llm_sampling_kwargs = chat_model_kwargs(
            self.llm_model,
            temperature_override=self.llm_temperature,
            seed_override=getattr(self.config, "llm_sampling_seed", None),
        )

        self.enabled_tools = enabled_tools or self.arch_config.enabled_tools

        self.coupling_mode = self.arch_config.coupling_mode

        self.contracts_loaded = False
        self.reverse_auction_contract = None
        self.identity_registry_contract = None
        self.reputation_registry_contract = None

        self.cost_tracker = cost_tracker or CostTracker(agent_id=agent_id, config=self.config)

        # Last block read; kept across cycles
        self._last_processed_block: int = 0

        self._bids_placed: List[Dict[str, Any]] = []
        self._react_stop_after_bid_attempt: bool = False
        self._bid_attempt_stats: Dict[str, int] = {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
        }
        self._bid_failures: List[Dict[str, Any]] = []
        self._own_current_bid_by_auction: Dict[int, int] = {}

        # Every price this agent has actually won at (the decline rule's input),
        # keyed by auction so re-reading the same AuctionEnded event cannot count
        # a win twice.
        self._own_winning_bids_micro: Dict[int, int] = {}

        # The margin of the most recent committed bid (shown only when
        # llm_last_margin_enabled); None until the first commit.
        self._llm_last_margin_pct: Optional[float] = None

        # LLM-History: one decision-outcome record per committed bid. Records start as
        # "pending" and are resolved as AuctionEnded events arrive; a win's reputation
        # effect is filled in once its feedback is on chain. _llm_outcome_by_auction
        # indexes the same records.
        self._llm_outcome_history: List[Dict[str, Any]] = []
        self._llm_outcome_by_auction: Dict[int, Dict[str, Any]] = {}
        self._recently_ended_auction_ids: List[int] = []

        # LLM-Reflect: a strategy memo the agent revises before every decision. Every
        # revision is logged (🧠 STRATEGY MEMO v..), so the trajectory survives the
        # overwrite.
        self._llm_strategy_memo: Optional[str] = None
        self._llm_memo_version: int = 0
        # The model's stated reasoning for its recent decisions, read by the reflection.
        self._llm_decision_rationales: List[Dict[str, Any]] = []
        self._RATIONALES_IN_REFLECTION = 3

        # Per-cycle state snapshot for the tool-call log.
        self._tool_audit_context: Dict[str, Any] = {}

        # Controller memory of the policies (policies.py), by strategy name.
        self._policy_states: Dict[str, "policies.PolicyState"] = {}

        self._default_two_way_effort_tier = "medium"
        self._auction_effort_tier_by_auction: Dict[int, str] = {}
        self._pending_feedback_effort_queue: List[Dict[str, Any]] = []
        self._last_reputation_feedback_count: Optional[int] = None
        self._last_reputation_average_score: Optional[float] = None

        # Local storage of the sealed bids' nonces, surviving restarts; per run, via
        # `AGENT_COMMITMENTS_DIR` (see `Config.commitments_directory`).
        _commit_dir = getattr(self.config, "commitments_directory", None)
        self._commitment_store = CommitmentStore(
            agent_id=agent_id,
            storage_dir=pathlib.Path(_commit_dir) if _commit_dir else None,
        )
        # Auctions already decided this session (including abstentions).
        self._decided_auctions: set = set()
        # Auctions bid on in the current LLM run, to block duplicate place_bid calls.
        self._bid_attempted_auctions_this_cycle: set = set()

        self._tools = self._build_tools()

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

    #: Appended to the prompt in structured mode; replaces the prompt's "call
    #: place_bid" step, since the agent places the bid from this JSON.
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

        Tolerates a code fence or surrounding prose, because ``strict: true`` is honoured by
        the endpoint but not by every model.
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
        """One schema-enforced LLM call; the agent parses the answer and places the bid itself.

        Returns a message dict shaped like the tool-calling agent's, so tracing, rationale
        capture and the outcome history work the same way.
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
            # Unparseable output is a no-bid, and is counted as one.
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
            # `place_bid` is a sync tool that calls `asyncio.run()` itself, so it runs in a
            # worker thread, as it does on the tool-calling path.
            outcome = await asyncio.to_thread(place_bid.invoke, args)
        except Exception as exc:  # surfaced, not swallowed
            logger.warning("🤖 structured place_bid failed: %s", exc)
            return {"messages": messages}
        messages.append(ToolMessage(content=json.dumps(outcome, default=str),
                                    tool_call_id="structured_bid"))
        return {"messages": messages}

    def _llm_kwargs_with_usage(self) -> Dict[str, Any]:
        """Sampling kwargs with OpenRouter's usage accounting merged into ``extra_body``.

        ``extra_body`` already carries per-model sampling (e.g. ``reasoning.effort``), and a
        second ``extra_body=`` would raise.
        """
        kwargs = dict(self.llm_sampling_kwargs)
        extra = dict(kwargs.get("extra_body") or {})
        usage = dict(extra.get("usage") or {})
        usage["include"] = True
        extra["usage"] = usage
        kwargs["extra_body"] = extra
        return kwargs

    def _build_tools(self) -> List:
        """The LLM agent's tools."""
        handler = self

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

            # One bid attempt per auction per reasoning run, so the LLM cannot retry after
            # NotInCommitPhase or another failure.
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
        """Strip known transport artefacts from a tool name the model emitted."""
        if not raw_name:
            return raw_name

        name = str(raw_name).strip().strip("`").strip('"').strip("'")
        name = re.sub(r"\s+", "", name)

        # Common malformed suffix: tool_name<|channel|>commentary
        if "<|" in name:
            name = name.split("<|", 1)[0]

        return name

    class _ToolNameCanonicalizationMiddleware(AgentMiddleware):
        """Canonicalise malformed tool names before the ToolNode dispatches them."""

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

            # ToolCallRequest.override does not include `tool`, so it is set explicitly.
            object.__setattr__(updated_request, "tool", tool_obj)

            logger.warning(f"Normalized malformed tool name '{raw_name}' -> '{canonical_name}'")
            return updated_request

        def wrap_tool_call(self, request, handler):
            return handler(self._rewrite_request_if_needed(request))

        async def awrap_tool_call(self, request, handler):
            return await handler(self._rewrite_request_if_needed(request))

    def _record_llm_last_margin(self, bid_amount: int) -> None:
        """Store the margin of this bid over the cost basis the prompt showed the LLM.

        Uses the same basis (execution EMA plus participation overhead), so the recorded
        margin is the one the model reasoned about. Skipped if the basis is unavailable.
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
        """Open a decision-outcome record for a committed bid (LLM-History).

        Holds what is known at decision time; the outcome fields are filled in by
        :meth:`_resolve_llm_outcome_history`. Always recorded, so the history is complete
        whichever prompt features are on.
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
        """Fill in the outcomes of past decisions as the chain reveals them.

        In two stages: AuctionEnded gives win or loss (and a win's revenue); a win's effect
        on reputation appears only after its feedback is posted.
        """
        current_rep = (state.get("agent_reputation") or {}).get("rating")
        current_feedback = (state.get("agent_reputation") or {}).get("feedback_count")
        balance_now = self.cost_tracker.get_net_balance()
        won_ids = {
            int(a.get("auction_id"))
            for a in (state.get("won_auctions") or [])
            if a.get("auction_id") is not None
        }

        # Stage 1: win or loss for auctions that ended this cycle.
        for auction_id in self._recently_ended_auction_ids:
            entry = self._llm_outcome_by_auction.get(int(auction_id))
            if entry is None or entry["outcome"] != "pending":
                continue
            if int(auction_id) in won_ids:
                entry["outcome"] = "won"
                # A loss has no feedback, so its reputation is settled now; a win waits for its
                # feedback (stage 2).
            else:
                entry["outcome"] = "lost"
                entry["reputation_after"] = current_rep
                entry["balance_after_usd"] = balance_now
        self._recently_ended_auction_ids = []

        # Stage 2: a win's reputation effect, once its feedback is counted.
        if isinstance(current_feedback, int):
            for entry in self._llm_outcome_history:
                if entry["outcome"] != "won" or entry["reputation_after"] is not None:
                    continue
                before = entry.get("feedback_count_before")
                if isinstance(before, int) and current_feedback > before:
                    entry["reputation_after"] = current_rep
                    entry["balance_after_usd"] = balance_now

    def _build_llm_outcome_history_view(self) -> List[Dict[str, Any]]:
        """The history as prompt rows, oldest first, optionally capped."""
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

        The reflection judges the reasoning as well as the outcome: a bad outcome from sound
        reasoning calls for a different revision than one from a wrong belief. The
        reasoning protocol already makes the model write compact structured lines, which
        are stored instead of whole prompts.
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
        # Only the newest few are shown, so older ones are dropped.
        excess = len(self._llm_decision_rationales) - self._RATIONALES_IN_REFLECTION
        if excess > 0:
            del self._llm_decision_rationales[:excess]

    def _render_outcome_table(self, rows: List[Dict[str, Any]]) -> str:
        """Compact numeric record of every past decision."""
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
        """The reflection prompt (LLM-Reflect): revise the memo from the newest evidence."""
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
        """Revise the strategy memo before deciding (LLM-Reflect); a no-op otherwise.

        Runs once per bidding cycle, once a decision exists to learn from. Its cost is
        charged as reasoning overhead. A failed revision never blocks bidding; the previous
        memo stays in force.
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
        """The state snapshot logged with each tool call (the TOOL_AUDIT lines)."""
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
        """Log a tool call with the state snapshot and the raw LLM arguments."""
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent_id": self.agent_id,
            "tool": tool_name,
            "llm_args": llm_args,
            "current_state": self._tool_audit_context or {},
        }
        logger.info(f"TOOL_AUDIT {json.dumps(event, default=str)}")

    def _build_graph(self) -> StateGraph:
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
        # The bidding path ends after reasoning; a cycle without place_bid is an abstention.
        # Pending commitments whose auction is in its reveal phase are revealed before
        # the state for this cycle's decisions is gathered.
        workflow.add_edge("reveal_commitments", "gather_state")
        workflow.add_edge("gather_state", "reasoning")
        workflow.add_edge("reasoning", END)

        return workflow.compile()

    async def initialize(self) -> bool:
        """Connect to the chain and load the contracts."""
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
        logger.info(f"🔀 Router: action={state['action']}")
        return state

    async def _fetch_reputation(self, agent_id: int) -> Dict[str, Any]:
        """An agent's rating (a float) and feedback count from the Reputation Registry.

        See :mod:`agents.common.reputation_lookup` for the registry's quirks.
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
        """Whether a commitment for this auction is stored locally.

        A cheap pre-filter; :meth:`_has_committed` is the authoritative check before a transaction.
        """
        return self._commitment_store.get_commitment(int(auction_id)) is not None

    async def _has_committed(self, auction_id: int) -> bool:
        """Whether this agent has committed to the auction: the local store, then the chain.

        The chain covers a commitment from an earlier process whose local record was lost.
        """
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

        On long phases the cap is 3 s; on short ones it shrinks with the window, so
        the desynchronisation does not consume the time the bid needs.
        """
        cap = self._BID_JITTER_MAX_SECONDS
        windows = [int(a.get("duration") or 0) for a in auctions if a.get("duration")]
        if windows:
            cap = min(cap, max(0.0, self._JITTER_WINDOW_FRACTION * min(windows)))
        return self._rng.uniform(0, cap)

    def _reveal_jitter_seconds(self, reveal_windows: List[int]) -> float:
        """Pre-reveal delay, capped to a fraction of the shortest reveal window.

        The reveal window is short (6 s in the experiments) and an unrevealed commitment is
        forfeited, so the delay stays a small fraction of it. Drawn from
        :attr:`_reveal_rng`, so it does not shift the bidding draws.
        """
        cap = self._REVEAL_JITTER_MAX_SECONDS
        windows = [int(w) for w in reveal_windows if w]
        if windows:
            cap = min(cap, max(0.0, self._JITTER_WINDOW_FRACTION * min(windows)))
        return self._reveal_rng.uniform(0, cap)

    async def _chain_timestamp(self) -> Optional[int]:
        """The current block timestamp, or None if the chain cannot be read.

        Block time, not wall time: on a snapshot-loaded Anvil the two are unrelated.
        """
        try:
            block = await self.client.get_block("latest")
            return int(block["timestamp"])
        except Exception as e:  # a stale value is better than none
            logger.debug(f"Could not read chain timestamp: {e}")
            return None

    def _fresh_time_remaining(self, auction: Dict[str, Any], decision_time: Optional[int]) -> int:
        """Seconds left in this auction's commit phase, as of now.

        The smaller of the value captured during state gathering and the one recomputed from
        the current block time: the captured value predates the jitter sleep, and for a
        deadline the safe reading of two numbers is the smaller one.
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
        """Seal and submit a bid.

        A fresh nonce is persisted locally before the transaction is broadcast, so a crash
        right after broadcast does not forfeit the reveal: the nonce exists nowhere else.
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

            # Persist before sending: an orphaned local record (the transaction never lands)
            # heals itself through the on-chain check; a lost nonce after a successful commit
            # does not.
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

            # Drop the local record only if the transaction was never broadcast; if only the
            # wait failed, it may still land.
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
        """Record whether this commit was mined before its phase closed.

        A commit that lands late does not raise; it is simply a bid that never
        counted. Counting them tells a too-short phase from a market where nobody
        wanted to bid.
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
                    "(%d late so far). Phase durations are too short for this "
                    "chain, or bid_freshness_min_seconds is too low.",
                    auction_id, mined_at - end_time, self.late_commits,
                )
        except Exception as e:
            logger.debug(f"Could not verify commit timing for auction {auction_id}: {e}")

    async def _reveal_bid(self, auction_id: int) -> Dict[str, Any]:
        """Reveal a committed bid with the locally stored amount and nonce."""
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
        """Reveal the pending commitments whose auction is in its reveal phase.

        A seeded delay precedes the first reveal of each cycle: the reveal is triggered by
        a shared on-chain deadline, so without it every agent would reveal in the same block,
        in an order decided by mempool arrival (see :attr:`_REVEAL_JITTER_MAX_SECONDS`).
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

            if phase == 0:  # commit phase: not yet
                continue

            # Once per cycle, and only when something is revealable, so an agent with nothing
            # to reveal never waits.
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

            if phase == 2:  # ended without a reveal
                logger.warning(
                    f"⏰ Reveal window for auction {auction_id} ended before we revealed; "
                    f"dropping local commitment"
                )
                self._commitment_store.remove_commitment(auction_id)
                continue

            # Reveal phase
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

            # A reveal that lands after the window closes forfeits the auction and
            # still pays gas, so skip when too little of the window is left.
            reveal_remaining = await self._reveal_seconds_remaining(auction_id)
            if reveal_remaining is not None and reveal_remaining <= self._bid_freshness_min_seconds:
                # Count the commitment, not the poll: the loop revisits a closing
                # window every cycle.
                if auction_id not in self._forfeited_commitments:
                    self._forfeited_commitments.add(auction_id)
                    self.missed_reveals += 1
                logger.warning(
                    "⏰ Skipping reveal for auction %s: only %ss of the reveal window left "
                    "(gate %.1fs). The commitment is forfeit; reveal_duration is too short "
                    "for this poll interval.",
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
            # Auction struct (ReverseAuction.sol): 4 commitDuration, 5 startTime,
            # 13 revealDuration. The reveal window opens when the commit phase ends.
            duration, start_time, reveal_duration = int(auction[4]), int(auction[5]), int(auction[13])
            block = await self.client.get_block("latest")
            return max(0, start_time + duration + reveal_duration - int(block["timestamp"]))
        except Exception as e:
            logger.debug(f"Could not read reveal window for auction {auction_id}: {e}")
            return None

    def _resolve_bid_effort_tier(self, selected_effort_tier: Optional[str]) -> Optional[str]:
        """The effort tier to bid with under two-way coupling; None otherwise."""
        if self.coupling_mode != "two_way":
            return None

        return self.cost_tracker.normalize_effort_tier(
            selected_effort_tier,
            default=self._default_two_way_effort_tier,
        )

    # --- Adapters onto policies.py: state dict and config into the policies' dataclasses ---
    def _observed_execution_cost_usd(self, state: AgentState) -> float:
        """The execution-cost estimate (USD), or the configured prior."""
        observed_cost_usd = state.get("estimated_execution_cost")
        if observed_cost_usd is None:
            observed_cost_usd = state.get("estimated_service_cost")
        if observed_cost_usd is None:
            observed_cost_usd = self.cost_tracker.get_initial_execution_cost_for_effort(
                getattr(self.config, "default_effort_tier", "medium")
            )
        return max(0.0, float(observed_cost_usd))

    def _policy_state_for(self, strategy: str) -> "policies.PolicyState":
        """Controller memory (margin, prev_wins, bids_placed) of one strategy, for the agent's lifetime."""
        return self._policy_states.setdefault(strategy, policies.PolicyState())

    def _policy_params(self, min_margin: float, max_margin: float) -> "policies.PolicyParams":
        return policies.PolicyParams(
            min_margin=min_margin,
            max_margin=max_margin,
            initial_margin_mode=getattr(self.config, "heuristic_initial_margin_mode", "max"),
            initial_reputation_default=self.initial_reputation_default,
            # The decline rule, judged against the prices this agent has won at.
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

        # The estimator supplies the shape, history the level. `multipliers` is
        # None for Adaptive, whose economics are then the history beliefs alone.
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
            # Wins: one per delivered job so far.
            wins=len(state.get("past_execution_costs") or []),
            # The decline rule's input; insertion order is win order.
            own_winning_bids_micro=tuple(self._own_winning_bids_micro.values()),
        )

    async def _prepare_task_multipliers(self, state: AgentState, auction: Dict[str, Any]) -> None:
        """Run the configured estimator for this auction, before the policy decides.

        Every estimator runs here, so the economics adapter stays synchronous and
        pure; LLM-Estimate awaits an LLM call, which is why this is the async site.
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
            # The prompt states the agent's own per-tier averages, the level the multiplier
            # is relative to; quality comes from its own feedback.
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
            # What is left of the commit window, minus what the bid still needs after the
            # estimate returns; otherwise the transport's own retry budget would apply.
            outcome = estimator.multiplier_for(
                requirement_id, tier, level_usd,
                deadline_seconds=self._estimator_deadline_seconds(auction),
            )
            multiplier, diagnostics = (
                await outcome if inspect.isawaitable(outcome) else outcome
            )
            multipliers[tier] = multiplier

            # The estimator's tokens are charged to the agent: `add_llm_cost` routes
            # them to the reasoning bucket, which feeds the per-participation
            # overhead the controller prices in. A cached call is charged the same
            # as a live one, from its recorded usage.
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

        Uses the captured ``time_remaining`` rather than a chain read, which would eat the
        budget it measures; the freshness gate re-checks against the chain afterwards.
        """
        try:
            remaining = float(auction.get("time_remaining") or 0.0)
        except (TypeError, ValueError):
            return None
        if remaining <= 0:
            return None
        budget = remaining - float(self._bid_freshness_min_seconds) - \
            cost_estimators.ESTIMATOR_DEADLINE_RESERVE_SECONDS
        # At this floor the estimator returns the heuristic almost at once.
        return max(1.0, budget)

    def _task_cost_multipliers(
        self, state: AgentState, tier_cost_micros: Dict[str, int]
    ) -> Optional[Dict[str, float]]:
        """Per-tier multipliers from the configured estimator, or None for Adaptive.

        A pure read of what `_prepare_task_multipliers` already computed, so the
        economics adapter stays free of I/O.
        """
        return state.get("_task_cost_multipliers")

    def _current_requirement_id(self, state: AgentState) -> Optional[str]:
        """The request behind the auction being priced; also loads its features into the estimator.

        The service CID is the dataset reference, read the way a live agent would read a
        job description.
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
        if getattr(self, "_estimator_sampler", None) is None:
            self._estimator_sampler = DatasetSampler(
                str(getattr(self.config, "sampled_dataset_dir", "")),
                getattr(self.config, "sampled_dataset_id", None),
            )
        return self._estimator_sampler

    def get_estimator_metrics(self) -> Dict[str, Any]:
        """What this agent's estimator did, for the run record.

        Includes the split between live and replayed (cached) calls, so a run
        served from the cache is recognisable from its own record.
        """
        estimator = getattr(self, "_cost_estimator", None)
        if estimator is None:
            return {"mode": "none"}

        metrics: Dict[str, Any] = {"mode": self.cost_estimator_mode}

        # Every estimator reports why it fell back to the heuristic, by reason.
        metrics["degraded"] = dict(getattr(estimator, "degraded", {}) or {})
        metrics["outcomes_len"] = len(getattr(estimator, "outcomes", []) or [])
        if isinstance(estimator, cost_estimators.DecomposedCostEstimator):
            metrics.update({
                "model": estimator.model_key,
                "calls_made": estimator.calls_made,
                "calls_cached": estimator.calls_cached,
                "calls_failed": estimator.calls_failed,
                "degraded": dict(getattr(estimator, "degraded", {}) or {}),
                "reasoning_cost_usd": round(estimator.reasoning_cost_usd, 6),
                "history_len": len(estimator.outcomes),
            })
            if isinstance(estimator, cost_estimators.DecomposedCostEstimator):
                # Calibration state, so a run where LLM-Estimate never priced is visible in the
                # status file.
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
        """Add a completed job to the estimator's history.

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
            "📒 Recorded job outcome for the estimator: requirement=%s tier=%s cost=$%.6f (history n=%d)",
            requirement_id, effort_tier, realized_cost_usd, len(self._cost_estimator.outcomes),
        )

    def _sync_effort_quality_from_reputation(self, agent_reputation: Dict[str, Any]) -> None:
        """Infer newly posted feedback from the reputation and attribute it to the delivered effort tiers."""
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
            "⭐ Effort-tier feedback sync: delta_count=%d, inferred_score=%.3f, applied=%d, fallback=%d",
            delta_count,
            inferred_score,
            applied,
            fallback_used,
        )

    def _route_action(self, state: AgentState) -> str:
        if state.get("action") == "complete_service":
            return "complete_service"
        return "monitor"

    async def _generate_feedback_auth_node(self, state: AgentState) -> AgentState:
        """Sign the ERC-8004 feedbackAuth for a completed job."""
        logger.info("🔐 Generating feedbackAuth...")

        try:
            client_address = state.get("client_address")
            if not client_address:
                raise ValueError("client_address is required")

            # Expiry from block time, not wall time.
            current_block = await self.client.w3.eth.get_block('latest')
            block_timestamp = current_block['timestamp']
            expiry = block_timestamp + 3600  # one hour

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
        """Whether the chain records the job as completed.

        The chain, not the receipt, decides: a send that fails locally may still have
        landed, after which retries revert with ``ServiceAlreadyCompleted()``.
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
        """Whether this auction's winning bid is already booked.

        Used by ``Orchestrator.get_status`` to avoid double-counting a job in DELIVERING.
        """
        try:
            return int(auction_id) in self._revenue_recorded
        except (TypeError, ValueError):
            return False

    async def _record_completion_revenue(self, auction_id: int) -> None:
        """Book the winning bid once, however many attempts completion took.

        A retry that finds the job already completed still books the revenue the failed
        attempt never booked, but never books it twice.
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
        """Call completeService, idempotently.

        A receipt read as failed, and a send reverting because the job is already completed,
        both re-read the chain before reporting an error, so a settled job is neither
        retried in a loop nor lost to the bookkeeping.
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
                # The receipt says failure but the chain says the job settled; the chain wins.
                # The failed receipt's gas is not booked.
                await self._record_completion_revenue(auction_id)
                state["tx_hash"] = tx_hash
                logger.warning(
                    "✅ Service completed for auction %s despite a failed receipt "
                    "(tx=%s); chain state confirms it.", auction_id, tx_hash
                )
            else:
                raise Exception("Transaction failed")

        except Exception as e:
            # `ServiceAlreadyCompleted()` (0xdb700e9d) surfaces as a gas-estimation revert
            # on a retry: the job is done, not failed.
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
        """Gather the market state: newly won auctions, active eligible auctions, reputations, own economics."""
        logger.info("📡 Gathering blockchain state...")

        try:
            current_block = await self.client.get_block_number()
            from_block = self._last_processed_block + 1 if self._last_processed_block > 0 else max(0, current_block - 100)

            # 1. Newly won auctions, from AuctionEnded events
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
                            # The clearing-price belief is updated when the auction
                            # resolves, which is when the agent learns the price.
                            # Index 8 = winningBid, micro-USDC.
                            try:
                                self._own_winning_bids_micro[auction_id] = int(auction[8])
                            except (TypeError, ValueError):
                                logger.warning(
                                    f"Unreadable winning bid for auction {auction_id}; "
                                    f"the decline rule will not count it"
                                )
                            logger.info(f"🎉 Won auction {auction_id}! Buyer: {auction[1]}, Service CID: {auction[2]}")
                        except Exception as e:
                            logger.error(f"Error fetching details for won auction {auction_id}: {e}")

                # The ended auctions are resolved in the outcome history after the reputation is
                # read below.
                self._recently_ended_auction_ids.extend(ended_auction_ids)

                for auction_id in ended_auction_ids:
                    self._auction_effort_tier_by_auction.pop(int(auction_id), None)
            except Exception as e:
                logger.warning(f"Error fetching AuctionEnded events: {e}")

            state["won_auctions"] = won_auctions

            # 2. Active auctions this agent is eligible for
            eligible_active_auctions = []
            try:
                auction_count = await self.client.call_contract_method(
                    "ReverseAuction",
                    "auctionIdCounter"
                )

                # One block read for the whole scan: the timestamp is the same for every auction.
                block = await self.client.get_block('latest')
                current_time = block['timestamp']

                # A settled auction can never become active again, so it is skipped; otherwise
                # the scan grows with every auction created and slows the loop late in a run.
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

                        # Auction struct: 0 id, 1 buyer, 2 serviceDescriptionCid, 3 maxPrice,
                        # 4 duration, 5 startTime, 6 eligibleAgentIds, 7 winningAgentId, 8 winningBid,
                        # 9 isActive, 10 isCompleted, 11 escrowAmount, 12 reputationWeight.

                        is_active = auction[9]
                        is_completed = auction[10]
                        start_time = auction[5]
                        duration = auction[4]
                        # Index 13 is revealDuration. Read defensively: an IndexError would drop the
                        # auction from the scan; without it the auction is just not classified as
                        # settled yet.
                        reveal_duration = auction[13] if len(auction) > 13 else 0

                        # Block time from the single read above, not wall time.
                        end_time = start_time + duration
                        time_remaining = max(0, end_time - current_time)

                        # Settled for good: nothing can be committed or revealed any more.
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

            agent_reputation = await self._fetch_reputation(self.agent_id)
            self._sync_effort_quality_from_reputation(agent_reputation)
            state["agent_reputation"] = agent_reputation
            logger.info(f"📊 Agent reputation: rating={agent_reputation['rating']}, feedback_count={agent_reputation['feedback_count']}")

            # Settle the outcomes of past bids now that the post-auction rating is known.
            self._resolve_llm_outcome_history(state)

            # Job descriptions from IPFS (live mode)
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
                        # Short timeout: this runs inside the commit window.
                        service_reqs = await ipfs_client.fetch_json(cid, timeout=5)
                        auction["service_requirements"] = service_reqs if service_reqs else {}
                        logger.debug(f"📥 Fetched service requirements for auction {auction['auction_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch IPFS data for auction {auction['auction_id']}: {e}")
                        auction["service_requirements"] = {}
                else:
                    auction["service_requirements"] = {}

            # Reputations of the competitors in all eligible auctions
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

            # Financial context, for the policy and the prompt
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

            # State by the architecture's state_level and coupling_mode
            if self.arch_config.state_level == 0:
                # State level 0: current state only (architecture 1)
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
                # State level 1: the agent's own performance history (architecture 2)
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
                    # Win rate: bids committed on chain, and auctions won (= completed jobs).
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
                    # Isolated coupling: no history
                    state["estimated_service_cost"] = default_execution_cost
                    state["estimated_execution_cost"] = default_execution_cost
                    state["non_execution_cost_per_participation"] = 0.0
                    state["unrecovered_overhead_balance"] = 0.0
                    state["effort_level_execution_costs"] = initial_execution_costs_by_effort
                    state["effort_level_execution_counts"] = {
                        effort: 0 for effort in self.cost_tracker.get_effort_levels()
                    }
                    state["effort_level_quality_stats"] = self.cost_tracker.get_quality_stats_by_effort()

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
        """The auctions the LLM still has to decide in this cycle."""
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
        """Decide and commit bids with the configured heuristic policy (and estimator)."""
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

        # Decide each auction once per session.
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
        # measure the thing it claims to measure.
        decision_time = await self._chain_timestamp()

        self._bids_placed = []

        try:
            for auction in new_auctions:
                auction_id = auction["auction_id"]
                max_price = auction["max_price"]
                time_remaining = self._fresh_time_remaining(auction, decision_time)
                # Marked decided first, so every exit path is covered.
                self._decided_auctions.add(auction_id)

                # Skip bids too late to be mined before the commit phase closes.
                if time_remaining <= self._bid_freshness_min_seconds:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: only {time_remaining}s remaining "
                        f"(gate {self._bid_freshness_min_seconds:.1f}s)"
                    )
                    continue

                observed_cost_usd = self._observed_execution_cost_usd(state)
                estimated_cost = max(1, int(round(observed_cost_usd * 1e6)))
                profit_cost_basis = estimated_cost
                selected_effort_tier = self._resolve_bid_effort_tier(None)

                # The pricing arithmetic lives in policies.py; everything
                # blockchain-facing (freshness gate, commit, audit trail) stays here.
                #   zi_random     : Random
                #   h2_strategy_1 : Random-in-band
                #   h2_strategy_3 : Adaptive
                if strategy in policies.POLICIES:
                    # Which auction is being priced: the estimators need the
                    # request behind it, and `_policy_economics` takes only the
                    # shared state dict.
                    state["_estimator_auction"] = auction

                    # The estimator runs here, inside the commit window and on the event loop. An LLM
                    # call takes a real fraction of the window, so the freshness gate is re-checked
                    # when it returns.
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

                else:
                    logger.error(f"Unknown heuristic strategy '{strategy}'; not bidding")
                    continue

                if bid_amount > max_price:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: bid {bid_amount} exceeds max_price {max_price}"
                    )
                    continue

                profit = bid_amount - profit_cost_basis
                profit_margin = (profit / profit_cost_basis) * 100

                logger.info(
                    f"📊 Auction {auction_id}: cost={profit_cost_basis/1e6:.6f} USDC, "
                    f"bid={bid_amount/1e6:.6f} USDC, margin={profit_margin:.3f}%"
                )

                # Re-read the auction before bidding; skip the bid if that fails.
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

                # A commitment is once per auction, ever; this is the authoritative check just
                # before sending.
                if await self._has_committed(auction_id):
                    logger.info(f"⏭️ Skipping auction {auction_id}: already committed")
                    continue

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
        """Decide and commit bids with the LLM price-setter."""
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

        # Mark every actionable auction decided before the LLM runs.
        for a in actionable_auctions:
            self._decided_auctions.add(int(a.get("auction_id", 0) or 0))

        # Desynchronise providers that detect the same new auction.
        jitter = self._rng.uniform(0, self._BID_JITTER_MAX_SECONDS)
        logger.info("⏳ Pre-reasoning jitter: sleeping %.2fs before LLM reasoning on %d auction(s)", jitter, len(actionable_auctions))
        await asyncio.sleep(jitter)

        # LLM-Reflect: revise the memo before deciding, so a new lesson informs this bid.
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

        self._bids_placed = []
        self._react_stop_after_bid_attempt = False
        self._bid_attempted_auctions_this_cycle = set()
        self._tool_audit_context = self._build_tool_audit_context(reasoning_state)

        try:
            auctions_context = json.dumps(actionable_auctions, indent=2)

            # A configured system prompt, or the architecture's prompt
            if self.system_prompt is not None:
                prompt = self.system_prompt.format(
                    agent_id=self.agent_id,
                    auctions_context=auctions_context
                )
                logger.info("Using custom system prompt from config")
            else:
                prompt = get_prompt(
                    architecture=self.arch_config.prompt_template,
                    agent_state=reasoning_state,
                )
                logger.info(f"Using {self.arch_config.name} prompt")

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

            # Keep the model's stated reasoning for what it just committed to, for the next
            # reflection.
            new_auction_ids = [
                e["auction_id"] for e in self._llm_outcome_history[decisions_before:]
            ]
            if new_auction_ids:
                ai_text = "\n".join(
                    str(m.content) for m in result.get("messages", [])
                    if m.__class__.__name__ == "AIMessage" and getattr(m, "content", None)
                )
                self._record_decision_rationale(ai_text, new_auction_ids)

            # Log the reasoning trace
            logger.info("\n" + "=" * 80)
            logger.info("🤖 AGENT REASONING TRACE")
            logger.info("=" * 80)
            for i, msg in enumerate(result["messages"], 1):
                msg_class = msg.__class__.__name__

                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    logger.info(f"\n[{i}] {msg_class}:\n{content}")

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"\n[{i}] {msg_class} - Tool Calls:")
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        logger.info(f"  🔧 {tool_name}({tool_args})")

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
            # No context carries over between cycles.
            self._tool_audit_context = {}

        return state

    async def monitor_auctions(self) -> Dict[str, Any]:
        """One monitoring cycle: returns the won auctions, the bids placed and any errors."""
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
        """Count a bid attempt's outcome, for the status file."""
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
        """Cumulative bidding counts, for the status file."""
        return {
            "attempted": int(self._bid_attempt_stats.get("attempted", 0)),
            "succeeded": int(self._bid_attempt_stats.get("succeeded", 0)),
            "failed": int(self._bid_attempt_stats.get("failed", 0)),
            "recent_failures": self._bid_failures[-50:],
        }

    async def complete_service(self, auction_id: int, client_address: str) -> Dict[str, Any]:
        """Complete a won job on chain; returns the transaction hash and the feedbackAuth."""
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
