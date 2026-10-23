"""
Cost tracking infrastructure for agent economic simulation.

Provides:
- CostTracker: Simple in-memory accounting for LLM costs, gas fees, and revenue
- LLMCostCallback: LangChain callback handler to automatically track LLM token usage
"""

import logging
from typing import Optional
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class LLMCostCallback(BaseCallbackHandler):
    """
    Callback handler to track LLM token usage and costs.
    
    Integrates with CostTracker to record actual LLM expenses.
    """
    
    def __init__(
        self,
        cost_tracker: "CostTracker",
        model: str,
        input_price_per_1k: float,
        output_price_per_1k: float,
        scope: str = "llm",
    ):
        super().__init__()
        self.cost_tracker = cost_tracker
        self.model = model
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self.scope = scope
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @staticmethod
    def _reported_cost(response):
        """The provider's own billed cost for this call, or None.

        OpenRouter surfaces it as `usage.cost`; LangChain carries the raw usage
        dict through either `llm_output["token_usage"]` or the message's
        `response_metadata["token_usage"]`. Returns None when absent so the
        caller can fall back to config pricing.
        """
        def _dig(usage):
            if not isinstance(usage, dict):
                return None
            value = usage.get("cost")
            if value is None:
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None

        try:
            llm_output = getattr(response, "llm_output", None) or {}
            found = _dig(llm_output.get("token_usage"))
            if found is not None:
                return found
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                meta = getattr(message, "response_metadata", None) or {}
                return _dig(meta.get("token_usage"))
        except Exception:
            return None
        return None

    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes - calculate cost from token usage."""
        try:
            input_tokens = 0
            output_tokens = 0

            def _to_non_negative_int(value) -> int:
                try:
                    return max(0, int(value or 0))
                except (TypeError, ValueError):
                    return 0

            # Format 1: llm_output.token_usage
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {}) or {}
                input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)

            # Format 2: generations[*][*].message.usage_metadata / response_metadata.token_usage
            if (input_tokens + output_tokens) == 0 and hasattr(response, "generations") and response.generations:
                try:
                    gen0 = response.generations[0][0]
                    msg = getattr(gen0, "message", None)

                    if msg is not None:
                        usage_meta = getattr(msg, "usage_metadata", None) or {}
                        input_tokens = usage_meta.get("input_tokens", 0) or usage_meta.get("prompt_tokens", 0)
                        output_tokens = usage_meta.get("output_tokens", 0) or usage_meta.get("completion_tokens", 0)

                        if (input_tokens + output_tokens) == 0:
                            resp_meta = getattr(msg, "response_metadata", None) or {}
                            token_usage = resp_meta.get("token_usage", {}) or {}
                            input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                            output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
                except Exception:
                    pass

            input_tokens = _to_non_negative_int(input_tokens)
            output_tokens = _to_non_negative_int(output_tokens)

            # Nothing to account for
            if (input_tokens + output_tokens) == 0:
                return

            # Prefer the provider's OWN billed cost when it is returned.
            #
            # OpenRouter returns `usage.cost` (real dollars billed, net of
            # prompt caching) when the request carries `usage: {include: true}`.
            # The config fallback below is a single flat rate applied to EVERY
            # model — $0.00075/1k in, $0.0045/1k out — which for gpt-oss-20b
            # overstates the true rate by ~27x input / ~33x output. Since only
            # the LLM arm makes calls, that error is subtracted from its net
            # balance alone and biases every LLM-vs-heuristic comparison against
            # the LLM. H2's estimator path (experiments/estimation/llm_client.py)
            # has always read `usage.cost`; this makes the react path agree, so
            # H1 and H2 price LLM calls the same way.
            reported = self._reported_cost(response)
            if reported is not None:
                total_cost = reported
            else:
                input_cost = (input_tokens / 1000) * self.input_price_per_1k
                output_cost = (output_tokens / 1000) * self.output_price_per_1k
                total_cost = input_cost + output_cost

            self.total_tokens += (input_tokens + output_tokens)
            self.total_cost += total_cost

            # Record in cost tracker
            self.cost_tracker.add_llm_cost(
                cost_usd=total_cost,
                context=f"{self.scope}:{self.model} ({input_tokens}+{output_tokens} tokens)",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
                
        except Exception as e:
            logger.warning(f"Failed to track LLM cost: {e}")


class CostTracker:
    """
    Tracks all operational costs for an agent.
    
    Simple in-memory accounting:
    - LLM API calls (reasoning + service execution)
    - Blockchain gas fees
    - Revenue from completed services
    - Service execution cost history (for bidding strategies)
    
    No persistence - tracks during agent lifetime only.
    """

    EFFORT_LEVELS = ("low", "medium", "high")
    
    def __init__(self, agent_id: int, config):
        self.agent_id = agent_id
        self.config = config
        
        # Cost accumulators (in USD)
        self.total_llm_costs = 0.0
        self.total_gas_costs = 0.0
        self.total_revenue = 0.0

        # LLM cost split by regime: reasoning (bidding/monitoring) vs service execution.
        # These always sum to total_llm_costs; the split is attributed at record time
        # by whether an execution window is open (see add_llm_cost).
        self.total_reasoning_llm_costs = 0.0
        self.total_service_llm_costs = 0.0

        # Call and token counts per regime, so a change in cost can be attributed to
        # more calls or to bigger calls rather than left ambiguous.
        self.reasoning_llm_calls = 0
        self.service_llm_calls = 0
        self.reasoning_input_tokens = 0
        self.reasoning_output_tokens = 0
        self.service_input_tokens = 0
        self.service_output_tokens = 0

        # Service execution tracking
        self.service_execution_history = []  # List of raw execution costs (USD)
        self._current_execution_llm_cost = 0.0
        self._is_executing_service = False
        self._participated_auction_ids = set()
        self._active_auction_participation_costs = {}
        self._execution_cost_ema = None
        self._non_execution_cost_per_auction_ema = None
        self._execution_cost_ema_by_effort = {
            effort: None for effort in self.EFFORT_LEVELS
        }
        self._execution_cost_history_by_effort = {
            effort: [] for effort in self.EFFORT_LEVELS
        }
        self._quality_stats_by_effort = {
            effort: {"count": 0, "mean": 0.0, "m2": 0.0}
            for effort in self.EFFORT_LEVELS
        }
        self._current_execution_effort_tier = "medium"
        self._current_execution_input_tokens = 0
        self._current_execution_output_tokens = 0
        self._execution_cost_ema_alpha = min(
            1.0,
            max(0.0, float(getattr(self.config, "execution_cost_ema_alpha", 0.25))),
        )
        self._non_execution_cost_ema_alpha = min(
            1.0,
            max(0.0, float(getattr(self.config, "non_execution_cost_ema_alpha", 0.20))),
        )

    @staticmethod
    def _update_ema(previous: float, sample: float, alpha: float) -> float:
        """Apply one EMA update step."""
        return (alpha * sample) + ((1.0 - alpha) * previous)

    def normalize_effort_tier(self, effort_tier: str = None, default: str = "medium") -> str:
        """Normalize effort tiers to strict low/medium/high values."""
        fallback = str(default or "medium").strip().lower()
        if fallback not in self.EFFORT_LEVELS:
            fallback = "medium"

        if effort_tier is None:
            return fallback

        normalized = str(effort_tier).strip().lower()
        return normalized if normalized in self.EFFORT_LEVELS else fallback

    def get_effort_levels(self) -> list:
        """Return supported canonical effort levels."""
        return list(self.EFFORT_LEVELS)

    def record_auction_participation(self, auction_id: int):
        """Record a unique auction participation for overhead amortization."""
        if auction_id is None:
            return
        auction_id = int(auction_id)
        self._participated_auction_ids.add(auction_id)
        self._active_auction_participation_costs.setdefault(
            auction_id,
            self._get_total_non_execution_costs(),
        )

    def finalize_auction_participation(self, auction_id: int):
        """Mark an auction participation as finished so its overhead becomes historical."""
        if auction_id is None:
            return
        snapshot = self._active_auction_participation_costs.pop(int(auction_id), None)
        if snapshot is None:
            return

        finalized_auctions = self.get_participated_auction_count() - self.get_active_participated_auction_count()
        if finalized_auctions <= 0:
            return

        historical_non_execution_costs = self._get_historical_non_execution_costs()
        sample_average = historical_non_execution_costs / finalized_auctions
        prior = max(0.0, float(getattr(self.config, "initial_non_execution_cost_per_participation", 0.0)))

        if self._non_execution_cost_per_auction_ema is None:
            self._non_execution_cost_per_auction_ema = self._update_ema(
                previous=prior,
                sample=sample_average,
                alpha=self._non_execution_cost_ema_alpha,
            )
        else:
            self._non_execution_cost_per_auction_ema = self._update_ema(
                previous=self._non_execution_cost_per_auction_ema,
                sample=sample_average,
                alpha=self._non_execution_cost_ema_alpha,
            )

    def get_participated_auction_count(self) -> int:
        """Return the number of unique auctions this agent has participated in."""
        return len(self._participated_auction_ids)

    def get_completed_service_count(self) -> int:
        """Return the number of completed service executions."""
        return len(self.service_execution_history)

    def get_active_participated_auction_count(self) -> int:
        """Return the number of participated auctions still in progress."""
        return len(self._active_auction_participation_costs)

    def get_average_execution_cost(self, default: float = 0.0) -> float:
        """Return a prior-seeded EMA estimate of execution costs."""
        if not self.service_execution_history:
            return float(default)

        if self._execution_cost_ema is not None:
            return float(self._execution_cost_ema)

        # Backward-compatible fallback if history exists but EMA was not initialized.
        return sum(self.service_execution_history) / len(self.service_execution_history)

    def get_initial_execution_cost_for_effort(self, effort_tier: str = None) -> float:
        """Resolve configured initial execution-cost prior for a given effort tier."""
        effort = self.normalize_effort_tier(effort_tier)

        resolver = getattr(self.config, "get_initial_service_execution_cost_for_effort", None)
        if callable(resolver):
            try:
                return max(0.0, float(resolver(effort)))
            except (TypeError, ValueError):
                pass

        fallback = max(0.0, float(getattr(self.config, "initial_service_execution_cost", 0.0)))
        per_effort = getattr(self.config, "initial_service_execution_cost_by_effort", None)
        if isinstance(per_effort, dict):
            raw = per_effort.get(effort)
            if raw is not None:
                try:
                    return max(0.0, float(raw))
                except (TypeError, ValueError):
                    pass
        return fallback

    def get_initial_quality_rating_for_effort(self, effort_tier: str = None) -> float:
        """Resolve configured initial quality-rating prior for a given effort tier."""
        effort = self.normalize_effort_tier(effort_tier)

        resolver = getattr(self.config, "get_initial_quality_rating_for_effort", None)
        if callable(resolver):
            try:
                return max(0.0, min(100.0, float(resolver(effort))))
            except (TypeError, ValueError):
                pass

        fallback = max(
            0.0,
            min(
                100.0,
                float(
                    getattr(
                        self.config,
                        "initial_quality_rating",
                        getattr(self.config, "initial_reputation_default", 50.0),
                    )
                ),
            ),
        )
        per_effort = getattr(self.config, "initial_quality_rating_by_effort", None)
        if isinstance(per_effort, dict):
            raw = per_effort.get(effort)
            if raw is not None:
                try:
                    return max(0.0, min(100.0, float(raw)))
                except (TypeError, ValueError):
                    pass
        return fallback
    
    def start_service_execution(self, effort_tier: str = None):
        """
        Begin tracking costs for a service execution.
        
        Call this before ServiceExecutor starts processing a service.
        """
        self._current_execution_llm_cost = 0.0
        self._is_executing_service = True
        self._current_execution_effort_tier = self.normalize_effort_tier(effort_tier)
        self._current_execution_input_tokens = 0
        self._current_execution_output_tokens = 0
        logger.debug(
            "📊 Started tracking service execution costs (effort=%s)",
            self._current_execution_effort_tier,
        )
    
    def add_llm_cost(
        self,
        cost_usd: float,
        context: str = "llm_call",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """
        Record LLM API cost.
        
        Args:
            cost_usd: Cost in USD
            context: Description of what the LLM call was for
        """
        self.total_llm_costs += cost_usd

        # If we're executing a service, track separately
        if self._is_executing_service:
            self._current_execution_llm_cost += cost_usd
            self._current_execution_input_tokens += int(input_tokens or 0)
            self._current_execution_output_tokens += int(output_tokens or 0)
            self.total_service_llm_costs += cost_usd
            self.service_llm_calls += 1
            self.service_input_tokens += int(input_tokens or 0)
            self.service_output_tokens += int(output_tokens or 0)
        else:
            self.total_reasoning_llm_costs += cost_usd
            self.reasoning_llm_calls += 1
            self.reasoning_input_tokens += int(input_tokens or 0)
            self.reasoning_output_tokens += int(output_tokens or 0)

        logger.debug(f"💰 LLM cost: ${cost_usd:.4f} ({context})")
    
    def add_gas_cost(self, gas_used: int, gas_price_wei: int, context: str = "transaction"):
        """
        Record blockchain gas cost using config constants.
        
        Args:
            gas_used: Gas units consumed
            gas_price_wei: Gas price in wei (from transaction receipt)
            context: Description of transaction type
        """
        gas_cost_mode = getattr(self.config, "gas_cost_mode", "receipt")

        # fixed: always use configured average gas price (good for scenario modeling)
        # receipt (default): use on-chain receipt price when available
        if gas_cost_mode == "fixed":
            gas_price_wei = int(self.config.gas_price_gwei * 1e9)
        else:
            # Some local/forked chains may return 0 effectiveGasPrice in receipts.
            # Fall back to configured simulation gas price to avoid undercounting.
            if not gas_price_wei or gas_price_wei <= 0:
                gas_price_wei = int(self.config.gas_price_gwei * 1e9)
                logger.debug(
                    f"effectiveGasPrice missing for {context}; "
                    f"falling back to config gas price {self.config.gas_price_gwei} gwei"
                )

        # Convert wei to gwei
        gas_price_gwei = gas_price_wei / 1e9
        cost_eth = (gas_used * gas_price_gwei) / 1e9
        cost_usd = cost_eth * self.config.eth_price_usd
        
        self.total_gas_costs += cost_usd
        logger.debug(f"⛽ Gas cost: ${cost_usd:.4f} ({context}, {gas_used} gas @ {gas_price_gwei:.4f} gwei)")
    
    def add_revenue(self, revenue_usd: float, context: str = "service_completion"):
        """
        Record revenue from winning an auction.
        
        Args:
            revenue_usd: Revenue in USD
            context: Description of revenue source
        """
        self.total_revenue += revenue_usd
        logger.info(f"💵 Revenue: ${revenue_usd:.4f} USD ({context})")
    
    def end_service_execution(self, effort_tier: str = None) -> float:
        """
        Complete service execution tracking and record to history.
        
        Returns:
            Service execution cost (USD)
        """
        if not self._is_executing_service:
            logger.warning("end_service_execution called but no service execution in progress")
            return 0.0
        
        execution_cost = self._current_execution_llm_cost
        resolved_effort_tier = self.normalize_effort_tier(
            effort_tier or self._current_execution_effort_tier
        )

        self.service_execution_history.append(execution_cost)
        self._execution_cost_history_by_effort[resolved_effort_tier].append(execution_cost)
        self._is_executing_service = False
        self._current_execution_effort_tier = "medium"

        global_prior = self.get_initial_execution_cost_for_effort(
            getattr(self.config, "default_effort_tier", "medium")
        )
        effort_prior = self.get_initial_execution_cost_for_effort(resolved_effort_tier)
        if self._execution_cost_ema is None:
            self._execution_cost_ema = self._update_ema(
                previous=global_prior,
                sample=execution_cost,
                alpha=self._execution_cost_ema_alpha,
            )
        else:
            self._execution_cost_ema = self._update_ema(
                previous=self._execution_cost_ema,
                sample=execution_cost,
                alpha=self._execution_cost_ema_alpha,
            )

        effort_prev = self._execution_cost_ema_by_effort.get(resolved_effort_tier)
        if effort_prev is None:
            self._execution_cost_ema_by_effort[resolved_effort_tier] = self._update_ema(
                previous=effort_prior,
                sample=execution_cost,
                alpha=self._execution_cost_ema_alpha,
            )
        else:
            self._execution_cost_ema_by_effort[resolved_effort_tier] = self._update_ema(
                previous=effort_prev,
                sample=execution_cost,
                alpha=self._execution_cost_ema_alpha,
            )
        
        logger.info(
            "📊 Service execution complete: $%.6f (effort=%s, tokens=%s+%s)",
            execution_cost,
            resolved_effort_tier,
            self._current_execution_input_tokens,
            self._current_execution_output_tokens,
        )

        self._current_execution_input_tokens = 0
        self._current_execution_output_tokens = 0

        return execution_cost
    
    def get_execution_cost_history(self) -> list:
        """
        Get raw array of past service execution costs.
        
        This is pure information relay - no interpretation or aggregation.
        BlockchainHandler decides how to use this history.
        
        Returns:
            List of execution costs (USD) in chronological order
        """
        return self.service_execution_history.copy()

    def get_execution_counts_by_effort(self) -> dict:
        """Get number of completed services by effort level."""
        return {
            effort: len(history)
            for effort, history in self._execution_cost_history_by_effort.items()
        }

    def get_average_execution_cost_by_effort(self, default=None) -> dict:
        """Return per-effort prior-seeded EMA estimates of execution costs."""
        estimates = {}

        def resolve_fallback(effort: str) -> float:
            if isinstance(default, dict):
                raw_value = default.get(effort)
                if raw_value is not None:
                    try:
                        return float(raw_value)
                    except (TypeError, ValueError):
                        pass
            elif default is not None:
                try:
                    return float(default)
                except (TypeError, ValueError):
                    pass
            return self.get_initial_execution_cost_for_effort(effort)

        for effort in self.EFFORT_LEVELS:
            history = self._execution_cost_history_by_effort.get(effort, [])
            if not history:
                estimates[effort] = resolve_fallback(effort)
                continue

            ema_value = self._execution_cost_ema_by_effort.get(effort)
            if ema_value is not None:
                estimates[effort] = float(ema_value)
            else:
                estimates[effort] = sum(history) / len(history)

        return estimates

    def record_effort_quality_feedback(self, effort_tier: str, score: float) -> None:
        """Update per-effort feedback quality statistics using Welford's algorithm."""
        effort = self.normalize_effort_tier(effort_tier)
        bounded_score = max(0.0, min(100.0, float(score)))
        stats = self._quality_stats_by_effort[effort]

        count = int(stats["count"]) + 1
        mean = float(stats["mean"])
        m2 = float(stats["m2"])

        delta = bounded_score - mean
        mean += delta / count
        delta2 = bounded_score - mean
        m2 += delta * delta2

        stats["count"] = count
        stats["mean"] = mean
        stats["m2"] = m2

    def get_quality_stats_by_effort(self, default_score=None) -> dict:
        """Return per-effort average score and variance from on-chain feedback."""
        result = {}

        def resolve_fallback(effort: str) -> float:
            if isinstance(default_score, dict):
                raw_value = default_score.get(effort)
                if raw_value is not None:
                    try:
                        return max(0.0, min(100.0, float(raw_value)))
                    except (TypeError, ValueError):
                        pass
            elif default_score is not None:
                try:
                    return max(0.0, min(100.0, float(default_score)))
                except (TypeError, ValueError):
                    pass
            return self.get_initial_quality_rating_for_effort(effort)

        for effort in self.EFFORT_LEVELS:
            stats = self._quality_stats_by_effort[effort]
            count = int(stats["count"])
            if count <= 0:
                result[effort] = {
                    "average_rating": resolve_fallback(effort),
                    "variance": 0.0,
                    "feedback_count": 0,
                }
                continue

            mean = float(stats["mean"])
            m2 = float(stats["m2"])
            variance = m2 / count
            result[effort] = {
                "average_rating": mean,
                "variance": variance,
                "feedback_count": count,
            }

        return result

    def _get_total_non_execution_costs(self) -> float:
        """Return cumulative non-execution costs accrued so far."""
        finalized_execution_cost = sum(self.service_execution_history)
        inflight_execution_cost = self._current_execution_llm_cost if self._is_executing_service else 0.0
        total_costs = self.total_llm_costs + self.total_gas_costs
        return max(0.0, total_costs - finalized_execution_cost - inflight_execution_cost)

    def _get_historical_non_execution_costs(self) -> float:
        """Return non-execution costs accrued before the earliest active auction began."""
        if not self._active_auction_participation_costs:
            return self._get_total_non_execution_costs()

        oldest_active_cost_snapshot = min(self._active_auction_participation_costs.values())
        return max(0.0, oldest_active_cost_snapshot)

    def get_average_non_execution_cost_per_auction(self, default: float = 0.0) -> float:
        """Estimate additive non-execution overhead per participated auction.

        This amortizes finalized bidding/reasoning overhead across completed
        auction participations only. Costs from currently active participations
        are excluded so the estimate does not drift during each reasoning loop.
        """
        finalized_auctions = self.get_participated_auction_count() - self.get_active_participated_auction_count()
        if finalized_auctions <= 0:
            return max(0.0, float(default))

        if self._non_execution_cost_per_auction_ema is not None:
            return float(self._non_execution_cost_per_auction_ema)

        historical_non_execution_costs = self._get_historical_non_execution_costs()
        return historical_non_execution_costs / finalized_auctions

    def get_unrecovered_overhead_balance(self) -> float:
        """Return historical non-execution overhead not yet recovered by service margin.

        Positive service margin from completed work can pay back historical bidding
        and reasoning overhead. This method keeps that recovery accounting
        separate from direct execution costs so pricing can recover overhead
        gradually without relying on a win-rate estimate.
        """
        finalized_execution_cost = sum(self.service_execution_history)
        historical_non_execution_costs = self._get_historical_non_execution_costs()

        recovered_overhead = max(0.0, self.total_revenue - finalized_execution_cost)
        return max(0.0, historical_non_execution_costs - recovered_overhead)
    
    def get_net_balance(self) -> float:
        """
        Get current net balance (revenue - costs).

        Returns:
            Net balance in USD
        """
        total_costs = self.total_llm_costs + self.total_gas_costs
        return self.total_revenue - total_costs

    def get_cost_breakdown(self) -> dict:
        """Return cost totals with LLM spend split by regime.

        `reasoning_llm_costs` covers bidding/monitoring calls made outside any
        service-execution window; `service_llm_costs` covers calls made inside
        one (including sampled-dataset replay, which records its cost inside a
        start/end execution window). The two always sum to `llm_costs`.
        """
        return {
            "llm_costs": self.total_llm_costs,
            "reasoning_llm_costs": self.total_reasoning_llm_costs,
            "service_llm_costs": self.total_service_llm_costs,
            "reasoning_llm_calls": self.reasoning_llm_calls,
            "service_llm_calls": self.service_llm_calls,
            "reasoning_input_tokens": self.reasoning_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "service_input_tokens": self.service_input_tokens,
            "service_output_tokens": self.service_output_tokens,
            "gas_costs": self.total_gas_costs,
            "total_costs": self.total_llm_costs + self.total_gas_costs,
        }

    def get_avg_gross_margin_per_win(self) -> Optional[float]:
        """Average gross margin earned on won auctions (USD per win).

        Both revenue and service execution costs accrue only on wins, so
        (total_revenue - total_execution_cost) / wins is exactly the mean
        per-win gross margin: how profitable an average win is BEFORE the
        overhead of bidding on auctions that were lost. This is deliberately
        distinct from net_balance / wins, which folds every losing auction's
        overhead into the numerator and can make a profitable win look like a
        loss. Returns None until the first completed service.
        """
        wins = len(self.service_execution_history)
        if wins <= 0:
            return None
        gross_margin_total = self.total_revenue - sum(self.service_execution_history)
        return gross_margin_total / wins

    def log_summary(self):
        """Log a summary of current financial state."""
        total_costs = self.total_llm_costs + self.total_gas_costs
        net_balance = self.get_net_balance()
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  Agent #{self.agent_id} - Financial Summary     
╠══════════════════════════════════════════════════════════╣
║  Revenue:      ${self.total_revenue:>10.4f} USD         
║  LLM Costs:    ${self.total_llm_costs:>10.4f} USD       
║  Gas Costs:    ${self.total_gas_costs:>10.4f} USD       
║  ────────────────────────────────────────────────        
║  Total Costs:  ${total_costs:>10.4f} USD                
║  NET BALANCE:  ${net_balance:>10.4f} USD                
╚══════════════════════════════════════════════════════════╝
        """)
