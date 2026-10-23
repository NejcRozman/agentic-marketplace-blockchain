"""Collects a run's metrics into `experiment_metrics.json`.

Auction and reputation outcomes come from the chain; bidding, execution, evaluation
and costs come from the agents' runtime status snapshots and final status files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from web3 import Web3

logger = logging.getLogger(__name__)


#: How many consumer status snapshots to keep in `experiment_metrics.json`;
#: 1 keeps the final one only (see `record_consumer_status_snapshot`).
CONSUMER_SNAPSHOT_RETENTION = 1


def _optional_float(value: Any) -> Optional[float]:
    """Float conversion that preserves 'not reported' as None rather than 0.0."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MetricsCollector:
    """Collects one run's metrics."""

    def __init__(self, experiment_id: str, metrics_dir: Path, logs_dir: Path):
        self.experiment_id = experiment_id
        self.metrics_dir = metrics_dir
        self.logs_dir = logs_dir

        self.metrics = {
            "experiment_id": experiment_id,
            "success": False,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,

            # What produced this run; set by the experiment runner at the start.
            "provenance": {},

            # One record per auction; the primary data.
            "auctions": [],

            # Counts only; rates and averages are computed in the analysis.
            "summary": {
                "total_auctions": 0,
                "completed_auctions": 0,
                "failed_auctions": 0,
                "total_bids_attempted": 0,
                "total_bids_on_chain": 0
            },

            # Snapshot schema: {after_auction, timestamp, reputations: {provider_id:
            # {score, feedback_count}}, source?}
            "reputation_evolution": [],

            "provider_financials": {
                "by_provider": {},
                "evolution": [],
                "runtime_snapshots": []
            },

            # Runtime snapshots, polled by the experiment runner.
            "runtime": {
                "consumer_status_snapshots": [],
                "provider_status_snapshots": []
            },

            "timing": {
                "phase_durations": {},
                "execution_times": [],
                "evaluation_times": []
            },

            "blockchain": {
                "total_transactions": 0,
                "failed_transactions": 0,
                "gas_used_per_tx": []
            },

            "errors": {
                "total_errors": 0,
                "total_warnings": 0,
                "critical_issues": [],
                "bid_failures": []
            }
        }

    def record_provenance(self, provenance: Dict[str, Any]):
        """Store the run's provenance record (see experiment_runner.build_provenance).

        Secrets must be redacted by the caller.
        """
        self.metrics["provenance"] = dict(provenance or {})

    def set_start_time(self, start_time: datetime):
        """Record experiment start time."""
        self.metrics["start_time"] = start_time.isoformat()

    def set_end_time(self, end_time: datetime):
        """Record experiment end time."""
        self.metrics["end_time"] = end_time.isoformat()

    def set_contracts(self, reverse_auction: str, payment_token: str,
                     identity_registry: str, reputation_registry: str):
        """Record deployed contract addresses."""
        self.metrics["contracts"] = {
            "reverse_auction": reverse_auction,
            "payment_token": payment_token,
            "identity_registry": identity_registry,
            "reputation_registry": reputation_registry
        }

    @staticmethod
    def _normalize_provider_profile(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize provider subtype metadata for storage and joins."""
        if not isinstance(profile, dict):
            return {}

        provider_id = profile.get("provider_id")
        try:
            provider_id = int(provider_id) if provider_id is not None else None
        except Exception:
            provider_id = None

        return {
            "provider_id": provider_id,
            "group_name": profile.get("group_name") or "default",
            "architecture": profile.get("architecture"),
            "reasoning_mode": profile.get("reasoning_mode"),
            "heuristic_strategy": profile.get("heuristic_strategy"),
            "heuristic_min_margin": profile.get("heuristic_min_margin"),
            "heuristic_max_margin": profile.get("heuristic_max_margin"),
            "blockchain_llm": profile.get("blockchain_llm"),
            "service_llm": profile.get("service_llm"),
        }

    def _provider_profile_map(self) -> Dict[int, Dict[str, Any]]:
        """Return provider subtype metadata keyed by provider id."""
        profiles = self.metrics.get("agents", {}).get("provider_profiles", []) or []
        out: Dict[int, Dict[str, Any]] = {}
        for profile in profiles:
            normalized = self._normalize_provider_profile(profile)
            provider_id = normalized.get("provider_id")
            if provider_id is None:
                continue
            out[int(provider_id)] = normalized
        return out

    def set_agents(self, consumer_id: int, provider_ids: List[int], provider_profiles: Optional[List[Dict[str, Any]]] = None):
        """Record agent IDs."""
        normalized_profiles = [
            self._normalize_provider_profile(profile)
            for profile in (provider_profiles or [])
            if self._normalize_provider_profile(profile)
        ]

        mix_counts: Dict[str, int] = {}
        for profile in normalized_profiles:
            label = profile.get("group_name") or "default"
            mix_counts[label] = mix_counts.get(label, 0) + 1

        self.metrics["agents"] = {
            "consumer_id": consumer_id,
            "provider_ids": provider_ids,
            "provider_count": len(provider_ids),
            "provider_profiles": normalized_profiles,
            "provider_mix": mix_counts,
        }

    def collect_auction_from_blockchain(self, w3: Web3, auction_contract, auction_id: int, reputation_contract=None) -> Dict[str, Any]:
        """Build one auction's record from the chain and the runtime snapshots.

        The record holds the auction's terms, every bid attempt (placed or failed), the
        winner, the execution, the evaluation, the reputation change and the timing.
        """
        auction_data = {
            "auction_id": auction_id,
            "timestamp_created": None,
            "timestamp_ended": None,  # From the AuctionEnded event
            "service_cid": None,
            "service_source_mode": "live_generation",
            "sampled_dataset_id": None,
            "sampled_requirement_id": None,
            "sampled_tier": None,
            "sampled_repeat_idx": None,
            "sampled_artifact_dir": None,
            "budget": 0,
            "duration": 0,
            "commit_duration": 0,
            "reveal_duration": 0,
            "creator_id": None,

            "bids_attempted": [],  # Every bid a provider tried to place
            "bids_on_chain": [],  # Only bids placed on chain
            "bid_failures": [],  # Failed attempts, with reasons
            "winner_id": None,
            "winning_bid_amount": 0,
            "bid_count": 0,

            "service_executed": False,
            "execution_start": None,
            "execution_end": None,
            "execution_duration_seconds": 0,
            "prompts_answered": 0,
            "prompts_total": 0,

            "quality_rating": None,
            "quality_method": "unknown",  # "tools", "fallback", or "failed"
            "quality_details": {},
            "evaluation_duration_seconds": 0,

            "feedback_submitted": False,
            "feedback_rating": None,
            "reputation_before": None,
            "reputation_after": None,
            "reputation_change": 0,

            "timing": {
                "creation_to_first_bid": 0,
                "first_bid_to_close": 0,
                "close_to_execution_start": 0,
                "execution": 0,
                "evaluation": 0,
                "feedback_submission": 0,
                "total_auction_cycle": 0
            },

            "status": "unknown",  # "completed", "failed", "timeout"
            "errors": []
        }

        try:
            # 1. Auction terms, from getAuctionDetails(), which also returns the
            # eligibleAgentIds array.
            auction_info = auction_contract.functions.getAuctionDetails(auction_id).call()
            # Auction struct: (id, buyer, serviceDescriptionCid, maxPrice, commitDuration, startTime,
            #                  eligibleAgentIds, winningAgentId, winningBid, isActive, isCompleted,
            #                  escrowAmount, reputationWeight, revealDuration)
            auction_data["creator_id"] = auction_info[1]  # buyer address
            auction_data["service_cid"] = auction_info[2]
            auction_data["budget"] = auction_info[3]
            auction_data["commit_duration"] = auction_info[4]
            auction_data["reveal_duration"] = auction_info[13]
            auction_data["duration"] = auction_info[4] + auction_info[13]  # commit + reveal
            auction_data["timestamp_created"] = auction_info[5]
            auction_data["eligible_agent_ids"] = list(auction_info[6])

            # 2. On-chain bids.
            bid_count = auction_contract.functions.getBidCount(auction_id).call()
            auction_data["bid_count"] = bid_count

            for i in range(bid_count):
                # Bid struct: (provider, agentId, amount, timestamp, reputation, score, reputationDecimals)
                bid_info = auction_contract.functions.auctionBids(auction_id, i).call()
                rep_raw = bid_info[4]
                rep_decimals = bid_info[6]
                rep_normalized = rep_raw / (10 ** rep_decimals) if rep_decimals else rep_raw
                auction_data["bids_on_chain"].append({
                    "provider_address": bid_info[0],
                    "agent_id": bid_info[1],
                    "amount": bid_info[2],
                    "timestamp": bid_info[3],
                    "reputation": rep_normalized,  # normalised to 0-100
                    "reputation_raw": rep_raw,
                    "reputation_decimals": rep_decimals,
                    "score": bid_info[5],  # weighted score (0-10000)
                })

            # 3. Winner, from the auction struct: [7] winningAgentId, [8] winningBid,
            # [9] isActive, [10] isCompleted.
            winning_agent_id = auction_info[7]  # 0 if no bids
            winning_bid_from_struct = auction_info[8]

            # The separate winningBid mapping is more reliable than the struct.
            try:
                winning_bid_from_mapping = auction_contract.functions.winningBid(auction_id).call()
            except:
                winning_bid_from_mapping = 0

            winning_bid = winning_bid_from_mapping if winning_bid_from_mapping > 0 else winning_bid_from_struct

            auction_data["winner_id"] = int(winning_agent_id) if winning_agent_id else None
            auction_data["winning_bid_amount"] = int(winning_bid) if winning_bid else 0

            is_active = auction_info[9]
            is_completed = auction_info[10]

            # 3a. Actual end time, from the AuctionEnded event.
            try:
                # Scoped to a recent block window: forked RPC endpoints reject eth_getLogs
                # ranges wider than about 10000 blocks.
                latest_block = w3.eth.block_number
                event_filter = auction_contract.events.AuctionEnded.create_filter(
                    from_block=max(0, latest_block - 9000),
                    argument_filters={'auctionId': auction_id}
                )
                events = event_filter.get_all_entries()
                if events:
                    block = w3.eth.get_block(events[0]['blockNumber'])
                    auction_data["timestamp_ended"] = block['timestamp']
                else:
                    # Fall back to the nominal end time.
                    auction_data["timestamp_ended"] = auction_data["timestamp_created"] + auction_data["duration"]
            except Exception as e:
                logger.warning(f"Could not fetch AuctionEnded event for auction {auction_id}: {e}")
                # Fall back to the nominal end time.
                auction_data["timestamp_ended"] = auction_data["timestamp_created"] + auction_data["duration"]

            # 4. Attempted and failed bids, from the runtime snapshots and the chain.
            runtime_bid_failures = self._runtime_bid_failures_for_auction(auction_id)

            for failure in runtime_bid_failures:
                attempt_amount = failure.get("amount")
                auction_data["bids_attempted"].append({
                    "provider_id": failure.get("provider_id"),
                    "amount": attempt_amount,
                    "source": "runtime_status"
                })
            auction_data["bid_failures"].extend(runtime_bid_failures)

            for bid in auction_data["bids_on_chain"]:
                auction_data["bids_attempted"].append({
                    "provider_id": bid.get("agent_id"),
                    "amount": bid.get("amount"),
                    "source": "on_chain"
                })

            # Keep attempts from eligible providers only.
            eligible_provider_ids = auction_data.get("eligible_agent_ids", [])
            if eligible_provider_ids:
                auction_data["bids_attempted"] = [
                    b for b in auction_data["bids_attempted"]
                    if b.get("provider_id") in eligible_provider_ids
                ]
                auction_data["bid_failures"] = [
                    b for b in auction_data["bid_failures"]
                    if b.get("provider_id") in eligible_provider_ids
                ]

            # De-duplicate, preserving order.
            deduped_attempts = []
            seen_attempts = set()
            for bid in auction_data["bids_attempted"]:
                key = (bid.get("provider_id"), bid.get("amount"))
                if key in seen_attempts:
                    continue
                seen_attempts.add(key)
                deduped_attempts.append(bid)
            auction_data["bids_attempted"] = deduped_attempts

            # De-duplicate, preserving order.
            deduped_failures = []
            seen_failures = set()
            for failure in auction_data["bid_failures"]:
                key = (
                    failure.get("provider_id"),
                    failure.get("error_code"),
                    failure.get("timestamp"),
                    failure.get("amount")
                )
                if key in seen_failures:
                    continue
                seen_failures.add(key)
                deduped_failures.append(failure)
            auction_data["bid_failures"] = deduped_failures

            # 5. Execution, from the providers' runtime snapshots.
            if auction_data["winner_id"]:
                runtime_job = self._runtime_job_outcome(auction_data["winner_id"], auction_id)
                if runtime_job:
                    status = str(runtime_job.get("status", "")).lower()
                    auction_data["service_executed"] = status in {"completed", "processing", "completing", "delivering"}
                    auction_data["feedback_submitted"] = status == "completed"

                    llm_cost = runtime_job.get("llm_cost")
                    if llm_cost is not None:
                        auction_data["quality_details"]["winner_llm_cost"] = llm_cost
                    if runtime_job.get("error"):
                        auction_data["errors"].append(f"provider_job_error: {runtime_job.get('error')}")

            # 5b. Completion and evaluation, from the consumer's runtime snapshot.
            runtime_consumer_auction = self._runtime_consumer_auction(auction_id)
            if runtime_consumer_auction:
                self._apply_runtime_consumer_auction(auction_data, runtime_consumer_auction)

            # 6. Reputation change of the winner.
            if auction_data["winner_id"] and reputation_contract:
                try:
                    from agents.common.reputation_lookup import fetch_reputation_summary, reputation_to_float

                    feedback_count, rep_value, rep_decimals = fetch_reputation_summary(
                        reputation_contract, auction_data["winner_id"]
                    )
                    auction_data["reputation_after"] = (
                        reputation_to_float(rep_value, rep_decimals) if feedback_count > 0 else 0.0
                    )

                    # Reputation after = before + (rating - 50), so before = after - (rating - 50).
                    if auction_data["reputation_before"] is None and auction_data["feedback_rating"]:
                        expected_change = auction_data["feedback_rating"] - 50
                        auction_data["reputation_before"] = auction_data["reputation_after"] - expected_change

                    if auction_data["reputation_before"] is not None:
                        auction_data["reputation_change"] = auction_data["reputation_after"] - auction_data["reputation_before"]
                except Exception as e:
                    logger.error(f"Failed to collect reputation for winner {auction_data['winner_id']}: {e}", exc_info=True)

            # 7. Timing.
            self._calculate_auction_timing(auction_data)

            # 8. Status, from the chain state and the runtime signals.
            if is_completed:
                # Service delivered and feedback received.
                auction_data["status"] = "completed"
            elif auction_data["feedback_submitted"]:
                auction_data["status"] = "completed"
            elif not is_active and auction_data["winner_id"] and auction_data["service_executed"]:
                # Executed; completion or feedback not yet final.
                auction_data["status"] = "incomplete"
            elif not is_active and auction_data["winner_id"]:
                # Winner, not executed yet.
                auction_data["status"] = "incomplete"
            elif not is_active and (auction_data["winner_id"] == 0 or auction_data["winner_id"] is None):
                # No bids, no winner.
                auction_data["status"] = "failed"
            elif len(auction_data["errors"]) > 0:
                auction_data["status"] = "error"
            else:
                auction_data["status"] = "incomplete"

        except Exception as e:
            logger.error(f"Error collecting auction {auction_id}: {e}")
            auction_data["errors"].append(str(e))
            auction_data["status"] = "error"

        self._validate_auction_data(auction_data)

        return auction_data

    def _calculate_auction_timing(self, auction_data: Dict[str, Any]):
        """Calculate timing metrics from raw timestamps."""
        timing = auction_data["timing"]

        try:
            if auction_data["bids_on_chain"] and len(auction_data["bids_on_chain"]) > 0:
                first_bid_ts = auction_data["bids_on_chain"][0]["timestamp"]
                timing["creation_to_first_bid"] = first_bid_ts - auction_data["timestamp_created"]

                # First bid to close, from the AuctionEnded time.
                if auction_data.get("timestamp_ended"):
                    timing["first_bid_to_close"] = auction_data["timestamp_ended"] - first_bid_ts
                else:
                    # Fall back to the last bid's time.
                    last_bid_ts = auction_data["bids_on_chain"][-1]["timestamp"]
                    timing["first_bid_to_close"] = last_bid_ts - first_bid_ts

            execution_start = self._parse_runtime_datetime(auction_data.get("execution_start"))
            execution_end = self._parse_runtime_datetime(auction_data.get("execution_end"))
            execution_duration = int(auction_data.get("execution_duration_seconds", 0) or 0)
            evaluation_duration = int(auction_data.get("evaluation_duration_seconds", 0) or 0)

            # Execution timing from runtime timestamps when available.
            execution_start_ts = None
            if execution_start and execution_end:
                timing["execution"] = max(0, int((execution_end - execution_start).total_seconds()))
                execution_start_ts = int(execution_start.timestamp())
            elif execution_duration > 0:
                timing["execution"] = execution_duration
                if execution_end:
                    execution_start_ts = int(execution_end.timestamp()) - timing["execution"]
                elif execution_start:
                    execution_start_ts = int(execution_start.timestamp())

            # Close to execution start.
            if execution_start_ts is not None and auction_data.get("timestamp_ended"):
                timing["close_to_execution_start"] = execution_start_ts - auction_data["timestamp_ended"]

                # Negative when the provider started before the AuctionEnded event, or with
                # clock skew between the chain and the local system.
            elif execution_start_ts is not None and auction_data["bids_on_chain"]:
                # Fall back to the last bid's time.
                last_bid_ts = auction_data["bids_on_chain"][-1]["timestamp"]
                timing["close_to_execution_start"] = execution_start_ts - last_bid_ts

            if evaluation_duration > 0:
                timing["evaluation"] = evaluation_duration

            # Total cycle time, only when every component is present.
            if timing["creation_to_first_bid"] > 0 and timing["execution"] > 0:
                timing["total_auction_cycle"] = (
                    timing["creation_to_first_bid"] +
                    timing["first_bid_to_close"] +
                    timing["close_to_execution_start"] +
                    timing["execution"] +
                    timing["evaluation"] +
                    timing["feedback_submission"]
                )
        except Exception as e:
            logger.warning(f"Error calculating timing metrics: {e}")

    @staticmethod
    def _parse_runtime_datetime(value: Any) -> Optional[datetime]:
        """Parse timestamps emitted by agent status snapshots."""
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _runtime_consumer_auction(self, auction_id: int) -> Optional[Dict[str, Any]]:
        """Return the latest consumer runtime snapshot for a specific auction."""
        snapshots = self.metrics.get("runtime", {}).get("consumer_status_snapshots", [])
        for snap in reversed(snapshots):
            status = snap.get("status") if isinstance(snap, dict) else None
            auctions = status.get("auctions") if isinstance(status, dict) else None
            if not isinstance(auctions, list):
                continue
            for auction in reversed(auctions):
                try:
                    if int(auction.get("auction_id", -1)) == int(auction_id):
                        return auction
                except Exception:
                    continue
        return None

    def _apply_runtime_consumer_auction(self, auction_data: Dict[str, Any], runtime_auction: Dict[str, Any]):
        """Merge consumer-side completion and evaluation data into an auction record."""
        ended_at = self._parse_runtime_datetime(runtime_auction.get("ended_at"))
        completed_at = self._parse_runtime_datetime(runtime_auction.get("completed_at"))
        evaluation_started_at = self._parse_runtime_datetime(runtime_auction.get("evaluation_started_at"))
        evaluation_completed_at = self._parse_runtime_datetime(runtime_auction.get("evaluation_completed_at"))
        feedback_submitted_at = self._parse_runtime_datetime(runtime_auction.get("feedback_submitted_at"))

        execution_duration = int(runtime_auction.get("execution_duration_seconds", 0) or 0)
        if execution_duration <= 0 and ended_at and completed_at:
            execution_duration = max(0, int((completed_at - ended_at).total_seconds()))

        evaluation_duration = int(runtime_auction.get("evaluation_duration_seconds", 0) or 0)
        if evaluation_duration <= 0 and evaluation_started_at and evaluation_completed_at:
            evaluation_duration = max(0, int((evaluation_completed_at - evaluation_started_at).total_seconds()))

        if ended_at and not auction_data.get("execution_start"):
            auction_data["execution_start"] = ended_at.isoformat(timespec="seconds")
        if completed_at and not auction_data.get("execution_end"):
            auction_data["execution_end"] = completed_at.isoformat(timespec="seconds")

        auction_data["service_executed"] = bool(runtime_auction.get("service_executed")) or bool(auction_data.get("service_executed"))
        auction_data["feedback_submitted"] = bool(runtime_auction.get("feedback_submitted")) or bool(auction_data.get("feedback_submitted"))

        if execution_duration > 0:
            auction_data["execution_duration_seconds"] = execution_duration
        if evaluation_duration > 0:
            auction_data["evaluation_duration_seconds"] = evaluation_duration

        quality_rating = runtime_auction.get("quality_rating")
        if quality_rating is not None:
            auction_data["quality_rating"] = quality_rating

        feedback_rating = runtime_auction.get("feedback_rating")
        if feedback_rating is not None:
            auction_data["feedback_rating"] = feedback_rating

        service_source_mode = runtime_auction.get("service_source_mode")
        if service_source_mode:
            auction_data["service_source_mode"] = service_source_mode

        for field_name in (
            "sampled_dataset_id",
            "sampled_requirement_id",
            "sampled_tier",
            "sampled_repeat_idx",
            "sampled_artifact_dir",
        ):
            if runtime_auction.get(field_name) is not None:
                auction_data[field_name] = runtime_auction.get(field_name)

        quality_method = runtime_auction.get("quality_method")
        if quality_method:
            auction_data["quality_method"] = quality_method

        quality_scores = runtime_auction.get("quality_scores")
        if isinstance(quality_scores, dict) and quality_scores:
            auction_data.setdefault("quality_details", {})["quality_scores"] = quality_scores

        quality_explanations = runtime_auction.get("quality_explanations")
        if isinstance(quality_explanations, list) and quality_explanations:
            auction_data.setdefault("quality_details", {})["explanations"] = quality_explanations

        if evaluation_duration > 0:
            auction_data["timing"]["evaluation"] = evaluation_duration
        if feedback_submitted_at and evaluation_completed_at:
            auction_data["timing"]["feedback_submission"] = max(
                0,
                int((feedback_submitted_at - evaluation_completed_at).total_seconds()),
            )

        runtime_error = runtime_auction.get("error")
        if runtime_error:
            auction_data["errors"].append(f"consumer_auction_error: {runtime_error}")

    def _validate_auction_data(self, auction_data: Dict[str, Any]):
        """Log a warning for every inconsistency between an auction record and the chain.

        Checks that every attempted and on-chain bid comes from an eligible provider, that
        the on-chain bid count matches the contract, that the winner placed a bid, that
        there are at least as many attempts as on-chain bids, and that timings are plausible.
        """
        auction_id = auction_data["auction_id"]
        eligible = set(auction_data.get("eligible_agent_ids", []))

        # 1. Attempted bids come from eligible providers.
        for bid in auction_data["bids_attempted"]:
            provider_id = bid.get("provider_id")
            if provider_id not in eligible:
                logger.error(
                    f"VALIDATION ERROR: Auction {auction_id} - "
                    f"Provider {provider_id} in bids_attempted but not in eligible_agent_ids {eligible}. "
                    f"This indicates a regex bug or incorrect eligibility filtering."
                )
                auction_data["errors"].append(f"Invalid bid attempt from ineligible provider {provider_id}")

        # 2. On-chain bids come from eligible providers.
        for bid in auction_data["bids_on_chain"]:
            agent_id = bid.get("agent_id")
            if agent_id and agent_id not in eligible:
                logger.error(
                    f"VALIDATION ERROR: Auction {auction_id} - "
                    f"Provider {agent_id} has on-chain bid but not in eligible_agent_ids {eligible}. "
                    f"This indicates a blockchain state inconsistency."
                )
                auction_data["errors"].append(f"On-chain bid from ineligible provider {agent_id}")

        # 3. On-chain bid count matches the contract.
        if len(auction_data["bids_on_chain"]) != auction_data["bid_count"]:
            logger.warning(
                f"VALIDATION WARNING: Auction {auction_id} - "
                f"Collected {len(auction_data['bids_on_chain'])} on-chain bids but "
                f"contract reports bid_count={auction_data['bid_count']}"
            )

        # 4. The winner placed an on-chain bid.
        if auction_data["winner_id"]:
            winner_bid = next(
                (b for b in auction_data["bids_on_chain"] if b.get("agent_id") == auction_data["winner_id"]),
                None
            )
            if not winner_bid:
                logger.warning(
                    f"VALIDATION WARNING: Auction {auction_id} - "
                    f"Winner {auction_data['winner_id']} has no on-chain bid recorded"
                )

        # 5. At least as many attempts as on-chain bids.
        attempted_count = len(auction_data["bids_attempted"])
        onchain_count = len(auction_data["bids_on_chain"])
        if attempted_count < onchain_count:
            logger.warning(
                f"VALIDATION WARNING: Auction {auction_id} - "
                f"Only {attempted_count} bids attempted but {onchain_count} made it on-chain. "
                f"This suggests incomplete log parsing."
            )

        # 6. No excessive number of attempts.
        if attempted_count > onchain_count * 10:
            logger.error(
                f"VALIDATION ERROR: Auction {auction_id} - "
                f"{attempted_count} bids attempted but only {onchain_count} on-chain. "
                f"Ratio of {attempted_count/max(onchain_count,1):.1f}x suggests phantom data from regex bug."
            )
            auction_data["errors"].append(
                f"Excessive bid attempts: {attempted_count} attempted vs {onchain_count} on-chain"
            )

        # 7. No large negative timings.
        timing = auction_data.get("timing", {})
        close_to_exec = timing.get("close_to_execution_start", 0)

        # Below 60 s, a negative value can come from clock skew or early execution;
        # above, it is a calculation error.
        if close_to_exec < -60:
            logger.warning(
                f"VALIDATION WARNING: Auction {auction_id} - "
                f"Execution started {abs(close_to_exec)}s before auction end. "
                f"This may indicate clock skew or incorrect timestamp parsing."
            )

        total_cycle = timing.get("total_auction_cycle", 0)
        if total_cycle < 0:
            logger.error(
                f"VALIDATION ERROR: Auction {auction_id} - "
                f"Negative total_auction_cycle ({total_cycle}s). "
                f"This indicates a timing calculation bug."
            )
            auction_data["errors"].append(f"Negative total cycle time: {total_cycle}s")

    def collect_system_completeness(self) -> Dict[str, Any]:
        """Build system completeness from runtime snapshots + auction outcomes."""
        completeness = {
            "auction_created": False,
            "bids_received": 0,
            "winner_selected": False,
            "service_executed": False,
            "service_evaluated": False,
            "feedback_submitted": False,
            "reputation_updated": False
        }

        try:
            auctions = self.metrics.get("auctions", [])
            consumer_snaps = self.metrics.get("runtime", {}).get("consumer_status_snapshots", [])
            provider_snaps = self.metrics.get("runtime", {}).get("provider_status_snapshots", [])

            completed_auctions = [a for a in auctions if str(a.get("status", "")).lower() == "completed"]
            completeness["auction_created"] = len(auctions) > 0 or len(consumer_snaps) > 0
            completeness["bids_received"] = sum(len(a.get("bids_on_chain", [])) for a in auctions)
            completeness["winner_selected"] = any((a.get("winner_id") or 0) > 0 for a in auctions)
            completeness["service_executed"] = any(bool(a.get("service_executed")) for a in auctions)
            completeness["service_evaluated"] = len(completed_auctions) > 0
            completeness["feedback_submitted"] = len(completed_auctions) > 0

            # Reputation is updated if a final snapshot or a completed auction exists.
            final_reputation = [
                s for s in self.metrics.get("reputation_evolution", [])
                if str(s.get("after_auction")) == "final"
            ]
            completeness["reputation_updated"] = bool(final_reputation) or len(completed_auctions) > 0

            # Completed jobs in the provider snapshots count as execution even when auction
            # extraction lags.
            if not completeness["service_executed"]:
                for snap in provider_snaps:
                    jobs = (snap.get("status") or {}).get("jobs") or {}
                    if jobs.get("completed"):
                        completeness["service_executed"] = True
                        break

        except Exception as e:
            logger.error(f"Error collecting system completeness: {e}")

        self.metrics["system_completeness"] = completeness
        return completeness

    def collect_errors(self) -> Dict[str, Any]:
        """Collect errors from runtime snapshots and auction-level collection issues."""
        errors = {
            "total_errors": 0,
            "total_warnings": 0,
            "critical_issues": [],  # Only blocking/important errors
            "bid_failures": []  # Also in the auction records
        }

        try:
            for auction in self.metrics.get("auctions", []):
                for issue in auction.get("errors", []) or []:
                    errors["total_errors"] += 1
                    errors["critical_issues"].append({
                        "source": "auction",
                        "auction_id": auction.get("auction_id"),
                        "message": str(issue)[:300],
                        "category": self._categorize_error(str(issue))
                    })

            for snap in self.metrics.get("runtime", {}).get("provider_status_snapshots", []):
                provider_id = snap.get("provider_id")
                status = snap.get("status") or {}
                bidding = status.get("bidding") if isinstance(status, dict) else {}
                failures = bidding.get("recent_failures") if isinstance(bidding, dict) else []
                for failure in failures or []:
                    errors["total_errors"] += 1
                    errors["bid_failures"].append({
                        "provider_id": provider_id,
                        "auction_id": failure.get("auction_id"),
                        "amount": failure.get("bid_amount"),
                        "error_code": failure.get("error_code"),
                        "reason": failure.get("error"),
                        "timestamp": failure.get("timestamp"),
                        "source": "runtime_status"
                    })

                status_error = status.get("last_error") if isinstance(status, dict) else None
                if status_error:
                    errors["total_errors"] += 1
                    errors["critical_issues"].append({
                        "source": "provider_status",
                        "provider_id": provider_id,
                        "message": str(status_error)[:300],
                        "category": self._categorize_error(str(status_error))
                    })

        except Exception as e:
            logger.error(f"Error collecting errors: {e}")

        self.metrics["errors"] = errors
        return errors

    def determine_success(self) -> bool:
        """True if every critical phase of the run completed."""
        completeness = self.collect_system_completeness()

        success = (
            completeness.get("auction_created", False) and
            completeness.get("bids_received", 0) >= 1 and
            completeness.get("winner_selected", False) and
            completeness.get("service_executed", False) and
            completeness.get("service_evaluated", False) and
            completeness.get("feedback_submitted", False) and
            self.metrics.get("blockchain", {}).get("failed_transactions", 0) == 0
        )

        self.metrics["success"] = success
        return success

    def calculate_duration(self):
        """Total run duration."""
        if self.metrics["start_time"] and self.metrics["end_time"]:
            start = datetime.fromisoformat(self.metrics["start_time"])
            end = datetime.fromisoformat(self.metrics["end_time"])
            self.metrics["duration_seconds"] = int((end - start).total_seconds())

    def save_metrics(self) -> Path:
        """Write the collected metrics to JSON and return the file's path."""
        self.calculate_duration()
        self._derive_blockchain_totals_from_auctions()
        self.collect_provider_financials()

        metrics_file = self.metrics_dir / "experiment_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Comprehensive metrics saved to {metrics_file}")
        return metrics_file

    def record_consumer_status_snapshot(self, timestamp: str, status: Dict[str, Any], source: str = "runtime_status"):
        """Record the consumer's runtime status, keeping only the most recent.

        Each snapshot re-serialises the consumer's whole auction list, so keeping all of
        them made up 85% of a metrics file. The last one is kept rather than none because
        `_runtime_consumer_auction` merges consumer-side completion and evaluation data from
        it. In 80 runs checked, the final snapshot contained every auction seen in any
        snapshot; raise ``CONSUMER_SNAPSHOT_RETENTION`` if a consumer ever prunes completed
        auctions from its status.
        """
        snapshots = self.metrics.setdefault("runtime", {}).setdefault("consumer_status_snapshots", [])
        payload = {
            "timestamp": timestamp,
            "status": status,
            "source": source,
        }
        if snapshots and snapshots[-1].get("status") == payload["status"]:
            return
        snapshots.append(payload)
        if CONSUMER_SNAPSHOT_RETENTION > 0 and len(snapshots) > CONSUMER_SNAPSHOT_RETENTION:
            del snapshots[:-CONSUMER_SNAPSHOT_RETENTION]

    def record_provider_status_snapshot(self, provider_id: int, timestamp: str, status: Dict[str, Any], provider_profile: Optional[Dict[str, Any]] = None, source: str = "runtime_status"):
        """Append provider runtime status snapshot with lightweight deduplication."""
        snapshots = self.metrics.setdefault("runtime", {}).setdefault("provider_status_snapshots", [])
        resolved_profile = self._normalize_provider_profile(provider_profile)
        if not resolved_profile:
            resolved_profile = self._provider_profile_map().get(int(provider_id), {})
        if not resolved_profile and isinstance(status, dict):
            resolved_profile = self._normalize_provider_profile(status.get("profile"))

        payload = {
            "timestamp": timestamp,
            "provider_id": int(provider_id),
            "profile": resolved_profile,
            "status": status,
            "source": source,
        }

        for prev in reversed(snapshots):
            if int(prev.get("provider_id", -1)) != int(provider_id):
                continue
            if prev.get("status") == payload["status"]:
                return
            break

        snapshots.append(payload)

    def _runtime_bid_failures_for_auction(self, auction_id: int) -> List[Dict[str, Any]]:
        """Collect per-auction bid failures from provider runtime snapshots."""
        out: List[Dict[str, Any]] = []
        snapshots = self.metrics.get("runtime", {}).get("provider_status_snapshots", [])
        for snap in snapshots:
            provider_id = snap.get("provider_id")
            status = snap.get("status") or {}
            bidding = status.get("bidding") if isinstance(status, dict) else {}
            failures = bidding.get("recent_failures") if isinstance(bidding, dict) else []
            for failure in failures or []:
                try:
                    if int(failure.get("auction_id", -1)) != int(auction_id):
                        continue
                except Exception:
                    continue
                out.append(
                    {
                        "provider_id": int(provider_id),
                        "timestamp": failure.get("timestamp"),
                        "amount": failure.get("bid_amount"),
                        "reason": failure.get("error") or "bid_rejected",
                        "error_code": failure.get("error_code"),
                        "source": "runtime_status",
                    }
                )

        # Periodic status polling repeats failures.
        unique: List[Dict[str, Any]] = []
        seen = set()
        for row in out:
            key = (row.get("provider_id"), row.get("timestamp"), row.get("amount"), row.get("error_code"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique

    def _runtime_job_outcome(self, provider_id: int, auction_id: int) -> Optional[Dict[str, Any]]:
        """Get latest runtime job status for a provider/auction from status snapshots."""
        snapshots = [
            s for s in self.metrics.get("runtime", {}).get("provider_status_snapshots", [])
            if int(s.get("provider_id", -1)) == int(provider_id)
        ]
        for snap in reversed(snapshots):
            status = snap.get("status") or {}
            jobs = status.get("jobs") if isinstance(status, dict) else {}
            if not isinstance(jobs, dict):
                continue
            for bucket in ("active", "completed"):
                for job in jobs.get(bucket, []) or []:
                    try:
                        if int(job.get("auction_id", -1)) == int(auction_id):
                            return job
                    except Exception:
                        continue
        return None

    def _derive_blockchain_totals_from_auctions(self):
        """Derive conservative blockchain transaction counters from collected auctions."""
        auctions = self.metrics.get("auctions", [])
        total_bids = sum(len(a.get("bids_on_chain", [])) for a in auctions)
        completed = sum(1 for a in auctions if str(a.get("status", "")).lower() == "completed")

        # Lower bound known from the auction records: placed bids plus one completion
        # per completed auction.
        self.metrics["blockchain"]["total_transactions"] = int(total_bids + completed)

    def _final_total_costs(self, provider_id: int) -> Optional[float]:
        """``total_costs`` from the agent's final status file, or None.

        Written at shutdown after every job settles, so its costs match the complete chain
        revenue, unlike a periodic snapshot, which can be up to one poll interval stale.
        """
        path = self.logs_dir / f"provider_{int(provider_id)}_status.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        financials = payload.get("financials")
        if not isinstance(financials, dict) or "total_costs" not in financials:
            return None
        try:
            return float(financials["total_costs"])
        except (TypeError, ValueError):
            return None

    def _auction_revenue_by_provider(self) -> Dict[int, float]:
        """Compute ground-truth provider revenue (USD) from completed auctions and winners."""
        revenue: Dict[int, float] = {}
        for auction in self.metrics.get("auctions", []):
            if str(auction.get("status", "")).lower() != "completed":
                continue
            winner = auction.get("winner_id")
            if winner in (None, 0):
                continue
            try:
                winner_id = int(winner)
            except Exception:
                continue
            winning_bid = auction.get("winning_bid_amount", 0)
            try:
                winning_bid_usd = float(winning_bid) / 1e6
            except Exception:
                winning_bid_usd = 0.0
            revenue[winner_id] = revenue.get(winner_id, 0.0) + max(0.0, winning_bid_usd)
        return revenue

    def record_provider_financial_snapshot(
        self,
        provider_id: int,
        timestamp: str,
        revenue: float,
        llm_costs: float,
        gas_costs: float,
        total_costs: float,
        net_balance: float,
        provider_profile: Optional[Dict[str, Any]] = None,
        source: str = "runtime_status",
        reasoning_llm_costs: Optional[float] = None,
        service_llm_costs: Optional[float] = None,
        cost_breakdown: Optional[Dict[str, Any]] = None,
    ):
        """Append a provider's runtime financial snapshot, skipping exact duplicates.

        The reasoning/service split stays None when the provider did not report it, never
        0.0: a heuristic agent has zero reasoning cost, whereas an unreported split is unknown.
        """
        runtime_snapshots = self.metrics.setdefault("provider_financials", {}).setdefault("runtime_snapshots", [])
        resolved_profile = self._normalize_provider_profile(provider_profile)
        if not resolved_profile:
            resolved_profile = self._provider_profile_map().get(int(provider_id), {})

        snapshot = {
            "timestamp": timestamp,
            "provider_id": int(provider_id),
            "profile": resolved_profile,
            "revenue": float(revenue),
            "llm_costs": float(llm_costs),
            "gas_costs": float(gas_costs),
            "total_costs": float(total_costs),
            "net_balance": float(net_balance),
            "source": source,
        }
        if reasoning_llm_costs is not None:
            snapshot["reasoning_llm_costs"] = float(reasoning_llm_costs)
        if service_llm_costs is not None:
            snapshot["service_llm_costs"] = float(service_llm_costs)
        # Call and token counts change on every snapshot, so they are left out of the
        # duplicate check.
        snapshot.update(cost_breakdown or {})

        for prev in reversed(runtime_snapshots):
            if int(prev.get("provider_id", -1)) != int(provider_id):
                continue
            if (
                abs(float(prev.get("revenue", 0.0)) - snapshot["revenue"]) < 1e-9
                and abs(float(prev.get("llm_costs", 0.0)) - snapshot["llm_costs"]) < 1e-9
                and abs(float(prev.get("gas_costs", 0.0)) - snapshot["gas_costs"]) < 1e-9
                and abs(float(prev.get("total_costs", 0.0)) - snapshot["total_costs"]) < 1e-9
                and abs(float(prev.get("net_balance", 0.0)) - snapshot["net_balance"]) < 1e-9
            ):
                return
            break

        runtime_snapshots.append(snapshot)

    def collect_provider_financials(self) -> Dict[str, Any]:
        """Finalize provider balance data from runtime snapshots with auction reconciliation."""
        runtime_snapshots = self.metrics.get("provider_financials", {}).get("runtime_snapshots", [])

        financials = {
            "by_provider": {},
            "evolution": list(runtime_snapshots),
            "runtime_snapshots": list(runtime_snapshots),
        }

        auction_revenue = self._auction_revenue_by_provider()

        provider_ids = self.metrics.get("agents", {}).get("provider_ids", [])
        provider_ids = [int(pid) for pid in provider_ids] if provider_ids else []

        # Take provider ids from the runtime snapshots if none are registered.
        if not provider_ids:
            provider_ids = sorted(
                {
                    int(row.get("provider_id"))
                    for row in financials["evolution"]
                    if row.get("provider_id") is not None
                }
            )

        for provider_id in provider_ids:
            snapshots = [
                row for row in financials["evolution"]
                if int(row.get("provider_id", -1)) == int(provider_id)
            ]
            if not snapshots:
                continue

            last = snapshots[-1]
            revenue_from_auctions = float(auction_revenue.get(provider_id, 0.0))
            revenue_from_runtime = float(last.get("revenue", 0.0))
            revenue_discrepancy = revenue_from_runtime - revenue_from_auctions

            # Revenue from completed auctions is authoritative.
            reconciled_revenue = revenue_from_auctions

            # Costs must describe the same instant as the revenue, which is complete; the
            # last periodic snapshot can be up to one poll interval stale and biased
            # net_balance upward. Prefer the final status file written at shutdown; fall
            # back to the snapshot and record the gap.
            costs_from_snapshot = float(last.get("total_costs", 0.0))
            costs_from_status = self._final_total_costs(provider_id)
            total_costs = (
                costs_from_status if costs_from_status is not None else costs_from_snapshot
            )
            costs_discrepancy = (
                None if costs_from_status is None
                else costs_from_status - costs_from_snapshot
            )
            if costs_discrepancy is not None and abs(costs_discrepancy) > 1e-6:
                logger.warning(
                    "Provider %s cost mismatch: snapshot=%.6f, final status=%.6f "
                    "(diff=%.6f). Using the final status file.",
                    provider_id, costs_from_snapshot, costs_from_status, costs_discrepancy,
                )
            reconciled_net_balance = reconciled_revenue - total_costs

            if abs(revenue_discrepancy) > 1e-6:
                logger.warning(
                    "Provider %s revenue mismatch: runtime=%.6f, auctions=%.6f (diff=%.6f). Using auction-derived revenue.",
                    provider_id,
                    revenue_from_runtime,
                    revenue_from_auctions,
                    revenue_discrepancy,
                )

            financials["by_provider"][provider_id] = {
                "profile": last.get("profile") or self._provider_profile_map().get(int(provider_id), {}),
                "revenue": reconciled_revenue,
                "revenue_from_auctions": revenue_from_auctions,
                "revenue_from_runtime": revenue_from_runtime,
                "revenue_discrepancy": revenue_discrepancy,
                "total_costs": total_costs,
                "total_costs_from_snapshot": costs_from_snapshot,
                "total_costs_from_status": costs_from_status,
                "total_costs_discrepancy": costs_discrepancy,
                "llm_costs": float(last.get("llm_costs", 0.0)),
                # None, not 0.0, when the provider did not report the reasoning/service split.
                "reasoning_llm_costs": _optional_float(last.get("reasoning_llm_costs")),
                "service_llm_costs": _optional_float(last.get("service_llm_costs")),
                "reasoning_llm_calls": _optional_float(last.get("reasoning_llm_calls")),
                "service_llm_calls": _optional_float(last.get("service_llm_calls")),
                "reasoning_input_tokens": _optional_float(last.get("reasoning_input_tokens")),
                "reasoning_output_tokens": _optional_float(last.get("reasoning_output_tokens")),
                "gas_costs": float(last.get("gas_costs", 0.0)),
                "net_balance": reconciled_net_balance,
                "net_balance_from_runtime": float(last.get("net_balance", 0.0)),
                "snapshots": len(snapshots)
            }

        if not financials["evolution"]:
            logger.warning(
                "No provider runtime financial snapshots found. "
                "Ensure experiment runner is collecting provider status financials during execution."
            )

        self.metrics["provider_financials"] = financials
        return financials

    def _identify_operation(self, tx: Dict) -> Optional[str]:
        """Identify operation type from transaction data."""
        try:
            # Function selector: the first 4 bytes of the input data.
            if tx.get('input') and len(tx['input']) >= 10:
                selector = tx['input'][:10]

                selectors = {
                    "0x095ea7b3": "approve_usdc",
                    "0xa9059cbb": "transfer_usdc",
                }

                return selectors.get(selector, "unknown")
        except:
            pass
        return None

    def _categorize_error(self, error_line: str) -> str:
        """Categorize error based on message content."""
        error_lower = error_line.lower()

        if "self-feedback" in error_lower or "self feedback" in error_lower:
            return "self_feedback_blocked"
        elif "gas" in error_lower:
            return "gas_error"
        elif "revert" in error_lower:
            return "transaction_reverted"
        elif "timeout" in error_lower:
            return "timeout"
        elif "api" in error_lower or "rate limit" in error_lower:
            return "api_error"
        elif "connection" in error_lower or "network" in error_lower:
            return "network_error"
        else:
            return "other"
