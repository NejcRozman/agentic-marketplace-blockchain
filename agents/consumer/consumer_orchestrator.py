"""The consumer agent's process: publishes jobs, runs their auctions, rates the results.

In the experiments the jobs come from the recorded dataset (``sampled_dataset``
mode), and the recorded rating of the replayed execution is submitted. In the
live mode the consumer generates the requests, publishes them on IPFS and rates
the delivered results with its LLM judge.
"""

import asyncio
import copy
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agents.config import Config
from agents.common.ipfs_client import IPFSClient
from agents.common.dataset_sampler import DatasetSampler
from agents.common.seeding import agent_rng
from agents.consumer.blockchain_handler import ConsumerBlockchainHandler
from agents.consumer.service_generator import ServiceGenerator
from agents.consumer.evaluator import ServiceEvaluator

config = Config()

logger = logging.getLogger(__name__)

# Mirrors ReverseAuction.sol's `enum Phase { Commit, Reveal, Ended }`.
AUCTION_PHASE_ENDED = 2


class AuctionStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    ENDED = "ended"
    COMPLETED = "completed"
    EVALUATED = "evaluated"
    FAILED = "failed"


@dataclass
class AuctionTracker:
    """One auction's lifecycle, as the consumer tracks it."""
    auction_id: int
    status: AuctionStatus
    created_at: datetime
    service_cid: str
    max_budget: int
    commit_duration: int
    reveal_duration: int
    eligible_providers: List[int]
    reputation_weight: int = 0
    
    winning_agent_id: Optional[int] = None
    winning_bid: Optional[int] = None
    ended_at: Optional[datetime] = None
    ended_block: Optional[int] = None
    
    result_path: Optional[Path] = None
    result: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    evaluation_started_at: Optional[datetime] = None
    evaluation_completed_at: Optional[datetime] = None
    feedback_submitted_at: Optional[datetime] = None
    
    evaluation: Optional[Dict[str, Any]] = None
    feedback_submitted: bool = False
    service_metadata: Dict[str, Any] = field(default_factory=dict)
    
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class Consumer:
    """The consumer agent: one auction at a time, from creation to rating."""
    
    def __init__(self, config: Config):
        self.config = config

        # The consumer's private RNG, derived from the run seed like the providers'. Its
        # draws (eligible providers, request order, poll jitter) are then identical
        # across conditions at the same seed.
        self._rng = agent_rng(
            getattr(config, "run_seed", None),
            getattr(config, "consumer_agent_id", 0),
        )

        self.blockchain_handler = ConsumerBlockchainHandler(config)
        self.ipfs_client = IPFSClient(config)
        self.service_generator = ServiceGenerator(config)
        self.evaluator = ServiceEvaluator(config)
        
        self.active_auctions: Dict[int, AuctionTracker] = {}
        self.completed_auctions: List[AuctionTracker] = []
        
        self.available_services: List[Dict[str, Any]] = []
        self.service_index: int = 0
        self._sampled_dataset_sampler: Optional[DatasetSampler] = None
        
        self.running = False
        configured_interval = getattr(config, "check_interval", config.consumer_check_interval)
        self.check_interval = float(configured_interval)
        self.check_interval_jitter = max(
            0.0,
            float(getattr(config, "consumer_check_interval_jitter", 0.0)),
        )
        # Must match the provider's `work_dir`: both resolve `AGENT_JOBS_DIR` through
        # `Config.jobs_directory`, unless `result_base_path` is set explicitly.
        self.result_base_path = Path(
            getattr(config, "result_base_path", None)
            or getattr(config, "jobs_directory",
                       Path(__file__).parent.parent / "data" / "jobs")
        )

        logger.info("Consumer initialized")

    def _is_sampled_dataset_mode(self) -> bool:
        """Whether jobs come from the recorded dataset."""
        mode = str(getattr(self.config, "service_source_mode", "live_generation") or "live_generation")
        return mode.strip().lower() == "sampled_dataset"

    def _get_service_selection_mode(self) -> str:
        """The request-selection mode.

        ``shuffled_pass`` needs the dataset's seeded permutation; in the live mode it
        falls back to ``sequential``.
        """
        mode = str(getattr(self.config, "service_selection", "sequential") or "sequential")
        normalized = mode.strip().lower()
        if normalized == "shuffled_pass":
            return "shuffled_pass" if self._is_sampled_dataset_mode() else "sequential"
        return normalized if normalized in {"sequential", "random"} else "sequential"

    def _next_shuffled_pass_index(self) -> int:
        """Index of the next request in the run's seeded pass.

        The sampler raises once all requests are used, rather than serving one again.
        """
        requirement_id = self._get_sampled_dataset_sampler().next_requirement_id()
        for index, service in enumerate(self.available_services):
            sampled = (service.get("service_metadata") or {}).get("sampled_dataset") or {}
            if sampled.get("requirement_id") == requirement_id:
                return index
        raise RuntimeError(
            f"shuffled_pass drew requirement {requirement_id!r}, which is not among "
            f"the {len(self.available_services)} loaded services"
        )

    def _get_sampled_dataset_sampler(self) -> DatasetSampler:
        if self._sampled_dataset_sampler is not None:
            return self._sampled_dataset_sampler

        dataset_dir = str(getattr(self.config, "sampled_dataset_dir", "") or "").strip()
        if not dataset_dir:
            raise RuntimeError("sampled_dataset_dir must be configured for sampled_dataset mode")

        dataset_id = getattr(self.config, "sampled_dataset_id", None)
        run_selection_mode = getattr(self.config, "sampled_dataset_run_selection_mode", "round_robin")
        # One seed per run: `sampled_dataset_selection_seed` wins if a config sets
        # it, but the experiments set only `run_seed`.
        base_seed = getattr(self.config, "sampled_dataset_selection_seed", None)
        if base_seed is None:
            base_seed = getattr(self.config, "run_seed", None)
        resolved_seed = None
        if base_seed is not None:
            try:
                resolved_seed = int(base_seed) + int(getattr(self.config, "consumer_agent_id", 0) or 0)
            except (TypeError, ValueError):
                resolved_seed = int(base_seed)

        # The permutation is drawn at construction, so the mode is passed here.
        requirement_selection_mode = str(
            getattr(self.config, "service_selection", "sequential") or "sequential"
        ).strip().lower()
        if requirement_selection_mode not in DatasetSampler.SELECTION_MODES:
            requirement_selection_mode = "round_robin"

        self._sampled_dataset_sampler = DatasetSampler(
            dataset_dir,
            dataset_id,
            requirement_selection_mode=requirement_selection_mode,
            run_selection_mode=run_selection_mode,
            random_seed=resolved_seed,
            requirement_pass_limit=getattr(self.config, "sampled_dataset_pass_limit", None),
        )
        return self._sampled_dataset_sampler

    @staticmethod
    def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        return value.isoformat(timespec="seconds")

    @staticmethod
    def _safe_duration_seconds(start: Optional[datetime], end: Optional[datetime]) -> int:
        """Elapsed seconds when both ends are known, never negative."""
        if start is None or end is None:
            return 0
        return max(0, int((end - start).total_seconds()))

    def _serialize_auction_tracker(self, tracker: AuctionTracker) -> Dict[str, Any]:
        """One auction's details, for the run's metrics."""
        evaluation = tracker.evaluation if isinstance(tracker.evaluation, dict) else {}
        service_metadata = tracker.service_metadata if isinstance(tracker.service_metadata, dict) else {}
        sampled_dataset = service_metadata.get("sampled_dataset") if isinstance(service_metadata.get("sampled_dataset"), dict) else {}
        quality_rating = evaluation.get("rating")
        quality_scores = evaluation.get("quality_scores") or {}
        quality_explanations = evaluation.get("explanations") or []
        evaluation_error = evaluation.get("error")
        service_executed = bool(tracker.result_path or tracker.result)

        if service_metadata.get("quality_method"):
            quality_method = str(service_metadata.get("quality_method"))
        elif evaluation.get("quality_method"):
            quality_method = str(evaluation.get("quality_method"))
        elif evaluation and evaluation_error:
            quality_method = "consumer_evaluator_failed"
        elif evaluation:
            quality_method = "consumer_evaluator"
        else:
            quality_method = "unknown"

        execution_duration_seconds = service_metadata.get("execution_duration_seconds")
        if execution_duration_seconds is None:
            execution_duration_seconds = (
                self._safe_duration_seconds(tracker.ended_at, tracker.completed_at) if service_executed else 0
            )

        evaluation_duration_seconds = service_metadata.get("evaluation_duration_seconds")
        if evaluation_duration_seconds is None:
            evaluation_duration_seconds = self._safe_duration_seconds(
                tracker.evaluation_started_at,
                tracker.evaluation_completed_at,
            )

        return {
            "auction_id": tracker.auction_id,
            "status": tracker.status.value if isinstance(tracker.status, AuctionStatus) else str(tracker.status),
            "created_at": self._serialize_datetime(tracker.created_at),
            "ended_at": self._serialize_datetime(tracker.ended_at),
            "completed_at": self._serialize_datetime(tracker.completed_at),
            "evaluation_started_at": self._serialize_datetime(tracker.evaluation_started_at),
            "evaluation_completed_at": self._serialize_datetime(tracker.evaluation_completed_at),
            "feedback_submitted_at": self._serialize_datetime(tracker.feedback_submitted_at),
            "service_cid": tracker.service_cid,
            "service_source_mode": service_metadata.get("source_mode", "live_generation"),
            "max_budget": tracker.max_budget,
            "commit_duration": tracker.commit_duration,
            "reveal_duration": tracker.reveal_duration,
            "reputation_weight": tracker.reputation_weight,
            "eligible_providers": list(tracker.eligible_providers),
            "winning_agent_id": tracker.winning_agent_id,
            "winning_bid": tracker.winning_bid,
            "service_executed": service_executed,
            "result_path": str(tracker.result_path) if tracker.result_path else None,
            "feedback_submitted": bool(tracker.feedback_submitted),
            "feedback_rating": quality_rating,
            "quality_rating": quality_rating,
            "quality_scores": quality_scores,
            "quality_explanations": quality_explanations,
            "quality_method": quality_method,
            "execution_duration_seconds": int(execution_duration_seconds or 0),
            "evaluation_duration_seconds": int(evaluation_duration_seconds or 0),
            "sampled_dataset_id": sampled_dataset.get("dataset_id"),
            "sampled_requirement_id": sampled_dataset.get("requirement_id"),
            "sampled_tier": sampled_dataset.get("tier"),
            "sampled_repeat_idx": sampled_dataset.get("repeat_idx"),
            "sampled_artifact_dir": sampled_dataset.get("artifact_dir"),
            "error": tracker.error or evaluation_error,
        }
    
    async def initialize(self, pdf_dir: Optional[Path] = None):
        """Connect to the chain and load the jobs (generating them from ``pdf_dir`` in the live mode)."""
        logger.info("Initializing consumer components...")
        
        success = await self.blockchain_handler.initialize()
        if not success:
            raise RuntimeError("Failed to initialize consumer blockchain handler")
        
        if pdf_dir is not None or self._is_sampled_dataset_mode():
            await self.load_services(pdf_dir)
        
        logger.info("✅ Consumer initialized successfully")
    
    async def load_services(self, pdf_dir: Optional[Path]):
        """Load the jobs: from the dataset, or generated from the PDFs and abstracts in ``pdf_dir``."""
        if self._is_sampled_dataset_mode():
            sampler = self._get_sampled_dataset_sampler()
            services: List[Dict[str, Any]] = []
            for requirement_id in sampler.list_requirement_ids():
                requirement_record = sampler.get_requirement(requirement_id)
                requirements = requirement_record.get("requirements") or {}
                sampled_metadata = {
                    "source_mode": "sampled_dataset",
                    "sampled_dataset": {
                        "dataset_id": sampler.dataset_id,
                        "requirement_id": requirement_id,
                        "selection_mode": self._get_service_selection_mode(),
                    },
                }
                services.append(
                    {
                        "service_cid": DatasetSampler.build_dataset_service_cid(
                            sampler.dataset_id,
                            requirement_id,
                        ),
                        "title": requirements.get("title") or f"Dataset Service {requirement_id}",
                        "prompts": list(requirements.get("prompts") or []),
                        "requirements": requirements,
                        "service_metadata": sampled_metadata,
                    }
                )

            self.available_services = services
            self.service_index = 0

            logger.info(
                "✅ Loaded %s dataset-backed services from %s",
                len(self.available_services),
                sampler.dataset_dir,
            )
            return

        if pdf_dir is None:
            raise ValueError("pdf_dir is required when service_source_mode=live_generation")

        logger.info(f"📚 Loading services from {pdf_dir}...")

        resolved_pdfs_per_service = max(
            1,
            int(getattr(self.config, "pdfs_per_service", 1) or 1),
        )
        resolved_sampling_mode = str(
            getattr(self.config, "service_generation_mode", "sequential_groups")
            or "sequential_groups"
        )
        resolved_sampling_iterations = max(
            1,
            int(getattr(self.config, "service_generation_iterations", 1) or 1),
        )
        resolved_sampling_seed = getattr(self.config, "service_generation_seed", None)
        logger.info(
            "Service generation config: mode=%s, pdfs_per_service=%s, iterations=%s, seed=%s",
            resolved_sampling_mode,
            resolved_pdfs_per_service,
            resolved_sampling_iterations,
            resolved_sampling_seed,
        )
        
        result = await self.service_generator.generate_services_from_pdfs(
            pdf_dir=pdf_dir,
            skip_processed=True,
            pdfs_per_service=resolved_pdfs_per_service,
            sampling_mode=resolved_sampling_mode,
            sampling_iterations=resolved_sampling_iterations,
            sampling_seed=resolved_sampling_seed,
        )
        
        self.available_services = result["processed"]
        self.service_index = 0
        
        logger.info(f"✅ Loaded {len(self.available_services)} services")
        if result["failed"]:
            logger.warning(f"⚠️  {len(result['failed'])} services failed to generate")
        if result["skipped"]:
            logger.info(f"⏭️  {len(result['skipped'])} services skipped (already processed)")
    
    async def create_auction(
        self,
        service_index: Optional[int] = None,
        max_budget: int = 10_000_000,  # 10 USDC
        commit_duration: Optional[int] = None,
        reveal_duration: Optional[int] = None,
        eligible_providers: Optional[List[int]] = None
    ) -> int:
        """Create the auction for the next job (or the job at ``service_index``); returns the auction id.

        ``max_budget`` is in micro-USDC; durations and eligible providers default to the config.
        """
        if not self.blockchain_handler._initialized:
            await self.initialize()

        if commit_duration is None:
            commit_duration = self.config.commit_duration
        if reveal_duration is None:
            reveal_duration = self.config.reveal_duration

        logger.info("📝 Creating new auction...")
        
        if not self.available_services:
            if self._is_sampled_dataset_mode():
                await self.load_services(None)

        if not self.available_services:
            raise RuntimeError(
                "No services available. Call load_services(pdf_dir) or initialize(pdf_dir) first."
            )
        
        if service_index is None:
            selection_mode = self._get_service_selection_mode()
            if selection_mode == "shuffled_pass":
                # Walk the sampler's seeded pass; the sampler raises when it runs out.
                idx = self._next_shuffled_pass_index()
            elif selection_mode == "random":
                idx = self._rng.randrange(len(self.available_services))
            else:
                # Next job, cycling
                idx = self.service_index % len(self.available_services)
                self.service_index += 1
        else:
            if service_index < 0 or service_index >= len(self.available_services):
                raise ValueError(
                    f"Invalid service_index {service_index}. "
                    f"Must be between 0 and {len(self.available_services) - 1}"
                )
            idx = service_index
        
        service_data = self.available_services[idx]
        service_cid = service_data["service_cid"]
        
        logger.info(f"Using service: {service_data['title']}")
        logger.info(f"Service CID: {service_cid}")
        
        if eligible_providers is None:
            if hasattr(self.config, 'provider_pool') and self.config.provider_pool:
                # Random draw from the provider pool
                pool = self.config.provider_pool
                count = getattr(self.config, 'eligible_per_auction', len(pool))
                
                if count >= len(pool):
                    eligible_providers = pool
                    logger.info(f"Using all {len(pool)} providers from pool")
                else:
                    eligible_providers = self._rng.sample(pool, count)
                    logger.info(f"Randomly selected {count} providers from pool of {len(pool)}: {eligible_providers}")
            else:
                eligible_providers = self.config.eligible_providers
                logger.info(f"Using static eligible providers: {eligible_providers}")
        else:
            logger.info(f"Using explicitly provided eligible providers: {eligible_providers}")
        
        logger.info(
            f"Creating auction on blockchain (budget: {max_budget}, "
            f"commit_duration: {commit_duration}s, reveal_duration: {reveal_duration}s)..."
        )

        result = await self.blockchain_handler.create_auction(
            service_cid=service_cid,
            max_price=max_budget,
            commit_duration=commit_duration,
            reveal_duration=reveal_duration,
            eligible_agent_ids=eligible_providers,
            reputation_weight=self.config.reputation_weight
        )
        
        if result['error']:
            raise RuntimeError(f"Failed to create auction: {result['error']}")
        
        auction_id = result['auction_id']
        tx_hash = result['tx_hash']
        
        logger.info(f"✓ Auction created on blockchain: ID={auction_id}, tx={tx_hash}")
        
        tracker = AuctionTracker(
            auction_id=auction_id,
            status=AuctionStatus.CREATED,
            created_at=datetime.now(),
            service_cid=service_cid,
            max_budget=max_budget,
            commit_duration=commit_duration,
            reveal_duration=reveal_duration,
            eligible_providers=eligible_providers,
            reputation_weight=self.config.reputation_weight,
            service_metadata=copy.deepcopy(service_data.get("service_metadata") or {}),
        )
        
        self.active_auctions[auction_id] = tracker
        
        logger.info(f"✅ Auction {auction_id} created")
        return auction_id
    
    async def monitor_auctions(self):
        """Advance every tracked auction: end it after its reveal phase, then collect, rate and give feedback."""
        if not self.blockchain_handler._initialized:
            logger.warning("Blockchain handler not initialized, skipping monitoring")
            return
        
        for auction_id, tracker in list(self.active_auctions.items()):
            try:
                auction_data = await self.blockchain_handler.get_auction_status(auction_id)
                
                if auction_data.get('error'):
                    logger.error(f"Error getting auction status: {auction_data['error']}")
                    continue
                
                is_active = auction_data['active']
                
                if not is_active and tracker.status in [AuctionStatus.CREATED, AuctionStatus.ACTIVE]:
                    current_block = await self.blockchain_handler.client.get_block_number()
                    logger.info(f"🏁 Auction {auction_id} has ended")
                    tracker.status = AuctionStatus.ENDED
                    tracker.ended_at = datetime.now()
                    tracker.ended_block = current_block
                    tracker.winning_agent_id = auction_data['winning_agent_id']
                    tracker.winning_bid = auction_data['winning_bid']

                    # Without a winner the auction is terminal and must not block the next one.
                    winner_id = tracker.winning_agent_id
                    if winner_id in (None, 0):
                        logger.info(
                            f"⚠️ Auction {auction_id} ended without winner (no valid bids); finalizing as failed"
                        )
                        tracker.status = AuctionStatus.FAILED
                        tracker.completed_at = datetime.now()
                        tracker.error = "Auction ended without winner"
                        self._finalize_terminal_auction(tracker)
                        continue
                    
                elif is_active:
                    tracker.status = AuctionStatus.ACTIVE

                    # End only after the reveal phase has expired on chain, so no provider is cut
                    # off before revealing.
                    phase = await self.blockchain_handler.get_auction_phase(auction_id)
                    phase_name = {0: "Commit", 1: "Reveal", 2: "Ended"}.get(phase, f"Unknown({phase})")
                    logger.info(f"📋 Auction {auction_id} active — phase={phase_name}")
                    if phase == AUCTION_PHASE_ENDED:
                        logger.info(f"⏰ Ending auction {auction_id} (commit+reveal phases expired)")
                        await self.end_auction(auction_id)
                
                is_completed = auction_data.get('completed', False)
                if is_completed and tracker.status == AuctionStatus.ENDED and not tracker.result_path:
                    logger.info(f"✅ Service {auction_id} completed by provider")
                    tracker.status = AuctionStatus.COMPLETED
                    tracker.completed_at = datetime.now()
                    await self._retrieve_result(tracker)
                    
            except Exception as e:
                logger.error(f"Error monitoring auction {auction_id}: {e}")
                tracker.error = str(e)
    
    async def end_auction(self, auction_id: int):
        logger.info(f"Ending auction {auction_id}...")
        
        result = await self.blockchain_handler.end_auction(auction_id)
        
        if result['error']:
            logger.error(f"Failed to end auction: {result['error']}")
            return
        
        logger.info(f"✓ Auction ended: tx={result['tx_hash']}, winner={result['winning_agent_id']}")
    
    async def _retrieve_result(self, tracker: AuctionTracker):
        """Read the winner's result from the shared jobs directory and rate it."""
        logger.info(f"📥 Retrieving result for auction {tracker.auction_id}...")
        
        # The winner writes the result to {result_base_path}/auction_{id}/result.json.
        result_path = self.result_base_path / f"auction_{tracker.auction_id}" / "result.json"
        
        if not result_path.exists():
            logger.warning(f"Result file not found: {result_path}")
            tracker.error = "Result file not found"
            return
        
        tracker.result_path = result_path
        logger.info(f"✓ Result retrieved from {result_path}")
        
        with open(result_path, 'r') as f:
            tracker.result = json.load(f)

        sampled_metadata = tracker.result.get("sampled_dataset") if isinstance(tracker.result, dict) else None
        if isinstance(sampled_metadata, dict):
            merged_metadata = copy.deepcopy(tracker.service_metadata)
            merged_metadata["source_mode"] = "sampled_dataset"
            merged_metadata["sampled_dataset"] = copy.deepcopy(sampled_metadata)
            if sampled_metadata.get("executor_duration_seconds") is not None:
                merged_metadata["execution_duration_seconds"] = int(
                    float(sampled_metadata.get("executor_duration_seconds") or 0)
                )
            if sampled_metadata.get("evaluator_duration_seconds") is not None:
                merged_metadata["evaluation_duration_seconds"] = int(
                    float(sampled_metadata.get("evaluator_duration_seconds") or 0)
                )
            if sampled_metadata.get("quality_method"):
                merged_metadata["quality_method"] = str(sampled_metadata.get("quality_method"))
            tracker.service_metadata = merged_metadata
        
        await self._evaluate_result(tracker)
    
    async def _evaluate_result(self, tracker: AuctionTracker):
        """Rate a result: the recorded rating in the dataset mode, the LLM judge in the live mode."""
        logger.info(f"🔍 Evaluating result for auction {tracker.auction_id}...")

        sampled_metadata = tracker.service_metadata.get("sampled_dataset") if isinstance(tracker.service_metadata, dict) else None
        if isinstance(sampled_metadata, dict):
            tracker.evaluation_started_at = datetime.now()
            evaluation = copy.deepcopy(sampled_metadata.get("evaluation") or {})
            if not evaluation:
                evaluation = {
                    "rating": sampled_metadata.get("evaluator_rating"),
                    "quality_scores": copy.deepcopy(sampled_metadata.get("evaluator_quality_scores") or {}),
                    "error": sampled_metadata.get("evaluator_error"),
                }
            evaluation.setdefault("quality_method", "sampled_dataset")
            tracker.evaluation_completed_at = datetime.now()
            tracker.evaluation = evaluation
            tracker.status = AuctionStatus.EVALUATED
            tracker.service_metadata["quality_method"] = evaluation.get("quality_method", "sampled_dataset")

            logger.info(f"✓ Sampled evaluation loaded: rating={evaluation.get('rating')}/100")
            await self._submit_feedback(tracker)
            return
        
        service_requirements = await self.ipfs_client.fetch_json(tracker.service_cid, retry_for=180)
        
        tracker.evaluation_started_at = datetime.now()
        evaluation = await self.evaluator.evaluate(
            service_requirements=service_requirements,
            result=tracker.result
        )
        tracker.evaluation_completed_at = datetime.now()
        
        tracker.evaluation = evaluation
        tracker.status = AuctionStatus.EVALUATED

        # If the judge produced no valid rating, give no feedback but finalise the auction.
        eval_error = evaluation.get("error")
        if eval_error and "LLM evaluator did not return a valid rating" in str(eval_error):
            logger.warning(
                f"⚠️ Skipping feedback for auction {tracker.auction_id}: evaluator could not produce valid rating"
            )
            tracker.error = str(eval_error)
            tracker.feedback_submitted = False
            tracker.status = AuctionStatus.COMPLETED
            self.completed_auctions.append(tracker)
            if tracker.auction_id in self.active_auctions:
                del self.active_auctions[tracker.auction_id]
            return
        
        logger.info(f"✓ Evaluation complete: rating={evaluation['rating']}/100")
        if evaluation.get('quality_scores'):
            logger.info(f"  Quality scores: {evaluation['quality_scores']}")
        
        await self._submit_feedback(tracker)
    
    async def _submit_feedback(self, tracker: AuctionTracker):
        """Submit the rating to the Reputation Registry, once the winner's feedbackAuth is on chain."""
        logger.info(f"📤 Submitting feedback for auction {tracker.auction_id}...")
        
        try:
            rating = tracker.evaluation.get('rating') if tracker.evaluation else None
            if not isinstance(rating, int):
                logger.warning(
                    f"⚠️ Skipping feedback for auction {tracker.auction_id}: invalid rating value {rating!r}"
                )
                tracker.feedback_submitted = False
                tracker.status = AuctionStatus.COMPLETED
                self._finalize_terminal_auction(tracker)
                return

            # The winner emits FeedbackAuthProvided in completeService(), a few blocks before
            # the consumer sees the auction completed, so search back from ended_block.
            if tracker.ended_block is not None:
                from_block = max(0, tracker.ended_block - 200)
            else:
                from_block = None

            feedback_auth = await self.blockchain_handler.get_feedback_auth(
                auction_id=tracker.auction_id,
                from_block=from_block,
                lookback_blocks=1000  # used when ended_block is unknown
            )
            
            if not feedback_auth:
                logger.error(f"FeedbackAuthProvided event not found for auction {tracker.auction_id}; skipping feedback")
                tracker.error = "FeedbackAuthProvided event not found"
                tracker.feedback_submitted = False
                tracker.status = AuctionStatus.COMPLETED
                self._finalize_terminal_auction(tracker)
                return
            
            quality_scores = tracker.evaluation.get('quality_scores', {})
            if quality_scores:
                feedback_text = ", ".join(
                    f"{k}: {v}" for k, v in quality_scores.items()
                )
            else:
                feedback_text = f"Rating: {tracker.evaluation['rating']}/100"
            
            # feedback_auth only shows that the winner completed the job; giveFeedback itself
            # takes no auth parameter.
            result = await self.blockchain_handler.submit_feedback(
                auction_id=tracker.auction_id,
                agent_id=tracker.winning_agent_id,
                rating=rating,
                feedback_text=feedback_text,
            )
            
            if result['error'] is not None:
                logger.error(f"Failed to submit feedback: {result['error']}")
                tracker.error = result['error']
                tracker.feedback_submitted = False
                tracker.status = AuctionStatus.COMPLETED
                self._finalize_terminal_auction(tracker)
                return
            
            tracker.feedback_submitted = True
            tracker.feedback_submitted_at = datetime.now()
            logger.info(f"✓ Feedback submitted: tx={result['tx_hash']}")
            
            tracker.status = AuctionStatus.COMPLETED
            self._finalize_terminal_auction(tracker)
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            tracker.error = str(e)

    def _finalize_terminal_auction(self, tracker: AuctionTracker):
        """Move a finished or failed auction out of active tracking."""
        if tracker not in self.completed_auctions:
            self.completed_auctions.append(tracker)
        if tracker.auction_id in self.active_auctions:
            del self.active_auctions[tracker.auction_id]
    
    async def run(self):
        logger.info("Starting consumer agent...")
        self.running = True
        
        try:
            while self.running:
                await self.monitor_auctions()
                
                await asyncio.sleep(self._next_check_delay())
                
        except Exception as e:
            logger.error(f"Consumer agent error: {e}", exc_info=True)
        finally:
            logger.info("Consumer agent stopped")
    
    def stop(self):
        logger.info("Stopping consumer agent...")
        self.running = False

    def _next_check_delay(self) -> float:
        """The next polling delay, with uniform jitter."""
        base_interval = max(0.0, float(self.check_interval))
        jitter = max(0.0, float(self.check_interval_jitter))

        if jitter == 0.0:
            return base_interval

        return max(0.1, base_interval + self._rng.uniform(-jitter, jitter))
    
    def get_status(self) -> Dict[str, Any]:
        failed_auctions = sum(1 for t in self.completed_auctions if t.status == AuctionStatus.FAILED)
        tracked_auctions: Dict[int, Dict[str, Any]] = {}
        for tracker in list(self.active_auctions.values()) + list(self.completed_auctions):
            tracked_auctions[tracker.auction_id] = self._serialize_auction_tracker(tracker)

        return {
            "active_auctions": len(self.active_auctions),
            "completed_auctions": len(self.completed_auctions),
            "failed_auctions": failed_auctions,
            "running": self.running,
            "auctions": [tracked_auctions[auction_id] for auction_id in sorted(tracked_auctions.keys())],
        }
    
    def write_status(self, status_file: Path):
        """Write the status file the runner reads."""
        try:
            with open(status_file, 'w') as f:
                json.dump(self.get_status(), f)
        except Exception as e:
            logger.error(f"Failed to write status file: {e}")


def _install_graceful_shutdown():
    """Write a final status on SIGTERM.

    The runner sends SIGTERM and reads the status file afterwards, but Python's default
    action terminates without running the ``finally`` that writes it.
    """
    import signal as _signal

    def _on_term(signum, _frame):
        logger.info("Received signal %s; writing final status and shutting down", signum)
        raise KeyboardInterrupt

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(sig, _on_term)
        except (ValueError, OSError):
            logger.debug("Could not install handler for %s", sig)


async def main(args):
    if args.agent_id:
        config.agent_id = args.agent_id
    if args.private_key:
        config.blockchain_private_key = args.private_key
    if args.eligible_providers:
        config.eligible_providers = args.eligible_providers
    if args.provider_pool:
        config.provider_pool = args.provider_pool
    if args.eligible_per_auction:
        config.eligible_per_auction = args.eligible_per_auction
    if args.max_budget:
        config.max_budget = args.max_budget
    if args.commit_duration:
        config.commit_duration = args.commit_duration
    if args.reveal_duration:
        config.reveal_duration = args.reveal_duration
    # `is not None`, not truthiness: w = 0 is a real setting (price alone).
    if args.reputation_weight is not None:
        config.reputation_weight = args.reputation_weight
    if args.check_interval is not None:
        config.consumer_check_interval = args.check_interval
    if args.check_interval_jitter is not None:
        config.consumer_check_interval_jitter = max(0.0, args.check_interval_jitter)
    if args.auto_create_auction:
        config.auto_create_auction = True
    if args.num_auctions:
        config.num_auctions = args.num_auctions
    if args.pdf_directory:
        config.pdf_directory = args.pdf_directory
    if args.service_complexity:
        config.service_complexity = args.service_complexity
    if args.auction_creation_delay is not None:
        config.auction_creation_delay = args.auction_creation_delay
    if args.inter_auction_delay is not None:
        config.inter_auction_delay = args.inter_auction_delay
    if args.pdfs_per_service is not None:
        config.pdfs_per_service = max(1, int(args.pdfs_per_service))
    if args.service_generation_mode:
        config.service_generation_mode = str(args.service_generation_mode)
    if args.service_generation_iterations is not None:
        config.service_generation_iterations = max(1, int(args.service_generation_iterations))
    if args.service_generation_seed is not None:
        config.service_generation_seed = int(args.service_generation_seed)
    
    consumer = Consumer(config)
    logger.info(
        "Consumer runtime polling: "
        f"check_interval={consumer.check_interval}, "
        f"check_interval_jitter={consumer.check_interval_jitter}"
    )
    
    pdf_dir = Path(config.pdf_directory) if config.pdf_directory and config.auto_create_auction else None
    await consumer.initialize(pdf_dir=pdf_dir)
    
    status_file = Path(args.status_file) if args.status_file else None

    _install_graceful_shutdown()
    
    auctions_created = 0
    consecutive_creation_failures = 0
    MAX_CONSECUTIVE_CREATION_FAILURES = 5
    BASE_RETRY_BACKOFF_SECONDS = 10
    MAX_RETRY_BACKOFF_SECONDS = 120

    try:
        if config.auto_create_auction:
            if config.auction_creation_delay > 0:
                logger.info(f"⏳ Waiting {config.auction_creation_delay}s before creating first auction...")
                await asyncio.sleep(config.auction_creation_delay)
            
            logger.info("🎬 Creating initial auction...")
            try:
                auction_id = await consumer.create_auction(
                    max_budget=config.max_budget,
                    commit_duration=config.commit_duration,
                    reveal_duration=config.reveal_duration
                )
                auctions_created = 1
                logger.info(f"✅ Created auction {auctions_created}/{config.num_auctions}: ID={auction_id}")
            except Exception as e:
                logger.error(f"❌ Failed to create initial auction: {e}")
                raise
        
        if status_file:
            consumer.write_status(status_file)
        
        consumer.running = True
        while consumer.running:
            await consumer.monitor_auctions()
            
            # Sequential creation: the next auction starts when the previous one is terminal
            if config.auto_create_auction and auctions_created < config.num_auctions:
                # Terminal auctions (completed or failed) count, so a round without bids does not block.
                completed_count = len(consumer.completed_auctions)
                
                logger.debug(f"Status check: completed={completed_count}, created={auctions_created}, target={config.num_auctions}")
                
                if completed_count >= auctions_created:
                    logger.info(f"✅ Auction {auctions_created} completed. Creating next auction...")
                    
                    if config.inter_auction_delay > 0:
                        logger.info(f"⏳ Waiting {config.inter_auction_delay}s before next auction...")
                        await asyncio.sleep(config.inter_auction_delay)
                    
                    try:
                        auction_id = await consumer.create_auction(
                            max_budget=config.max_budget,
                            commit_duration=config.commit_duration,
                            reveal_duration=config.reveal_duration,
                        )
                        auctions_created += 1
                        consecutive_creation_failures = 0
                        logger.info(f"✅ Created auction {auctions_created}/{config.num_auctions}: ID={auction_id}")
                    except Exception as e:
                        consecutive_creation_failures += 1
                        logger.error(
                            f"❌ Failed to create auction {auctions_created + 1} "
                            f"(attempt {consecutive_creation_failures}/{MAX_CONSECUTIVE_CREATION_FAILURES}): {e}"
                        )
                        if consecutive_creation_failures >= MAX_CONSECUTIVE_CREATION_FAILURES:
                            raise RuntimeError(
                                f"Giving up on auction {auctions_created + 1} after "
                                f"{consecutive_creation_failures} consecutive creation failures"
                            ) from e
                        backoff_seconds = min(
                            MAX_RETRY_BACKOFF_SECONDS,
                            BASE_RETRY_BACKOFF_SECONDS * (2 ** (consecutive_creation_failures - 1)),
                        )
                        logger.info(f"⏳ Backing off {backoff_seconds}s before retrying auction creation...")
                        await asyncio.sleep(backoff_seconds)
            
            if (config.auto_create_auction and 
                len(consumer.completed_auctions) >= config.num_auctions):
                logger.info(f"🎯 All {config.num_auctions} auctions completed!")
                consumer.stop()
            
            if status_file:
                consumer.write_status(status_file)
            
            await asyncio.sleep(consumer._next_check_delay())
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        consumer.stop()
        if status_file:
            consumer.write_status(status_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consumer Agent Orchestrator")
    parser.add_argument("--agent-id", type=int, help="Consumer agent ID")
    parser.add_argument("--private-key", type=str, help="Blockchain private key")
    parser.add_argument("--eligible-providers", type=int, nargs="+", help="List of eligible provider agent IDs")
    parser.add_argument("--provider-pool", type=int, nargs="+", help="Provider pool for random selection")
    parser.add_argument("--eligible-per-auction", type=int, help="Number of providers to randomly select per auction")
    parser.add_argument("--max-budget", type=int, help="Maximum budget for auctions (in token units)")
    parser.add_argument("--commit-duration", type=int, help="Commit phase duration in seconds")
    parser.add_argument("--reveal-duration", type=int, help="Reveal phase duration in seconds")
    parser.add_argument("--reputation-weight", type=int, help="Weight of reputation in bid scoring (0-100)")
    parser.add_argument("--check-interval", type=float, help="Blockchain check interval in seconds")
    parser.add_argument(
        "--check-interval-jitter",
        type=float,
        help="Symmetric uniform jitter bound in seconds (applies +/- jitter around check interval)",
    )
    parser.add_argument("--status-file", type=str, help="Path to write status JSON file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Automatic auctions
    parser.add_argument("--auto-create-auction", action="store_true", help="Automatically create auction(s) on startup")
    parser.add_argument("--num-auctions", type=int, help="Number of auctions to create sequentially")
    parser.add_argument("--pdf-directory", type=str, help="Directory containing PDFs for service generation")
    parser.add_argument("--service-complexity", type=str, choices=["low", "medium", "high"], help="Service complexity level")
    parser.add_argument("--auction-creation-delay", type=int, help="Seconds to wait before creating first auction")
    parser.add_argument("--inter-auction-delay", type=int, help="Seconds to wait between sequential auctions")
    parser.add_argument(
        "--pdfs-per-service",
        type=int,
        help="Number of PDFs to bundle into a single generated service",
    )
    parser.add_argument(
        "--service-generation-mode",
        choices=["sequential_groups", "random_iterations"],
        help="Strategy for generating services from input PDFs",
    )
    parser.add_argument(
        "--service-generation-iterations",
        type=int,
        help="Number of service-generation iterations in random_iterations mode",
    )
    parser.add_argument(
        "--service-generation-seed",
        type=int,
        help="Optional random seed for deterministic service sampling",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main(args))
