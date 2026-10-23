"""
Orchestrator for managing service execution in the agentic marketplace.

The orchestrator coordinates between:
- BlockchainHandler: Monitor auctions and handle blockchain interactions
- ServiceExecutor: Execute the actual service
- IPFS/P2P: File transfer

Workflow:
1. Periodically check for won auctions
2. Fetch service requirements from IPFS
3. Download input files (PDFs)
4. Execute literature analysis service
5. Deliver results to customer
6. Complete service on blockchain
7. Track costs and handle retries
"""

import asyncio
import copy
import logging
import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from agents.provider.blockchain_handler import BlockchainHandler
from agents.provider.service_executor import ServiceExecutor
from agents.common.ipfs_client import IPFSClient
from agents.common.cost_tracker import CostTracker
from agents.common.executor_tier_dataset_sampler import ExecutorTierDatasetSampler
from agents.config import Config

logger = logging.getLogger(__name__)


class SampledReplayUnavailable(RuntimeError):
    """Dataset v2 has no recorded run for this (requirement, tier).

    Distinct from a service *execution* failure on purpose. A live executor that
    fails is a market event — the provider tried, delivered nothing, and should
    be judged for it. A missing replay is not a market event: it means the run is
    reading a dataset that cannot answer, and continuing would deliver a
    fabricated result, collect the escrow, and record zero execution cost. That
    is invented profit in the primary DV (S7 §F.1), so it must stop the job
    rather than "unblock the workflow".
    """


class JobStatus(str, Enum):
    """Status of a service execution job."""
    WON = "won"
    FETCHING_REQUIREMENTS = "fetching_requirements"
    FETCHING_FILES = "fetching_files"
    PROCESSING = "processing"
    DELIVERING = "delivering"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a service execution job."""
    auction_id: int
    status: JobStatus
    started_at: datetime
    buyer_address: str
    service_cid: str
    service_requirements: Optional[Dict[str, Any]] = None
    pdf_directory: Optional[Path] = None
    prompts: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    llm_cost: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    effort_tier: Optional[str] = None
    service_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for logging/storage."""
        return {
            "auction_id": self.auction_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "buyer_address": self.buyer_address,
            "service_cid": self.service_cid,
            "retry_count": self.retry_count,
            "llm_cost": self.llm_cost,
            "error": self.error
        }


class Orchestrator:
    """
    Orchestrator for coordinating service execution.

    This is a simple sequential controller (not ReAct-based) that:
    - Monitors blockchain for won auctions
    - Executes services using specialized agents
    - Manages job lifecycle and state
    - Handles retries and errors
    """

    #: Default polling interval for the main loop (seconds). S7 overrides it via
    #: `provider_monitor_interval` (§A.6.5): at 5s a provider cannot reliably
    #: catch a 5s reveal window, which is how bids get committed and then
    #: forfeited. On a local chain mining every 1s, polling at 1s is within one
    #: block of a WebSocket event subscription — the same reaction time for none
    #: of the connection-lifecycle risk.
    _DEFAULT_MONITOR_INTERVAL = 5.0

    def __init__(
        self,
        config: Optional[Config] = None,
        architecture: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        """Initialize the orchestrator."""
        self.config = config or Config()
        resolved_architecture = architecture or getattr(self.config, "architecture", "1")
        resolved_reasoning_mode = reasoning_mode or getattr(self.config, "reasoning_mode", "llm_react")
        
        # Create shared cost tracker for all components
        self.cost_tracker = CostTracker(agent_id=self.config.agent_id, config=self.config)
        
        # Initialize agents with shared cost tracker
        self.blockchain_handler = BlockchainHandler(
            self.config.agent_id,
            architecture=resolved_architecture,
            reasoning_mode=resolved_reasoning_mode,
            cost_tracker=self.cost_tracker
        )
        self.service_executor = ServiceExecutor(
            str(self.config.agent_id),
            cost_tracker=self.cost_tracker,
            architecture=resolved_architecture
        )
        self.ipfs_client = IPFSClient()

        self.provider_profile = {
            "group_name": group_name or "default",
            "architecture": resolved_architecture,
            "reasoning_mode": resolved_reasoning_mode,
            "coupling_mode": self.blockchain_handler.coupling_mode,
            "heuristic_strategy": getattr(self.config, "heuristic_strategy", None),
            "heuristic_min_margin": getattr(self.config, "heuristic_min_margin", None),
            "heuristic_max_margin": getattr(self.config, "heuristic_max_margin", None),
            "blockchain_llm": self.blockchain_handler.llm_model,
            "service_llm": self.service_executor.llm_model,
        }
        
        # Job tracking
        self.active_jobs: Dict[int, Job] = {}  # auction_id -> Job
        self.completed_jobs: List[Job] = []
        
        # Working directory for downloaded files, and where finished results are
        # written for the consumer to read. Per-run via `AGENT_JOBS_DIR`; see
        # `Config.jobs_directory` for why sharing it corrupts results.
        self.work_dir = Path(getattr(
            self.config, "jobs_directory",
            Path(__file__).parent.parent / "data" / "jobs",
        ))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Runtime state
        self.running = False
        self.monitor_interval = float(getattr(
            self.config, "provider_monitor_interval", self._DEFAULT_MONITOR_INTERVAL
        ))
        self.last_monitor_error: Optional[str] = None
        #: Jobs abandoned because dataset v2 had no run to replay. Must be zero
        #: in any run whose numbers are used (S7 §F.1).
        self.replay_failures = 0
        self._sampled_dataset_sampler: Optional[ExecutorTierDatasetSampler] = None
        
        logger.info(f"Orchestrator initialized for agent {self.config.agent_id}")
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing orchestrator components...")
        await self.blockchain_handler.initialize()
        logger.info("✓ Orchestrator ready")

    def _get_sampled_dataset_sampler(self) -> ExecutorTierDatasetSampler:
        """Return cached dataset sampler for sampled execution mode."""
        if self._sampled_dataset_sampler is not None:
            return self._sampled_dataset_sampler

        dataset_dir = str(getattr(self.config, "sampled_dataset_dir", "") or "").strip()
        if not dataset_dir:
            raise RuntimeError("sampled_dataset_dir must be configured for sampled dataset execution")

        dataset_id = getattr(self.config, "sampled_dataset_id", None)
        run_selection_mode = getattr(self.config, "sampled_dataset_run_selection_mode", "round_robin")
        base_seed = getattr(self.config, "sampled_dataset_selection_seed", None)
        if base_seed is None:
            # S7 §A.3: fall back to the run seed so one number drives the run.
            base_seed = getattr(self.config, "run_seed", None)
        resolved_seed = None
        if base_seed is not None:
            try:
                resolved_seed = int(base_seed) + int(getattr(self.config, "agent_id", 0) or 0)
            except (TypeError, ValueError):
                resolved_seed = int(base_seed)

        self._sampled_dataset_sampler = ExecutorTierDatasetSampler(
            dataset_dir,
            dataset_id,
            run_selection_mode=run_selection_mode,
            random_seed=resolved_seed,
        )
        return self._sampled_dataset_sampler
    
    async def run(self):
        """
        Main orchestrator loop.
        
        Periodically:
        1. Check for new won auctions
        2. Process active jobs
        3. Clean up completed jobs
        """
        self.running = True
        logger.info("🚀 Orchestrator started")
        
        try:
            while self.running:
                try:
                    # Check for new won auctions
                    await self._check_won_auctions()
                    
                    # Process active jobs
                    await self._process_active_jobs()
                    
                    # Clean up old completed jobs
                    self._cleanup_completed_jobs()
                    
                except Exception as e:
                    logger.error(f"Error in orchestrator loop: {e}", exc_info=True)
                
                # Wait before next check
                await asyncio.sleep(self.monitor_interval)
                
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            self.running = False
            logger.info("Orchestrator stopped")
    
    def stop(self):
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self.running = False
    
    async def _check_won_auctions(self):
        """Check blockchain for newly won auctions."""
        try:
            result = await self.blockchain_handler.monitor_auctions()
            self.last_monitor_error = result.get("error")
            won_auctions = result.get("won_auctions", [])
            
            for auction_details in won_auctions:
                auction_id = auction_details["auction_id"]
                if auction_id not in self.active_jobs:
                    await self._start_job(auction_details)
                    
        except Exception as e:
            logger.error(f"Error checking won auctions: {e}")
    
    async def _start_job(self, auction_details: Dict[str, Any]):
        """Start a new job for a won auction."""
        auction_id = auction_details["auction_id"]
        logger.info(f"🎉 Starting job for won auction {auction_id}")
        
        try:
            job = Job(
                auction_id=auction_id,
                status=JobStatus.WON,
                started_at=datetime.now(),
                buyer_address=auction_details["buyer_address"],
                service_cid=auction_details["service_cid"],
                effort_tier=auction_details["effort_tier"]
            )
            
            self.active_jobs[auction_id] = job
            logger.info(f"✓ Job {auction_id} created: {job.to_dict()}")
            
        except Exception as e:
            logger.error(f"Error starting job {auction_id}: {e}", exc_info=True)
    
    async def _process_active_jobs(self):
        """Process all active jobs through their lifecycle.

        Each job is driven to completion (or failure) in a single call rather
        than advancing by one state per outer loop iteration.  The outer loop
        still sleeps between calls, so new won-auctions are picked up promptly;
        within a single job the steps are just awaited back-to-back, which is
        safe because every step already uses await internally for any real I/O.
        """
        for auction_id in list(self.active_jobs.keys()):
            job = self.active_jobs[auction_id]

            try:
                while True:
                    prev_status = job.status

                    if job.status == JobStatus.WON:
                        await self._fetch_requirements(job)
                    elif job.status == JobStatus.FETCHING_REQUIREMENTS:
                        await self._fetch_files(job)
                    elif job.status == JobStatus.FETCHING_FILES:
                        await self._execute_service(job)
                    elif job.status == JobStatus.PROCESSING:
                        await self._deliver_result(job)
                    elif job.status == JobStatus.DELIVERING:
                        await self._complete_service(job)
                    elif job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        self.completed_jobs.append(job)
                        del self.active_jobs[auction_id]
                        logger.info(f"✓ Job {auction_id} finished with status: {job.status.value}")
                        break

                    # Guard: if a step left the status unchanged something
                    # unexpected stalled — break to avoid an infinite loop.
                    if job.status == prev_status:
                        logger.warning(f"Job {auction_id} status unchanged at {job.status.value}, deferring")
                        break

            except SampledReplayUnavailable as e:
                # Not a market event: the dataset cannot answer, so there is
                # nothing to deliver and nothing to be paid for.
                logger.error(
                    "🛑 Job %s aborted — replay data missing: %s", auction_id, e
                )
                self.replay_failures += 1
                job.error = str(e)
                job.status = JobStatus.FAILED
            except Exception as e:
                logger.error(f"Error processing job {auction_id}: {e}", exc_info=True)
                await self._handle_job_error(job, str(e))
    
    async def _fetch_requirements(self, job: Job):
        """Fetch service requirements from IPFS."""
        logger.info(f"📥 Fetching requirements for job {job.auction_id} from {job.service_cid}")
        
        try:
            sampled_reference = ExecutorTierDatasetSampler.parse_dataset_service_cid(job.service_cid)
            if sampled_reference is not None:
                job.service_requirements = {
                    "execution_mode": "sampled_dataset",
                    **sampled_reference,
                }
                job.service_metadata = {
                    "source_mode": "sampled_dataset",
                    "sampled_dataset": copy.deepcopy(sampled_reference),
                }
                job.prompts = []
                logger.info(
                    "✓ Parsed sampled dataset reference for job %s: dataset=%s, requirement=%s",
                    job.auction_id,
                    sampled_reference.get("dataset_id"),
                    sampled_reference.get("requirement_id"),
                )
                job.status = JobStatus.FETCHING_REQUIREMENTS
                return

            # Fetch service description from IPFS
            requirements = await self.ipfs_client.fetch_json(job.service_cid)
            
            if not requirements:
                raise Exception(f"Failed to fetch service requirements from IPFS: {job.service_cid}")
            
            job.service_requirements = requirements
            
            # Extract prompts for literature analysis
            job.prompts = requirements.get("prompts", requirements.get("requirements", []))
            
            logger.info(f"✓ Requirements fetched: {len(job.prompts)} prompts")
            job.status = JobStatus.FETCHING_REQUIREMENTS
            
        except Exception as e:
            raise Exception(f"Failed to fetch requirements: {e}")
    
    async def _fetch_files(self, job: Job):
        """
        Fetch input files (PDFs) for the service.
        
        Supports two formats:
        1. input_files_cid: Single CID pointing to one file
        2. input_files: Dict/list of {filename: cid} mappings
        """
        logger.info(f"📥 Fetching input files for job {job.auction_id}")
        
        try:
            # Create job-specific directory
            job.pdf_directory = self.work_dir / f"auction_{job.auction_id}"
            if job.pdf_directory.exists():
                shutil.rmtree(job.pdf_directory)
            job.pdf_directory.mkdir(parents=True, exist_ok=True)

            if (job.service_requirements or {}).get("execution_mode") == "sampled_dataset":
                logger.info("✓ Sampled dataset mode: skipping input-file download for job %s", job.auction_id)
                job.status = JobStatus.FETCHING_FILES
                return
            
            downloaded_count = 0
            source_pdfs = job.service_requirements.get("source_pdfs") or []
            
            # Check for single file CID
            input_cid = job.service_requirements.get("input_files_cid")
            if input_cid:
                logger.info(f"Downloading file from IPFS: {input_cid}")
                output_name = (
                    job.service_requirements.get("input_file_name")
                    or (source_pdfs[0] if source_pdfs else None)
                    or "input.pdf"
                )
                output_path = job.pdf_directory / output_name
                success = await self.ipfs_client.download_file(input_cid, output_path)
                if success:
                    downloaded_count += 1
                else:
                    logger.warning(f"Failed to download file from {input_cid}")
            
            # Check for multiple files (dict or list of CIDs)
            input_files = job.service_requirements.get("input_files")
            if input_files:
                if isinstance(input_files, dict):
                    # Format: {"file1.pdf": "Qm...", "file2.pdf": "Qm..."}
                    for filename, cid in input_files.items():
                        logger.info(f"Downloading {filename} from IPFS: {cid}")
                        output_path = job.pdf_directory / filename
                        success = await self.ipfs_client.download_file(cid, output_path)
                        if success:
                            downloaded_count += 1
                        else:
                            logger.warning(f"Failed to download {filename} from {cid}")
                            
                elif isinstance(input_files, list):
                    # Format: ["Qm...", "Qm..."]
                    for i, cid in enumerate(input_files):
                        logger.info(f"Downloading file {i+1} from IPFS: {cid}")
                        output_path = job.pdf_directory / f"input_{i+1}.pdf"
                        success = await self.ipfs_client.download_file(cid, output_path)
                        if success:
                            downloaded_count += 1
                        else:
                            logger.warning(f"Failed to download file from {cid}")
            
            if downloaded_count == 0 and not input_cid and not input_files:
                logger.warning(f"No input files specified for job {job.auction_id}")
            elif downloaded_count > 0:
                logger.info(f"✓ Downloaded {downloaded_count} files to {job.pdf_directory}")
            
            job.status = JobStatus.FETCHING_FILES
            
        except Exception as e:
            raise Exception(f"Failed to fetch files: {e}")
    
    async def _execute_service(self, job: Job):
        """Execute the service."""
        logger.info(f"⚙️  Executing service for job {job.auction_id}")
        
        tracking_started = False
        try:
            if (job.service_requirements or {}).get("execution_mode") == "sampled_dataset":
                await self._execute_sampled_service(job)
                return

            # Track only service-execution LLM costs
            if self.cost_tracker:
                self.cost_tracker.start_service_execution(effort_tier=job.effort_tier)
                tracking_started = True

            # Execute service
            # Only pass effort_tier if two_way coupling is enabled
            service_requirements = job.service_requirements or {}
            kwargs = {
                "pdf_directory": str(job.pdf_directory),
                "prompts": job.prompts,
                "force_rebuild": True,
            }
            if self.blockchain_handler.coupling_mode == "two_way":
                kwargs["effort_tier"] = job.effort_tier
            
            result = self.service_executor.perform_analysis(**kwargs)
            
            if not result["success"]:
                raise Exception(f"Service execution failed: {result.get('error')}")
            
            job.result = result

            if self.cost_tracker and tracking_started:
                job.llm_cost = self.cost_tracker.end_service_execution(effort_tier=job.effort_tier)
            
            logger.info(f"✓ Service executed: {len(result['responses'])} responses generated")
            job.status = JobStatus.PROCESSING
            
        except SampledReplayUnavailable:
            raise
        except Exception as e:
            if self.cost_tracker and tracking_started:
                # Close tracking window even on failure (captures partial execution cost)
                job.llm_cost = self.cost_tracker.end_service_execution(effort_tier=job.effort_tier)
            raise Exception(f"Failed to execute service: {e}")

    async def _execute_sampled_service(self, job: Job):
        """Replay a sampled executor run instead of performing live RAG execution."""
        service_requirements = job.service_requirements or {}
        dataset_id = service_requirements.get("dataset_id")
        requirement_id = service_requirements.get("requirement_id")
        if not dataset_id or not requirement_id:
            raise Exception("Sampled dataset execution requires dataset_id and requirement_id")

        sampler = self._get_sampled_dataset_sampler()
        if sampler.dataset_id != dataset_id:
            raise Exception(
                f"Configured sampled dataset '{sampler.dataset_id}' does not match requested dataset '{dataset_id}'"
            )

        resolved_tier = self.cost_tracker.normalize_effort_tier(
            job.effort_tier,
            default=getattr(self.config, "default_effort_tier", "medium"),
        )
        try:
            run_record = sampler.sample_run(requirement_id, resolved_tier)
        except KeyError as e:
            raise SampledReplayUnavailable(
                f"dataset '{dataset_id}' has no run for requirement_id={requirement_id!r} "
                f"tier={resolved_tier!r}. Check that the config points at the canonical "
                f"dataset (…/executor_tier_dataset_v2/final), not the append-only "
                f"collection directory."
            ) from e
        sampled_result = copy.deepcopy(run_record.get("service_output") or {})
        if not sampled_result:
            raise Exception(
                f"Sampled run missing service_output for requirement_id={requirement_id}, tier={resolved_tier}"
            )

        execution_cost = float(run_record.get("execution_cost_usd", 0.0) or 0.0)
        if self.cost_tracker:
            self.cost_tracker.start_service_execution(effort_tier=job.effort_tier)
            if execution_cost > 0:
                self.cost_tracker.add_llm_cost(
                    cost_usd=execution_cost,
                    context=f"sampled_service_execution:{requirement_id}:{resolved_tier}",
                )
            job.llm_cost = self.cost_tracker.end_service_execution(effort_tier=job.effort_tier)

        # S7 §B.2: arm R regresses on the agent's *own* completed jobs, so it
        # needs (requirement, tier, realized cost). The cost tracker keeps costs
        # per tier but not which requirement produced them. Recorded here, at the
        # moment the job actually completes, so the fit history can never contain
        # a job the agent has not finished — an agent may only regress on what it
        # already knows.
        self.blockchain_handler.record_job_outcome(
            requirement_id=requirement_id,
            effort_tier=resolved_tier,
            realized_cost_usd=execution_cost,
        )

        sampled_metadata = {
            "dataset_id": dataset_id,
            "requirement_id": requirement_id,
            "tier": resolved_tier,
            "repeat_idx": run_record.get("repeat_idx"),
            "artifact_dir": run_record.get("artifact_dir"),
            "quality_method": "sampled_dataset",
            "evaluation": copy.deepcopy(run_record.get("evaluation") or {}),
            "evaluator_rating": run_record.get("evaluator_rating"),
            "evaluator_quality_scores": copy.deepcopy(run_record.get("evaluator_quality_scores") or {}),
            "evaluator_error": run_record.get("evaluator_error"),
            "executor_duration_seconds": float(run_record.get("executor_duration_seconds", 0.0) or 0.0),
            "evaluator_duration_seconds": float(run_record.get("evaluator_duration_seconds", 0.0) or 0.0),
            "execution_cost_usd": execution_cost,
            # Dataset v3 ground truth: how much digging the recorded execution
            # actually did, per prompt. Carried here so an analysis can score the
            # estimator's predicted depth against what happened on the very jobs
            # it bid on.
            #
            # ⚠️ The depth of the job being *priced* is never readable by any
            # estimator — arm X predicts these numbers from the prompt text
            # before bidding, and observing them would be reading the answer
            # key. Absent on v2 records, which is why the default is None/False rather
            # than 0: v2's rounds were never captured and must not look like zero
            # digging.
            "retrieval_rounds_per_prompt": copy.deepcopy(
                run_record.get("retrieval_rounds_per_prompt") or []
            ),
            "retrieval_rounds_total": run_record.get("retrieval_rounds_total"),
            "retrieval_rounds_measured": bool(run_record.get("retrieval_rounds_measured", False)),
            "selection_mode": getattr(self.config, "sampled_dataset_run_selection_mode", "round_robin"),
        }

        sampled_result.setdefault(
            "success",
            bool(run_record.get("executor_success", sampled_result.get("success", False))),
        )
        sampled_result.setdefault("responses", [])
        sampled_result["sampled_dataset"] = sampled_metadata

        job.result = sampled_result
        job.service_metadata = {
            "source_mode": "sampled_dataset",
            "sampled_dataset": copy.deepcopy(sampled_metadata),
        }
        job.status = JobStatus.PROCESSING

        logger.info(
            "✓ Sampled service replayed for job %s: tier=%s, repeat=%s, cost=$%.6f",
            job.auction_id,
            resolved_tier,
            sampled_metadata.get("repeat_idx"),
            job.llm_cost,
        )
    
    async def _deliver_result(self, job: Job):
        """
        Deliver results to customer.
        
        For v1: Save to local file
        For v2: Use A2A protocol to POST to customer endpoint
        """
        logger.info(f"📤 Delivering result for job {job.auction_id}")
        
        try:
            # For v1: Save result to file
            result_file = job.pdf_directory / "result.json"
            with open(result_file, "w") as f:
                json.dump(job.result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Result saved to {result_file}")
            
            # TODO: For v2, send via A2A protocol
            # customer_endpoint = job.service_requirements.get("customer_endpoint")
            # if customer_endpoint:
            #     await self._send_via_a2a(customer_endpoint, job.result)
            
            job.status = JobStatus.DELIVERING
            
        except Exception as e:
            raise Exception(f"Failed to deliver result: {e}")
    
    async def _complete_service(self, job: Job):
        """Complete the service on blockchain."""
        logger.info(f"✅ Completing service for job {job.auction_id}")
        
        try:
            # Call blockchain handler to complete service
            completion_result = await self.blockchain_handler.complete_service(
                auction_id=job.auction_id,
                client_address=job.buyer_address
            )
            
            if completion_result.get("error"):
                raise Exception(f"Blockchain completion failed: {completion_result['error']}")
            
            logger.info(f"✓ Service completed on-chain: tx={completion_result.get('tx_hash')}")
            logger.info(f"✓ Feedback auth: {completion_result.get('feedback_auth')}")
            
            job.status = JobStatus.COMPLETED
            
        except Exception as e:
            raise Exception(f"Failed to complete service: {e}")
    
    async def _handle_job_error(self, job: Job, error: str):
        """Handle job errors with retry logic."""
        job.error = error
        job.retry_count += 1
        
        logger.warning(f"⚠️  Job {job.auction_id} error (attempt {job.retry_count}/{job.max_retries}): {error}")
        
        if job.retry_count >= job.max_retries:
            if job.status == JobStatus.FETCHING_FILES:
                logger.error(
                    f"❌ Job {job.auction_id} service execution failed after {job.retry_count} attempts; "
                    "continuing with failure-marked result to unblock workflow"
                )
                self._continue_after_service_failure(job)
                return

            logger.error(f"❌ Job {job.auction_id} failed after {job.retry_count} attempts")
            job.status = JobStatus.FAILED
        else:
            # Retry: Reset status to previous step
            logger.info(f"🔄 Retrying job {job.auction_id}...")
            # Keep current status to retry the same step

    def _continue_after_service_failure(self, job: Job):
        """Create a synthetic failed result and continue pipeline to on-chain completion.

        This keeps the marketplace flow moving even when local service execution repeatedly fails.
        """
        failure_result = {
            "success": False,
            "responses": [],
            "error": job.error or "Service execution failed after max retries",
            "service_failed": True,
            "auction_id": job.auction_id,
            "retry_count": job.retry_count,
            "failure_timestamp": datetime.now().isoformat()
        }

        # Persist failure outcome so downstream analysis can treat this as poor delivery quality.
        job.result = failure_result
        job.status = JobStatus.PROCESSING
    
    def _cleanup_completed_jobs(self, max_completed: int = 100):
        """Clean up old completed jobs to prevent memory bloat."""
        if len(self.completed_jobs) > max_completed:
            removed = len(self.completed_jobs) - max_completed
            self.completed_jobs = self.completed_jobs[-max_completed:]
            logger.debug(f"Cleaned up {removed} old completed jobs")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        cost_breakdown = self.cost_tracker.get_cost_breakdown()

        # Jobs in DELIVERING state have had their execution cost charged but have not yet
        # received revenue (the completeService tx is still in flight).  Adding these costs
        # back prevents a transient balance dip in status snapshots that are written between
        # cost recording and revenue recording inside wait_for_transaction.
        #
        # ⚠️ The revenue check is load-bearing, not defensive. `DELIVERING` does not
        # end when revenue is booked: the winning bid is recorded inside a LangGraph
        # node (`_record_completion_revenue`), while `job.status = COMPLETED` is set
        # only after `graph.ainvoke` returns. The whole tail of that traversal is
        # `await` points, and the status writer is a coroutine on the same loop
        # (`await asyncio.sleep(10)` in `main`), so it can and eventually will
        # observe a job that is still DELIVERING *and* already paid. Compensating
        # then reports `net + revenue` — overstating by exactly the execution cost.
        #
        # This matters more than the odds suggest: `financials.net_balance` from
        # this status file is the primary DV named in `docs/preregistration.md` §4.
        # Never observed in block E's 320 provider status files — every final write
        # happens after the drain, with no job in flight — so no published result
        # moves. Closed before S9 because 960 runs is the wrong place to carry a
        # latent race that costs four lines to remove.
        pending_execution_cost = sum(
            job.llm_cost
            for job in self.active_jobs.values()
            if job.status == JobStatus.DELIVERING
            and not self.blockchain_handler.revenue_booked(job.auction_id)
        )

        return {
            "running": self.running,
            "profile": self.provider_profile,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "last_error": self.last_monitor_error,
            "bidding": self.blockchain_handler.get_bidding_metrics(),
            # S7 §A.4/§B.3: which arm ran, what it spent, and how much of it was
            # replayed from cache rather than called live.
            "estimator": self.blockchain_handler.get_estimator_metrics(),
            "financials": {
                **cost_breakdown,
                "revenue": self.cost_tracker.total_revenue,
                "net_balance": self.cost_tracker.get_net_balance() + pending_execution_cost,
            },
            "jobs": {
                "active": [job.to_dict() for job in self.active_jobs.values()],
                "completed": [job.to_dict() for job in self.completed_jobs[-10:]]  # Last 10
            }
        }


def _install_graceful_shutdown(orchestrator, status_file, get_status):
    """Write a final status on SIGTERM instead of dying mid-flight.

    The runner stops agents with SIGTERM and waits 5s before SIGKILL. Python's
    default SIGTERM action terminates the process without unwinding, so the
    ``finally`` that writes the closing status never ran: the newest status file
    was whatever the 10-second periodic loop last wrote, up to ten seconds
    stale. A job settling inside that window was missing from
    ``completed_jobs`` and its revenue from ``net_balance`` — seed 1008's H arm
    reported $0.1566 against a true $0.1850, and 3 of 20 arm-runs were wrong by
    up to $0.047.

    That matters beyond bookkeeping: every profit analysis reads these files
    (``experiments/analysis/compare_block_e.py`` and friends), so the headline
    numbers were being taken from a stale snapshot.

    Raising ``KeyboardInterrupt`` reuses the shutdown path that already exists
    rather than adding a second one.
    """
    import signal as _signal

    def _on_term(signum, _frame):
        logger.info("Received signal %s; writing final status and shutting down", signum)
        raise KeyboardInterrupt

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(sig, _on_term)
        except (ValueError, OSError):
            # Not on the main thread, or the platform disallows it: the periodic
            # write still runs, so this degrades to the previous behaviour.
            logger.debug("Could not install handler for %s", sig)


async def main(args):
    """Run the orchestrator with CLI arguments."""
    # Build runtime config from environment and CLI overrides
    runtime_config = Config()

    if args.agent_id:
        runtime_config.agent_id = args.agent_id
    if args.private_key:
        runtime_config.private_key = args.private_key

    if args.heuristic_strategy:
        runtime_config.heuristic_strategy = args.heuristic_strategy
    if args.heuristic_min_margin is not None:
        runtime_config.heuristic_min_margin = args.heuristic_min_margin
    if args.heuristic_max_margin is not None:
        runtime_config.heuristic_max_margin = args.heuristic_max_margin
    if args.heuristic_bid_tick_usd is not None:
        runtime_config.heuristic_bid_tick_usd = args.heuristic_bid_tick_usd

    architecture = args.architecture or runtime_config.architecture
    reasoning_mode = args.reasoning_mode

    orchestrator = Orchestrator(
        config=runtime_config,
        architecture=architecture,
        reasoning_mode=reasoning_mode,
        group_name=args.group_name,
    )

    logger.info(
        "Provider runtime: "
        f"group_name={args.group_name or 'default'}, "
        f"architecture={architecture}, reasoning_mode={reasoning_mode or 'default'}, "
        f"heuristic_strategy={runtime_config.heuristic_strategy}, "
        f"heuristic_min_margin={runtime_config.heuristic_min_margin}, "
        f"heuristic_max_margin={runtime_config.heuristic_max_margin}, "
        f"heuristic_bid_tick_usd={runtime_config.heuristic_bid_tick_usd}, "
        f"monitor_interval={orchestrator.monitor_interval}s"
    )

    await orchestrator.initialize()
    
    # Create status file path if provided
    status_file = Path(args.status_file) if args.status_file else None

    _install_graceful_shutdown(orchestrator, status_file, orchestrator.get_status)
    
    try:
        # Write initial status
        if status_file:
            with open(status_file, 'w') as f:
                json.dump(orchestrator.get_status(), f)
        
        # Start orchestrator main loop
        orchestrator_task = asyncio.create_task(orchestrator.run())
        
        # Periodically write status while orchestrator is running
        while not orchestrator_task.done():
            if status_file:
                with open(status_file, 'w') as f:
                    json.dump(orchestrator.get_status(), f)
            await asyncio.sleep(10)  # Update status every 10 seconds
        
        # Await the orchestrator task to catch any exceptions
        await orchestrator_task
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        orchestrator.stop()
        if status_file:
            with open(status_file, 'w') as f:
                json.dump(orchestrator.get_status(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provider Agent Orchestrator")
    parser.add_argument("--agent-id", type=int, help="Provider agent ID")
    parser.add_argument("--private-key", type=str, help="Blockchain private key")
    parser.add_argument("--group-name", type=str, help="Provider subgroup label")
    parser.add_argument("--architecture", type=str, help="Architecture profile (e.g., 1)")
    parser.add_argument(
        "--reasoning-mode",
        type=str,
        choices=["deterministic", "heuristic", "llm_react", "llm_strategic"],
        help="Provider reasoning mode override"
    )
    parser.add_argument(
        "--heuristic-strategy",
        type=str,
        choices=[
            "cost_plus",
            "zi_random",
            "h2_strategy_1",
            "h2_strategy_2",
            "h2_strategy_3",
            "h3_strategy_1",
            "h3_strategy_2",
        ],
        help="Heuristic bidding strategy"
    )
    parser.add_argument("--heuristic-min-margin", type=float, help="Heuristic minimum markup margin")
    parser.add_argument("--heuristic-max-margin", type=float, help="Heuristic maximum markup margin")
    parser.add_argument("--heuristic-bid-tick-usd", type=float, help="Bid tick size for legacy strategies")
    parser.add_argument("--status-file", type=str, help="Path to write status JSON file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main(args))
