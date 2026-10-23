"""The provider agent's process: bids through the BlockchainHandler and delivers the jobs it wins.

A won job is fetched (from IPFS in the live mode), executed with the
ServiceExecutor or replayed from the dataset, delivered to the shared jobs
directory, and completed on chain, which pays the provider.
"""

import asyncio
import copy
import logging
import argparse
import json
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
from agents.common.dataset_sampler import DatasetSampler
from agents.config import Config

logger = logging.getLogger(__name__)

#: How long a job keeps retrying IPFS reads before it fails.
IPFS_RETRY_SECONDS = 180


class SampledReplayUnavailable(RuntimeError):
    """The dataset has no recorded run for this (request, tier).

    Unlike a failed execution, which is a market event the provider is judged for,
    continuing here would deliver a fabricated result, collect the escrow and record
    no execution cost, so the job stops.
    """


class JobStatus(str, Enum):
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
    """One won auction's job."""
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
    """Drives each won job from fetching its requirements to on-chain completion, with retries."""

    # Default polling interval of the main loop (seconds); the experiments set it via
    # `provider_monitor_interval`. At 5 s a provider cannot reliably catch a 5 s
    # reveal window; on a local chain mining every second, 1 s is enough.
    _DEFAULT_MONITOR_INTERVAL = 5.0

    def __init__(
        self,
        config: Optional[Config] = None,
        architecture: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        self.config = config or Config()
        resolved_architecture = architecture or getattr(self.config, "architecture", "1")
        resolved_reasoning_mode = reasoning_mode or getattr(self.config, "reasoning_mode", "llm_react")

        # One cost tracker, shared by the handler and the executor.
        self.cost_tracker = CostTracker(agent_id=self.config.agent_id, config=self.config)

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

        self.active_jobs: Dict[int, Job] = {}  # auction_id -> Job
        self.completed_jobs: List[Job] = []

        # Where downloaded files and finished results live; per run, via `AGENT_JOBS_DIR`
        # (see `Config.jobs_directory`).
        self.work_dir = Path(getattr(
            self.config, "jobs_directory",
            Path(__file__).parent.parent / "data" / "jobs",
        ))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.running = False
        self.monitor_interval = float(getattr(
            self.config, "provider_monitor_interval", self._DEFAULT_MONITOR_INTERVAL
        ))
        self.last_monitor_error: Optional[str] = None
        #: Jobs abandoned because the dataset had no run to replay. The campaign
        #: check rejects a run with any.
        self.replay_failures = 0
        self._sampled_dataset_sampler: Optional[DatasetSampler] = None

        logger.info(f"Orchestrator initialized for agent {self.config.agent_id}")

    async def initialize(self):
        logger.info("Initializing orchestrator components...")
        await self.blockchain_handler.initialize()
        logger.info("✓ Orchestrator ready")

    def _get_sampled_dataset_sampler(self) -> DatasetSampler:
        if self._sampled_dataset_sampler is not None:
            return self._sampled_dataset_sampler

        dataset_dir = str(getattr(self.config, "sampled_dataset_dir", "") or "").strip()
        if not dataset_dir:
            raise RuntimeError("sampled_dataset_dir must be configured for sampled dataset execution")

        dataset_id = getattr(self.config, "sampled_dataset_id", None)
        run_selection_mode = getattr(self.config, "sampled_dataset_run_selection_mode", "round_robin")
        base_seed = getattr(self.config, "sampled_dataset_selection_seed", None)
        if base_seed is None:
            # Fall back to the run seed so one number drives the run.
            base_seed = getattr(self.config, "run_seed", None)
        resolved_seed = None
        if base_seed is not None:
            try:
                resolved_seed = int(base_seed) + int(getattr(self.config, "agent_id", 0) or 0)
            except (TypeError, ValueError):
                resolved_seed = int(base_seed)

        self._sampled_dataset_sampler = DatasetSampler(
            dataset_dir,
            dataset_id,
            run_selection_mode=run_selection_mode,
            random_seed=resolved_seed,
        )
        return self._sampled_dataset_sampler

    async def run(self):
        """Main loop: pick up won auctions, drive active jobs, drop old completed ones."""
        self.running = True
        logger.info("🚀 Orchestrator started")

        try:
            while self.running:
                try:
                    await self._check_won_auctions()

                    await self._process_active_jobs()

                    self._cleanup_completed_jobs()

                except Exception as e:
                    logger.error(f"Error in orchestrator loop: {e}", exc_info=True)

                await asyncio.sleep(self.monitor_interval)

        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            self.running = False
            logger.info("Orchestrator stopped")

    def stop(self):
        logger.info("Stopping orchestrator...")
        self.running = False

    async def _check_won_auctions(self):
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
        """Drive every active job to completion or failure.

        Each job's steps are awaited back to back within one call; the main loop still
        sleeps between calls, so new wins are picked up promptly.
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

                    # A step that left the status unchanged has stalled; stop rather than loop.
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
        """The job's requirements: the dataset record when replaying, else the description on IPFS."""
        logger.info(f"📥 Fetching requirements for job {job.auction_id} from {job.service_cid}")

        try:
            sampled_reference = DatasetSampler.parse_dataset_service_cid(job.service_cid)
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

            # Freshly pinned content can take a while to become readable.
            requirements = await self.ipfs_client.fetch_json(job.service_cid, retry_for=IPFS_RETRY_SECONDS)

            if not requirements:
                raise Exception(f"Failed to fetch service requirements from IPFS: {job.service_cid}")

            job.service_requirements = requirements

            job.prompts = requirements.get("prompts", requirements.get("requirements", []))

            logger.info(f"✓ Requirements fetched: {len(job.prompts)} prompts")
            job.status = JobStatus.FETCHING_REQUIREMENTS

        except Exception as e:
            raise Exception(f"Failed to fetch requirements: {e}")

    async def _fetch_files(self, job: Job):
        """Download the job's input files from IPFS.

        Either ``input_files_cid`` (one file) or ``input_files`` (a dict or list of CIDs).
        """
        logger.info(f"📥 Fetching input files for job {job.auction_id}")

        try:
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

            # One file
            input_cid = job.service_requirements.get("input_files_cid")
            if input_cid:
                logger.info(f"Downloading file from IPFS: {input_cid}")
                output_name = (
                    job.service_requirements.get("input_file_name")
                    or (source_pdfs[0] if source_pdfs else None)
                    or "input.pdf"
                )
                output_path = job.pdf_directory / output_name
                success = await self.ipfs_client.download_file(input_cid, output_path, retry_for=IPFS_RETRY_SECONDS)
                if success:
                    downloaded_count += 1
                else:
                    logger.warning(f"Failed to download file from {input_cid}")

            # Several files: a dict or a list of CIDs
            input_files = job.service_requirements.get("input_files")
            if input_files:
                if isinstance(input_files, dict):
                    # {"file1.pdf": "Qm...", ...}
                    for filename, cid in input_files.items():
                        logger.info(f"Downloading {filename} from IPFS: {cid}")
                        output_path = job.pdf_directory / filename
                        success = await self.ipfs_client.download_file(cid, output_path, retry_for=IPFS_RETRY_SECONDS)
                        if success:
                            downloaded_count += 1
                        else:
                            logger.warning(f"Failed to download {filename} from {cid}")

                elif isinstance(input_files, list):
                    # ["Qm...", ...]
                    for i, cid in enumerate(input_files):
                        logger.info(f"Downloading file {i+1} from IPFS: {cid}")
                        output_path = job.pdf_directory / f"input_{i+1}.pdf"
                        success = await self.ipfs_client.download_file(cid, output_path, retry_for=IPFS_RETRY_SECONDS)
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
        """Execute the job with the ServiceExecutor at its effort tier."""
        logger.info(f"⚙️  Executing service for job {job.auction_id}")

        tracking_started = False
        try:
            if (job.service_requirements or {}).get("execution_mode") == "sampled_dataset":
                await self._execute_sampled_service(job)
                return

            # Only the execution's own LLM costs go into the execution window.
            if self.cost_tracker:
                self.cost_tracker.start_service_execution(effort_tier=job.effort_tier)
                tracking_started = True

            # effort_tier is passed only with two-way coupling.
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
                # Close the window even on failure, so a partial execution is still charged.
                job.llm_cost = self.cost_tracker.end_service_execution(effort_tier=job.effort_tier)
            raise Exception(f"Failed to execute service: {e}")

    async def _execute_sampled_service(self, job: Job):
        """Replay a recorded execution from the dataset instead of executing."""
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

        # The estimators learn from the agent's own completed jobs, so they need
        # (request, tier, realised cost). Recorded here, when the job completes,
        # so their history never contains a job the agent has not finished.
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
            # How many retrieval rounds the recorded execution took, per prompt, so
            # an analysis can compare LLM-Estimate's predicted depths with them.
            # No estimator ever reads this for the job it is pricing.
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
        """Write the result to the shared jobs directory, where the consumer reads it."""
        logger.info(f"📤 Delivering result for job {job.auction_id}")

        try:
            result_file = job.pdf_directory / "result.json"
            with open(result_file, "w") as f:
                json.dump(job.result, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Result saved to {result_file}")

            job.status = JobStatus.DELIVERING

        except Exception as e:
            raise Exception(f"Failed to deliver result: {e}")

    async def _complete_service(self, job: Job):
        """Complete the job on chain, which releases the payment."""
        logger.info(f"✅ Completing service for job {job.auction_id}")

        try:
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
        """Retry the failed step, up to the job's retry limit."""
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
            logger.info(f"🔄 Retrying job {job.auction_id}...")
            # Keep the status, so the same step is retried.

    def _continue_after_service_failure(self, job: Job):
        """Deliver a failed result and complete the job on chain anyway.

        The consumer then rates the failure, so a failed delivery is judged like any other.
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

        # Recorded, so the analysis can treat it as a poor delivery.
        job.result = failure_result
        job.status = JobStatus.PROCESSING

    def _cleanup_completed_jobs(self, max_completed: int = 100):
        if len(self.completed_jobs) > max_completed:
            removed = len(self.completed_jobs) - max_completed
            self.completed_jobs = self.completed_jobs[-max_completed:]
            logger.debug(f"Cleaned up {removed} old completed jobs")

    def get_status(self) -> Dict[str, Any]:
        """The status the runner and the analysis read."""
        cost_breakdown = self.cost_tracker.get_cost_breakdown()

        # A job in DELIVERING has been charged its execution cost but may not have been
        # paid yet (completeService is in flight), so its cost is added back to avoid a
        # transient dip in the status. Only while the revenue is not yet booked: the
        # status writer runs on the same event loop and can see a job that is still
        # DELIVERING but already paid, which would otherwise overstate the net balance.
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
            # Which estimator ran, what it spent, and how much was replayed from
            # the cache rather than called live.
            "estimator": self.blockchain_handler.get_estimator_metrics(),
            "financials": {
                **cost_breakdown,
                "revenue": self.cost_tracker.total_revenue,
                "net_balance": self.cost_tracker.get_net_balance() + pending_execution_cost,
            },
            "jobs": {
                "active": [job.to_dict() for job in self.active_jobs.values()],
                "completed": [job.to_dict() for job in self.completed_jobs[-10:]]  # the last 10
            }
        }


def _install_graceful_shutdown(orchestrator, status_file, get_status):
    """Write a final status on SIGTERM.

    The runner stops agents with SIGTERM and waits 5 s before SIGKILL; Python's default
    action would terminate without running the ``finally`` that writes the closing
    status, leaving a status up to ten seconds stale, without the jobs that settled
    meanwhile. Raising ``KeyboardInterrupt`` reuses the existing shutdown path.
    """
    import signal as _signal

    def _on_term(signum, _frame):
        logger.info("Received signal %s; writing final status and shutting down", signum)
        raise KeyboardInterrupt

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(sig, _on_term)
        except (ValueError, OSError):
            # Not on the main thread, or the platform disallows it: the periodic status write
            # still runs.
            logger.debug("Could not install handler for %s", sig)


async def main(args):
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

    status_file = Path(args.status_file) if args.status_file else None

    _install_graceful_shutdown(orchestrator, status_file, orchestrator.get_status)

    try:
        if status_file:
            with open(status_file, 'w') as f:
                json.dump(orchestrator.get_status(), f)

        orchestrator_task = asyncio.create_task(orchestrator.run())

        while not orchestrator_task.done():
            if status_file:
                with open(status_file, 'w') as f:
                    json.dump(orchestrator.get_status(), f)
            await asyncio.sleep(10)  # status every 10 seconds

        # Await the task, so its exceptions surface.
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

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main(args))
