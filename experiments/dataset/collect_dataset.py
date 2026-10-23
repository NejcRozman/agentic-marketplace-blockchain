#!/usr/bin/env python3
"""Build the recorded service dataset that the market runs replay. Makes paid LLM calls.

Each request is written to a slot of the requirement plan (experiments/dataset/
requirement_plan.json), which fixes its papers, its designed synthesis depth and
its countable structure. The dataset was built in two collections:

1. generate the request text for every slot, and stop:

       python -m experiments.dataset.collect_dataset --collection-id requests --requirements-only

2. execute each request three times at the medium tier and have the consumer's
   judge rate each answer, reusing the text from step 1:

       python -m experiments.dataset.collect_dataset --collection-id full --reuse-requirements-from requests

Records are appended to data/dataset/{requirements,runs}.jsonl, with per-run
artifacts under data/dataset/artifacts/; finalize_dataset.py then selects one
collection into data/dataset/final/, the layout the market reads. The source
papers are fetched with corpus/arxiv_corpus.py into data/corpus/.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import os
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.config import Config, get_architecture
from agents.config import config as runtime_config_singleton
from agents.consumer.evaluator import ServiceEvaluator
from agents.consumer.service_generator import DEPTH_LADDER, ServiceGenerator
from agents.common.cost_tracker import CostTracker
from agents.provider.service_executor import ServiceExecutor
from agents.common import paths
from agents.provider.estimation.requirement import VisibleRequirement, mechanical_features

logger = logging.getLogger("executor_tier_dataset")


DEFAULT_QUALITY_CRITERIA = {
    "completeness": "All prompts must be answered thoroughly",
    "depth": "Answers should be detailed and well-supported with evidence from the paper",
    "citations": "Include relevant citations and specific references to the paper's content",
    "clarity": "Clear, well-structured responses that demonstrate deep understanding",
}


def usage_columns(exec_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten per-prompt execution usage into top-level dataset columns.

    Dataset v3's reason for existing: how much digging each prompt actually
    took. ``service_output`` already carries these nested inside the executor
    result, but every analysis reads top-level columns, and a field nobody can
    find is a field nobody uses.

    ``retrieval_rounds_measured`` is False when *any* prompt's usage was not
    observed — the recursion-limit fallback returns no graph state — so a
    consumer can drop partial rows rather than trust a total that silently
    omits the hardest prompt in the job. For the same reason unmeasured prompts
    appear as ``None`` in the per-prompt lists rather than 0: a silent zero is
    indistinguishable from "needed no digging" and would poison a regression
    fitted on it.
    """
    responses = list(exec_result.get("responses") or [])
    measured = bool(responses) and all(
        r.get("usage_measured") and r.get("retrieval_rounds_used") is not None
        for r in responses
    )
    return {
        "retrieval_rounds_per_prompt": [r.get("retrieval_rounds_used") for r in responses],
        "retrieval_rounds_total": exec_result.get("retrieval_rounds_total"),
        "retrieval_rounds_measured": measured,
        "tool_calls_per_prompt": [r.get("tool_calls_used") for r in responses],
        "prompt_durations_seconds": [r.get("prompt_duration_seconds") for r in responses],
    }


@dataclass
class ExecutorTierDatasetConfig:
    dataset_id: str
    collection_id: str
    source_pdf_dir: str
    file_glob: str
    workspace_dir: str
    provider_agent_id: str
    architecture: str
    effort_tiers: List[Optional[str]]
    executor_repeats_per_requirement: int
    force_rebuild_rag: bool
    reset_executor_between_runs: bool
    include_partial_fallback_runs: bool
    evaluation_timeout_seconds: Optional[float]
    heartbeat_interval_seconds: float
    overwrite: bool
    plan_path: str
    plan_slot_ids: List[str] = field(default_factory=list)
    plan_splits: List[str] = field(default_factory=list)
    prompt_char_tolerance: float = 0.15
    prompt_generation_max_attempts: int = 3
    reuse_requirements_from: Optional[str] = None
    # The effort budget each tier buys. Carried in the collection config so a
    # run reproduces without environment variables — the executor reads the module
    # singleton, so forgetting them silently reverts to the design's caps.
    effort_budget: Optional[Dict[str, Dict[str, int]]] = None
    #: Generate and persist requirements, then stop. Execution is the
    #: whole cost and runtime; separating it buys the chance to read the requirements
    #: before paying to execute them.
    requirements_only: bool = False


#: Requirement metadata the plan carries through to `requirements.jsonl`, so the
#: designed depth of every request stays attached to its records.
#: The retrieval caps the plan's depth ladder is graded to. Runtime values are
#: environment-overridable; when they differ, the collection says so.
DESIGN_RETRIEVAL_CAPS = {"low": 1, "medium": 3, "high": 10}


PLAN_METADATA_FIELDS = (
    "demanded_synthesis_depth",
    "demanded_synthesis_rank",
    "demanded_retrieval_rounds",
    "evidence_dispersion",
    "requires_reconciliation",
    "n_conflict_pairs",
    "shares_documents_with",
    "source_areas",
)


class ExecutorTierDatasetCollector:
    def __init__(self, cfg: ExecutorTierDatasetConfig):
        self.cfg = cfg
        self.runtime_config = Config()

        self._apply_effort_budget()

        self.requirement_source = "plan_driven"
        self.source_pdf_dir = self._abs_path(cfg.source_pdf_dir)
        self.dataset_root = paths.DATASET_DIR
        self.collection_manifest_dir = self.dataset_root / "collections"
        self.collection_manifest_path = self.collection_manifest_dir / f"{cfg.collection_id}.json"
        self.collection_artifacts_root = paths.DATASET_ARTIFACTS_DIR / cfg.collection_id

        self.requirements_jsonl_path = self.dataset_root / "requirements.jsonl"
        self.runs_jsonl_path = self.dataset_root / "runs.jsonl"
        self.dataset_manifest_path = self.dataset_root / "dataset_manifest.json"
        self.requirement_summary_json_path = self.dataset_root / "requirement_summary.json"
        self.requirement_summary_csv_path = self.dataset_root / "requirement_summary.csv"
        self.tier_summary_json_path = self.dataset_root / "tier_summary.json"
        self.depth_tier_summary_json_path = self.dataset_root / "depth_tier_summary.json"
        self.depth_tier_summary_csv_path = self.dataset_root / "depth_tier_summary.csv"
        self.tier_summary_csv_path = self.dataset_root / "tier_summary.csv"
        self.overall_summary_path = self.dataset_root / "overall_summary.json"

    @staticmethod
    def _abs_path(path_str: str) -> Path:
        return paths.resolve(path_str)

    @staticmethod
    def _repo_relative(path: Path) -> str:
        """``data/...`` for data files, repo-relative for files in the tree, absolute otherwise."""
        for root, prefix in ((paths.DATA_DIR, "data/"), (REPO_ROOT, "")):
            try:
                return prefix + str(path.relative_to(root))
            except ValueError:
                continue
        return str(path)

    @staticmethod
    def _safe_id_fragment(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
        slug = slug.strip("_")
        return slug[:60] if slug else "sample"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _normalize_effort_tier(effort_tier: Optional[str]) -> Optional[str]:
        if effort_tier is None:
            return None
        normalized = str(effort_tier).strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return None

    def _resolve_effort_tier(self, effort_tier: Optional[str]) -> str:
        normalized = self._normalize_effort_tier(effort_tier)
        if normalized is not None:
            return normalized
        fallback = str(getattr(self.runtime_config, "default_effort_tier", "medium")).strip().lower()
        if fallback in {"low", "medium", "high"}:
            return fallback
        return "medium"

    @staticmethod
    def _quality_criteria_for_depth(depth: str, multi_paper: bool) -> Dict[str, str]:
        """The criteria a consumer asking *this* task would actually state.

        Synthesis is asked for only where the task demands it (contrast, reconcile),
        and lookups are judged on precision rather than elaboration, so a correct
        one-line answer is not penalised. The criteria are recorded per requirement.
        """
        evidence_noun = "papers" if multi_paper else "paper"
        criteria = {
            "completeness": "All prompts must be answered thoroughly",
            "depth": f"Answers should be detailed and well-supported with evidence from the {evidence_noun}",
            "citations": f"Include relevant citations and specific references to the {evidence_noun}'s content",
            "clarity": "Clear, well-structured responses that demonstrate deep understanding",
        }

        if depth in {"locate", "enumerate"}:
            # A correct answer to a lookup is one line. Judged on `depth` — "extensive
            # explanation, multiple supporting arguments" — a right answer scores badly
            # for being appropriately terse, which is what sank `locate` in the pilot.
            criteria["depth"] = (
                "Precision: the exact detail requested is stated correctly, in the source's own "
                "terms, with no padding or unrequested elaboration"
            )
        if depth in {"contrast", "reconcile"}:
            criteria["synthesis"] = (
                "Explain how the sources differ and why, using evidence from each"
                if depth == "contrast"
                else "Reconcile the disagreement: what is in tension, what produces it, what would settle it"
            )
        return criteria

    @staticmethod
    def _build_requirements(
        pdf_paths: List[Path],
        prompts: List[str],
        depth: str,
    ) -> Dict[str, Any]:
        multi_paper = len(pdf_paths) > 1
        if not multi_paper:
            title = f"Executor Tier Dataset Task: {pdf_paths[0].stem}"
            description = f"Collect tiered executor outputs for {pdf_paths[0].stem}"
        else:
            preview_titles = [pdf.stem for pdf in pdf_paths[:2]]
            suffix = f" + {len(pdf_paths) - 2} more" if len(pdf_paths) > 2 else ""
            title = f"Executor Tier Dataset Task: {' & '.join(preview_titles)}{suffix}"
            description = (
                f"Collect tiered executor outputs for comparative synthesis across {len(pdf_paths)} papers"
            )

        quality_criteria = ExecutorTierDatasetCollector._quality_criteria_for_depth(depth, multi_paper)

        return {
            "title": title,
            "description": description,
            "service_type": "literature_review",
            "prompts": prompts,
            "input_file": "input.pdf",
            "source_pdfs": [pdf_path.name for pdf_path in pdf_paths],
            "quality_criteria": quality_criteria,
        }

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        return rows

    @staticmethod
    def _rewrite_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _numeric_stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "variance": None,
                "std": None,
                "min": None,
                "max": None,
                "range": None,
            }

        if len(values) == 1:
            v = float(values[0])
            return {
                "count": 1,
                "mean": v,
                "variance": 0.0,
                "std": 0.0,
                "min": v,
                "max": v,
                "range": 0.0,
            }

        mean_v = statistics.mean(values)
        var_v = statistics.pvariance(values)
        std_v = statistics.pstdev(values)
        min_v = min(values)
        max_v = max(values)
        return {
            "count": len(values),
            "mean": float(mean_v),
            "variance": float(var_v),
            "std": float(std_v),
            "min": float(min_v),
            "max": float(max_v),
            "range": float(max_v - min_v),
        }

    def _discover_pdf_paths(self) -> List[Path]:
        if not self.source_pdf_dir.exists():
            raise FileNotFoundError(f"Source PDF directory does not exist: {self.source_pdf_dir}")

        pdf_paths = sorted(self.source_pdf_dir.glob(self.cfg.file_glob))
        pdf_paths = [p for p in pdf_paths if p.is_file() and p.suffix.lower() == ".pdf"]

        with_abstract = []
        skipped_missing_abstract = []
        for pdf in pdf_paths:
            if pdf.with_suffix(".txt").exists():
                with_abstract.append(pdf)
            else:
                skipped_missing_abstract.append(pdf.name)

        if skipped_missing_abstract:
            logger.warning(
                "Skipping %s PDFs with no matching abstract .txt: %s",
                len(skipped_missing_abstract),
                ", ".join(skipped_missing_abstract),
            )

        return with_abstract

    def _load_plan_slots(self) -> List[Dict[str, Any]]:
        """Read the requirement plan and select the slots this collection covers."""
        plan_path = self._abs_path(str(self.cfg.plan_path))
        if not plan_path.exists():
            raise FileNotFoundError(f"Requirement plan not found: {plan_path}")

        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        slots = payload.get("slots") or []
        if not slots:
            raise ValueError(f"Requirement plan contains no slots: {plan_path}")

        wanted_ids = [str(s).strip() for s in (self.cfg.plan_slot_ids or []) if str(s).strip()]
        wanted_splits = [str(s).strip().lower() for s in (self.cfg.plan_splits or []) if str(s).strip()]

        selected: List[Dict[str, Any]] = []
        for slot in slots:
            slot_id = str(slot.get("slot_id") or "")
            split = str(slot.get("split") or "").strip().lower()
            if wanted_ids and slot_id not in wanted_ids:
                continue
            if wanted_splits and split not in wanted_splits:
                continue
            selected.append(slot)

        if wanted_ids:
            missing = sorted(set(wanted_ids) - {str(s.get("slot_id")) for s in selected})
            if missing:
                raise ValueError(f"plan_slot_ids not found in {plan_path}: {', '.join(missing)}")

        if not selected:
            raise ValueError(
                f"No plan slots selected from {plan_path} "
                f"(slot_ids={wanted_ids or 'all'}, splits={wanted_splits or 'all'})"
            )

        for slot in selected:
            self._validate_plan_slot(slot)

        logger.info(
            "Loaded %s/%s plan slots from %s",
            len(selected),
            len(slots),
            plan_path,
        )
        return selected

    @staticmethod
    def _validate_plan_slot(slot: Dict[str, Any]) -> None:
        slot_id = slot.get("slot_id")
        documents = slot.get("documents") or []
        breadths = slot.get("prompt_breadths") or []
        prompt_count = int(slot.get("prompt_count") or 0)

        if not slot_id:
            raise ValueError("Plan slot is missing slot_id")
        if not documents:
            raise ValueError(f"Plan slot {slot_id} has no documents")
        if not breadths:
            raise ValueError(f"Plan slot {slot_id} has no prompt_breadths")
        if prompt_count != len(breadths):
            raise ValueError(
                f"Plan slot {slot_id}: prompt_count={prompt_count} disagrees with "
                f"{len(breadths)} prompt_breadths"
            )
        if not slot.get("target_prompt_chars"):
            raise ValueError(f"Plan slot {slot_id} has no target_prompt_chars")

        metadata = slot.get("metadata") or {}
        depth = metadata.get("demanded_synthesis_depth")
        if depth not in DEPTH_LADDER:
            raise ValueError(
                f"Plan slot {slot_id}: unknown demanded_synthesis_depth {depth!r}"
            )

    def _resolve_plan_documents(self, slot: Dict[str, Any], pdf_paths: List[Path]) -> List[Path]:
        """Map a slot's document stems onto the corpus."""
        by_stem = {path.stem: path for path in pdf_paths}
        resolved: List[Path] = []
        missing: List[str] = []
        for document in slot.get("documents") or []:
            path = by_stem.get(str(document))
            if path is None:
                missing.append(str(document))
            else:
                resolved.append(path)

        if missing:
            raise FileNotFoundError(
                f"Plan slot {slot.get('slot_id')} references documents absent from "
                f"{self.source_pdf_dir} (or missing their abstract .txt): {', '.join(missing)}"
            )

        return resolved

    def _build_plan_requirement_specs(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for slot in self._load_plan_slots():
            specs.append(
                {
                    "logical_requirement_idx": len(specs) + 1,
                    "repeat_idx": 1,
                    "pdf_group": self._resolve_plan_documents(slot, pdf_paths),
                    "slot": slot,
                }
            )
        return specs

    def _build_plan_requirement_metadata(
        self,
        slot: Dict[str, Any],
        slot_metadata: Dict[str, Any],
        prompt_generation: Dict[str, Any],
        prompts: List[str],
    ) -> Dict[str, Any]:
        """The per-requirement design and realised-structure fields.

        Designed values come from the plan; realized values are measured on the
        prompts that were actually written, and the countable features are
        computed with `estimation.dataset.mechanical_features` - the same code
        the estimators use.
        """
        source_stems = [str(document) for document in slot.get("documents") or []]
        visible = VisibleRequirement(
            probe_id=str(slot.get("slot_id")),
            title="",
            description="",
            service_type="literature_review",
            prompts=tuple(prompts),
            source_pdf_count=len(source_stems),
            source_pdf_names=tuple(source_stems),
        )

        metadata: Dict[str, Any] = {
            "slot_id": str(slot.get("slot_id")),
            "pair_id": slot.get("pair_id"),
            "split": slot.get("split") or slot_metadata.get("split"),
        }
        for field_name in PLAN_METADATA_FIELDS:
            metadata[field_name] = slot_metadata.get(field_name)

        metadata.update(
            {
                "designed_prompt_count": int(slot.get("prompt_count") or 0),
                "designed_prompt_breadths": list(slot.get("prompt_breadths") or []),
                "realized_prompt_breadths": list(prompt_generation["realized_prompt_breadths"]),
                "target_prompt_chars": int(prompt_generation["target_prompt_chars"]),
                "realized_prompt_chars": int(prompt_generation["realized_prompt_chars"]),
                "prompt_char_band": list(prompt_generation["char_band"]),
                "prompt_length_in_band": bool(prompt_generation["length_in_band"]),
                "prompt_generation_attempts": int(prompt_generation["attempts"]),
                "prompt_generation_fallback_used": bool(prompt_generation["fallback_used"]),
                "designed_countable_features": dict(slot.get("countable_features") or {}),
                "realized_countable_features": mechanical_features(visible),
            }
        )
        return metadata

    def _prepare_output_dirs(self) -> None:
        collection_exists = self.collection_manifest_path.exists() or self.collection_artifacts_root.exists()
        if collection_exists and not self.cfg.overwrite:
            raise FileExistsError(
                f"Collection already exists: {self.cfg.collection_id}. Use overwrite=true or a new collection_id."
            )

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.collection_manifest_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.overwrite:
            self._drop_collection_records(self.requirements_jsonl_path)
            self._drop_collection_records(self.runs_jsonl_path)
            if self.collection_manifest_path.exists():
                self.collection_manifest_path.unlink()
            if self.collection_artifacts_root.exists():
                shutil.rmtree(self.collection_artifacts_root)

        self.collection_artifacts_root.mkdir(parents=True, exist_ok=True)

    def _drop_collection_records(self, path: Path) -> None:
        rows = self._load_jsonl(path)
        if not rows:
            return

        filtered = [row for row in rows if row.get("collection_id") != self.cfg.collection_id]
        if len(filtered) != len(rows):
            self._rewrite_jsonl(path, filtered)

    def _prepare_requirement_dir(self, requirement_id: str, pdf_paths: List[Path]) -> Path:
        req_dir = self.collection_artifacts_root / requirement_id
        req_dir.mkdir(parents=True, exist_ok=True)

        if len(pdf_paths) == 1:
            shutil.copy2(pdf_paths[0], req_dir / "input.pdf")
            abstract_src = pdf_paths[0].with_suffix(".txt")
            if abstract_src.exists():
                shutil.copy2(abstract_src, req_dir / "input.txt")
            return req_dir

        for pdf_path in pdf_paths:
            shutil.copy2(pdf_path, req_dir / pdf_path.name)
            abstract_src = pdf_path.with_suffix(".txt")
            if abstract_src.exists():
                shutil.copy2(abstract_src, req_dir / abstract_src.name)

        return req_dir

    def _load_reusable_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Requirement text from an earlier collection, keyed by slot_id.

        A cap sweep has to hold the task fixed: if every collection regenerates its
        prompts, the difference between two cap settings is confounded with the
        difference between two prompt sets.
        """
        if not self.cfg.reuse_requirements_from:
            return {}

        rows = [
            r for r in self._load_jsonl(self.requirements_jsonl_path)
            if r.get("collection_id") == self.cfg.reuse_requirements_from and r.get("slot_id")
        ]
        if not rows:
            raise ValueError(
                f"reuse_requirements_from={self.cfg.reuse_requirements_from} matched no requirements "
                f"in {self.requirements_jsonl_path}"
            )

        logger.info(
            "Reusing %s requirement(s) from collection %s", len(rows), self.cfg.reuse_requirements_from
        )
        return {str(r["slot_id"]): r for r in rows}

    def _resolve_models(self) -> Dict[str, Optional[str]]:
        """The three models this collection ran on.

        Recorded per collection because the models are the instrument: the judge's
        discrimination is what lets the tier interaction show up in ratings at all,
        and a dataset appended across a model change would silently mix two
        instruments.
        """
        try:
            architecture = get_architecture(str(self.cfg.architecture))
            executor_model = self.runtime_config.service_llm_override or architecture.service_llm
        except ValueError:
            executor_model = self.runtime_config.service_llm_override

        return {
            "requirement_generation": self.runtime_config.service_generation_model,
            "service_execution": executor_model,
            "service_execution_by_tier": self._resolve_tier_models(),
            "evaluation": self.runtime_config.openrouter_model,
        }

    def _apply_effort_budget(self) -> None:
        """Push the collection's effort budget onto the runtime config.

        `ServiceExecutor` resolves its budget from the module-level config singleton,
        so a budget expressed only in this collector would be ignored. Applying it to
        both keeps the YAML the single source of truth for what a tier buys.
        """
        budget = self.cfg.effort_budget
        if not budget:
            return

        fields = {
            "retrieval_rounds": "service_max_retrieval_rounds_by_effort",
            "tool_calls": "service_max_tool_calls_by_effort",
            "retrieval_k": "service_retrieval_k_by_effort",
        }
        # The tier may also name the model it runs, with its own prices so a
        # cheaper subcontractor shows up as cheaper in the record.
        for attribute, key in (("service_llm_by_effort", "model"), ("service_llm_prices_by_effort", "prices")):
            values = budget.get(key)
            if not values:
                continue
            for target in (self.runtime_config, runtime_config_singleton):
                current = dict(getattr(target, attribute, {}) or {})
                current.update({tier: values[tier] for tier in ("low", "medium", "high") if tier in values})
                setattr(target, attribute, current)
        for key, attribute in fields.items():
            values = budget.get(key)
            if key in {"model", "prices"}:
                continue
            if not values:
                continue
            resolved = {
                tier: int(values[tier]) for tier in ("low", "medium", "high") if tier in values
            }
            if not resolved:
                continue
            for target in (self.runtime_config, runtime_config_singleton):
                current = dict(getattr(target, attribute, {}) or {})
                current.update(resolved)
                setattr(target, attribute, current)

        logger.info(
            "Applied effort budget from config: rounds=%s, tool_calls=%s, retrieval_k=%s",
            self.runtime_config.service_max_retrieval_rounds_by_effort,
            self.runtime_config.service_max_tool_calls_by_effort,
            getattr(self.runtime_config, "service_retrieval_k_by_effort", None),
        )

    def _resolve_effort_caps(self) -> Dict[str, Dict[str, int]]:
        """The search budgets this collection ran under.

        `{low: 1, medium: 3, high: 10}` is what the design's depth ladder is graded
        to, but the caps are environment-overridable, so a re-tuned pilot must carry
        its own values — the caps are part of what produced the numbers.
        """
        return {
            "retrieval_rounds": {
                tier: int(self.runtime_config.service_max_retrieval_rounds_by_effort.get(tier, 0))
                for tier in ("low", "medium", "high")
            },
            "tool_calls": {
                tier: int(self.runtime_config.service_max_tool_calls_by_effort.get(tier, 0))
                for tier in ("low", "medium", "high")
            },
            "retrieval_k": {
                tier: int(
                    getattr(self.runtime_config, "service_retrieval_k_by_effort", {}).get(
                        tier, self.runtime_config.rag_retrieval_k
                    )
                )
                for tier in ("low", "medium", "high")
            },
        }

    def _resolve_tier_models(self) -> Dict[str, Optional[str]]:
        """The executor model each tier runs, when the tier names one."""
        by_effort = getattr(self.runtime_config, "service_llm_by_effort", {}) or {}
        return {tier: by_effort.get(tier) for tier in ("low", "medium", "high")}

    def _warn_on_effort_cap_drift(self) -> bool:
        """Say so, loudly, when the caps no longer match the design's assumption."""
        realized = self._resolve_effort_caps()["retrieval_rounds"]
        if realized == DESIGN_RETRIEVAL_CAPS:
            return False

        logger.warning(
            "Effort retrieval caps %s differ from the design's %s, to which the plan's depth "
            "ladder is graded",
            realized,
            DESIGN_RETRIEVAL_CAPS,
        )
        return True

    def _ensure_runtime_ready(self, dry_run: bool) -> None:
        if dry_run:
            return

        if not self.runtime_config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for executor tier dataset collection")

    def _build_executor(self) -> tuple[ServiceExecutor, CostTracker]:
        cost_tracker = CostTracker(agent_id=0, config=self.runtime_config)
        executor = ServiceExecutor(
            agent_id=self.cfg.provider_agent_id,
            workspace_dir=str(self._abs_path(self.cfg.workspace_dir)),
            cost_tracker=cost_tracker,
            architecture=self.cfg.architecture,
        )
        return executor, cost_tracker

    @staticmethod
    async def _evaluate_with_observability(
        evaluator: ServiceEvaluator,
        requirements: Dict[str, Any],
        service_output: Dict[str, Any],
        run_label: str,
        timeout: Optional[float],
        heartbeat_interval_seconds: float,
    ) -> Dict[str, Any]:
        started = time.time()
        heartbeat = max(1.0, float(heartbeat_interval_seconds))

        task = asyncio.create_task(evaluator.evaluate(requirements, service_output))
        while True:
            elapsed = time.time() - started
            remaining: Optional[float] = None
            if timeout is not None:
                remaining = timeout - elapsed
                if remaining <= 0:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    return {
                        "rating": None,
                        "quality_scores": {},
                        "error": f"Timed out after {timeout:.1f}s",
                        "duration_seconds": round(time.time() - started, 3),
                    }

            wait_for = heartbeat if remaining is None else min(heartbeat, remaining)
            done, _ = await asyncio.wait({task}, timeout=wait_for)
            if task in done:
                try:
                    result = task.result()
                except Exception as exc:
                    return {
                        "rating": None,
                        "quality_scores": {},
                        "error": str(exc),
                        "duration_seconds": round(time.time() - started, 3),
                    }

                duration = round(time.time() - started, 3)
                if not isinstance(result, dict):
                    return {
                        "rating": None,
                        "quality_scores": {},
                        "error": "Evaluator returned non-dict result",
                        "duration_seconds": duration,
                    }

                result.setdefault("rating", None)
                result.setdefault("quality_scores", {})
                result.setdefault("error", None)
                result["duration_seconds"] = duration
                return result

            if timeout is None:
                logger.info("Heartbeat: %s still running (elapsed=%.1fs)", run_label, elapsed)
            else:
                logger.info(
                    "Heartbeat: %s still running (elapsed=%.1fs, timeout=%.1fs)",
                    run_label,
                    elapsed,
                    timeout,
                )

    def _summarize_requirements(
        self,
        requirement_records: List[Dict[str, Any]],
        run_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        runs_by_requirement: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in run_records:
            runs_by_requirement[row.get("requirement_id")].append(row)

        summaries: List[Dict[str, Any]] = []
        for req in requirement_records:
            requirement_id = req["requirement_id"]
            runs = runs_by_requirement.get(requirement_id, [])
            evaluator_success_runs = [
                r
                for r in runs
                if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
            ]
            executor_success_runs = [r for r in runs if r.get("executor_success")]
            failed_runs = [r for r in runs if r.get("evaluator_error") is not None]
            fallback_runs = [r for r in runs if r.get("partial_fallback_used")]

            ratings = [float(r["evaluator_rating"]) for r in evaluator_success_runs]
            execution_costs = [float(r.get("execution_cost_usd", 0.0)) for r in runs]
            exec_durations = [float(r.get("executor_duration_seconds", 0.0)) for r in runs]
            eval_durations = [float(r.get("evaluator_duration_seconds", 0.0)) for r in runs]

            dim_values: Dict[str, List[float]] = {}
            for r in evaluator_success_runs:
                for dim, score in (r.get("evaluator_quality_scores") or {}).items():
                    if isinstance(score, (int, float)):
                        dim_values.setdefault(dim, []).append(float(score))

            tier_counts: Dict[str, int] = {}
            for r in runs:
                tier = str(r.get("tier") or "unknown")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            summaries.append(
                {
                    "requirement_id": requirement_id,
                    "collection_id": req.get("collection_id"),
                    "slot_id": req.get("slot_id"),
                    "pair_id": req.get("pair_id"),
                    "split": req.get("split"),
                    "demanded_synthesis_depth": req.get("demanded_synthesis_depth"),
                    "demanded_retrieval_rounds": req.get("demanded_retrieval_rounds"),
                    "source_pdfs": req.get("source_pdfs") or [],
                    "source_pdf_count": req.get("source_pdf_count"),
                    "prompt_count": req.get("prompt_count"),
                    "realized_prompt_chars": req.get("realized_prompt_chars"),
                    "target_prompt_chars": req.get("target_prompt_chars"),
                    "runs_total": len(runs),
                    "runs_executor_success": len(executor_success_runs),
                    "runs_evaluator_success": len(evaluator_success_runs),
                    "runs_failed": len(failed_runs),
                    "partial_fallback_runs": len(fallback_runs),
                    "tier_counts": tier_counts,
                    "rating_stats": self._numeric_stats(ratings),
                    "execution_cost_stats": self._numeric_stats(execution_costs),
                    "evaluation_cost_stats": self._numeric_stats(
                        self._cost_series(runs, "evaluation_cost_usd")
                    ),
                    "total_cost_stats": self._numeric_stats(self._cost_series(runs, "total_cost_usd")),
                    "executor_duration_stats": self._numeric_stats(exec_durations),
                    "evaluator_duration_stats": self._numeric_stats(eval_durations),
                    "dimension_stats": {
                        dim: self._numeric_stats(vals) for dim, vals in sorted(dim_values.items())
                    },
                }
            )

        return summaries

    @staticmethod
    def _cost_series(runs: List[Dict[str, Any]], key: str) -> List[float]:
        return [float(r.get(key, 0.0) or 0.0) for r in runs]

    def _summarize_depth_tiers(self, run_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rating and cost per (demanded depth x tier).

        The design's claim is conditional, not marginal: quality should be flat
        across tiers on `locate` and collapse at the low tier on `reconcile`. Pooled
        tier means cannot show that, so the cross-tabulation is materialised here
        rather than left for each consumer to re-derive.
        """
        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for row in run_records:
            depth = row.get("demanded_synthesis_depth")
            if not depth:
                continue
            grouped[(str(depth), str(row.get("tier") or "unknown"))].append(row)

        depth_order = {"locate": 0, "enumerate": 1, "contrast": 2, "reconcile": 3}
        tier_order = {"low": 0, "medium": 1, "high": 2}

        summaries: List[Dict[str, Any]] = []
        for (depth, tier) in sorted(
            grouped,
            key=lambda k: (depth_order.get(k[0], 99), tier_order.get(k[1], 99)),
        ):
            runs = grouped[(depth, tier)]
            rated = [
                r
                for r in runs
                if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
            ]
            summaries.append(
                {
                    "demanded_synthesis_depth": depth,
                    "tier": tier,
                    "requirement_count": len({r.get("requirement_id") for r in runs}),
                    "runs_total": len(runs),
                    "runs_evaluator_success": len(rated),
                    "rating_stats": self._numeric_stats([float(r["evaluator_rating"]) for r in rated]),
                    "execution_cost_stats": self._numeric_stats(
                        self._cost_series(runs, "execution_cost_usd")
                    ),
                    "evaluation_cost_stats": self._numeric_stats(
                        self._cost_series(runs, "evaluation_cost_usd")
                    ),
                    "total_cost_stats": self._numeric_stats(self._cost_series(runs, "total_cost_usd")),
                    "executor_duration_stats": self._numeric_stats(
                        self._cost_series(runs, "executor_duration_seconds")
                    ),
                }
            )

        return summaries

    @staticmethod
    def _flatten_depth_tier_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "demanded_synthesis_depth": row.get("demanded_synthesis_depth"),
                "tier": row.get("tier"),
                "requirement_count": row.get("requirement_count"),
                "runs_total": row.get("runs_total"),
                "runs_evaluator_success": row.get("runs_evaluator_success"),
                "rating_mean": row.get("rating_stats", {}).get("mean"),
                "rating_std": row.get("rating_stats", {}).get("std"),
                "execution_cost_mean": row.get("execution_cost_stats", {}).get("mean"),
                "evaluation_cost_mean": row.get("evaluation_cost_stats", {}).get("mean"),
                "total_cost_mean": row.get("total_cost_stats", {}).get("mean"),
                "executor_duration_mean_seconds": row.get("executor_duration_stats", {}).get("mean"),
            }
            for row in rows
        ]

    def _summarize_tiers(self, run_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        runs_by_tier: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in run_records:
            runs_by_tier[str(row.get("tier") or "unknown")].append(row)

        summaries: List[Dict[str, Any]] = []
        for tier, runs in sorted(runs_by_tier.items()):
            evaluator_success_runs = [
                r
                for r in runs
                if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
            ]
            executor_success_runs = [r for r in runs if r.get("executor_success")]
            failed_runs = [r for r in runs if r.get("evaluator_error") is not None]
            fallback_runs = [r for r in runs if r.get("partial_fallback_used")]

            ratings = [float(r["evaluator_rating"]) for r in evaluator_success_runs]
            execution_costs = [float(r.get("execution_cost_usd", 0.0)) for r in runs]
            exec_durations = [float(r.get("executor_duration_seconds", 0.0)) for r in runs]
            eval_durations = [float(r.get("evaluator_duration_seconds", 0.0)) for r in runs]

            dim_values: Dict[str, List[float]] = {}
            for r in evaluator_success_runs:
                for dim, score in (r.get("evaluator_quality_scores") or {}).items():
                    if isinstance(score, (int, float)):
                        dim_values.setdefault(dim, []).append(float(score))

            summaries.append(
                {
                    "tier": tier,
                    "requirement_count": len({r.get("requirement_id") for r in runs}),
                    "runs_total": len(runs),
                    "runs_executor_success": len(executor_success_runs),
                    "runs_evaluator_success": len(evaluator_success_runs),
                    "runs_failed": len(failed_runs),
                    "partial_fallback_runs": len(fallback_runs),
                    "rating_stats": self._numeric_stats(ratings),
                    "execution_cost_stats": self._numeric_stats(execution_costs),
                    "evaluation_cost_stats": self._numeric_stats(
                        self._cost_series(runs, "evaluation_cost_usd")
                    ),
                    "total_cost_stats": self._numeric_stats(self._cost_series(runs, "total_cost_usd")),
                    "executor_duration_stats": self._numeric_stats(exec_durations),
                    "evaluator_duration_stats": self._numeric_stats(eval_durations),
                    "dimension_stats": {
                        dim: self._numeric_stats(vals) for dim, vals in sorted(dim_values.items())
                    },
                }
            )

        return summaries

    @staticmethod
    def _flatten_requirement_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dim_names = set()
        for row in rows:
            dim_names.update(row.get("dimension_stats", {}).keys())

        ordered_dims = sorted(dim_names)
        flat: List[Dict[str, Any]] = []
        for row in rows:
            base = {
                "requirement_id": row.get("requirement_id"),
                "collection_id": row.get("collection_id"),
                "slot_id": row.get("slot_id"),
                "pair_id": row.get("pair_id"),
                "split": row.get("split"),
                "demanded_synthesis_depth": row.get("demanded_synthesis_depth"),
                "demanded_retrieval_rounds": row.get("demanded_retrieval_rounds"),
                "source_pdfs": ", ".join(row.get("source_pdfs") or []),
                "source_pdf_count": row.get("source_pdf_count"),
                "prompt_count": row.get("prompt_count"),
                "realized_prompt_chars": row.get("realized_prompt_chars"),
                "target_prompt_chars": row.get("target_prompt_chars"),
                "runs_total": row.get("runs_total"),
                "runs_executor_success": row.get("runs_executor_success"),
                "runs_evaluator_success": row.get("runs_evaluator_success"),
                "runs_failed": row.get("runs_failed"),
                "partial_fallback_runs": row.get("partial_fallback_runs"),
                "rating_mean": row.get("rating_stats", {}).get("mean"),
                "rating_variance": row.get("rating_stats", {}).get("variance"),
                "rating_std": row.get("rating_stats", {}).get("std"),
                "execution_cost_mean": row.get("execution_cost_stats", {}).get("mean"),
                "execution_cost_std": row.get("execution_cost_stats", {}).get("std"),
                "evaluation_cost_mean": row.get("evaluation_cost_stats", {}).get("mean"),
                "total_cost_mean": row.get("total_cost_stats", {}).get("mean"),
                "executor_duration_mean_seconds": row.get("executor_duration_stats", {}).get("mean"),
                "evaluator_duration_mean_seconds": row.get("evaluator_duration_stats", {}).get("mean"),
            }

            for dim in ordered_dims:
                stats = row.get("dimension_stats", {}).get(dim, {})
                base[f"{dim}_mean"] = stats.get("mean")
                base[f"{dim}_variance"] = stats.get("variance")
                base[f"{dim}_std"] = stats.get("std")

            flat.append(base)

        return flat

    @staticmethod
    def _flatten_tier_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dim_names = set()
        for row in rows:
            dim_names.update(row.get("dimension_stats", {}).keys())

        ordered_dims = sorted(dim_names)
        flat: List[Dict[str, Any]] = []
        for row in rows:
            base = {
                "tier": row.get("tier"),
                "requirement_count": row.get("requirement_count"),
                "runs_total": row.get("runs_total"),
                "runs_executor_success": row.get("runs_executor_success"),
                "runs_evaluator_success": row.get("runs_evaluator_success"),
                "runs_failed": row.get("runs_failed"),
                "partial_fallback_runs": row.get("partial_fallback_runs"),
                "rating_mean": row.get("rating_stats", {}).get("mean"),
                "rating_variance": row.get("rating_stats", {}).get("variance"),
                "rating_std": row.get("rating_stats", {}).get("std"),
                "execution_cost_mean": row.get("execution_cost_stats", {}).get("mean"),
                "execution_cost_variance": row.get("execution_cost_stats", {}).get("variance"),
                "execution_cost_std": row.get("execution_cost_stats", {}).get("std"),
                "execution_cost_min": row.get("execution_cost_stats", {}).get("min"),
                "execution_cost_max": row.get("execution_cost_stats", {}).get("max"),
                "evaluation_cost_mean": row.get("evaluation_cost_stats", {}).get("mean"),
                "total_cost_mean": row.get("total_cost_stats", {}).get("mean"),
                "executor_duration_mean_seconds": row.get("executor_duration_stats", {}).get("mean"),
                "evaluator_duration_mean_seconds": row.get("evaluator_duration_stats", {}).get("mean"),
            }

            for dim in ordered_dims:
                stats = row.get("dimension_stats", {}).get(dim, {})
                base[f"{dim}_mean"] = stats.get("mean")
                base[f"{dim}_variance"] = stats.get("variance")
                base[f"{dim}_std"] = stats.get("std")

            flat.append(base)

        return flat

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    async def run(self, dry_run: bool = False) -> Dict[str, Any]:
        self._ensure_runtime_ready(dry_run=dry_run)

        pdf_paths = self._discover_pdf_paths()
        if not pdf_paths:
            raise RuntimeError("No eligible PDF files found for executor tier dataset collection")

        requirement_specs = self._build_plan_requirement_specs(pdf_paths)

        resolved_tiers = [self._resolve_effort_tier(tier) for tier in self.cfg.effort_tiers]
        total_planned_runs = (
            0 if self.cfg.requirements_only
            else len(requirement_specs) * len(self.cfg.effort_tiers) * self.cfg.executor_repeats_per_requirement
        )

        if dry_run:
            plan_slots = [spec["slot"] for spec in requirement_specs if spec.get("slot")]
            depth_counts: Dict[str, int] = {}
            for slot in plan_slots:
                depth = str((slot.get("metadata") or {}).get("demanded_synthesis_depth"))
                depth_counts[depth] = depth_counts.get(depth, 0) + 1

            return {
                "dataset_id": self.cfg.dataset_id,
                "collection_id": self.cfg.collection_id,
                "source_pdf_dir": str(self.source_pdf_dir),
                "requirement_source": self.requirement_source,
                "models": self._resolve_models(),
                "effort_caps": self._resolve_effort_caps(),
                "effort_caps_match_design": (
                    self._resolve_effort_caps()["retrieval_rounds"] == DESIGN_RETRIEVAL_CAPS
                ),
                "plan_path": str(self._abs_path(self.cfg.plan_path)),
                "plan_slot_ids": [str(slot.get("slot_id")) for slot in plan_slots],
                "plan_depth_counts": depth_counts,
                "plan_splits": sorted({str(slot.get("split")) for slot in plan_slots}),
                "requirement_count": len(requirement_specs),
                "effort_tiers": resolved_tiers,
                "executor_repeats_per_requirement": self.cfg.executor_repeats_per_requirement,
                "planned_total_runs": total_planned_runs,
            }

        self._prepare_output_dirs()

        service_generator = ServiceGenerator(self.runtime_config)
        evaluator = ServiceEvaluator(self.runtime_config)
        reused_requirements = self._load_reusable_requirements()
        models = self._resolve_models()
        evaluator_model = getattr(evaluator, "model", models["evaluation"])
        logger.info(
            "Models: requirement_generation=%s, service_execution=%s, evaluation=%s",
            models["requirement_generation"],
            models["service_execution"],
            models["evaluation"],
        )
        self._warn_on_effort_cap_drift()

        logger.info(
            "Running dataset collection: requirements=%s, tiers=%s, repeats=%s",
            len(requirement_specs),
            resolved_tiers,
            self.cfg.executor_repeats_per_requirement,
        )

        started = time.time()
        collection_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        collection_prefix = self._safe_id_fragment(self.cfg.collection_id)[:24] or "collection"

        requirement_records: List[Dict[str, Any]] = []
        requirements_data: List[Dict[str, Any]] = []
        for generated_idx, spec in enumerate(requirement_specs, start=1):
            pdf_group = spec["pdf_group"]
            logical_requirement_idx = int(spec["logical_requirement_idx"])
            repeat_idx = int(spec["repeat_idx"])
            slot = spec["slot"]
            requirement_id = f"{collection_prefix}_{self._safe_id_fragment(str(slot['slot_id']))}"
            req_dir = self._prepare_requirement_dir(requirement_id, pdf_group)

            paper_abstracts = [
                {
                    "source_id": pdf_path.stem,
                    "abstract": service_generator._read_abstract(pdf_path),
                }
                for pdf_path in pdf_group
            ]

            reused = reused_requirements.get(str(slot["slot_id"]))
            if reused is not None:
                # Same task text, different execution conditions: the only way a cap
                # sweep measures caps rather than prompt variation.
                prompts = list((reused.get("requirements") or {}).get("prompts") or [])
                prompt_generation = reused.get("prompt_generation")
                plan_metadata = {
                    k: v for k, v in reused.items()
                    if k in PLAN_METADATA_FIELDS
                    or k in {
                        "slot_id", "pair_id", "split", "designed_prompt_count",
                        "designed_prompt_breadths", "realized_prompt_breadths",
                        "target_prompt_chars", "realized_prompt_chars", "prompt_char_band",
                        "prompt_length_in_band", "prompt_generation_attempts",
                        "prompt_generation_fallback_used", "designed_countable_features",
                        "realized_countable_features",
                    }
                }
                logger.info(
                    "Slot %s: reusing requirement text from collection %s",
                    slot["slot_id"],
                    self.cfg.reuse_requirements_from,
                )
            else:
                slot_metadata = dict(slot.get("metadata") or {})
                prompt_generation = await service_generator.generate_prompts_for_plan_slot(
                    paper_abstracts=paper_abstracts,
                    depth=str(slot_metadata["demanded_synthesis_depth"]),
                    prompt_breadths=list(slot.get("prompt_breadths") or []),
                    target_prompt_chars=int(slot["target_prompt_chars"]),
                    char_tolerance=self.cfg.prompt_char_tolerance,
                    max_attempts=self.cfg.prompt_generation_max_attempts,
                    evidence_dispersion=str(slot_metadata.get("evidence_dispersion") or "localised"),
                )
                prompts = prompt_generation["prompts"]
                plan_metadata = self._build_plan_requirement_metadata(
                    slot=slot,
                    slot_metadata=slot_metadata,
                    prompt_generation=prompt_generation,
                    prompts=prompts,
                )
                logger.info(
                    "Slot %s: depth=%s, prompts=%s, chars=%s (target %s, in_band=%s)",
                    slot["slot_id"],
                    plan_metadata["demanded_synthesis_depth"],
                    len(prompts),
                    plan_metadata["realized_prompt_chars"],
                    plan_metadata["target_prompt_chars"],
                    plan_metadata["prompt_length_in_band"],
                )

            if reused is not None:
                requirements = dict(reused["requirements"])
            else:
                requirements = self._build_requirements(
                    pdf_group, prompts, depth=str(slot_metadata["demanded_synthesis_depth"])
                )
            self._write_json(req_dir / "requirements.json", requirements)

            source_pdfs = [self._repo_relative(pdf_path) for pdf_path in pdf_group]

            requirement_record = {
                "dataset_id": self.cfg.dataset_id,
                "collection_id": self.cfg.collection_id,
                "requirement_id": requirement_id,
                "logical_requirement_idx": logical_requirement_idx,
                "repeat_idx": repeat_idx,
                "source_pdfs": source_pdfs,
                "source_pdf_count": len(pdf_group),
                "artifact_dir": self._repo_relative(req_dir),
                "prompt_count": len(prompts),
                "requirements": requirements,
                "requirement_source": self.requirement_source,
                "reused_requirements_from": self.cfg.reuse_requirements_from if reused else None,
                "created_at": collection_started_at,
                **plan_metadata,
            }
            if prompt_generation is not None:
                requirement_record["prompt_generation"] = prompt_generation
                self._write_json(req_dir / "prompt_generation.json", prompt_generation)

            self._append_jsonl(self.requirements_jsonl_path, requirement_record)
            requirement_records.append(requirement_record)
            requirements_data.append(
                {
                    "requirement_id": requirement_id,
                    "logical_requirement_idx": logical_requirement_idx,
                    "repeat_idx": repeat_idx,
                    "source_pdfs": source_pdfs,
                    "source_pdf_count": len(pdf_group),
                    "req_dir": req_dir,
                    "requirements": requirements,
                    "plan_metadata": plan_metadata,
                }
            )

        if self.cfg.requirements_only:
            logger.info(
                "requirements_only: generated %s requirement(s); skipping execution and evaluation",
                len(requirement_records),
            )

        completed_runs = 0
        current_collection_run_records: List[Dict[str, Any]] = []
        service_executor: Optional[ServiceExecutor] = None
        cost_tracker: Optional[CostTracker] = None

        for req_idx, req in enumerate([] if self.cfg.requirements_only else requirements_data, start=1):
            requirement_id = req["requirement_id"]
            logical_requirement_idx = int(req.get("logical_requirement_idx", req_idx))
            requirements = req["requirements"]
            prompts = requirements["prompts"]
            req_dir = req["req_dir"]
            source_pdfs = req["source_pdfs"]
            source_pdf_count = req["source_pdf_count"]
            plan_metadata = req.get("plan_metadata") or {}
            # Carried onto every run so by-depth analyses need no join.
            run_plan_fields = {
                key: plan_metadata.get(key)
                for key in (
                    "slot_id",
                    "pair_id",
                    "split",
                    "demanded_synthesis_depth",
                    "demanded_retrieval_rounds",
                )
            } if plan_metadata else {}

            logger.info(
                "[%s/%s] Running tier loops for %s",
                req_idx,
                len(requirements_data),
                requirement_id,
            )

            for effort_tier in self.cfg.effort_tiers:
                resolved_tier = self._resolve_effort_tier(effort_tier)
                for repeat_idx in range(1, self.cfg.executor_repeats_per_requirement + 1):
                    run_label = (
                        f"requirement {requirement_id} [{req_idx}/{len(requirements_data)}], "
                        f"tier {resolved_tier}, repeat {repeat_idx}/{self.cfg.executor_repeats_per_requirement}"
                    )
                    logger.info("Starting %s", run_label)

                    if service_executor is None or cost_tracker is None or self.cfg.reset_executor_between_runs:
                        service_executor, cost_tracker = self._build_executor()

                    assert cost_tracker is not None
                    cost_tracker.start_service_execution(resolved_tier)
                    exec_started = time.time()
                    exec_result = service_executor.perform_analysis(
                        pdf_directory=str(req_dir),
                        prompts=prompts,
                        force_rebuild=self.cfg.force_rebuild_rag,
                        effort_tier=effort_tier,
                    )
                    exec_duration = round(time.time() - exec_started, 3)
                    execution_cost = float(cost_tracker.end_service_execution(resolved_tier))

                    if not exec_result.get("success", False) or not exec_result.get("responses"):
                        evaluator_payload = {
                            "rating": None,
                            "quality_scores": {},
                            "error": "Skipped evaluator due to failed/empty executor output",
                            "duration_seconds": 0.0,
                        }
                    elif (
                        bool(exec_result.get("partial_fallback_used", False))
                        and not self.cfg.include_partial_fallback_runs
                    ):
                        evaluator_payload = {
                            "rating": None,
                            "quality_scores": {},
                            "error": "Excluded: partial fallback run",
                            "duration_seconds": 0.0,
                        }
                    else:
                        evaluator_payload = await self._evaluate_with_observability(
                            evaluator=evaluator,
                            requirements=requirements,
                            service_output=exec_result,
                            run_label=run_label,
                            timeout=self.cfg.evaluation_timeout_seconds,
                            heartbeat_interval_seconds=self.cfg.heartbeat_interval_seconds,
                        )

                    run_artifact_dir = req_dir / "runs" / resolved_tier / f"repeat_{repeat_idx:03d}"
                    run_artifact_dir.mkdir(parents=True, exist_ok=True)
                    self._write_json(run_artifact_dir / "result.json", exec_result)
                    self._write_json(run_artifact_dir / "evaluation.json", evaluator_payload)

                    run_record = {
                        "dataset_id": self.cfg.dataset_id,
                        "collection_id": self.cfg.collection_id,
                        "requirement_id": requirement_id,
                        "logical_requirement_idx": logical_requirement_idx,
                        "source_pdfs": source_pdfs,
                        "source_pdf_count": source_pdf_count,
                        "tier": resolved_tier,
                        "tier_input": effort_tier,
                        "repeat_idx": repeat_idx,
                        "artifact_dir": self._repo_relative(run_artifact_dir),
                        "executor_success": bool(exec_result.get("success", False)),
                        "executor_error": exec_result.get("error"),
                        "partial_fallback_used": bool(exec_result.get("partial_fallback_used", False)),
                        "response_count": len(exec_result.get("responses", [])),
                        **usage_columns(exec_result),
                        "executor_duration_seconds": exec_duration,
                        "execution_cost_usd": execution_cost,
                        "evaluator_rating": evaluator_payload.get("rating"),
                        "evaluator_quality_scores": evaluator_payload.get("quality_scores", {}),
                        "evaluator_error": evaluator_payload.get("error"),
                        "evaluator_duration_seconds": float(evaluator_payload.get("duration_seconds", 0.0)),
                        # Consumer-side spend per run: the judge makes prompt_count+1
                        # calls, so it is a real cost line.
                        "evaluator_model": evaluator_payload.get("evaluator_model", evaluator_model),
                        "evaluator_llm_calls": int(evaluator_payload.get("evaluator_llm_calls", 0) or 0),
                        "evaluator_input_tokens": int(evaluator_payload.get("evaluator_input_tokens", 0) or 0),
                        "evaluator_output_tokens": int(evaluator_payload.get("evaluator_output_tokens", 0) or 0),
                        "evaluation_cost_usd": float(evaluator_payload.get("evaluator_cost_usd", 0.0) or 0.0),
                        "total_cost_usd": round(
                            execution_cost + float(evaluator_payload.get("evaluator_cost_usd", 0.0) or 0.0),
                            8,
                        ),
                        "service_output": exec_result,
                        "evaluation": evaluator_payload,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        **run_plan_fields,
                    }
                    self._write_json(run_artifact_dir / "run_record.json", run_record)
                    self._append_jsonl(self.runs_jsonl_path, run_record)
                    current_collection_run_records.append(run_record)

                    completed_runs += 1
                    avg_run_seconds = (time.time() - started) / completed_runs
                    remaining_runs = total_planned_runs - completed_runs
                    eta_seconds = remaining_runs * avg_run_seconds

                    logger.info(
                        "Finished %s in %.1fs (cost=$%.6f, rating=%s, evaluator_error=%s)",
                        run_label,
                        exec_duration + run_record["evaluator_duration_seconds"],
                        execution_cost,
                        run_record["evaluator_rating"],
                        run_record["evaluator_error"],
                    )
                    logger.info(
                        "Progress: %s/%s runs (%.1f%%), avg %.1fs/run, ETA ~%s",
                        completed_runs,
                        total_planned_runs,
                        (completed_runs / total_planned_runs) * 100.0,
                        avg_run_seconds,
                        self._format_duration(eta_seconds),
                    )

        all_requirement_records = self._load_jsonl(self.requirements_jsonl_path)
        all_run_records = self._load_jsonl(self.runs_jsonl_path)
        requirement_summaries = self._summarize_requirements(all_requirement_records, all_run_records)
        tier_summaries = self._summarize_tiers(all_run_records)
        depth_tier_summaries = self._summarize_depth_tiers(all_run_records)

        successful_ratings = [
            float(r["evaluator_rating"])
            for r in all_run_records
            if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
        ]
        execution_costs = [float(r.get("execution_cost_usd", 0.0)) for r in all_run_records]
        collection_ids = sorted({str(r.get("collection_id")) for r in all_requirement_records if r.get("collection_id")})
        tiers = sorted({str(r.get("tier")) for r in all_run_records if r.get("tier")})
        total_fallback_runs = sum(1 for r in all_run_records if r.get("partial_fallback_used"))
        total_failed_runs = sum(1 for r in all_run_records if r.get("evaluator_error") is not None)
        total_executor_success_runs = sum(1 for r in all_run_records if r.get("executor_success"))
        total_evaluator_success_runs = sum(
            1
            for r in all_run_records
            if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
        )

        overall = {
            "dataset_id": self.cfg.dataset_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "collection_count": len(collection_ids),
            "collection_ids": collection_ids,
            "requirement_count": len(all_requirement_records),
            "total_runs": len(all_run_records),
            "tiers": tiers,
            "executor_successful_runs": total_executor_success_runs,
            "evaluator_successful_runs": total_evaluator_success_runs,
            "failed_runs": total_failed_runs,
            "partial_fallback_runs": total_fallback_runs,
            "overall_rating_stats": self._numeric_stats(successful_ratings),
            "overall_execution_cost_stats": self._numeric_stats(execution_costs),
            "overall_evaluation_cost_stats": self._numeric_stats(
                self._cost_series(all_run_records, "evaluation_cost_usd")
            ),
            "overall_total_cost_stats": self._numeric_stats(
                self._cost_series(all_run_records, "total_cost_usd")
            ),
            "duration_seconds_total": round(time.time() - started, 3),
            "output_files": {
                "requirements_jsonl": str(self.requirements_jsonl_path),
                "runs_jsonl": str(self.runs_jsonl_path),
                "dataset_manifest_json": str(self.dataset_manifest_path),
                "requirement_summary_json": str(self.requirement_summary_json_path),
                "requirement_summary_csv": str(self.requirement_summary_csv_path),
                "tier_summary_json": str(self.tier_summary_json_path),
                "tier_summary_csv": str(self.tier_summary_csv_path),
                "depth_tier_summary_json": str(self.depth_tier_summary_json_path),
                "depth_tier_summary_csv": str(self.depth_tier_summary_csv_path),
                "overall_summary_json": str(self.overall_summary_path),
            },
        }

        unstable_requirements = sorted(
            requirement_summaries,
            key=lambda s: (s.get("rating_stats", {}).get("std") or 0.0),
            reverse=True,
        )
        overall["top_unstable_requirements"] = [
            {
                "requirement_id": s.get("requirement_id"),
                "collection_id": s.get("collection_id"),
                "source_pdfs": s.get("source_pdfs"),
                "rating_std": s.get("rating_stats", {}).get("std"),
                "rating_mean": s.get("rating_stats", {}).get("mean"),
                "execution_cost_mean": s.get("execution_cost_stats", {}).get("mean"),
            }
            for s in unstable_requirements[: min(10, len(unstable_requirements))]
        ]

        collection_ratings = [
            float(r["evaluator_rating"])
            for r in current_collection_run_records
            if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
        ]
        collection_costs = [float(r.get("execution_cost_usd", 0.0)) for r in current_collection_run_records]
        collection_summary = {
            "dataset_id": self.cfg.dataset_id,
            "collection_id": self.cfg.collection_id,
            "created_at": collection_started_at,
            "source_pdf_dir": str(self.source_pdf_dir),
            "requirements_added": len(requirement_records),
            "runs_added": len(current_collection_run_records),
            "planned_total_runs": total_planned_runs,
            "effort_tiers": resolved_tiers,
            "executor_repeats_per_requirement": self.cfg.executor_repeats_per_requirement,
            "requirements_only": self.cfg.requirements_only,
            "successful_executor_runs": sum(1 for r in current_collection_run_records if r.get("executor_success")),
            "successful_evaluator_runs": sum(
                1
                for r in current_collection_run_records
                if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
            ),
            "failed_runs": sum(1 for r in current_collection_run_records if r.get("evaluator_error") is not None),
            "partial_fallback_runs": sum(1 for r in current_collection_run_records if r.get("partial_fallback_used")),
            "rating_stats": self._numeric_stats(collection_ratings),
            "execution_cost_stats": self._numeric_stats(collection_costs),
            "evaluation_cost_stats": self._numeric_stats(
                self._cost_series(current_collection_run_records, "evaluation_cost_usd")
            ),
            "total_cost_stats": self._numeric_stats(
                self._cost_series(current_collection_run_records, "total_cost_usd")
            ),
            # What this collection actually spent, split by stage.
            "spend_usd": {
                "requirement_generation": service_generator.generation_usage()["cost_usd"],
                "service_execution": round(sum(collection_costs), 8),
                "evaluation": round(
                    sum(self._cost_series(current_collection_run_records, "evaluation_cost_usd")), 8
                ),
                "total": round(
                    service_generator.generation_usage()["cost_usd"]
                    + sum(self._cost_series(current_collection_run_records, "total_cost_usd")),
                    8,
                ),
            },
            "requirement_generation_usage": service_generator.generation_usage(),
            "effort_caps": self._resolve_effort_caps(),
            "effort_caps_match_design": (
                self._resolve_effort_caps()["retrieval_rounds"] == DESIGN_RETRIEVAL_CAPS
            ),
            "duration_seconds_total": round(time.time() - started, 3),
            "artifacts_dir": str(self.collection_artifacts_root),
            "models": self._resolve_models(),
            "output_files": {
                "collection_manifest_json": str(self.collection_manifest_path),
                "requirements_jsonl": str(self.requirements_jsonl_path),
                "runs_jsonl": str(self.runs_jsonl_path),
            },
            "config": {
                "architecture": self.cfg.architecture,
                "requirement_source": self.requirement_source,
                "plan_path": self.cfg.plan_path,
                "plan_slot_ids": list(self.cfg.plan_slot_ids or []),
                "plan_splits": list(self.cfg.plan_splits or []),
                "prompt_char_tolerance": self.cfg.prompt_char_tolerance,
                "prompt_generation_max_attempts": self.cfg.prompt_generation_max_attempts,
                "executor_repeats_per_requirement": self.cfg.executor_repeats_per_requirement,
                "reuse_requirements_from": self.cfg.reuse_requirements_from,
                "force_rebuild_rag": self.cfg.force_rebuild_rag,
                "reset_executor_between_runs": self.cfg.reset_executor_between_runs,
                "include_partial_fallback_runs": self.cfg.include_partial_fallback_runs,
                "evaluation_timeout_seconds": self.cfg.evaluation_timeout_seconds,
                "heartbeat_interval_seconds": self.cfg.heartbeat_interval_seconds,
            },
        }

        dataset_manifest = {
            "dataset_id": self.cfg.dataset_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "collection_count": len(collection_ids),
            "collections": collection_ids,
            "requirement_count": len(all_requirement_records),
            "run_count": len(all_run_records),
            "tiers": tiers,
            "requirements_jsonl": str(self.requirements_jsonl_path),
            "runs_jsonl": str(self.runs_jsonl_path),
            "artifacts_root": str(self.collection_artifacts_root.parent),
        }

        self._write_json(self.collection_manifest_path, collection_summary)
        self._write_json(self.dataset_manifest_path, dataset_manifest)
        self._write_json(self.requirement_summary_json_path, {"requirements": requirement_summaries})
        self._write_json(self.tier_summary_json_path, {"tiers": tier_summaries})
        self._write_json(
            self.depth_tier_summary_json_path, {"depth_tiers": depth_tier_summaries}
        )
        self._write_json(self.overall_summary_path, overall)

        self._write_csv(
            self.requirement_summary_csv_path,
            self._flatten_requirement_summary_rows(requirement_summaries),
        )
        self._write_csv(
            self.tier_summary_csv_path,
            self._flatten_tier_summary_rows(tier_summaries),
        )
        self._write_csv(
            self.depth_tier_summary_csv_path,
            self._flatten_depth_tier_summary_rows(depth_tier_summaries),
        )

        logger.info("Executor tier dataset collection complete: %s", json.dumps(collection_summary, indent=2))
        return collection_summary


def _expand_env_vars(raw_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Environment variable {var_name} referenced in config but not set")
        return value

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, raw_text)


def load_config(path: Path) -> ExecutorTierDatasetConfig:
    raw = path.read_text(encoding="utf-8")
    section = (yaml.safe_load(_expand_env_vars(raw)) or {}).get("collection", {})
    if not section:
        raise ValueError("Missing 'collection' section in config")

    repeats = int(section.get("executor_repeats_per_requirement", 1))
    if repeats < 1:
        raise ValueError("executor_repeats_per_requirement must be >= 1")

    heartbeat_interval_seconds = float(section.get("heartbeat_interval_seconds", 30))
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be > 0")

    timeout_raw = section.get("evaluation_timeout_seconds")
    if timeout_raw is None:
        evaluation_timeout_seconds = None
    else:
        evaluation_timeout_seconds = float(timeout_raw)
        if evaluation_timeout_seconds <= 0:
            raise ValueError("evaluation_timeout_seconds must be > 0 when provided")

    raw_tiers = section.get("effort_tiers")
    if raw_tiers is None:
        raw_tiers = [section.get("effort_tier")]
    elif not isinstance(raw_tiers, list):
        raw_tiers = [raw_tiers]
    effort_tiers = [None if t is None else str(t).strip().lower() for t in raw_tiers] or [None]

    plan_path = section.get("plan_path")
    if not plan_path:
        raise ValueError("plan_path is required")

    def _as_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    prompt_char_tolerance = float(section.get("prompt_char_tolerance", 0.15))
    if not 0.0 < prompt_char_tolerance < 1.0:
        raise ValueError("prompt_char_tolerance must be in (0, 1)")

    prompt_generation_max_attempts = int(section.get("prompt_generation_max_attempts", 3))
    if prompt_generation_max_attempts < 1:
        raise ValueError("prompt_generation_max_attempts must be >= 1")

    return ExecutorTierDatasetConfig(
        dataset_id=str(section.get("dataset_id", "executor_tier_dataset_v3")),
        collection_id=str(section.get("collection_id", "collection_001")),
        source_pdf_dir=section.get("source_pdf_dir", "data/corpus"),
        file_glob=section.get("file_glob", "*.pdf"),
        workspace_dir=section.get("workspace_dir", "data/workspaces/collection"),
        provider_agent_id=str(section.get("provider_agent_id", "dataset-provider")),
        architecture=str(section.get("architecture", "1")),
        effort_tiers=effort_tiers,
        executor_repeats_per_requirement=repeats,
        force_rebuild_rag=bool(section.get("force_rebuild_rag", True)),
        reset_executor_between_runs=bool(section.get("reset_executor_between_runs", False)),
        include_partial_fallback_runs=bool(section.get("include_partial_fallback_runs", True)),
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        overwrite=bool(section.get("overwrite", False)),
        plan_path=str(plan_path),
        plan_slot_ids=_as_str_list(section.get("plan_slot_ids")),
        plan_splits=[s.lower() for s in _as_str_list(section.get("plan_splits"))],
        prompt_char_tolerance=prompt_char_tolerance,
        prompt_generation_max_attempts=prompt_generation_max_attempts,
        reuse_requirements_from=(
            None
            if section.get("reuse_requirements_from") is None
            else str(section.get("reuse_requirements_from"))
        ),
        effort_budget=section.get("effort_budget") or None,
        requirements_only=bool(section.get("requirements_only", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the recorded service dataset (makes paid LLM calls)")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/dataset/collection.yaml",
        help="Path to the collection YAML config",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional runtime override for executor repeats per requirement",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Optional runtime override for collection_id",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing collection with the same collection_id",
    )
    parser.add_argument(
        "--slots",
        type=str,
        default=None,
        help=(
            "Comma-separated plan slot_ids to collect, e.g. for a pilot before spending the rest"
        ),
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help=(
            "Override workspace_dir. Each parallel worker needs its own: concurrent "
            "RAG index rebuilds in one directory corrupt each other"
        ),
    )
    parser.add_argument(
        "--requirements-only",
        action="store_true",
        help="Generate and persist requirements, then stop",
    )
    parser.add_argument(
        "--reuse-requirements-from",
        type=str,
        default=None,
        help=(
            "Reuse another collection's requirement text instead of generating new prompts"
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated plan splits to collect, e.g. dev",
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout per evaluator call; timed-out calls are recorded as failed",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=None,
        help="Heartbeat log interval while waiting for evaluator calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input/config and print planned run size",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def _async_main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config)
    if args.repeats is not None:
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        cfg.executor_repeats_per_requirement = int(args.repeats)
    if args.collection_id is not None:
        cfg.collection_id = str(args.collection_id)
    if args.slots is not None:
        cfg.plan_slot_ids = [part.strip() for part in args.slots.split(",") if part.strip()]
    if args.splits is not None:
        cfg.plan_splits = [part.strip().lower() for part in args.splits.split(",") if part.strip()]
    if args.reuse_requirements_from is not None:
        cfg.reuse_requirements_from = str(args.reuse_requirements_from)
    if args.requirements_only:
        cfg.requirements_only = True
    if args.workspace_dir is not None:
        cfg.workspace_dir = str(args.workspace_dir)
    if args.overwrite:
        cfg.overwrite = True
    if args.evaluation_timeout_seconds is not None:
        if args.evaluation_timeout_seconds <= 0:
            raise ValueError("--evaluation-timeout-seconds must be > 0")
        cfg.evaluation_timeout_seconds = float(args.evaluation_timeout_seconds)
    if args.heartbeat_interval_seconds is not None:
        if args.heartbeat_interval_seconds <= 0:
            raise ValueError("--heartbeat-interval-seconds must be > 0")
        cfg.heartbeat_interval_seconds = float(args.heartbeat_interval_seconds)

    collector = ExecutorTierDatasetCollector(cfg)
    result = await collector.run(dry_run=args.dry_run)
    if args.dry_run:
        logger.info("Dry-run plan: %s", json.dumps(result, indent=2))
    return 0


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Dataset collection failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())