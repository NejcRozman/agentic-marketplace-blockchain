"""The consumer's request generator: an LLM writes the prompts of a literature-review job from paper abstracts.

Used by the live market mode and by the dataset collection, which writes each
request to a slot of the requirement plan (:meth:`ServiceGenerator.generate_prompts_for_plan_slot`).
"""

import logging
import asyncio
import random
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import json
from langchain_openai import ChatOpenAI

try:
    from ..config import Config
    from ..common.ipfs_client import IPFSClient
except ImportError:
    from config import Config
    from common.ipfs_client import IPFSClient

logger = logging.getLogger(__name__)


#: The designed synthesis depths of the requirement plan. Depth drives *demanded*
#: retrieval; the effort tier caps *allowed* retrieval.
DEPTH_LADDER: Dict[str, Dict[str, Any]] = {
    "locate": {
        "demanded_retrieval_rounds": 1,
        "evidence_items": "exactly ONE specific detail, in total, across the whole prompt",
        "instruction": (
            "Ask for one specific detail the assigned source *foregrounds* - its headline "
            "result, the name of the system or method it proposes, its main dataset or "
            "benchmark. One well-formed search must be able to surface it, so do not ask for "
            "incidental settings or details buried in an appendix. Do not ask for comparison or "
            "synthesis, and do not ask for a second detail."
        ),
    },
    "enumerate": {
        "demanded_retrieval_rounds": 3,
        "evidence_items": "ONE specific detail from EACH assigned source",
        "instruction": (
            "Ask what each assigned source reports about one specific aspect, source by source, "
            "naming the concrete detail wanted from each (its measured result, its setup, its "
            "stated limitation). It must be answerable by visiting each source in turn. Do not "
            "ask the reader to argue across the sources or resolve differences."
        ),
    },
    "contrast": {
        "demanded_retrieval_rounds": 6,
        "evidence_items": (
            "TWO specific details from EACH assigned source (what it measured and under what "
            "setup), plus the comparison between them"
        ),
        "instruction": (
            "Ask how the assigned sources differ on a specific technical point - their measured "
            "results, evaluation setup, or assumptions - and why that difference arises. It must "
            "require locating comparable concrete evidence in every assigned source and "
            "explaining the difference, not merely listing both positions."
        ),
    },
    "reconcile": {
        "demanded_retrieval_rounds": 10,
        "evidence_items": (
            "THREE specific details from EACH assigned source (its reported result, the setup or "
            "assumption behind it, and its stated limitation), plus what would settle the "
            "disagreement between them"
        ),
        "instruction": (
            "Ask the reader to reconcile a genuine disagreement between the assigned sources: "
            "which specific reported results are in tension, which differences in method, "
            "setting, or measurement produce the disagreement, and what evidence would settle "
            "it. It must require repeatedly returning to the sources for details that are never "
            "stated side by side."
        ),
    },
}

#: Whether the evidence a prompt needs sits in one place or has to be assembled.
#: Carried per requirement by the plan and stated in the generation prompt.
DISPERSION_INSTRUCTIONS: Dict[str, str] = {
    "localised": (
        "Evidence placement: each prompt should target detail that sits together in one part of "
        "one source, so a reader who finds the right passage has the whole answer."
    ),
    "scattered": (
        "Evidence placement: each prompt must require detail from at least two different places - "
        "different sources, or different parts of one source (its setup versus its results versus "
        "its stated limitations). A reader must not be able to answer from a single passage."
    ),
}

# Prompts must demand detail that cannot be recalled, only looked up: otherwise
# the executor can answer well-known papers from prior knowledge.
SPECIFICITY_RULES = """Every prompt must be unanswerable from general familiarity with this
literature. Someone who has read about this topic but does not have these specific documents open
must not be able to answer. Demand the kind of detail that only the document carries: reported
numbers, parameter values, dataset and baseline names, named components, stated limitations, exact
conditions of an experiment.

Ask for the kind of detail, not its location: you are working from abstracts and cannot see tables,
figures, or section numbers, so never refer to "Table 3", "Section 4", or "the abstract". The
analyst answering these prompts has the full papers and will search them."""


class ServiceGenerator:
    """Generates literature-review requests from one or more paper abstracts."""
    
    def __init__(self, config: Config):
        self.config = config
        self.ipfs_client = IPFSClient()
        self.processed_pdfs: Set[str] = set()  # PDFs already used in this session
        self.service_generation_timeout = max(1, int(getattr(config, "service_generation_timeout", 180)))
        
        # The model comes from the config, so it is recorded per collection.
        self.model_name = str(
            getattr(config, "service_generation_model", None) or "openai/gpt-oss-20b"
        )
        self.model = ChatOpenAI(
            model=self.model_name,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=0.7
        )

        # Consumer-side spend of all generation calls, for the collection manifest.
        self.generation_llm_calls = 0
        self.generation_input_tokens = 0
        self.generation_output_tokens = 0
        self._input_price_per_1k = float(getattr(config, "consumer_llm_input_price_per_1k", 0.0) or 0.0)
        self._output_price_per_1k = float(getattr(config, "consumer_llm_output_price_per_1k", 0.0) or 0.0)

        logger.info("ServiceGenerator initialized with OpenRouter LLM %s", self.model_name)

    def _price(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000.0) * self._input_price_per_1k
            + (output_tokens / 1000.0) * self._output_price_per_1k
        )

    def generation_usage(self) -> Dict[str, Any]:
        """Total request-generation spend so far, for the collection manifest."""
        return {
            "model": self.model_name,
            "llm_calls": self.generation_llm_calls,
            "input_tokens": self.generation_input_tokens,
            "output_tokens": self.generation_output_tokens,
            "cost_usd": round(
                self._price(self.generation_input_tokens, self.generation_output_tokens), 8
            ),
        }

    def _record_model_usage(self, response: Any) -> Dict[str, int]:
        """Add one LLM response's token usage to the running totals.

        Both of LangChain's usage shapes are tried; an unusable response counts as a call
        with zero tokens.
        """
        input_tokens = 0
        output_tokens = 0

        def _non_negative(value: Any) -> int:
            try:
                return max(0, int(value or 0))
            except (TypeError, ValueError):
                return 0

        try:
            usage_meta = getattr(response, "usage_metadata", None) or {}
            input_tokens = _non_negative(
                usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens")
            )
            output_tokens = _non_negative(
                usage_meta.get("output_tokens") or usage_meta.get("completion_tokens")
            )

            if (input_tokens + output_tokens) == 0:
                token_usage = (getattr(response, "response_metadata", None) or {}).get(
                    "token_usage", {}
                ) or {}
                input_tokens = _non_negative(
                    token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                )
                output_tokens = _non_negative(
                    token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                )
        except Exception as exc:  # provider metadata is best-effort
            logger.warning("Could not read token usage from generation response: %s", exc)
            input_tokens = 0
            output_tokens = 0

        self.generation_llm_calls += 1
        self.generation_input_tokens += input_tokens
        self.generation_output_tokens += output_tokens
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
    
    async def generate_services_from_pdfs(
        self,
        pdf_dir: Path,
        skip_processed: bool = True,
        pdfs_per_service: int = 1,
        sampling_mode: str = "sequential_groups",
        sampling_iterations: int = 1,
        sampling_seed: int = None,
    ) -> Dict[str, Any]:
        """Generate job descriptions from the PDFs (with abstracts) in a directory, for the live market mode.

        For each job: read the abstracts, generate the prompts, pin the PDFs and the job
        description to IPFS. ``sampling_mode`` "sequential_groups" chunks the PDFs in
        file-name order; "random_iterations" draws ``pdfs_per_service`` PDFs per job.
        Returns the ``processed``, ``skipped`` and ``failed`` jobs.
        """
        logger.info(f"Generating services from PDFs in {pdf_dir}")
        
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        
        if not pdf_paths:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {"processed": [], "skipped": [], "failed": []}
        
        logger.info(f"Found {len(pdf_paths)} PDF files")

        configured_pdfs_per_service = int(
            pdfs_per_service
            or getattr(self.config, "pdfs_per_service", 1)
            or 1
        )
        resolved_pdfs_per_service = max(1, configured_pdfs_per_service)

        configured_sampling_mode = str(
            sampling_mode
            or getattr(self.config, "service_generation_mode", "sequential_groups")
            or "sequential_groups"
        ).strip().lower()
        if configured_sampling_mode not in {"sequential_groups", "random_iterations"}:
            logger.warning(
                "Unknown sampling_mode '%s'; using sequential_groups",
                configured_sampling_mode,
            )
            configured_sampling_mode = "sequential_groups"

        configured_sampling_iterations = int(
            sampling_iterations
            or getattr(self.config, "service_generation_iterations", 1)
            or 1
        )
        resolved_sampling_iterations = max(1, configured_sampling_iterations)

        resolved_sampling_seed = sampling_seed
        if resolved_sampling_seed is None:
            resolved_sampling_seed = getattr(self.config, "service_generation_seed", None)

        logger.info(
            "Service generation settings: timeout=%ss, pdfs_per_service=%s, mode=%s, iterations=%s, seed=%s",
            self.service_generation_timeout,
            resolved_pdfs_per_service,
            configured_sampling_mode,
            resolved_sampling_iterations,
            resolved_sampling_seed,
        )
        
        processed = []
        skipped = []
        failed = []
        
        if configured_sampling_mode == "random_iterations":
            if skip_processed and self.processed_pdfs:
                logger.info(
                    "random_iterations mode ignores processed-pdf cache and samples from full pool"
                )

            sample_size = min(resolved_pdfs_per_service, len(pdf_paths))
            if sample_size < resolved_pdfs_per_service:
                logger.warning(
                    "Requested pdfs_per_service=%s but only %s PDFs are available; sampling %s per iteration",
                    resolved_pdfs_per_service,
                    len(pdf_paths),
                    sample_size,
                )

            rng = random.Random(resolved_sampling_seed)
            logger.info(
                "Creating %s service(s) via random sampling from %s PDF(s)",
                resolved_sampling_iterations,
                len(pdf_paths),
            )

            for iteration_idx in range(1, resolved_sampling_iterations + 1):
                sampled_group = rng.sample(pdf_paths, k=sample_size)
                group_names = [pdf.name for pdf in sampled_group]
                group_label = ", ".join(group_names)

                try:
                    result = await asyncio.wait_for(
                        self._generate_service_for_pdf_group(sampled_group),
                        timeout=self.service_generation_timeout
                    )
                    result["sampling_iteration"] = iteration_idx
                    processed.append(result)
                    logger.info(
                        "✓ Generated sampled service %s/%s for PDFs: %s",
                        iteration_idx,
                        resolved_sampling_iterations,
                        group_label,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "❌ Timeout generating sampled service %s/%s for PDFs [%s] after %ss",
                        iteration_idx,
                        resolved_sampling_iterations,
                        group_label,
                        self.service_generation_timeout,
                    )
                    failed.append({
                        "iteration": iteration_idx,
                        "pdf_names": group_names,
                        "error": f"Service generation timeout after {self.service_generation_timeout}s"
                    })
                    continue
                except Exception as e:
                    logger.error(
                        "❌ Failed to generate sampled service %s/%s for PDFs [%s]: %s",
                        iteration_idx,
                        resolved_sampling_iterations,
                        group_label,
                        e,
                    )
                    failed.append({
                        "iteration": iteration_idx,
                        "pdf_names": group_names,
                        "error": str(e),
                    })
                    continue
        else:
            candidate_pdf_paths: List[Path] = []
            for pdf_path in pdf_paths:
                pdf_name = pdf_path.name
                if skip_processed and pdf_name in self.processed_pdfs:
                    logger.info(f"⏭️  Skipping {pdf_name} (already processed)")
                    skipped.append(pdf_name)
                    continue
                candidate_pdf_paths.append(pdf_path)

            if not candidate_pdf_paths:
                logger.info("No new PDFs to process after applying skip_processed filter")
                return {"processed": [], "skipped": skipped, "failed": []}

            pdf_groups: List[List[Path]] = [
                candidate_pdf_paths[i:i + resolved_pdfs_per_service]
                for i in range(0, len(candidate_pdf_paths), resolved_pdfs_per_service)
            ]

            logger.info(
                "Creating %s service(s) from %s PDF(s)",
                len(pdf_groups),
                len(candidate_pdf_paths),
            )

            for group_idx, pdf_group in enumerate(pdf_groups, start=1):
                group_names = [pdf.name for pdf in pdf_group]
                group_label = ", ".join(group_names)

                try:
                    result = await asyncio.wait_for(
                        self._generate_service_for_pdf_group(pdf_group),
                        timeout=self.service_generation_timeout
                    )
                    processed.append(result)
                    for pdf_name in group_names:
                        self.processed_pdfs.add(pdf_name)
                    logger.info(
                        "✓ Generated service %s/%s for PDFs: %s",
                        group_idx,
                        len(pdf_groups),
                        group_label,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "❌ Timeout generating service for PDFs [%s] after %ss",
                        group_label,
                        self.service_generation_timeout,
                    )
                    if len(group_names) == 1:
                        failed.append({
                            "pdf_name": group_names[0],
                            "error": f"Service generation timeout after {self.service_generation_timeout}s"
                        })
                    else:
                        failed.append({
                            "pdf_names": group_names,
                            "error": f"Service generation timeout after {self.service_generation_timeout}s"
                        })
                    continue
                except Exception as e:
                    logger.error(f"❌ Failed to generate service for PDFs [{group_label}]: {e}")
                    if len(group_names) == 1:
                        failed.append({"pdf_name": group_names[0], "error": str(e)})
                    else:
                        failed.append({"pdf_names": group_names, "error": str(e)})
                    continue
        
        logger.info(
            f"✓ Results: {len(processed)} processed, "
            f"{len(skipped)} skipped, {len(failed)} failed"
        )
        
        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed
        }
    
    async def _generate_service_for_pdf_group(
        self,
        pdf_paths: List[Path]
    ) -> Dict[str, Any]:
        """Generate and publish one job description from one or more PDFs."""
        if not pdf_paths:
            raise ValueError("pdf_paths cannot be empty")

        pdf_names = [pdf_path.name for pdf_path in pdf_paths]

        paper_abstracts: List[Dict[str, str]] = []
        for pdf_path in pdf_paths:
            abstract = self._read_abstract(pdf_path)
            paper_abstracts.append(
                {
                    "source_id": pdf_path.stem,
                    "abstract": abstract,
                }
            )

        prompts = await self._generate_prompts_from_abstracts(paper_abstracts)

        pdf_cids: Dict[str, str] = {}
        for pdf_path in pdf_paths:
            logger.info(f"  Uploading {pdf_path.name}...")
            pdf_result = await self.ipfs_client.pin_file(
                file_path=pdf_path,
                name=pdf_path.name,
                metadata={"type": "input_file", "format": "pdf"}
            )

            if not pdf_result.success:
                raise Exception(f"Failed to upload PDF {pdf_path.name}: {pdf_result.error}")

            pdf_cids[pdf_path.name] = str(pdf_result.cid)
            logger.info(f"  ✅ PDF {pdf_path.name}: {pdf_result.cid}")

        if len(pdf_paths) == 1:
            primary_stem = pdf_paths[0].stem
            service_title = f"Literature Review: {primary_stem}"
            service_description = (
                f"Comprehensive literature review and analysis of {primary_stem}"
            )
        else:
            preview_titles = [pdf.stem for pdf in pdf_paths[:2]]
            suffix = f" + {len(pdf_paths) - 2} more" if len(pdf_paths) > 2 else ""
            service_title = f"Literature Review: {' & '.join(preview_titles)}{suffix}"
            service_description = (
                f"Comprehensive comparative literature review and synthesis across {len(pdf_paths)} papers."
            )

        service_desc = {
            "title": service_title,
            "description": service_description,
            "service_type": "literature_review",
            "prompts": prompts,
            "source_pdfs": pdf_names,
            "quality_criteria": {
                "completeness": "All prompts must be answered thoroughly",
                "depth": "Answers should be detailed and well-supported with evidence from the papers",
                "citations": "Include relevant citations and specific references to the papers' content",
                "clarity": "Clear, well-structured responses that demonstrate deep understanding",
                "synthesis": "Identify agreements, disagreements, and cross-paper insights when applicable"
            }
        }

        if len(pdf_paths) == 1:
            service_desc["input_files_cid"] = next(iter(pdf_cids.values()))
            service_desc["input_file_name"] = pdf_paths[0].name
        else:
            service_desc["input_files"] = pdf_cids

        service_result = await self.ipfs_client.pin_json(
            data=service_desc,
            name=f"service-{service_title[:30]}",
            metadata={"type": "service_description"}
        )

        if not service_result.success:
            raise Exception(f"Failed to upload service: {service_result.error}")

        service_cid = service_result.cid
        logger.info(f"  ✅ Service: {service_cid}")

        if len(pdf_paths) == 1:
            return {
                "service_cid": service_cid,
                "pdf_cid": next(iter(pdf_cids.values())),
                "pdf_name": pdf_paths[0].name,
                "title": service_desc["title"],
                "prompts": prompts
            }

        return {
            "service_cid": service_cid,
            "pdf_names": pdf_names,
            "pdf_cids": pdf_cids,
            "pdf_count": len(pdf_paths),
            "title": service_desc["title"],
            "prompts": prompts
        }
    
    def _read_abstract(self, pdf_path: Path) -> str:
        """The abstract in the ``.txt`` file next to a PDF."""
        abstract_path = pdf_path.with_suffix('.txt')
        
        if not abstract_path.exists():
            raise FileNotFoundError(
                f"Abstract file not found: {abstract_path}. "
                f"Each PDF must have a corresponding .txt file with the abstract."
            )
        
        with open(abstract_path, 'r', encoding='utf-8') as f:
            abstract = f.read().strip()
        
        if not abstract:
            raise ValueError(f"Abstract file is empty: {abstract_path}")
        
        logger.info(f"  Read abstract ({len(abstract)} chars)")
        return abstract
    
    @staticmethod
    def _normalize_prompt_source_ids(prompts: List[str], source_ids: List[str]) -> List[str]:
        """Replace "Paper N" references with the source ids."""
        if not source_ids:
            return [str(prompt) for prompt in prompts]

        joined_ids = ", ".join(source_ids)
        normalized_prompts: List[str] = []
        for prompt in prompts:
            updated = str(prompt)
            for idx, source_id in enumerate(source_ids, start=1):
                updated = re.sub(rf"\bPaper\s*{idx}\b", source_id, updated, flags=re.IGNORECASE)

            if len(source_ids) == 1:
                source_id = source_ids[0]
                updated = re.sub(r"\bthis paper\b", source_id, updated, flags=re.IGNORECASE)
                updated = re.sub(r"\bthe paper\b", source_id, updated, flags=re.IGNORECASE)
                if source_id not in updated:
                    updated = f"For {source_id}, {updated[0].lower()}{updated[1:]}" if updated else source_id
            else:
                updated = re.sub(r"\bthis paper\b", "the relevant source", updated, flags=re.IGNORECASE)
                updated = re.sub(r"\bthe paper\b", "the relevant source", updated, flags=re.IGNORECASE)
                updated = re.sub(r"\bthese papers\b", joined_ids, updated, flags=re.IGNORECASE)
                updated = re.sub(r"\bthe papers\b", joined_ids, updated, flags=re.IGNORECASE)
                updated = re.sub(r"\bthese sources\b", joined_ids, updated, flags=re.IGNORECASE)
                if not any(source_id in updated for source_id in source_ids):
                    updated = f"Across {joined_ids}, {updated[0].lower()}{updated[1:]}" if updated else joined_ids

            normalized_prompts.append(updated)

        return normalized_prompts

    async def _generate_prompts_from_abstracts(
        self,
        paper_abstracts: List[Dict[str, str]],
    ) -> List[str]:
        if not paper_abstracts:
            return self._get_default_prompts()

        multi_paper = len(paper_abstracts) > 1
        source_ids = [
            str(item.get("source_id") or item.get("paper_name") or f"source_{idx + 1}")
            for idx, item in enumerate(paper_abstracts)
        ]

        if multi_paper:
            system_prompt = """You are an expert at creating synthesis questions for multi-paper literature reviews.

Given multiple research paper abstracts identified by source IDs, generate 10 specific, thoughtful questions that require comparing, contrasting, and integrating insights across the papers.

Requirements:
- Every question must require evidence from at least 3 source IDs.
- At least half of the questions must require evidence from all provided papers.
- Avoid questions answerable through direct extraction from a single paper.
- Prefer synthesis, integration, contradiction detection, prioritization, and framework-building questions.
- Questions should require combining evidence from multiple papers before an answer can be produced.
- Avoid simple summary, methodology description, or contribution listing questions.

Your questions should cover:
- Relative research goals and contributions
- Methodological similarities/differences
- Agreements and contradictions in findings
- Shared and unique limitations
- Combined implications and future directions

Make questions specific to these papers, not generic.

Output ONLY a valid JSON array of strings with no additional text or formatting.
Example: ["How do Source_A and Source_B differ in...", "Which findings are consistent across Source_A, Source_B, and Source_C?"]"""
            user_prompt = """
Source IDs and abstracts:
{papers_block}

Generate 10 synthesis questions for this multi-paper literature review:
""".strip().format(
                papers_block="\n\n".join(
                    [
                        f"Source ID: {source_ids[idx]}\n"
                        f"Abstract:\n{item.get('abstract', '')}"
                        for idx, item in enumerate(paper_abstracts)
                    ]
                )
            )
        else:
            source_id = source_ids[0]
            abstract = str(paper_abstracts[0].get("abstract", ""))
            system_prompt = """You are an expert at creating insightful questions for literature reviews.

Given a research paper abstract and its source ID, generate 5 specific, thoughtful questions that would help someone thoroughly understand and analyze the paper.

Requirements:
- Use the exact source ID provided below when referring to the paper.
- Do not rename the paper.
- Never refer to it as Paper 1.

Your questions should cover:
- Main research question and contribution
- Methodology and approach
- Key findings and results
- Limitations and future directions
- Broader implications or applications

Make questions specific to the paper's content, not generic.

Output ONLY a valid JSON array of strings with no additional text or formatting.
Example: ["What is the main...", "How does the paper...", "What are the key..."]"""
            user_prompt = f"""Source ID: {source_id}

Abstract:
{abstract}

Generate 5 specific questions for this paper's literature review:"""
        
        try:
            response = self.model.invoke(
                f"{system_prompt}\n\n{user_prompt}"
            )
            self._record_model_usage(response)

            response_text = response.content.strip()
            
            # Strip markdown code fences.
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            prompts = json.loads(response_text)
            
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("LLM did not return a valid list of prompts")
            
            prompts = self._normalize_prompt_source_ids([str(p) for p in prompts], source_ids)
            
            logger.info(f"  Generated {len(prompts)} prompts")
            return prompts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response: {response.content}")
            return self._get_default_prompts(multi_paper=multi_paper, source_ids=source_ids)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_default_prompts(multi_paper=multi_paper, source_ids=source_ids)
    
    
    # --- Plan-driven prompt generation (dataset collection) ---

    @staticmethod
    def _assign_prompt_sources(
        source_ids: List[str],
        prompt_breadths: List[int],
    ) -> List[List[str]]:
        """The source ids each planned prompt must span.

        A rotating window, without randomness: prompt i starts at source i and takes
        ``breadth`` sources, wrapping around.
        """
        if not source_ids:
            raise ValueError("source_ids cannot be empty")

        assignments: List[List[str]] = []
        n = len(source_ids)
        for idx, raw_breadth in enumerate(prompt_breadths):
            breadth = max(1, int(raw_breadth))
            if breadth > n:
                logger.warning(
                    "Prompt %s requests breadth %s but only %s sources are available; clamping",
                    idx + 1,
                    breadth,
                    n,
                )
                breadth = n
            start = idx % n
            assignments.append([source_ids[(start + offset) % n] for offset in range(breadth)])

        return assignments

    @staticmethod
    def _mentioned_source_ids(prompt: str, source_ids: List[str]) -> List[str]:
        """The source ids a prompt names (case-insensitive)."""
        lowered = str(prompt).lower()
        return [source_id for source_id in source_ids if source_id.lower() in lowered]

    @staticmethod
    def _repair_prompt_source_mentions(
        prompts: List[str],
        assignments: List[List[str]],
    ) -> Tuple[List[str], bool]:
        """Make every prompt name the sources its slot assigns to it.

        The number of sources a prompt spans is part of the design, so missing ones are
        prefixed; extra mentions are left in place and reported by the caller.
        """
        repaired: List[str] = []
        repaired_any = False
        for prompt, assigned in zip(prompts, assignments):
            text = str(prompt).strip()
            missing = [
                source_id
                for source_id in assigned
                if source_id.lower() not in text.lower()
            ]
            if missing:
                repaired_any = True
                text = f"Drawing on {', '.join(missing)}: {text}"
            repaired.append(text)

        return repaired, repaired_any

    def _depth_fallback_prompts(self, assignments: List[List[str]], depth: str) -> List[str]:
        """Fixed prompts shaped by depth, used only when the LLM fails."""
        templates = {
            "locate": "In {sources}, what specific result, definition, or reported number is stated for the paper's central claim?",
            "enumerate": "For each of {sources}, list what that source reports about its method and its main measured outcome.",
            "contrast": "How do the approaches and measured results of {sources} differ on their shared question, and what explains the difference?",
            "reconcile": (
                "{sources} disagree about their shared question. Reconcile them: state precisely what is in "
                "tension, identify which differences in method, setting, or measurement produce the "
                "disagreement, and say what evidence would settle it."
            ),
        }
        template = templates[depth]
        return [template.format(sources=", ".join(assigned)) for assigned in assignments]

    def _build_slot_prompt_request(
        self,
        paper_abstracts: List[Dict[str, str]],
        source_ids: List[str],
        assignments: List[List[str]],
        depth: str,
        target_prompt_chars: int,
        char_band: Tuple[int, int],
        length_feedback: Optional[str],
        dispersion: str,
    ) -> str:
        ladder = DEPTH_LADDER[depth]
        evidence_items = ladder["evidence_items"]
        prompt_count = len(assignments)
        per_prompt_chars = max(60, int(round(target_prompt_chars / prompt_count)))
        approx_sentences = max(1, round(per_prompt_chars / 120))
        low, high = char_band
        dispersion_rule = DISPERSION_INSTRUCTIONS.get(
            str(dispersion).strip().lower(), DISPERSION_INSTRUCTIONS["localised"]
        )

        system_prompt = f"""You are designing the prompt set for a literature-analysis service over a fixed set of research papers.

Every prompt must be at exactly one demanded synthesis depth: {depth}.
{depth}: {ladder['instruction']}

{SPECIFICITY_RULES}

{dispersion_rule}

Hard requirements:
- Produce exactly {prompt_count} prompts, in the order of the specifications below.
- Each prompt must name every Source ID assigned to it, using the exact Source ID string, and must not name any other source.
- Each prompt must stay at the {depth} depth. Do not make a shallow prompt deeper or a deep prompt shallower.
- Each prompt must require exactly this much evidence, no more and no less: {evidence_items}. This is what separates the depths and it must not drift.
- Each prompt must be about {per_prompt_chars} characters — roughly {approx_sentences} sentence(s) — so the set totals close to {target_prompt_chars} characters (acceptable range {low}-{high}).
- **Length and evidence are independent.** Reach the character target with framing — why the detail matters, what the analyst should do with it, how to present it — and NEVER by asking for additional details. A long prompt at {depth} depth asks for the same evidence as a short one, in more words.
- Prompts must be specific to the content of the abstracts below, not generic.

Output ONLY a valid JSON array of {prompt_count} strings, with no additional text or formatting."""

        papers_block = "\n\n".join(
            f"Source ID: {source_ids[idx]}\nAbstract:\n{item.get('abstract', '')}"
            for idx, item in enumerate(paper_abstracts)
        )
        spec_block = "\n".join(
            f"{idx}. Sources: {', '.join(assigned)} | about {per_prompt_chars} characters"
            for idx, assigned in enumerate(assignments, start=1)
        )

        user_prompt = f"""Source IDs and abstracts:
{papers_block}

Prompt specifications:
{spec_block}

Generate the {prompt_count} prompts at {depth} depth:"""

        if length_feedback:
            user_prompt = f"{user_prompt}\n\n{length_feedback}"

        return f"{system_prompt}\n\n{user_prompt}"

    def _parse_prompt_array(self, response_text: str, expected_count: int) -> Optional[List[str]]:
        """Parse the model's JSON array; None when unusable."""
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse slot prompt response as JSON: %s", exc)
            return None

        if not isinstance(parsed, list) or not parsed:
            logger.error("Slot prompt response was not a non-empty JSON array")
            return None

        prompts = [str(item).strip() for item in parsed if str(item).strip()]
        if len(prompts) != expected_count:
            logger.error(
                "Slot prompt response returned %s prompts, expected %s",
                len(prompts),
                expected_count,
            )
            return None

        return prompts

    async def generate_prompts_for_plan_slot(
        self,
        paper_abstracts: List[Dict[str, str]],
        depth: str,
        prompt_breadths: List[int],
        target_prompt_chars: int,
        char_tolerance: float = 0.15,
        max_attempts: int = 3,
        evidence_dispersion: str = "localised",
    ) -> Dict[str, Any]:
        """Generate one plan slot's prompts under depth, breadth and length control.

        The total length is measured after every attempt and the prompts regenerated while
        it falls outside the slot's band, so length is set independently of depth.
        Returns the prompts and the realised values recorded with the request.
        """
        if not paper_abstracts:
            raise ValueError("paper_abstracts cannot be empty")

        depth_key = str(depth).strip().lower()
        if depth_key not in DEPTH_LADDER:
            raise ValueError(
                f"Unknown demanded_synthesis_depth '{depth}'; expected one of {sorted(DEPTH_LADDER)}"
            )

        breadths = [max(1, int(b)) for b in (prompt_breadths or [])]
        if not breadths:
            raise ValueError("prompt_breadths cannot be empty")

        target_chars = int(target_prompt_chars)
        if target_chars <= 0:
            raise ValueError("target_prompt_chars must be > 0")

        tolerance = float(char_tolerance)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("char_tolerance must be in (0, 1)")

        attempts_allowed = max(1, int(max_attempts))

        source_ids = [
            str(item.get("source_id") or item.get("paper_name") or f"source_{idx + 1}")
            for idx, item in enumerate(paper_abstracts)
        ]
        assignments = self._assign_prompt_sources(source_ids, breadths)
        char_band = (
            int(round(target_chars * (1.0 - tolerance))),
            int(round(target_chars * (1.0 + tolerance))),
        )

        best_prompts: Optional[List[str]] = None
        best_total = 0
        repaired_any = False
        attempts_made = 0
        length_feedback: Optional[str] = None
        slot_input_tokens = 0
        slot_output_tokens = 0

        for attempt in range(1, attempts_allowed + 1):
            attempts_made = attempt
            request = self._build_slot_prompt_request(
                paper_abstracts=paper_abstracts,
                source_ids=source_ids,
                assignments=assignments,
                depth=depth_key,
                target_prompt_chars=target_chars,
                char_band=char_band,
                length_feedback=length_feedback,
                dispersion=evidence_dispersion,
            )

            try:
                response = self.model.invoke(request)
                call_usage = self._record_model_usage(response)
                slot_input_tokens += call_usage["input_tokens"]
                slot_output_tokens += call_usage["output_tokens"]
                prompts = self._parse_prompt_array(str(response.content), len(assignments))
            except Exception as exc:
                logger.error("Slot prompt generation failed on attempt %s: %s", attempt, exc)
                prompts = None

            if prompts is None:
                length_feedback = None
                continue

            prompts, repaired = self._repair_prompt_source_mentions(prompts, assignments)
            repaired_any = repaired_any or repaired
            total_chars = sum(len(p) for p in prompts)

            if best_prompts is None or abs(total_chars - target_chars) < abs(best_total - target_chars):
                best_prompts = prompts
                best_total = total_chars

            if char_band[0] <= total_chars <= char_band[1]:
                break

            per_prompt_delta = round((target_chars - total_chars) / len(assignments))
            if per_prompt_delta > 0:
                adjustment = (
                    f"add about {per_prompt_delta} characters to EACH prompt by expanding the "
                    "framing - why the detail matters, how the analyst should present it - while "
                    "asking for exactly the same evidence as before"
                )
            else:
                adjustment = (
                    f"cut about {abs(per_prompt_delta)} characters from EACH prompt by dropping "
                    "context sentences, keeping every specific being asked for"
                )
            length_feedback = (
                f"Your previous attempt totalled {total_chars} characters across "
                f"{len(assignments)} prompts (about {round(total_chars / len(assignments))} each), "
                f"but the target is {target_chars} (about {round(target_chars / len(assignments))} "
                f"each, acceptable range {char_band[0]}-{char_band[1]}). Rewrite the same "
                f"{len(assignments)} questions at the same {depth_key} depth and "
                f"{adjustment}. Keep each prompt's assigned sources unchanged."
            )
            logger.info(
                "Slot prompt attempt %s/%s landed at %s chars (band %s-%s); regenerating",
                attempt,
                attempts_allowed,
                total_chars,
                char_band[0],
                char_band[1],
            )

        fallback_used = best_prompts is None
        if fallback_used:
            logger.error(
                "Slot prompt generation exhausted %s attempts; using deterministic %s fallback",
                attempts_allowed,
                depth_key,
            )
            best_prompts = self._depth_fallback_prompts(assignments, depth_key)
            best_total = sum(len(p) for p in best_prompts)

        realized_breadths = [
            len(self._mentioned_source_ids(prompt, source_ids)) for prompt in best_prompts
        ]
        designed_breadths = [len(assigned) for assigned in assignments]
        length_in_band = char_band[0] <= best_total <= char_band[1]

        if not length_in_band:
            logger.warning(
                "Slot prompts total %s chars, outside band %s-%s after %s attempt(s)",
                best_total,
                char_band[0],
                char_band[1],
                attempts_made,
            )

        logger.info(
            "  Generated %s %s-depth prompts (%s chars, band %s-%s, attempts=%s)",
            len(best_prompts),
            depth_key,
            best_total,
            char_band[0],
            char_band[1],
            attempts_made,
        )

        return {
            "prompts": best_prompts,
            "depth": depth_key,
            "evidence_dispersion": str(evidence_dispersion).strip().lower(),
            "generation_model": self.model_name,
            "generation_input_tokens": slot_input_tokens,
            "generation_output_tokens": slot_output_tokens,
            # Every attempt is counted, including those a length retry discarded.
            "generation_cost_usd": round(self._price(slot_input_tokens, slot_output_tokens), 8),
            "demanded_retrieval_rounds": DEPTH_LADDER[depth_key]["demanded_retrieval_rounds"],
            "source_ids": source_ids,
            "prompt_source_assignments": assignments,
            "designed_prompt_breadths": designed_breadths,
            "realized_prompt_breadths": realized_breadths,
            "breadths_match_design": realized_breadths == designed_breadths,
            "target_prompt_chars": target_chars,
            "realized_prompt_chars": best_total,
            "char_band": list(char_band),
            "length_in_band": length_in_band,
            "attempts": attempts_made,
            "source_mentions_repaired": repaired_any,
            "fallback_used": fallback_used,
        }

    def _get_default_prompts(
        self,
        multi_paper: bool = False,
        source_ids: List[str] = None,
    ) -> List[str]:
        """Default prompts, as a fallback."""
        source_ids = [str(source_id) for source_id in (source_ids or []) if str(source_id).strip()]

        if multi_paper:
            joined_ids = ", ".join(source_ids) if source_ids else "the provided sources"
            first_source = source_ids[0] if len(source_ids) >= 1 else "the first source"
            second_source = source_ids[1] if len(source_ids) >= 2 else first_source
            return [
                f"How do {first_source} and {second_source} differ in their main contributions, and where do they overlap?",
                f"How do the methodologies compare across {joined_ids}?",
                f"Which findings agree or conflict across {joined_ids}?",
                f"What limitations are shared across {joined_ids}, and which are unique to individual sources?",
                f"What combined implications and future research directions emerge from {joined_ids}?",
            ]

        source_id = source_ids[0] if source_ids else "this source"
        return [
            f"What is the main research question or contribution of {source_id}?",
            f"What methodology or approach does {source_id} use?",
            f"What are the key findings and results of {source_id}?",
            f"What are the limitations and potential areas for future work in {source_id}?",
            f"What are the broader implications or applications of the research in {source_id}?"
        ]
    
