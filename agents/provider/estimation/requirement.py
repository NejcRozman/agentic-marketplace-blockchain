"""What an estimator sees of a request: its text and countable features, never its outcomes."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class VisibleRequirement:
    """Exactly what an estimator may condition on.

    ``probe_id`` is a label for the caller and is never rendered.
    """

    probe_id: str
    title: str
    description: str
    service_type: str
    prompts: Tuple[str, ...]
    source_pdf_count: int
    source_pdf_names: Tuple[str, ...]

    def as_payload(self) -> Dict[str, Any]:
        """The dict rendered into the prompt. No identifiers, by construction."""
        return {
            "title": self.title,
            "description": self.description,
            "service_type": self.service_type,
            "source_pdf_count": self.source_pdf_count,
            "source_pdf_names": list(self.source_pdf_names),
            "prompts": list(self.prompts),
        }


def _pdf_stem(path: str) -> str:
    """``Rozman_Emergence.pdf`` -> ``Rozman_Emergence``: the file name without its directory."""
    return Path(str(path)).stem


def visible_view(record: Mapping[str, Any], probe_id: str) -> VisibleRequirement:
    """Project a ``requirements.jsonl`` record onto the allow-list.

    Reads only the named fields. A record carrying a new identifier field will
    produce the same view it produced before that field existed.
    """
    inner = record.get("requirements") or {}
    prompts = tuple(str(p) for p in (inner.get("prompts") or []))
    # Source PDFs appear both top-level and nested; prefer the top-level list
    # and fall back, because a record may carry only one of them.
    raw_pdfs = record.get("source_pdfs") or inner.get("source_pdfs") or []
    return VisibleRequirement(
        probe_id=probe_id,
        title=str(inner.get("title") or ""),
        description=str(inner.get("description") or ""),
        service_type=str(inner.get("service_type") or ""),
        prompts=prompts,
        source_pdf_count=int(record.get("source_pdf_count") or len(raw_pdfs)),
        source_pdf_names=tuple(_pdf_stem(p) for p in raw_pdfs),
    )


#: The countable features Regression fits on.
SURFACE_FEATURES: Tuple[str, ...] = (
    "total_prompt_chars",
    "source_pdf_count",
    "prompt_count",
)

#: A prompt is "wide" when it names at least this many distinct source papers.
WIDE_PROMPT_THRESHOLD = 3


def _stem_pattern(stem: str) -> re.Pattern:
    # Word-ish boundaries that treat "_" as part of the token, so
    # "Rozman_Emergence" does not match inside "Rozman_EmergenceExtra".
    return re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(stem) + r"(?![A-Za-z0-9_])"
    )


def count_paper_mentions(prompt: str, pdf_names: Sequence[str]) -> int:
    """Distinct source papers named in one prompt, counted by exact stem match.

    Exact matching means the same thing in every request; accepting bare author
    prefixes would depend on which papers a request happens to include.
    """
    return sum(1 for stem in pdf_names if _stem_pattern(stem).search(prompt))


def mechanical_features(visible: VisibleRequirement) -> Dict[str, float]:
    """Features B1/B1b regress on, derived only from the visible view.

    Identical inputs to the LLM's, so "the LLM beat B1b" cannot be an artefact
    of the baseline being shown less.
    """
    mentions = [
        count_paper_mentions(prompt, visible.source_pdf_names)
        for prompt in visible.prompts
    ]
    return {
        "total_prompt_chars": float(sum(len(p) for p in visible.prompts)),
        "source_pdf_count": float(visible.source_pdf_count),
        "prompt_count": float(len(visible.prompts)),
        "mean_papers_per_prompt": (
            float(statistics.mean(mentions)) if mentions else 0.0
        ),
        "wide_prompt_count": float(
            sum(1 for m in mentions if m >= WIDE_PROMPT_THRESHOLD)
        ),
    }
