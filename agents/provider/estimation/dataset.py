"""Dataset v1 as the probe sees it — and the leakage boundary (S2 §2).

Two jobs, kept in one file because they are the same job seen from two sides.

**What the model may see.** :func:`visible_view` is an explicit *allow-list*, not
a deny-list. A deny-list silently admits any field added to the dataset later,
which on this dataset would be a quiet disaster: ``requirement_id`` embeds the
PDF stems *and* a repeat index, so a model keying on it could match repeats to
each other without reading a word. The allow-list means a new field is invisible
until someone deliberately admits it.

**What a baseline may count.** B1/B1b are the controls that decide whether the
LLM is reading meaning or approximating ``len()``, so their features must be
derivable from exactly the same visible view — otherwise the comparison is
rigged in whichever direction the extra information points.

One thing worth knowing before reading any S2 number: on v1, ``source_pdf_count``
is 5 and ``prompt_count`` is 10 for **all ten requirements**. Two of B1's three
features carry no variance at all, so B1 collapses to a univariate regression on
prompt length. That is a property of the dataset, not a bug here, and
:func:`feature_variance_report` surfaces it rather than letting a degenerate
design matrix be discovered later as a strange coefficient.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agents.common.paths import REPO_ROOT  # noqa: E402

DATASET_V1_DIR = (
    REPO_ROOT
    / "experiments"
    / "executor_analysis"
    / "data"
    / "datasets"
    / "executor_tier_dataset_v1"
)

EFFORT_TIERS: Tuple[str, str, str] = ("low", "medium", "high")

# ---------------------------------------------------------------------------
# The leakage boundary (S2 §2)
# ---------------------------------------------------------------------------

#: Top-level record fields the probe may read. Anything not named here is
#: invisible to the model, including fields that do not exist yet.
VISIBLE_TOP_LEVEL_FIELDS: Tuple[str, ...] = ("source_pdf_count",)

#: Fields inside ``record["requirements"]`` the probe may read. ``input_file``
#: and ``source_pdfs`` are excluded here because the filenames are surfaced
#: separately as bare stems — the full ``utils/files/...`` path is a filesystem
#: location, not task information a provider would be given.
VISIBLE_REQUIREMENT_FIELDS: Tuple[str, ...] = (
    "title",
    "description",
    "service_type",
    "prompts",
)

#: Named for the leakage test, which asserts none of these values survives into
#: the rendered prompt string. ``requirement_id`` is the dangerous one: it is
#: the only field that would let a model recognise the same requirement across
#: repeats and answer from memory rather than from the text.
FORBIDDEN_FIELDS: Tuple[str, ...] = (
    "artifact_dir",
    "requirement_id",
    "dataset_id",
    "collection_id",
    "logical_requirement_idx",
    "repeat_idx",
    "created_at",
)

#: Everything in ``runs.jsonl`` is an outcome. None of it may reach the model —
#: that is the answer sheet.
FORBIDDEN_RUN_FIELDS: Tuple[str, ...] = (
    "execution_cost_usd",
    "evaluator_rating",
    "evaluator_quality_scores",
    "executor_duration_seconds",
    "evaluator_duration_seconds",
    "service_output",
    "evaluation",
    "response_count",
    "tier",
)


class LeakageError(AssertionError):
    """A forbidden value reached something the model would be shown."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisibleRequirement:
    """Exactly what the model is allowed to condition on.

    ``probe_id`` is *not* part of the view — it is the harness's join key back
    to the realized outcomes and is never rendered. It is a positional label
    (``req_01`` …) rather than a hash of the content so that a human reading
    ``results_v1.csv`` can find the row in ``requirements.jsonl``, and it is
    deliberately meaningless to a model that never sees it.
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


@dataclass(frozen=True)
class RealizedOutcome:
    """The recorded truth for one (requirement, tier) cell.

    On v1 a cell is a single run and the spread fields are zero. On v2 a cell is
    the mean of 3 repeats, and ``cost_sd`` / ``quality_sd`` carry the *within*-
    cell spread — the environment's own noise, which S5 §3 needs in order to say
    which between-requirement differences are resolvable at all.
    """

    tier: str
    execution_cost_usd: float
    quality_rating: float
    n_runs: int = 1
    cost_sd: float = 0.0
    quality_sd: float = 0.0


@dataclass
class ProbeRequirement:
    """A requirement joined to its realized outcomes.

    The two halves are kept in separate attributes rather than flattened so
    that handing the visible half to a renderer cannot accidentally carry the
    outcome half along with it.
    """

    probe_id: str
    requirement_id: str
    visible: VisibleRequirement
    outcomes: Dict[str, RealizedOutcome] = field(default_factory=dict)

    #: v2 design metadata. Never rendered — it is the answer sheet for S5 §4
    #: (``depth_rank`` *is* the property the estimator is being asked to infer)
    #: and the split label that decides what may be fitted on. Empty on v1.
    split: str = ""
    pair_id: str = ""
    slot_id: str = ""
    depth: str = ""
    depth_rank: int = 0

    def realized_cost(self, tier: str) -> float:
        return self.outcomes[tier].execution_cost_usd

    def realized_quality(self, tier: str) -> float:
        return self.outcomes[tier].quality_rating

    def has(self, tier: str) -> bool:
        return tier in self.outcomes


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _pdf_stem(path: str) -> str:
    """``utils/files/Rozman_Emergence.pdf`` → ``Rozman_Emergence``.

    The stem is admitted (S2 §2: the filenames carry topic information a real
    provider would see when handed the documents); the directory is not.
    """
    return Path(str(path)).stem


def visible_view(record: Mapping[str, Any], probe_id: str) -> VisibleRequirement:
    """Project a ``requirements.jsonl`` record onto the allow-list.

    Reads only the named fields. A record carrying a new identifier field will
    produce the same view it produced before that field existed.
    """
    inner = record.get("requirements") or {}
    prompts = tuple(str(p) for p in (inner.get("prompts") or []))
    # Source PDFs appear both top-level and nested; prefer the top-level list
    # and fall back, because a v2 record may only carry one of them.
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


class ProbeDataset:
    """Dataset v1 in probe form: visible views, outcomes, and tier levels.

    The **tier level** is the reference the §1 multiplier is expressed against.
    It is the arithmetic mean over requirements, matching how S1 computed the
    task-blind σ of 0.1287, so B0's error here and the σ quoted in
    ``docs/s1_validation.md`` §2 measure the same thing. It is computed
    in-sample, which flatters B0 very slightly; that is deliberate, because B0
    is the bar every other arm must clear and the bar should not be soft.
    """

    def __init__(
        self,
        requirements: Sequence[ProbeRequirement],
        *,
        levels_from: Optional["ProbeDataset"] = None,
    ) -> None:
        if not requirements:
            raise ValueError("probe dataset is empty")
        self.requirements: List[ProbeRequirement] = list(requirements)
        self._tier_cost_level: Dict[str, float] = {}
        self._tier_quality_level: Dict[str, float] = {}
        for tier in EFFORT_TIERS:
            costs = [r.realized_cost(tier) for r in self.requirements if r.has(tier)]
            ratings = [r.realized_quality(tier) for r in self.requirements if r.has(tier)]
            if costs:
                self._tier_cost_level[tier] = statistics.mean(costs)
            if ratings:
                self._tier_quality_level[tier] = statistics.mean(ratings)
        if levels_from is not None:
            # S5 §2: the level a multiplier is relative to is part of what B0
            # predicts, so on a held-out slice it must come from the dev slice
            # or B0 has been fitted on the test set. Only the levels are
            # borrowed; ``self.requirements`` stays the slice being scored.
            self._tier_cost_level = dict(levels_from._tier_cost_level)
            self._tier_quality_level = dict(levels_from._tier_quality_level)

    # -- levels ------------------------------------------------------------

    @property
    def tiers(self) -> Tuple[str, ...]:
        return tuple(t for t in EFFORT_TIERS if t in self._tier_cost_level)

    def tier_cost_level(self, tier: str) -> float:
        """The historical tier mean cost a multiplier of 1.0 refers to."""
        return self._tier_cost_level[tier]

    def tier_quality_level(self, tier: str) -> float:
        """The historical tier mean rating a delta of 0.0 refers to."""
        return self._tier_quality_level[tier]

    # -- convenience -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.requirements)

    def __iter__(self) -> Iterable[ProbeRequirement]:
        return iter(self.requirements)

    def by_probe_id(self, probe_id: str) -> ProbeRequirement:
        for requirement in self.requirements:
            if requirement.probe_id == probe_id:
                return requirement
        raise KeyError(probe_id)

    def history_profile(self) -> Dict[str, Any]:
        """The aggregate shape of the agent's completed jobs, for the prompt.

        What a deployed agent knows about its own workload, and the reference
        population the §1 multiplier is expressed against. Aggregate and
        identical for every requirement in the dataset, so it cannot help the
        model discriminate between them — it can only fix the level, which is
        exactly the failure it exists to correct. See
        :func:`prompts.render_user_message` for the measurement that shows why
        it is needed.
        """
        pdf_counts = [r.visible.source_pdf_count for r in self.requirements]
        prompt_counts = [len(r.visible.prompts) for r in self.requirements]
        breadth = [
            mechanical_features(r.visible)["mean_papers_per_prompt"]
            for r in self.requirements
        ]
        return {
            "n_jobs": len(self.requirements),
            "mean_pdfs": round(statistics.mean(pdf_counts)) if pdf_counts else 0,
            "mean_prompts": round(statistics.mean(prompt_counts)) if prompt_counts else 0,
            "mean_papers_per_prompt": (
                round(statistics.mean(breadth)) if breadth else 0
            ),
        }

    def cells(self) -> List[Tuple[ProbeRequirement, str]]:
        """Every (requirement, tier) cell with a recorded outcome."""
        return [
            (requirement, tier)
            for requirement in self.requirements
            for tier in EFFORT_TIERS
            if requirement.has(tier)
        ]


def load_probe_dataset(
    dataset_dir: Path | str = DATASET_V1_DIR,
) -> ProbeDataset:
    """Load ``requirements.jsonl`` + ``runs.jsonl`` into probe form.

    Requirements with no recorded run are dropped rather than carried with
    empty outcomes: a cell that cannot be scored should not appear in a
    denominator anywhere downstream.
    """
    directory = Path(dataset_dir)
    requirements_path = directory / "requirements.jsonl"
    runs_path = directory / "runs.jsonl"

    records = _read_jsonl(requirements_path)
    runs = _read_jsonl(runs_path)

    outcomes: Dict[str, Dict[str, RealizedOutcome]] = {}
    for row in runs:
        requirement_id = str(row.get("requirement_id") or "").strip()
        tier = str(row.get("tier") or "").strip().lower()
        if not requirement_id or tier not in EFFORT_TIERS:
            continue
        if row.get("execution_cost_usd") is None or row.get("evaluator_rating") is None:
            continue
        outcomes.setdefault(requirement_id, {})[tier] = RealizedOutcome(
            tier=tier,
            execution_cost_usd=float(row["execution_cost_usd"]),
            quality_rating=float(row["evaluator_rating"]),
        )

    probe_requirements: List[ProbeRequirement] = []
    for index, record in enumerate(records, start=1):
        requirement_id = str(record.get("requirement_id") or "").strip()
        cell_outcomes = outcomes.get(requirement_id) or {}
        if not cell_outcomes:
            continue
        probe_id = f"req_{index:02d}"
        probe_requirements.append(
            ProbeRequirement(
                probe_id=probe_id,
                requirement_id=requirement_id,
                visible=visible_view(record, probe_id),
                outcomes=cell_outcomes,
            )
        )

    return ProbeDataset(probe_requirements)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Leakage assertion (S2 §2)
# ---------------------------------------------------------------------------

def forbidden_values(record: Mapping[str, Any]) -> List[str]:
    """The literal strings that must not appear in a rendered prompt.

    Only values long enough to be identifying are returned. ``repeat_idx = 1``
    is a forbidden *field*, but the character "1" appears in any prompt that
    mentions a number, so asserting on it would fail on every input and teach
    nobody anything. The index fields are instead covered by
    :func:`assert_no_leakage`'s pattern check on ``repeat_NNN``-style tokens.
    """
    values: List[str] = []
    for name in FORBIDDEN_FIELDS:
        value = record.get(name)
        if isinstance(value, str) and len(value.strip()) >= 8:
            values.append(value.strip())
    return values


#: Index-bearing tokens that must never appear: the repeat suffix and the
#: zero-padded logical index the collection ids use.
_IDENTIFIER_PATTERNS = (
    re.compile(r"repeat_\d{2,}", re.IGNORECASE),
    re.compile(r"collection_\d{4,}", re.IGNORECASE),
    re.compile(r"executor_tier_dataset", re.IGNORECASE),
)


def assert_no_leakage(rendered: str, record: Mapping[str, Any]) -> None:
    """Raise :class:`LeakageError` if the rendered prompt exposes identity.

    This is called on every prompt actually sent, not only in tests. A probe
    whose leakage guard runs solely under ``unittest`` guards the test suite,
    not the experiment.
    """
    for value in forbidden_values(record):
        if value in rendered:
            raise LeakageError(f"forbidden value present in prompt: {value!r}")
    for pattern in _IDENTIFIER_PATTERNS:
        match = pattern.search(rendered)
        if match:
            raise LeakageError(
                f"identifier-shaped token present in prompt: {match.group()!r}"
            )


# ---------------------------------------------------------------------------
# Mechanical features — what B1 and B1b are allowed to count (S2 §3)
# ---------------------------------------------------------------------------

#: B1: countable surface features. No reading required.
SURFACE_FEATURES: Tuple[str, ...] = (
    "total_prompt_chars",
    "source_pdf_count",
    "prompt_count",
)

#: B1b: B1 plus mechanically-counted cross-document breadth. Still no reading —
#: a regex over the PDF stems is the whole of it.
BREADTH_FEATURES: Tuple[str, ...] = SURFACE_FEATURES + (
    "mean_papers_per_prompt",
    "wide_prompt_count",
)

#: A prompt is "wide" when it names at least this many distinct source papers.
#: Three is the threshold S2 §1 quotes its 4 → 10 range against.
WIDE_PROMPT_THRESHOLD = 3


def _stem_pattern(stem: str) -> re.Pattern:
    # Word-ish boundaries that treat "_" as part of the token, so
    # "Rozman_Emergence" does not match inside "Rozman_EmergenceExtra".
    return re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(stem) + r"(?![A-Za-z0-9_])"
    )


def count_paper_mentions(prompt: str, pdf_names: Sequence[str]) -> int:
    """Distinct source papers named in one prompt, counted by exact stem match.

    Exact-stem matching only. The obvious extension — also accepting the bare
    author prefix, so "(Lupi)" counts as a mention of
    ``Lupi_BlockchainBasedShared`` — was tried and rejected: "Rozman" prefixes
    four of the six papers, so the rule would have to be "accept the author
    prefix only when it is unique in this requirement's set", which makes the
    feature's meaning depend on the composition of the PDF set rather than on
    the prompt. Exact matching is the rule that means the same thing in every
    requirement, and it is the one that reproduces S2 §1's stated range
    (mean papers/prompt 2.70 → 5.00, wide prompts 4 → 10) exactly.
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


def redacted_payload(visible: VisibleRequirement) -> Dict[str, Any]:
    """The B2 ablation input: structure kept, prompt bodies removed (S2 §3).

    Each prompt becomes its character count and its paper-mention count — the
    two things B1b is allowed to know — so B2 isolates *content* from countable
    structure. If B2 scores like the full-text arm, the full-text arm was never
    reading the prompts.
    """
    payload = visible.as_payload()
    payload["prompts"] = [
        {
            "prompt_chars": len(prompt),
            "papers_mentioned": count_paper_mentions(
                prompt, visible.source_pdf_names
            ),
        }
        for prompt in visible.prompts
    ]
    payload["prompts_redacted"] = True
    return payload


def feature_variance_report(dataset: ProbeDataset) -> Dict[str, Dict[str, float]]:
    """Per-feature spread across requirements, and whether it is degenerate.

    Exists because on v1 two of B1's three features are constant. A regression
    on a constant column is not wrong so much as *empty*, and the report must
    say so rather than quoting a coefficient nobody can interpret.
    """
    report: Dict[str, Dict[str, float]] = {}
    for name in BREADTH_FEATURES:
        values = [
            mechanical_features(r.visible)[name] for r in dataset.requirements
        ]
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        report[name] = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "sd": spread,
            "degenerate": float(spread == 0.0),
        }
    return report


def log_cost_deviations(dataset: ProbeDataset) -> Dict[str, List[float]]:
    """``log(realized / tier level)`` per tier — the target B1/B1b predict.

    Working in log-ratio space is what makes the target comparable across tiers
    and puts the answer directly in ``noisy_oracle``'s ``cost_sigma`` units.
    """
    out: Dict[str, List[float]] = {}
    for tier in dataset.tiers:
        level = dataset.tier_cost_level(tier)
        out[tier] = [
            math.log(r.realized_cost(tier) / level)
            for r in dataset.requirements
            if r.has(tier)
        ]
    return out
