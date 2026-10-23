"""Dataset v2 as the S5 probe sees it: the split, the pairs, and the levels.

S2's :mod:`dataset` module supplies everything about *what the model may see* —
the allow-list, the mechanical features, the leakage guard — and this module
reuses it unchanged. That reuse is not tidiness: if the visible view or the
countable features differed between S2 and S5, the frozen protocol would be
running against a different input than the one it was frozen for, and the
comparison against B1b would be rigged by construction.

What is new here is everything v2 has and v1 did not.

**The split.** 20 dev requirements, 10 held out. :class:`SplitDataset` is the
only supported way to load v2, and it hands back the two slices separately, so
"fit on dev, evaluate on held-out" is the path of least resistance and pooling
them takes a deliberate act.

**Repeats.** v1 recorded one run per (requirement, tier); v2 records three. A
cell's realized outcome is the mean of its repeats, and the within-cell spread
is carried alongside on :class:`~agents.provider.estimation.dataset.RealizedOutcome`
rather than discarded — S5 §3 needs it to say which differences are resolvable.

**The levels.** The §1 multiplier is relative to a per-tier mean, and B0 *is*
that mean. Computing it over all 30 requirements would fit the floor on the test
set, so held-out slices borrow their levels from dev via ``levels_from``. Same
for :meth:`ProbeDataset.history_profile`, which is rendered into the prompt: it
describes the agent's completed jobs, and the held-out requirements are by
construction ones it has not run.

**The pairs.** Ten matched pairs, each a shallow and a deep prompt set over an
identical document set. :func:`matched_pairs` is what S5 §4's contrast runs on.

> **A measured caveat on §4.** The spec describes pair members as carrying
> "identical values on every countable feature". That is true of
> ``source_pdf_count`` and ``prompt_count`` (10/10 pairs) and nearly true of
> ``wide_prompt_count`` (9/10) and ``mean_papers_per_prompt`` (7/10), but it is
> **false for ``total_prompt_chars``, which differs in 10 of 10 pairs** — and
> the deeper member is the longer one in 8 of them. B1b can therefore separate
> pair members after all, by length alone. The consequence is not that the
> contrast is invalid; it is that B1b must be *run through the same contrast*
> and reported next to the LLM, rather than assumed to score at chance.
> :func:`pair_feature_gaps` produces that table.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.provider.estimation import dataset as ds

DATASET_V2_DIR = (
    ds.REPO_ROOT
    / "experiments"
    / "executor_analysis"
    / "data"
    / "datasets"
    / "executor_tier_dataset_v2"
    / "final"
)

#: ``dev`` may be fitted on; ``heldout`` may only be scored.
SPLITS: Tuple[str, str] = ("dev", "heldout")

#: v2 fields whose values are **identifiers**, so a substring search for them is
#: meaningful. ``slot_id`` (``pair_01a``) and ``shares_documents_with`` would let
#: a model recognise that two requirements share a document set.
FORBIDDEN_V2_ID_FIELDS: Tuple[str, ...] = (
    "slot_id",
    "pair_id",
    "shares_documents_with",
    "reused_requirements_from",
    "requirement_source",
)

#: v2 fields that are the **answer sheet** — ``demanded_synthesis_depth`` is
#: literally the latent variable S5 §4 asks the model to infer, and ``split``
#: says which requirements are being graded.
#:
#: These are deliberately **not** substring-checked. Their values are ordinary
#: English words (``locate``, ``contrast``, ``localised``) that a good-faith
#: prompt may legitimately contain — "contrast the two approaches" is a normal
#: instruction, and a substring rule would reject valid input while proving
#: nothing. What keeps them out is :func:`dataset.visible_view`, which is an
#: allow-list: a field must be named there to be rendered at all. That is the
#: stronger guarantee, and ``test_estimation_probe_v2`` asserts it directly by
#: pinning the exact key set of every rendered payload.
FORBIDDEN_V2_DESIGN_FIELDS: Tuple[str, ...] = (
    "split",
    "demanded_synthesis_depth",
    "demanded_synthesis_rank",
    "demanded_retrieval_rounds",
    "evidence_dispersion",
    "requires_reconciliation",
    "designed_countable_features",
    "realized_countable_features",
    "designed_prompt_breadths",
    "realized_prompt_breadths",
    "n_conflict_pairs",
    "source_areas",
)

#: The exact keys a full-arm payload may carry. Pinned so that admitting a new
#: field to the allow-list is a deliberate edit in two places, not a side effect
#: of v2 records having more columns than v1's.
ALLOWED_PAYLOAD_KEYS: Tuple[str, ...] = (
    "description",
    "prompts",
    "service_type",
    "source_pdf_count",
    "source_pdf_names",
    "title",
)

#: Probe ids are namespaced ``v2_req_NN``. Not cosmetic: the LLM cache is keyed
#: on ``(protocol fingerprint, model, arm, probe_id, repeat)``, so reusing v1's
#: ``req_01`` would serve a v1 answer for a v2 requirement wherever the two runs
#: share a model — silently, and with no way to notice from the results CSV.
PROBE_ID_PREFIX = "v2_req"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchedPair:
    """One S5 §4 pair: the same documents, two demanded synthesis depths.

    ``shallow`` and ``deep`` are assigned from ``demanded_synthesis_rank``, not
    from the ``a``/``b`` slot suffix, so the ordering comes from the design
    variable rather than from a naming convention that could drift.
    """

    pair_id: str
    shallow: ds.ProbeRequirement
    deep: ds.ProbeRequirement

    @property
    def split(self) -> str:
        return self.shallow.split

    def realized_cost_ratio(self, tier: str) -> Optional[float]:
        if not (self.shallow.has(tier) and self.deep.has(tier)):
            return None
        shallow_cost = self.shallow.realized_cost(tier)
        return self.deep.realized_cost(tier) / shallow_cost if shallow_cost > 0 else None


class SplitDataset:
    """v2's dev and held-out slices, plus the pairs that span both.

    ``heldout`` is constructed with ``levels_from=dev``: every tier level it
    reports is the dev mean. ``pooled`` exists only for S5 §4, which is
    explicitly reported over all ten pairs with the held-out ones marked.
    """

    def __init__(
        self,
        requirements: Sequence[ds.ProbeRequirement],
        raw_records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.raw_records = dict(raw_records)
        self.dev = ds.ProbeDataset([r for r in requirements if r.split == "dev"])
        self.heldout = ds.ProbeDataset(
            [r for r in requirements if r.split == "heldout"], levels_from=self.dev
        )
        self.pooled = ds.ProbeDataset(list(requirements), levels_from=self.dev)

    def slice_for(self, split: str) -> ds.ProbeDataset:
        if split == "dev":
            return self.dev
        if split == "heldout":
            return self.heldout
        if split == "pooled":
            return self.pooled
        raise KeyError(f"unknown split {split!r}; expected one of dev, heldout, pooled")

    def raw_record(self, requirement: ds.ProbeRequirement) -> Mapping[str, Any]:
        """The unprojected record, for :func:`dataset.assert_no_leakage`."""
        return self.raw_records.get(requirement.requirement_id, {})

    def matched_pairs(self) -> List[MatchedPair]:
        return matched_pairs(self.pooled)


def load_split_dataset(dataset_dir: Path | str = DATASET_V2_DIR) -> SplitDataset:
    """Load v2's finalised requirements and runs into probe form.

    Only ``final/`` is read. The parent directory still holds the pilots and the
    superseded chunks described in ``final/manifest.json``; scoring against
    those would mix tuning runs into the gate.
    """
    directory = Path(dataset_dir)
    records = ds._read_jsonl(directory / "requirements.jsonl")
    runs = ds._read_jsonl(directory / "runs.jsonl")

    outcomes = _aggregate_runs(runs)

    requirements: List[ds.ProbeRequirement] = []
    for index, record in enumerate(sorted(records, key=_record_sort_key), start=1):
        requirement_id = str(record.get("requirement_id") or "").strip()
        cell_outcomes = outcomes.get(requirement_id) or {}
        if not cell_outcomes:
            continue
        split = str(record.get("split") or "").strip().lower()
        if split not in SPLITS:
            raise ValueError(
                f"requirement {requirement_id!r} has split {split!r}; "
                f"every v2 requirement must be labelled one of {SPLITS}"
            )
        probe_id = f"{PROBE_ID_PREFIX}_{index:02d}"
        requirements.append(
            ds.ProbeRequirement(
                probe_id=probe_id,
                requirement_id=requirement_id,
                visible=ds.visible_view(record, probe_id),
                outcomes=cell_outcomes,
                split=split,
                pair_id=str(record.get("pair_id") or ""),
                slot_id=str(record.get("slot_id") or ""),
                depth=str(record.get("demanded_synthesis_depth") or ""),
                depth_rank=int(record.get("demanded_synthesis_rank") or 0),
            )
        )

    by_id = {str(r.get("requirement_id") or ""): r for r in records}
    return SplitDataset(requirements, by_id)


def _record_sort_key(record: Mapping[str, Any]) -> Tuple[int, str]:
    """Order requirements so probe ids are stable across reloads.

    Sorted by ``slot_id``, which is the design's own label (``pair_01a``,
    ``single_07``) and is independent of which collection a requirement was
    executed in — a requirement re-run in a later chunk keeps its position.
    """
    slot = str(record.get("slot_id") or "")
    return (0 if slot.startswith("pair") else 1, slot)


def _aggregate_runs(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, ds.RealizedOutcome]]:
    """Collapse repeats to one outcome per (requirement, tier) cell.

    The mean, not the median: the downstream σ is an RMSE, the simulator's cost
    model is multiplicative in the mean, and with three repeats a median simply
    discards a third of the measurement.
    """
    grouped: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in runs:
        requirement_id = str(row.get("requirement_id") or "").strip()
        tier = str(row.get("tier") or "").strip().lower()
        if not requirement_id or tier not in ds.EFFORT_TIERS:
            continue
        if row.get("execution_cost_usd") is None or row.get("evaluator_rating") is None:
            continue
        grouped.setdefault((requirement_id, tier), []).append(
            (float(row["execution_cost_usd"]), float(row["evaluator_rating"]))
        )

    outcomes: Dict[str, Dict[str, ds.RealizedOutcome]] = {}
    for (requirement_id, tier), values in grouped.items():
        costs = [c for c, _ in values]
        ratings = [q for _, q in values]
        outcomes.setdefault(requirement_id, {})[tier] = ds.RealizedOutcome(
            tier=tier,
            execution_cost_usd=statistics.mean(costs),
            quality_rating=statistics.mean(ratings),
            n_runs=len(values),
            cost_sd=statistics.stdev(costs) if len(costs) > 1 else 0.0,
            quality_sd=statistics.stdev(ratings) if len(ratings) > 1 else 0.0,
        )
    return outcomes


# ---------------------------------------------------------------------------
# Matched pairs (S5 §4)
# ---------------------------------------------------------------------------

def matched_pairs(data: ds.ProbeDataset) -> List[MatchedPair]:
    """The pairs present in ``data``, ordered by ``pair_id``.

    A pair with only one member present is dropped rather than half-reported:
    the contrast is a within-pair comparison and has no meaning with one side.
    """
    grouped: Dict[str, List[ds.ProbeRequirement]] = {}
    for requirement in data.requirements:
        if requirement.pair_id:
            grouped.setdefault(requirement.pair_id, []).append(requirement)

    pairs: List[MatchedPair] = []
    for pair_id, members in sorted(grouped.items()):
        if len(members) != 2:
            continue
        shallow, deep = sorted(members, key=lambda r: r.depth_rank)
        if shallow.depth_rank == deep.depth_rank:
            raise ValueError(
                f"{pair_id} has two members at synthesis rank {shallow.depth_rank}; "
                "the contrast needs a designed depth difference"
            )
        pairs.append(MatchedPair(pair_id=pair_id, shallow=shallow, deep=deep))
    return pairs


def pair_feature_gaps(data: ds.ProbeDataset) -> Dict[str, Dict[str, float]]:
    """Per countable feature: how often it separates a pair, and by how much.

    The evidence for the caveat in this module's docstring. ``identical_pairs``
    is the count of pairs where the feature takes the same value on both
    members — the property S5 §4 assumes holds for all five. ``deep_larger`` is
    how often the deeper member scores higher, which is what decides whether a
    counting model can win the contrast by accident.
    """
    report: Dict[str, Dict[str, float]] = {}
    pairs = matched_pairs(data)
    if not pairs:
        return report

    for name in ds.BREADTH_FEATURES:
        identical = 0
        deep_larger = 0
        gaps: List[float] = []
        for pair in pairs:
            shallow = ds.mechanical_features(pair.shallow.visible)[name]
            deep = ds.mechanical_features(pair.deep.visible)[name]
            gaps.append(deep - shallow)
            if deep == shallow:
                identical += 1
            elif deep > shallow:
                deep_larger += 1
        report[name] = {
            "n_pairs": float(len(pairs)),
            "identical_pairs": float(identical),
            "deep_larger": float(deep_larger),
            "mean_gap": statistics.mean(gaps),
            "separates": float(len(pairs) - identical),
        }
    return report


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def assert_no_v2_leakage(rendered: str, record: Mapping[str, Any]) -> None:
    """S2's guard plus the v2 design fields, run on every prompt actually sent.

    Two different guarantees, and it is worth being clear about which covers
    what. The **identifier** fields are substring-checked here, because their
    values (``pair_01a``, ``v2_dev_c1``) cannot occur by chance. The **design**
    fields are not checked here at all — see
    :data:`FORBIDDEN_V2_DESIGN_FIELDS` for why a substring rule on ``contrast``
    would reject valid prompts — and are instead kept out by the allow-list,
    which :func:`assert_payload_allowed` verifies directly.
    """
    ds.assert_no_leakage(rendered, record)
    for name in FORBIDDEN_V2_ID_FIELDS:
        value = record.get(name)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if len(value) >= 6 and value in rendered:
            raise ds.LeakageError(
                f"v2 identifier field {name!r} leaked into the prompt: {value!r}"
            )


def assert_payload_allowed(payload: Mapping[str, Any]) -> None:
    """Raise if a payload carries any key outside the allow-list.

    The real defence against v2's design metadata. Run on the payload rather
    than on the rendered string, so it catches a leaked field regardless of how
    its value happens to serialise.
    """
    extra = set(payload) - set(ALLOWED_PAYLOAD_KEYS) - {"prompts_redacted"}
    if extra:
        raise ds.LeakageError(
            f"payload carries non-allow-listed keys: {sorted(extra)}"
        )


def visible_payload_keys(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    """The keys a rendered payload carries, for the allow-list test."""
    return tuple(sorted(payload))
