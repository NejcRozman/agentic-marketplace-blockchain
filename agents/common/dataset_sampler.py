"""Replay of the recorded service dataset: requests, and the recorded runs of each."""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse


class DatasetSampler:
    """Load and replay requests and their recorded runs from the dataset."""

    SERVICE_CID_SCHEME = "sampled-executor"

    #: ``shuffled_pass`` is the mode the experiments use for requests: permute
    #: the dataset under the run seed, then consume it in that order exactly once.
    #:
    #: ``random`` draws *with replacement*, so a request can recur, and a
    #: recurring request is already in the agent's own cost history.
    #: ``round_robin`` makes one pass but always the same one, so the job stream
    #: would be identical across seeds.
    SELECTION_MODES = {"round_robin", "random", "shuffled_pass"}

    def __init__(
        self,
        dataset_root: str | Path,
        dataset_id: Optional[str] = None,
        *,
        requirement_selection_mode: str = "round_robin",
        run_selection_mode: str = "round_robin",
        random_seed: Optional[int] = None,
        requirement_pass_limit: Optional[int] = None,
    ):
        root_path = Path(dataset_root)
        dataset_dir = self._resolve_dataset_dir(root_path, dataset_id)

        self.dataset_dir = dataset_dir
        self.dataset_id = dataset_id or dataset_dir.name
        self.requirement_selection_mode = self._normalize_selection_mode(requirement_selection_mode)
        self.run_selection_mode = self._normalize_selection_mode(run_selection_mode)
        self._rng = random.Random(random_seed)

        self.requirements_jsonl_path = self.dataset_dir / "requirements.jsonl"
        self.runs_jsonl_path = self.dataset_dir / "runs.jsonl"

        self._requirement_records = self._load_jsonl(self.requirements_jsonl_path)
        self._run_records = self._load_jsonl(self.runs_jsonl_path)

        self._requirements_by_id: Dict[str, Dict[str, Any]] = {}
        self._requirement_order: List[str] = []
        self._runs_by_requirement_and_tier: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self._requirement_cursor = 0
        self._run_cursors: Dict[Tuple[str, str], int] = defaultdict(int)

        self._index_records()

        # The shuffled pass is drawn once, at construction, so the whole order is
        # a deterministic function of the seed rather than of how many times the
        # agent happened to call in.
        self._shuffled_requirement_order: List[str] = list(self._requirement_order)
        if self.requirement_selection_mode == "shuffled_pass":
            self._rng.shuffle(self._shuffled_requirement_order)

        # The pass limit is the run horizon. It makes the horizon explicit and
        # keeps exhaustion a hard error.
        self.requirement_pass_limit = (
            None if requirement_pass_limit is None else int(requirement_pass_limit)
        )
        if self.requirement_pass_limit is not None:
            if self.requirement_pass_limit > len(self._requirement_order):
                raise ValueError(
                    f"requirement_pass_limit={self.requirement_pass_limit} exceeds the "
                    f"{len(self._requirement_order)} requirements in "
                    f"'{self.dataset_id}'; a run cannot draw more without replacement"
                )
            self._shuffled_requirement_order = self._shuffled_requirement_order[
                : self.requirement_pass_limit
            ]

    @classmethod
    def build_dataset_service_cid(cls, dataset_id: str, requirement_id: str) -> str:
        """Encode dataset identifiers into a synthetic service CID string."""
        encoded_dataset = quote(str(dataset_id), safe="")
        encoded_requirement = quote(str(requirement_id), safe="")
        return f"{cls.SERVICE_CID_SCHEME}://{encoded_dataset}/{encoded_requirement}"

    @classmethod
    def parse_dataset_service_cid(cls, value: str) -> Optional[Dict[str, str]]:
        """Parse synthetic dataset service CID metadata."""
        if not value or not isinstance(value, str):
            return None

        parsed = urlparse(value)
        if parsed.scheme != cls.SERVICE_CID_SCHEME:
            return None

        requirement_path = parsed.path.lstrip("/")
        if not parsed.netloc or not requirement_path:
            return None

        return {
            "dataset_id": unquote(parsed.netloc),
            "requirement_id": unquote(requirement_path),
        }

    def list_requirement_ids(self) -> List[str]:
        """Return requirement ids in deterministic dataset order."""
        return list(self._requirement_order)

    def requirement_pass_order(self) -> List[str]:
        """Return the order this run consumes requirements in.

        Under ``shuffled_pass`` that is the seeded permutation; otherwise it is
        dataset order. Callers that hold their own service list — the consumer
        does — order it by this so the pass they walk is the pass the sampler
        drew.
        """
        if self.requirement_selection_mode == "shuffled_pass":
            return list(self._shuffled_requirement_order)
        return list(self._requirement_order)

    def next_requirement_id(self) -> str:
        """Return the next requirement id in the run's pass, without replacement.

        Raises once the pass is exhausted. That is the point of the mode: a 31st
        auction against a 30-requirement dataset is a misconfiguration, and the
        run must stop rather than quietly serve a requirement the agent has
        already executed and priced.
        """
        order = self.requirement_pass_order()
        if not order:
            raise ValueError(f"No requirement records found in {self.requirements_jsonl_path}")

        if self.requirement_selection_mode != "shuffled_pass":
            requirement_id = order[self._requirement_cursor % len(order)]
            self._requirement_cursor += 1
            return requirement_id

        if self._requirement_cursor >= len(order):
            raise RuntimeError(
                f"shuffled_pass exhausted: dataset '{self.dataset_id}' holds "
                f"{len(order)} requirements and all have been consumed. A run may "
                f"not reuse a requirement — it would already be in the agent's own "
                f"cost history."
            )

        requirement_id = order[self._requirement_cursor]
        self._requirement_cursor += 1
        return requirement_id

    def get_requirement(self, requirement_id: str) -> Dict[str, Any]:
        """Return a requirement record by id."""
        if requirement_id not in self._requirements_by_id:
            raise KeyError(f"Unknown requirement_id: {requirement_id}")
        return copy.deepcopy(self._requirements_by_id[requirement_id])

    def sample_run(
        self,
        requirement_id: str,
        effort_tier: str,
        selection_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select a run for a given requirement and effort tier.

        ``shuffled_pass`` is a *requirement*-level mode and falls through to
        round-robin here: choosing among the recordings of one request and tier
        is a different question, and round-robin answers it deterministically.
        """
        resolved_tier = self._normalize_effort_tier(effort_tier)
        key = (str(requirement_id), resolved_tier)
        candidates = self._runs_by_requirement_and_tier.get(key, [])

        if not candidates:
            raise KeyError(
                f"No dataset runs found for requirement_id={requirement_id!r}, effort_tier={resolved_tier!r}"
            )

        resolved_mode = self._normalize_selection_mode(selection_mode or self.run_selection_mode)
        if resolved_mode == "random":
            row = self._rng.choice(candidates)
        else:
            cursor = self._run_cursors[key]
            row = candidates[cursor % len(candidates)]
            self._run_cursors[key] = cursor + 1

        return copy.deepcopy(row)

    @staticmethod
    def _resolve_dataset_dir(dataset_root: Path, dataset_id: Optional[str]) -> Path:
        if (dataset_root / "requirements.jsonl").exists() or (dataset_root / "runs.jsonl").exists():
            return dataset_root
        if dataset_id:
            return dataset_root / str(dataset_id)
        return dataset_root

    @classmethod
    def _normalize_selection_mode(cls, value: Optional[str]) -> str:
        normalized = str(value or "round_robin").strip().lower()
        if normalized not in cls.SELECTION_MODES:
            raise ValueError(
                f"Unsupported selection mode: {value!r}. Expected one of {sorted(cls.SELECTION_MODES)}"
            )
        return normalized

    @staticmethod
    def _normalize_effort_tier(value: Optional[str]) -> str:
        normalized = str(value or "medium").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError(f"Unsupported effort tier: {value!r}")
        return normalized

    def _index_records(self) -> None:
        for row in self._requirement_records:
            requirement_id = str(row.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            if requirement_id in self._requirements_by_id:
                continue
            self._requirements_by_id[requirement_id] = row
            self._requirement_order.append(requirement_id)

        for row in self._run_records:
            requirement_id = str(row.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            try:
                tier = self._normalize_effort_tier(row.get("tier"))
            except ValueError:
                continue
            self._runs_by_requirement_and_tier[(requirement_id, tier)].append(row)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        return rows