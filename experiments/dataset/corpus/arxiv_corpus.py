"""Fetch an arXiv paper corpus into the ``utils/files/`` layout the pipeline reads.

Two subcommands:

``search``
    Candidate discovery. Runs arXiv queries and dumps id/title/date/categories
    so a corpus can be *curated* rather than sampled. S3 §0 asks for topical
    clusters and deliberate conflicting pairs; a search query returns neither
    reliably, so the workflow is search → inspect → pin ids into a spec file.

``build``
    Reads the pinned spec, fetches metadata by id, downloads each PDF, writes
    the abstract to a sibling ``.txt``, and emits a manifest. The ``.pdf`` +
    ``.txt`` pairing is the convention ``ServiceGenerator._read_abstract`` and
    ``generate_dataset._discover_pdf_paths`` both depend on.

Every PDF is validated with pypdf before it is kept: the executor ingests via
``PyPDFDirectoryLoader``, so a scanned or image-only PDF would silently
contribute nothing to the RAG index while still counting as a document.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPEC = Path(__file__).resolve().parent / "corpus_v2.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "utils" / "files" / "v2"

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "agentic-marketplace-corpus/1.0 (research; nejc.rozman@fs.uni-lj.si)"

#: arXiv asks for at least 3 s between API calls. PDF downloads hit a different
#: host and tolerate a shorter gap, but stay polite: this runs once.
API_DELAY_S = 3.0
PDF_DELAY_S = 1.0

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

#: A PDF yielding less than this many extracted characters is almost certainly
#: image-only; the RAG index would see an empty document.
MIN_EXTRACTED_CHARS = 4000

_STOPWORDS = frozenset(
    """a an the of on in for to and or with without via using by from at as is are
    be does do how what why when where towards toward into over under between
    across can could should would we our their its it this that these those
    new novel some more most less than not no""".split()
)

#: Words that are technically content but carry almost no discriminating power
#: in this corpus — "Large Language Models" appears in a third of the titles.
#: Kept, but only after every stronger word, so the distinguishing term
#: ("Mirage", "Sharding") survives the truncation.
_WEAK_WORDS = frozenset(
    """large language model models llm llms learning deep neural network networks
    system systems study survey approach approaches analysis method methods
    framework based""".split()
)


# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------

@dataclass
class ArxivPaper:
    """One arXiv entry, flattened to the fields the manifest records."""

    arxiv_id: str          # version-pinned, e.g. "2304.15004v2"
    title: str
    authors: List[str]
    abstract: str
    published: str         # first submission (ISO 8601)
    updated: str           # this version (ISO 8601)
    primary_category: str
    categories: List[str]
    pdf_url: str
    doi: str = ""
    journal_ref: str = ""

    @property
    def year(self) -> int:
        return int(self.published[:4])


def _http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _clean(text: Optional[str]) -> str:
    """Collapse arXiv's hard-wrapped fields to a single normalised line."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_feed(payload: bytes) -> List[ArxivPaper]:
    root = ET.fromstring(payload)
    papers: List[ArxivPaper] = []

    for entry in root.findall("atom:entry", NS):
        raw_id = _clean(entry.findtext("atom:id", "", NS))
        # "http://arxiv.org/abs/2304.15004v2" -> "2304.15004v2"
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        if not arxiv_id or arxiv_id == raw_id:
            continue  # an error entry, not a paper

        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        primary = entry.find("arxiv:primary_category", NS)

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=_clean(entry.findtext("atom:title", "", NS)),
                authors=[
                    _clean(name.text)
                    for name in entry.findall("atom:author/atom:name", NS)
                ],
                abstract=_clean(entry.findtext("atom:summary", "", NS)),
                published=_clean(entry.findtext("atom:published", "", NS)),
                updated=_clean(entry.findtext("atom:updated", "", NS)),
                primary_category=primary.get("term", "") if primary is not None else "",
                categories=[
                    c.get("term", "") for c in entry.findall("atom:category", NS)
                ],
                pdf_url=pdf_url,
                doi=_clean(entry.findtext("arxiv:doi", "", NS)),
                journal_ref=_clean(entry.findtext("arxiv:journal_ref", "", NS)),
            )
        )

    return papers


def search_arxiv(query: str, max_results: int = 25, sort: str = "relevance") -> List[ArxivPaper]:
    """Run one arXiv search query."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": {"relevance": "relevance", "recent": "submittedDate"}[sort],
            "sortOrder": "descending",
        }
    )
    return _parse_feed(_http_get(f"{ARXIV_API}?{params}"))


def fetch_by_ids(arxiv_ids: Sequence[str], batch_size: int = 40) -> Dict[str, ArxivPaper]:
    """Resolve pinned ids to metadata, keyed by the id as requested.

    arXiv returns the exact version when the id carries one and the latest
    otherwise, so the returned key is the *requested* id while
    ``paper.arxiv_id`` is what arXiv actually served.
    """
    resolved: Dict[str, ArxivPaper] = {}

    for start in range(0, len(arxiv_ids), batch_size):
        batch = list(arxiv_ids[start : start + batch_size])
        params = urllib.parse.urlencode(
            {"id_list": ",".join(batch), "max_results": len(batch)}
        )
        papers = _parse_feed(_http_get(f"{ARXIV_API}?{params}"))

        by_bare = {p.arxiv_id.split("v")[0]: p for p in papers}
        for requested in batch:
            match = by_bare.get(requested.split("v")[0])
            if match is None:
                logger.error("arXiv returned no entry for %s", requested)
                continue
            resolved[requested] = match

        if start + batch_size < len(arxiv_ids):
            time.sleep(API_DELAY_S)

    return resolved


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def _surname(author: str) -> str:
    """Last whitespace-separated token, letters only. 'Rylan Schaeffer' -> 'Schaeffer'."""
    tokens = [t for t in re.split(r"\s+", author.strip()) if t]
    if not tokens:
        return "Unknown"
    surname = re.sub(r"[^A-Za-z]", "", tokens[-1])
    return surname.capitalize() if surname else "Unknown"


def _title_camel(title: str, max_words: int = 3) -> str:
    """'Are Emergent Abilities of LLMs a Mirage?' -> 'EmergentAbilitiesMirage'.

    Strong words keep their title order; weak words fill any remaining slots.
    Truncating a title to three words throws away most of it, so what survives
    has to be the discriminating part.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", title)
    content = [w for w in words if w.lower() not in _STOPWORDS] or words
    strong = [w for w in content if w.lower() not in _WEAK_WORDS]
    weak = [w for w in content if w.lower() in _WEAK_WORDS]
    chosen = (strong + weak)[:max_words]
    # Restore title order so the stem still reads as English.
    chosen.sort(key=content.index)
    return "".join(w.capitalize() for w in chosen) or "Untitled"


def make_stem(paper: ArxivPaper, taken: Iterable[str], override: str = "") -> str:
    """Build the ``Surname_TitleCamel`` stem, matching the v1 convention.

    The stem is not cosmetic: the executor exposes it to the RAG as
    ``source_id`` (``service_executor._extract_paper_metadata``) and the
    estimation probe sees it as ``source_pdf_names``
    (``experiments/estimation/dataset._pdf_stem``). It is task information the
    provider would really have, so it has to stay topically informative and
    unique — which is why the spec may override the heuristic per paper.
    """
    taken = set(taken)
    if override:
        if override in taken:
            raise ValueError(f"spec assigns the stem {override!r} to more than one paper")
        return override

    base = f"{_surname(paper.authors[0] if paper.authors else '')}_{_title_camel(paper.title)}"

    if base not in taken:
        return base

    # Widen the title before falling back to a numeric suffix — a longer title
    # stays informative where "_2" does not.
    wider = f"{_surname(paper.authors[0] if paper.authors else '')}_{_title_camel(paper.title, max_words=5)}"
    if wider not in taken:
        return wider

    for n in range(2, 100):
        candidate = f"{base}{n}"
        if candidate not in taken:
            return candidate
    raise RuntimeError(f"cannot build a unique stem for {paper.arxiv_id}")


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

@dataclass
class PdfStats:
    pages: int
    chars: int
    bytes: int


def inspect_pdf(path: Path) -> PdfStats:
    """Page/character counts via pypdf — the same extractor the executor uses."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chars = sum(len(page.extract_text() or "") for page in reader.pages)
    return PdfStats(pages=len(reader.pages), chars=chars, bytes=path.stat().st_size)


def download_pdf(paper: ArxivPaper, dest: Path) -> None:
    dest.write_bytes(_http_get(paper.pdf_url, timeout=180))
    if dest.read_bytes()[:5] != b"%PDF-":
        dest.unlink(missing_ok=True)
        raise ValueError(f"{paper.arxiv_id}: downloaded file is not a PDF")


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass
class SpecEntry:
    arxiv_id: str
    area: str
    cluster: str
    conflicts_with: List[str] = field(default_factory=list)
    note: str = ""
    stem: str = ""  # optional override; see make_stem


def load_spec(path: Path) -> List[SpecEntry]:
    """Flatten the nested ``areas → clusters → papers`` spec into entries."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries: List[SpecEntry] = []

    for area in spec.get("areas", []):
        area_name = area["name"]
        for cluster in area.get("clusters", []):
            cluster_name = cluster["name"]
            for paper in cluster.get("papers", []):
                entries.append(
                    SpecEntry(
                        arxiv_id=str(paper["id"]),
                        area=area_name,
                        cluster=cluster_name,
                        conflicts_with=[str(c) for c in paper.get("conflicts_with", [])],
                        note=str(paper.get("note", "")),
                        stem=str(paper.get("stem", "")),
                    )
                )

    seen = [e.arxiv_id for e in entries]
    duplicates = {i for i in seen if seen.count(i) > 1}
    if duplicates:
        raise ValueError(f"spec lists these ids more than once: {sorted(duplicates)}")

    return entries


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_corpus(
    spec_path: Path,
    out_dir: Path,
    manifest_path: Path,
    *,
    force: bool = False,
) -> List[Dict[str, Any]]:
    entries = load_spec(spec_path)
    logger.info("spec lists %d papers", len(entries))

    papers = fetch_by_ids([e.arxiv_id for e in entries])
    missing = [e.arxiv_id for e in entries if e.arxiv_id not in papers]
    if missing:
        raise RuntimeError(f"arXiv did not resolve: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Stems are assigned in spec order so the mapping is stable across reruns.
    stems: Dict[str, str] = {}
    for entry in entries:
        stems[entry.arxiv_id] = make_stem(
            papers[entry.arxiv_id], stems.values(), override=entry.stem
        )

    records: List[Dict[str, Any]] = []
    rejected: List[str] = []

    for index, entry in enumerate(entries, start=1):
        paper = papers[entry.arxiv_id]
        stem = stems[entry.arxiv_id]
        pdf_path = out_dir / f"{stem}.pdf"
        txt_path = out_dir / f"{stem}.txt"

        if pdf_path.exists() and not force:
            logger.info("[%2d/%d] %s (cached)", index, len(entries), stem)
        else:
            logger.info("[%2d/%d] %s <- %s", index, len(entries), stem, paper.pdf_url)
            download_pdf(paper, pdf_path)
            time.sleep(PDF_DELAY_S)

        stats = inspect_pdf(pdf_path)
        if stats.chars < MIN_EXTRACTED_CHARS:
            logger.error(
                "%s: only %d extractable chars over %d pages — dropping "
                "(the RAG index would see an empty document)",
                stem, stats.chars, stats.pages,
            )
            pdf_path.unlink(missing_ok=True)
            rejected.append(entry.arxiv_id)
            continue

        # The .txt is the *abstract*, not the extracted body: that is what
        # ServiceGenerator._read_abstract expects, and what v1's files hold.
        txt_path.write_text(paper.abstract + "\n", encoding="utf-8")

        records.append(
            {
                "stem": stem,
                "arxiv_id": paper.arxiv_id,
                "requested_id": entry.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "updated": paper.updated,
                "year": paper.year,
                "area": entry.area,
                "cluster": entry.cluster,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref,
                "abstract_chars": len(paper.abstract),
                "pdf_pages": stats.pages,
                "pdf_bytes": stats.bytes,
                "extracted_chars": stats.chars,
                "conflicts_with": entry.conflicts_with,
                "note": entry.note,
                "pdf_url": paper.pdf_url,
            }
        )

    # Conflict links are written as arXiv ids in the spec; resolve them to
    # stems too, since stems are what every downstream consumer works in.
    by_id = {r["requested_id"]: r["stem"] for r in records}
    for record in records:
        record["conflicts_with_stems"] = [
            by_id[i] for i in record["conflicts_with"] if i in by_id
        ]
        dangling = [i for i in record["conflicts_with"] if i not in by_id]
        if dangling:
            logger.warning("%s: conflict target not in corpus: %s", record["stem"], dangling)

    manifest = {
        "corpus_id": "v2",
        "source": "arXiv",
        "spec": str(spec_path.relative_to(REPO_ROOT)),
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
        "n_papers": len(records),
        "n_rejected": len(rejected),
        "rejected_ids": rejected,
        "papers": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info("wrote %d papers to %s", len(records), out_dir)
    logger.info("manifest: %s", manifest_path)
    if rejected:
        logger.warning("%d papers rejected for unextractable text: %s", len(rejected), rejected)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_search(args: argparse.Namespace) -> int:
    queries: List[str] = list(args.query or [])
    if args.queries_file:
        loaded = yaml.safe_load(Path(args.queries_file).read_text(encoding="utf-8"))
        queries.extend(loaded["queries"] if isinstance(loaded, dict) else loaded)

    if not queries:
        print("no queries given", file=sys.stderr)
        return 2

    dump: Dict[str, List[Dict[str, Any]]] = {}
    for position, query in enumerate(queries):
        if position:
            time.sleep(API_DELAY_S)
        papers = search_arxiv(query, max_results=args.max_results, sort=args.sort)
        dump[query] = [asdict(p) for p in papers]

        print(f"\n### {query}  ({len(papers)} hits)")
        for paper in papers:
            print(f"  {paper.arxiv_id:<16} {paper.published[:7]}  {paper.primary_category:<10} {paper.title[:96]}")

    if args.out:
        Path(args.out).write_text(json.dumps(dump, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote candidates to {args.out}")

    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    build_corpus(
        spec_path=Path(args.spec),
        out_dir=Path(args.out_dir),
        manifest_path=Path(args.manifest),
        force=args.force,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="run arXiv queries and dump candidates for curation")
    search.add_argument("-q", "--query", action="append", help="arXiv search_query (repeatable)")
    search.add_argument("--queries-file", help="YAML file with a 'queries' list")
    search.add_argument("--max-results", type=int, default=25)
    search.add_argument("--sort", choices=["relevance", "recent"], default="relevance")
    search.add_argument("--out", help="write candidate metadata to this JSON file")
    search.set_defaults(func=_cmd_search)

    build = sub.add_parser("build", help="download the pinned corpus into utils/files/v2")
    build.add_argument("--spec", default=str(DEFAULT_SPEC))
    build.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    build.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "manifest_v2.json"))
    build.add_argument("--force", action="store_true", help="re-download PDFs that already exist")
    build.set_defaults(func=_cmd_build)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
