"""Load extended paper and chain research text from bundled taxonomy data."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from graph_surgeon.taxonomy.chain_catalog import CHAIN_DETECTION_RATIONALE

_DATA_DIR = Path(__file__).parent / "data"
_TAXONOMY_JSON = _DATA_DIR / "papers_taxonomy.json"
_ATTACK_RESEARCH_NOTES = _DATA_DIR / "attack_research_notes.md"

# Slugs not in numeric taxonomy tables (registry-only identifiers)
SLUG_ALIASES: Dict[str, Dict[str, str]] = {
    "BNAttack-2020": {
        "title": "BatchNorm-targeted / distribution-shift attacks",
        "summary": "Literature on attacking BatchNorm running statistics and channel distributions at inference.",
    },
    "HiddenLayer-ShadowLogic-2024": {
        "title": "ShadowLogic (HiddenLayer, 2024)",
        "summary": "Graph-level backdoor injection via conditional ONNX subgraphs; trigger-activated malicious paths.",
        "url": "https://hiddenlayer.com/innovation-hub/shadowlogic/",
    },
    "HiddenLayer-PersistentBackdoors-2024": {
        "title": "Persistent ML backdoors (HiddenLayer, 2024)",
        "summary": "Backdoors that survive fine-tuning and format conversion when embedded in the graph.",
    },
    "arXiv-2511.00664": {
        "title": "ShadowLogic academic reference (arXiv:2511.00664)",
        "summary": "Peer-reviewed treatment of graph-embedded backdoor mechanisms.",
        "url": "https://arxiv.org/abs/2511.00664",
    },
    "ResNet-2015": {
        "title": "ResNet skip-connection architecture",
        "summary": "Residual networks provide gradient highways relevant to deep-network adversarial optimization.",
    },
    "SENet-2018": {
        "title": "Squeeze-and-Excitation Networks",
        "summary": "Spatial attention via channel recalibration; defensive when present before pooling.",
    },
    "CBAM-2018": {
        "title": "Convolutional Block Attention Module",
        "summary": "Combined channel and spatial attention; can filter anomalous regions.",
    },
}

ARXIV_BY_NUMERIC_ID: Dict[str, str] = {
    "36": "https://arxiv.org/abs/1712.09665",
    "39": "https://arxiv.org/abs/1801.02608",
    "38": "https://arxiv.org/abs/1707.07397",
    "60": "https://arxiv.org/abs/1707.08945",
    "84": "https://arxiv.org/abs/1904.08653",
    "111": "https://arxiv.org/abs/2506.18516",
}


# Research-doc scaffolding (project logs, phase plans) — not for catalog output.
_CATALOG_EXCLUDED_MARKERS: tuple[str, ...] = (
    "\n## Cross-Cutting",
    "\n## Proposed New Gadget",
    "\n## Session Log",
    "\n## Bibliography",
    "\n## Phase ",
    "\n## ShadowLogic",
    "\n## Object Detector Vulnerability Summary",
    "\n**Next Steps:**",
    "\n**Questions for Discussion:**",
    "\n### Papers for Phase",
    "\n### Papers Researched:",
    "\n### Supporting Literature:",
    "\n### Research Summary:",
    "\n### New Gadget Types to Implement",
    "\n### Updates to Existing Gadgets",
    "\n### New Chain Patterns",
)

_PAPER_SECTION_END = re.compile(r"(?=\n### \[|\n## |\Z)")


@lru_cache(maxsize=1)
def load_taxonomy_papers() -> Dict[str, Dict[str, Any]]:
    if _TAXONOMY_JSON.exists():
        return json.loads(_TAXONOMY_JSON.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def _attack_research_notes_text() -> Optional[str]:
    if _ATTACK_RESEARCH_NOTES.exists():
        return _ATTACK_RESEARCH_NOTES.read_text(encoding="utf-8")
    return None


def _sanitize_catalog_research_text(body: str) -> str:
    """Drop project meta, phase logs, and unrelated research blocks."""
    cut_at = len(body)
    for marker in _CATALOG_EXCLUDED_MARKERS:
        idx = body.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    body = body[:cut_at].strip()
    # Drop session-log style ### subsections (not ### [NN] paper headers).
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("### ") and not re.match(r"### \[\d+\]", line):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def extract_paper_section(numeric_id: str) -> Optional[str]:
    """Return ### [ID] paper analysis from bundled attack_research_notes.md."""
    text = _attack_research_notes_text()
    if not text:
        return None
    pattern = rf"### \[{re.escape(numeric_id)}\][^\n]*\n(.*?){_PAPER_SECTION_END.pattern}"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    header_m = re.search(rf"### \[{numeric_id}\][^\n]*", text)
    header = header_m.group(0) if header_m else f"### [{numeric_id}]"
    body = _sanitize_catalog_research_text(m.group(1))
    if not body:
        return None
    return f"{header}\n{body}"


def extract_chain_rationale(chain_id: str) -> Optional[str]:
    """Return catalog-safe chain detection rationale from chain_catalog."""
    return CHAIN_DETECTION_RATIONALE.get(chain_id)


def resolve_paper(slug: str) -> Dict[str, Any]:
    """Merge taxonomy table row, aliases, and bundled research notes section."""
    out: Dict[str, Any] = {"slug": slug}
    papers = load_taxonomy_papers()
    if slug in papers:
        out.update(papers[slug])
    elif slug in SLUG_ALIASES:
        out.update(SLUG_ALIASES[slug])

    numeric = out.get("numeric_id") or _numeric_id_from_slug(slug)
    if numeric:
        out["numeric_id"] = numeric
        if numeric in ARXIV_BY_NUMERIC_ID:
            out["url"] = ARXIV_BY_NUMERIC_ID[numeric]
        section = extract_paper_section(numeric)
        if section:
            out["research_notes_section"] = section

    if "short_name" in out and "year" in out:
        out.setdefault("title", f"{out['short_name']} ({out['year']})")
    return out


def _numeric_id_from_slug(slug: str) -> Optional[str]:
    m = re.match(r"^(\d+)-", slug)
    return m.group(1) if m else None


def format_paper_catalog_block(slug: str) -> str:
    """Format one paper entry for catalog output."""
    from graph_surgeon.taxonomy.research_coverage import paper_coverage_status

    info = resolve_paper(slug)
    lines = [f"### {slug}"]
    numeric = info.get("numeric_id") or _numeric_id_from_slug(slug)
    if numeric:
        cov = paper_coverage_status(numeric)
        if cov == "missing":
            lines.append("Detailed analysis: not yet in corpus.")

    if info.get("title"):
        lines.append(f"Title: {info['title']}")
    if info.get("venue") and info.get("year"):
        lines.append(f"Venue: {info['venue']} {info['year']}")
    if info.get("attack_form"):
        lines.append(f"Attack form: {info['attack_form']}")
    if info.get("detection_status"):
        lines.append(f"Taxonomy coverage: {info['detection_status']}")
    if info.get("url"):
        lines.append(f"Reference: {info['url']}")
    if info.get("summary"):
        lines.append(f"Summary: {info['summary']}")

    section = info.get("research_notes_section")
    if section:
        lines.append("")
        lines.append("Research notes:")
        lines.append(section)
    elif slug not in load_taxonomy_papers() and slug not in SLUG_ALIASES:
        lines.append("(No extended research notes found for this slug.)")

    return "\n".join(lines)
