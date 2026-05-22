"""Research corpus coverage tracking for catalog and tooling."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).parent / "data"
_COVERAGE_JSON = _DATA_DIR / "research_coverage.json"


@lru_cache(maxsize=1)
def load_research_coverage() -> Dict[str, Any]:
    if _COVERAGE_JSON.exists():
        return json.loads(_COVERAGE_JSON.read_text(encoding="utf-8"))
    return {"papers": {}, "summary": {}}


def paper_coverage_status(numeric_id: str) -> Optional[str]:
    papers = load_research_coverage().get("papers", {})
    entry = papers.get(str(numeric_id))
    return entry.get("status") if entry else None


def registry_slugs_missing_notes(registry_slugs: List[str]) -> List[str]:
    """Return slugs whose numeric id has no complete/out_of_scope analysis in corpus."""
    missing: List[str] = []
    for slug in registry_slugs:
        m = re.match(r"^(\d+)-", slug)
        if not m:
            continue
        status = paper_coverage_status(m.group(1))
        if status not in ("complete", "out_of_scope"):
            missing.append(slug)
    return missing


def format_coverage_report() -> str:
    data = load_research_coverage()
    summary = data.get("summary", {})
    papers: Dict[str, Any] = data.get("papers", {})
    lines = [
        "GraphSurgeon research corpus coverage",
        "=" * 40,
        f"Complete:      {summary.get('complete', 0)}",
        f"Out of scope:  {summary.get('out_of_scope', 0)}",
        f"Missing:       {summary.get('missing', 0)}",
        "",
    ]
    missing_ids = [pid for pid, p in sorted(papers.items(), key=lambda x: int(x[0])) if p.get("status") == "missing"]
    if missing_ids:
        lines.append("Missing analysis:")
        for pid in missing_ids:
            lines.append(f"  [{pid}] {papers[pid].get('slug', '')}")
    else:
        lines.append("All taxonomy papers have terminal status (complete or out_of_scope).")
    oos = [pid for pid, p in papers.items() if p.get("status") == "out_of_scope"]
    if oos:
        lines.extend(["", f"Out-of-scope ({len(oos)}): " + ", ".join(sorted(oos, key=int))])
    return "\n".join(lines)
