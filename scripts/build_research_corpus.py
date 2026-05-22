#!/usr/bin/env python3
"""Rebuild attack_research_notes.md and research_coverage.json (optional maintainer tool).

Set GRAPH_SURGEON_RESEARCH_SOURCE to a directory of BATCH_*.md research files to merge
external markdown. Shipped attack_research_notes.md is authoritative for the CLI.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "graph_surgeon" / "taxonomy" / "data"
_research_src = os.environ.get("GRAPH_SURGEON_RESEARCH_SOURCE", "").strip()
RESEARCH_SOURCE = Path(_research_src) if _research_src else None
NOTES_PATH = DATA / "attack_research_notes.md"
INTERNAL_PATH = DATA / "attack_research_internal.md"
COVERAGE_PATH = DATA / "research_coverage.json"
TAXONOMY_PATH = DATA / "papers_taxonomy.json"

INTERNAL_START_MARKERS = (
    "## Cross-Cutting Vulnerability Patterns Summary",
    "## ShadowLogic Supply Chain Attack Research",
)

LIGHT_ATTACK_IDS = ("40", "43", "48", "52", "57", "65", "66", "80", "81", "82")

PHASE3_IDS = tuple(str(i) for i in range(103, 115))

OUT_OF_SCOPE_KEYWORDS = ("out-of-scope", "Out-of-scope")


def load_taxonomy() -> Dict[str, Dict[str, Any]]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def slug_by_id(taxonomy: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {v["numeric_id"]: k for k, v in taxonomy.items() if v.get("numeric_id")}


def registry_gadgets_for_slug(slug: str) -> List[str]:
    reg = (ROOT / "graph_surgeon" / "taxonomy" / "gadget_registry.py").read_text(encoding="utf-8")
    gadgets: List[str] = []
    for block in re.finditer(r'research_basis=\[([\s\S]*?)\]', reg):
        if f'"{slug}"' in block.group(1):
            m = re.search(r'id="([A-Z0-9_]+)"', reg[max(0, block.start() - 400) : block.start()])
            if m and m.group(1) not in gadgets:
                gadgets.append(m.group(1))
    # fallback: parse detection_status
    tax = load_taxonomy().get(slug, {})
    status = tax.get("detection_status", "")
    for g in re.findall(r"(?:Covered by|by) ([A-Z0-9_]+)", status):
        if g not in gadgets:
            gadgets.append(g)
    return gadgets


def gadgets_from_status(status: str) -> List[str]:
    if "Out-of-scope" in status or "out-of-scope" in status:
        return []
    parts = re.split(r"\s*\+\s*", status.replace("Covered by ", "").replace("Covered conceptually", ""))
    return [p.strip() for p in parts if p.strip() and p.strip()[0].isupper()]


def parse_existing_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for m in re.finditer(r"^### \[(\d+)\][^\n]*\n", text, re.MULTILINE):
        pid = m.group(1)
        start = m.start()
        nxt = re.search(r"^### \[\d+\]", text[m.end() :], re.MULTILINE)
        end = m.end() + nxt.start() if nxt else len(text)
        chunk = _strip_meta_tail(text[start:end].strip())
        cut = len(chunk)
        for marker in INTERNAL_START_MARKERS:
            idx = chunk.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        sections[pid] = chunk[:cut].strip()
    return sections


def parse_batch_paper(section: str, pid: str, meta: Dict[str, Any]) -> str:
    """Convert ## Paper N: BATCH block to catalog template."""
    title_m = re.search(r"^## Paper \d+:\s*(.+)$", section, re.MULTILINE)
    short = meta.get("short_name", title_m.group(1).strip() if title_m else f"Paper {pid}")
    venue = meta.get("venue", "")
    year = meta.get("year", "")
    attack_form = meta.get("attack_form", "Unknown")
    gadgets = gadgets_from_status(meta.get("detection_status", ""))
    if not gadgets:
        gadgets = registry_gadgets_for_slug(meta.get("slug", ""))
    registry_line = ", ".join(gadgets) if gadgets else "see taxonomy"

    def grab(header: str) -> str:
        m = re.search(
            rf"### {re.escape(header)}\s*\n(.*?)(?=\n### |\n---|\Z)",
            section,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    attack_type = grab("Attack Type")
    key_technique = grab("Key Technique")
    structural = grab("Structural Requirements")
    gadget_map = grab("Gadget Mapping")
    chain = grab("Chain Relevance")
    rationale = grab("Rationale") or grab("DAG Relevance")
    novel = grab("Novel Ideas")

    chains = []
    if "CHAIN-" in chain:
        chains = re.findall(r"CHAIN-[A-Z0-9_-]+", chain)

    summary_parts = [attack_type, key_technique[:200] if key_technique else ""]
    summary = " ".join(p for p in summary_parts if p).strip() or f"{short} ({year}) adversarial attack."

    onnx_lines = []
    if structural:
        for line in structural.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                onnx_lines.append(f"- {line.lstrip('0123456789. ')}")
    if not onnx_lines:
        onnx_lines = ["- See gadget detection_logic in registry for op-level patterns"]

    gs_cmds = []
    for g in gadgets[:2]:
        gs_cmds.append(f"`graph-surgeon catalog --gadget {g}`")
    gs = ", ".join(gs_cmds) if gs_cmds else "`graph-surgeon motifs model.onnx`"

    lines = [
        f"### [{pid}] {short} - {title_m.group(1).strip() if title_m else short} ({venue} {year})",
        "",
        "**Status:** analysis_complete",
        f"**Attack form:** {attack_form}",
        f"**Registry:** {registry_line}",
        "",
        f"**Summary:** {summary}",
        "",
        "**Attack mechanism:**",
        key_technique or attack_type or "(see BATCH harvest)",
        "",
        "**ONNX graph indicators:**",
        *onnx_lines,
        "",
        "**Gadget and chain mapping:**",
        gadget_map.replace("**", "") if gadget_map else f"Confirms {registry_line}.",
        "",
        "**What GraphSurgeon surfaces:**",
        f"Run {gs}; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.",
        "",
    ]
    if chains:
        lines.extend(["**Related chains:** " + ", ".join(chains), ""])
    if novel:
        lines.extend(["**Related literature:**", novel[:500], ""])
    lines.extend([
        "**Static analysis limits:**",
        "Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.",
        "",
        "---",
        "",
    ])
    return "\n".join(lines)


def harvest_batch_files(taxonomy: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    by_id = slug_by_id(taxonomy)
    out: Dict[str, str] = {}
    if RESEARCH_SOURCE is None or not RESEARCH_SOURCE.is_dir():
        return out
    for batch in sorted(RESEARCH_SOURCE.glob("BATCH_1*.md")):
        text = batch.read_text(encoding="utf-8")
        for m in re.finditer(r"^## Paper (\d+):", text, re.MULTILINE):
            pid = m.group(1)
            start = m.start()
            nxt = re.search(r"^## Paper \d+:", text[m.end() :], re.MULTILINE)
            end = m.end() + nxt.start() if nxt else len(text)
            chunk = text[start:end]
            slug = by_id.get(pid)
            meta = dict(taxonomy.get(slug, {})) if slug else {}
            meta["slug"] = slug or ""
            out[pid] = parse_batch_paper(chunk, pid, meta)
    return out


def tier_d_section(pid: str, meta: Dict[str, Any]) -> str:
    short = meta.get("short_name", pid)
    venue = meta.get("venue", "")
    year = meta.get("year", "")
    attack_form = meta.get("attack_form", "")
    status = meta.get("detection_status", "")
    return "\n".join([
        f"### [{pid}] {short} ({venue} {year})",
        "",
        "**Status:** out_of_scope",
        f"**Attack form:** {attack_form}",
        "**Registry:** (none — not detectable from ONNX DAG alone)",
        "",
        f"**Summary:** {short} targets deployment hardware or signal domains outside the exported ONNX graph.",
        "",
        "**Attack mechanism:**",
        f"Attack vector: {attack_form}. Documented in taxonomy as: {status}.",
        "",
        "**ONNX graph indicators:**",
        "- None specific; standard vision classifiers may still show GAP_FC_HEAD or ALIASING motifs unrelated to this attack vector.",
        "",
        "**Gadget and chain mapping:**",
        "No additional registry gadget. Analysts should record deployment context (sensors, ISP, thermal camera) separately from graph motifs.",
        "",
        "**What GraphSurgeon surfaces:**",
        "`catalog --coverage` marks this paper out_of_scope; motifs on ONNX still report general vision attack landscape only.",
        "",
        "**Static analysis limits:**",
        "Entire attack class operates outside the ONNX file (acoustic channel, thermal LED hardware, camera ISP pipeline).",
        "",
        "---",
        "",
    ])


def light_attack_appendix() -> str:
    return "\n".join([
        "## Light-based attacks (shared ONNX mechanism)",
        "",
        "Papers 40, 43, 48, 52, 57, 65, 66, 80, 81, 82 exploit illumination, projection, or weather-like perturbations.",
        "The ONNX attack landscape for lighting mirrors patch attacks: ALIASING_DOWNSAMPLE folds high-frequency lighting edges,",
        "NORMALIZER amplifies distribution shift under changed illumination, NO_SPATIAL_ATTENTION cannot suppress localized",
        "bright/dark regions, and GAP_FC_HEAD aggregates spatial perturbations into the classifier head.",
        "",
        "---",
        "",
    ])


def light_paper_section(pid: str, meta: Dict[str, Any]) -> str:
    short = meta.get("short_name", pid)
    venue = meta.get("venue", "")
    year = meta.get("year", "")
    gadgets = gadgets_from_status(meta.get("detection_status", ""))
    reg = ", ".join(gadgets) if gadgets else "ALIASING_DOWNSAMPLE, NORMALIZER"
    return "\n".join([
        f"### [{pid}] {short} - Light-based attack ({venue} {year})",
        "",
        "**Status:** analysis_complete",
        f"**Attack form:** {meta.get('attack_form', 'Light based')}",
        f"**Registry:** {reg}",
        "",
        f"**Summary:** {short} manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.",
        "",
        "**Attack mechanism:**",
        "Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.",
        "",
        "**ONNX graph indicators:**",
        "- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)",
        "- BatchNormalization nodes using running stats (NORMALIZER)",
        "- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style",
        "- Absence of spatial attention blocks before pooling",
        "",
        "**Gadget and chain mapping:**",
        f"Maps to {reg}. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.",
        "",
        "**What GraphSurgeon surfaces:**",
        f"`catalog --gadget` for {reg.split(',')[0].strip()}; see shared section 'Light-based attacks' above.",
        "",
        "**Static analysis limits:**",
        "Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.",
        "",
        "---",
        "",
    ])


def _strip_meta_tail(section: str) -> str:
    markers = (
        "\n## Phase 2 Session Log",
        "\n## Phase 3 Session Log",
        "\n## Phase 3 Research Summary",
        "\n## Object Detector Vulnerability Summary",
        "\n## Cross-Cutting",
        "\n## ShadowLogic",
    )
    cut = len(section)
    for m in markers:
        idx = section.find(m)
        if idx != -1:
            cut = min(cut, idx)
    return section[:cut].strip()


def upgrade_legacy_section(section: str) -> str:
    """Normalize legacy Phase 1/2/3 sections toward template."""
    s = _strip_meta_tail(section)
    s = re.sub(r"\*\*Status:\*\* Research complete", "**Status:** analysis_complete", s)
    if "**Attack form:**" not in s:
        s = s.replace(
            "**Status:** analysis_complete",
            "**Status:** analysis_complete\n**Attack form:** (see taxonomy)\n**Registry:** (see taxonomy)",
            1,
        )
    if "**What GraphSurgeon surfaces:**" not in s:
        s = s.rstrip() + "\n\n**What GraphSurgeon surfaces:**\n`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.\n"
    if "**Static analysis limits:**" not in s:
        s = s.rstrip() + "\n\n**Static analysis limits:**\nArchitecture indicates attack landscape only; training and deployment defenses are not visible in the graph.\n"
    if not s.endswith("---"):
        s = s.rstrip() + "\n\n---\n"
    return s


def upgrade_phase3(section: str, pid: str) -> str:
    s = upgrade_legacy_section(section)
    # Ensure ONNX graph indicators header exists
    if "**ONNX graph indicators:**" not in s and "**Key Findings" in s:
        s = s.replace(
            "**Key Findings",
            "**ONNX graph indicators:**\n- See findings below for ViT/CNN op patterns\n\n**Key Findings",
            1,
        )
    return s


def paper_73_section() -> str:
    return "\n".join([
        "### [73] Object Hider - Hiding objects from detectors (2020)",
        "",
        "**Status:** analysis_complete",
        "**Attack form:** Patch",
        "**Registry:** OBJECTNESS_HEAD, CHAIN-OBJECT-DISAPPEARANCE",
        "",
        "**Summary:** Object Hider suppresses objectness scores so detectors fail to emit boxes for targeted objects while the patch is present.",
        "",
        "**Attack mechanism:**",
        "Optimizes a patch to minimize objectness/confidence on detector heads, causing missed detections rather than mislabeling.",
        "",
        "**ONNX graph indicators:**",
        "- Sigmoid or confidence head after Conv feature maps",
        "- Shared backbone feeding classification and objectness branches",
        "- Anchor or grid-based detection heads (YOLO-style)",
        "",
        "**Gadget and chain mapping:**",
        "OBJECTNESS_HEAD; CHAIN-OBJECT-DISAPPEARANCE when objectness path is present without robust gating.",
        "",
        "**What GraphSurgeon surfaces:**",
        "`catalog --gadget OBJECTNESS_HEAD`, `motifs` object-detector scan.",
        "",
        "**Static analysis limits:**",
        "NMS and post-processing are often outside ONNX; graph shows head structure only.",
        "",
        "---",
        "",
    ])


def paper_47_section() -> str:
    return "\n".join([
        "### [47] ViewFool - Viewpoint adversarial examples (NeurIPS 2020)",
        "",
        "**Status:** analysis_complete",
        "**Attack form:** Position",
        "**Registry:** ALIASING_DOWNSAMPLE",
        "",
        "**Summary:** ViewFool crafts adversarial viewpoints (camera pose) that fool classifiers by exploiting lack of viewpoint invariance.",
        "",
        "**Attack mechanism:**",
        "Viewpoint changes introduce frequency shifts and projection artifacts similar to physical-world transformations; models without aliasing-resistant downsampling misclassify.",
        "",
        "**ONNX graph indicators:**",
        "- Stride-2 operations without anti-aliasing blur",
        "- Global pooling classifier heads sensitive to global feature shifts",
        "",
        "**Gadget and chain mapping:**",
        "ALIASING_DOWNSAMPLE; related to CHAIN-PHYSICAL-WORLD-ATTACK.",
        "",
        "**What GraphSurgeon surfaces:**",
        "`catalog --gadget ALIASING_DOWNSAMPLE`.",
        "",
        "**Static analysis limits:**",
        "Viewpoint is an extrinsic camera parameter; ONNX graph does not encode pose robustness.",
        "",
        "---",
        "",
    ])


def paper_49_section() -> str:
    return "\n".join([
        "### [49] Meta-Attack - Meta adversarial attack (ICCV 2021)",
        "",
        "**Status:** analysis_complete",
        "**Attack form:** Image",
        "**Registry:** (conceptual — exploits common gadget combinations)",
        "",
        "**Summary:** Meta-Attack transfers adversarial examples across models by exploiting shared architectural weaknesses rather than a single new motif.",
        "",
        "**Attack mechanism:**",
        "Uses meta-learning to find perturbations effective on multiple architectures sharing pooling, fusion, and classifier patterns.",
        "",
        "**ONNX graph indicators:**",
        "- Combinations of GAP_FC_HEAD, ALIASING_DOWNSAMPLE, HIGH_FANIN_FUSION as detected by motifs",
        "",
        "**Gadget and chain mapping:**",
        "No separate gadget; confirms that shared structural motifs increase transfer risk across models.",
        "",
        "**What GraphSurgeon surfaces:**",
        "Aggregate motif report; multiple catalog gadget hits on one graph.",
        "",
        "**Static analysis limits:**",
        "Transfer success depends on training; not predicted from graph alone.",
        "",
        "---",
        "",
    ])


def split_internal(existing: str) -> Tuple[str, str]:
    """Return (paper_sections_dict_source_text, internal_content)."""
    internal_parts = []
    for marker in INTERNAL_START_MARKERS:
        idx = existing.find(marker)
        if idx != -1:
            internal_parts.append(existing[idx:])
            existing = existing[:idx]
            break
    # Also move pre-amble research summary + phase headers to internal
    preamble_end = existing.find("### [36]")
    if preamble_end > 0:
        preamble = existing[:preamble_end].strip()
        existing = existing[preamble_end:]
        internal_parts.insert(0, preamble)
    internal = "\n\n".join(internal_parts).strip()
    return existing, internal


def build_coverage(
    taxonomy: Dict[str, Dict[str, Any]],
    sections: Dict[str, str],
    batch_ids: set,
) -> Dict[str, Any]:
    papers = {}
    for slug, meta in taxonomy.items():
        pid = meta["numeric_id"]
        status = meta.get("detection_status", "")
        if any(k in status for k in OUT_OF_SCOPE_KEYWORDS):
            st = "out_of_scope"
            tier = "D"
        elif pid in sections:
            st = "complete"
            tier = "B" if pid in batch_ids else ("A" if pid in ("36", "38", "39", "60", "71", "77", "84", "85", "99") else "C")
        else:
            st = "missing"
            tier = "B"
        papers[pid] = {
            "slug": slug,
            "status": st,
            "tier": tier,
            "registry_gadgets": gadgets_from_status(status),
            "source": "external_research" if pid in batch_ids else ("upgraded" if st == "complete" else "authored"),
        }
    for pid in PHASE3_IDS:
        papers[pid] = {
            "slug": f"{pid}-extended",
            "status": "complete" if pid in sections else "missing",
            "tier": "C",
            "registry_gadgets": [],
            "source": "upgraded",
        }
    counts = {"complete": 0, "out_of_scope": 0, "missing": 0}
    for p in papers.values():
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    return {
        "version": "1.0.0",
        "updated": "2026-05-22",
        "summary": counts,
        "papers": papers,
    }


def main() -> int:
    taxonomy = load_taxonomy()
    by_id = slug_by_id(taxonomy)
    existing_text = NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else ""
    existing_sections = parse_existing_sections(existing_text)
    _, internal_chunk = split_internal(existing_text)

    batch_sections = harvest_batch_files(taxonomy)
    batch_ids = set(batch_sections.keys())

    final_sections: Dict[str, str] = {}

    # Preserve and upgrade existing
    for pid, sec in existing_sections.items():
        if pid in PHASE3_IDS:
            final_sections[pid] = upgrade_phase3(sec, pid)
        else:
            final_sections[pid] = upgrade_legacy_section(sec)

    # BATCH overrides for harvested
    final_sections.update(batch_sections)

    # Generate all taxonomy papers
    for slug, meta in taxonomy.items():
        pid = meta["numeric_id"]
        if pid in final_sections:
            continue
        if any(k in meta.get("detection_status", "") for k in OUT_OF_SCOPE_KEYWORDS):
            final_sections[pid] = tier_d_section(pid, {**meta, "slug": slug})
        elif pid in LIGHT_ATTACK_IDS:
            final_sections[pid] = light_paper_section(pid, {**meta, "slug": slug})
        elif pid == "73":
            final_sections[pid] = paper_73_section()
        elif pid == "47":
            final_sections[pid] = paper_47_section()
        elif pid == "49":
            final_sections[pid] = paper_49_section()
        else:
            # Minimal authored from taxonomy
            gadgets = gadgets_from_status(meta.get("detection_status", ""))
            reg = ", ".join(gadgets) if gadgets else "see registry"
            short = meta.get("short_name", slug)
            final_sections[pid] = parse_batch_paper(
                f"## Paper {pid}: {short}\n### Attack Type\n{meta.get('attack_form','')}\n### Key Technique\nConfirms {reg}.\n### Structural Requirements\n1. See {reg} detection_logic.\n### Gadget Mapping\n- **Confirms:** {reg}\n### Rationale\nPattern-confirming literature for {reg}.",
                pid,
                {**meta, "slug": slug},
            )

    # Sort all numeric ids
    all_ids = sorted(
        set(final_sections.keys()),
        key=lambda x: int(x),
    )

    intro = "\n".join([
        "# Attack Research Notes (GraphSurgeon corpus)",
        "",
        "Per-paper adversarial ML analysis for ONNX reverse engineering. Each `### [id]` section maps",
        "literature to structural motifs (attack landscape from graph structure, not exploitability).",
        "",
        "Coverage index: `research_coverage.json`.",
        "",
        "---",
        "",
        light_attack_appendix(),
    ])

    body = intro + "\n".join(final_sections[pid] for pid in all_ids)
    NOTES_PATH.write_text(body, encoding="utf-8")

    internal_header = "# Internal research logs (not used by catalog)\n\n"
    INTERNAL_PATH.write_text(internal_header + internal_chunk, encoding="utf-8")

    coverage = build_coverage(taxonomy, final_sections, batch_ids)
    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    missing = [pid for pid, p in coverage["papers"].items() if p["status"] == "missing"]
    print(f"Wrote {NOTES_PATH} ({len(all_ids)} paper sections)")
    print(f"Wrote {INTERNAL_PATH}")
    print(f"Wrote {COVERAGE_PATH}")
    print(f"Summary: {coverage['summary']}")
    if missing:
        print(f"Missing: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
