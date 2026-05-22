# Research note template (GraphSurgeon corpus)

Author new entries in `attack_research_notes.md` using this structure. One section per paper: `### [numeric_id] ShortName - Title (Venue YEAR)`.

Required fields:

- **Status:** `analysis_complete` or `out_of_scope`
- **Attack form:** from taxonomy
- **Registry:** comma-separated gadget and chain IDs from `gadget_registry.py`

Body sections:

1. **Summary** — 2-3 sentences (attack goal, domain)
2. **Attack mechanism** — how the attack works
3. **ONNX graph indicators** — ops and subgraph patterns visible in a static DAG
4. **Gadget and chain mapping** — which motifs apply and why (attack landscape, not exploitability)
5. **What GraphSurgeon surfaces** — `motifs`, `patterns`, `catalog --gadget`, `topology`
6. **Related literature** — optional cross-refs
7. **Static analysis limits** — what the graph alone cannot establish

Do not include: phase logs, session logs, implementation todos, ShadowLogic build notes, or proposed-gadget brainstorming tables.
