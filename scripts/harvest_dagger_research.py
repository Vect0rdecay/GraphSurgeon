#!/usr/bin/env python3
"""One-time harvest: merge Dagger BATCH analyses into GraphSurgeon corpus. Runs build_research_corpus."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "build_research_corpus.py"), run_name="__main__")
