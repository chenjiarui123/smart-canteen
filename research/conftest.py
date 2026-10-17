"""Pytest bootstrap for the flat research source layout."""

from __future__ import annotations

import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
