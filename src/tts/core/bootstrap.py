"""Ensure repo root and ``src/`` are on ``sys.path`` (mirror of root ``repo_bootstrap.py``)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    ps = str(p)
    if p.is_dir() and ps not in sys.path:
        sys.path.insert(0, ps)
