"""Ensure ``src/`` is on ``sys.path`` (no ``tts`` imports — safe to load first)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    ps = str(p)
    if p.is_dir() and ps not in sys.path:
        sys.path.insert(0, ps)
