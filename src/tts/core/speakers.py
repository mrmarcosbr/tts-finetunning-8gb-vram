"""Speaker ID extraction from LapsBM / HF dataset rows."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def extract_speaker_id(item: Optional[Dict[str, Any]] = None, *, missing: str = "unknown") -> str:
    if not item:
        return missing
    for k in ("speaker_id", "speaker", "user_id", "client_id"):
        if k in item and item[k] is not None:
            return str(item[k])
    possible_paths = [
        item.get("__url__", ""),
        item.get("wav", {}).get("path", "") if isinstance(item.get("wav"), dict) else "",
        item.get("audio", {}).get("path", "") if isinstance(item.get("audio"), dict) else "",
        item.get("audio", "") if isinstance(item.get("audio"), str) else "",
    ]
    for raw in possible_paths:
        p = str(raw)
        if not p:
            continue
        if "LapsBM-" in p:
            fn = os.path.basename(p.replace("/", os.sep))
            if fn.startswith("LapsBM-"):
                return fn.split(".")[0].replace("LapsBM-", "")
        parts = p.replace("\\", "/").split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            if folder.lower() not in {"audio", "wavs", "clips"} and len(folder) > 1:
                return folder
    return missing
