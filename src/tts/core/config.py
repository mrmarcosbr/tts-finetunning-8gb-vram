"""YAML configuration loading."""
from __future__ import annotations

from typing import Any, Dict

import yaml


def load_config(config_path: str = "configs/config_train.yaml") -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
