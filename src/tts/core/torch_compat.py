"""PyTorch compatibility shims applied before model load."""
from __future__ import annotations

import importlib

import torch


def apply_torch_load_compat() -> None:
    """Allow numpy arrays in checkpoints (PyTorch 2.6+ weights_only default)."""
    _original_load = torch.load

    def safe_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = safe_load


def import_distributed_tensor_early() -> None:
    try:
        importlib.import_module("torch.distributed.tensor")
    except ImportError:
        pass
