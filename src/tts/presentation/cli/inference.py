"""CLI: batch inference and metrics (`test_inference_exhaustive.py`)."""
from __future__ import annotations

import sys

from dotenv import load_dotenv

sys.modules.setdefault("torchcodec", None)

import repo_bootstrap  # noqa: F401
from tts.core.patches import apply_runtime_patches, configure_runtime_warning_filters
from tts.core.torch_compat import import_distributed_tensor_early

load_dotenv()

configure_runtime_warning_filters()
apply_runtime_patches()
import_distributed_tensor_early()

from tts.application.inference.pipeline import main

if __name__ == "__main__":
    main()
