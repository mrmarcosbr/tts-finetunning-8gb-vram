"""CLI: LoRA training (`train_exhaustive.py`)."""
from __future__ import annotations

from dotenv import load_dotenv

import repo_bootstrap  # noqa: F401
from tts.core.patches import apply_runtime_patches, configure_runtime_warning_filters
from tts.core.torch_compat import (
    apply_torch_load_compat,
    bypass_transformers_torch_version_guard,
    import_distributed_tensor_early,
)

load_dotenv()

configure_runtime_warning_filters()
apply_runtime_patches()
import_distributed_tensor_early()
apply_torch_load_compat()
bypass_transformers_torch_version_guard()

from tts.application.training.pipeline import main

if __name__ == "__main__":
    main()
