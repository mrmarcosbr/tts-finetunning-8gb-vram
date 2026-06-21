"""CLI: F0 RMSE / metrics CSV on inference folder."""
import repo_bootstrap  # noqa: F401

from tts.infrastructure.metrics.f0_infer_metrics import main

if __name__ == "__main__":
    raise SystemExit(main())
