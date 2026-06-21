"""CLI: standalone WER/CER evaluation."""
import repo_bootstrap  # noqa: F401

from tts.infrastructure.metrics.eval_metrics_aval import main

if __name__ == "__main__":
    raise SystemExit(main())
