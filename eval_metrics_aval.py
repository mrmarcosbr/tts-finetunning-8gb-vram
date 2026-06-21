"""Backward-compatible CLI entry for WER/CER evaluation."""
import repo_bootstrap  # noqa: F401

from tts.presentation.cli.eval_metrics import main

if __name__ == "__main__":
    raise SystemExit(main())
