"""Backward-compatible CLI entry for F0 metrics."""
import repo_bootstrap  # noqa: F401

from tts.presentation.cli.f0_metrics import main

if __name__ == "__main__":
    raise SystemExit(main())
