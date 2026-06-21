"""Entry point: training. Implementation: ``tts.application.training.pipeline``."""
import repo_bootstrap  # noqa: F401

from tts.presentation.cli.train import main

if __name__ == "__main__":
    main()
