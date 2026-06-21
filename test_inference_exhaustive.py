"""Entry point: inference. Implementation: ``tts.application.inference.pipeline``."""
import repo_bootstrap  # noqa: F401

from tts.presentation.cli.inference import main

if __name__ == "__main__":
    main()
