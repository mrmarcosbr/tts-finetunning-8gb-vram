# Architecture (Clean / layered)

```
presentation/     CLI, env bootstrap, warnings
    └── cli/        train.py, inference.py, f0_metrics.py, eval_metrics.py

application/        Use cases (orchestration)
    ├── training/   pipeline.py — LoRA fine-tuning
    └── inference/  pipeline.py — synthesis + metrics batch

core/               Config, paths, hardware, shared domain helpers
    config.py       load_config()
    config_profile.py
    data_paths.py
    speakers.py     extract_speaker_id()
    hardware.py     detect_hardware_profile(), device_for_speechbrain()
    patches.py      HF / SpeechBrain runtime patches
    torch_compat.py

infrastructure/     External libs & I/O adapters
    audio/          mel, noise reduce, post-EQ
    metrics/        F0 RMSE, WER/CER
```

## Dependency rule

Dependencies flow **inward only**:

`presentation → application → (core + infrastructure)`

- `core` must not import from `application` or `presentation`.
- `infrastructure` may import `core` but not `application`.

## Entry points (repo root)

| File | Layer |
|------|--------|
| `train_exhaustive.py` | → `presentation.cli.train` |
| `test_inference_exhaustive.py` | → `presentation.cli.inference` |
| `f0_infer_metrics.py` | → `presentation.cli.f0_metrics` |
| `eval_metrics_aval.py` | → `presentation.cli.eval_metrics` |

## Backward-compatible imports

Old paths still work via thin shims:

- `tts.config_profile` → `tts.core.config_profile`
- `tts.data_paths` → `tts.core.data_paths`
- `repo_bootstrap` → `tts.core.bootstrap`
