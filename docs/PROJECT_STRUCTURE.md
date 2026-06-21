# Project structure

Fluxo principal: **SpeechT5 fine-tuning + inferência** no LapsBM.  
Arquitetura em camadas: ver [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Layout

```
tts-finetunning-8gb-vram/
├── train_exhaustive.py              # → presentation.cli.train
├── test_inference_exhaustive.py     # → presentation.cli.inference
├── f0_infer_metrics.py              # → presentation.cli.f0_metrics
├── eval_metrics_aval.py             # → presentation.cli.eval_metrics
├── repo_bootstrap.py                # → tts.core.bootstrap
├── run_with_log.ps1
├── configs/
├── src/tts/
│   ├── core/                        # config, paths, hardware, patches
│   ├── infrastructure/
│   │   ├── audio/
│   │   └── metrics/
│   ├── application/
│   │   ├── training/pipeline.py
│   │   └── inference/pipeline.py
│   └── presentation/cli/
├── scripts/setup/                   # venv patches
├── scripts/data/                    # download, export embeddings
├── embeddings_test_speakers/
├── data/                            # raw, repository, cache, exports
├── logs/
└── legacy/
```

## Entry points

| Script | Função |
|--------|--------|
| `train_exhaustive.py` | Treino LoRA |
| `test_inference_exhaustive.py` | Síntese + métricas |
| `run_with_log.ps1` | Wrapper → `logs/out-<timestamp>.txt` |

## Configuração

| Ficheiro | Uso |
|----------|-----|
| `configs/config_train.yaml` | Perfil unificado `profiles.lapsbm_speecht5` |
| `configs/config_inference.yaml` | Defaults de inferência |

Resolução: `tts.core.config_profile.resolve_training_profile`.

## Dados auxiliares

| Pasta | Uso |
|-------|-----|
| `data/raw/` | Espelho offline Hugging Face |
| `data/repository/` | Dataset serializado (`load_from_disk`) |
| `data/cache/` | Cache de pré-processamento |
| `data/exports/` | WAVs de referência para inferência |
| `embeddings_test_speakers/` | x-vectors (`--use-test-embeddings`) |

Paths: `tts.core.data_paths`.

## Legacy

Ver [`legacy/README.md`](../legacy/README.md).
