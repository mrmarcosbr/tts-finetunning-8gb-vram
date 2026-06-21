# Legacy / archived material

Arquivos **fora do fluxo principal** (`train_exhaustive.py`, `test_inference_exhaustive.py`).
Mantidos para referência, experimentos antigos e scripts de debug — não apagar sem revisar.

**Critério:** só entra aqui o que **não** é referenciado por parâmetros, configs ou scripts ativos na raiz (`train_exhaustive.py`, `test_inference_exhaustive.py`, `scripts/data/`, `configs/`). Recursos opcionais do pipeline (ex.: `--use-test-embeddings` → `embeddings_test_speakers/`) ficam **fora** de `legacy/`, mesmo desligados por default.

## Estrutura

| Pasta | Conteúdo |
|-------|----------|
| `scripts/` | Treino comentado, testes unitários soltos, utilitários WER/NR, pasta `other_test_scripts/` |
| `logs/` | Logs antigos (`out-*.txt`, `error_log.txt`) |
| `docs/` | Cópias de config, notas de avaliação, `valores_referencia_treinamento.txt`, `prompt.txt` |
| `figures/` | Imagens (ex.: diagramas para relatório) |
| `audio/` | WAVs de teste soltos na raiz do repo |
| `aval_audios/` | Áudios de avaliação antigos (não referenciados por nenhum script ativo) |

## Scripts legacy ainda úteis (rodar da **raiz** do repo)

```powershell
# WER/CER só em pasta de inferência antiga
python legacy/scripts/other_test_scripts/retro_inference_wer_cer_csv.py ".\output_cuda_16gb\...\inference_batch_..."

# Médias WER/CER no modo default_texts triplo
python legacy/scripts/triple_treinado_wer_cer_means.py "...\treinado_wav"
```

## O que **não** está aqui

- Checkpoints: `output_cuda_*` (gitignore)
- Datasets locais: `data/` (`raw/`, `repository/`, `cache/`, `exports/`)
- Ambientes: `venv_global/`, `venv_fs2/`
Scripts em `legacy/` podem depender de imports antigos (módulos na raiz). Use apenas para referência; o código ativo está em `src/tts/` — ver `docs/PROJECT_STRUCTURE.md`.
