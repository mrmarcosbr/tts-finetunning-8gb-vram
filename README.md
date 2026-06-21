.\run_with_log.ps1 train_exhaustive.py --config configs/config_train.yaml 

.\run_with_log.ps1 train_exhaustive.py --config configs/config_train.yaml --resume_from ".\output_cuda_16gb\speecht5-lapsbm_speecht5-2026-04-30-20-14-15\checkpoint-4900" 

python test_inference_exhaustive.py --model_path ".\output_cuda_16gb\speecht5-lapsbm_speecht5-2026-05-01-00-37-27\checkpoint-9900" --config configs/config_train.yaml --compute_f0_rmse --dataset_reference_audios --infer_all_test_sentences

python test_inference_exhaustive.py --model_path ".\output_cuda_16gb\speecht5-lapsbm_speecht5-2026-04-27-02-20-47\checkpoint-9500" --config configs/config_train.yaml --dataset_reference_audios --infer_all_test_sentences --speecht5_zero_speaker_embedding


# tts-finetunning-8gb-vram

## Estrutura do repositório

- **Ativo:** entry points na raiz → `src/tts/` (camadas clean) — ver [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Logs novos:** `logs/out-<timestamp>.txt` (via `run_with_log.ps1`)
- **Dados locais:** `data/` — ver secção [Estrutura de dados](#estrutura-de-dados-data) abaixo
- **Embeddings opcionais:** `embeddings_test_speakers/` — usado com `--use-test-embeddings` (pipeline ativo; não é legacy)
- **Arquivo:** só material **sem** referência nos scripts/configs ativos — ver [`legacy/`](legacy/README.md)

- Testado com python 3.11.9 (Windows) e 3.12.3 (Linux)
- Drivers CUDA da NVIDA para rodar pytorch usando a GPU

# Cria ambientes com uv
- Instale o `uv` se ainda não tiver: https://docs.astral.sh/uv/getting-started/installation/
- `uv venv venv_global`
- `.\venv_global\Scripts\Activate.ps1`
- `uv pip install -r requirements.txt --index-strategy unsafe-best-match`
- Depois de concluir o ambiente global, crie o ambiente do FS2: `uv venv venv_fs2`
- Para instalar o FS2 sem baixar o torch de novo, use o cache do uv com hardlinks: `uv pip install --python .\venv_fs2 --requirement requirements_fs2.txt --link-mode hardlink --index-strategy unsafe-best-match`

# Instala dependências
- O ambiente `venv_global` passa a conter os pacotes do projeto via `uv`.
- O ambiente `venv_fs2` recebe as dependências específicas do FastSpeech 2 via `uv`, reaproveitando o cache do `torch` já baixado.
- Se preferir usar o fluxo do projeto com `pyproject.toml`, `uv sync` também está configurado.

# Token Hugging Face
- Crie uma conta na Hugging Face - https://huggingface.co/settings/tokens - e gere um Token de Acesso (permissão de leitura) para fazer o Download os Modelos pré-treinados necessários para o fine-tuning e inferência.
- Configure a variável de ambiente `HF_TOKEN` com o token no arquivo .env ou via terminal/variáveis de ambiente do sistema operacional. 
 
# Aplicar Patches e Correções Nativas 
Algumas bibliotecas como SpeechBrain e HuggingFace podem apresentar erros e alertas conflitantes (ex: warnings do `autocast`, bugs de `HF_TOKEN` suprimido e erros falsos 404 HTTP). Para ter um ambiente de console limpo e livre de interrupções falsas, execute os scripts de correção que injetam os reparos diretamente na sua subpasta `venv_global` recém instalada (somente depois de já ter instalado os pacotes com `uv pip install`):
- `python scripts/setup/apply_patches_no_warnings.py`

# Estrutura de dados (`data/`)

Todo o material de datasets fica sob **`data/`** (configurável em `configs/config_train.yaml` → `settings`). Nada disto é versionado no Git.

```
data/
├── raw/              # Espelho offline do Hugging Face (download_datasets.py)
│   └── lapsbm/       #   Ficheiros brutos do repo HF (tar.gz, etc.)
├── repository/       # Dataset para treino/inferência (formato save_to_disk)
│   └── lapsbm/       #   Criado na 1.ª corrida ou ao gravar após streaming HF
├── cache/            # Cache de pré-processamento (mel, NR, multi-frase)
│   └── speecht5_lapsbm_speecht5_16000hz_.../
└── exports/          # Exportações auxiliares para inferência
    └── lapsbm_speecht5/
        ├── audio/           # WAVs de referência por locutor/frase
        └── hf_test_split/   # Split HF exportado (opcional)
```

| Pasta | Função | Como é populada |
|-------|--------|-----------------|
| **`raw/`** | Backup offline permanente do Hub | `python scripts/data/download_datasets.py --repo falabrasil/lapsbm` |
| **`repository/`** | Leitura em treino/inferência (`load_from_disk`) | Streaming HF na 1.ª corrida ou migração manual |
| **`cache/`** | Evita recomputar mel/NR/splits processados | Gerado automaticamente pelo treino |
| **`exports/`** | WAVs de referência para métricas/inferência | Scripts de export (ex. legacy `reconstruct_and_export_test_split`) |

**Nota:** `raw/` e `repository/` têm formatos diferentes. O treino usa **`repository/`**; `raw/` serve só como espelho offline opcional.

### Embeddings de teste (`embeddings_test_speakers/`)

Pasta do pipeline de inferência ( **não** legacy): x-vectors pré-exportados dos locutores M031–M034 para o modo `--use-test-embeddings`. Geração:

```powershell
python scripts/data/export_test_speaker_embeddings.py --device cuda
```

Com `use_test_embeddings: false` no `config_inference.yaml` (default), a inferência usa `--dataset_reference_audios` em vez desta pasta.

### Migrar pastas antigas

Se ainda tiver `meus_datasets/` ou `datasets/` na raiz:

```powershell
# na raiz do repo
New-Item -ItemType Directory -Force data/raw, data/repository, data/cache, data/exports
Move-Item meus_datasets/* data/raw/ -ErrorAction SilentlyContinue
Move-Item datasets/lapsbm data/repository/ -ErrorAction SilentlyContinue
Move-Item datasets/cache_processado/* data/cache/ -ErrorAction SilentlyContinue
Move-Item datasets/test_split_inferencia/* data/exports/ -ErrorAction SilentlyContinue
```

# Trabalhando com Datasets Offline (Opcional)
Para espelhar os repositórios Hugging Face localmente (sem depender da internet no download):

- `python scripts/data/download_datasets.py --repo falabrasil/lapsbm`
- `python scripts/data/download_datasets.py --all` (LapsBM + pt-br_char)

Os ficheiros vão para **`data/raw/`**. Na primeira corrida de treino, se `data/repository/lapsbm/` ainda não existir, o script faz streaming do Hub e grava em **`data/repository/`** automaticamente.

# Configuração do Treinamento/Fine-Tuning
Verificar `configs/config_train.yaml` (perfil unificado `profiles.lapsbm_speecht5`: hardware + SpeechT5 + LapsBM) e `configs/config_inference.yaml` (defaults opcionais para inferência; ver `--inference_config`).

# Treina Modelo (Fine Tunning)
Usando Detecção Automática de Hardware: 
- `python train_exhaustive.py` 

Usando profile explícito (opcional; o default é `lapsbm_speecht5`):
- `python train_exhaustive.py --profile lapsbm_speecht5`

Realiza o treinamento de modelo existente em inglês com novos áudios transcritos em português. O objetivo deste fine tunning é permitir que um modelo construído puramente para responder audios em inglês, consigo também responder em português.

- Levou em torno de 45 Minutos para treinar com uma RTX 5060 TI 16GB de VRAM (Ryzen 9950X3D - Desktop)
- Levou em torno de 1 hora para treinar com uma RTX 4070 Mobile de 8GB de VRAM (Notebook)
- O arquivo final com os pesos do modelo (Apenas último checkpoint) ficou com 500 MB.

# Realiza Inferência (Converte texto em português para Áudio com o Modelo retreinado)
Usando Detecção Automática de Hardware: 
- `python test_inference_exhaustive.py`

Com defaults de `configs/config_inference.yaml` (perfil `lapsbm_speecht5`):
- `python test_inference_exhaustive.py --model_path ".\output_cuda_16gb\...\checkpoint-9100"`
