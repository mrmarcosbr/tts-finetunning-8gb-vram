# Descrição do pipeline (TCC): SpeechT5 e FastSpeech2 no LapsBM (Fala Brasil)

Este documento resume **apenas** as funcionalidades do repositório relacionadas ao **treino e inferência** dos modelos **SpeechT5** (Hugging Face + PEFT) e **FastSpeech2** (Coqui TTS / FastPitch com LoRA), utilizando o dataset **`falabrasil/lapsbm`** (LapsBM, Fala Brasil). Outros arquétipos (por exemplo Matcha, Glow-TTS, XTTS) e outros conjuntos de dados estão **fora do escopo** intencional desta descrição.

---

## 1. Objetivo científico (enquadramento do TCC)

O trabalho parte de modelos TTS **pré-treinados predominantemente em inglês** (SpeechT5 TTS; FastSpeech2/FastPitch no checkpoint LJSpeech) e investiga o **fine-tuning com LoRA** para aproximar a síntese ao **português brasileiro**, usando o LapsBM como fonte de fala real. A linha narrativa experimental é progressiva:

1. **Um locutor** — reduzir variabilidade e validar o encadeamento texto–áudio e o pipeline (por exemplo `num_speakers: 1` no perfil de dataset, quando aplicável).
2. **Dataset completo “simples”** — todas as amostras como **uma sentença por clipe** (`speecht5_train_spk_pool_size: 1` ou equivalente; sem fusão multi-frase).
3. **Formato misto com geração de amostras concatenadas** — modos **`replace`** e **`append`** no pré-processamento de treino SpeechT5 (ver secção 3), que alteram o conjunto de treino sem mudar a cardinalidade de validação da mesma forma.

Em síntese: demonstrar **desafios reais** de adaptação cruzada de língua com dados limitados, áudios curtos e condicionamento por locutor (SpeechT5 via x-vector).

---

## 2. Dataset LapsBM e preparação comum

- **Identificador**: `falabrasil/lapsbm` (Hugging Face Datasets).
- **Perfis no `config.yaml`**: `lapsbm_speecht5` e `lapsbm_fastspeech2`, ambos apontando para o mesmo repositório; o split usado nos perfis é configurável (`dataset_split`; no YAML atual figura `test` como split carregado — o código materializa uma lista e constrói um `Dataset`).
- **Identificação de locutor**: função `extract_speaker_id` — extrai ID a partir de metadados ou do caminho (padrões `LapsBM-…`, pastas `M-…` / `f001`, etc.).
- **Divisão treino / validação / teste por locutor** (`zero_shot_split`): parâmetros `train_speakers`, `val_speakers`, `test_speakers` fixam quantos locutores entram em cada conjunto; amostras são roteadas conforme o locutor. Isso permite avaliar **generalização a locutores não vistos no treino** quando há locutores suficientes.
- **Limite de amostras por locutor**: `num_samples_per_speaker` (0 = ilimitado no recorte aplicado).
- **Idioma da preparação**: `language` no perfil (padrão `pt` quando definido) — usado na tokenização / fonemização do FastSpeech2 e no fluxo do SpeechT5.
- **Cache em disco**: após o primeiro `map` com `prepare_item`, os tensores processados são guardados sob `datasets/cache_processado/…`, com sufixos que incluem versão de F0 para FastSpeech2 e metadados do “speaker pool” para SpeechT5, evitão recomputação.

**Problema central para o TCC (dados)**: o LapsBM, neste uso, é **relativamente pequeno** para adaptação forte de um modelo inglês, e muitos clipes são **muito curtos**. Isso afeta de forma desproporcional o ramo de **identidade de locutor** no SpeechT5: o codificador x-vector (SpeechBrain) precisa de trecho de áudio suficiente para um embedding estável; clipes curtos tendem a produzir representações **ruidosas ou fracas**, dificultando a consistência timbrística mesmo quando a perda de mel melhora.

---

## 3. SpeechT5: treino (LoRA e multi-frase)

### 3.1 Modelo e LoRA

- Base: `microsoft/speecht5_tts`; vocoder HiFi-GAN associado carregado para uso em inferência (no treino, o forward usa sobretudo mel-espectrograma como alvo).
- **LoRA** (configurável em `models.speecht5.lora`): rank `r`, `alpha`, `dropout`, `target_modules` (por exemplo projeções de atenção `q_proj`, `k_proj`, `v_proj`, `out_proj`). O perfil de dataset pode sobrescrever com `model_overrides`.
- **Resume**: suporte a `PeftModel.from_pretrained` ou `--resume_model_only` com verificação de compatibilidade do rank do adapter.

### 3.2 Preparação de cada amostra (`SpeechT5Handler.prepare_item`)

- Texto normalizado (`normalize_text`).
- **Mel log** como rótulo (librosa melspectrogram + log10, alinhado à taxa do modelo, tipicamente 16 kHz).
- **Embedding de locutor**: por defeito, **x-vector** via SpeechBrain (`speaker_encoder_id`), com normalização L2. Se a linha de treino trouxer `speaker_embedding_override` (ver multi-frase), pode usar-se vetor **médio** em modo `mean` em vez de reencodar só o áudio fundido.

### 3.3 Treino multi-frase (mesmo locutor): `replace` vs `append`

Implementação: `merge_speecht5_train_dataset_multi_sentence` e funções auxiliares (`_speecht5_build_multi_record`, `join_speecht5_training_texts`).

- **Parâmetros principais** (no bloco `training` do perfil `lapsbm_speecht5`):  
  `speecht5_train_spk_pool_size` (K), `speecht5_train_spk_pool_mode` (`concat` | `mean`), `speecht5_train_spk_pool_gap_sec`, `speecht5_train_spk_pool_mix_mode` (`replace` | `append`), `speecht5_train_spk_pool_multi_fraction`, `speecht5_train_spk_pool_append_multi_ratio`, `speecht5_train_spk_pool_seed`.

- **Modo `concat`**: funde áudios do mesmo locutor com silêncio (gap) entre clipes; o x-vector pode ser calculado sobre o **áudio concatenado** no `prepare_item` (alinhado ao mel-alvo fundido).

- **Modo `mean`**: calcula x-vector por clipe e usa **média L2-normalizada** como override — útil quando se quer estabilizar identidade a partir de vários curtos.

- **`replace`**: para cada linha, com probabilidade `multi_fraction`, substitui a amostra por uma versão **multi-frase**; mantém o tamanho do conjunto em número de linhas (salvo `multi_fraction` extremos).

- **`append`**: mantém todas as linhas **single** e **acrescenta** `round(n × append_multi_ratio)` linhas multi-frase, depois embaralha — aumenta o dataset efetivo de treino sem remover singles.

- **Validação**: o código deixa explícito que a fusão multi-frase aplica-se ao **treino**; a validação permanece **sentença a sentença**, para métricas comparáveis.

### 3.4 Detalhes de robustez no collator

- Padding de labels com **−100.0** nas regiões padded (máscara compatível com a loss do SpeechT5), evitando contaminar a loss com valores tipo −5 em batches com comprimentos mistos — relevante quando se misturam clipes curtos e fundidos longos.

### 3.5 Callbacks específicos

- **`SpeechT5GuidedAttentionLayerdropCallback`**: quando há *guided attention*, força o **layer drop efetivo do decoder a zero** (o módulo copia `layerdrop` na construção; alterar só o config não basta), evitando falhas no treino com `cross_attentions` vazios.

- **`PeftAdapterSaveCallback`**: em cada `checkpoint-*`, grava também **adapter PEFT** (`adapter_config` + pesos) para inferência leve.

---

## 4. FastSpeech2 (Coqui / FastPitch): treino (LoRA)

### 4.1 Modelo e loss

- Checkpoint base: `tts_models/en/ljspeech/fast_pitch` (config JSON + `model.pth` via `ModelManager`).
- O treino **não** delega só ao `forward` bruto: `FastSpeech2Wrapper` recompõe a **loss do Coqui** (mel, duração, **pitch**, energia, alinhamento MAS) via `get_criterion()`, condizente com o pré-treino LJSpeech.

### 4.2 Pitch e “isolar variáveis”

- **`pitch_loss_alpha`** em `config.yaml` (`models.fastspeech2`): escalonar a contribuição da parcela de pitch no objetivo. Colocar **0** remove o gradiente no ramo de pitch **na loss**; valores intermediários (comentários no YAML sugerem ~0,08–0,12) equilibram F0 vs estabilidade; valores altos podem **instabilizar** cedo.
- **Normalização de F0**: `prepare_item` calcula F0 com `AudioProcessor.compute_f0`, aplica **z-score** nos frames não nulos usando **média e desvio globais** estimados no **conjunto de treino** (`compute_f0_zscore_from_train_dataset`), gravos em `f0_stats.json` no cache. Isso alinha o alvo ao estilo **F0Dataset** do Coqui, diferente de usar Hz crus sem coerência com o checkpoint LJS.
- **`f0_input_mode`**: o modo `raw_hz` é desaconselhado/rejeitado para este checkpoint; mantém-se `zscore`.
- **Forward em FP32 no wrapper**: autocast desativado no forward do modelo Coqui para evitar NaN no alinhador MAS / Cython com FP16.

### 4.3 Alinhamento binário (warmup)

- **`binary_align_warmup_steps`**: callback `BinaryAlignWarmupCallback` faz rampa linear do peso efetivo da parcela de **binary alignment** no total da loss, para o MAS “assentar” antes de penalizar plenamente — comentários ligam isso à **briga** entre ramos de pitch e alinhamento nos primeiros passos.

### 4.4 Português sobre modelo inglês (fonemas)

- Substituição do fonemizador por **Gruut (pt)** com **wrapper** que mapeia alguns fonemas/nasalizações para símbolos **presentes no vocabulário IPA** do modelo base (evitar substituições globais destrutivas). Isto é indispensável para texto em PT, mas introduz **tensão** entre cobertura fonética do inglês e necessidades do português.

### 4.5 LoRA no FastSpeech2

- `target_modules` inclui embedding, projeções de atenção convolucionais, preditores de duração e pitch, etc. — reflete a hipótese de adaptar **lexicalização** (embeddings) e **ritmo/prosódia** sem retreinar o modelo inteiro.

---

## 5. Inferência (ferramentas relevantes no `test_inference_exhaustive.py`)

### 5.1 SpeechT5

- Carregamento do checkpoint do Trainer + adapter LoRA; síntese com vocoder HiFi-GAN.
- **Referência de locutor**: extração de embedding a partir de áudios (com opções de *pool* por vários clipes, mistura com vetor neutro, chunking de texto longo, silêncio entre chunks).
- **`--speecht5_zero_speaker_embedding`**: força embedding nulo — útil como **ablação** para separar efeito de identidade de locutor vs conteúdo linguístico.
- **`--dataset_reference_audios` / treino**: usar áudios do próprio LapsBM como referência de voz para métricas ou lotes de inferência.
- Pós-processamento opcional (high-pass, normalização, etc.) conforme argumentos.

### 5.2 FastSpeech2

- Pipeline **pt_base** vs **pt_lora** (base LJS vs adaptador treinado).
- Reutiliza a mesma lógica de fonemização PT (Gruut + mapeamentos) alinhada ao treino.
- Avisos no código para apontar a pasta correta de checkpoint (estrutura Hugging Face + pesos Coqui).

---

## 6. Síntese dos desafios (para discussão no TCC)

### 6.1 SpeechT5 + LapsBM

- **Dados escassos e clipes curtos** prejudicam **x-vectors** estáveis e, por arrastamento, a condição de locutor e a naturalidade.
- O treino **multi-frase** (`replace` / `append`, `concat` / `mean`) endereça diretamente a **falta de contexto acústico** por amostra, aproximando o comprimento efetivo das janelas de fala — alinhado à observação de que **o formato misto** melhora resultados auditivos em relação ao treino estritamente sentença-a-sentença, embora com **teto de qualidade** imposto pelo corpus.

### 6.2 FastSpeech2 + LapsBM

- Mesmo **isolando** o peso de pitch via `pitch_loss_alpha`, o modelo **continua acoplado** a preditores de duração, alinhamento MAS e energia; o checkpoint **inglês** e o vocabulário fonético mapeado não deixam de ser variáveis confundidoras.
- A extração de F0 em áudios **muito curtos** ou ruidosos introduce ruído nos alvos; o warmup do alinhamento binário e o z-score de F0 são mitigações **parciais**, não eliminação do problema fundamental de domínio (inglês → português, corpus pequeno).

### 6.3 Expectativa de qualidade e contribuição do trabalho

É **pouco realista** esperar síntese em português com **clareza e naturalidade de nível comercial** apenas com este fine-tuning LoRA sobre o LapsBM e bases inglesas, dadas as limitações de dados e de duração dos clipes. Contudo, a evidência empírica reunida no projeto aponta que o **treino misto** (single + amostras concatenadas por locutor, com modos `replace` e `append`) é **promissor**: produz saídas **audíveis** e **superiores** ao baselines mais ingenuos, ainda com **qualidade restrita** (artefatos, instabilidade prosódica, timbre incerto). Isto sugere que **um dataset maior e com frases mais longas e melhor gravadas** provavelmente **elevaria** o teto de qualidade sem mudar a arquitetura fundamental do pipeline descrito neste repositório.

---

## 7. Entrada principal do treino e inferência

- **Treino unificado**: `train_exhaustive.py` — seleção via `--dataset lapsbm_speecht5` ou `lapsbm_fastspeech2` e `--profile` de hardware.
- **Inferência batch**: `test_inference_exhaustive.py` — `--model_path` apontando para `checkpoint-*` com adapter, `--dataset` coerente com o perfil.

*(Scripts em `other_test_scripts/` e utilitários de exportação de embeddings servem de suporte experimental e não substituem o pipeline principal acima.)*
