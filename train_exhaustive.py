import os
import sys
import time
import warnings
from datetime import datetime
from dotenv import load_dotenv

# .env (ex.: HF_TOKEN para Hugging Face Hub) — uma chamada basta; não duplicar em baixo
load_dotenv()

# Evita spam de warning conhecido do SpeechBrain/Torchaudio durante o treino.
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Garante que o pacote *real* torch.distributed.tensor carrega cedo. Não usar stub aqui: um ModuleType
# falso substitui o pacote e o Adam/compile importa DTensor/FSdP e rebenta com ImportError.
try:
    importlib.import_module("torch.distributed.tensor")
except ImportError:
    pass  # instalação sem distributed tensor; single-GPU / treino ainda costuma seguir

# Fix para PyTorch 2.6 e resume do Trainer
# O PyTorch 2.6 restringe load() de estados do numpy salvos pelo transformers.
# Como confiamos nos nossos próprios checkpoints, vamos permitir o carregamento.
_original_load = torch.load
def safe_load(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

import json
import yaml
import random
import torchaudio
import unicodedata
import argparse
import copy
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from collections import Counter, defaultdict

from datasets import load_dataset, Audio, Dataset, load_from_disk
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments, SpeechT5HifiGan, TrainerCallback
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from speechbrain.inference.speaker import EncoderClassifier


def _device_for_speechbrain(device: str) -> str:
    # SpeechBrain faz parse de 'device' e, em algumas versões, 'cuda' sem índice → "expected 2, got 1"
    if device == "cuda" and torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return device


def _speecht5_zero_decoder_layerdrop_for_guided_attention(model: torch.nn.Module) -> None:
    """
    O SpeechT5Decoder copia decoder_layerdrop para self.layerdrop no __init__; o forward usa self.layerdrop.
    Só alterar config.decoder_layerdrop não desativa o LayerDrop — causa cross_attentions vazio e falha no loss.
    """
    inner = model
    if hasattr(inner, "get_base_model"):
        try:
            inner = inner.get_base_model()
        except Exception:
            pass
    cfg = getattr(inner, "config", None)
    if cfg is None or not getattr(cfg, "use_guided_attention_loss", False):
        return
    dec_mod = getattr(getattr(inner, "speecht5", None), "decoder", None)
    wrapped = getattr(dec_mod, "wrapped_decoder", None) if dec_mod is not None else None
    drop_cfg = float(getattr(cfg, "decoder_layerdrop", 0.0) or 0.0)
    drop_mod = float(getattr(wrapped, "layerdrop", 0.0) or 0.0) if wrapped is not None else 0.0
    if drop_cfg > 0.0 or drop_mod > 0.0:
        cfg.decoder_layerdrop = 0.0
        if wrapped is not None and hasattr(wrapped, "layerdrop"):
            wrapped.layerdrop = 0.0


def _get_peft_model_for_save(model) -> Optional[PeftModel]:
    """Encontra o PeftModel embutido (wrapper FS2, SpeechT5, XTTS.gpt, Glow-TTS .encoder, etc.)."""
    if isinstance(model, PeftModel):
        return model
    m = getattr(model, "model", None)
    if m is not None and isinstance(m, PeftModel):
        return m
    gpt = getattr(getattr(model, "xtts", None), "gpt", None)
    if gpt is not None and isinstance(gpt, PeftModel):
        return gpt
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "encoder") and isinstance(base.encoder, PeftModel):
        return base.encoder
    return None


def save_peft_adapter_for_inference(model, out_dir: str) -> bool:
    """
    Garante adapter_config.json + adapter_model.safetensors (ou .bin) em out_dir, formato p/ inferência.
    A backbone (ex.: model.pth LJS) continua a vir do repositório Coqui na inferência.
    """
    peft = _get_peft_model_for_save(model)
    if peft is None:
        return False
    os.makedirs(out_dir, exist_ok=True)
    try:
        peft.save_pretrained(out_dir, safe_serialization=True)
    except (ImportError, OSError, TypeError, ValueError):
        peft.save_pretrained(out_dir, safe_serialization=False)
    return True


class SpeechT5GuidedAttentionLayerdropCallback(TrainerCallback):
    """
    Garante decoder_layerdrop efetivo a 0 quando há guided attention (ver _speecht5_zero_decoder_layerdrop_*).

    on_train_begin: após Trainer.load_adapter no resume.
    on_step_begin: evita que layerdrop volte a >0 durante o treino (p.ex. referência antiga ou ordem de init);
    o forward do SpeechT5Decoder usa self.layerdrop copiado no __init__, não o config.
    """

    def __init__(self, model: torch.nn.Module):
        self._model = model

    def on_train_begin(self, args, state, control, **kwargs):
        _speecht5_zero_decoder_layerdrop_for_guided_attention(self._model)

    def on_step_begin(self, args, state, control, **kwargs):
        _speecht5_zero_decoder_layerdrop_for_guided_attention(self._model)


class PeftAdapterSaveCallback(TrainerCallback):
    """O Trainer grava o state do wrapper; este callback acrescenta o bundle PEFT em cada checkpoint-*."""

    def __init__(self, model: torch.nn.Module):
        self._model = model

    def on_save(self, args, state, control, **kwargs):
        sub = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(sub):
            return
        if save_peft_adapter_for_inference(self._model, sub):
            print(f"   💾 PEFT (adapter_config + adapter) → {sub}")


class LogMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            model = kwargs.get("model")
            
            # Tentar encontrar as métricas no modelo (pode estar dentro de um wrapper)
            m = model
            if hasattr(m, "module"): m = m.module
            
            if hasattr(m, "last_metrics"):
                # Mapeamento de possíveis nomes de chaves do Coqui
                key_map = {
                    "mel_loss": ["loss_spec", "mel_loss", "loss_mel"],
                    "duration_loss": ["loss_dur", "duration_loss", "loss_duration"],
                    "pitch_mse": ["pitch_mse", "loss_pitch", "pitch_loss"],
                    "f0_rmse_hz": "f0_rmse_hz",
                    "aligner_sharpness": ["loss_binary_alignment", "loss_aligner", "binary_loss"]
                }
                
                for display_name, possible_keys in key_map.items():
                    if isinstance(possible_keys, str):
                        if possible_keys in m.last_metrics:
                            val = m.last_metrics[possible_keys]
                            logs[display_name] = round(float(val.detach() if hasattr(val, 'detach') else val), 4)
                    else:
                        for pk in possible_keys:
                            if pk in m.last_metrics:
                                val = m.last_metrics[pk]
                                logs[display_name] = round(float(val.detach() if hasattr(val, 'detach') else val), 4)
                                break

# Configuração de encoding para Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Funções Utilitárias ---
def normalize_text(text):
    if not text: return ""
    text = text.lower().replace('\n', ' ').strip()
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def extract_speaker_id(item):
    for k in ['speaker_id', 'speaker', 'user_id', 'client_id']:
        if k in item and item[k] is not None:
            return str(item[k])
            
    # Para lidar com caminhos remotos (__url__) e locais (audio['path'])
    possible_paths = [
        item.get("__url__", ""),
        item.get("audio", {}).get("path", "") if isinstance(item.get("audio"), dict) else "",
        item.get("audio", "") if isinstance(item.get("audio"), str) else ""
    ]
    
    for p in possible_paths:
        p_str = str(p)
        if not p_str: continue
        
        # Padrão LapsBM-M-001 ou LapsBM-F-001
        if "LapsBM-" in p_str:
            filename = os.path.basename(p_str)
            if filename.startswith("LapsBM-"):
                return filename.split(".")[0].replace("LapsBM-", "")
        
        # Padrão de pastas lapsbm/M-001/... ou lapsbm/f001/...
        parts = p_str.replace("\\", "/").split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            if folder.lower() != "audio" and len(folder) > 1:
                return folder

    return "default"

def join_speecht5_training_texts(sentence_parts: List[str]) -> str:
    """Junta frases com '. ' (ponto + espaço) para simular encadeamento em falas longas."""
    bits: List[str] = []
    for p in sentence_parts:
        s = str(p).strip()
        if not s:
            continue
        s = s.rstrip().rstrip(".,; ")
        if s:
            bits.append(s)
    if not bits:
        return ""
    return ". ".join(bits) + "."


def _pick_train_spk_pool_indices(
    anchor_idx: int,
    peer_indices: List[int],
    k: int,
    rng: Optional[random.Random],
) -> List[int]:
    """Âncora incluída; restantes do mesmo locutor até k índices (únicos, ordenados por índice no dataset)."""
    peers = sorted(set(int(x) for x in peer_indices))
    if len(peers) <= k:
        return peers
    if anchor_idx not in peers:
        anchor_idx = peers[0]
    rest = [p for p in peers if p != anchor_idx]
    if rng is not None:
        rest = rest[:]
        rng.shuffle(rest)
    chosen = [int(anchor_idx)] + rest[: max(0, k - 1)]
    return sorted(chosen[:k])


def _speecht5_xvector_numpy(
    y_1d: np.ndarray,
    speaker_model: Any,
) -> np.ndarray:
    with torch.no_grad():
        emb = speaker_model.encode_batch(torch.tensor(np.asarray(y_1d, dtype=np.float32).reshape(-1)))
        v = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def _speecht5_train_row_as_single_record(row: Dict[str, Any], target_sr: int) -> Dict[str, Any]:
    """Uma sentença, um áudio; prepare_item calcula o x-vector no encode_batch (sem override)."""
    au = row["audio"]
    y = np.asarray(au["array"], dtype=np.float32).reshape(-1)
    t = row.get("text") or row.get("txt") or ""
    return {
        "text": str(t),
        "audio": {"array": y.astype(np.float32), "sampling_rate": int(target_sr)},
        "speaker_embedding_override": None,
    }


def _speecht5_build_multi_record(
    train_dataset: Dataset,
    anchor_idx: int,
    spk_to_idxs: Dict[str, List[int]],
    pool_size: int,
    rng_pool: Optional[random.Random],
    pool_mode: str,
    gap: np.ndarray,
    target_sr: int,
    speaker_model: Any,
) -> Dict[str, Any]:
    """Uma linha de treino multi: âncora + até pool_size−1 pares do mesmo locutor."""
    row_i = train_dataset[anchor_idx]
    spk = extract_speaker_id(row_i)
    peers = spk_to_idxs.get(spk, [anchor_idx])
    pool_idx = _pick_train_spk_pool_indices(
        anchor_idx,
        peers,
        pool_size,
        rng_pool,
    )
    texts: List[str] = []
    audio_pieces: List[np.ndarray] = []
    seg_vecs: List[np.ndarray] = []
    for j in pool_idx:
        row = train_dataset[j]
        t = row.get("text") or row.get("txt") or ""
        texts.append(str(t))
        au = row["audio"]
        y = np.asarray(au["array"], dtype=np.float32).reshape(-1)
        audio_pieces.append(y)
        if pool_mode == "mean":
            if speaker_model is None:
                raise RuntimeError(
                    "speecht5_train_spk_pool_mode=mean requer speaker_encoder_id e modelo SpeechBrain carregado."
                )
            seg_vecs.append(_speecht5_xvector_numpy(y, speaker_model))

    merged_text = join_speecht5_training_texts(texts)
    merged_parts: List[np.ndarray] = []
    for pi, arr in enumerate(audio_pieces):
        merged_parts.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        if pi < len(audio_pieces) - 1 and gap.size > 0:
            merged_parts.append(gap.copy())
    merged_audio = np.concatenate(merged_parts) if merged_parts else np.zeros(0, dtype=np.float32)

    override: Optional[np.ndarray] = None
    if pool_mode == "mean" and seg_vecs:
        st = np.stack(seg_vecs, axis=0)
        mvec = st.mean(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(mvec))
        if nrm > 1e-12:
            mvec = mvec / nrm
        override = mvec

    return {
        "text": merged_text,
        "audio": {"array": merged_audio.astype(np.float32), "sampling_rate": int(target_sr)},
        "speaker_embedding_override": override,
    }


def merge_speecht5_train_dataset_multi_sentence(
    train_dataset: Dataset,
    *,
    pool_size: int,
    pool_mode: str,
    gap_sec: float,
    seed: Optional[int],
    target_sr: int,
    speaker_model: Any,
    multi_sample_fraction: float = 1.0,
    mix_mode: str = "replace",
    append_multi_ratio: float = 1.0,
) -> Dataset:
    """
    Agrupa amostras multi-frase (mesmo locutor), texto com '. ' e áudio com gap.

    mix_mode 'replace' (padrão): em cada linha i, com probabilidade multi_sample_fraction
    substitui por multi; senão single. mf=0.3 ⇒ ~30% multi + ~70% single; n linhas no total.

    mix_mode 'append': mantém as n linhas originais como single e acrescenta k linhas multi,
    com k = round(n * append_multi_ratio). Ex.: n=560 e append_multi_ratio=1.0 ⇒ 560+560=1120;
    ratio=0.5 ⇒ 560+280=840. Âncoras das k multis: t % n para t=0..k−1.

    pool_mode 'concat' ⇒ x-vector no prepare_item sobre o áudio fundido; 'mean' ⇒ média L2
    dos x-vectors por clipe em speaker_embedding_override.
    """
    n = len(train_dataset)
    if pool_size <= 1 or n == 0:
        return train_dataset

    mode = (pool_mode or "concat").strip().lower()
    if mode not in ("mean", "concat"):
        raise ValueError("speecht5_train_spk_pool_mode deve ser 'mean' ou 'concat'")

    spk_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for idx in range(n):
        row = train_dataset[idx]
        spk = extract_speaker_id(row)
        spk_to_idxs[spk].append(idx)
    for spk in spk_to_idxs:
        spk_to_idxs[spk] = sorted(spk_to_idxs[spk])

    gap_samples = int(max(0.0, float(gap_sec)) * target_sr)
    gap = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else np.zeros(0, dtype=np.float32)

    mm = (mix_mode or "replace").strip().lower()
    if mm not in ("replace", "append"):
        raise ValueError("speecht5_train_spk_pool_mix_mode deve ser 'replace' ou 'append'")

    rng_pool: Optional[random.Random] = random.Random(int(seed)) if seed is not None else None

    if mm == "append":
        ar = max(0.0, float(append_multi_ratio))
        k = int(round(n * ar))
        singles = [_speecht5_train_row_as_single_record(train_dataset[i], target_sr) for i in range(n)]
        if k <= 0:
            print(
                f"   🔀 SpeechT5 append multi: ratio={ar:g} ⇒ 0 extras; dataset inalterado ({n} amostras)."
            )
            return train_dataset
        multi_rows = [
            _speecht5_build_multi_record(
                train_dataset,
                t % n,
                spk_to_idxs,
                pool_size,
                rng_pool,
                mode,
                gap,
                target_sr,
                speaker_model,
            )
            for t in range(k)
        ]
        new_rows = singles + multi_rows
        if seed is not None:
            rng_shuf = random.Random(int(seed) + 100003)
            rng_shuf.shuffle(new_rows)
        else:
            random.shuffle(new_rows)
        print(
            f"   🔀 SpeechT5 multi-frase APPEND: {n} single + {k} multi → {len(new_rows)} amostras | "
            f"K={pool_size} modo={mode} | gap={gap_sec}s | append_multi_ratio={ar:g} (k=round(n×ratio))."
        )
        out = Dataset.from_list(new_rows)
        return out.cast_column("audio", Audio(sampling_rate=int(target_sr)))

    mf = max(0.0, min(1.0, float(multi_sample_fraction)))
    if mf <= 0.0:
        return train_dataset

    rng_mix: Optional[random.Random] = None
    if mf < 1.0:
        rng_mix = random.Random(int(seed)) if seed is not None else random.Random()

    new_rows: List[Dict[str, Any]] = []
    n_multi = 0
    n_single = 0
    for i in range(n):
        use_multi = mf >= 1.0 or (rng_mix is not None and rng_mix.random() < mf)
        if not use_multi:
            new_rows.append(_speecht5_train_row_as_single_record(train_dataset[i], target_sr))
            n_single += 1
            continue

        n_multi += 1
        new_rows.append(
            _speecht5_build_multi_record(
                train_dataset,
                i,
                spk_to_idxs,
                pool_size,
                rng_pool if mf >= 1.0 else rng_mix,
                mode,
                gap,
                target_sr,
                speaker_model,
            )
        )

    out = Dataset.from_list(new_rows)
    mix_note = (
        f" | mistura replace: {100.0 * mf:.1f}% multi (≈{n_multi} linhas) + {n_single} single"
        if mf < 1.0
        else ""
    )
    print(
        f"   🔀 SpeechT5 multi-frase REPLACE: {n} → {len(out)} amostras | K={pool_size} modo={mode} | "
        f"gap={gap_sec}s entre clipes{mix_note} | texto multi ligado com '. '."
    )
    return out.cast_column("audio", Audio(sampling_rate=int(target_sr)))


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Sufixo de pasta de cache: incrementar se mudar F0 / mel em prepare_item (FastPitch)
FS2_F0_DATASET_VERSION = "f0zscore_v1"

def detect_hardware_profile(full_cfg):
    """Tenta identificar o melhor perfil de hardware automaticamente."""
    print("🔍 Autodetectando hardware...")
    
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   🖥️  GPU NVIDIA Detectada: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB VRAM)")
        if total_vram >= 14.0: return "cuda_16gb"
        return "cuda_8gb"
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("   🍎 Apple Silicon Detectado (MPS Backend)")
        return "macbook"
    
    print("   ⚠️ Nenhuma GPU de alta performance detectada. Usando CPU.")
    return "cpu"


class FS2DataCollator:
    """Collator em escopo de módulo para suportar DataLoader multiprocess no Windows."""
    def __call__(self, features):
        from torch.nn.utils.rnn import pad_sequence
        x = pad_sequence([f["x"] for f in features], batch_first=True)
        y = pad_sequence([f["y"] for f in features], batch_first=True)
        pitch = pad_sequence([f["pitch"].squeeze(0) for f in features], batch_first=True).unsqueeze(1)
        x_lengths = torch.stack([f["x_lengths"] for f in features])
        y_lengths = torch.stack([f["y_lengths"] for f in features])

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "pitch": pitch
        }

# ==============================================================================
# MODEL HANDLERS (Um para cada arquitetura)
# ==============================================================================

class ModelHandler:
    """Base para suporte a diferentes arquiteturas de TTS."""
    def __init__(self, model_cfg: Dict, device: str):
        self.model_cfg = model_cfg
        self.device = device
        self.processor = None
        self.model = None

    def load(self, resume_from=None, resume_model_only=False):
        raise NotImplementedError()

    def prepare_item(self, item, processor, speaker_model):
        raise NotImplementedError()

    def get_collator(self):
        raise NotImplementedError()


@dataclass
class SpeechT5TTSDataCollatorWithPadding:
    """
    Tem de estar ao nível do módulo (não como classe local) para o DataLoader com
    dataloader_num_workers>0 no Windows conseguir fazer pickle do collate_fn (spawn).
    """

    processor: Any

    def __call__(self, features):
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        labels = [feature["labels"] for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]
        batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")
        max_len = max([l.shape[0] for l in labels])
        if max_len % 2 != 0:
            max_len += 1
        padded_labels = [
            # SpeechT5SpectrogramLoss (HF) usa padding_mask = labels != -100.0; outro valor (−5) contamina
            # o L1/BCE em regiões padeadas — com batches mistos curto+longo (mix multi/single) piora muito.
            torch.nn.functional.pad(l, (0, 0, 0, max_len - l.shape[0]), value=-100.0) for l in labels
        ]
        batch["labels"] = torch.stack(padded_labels)
        batch["speaker_embeddings"] = torch.stack(
            [
                s.clone().detach() if isinstance(s, torch.Tensor) else torch.tensor(s)
                for s in speaker_features
            ]
        )
        return batch


class SpeechT5Handler(ModelHandler):
    def load(self, resume_from=None, resume_model_only=False):
        print(f"📥 Carregando SpeechT5: {self.model_cfg['id']}")
        self.processor = SpeechT5Processor.from_pretrained(self.model_cfg['id'])
        # Treino (mel + loss) não usa o vocoder no forward; carregar em GPU só desperdiça VRAM (~centenas MB–1GB+).
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_cfg["vocoder_id"]) #.to(self.device)
        self.vocoder.eval()
        self.vocoder = self.vocoder.to("cpu")
        
        SpeechT5ForTextToSpeech._keys_to_ignore_on_load_unexpected = [r'.*encode_positions\.pe']
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_cfg['id'])
        
        # Aplicar LoRA ou Carregar existente
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none", task_type=None
        )
        
        has_adapter = resume_from and os.path.exists(os.path.join(resume_from, "adapter_config.json"))
        ckpt_r: Optional[int] = None
        if has_adapter:
            cfg_path = os.path.join(resume_from, "adapter_config.json")
            try:
                with open(cfg_path, "r", encoding="utf-8") as af:
                    acfg = json.load(af)
                ckpt_r = acfg.get("r")
                if ckpt_r is None and isinstance(acfg.get("peft_config"), dict):
                    ckpt_r = acfg["peft_config"].get("r")
                if ckpt_r is not None:
                    ckpt_r = int(ckpt_r)
            except Exception as e:
                print(f"   ⚠️ Não foi possível ler r do adapter checkpoint: {e}")
        desired_r = int(lora_cfg["r"])
        if has_adapter and ckpt_r is not None:
            if resume_model_only and ckpt_r != desired_r:
                print(
                    f"   ⚡ Checkpoint LoRA r={ckpt_r}, config atual r={desired_r}. "
                    f"--resume_model_only: tensores com shape incompatível serão ignorados (LoRA efectivamente novo nesse rank)."
                )
            elif not resume_model_only and ckpt_r != desired_r:
                raise ValueError(
                    f"Checkpoint LoRA tem r={ckpt_r}; o config atual pede r={desired_r}. "
                    f"Tensores não são compatíveis para PeftModel.from_pretrained. Opções: (1) repor "
                    f"models.speecht5.lora/model_overrides ao mesmo rank do checkpoint; ou (2) treinar novo rank com "
                    f"--resume_model_only (pesos compatíveis carregados; novos filtros ficam aleatórios)."
                )
        if has_adapter and not resume_model_only:
            print(f"🔄 Carregando adaptadores LoRA existentes de: {resume_from}")
            self.model = PeftModel.from_pretrained(self.model, resume_from, is_trainable=True)
        else:
            self.model = get_peft_model(self.model, peft_config)
            if resume_from and resume_model_only and has_adapter:
                wpath = os.path.join(resume_from, "adapter_model.safetensors")
                if not os.path.exists(wpath):
                    wpath = os.path.join(resume_from, "pytorch_model.bin")
                if not os.path.exists(wpath):
                    print(
                        f"   ⚠️ --resume_model_only: nenhum adapter_model.safetensors/pytorch_model.bin em {resume_from}; "
                        f"iniciando LoRA novo a partir de zero."
                    )
                else:
                    if wpath.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        st = load_file(wpath)
                    else:
                        st = torch.load(wpath, map_location="cpu")
                    miss, uexp = self.model.load_state_dict(st, strict=False)
                    print(
                        f"   🔀 --resume_model_only: load_state_dict(strict=False) — "
                        f"{len(st)} tensores no ficheiro; missing={len(miss)} unexpected={len(uexp)} (normal se alterou LoRA)."
                    )
        # Garantir contiguidade em parâmetros e BUFFERS (essencial p/ pos_encoder.pe)
        for p in self.model.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
        for b in self.model.buffers():
            if not b.is_contiguous(): b.data = b.data.contiguous()

        _m = self.model
        _inner = _m.get_base_model() if hasattr(_m, "get_base_model") else _m
        drop_cfg_before = float(getattr(getattr(_inner, "config", None), "decoder_layerdrop", 0.0) or 0.0)
        _wd = getattr(
            getattr(getattr(_inner, "speecht5", None), "decoder", None),
            "wrapped_decoder",
            None,
        )
        drop_mod_before = float(getattr(_wd, "layerdrop", 0.0) or 0.0) if _wd is not None else 0.0
        _speecht5_zero_decoder_layerdrop_for_guided_attention(self.model)
        if getattr(getattr(_inner, "config", None), "use_guided_attention_loss", False) and (
            drop_cfg_before > 0 or drop_mod_before > 0
        ):
            print(
                f"   ℹ️ Guided attention: decoder_layerdrop config={drop_cfg_before} módulo.layerdrop={drop_mod_before} → 0"
            )

        return self.model, self.processor

    def prepare_item(self, batch, processor, speaker_model, language=None):
        audio = batch["audio"]
        override = batch.pop("speaker_embedding_override", None)
        clean_text = normalize_text(batch.get("text", "dummy"))
        batch["input_ids"] = processor(text=clean_text, return_tensors="pt").input_ids[0]
        y = np.array(audio["array"])
        mel_sr = int(self.model_cfg.get("sampling_rate", 16000))
        
        # SpeechT5 espera mel-spectrogram como labels (alinhado a target_sr do cast_column)
        mel = librosa.feature.melspectrogram(
            y=y, sr=mel_sr, n_fft=1024, hop_length=256, n_mels=80, fmin=80, fmax=7600
        )
        log_mel = np.log10(np.clip(mel, 1e-5, None)).T
        batch["labels"] = torch.tensor(log_mel)

        used_override = False
        if override is not None and speaker_model is not None:
            ov = np.asarray(override, dtype=np.float32).reshape(-1)
            if ov.size == 512:
                batch["speaker_embeddings"] = ov
                used_override = True
        if not used_override:
            if speaker_model is None:
                raise RuntimeError("SpeechT5 prepare_item: speaker_model ausente e sem speaker_embedding_override.")
            with torch.no_grad():
                emb = speaker_model.encode_batch(torch.tensor(y))
                batch["speaker_embeddings"] = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
        return batch

    def get_collator(self):
        return SpeechT5TTSDataCollatorWithPadding(processor=self.processor)

# --- Placeholders para futuros Handlers ---
class F5CFMWrapper(torch.nn.Module):
    def __init__(self, cfm):
        super().__init__()
        self.cfm = cfm
        
    def forward(self, mel=None, text=None, **kwargs):
        # O CFM.forward retorna (loss, cond, pred)
        loss, _, _ = self.cfm(mel, text)
        return {"loss": loss}

class F5TTSHandler(ModelHandler):
    def load(self, resume_from=None, resume_model_only=False):
        try:
            from f5_tts.model import DiT, CFM
            from f5_tts.model.utils import get_tokenizer
        except ImportError:
            print("❌ Erro: Biblioteca 'f5-tts' não encontrada. Instale com 'pip install f5-tts'")
            sys.exit(1)

        print(f"📥 Carregando F5-TTS (DiT Backbone): {self.model_cfg['id']}")
        
        import importlib.resources
        import f5_tts
        try:
            vocab_path = str(importlib.resources.files("f5_tts").joinpath("infer/examples/vocab.txt"))
        except:
            vocab_path = os.path.join(os.path.dirname(f5_tts.__file__), "infer", "examples", "vocab.txt")
            
        self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_path, "custom")
        
        # DiT Architecture
        # Utilizamos self.vocab_size ao invés do valor embutido fixo (256)
        self.dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_num_embeds=self.vocab_size, mel_dim=80)
        
        # Aplicar LoRA no DiT
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none"
        )
        from peft import get_peft_model
        self.dit = get_peft_model(self.dit, peft_config)
        for p in self.dit.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
        self.dit.to(self.device)
        
        # CFM Wrapper para HF Trainer
        cfm = CFM(transformer=self.dit, sigma=0.0).to(self.device)
        self.model = F5CFMWrapper(cfm)
        
        return self.model, self.vocab_char_map

    def prepare_item(self, batch, processor, speaker_model, language=None):
        # Preparação específica para o treino do CFM
        from f5_tts.model.utils import convert_char_to_pinyin
        audio = batch["audio"]
        y = np.array(audio["array"])
        
        # Extrair Mel-Spectrogram (80 bins conforme F5-TTS)
        mel = librosa.feature.melspectrogram(y=y, sr=24000, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = torch.tensor(np.log10(np.clip(mel, 1e-5, None)).T)
        
        batch["mel"] = log_mel
        
        text = batch.get("text", "")
        # Converter texto para pinyin e depois para IDs
        pinyin_list = convert_char_to_pinyin([text])[0]
        text_ids = [self.vocab_char_map.get(c, 0) for c in pinyin_list]
        batch["text_ids"] = torch.tensor(text_ids, dtype=torch.long)
        
        return batch

    def get_collator(self):
        @dataclass
        class F5DataCollator:
            def __call__(self, features):
                # Collation para o CFM (Padding de mel e text_ids)
                mels = [f["mel"] for f in features]
                texts = [f["text_ids"] for f in features]
                
                # Padding manual simples (truncagem ou pad conforme necessário)
                from torch.nn.utils.rnn import pad_sequence
                batch_mels = pad_sequence(mels, batch_first=True, padding_value=-5.0)
                batch_texts = pad_sequence(texts, batch_first=True, padding_value=0)
                
                return {"mel": batch_mels, "text": batch_texts}
        return F5DataCollator()

class XTTSWrapper(torch.nn.Module):
    def __init__(self, xtts):
        super().__init__()
        self.xtts = xtts
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, input_ids=None, text_len=None, audio_codes=None, wav_len=None, gpt_cond_latent=None, **kwargs):
        # O GPT do XTTS v2 preve os audio_codes (tokens auditivos)
        # cond_latents so os latents de voz (gpt_cond_latent)
        # Retorna logits: [batch, seq_len, vocab_size]
        logits = self.xtts.gpt(
            text_inputs=input_ids,
            text_lengths=text_len,
            audio_codes=audio_codes,
            wav_lengths=wav_len,
            cond_latents=gpt_cond_latent
        )
        
        # Shift para treino autoregressivo: prever o prximo token
        # Os labels so os prprios audio_codes
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = audio_codes[..., 1:].contiguous()
        
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}

class XTTSHandler(ModelHandler):
    def load(self, resume_from=None, resume_model_only=False):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv.")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2: {self.model_cfg['id']}")
        
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.utils.manage import ModelManager
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv.")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2 via ModelManager...")
        
        # O XTTS v2 precisa de pesos reais para inicializar o GPT
        manager = ModelManager()
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        res = manager.download_model(model_name)
        
        # O ModelManager pode retornar uma tupla (path, config, etc) ou apenas o path
        model_path = res[0] if isinstance(res, (list, tuple)) else res
        
        config_path = os.path.join(model_path, "config.json")
        config = XttsConfig()
        config.load_json(config_path)
        
        # Monkey-patch para contornar restrições de segurança do PyTorch 2.6+
        # O XTTS usa muitas classes customizadas no pickle que seriam bloqueadas pelo novo padrão weights_only=True
        orig_load = torch.load
        def trusted_load(*args, **kwargs):
            kwargs.pop("weights_only", None) # Remove se existir
            return orig_load(*args, weights_only=False, **kwargs)
        
        torch.load = trusted_load
        try:
            self.xtts = Xtts.init_from_config(config)
            self.xtts.load_checkpoint(config, checkpoint_dir=model_path)
        finally:
            torch.load = orig_load
        
        # Aplicar LoRA no GPT parte do XTTS
        if self.xtts.gpt is not None:
            lora_cfg = self.model_cfg['lora']
            peft_config = LoraConfig(
                r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
                target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
                bias="none",
                fan_in_fan_out=True # Necessário para as camadas Conv1D do GPT-2 do XTTS
            )
            # Aplicar LoRA especificamente ao sub-módulo GPT
            self.xtts.gpt = get_peft_model(self.xtts.gpt, peft_config)
            for p in self.xtts.gpt.parameters():
                if not p.is_contiguous(): p.data = p.data.contiguous()
            print(f"✅ LoRA aplicado ao GPT do XTTS v2.")
        
        self.xtts.to(self.device)
        self.model = XTTSWrapper(self.xtts)
        # O XTTS usa seu próprio tokenizer e processador de áudio incorporado
        return self.model, self.xtts.tokenizer

    def prepare_item(self, batch, processor, speaker_model, language="pt"):
        import tempfile
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 24000)
        text = normalize_text(batch.get("text", ""))
        
        # O XTTS v2 oficial da Coqui requer caminhos de arquivos para extrair condicionamento
        # Usamos arquivos temporários para compatibilidade total com a API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            temp_path = f.name
            
        try:
            # Extrair latents de condicionamento (voz)
            gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                audio_path=[temp_path] 
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Extrair Audio Codes (tokens VQ) do áudio original
        # Isso é o que o GPT tentará prever no treinamento
        audio_codes = self.xtts.gpt.encode_audio(torch.tensor(y).unsqueeze(0).to(self.device))
        
        input_ids = self.xtts.tokenizer.encode(text, lang=language)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "text_len": torch.tensor(len(input_ids)),
            "gpt_cond_latent": gpt_cond_latent.squeeze(0),
            "speaker_embedding": speaker_embedding.squeeze(0),
            "audio_codes": audio_codes.squeeze(0),
            "wav_len": torch.tensor(audio_codes.shape[-1]),
            "audio": torch.tensor(y)
        }

    def get_collator(self):
        @dataclass
        class XTTSDataCollator:
            def __call__(self, features):
                from torch.nn.utils.rnn import pad_sequence
                input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=0)
                text_len = torch.tensor([f["text_len"] for f in features])
                gpt_cond_latent = torch.stack([f["gpt_cond_latent"] for f in features])
                # Corrigido: audio_codes precisam de padding e ter o valor -1 para ignorar na perda
                audio_codes = pad_sequence([f["audio_codes"] for f in features], batch_first=True, padding_value=-1)
                wav_len = torch.tensor([f["wav_len"] for f in features])
                
                return {
                    "input_ids": input_ids,
                    "text_len": text_len,
                    "gpt_cond_latent": gpt_cond_latent,
                    "audio_codes": audio_codes,
                    "wav_len": wav_len
                }
        return XTTSDataCollator()

class FastSpeech2Wrapper(torch.nn.Module):
    # Trainer._issue_warnings_after_load (resume) acede a isto em PreTrainedModel; wrapper puro não tem.
    _keys_to_ignore_on_save = None

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        
        # Recuperar a função de loss oficial do Coqui para treinar Pitch/Duration/Energy
        base_model = getattr(self.model, "base_model", self.model)
        if hasattr(base_model, "model"): base_model = base_model.model
        self.criterion = base_model.get_criterion()
        self._pitch_loss_alpha = 0.0
        m = self.model
        for _ in range(8):
            if m is not None and hasattr(m, "config") and m.config is not None and hasattr(m.config, "pitch_loss_alpha"):
                self._pitch_loss_alpha = float(m.config.pitch_loss_alpha)
                break
            m = getattr(m, "base_model", None) or getattr(m, "model", None)
        # Para f0_rmse_hz: mesmos mean/std (Hz) usados no z-score dos alvos; definir com set_f0_denorm após f0_stats
        self.f0_denorm_mean: Optional[float] = None
        self.f0_denorm_std: Optional[float] = None
        # 0.0 = sem contrib. binária no loss total; 1.0 = pleno. Atualizado por BinaryAlignWarmupCallback
        self._binary_warm: float = 1.0

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                self.model.gradient_checkpointing_enable()
            print("OK: gradient_checkpointing repassado ao PeftModel/base (se suportado).")
        else:
            print("Aviso: gradient_checkpointing_enable ignorado — base sem suporte (ForwardTTS não-HF puro).")

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def is_gradient_checkpointing(self):
        if hasattr(self.model, "is_gradient_checkpointing"):
            return bool(self.model.is_gradient_checkpointing())
        return False

    def set_f0_denorm(self, mean: Optional[float], std: Optional[float]) -> None:
        """Estatísticas de treino (Hz) para reportar RMSE de F0 em Hz (não em z-score)."""
        self.f0_denorm_mean = None if mean is None else float(mean)
        self.f0_denorm_std = None if std is None else max(float(std), 1.0)
        
    def forward(self, x=None, x_lengths=None, y=None, y_lengths=None, **kwargs):
        pitch = kwargs.pop('pitch', None)
        
        # Cython MAS Aligner e algumas operações do Coqui quebram/geram NaN com FP16
        # Desabilitamos o autocast especificamente para o forward do modelo
        with torch.amp.autocast('cuda', enabled=False):
            if y is not None: y = y.float()
            if pitch is not None: pitch = pitch.float()
            
            outputs = self.model(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                pitch=pitch,
                **kwargs
            )
            
            if isinstance(outputs, dict) and "loss" in outputs:
                return outputs
                
            # Calcular a Loss Completa do FastPitch (inclui Mel, Duration, Pitch)
            # Isso é crucial para o LoRA aprender a entonação e ritmo em português!
            loss_dict = self.criterion(
                decoder_output=outputs["model_outputs"],
                decoder_target=y,
                decoder_output_lens=y_lengths,
                dur_output=outputs["durations_log"],
                dur_target=outputs.get("o_alignment_dur", None),
                pitch_output=outputs.get("pitch_avg", None),
                pitch_target=outputs.get("pitch_avg_gt", None),
                energy_output=outputs.get("energy_avg", None),
                energy_target=outputs.get("energy_avg_gt", None),
                input_lens=x_lengths,
                alignment_logprob=outputs.get("alignment_logprob", None),
                alignment_soft=outputs["alignment_soft"],
                alignment_hard=outputs["alignment_mas"],
                binary_loss_weight=1.0,
            )
            # Ajuste de warmup: em Coqui, return_dict['loss'] já inclui o binary completo; loss_binary_alignment (com weight=1)
            # coincide com a parcela a subtrair/reintroduzir. Ver config binary_align_warmup_steps.
            if (
                "loss_binary_alignment" in loss_dict
                and self._binary_warm < 1.0 - 1e-8
            ):
                b = loss_dict["loss_binary_alignment"]
                if hasattr(b, "float"):
                    b = b.float()
                loss_dict["loss"] = loss_dict["loss"] - (1.0 - self._binary_warm) * b
            
            # Depuração: imprimir as chaves na primeira vez para sabermos os nomes exatos
            if not hasattr(self, "_keys_printed"):
                keys = list(loss_dict.keys())
                print(f"\n🔍 [DEBUG] Métricas detectadas no Coqui: {keys}")
                self._keys_printed = True

            # MSE de pitch (espaço z-score de F0, não Hz) — só se pitch_loss_alpha>0
            if "loss_pitch" in loss_dict and self._pitch_loss_alpha > 0:
                lp = loss_dict["loss_pitch"]
                v = float(lp.detach() if hasattr(lp, "detach") else lp) / (self._pitch_loss_alpha + 1e-12)
                loss_dict["pitch_mse"] = v
                loss_dict["pitch_rmse_zscore"] = v ** 0.5  # raiz do MSE no espaço z-score, não Hz

            # RMSE de F0 em Hz (só com mean/std e tensores de pitch; máscara: alvo com F0 ativo, estilo TTS)
            if (
                self.f0_denorm_mean is not None
                and self.f0_denorm_std is not None
                and outputs.get("pitch_avg") is not None
                and outputs.get("pitch_avg_gt") is not None
            ):
                m0, s0 = self.f0_denorm_mean, self.f0_denorm_std
                gto = outputs["pitch_avg_gt"]
                prz = outputs["pitch_avg"]
                with torch.no_grad():
                    g_hz = gto * s0 + m0
                    p_hz = prz * s0 + m0
                    mask = gto.abs() > 1e-5
                    if mask.any():
                        d = (g_hz[mask] - p_hz[mask]).float()
                        loss_dict["f0_rmse_hz"] = float((d * d).mean().sqrt().item())
                    else:
                        loss_dict["f0_rmse_hz"] = 0.0

            # Salvar no modelo para o Callback de log acessar
            if "loss_binary_alignment" in loss_dict:
                loss_dict["aligner_sharpness"] = loss_dict["loss_binary_alignment"]
            elif "binary_loss" in loss_dict:
                loss_dict["aligner_sharpness"] = loss_dict["binary_loss"]
                
            self.last_metrics = loss_dict

            return {"loss": loss_dict["loss"], "model_outputs": outputs["model_outputs"]}

class FastSpeech2Handler(ModelHandler):
    def load(self, resume_from=None, resume_model_only=False):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.forward_tts import ForwardTTS as FastSpeech2
            from TTS.tts.configs.fastspeech2_config import Fastspeech2Config as FastSpeech2Config
            from TTS.utils.audio import AudioProcessor
            from peft import get_peft_model, LoraConfig, PeftModel
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv_fs2.")
            sys.exit(1)

        print(f"📥 Carregando FastSpeech 2: {self.model_cfg['id']}")
        
        manager = ModelManager()
        model_path = manager.download_model(self.model_cfg['id'])
        if isinstance(model_path, tuple): model_path = model_path[0]
        
        # Se model_path for um arquivo (ex: model.pth), pegar o diretório pai
        if os.path.isfile(model_path):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path

        config_path = os.path.join(model_dir, "config.json")
        config = FastSpeech2Config()
        config.load_json(config_path)
        if "pitch_loss_alpha" in self.model_cfg and self.model_cfg["pitch_loss_alpha"] is not None:
            config.pitch_loss_alpha = float(self.model_cfg["pitch_loss_alpha"])
        
        # Inicializar o AudioProcessor oficial com a config do modelo
        self.ap = AudioProcessor.init_from_config(config)
        self.f0_norm_mean: Optional[float] = None
        self.f0_norm_std: Optional[float] = None
        
        # Ajuste de segurança para detecção de Pitch (F0)
        # Valores abaixo de 50 Hz causam erro no librosa.pyin com janelas de 1024
        self.ap.pitch_fmin = 50
        self.ap.pitch_fmax = 1000
        
        self.model_fs2 = FastSpeech2.init_from_config(config)
        self.model_fs2.load_checkpoint(config, checkpoint_path=os.path.join(model_dir, "model.pth"))
        
        # Aplicar LoRA ou Carregar existente
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['alpha'],
            target_modules=lora_cfg['target_modules'],
            lora_dropout=lora_cfg['dropout'],
            bias="none",
            task_type=None
        )

        if resume_from and (os.path.exists(os.path.join(resume_from, "adapter_config.json")) or os.path.exists(os.path.join(resume_from, "model.safetensors"))):
            print(f"🔄 Carregando adaptadores LoRA existentes de: {resume_from}")
            # Se for um diretório de modelo final (com model.safetensors mas sem adapter_config.json pela estrutura do wrapper)
            # precisamos inicializar o LoRA primeiro e depois carregar os pesos
            self.model_fs2 = get_peft_model(self.model_fs2, peft_config)
            
            # Tentar carregar pesos (lidando com o prefixo 'model.' se necessário)
            weights_path = os.path.join(resume_from, "model.safetensors")
            if not os.path.exists(weights_path): weights_path = os.path.join(resume_from, "pytorch_model.bin")
            
            if os.path.exists(weights_path):
                if weights_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    sd = load_file(weights_path)
                else:
                    sd = torch.load(weights_path, map_location="cpu")
                
                # Ajustar prefixos do wrapper
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("model."): new_sd[k[6:]] = v
                    else: new_sd[k] = v
                self.model_fs2.load_state_dict(new_sd, strict=False)
        else:
            self.model_fs2 = get_peft_model(self.model_fs2, peft_config)
        
        # Fix contiguidade (Parâmetros e Buffers)
        for p in self.model_fs2.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
        for b in self.model_fs2.buffers():
            if not b.is_contiguous(): b.data = b.data.contiguous()
            
        print(f"✅ LoRA aplicado ao FastSpeech 2.")
        
        self.model = FastSpeech2Wrapper(self.model_fs2, self.device)
        self.f0_input_mode: str = str(self.model_cfg.get("f0_input_mode", "zscore")).lower()
        if self.f0_input_mode == "raw_hz":
            print(
                "⚠️ f0_input_mode=raw_hz não é suportado com tts_models/en/ljspeech/fast_pitch (LJS usou F0 em z-score). "
                "A usar 'zscore' (mean/std a partir de f0_stats do corpus / LJS).",
            )
            self.f0_input_mode = "zscore"

        # Substituir o fonemizador original pelo de português (Gruut)
        try:
            from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name
            base_phonemizer = get_phonemizer_by_name("gruut", language="pt")
            
            # Criamos um wrapper para mapear fonemas do PT que não existem no modelo Base (Inglês)
            class PTPhonemizerWrapper:
                def __init__(self, p): self.p = p
                def phonemize(self, text, separator="", language="pt"):
                    ph = self.p.phonemize(text, separator=separator, language=language)
                    # Mapeamento de Precisão: Usamos 'ɐ' (presente no vocab base) para o 'a' fechado/nasal.
                    # Não fazer replace global de 'g' (destrói IPA). Apenas vogais nasais mapeáveis.
                    replacements = {
                        'ã': 'ɐn', 'ẽ': 'en', 'ĩ': 'in', 'õ': 'on', 'ũ': 'un',
                        '\u0303': 'n',
                    }
                    for k, v in replacements.items():
                        ph = ph.replace(k, v)
                    return ph
                def name(self): return "pt_wrapper"
                def print_logs(self, level): pass
                
            new_phonemizer = PTPhonemizerWrapper(base_phonemizer)
            self.model_fs2.tokenizer.phonemizer = new_phonemizer
            print(f"🌐 Phonemizer do FastSpeech 2 substituído por Gruut (pt) com mapeamento cross-lingual")
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível trocar o phonemizer: {e}")
            
        self.model_fs2.to(self.device)
        return self.model, self.model_fs2.tokenizer

    def set_f0_zscore_stats(self, mean: float, std: float):
        """Estatísticas globais (Hz) em frames não nulos, estilo TTS/tts/datasets F0Dataset."""
        self.f0_norm_mean = float(mean)
        self.f0_norm_std = max(float(std), 1.0)

    def compute_f0_zscore_from_train_dataset(self, train_dataset, max_samples: Optional[int] = None) -> None:
        """Uma passagem pelo treino: concatena f0>0 (Hz) e define mean/std (Coqui-style)."""
        import librosa
        vals: List[np.ndarray] = []
        n = len(train_dataset) if max_samples is None else min(len(train_dataset), max_samples)
        for i in range(n):
            item = train_dataset[i]
            y = np.array(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"].get("sampling_rate", self.ap.sample_rate)
            if int(sr) != int(self.ap.sample_rate):
                y = librosa.resample(y, orig_sr=int(sr), target_sr=int(self.ap.sample_rate))
            f0 = self.ap.compute_f0(y)
            f0 = np.nan_to_num(f0)
            nz = f0[f0 > 0.0]
            if nz.size:
                vals.append(nz)
        if not vals:
            self.set_f0_zscore_stats(150.0, 40.0)
            print("⚠️ F0: sem amostras não nulas; usando mean=150, std=40 (fallback).")
            return
        cat = np.concatenate(vals)
        self.set_f0_zscore_stats(float(np.mean(cat)), float(np.std(cat)))
        print(f"📈 F0 z-score: mean={self.f0_norm_mean:.2f} Hz, std={self.f0_norm_std:.2f} Hz (n={len(cat)} frame values)")

    def prepare_item(self, batch, processor, speaker_model, language="en"):
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 22050)
        text = normalize_text(batch.get("text", ""))
        
        # Tokenização (FS2 usa fonemas internamente via tokenizer)
        input_ids = processor.text_to_ids(text, language=language)
        
        # Extração de Mel e Pitch usando o AudioProcessor (garante escalas corretas)
        import librosa
        if sr != self.ap.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.ap.sample_rate)
            sr = self.ap.sample_rate
            
        # O melspectrogram do AP já aplica log e normalização (symmetric se configurado)
        mel_spec = self.ap.melspectrogram(y) 
        y_tensor = torch.tensor(mel_spec.T) # [T, C]
        
        # F0: Hz, depois z-score (mean/std) em frames f0>0, zeros mantidos — alinhado a TTS/tts/datasets F0Dataset
        pitch = self.ap.compute_f0(y)
        pitch = np.nan_to_num(pitch).astype(np.float32)
        if self.f0_norm_mean is not None and self.f0_norm_std is not None:
            m, s = self.f0_norm_mean, self.f0_norm_std
            nz = pitch > 0.0
            if np.any(nz):
                pitch = pitch.copy()
                pitch[nz] = (pitch[nz] - m) / s
        if len(pitch) > y_tensor.shape[0]: pitch = pitch[:y_tensor.shape[0]]
        elif len(pitch) < y_tensor.shape[0]: pitch = np.pad(pitch, (0, y_tensor.shape[0] - len(pitch)))
        
        return {
            "x": torch.tensor(input_ids),
            "x_lengths": torch.tensor(len(input_ids)),
            "y": y_tensor.float(),
            "y_lengths": torch.tensor(y_tensor.shape[0]),
            "pitch": torch.tensor(pitch).unsqueeze(0).float() # [1, T]
        }

    def get_collator(self):
        return FS2DataCollator()

class MatchaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x=None, x_lengths=None, y=None, y_lengths=None, **kwargs):
        # Matcha-TTS (Lightning module) retorna loss no método forward ou compute_loss
        # x: input_ids, y: mel
        outputs = self.model(
            x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, **kwargs
        )
        return {"loss": outputs["loss"]}

class MatchaHandler(ModelHandler):
    def load(self, resume_from=None, resume_model_only=False):
        try:
            from matcha.models.matcha_tts import MatchaTTS
        except ImportError:
            print("❌ Erro: Biblioteca 'matcha-tts' não encontrada no .venv_matcha.")
            sys.exit(1)

        print(f"📥 Carregando Matcha-TTS: {self.model_cfg['id']}")
        
        # O Matcha-TTS costuma ser carregado via checkpoint (.ckpt)
        # Para simplificar o LoRA, assumimos que o modelo base está disponível ou baixado
        # Ex: self.model_matcha = MatchaTTS.load_from_checkpoint(checkpoint_path)
        # Aqui inicializamos um modelo base para aplicar o LoRA
        # Nota: Uma implementação real precisaria do config exato.
        self.model_matcha = MatchaTTS(...) # Placeholder para inicialização real
        
        # Aplicar LoRA no Decoder (DiT/Transformer) do Matcha
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none"
        )
        self.model_matcha.decoder = get_peft_model(self.model_matcha.decoder, peft_config)
        for p in self.model_matcha.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
            
        self.model_matcha.to(self.device)
        self.model = MatchaWrapper(self.model_matcha)
        return self.model, None # Matcha usa phonemizer diretamente no prepare

    def prepare_item(self, batch, processor, speaker_model, language="en"):
        # Preparação específica para o Flow Matching do Matcha
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 22050)
        text = normalize_text(batch.get("text", ""))
        
        # Matcha usa phonemizer externo
        from matcha.utils.cleaners import english_cleaners
        clean_text = english_cleaners(text)
        
        # Extração de Mel
        import librosa
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = np.log10(np.clip(mel, 1e-5, None)).T
        
        return {
            "x": torch.tensor([0]), # IDs do texto (Placeholder)
            "y": torch.tensor(log_mel),
            "x_lengths": torch.tensor(1),
            "y_lengths": torch.tensor(log_mel.shape[0])
        }

    def get_collator(self):
        @dataclass
        class MatchaDataCollator:
            def __call__(self, features):
                from torch.nn.utils.rnn import pad_sequence
                x = pad_sequence([f["x"] for f in features], batch_first=True)
                y = pad_sequence([f["y"] for f in features], batch_first=True, padding_value=-5.0)
                return {"x": x, "y": y, "x_lengths": torch.tensor([f["x_lengths"] for f in features]), "y_lengths": torch.tensor([f["y_lengths"] for f in features])}
        return MatchaDataCollator()

class GlowTTSHandler(FastSpeech2Handler):
    # Glow-TTS compartilha muita lógica com o FastSpeech 2 na Coqui
    def load(self, resume_from=None, resume_model_only=False):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.glow_tts import GlowTTS
            from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv_fs2.")
            sys.exit(1)

        print(f"📥 Carregando Glow-TTS: {self.model_cfg['id']}")
        manager = ModelManager()
        model_path = manager.download_model(self.model_cfg['id'])
        if isinstance(model_path, tuple): model_path = model_path[0]
        
        config = GlowTTSConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model_glow = GlowTTS.init_from_config(config)
        self.model_glow.load_checkpoint(config, checkpoint_dir=model_path)
        
        # LoRA no Encoder do Glow-TTS
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(target_modules=lora_cfg['target_modules'], r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'])
        self.model_glow.encoder = get_peft_model(self.model_glow.encoder, peft_config)
        for p in self.model_glow.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
            
        self.model_glow.to(self.device)
        self.model = FastSpeech2Wrapper(self.model_glow, self.device) # Reutiliza wrapper de loss
        return self.model, self.model_glow.tokenizer

class GenericSOTAHandler(ModelHandler):
    def load(self, resume_from=None, resume_model_only=False):
        print(f"⚠️ O modelo {self.model_cfg['id']} requer implementações customizadas.")
        sys.exit(1)

def merge_dataset_model_overrides(model_cfg: Dict, ds_cfg: Dict) -> Dict:
    """Mescla `dataset_profiles[*].model_overrides` sobre cópia de `models.<model_type>` (ex.: lora)."""
    base = copy.deepcopy(model_cfg)
    overrides = ds_cfg.get("model_overrides") if isinstance(ds_cfg.get("model_overrides"), dict) else None
    if not overrides:
        return base
    for key, val in overrides.items():
        cur = base.get(key)
        if isinstance(val, dict) and isinstance(cur, dict):
            merged = dict(cur)
            merged.update(val)
            base[key] = merged
            print(f"   📎 model_overrides (dataset): '{key}' mesclado → {merged}")
        else:
            base[key] = val
            print(f"   📎 model_overrides (dataset): '{key}' = {val}")
    return base


def get_handler(model_type, model_cfg, device):
    if model_type == 'speecht5':
        return SpeechT5Handler(model_cfg, device)
    elif model_type == 'f5_tts':
        return F5TTSHandler(model_cfg, device)
    elif model_type == 'fastspeech2':
        return FastSpeech2Handler(model_cfg, device)
    elif model_type == 'matcha':
        return MatchaHandler(model_cfg, device)
    elif model_type == 'glow_tts':
        return GlowTTSHandler(model_cfg, device)
    elif model_type in ['xtts_v2', 'your_tts']:
        return XTTSHandler(model_cfg, device)
    else:
        return GenericSOTAHandler(model_cfg, device)

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

class ValidationCallback(TrainerCallback):
    """Callback para mostrar detalhes extras da validação no terminal."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n✨ [VALIDAÇÃO] Época: {state.epoch:.2f} | Passo: {state.global_step}")
            for key, value in metrics.items():
                name = key.replace("eval_", "📊 ").replace("_", " ").title()
                if isinstance(value, float):
                    print(f"   {name}: {value:.4f}")
                else:
                    print(f"   {name}: {value}")
            print("--------------------------------------------------\n")


class BinaryAlignWarmupCallback(TrainerCallback):
    """Rampa linear 0→1 do peso efetivo do binary alignment (só a parcela reescrita no wrapper FastSpeech2)."""
    def __init__(self, warmup_steps: int = 0, model_ref: Any = None):
        self.warmup_steps = max(0, int(warmup_steps))
        self._model_ref = model_ref

    @staticmethod
    def _set_binary_warm(model: Any, w: float) -> None:
        cur: Any = model
        for _ in range(12):
            if cur is not None and hasattr(cur, "_binary_warm"):
                cur._binary_warm = float(w)
                return
            nxt = getattr(cur, "module", None) or getattr(cur, "base_model", None) or getattr(cur, "model", None)
            if nxt is None or nxt is cur:
                return
            cur = nxt

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.warmup_steps <= 0:
            return
        m = self._model_ref or model or kwargs.get("model")
        if m is not None:
            self._set_binary_warm(m, 0.0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.warmup_steps <= 0:
            return
        m = self._model_ref or model or kwargs.get("model")
        if m is None:
            return
        w = min(1.0, float(state.global_step) / float(self.warmup_steps))
        self._set_binary_warm(m, w)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class TrainingETACallback(TrainerCallback):
    """Mostra ETA estimado a partir do progresso real do treino."""

    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()

    def _print_eta(self, prefix, state):
        if not self.start_time or not state.max_steps or state.global_step <= 0:
            return

        elapsed = time.perf_counter() - self.start_time
        avg_step = elapsed / max(state.global_step, 1)
        remaining_steps = max(state.max_steps - state.global_step, 0)
        remaining_seconds = remaining_steps * avg_step

        print(
            f"⏳ [{prefix}] Passo {state.global_step}/{state.max_steps} | "
            f"Média: {avg_step:.2f}s/step | ETA: {format_duration(remaining_seconds)}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and ("loss" in logs or "eval_loss" in logs):
            self._print_eta("ETA", state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._print_eta("ETA/VAL", state)

def main():
    parser = argparse.ArgumentParser(description="Treinamento TTS Generalizado con LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional)")
    parser.add_argument("--dataset", type=str, default=None, help="Perfil do dataset configurado no YAML")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo config.yaml")
    parser.add_argument("--resume_from", type=str, default=None, help="Caminho para checkpoint ou modelo para continuar treinamento")
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Com --resume_from: carrega só os pesos no modelo (ex.: após mudar target_modules LoRA); "
        "NÃO reutiliza optimizer/scheduler/global_step do checkpoint (evita incompat. de parâmetros).",
    )
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    
    # 1. Identificar Perfis e Modelos
    dataset_name = args.dataset or full_cfg.get('settings', {}).get('default_dataset_profile', 'lapsbm')
    if dataset_name not in full_cfg['dataset_profiles']:
        print(f"❌ Perfil de dataset '{dataset_name}' não encontrado.")
        sys.exit(1)
        
    ds_cfg = full_cfg['dataset_profiles'][dataset_name]
    model_type = ds_cfg.get('model_type', 'speecht5')
    if model_type not in full_cfg['models']:
        print(f"❌ Modelo '{model_type}' não definido no YAML.")
        sys.exit(1)
    
    model_cfg = merge_dataset_model_overrides(full_cfg["models"][model_type], ds_cfg)

    # 2. Hardware
    hw_profile = args.profile or detect_hardware_profile(full_cfg)
    hw_cfg = full_cfg['hardware_profiles'].get(hw_profile, full_cfg['hardware_profiles']['cpu'])
    device = hw_cfg.get('device', 'cpu')
    
    # TF32 acelera matmul em FP32 (Ampere+). cudnn.benchmark escolhe algoritmos conv para o tamanho de lote actual.
    # Com bf16/fp16 no HuggingFace Trainer, o passo de treino do modelo usa precisão mista onde o autocast aplicar.
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        print(
            "🚀 CUDA: matmul TF32 (high) + cudnn.benchmark=True; Trainer usa bf16/fp16 quando activos no perfil/YAML.",
        )

    if model_type == 'f5_tts' and device == 'cuda' and hw_cfg['batch_size'] > 2:
        multiplier = hw_cfg['batch_size'] // 2
        hw_cfg['gradient_accumulation_steps'] = hw_cfg.get('gradient_accumulation_steps', 1) * multiplier
        hw_cfg['batch_size'] = 2
        print(f"⚠️ Otimizando p/ F5-TTS: Batch size reduzido para 2 e Gradient Accumulation na GPU para {hw_cfg['gradient_accumulation_steps']}x para evitar Timeouts.")

    # 3. Output Dir
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(hw_cfg['output_dir'], f"{model_type}-{dataset_name}-{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 Iniciando Pipeline para: {model_type.upper()}")
    print(f"📂 Dataset: {dataset_name} | Hardware: {hw_profile.upper()}")
    
    start_time = time.time()

    # 4. Handler e Carregamento do Modelo
    handler = get_handler(model_type, model_cfg, device)
    model, processor = handler.load(
        resume_from=args.resume_from, resume_model_only=bool(getattr(args, "resume_model_only", False))
    )

    # 5. Dataset Loading
    local_root = full_cfg.get('settings', {}).get('local_datasets_dir', './datasets')
    repo_name = ds_cfg['dataset_id'].split('/')[-1]
    local_ds_path = os.path.join(local_root, repo_name)
    
    # Se o diretório existir e contiver arquivos, consideramos local
    is_local = os.path.exists(local_ds_path) and len(os.listdir(local_ds_path)) > 0 if os.path.exists(local_ds_path) else False
    
    print(f"📥 Carregando dataset '{ds_cfg['dataset_id']}' (Local: {is_local})...")
    
    all_data = []
    if ds_cfg['dataset_id'] == "firstpixel/pt-br_char":
        print("⚠️ Aplicando patch para o 'pt-br_char' (lendo repositório raw p/ contornar erro da plataforma)...")
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        try:
            if is_local:
                csv_path = os.path.join(local_ds_path, "metadata.csv")
            else:
                csv_path = hf_hub_download(repo_id=ds_cfg['dataset_id'], filename="metadata.csv", repo_type="dataset")
                
            df = pd.read_csv(csv_path, sep="|")
            
            for idx, row in df.iterrows():
                filename = row.iloc[0]
                text = row.iloc[1]
                
                if is_local:
                    audio_url = os.path.join(local_ds_path, filename.replace("/", os.sep))
                else:
                    audio_url = f"hf://datasets/{ds_cfg['dataset_id']}/{filename}"
                    
                # O HuggingFace Datasets permite instanciar áudios localmente ou via protocolo direto
                all_data.append({"audio": audio_url, "text": text})
                    
            full_ds = Dataset.from_list(all_data)
        except Exception as e:
            print(f"❌ Erro de permissão ou leitura! O dataset '{ds_cfg['dataset_id']}' pode estar fechado (Gated) online ou corrompido.\n{str(e)}")
            sys.exit(1)
    else:
        load_kwargs = {"split": ds_cfg['dataset_split']}
        if is_local:
            print(f"📂 Carregando dataset local de: {local_ds_path}")
            dataset_stream = load_from_disk(local_ds_path)
        else:
            load_kwargs["path"] = ds_cfg['dataset_id']
            load_kwargs["streaming"] = ds_cfg.get('streaming', True)
            if 'dataset_config' in ds_cfg: load_kwargs["name"] = ds_cfg['dataset_config']
            dataset_stream = load_dataset(**load_kwargs)
        
        for item in dataset_stream:
            if "wav" in item and "audio" not in item: item["audio"] = item["wav"]
            if "txt" in item and "text" not in item: item["text"] = item["txt"]
            all_data.append(item)
        full_ds = Dataset.from_list(all_data)
        
        # Salvar localmente se não for local para evitar downloads futuros
        if not is_local:
            print(f"💾 Salvando dataset localmente em: {local_ds_path}")
            os.makedirs(os.path.dirname(local_ds_path), exist_ok=True)
            full_ds.save_to_disk(local_ds_path)
    
    # Seleção de Locutores e Amostragem Balanceada
    all_speakers = [extract_speaker_id(x) for x in all_data]
    counts = Counter(all_speakers)
    num_spk = ds_cfg.get('num_speakers', 1)
    max_samples = ds_cfg.get('num_samples_per_speaker', 0) # 0 para ilimitado
    zero_shot_split = ds_cfg.get('zero_shot_split', None)
    
    if num_spk == 0:
        valid_spks = set(all_speakers)
    else:
        valid_spks = set(s for s, c in counts.most_common(num_spk))
        
    train_spks, val_spks, test_spks = valid_spks, set(), set()
    
    if zero_shot_split and len(valid_spks) > 0:
        t_spk = zero_shot_split.get('train_speakers', 28)
        v_spk = zero_shot_split.get('val_speakers', 3)
        ts_spk = zero_shot_split.get('test_speakers', 4)
        
        sorted_spks = sorted(list(valid_spks))
        if len(sorted_spks) >= (t_spk + v_spk + ts_spk):
            train_spks = set(sorted_spks[:t_spk])
            val_spks = set(sorted_spks[t_spk:t_spk+v_spk])
            test_spks = set(sorted_spks[t_spk+v_spk:t_spk+v_spk+ts_spk])
            valid_spks = train_spks | val_spks | test_spks
        else:
            print(f"⚠️ Atenção: Não há locutores suficientes para a divisão zero-shot ({len(sorted_spks)} < {t_spk+v_spk+ts_spk}). Usando todos para treino.")

    train_indices = []
    val_indices = []
    test_indices = []
    spk_added_counts = {s: 0 for s in valid_spks}
    
    for i, (item, s) in enumerate(zip(all_data, all_speakers)):
        if s in valid_spks:
            if max_samples == 0 or spk_added_counts[s] < max_samples:
                if s in val_spks:
                    val_indices.append(i)
                elif s in test_spks:
                    test_indices.append(i)
                else: # train
                    train_indices.append(i)
                spk_added_counts[s] += 1
                
    train_dataset = full_ds.select(train_indices)
    val_dataset = full_ds.select(val_indices) if val_indices else None
    test_dataset = full_ds.select(test_indices) if test_indices else None
    
    print("\n📊 --- Resumo do Dataset (Divisão 80/10/10 por Locutor) ---")
    print(f"   👥 Locutores: Total({len(counts)}) | Selecionados({len(valid_spks)})")
    print(f"   🔀 Divisão por Locutor: Treino({len(train_spks)}) | Validação({len(val_spks)}) | Teste({len(test_spks)})")
    print(f"   🎙️ Amostras: Treino({len(train_dataset)}) | Validação({len(val_dataset) if val_dataset else 0}) | Teste({len(test_dataset) if test_dataset else 0})")
    print(f"   ⚖️ Teto: {max_samples if max_samples > 0 else 'Ilimitado'} audios/locutor")
    print("----------------------------------------------------------\n")
    
    # Exportar dataset_split.json
    split_info = {
        "train_speakers": list(train_spks),
        "val_speakers": list(val_spks),
        "test_speakers": list(test_spks),
        "counts": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "test_samples": len(test_dataset) if test_dataset else 0
        }
    }
    with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)
        
    train_params = ds_cfg.get("training") or {}

    # Configurar Sampling Rate
    target_sr = model_cfg.get('sampling_rate', 16000)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=target_sr))
    if val_dataset:
        val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    # 6. Speaker Encoder & Processing
    print(f"🔊 Processando áudio em {target_sr} Hz...")
    
    speaker_model = None
    if 'speaker_encoder_id' in model_cfg:
        from speechbrain.inference.classifiers import EncoderClassifier
        speaker_model = EncoderClassifier.from_hparams(
            source=model_cfg['speaker_encoder_id'], run_opts={"device": _device_for_speechbrain(device)}
        )
    
    language = ds_cfg.get("language", "pt")

    _pool_sz = 1
    _pool_md = "concat"
    _pool_gap = 0.25
    _pool_sd = None
    _pool_multi_frac = 1.0
    _pool_mix_mode = "replace"
    _pool_append_ratio = 1.0
    if model_type == "speecht5":
        _pool_sz = int(train_params.get("speecht5_train_spk_pool_size", 1) or 1)
        _pool_md = str(train_params.get("speecht5_train_spk_pool_mode", "concat")).strip().lower()
        _pool_gap = float(train_params.get("speecht5_train_spk_pool_gap_sec", 0.25))
        _pool_sd = train_params.get("speecht5_train_spk_pool_seed", None)
        _pool_multi_frac = max(0.0, min(1.0, float(train_params.get("speecht5_train_spk_pool_multi_fraction", 1.0))))
        _pool_mix_mode = str(train_params.get("speecht5_train_spk_pool_mix_mode", "replace")).strip().lower()
        _pool_append_ratio = max(0.0, float(train_params.get("speecht5_train_spk_pool_append_multi_ratio", 1.0)))
        if _pool_sz > 1:
            if _pool_md not in ("mean", "concat"):
                print(f"❌ speecht5_train_spk_pool_mode inválido: {_pool_md!r} (use mean ou concat).")
                sys.exit(1)
            if _pool_md == "mean" and speaker_model is None:
                print("❌ speecht5_train_spk_pool_mode=mean requer speaker_encoder_id em models.speecht5.")
                sys.exit(1)
            if _pool_mix_mode not in ("replace", "append"):
                print(
                    f"❌ speecht5_train_spk_pool_mix_mode inválido: {_pool_mix_mode!r} "
                    f"(use 'replace' ou 'append')."
                )
                sys.exit(1)

            if _pool_mix_mode == "append":
                if _pool_append_ratio <= 0.0:
                    print(
                        f"   📎 speecht5_train_spk_pool_append_multi_ratio=0: sem linhas multi extra; "
                        f"treino com dataset original ({len(train_dataset)} amostras), apesar de K={_pool_sz}."
                    )
                    split_info["speecht5_train_spk_pool"] = {
                        "pool_size": _pool_sz,
                        "mode": _pool_md,
                        "gap_sec": _pool_gap,
                        "seed": _pool_sd,
                        "mix_mode": "append",
                        "append_multi_ratio": 0.0,
                        "merged_train": False,
                    }
                    with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
                        json.dump(split_info, f, indent=4)
                else:
                    print(
                        f"   📎 Multi-frase APPEND: K={_pool_sz}, modo={_pool_md}, gap={_pool_gap}s (seed={_pool_sd!r}), "
                        f"append_multi_ratio={_pool_append_ratio:g} ⇒ +round(n×ratio) linhas multi — só train; "
                        "val permanece sentença-a-sentença."
                    )
                    train_dataset = merge_speecht5_train_dataset_multi_sentence(
                        train_dataset,
                        pool_size=_pool_sz,
                        pool_mode=_pool_md,
                        gap_sec=_pool_gap,
                        seed=int(_pool_sd) if _pool_sd is not None else None,
                        target_sr=target_sr,
                        speaker_model=speaker_model,
                        multi_sample_fraction=_pool_multi_frac,
                        mix_mode="append",
                        append_multi_ratio=_pool_append_ratio,
                    )
                    split_info["counts"]["train_samples"] = len(train_dataset)
                    split_info["speecht5_train_spk_pool"] = {
                        "pool_size": _pool_sz,
                        "mode": _pool_md,
                        "gap_sec": _pool_gap,
                        "seed": _pool_sd,
                        "mix_mode": "append",
                        "append_multi_ratio": _pool_append_ratio,
                        "merged_train": True,
                    }
                    with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
                        json.dump(split_info, f, indent=4)
            elif _pool_multi_frac <= 0.0:
                print(
                    f"   📎 Multi-frase desactivada (speecht5_train_spk_pool_multi_fraction=0): "
                    f"treino 100% single apesar de K={_pool_sz} na config."
                )
                split_info["speecht5_train_spk_pool"] = {
                    "pool_size": _pool_sz,
                    "mode": _pool_md,
                    "gap_sec": _pool_gap,
                    "seed": _pool_sd,
                    "mix_mode": "replace",
                    "multi_fraction": 0.0,
                    "merged_train": False,
                }
                with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
                    json.dump(split_info, f, indent=4)
            else:
                print(
                    f"   📎 Multi-frase REPLACE: K={_pool_sz}, modo={_pool_md}, gap={_pool_gap}s "
                    f"(seed={_pool_sd!r}), multi_fraction={_pool_multi_frac:.1%} (Bernoulli por linha) — só train; "
                    "val permanece sentença-a-sentença."
                )
                train_dataset = merge_speecht5_train_dataset_multi_sentence(
                    train_dataset,
                    pool_size=_pool_sz,
                    pool_mode=_pool_md,
                    gap_sec=_pool_gap,
                    seed=int(_pool_sd) if _pool_sd is not None else None,
                    target_sr=target_sr,
                    speaker_model=speaker_model,
                    multi_sample_fraction=_pool_multi_frac,
                    mix_mode="replace",
                    append_multi_ratio=1.0,
                )
                split_info["counts"]["train_samples"] = len(train_dataset)
                split_info["speecht5_train_spk_pool"] = {
                    "pool_size": _pool_sz,
                    "mode": _pool_md,
                    "gap_sec": _pool_gap,
                    "seed": _pool_sd,
                    "mix_mode": "replace",
                    "multi_fraction": _pool_multi_frac,
                    "merged_train": True,
                }
                with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
                    json.dump(split_info, f, indent=4)
    
    # Sistema de Cache Hard (Disco) — sufixo fs2 força recache ao mudar F0 (f0zscore_v1, etc.)
    _fs2_cache_suffix = f"_{FS2_F0_DATASET_VERSION}" if model_type == "fastspeech2" else ""
    _spk_pool_cache = ""
    if model_type == "speecht5" and _pool_sz > 1:
        _use_spk_pool_cache = (_pool_mix_mode == "append" and _pool_append_ratio > 0) or (
            _pool_mix_mode == "replace" and _pool_multi_frac > 0
        )
        if _use_spk_pool_cache:
            _spk_pool_cache = f"_spkpool{_pool_sz}_{_pool_md}_g{_pool_gap}_sd{_pool_sd}"
            if _pool_mix_mode == "append":
                _spk_pool_cache += f"_mixappend_r{int(round(_pool_append_ratio * 100))}"
            elif _pool_mix_mode == "replace" and _pool_multi_frac < 1.0:
                _spk_pool_cache += f"_mf{int(round(_pool_multi_frac * 100))}"
    cache_base_dir = os.path.join(
        full_cfg.get('settings', {}).get('local_datasets_dir', './datasets'),
        "cache_processado",
        f"{model_type}_{dataset_name}_{target_sr}hz{_fs2_cache_suffix}{_spk_pool_cache}",
    )
    train_cache_path = os.path.join(cache_base_dir, "train")
    val_cache_path = os.path.join(cache_base_dir, "val")

    if model_type == "fastspeech2" and isinstance(handler, FastSpeech2Handler):
        f0_path = os.path.join(cache_base_dir, "f0_stats.json")
        if os.path.isfile(f0_path):
            with open(f0_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            handler.set_f0_zscore_stats(st["mean"], st["std"])
            print(f"📈 F0 z-score (Hz → normalizado) carregado de: {f0_path}")
        else:
            mxs = ds_cfg.get("f0_stats_max_train_samples", None)
            handler.compute_f0_zscore_from_train_dataset(train_dataset, max_samples=mxs)
            os.makedirs(cache_base_dir, exist_ok=True)
            with open(f0_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mean": handler.f0_norm_mean,
                        "std": handler.f0_norm_std,
                        "version": FS2_F0_DATASET_VERSION,
                    },
                    f,
                    indent=2,
                )
            print(f"💾 f0_stats.json salvo em: {f0_path}")
        if handler.f0_norm_mean is not None and hasattr(handler.model, "set_f0_denorm"):
            handler.model.set_f0_denorm(handler.f0_norm_mean, handler.f0_norm_std)
    
    if os.path.exists(train_cache_path):
        print(f"⚡ Carregando dataset TRAIN processado do cache: {train_cache_path}")
        processed_train_ds = load_from_disk(train_cache_path)
    else:
        processed_train_ds = train_dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=train_dataset.column_names)
        processed_train_ds.set_format(type="torch")
        processed_train_ds.save_to_disk(train_cache_path)
        print(f"💾 Cache TRAIN salvo em: {train_cache_path}")
    
    processed_val_ds = None
    if val_dataset:
        if os.path.exists(val_cache_path):
            print(f"⚡ Carregando dataset VAL processado do cache: {val_cache_path}")
            processed_val_ds = load_from_disk(val_cache_path)
        else:
            processed_val_ds = val_dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=val_dataset.column_names)
            processed_val_ds.set_format(type="torch")
            processed_val_ds.save_to_disk(val_cache_path)
            print(f"💾 Cache VAL salvo em: {val_cache_path}")

    # 7. Training
    # train_params já definido (secção cache / multi-frase SpeechT5)
    
    # Prioridade de parâmetros:
    # Se o usuário passar um perfil CUDA explicitamente via CLI, o Hardware Profile manda.
    # Caso contrário, o Dataset Profile manda (para manter reprodutibilidade por modelo).
    if args.profile and "cuda" in args.profile:
        print(f"🎮 Modo Override Ativo: Usando Batch ({hw_cfg['batch_size']}) e GradAcc ({hw_cfg['gradient_accumulation_steps']}) do perfil {hw_profile}")
        effective_batch_size = hw_cfg['batch_size']
        effective_grad_acc = hw_cfg['gradient_accumulation_steps']
    else:
        effective_batch_size = train_params.get('batch_size', hw_cfg['batch_size'])
        effective_grad_acc = train_params.get('gradient_accumulation_steps', hw_cfg['gradient_accumulation_steps'])
    effective_num_workers = train_params.get('dataloader_num_workers', 0)
    is_windows = os.name == "nt"
    if is_windows and effective_num_workers > 4:
        print(
            f"⚙️ Windows: dataloader_num_workers {effective_num_workers}→4 (spawn/IPC; subir manualmente se medires melhora).",
        )
        effective_num_workers = 4

    # bf16: training.bf16 true força; false desliga; null ausente = herda hardware (ex. cuda_16gb.bf16: true)
    _tpbf = train_params.get("bf16", None)
    if _tpbf is True:
        use_bf16 = device == "cuda"
    elif _tpbf is False:
        use_bf16 = False
    else:
        use_bf16 = bool(hw_cfg.get("bf16", False)) and device == "cuda"
    if use_bf16:
        print("🧮 bf16=True, fp16=False (ou dataset/hardware sem fp16)")

    _gck = train_params.get("gradient_checkpointing", None)
    if _gck is not None:
        use_grad_ckpt = bool(_gck)
    else:
        use_grad_ckpt = bool(hw_cfg.get("gradient_checkpointing", False))
    if use_grad_ckpt:
        print("📎 gradient_checkpointing=True (reencaminhado p/ base/Peft; use_reentrant=False no kwargs)")

    gbatch = effective_batch_size * effective_grad_acc
    print(
        f"📦 Lote efetivo: batch {effective_batch_size} × grad_acc {effective_grad_acc} = {gbatch} (menos acum. ⇒ menos overhead por passo, ex. 4 vs 8).",
    )

    _max_st = int(train_params.get("max_steps") or 0)
    # None / omit = sem limite (guarda todos os checkpoint-*); inteiro N = manter no máximo N.
    _stl = train_params.get("save_total_limit", None)
    if _stl is not None and str(_stl).strip() != "":
        _sv = int(_stl)
        _save_limit = None if _sv <= 0 else max(1, _sv)  # 0 ou negativo = sem limite (como null)
    else:
        _save_limit = None
    training_kwargs = {
        "output_dir": out_dir,
        "per_device_train_batch_size": effective_batch_size,
        "gradient_accumulation_steps": effective_grad_acc,
        "learning_rate": train_params['learning_rate'],
        "num_train_epochs": train_params['num_epochs'],
        "logging_steps": train_params['logging_steps'],
        "save_strategy": "steps",
        "save_steps": train_params['save_steps'], # Salvando a cada 10 épocas (aprox. 1750 passos com batch 4)
        "save_safetensors": False,
        "weight_decay": train_params.get('weight_decay', 0.0),
        "dataloader_num_workers": effective_num_workers,
        "dataloader_pin_memory": device == "cuda",
        "remove_unused_columns": False,
        "report_to": "none",
        "max_grad_norm": 5.0 # Estabilizando gradientes para evitar saltos bruscos na Loss
    }
    if _save_limit is not None:
        training_kwargs["save_total_limit"] = _save_limit
    if _max_st > 0:
        training_kwargs["max_steps"] = _max_st
        print(
            f"🧭 max_steps={_max_st} (limita o treino a estes passos; útil c/ runs longos a partir de zero).",
        )
    if _save_limit is not None:
        print(f"📁 Checkpoints: manter no máximo os últimos {_save_limit} (rotação no disco).")
    else:
        print("📁 Checkpoints: sem limite — todos os `checkpoint-*` no output_dir serão conservados.")
    _wr = train_params.get("warmup_ratio", None)
    if _wr is not None and float(_wr) > 0:
        training_kwargs["warmup_ratio"] = float(_wr)
    _lst = train_params.get("lr_scheduler_type", None)
    if _lst is not None and str(_lst).strip():
        training_kwargs["lr_scheduler_type"] = str(_lst).strip()
    if use_bf16:
        training_kwargs["bf16"] = True
        training_kwargs["fp16"] = False
    else:
        _tpfp = train_params.get("fp16", None)
        training_kwargs["fp16"] = _tpfp if _tpfp is not None else hw_cfg.get("fp16", False)

    if use_grad_ckpt and device == "cuda":
        training_kwargs["gradient_checkpointing"] = True
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # Ajuste de comportamento por sistema operacional.
    # No Linux, workers persistentes tendem a melhorar o throughput de input pipeline.
    # No Windows, manter desabilitado evita arestas com spawn em alguns cenários.
    if effective_num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = train_params.get("dataloader_prefetch_factor", 2)
        training_kwargs["dataloader_persistent_workers"] = not is_windows

    if processed_val_ds:
        training_kwargs["eval_strategy"] = "steps"
        # null/ausente: usa save_steps (menos overhead do que validar a cada logging_steps)
        _es = train_params.get("eval_steps", None)
        if _es is None:
            _es = train_params.get("save_steps", train_params["logging_steps"])
        training_kwargs["eval_steps"] = int(_es)
        _nval = len(processed_val_ds)
        _ls = train_params.get("logging_steps", 50)
        print(
            f"📊 Validação: {_nval} amostras no eval_dataset | "
            f"eval a cada {training_kwargs['eval_steps']} passos (eval_steps no YAML = null → igual a save_steps) | "
            f"log de treino a cada {_ls} passos (não confundir com eval).",
        )
    else:
        print("⚠️ Sem dataset de validação: Trainer sem eval (só treino + save por save_steps se aplicável).")

    _lb = train_params.get("load_best_model_at_end", None)
    if _lb is True and processed_val_ds:
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = str(train_params.get("metric_for_best_model", "eval_loss"))
        training_kwargs["greater_is_better"] = bool(train_params.get("greater_is_better", False))
        print(
            f"📌 load_best_model_at_end=True métrica={training_kwargs['metric_for_best_model']} "
            f"greater_is_better={training_kwargs['greater_is_better']}",
        )
    elif _lb is True and not processed_val_ds:
        print("⚠️ load_best_model_at_end omitido — sem eval_dataset.")

    training_args = TrainingArguments(**training_kwargs)

    cb = [TrainingETACallback(), ValidationCallback(), PeftAdapterSaveCallback(model)]
    if model_type == "speecht5":
        cb.append(SpeechT5GuidedAttentionLayerdropCallback(model))
    if _get_peft_model_for_save(model) is not None:
        print("💾 Após cada save: adapter_config + adapter (PEFT) em checkpoint-* (além do state do Trainer).")
    _baw = int(train_params.get("binary_align_warmup_steps", 0) or 0)
    if _baw > 0 and model_type == "fastspeech2":
        cb.append(BinaryAlignWarmupCallback(_baw, model_ref=model))
        print(f"🔥 Binary align warmup: linear até passo ~{_baw} (1.0 = binário pleno)")

    trainer_kwargs = {
        "model": model, 
        "args": training_args, 
        "train_dataset": processed_train_ds, 
        "eval_dataset": processed_val_ds,
        "data_collator": handler.get_collator(),
        "callbacks": cb,
    }

    trainer_kwargs["callbacks"].append(LogMetricsCallback())
    trainer = Trainer(**trainer_kwargs)
    
    # Configurar logging para incluir métricas customizadas como f0_rmse_hz
    # Nome do tensor-alvo usado no eval: tem de bater com o collator (senão has_labels=False e não há eval_loss).
    _label_for_eval = {
        "fastspeech2": "y",
        "glow_tts": "y",
        "speecht5": "labels",
        "f5_tts": "mel",
        "xtts_v2": "audio_codes",
        "matcha": "y",
    }
    if model_type in _label_for_eval:
        trainer.label_names = [_label_for_eval[model_type]]
    else:
        trainer.label_names = None

    print("\n🏁 Iniciando Treinamento...")
    _trainer_resume = None
    if args.resume_from and getattr(args, "resume_model_only", False):
        print(f"🔄 Pesos a partir de (modelo): {args.resume_from}")
        print(
            "   ⚡ --resume_model_only: optimizer, scheduler e contador de passo recomeçam (treino 'novo' na pasta out_dir). "
            "Use ao alterar target_modules/LoRA ou quando o state do otimizador deixa de bater com o modelo."
        )
    elif args.resume_from:
        print(f"🔄 Retomando treinamento a partir de: {args.resume_from}")
        _trainer_resume = args.resume_from
        # Patch para compatibilidade de versão do TrainerState
        # Remove chaves obsoletas como 'best_global_step' que causam erro no transformers 4.44+
        state_path = os.path.join(args.resume_from, "trainer_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                
                dirty = False
                for key in ["best_global_step"]:
                    if key in state_data:
                        del state_data[key]
                        dirty = True
                
                if dirty:
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state_data, f, indent=2)
                    print(f"   🩹 Patch aplicado ao trainer_state.json (removidas chaves obsoletas).")
            except Exception as e:
                print(f"   ⚠️ Aviso ao aplicar patch no state: {e}")
    
    trainer.train(resume_from_checkpoint=_trainer_resume)
    
    # 8. Saving: bundle PEFT (adapter_config + adapter) em out_dir; processor HF se houver; senão state genérico
    print("\n💾 Salvando modelo final (adapter PEFT p/ inferência, quando aplicável)...")
    try:
        if save_peft_adapter_for_inference(model, out_dir):
            print("   ✅ adapter_config.json + adapter_model.* (PEFT) no diretório de saída")
        else:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(out_dir)  # type: ignore[union-attr]
            elif hasattr(model, "model") and hasattr(model.model, "save_pretrained"):
                model.model.save_pretrained(out_dir)  # type: ignore[union-attr]
            else:
                trainer.save_model(out_dir)

        if hasattr(processor, "save_pretrained") and processor is not None:
            processor.save_pretrained(out_dir)

        print(f"✅ Treinamento concluído com sucesso! Pasta do run: {out_dir}")
    except Exception as e:
        print(f"⚠️ Atenção ao salvar modelo final: {e}")
        print("💡 Nota: Os checkpoints do treinamento (ex: checkpoint-3000) já foram salvos automaticamente pelo Trainer e podem ser utilizados!")

    # 9. Saving Cost-Benefit Metrics
    print("\n📝 Gerando README com parâmetros da fórmula Custo-Benefício...")
    end_time = time.time()
    training_latency = end_time - start_time
    
    final_loss = None
    if trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if 'loss' in log:
                final_loss = log['loss']
                break
                
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_allocated_gb = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    
    readme_appendix = f"""
---
## Parâmetros da Fórmula Custo-Benefício (Treinamento)

**Fórmula Original:**
`CustoBeneficio = [w1 * (1 - WER) + w2 * SimLocutor + w3 * Qualidade] / [CustoGPU$ + lambda * Latência]`

### 🔻 Custos (Denominador)
- **CustoGPU$ (Hardware + Uso VRAM)**: {gpu_name} (Pico VRAM: {vram_allocated_gb:.2f} GB)
- **Latência (Tempo de Treinamento)**: {training_latency:.2f} segundos ({training_latency/60:.2f} minutos)

### 🌟 Benefícios (Numerador - Proxies Atuais)
*Métricas como WER, Similaridade e MOS (Qualidade) requerem uma etapa de `inferência + validação` ao final do processo. Aqui usamos a Loss do treino como indicativo de melhora na capacidade do modelo.*
- **Loss Final do Treino**: {final_loss if final_loss is not None else 'N/A'}
"""
    readme_path = os.path.join(out_dir, "README.md")
    mode = "a" if os.path.exists(readme_path) else "w"
    try:
        with open(readme_path, mode, encoding="utf-8") as f:
            f.write(readme_appendix)
        print(f"📄 Parâmetros de Custo-Benefício adicionados ao README em: {readme_path}")
    except Exception as e:
        print(f"⚠️ Erro ao atualizar README: {e}")

if __name__ == "__main__":
    main()
