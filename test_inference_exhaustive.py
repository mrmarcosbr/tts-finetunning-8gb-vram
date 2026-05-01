from __future__ import annotations

import os
import sys
sys.modules['torchcodec'] = None # Mock para evitar crash do torchaudio e do transformers no Windows

import warnings

# Torchaudio/HF/SpeechBrain: avisos conhecidos e barulhentos durante import e inferência.
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"speechbrain\.utils\.torch_audio_backend",
)
warnings.filterwarnings(
    "ignore",
    message=r".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

import importlib
import time
import random
import yaml
import torch
import torchaudio

try:
    importlib.import_module("torch.distributed.tensor")
except ImportError:
    pass
import re
import unicodedata
import argparse
import csv
import json
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import subprocess

# Carregar variáveis de ambiente (HF_TOKEN)
load_dotenv()

# --- 1. Monkey Patch & Setup ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from datasets import load_dataset, Audio, Dataset, load_from_disk
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from collections import Counter


def _device_for_speechbrain(device: str) -> str:
    raw = str(device or "cpu").strip()
    if raw.lower() == "cuda":
        # SpeechBrain (run_opts): 'cuda' sem índice → parse falha ("expected 2, got 1"); usar cuda:N.
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    return raw


# Configuração de encoding para Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def normalize_text(text):
    if not text: return ""
    text = text.lower() # Converter para minúsculo para bater com vocabulário
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

# Alvo de pico ao gravar WAV (evita clipping comum em vocoder / SpeechT5)
INFERENCE_PEAK_TARGET = 0.95
# Inferência "quase muda": se max(|x|) já é isto ou menos, não aplicamos ganho até peak_target —
# senão amplificamos apenas chão/quantização (soa ruído forte, muitas vezes perceptível nos graves).
PEAK_NORMALIZE_MIN_ABS = 2e-3


def peak_normalize_audio(
    audio,
    peak_target: float = INFERENCE_PEAK_TARGET,
    *,
    min_peak_abs: float = PEAK_NORMALIZE_MIN_ABS,
):
    """Escala o sinal para que max(|x|) == peak_target, exceto se já for quase silêncio (evita amplifier ruído de chão)."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.asarray([], dtype=np.float32)
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12:
        return np.zeros(x.shape, dtype=np.float32)
    if peak < float(min_peak_abs):
        return x.astype(np.float32)
    return (x * (peak_target / peak)).astype(np.float32)


_MP3_NOTICE_SHOWN = False


def _try_export_mp3_via_ffmpeg(path_wav: str, mp3_path: Path, bitrate: str) -> bool:
    """Encode MP3 from existing WAV via `ffmpeg` subprocess (robusto no Windows vs. pydub)."""
    exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not exe:
        return False
    br = (str(bitrate).strip() or "192k").replace(" ", "")
    try:
        r = subprocess.run(
            [
                exe,
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                os.path.normpath(os.path.abspath(path_wav)),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                br,
                os.path.normpath(str(mp3_path.resolve())),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    ok = r.returncode == 0 and mp3_path.is_file() and mp3_path.stat().st_size > 0
    return ok


def _try_export_mp3_via_pydub(path_wav: str, sr: int, audio_arr: np.ndarray, mp3_path: Path, bitrate: str) -> bool:
    try:
        from pydub import AudioSegment
    except ImportError:
        return False
    x = np.asarray(audio_arr, dtype=np.float64).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm_i16 = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    seg = AudioSegment(
        pcm_i16.tobytes(),
        frame_rate=int(sr),
        sample_width=2,
        channels=1,
    )
    try:
        seg.export(str(mp3_path), format="mp3", bitrate=str(bitrate))
    except Exception:
        return False
    return mp3_path.is_file() and mp3_path.stat().st_size > 0


def write_wav_and_mp3(
    path_wav: str,
    audio_arr: np.ndarray,
    sr: int,
    *,
    also_mp3: bool = True,
    mp3_bitrate: str = "192k",
) -> None:
    """Grava WAV; cópia .mp3 ao lado (preferencialmente ffmpeg a partir do WAV; fallback pydub)."""
    sr = int(sr)
    sf.write(path_wav, audio_arr, sr)
    global _MP3_NOTICE_SHOWN
    if not also_mp3:
        return
    mp3_path = Path(path_wav).with_suffix(".mp3")
    if _try_export_mp3_via_ffmpeg(path_wav, mp3_path, mp3_bitrate):
        return
    if _try_export_mp3_via_pydub(path_wav, sr, audio_arr, mp3_path, mp3_bitrate):
        return
    if not _MP3_NOTICE_SHOWN:
        print(
            "   ⚠️ MP3 não gerado: instala FFmpeg no PATH (https://ffmpeg.org) "
            "(libmp3lame) ou `pip install pydub` + ffmpeg. Saídas em WAV apenas."
        )
        _MP3_NOTICE_SHOWN = True

from trained_post_eq import apply_trained_output_waveform_eq, inference_dc_and_highpass

def split_text_for_tts(text: str, max_chars: int = 180) -> List[str]:
    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    separators = [". ", "? ", "! ", "; ", ", "]
    chunks = [clean]
    for sep in separators:
        next_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                next_chunks.append(chunk)
                continue
            parts = [p.strip() for p in chunk.split(sep) if p.strip()]
            if len(parts) == 1:
                next_chunks.append(chunk)
                continue
            for idx, part in enumerate(parts):
                if idx < len(parts) - 1 and not part.endswith(tuple(".!?;,:")):
                    part = part + sep.strip()
                next_chunks.append(part)
        chunks = next_chunks

    final_chunks = []
    buffer = ""
    for chunk in chunks:
        candidate = chunk if not buffer else f"{buffer} {chunk}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                final_chunks.append(buffer)
            buffer = chunk
    if buffer:
        final_chunks.append(buffer)

    return final_chunks if final_chunks else [clean]

def compress_audio_silence(
    audio,
    sr: int,
    top_db: float = 35.0,
    min_gap_ms: int = 90,
):
    """Remove silêncios longos no áudio, preservando pequenas pausas naturais."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or sr <= 0:
        return x

    try:
        intervals = librosa.effects.split(x, top_db=top_db)
    except Exception:
        return x

    if intervals.size == 0:
        return np.asarray([], dtype=np.float32)

    gap = np.zeros(int(sr * (min_gap_ms / 1000.0)), dtype=np.float32)
    pieces = []
    for idx, (start, end) in enumerate(intervals):
        piece = x[start:end]
        if piece.size == 0:
            continue
        pieces.append(piece)
        if idx < len(intervals) - 1 and gap.size > 0:
            pieces.append(gap)

    if not pieces:
        return np.asarray([], dtype=np.float32)

    return np.concatenate(pieces).astype(np.float32)

def trim_and_compress_silence(audio, sr: int, top_db: float = 35.0):
    """Tira silêncio das bordas e comprime silêncios longos internos."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or sr <= 0:
        return x
    try:
        trimmed, _ = librosa.effects.trim(x, top_db=top_db)
    except Exception:
        trimmed = x
    return compress_audio_silence(trimmed, sr, top_db=top_db, min_gap_ms=90)

def inference_wav_prefix(args: argparse.Namespace, index: int) -> str:
    """Mesmo prefixo usado nos ficheiros WAV deste run (custom vs sentenca_N)."""
    ut = getattr(args, "text", None)
    stripped = ut.strip() if isinstance(ut, str) else ""
    return "custom" if (stripped and index == 0) else f"sentenca_{index + 1}"


def write_per_sentence_transcripts_and_tsv(
    out_dir: str,
    texts: List[str],
    args: argparse.Namespace,
) -> None:
    """
    Grava sentenca_N.txt (só a transcrição bruta) e transcricoes.tsv (prefix<TAB>texto, sem cabeçalho),
    alinhado aos prefixos dos WAV deste run.
    """
    tsv_path = os.path.join(out_dir, "transcricoes.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as tsv_f:
        w = csv.writer(tsv_f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for i, raw in enumerate(texts):
            prefix = inference_wav_prefix(args, i)
            body = (raw or "").strip()
            txt_path = os.path.join(out_dir, f"{prefix}.txt")
            with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
                if body:
                    f.write("\n")
            w.writerow([prefix, body])
    print(f"   📄 Transcrições por sentença: {os.path.basename(out_dir)}/*.txt + {os.path.basename(tsv_path)}")


def save_speecht5_reference_run_extras(
    out_dir: str,
    *,
    prefix: str,
    text_raw: str,
    p_refs_wav: str,
    speaker_model: Optional[Any],
    encoder_sr: int,
    device: str,
    manifest_rows: List[Dict[str, Any]],
) -> None:
    """
    Dentro da pasta de inferência: referencia_embeddings (*.npy + manifest acumulado em manifest_rows),
    referencia_sentencas (*.txt), referencia_wav (cópia só do .wav na raiz — ignora .mp3).
    O x-vector é calculado apenas a partir do ficheiro *_referencia.wav (1 WAV = 1 embedding; sem áudio
    agregado nem mistura do pool usado no gen_spk).
    """
    root = Path(out_dir)
    emb_dir = root / "referencia_embeddings"
    txt_dir = root / "referencia_sentencas"
    wav_dir = root / "referencia_wav"
    emb_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {
        "prefix": prefix,
        "root_referencia_wav": os.path.basename(p_refs_wav),
    }

    src = Path(p_refs_wav)
    y_from_wav: Optional[np.ndarray] = None
    sr_from_wav: Optional[int] = None
    if src.is_file() and src.suffix.lower() == ".wav":
        try:
            y_lr, sr_lr = librosa.load(str(src), sr=None, mono=True)
            y_from_wav = np.asarray(y_lr, dtype=np.float32).reshape(-1)
            sr_from_wav = int(sr_lr)
        except Exception:
            y_from_wav = None
            sr_from_wav = None

    if speaker_model is not None and y_from_wav is not None and y_from_wav.size > 0 and sr_from_wav is not None:
        ref_t = speecht5_speaker_embedding_from_numpy(
            y_from_wav, sr_from_wav, int(encoder_sr), speaker_model, device
        )
        vec = ref_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        emb_name = f"{prefix}_referencia_emb.npy"
        np.save(str(emb_dir / emb_name), vec)
        row["embedding_npy"] = emb_name
        row["shape"] = [int(vec.shape[0])]
        row["embedding_source"] = os.path.basename(p_refs_wav)
    else:
        row["embedding_npy"] = None
        if speaker_model is None:
            row["note"] = "sem SpeechBrain encoder"
        elif not src.is_file() or src.suffix.lower() != ".wav":
            row["note"] = "falta *_referencia.wav (só .wav é usado para x-vector e cópia)"
        else:
            row["note"] = "falha ao ler WAV de referência ou sinal vazio"

    body = (text_raw or "").strip()
    txt_name = f"{prefix}.txt"
    (txt_dir / txt_name).write_text(body + "\n" if body else "", encoding="utf-8")
    row["sentenca_txt"] = txt_name

    if src.is_file() and src.suffix.lower() == ".wav":
        dst_name = f"{prefix}_referencia.wav"
        shutil.copy2(src, wav_dir / dst_name)
        row["referencia_wav_subfolder"] = dst_name
    manifest_rows.append(row)


def copy_treinado_wavs_to_subfolder(out_dir: str, subfolder: str = "treinado_wav") -> int:
    """Cria `out_dir/subfolder` com cópias só dos .wav `*_treinado.wav` na raiz (ignora .mp3 e outros)."""
    root = Path(out_dir)
    if not root.is_dir():
        return 0
    dest = root / subfolder
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(root.glob("*_treinado.*")):
        if not src.is_file() or src.suffix.lower() != ".wav":
            continue
        if not src.stem.lower().endswith("_treinado"):
            continue
        shutil.copy2(src, dest / src.name)
        n += 1
    if n:
        print(f"   📁 Cópias só *_treinado.wav → {dest} ({n} ficheiro(s); .mp3 ignorados)")
    return n


def write_inference_sentences_txt(out_dir: str, texts: List[str]) -> None:
    path = os.path.join(out_dir, "sentencas.txt")
    lines = [
        "Sentenças usadas nesta inferência",
        "(entrada do modelo = texto normalizado: minúsculas, sem acentos)",
        "",
    ]
    for i, raw in enumerate(texts, 1):
        clean = normalize_text(raw)
        lines.append(f"{i}. Texto bruto:")
        lines.append(f"   {raw}")
        lines.append(f"   Normalizado (TTS):")
        lines.append(f"   {clean}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def calculate_wer(reference_text, hypothesis_text):
    import re
    # Limpa pontuação e deixa tudo minúsculo
    def clean(s): return re.sub(r'[^\w\s]', '', s.lower().strip())
    
    ref_words = clean(reference_text).split()
    hyp_words = clean(hypothesis_text).split()
    if not ref_words: return 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint8)
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j]) + 1
                
    return float(d[len(ref_words)][len(hyp_words)]) / len(ref_words)
    
def find_checkpoint(model_path):
    """Detecta se há subpastas de checkpoint e retorna a mais recente."""
    if not os.path.isdir(model_path):
        return model_path
    
    checkpoints = [d for d in os.listdir(model_path) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, d))]
    
    if not checkpoints:
        return model_path
        
    # Ordenar por número do passo (checkpoint-1000, checkpoint-2000, etc)
    try:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest = checkpoints[-1]
        print(f"   🔄 Checkpoint detectado automaticamente: {latest}")
        return os.path.join(model_path, latest)
    except:
        return model_path


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_hardware_profile():
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram >= 14.0: return "cuda_16gb"
        return "cuda_8gb"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "macbook"
    return "cpu"

def extract_speaker_id(item: Optional[Dict[str, Any]]) -> str:
    """Identificador do locutor (ex. M031) a partir de metadados ou caminhos LapsBM."""
    if not item:
        return "unknown"
    for k in ["speaker_id", "speaker", "user_id", "client_id"]:
        if k in item and item[k] is not None:
            return str(item[k])
    possible_paths = [
        item.get("__url__", ""),
        item.get("wav", {}).get("path", "") if isinstance(item.get("wav"), dict) else "",
        item.get("audio", {}).get("path", "") if isinstance(item.get("audio"), dict) else "",
    ]
    for raw in possible_paths:
        p = str(raw)
        if not p:
            continue
        if "LapsBM-" in p:
            fn = os.path.basename(p.replace("/", os.sep))
            if fn.startswith("LapsBM-"):
                return fn.split(".")[0].replace("LapsBM-", "")
        parts = p.replace("\\", "/").split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            if folder.lower() not in {"audio", "wavs", "clips"} and len(folder) > 1:
                return folder
    return "unknown"


def format_sentence_id_display(item: Optional[Dict[str, Any]]) -> str:
    """ID de sentença estilo LapsBM-0678 a partir de __key__ ou nome do WAV."""
    if not item:
        return "—"
    base = ""
    ky = item.get("__key__")
    if isinstance(ky, str) and ky.strip():
        base = os.path.splitext(os.path.basename(ky.strip().replace("\\", "/")))[0]
    if not base:
        au = item.get("audio") or item.get("wav")
        if isinstance(au, dict) and au.get("path"):
            base = os.path.splitext(os.path.basename(str(au["path"]).replace("/", os.sep)))[0]
    if not base:
        return "—"
    m = re.match(r"^LapsBM[_-](.+)$", base, re.IGNORECASE)
    if m:
        return "LapsBM-" + m.group(1)
    return base


def reference_wav_stem_for_export(item: Dict[str, Any]) -> str:
    ky = item.get("__key__")
    if isinstance(ky, str) and ky.strip():
        return os.path.splitext(os.path.basename(ky.strip().replace("\\", "/")))[0]
    au = item.get("audio") or item.get("wav") or {}
    if isinstance(au, dict) and au.get("path"):
        return os.path.splitext(os.path.basename(str(au["path"]).replace("/", os.sep)))[0]
    return ""


def _speaker_id_from_audio_path_dict(dct: Dict[str, Any]) -> Optional[str]:
    """Locutor a partir de .../audio/<speaker_id>/ficheiro.wav quando a coluna HF não traz speaker_id."""
    for key in ("audio", "wav"):
        au = dct.get(key)
        if not isinstance(au, dict):
            continue
        p = au.get("path")
        if not p:
            continue
        norm = os.path.normpath(str(p).replace("/", os.sep))
        parent = os.path.basename(os.path.dirname(norm))
        if not parent:
            continue
        low = parent.lower()
        if low in {"", ".", "audio", "wavs", "clips"}:
            continue
        return parent
    return None


def exported_reference_wav_fallback_path(
    item: Dict[str, Any], full_cfg: Dict[str, Any], dataset_profile: str
) -> Optional[str]:
    """WAV em test_split_inferencia/.../audio/<locutor>/<stem>.wav (reconstruct_and_export)."""
    stem = reference_wav_stem_for_export(item)
    if not stem:
        return None
    lr = full_cfg.get("settings", {}).get("local_datasets_dir", "./datasets")
    audio_root = os.path.abspath(
        os.path.join(lr, "test_split_inferencia", dataset_profile, "audio"),
    )
    spk = extract_speaker_id(item)
    direct: Optional[str] = None
    if spk != "unknown":
        p = os.path.join(audio_root, spk, f"{stem}.wav")
        if os.path.isfile(p):
            direct = os.path.abspath(p)
    if direct:
        return direct
    # Dataset exportado muitas vezes não tem speaker_id na linha HF; procurar stem em qualquer subpasta.
    try:
        if os.path.isdir(audio_root):
            for sub in sorted(os.listdir(audio_root)):
                subdir = os.path.join(audio_root, sub)
                if not os.path.isdir(subdir):
                    continue
                cand = os.path.join(subdir, f"{stem}.wav")
                if os.path.isfile(cand):
                    return os.path.abspath(cand)
    except OSError:
        pass
    return None


def resolve_reference_waveform_for_inference(
    item: Optional[Dict[str, Any]],
    target_sr: int,
    *,
    full_cfg: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], int]:
    """Áudio de referência do locutor: Arrow/HF; se vazio, WAV em test_split_inferencia/.../audio."""
    y, sr = resolve_reference_waveform_to_mono_sr(item, target_sr)
    if y is not None and y.size > 0:
        return y, sr
    if full_cfg and dataset_profile and item:
        p = exported_reference_wav_fallback_path(item, full_cfg, dataset_profile)
        if p:
            try:
                y2, _ = librosa.load(p, sr=int(target_sr), mono=True)
                return np.asarray(y2, dtype=np.float32).reshape(-1), int(target_sr)
            except Exception:
                pass
    return None, int(target_sr)


def speecht5_speaker_embedding_from_numpy(
    y: np.ndarray,
    sr_audio: int,
    encoder_sr: int,
    speaker_model: Any,
    device: str,
) -> torch.Tensor:
    """Idêntico a train_exhaustive SpeechT5Handler.prepare_item (waveform 1D → x-vector), depois [1,512] float32."""
    empty = torch.zeros((1, 512), dtype=torch.float32, device=device)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return empty
    if int(sr_audio) != int(encoder_sr):
        y = librosa.resample(y, orig_sr=int(sr_audio), target_sr=int(encoder_sr)).astype(np.float32)
    # O treino usa tensor 1D sobre o array (sem batch explícito); manter igual.
    wav = torch.tensor(np.asarray(y))
    with torch.no_grad():
        emb = speaker_model.encode_batch(wav)
        emb = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    if vec.size != 512:
        raise RuntimeError(f"speaker embedding shape inesperada: {vec.shape}, esperado (512,)")
    return torch.from_numpy(vec.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)


def speecht5_embedding_for_inference_candidate(
    effective_cand: Optional[Any],
    *,
    full_cfg: Dict[str, Any],
    dataset_profile: str,
    model_cfg: Dict[str, Any],
    speaker_model: Optional[Any],
    device: str,
) -> Tuple[torch.Tensor, str]:
    """Um embedding [1,512] por sentença (mesmo WAV que *_referencia.wav; alinhado ao treino)."""
    encoder_sr = int(model_cfg.get("sampling_rate", 16000))
    z = torch.zeros((1, 512), dtype=torch.float32, device=device)
    if speaker_model is None:
        return z, "sem_encoder_de_locutor"
    if effective_cand is None or effective_cand.dataset_item is None:
        return z, "sem_linha_dataset"
    y, sr = resolve_reference_waveform_for_inference(
        effective_cand.dataset_item,
        encoder_sr,
        full_cfg=full_cfg,
        dataset_profile=dataset_profile,
    )
    if y is None or y.size == 0:
        return z, "sem_wav_de_referencia"
    yv = np.asarray(y, dtype=np.float32).reshape(-1)
    pk = float(np.max(np.abs(yv)))
    if pk < 1e-6:
        print(
            "   ⚠️ WAV de referência efectivamente nulo (peak≈0): x-vector será fraco/mudo; "
            "verifica paths no split exportado ou ficheiros em test_split_inferencia/.../audio/."
        )
    else:
        print(f"   📊 Áudio de referência para x-vector: peak|x|={pk:.5f} ({yv.size} amostras @ {sr} Hz)")
    emb = speecht5_speaker_embedding_from_numpy(y, sr, encoder_sr, speaker_model, device)
    return emb, "x-vector_do_WAV_referência"


def build_speecht5_speaker_pool_index(
    full_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    dataset_name: str,
    *,
    include_train: bool,
) -> Dict[str, List[InferenceCandidate]]:
    """Agrupa candidatos com WAV por locutor; dedupe por texto normalizado."""
    by_spk: Dict[str, List[InferenceCandidate]] = defaultdict(list)
    seen_per: Dict[str, set] = defaultdict(set)

    def ingest(lst: List[InferenceCandidate]) -> None:
        for c in lst:
            if not c.dataset_item:
                continue
            spk = extract_speaker_id(c.dataset_item)
            if not spk or spk == "unknown":
                continue
            nk = normalize_text(c.text_raw)
            if not nk or nk in seen_per[spk]:
                continue
            seen_per[spk].add(nk)
            by_spk[spk].append(c)

    ingest(load_inference_test_candidates(full_cfg, ds_cfg, dataset_name))
    if include_train:
        ingest(load_inference_train_candidates(full_cfg, ds_cfg, dataset_name))

    return dict(by_spk)


def select_speecht5_embedding_pool_candidates(
    current: InferenceCandidate,
    pool_by_speaker: Dict[str, List[InferenceCandidate]],
    k: int,
    *,
    seed: Optional[int],
) -> List[InferenceCandidate]:
    """Inclui a sentença actual primeiro; completa até k com outras do mesmo locutor."""
    if k <= 1 or not current.dataset_item:
        return [current]
    spk = extract_speaker_id(current.dataset_item)
    peers = list(pool_by_speaker.get(spk, []))
    if not peers:
        return [current]
    cur_key = normalize_text(current.text_raw)
    others = [c for c in peers if normalize_text(c.text_raw) != cur_key]
    if seed is not None:
        rng = random.Random(int(seed))
        others = others[:]
        rng.shuffle(others)
    else:
        others = sorted(others, key=lambda c: format_sentence_id_display(c.dataset_item))
    ordered = [current] + others
    seen: set = set()
    out: List[InferenceCandidate] = []
    for c in ordered:
        nk = normalize_text(c.text_raw)
        if not nk or nk in seen:
            continue
        seen.add(nk)
        out.append(c)
        if len(out) >= k:
            break
    return out if out else [current]


def speecht5_pooled_speaker_embedding_from_candidates(
    candidates: List[InferenceCandidate],
    *,
    pool_mode: str,
    encoder_sr: int,
    speaker_model: Any,
    device: str,
    full_cfg: Dict[str, Any],
    dataset_profile: str,
    concat_gap_sec: float,
) -> Tuple[torch.Tensor, str, List[InferenceCandidate]]:
    """
    pool_mode: mean — média dos x-vectors (cada clip L2-normalizado pelo encoder); concat — um só forward
    sobre áudios concatenados (silêncio curto entre clipes).
    """
    z = torch.zeros((1, 512), dtype=torch.float32, device=device)
    mode = (pool_mode or "mean").strip().lower()
    waves: List[Tuple[np.ndarray, int]] = []
    used: List[InferenceCandidate] = []
    for c in candidates:
        if not c.dataset_item:
            continue
        y, sr = resolve_reference_waveform_for_inference(
            c.dataset_item,
            encoder_sr,
            full_cfg=full_cfg,
            dataset_profile=dataset_profile,
        )
        if y is None or y.size == 0:
            continue
        waves.append((np.asarray(y, dtype=np.float32).reshape(-1), int(sr)))
        used.append(c)

    if not waves:
        return z, "pool_sem_wav", []

    if mode == "concat":
        gap_n = int(max(0.0, float(concat_gap_sec)) * encoder_sr)
        gap = np.zeros(gap_n, dtype=np.float32) if gap_n > 0 else np.zeros(0, dtype=np.float32)
        pieces: List[np.ndarray] = []
        for j, (w, wsr) in enumerate(waves):
            if int(wsr) != int(encoder_sr):
                w = librosa.resample(w, orig_sr=int(wsr), target_sr=int(encoder_sr)).astype(np.float32)
            pieces.append(np.asarray(w, dtype=np.float32).reshape(-1))
            if j < len(waves) - 1 and gap.size > 0:
                pieces.append(gap.copy())
        y_all = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        pk = float(np.max(np.abs(y_all))) if y_all.size else 0.0
        print(
            f"   📊 x-vector concat: {len(used)} clip(s), ~{y_all.size / max(encoder_sr, 1):.2f}s @ {encoder_sr} Hz, peak|x|={pk:.5f}"
        )
        emb = speecht5_speaker_embedding_from_numpy(y_all, encoder_sr, encoder_sr, speaker_model, device)
        return emb, f"x-vector_concat_{len(used)}_clips", used

    row_vecs: List[torch.Tensor] = []
    for w, wsr in waves:
        e = speecht5_speaker_embedding_from_numpy(w, wsr, encoder_sr, speaker_model, device)
        row_vecs.append(e.detach().squeeze(0).float().cpu())
    stack = torch.stack(row_vecs, dim=0)
    m = stack.mean(dim=0)
    emb = torch.nn.functional.normalize(m.unsqueeze(0), dim=-1, eps=1e-12).to(device=device, dtype=torch.float32)
    print(f"   📊 x-vector média: {len(used)} clip(s) (modo mean), L2 após média aritmética.")
    return emb, f"x-vector_mean_{len(used)}_clips", used


_NEUTRAL_SPK_KEYS: Dict[Tuple[Any, torch.dtype, int], torch.Tensor] = {}


def neutral_speaker_unit_vector_like(emb: torch.Tensor) -> torch.Tensor:
    """Vector unitário fixo determinístico (mesma dim.) — ancoragem estável ao misturar com o x-vector."""
    d = int(emb.shape[-1])
    key = (str(emb.device), emb.dtype, d)
    if key not in _NEUTRAL_SPK_KEYS:
        v = torch.ones((1, d), dtype=emb.dtype, device=emb.device)
        _NEUTRAL_SPK_KEYS[key] = torch.nn.functional.normalize(v, dim=-1, eps=1e-12)
    return _NEUTRAL_SPK_KEYS[key]


def blend_speecht5_speaker_embedding(
    emb: torch.Tensor,
    mix: float,
    *,
    mode: str = "toward_neutral",
) -> torch.Tensor:
    """
    - toward_neutral (por defeito): emb_out = normalize((1−w)·emb + w·µ); µ=[1,...,1]/||·|| — mantém ||
      unitário quando w∈(0,1]; evita aplastar como w·emb (x-vector já L2‑normalizado) que causa síntese mudas.
    - linear_gain (legado): emb_out = w·emb — w baixo implica norma entrada ≪ 1 (= treino típico) e pode colapsar para mudo.
    """
    w = float(max(0.0, min(1.0, mix)))
    md = mode.strip().lower() if isinstance(mode, str) else "toward_neutral"
    if md == "linear_gain":
        return emb * w
    if md != "toward_neutral":
        raise ValueError(f"mode desconhecido: {mode!r} (usa toward_neutral ou linear_gain)")
    if w <= 0.0:
        return emb
    if w >= 1.0:
        return neutral_speaker_unit_vector_like(emb)
    neu = neutral_speaker_unit_vector_like(emb)
    v = (1.0 - w) * emb + w * neu
    return torch.nn.functional.normalize(v, dim=-1, eps=1e-12)


def _pick_text_from_item(item: Dict) -> str:
    for key in ["text", "txt", "sentence", "transcription"]:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""

def _merge_unique_texts(*groups: List[str]) -> List[str]:
    merged = []
    seen = set()
    for group in groups:
        for text in group:
            raw = (text or "").strip()
            if not raw:
                continue
            # Deduplicar por versão normalizada evita frases iguais com acentuação/pontuação diferente.
            key = normalize_text(raw)
            if key in seen:
                continue
            seen.add(key)
            merged.append(raw)
    return merged


def load_test_exported_embedding_cycle(embed_dir: str, device: str) -> Tuple[List[str], List[torch.Tensor]]:
    """Carrega spk_*.npy (512-D) na ordem de manifest.test_speakers ou M031…M034."""
    root = Path(embed_dir)
    if not root.is_dir():
        print(f"❌ --use-test-embeddings: pasta inexistente: {root.resolve()}", file=sys.stderr)
        sys.exit(1)
    order: List[str] = ["M031", "M032", "M033", "M034"]
    manifest_path = root / "manifest.json"
    npy_names: List[str] = []
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        order = list(data.get("test_speakers") or order)
        spk_info = data.get("speakers") or {}
        for spk in order:
            info = spk_info.get(spk) or {}
            npy_names.append(str(info.get("npy") or f"spk_{spk}.npy"))
    else:
        for spk in order:
            npy_names.append(f"spk_{spk}.npy")
    tensors: List[torch.Tensor] = []
    for fn in npy_names:
        path = root / fn
        if not path.is_file():
            print(f"❌ --use-test-embeddings: falta {path.resolve()}", file=sys.stderr)
            sys.exit(1)
        arr = np.load(str(path), allow_pickle=False).astype(np.float32).reshape(-1)
        if arr.size != 512:
            print(f"❌ {path.name}: esperado vetor 512-D, shape={arr.shape}", file=sys.stderr)
            sys.exit(1)
        tensors.append(torch.from_numpy(arr.copy()).unsqueeze(0).to(device=device, dtype=torch.float32))
    return order, tensors


def _default_texts_prefix_alignment_mask(test_texts: List[str], defaults_unique: List[str]) -> List[bool]:
    """True em i só se test_texts[i] coincidir com o i-ésimo default após _merge_unique_texts(default_texts)."""
    out: List[bool] = []
    for i in range(len(test_texts)):
        ok = (
            i < len(defaults_unique)
            and normalize_text((test_texts[i] or "").strip())
            == normalize_text((defaults_unique[i] or "").strip())
        )
        out.append(ok)
    return out


def _random_pick_texts(texts: List[str], limit: int, seed: Optional[int] = None) -> List[str]:
    clean_unique = _merge_unique_texts(texts)
    if limit <= 0 or not clean_unique:
        return []
    if len(clean_unique) <= limit:
        return clean_unique
    if seed is None:
        return random.SystemRandom().sample(clean_unique, limit)
    return random.Random(seed).sample(clean_unique, limit)


@dataclass
class InferenceCandidate:
    """Texto da amostra + item do dataset HF (preserva áudio de referência, se existir)."""

    text_raw: str
    dataset_item: Optional[Dict[str, Any]] = None


def random_pick_inference_candidates(
    candidates: List[InferenceCandidate],
    limit: int,
    seed: Optional[int] = None,
    *,
    take_all: bool = False,
) -> List[InferenceCandidate]:
    """Deduplica por texto normalizado (mantém o primeiro candidato); depois amostragem até `limit`, ou tudo se take_all."""
    uniq_order: Dict[str, InferenceCandidate] = {}
    for c in candidates:
        key = normalize_text(c.text_raw)
        if not key or key in uniq_order:
            continue
        uniq_order[key] = c
    pool = list(uniq_order.values())
    if take_all:
        return pool[:]
    if limit <= 0 or not pool:
        return []
    if len(pool) <= limit:
        return pool[:]
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    return rng.sample(pool, limit)


def print_inference_sample_banner(candidates: List[InferenceCandidate], heading: str) -> None:
    """Lista locutor + id LapsBM (ex.: M034, LapsBM-0678) para o subset aleatório de teste."""
    print(f"   {heading}")
    if not candidates:
        print("      (vazio)")
        return
    for j, c in enumerate(candidates, 1):
        if not c.dataset_item:
            print(f"      [{j}] — (sem linha HF / só texto)")
            continue
        spk = extract_speaker_id(c.dataset_item)
        sid = format_sentence_id_display(c.dataset_item)
        print(f"      [{j}] locutor={spk} | sentença={sid}")


def _hf_row_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return deepcopy(item)
    try:
        return dict(item)
    except Exception:
        return {}


def resolve_reference_waveform_to_mono_sr(
    item: Optional[Dict[str, Any]], target_sr: int
) -> tuple[Optional[np.ndarray], int]:
    """Devolve waveform mono em target_sr ou None se não houver áudio utilizável."""
    if not item:
        return None, target_sr
    au = item.get("audio") or item.get("wav")
    if isinstance(au, dict):
        arr = au.get("array")
        sr = au.get("sampling_rate") or target_sr
        if arr is not None:
            y = np.asarray(arr, dtype=np.float32).reshape(-1)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            return np.asarray(y, dtype=np.float32), int(target_sr)
        path = au.get("path")
        if path and isinstance(path, str) and os.path.isfile(path):
            try:
                y, _sr = librosa.load(path, sr=int(target_sr), mono=True)
                return np.asarray(y, dtype=np.float32), int(target_sr)
            except Exception:
                return None, target_sr
    return None, target_sr


def load_inference_test_candidates_exported(full_cfg: Dict, dataset_name: str) -> List[InferenceCandidate]:
    local_root = full_cfg.get("settings", {}).get("local_datasets_dir", "./datasets")
    exported_path = os.path.join(local_root, "test_split_inferencia", dataset_name, "hf_test_split")
    if not os.path.exists(exported_path):
        return []

    try:
        ds = load_from_disk(exported_path)
    except Exception as e:
        print(f"⚠️ Não foi possível abrir split exportado em '{exported_path}': {e}")
        return []

    out: List[InferenceCandidate] = []
    n = len(ds)
    for i in range(n):
        row = ds[i]
        dct = _hf_row_to_dict(row)
        if "wav" in dct and "audio" not in dct:
            dct["audio"] = dct["wav"]
        if "txt" in dct and "text" not in dct:
            dct["text"] = dct["txt"]
        inferred = _speaker_id_from_audio_path_dict(dct)
        if inferred and not dct.get("speaker_id"):
            dct["speaker_id"] = inferred
        txt = _pick_text_from_item(dct)
        if txt:
            out.append(InferenceCandidate(txt, dct))
    return out


def load_inference_candidates_from_stream(
    full_cfg: Dict,
    ds_cfg: Dict,
    dataset_name: str,
    *,
    speaker_subset: str,
) -> List[InferenceCandidate]:
    """Índices HF locais como no train_exaustivo: 'train', 'test' ou 'val' segundo zero_shot_split por locutor."""
    subset = str(speaker_subset or "test").strip().lower()
    if subset not in {"train", "val", "test"}:
        raise ValueError(f"speaker_subset deve ser train|val|test, não {speaker_subset!r}")

    dataset_id = ds_cfg.get("dataset_id")
    if not dataset_id:
        return []

    local_root = full_cfg.get("settings", {}).get("local_datasets_dir", "./datasets")
    repo_name = dataset_id.split("/")[-1]
    local_ds_path = os.path.join(local_root, repo_name)
    is_local = os.path.exists(local_ds_path) and bool(os.listdir(local_ds_path)) if os.path.exists(local_ds_path) else False

    print(
        f"📥 Dataset '{dataset_id}' subset locutor [{subset}] (HF split `{ds_cfg.get('dataset_split', 'train')}`; Local: {is_local})...",
    )

    all_data: List[Any] = []
    try:
        if dataset_id == "firstpixel/pt-br_char":
            import pandas as pd
            from huggingface_hub import hf_hub_download

            if is_local:
                csv_path = os.path.join(local_ds_path, "metadata.csv")
            else:
                csv_path = hf_hub_download(repo_id=dataset_id, filename="metadata.csv", repo_type="dataset")

            df = pd.read_csv(csv_path, sep="|")
            for _, row in df.iterrows():
                all_data.append({"text": str(row.iloc[1]), "speaker_id": "default"})
        else:
            load_kwargs = {"split": ds_cfg.get("dataset_split", "train")}
            if is_local:
                dataset_obj = load_from_disk(local_ds_path)
                if isinstance(dataset_obj, dict):
                    split_name = ds_cfg.get("dataset_split", "train")
                    dataset_stream = dataset_obj.get(split_name) or next(iter(dataset_obj.values()))
                else:
                    dataset_stream = dataset_obj
            else:
                load_kwargs["path"] = dataset_id
                load_kwargs["streaming"] = ds_cfg.get("streaming", True)
                if "dataset_config" in ds_cfg:
                    load_kwargs["name"] = ds_cfg["dataset_config"]
                dataset_stream = load_dataset(**load_kwargs)

            for item in dataset_stream:
                item = dict(item) if hasattr(item, "keys") else _hf_row_to_dict(item)
                if "wav" in item and "audio" not in item:
                    item["audio"] = item["wav"]
                if "txt" in item and "text" not in item:
                    item["text"] = item["txt"]
                all_data.append(item)
    except Exception as e:
        print(f"⚠️ Não foi possível carregar dataset para inferência: {e}")
        return []

    if not all_data:
        print("⚠️ Dataset sem amostras utilizáveis.")
        return []

    all_speakers = [extract_speaker_id(x) for x in all_data]
    counts = Counter(all_speakers)
    num_spk = ds_cfg.get("num_speakers", 1)
    max_samples = ds_cfg.get("num_samples_per_speaker", 0)
    zero_shot_split = ds_cfg.get("zero_shot_split", None)

    if num_spk == 0:
        valid_spks = set(all_speakers)
    else:
        valid_spks = set(s for s, _ in counts.most_common(num_spk))

    train_spks: set[str] = set()
    val_spks: set[str] = set()
    test_spks: set[str] = set()
    if zero_shot_split and len(valid_spks) > 0:
        t_spk = zero_shot_split.get("train_speakers", 28)
        v_spk = zero_shot_split.get("val_speakers", 3)
        ts_spk = zero_shot_split.get("test_speakers", 4)

        sorted_spks = sorted(list(valid_spks))
        if len(sorted_spks) >= (t_spk + v_spk + ts_spk):
            train_spks = set(sorted_spks[:t_spk])
            val_spks = set(sorted_spks[t_spk : t_spk + v_spk])
            test_spks = set(sorted_spks[t_spk + v_spk : t_spk + v_spk + ts_spk])
            valid_spks = train_spks | val_spks | test_spks
        else:
            print("⚠️ Poucos locutores para 80/10/10; sem split de teste dedicado.")

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    spk_added_counts = {s: 0 for s in valid_spks}
    for i, (_, s) in enumerate(zip(all_data, all_speakers)):
        if s in valid_spks and (max_samples == 0 or spk_added_counts[s] < max_samples):
            if s in val_spks:
                val_indices.append(i)
            elif s in test_spks:
                test_indices.append(i)
            else:
                train_indices.append(i)
            spk_added_counts[s] += 1

    bucket_map = {"train": train_indices, "val": val_indices, "test": test_indices}
    indices = bucket_map[subset]

    out: List[InferenceCandidate] = []
    for idx in indices:
        d = _hf_row_to_dict(all_data[idx])
        inferred = _speaker_id_from_audio_path_dict(d)
        if inferred and not d.get("speaker_id"):
            d["speaker_id"] = inferred
        txt = _pick_text_from_item(d)
        if txt:
            out.append(InferenceCandidate(txt, d))
    return out


def load_inference_test_candidates_from_stream(full_cfg: Dict, ds_cfg: Dict, dataset_name: str) -> List[InferenceCandidate]:
    return load_inference_candidates_from_stream(
        full_cfg, ds_cfg, dataset_name, speaker_subset="test"
    )


def load_inference_train_candidates_from_stream(full_cfg: Dict, ds_cfg: Dict, dataset_name: str) -> List[InferenceCandidate]:
    return load_inference_candidates_from_stream(
        full_cfg, ds_cfg, dataset_name, speaker_subset="train"
    )


def load_inference_test_candidates(
    full_cfg: Dict,
    ds_cfg: Dict,
    dataset_name: str,
) -> List[InferenceCandidate]:
    exported = load_inference_test_candidates_exported(full_cfg, dataset_name)
    if exported:
        return exported
    return load_inference_test_candidates_from_stream(full_cfg, ds_cfg, dataset_name)


def load_inference_train_candidates(
    full_cfg: Dict,
    ds_cfg: Dict,
    dataset_name: str,
) -> List[InferenceCandidate]:
    """Como train_exaustivo: amostras cujos locutores pertencem ao subset de treino (não há hf_train_split exportado)."""
    return load_inference_train_candidates_from_stream(full_cfg, ds_cfg, dataset_name)


def load_test_sentences_from_dataset(
    full_cfg: Dict,
    ds_cfg: Dict,
    dataset_name: str,
    limit: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[InferenceCandidate]]:
    cands = load_inference_test_candidates(full_cfg, ds_cfg, dataset_name)
    if not cands:
        return [], []

    chosen = random_pick_inference_candidates(cands, limit, seed=seed)
    exported_path = os.path.join(
        full_cfg.get("settings", {}).get("local_datasets_dir", "./datasets"),
        "test_split_inferencia",
        dataset_name,
        "hf_test_split",
    )
    src = "split exportado" if os.path.exists(exported_path) else "subset teste por locutor (80/10/10)"
    print(
        f"   🧪 Sentenças de dataset ({src}): {len(chosen)} | alvo: {limit}"
        + (f" | seed: {seed}" if seed is not None else ""),
    )
    print_inference_sample_banner(chosen, "🧾 Subset aleatório (locutor + sentença):")
    return [c.text_raw for c in chosen], chosen

# ==============================================================================
# INFERENCE HANDLERS
# ==============================================================================

class InferenceHandler:
    def __init__(self, model_cfg: Dict, device: str, model_path: str):
        self.model_cfg = model_cfg
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError()

    def generate(self, text: str, speaker_emb: torch.Tensor):
        raise NotImplementedError()

class SpeechT5Inference(InferenceHandler):
    def load(self):
        actual = find_checkpoint(self.model_path)
        print(f"📥 Carregando SpeechT5 (Base + LoRA) de: {self.model_path}" + (f" → {actual}" if actual != self.model_path else ""))
        self.processor = None
        for p in (self.model_path, actual):
            if os.path.isdir(p):
                try:
                    self.processor = SpeechT5Processor.from_pretrained(p)
                    break
                except OSError:
                    pass
        if self.processor is None:
            self.processor = SpeechT5Processor.from_pretrained(self.model_cfg["id"])
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_cfg['vocoder_id']).to(self.device)
        
        base_model = SpeechT5ForTextToSpeech.from_pretrained(self.model_cfg['id'])
        self.model = PeftModel.from_pretrained(base_model, actual)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, text, speaker_emb, use_lora=True, **kwargs):
        max_ch = int(kwargs.get("speecht5_chunk_max_chars", getattr(self, "chunk_max_chars", 100)))
        gap_sec = float(
            kwargs.get("speecht5_inter_chunk_silence_sec", getattr(self, "inter_chunk_silence_sec", 0.12))
        )
        sr = int(self.model_cfg.get("sampling_rate", 16000))
        chunks = split_text_for_tts(text, max_chars=max_ch)
        chunk_audio = []
        gap_samples = int(max(0.0, gap_sec) * sr)

        for idx, chunk in enumerate(chunks):
            print(f"   (SpeechT5) Parte {idx + 1}/{len(chunks)}: {chunk}...")
            inputs = self.processor(text=chunk, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            # Mesmo dtype que o Peft + SpeechT5 (evita mismatch em embeddings / decoder).
            md = next(self.model.parameters()).dtype
            spk = speaker_emb.to(device=self.device, dtype=md)
            with torch.no_grad():
                if use_lora:
                    wave = self.model.generate(
                        input_ids,
                        speaker_embeddings=spk,
                        vocoder=self.vocoder,
                    )
                else:
                    with self.model.disable_adapter():
                        wave = self.model.generate(
                            input_ids,
                            speaker_embeddings=spk,
                            vocoder=self.vocoder,
                        )
            audio_piece = wave.squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32)
            chunk_audio.append(audio_piece)
            if idx < len(chunks) - 1 and gap_samples > 0:
                chunk_audio.append(np.zeros(gap_samples, dtype=np.float32))

        if not chunk_audio:
            return np.asarray([], dtype=np.float32)

        return np.concatenate(chunk_audio)

class F5Inference(InferenceHandler):
    def load(self):
        try:
            from f5_tts.model import DiT, CFM
            from f5_tts.model.utils import get_tokenizer
            from vocos import Vocos
        except ImportError:
            print("❌ Erro: Requer 'pip install f5-tts vocos'")
            sys.exit(1)

        print(f"📥 Carregando F5-TTS de: {self.model_path}")
        self.tokenizer = get_tokenizer("pinyin")
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        
        # Base DiT
        dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_num_embeds=256, mel_dim=80)
        
        # Load LoRA
        self.model = PeftModel.from_pretrained(dit, self.model_path)
        self.model.to(self.device).eval()
        
        # CFM sampler
        self.cfm = CFM(transformer=self.model, sigma=0.0).to(self.device)

    def generate(self, text, speaker_emb, use_lora=True, **kwargs):
        print(f"   (F5-TTS) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        text_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        with torch.no_grad():
            cond_mel = torch.zeros((1, 5, 80)).to(self.device) 
            if use_lora:
                sampled_mel, _ = self.cfm.sample(cond_mel, text_ids, duration=200, steps=32)
            else:
                with self.model.disable_adapter():
                    sampled_mel, _ = self.cfm.sample(cond_mel, text_ids, duration=200, steps=32)
            audio_out = self.vocos.decode(sampled_mel.transpose(1, 2))
        return audio_out.squeeze().cpu().numpy()

class XTTSInference(InferenceHandler):
    def load(self):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError:
            print("❌ Erro: Requer 'pip install TTS'")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2 de: {self.model_path}")
        config = XttsConfig()
        self.xtts = Xtts(config)
        self.model = PeftModel.from_pretrained(self.xtts.gpt, self.model_path)
        self.xtts.gpt = self.model # Substitui o gpt pelo PeftModel
        self.xtts.to(self.device).eval()

    def generate(self, text, speaker_emb, use_lora=True, **kwargs):
        print(f"   (XTTS v2) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        with torch.no_grad():
            if use_lora:
                output = self.xtts.inference(
                    text=text,
                    language="pt",
                    gpt_cond_latent=speaker_emb[0],
                    speaker_embedding=speaker_emb[1],
                    temperature=0.7
                )
            else:
                with self.model.disable_adapter():
                    output = self.xtts.inference(
                        text=text,
                        language="pt",
                        gpt_cond_latent=speaker_emb[0],
                        speaker_embedding=speaker_emb[1],
                        temperature=0.7
                    )
            audio_out = output["wav"]
        return audio_out

class FastSpeech2Inference(InferenceHandler):
    def load(self):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.forward_tts import ForwardTTS as FastSpeech2
            from TTS.tts.configs.fastspeech2_config import Fastspeech2Config as FastSpeech2Config
            from TTS.vocoder.models.gan import GAN as HifiGan
            from TTS.utils.audio import AudioProcessor
        except ImportError:
            print("❌ Erro: Requer 'pip install coqui-tts'")
            sys.exit(1)

        mp = os.path.abspath(os.path.normpath(self.model_path))
        if not os.path.exists(mp):
            print(
                f"❌ Caminho do modelo/checkpoint inexistente:\n   {mp}\n"
                "   Use a pasta *real* do run (ex. output_cuda_16gb\\fastspeech2-lapsbm_fastspeech2-AAAA-MM-DD-...\\checkpoint-1000), "
                "não '...' nem atalhos de exemplo."
            )
            sys.exit(1)
        self.model_path = mp

        print(f"📥 Carregando FastSpeech 2 de: {self.model_path}")
        config_path = os.path.join(self.model_path, "config.json")
        model_file = os.path.join(self.model_path, "model.pth") # Caso padrão
        
        if not os.path.exists(config_path):
            print(f"   🔍 Config não encontrada em {self.model_path}, baixando modelo base...")
            manager = ModelManager()
            model_base_path = manager.download_model(self.model_cfg['id'])
            if isinstance(model_base_path, tuple): model_base_path = model_base_path[0]
            
            if os.path.isfile(model_base_path):
                model_dir = os.path.dirname(model_base_path)
                model_file = model_base_path
            else:
                model_dir = model_base_path
                model_file = os.path.join(model_dir, "model.pth")
                
            config_path = os.path.join(model_dir, "config.json")
            
        config = FastSpeech2Config()
        config.load_json(config_path)
        mcfg = self.model_cfg.get("pitch_loss_alpha", None)
        if mcfg is not None:
            config.pitch_loss_alpha = float(mcfg)
        # Mesmo processamento de espectro do treino (LJS: signal_norm false → mel em dB, sem [-4,4] manual)
        self.ap = AudioProcessor.init_from_config(config)
        base_model = FastSpeech2.init_from_config(config)
        
        # Tentar carregar o checkpoint base antes de aplicar o Peft
        if os.path.exists(model_file):
            print(f"   基础 (Base) model weights: {model_file}")
            base_model.load_checkpoint(config, checkpoint_path=model_file)
            
        # 🔍 Lógica de Detecção de Checkpoint
        actual_model_path = find_checkpoint(self.model_path)

        # Carregar LoRA (tentando PeftModel.from_pretrained primeiro, depois fallback manual)
        if os.path.exists(os.path.join(actual_model_path, "adapter_config.json")):
            print(f"   ✅ Carregando adaptadores LoRA via PEFT de {actual_model_path}")
            self.model = PeftModel.from_pretrained(base_model, actual_model_path)
        else:
            print("   ⚠️ adapter_config.json não encontrado. Tentando reconstrução manual do LoRA...")
            from peft import LoraConfig, get_peft_model
            lora_cfg = self.model_cfg['lora']
            peft_config = LoraConfig(
                r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
                target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
                bias="none"
            )
            self.model = get_peft_model(base_model, peft_config)
            
            # Carregar pesos do state_dict (Safetensors ou PyTorch Bin)
            weights_file = None
            for wf in [
                "adapter_model.safetensors",
                "model.safetensors",
                "pytorch_model.bin",
                "adapter_model.bin",
            ]:
                if os.path.exists(os.path.join(actual_model_path, wf)):
                    weights_file = os.path.join(actual_model_path, wf)
                    break
            
            if weights_file:
                print(f"   📥 Carregando pesos de: {os.path.basename(weights_file)}")
                if weights_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_file)
                else:
                    state_dict = torch.load(weights_file, map_location=self.device)
                
                # Limpeza cirúrgica de prefixos
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    # O Trainer salva com 'model.base_model.model...', 
                    # o PeftModel espera 'base_model.model...'
                    if name.startswith("model.base_model.model."): 
                        name = name[6:] # Remove apenas o primeiro 'model.'
                    new_state_dict[name] = v
                
                # Carregar pesos no PeftModel
                missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                loaded_keys = len(new_state_dict) - len(unexpected)
                print(f"   ✨ Sucesso: {loaded_keys} tensores carregados (LoRA + Base).")
                if unexpected:
                    print(f"   ℹ️ {len(unexpected)} chaves ignoradas (provavelmente optimizers ou metadados).")
            else:
                print(f"   ❌ ERRO: Nenhum arquivo de pesos PEFT/Trainer em {actual_model_path}")
                if os.path.isdir(actual_model_path):
                    print(f"      Ficheiros na pasta: {os.listdir(actual_model_path)}")
                else:
                    print("      (a pasta não existe — verifica o caminho passado a --model_path)")
                sys.exit(1)

        self.model.to(self.device).eval()
        
        # Load Vocoder
        manager = ModelManager()
        vocoder_res = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
        if isinstance(vocoder_res, tuple): vocoder_res = vocoder_res[0]
        
        if os.path.isfile(vocoder_res):
            vocoder_dir = os.path.dirname(vocoder_res)
            vocoder_file = vocoder_res
        else:
            vocoder_dir = vocoder_res
            vocoder_file = os.path.join(vocoder_dir, "model.pth")
            
        from TTS.vocoder.configs.hifigan_config import HifiganConfig as HifiGanConfig
        v_config = HifiGanConfig()
        
        # Carregar config limpando comentários (// e /* */) que o JSON padrão não suporta
        import re
        import json
        with open(os.path.join(vocoder_dir, "config.json"), 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        v_config.from_dict(json.loads(content))

        self.vocoder = HifiGan.init_from_config(v_config)
        self.vocoder.load_checkpoint(v_config, checkpoint_path=vocoder_file)
        self.vocoder.to(self.device).eval()

        self.tokenizer = base_model.tokenizer
        # Phonemizer/token default do LJS (EN) no *mesmo* model.pth — antes de Gruut; usado só em _ljs_english.wav
        self._phonemizer_ljs_en = self.tokenizer.phonemizer
        self._phonemizer_pt: Optional[object] = None

        # Tentar trocar para o phonemizer de PT se possível (como feito no treino)
        try:
            from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name
            base_phonemizer = get_phonemizer_by_name("gruut", language="pt")
            
            # Criamos um wrapper para mapear fonemas do PT que não existem no modelo Base (Inglês)
            class PTPhonemizerWrapper:
                def __init__(self, p): self.p = p
                def phonemize(self, text, separator="", language="pt"):
                    ph = self.p.phonemize(text, separator=separator, language=language)
                    # Mapeamento de Precisão (Vogal + n) usando 'ɐ' para o som de 'ã' fechado.
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
            self.tokenizer.phonemizer = new_phonemizer
            self._phonemizer_pt = new_phonemizer
            print(f"   🌐 Phonemizer do FastSpeech 2 substituído por Gruut (pt) com mapeamento cross-lingual")
        except Exception as e:
            print(f"   ⚠️ Aviso: Não foi possível trocar o phonemizer: {e}")
            print(f"   ℹ️  Baselines 'pt' usarão o phonemizer ainda ativo (p. ex. stock EN).")

    def _forward_tts_module(self):
        """Peft/ForwardTTS: onde vivem inference() e length_scale (Coqui)."""
        m = self.model
        for _ in range(16):
            if m is None:
                return None
            if hasattr(m, "inference") and hasattr(m, "length_scale") and hasattr(m, "format_durations"):
                return m
            m = getattr(m, "base_model", None) or getattr(m, "model", None)
        return None

    def set_length_scale(self, scale: Optional[float]):
        """>1.0 = mais lento/mais longo (Coqui: multiplica durações em format_durations). None = não altera."""
        if scale is None:
            return
        s = float(scale)
        core = self._forward_tts_module()
        if core is not None:
            core.length_scale = s
            print(f"   📏 length_scale = {s} (duração ≈ fator nas durações previstas)")

    def _mel_to_wav(self, mel_output, n_tokens, ap, length_scale_from=None):
        """Ajusta shape do mel, log de métricas, HiFi-GAN. `length_scale_from`: módulo com atributo .length_scale."""
        if mel_output.ndim == 3:
            _, a, b = mel_output.shape
            if a != 80 and b in (80, 90):
                mel_output = mel_output.transpose(1, 2)
        if mel_output.shape[1] > 80:
            mel_output = mel_output[:, :80, :]
        elif mel_output.shape[1] < 80:
            mel_output = torch.nn.functional.pad(mel_output, (0, 0, 0, 80 - mel_output.shape[1]))
        n_frames = int(mel_output.shape[-1])
        hop = int(getattr(ap, "hop_length", 256))
        sr = int(getattr(ap, "sample_rate", 22050))
        est_s = n_frames * hop / max(sr, 1)
        ls = None
        if length_scale_from is not None and hasattr(length_scale_from, "length_scale"):
            ls = float(length_scale_from.length_scale)
        _ls = f", length_scale={ls}" if ls is not None else ""
        print(
            f"   📐 tokens_fon/ids={n_tokens} | mel_frames={n_frames} | ≈{est_s:.2f}s @ {sr}Hz hop={hop}{_ls} "
            f"(duração vem do duration predictor; duration_loss no treino ≠ segundos)"
        )
        return self.vocoder.inference(mel_output).squeeze().cpu().numpy()

    def generate(
        self,
        text,
        speaker_emb,
        use_lora=True,
        fastspeech2_pipeline: Optional[str] = None,
        **kwargs,
    ):
        # ljs_en:  model.pth + EN (sem LoRA)
        # pt_base: model.pth + PT (Gruut), sem LoRA
        # pt_lora: LoRA + PT (Gruut)
        if fastspeech2_pipeline is None:
            fastspeech2_pipeline = "pt_lora" if use_lora else "pt_base"
        if fastspeech2_pipeline == "ljs_en":
            label = "model.pth, pipeline EN (sem LoRA)"
            print(f"   (FastSpeech 2) {label}: {text[:30]}...")
            self.tokenizer.phonemizer = self._phonemizer_ljs_en
            input_ids = torch.tensor(
                [self.tokenizer.text_to_ids(text, language="en")]
            ).to(self.device)
            n_tokens = int(input_ids.shape[1])
            with torch.no_grad():
                with self.model.disable_adapter():
                    outputs = self.model.inference(
                        input_ids, aux_input={"d_vectors": None, "speaker_ids": None}
                    )
            return self._mel_to_wav(
                outputs["model_outputs"],
                n_tokens,
                self.ap,
                length_scale_from=self._forward_tts_module(),
            )
        if fastspeech2_pipeline == "pt_base":
            if self._phonemizer_pt is not None:
                self.tokenizer.phonemizer = self._phonemizer_pt
            lang = "pt"
            do_lora = False
            label = "model.pth + PT (tokenizer + Gruut), sem LoRA"
        elif fastspeech2_pipeline == "pt_lora":
            if self._phonemizer_pt is not None:
                self.tokenizer.phonemizer = self._phonemizer_pt
            lang = "pt"
            do_lora = True
            label = "LoRA + PT (Gruut)"
        else:
            raise ValueError(
                f"fastspeech2_pipeline desconhecido: {fastspeech2_pipeline!r} "
                "(use 'ljs_en', 'pt_base', 'pt_lora' ou omita e use use_lora=True/False)."
            )

        print(f"   (FastSpeech 2) {label}: {text[:30]}...")
        input_ids = torch.tensor([self.tokenizer.text_to_ids(text, language=lang)]).to(self.device)
        n_tokens = int(input_ids.shape[1])
        with torch.no_grad():
            if do_lora:
                outputs = self.model.inference(
                    input_ids, aux_input={"d_vectors": None, "speaker_ids": None}
                )
            else:
                with self.model.disable_adapter():
                    outputs = self.model.inference(
                        input_ids, aux_input={"d_vectors": None, "speaker_ids": None}
                    )
        return self._mel_to_wav(
            outputs["model_outputs"],
            n_tokens,
            self.ap,
            length_scale_from=self._forward_tts_module(),
        )

class MatchaInference(InferenceHandler):
    def load(self):
        try:
            from matcha.models.matcha_tts import MatchaTTS
        except ImportError:
            print("❌ Erro: Requer 'pip install matcha-tts'")
            sys.exit(1)
        print(f"📥 Carregando Matcha-TTS de: {self.model_path}")
        pass
    def generate(self, text, speaker_emb, use_lora=True, **kwargs):
        print(f"   (Matcha-TTS) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        return np.zeros(16000)

class GlowTTSInference(FastSpeech2Inference):
    def load(self):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.glow_tts import GlowTTS
            from TTS.tts.configs.glow_tts_config import GlowTTSConfig
            from TTS.vocoder.models.gan import GAN as HifiGan
        except ImportError:
            print("❌ Erro: Requer 'pip install coqui-tts'")
            sys.exit(1)
        print(f"📥 Carregando Glow-TTS de: {self.model_path}")
        manager = ModelManager()
        model_base_path = manager.download_model(self.model_cfg['id'])
        if isinstance(model_base_path, tuple): model_base_path = model_base_path[0]
        config = GlowTTSConfig()
        config.load_json(os.path.join(model_base_path, "config.json"))
        base_model = GlowTTS.init_from_config(config)
        self.model = PeftModel.from_pretrained(base_model.encoder, self.model_path)
        self.model.to(self.device).eval()
        
        # Load Vocoder
        manager = ModelManager()
        vocoder_res = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
        if isinstance(vocoder_res, tuple): vocoder_res = vocoder_res[0]
        
        if os.path.isfile(vocoder_res):
            vocoder_dir = os.path.dirname(vocoder_res)
            vocoder_file = vocoder_res
        else:
            vocoder_dir = vocoder_res
            vocoder_file = os.path.join(vocoder_dir, "model.pth")
            
        from TTS.vocoder.configs.hifigan_config import HifiganConfig as HifiGanConfig
        v_config = HifiGanConfig()
        
        # Carregar config limpando comentários (// e /* */) que o JSON padrão não suporta
        import re
        import json
        with open(os.path.join(vocoder_dir, "config.json"), 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        v_config.from_dict(json.loads(content))

        from TTS.vocoder.models.gan import GAN as HifiGan
        self.vocoder = HifiGan.init_from_config(v_config)
        self.vocoder.load_checkpoint(v_config, checkpoint_path=vocoder_file)
        self.vocoder.to(self.device).eval()
        self.tokenizer = base_model.tokenizer

    def generate(self, text, speaker_emb, use_lora=True, **kwargs):
        print(f"   (Glow-TTS) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        input_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        with torch.no_grad():
            if use_lora:
                outputs = self.model.inference(input_ids)
            else:
                with self.model.disable_adapter():
                    outputs = self.model.inference(input_ids)
            mel_output = outputs["model_outputs"]
            
            # Ajustar dimensões para o vocoder (espera [B, C, T])
            if mel_output.ndim == 3:
                # Se for [B, T, C], transpõe para [B, C, T]
                if mel_output.shape[2] == 80 or mel_output.shape[2] == 90:
                    mel_output = mel_output.transpose(1, 2)
            
            # Garantir 80 canais
            if mel_output.shape[1] > 80:
                mel_output = mel_output[:, :80, :]
            elif mel_output.shape[1] < 80:
                mel_output = torch.nn.functional.pad(mel_output, (0, 0, 0, 80 - mel_output.shape[1]))

            audio_out = self.vocoder.inference(mel_output)
        return audio_out.squeeze().cpu().numpy()

def get_inference_handler(model_type, model_cfg, device, model_path):
    if model_type == 'speecht5': return SpeechT5Inference(model_cfg, device, model_path)
    elif model_type == 'f5_tts': return F5Inference(model_cfg, device, model_path)
    elif model_type == 'fastspeech2': return FastSpeech2Inference(model_cfg, device, model_path)
    elif model_type == 'matcha': return MatchaInference(model_cfg, device, model_path)
    elif model_type == 'glow_tts': return GlowTTSInference(model_cfg, device, model_path)
    elif model_type == 'xtts_v2': return XTTSInference(model_cfg, device, model_path)
    else:
        print(f"⚠️ Inferência para {model_type} não implementada.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--model_dir", type=str, default=None, help="Pasta do modelo treinado (ex: ./output_cuda_8gb/fastspeech2-...)")
    parser.add_argument("--text", type=str, default=None, help="Texto customizado para gerar áudio")
    parser.add_argument(
        "--length_scale",
        type=float,
        default=None,
        help="FastPitch/Coqui: multiplica durações previstas (>1 = áudio mais longo). Ex.: 1.15 – 1.35 se sair tudo ~1s.",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--apply_peak_normalization",
        action="store_true",
        help="Aplica normalização de pico antes de salvar WAV. Padrão: desativado.",
    )
    parser.add_argument(
        "--test_text_seed",
        type=int,
        default=None,
        help="Seed opcional para tornar reproduzível a seleção das sentenças de teste.",
    )
    parser.add_argument(
        "--use-test-embeddings",
        action="store_true",
        dest="use_test_embeddings",
        help=(
            "SpeechT5: linhas alinhadas ao prefixo default_texts (ordem _merge_unique_texts só dos defaults) "
            "usam x-vectors em --use-test-embeddings-dir (ciclo: 1→M031, 2→M032, 3→M033, 4→M034, 5→M031, …)."
        ),
    )
    parser.add_argument(
        "--use-test-embeddings-dir",
        type=str,
        default="./embeddings_test_speakers",
        dest="use_test_embeddings_dir",
        help="Pasta com manifest.json + spk_*.npy (export_test_speaker_embeddings.py).",
    )
    parser.add_argument(
        "--trained_bass_cut_hz",
        type=float,
        default=120.0,
        metavar="HZ",
        help="Atenuação de graves: frequência até onde a banda grave é atenuada (Hz). Só com --apply_trained_bass_treatment.",
    )
    parser.add_argument(
        "--trained_bass_atten_db",
        type=float,
        default=1.2,
        metavar="DB",
        help="Atenuação de graves em dB (abaixo de --trained_bass_cut_hz). Só com --apply_trained_bass_treatment.",
    )
    parser.add_argument(
        "--trained_bass_transition_hz",
        type=float,
        default=120.0,
        metavar="HZ",
        help="Largura da rampa (Hz) entre zona atenuada e zona intacta na curva de graves.",
    )
    parser.add_argument(
        "--trained_grave_highpass_hz",
        type=float,
        default=0.0,
        metavar="HZ",
        help=(
            "High-pass extra (Hz) aplicado só ao *_treinado após o HP global SpeechT5; 0 = desligado. "
            "Corte de subgrave mais duro que a curva de atenuação."
        ),
    )
    parser.add_argument(
        "--trained_treble_shelf_hz",
        type=float,
        default=3500.0,
        metavar="HZ",
        help="Início da prateleira de agudos (Hz) quando --trained_treble_boost_db > 0.",
    )
    parser.add_argument(
        "--trained_treble_transition_hz",
        type=float,
        default=1000.0,
        metavar="HZ",
        help="Largura da rampa (Hz) até o ganho de agudos ficar pleno.",
    )
    parser.add_argument(
        "--trained_treble_boost_db",
        type=float,
        default=0.0,
        metavar="DB",
        help="Ganho em agudos (dB) só no áudio treinado; 0 = desliga (não altera o sinal).",
    )
    parser.add_argument(
        "--apply_trained_bass_treatment",
        action="store_true",
        help="Ativa atenuação suave de graves (--trained_bass_*). Áudio treinado (LoRA) apenas.",
    )
    parser.add_argument(
        "--apply_silence_cleanup",
        action="store_true",
        help="Remove/comprime silêncios no áudio gerado pelo SpeechT5. Padrão: desativado.",
    )
    parser.add_argument(
        "--silence_top_db",
        type=float,
        default=35.0,
        help="Limiar top_db (effects.split/trim) quando --apply_silence_cleanup está ativo. Default: 35.",
    )
    parser.add_argument(
        "--speecht5_inference_highpass_hz",
        type=float,
        default=75.0,
        metavar="HZ",
        help=(
            "SpeechT5 após vocoder: high-pass (~Butterworth) contra rumor subgrave/DC; "
            "0 desativa. Default: 75."
        ),
    )
    parser.add_argument(
        "--only_f0_rmse_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Só calcula F0 RMSE (treinado vs original) na pasta e grava f0_rmse.csv; não carrega modelo nem gera áudio.",
    )
    parser.add_argument(
        "--compute_f0_rmse",
        action="store_true",
        help="Após inferência SpeechT5: calcula F0 RMSE na pasta de saída (treinado vs base sintético).",
    )
    parser.add_argument("--f0_sample_rate", type=int, default=16000, help="SR para YIN (deve bater com o WAV de inferência).")
    parser.add_argument("--f0_hop_length", type=int, default=256, help="hop_length do librosa.yin (reprodutibilidade).")
    parser.add_argument("--f0_fmin", type=float, default=50.0)
    parser.add_argument("--f0_fmax", type=float, default=500.0)
    parser.add_argument(
        "--dataset_reference_audios",
        action="store_true",
        help=(
            "SpeechT5: subset de locutores de teste (holdout) + WAV *_referencia; x-vector no mesmo áudio que o treino."
        ),
    )
    parser.add_argument(
        "--dataset_sentence_limit",
        type=int,
        default=5,
        help=(
            "Com --dataset_reference_audios: máx. frases do bloco teste. Com treino também activo: 5+5=10 por defeito. "
            "É ignorado se --infer_all_test_sentences."
        ),
    )
    parser.add_argument(
        "--infer_all_test_sentences",
        action="store_true",
        help=(
            "SpeechT5 + --dataset_reference_audios: inferir sobre todas as frases do conjunto de teste (sem subamostragem). "
            "Ignora --dataset_train_reference_audios neste run. Não combinar com --text."
        ),
    )
    parser.add_argument(
        "--dataset_train_reference_audios",
        action="store_true",
        help=(
            "SpeechT5: acrescenta frases de locutores do subset **treino** (HF/cache; não é o hf_test_split exportado). "
            "Não substitui --dataset_reference_audios: em conjunto fica teste primeiro + treino depois (ex.: 5+5=10)."
        ),
    )
    parser.add_argument(
        "--dataset_train_sentence_limit",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Frases do bloco treino. Default 5 ⇒ com --dataset_reference_audios são 5 teste + 5 treino (=10 frases)."
        ),
    )
    parser.add_argument(
        "--no_mp3",
        action="store_true",
        help="Não gera cópias .mp3 (por defeito: WAV + MP3 quando pydub/ffmpeg estão disponíveis).",
    )
    parser.add_argument(
        "--mp3_bitrate",
        type=str,
        default="192k",
        metavar="RATE",
        help="Bitrate ao exportar MP3 (ex.: 192k, 128k). Requer pydub + ffmpeg.",
    )
    parser.add_argument(
        "--speecht5_zero_speaker_embedding",
        action="store_true",
        help=(
            "SpeechT5: não carrega SpeechBrain; usa embedding nulo (1×512), como antes. "
            "Útil para comparar com x-vector; métricas F0 vs referência perdem sentido."
        ),
    )
    parser.add_argument(
        "--speecht5_speaker_emb_mix",
        type=float,
        default=0.0,
        metavar="W",
        help=(
            "SpeechT5: valor interpretado segundo --speecht5_speaker_emb_mix_mode. "
            "toward_neutral (por defeito): emb = normalize((1−w)·x-vector + w·µ) com µ vetor unitário fixo; "
            "w∈[0,1], w=0 sem mistura (igual ao treino), w→1 tende a pseudo-neutro sem esmagar a norma (evita síntese muda). "
            "linear_gain (legado): emb = w·x-vector (mantém norma baixa se w≪1 e costuma colapsar)."
        ),
    )
    parser.add_argument(
        "--speecht5_speaker_emb_mix_mode",
        type=str,
        default="toward_neutral",
        choices=("toward_neutral", "linear_gain"),
        help=(
            "Como aplicar --speecht5_speaker_emb_mix: toward_neutral (recom.; interpola antes de gerar norma estável); "
            "linear_gain comportamento antigo w·embedding (mudas se w pouco)."
        ),
    )
    parser.add_argument(
        "--speecht5_chunk_max_chars",
        type=int,
        default=100,
        metavar="N",
        help=(
            "SpeechT5: limite de caracteres por parte ao partir o texto (split em pontuação). "
            "Valor grande (ex.: 10000) ≈ uma única chamada generate por frase — alinha à gravação contínua; "
            "documentar no TCC se alterar face a um protocolo base com partes curtas. Default 100 = comportamento "
            "histórico deste script."
        ),
    )
    parser.add_argument(
        "--speecht5_inter_chunk_silence_sec",
        type=float,
        default=0.12,
        metavar="SEC",
        help="SpeechT5: silêncio entre partes quando há vários chunks. 0 = sem pausa artificial.",
    )
    parser.add_argument(
        "--speecht5_speaker_emb_pool_size",
        type=int,
        default=1,
        metavar="K",
        help=(
            "SpeechT5: condicionamento com K clips do mesmo locutor (média ou concat de x-vectors). "
            "1 = só o WAV desta sentença (baseline). K>1 usa o índice do split teste exportado; opcionalmente treino."
        ),
    )
    parser.add_argument(
        "--speecht5_speaker_emb_pool_mode",
        type=str,
        default="mean",
        choices=("mean", "concat"),
        help=(
            "Com --speecht5_speaker_emb_pool_size>1: mean = média dos x-vectors por clip (L2 após média); "
            "concat = áudios concatenados (--speecht5_speaker_emb_pool_concat_gap_sec) + um encode."
        ),
    )
    parser.add_argument(
        "--speecht5_speaker_emb_pool_concat_gap_sec",
        type=float,
        default=0.25,
        metavar="SEC",
        help="Só modo concat: silêncio entre clipes antes do SpeechBrain.",
    )
    parser.add_argument(
        "--speecht5_speaker_emb_pool_include_train",
        action="store_true",
        help=(
            "Inclui frases do subset de treino no índice por locutor (mais clipes para K>1). "
            "Declarar no relatório se o condicionamento não for só do conjunto de teste."
        ),
    )
    parser.add_argument(
        "--speecht5_speaker_emb_pool_seed",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Opcional: após a sentença actual, clipes extra do mesmo locutor pseudo-aleatórios (reproducível). "
            "Omitir = ordem determinística por id de sentença."
        ),
    )
    args = parser.parse_args()

    if getattr(args, "only_f0_rmse_dir", None):
        from f0_infer_metrics import (
            compute_metrics_for_inference_dir,
            summarize_f0_metrics_rows,
            summarize_f0_metrics_vs_reference,
            write_metrics_csv,
        )

        od = args.only_f0_rmse_dir.strip()
        if not od:
            print("❌ --only_f0_rmse_dir vazio.")
            sys.exit(1)
        rows = compute_metrics_for_inference_dir(
            od,
            sample_rate=args.f0_sample_rate,
            hop_length=args.f0_hop_length,
            fmin=args.f0_fmin,
            fmax=args.f0_fmax,
        )
        out_csv = os.path.join(od, "f0_rmse.csv")
        write_metrics_csv(rows, out_csv)
        mean_hz, valid_n, total_n = summarize_f0_metrics_rows(rows)
        mo, ko, mt, kt = summarize_f0_metrics_vs_reference(rows)
        print(f"\n📈 F0 RMSE só-métrica: {total_n} linhas | treinado-vs-base válidos={valid_n}")
        if np.isfinite(mean_hz):
            print(f"   Média treinado(LoRA) vs base sintét.: {mean_hz:.4f} Hz")
        if ko > 0 or kt > 0:
            print(f"   Base vs WAV dataset (n={ko}): {mo:.4f} Hz")
            print(f"   LoRA vs WAV dataset (n={kt}): {mt:.4f} Hz")
        print(f"   📄 {out_csv}")
        sys.exit(0)

    if not getattr(args, "model_path", None):
        parser.error("--model_path (--model_dir) é obrigatório quando não usas --only_f0_rmse_dir")

    full_cfg = load_config(args.config)
    dataset_name = args.dataset or full_cfg.get('settings', {}).get('default_dataset_profile', 'lapsbm_fastspeech2')
    ds_cfg = full_cfg['dataset_profiles'][dataset_name]
    model_type = ds_cfg.get('model_type', 'fastspeech2')
    model_cfg = full_cfg['models'][model_type]

    infer_all_test = bool(getattr(args, "infer_all_test_sentences", False))
    if infer_all_test:
        if model_type != "speecht5":
            print("❌ --infer_all_test_sentences só é suportado com perfil SpeechT5.")
            sys.exit(1)
        if not getattr(args, "dataset_reference_audios", False):
            print("❌ --infer_all_test_sentences requer --dataset_reference_audios.")
            sys.exit(1)
    
    hw_name = args.profile or detect_hardware_profile()
    hw_cfg = full_cfg['hardware_profiles'].get(hw_name, full_cfg['hardware_profiles']['cpu'])
    device = hw_cfg.get('device', 'cpu')

    # Ajuste para aceitar o caminho do checkpoint diretamente ou a pasta pai
    model_path = args.model_path

    print(f"🚀 Iniciando Teste de Inferência...")
    print(f"   📂 Modelo: {model_path}")
    print(f"   💻 Hardware: {hw_name.upper()} ({device})")
    if args.test_text_seed is None:
        print("   🎲 Seleção de sentenças de teste: aleatória (sem seed)")
    else:
        print(f"   🎲 Seleção de sentenças de teste: reproduzível (seed={args.test_text_seed})")
    _treb = float(getattr(args, "trained_treble_boost_db", 0) or 0)
    _xhp = float(getattr(args, "trained_grave_highpass_hz", 0) or 0)
    _eq_parts: List[str] = []
    if args.apply_trained_bass_treatment:
        _eq_parts.append(
            f"graves ≤{args.trained_bass_cut_hz:.0f}Hz −{args.trained_bass_atten_db:.1f}dB "
            f"(transição {args.trained_bass_transition_hz:.0f}Hz)"
        )
    if _xhp > 0:
        _eq_parts.append(f"HP extra {_xhp:.0f}Hz (só treinado)")
    if _treb > 1e-6:
        _eq_parts.append(
            f"agudos +{_treb:.1f}dB @≥{args.trained_treble_shelf_hz:.0f}Hz "
            f"(trans. {args.trained_treble_transition_hz:.0f}Hz)"
        )
    if _eq_parts:
        print("   🎛️ Pós-processo apenas em áudio treinado (LoRA): " + " | ".join(_eq_parts))
    else:
        print(
            "   🎛️ Pós-processo apenas em áudio treinado (LoRA): nenhum "
            "(graves / HP extra / agudos desativados)"
        )
    print(
        "   📈 Normalização de pico: "
        + ("ativada" if args.apply_peak_normalization else "desativada")
    )
    print(
        "   🧹 Cleanup de silêncio (SpeechT5): "
        + (f"ativado (top_db={args.silence_top_db:.0f})" if args.apply_silence_cleanup else "desativado")
    )
    if model_type == "speecht5":
        hpf = float(getattr(args, "speecht5_inference_highpass_hz", 0) or 0)
        print(
            "   🔊 Pós-Vocoder SpeechT5: "
            + (f"DC + HP {hpf:.0f} Hz anti-subgrave rum" if hpf > 0 else "sem HP (--speecht5_inference_highpass_hz 0)")
        )
    print(
        "   📉 F0 RMSE inferência:"
        + (
            " ativo (--compute_f0_rmse após SpeechT5)"
            if getattr(args, "compute_f0_rmse", False)
            else " desativo"
        )
    )
    if getattr(args, "no_mp3", False):
        print("   📼 Exportação ficheiros: apenas WAV (--no_mp3)")
    else:
        print(
            f"   📼 Exportação ficheiros: WAV + MP3 (bitrate {getattr(args, 'mp3_bitrate', '192k')}; "
            "pydub + ffmpeg no PATH)",
        )
    spk_mix = float(getattr(args, "speecht5_speaker_emb_mix", 0.0))
    spk_mix_md = getattr(args, "speecht5_speaker_emb_mix_mode", "toward_neutral") or "toward_neutral"
    if model_type == "speecht5":
        if getattr(args, "speecht5_zero_speaker_embedding", False):
            print("   🔀 SpeechT5 mistura emb.: omitida (--speecht5_zero_speaker_embedding → sempre zeros)")
        else:
            wm = max(0.0, min(1.0, spk_mix))
            if wm != spk_mix:
                print(f"   🔀 SpeechT5 emb. (--speecht5_speaker_emb_mix={spk_mix} → clamp {wm}) ")
            label = "'toward_neutral': w peso vetor µ" if spk_mix_md == "toward_neutral" else "'linear_gain': w amplitude em embedding"
            if spk_mix_md == "toward_neutral":
                hint = "(w=0 = x-vector integral; maior w = mais próximo µ pseudo-neutra; norma antes do modelo mantém‑se)"
            else:
                hint = "(w baixo ⇒ ||emb|| pequeno; treino ≈ vectores norma ~1 ⇒ risco síntese muda)"
                if wm > 0.0 and wm < 0.5:
                    hint += " ⚠️ considera --speecht5_speaker_emb_mix_mode toward_neutral"
            print(f"      modo {label} → w={wm:g}. {hint}")

    # 1. Carregar Handler e Modelo
    handler = get_inference_handler(model_type, model_cfg, device, model_path)
    handler.load()
    if model_type == "speecht5":
        handler.chunk_max_chars = int(getattr(args, "speecht5_chunk_max_chars", 100))
        handler.inter_chunk_silence_sec = float(getattr(args, "speecht5_inter_chunk_silence_sec", 0.12))
        print(
            "   🎛️ SpeechT5 chunking: "
            f"max_chars={handler.chunk_max_chars}, "
            f"silêncio_entre_partes={handler.inter_chunk_silence_sec:g}s "
            "(pontuação em `split_text_for_tts`; valor alto ≈ uma parte por frase)."
        )
    if model_type == "fastspeech2" and getattr(args, "length_scale", None) is not None and hasattr(handler, "set_length_scale"):
        handler.set_length_scale(args.length_scale)

    sampling_rate = model_cfg.get('sampling_rate', 22050)
    
    # 2. Obter Speaker Embedding (SpeechT5: x-vector por sentença a partir do WAV de referência; XTTS: placeholder)
    speaker_emb = None
    speecht5_speaker_model = None
    if model_type == "speecht5":
        spk_enc_id = model_cfg.get("speaker_encoder_id", "speechbrain/spkrec-xvect-voxceleb")
        sb_savedir = os.path.join(
            full_cfg.get("settings", {}).get("local_datasets_dir", "./datasets"),
            ".speechbrain_cache",
            spk_enc_id.replace("/", "__"),
        )
        os.makedirs(sb_savedir, exist_ok=True)
        if getattr(args, "speecht5_zero_speaker_embedding", False):
            print(
                "   🔊 SpeechT5 encoder de locutor: omitido (--speecht5_zero_speaker_embedding); "
                "condicionamento 1×512 nulo (igual texto HF / antes do x-vector)."
            )
        else:
            print(f"   📥 Encoder de locutor para inferência SpeechT5: {spk_enc_id}")
            # COPY evita symlink (WinError 1314 sem "Developer Mode" / admin nos symlinks HF→savedir).
            speecht5_speaker_model = EncoderClassifier.from_hparams(
                source=spk_enc_id,
                run_opts={"device": _device_for_speechbrain(device)},
                savedir=sb_savedir,
                local_strategy=LocalStrategy.COPY,
            )
    elif model_type == "xtts_v2":
        speaker_emb = [torch.zeros((1, 1, 1024)).to(device), torch.zeros((1, 512)).to(device)]

    # 3. Preparar lista de textos
    default_texts = [
        "Em dois mil e vinte e cinco, a taxa caiu de doze virgula cinco por cento para nove virgula oito por cento.",
        "O doutor Silva atende na avenida Paulista, no apartamento mil duzentos e tres.",
        "Eu preciso levar a colher para colher as ervas.",
        "A equipe validou os dados antes de iniciar os experimentos.",
        "Voce confirmou se o arquivo de audio esta correto",
        "Que otima noticia, o treinamento terminou sem erros",
        "O treinamento exaustivo foi finalizado com sucesso.",
        "Esta é uma demonstração de voz gerada por inteligência artificial.",
        # "A qualidade do áudio melhorou significativamente após o ajuste fino.",
        # "Estamos testando a comparação entre o modelo base e o modelo treinado.",
        # "Agora vamos testar uma sentença um pouco mais longa. A ideia é avaliar se nosso novo modelo Text To Speech consegue lidar com frases um pouco maiores.",
        # "Este é um teste longo de texto. A ideia é avaliar se nosso modelo Text To Speech consegue lidar com frases mais complexas e variadas, incluindo vírgula e ponto final. Vamos ver como ele se sai com esta sentença que tem mais de vinte palavras, o que é um bom desafio para a geração de áudio realista e fluída... Se tudo correr bem, o resultado deve soar natural e coerente, mesmo com a extensão do texto.",
        # "Era uma vez um pequeno robô chamado Pip que vivia numa floresta de cristal onde as árvores tilintavam ao vento como sinos de vidro. Ele passava seus dias colecionando sons esquecidos. O suspiro de uma flor ao amanhecer, o ronco distante de um trovão tímido, o risinho abafado de um cogumelo quando a chuva fazia cócegas em seu chapéu. Até que certa manhã encontrou uma velhinha chamada Zara que carregava numa mala de couro uma única palavra que havia perdido em sonho, e Pip, com todo o carinho de seus circuitos dourados, abriu o peito e reproduziu o som exato, e a palavra voltou a existir, e a velhinha sorriu tão amplamente que as estrelas, mesmo sendo de dia, resolveram aparecer só pra ver.",
    ]
    
    dataset_test_limit = getattr(args, "dataset_sentence_limit", 5)
    train_limit = int(max(1, getattr(args, "dataset_train_sentence_limit", 5) or 5))

    inference_run_rows: Optional[List[InferenceCandidate]] = None
    dataset_picked_candidates: Optional[List[InferenceCandidate]] = None
    user_text_early = getattr(args, "text", None)

    stripped_user = user_text_early.strip() if isinstance(user_text_early, str) else ""

    has_ds_test = getattr(args, "dataset_reference_audios", False) and model_type == "speecht5"
    has_ds_train = getattr(args, "dataset_train_reference_audios", False) and model_type == "speecht5"

    if infer_all_test:
        if stripped_user:
            print("❌ Não combines --text com --infer_all_test_sentences.")
            sys.exit(1)
        if has_ds_train:
            print(
                "ℹ️ --infer_all_test_sentences: ignorando --dataset_train_reference_audios (apenas o conjunto de teste)."
            )
            has_ds_train = False

    if has_ds_test or has_ds_train:
        merged_rows: List[InferenceCandidate] = []
        n_frm_test_split = 0
        n_frm_train_split = 0

        if has_ds_test:
            if not stripped_user:
                test_pool = load_inference_test_candidates(full_cfg, ds_cfg, dataset_name)
                if infer_all_test:
                    test_part = random_pick_inference_candidates(
                        test_pool,
                        limit=1,
                        seed=None,
                        take_all=True,
                    )
                    print(
                        f"   🎙️ Referência dataset: teste completo — {len(test_part)} sentença(s) (sem subamostragem)."
                    )
                else:
                    test_part = random_pick_inference_candidates(
                        test_pool,
                        limit=max(1, dataset_test_limit),
                        seed=args.test_text_seed,
                    )
                    print(
                        f"   🎙️ Referência dataset (subset teste por locutor): {len(test_part)} sentença(s)"
                    )
                if not test_part:
                    print("❌ Sem amostras de teste no dataset (--dataset_reference_audios)...")
                    sys.exit(1)
                merged_rows.extend(test_part)
                n_frm_test_split = len(test_part)
                if infer_all_test and len(test_part) > 48:
                    print(
                        f"   🧾 Lista de amostras omitida no terminal ({len(test_part)} itens); "
                        "ver sentencas.txt na pasta de saída."
                    )
                else:
                    print_inference_sample_banner(
                        test_part,
                        (
                            "🧾 Conjunto de teste completo — F0 RMSE vs WAV do mesmo locutor:"
                            if infer_all_test
                            else "🧾 Amostras escolhidas (teste) — F0 RMSE vs WAV do mesmo locutor:"
                        ),
                    )
                    print("   📌 Sugestão: --compute_f0_rmse para RMSE contra *_referencia.wav.")
            else:
                pool = random_pick_inference_candidates(
                    load_inference_test_candidates(full_cfg, ds_cfg, dataset_name),
                    limit=max(1, max(0, dataset_test_limit - 1)),
                    seed=args.test_text_seed,
                )
                merged_rows.append(InferenceCandidate(stripped_user, None))
                merged_rows.extend(pool)
                n_frm_test_split = 1 + len(pool)
                print(f"   🎙️ Texto CLI + subset teste dataset: {n_frm_test_split} sentença(s)")
                print_inference_sample_banner(pool, "🧾 Amostras aleatórias do teste (2ª em diante):")
                print("   📌 --compute_f0_rmse quando existir WAV de referência.")

        if has_ds_train:
            train_pool = load_inference_train_candidates(full_cfg, ds_cfg, dataset_name)
            tlim = train_limit
            if stripped_user and not has_ds_test:
                tlim = max(1, train_limit - 1)
            train_part = random_pick_inference_candidates(
                train_pool,
                limit=max(1, tlim),
                seed=args.test_text_seed,
            )
            if not train_part:
                print(
                    "❌ Sem amostras no subset de treino (--dataset_train_reference_audios). "
                    "Garante cache local de falabrasil/lapsbm (settings.local_datasets_dir).",
                )
                sys.exit(1)
            merged_rows.extend(train_part)
            n_frm_train_split = len(train_part)
            print(f"   🎓 Referência dataset (subset treino por locutor): {len(train_part)} sentença(s)")
            print_inference_sample_banner(
                train_part,
                "🧾 Amostras de treino (locutor visto na fase train — áudio HF/local):",
            )

        if stripped_user and has_ds_train and not has_ds_test:
            merged_rows.insert(0, InferenceCandidate(stripped_user, None))

        if not merged_rows:
            print("❌ Lista de inferência vazia (flags dataset).")
            sys.exit(1)

        inference_run_rows = merged_rows
        test_texts = [x.text_raw for x in inference_run_rows]
        if has_ds_test and has_ds_train and not stripped_user:
            print(
                "   📊 Total de sentenças neste run (teste + treino): "
                f"{len(inference_run_rows)} (= {n_frm_test_split} teste + {n_frm_train_split} treino; "
                "sentenca_1…N=teste, depois treino).\n"
                "   📎 Ordem: subset teste por locutor (holdout); a seguir subset treino (visto no treino).",
            )
        elif has_ds_test and has_ds_train:
            print(
                "   📊 Total neste run: "
                f"{len(inference_run_rows)} (= {n_frm_test_split} teste + {n_frm_train_split} treino).\n"
                "   📎 Ordem: subset teste; depois subset treino.",
            )
    elif stripped_user:
        test_texts = [stripped_user]
        dataset_test_texts, dataset_picked_candidates = load_test_sentences_from_dataset(
            full_cfg,
            ds_cfg,
            dataset_name,
            limit=dataset_test_limit,
            seed=args.test_text_seed,
        )
        test_texts = _merge_unique_texts(test_texts, default_texts, dataset_test_texts)
    else:
        dataset_test_texts, dataset_picked_candidates = load_test_sentences_from_dataset(
            full_cfg,
            ds_cfg,
            dataset_name,
            limit=dataset_test_limit,
            seed=args.test_text_seed,
        )
        test_texts = _merge_unique_texts([], default_texts, dataset_test_texts)

    defaults_unique_for_emb = _merge_unique_texts(default_texts)
    test_exported_emb_apply: Optional[List[bool]] = None
    test_exported_emb_order: List[str] = []
    test_exported_emb_tensors: List[torch.Tensor] = []
    test_exported_emb_counter = 0
    if getattr(args, "use_test_embeddings", False):
        if model_type != "speecht5":
            print(
                "⚠️ --use-test-embeddings só aplica a SpeechT5; flag ignorada para este modelo.",
                file=sys.stderr,
            )
        else:
            test_exported_emb_order, test_exported_emb_tensors = load_test_exported_embedding_cycle(
                str(args.use_test_embeddings_dir),
                device,
            )
            test_exported_emb_apply = _default_texts_prefix_alignment_mask(test_texts, defaults_unique_for_emb)
            _nh = sum(test_exported_emb_apply)
            print(
                f"   📎 --use-test-embeddings ({args.use_test_embeddings_dir}): "
                f"locutores ciclo = {test_exported_emb_order} | "
                f"{_nh} linha(s) com condicionamento .npy (prefixo default_texts)."
            )

    # 4. Criar pasta de saída
    type_str = "custom" if args.text else "batch"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(model_path, f"inference_{type_str}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    write_inference_sentences_txt(out_dir, test_texts)
    write_per_sentence_transcripts_and_tsv(out_dir, test_texts, args)

    if model_type == "fastspeech2":
        print(
            f"\n   🎙️ Iniciando geração de {len(test_texts)} sentenças (3 WAV por frase): "
            f"(1) _ljs_english = model.pth, EN, sem LoRA  "
            f"(2) _pt_original = model.pth + PT  "
            f"(3) _pt_treinado = LoRA + PT"
        )
    else:
        print(f"\n   🎙️ Iniciando geração de {len(test_texts)} sentenças...")
    print(f"   🎯 Resultados em: {out_dir}")
    print(
        "   📏 (mesmo SR) tamanho WAV ≈ proporcional à duração; treinado muito menor ⇒ geralmente "
        "menos frames de mel (duration predictor), não 'qualidade de ficheiro'."
    )

    def _wav_report(path: str, audio_arr, sr: int) -> None:
        if audio_arr is None:
            return
        xa = np.asarray(audio_arr, dtype=np.float32).reshape(-1)
        n = len(xa)
        dur = n / float(sr) if sr else 0.0
        pk = float(np.max(np.abs(xa))) if n else 0.0
        b = os.path.getsize(path) if os.path.isfile(path) else 0
        print(
            f"      → {os.path.basename(path)} | {b:,} bytes | ~{dur:.2f}s @ {sr}Hz | peak|y|≈{pk:.4f}"
        )
        if model_type == "speecht5" and n > 64 and pk < 1e-5:
            print(
                "      ⚠️ Este WAV parece mudo (pico ~0). Verifica se o x-vector veio de áudio de referência "
                "(📊 acima) e export em test_split_inferencia/.../audio/."
            )
        p3 = Path(path).with_suffix(".mp3")
        if p3.is_file():
            print(f"      → {p3.name} | {p3.stat().st_size:,} bytes (mp3)")

    def _finalize_audio(audio_arr, is_trained: bool = False):
        out = np.asarray(audio_arr, dtype=np.float32).reshape(-1)
        if model_type == "speecht5":
            hpf = float(getattr(args, "speecht5_inference_highpass_hz", 0) or 0)
            if hpf > 0:
                out = inference_dc_and_highpass(out, sampling_rate, hpf)
        if is_trained:
            out = apply_trained_output_waveform_eq(out, sampling_rate, args, speecht5_highpass_hz=None)
        norm_first = model_type == "speecht5" and args.apply_silence_cleanup and args.apply_peak_normalization
        if norm_first:
            out = peak_normalize_audio(out)
        if model_type == "speecht5" and args.apply_silence_cleanup:
            before_n = int(out.size)
            out2 = trim_and_compress_silence(out, sampling_rate, top_db=args.silence_top_db)
            thr = max(int(sampling_rate * 0.05), int(max(1, before_n) * 0.08))
            if before_n > 0 and out2.size < thr:
                print(
                    "      ⚠️ Cleanup de silêncio quase eliminou o sinal; ignorando esta etapa nesta amostra. "
                    "(Desliga --apply_silence_cleanup ou sobe --silence_top_db; com dois passos ligados já "
                    " aplicámos normalização de pico antes do trim.)"
                )
            else:
                out = out2
        if args.apply_peak_normalization and not norm_first:
            out = peak_normalize_audio(out)
        return out

    dataset_pick_lookup: Dict[str, InferenceCandidate] = {}
    if dataset_picked_candidates:
        for c in dataset_picked_candidates:
            nk = normalize_text(c.text_raw)
            if nk and nk not in dataset_pick_lookup:
                dataset_pick_lookup[nk] = c

    speecht5_spk_pool_index: Dict[str, List[InferenceCandidate]] = {}
    _pool_k = int(getattr(args, "speecht5_speaker_emb_pool_size", 1) or 1)
    if (
        model_type == "speecht5"
        and _pool_k > 1
        and not getattr(args, "speecht5_zero_speaker_embedding", False)
    ):
        speecht5_spk_pool_index = build_speecht5_speaker_pool_index(
            full_cfg,
            ds_cfg,
            dataset_name,
            include_train=bool(getattr(args, "speecht5_speaker_emb_pool_include_train", False)),
        )
        _n_spk = len(speecht5_spk_pool_index)
        _n_tot = sum(len(v) for v in speecht5_spk_pool_index.values())
        _pm = getattr(args, "speecht5_speaker_emb_pool_mode", "mean")
        _ix_note = (
            "teste+treino no índice"
            if getattr(args, "speecht5_speaker_emb_pool_include_train", False)
            else "preferencialmente split teste exportado"
        )
        print(
            f"   🎛️ Pool x-vector: K={_pool_k}, modo={_pm}, locutores={_n_spk}, "
            f"frases indexadas={_n_tot} ({_ix_note})."
        )

    _mp3_kw = {
        "also_mp3": not getattr(args, "no_mp3", False),
        "mp3_bitrate": getattr(args, "mp3_bitrate", "192k"),
    }
    ref_extra_manifest: List[Dict[str, Any]] = []
    speecht5_encoder_id_str = str(
        model_cfg.get("speaker_encoder_id", "speechbrain/spkrec-xvect-voxceleb")
    )
    speecht5_encoder_sr = int(model_cfg.get("sampling_rate", 16000))

    # 5. Executar Inferência
    for i, text in enumerate(test_texts):
        cand_row = inference_run_rows[i] if inference_run_rows is not None and i < len(inference_run_rows) else None
        effective_cand = cand_row
        if effective_cand is None and dataset_pick_lookup:
            effective_cand = dataset_pick_lookup.get(normalize_text(text.strip()))

        print(f"\n   --- Teste {i+1}/{len(test_texts)} ---")
        if (
            model_type == "speecht5"
            and effective_cand is not None
            and effective_cand.dataset_item is not None
        ):
            print(
                f"   🏷️ Locutor={extract_speaker_id(effective_cand.dataset_item)} | "
                f"sentença={format_sentence_id_display(effective_cand.dataset_item)}"
            )
        clean_text = normalize_text(text)
        prefix = inference_wav_prefix(args, i)

        gen_spk = speaker_emb
        if model_type == "speecht5":
            exported_npy_this_row = (
                getattr(args, "use_test_embeddings", False)
                and test_exported_emb_apply is not None
                and i < len(test_exported_emb_apply)
                and test_exported_emb_apply[i]
            )
            if exported_npy_this_row:
                s_ix = test_exported_emb_counter % len(test_exported_emb_order)
                gen_spk = test_exported_emb_tensors[s_ix]
                spk_lab = test_exported_emb_order[s_ix]
                print(
                    f"   🔊 Speaker conditioning: x-vector exportado {spk_lab} "
                    f"(--use-test-embeddings; uso #{test_exported_emb_counter + 1} no ciclo "
                    f"{test_exported_emb_order})"
                )
                test_exported_emb_counter += 1
            elif getattr(args, "speecht5_zero_speaker_embedding", False):
                gen_spk = torch.zeros((1, 512), dtype=torch.float32, device=device)
                print(
                    "   🔊 Speaker conditioning: vetor zero (--speecht5_zero_speaker_embedding); "
                    "compara com runs x-vector quando necessário."
                )
            else:
                pool_k = int(getattr(args, "speecht5_speaker_emb_pool_size", 1) or 1)
                emb_how = ""
                if (
                    pool_k > 1
                    and speecht5_spk_pool_index
                    and effective_cand is not None
                    and effective_cand.dataset_item is not None
                ):
                    pool_cands = select_speecht5_embedding_pool_candidates(
                        effective_cand,
                        speecht5_spk_pool_index,
                        pool_k,
                        seed=getattr(args, "speecht5_speaker_emb_pool_seed", None),
                    )
                    gen_spk, emb_how, used_pool = speecht5_pooled_speaker_embedding_from_candidates(
                        pool_cands,
                        pool_mode=str(getattr(args, "speecht5_speaker_emb_pool_mode", "mean")),
                        encoder_sr=int(model_cfg.get("sampling_rate", 16000)),
                        speaker_model=speecht5_speaker_model,
                        device=device,
                        full_cfg=full_cfg,
                        dataset_profile=dataset_name,
                        concat_gap_sec=float(
                            getattr(args, "speecht5_speaker_emb_pool_concat_gap_sec", 0.25)
                        ),
                    )
                    if emb_how != "pool_sem_wav":
                        id_parts = [
                            format_sentence_id_display(c.dataset_item)
                            for c in used_pool[:6]
                            if c.dataset_item
                        ]
                        id_str = ", ".join(id_parts)
                        if len(used_pool) > 6:
                            id_str += ", …"
                        print(
                            f"   🔊 Speaker conditioning: {emb_how} (clipes: {id_str})"
                        )
                    else:
                        gen_spk, emb_how = speecht5_embedding_for_inference_candidate(
                            effective_cand,
                            full_cfg=full_cfg,
                            dataset_profile=dataset_name,
                            model_cfg=model_cfg,
                            speaker_model=speecht5_speaker_model,
                            device=device,
                        )
                        print("   ⚠️ Pool sem WAV útil — recuou para x-vector só desta sentença.")
                else:
                    gen_spk, emb_how = speecht5_embedding_for_inference_candidate(
                        effective_cand,
                        full_cfg=full_cfg,
                        dataset_profile=dataset_name,
                        model_cfg=model_cfg,
                        speaker_model=speecht5_speaker_model,
                        device=device,
                    )
                if emb_how == "x-vector_do_WAV_referência":
                    print(
                        "   🔊 Speaker conditioning: SpeechBrain x-vector (mesmo áudio "
                        "que alimenta F0/ref. e *_referencia.wav)."
                    )
                elif emb_how and (
                    emb_how.startswith("x-vector_mean_") or emb_how.startswith("x-vector_concat_")
                ):
                    pass
                else:
                    print(
                        f"   ⚠️ Speaker conditioning: embedding zero ou inválido ({emb_how}); "
                        "preferir WAV de referência na linha HF ou export "
                        "`test_split_inferencia/...` ou `--speecht5_zero_speaker_embedding`."
                    )
                mix_mode = getattr(args, "speecht5_speaker_emb_mix_mode", "toward_neutral") or "toward_neutral"
                wmix = float(max(0.0, min(1.0, getattr(args, "speecht5_speaker_emb_mix", 0.0))))
                blended = False
                if mix_mode == "toward_neutral" and wmix > 0.0:
                    gen_spk = blend_speecht5_speaker_embedding(
                        gen_spk, wmix, mode=mix_mode
                    )
                    blended = True
                elif mix_mode == "linear_gain" and wmix < 1.0:
                    gen_spk = blend_speecht5_speaker_embedding(
                        gen_spk, wmix, mode="linear_gain"
                    )
                    blended = True
                if blended:
                    print(
                        f"   🔀 mistura aplicada (--speecht5_speaker_emb_mix_mode {mix_mode} w={wmix:g}) "
                        "antes de SpeechT5.generate."
                    )

        if model_type == "fastspeech2":
            # 1) model.pth, EN, sem LoRA
            print(f"   (1) _ljs_english — model.pth, EN, sem LoRA...")
            try:
                audio_ljs = handler.generate(
                    clean_text, speaker_emb, use_lora=False, fastspeech2_pipeline="ljs_en"
                )
                p_ljs = os.path.join(out_dir, f"{prefix}_ljs_english.wav")
                audio_ljs_out = _finalize_audio(audio_ljs, is_trained=False)
                write_wav_and_mp3(p_ljs, audio_ljs_out, sampling_rate, **_mp3_kw)
                _wav_report(p_ljs, audio_ljs_out, sampling_rate)
            except Exception as e:
                print(f"   ⚠️ Erro (LJS/EN): {e}")

            # 2) model.pth + PT (Gruut), sem LoRA
            print(f"   (2) _pt_original — model.pth + PT (tokenizer + Gruut)...")
            try:
                audio_pt_base = handler.generate(
                    clean_text, speaker_emb, use_lora=False, fastspeech2_pipeline="pt_base"
                )
                p_pt_base = os.path.join(out_dir, f"{prefix}_pt_original.wav")
                audio_pt_base_out = _finalize_audio(audio_pt_base, is_trained=False)
                write_wav_and_mp3(p_pt_base, audio_pt_base_out, sampling_rate, **_mp3_kw)
                _wav_report(p_pt_base, audio_pt_base_out, sampling_rate)
            except Exception as e:
                print(f"   ⚠️ Erro (base+PT): {e}")
                audio_pt_base = None

            # 3) LoRA + PT
            print(f"   (3) _pt_treinado — LoRA + PT...")
            try:
                audio_lora = handler.generate(
                    clean_text, speaker_emb, use_lora=True, fastspeech2_pipeline="pt_lora"
                )
                p_lora = os.path.join(out_dir, f"{prefix}_pt_treinado.wav")
                audio_lora_out = _finalize_audio(audio_lora, is_trained=True)
                write_wav_and_mp3(p_lora, audio_lora_out, sampling_rate, **_mp3_kw)
                _wav_report(p_lora, audio_lora_out, sampling_rate)
                if audio_pt_base is not None and len(np.asarray(audio_pt_base).reshape(-1)) > 0:
                    ratio = len(np.asarray(audio_lora).reshape(-1)) / len(np.asarray(audio_pt_base).reshape(-1))
                    print(f"      📊 Razão duração pt_treinado / pt_base ≈ {ratio:.2f}×")
                if args.text and i == 0:
                    write_wav_and_mp3("output_inference.wav", audio_lora_out, sampling_rate, **_mp3_kw)
            except Exception as e:
                print(f"   ⚠️ Erro (LoRA+PT): {e}")
        else:
            # Outros modelos: mantém pares base vs LoRA
            print(f"   🔊 Gerando ORIGINAL (base sem LoRA)...")
            try:
                audio_orig = handler.generate(clean_text, gen_spk, use_lora=False)
                p_orig = os.path.join(out_dir, f"{prefix}_original.wav")
                audio_orig_out = _finalize_audio(audio_orig, is_trained=False)
                write_wav_and_mp3(p_orig, audio_orig_out, sampling_rate, **_mp3_kw)
                _wav_report(p_orig, audio_orig_out, sampling_rate)
                if effective_cand is not None and model_type == "speecht5" and effective_cand.dataset_item is not None:
                    ya, sra = resolve_reference_waveform_for_inference(
                        effective_cand.dataset_item,
                        int(sampling_rate),
                        full_cfg=full_cfg,
                        dataset_profile=dataset_name,
                    )
                    if ya is not None and ya.size > 0:
                        p_refs = os.path.join(out_dir, f"{prefix}_referencia.wav")
                        write_wav_and_mp3(p_refs, np.asarray(ya, dtype=np.float32), int(sra), **_mp3_kw)
                        _wav_report(p_refs, ya, int(sra))
                        if model_type == "speecht5":
                            save_speecht5_reference_run_extras(
                                out_dir,
                                prefix=prefix,
                                text_raw=text,
                                p_refs_wav=p_refs,
                                speaker_model=speecht5_speaker_model,
                                encoder_sr=speecht5_encoder_sr,
                                device=device,
                                manifest_rows=ref_extra_manifest,
                            )
                    elif getattr(args, "dataset_reference_audios", False):
                        print(
                            "      ⚠️ WAV de referência não gravado "
                            "(HF sem array/path e sem correspondente em "
                            f"test_split_inferencia/{dataset_name}/audio/<locutor>/)."
                        )
            except Exception as e:
                print(f"   ⚠️ Erro no modelo original: {e}")
                audio_orig = None

            print(f"   🔥 Gerando TREINADO (LoRA)...")
            try:
                audio_lora = handler.generate(clean_text, gen_spk, use_lora=True)
                p_lora = os.path.join(out_dir, f"{prefix}_treinado.wav")
                audio_lora_out = _finalize_audio(audio_lora, is_trained=True)
                write_wav_and_mp3(p_lora, audio_lora_out, sampling_rate, **_mp3_kw)
                _wav_report(p_lora, audio_lora_out, sampling_rate)
                if audio_orig is not None and len(np.asarray(audio_orig).reshape(-1)) > 0:
                    ratio = len(np.asarray(audio_lora).reshape(-1)) / len(np.asarray(audio_orig).reshape(-1))
                    print(f"      📊 Razão duração treinado/original ≈ {ratio:.2f}×")
                if args.text and i == 0:
                    write_wav_and_mp3("output_inference.wav", audio_lora_out, sampling_rate, **_mp3_kw)
            except Exception as e:
                print(f"   ⚠️ Erro no modelo treinado: {e}")
            
    if model_type == "speecht5" and getattr(args, "compute_f0_rmse", False):
        from f0_infer_metrics import (
            compute_metrics_for_inference_dir,
            summarize_f0_metrics_rows,
            summarize_f0_metrics_vs_reference,
            write_metrics_csv,
        )

        rows = compute_metrics_for_inference_dir(
            out_dir,
            sample_rate=args.f0_sample_rate,
            hop_length=args.f0_hop_length,
            fmin=args.f0_fmin,
            fmax=args.f0_fmax,
        )
        if rows:
            out_csv = os.path.join(out_dir, "f0_rmse.csv")
            write_metrics_csv(rows, out_csv)
            mean_hz, valid_n, total_n = summarize_f0_metrics_rows(rows)
            mo, ko, mt, kt = summarize_f0_metrics_vs_reference(rows)
            print(f"\n   📉 F0 RMSE treinado-vs-base sintét.: {total_n} linhas | válidos={valid_n}")
            if np.isfinite(mean_hz):
                print(f"      Média: {mean_hz:.4f} Hz")
            if ko > 0 or kt > 0:
                print(f"      Base vs WAV referência dataset (n={ko}): {mo:.4f} Hz")
                print(f"      LoRA vs WAV referência dataset (n={kt}): {mt:.4f} Hz")
            print(f"      📄 {out_csv}")
        else:
            print("\n   ⚠️ compute_f0_rmse: sem pares *_original.wav / *_treinado.wav nesta pasta.")

    if model_type == "speecht5" and ref_extra_manifest:
        man_path = os.path.join(out_dir, "referencia_embeddings", "manifest.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "speaker_encoder_id": speecht5_encoder_id_str,
                    "encoder_sampling_rate_hz": speecht5_encoder_sr,
                    "n_entries": len(ref_extra_manifest),
                    "entries": ref_extra_manifest,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        n_emb = sum(1 for e in ref_extra_manifest if e.get("embedding_npy"))
        print(
            f"\n   📎 Referência organizada: referencia_embeddings/ ({n_emb} x-vector .npy + manifest.json) | "
            f"referencia_sentencas/ ({len(ref_extra_manifest)} .txt) | "
            f"referencia_wav/ ({len(ref_extra_manifest)} .wav)"
        )
        print(f"      📄 {man_path}")

    copy_treinado_wavs_to_subfolder(out_dir)

    print(f"\n✅ Concluído! Todos os arquivos estão na pasta: {out_dir}")

if __name__ == "__main__":
    main()
