#!/usr/bin/env python3
"""
Exporta x-vectors SpeechBrain (512-D, L2) dos locutores do *split teste* zero-shot,
com a mesma lógica de `extract_speaker_id` e divisão train/val/test que `train_exhaustive.py`.

Saída:
  - manifest.json + `spk_<id>.npy` agregado (ver --aggregate);
  - com --export-all-utterances (padrão): todos os clipes do teste por locutor como
    `.wav`, `.txt` e `spk_<id>_idx<dataset_index>_emb.npy` (embedding só se o clip for válido).

Exemplo:
  python export_test_speaker_embeddings.py --config config.yaml --dataset lapsbm_speecht5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset, load_from_disk

from train_exhaustive import (
    _device_for_speechbrain,
    extract_speaker_id,
    load_config,
    merge_dataset_model_overrides,
)


def load_all_items(full_cfg: Dict, ds_cfg: Dict) -> Tuple[List[Dict], Dataset]:
    local_root = full_cfg.get("settings", {}).get("local_datasets_dir", "./datasets")
    repo_name = ds_cfg["dataset_id"].split("/")[-1]
    local_ds_path = os.path.join(local_root, repo_name)
    is_local = os.path.exists(local_ds_path) and len(os.listdir(local_ds_path)) > 0

    all_data: List[Dict] = []
    if is_local:
        print(f"📂 Dataset local: {local_ds_path}")
        dataset_stream = load_from_disk(local_ds_path)
    else:
        load_kwargs: Dict[str, Any] = {"split": ds_cfg["dataset_split"], "path": ds_cfg["dataset_id"]}
        load_kwargs["streaming"] = ds_cfg.get("streaming", True)
        if "dataset_config" in ds_cfg:
            load_kwargs["name"] = ds_cfg["dataset_config"]
        dataset_stream = load_dataset(**load_kwargs)

    for item in dataset_stream:
        if "wav" in item and "audio" not in item:
            item["audio"] = item["wav"]
        if "txt" in item and "text" not in item:
            item["text"] = item["txt"]
        all_data.append(item)

    return all_data, Dataset.from_list(all_data)


def zero_shot_split_indices(
    all_data: List[Dict],
    ds_cfg: Dict,
) -> Tuple[Set[str], List[int]]:
    """
    Mesma regra que train_exhaustive: zero_shot_split com contagem de locutores;
    devolve o conjunto de IDs de locutor de teste e índices globais das amostras de teste.
    """
    all_speakers = [extract_speaker_id(x) for x in all_data]
    counts = Counter(all_speakers)
    num_spk = ds_cfg.get("num_speakers", 1)
    max_samples = ds_cfg.get("num_samples_per_speaker", 0)
    zero_shot_split = ds_cfg.get("zero_shot_split", None)

    if num_spk == 0:
        valid_spks = set(all_speakers)
    else:
        valid_spks = set(s for s, _ in counts.most_common(num_spk))

    train_spks: Set[str]
    val_spks: Set[str]
    test_spks: Set[str]

    train_spks, val_spks, test_spks = valid_spks, set(), set()
    if zero_shot_split and len(valid_spks) > 0:
        t_spk = int(zero_shot_split.get("train_speakers", 28))
        v_spk = int(zero_shot_split.get("val_speakers", 3))
        ts_spk = int(zero_shot_split.get("test_speakers", 4))
        sorted_spks = sorted(valid_spks)
        need = t_spk + v_spk + ts_spk
        if len(sorted_spks) >= need:
            train_spks = set(sorted_spks[:t_spk])
            val_spks = set(sorted_spks[t_spk : t_spk + v_spk])
            test_spks = set(sorted_spks[t_spk + v_spk : t_spk + v_spk + ts_spk])
            valid_spks = train_spks | val_spks | test_spks
        else:
            print(
                f"⚠️ Locutores insuficientes para zero-shot ({len(sorted_spks)} < {need}); "
                "test_spks fica vazio.",
                file=sys.stderr,
            )
            test_spks = set()

    spk_added_counts = {s: 0 for s in valid_spks}
    test_indices: List[int] = []
    for i, (item, s) in enumerate(zip(all_data, all_speakers)):
        if s not in valid_spks:
            continue
        if max_samples > 0 and spk_added_counts[s] >= max_samples:
            continue
        if s in val_spks:
            spk_added_counts[s] += 1
        elif s in test_spks:
            test_indices.append(i)
            spk_added_counts[s] += 1
        else:
            spk_added_counts[s] += 1

    return test_spks, test_indices


def describe_audio_provenance(item: Dict[str, Any]) -> str:
    """Identificação legível do clip (para log / manifest)."""
    u = item.get("__url__")
    if u:
        return str(u).strip()
    au = item.get("audio")
    if isinstance(au, dict):
        p = au.get("path")
        if p:
            return str(p).strip()
    for k in ("path", "file", "audio_path"):
        if item.get(k):
            return str(item[k]).strip()
    return "(origem desconhecida — só array em memória)"


def waveform_is_valid(y: np.ndarray, *, min_samples: int, min_peak: float) -> bool:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < min_samples:
        return False
    if float(np.max(np.abs(y))) < min_peak:
        return False
    return True


def row_transcript(item: Dict[str, Any]) -> str:
    return str(item.get("text") or item.get("txt") or "").strip()


def save_wav_and_transcript_sidecar(
    out_dir: Path,
    *,
    safe_spk: str,
    dataset_index: int,
    y: np.ndarray,
    sr: int,
    transcript: str,
) -> Dict[str, str]:
    """
    Grava na pasta de saída o clip (WAV PCM16, mono) e a transcrição (.txt)
    com o mesmo prefixo de nome.
    """
    base = f"spk_{safe_spk}_idx{dataset_index}"
    wav_path = out_dir / f"{base}.wav"
    txt_path = out_dir / f"{base}.txt"
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = np.clip(y, -1.0, 1.0)
    sf.write(
        str(wav_path),
        (y * 32767.0).astype(np.int16),
        int(sr),
        subtype="PCM_16",
    )
    body = transcript.strip()
    txt_path.write_text(body + "\n" if body else "", encoding="utf-8")
    return {"wav": wav_path.name, "txt": txt_path.name}


def save_embedding_npy_sidecar(
    out_dir: Path,
    *,
    safe_spk: str,
    dataset_index: int,
    vec: np.ndarray,
) -> str:
    """Grava o x-vector de uma utterance (L2 já aplicado em encode_one_utterance)."""
    base = f"spk_{safe_spk}_idx{dataset_index}_emb"
    path = out_dir / f"{base}.npy"
    np.save(str(path), np.asarray(vec, dtype=np.float32).reshape(-1))
    return path.name


def encode_one_utterance(
    y: np.ndarray,
    sr: int,
    encoder_sr: int,
    speaker_model: Any,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if int(sr) != int(encoder_sr):
        y = librosa.resample(y, orig_sr=int(sr), target_sr=int(encoder_sr)).astype(np.float32)
    wav = torch.tensor(np.asarray(y))
    with torch.no_grad():
        emb = speaker_model.encode_batch(wav)
        v = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def aggregate_embeddings(vectors: List[np.ndarray], mode: str) -> np.ndarray:
    if not vectors:
        raise ValueError("sem vectores")
    if mode == "first":
        out = vectors[0].astype(np.float32)
    else:
        st = np.stack([v.astype(np.float32) for v in vectors], axis=0)
        out = st.mean(axis=0)
    nrm = float(np.linalg.norm(out))
    if nrm > 1e-12:
        out = out / nrm
    return out.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="x-vectors dos locutores do split teste (zero-shot).")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default="lapsbm_speecht5")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings_test_speakers",
        help="Pasta de saída (.npy + manifest.json).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu (SpeechBrain).")
    parser.add_argument(
        "--aggregate",
        type=str,
        default="first",
        choices=("mean", "first"),
        help=(
            "first = um embedding / locutor: primeiro áudio *válido* na ordem do split teste; "
            "mean = média L2 de vários x-vectors."
        ),
    )
    parser.add_argument(
        "--min_audio_samples",
        type=int,
        default=800,
        help="Mínimo de amostras no clip para ser aceite com --aggregate first (ex.: 800 ≈ 50 ms @ 16 kHz).",
    )
    parser.add_argument(
        "--min_audio_peak",
        type=float,
        default=1e-5,
        help="Pico mínimo |y| para o clip contar como válido (evita áudio mudo).",
    )
    parser.add_argument(
        "--max_utts_per_speaker",
        type=int,
        default=0,
        help="Limita utterances por locutor no agregador (0 = todas as do teste).",
    )
    parser.add_argument(
        "--export-all-utterances",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exporta todos os áudios do split teste (wav+txt por clip). "
            "Gera também spk_*_idx*_emb.npy para clipes válidos. "
            "Desligar reproduz o modo antigo: só os clipes usados no agregador."
        ),
    )
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    if args.dataset not in full_cfg.get("dataset_profiles", {}):
        print(f"❌ dataset_profiles['{args.dataset}'] não encontrado.", file=sys.stderr)
        return 1

    ds_cfg = full_cfg["dataset_profiles"][args.dataset]
    model_type = ds_cfg.get("model_type", "speecht5")
    if model_type != "speecht5":
        print("⚠️ Este script visa SpeechT5 + SpeechBrain; continua com encoder do YAML.", file=sys.stderr)

    models = full_cfg.get("models", {})
    if model_type not in models:
        print(f"❌ models['{model_type}'] ausente.", file=sys.stderr)
        return 1

    model_cfg = merge_dataset_model_overrides(models[model_type], ds_cfg)
    encoder_id = model_cfg.get("speaker_encoder_id")
    if not encoder_id:
        print("❌ Config sem speaker_encoder_id (models.speecht5).", file=sys.stderr)
        return 1

    target_sr = int(model_cfg.get("sampling_rate", 16000))

    dev = str(args.device).strip()
    if dev.lower() == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
        print("⚠️ CUDA indisponível; usando CPU.", file=sys.stderr)

    from speechbrain.inference.classifiers import EncoderClassifier

    speaker_model = EncoderClassifier.from_hparams(
        source=encoder_id,
        run_opts={"device": _device_for_speechbrain(dev)},
    )

    all_data, full_ds = load_all_items(full_cfg, ds_cfg)
    test_spks, test_indices = zero_shot_split_indices(all_data, ds_cfg)

    if not test_spks:
        print("❌ Nenhum locutor de teste (revisa zero_shot_split / num_speakers).", file=sys.stderr)
        return 1

    print(f"👥 Locutores teste ({len(test_spks)}): {sorted(test_spks)}")
    print(f"🎙️ Amostras indexadas no teste: {len(test_indices)}")

    by_spk: Dict[str, List[int]] = defaultdict(list)
    for i in test_indices:
        s = extract_speaker_id(all_data[i])
        if s in test_spks:
            by_spk[s].append(i)

    ds_audio = full_ds.cast_column("audio", Audio(sampling_rate=target_sr))
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "dataset_profile": args.dataset,
        "speaker_encoder_id": encoder_id,
        "target_sr": target_sr,
        "aggregate": args.aggregate,
        "export_all_utterances": bool(args.export_all_utterances),
        "test_speakers": sorted(test_spks),
        "speakers": {},
    }

    export_all = bool(args.export_all_utterances)
    min_samp = max(1, args.min_audio_samples)
    min_pk = float(args.min_audio_peak)

    for spk in sorted(test_spks):
        idxs = by_spk.get(spk, [])
        if not idxs:
            print(f"⚠️ Locutor {spk!r} sem amostras no teste — omitido.", file=sys.stderr)
            continue
        lim = idxs[: args.max_utts_per_speaker] if args.max_utts_per_speaker > 0 else idxs
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in spk)
        saved_exports: List[Dict[str, Any]] = []

        vecs_agg: List[np.ndarray] = []
        used_indices_agg: List[int] = []
        chosen_source = ""
        chosen_index: Optional[int] = None
        clip_meta: Dict[str, Any] = {}

        for j in lim:
            row = ds_audio[j]
            au = row["audio"]
            y = np.asarray(au["array"], dtype=np.float32).reshape(-1)
            sr = int(au.get("sampling_rate", target_sr))
            transcript = row_transcript(all_data[j])
            prov = describe_audio_provenance(all_data[j])
            is_valid = waveform_is_valid(y, min_samples=min_samp, min_peak=min_pk)
            dur_s = float(y.size) / max(sr, 1)

            v_enc: Optional[np.ndarray] = None

            if args.aggregate == "mean":
                v_enc = encode_one_utterance(y, sr, target_sr, speaker_model)
                vecs_agg.append(v_enc)
                used_indices_agg.append(j)
            elif args.aggregate == "first" and not vecs_agg and is_valid:
                v_enc = encode_one_utterance(y, sr, target_sr, speaker_model)
                vecs_agg.append(v_enc)
                used_indices_agg.append(j)
                chosen_index = j
                chosen_source = prov
                clip_meta = {
                    "clip_n_samples": int(y.size),
                    "clip_sample_rate_hz": sr,
                    "clip_duration_sec": round(dur_s, 4),
                }

            write_wav_txt = export_all or args.aggregate == "mean" or (
                args.aggregate == "first" and not export_all and is_valid and len(vecs_agg) > 0 and len(saved_exports) == 0
            )

            if write_wav_txt:
                ex = save_wav_and_transcript_sidecar(
                    out_path,
                    safe_spk=safe,
                    dataset_index=j,
                    y=y,
                    sr=sr,
                    transcript=transcript,
                )
                u_entry: Dict[str, Any] = {
                    **ex,
                    "dataset_index": j,
                    "is_valid_clip": is_valid,
                    "audio_source": prov,
                    "clip_n_samples": int(y.size),
                    "clip_sample_rate_hz": sr,
                    "clip_duration_sec": round(dur_s, 4),
                }
                if export_all and is_valid:
                    if v_enc is None:
                        v_enc = encode_one_utterance(y, sr, target_sr, speaker_model)
                    u_entry["embedding_npy"] = save_embedding_npy_sidecar(
                        out_path, safe_spk=safe, dataset_index=j, vec=v_enc
                    )
                    print(
                        f"   🎤 {spk} idx={j} | {y.size} amostras (~{dur_s:.2f}s @ {sr} Hz) | "
                        f"{ex['wav']} + {ex['txt']} + {u_entry['embedding_npy']}"
                    )
                elif not (args.aggregate == "first" and not export_all):
                    print(f"        → {ex['wav']} + {ex['txt']} (idx={j})")
                saved_exports.append(u_entry)

            if args.aggregate == "first" and not export_all:
                if vecs_agg:
                    print(
                        f"   🎤 {spk}  |  índice_dataset={chosen_index}  |  "
                        f"clip: {clip_meta.get('clip_n_samples')} amostras "
                        f"(~{clip_meta.get('clip_duration_sec')}s @ {clip_meta.get('clip_sample_rate_hz')} Hz)\n"
                        f"        origem_meta: {chosen_source}",
                    )
                    break

        if not vecs_agg:
            print(
                f"⚠️ Locutor {spk!r}: nenhum áudio válido para agregação entre {len(lim)} candidato(s) — omitido.",
                file=sys.stderr,
            )
            continue

        if args.aggregate == "mean":
            print(
                f"   📊 {spk}: agregado média sobre {len(vecs_agg)} utterance(s); "
                f"índices {used_indices_agg[:6]}{'…' if len(used_indices_agg) > 6 else ''}",
            )
        elif export_all:
            n_emb = sum(1 for e in saved_exports if e.get("embedding_npy"))
            print(f"   📂 {spk}: {len(saved_exports)} clipes exportados ({n_emb} com _emb.npy válidos)")

        emb = aggregate_embeddings(vecs_agg, args.aggregate)
        npy_file = out_path / f"spk_{safe}.npy"
        np.save(npy_file, emb)
        entry: Dict[str, Any] = {
            "npy": str(npy_file.name),
            "n_utts_used": len(vecs_agg),
            "indices": used_indices_agg,
        }
        if chosen_index is not None:
            entry["dataset_index"] = chosen_index
            entry["audio_source"] = chosen_source
            entry.update(clip_meta)
        if saved_exports:
            entry["exported_clips"] = saved_exports
        manifest["speakers"][spk] = entry
        print(f"      → gravado {npy_file.name}  shape={emb.shape}")

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n📄 manifest: {out_path / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
