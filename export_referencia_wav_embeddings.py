#!/usr/bin/env python3
"""
Exporta um x-vector SpeechBrain (512-D, L2-normalizado) por ficheiro `*_referencia.wav`
numa pasta de batch de inferência.

Usa o mesmo `speaker_encoder_id` e `sampling_rate` que o perfil SpeechT5 no `config.yaml`.

Exemplo:
  python export_referencia_wav_embeddings.py \\
    --input_dir ".\\output_cuda_16gb\\speecht5-lapsbm_speecht5-2026-04-29-21-20-04\\checkpoint-11600\\inference_batch_20260429_235921" \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import torch

from train_exhaustive import (
    _device_for_speechbrain,
    load_config,
    merge_dataset_model_overrides,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embeddings x-vector por *_referencia.wav (pasta de inferência)."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=Path,
        required=True,
        help="Pasta com ficheiros *_referencia.wav",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lapsbm_speecht5",
        help="Chave em dataset_profiles para resolver speaker_encoder_id / sampling_rate.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=None,
        help="Onde gravar os .npy (predef.: <input_dir>/referencia_embeddings).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_referencia.wav",
        help="Glob relativamente a input_dir.",
    )
    parser.add_argument(
        "--speaker_encoder_id",
        type=str,
        default=None,
        help="Sobrepor o ID HuggingFace/SpeechBrain do classificador.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=None,
        help="Sobrepor taxa de amostragem esperada do encoder (predef.: modelo no YAML).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    inp: Path = args.input_dir.expanduser().resolve()
    if not inp.is_dir():
        print(f"❌ Pasta inexistente: {inp}", file=sys.stderr)
        return 1

    wav_paths = sorted(inp.glob(args.pattern))
    if not wav_paths:
        print(f"❌ Nenhum ficheiro com padrão {args.pattern!r} em {inp}", file=sys.stderr)
        return 1

    full_cfg = load_config(args.config)
    if args.dataset not in full_cfg.get("dataset_profiles", {}):
        print(f"❌ dataset_profiles['{args.dataset}'] não encontrado.", file=sys.stderr)
        return 1

    ds_cfg = full_cfg["dataset_profiles"][args.dataset]
    model_type = ds_cfg.get("model_type", "speecht5")
    models = full_cfg.get("models", {})
    if model_type not in models:
        print(f"❌ models['{model_type}'] ausente.", file=sys.stderr)
        return 1

    model_cfg = merge_dataset_model_overrides(models[model_type], ds_cfg)
    encoder_id = args.speaker_encoder_id or model_cfg.get("speaker_encoder_id")
    if not encoder_id:
        print("❌ Sem speaker_encoder_id (use --speaker_encoder_id ou config).", file=sys.stderr)
        return 1

    target_sr = int(args.sampling_rate or model_cfg.get("sampling_rate", 16000))

    dev = str(args.device).strip()
    if dev.lower() == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
        print("⚠️ CUDA indisponível; usando CPU.", file=sys.stderr)

    from speechbrain.inference.classifiers import EncoderClassifier

    speaker_model = EncoderClassifier.from_hparams(
        source=encoder_id,
        run_opts={"device": _device_for_speechbrain(dev)},
    )

    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else inp / "referencia_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []

    print(f"📂 Entrada: {inp}  |  {len(wav_paths)} wav | encoder={encoder_id} | sr={target_sr}")

    for wav_path in wav_paths:
        try:
            y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        except Exception as e:
            print(f"⚠️ Falha ao ler {wav_path.name}: {e}", file=sys.stderr)
            continue
        vec = encode_one_utterance(np.asarray(y, dtype=np.float32), int(sr), target_sr, speaker_model)
        out_npy = out_dir / f"{wav_path.stem}_emb.npy"
        np.save(str(out_npy), vec.astype(np.float32))
        entries.append(
            {
                "source_wav": str(wav_path.relative_to(inp)),
                "embedding_npy": out_npy.name,
                "shape": list(vec.shape),
            }
        )
        print(f"   ✓ {wav_path.name}  →  {out_npy.name}  shape={vec.shape}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(inp),
                "pattern": args.pattern,
                "speaker_encoder_id": encoder_id,
                "sampling_rate": target_sr,
                "n_files": len(entries),
                "files": entries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"📄 {manifest_path}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
