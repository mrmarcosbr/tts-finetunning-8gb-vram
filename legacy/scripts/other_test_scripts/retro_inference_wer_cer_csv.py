#!/usr/bin/env python3
"""
Escreve inference_metrics_wer_cer.csv numa pasta de inferência antiga (ex.: *_original.wav).

Reutiliza ``compute_metrics_for_inference_dir`` (mesmo Whisper + WER/CER que o projeto).
Última linha: médias de wer_base, wer_treinado, cer_base, cer_treinado (só linhas com valores finitos).

Exemplo:
  python legacy/scripts/other_test_scripts/retro_inference_wer_cer_csv.py ^
    "E:/.../inference_batch_20260501_093052"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Raiz do repo (legacy/scripts/other_test_scripts → três níveis acima)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from f0_infer_metrics import (  # noqa: E402
    INFERENCE_METRICS_CSV_COLUMN_ORDER,
    _mean_column,
    compute_metrics_for_inference_dir,
    inference_pair_prefix_sort_key,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="WER/CER retroactivos → CSV + linha MÉDIA.")
    ap.add_argument("out_dir", type=Path, help="Pasta inference_batch_*")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Caminho do CSV (default: <out_dir>/inference_metrics_wer_cer.csv)",
    )
    ap.add_argument(
        "--whisper-model",
        default="openai/whisper-large-v3",
        help="Igual ao típico config_inference.yaml do repo.",
    )
    ap.add_argument("--language", default="portuguese")
    ap.add_argument("--device", default=None, help="cuda | cpu (omisso: cuda se disponível)")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    if not out_dir.is_dir():
        print(f"Pasta inexistente: {out_dir}", file=sys.stderr)
        return 1

    out_csv = (args.out_csv or (out_dir / "inference_metrics_wer_cer.csv")).resolve()

    dev = (args.device or "cuda").strip()
    print(f"Dir: {out_dir}")
    print(f"CSV: {out_csv}")
    print(f"Whisper: {args.whisper_model} | lang={args.language} | device={dev}")

    rows = compute_metrics_for_inference_dir(
        out_dir,
        compute_f0=False,
        wer_cer=True,
        whisper_model=str(args.whisper_model),
        whisper_language=str(args.language),
        asr_device=dev,
    )

    rows_sorted = sorted(
        rows,
        key=lambda r: inference_pair_prefix_sort_key(str(r.get("pair_prefix", "") or "")),
    )

    mb, nb = _mean_column(rows_sorted, "wer_base")
    mt, nt = _mean_column(rows_sorted, "wer_treinado")
    mcb, ncb = _mean_column(rows_sorted, "cer_base")
    mct, nct = _mean_column(rows_sorted, "cer_treinado")

    mean_row = {
        "pair_prefix": "MÉDIA",
        "wer_base": mb if nb else "",
        "wer_treinado": mt if nt else "",
        "cer_base": mcb if ncb else "",
        "cer_treinado": mct if nct else "",
        "f0_rmse_base": "",
        "f0_rmse_treinado": "",
        "failure_reason": "",
        "wav_base": "",
        "wav_treinado": "",
        "wav_referencia": "",
    }

    keys = list(INFERENCE_METRICS_CSV_COLUMN_ORDER)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k, "") for k in keys})
        w.writerow(mean_row)

    print(f"OK: {len(rows_sorted)} linhas + MEDIA -> {out_csv.name}")
    if nb:
        print(f"   WER médio base:      {mb:.6f} (n={nb})")
    if nt:
        print(f"   WER médio treinado: {mt:.6f} (n={nt})")
    if ncb:
        print(f"   CER médio base:      {mcb:.6f} (n={ncb})")
    if nct:
        print(f"   CER médio treinado: {mct:.6f} (n={nct})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
