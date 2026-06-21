"""
WER/CER médios e por ficheiro (Whisper) só sobre `*_base_teste_treinado.wav`,
`*_base_treino_treinado.wav`, `*_sem_embedding_treinado.wav` numa pasta (ex.: `.../treinado_wav`).

Referências: `transcricoes.tsv` na pasta-mãe do run (col1 = sentenca_N, col2 = texto).

No CSV, as linhas por sentença vêm ordenadas como sentenca_1 … sentenca_13 (não lexicográfico).
No ficheiro acrescentam-se linhas de resumo: ``{grupo}_media`` e ``{grupo}_media_curtas`` (sent. 1–10).

  python triple_treinado_wer_cer_means.py "E:\\...\\treinado_wav"
  python triple_treinado_wer_cer_means.py "E:\\...\\treinado_wav" --no-csv --verbose
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval_metrics_aval import (
    load_reference_map,
    transcribe_wav,
    word_error_rate,
    char_error_rate,
)


GROUPS: Tuple[Tuple[str, str], ...] = (
    ("*_base_teste_treinado.wav", "base_teste"),
    ("*_base_treino_treinado.wav", "base_treino"),
    ("*_sem_embedding_treinado.wav", "sem_embedding"),
)


def _sentence_key_from_stem(stem: str) -> Optional[str]:
    m = re.match(r"(?i)^(sentenca_\d+)_", stem)
    return m.group(1).lower() if m else None


def _sentence_index_from_key(sentenca_key: str) -> Optional[int]:
    m = re.match(r"(?i)^sentenca_(\d+)$", (sentenca_key or "").strip())
    return int(m.group(1)) if m else None


def _treinado_wav_sort_key(path: Path) -> Tuple[int, int, str]:
    """sentenca_1 … sentenca_13 por número; restantes (sem índice) ao fim."""
    sk = _sentence_key_from_stem(path.stem)
    n = _sentence_index_from_key(sk or "")
    if n is not None:
        return (0, n, path.name.lower())
    return (1, 0, path.name.lower())


def _mean(vals: List[float]) -> Tuple[float, int]:
    good = [float(x) for x in vals if np.isfinite(x)]
    if not good:
        return float("nan"), 0
    return float(np.mean(good)), len(good)


def _summary_csv_row(
    resumo_id: str,
    wers: List[float],
    cers: List[float],
) -> Dict[str, object]:
    mw, nw = _mean(wers)
    mc, nc = _mean(cers)
    row: Dict[str, object] = {
        "grupo": resumo_id,
        "sentenca_key": "",
        "wav": "",
        "referencia": "",
        "hipotese_whisper": "",
        "erro": "",
    }
    if nw > 0 and nc > 0:
        row["wer"] = mw
        row["cer"] = mc
        row["wer_percent"] = round(100.0 * mw, 6)
        row["cer_percent"] = round(100.0 * mc, 6)
        row["erro"] = f"resumo_n={nw}"
    else:
        row["wer"] = ""
        row["cer"] = ""
        row["wer_percent"] = ""
        row["cer_percent"] = ""
        row["erro"] = "resumo_sem_valores"
    return row


def build_summary_rows(detail_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Linhas finais: ``base_teste_media``, ``base_teste_media_curtas``, …"""
    out: List[Dict[str, object]] = []
    for _glob_pat, label in GROUPS:
        all_w: List[float] = []
        all_c: List[float] = []
        curtas_w: List[float] = []
        curtas_c: List[float] = []
        for r in detail_rows:
            if str(r.get("grupo") or "") != label:
                continue
            w = r.get("wer")
            c = r.get("cer")
            if not isinstance(w, (int, float)) or not isinstance(c, (int, float)):
                continue
            wf, cf = float(w), float(c)
            if not (np.isfinite(wf) and np.isfinite(cf)):
                continue
            n = _sentence_index_from_key(str(r.get("sentenca_key") or ""))
            if n is None:
                continue
            all_w.append(wf)
            all_c.append(cf)
            if 1 <= n <= 10:
                curtas_w.append(wf)
                curtas_c.append(cf)
        out.append(_summary_csv_row(f"{label}_media", all_w, all_c))
        out.append(_summary_csv_row(f"{label}_media_curtas", curtas_w, curtas_c))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="WER/CER por ficheiro + médios por grupo (default_texts triplo, só *_treinado.wav)."
    )
    p.add_argument("wav_dir", type=Path, help="Pasta com os WAV treinados (ex.: …/treinado_wav).")
    p.add_argument(
        "--references",
        type=Path,
        default=None,
        help="TSV prefixo TAB texto (default: pai/wav_dir/transcricoes.tsv).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV por ficheiro (default: pasta do run — pai de treinado_wav — /wer_cer_triplo_por_ficheiro.csv).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Não gravar CSV (só resumo no stdout).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Imprime uma linha por WAV (WER/CER) no stdout.",
    )
    p.add_argument("--wer_cer_whisper_model", type=str, default="openai/whisper-large-v3")
    p.add_argument("--wer_cer_whisper_language", type=str, default="portuguese")
    p.add_argument("--wer_cer_device", type=str, default="", help="cuda (default), cuda:N ou cpu.")
    p.add_argument(
        "--no-whisper-soft-prompt",
        action="store_true",
        help=(
            "Não aplicar texto auxiliar ao Whisper (pedido de português e números por extenso). "
            "Mantém apenas task=transcribe + language (--wer-cer-whisper-language)."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    wav_dir = args.wav_dir
    if not wav_dir.is_dir():
        print(f"[erro] Pasta inexistente: {wav_dir}", file=sys.stderr)
        return 2

    ref_path = args.references if args.references is not None else wav_dir.parent / "transcricoes.tsv"
    if not ref_path.is_file():
        print(f"[erro] Falta referência TSV: {ref_path}", file=sys.stderr)
        return 2

    out_csv = args.out_csv
    if out_csv is None and not args.no_csv:
        out_csv = wav_dir.parent / "wer_cer_triplo_por_ficheiro.csv"

    refs: Dict[str, str] = load_reference_map(Path(ref_path).parent, str(ref_path))
    dev = str(args.wer_cer_device).strip()
    if not dev or dev.lower() == "cuda":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = str(args.wer_cer_whisper_model)
    whisper_lang = str(args.wer_cer_whisper_language)

    print(f"Pasta WAV: {wav_dir.resolve()}")
    print(f"Referências: {ref_path.resolve()}")
    print(f"Whisper: {whisper_model} | idioma={whisper_lang} | device={dev}")
    if not args.no_csv and out_csv is not None:
        print(f"CSV por ficheiro: {out_csv.resolve()}\n")
    else:
        print()

    rows_out: List[Dict[str, object]] = []
    per_group_wer: Dict[str, List[float]] = {label: [] for _, label in GROUPS}
    per_group_cer: Dict[str, List[float]] = {label: [] for _, label in GROUPS}

    for glob_pat, label in GROUPS:
        paths = sorted(wav_dir.glob(glob_pat), key=_treinado_wav_sort_key)
        missing_ref = 0
        for wp in paths:
            key = _sentence_key_from_stem(wp.stem)
            base_row: Dict[str, object] = {
                "grupo": label,
                "sentenca_key": key or "",
                "wav": wp.name,
                "referencia": "",
                "hipotese_whisper": "",
                "wer": "",
                "cer": "",
                "wer_percent": "",
                "cer_percent": "",
                "erro": "",
            }
            if not key:
                base_row["erro"] = "nome_inesperado"
                rows_out.append(base_row)
                print(f"   [{label}] [ignorado] nome inesperado: {wp.name}")
                continue
            ref_t = (refs.get(key) or "").strip()
            base_row["referencia"] = ref_t
            if not ref_t:
                missing_ref += 1
                base_row["erro"] = "sem_texto_em_tsv"
                rows_out.append(base_row)
                print(f"   [{label}] [sem ref] {wp.name} (chave {key!r})")
                continue
            try:
                _tw_kw: Dict[str, object] = dict(
                    language=whisper_lang,
                    model_id=whisper_model,
                    device=dev,
                )
                if args.no_whisper_soft_prompt:
                    _tw_kw["transcribe_prompt"] = ""
                hyp = transcribe_wav(wp, **_tw_kw)
                w = float(word_error_rate(ref_t, hyp))
                c = float(char_error_rate(ref_t, hyp))
                base_row["hipotese_whisper"] = hyp
                base_row["wer"] = w
                base_row["cer"] = c
                base_row["wer_percent"] = round(100.0 * w, 6)
                base_row["cer_percent"] = round(100.0 * c, 6)
                per_group_wer[label].append(w)
                per_group_cer[label].append(c)
                if args.verbose:
                    print(
                        f"   [{label}] {wp.name}\tWER={100.0 * w:.4f}%\tCER={100.0 * c:.4f}%"
                    )
            except Exception as exc:
                base_row["erro"] = str(exc)
                rows_out.append(base_row)
                print(f"   [{label}] [erro ASR] {wp.name}: {exc}")
                continue
            rows_out.append(base_row)

        mw, nw = _mean(per_group_wer[label])
        mc, nc = _mean(per_group_cer[label])
        print(f"=== {label} ({glob_pat}) — ficheiros={len(paths)} válidos_WER/CER=n={nw} ===")
        if missing_ref:
            print(f"   (avisos: {missing_ref} sem texto em TSV)")
        if nw:
            print(f"   WER médio: {100.0 * mw:.4f}%")
            print(f"   CER médio: {100.0 * mc:.4f}%")
        else:
            print("   Sem valores (ver erros acima).")
        print()

    if not args.no_csv and out_csv is not None:
        fields = [
            "grupo",
            "sentenca_key",
            "wav",
            "wer",
            "cer",
            "wer_percent",
            "cer_percent",
            "referencia",
            "hipotese_whisper",
            "erro",
        ]
        summary = build_summary_rows(rows_out)
        csv_rows = rows_out + summary
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"Gravado: {out_csv.resolve()} ({len(rows_out)} linhas detalhe + {len(summary)} resumo)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
