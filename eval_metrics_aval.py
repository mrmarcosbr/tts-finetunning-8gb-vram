"""
Avaliação em pasta metrics_aval: WER (via ASR) e F0 RMSE (YIN + alinhamento temporal).

Convénio de ficheiros (igual a f0_infer_metrics.py):
  {prefix}_referencia.wav — gravação de referência (dataset / locutor)
  {prefix}_original.wav   — síntese modelo base
  {prefix}_treinado.wav   — síntese após fine-tuning

WER:
  Whisper transcreve *_original.wav e *_treinado.wav. O texto de referência (WER) pode ser:
    • {prefix}.txt — ex.: sentenca_1.txt (comum a sentenca_1_original/referencia/treinado.wav)
    • {prefix}_referencia.txt — ex.: sentenca_1_referencia.txt (alinhado ao nome do WAV de referência)
  Ou TSV/JSON em lote (ver load_reference_map).

  Modo alternativo (--wer-vs-asr-referencia): referência = ASR(*_referencia.wav), sem ficheiros .txt.

Dependências: librosa, numpy, torch, transformers, soundfile (já usadas no projeto).

Pós-processo só no treinado (opcional): os mesmos parâmetros que em test_inference_exhaustive.py
(--apply_trained_bass_treatment, --trained_* , --eval_reapply_speecht5_highpass_hz).
Quando activos, o *treinado* é reprocessado numa pasta temporária; F0 RMSE e WER (hipótese treinado)
usam esse sinal. Não combines com ficheiros que já foram gravados com o mesmo EQ na inferência
(re-aplicação dupla).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

import shutil
import tempfile

import librosa
import soundfile as sf

from f0_infer_metrics import (
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SR,
    _INFERENCE_PAIR_EXTS,
    compute_metrics_for_inference_dir,
    iter_inference_audio_triplets,
)
from trained_post_eq import apply_trained_output_waveform_eq, trained_eq_affects_audio


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def wer_prepare(s: str) -> str:
    s = normalize_text(s)
    return re.sub(r"[^\w\s]", "", s.lower().strip())


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """WER por palavra, mesmo critério que test_inference_exhaustive.calculate_wer."""
    ref_words = wer_prepare(reference_text).split()
    hyp_words = wer_prepare(hypothesis_text).split()
    if not ref_words:
        return 0.0
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint32)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j]) + 1
    return float(d[len(ref_words)][len(hyp_words)]) / len(ref_words)


def load_reference_map(metrics_dir: Path, explicit_path: Optional[str]) -> Dict[str, str]:
    """
    Ordem de precedência:
      1) --references (JSON ou TSV: prefix<TAB>texto); o ficheiro tem de existir se for passado.
      2) metrics_dir / reference_transcripts.json
      3) metrics_dir / reference_transcripts.tsv ou reference_texts.tsv
      4) Todos os metrics_dir / *_referencia.txt → chave = parte antes de _referencia
      5) metrics_dir / {prefix}.txt para cada prefixo de WAV ainda em falta
    """
    refs: Dict[str, str] = {}
    tried: List[str] = []

    if explicit_path:
        ep = Path(explicit_path)
        if not ep.is_file():
            raise FileNotFoundError(f"--references inexistente: {ep.resolve()}")

    def merge_from_json(path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON deve ser objeto prefixo→texto: {path}")
        for k, v in data.items():
            refs[str(k).strip()] = str(v).strip()

    def merge_from_tsv(path: Path) -> None:
        with open(path, encoding="utf-8", newline="") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                refs[str(parts[0]).strip()] = str(parts[1]).strip()

    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    else:
        candidates.extend(
            [
                metrics_dir / "reference_transcripts.json",
                metrics_dir / "reference_transcripts.tsv",
                metrics_dir / "reference_texts.tsv",
            ]
        )

    for p in candidates:
        if not p.is_file():
            tried.append(str(p))
            continue
        tried.append(str(p))
        if p.suffix.lower() == ".json":
            merge_from_json(p)
        else:
            merge_from_tsv(p)
        if refs:
            return refs

    # Por-utterance .txt (ex.: sentenca_1.txt ou sentenca_1_referencia.txt)
    prefixes = set()
    for ext in _INFERENCE_PAIR_EXTS:
        for tail in (f"_original{ext}", f"_treinado{ext}", f"_referencia{ext}"):
            for aud in metrics_dir.glob(f"*{tail}"):
                if not aud.is_file() or not aud.name.lower().endswith(tail.lower()):
                    continue
                prefixes.add(aud.name[: -len(tail)])

    for p in sorted(metrics_dir.glob("*_referencia.txt")):
        m = re.match(r"(.+)_referencia\.txt$", p.name)
        if m:
            refs[m.group(1)] = p.read_text(encoding="utf-8").strip()

    for pref in sorted(prefixes):
        if pref in refs:
            continue
        tp = metrics_dir / f"{pref}.txt"
        if tp.is_file():
            refs[pref] = tp.read_text(encoding="utf-8").strip()

    return refs


_asr_pipe = None


def get_asr_pipeline(
    model_id: str,
    device: str,
    torch_dtype: Optional[Any] = None,
):
    global _asr_pipe
    if _asr_pipe is not None:
        return _asr_pipe
    import torch
    from transformers import pipeline

    if torch_dtype is None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    dev: Union[int, str]
    if device == "cpu" or device == "-1":
        dev = -1
    elif device == "cuda" or device.startswith("cuda:"):
        try:
            dev = int(device.split(":")[1]) if ":" in device else 0
        except (IndexError, ValueError):
            dev = 0
    else:
        try:
            dev = int(device)
        except ValueError:
            dev = device

    try:
        _asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=dev,
            chunk_length_s=30,
        )
    except TypeError:
        _asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            dtype=torch_dtype,
            device=dev,
            chunk_length_s=30,
        )
    return _asr_pipe


def transcribe_wav(
    wav_path: Path,
    *,
    language: Optional[str],
    model_id: str,
    device: str,
) -> str:
    pipe = get_asr_pipeline(model_id, device)
    kw: Dict[str, Any] = {}
    if language:
        kw["language"] = language
        kw["task"] = "transcribe"
    # Caminho ou array; o pipeline reamostra para a taxa do modelo
    if kw:
        out = pipe(str(wav_path), generate_kwargs=kw)
    else:
        out = pipe(str(wav_path))
    if isinstance(out, dict):
        return str(out.get("text", "")).strip()
    return str(out).strip()


def iter_triplets(metrics_dir: Path) -> Iterable[Tuple[str, Path, Path, Optional[Path]]]:
    """Delega em f0_infer_metrics (wav / mp3 / …)."""
    yield from iter_inference_audio_triplets(metrics_dir)


def _eval_reprocess_treinado_waveform(args: argparse.Namespace) -> bool:
    hpre = float(getattr(args, "eval_reapply_speecht5_highpass_hz", 0) or 0)
    if hpre > 0:
        return True
    return trained_eq_affects_audio(args)


def prepare_triplets_and_f0_dir_for_eval(
    metrics_dir: Path,
    triplets: List[Tuple[str, Path, Path, Optional[Path]]],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, Path, Path, Optional[Path]]], Path, Optional[Path]]:
    """
    Se o pós-processo treinado estiver activo, grava cópias em pasta temporária
    (original/ref copiados; treinado filtrado) e devolve esse caminho para F0.

    Retorno: (triplets_para_wer paths, f0_dir, work_dir_ou_None_para_cleanup)
    """
    if not _eval_reprocess_treinado_waveform(args):
        return triplets, metrics_dir, None

    work = Path(tempfile.mkdtemp(prefix="eval_trained_eq_"))
    hpre = float(getattr(args, "eval_reapply_speecht5_highpass_hz", 0) or 0)
    hpre_opt = hpre if hpre > 0 else None
    out: List[Tuple[str, Path, Path, Optional[Path]]] = []
    try:
        for prefix, p_orig, p_train, p_ref in triplets:
            shutil.copy2(p_orig, work / p_orig.name)
            if p_ref is not None:
                shutil.copy2(p_ref, work / p_ref.name)
            y, sr = librosa.load(str(p_train), sr=int(args.sample_rate), mono=True)
            y2 = apply_trained_output_waveform_eq(
                np.asarray(y, dtype=np.float32), int(sr), args, speecht5_highpass_hz=hpre_opt
            )
            p_t_out = work / f"{prefix}_treinado.wav"
            sf.write(str(p_t_out), y2, int(sr))
            r_out: Optional[Path] = (work / p_ref.name) if p_ref is not None else None
            out.append((prefix, work / p_orig.name, p_t_out, r_out))
        return out, work, work
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def mean_finite(values: List[float]) -> Tuple[float, int]:
    xs = [x for x in values if np.isfinite(x)]
    if not xs:
        return float("nan"), 0
    return float(np.mean(xs)), len(xs)


def _parse_float_metric(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return x if np.isfinite(x) else None
    try:
        x = float(str(v).strip())
        return x if np.isfinite(x) else None
    except ValueError:
        return None


def print_terminal_per_audio_report(
    rows: List[Dict[str, Any]],
    *,
    include_wer: bool,
    include_f0: bool,
) -> None:
    """Resumo legível no terminal (poucas casas decimais), uma linha por prefixo de áudio."""
    if not rows or (not include_wer and not include_f0):
        return
    print("\n=== Por áudio ===")
    print("WER: percentagem de palavras erradas (menor é melhor). F0 RMSE: Hz (menor é melhor).")
    body: List[List[str]] = []
    for r in rows:
        p = str(r.get("pair_prefix", ""))
        line: List[str] = [p]
        if include_wer:
            wo = _parse_float_metric(r.get("wer_original"))
            wt = _parse_float_metric(r.get("wer_treinado"))
            line.append(f"{100.0 * wo:.1f}%" if wo is not None else "—")
            line.append(f"{100.0 * wt:.1f}%" if wt is not None else "—")
        if include_f0:
            fb = _parse_float_metric(r.get("f0_rmse_hz_vs_base"))
            fo = _parse_float_metric(r.get("f0_rmse_hz_original_vs_ref"))
            ft = _parse_float_metric(r.get("f0_rmse_hz_treinado_vs_ref"))
            line.append(f"{fb:.2f}" if fb is not None else "—")
            line.append(f"{fo:.2f}" if fo is not None else "—")
            line.append(f"{ft:.2f}" if ft is not None else "—")
        body.append(line)

    headers: List[str] = ["prefix"]
    if include_wer:
        headers.extend(["WER_orig", "WER_treinado"])
    if include_f0:
        headers.extend(["F0_tr↔base", "F0_orig↔ref", "F0_tr↔ref"])

    widths = [
        max(len(headers[i]), max((len(row[i]) for row in body), default=0))
        for i in range(len(headers))
    ]
    head_s = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(head_s)
    print("-" * len(head_s))
    for row in body:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    if include_f0:
        print(
            "Nota: F0_tr↔base = treinado vs original sintético | "
            "F0_orig↔ref / F0_tr↔ref = vs áudio de referência (se existir)."
        )


def _round_metric_value_for_csv(key: str, v: Any) -> Any:
    x = _parse_float_metric(v)
    if x is None:
        return v
    if key in ("wer_original", "wer_treinado"):
        return f"{x:.2f}"
    if key in (
        "f0_rmse_hz_vs_base",
        "f0_rmse_hz_original_vs_ref",
        "f0_rmse_hz_treinado_vs_ref",
    ):
        return f"{x:.2f}"
    return v


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="WER + F0 RMSE sobre metrics_aval (tripletos *_referencia/original/treinado)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="metrics_aval",
        help="Pasta com WAVs (predef.: metrics_aval).",
    )
    parser.add_argument(
        "--references",
        type=str,
        default=None,
        help="JSON ou TSV com transcrições (prefixo → texto). Se omitido, procura ficheiros padrão na pasta.",
    )
    parser.add_argument(
        "--wer-vs-asr-referencia",
        action="store_true",
        help="WER com referência = ASR(*_referencia.wav) em vez de texto fornecido.",
    )
    parser.add_argument(
        "--skip-wer",
        action="store_true",
        help="Só F0 RMSE.",
    )
    parser.add_argument(
        "--skip-f0",
        action="store_true",
        help="Só WER.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-small",
        help="Checkpoint Hugging Face para ASR.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu ou índice GPU.")
    parser.add_argument("--whisper-language", type=str, default="portuguese", help="ISO ou nome (ex.: portuguese).")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--hop_length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--fmin", type=float, default=DEFAULT_FMIN)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="CSV de saída (predef.: <dir>/metrics_wer_f0.csv).",
    )
    parser.add_argument(
        "--eval_reapply_speecht5_highpass_hz",
        type=float,
        default=0.0,
        metavar="HZ",
        help=(
            "Se > 0, reaplica high-pass tipo SpeechT5 ao *_treinado antes do resto da cadeia EQ. "
            "Use só se os WAVs não reflectirem esse passo (normalmente 0 para não duplicar)."
        ),
    )
    parser.add_argument(
        "--trained_bass_cut_hz",
        type=float,
        default=120.0,
        metavar="HZ",
        help="Com --apply_trained_bass_treatment: até que Hz atenuar graves.",
    )
    parser.add_argument(
        "--trained_bass_atten_db",
        type=float,
        default=1.2,
        metavar="DB",
        help="Com --apply_trained_bass_treatment: atenuação em dB na banda grave.",
    )
    parser.add_argument(
        "--trained_bass_transition_hz",
        type=float,
        default=120.0,
        metavar="HZ",
        help="Largura da rampa (Hz) na curva de graves.",
    )
    parser.add_argument(
        "--trained_grave_highpass_hz",
        type=float,
        default=0.0,
        metavar="HZ",
        help="High-pass extra (Hz) só no treinado; 0 = desligado.",
    )
    parser.add_argument(
        "--trained_treble_shelf_hz",
        type=float,
        default=3500.0,
        metavar="HZ",
        help="Início da prateleira de agudos quando --trained_treble_boost_db > 0.",
    )
    parser.add_argument(
        "--trained_treble_transition_hz",
        type=float,
        default=1000.0,
        metavar="HZ",
        help="Rampa (Hz) até ganho de agudos pleno.",
    )
    parser.add_argument(
        "--trained_treble_boost_db",
        type=float,
        default=0.0,
        metavar="DB",
        help="Boost de agudos (dB) só no treinado; 0 = off.",
    )
    parser.add_argument(
        "--apply_trained_bass_treatment",
        action="store_true",
        help="Ativa atenuação suave de graves (--trained_bass_* ) no treinado (eval/F0/WER hypo treinado).",
    )
    args = parser.parse_args(argv)

    metrics_dir = Path(args.dir)
    if not metrics_dir.is_dir():
        print(f"[erro] Pasta inexistente: {metrics_dir.resolve()}", file=sys.stderr)
        return 1

    triplets_raw = list(iter_triplets(metrics_dir))
    if not triplets_raw:
        print(
            f"[erro] Nenhum par *_original + *_treinado em {metrics_dir} (.wav/.mp3/…; extensões podem misturar).",
            file=sys.stderr,
        )
        return 1

    if not args.skip_wer:
        if args.wer_vs_asr_referencia:
            missing_ref = [p for p, _, _, pr in triplets_raw if pr is None]
            if missing_ref:
                print(
                    "[erro] --wer-vs-asr-referencia exige *_referencia.wav em todas as sentenças. "
                    f"Faltam referência em: {len(missing_ref)} caso(s).",
                    file=sys.stderr,
                )
                return 1
        else:
            ref_map = load_reference_map(metrics_dir, args.references)
    else:
        ref_map = {}

    triplets, f0_dir, work_eq_dir = prepare_triplets_and_f0_dir_for_eval(
        metrics_dir, triplets_raw, args
    )
    if work_eq_dir is not None:
        print(
            "📌 Pós-processo treinado activo: F0 RMSE e WER (treinado) usam WAV reprocessado (temp).",
            file=sys.stderr,
        )

    # Device string
    device = str(args.device).strip()
    if device.lower() == "cuda":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows_out: List[Dict[str, Any]] = []
    wer_orig_list: List[float] = []
    wer_train_list: List[float] = []
    f0_rows: List[Dict[str, Any]] = []

    try:
        for prefix, p_orig, p_train, p_ref_wav in triplets:
            row: Dict[str, Any] = {"pair_prefix": prefix}
            ref_text_for_wer = ""
            if not args.skip_wer:
                if args.wer_vs_asr_referencia:
                    assert p_ref_wav is not None
                    try:
                        ref_text_for_wer = transcribe_wav(
                            p_ref_wav,
                            language=args.whisper_language,
                            model_id=args.whisper_model,
                            device=device,
                        )
                    except Exception as exc:
                        print(f"[aviso] ASR referência falhou {prefix}: {exc}", file=sys.stderr)
                        ref_text_for_wer = ""
                    row["ref_asr_text"] = ref_text_for_wer
                else:
                    ref_text_for_wer = (ref_map.get(prefix) or "").strip()
                    if not ref_text_for_wer and p_ref_wav is not None:
                        for sidecar in (
                            p_ref_wav.with_suffix(".txt"),
                            metrics_dir / Path(p_ref_wav.name).with_suffix(".txt"),
                        ):
                            if sidecar.is_file():
                                ref_text_for_wer = sidecar.read_text(encoding="utf-8").strip()
                                break
                    if not ref_text_for_wer:
                        tp = metrics_dir / f"{prefix}.txt"
                        if tp.is_file():
                            ref_text_for_wer = tp.read_text(encoding="utf-8").strip()
                    if not ref_text_for_wer:
                        print(
                            f"[aviso] Sem texto de referência para prefixo {prefix!r} — WER omitido.",
                            file=sys.stderr,
                        )

                if ref_text_for_wer:
                    try:
                        hyp_o = transcribe_wav(
                            p_orig,
                            language=args.whisper_language,
                            model_id=args.whisper_model,
                            device=device,
                        )
                        hyp_t = transcribe_wav(
                            p_train,
                            language=args.whisper_language,
                            model_id=args.whisper_model,
                            device=device,
                        )
                        wo = word_error_rate(ref_text_for_wer, hyp_o)
                        wt = word_error_rate(ref_text_for_wer, hyp_t)
                        row["wer_original"] = wo
                        row["wer_treinado"] = wt
                        row["asr_original"] = hyp_o
                        row["asr_treinado"] = hyp_t
                        wer_orig_list.append(wo)
                        wer_train_list.append(wt)
                    except Exception as exc:
                        row["wer_failure"] = str(exc)
                        print(f"[aviso] WER falhou {prefix}: {exc}", file=sys.stderr)

            rows_out.append(row)

        if not args.skip_f0:
            f0_rows = compute_metrics_for_inference_dir(
                f0_dir,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            by_prefix = {r["pair_prefix"]: r for r in f0_rows}
            for r in rows_out:
                p = r["pair_prefix"]
                if p in by_prefix:
                    for k, v in by_prefix[p].items():
                        if k != "pair_prefix" and k not in r:
                            r[k] = v
    finally:
        if work_eq_dir is not None:
            shutil.rmtree(work_eq_dir, ignore_errors=True)

    out_csv = Path(args.out_csv) if args.out_csv else metrics_dir / "metrics_wer_f0.csv"
    fieldnames: List[str] = []
    for r in rows_out:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    pref = [
        "pair_prefix",
        "wer_original",
        "wer_treinado",
        "f0_rmse_hz_vs_base",
        "f0_rmse_hz_original_vs_ref",
        "f0_rmse_hz_treinado_vs_ref",
    ]
    ordered = [k for k in pref if k in fieldnames] + [k for k in fieldnames if k not in pref]
    _csv_metric_keys = (
        "wer_original",
        "wer_treinado",
        "f0_rmse_hz_vs_base",
        "f0_rmse_hz_original_vs_ref",
        "f0_rmse_hz_treinado_vs_ref",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            row_csv = {fn: r.get(fn, "") for fn in ordered}
            for mk in _csv_metric_keys:
                if mk in row_csv:
                    row_csv[mk] = _round_metric_value_for_csv(mk, row_csv[mk])
            w.writerow(row_csv)

    mo, no = mean_finite(wer_orig_list)
    mt, nt = mean_finite(wer_train_list)
    print(f"[ok] Linhas: {len(rows_out)} | CSV: {out_csv.resolve()}")

    print_terminal_per_audio_report(
        rows_out,
        include_wer=not args.skip_wer,
        include_f0=not args.skip_f0,
    )

    print("\n--- Médias ---")
    if not args.skip_wer and (no > 0 or nt > 0):
        print(
            f"  WER médio (original):   {100.0 * mo:.2f}%  (n={no})"
            if np.isfinite(mo)
            else "  WER médio (original):   (sem valores)"
        )
        print(
            f"  WER médio (treinado):   {100.0 * mt:.2f}%  (n={nt})"
            if np.isfinite(mt)
            else "  WER médio (treinado):   (sem valores)"
        )
    if not args.skip_f0:
        from f0_infer_metrics import summarize_f0_metrics_rows, summarize_f0_metrics_vs_reference

        hz_b, v_b, _ = summarize_f0_metrics_rows(f0_rows)
        mo_f0, ko, mt_f0, kt = summarize_f0_metrics_vs_reference(f0_rows)
        if np.isfinite(hz_b):
            print(
                f"  F0 RMSE médio treinado vs base:    {hz_b:.2f} Hz (válidos={v_b}) "
                "(LoRA vs síntese sem LoRA; mesmo texto)"
            )
        if ko > 0 and np.isfinite(mo_f0):
            print(
                f"  F0 RMSE médio base vs ref:         {mo_f0:.2f} Hz (n={ko}) "
                "(*_original.wav = síntese sem LoRA vs gravação)"
            )
        if kt > 0 and np.isfinite(mt_f0):
            print(
                f"  F0 RMSE médio treinado vs ref:     {mt_f0:.2f} Hz (n={kt}) "
                "(*_treinado.wav = síntese LoRA vs gravação)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
