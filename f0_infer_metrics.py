"""
Métricas de F0 no domínio de inferência (WAV × WAV).

Compara dois sinais (ex.: modelo treinado vs base SpeechT5) via librosa.yin e RMSE em Hz,
com alinhamento por tempo-normalização dos contours quando as durações divergem.

Uso:
  python f0_infer_metrics.py ./pasta_inference_batch_xxxxx

Ou importar `compute_metrics_for_inference_dir` a partir do script de inferência.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np

DEFAULT_SR = 16000
DEFAULT_HOP_LENGTH = 256
DEFAULT_FMIN = 50.0
DEFAULT_FMAX = 500.0

_INFERENCE_PAIR_EXTS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg")


def shorten_path_for_csv(path_like: str | Path | None) -> str:
    """
    Caminho apenas a partir do segmento checkpoint-NNNN/... para não encher o CSV em Windows.

    Sem esse segmento, devolve as últimas 3 partes ou só o basename.
    """
    if path_like is None:
        return ""
    s = str(path_like).strip()
    if not s:
        return ""
    norm = os.path.normpath(str(Path(s).resolve())).replace("\\", "/")
    m = re.search(r"(?i)(checkpoint-\d+/.*)", norm)
    if m:
        return m.group(1)
    parts = Path(s).resolve().parts
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return Path(s).name


def compute_yin_f0(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0 or sr <= 0:
        return np.array([], dtype=np.float64)
    return librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)


def _align_f0_time_normalize(f_a: np.ndarray, f_b: np.ndarray, num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    fa = np.asarray(f_a, dtype=np.float64).reshape(-1)
    fb = np.asarray(f_b, dtype=np.float64).reshape(-1)
    if fa.size == 0 or fb.size == 0:
        raise ValueError("contour F0 vazio")
    n = num_points if num_points is not None else max(fa.size, fb.size)
    n = max(int(n), 2)
    t_target = np.linspace(0.0, 1.0, num=n)
    ta = np.linspace(0.0, 1.0, num=len(fa))
    tb = np.linspace(0.0, 1.0, num=len(fb))
    ia = np.interp(t_target, ta, fa, left=np.nan, right=np.nan)
    ib = np.interp(t_target, tb, fb, left=np.nan, right=np.nan)
    return ia, ib


def f0_rmse_hz_between_contours(
    gen_f0: np.ndarray,
    ref_f0: np.ndarray,
    *,
    voiced_min_hz: float = 50.0,
) -> float:
    """RMSE(Hz) após igualar comprimento ao longo do eixo tempo normalizado; só quadros voiced em ambos."""
    g, r = _align_f0_time_normalize(gen_f0, ref_f0)
    mask = np.isfinite(g) & np.isfinite(r) & (g >= voiced_min_hz) & (r >= voiced_min_hz)
    if mask.sum() < 3:
        return float("nan")
    d = g[mask] - r[mask]
    return float(np.sqrt(np.mean(d * d)))


def f0_rmse_hz_between_wavs(
    path_generated: Path | str,
    path_reference: Path | str,
    *,
    sample_rate: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    voiced_min_hz: float = DEFAULT_FMIN,
) -> float:
    g, sr_g = librosa.load(str(path_generated), sr=sample_rate, mono=True)
    r, sr_r = librosa.load(str(path_reference), sr=sample_rate, mono=True)
    sample_rate = int(sr_g or sr_r or sample_rate)
    gf = compute_yin_f0(g, sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
    rf = compute_yin_f0(r, sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
    return f0_rmse_hz_between_contours(gf, rf, voiced_min_hz=voiced_min_hz)


def find_referencia_path(out_dir: Path, prefix: str) -> Optional[Path]:
    """
    Ficheiro `prefix_referencia.*`. Se existir mais do que uma extensão, prefere-se a ordem em
    `_INFERENCE_PAIR_EXTS` (.wav antes de .mp3) para métricas mais estáveis, independentemente do original.
    """
    candidates: List[Path] = []
    for e in _INFERENCE_PAIR_EXTS:
        p = out_dir / f"{prefix}_referencia{e}"
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None
    for e in _INFERENCE_PAIR_EXTS:
        for c in candidates:
            if c.suffix.lower() == e.lower():
                return c
    return candidates[0]


def _pick_treinado_for_original(out_dir: Path, prefix: str, orig_suffix: str) -> Optional[Path]:
    """Procura `{prefix}_treinado.*`: preferindo a mesma extensão que o original, depois as outras."""
    suf = orig_suffix.lower() if orig_suffix.startswith(".") else f".{orig_suffix.lower()}"
    order = [suf] + [e for e in _INFERENCE_PAIR_EXTS if e.lower() != suf]
    for e in order:
        p = out_dir / f"{prefix}_treinado{e}"
        if p.is_file():
            return p
    return None


def iter_speecht5_original_treinado_pairs(out_dir: Path | str) -> Iterable[Tuple[str, Path, Path]]:
    """
    Emparelha `{prefix}_original.*` com `{prefix}_treinado.*`.
    Extensões: .wav, .mp3, .flac, .ogg.
    O treinado pode ter extensão diferente do original (ex.: .wav + .mp3).
    Por prefixo conta-se um par (prioridade do original: wav, mp3, …).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    seen: set[str] = set()
    for ext in _INFERENCE_PAIR_EXTS:
        tail = f"_original{ext}"
        for orig in sorted(out_dir.glob(f"*{tail}")):
            if not orig.is_file():
                continue
            if not orig.name.lower().endswith(tail.lower()):
                continue
            prefix = orig.name[: -len(tail)]
            if prefix in seen:
                continue
            treina = _pick_treinado_for_original(out_dir, prefix, orig.suffix)
            if treina is not None:
                seen.add(prefix)
                yield prefix, orig, treina


def iter_inference_audio_triplets(
    out_dir: Path | str,
) -> Iterable[Tuple[str, Path, Path, Optional[Path]]]:
    """prefix, original, treinado, referencia (opcional)."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    for prefix, orig, train in iter_speecht5_original_treinado_pairs(out_dir):
        ref = find_referencia_path(out_dir, prefix)
        yield prefix, orig, train, ref


def compute_metrics_for_inference_dir(
    out_dir: Path | str,
    *,
    sample_rate: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> List[Dict[str, Any]]:
    """
    Por `{prefix}_original.*` + `{prefix}_treinado.*` (extensões podem diferir, ex. wav vs mp3):
      Treinado vs base sintético: f0_rmse_hz_vs_base
    Se existir `{prefix}_referencia.*`:
      Base vs disco: f0_rmse_hz_original_vs_ref ; Treinado vs disco: f0_rmse_hz_treinado_vs_ref
    """
    rows: List[Dict[str, Any]] = []
    out_dir = Path(out_dir)

    for prefix, p_orig, p_train in iter_speecht5_original_treinado_pairs(out_dir):
        p_ref = find_referencia_path(out_dir, prefix)
        row: Dict[str, Any] = {
            "pair_prefix": prefix,
            "wav_treinado": shorten_path_for_csv(p_train),
            "wav_original": shorten_path_for_csv(p_orig),
            "wav_referencia": shorten_path_for_csv(p_ref) if p_ref is not None else "",
            "f0_rmse_hz_vs_base": "",
            "f0_rmse_hz_original_vs_ref": "",
            "f0_rmse_hz_treinado_vs_ref": "",
            "failure_reason": "",
        }
        errs: List[str] = []

        try:
            r_base = f0_rmse_hz_between_wavs(
                p_train,
                p_orig,
                sample_rate=sample_rate,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
            )
            row["f0_rmse_hz_vs_base"] = r_base if np.isfinite(r_base) else ""
        except Exception as exc:
            errs.append(f"trein-vs-base:{exc}")

        if p_ref is not None:
            try:
                r_o = f0_rmse_hz_between_wavs(
                    p_orig,
                    p_ref,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    fmin=fmin,
                    fmax=fmax,
                )
                row["f0_rmse_hz_original_vs_ref"] = r_o if np.isfinite(r_o) else ""
            except Exception as exc:
                errs.append(f"original-vs-ref:{exc}")
            try:
                r_t = f0_rmse_hz_between_wavs(
                    p_train,
                    p_ref,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    fmin=fmin,
                    fmax=fmax,
                )
                row["f0_rmse_hz_treinado_vs_ref"] = r_t if np.isfinite(r_t) else ""
            except Exception as exc:
                errs.append(f"trein-vs-ref:{exc}")

        if errs:
            row["failure_reason"] = "; ".join(errs)
        rows.append(row)
    return rows


def write_metrics_csv(rows: List[Dict[str, Any]], out_csv: Path | str) -> None:
    out_csv = Path(out_csv)
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    pref_order = ["pair_prefix", "wav_original", "wav_treinado", "wav_referencia"]
    rest = sorted(k for k in keys if k not in pref_order and k != "failure_reason")
    fieldnames = [k for k in pref_order if k in keys] + rest + (["failure_reason"] if "failure_reason" in keys else [])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _mean_column(rows: List[Dict[str, Any]], key: str) -> Tuple[float, int]:
    vals: List[float] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            vals.append(float(v))
        elif isinstance(v, str) and str(v).strip() != "":
            try:
                fv = float(v)
                if np.isfinite(fv):
                    vals.append(fv)
            except ValueError:
                pass
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def summarize_f0_metrics_rows(rows: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    """Retrocompatível: médias na coluna trein-vs-base sintético."""
    total = len(rows)
    mv, kv = _mean_column(rows, "f0_rmse_hz_vs_base")
    return mv, kv, total


def summarize_f0_metrics_vs_reference(rows: List[Dict[str, Any]]) -> Tuple[float, int, float, int]:
    """Médias: base vs WAV do dataset ; treinado vs WAV do dataset."""
    mo, ko = _mean_column(rows, "f0_rmse_hz_original_vs_ref")
    mt, kt = _mean_column(rows, "f0_rmse_hz_treinado_vs_ref")
    return mo, ko, mt, kt



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calcula F0 RMSE (treinado vs original) sobre pastas de inferência SpeechT5.")
    parser.add_argument("out_dir", type=str, help="Pasta com pares *_original / *_treinado (.wav, .mp3, …)")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--hop_length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--fmin", type=float, default=DEFAULT_FMIN)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"[erro] Pasta inexistente: {out_dir}", file=sys.stderr)
        return 1

    rows = compute_metrics_for_inference_dir(
        out_dir,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    if not rows:
        print(f"[aviso] Nenhum par *_original / *_treinado (.wav, .mp3, …) em {out_dir}")
        return 0

    csv_path = out_dir / "f0_rmse.csv"
    write_metrics_csv(rows, csv_path)
    mean_hz, valid_n, total_n = summarize_f0_metrics_rows(rows)
    mo, ko, mt, kt = summarize_f0_metrics_vs_reference(rows)
    print(f"[ok] {total_n} prefixos avaliados | RMSE train-vs-base validos={valid_n}")
    print(
        f"   F0 RMSE medio treinado(LoRA) vs base sintetico: {mean_hz:.4f} Hz"
        if np.isfinite(mean_hz)
        else "   [aviso] Sem RMSE treinado(LoRA) vs base sintético"
    )
    if ko > 0 or kt > 0:
        if np.isfinite(mo):
            print(f"   F0 RMSE medio base sintet vs WAV dataset ({ko} valores): {mo:.4f} Hz")
        if np.isfinite(mt):
            print(f"   F0 RMSE medio LoRA vs WAV dataset ({kt} valores): {mt:.4f} Hz")
    print(f"   CSV: {csv_path.resolve()}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
