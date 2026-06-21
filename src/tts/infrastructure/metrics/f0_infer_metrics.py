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
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from tts.infrastructure.audio.inference_noise_reduce import (
    DEFAULT_NR_NOISE_CLIP_SECONDS,
    DEFAULT_NR_PROP_DECREASE,
    DEFAULT_NR_PEAK_MATCH,
)

DEFAULT_SR = 16000
DEFAULT_HOP_LENGTH = 256
DEFAULT_FMIN = 50.0
DEFAULT_FMAX = 500.0

_INFERENCE_PAIR_EXTS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg")


def inference_pair_prefix_sort_key(prefix: str) -> Tuple[int, int, str]:
    """
    Ordem de leitura/CSV alinhada a ``sentenca_1``, ``sentenca_2``, …, ``sentenca_10`` (não lexicográfica).

    ``custom`` (inferência com um único texto) ordena antes das sentenças numeradas.
    Outros prefixos vão depois, por nome em minúsculas.
    """
    p = (prefix or "").strip()
    if not p:
        return (99, 0, "")
    if p.lower() == "custom":
        return (-1, 0, p)
    if p.upper() == "MEDIA":
        return (100, 0, p)
    m = re.match(r"(?i)^sentenca_(\d+)$", p)
    if m:
        return (0, int(m.group(1)), p)
    return (1, 0, p.lower())


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


def _load_mono_resampled(path: Path | str, sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def f0_rmse_hz_between_float_audio(
    y_gen: np.ndarray,
    y_ref: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    voiced_min_hz: float = DEFAULT_FMIN,
) -> float:
    y_gen = np.asarray(y_gen, dtype=np.float32).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=np.float32).reshape(-1)
    if y_gen.size == 0 or y_ref.size == 0:
        return float("nan")
    gf = compute_yin_f0(y_gen, sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
    rf = compute_yin_f0(y_ref, sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
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


_DEFAULT_TEXTS_TRIPLE_WAV_PREFIX_TAGS: Tuple[str, ...] = ("_sem_embedding", "_base_treino", "_base_teste")


def wav_pair_prefix_without_triple_emb_tag(pair_prefix: str) -> Optional[str]:
    """
    Inferência ``default_texts`` tripla: prefixo do WAV = ``sentenca_N_<rótulo>``.
    ``sentenca_4_base_treino`` → ``sentenca_4`` (onde costuma estar o .txt da frase sintetizada).
    """
    for suf in _DEFAULT_TEXTS_TRIPLE_WAV_PREFIX_TAGS:
        if pair_prefix.endswith(suf):
            return pair_prefix[: -len(suf)]
    return None


def treinado_nr_wav_path(out_dir: Path, prefix: str) -> Path:
    """Ficheiro opcional gravado na inferência com ``--noise_reduce``: ``{prefix}_treinado_nr.wav``."""
    return out_dir / f"{prefix}_treinado_nr.wav"


def base_nr_wav_path(out_dir: Path, prefix: str) -> Path:
    """Inferência + ``--noise_reduce``: ``{prefix}_base_nr.wav``."""
    return out_dir / f"{prefix}_base_nr.wav"


def referencia_nr_wav_path(out_dir: Path, prefix: str) -> Path:
    """Inferência + ``--noise_reduce``: ``{prefix}_referencia_nr.wav``."""
    return out_dir / f"{prefix}_referencia_nr.wav"


def _pick_treinado_for_base(out_dir: Path, prefix: str, base_suffix: str) -> Optional[Path]:
    """Procura `{prefix}_treinado.*`: preferindo a mesma extensão que o ficheiro base, depois as outras."""
    suf = base_suffix.lower() if base_suffix.startswith(".") else f".{base_suffix.lower()}"
    order = [suf] + [e for e in _INFERENCE_PAIR_EXTS if e.lower() != suf]
    for e in order:
        p = out_dir / f"{prefix}_treinado{e}"
        if p.is_file():
            return p
    return None


def iter_speecht5_base_treinado_pairs(out_dir: Path | str) -> Iterable[Tuple[str, Path, Path]]:
    """
    Emparelha ``{prefix}_base.*`` (ou legado ``_original.*``) com ``{prefix}_treinado.*``.
    Preferência: ``_base`` antes de ``_original``; extensões em ``_INFERENCE_PAIR_EXTS``.
    Por prefixo conta-se um só par (prioridade do WAV base: wav, mp3, … dentro de cada grupo).

    A sequência final ordena prefixos com ``inference_pair_prefix_sort_key`` (ex.: sentenca_2 antes de sentenca_10).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    seen: set[str] = set()
    pairs: List[Tuple[str, Path, Path]] = []

    def collect(tail_lower: str) -> None:
        for ext in _INFERENCE_PAIR_EXTS:
            tail = f"{tail_lower}{ext}"
            for pb in out_dir.glob(f"*{tail}"):
                if not pb.is_file() or not pb.name.lower().endswith(tail.lower()):
                    continue
                prefix = pb.name[: -len(tail)]
                if prefix in seen:
                    continue
                treina = _pick_treinado_for_base(out_dir, prefix, pb.suffix)
                if treina is not None:
                    seen.add(prefix)
                    pairs.append((prefix, pb, treina))

    collect("_base")
    collect("_original")
    pairs.sort(key=lambda t: inference_pair_prefix_sort_key(t[0]))
    yield from pairs


def iter_speecht5_original_treinado_pairs(out_dir: Path | str) -> Iterable[Tuple[str, Path, Path]]:
    """Compatibilidade com código antigo: igual a ``iter_speecht5_base_treinado_pairs``."""
    yield from iter_speecht5_base_treinado_pairs(out_dir)


def iter_inference_audio_triplets(
    out_dir: Path | str,
) -> Iterable[Tuple[str, Path, Path, Optional[Path]]]:
    """prefix, base_wav (ou legado *_original.*), treinado, referencia (opcional)."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    for prefix, p_base, train in iter_speecht5_base_treinado_pairs(out_dir):
        ref = find_referencia_path(out_dir, prefix)
        yield prefix, p_base, train, ref


def compute_metrics_for_inference_dir(
    out_dir: Path | str,
    *,
    sample_rate: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    compute_f0: bool = True,
    noise_reduce: bool = False,
    noise_clip_seconds: float = DEFAULT_NR_NOISE_CLIP_SECONDS,
    nr_prop_decrease: float = DEFAULT_NR_PROP_DECREASE,
    nr_peak_match: bool = DEFAULT_NR_PEAK_MATCH,
    wer_cer: bool = False,
    whisper_model: str = "openai/whisper-large-v3",
    whisper_language: str = "portuguese",
    asr_device: str = "cuda",
    references_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Métricas só sobre áudio **original** (sem NR): WAV base sintético, treinado e referência do dataset.

    • **WER/CER**: texto de referência (input do TTS) vs transcrição Whisper do áudio sintético (base ou LoRA).
    • **F0 RMSE**: contour do sintético vs contour do WAV de referência (mesmo protocolo YIN/RMSE).

    Os parâmetros ``noise_reduce``, ``noise_clip_seconds``, ``nr_prop_decrease`` e ``nr_peak_match``
    mantêm a assinatura por compatibilidade com chamadores; **não entram no cálculo** das métricas
    (o NR na inferência só gera ficheiros ``*_nr.wav`` quando activo no script de inferência).

    Prefixo de ficheiros: ``*_base.*`` ou legado ``*_original.*`` ; ``*_treinado.*`` ; ``*_referencia.*``.
    """
    _ = (noise_reduce, noise_clip_seconds, nr_prop_decrease, nr_peak_match)
    rows: List[Dict[str, Any]] = []
    out_dir = Path(out_dir)

    def _fcell(x: float) -> str | float:
        return "" if not np.isfinite(x) else float(x)

    ref_map: Dict[str, str] = {}
    if wer_cer:
        from tts.infrastructure.metrics.eval_metrics_aval import load_reference_map, transcribe_wav, word_error_rate, char_error_rate

        ref_map = dict(references_map) if references_map else {}
        if not ref_map:
            ref_map = load_reference_map(out_dir, None)

    dev = str(asr_device).strip()
    if wer_cer and dev.lower() == "cuda":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"

    tmpdir: Optional[str] = None
    if wer_cer:
        tmpdir = tempfile.mkdtemp(prefix="infer_asr_")

    try:
        for prefix, p_base, p_train in iter_speecht5_base_treinado_pairs(out_dir):
            p_ref = find_referencia_path(out_dir, prefix)

            pair_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

            def _get_mono(path: Path) -> np.ndarray:
                k = ("mono", str(path.resolve()), int(sample_rate))
                if k not in pair_cache:
                    pair_cache[k] = _load_mono_resampled(path, sample_rate)
                return pair_cache[k]

            def base_raw() -> np.ndarray:
                k = ("braw", str(p_base.resolve()), int(sample_rate))
                if k not in pair_cache:
                    pair_cache[k] = _get_mono(p_base)
                return pair_cache[k]

            def treinado_raw() -> np.ndarray:
                k = ("traw", str(p_train.resolve()), int(sample_rate))
                if k not in pair_cache:
                    pair_cache[k] = _get_mono(p_train)
                return pair_cache[k]

            row: Dict[str, Any] = {
                "pair_prefix": prefix,
                "wer_base": "",
                "wer_treinado": "",
                "cer_base": "",
                "cer_treinado": "",
                "f0_rmse_base": "",
                "f0_rmse_treinado": "",
                "failure_reason": "",
                "wav_base": shorten_path_for_csv(p_base),
                "wav_treinado": shorten_path_for_csv(p_train),
                "wav_referencia": shorten_path_for_csv(p_ref) if p_ref is not None else "",
            }
            errs: List[str] = []

            def _metric_f0(gen: np.ndarray, ref: np.ndarray) -> float:
                return float(
                    f0_rmse_hz_between_float_audio(
                        gen,
                        ref,
                        sample_rate,
                        hop_length=hop_length,
                        fmin=fmin,
                        fmax=fmax,
                    )
                )

            if compute_f0:
                if p_ref is None:
                    errs.append("sem_wav_referencia_f0")
                else:
                    try:
                        y_b = base_raw()
                        y_t = treinado_raw()
                        y_r = _get_mono(p_ref)
                        row["f0_rmse_base"] = _fcell(_metric_f0(y_b, y_r))
                        row["f0_rmse_treinado"] = _fcell(_metric_f0(y_t, y_r))
                    except Exception as exc:
                        errs.append(f"f0-vs-ref:{exc}")

            if wer_cer and tmpdir is not None:
                safe = re.sub(r"[^\w\-]+", "_", prefix)[:80]
                ref_text_for_wer = (ref_map.get(prefix) or "").strip()
                if not ref_text_for_wer and p_ref is not None:
                    for sidecar in (
                        p_ref.with_suffix(".txt"),
                        out_dir / Path(p_ref.name).with_suffix(".txt"),
                    ):
                        if sidecar.is_file():
                            ref_text_for_wer = sidecar.read_text(encoding="utf-8").strip()
                            break
                if not ref_text_for_wer:
                    tp = out_dir / f"{prefix}.txt"
                    if tp.is_file():
                        ref_text_for_wer = tp.read_text(encoding="utf-8").strip()
                rtxt = out_dir / "referencia_sentencas" / f"{prefix}_referencia.txt"
                if not ref_text_for_wer and rtxt.is_file():
                    ref_text_for_wer = rtxt.read_text(encoding="utf-8").strip()

                if not ref_text_for_wer:
                    stripped_pf = wav_pair_prefix_without_triple_emb_tag(prefix)
                    if stripped_pf:
                        tp3 = out_dir / f"{stripped_pf}.txt"
                        if tp3.is_file():
                            ref_text_for_wer = tp3.read_text(encoding="utf-8").strip()

                if not ref_text_for_wer:
                    errs.append("sem_texto_referencia_wer_cer")
                else:
                    try:
                        yo = base_raw()
                        yt0 = treinado_raw()
                        pb0 = Path(tmpdir) / f"{safe}_asr_base.wav"
                        pt0 = Path(tmpdir) / f"{safe}_asr_treinado.wav"
                        sf.write(str(pb0), yo, sample_rate)
                        sf.write(str(pt0), yt0, sample_rate)
                        hyp_b = transcribe_wav(pb0, language=whisper_language, model_id=whisper_model, device=dev)
                        hyp_t0 = transcribe_wav(pt0, language=whisper_language, model_id=whisper_model, device=dev)
                        row["wer_base"] = word_error_rate(ref_text_for_wer, hyp_b)
                        row["wer_treinado"] = word_error_rate(ref_text_for_wer, hyp_t0)
                        row["cer_base"] = char_error_rate(ref_text_for_wer, hyp_b)
                        row["cer_treinado"] = char_error_rate(ref_text_for_wer, hyp_t0)
                    except Exception as exc:
                        errs.append(f"wer_cer:{exc}")

            if errs:
                row["failure_reason"] = "; ".join(errs)
            rows.append(row)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return rows


INFERENCE_METRICS_CSV_COLUMN_ORDER: List[str] = [
    "pair_prefix",
    "wer_base",
    "wer_treinado",
    "cer_base",
    "cer_treinado",
    "f0_rmse_base",
    "f0_rmse_treinado",
    "failure_reason",
    "wav_base",
    "wav_treinado",
    "wav_referencia",
]


def build_inference_metrics_mean_row(
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
    *,
    label: str = "MEDIA",
) -> Dict[str, Any]:
    """Linha de resumo: média por coluna numérica; demais campos vazios."""
    mean_row: Dict[str, Any] = {"pair_prefix": label}
    for key in fieldnames:
        if key == "pair_prefix":
            continue
        mean, n = _mean_column(rows, key)
        if n > 0 and np.isfinite(mean):
            mean_row[key] = mean
        else:
            mean_row[key] = ""
    return mean_row


def write_metrics_csv(
    rows: List[Dict[str, Any]],
    out_csv: Path | str,
    *,
    append_mean_row: bool = True,
) -> None:
    out_csv = Path(out_csv)
    data_rows = [
        r for r in rows if str(r.get("pair_prefix", "") or "").strip().upper() != "MEDIA"
    ]
    rows_sorted = sorted(
        data_rows,
        key=lambda r: inference_pair_prefix_sort_key(str(r.get("pair_prefix", "") or "")),
    )
    keys: set[str] = set()
    for r in rows_sorted:
        keys.update(r.keys())
    ordered: List[str] = []
    for k in INFERENCE_METRICS_CSV_COLUMN_ORDER:
        if k in keys:
            ordered.append(k)
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        for row in rows_sorted:
            w.writerow(row)
        if append_mean_row and rows_sorted:
            w.writerow(build_inference_metrics_mean_row(rows_sorted, ordered))


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


def summarize_f0_rmse_inference(rows: List[Dict[str, Any]]) -> Tuple[float, int, float, int]:
    """Médias ``f0_rmse_base`` e ``f0_rmse_treinado`` (sintético vs WAV de referência)."""
    mo, ko = _mean_column(rows, "f0_rmse_base")
    mt, kt = _mean_column(rows, "f0_rmse_treinado")
    return mo, ko, mt, kt


def summarize_wer_cer_inference(rows: List[Dict[str, Any]]) -> Tuple[float, int, float, int]:
    """Médias ``wer_base`` e ``wer_treinado``."""
    mb, nb = _mean_column(rows, "wer_base")
    mt, nt = _mean_column(rows, "wer_treinado")
    return mb, nb, mt, nt



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calcula métricas (F0 RMSE; opcional WER/CER) sobre pastas de inferência SpeechT5."
    )
    parser.add_argument("out_dir", type=str, help="Pasta com pares *_base (ou legado *_original) / *_treinado (.wav, .mp3, …)")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--hop_length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--fmin", type=float, default=DEFAULT_FMIN)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    parser.add_argument(
        "--noise_reduce",
        action="store_true",
        help=(
            "Compatibilidade de CLI: as métricas em CSV não usam NR; só o script de inferência "
            "grava *_nr.wav quando activo."
        ),
    )
    parser.add_argument(
        "--noise_clip_sec",
        type=float,
        default=DEFAULT_NR_NOISE_CLIP_SECONDS,
        help="(Ignorado aqui; só inferência usa NR.) Mantido por compatibilidade de CLI.",
    )
    parser.add_argument(
        "--noise_prop_decrease",
        type=float,
        default=DEFAULT_NR_PROP_DECREASE,
        metavar="P",
        help="(Ignorado aqui.) Mantido por compatibilidade de CLI.",
    )
    parser.add_argument(
        "--no_nr_peak_match",
        action="store_true",
        help="(Ignorado aqui.) Mantido por compatibilidade de CLI.",
    )
    parser.add_argument(
        "--no-f0",
        dest="compute_f0",
        action="store_false",
        help="Não calcular F0 RMSE (útil com --wer-cer para só ASR).",
    )
    parser.set_defaults(compute_f0=True)
    parser.add_argument("--wer-cer", dest="wer_cer", action="store_true", help="WER/CER com Whisper + texto de referência.")
    parser.add_argument("--wer_cer_whisper_model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--wer_cer_whisper_language", type=str, default="portuguese")
    parser.add_argument(
        "--wer_cer_device",
        type=str,
        default=None,
        help="cuda | cpu. Omisso: cuda se disponível, senão cpu.",
    )
    args = parser.parse_args(argv)

    if not args.compute_f0 and not args.wer_cer:
        print("[erro] Sem métricas: use F0 (por defeito), ou --wer-cer, ou ambos (--no-f0 --wer-cer para só ASR).", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"[erro] Pasta inexistente: {out_dir}", file=sys.stderr)
        return 1

    _dev = args.wer_cer_device or "cuda"
    rows = compute_metrics_for_inference_dir(
        out_dir,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        compute_f0=bool(args.compute_f0),
        noise_reduce=bool(args.noise_reduce),
        noise_clip_seconds=float(args.noise_clip_sec),
        nr_prop_decrease=float(args.noise_prop_decrease),
        nr_peak_match=not bool(args.no_nr_peak_match),
        wer_cer=bool(args.wer_cer),
        whisper_model=str(args.wer_cer_whisper_model),
        whisper_language=str(args.wer_cer_whisper_language),
        asr_device=str(_dev),
        references_map=None,
    )
    if not rows:
        print(f"[aviso] Nenhum par *_base / *_treinado (.wav, .mp3, …) nem legado *_original em {out_dir}")
        return 0

    csv_path = out_dir / "inference_metrics.csv"
    write_metrics_csv(rows, csv_path)
    print(f"[ok] {len(rows)} prefixos avaliados -> {csv_path.resolve()}")
    if args.compute_f0:
        mo, ko, mt, kt = summarize_f0_rmse_inference(rows)
        if ko > 0 or kt > 0:
            if np.isfinite(mo):
                print(f"   F0 RMSE médio base vs WAV ref. (n={ko}): {mo:.4f} Hz")
            if np.isfinite(mt):
                print(f"   F0 RMSE médio LoRA vs WAV ref. (n={kt}): {mt:.4f} Hz")
    if args.wer_cer:
        mb, nb, mt_w, nt_w = summarize_wer_cer_inference(rows)
        if nb > 0 and np.isfinite(mb):
            print(f"   WER médio base (n={nb}): {100.0 * mb:.2f}%")
        if nt_w > 0 and np.isfinite(mt_w):
            print(f"   WER médio LoRA (n={nt_w}): {100.0 * mt_w:.2f}%")
        print("   CER: colunas cer_base / cer_treinado no CSV.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
