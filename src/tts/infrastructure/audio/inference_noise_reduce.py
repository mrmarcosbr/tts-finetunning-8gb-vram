"""Redução de ruído (noisereduce) — mesma lógica que ``noise_remove_test.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np


# --- Predefinições partilhadas (treino, inferência, métricas, noise_remove_test.py) ---
DEFAULT_NR_NOISE_CLIP_SECONDS = 1.0
DEFAULT_NR_PROP_DECREASE = 0.93
DEFAULT_NR_PEAK_MATCH = True


def _nr_module() -> Any:
    try:
        import noisereduce as nr  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Instale noisereduce: python -m pip install noisereduce"
        ) from e
    return nr


def waveform_to_float_mono_like_noise_remove_test(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """
    Converte o sinal para mono float32 em [-1, 1], como
    ``noise_remove_test.py`` (passos após ``wavfile.read``).
    """
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros(0, dtype=np.float32), int(sr)
    if y.ndim > 1:
        y = y.mean(axis=1)
    raw = y
    y_work = raw.astype(np.float64).reshape(-1)
    if np.issubdtype(raw.dtype, np.integer):
        y_work = y_work / float(np.iinfo(np.dtype(raw.dtype)).max)
    y_f32 = np.clip(y_work.astype(np.float32), -1.0, 1.0)
    return y_f32, int(sr)


def auto_pick_stationary_for_clip(
    y: np.ndarray,
    sr: int,
    noise_clip_seconds: float,
    *,
    head_vs_rest_ratio_max: float = 0.72,
) -> bool:
    """
    Escolhe se ``noisereduce`` deve usar ``stationary=True``.

    No modo estacionário, os primeiros ``noise_clip_seconds`` s são ``y_noise``. Se esse
    trecho contiver **fala** (energia semelhante ao resto do clip), o perfil deixa de ser
    “só ruído” e o NR costuma **pouco efectivo** — o fundo continua audível.

    Heurística: compara a energia média (valor absoluto) do trecho inicial com a do *resto* do
    sinal. Se ``mean(|head|) / mean(|rest|) >= head_vs_rest_ratio_max``, assume-se fala no
    início e devolve-se ``False`` (gating não estacionário). Caso contrário ``True``.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 2048:
        return True
    n_clip = float(max(1e-6, noise_clip_seconds)) * float(sr)
    n = int(max(1, min(y.size - 1, n_clip)))
    head = y[:n]
    rest = y[n:]
    if rest.size < 512:
        return True
    m_head = float(np.mean(np.abs(head)))
    m_rest = float(np.mean(np.abs(rest)))
    if m_rest < 1e-12:
        return True
    ratio = m_head / (m_rest + 1e-12)
    return bool(ratio < float(head_vs_rest_ratio_max))


def peak_match_wavforms(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Escala ``output`` para igualar o pico absoluto de ``reference`` (mantém sensação de volume).
    Útil após noisereduce, que costuma baixar o nível global.
    """
    out = np.asarray(output, dtype=np.float32).reshape(-1)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    if out.size == 0 or ref.size == 0:
        return out
    p_ref = float(np.max(np.abs(ref)))
    p_out = float(np.max(np.abs(out)))
    if p_out < 1e-12 or p_ref < 1e-12:
        return np.clip(out, -1.0, 1.0)
    scaled = out * (p_ref / p_out)
    return np.clip(scaled.astype(np.float32), -1.0, 1.0)


def noise_reduce_waveform(
    y: np.ndarray,
    sr: int,
    *,
    noise_clip_seconds: float = DEFAULT_NR_NOISE_CLIP_SECONDS,
    stationary: bool = True,
    prop_decrease: float = DEFAULT_NR_PROP_DECREASE,
    peak_match: bool = DEFAULT_NR_PEAK_MATCH,
) -> np.ndarray:
    """
    Redução espectral com ``noisereduce``.

    Com ``stationary=True`` (recomendado quando o início do clip tem sobretudo ruído
    ou silêncio de sala): ``y_noise`` = primeiros ``noise_clip_seconds`` s — estes
    amostras são usados pelo algoritmo **estacionário** (perfil de ruído explícito).

    Com ``stationary=False``: gating **não estacionário** sobre ``y`` inteiro; o parâmetro
    ``noise_clip_seconds`` **não** altera o algoritmo (o ``y_noise`` é ignorado pela
    biblioteca — comportamento de ``noisereduce`` 3.x).

    ``prop_decrease`` (0–1): fração de atenuação do ruído; valores mais baixos = NR mais suave
    (menos “sucção” de energia na fala).

    ``peak_match``: se True, após o NR o sinal é escalado para o mesmo pico absoluto que a entrada,
    atenuando o efeito de “áudio mais baixo”.
    """
    nr = _nr_module()
    y_f32, sr_i = waveform_to_float_mono_like_noise_remove_test(y, sr)
    if y_f32.size == 0:
        return y_f32

    n_noise = int(max(1, min(y_f32.size, float(noise_clip_seconds) * sr_i)))
    ruido_fundo = y_f32[:n_noise].copy()

    pd = float(np.clip(prop_decrease, 0.0, 1.0))

    if stationary:
        reduced_noise = nr.reduce_noise(
            y=y_f32,
            sr=sr_i,
            stationary=True,
            y_noise=ruido_fundo,
            prop_decrease=pd,
        )
    else:
        reduced_noise = nr.reduce_noise(
            y=y_f32,
            sr=sr_i,
            stationary=False,
            prop_decrease=pd,
        )
    out = np.asarray(reduced_noise, dtype=np.float32).reshape(-1)

    if peak_match:
        out = peak_match_wavforms(out, y_f32)
    else:
        amax = float(np.max(np.abs(out))) if out.size else 0.0
        if amax > 1.0:
            out = (out / amax).astype(np.float32)
        out = np.clip(out, -1.0, 1.0)
    return out


def apply_noise_reduce_to_wav_file(
    path_in: Union[str, Path],
    path_out: Union[str, Path],
    *,
    noise_clip_seconds: float = DEFAULT_NR_NOISE_CLIP_SECONDS,
    stationary: bool = True,
    prop_decrease: float = DEFAULT_NR_PROP_DECREASE,
    peak_match: bool = DEFAULT_NR_PEAK_MATCH,
) -> Tuple[Optional[str], float, float]:
    """
    Mesmo pipeline que ``noise_remove_test.py``: lê o WAV em disco com
    ``scipy.io.wavfile.read``, aplica ``noise_reduce_waveform``, grava com ``wavfile.write``.

    Isto evita diferenças de escala entre o array em memória (pré-quantização soundfile) e o ficheiro
    PCM — que era o que o teste manual usava e a inferência não.

    Retorna ``(aviso_ou_none, delta_máx_abs_vs_entrada_float, escala_pico_entrada)``.
    """
    from scipy.io import wavfile

    path_in = Path(path_in)
    path_out = Path(path_out)
    rate, raw = wavfile.read(str(path_in))
    sr_i = int(rate)
    y_in, _ = waveform_to_float_mono_like_noise_remove_test(raw, sr_i)
    yscale = float(max(1e-12, np.max(np.abs(y_in)))) if y_in.size else 1.0
    try:
        y_nr = noise_reduce_waveform(
            raw,
            sr_i,
            noise_clip_seconds=noise_clip_seconds,
            stationary=stationary,
            prop_decrease=prop_decrease,
            peak_match=peak_match,
        )
    except Exception as exc:
        return str(exc), 0.0, yscale
    if y_in.size != y_nr.size or y_in.size == 0:
        return "tamanho_sinal_inválido_apos_nr", 0.0, yscale
    dmax = float(np.max(np.abs(y_nr.astype(np.float64) - y_in.astype(np.float64))))
    out = np.asarray(y_nr, dtype=np.float32).reshape(-1)
    wavfile.write(str(path_out), sr_i, out)
    return None, dmax, yscale


def try_noise_reduce_waveform(
    y: np.ndarray,
    sr: int,
    *,
    noise_clip_seconds: float = DEFAULT_NR_NOISE_CLIP_SECONDS,
    stationary: bool = True,
    prop_decrease: float = DEFAULT_NR_PROP_DECREASE,
    peak_match: bool = DEFAULT_NR_PEAK_MATCH,
) -> Tuple[np.ndarray, Optional[str]]:
    """Como ``noise_reduce_waveform``; em erro devolve cópia do original (float32) + razão."""
    try:
        return (
            noise_reduce_waveform(
                y,
                sr,
                noise_clip_seconds=noise_clip_seconds,
                stationary=stationary,
                prop_decrease=prop_decrease,
                peak_match=peak_match,
            ),
            None,
        )
    except Exception as exc:
        return np.asarray(y, dtype=np.float32).reshape(-1), str(exc)
