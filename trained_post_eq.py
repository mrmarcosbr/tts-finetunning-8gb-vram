"""
Pós-EQ aplicado só ao áudio «treinado» (LoRA) na inferência e, opcionalmente, na avaliação F0/WER.

Dependências: numpy, scipy (butter/sosfiltfilt para high-pass).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def inference_dc_and_highpass(audio: np.ndarray, sr: int, highpass_hz: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size < 32 or sr <= 0 or highpass_hz <= 0:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    x = x - np.mean(x)
    hp = float(np.clip(highpass_hz, 1.0, 0.49 * float(sr)))
    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = 0.5 * float(sr)
        w = hp / nyq
        w = float(np.clip(w, 0.002, 0.99))
        sos = butter(4, w, btype="highpass", output="sos")
        x = sosfiltfilt(sos, x)
    except Exception:
        pass
    return x.astype(np.float32)


def attenuate_low_frequencies(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 120.0,
    attenuation_db: float = 1.2,
    transition_hz: float = 120.0,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or sr <= 0:
        return x
    x = x.astype(np.float64) - float(np.mean(x))

    min_cutoff = 40.0
    max_cutoff = max(min_cutoff, (sr * 0.5) - 200.0)
    cutoff = float(np.clip(cutoff_hz, min_cutoff, max_cutoff))
    transition = max(20.0, float(transition_hz))
    low_gain = 10.0 ** (-max(0.0, float(attenuation_db)) / 20.0)

    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    gains = np.ones_like(freqs, dtype=np.float32)

    low_mask = freqs <= cutoff
    trans_mask = (freqs > cutoff) & (freqs < cutoff + transition)

    gains[low_mask] = low_gain
    if np.any(trans_mask):
        t = (freqs[trans_mask] - cutoff) / transition
        gains[trans_mask] = low_gain + (1.0 - low_gain) * t

    filtered = np.fft.irfft(spec * gains, n=x.size)
    return np.asarray(filtered, dtype=np.float32)


def boost_high_frequencies(
    audio: np.ndarray,
    sr: int,
    shelf_start_hz: float = 3500.0,
    boost_db: float = 1.5,
    transition_hz: float = 1000.0,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or sr <= 0 or float(boost_db) <= 1e-6:
        return x
    x = x.astype(np.float64) - float(np.mean(x))

    nyq = 0.5 * float(sr)
    f0 = float(np.clip(float(shelf_start_hz), 400.0, max(400.0, nyq - 500.0)))
    trans = max(100.0, float(transition_hz))
    high_gain = 10.0 ** (max(0.0, float(boost_db)) / 20.0)

    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    gains = np.ones_like(freqs, dtype=np.float32)

    below = freqs < f0
    ramp = (freqs >= f0) & (freqs <= f0 + trans)
    above = freqs > f0 + trans

    gains[below] = 1.0
    gains[above] = high_gain
    if np.any(ramp):
        tr = (freqs[ramp] - f0) / trans
        gains[ramp] = 1.0 + (high_gain - 1.0) * tr

    filtered = np.fft.irfft(spec * gains, n=x.size)
    return np.asarray(filtered, dtype=np.float32)


def apply_trained_output_waveform_eq(
    y: np.ndarray,
    sr: int,
    args: Any,
    *,
    speecht5_highpass_hz: Optional[float] = None,
) -> np.ndarray:
    """
    Mesma cadeia que test_inference_exhaustive._finalize_audio para is_trained=True
    (sem cleanup de silêncio nem peak norm).

    speecht5_highpass_hz: se não None, aplica antes da cadeia treinada (útil se o WAV ainda não passou pelo HP do vocoder).
    Normalmente na avaliação deixar None — os ficheiros já refletem a inferência.
    """
    out = np.asarray(y, dtype=np.float32).reshape(-1)
    hpf_g = float(speecht5_highpass_hz if speecht5_highpass_hz is not None else 0.0)
    if hpf_g > 0:
        out = inference_dc_and_highpass(out, sr, hpf_g)
    xhp = float(getattr(args, "trained_grave_highpass_hz", 0) or 0)
    if xhp > 0:
        out = inference_dc_and_highpass(out, sr, xhp)
    if bool(getattr(args, "apply_trained_bass_treatment", False)):
        out = attenuate_low_frequencies(
            out,
            sr,
            cutoff_hz=float(getattr(args, "trained_bass_cut_hz", 120.0)),
            attenuation_db=float(getattr(args, "trained_bass_atten_db", 1.2)),
            transition_hz=float(getattr(args, "trained_bass_transition_hz", 120.0)),
        )
    tboost = float(getattr(args, "trained_treble_boost_db", 0) or 0)
    if tboost > 1e-6:
        out = boost_high_frequencies(
            out,
            sr,
            shelf_start_hz=float(getattr(args, "trained_treble_shelf_hz", 3500.0)),
            boost_db=tboost,
            transition_hz=float(getattr(args, "trained_treble_transition_hz", 1000.0)),
        )
    return out


def trained_eq_affects_audio(args: Any) -> bool:
    if bool(getattr(args, "apply_trained_bass_treatment", False)):
        return True
    if float(getattr(args, "trained_grave_highpass_hz", 0) or 0) > 0:
        return True
    if float(getattr(args, "trained_treble_boost_db", 0) or 0) > 1e-6:
        return True
    return False
