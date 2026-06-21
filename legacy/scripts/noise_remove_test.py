# Requer: pip install noisereduce  (ou pip install -r requirements.txt)
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from inference_noise_reduce import (
    DEFAULT_NR_NOISE_CLIP_SECONDS,
    DEFAULT_NR_PROP_DECREASE,
    DEFAULT_NR_PEAK_MATCH,
    noise_reduce_waveform,
)

# Opcional: sobrescrever aqui; por omissão = mesmos defaults que treino / inferência (inference_noise_reduce).
NOISE_CLIP_SEC = DEFAULT_NR_NOISE_CLIP_SECONDS
PROP_DECREASE = DEFAULT_NR_PROP_DECREASE
PEAK_MATCH = DEFAULT_NR_PEAK_MATCH

WAV_IN = (
    Path("datasets")
    / "cache_processado"
    / "speecht5_lapsbm_speecht5_16000hz_nr_gate_st4_hf_fe_v1_spkpool2_concat_g0.25_sdNone_mixappend_r50_log10"
    / "nr_preview"
    / "nr_preview_01_F004_idx0.wav"
)

if not WAV_IN.is_file():
    raise SystemExit(f"Ficheiro não encontrado: {WAV_IN.resolve()}")
rate, raw = wavfile.read(str(WAV_IN))

out = noise_reduce_waveform(
    raw,
    int(rate),
    noise_clip_seconds=NOISE_CLIP_SEC,
    stationary=True,
    prop_decrease=PROP_DECREASE,
    peak_match=PEAK_MATCH,
)

out_wav = Path("audio_limpo.wav")
dest = out_wav.resolve()
wavfile.write(str(dest), int(rate), np.asarray(out, dtype=np.float32))
size = dest.stat().st_size if dest.is_file() else 0

y_in = raw.astype(np.float64) if not np.issubdtype(raw.dtype, np.integer) else raw.astype(np.float64) / float(
    np.iinfo(raw.dtype).max
)
if y_in.ndim > 1:
    y_in = y_in.mean(axis=1)
print(
    f"Gravado: {dest} ({size:,} bytes) | "
    f"pico in={float(np.max(np.abs(y_in))):.4f} pico out={float(np.max(np.abs(out))):.4f} | "
    f"PROP_DECREASE={PROP_DECREASE} PEAK_MATCH={PEAK_MATCH}"
)
