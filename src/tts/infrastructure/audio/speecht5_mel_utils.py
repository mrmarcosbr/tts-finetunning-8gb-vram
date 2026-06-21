"""Alvo log-mel para treino SpeechT5 TTS, alinhado ao `SpeechT5FeatureExtractor` (HF)."""

from __future__ import annotations

import numpy as np
import torch


def waveform_to_speecht5_label_tensor(
    y: np.ndarray | list[float],
    feature_extractor,
    sampling_rate: int,
) -> torch.Tensor:
    """
    Mesma representação que o pré-treino SpeechT5: STFT amplitude → mel Slaney → log10
    (`transformers` `spectrogram`, não librosa potência+log10).

    `sampling_rate` deve coincidir com `feature_extractor.sampling_rate` (ex.: 16000).
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    out = feature_extractor(
        audio_target=y,
        sampling_rate=int(sampling_rate),
        return_tensors="pt",
    )
    lab = out["input_values"]
    if lab.dim() == 3:
        lab = lab.squeeze(0)
    return lab.float()
