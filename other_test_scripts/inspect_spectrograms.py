import sys
from pathlib import Path

import librosa
import numpy as np
from datasets import Audio, load_dataset
from transformers import SpeechT5Processor

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from speecht5_mel_utils import waveform_to_speecht5_label_tensor


def main():
    print("🔍 Inspecting Spectrograms...")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    fe = processor.feature_extractor

    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    if "wav" in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset[0]
    audio_array = np.asarray(sample["audio"]["array"], dtype=np.float32).reshape(-1)
    sr = sample["audio"]["sampling_rate"]

    print(f"Audio Shape: {audio_array.shape}, SR: {sr}")

    lab = waveform_to_speecht5_label_tensor(audio_array, fe, int(sr))
    lm = lab.numpy()
    print("\n--- SpeechT5 training labels (SpeechT5FeatureExtractor / train_exhaustive) ---")
    print(f"Shape: {lm.shape}")
    print(f"Min: {lm.min():.4f}")
    print(f"Max: {lm.max():.4f}")
    print(f"Mean: {lm.mean():.4f}")

    mel_legacy = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=80,
        fmax=7600,
    )
    log_mel_legacy = np.log10(np.clip(mel_legacy, 1e-5, None)).T
    print("\n--- Legacy: librosa power spectrogram + log10 (rótulos antigos, desalinhados) ---")
    print(f"Shape: {log_mel_legacy.shape}")
    print(f"Min: {log_mel_legacy.min():.4f}")
    print(f"Max: {log_mel_legacy.max():.4f}")
    print(f"Mean: {log_mel_legacy.mean():.4f}")

    log_mel_db = librosa.power_to_db(mel_legacy, ref=np.max).T
    print("\n--- Librosa power_to_db(ref=max) ---")
    print(f"Shape: {log_mel_db.shape}")
    print(f"Min: {log_mel_db.min():.4f}")
    print(f"Max: {log_mel_db.max():.4f}")
    print(f"Mean: {log_mel_db.mean():.4f}")

    print("\n--- SpeechT5FeatureExtractor config ---")
    print(f"Feature Extractor Class: {type(fe)}")
    print(f"Sampling Rate: {fe.sampling_rate}")
    print(f"Do Normalize: {getattr(fe, 'do_normalize', 'Unknown')}")
    print(f"Return Attention Mask: {getattr(fe, 'return_attention_mask', 'Unknown')}")


if __name__ == "__main__":
    main()
