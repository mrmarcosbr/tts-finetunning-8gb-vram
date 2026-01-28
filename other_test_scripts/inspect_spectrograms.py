import torch
from transformers import SpeechT5Processor
from datasets import load_dataset, Audio
import librosa
import numpy as np

def main():
    print("🔍 Inspecting Spectrograms...")
    
    # Load Processor
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    # Load one sample
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    if "wav" in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset[0]
    audio_array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    
    print(f"Audio Shape: {audio_array.shape}, SR: {sr}")
    
    # 1. Manual Extraction (Current Implementation)
    mel = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600)
    log_mel = librosa.power_to_db(mel, ref=np.max).T
    
    print("\n--- Manual Librosa (Current) ---")
    print(f"Shape: {log_mel.shape}")
    print(f"Min: {log_mel.min():.4f}")
    print(f"Max: {log_mel.max():.4f}")
    print(f"Mean: {log_mel.mean():.4f}")
    
    # 2. Processor Inspection
    print("\n--- SpeechT5FeatureExtractor Config ---")
    fe = processor.feature_extractor
    print(f"Feature Extractor Class: {type(fe)}")
    print(f"Sampling Rate: {fe.sampling_rate}")
    print(f"Do Normalize: {getattr(fe, 'do_normalize', 'Unknown')}")
    print(f"Return Attention Mask: {getattr(fe, 'return_attention_mask', 'Unknown')}")
    print(f"Keys in __dict__: {fe.__dict__.keys()}")
    
    if hasattr(fe, "mean") and fe.mean is not None:
         print(f"Mean (first 10): {fe.mean[:10]}")
    else:
         print("No 'mean' attribute found.")
         
    if hasattr(fe, "var") and fe.var is not None:
         print(f"Var (first 10): {fe.var[:10]}")
    else:
         print("No 'var' attribute found.")

if __name__ == "__main__":
    main()
