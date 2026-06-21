from transformers import SpeechT5FeatureExtractor
from datasets import load_dataset, Audio
import torch
import numpy as np

def main():
    print("🔍 Testing Feature Extractor...")
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
    
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    if "wav" in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    sample = dataset[0]["audio"]["array"]
    print(f"Input Audio Shape: {sample.shape}")
    
    # Try standard call
    try:
        # Note: SpeechT5 feature extractor often returns raw audio for some tasks, 
        # but for TTS training we need mels.
        # Let's inspect the code or doc logic via trial.
        res = feature_extractor(audio=sample, sampling_rate=16000, return_tensors="pt")
        print(f"Result Keys: {res.keys()}")
        if "input_values" in res:
             print(f"Input Values Shape: {res['input_values'].shape}")
             # If shape is (1, N), it's raw audio. 
             # If shape is (1, T, 80), it's spectrogram.
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
