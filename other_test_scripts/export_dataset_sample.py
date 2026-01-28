from datasets import load_dataset, Audio
import soundfile as sf
import os
import torch

def main():
    print("📦 Carregando dataset para verificação...")
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    
    # Rename column if needed
    if "wav" in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")
        
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Pick a sample
    idx = 0
    sample = dataset[idx]
    
    audio_data = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    text = sample.get("text", sample.get("txt", "Unknown"))
    
    print(f"Sample {idx}:")
    print(f"Text: {text}")
    print(f"Audio Shape: {audio_data.shape}")
    print(f"SR: {sr}")
    
    filename = "dataset_sample_raw.wav"
    sf.write(filename, audio_data, sr)
    print(f"✅ Áudio original salvo em: {filename}")

if __name__ == "__main__":
    main()
