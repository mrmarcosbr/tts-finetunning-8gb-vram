import torch
from transformers import SpeechT5HifiGan
from datasets import load_dataset, Audio
import librosa
import numpy as np
import soundfile as sf

def main():
    print("🔊 Testing Vocoder Compatibility...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    vocoder.eval()

    # Load Sample
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    if "wav" in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset[0]["audio"]["array"]
    
    print(f"Original Audio Shape: {sample.shape}")
    sf.write("vocoder_input_original.wav", sample, 16000)
    
    # --- 1. Current Method (Librosa + PowerToDB) ---
    print("\n--- Method 1: Librosa + PowerToDB (ref=max) ---")
    mel = librosa.feature.melspectrogram(y=sample, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600)
    log_mel = librosa.power_to_db(mel, ref=np.max).T
    
    tensor_mel = torch.tensor(log_mel).unsqueeze(0).to(device) # (1, Time, 80)
    print(f"Spectrogram Shape: {tensor_mel.shape}")
    
    with torch.no_grad():
        out_audio = vocoder(tensor_mel)
        if out_audio.dim() > 1:
            out_audio = out_audio.squeeze()
            
    print(f"Output Audio Shape: {out_audio.shape}")
    sf.write("vocoder_output_method1.wav", out_audio.cpu().numpy(), 16000)
    print("Saved 'vocoder_output_method1.wav'")

    # --- 2. Normalized Method (Simulating V5) ---
    print("\n--- Method 2: Normalized (Method 1 - Mean / Std) ---")
    # We don't have global mean here, but let's just center THIS sample to test relative shape
    mean = tensor_mel.mean()
    std = tensor_mel.std()
    norm_mel = (tensor_mel - mean) / std
    
    # NOTE: The vocoder expects the absolute scale it was trained on. 
    # If it was trained on unnormalized log-mel, feeding it normalized might reduce gain but should sound okayish?
    # Or does the vocoder expect normalized input? Usually vocoders are trained on the TARGET domain.
    # If SpeechT5 predicts spectrograms, does it predict them in range [-4, 4] (normalized)? 
    # If so, does the VOCODER expect [-4, 4]? 
    # Let's try feeding the normalized one to the vocoder.
    
    with torch.no_grad():
        out_audio_norm = vocoder(norm_mel)
        if out_audio_norm.dim() > 1:
            out_audio_norm = out_audio_norm.squeeze()
        
    sf.write("vocoder_output_method2.wav", out_audio_norm.cpu().numpy(), 16000)
    print("Saved 'vocoder_output_method2.wav'")
    
    # --- 3. Pure Log (No dB scaling) ---
    print("\n--- Method 3: Pure Log (np.log10) ---")
    mel3 = librosa.feature.melspectrogram(y=sample, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600)
    # Clamp to avoid log(0)
    mel3 = np.clip(mel3, a_min=1e-5, a_max=None)
    log_mel3 = np.log10(mel3).T
    
    tensor_mel3 = torch.tensor(log_mel3).unsqueeze(0).to(device)
    with torch.no_grad():
        out_audio3 = vocoder(tensor_mel3)
        if out_audio3.dim() > 1:
            out_audio3 = out_audio3.squeeze()
        
    sf.write("vocoder_output_method3.wav", out_audio3.cpu().numpy(), 16000)
    print("Saved 'vocoder_output_method3.wav'")

if __name__ == "__main__":
    main()
