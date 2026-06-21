import torch
import torchaudio

# --- Monkey Patch for Torchaudio ---
# Must be applied BEFORE importing speechbrain
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
import os
import numpy as np

def main():
    print("🚀 Iniciando Teste do Modelo Fine-tuned...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Paths
    base_model_id = "microsoft/speecht5_tts"
    adapter_path = "./speecht5_laps_lora_v7"
    
    # 1. Load Models
    print("📦 Carregando modelos base...")
    processor = SpeechT5Processor.from_pretrained(adapter_path) # Load processor from saved dir to ensure config match
    model = SpeechT5ForTextToSpeech.from_pretrained(base_model_id)
    
    # Load Vocoder
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # 2. Load LoRA Adapter
    print(f"🔗 Carregando Adapter LoRA de: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    
    # 3. Speaker Embeddings
    print("🗣️  Carregando encoder de speaker...")
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name, 
        run_opts={"device": device}, 
        savedir=os.path.join("/tmp", spk_model_name)
    )
    
    # Load a real sample from the dataset to get a valid speaker embedding
    from datasets import load_dataset, Audio
    print("📊 Carregando um exemplo do dataset para extrair embedding real...")
    # Using the same dataset ID as training
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    print(f"   Colunas encontradas: {dataset.column_names}")
    
    if "wav" in dataset.column_names and "audio" not in dataset.column_names:
        print("   Renomeando coluna 'wav' para 'audio'...")
        dataset = dataset.rename_column("wav", "audio")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset[0] # Use standard indexing for non-streaming
    
    print(f"   Sample keys: {sample.keys()}")
    print(f"   Speaker ID do exemplo: {sample.get('speaker_id', 'unknown')}")
    
    audio_data = sample["audio"]["array"]
    waveform = torch.tensor(audio_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(waveform)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze(1).cpu()
    
    speaker_embeddings = speaker_embeddings.to(device)
    print(f"   Embedding shape: {speaker_embeddings.shape}")
    

    # Load Stats for Denormalization
    stats_path = adapter_path
    mean_path = os.path.join(stats_path, "mean.npy")
    std_path = os.path.join(stats_path, "std.npy")
    
    use_denorm = False
    if os.path.exists(mean_path) and os.path.exists(std_path):
        print(f"📉 Carregando estatísticas para denormalização de: {stats_path}")
        stats_mean = np.load(mean_path)
        stats_std = np.load(std_path)
        # Convert to tensor
        stats_mean = torch.tensor(stats_mean).to(device)
        stats_std = torch.tensor(stats_std).to(device)
        use_denorm = True
    else:
        print("⚠️  Estatísticas não encontradas. Gerando sem denormalização.")

    # 4. Inference
    text = "O treinamento foi finalizado e agora estamos testando a voz em português."
    print(f"📝 Gerando áudio para: '{text}'")
    
    inputs = processor(text=text, return_tensors="pt")
    
    # Generate Spectrogram
    print("🔮 Gerando espectrograma...")
    with torch.no_grad():
        # model.generate returns the spectrogram
        spectrogram = model.generate(
            input_ids=inputs["input_ids"].to(device), 
            speaker_embeddings=speaker_embeddings
        )
    
    print(f"DEBUG Spectrogram Shape (Predicted): {spectrogram.shape}")
    
    
    # Denormalize
    # V7 uses Normalized Log10. We must DE-NORMALIZE to get Log10 for Vocoder.
    if use_denorm:
       print("🔄 Denormalizando espectrograma...")
       spectrogram = (spectrogram * stats_std) + stats_mean
        
    # Vocoder
    print("🔊 Convertendo para áudio (Vocoder)...")
    with torch.no_grad():
        speech = vocoder(spectrogram)
    
    # --- Debug Stats ---
    print(f"DEBUG Stats:")
    print(f"  Shape: {speech.shape}")
    print(f"  Min: {speech.min().item()}")
    print(f"  Max: {speech.max().item()}")
    print(f"  Mean: {speech.mean().item()}")
    print(f"  NaNs: {torch.isnan(speech).sum().item()}")
    if torch.allclose(speech, torch.zeros_like(speech)):
        print("  ⚠️ WARNING: Output is all zeros (Silence)")
    
    output_filename = "teste_finetuned.wav"
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    print(f"✅ Áudio salvo em: {output_filename}")

    # 4b. Test with Random Speaker Embedding (To rule out noisy embedding issues)
    print("🧪 Testando com Embedding Aleatório (Random Speaker)...")
    # Random tensor (1, 512) normalized
    random_embedding = torch.randn(1, 512).to(device)
    random_embedding = torch.nn.functional.normalize(random_embedding, dim=1) 
    
    print("📝 Gerando áudio normalizado para Random Speaker...")
    with torch.no_grad():
        spec_random = model.generate(
            input_ids=inputs["input_ids"].to(device), 
            speaker_embeddings=random_embedding
        )
        
    if use_denorm:
         spec_random = (spec_random * stats_std) + stats_mean
         
    with torch.no_grad():
        speech_random = vocoder(spec_random)
        
    sf.write("teste_random_speaker.wav", speech_random.cpu().numpy(), samplerate=16000)
    print("✅ Áudio Random Speaker salvo em: teste_random_speaker.wav")
    
    # Teste comparativo (Sem LoRA - Zero Shot)
    print("Comparando com Zero-Shot (Base Model)...")
    # Disable adapters to use the base model
    with model.disable_adapter():
        with torch.no_grad():
            speech_base = model.generate_speech(
                inputs["input_ids"].to(device), 
                speaker_embeddings, 
                vocoder=vocoder
            )
    
    sf.write("teste_base_zeroshot.wav", speech_base.cpu().numpy(), samplerate=16000)
    print("✅ Áudio base (Zero-Shot) salvo em: teste_base_zeroshot.wav")
    
    # --- Debug Stats for Zero-Shot ---
    print(f"DEBUG Stats (Zero-Shot):")
    print(f"  Shape: {speech_base.shape}")
    print(f"  Min: {speech_base.min().item()}")
    print(f"  Max: {speech_base.max().item()}")
    print(f"  Mean: {speech_base.mean().item()}")

if __name__ == "__main__":
    main()
