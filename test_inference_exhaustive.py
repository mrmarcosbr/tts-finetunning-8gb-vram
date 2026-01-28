import os
import torch
import torchaudio
import unicodedata
import soundfile as sf
import numpy as np

# --- 1. Monkey Patch & Setup (MUST BE BEFORE SPEECHBRAIN IMPORT) ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel
from speechbrain.inference.speaker import EncoderClassifier
from collections import Counter

# Remove acentos e caracteres especiais de uma string
def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def main():
    print("🚀 Iniciando Teste de Inferência dos 3 Casos (Modelo Exaustivo)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./speecht5_laps_lora_exhaustive"
    
    # --- Load Models ---
    print("Carregando modelos...")
    processor = SpeechT5Processor.from_pretrained(output_dir)
    
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # Load Base + LoRA
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    # Load the specific checkpoint or the final output
    model = PeftModel.from_pretrained(model, output_dir)
    model.to(device)
    model.eval()
    
    # --- Load Speaker Embedding ---
    print("Carregando Speaker Embedding...")
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    if "wav" in dataset.column_names: dataset = dataset.rename_column("wav", "audio")
    
    # Extract Speaker IDs (same logic as training)
    def extract_speaker_id(url):
        if "LapsBM-" in url:
            parts = url.split('/')
            for p in parts:
                if p.startswith("LapsBM-") and ".tar" not in p:
                    return p.replace("LapsBM-", "")
                if p.startswith("LapsBM-") and ".tar" in p:
                     return p.split(".")[0].replace("LapsBM-", "")
        return "unknown"

    all_speakers = [extract_speaker_id(x) for x in dataset["__url__"]]
    counts = Counter(all_speakers)
    most_common_spk, _ = counts.most_common(1)[0]
    indices = [i for i, x in enumerate(all_speakers) if x == most_common_spk]
    
    # Get first sample for embedding
    sample = dataset[indices[0]]
    audio = sample["audio"]["array"]
    
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(source=spk_model_name, run_opts={"device": device}, savedir="/tmp/speechbrain_xvector")
    
    with torch.no_grad():
        emb = speaker_model.encode_batch(torch.tensor(audio))
        emb = torch.nn.functional.normalize(emb, dim=2)
        emb = emb.squeeze().cuda() # Move directly to cuda for generation
        emb = emb.unsqueeze(0) # [1, 512]

    # --- Test Cases ---
    cases = [
        ("caso1_standard", "O treinamento foi finalizado e agora estamos testando a voz em português."),
        ("caso2_paris", "Há algumas coisas que não podem deixar de serem vistas em Paris."),
        ("caso3_extra", "O modelo exaustivo deve ser capaz de clonar a voz com alta fidelidade.")
    ]
    
    print("\n🎧 Gerando áudios...")
    for filename, text in cases:
        clean_text = normalize_text(text)
        print(f"   📝 {filename}: '{clean_text}'")
        
        inputs = processor(text=clean_text, return_tensors="pt")
        
        with torch.no_grad():
            spec = model.generate(input_ids=inputs["input_ids"].to(device), speaker_embeddings=emb)
            audio_out = vocoder(spec)
            
        out_path = f"{filename}_exhaustive.wav"
        sf.write(out_path, audio_out.squeeze().cpu().numpy(), 16000)
        print(f"      ✅ Salvo: {out_path}")

if __name__ == "__main__":
    main()
