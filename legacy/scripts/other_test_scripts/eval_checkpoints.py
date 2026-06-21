import os
import torch
import torchaudio

# Patch MUST be prior to speechbrain import
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import unicodedata
import soundfile as sf
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel
from datasets import load_dataset, Audio
from speechbrain.inference.speaker import EncoderClassifier
from collections import Counter

def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def main():
    print("🚀 Avaliando Checkpoints...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "microsoft/speecht5_tts"
    vocoder_id = "microsoft/speecht5_hifigan"
    
    # 1. Load Fixed Components
    processor = SpeechT5Processor.from_pretrained(base_model)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id).to(device)
    
    # 2. Get Embedding (Sample 0)
    dataset = load_dataset("falabrasil/lapsbm", split="test", streaming=False)
    if "wav" in dataset.column_names: dataset = dataset.rename_column("wav", "audio")
    # Identify F004
    def extract_id(url): return url.split('/')[-1].split('.')[0].replace("LapsBM-", "")
    ids = [extract_id(x) for x in dataset["__url__"]]
    idx = ids.index("F004")
    
    # Extract
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(source=spk_model_name, run_opts={"device": device}, savedir=os.path.join("/tmp", spk_model_name))
    
    with torch.no_grad():
        wav = torch.tensor(dataset[idx]["audio"]["array"])
        # Fix: Normalize (1, 1, 512) -> dim=2, THEN reshape
        emb = speaker_model.encode_batch(wav)
        emb = torch.nn.functional.normalize(emb, dim=2)
        emb = emb.squeeze().unsqueeze(0).to(device)
    
    # 3. Test Sentence
    text = "há algumas coisas que não podem deixar de serem vistas em paris."
    clean_text = normalize_text(text)
    print(f"📝 Frase: {clean_text}")
    inputs = processor(text=clean_text, return_tensors="pt")

    # 4. Iterate Checkpoints
    checkpoints = [100, 200, 300, 400, 500]
    
    for cp in checkpoints:
        cp_path = f"./speecht5_laps_lora_v17_log_fix/checkpoint-{cp}"
        if not os.path.exists(cp_path):
            print(f"⚠️ Checkpoint {cp} não encontrado. Pulando.")
            continue
            
        print(f"🔄 Testando Checkpoint-{cp}...")
        
        # Load Base Model FRESH each time to be safe
        model = SpeechT5ForTextToSpeech.from_pretrained(base_model)
        # Load LoRA Adapter
        model = PeftModel.from_pretrained(model, cp_path)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            spec = model.generate(input_ids=inputs["input_ids"].to(device), speaker_embeddings=emb)
            audio = vocoder(spec)
            
        fname = f"eval_v17_cp{cp}.wav"
        sf.write(fname, audio.squeeze().cpu().numpy(), 16000)
        print(f"   ✅ Salvo: {fname}")

if __name__ == "__main__":
    main()
