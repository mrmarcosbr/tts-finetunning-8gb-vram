import os
import sys
import yaml
import torch
import torchaudio
import unicodedata
import argparse
import soundfile as sf
import numpy as np
from datetime import datetime

# --- 1. Monkey Patch & Setup (MUST BE BEFORE SPEECHBRAIN IMPORT) ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel
from speechbrain.inference.speaker import EncoderClassifier
from collections import Counter

fine_tuning_model_name = "SpeechT5"

# Remove acentos e caracteres especiais de uma string
def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_hardware_profile():
    """Tenta identificar o melhor perfil de hardware automaticamente."""
    print("🔍 Autodetectando hardware...")
    
    # 1. Verificar NVIDIA CUDA
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   🖥️  GPU NVIDIA Detectada: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB VRAM)")
        
        # Se tiver 14GB ou mais (margem para 16GB totais), usa 16gb. Caso contrário, 8gb.
        if total_vram >= 14.0:
            return "cuda_16gb"
        return "cuda_8gb"
    
    # 2. Verificar Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("   🍎 Apple Silicon Detectado (MPS Backend)")
        return "macbook"
    
    # 3. Fallback para CPU
    print("   ⚠️ Nenhuma GPU de alta performance detectada. Usando CPU.")
    return "cpu"

def main():
    # 1. Parsing de Argumentos
    parser = argparse.ArgumentParser(description="Teste de Inferência SpeechT5 LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional)")
    parser.add_argument("--model_name", type=str, default=None, help="Nome da subpasta do modelo (ex: SpeechT5-2026-03-30-23-33-05)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo config.yaml")
    args = parser.parse_args()

    # 2. Carregar Configurações
    full_cfg = load_config(args.config)
    
    # 3. Identificar o Perfil (Manual ou Automático)
    profile_name = args.profile
    if not profile_name:
        profile_name = detect_hardware_profile()
    
    if profile_name not in full_cfg['profiles']:
        print(f"❌ Perfil '{profile_name}' não existe no YAML. Usando 'cpu' como fallback.")
        profile_name = "cpu"
    
    profile_cfg = full_cfg['profiles'][profile_name]
    print(f"🚀 Perfil Ativo para Inferência: {profile_name.upper()}")

    # 4. Localizar Pasta do Modelo
    profile_output_dir = profile_cfg['output_dir']
    
    if args.model_name:
        model_path = os.path.join(profile_output_dir, args.model_name)
        if not os.path.exists(model_path):
            print(f"❌ Modelo especificado não encontrado: {model_path}")
            sys.exit(1)
    else:
        # Tentar encontrar o modelo mais recente
        if not os.path.exists(profile_output_dir):
             print(f"⚠️ Diretório do perfil não encontrado: {profile_output_dir}. Usando diretório padrão fallback.")
             model_path = "./speecht5_laps_lora_exhaustive"
        else:
             subfolders = [f for f in os.listdir(profile_output_dir) if os.path.isdir(os.path.join(profile_output_dir, f)) and f.startswith(fine_tuning_model_name)]
             if not subfolders:
                 print(f"⚠️ Nenhum modelo treinado encontrado em {profile_output_dir}. Usando diretório padrão fallback.")
                 model_path = "./speecht5_laps_lora_exhaustive"
             else:
                 subfolders.sort() # Nome formatado YYYY-MM-DD-HH-MM-SS ordena bem alfabeticamente
                 latest_model = subfolders[-1]
                 model_path = os.path.join(profile_output_dir, latest_model)
                 print(f"📂 Modelo mais recente detectado: {latest_model}")

    print(f"📍 Usando diretório do modelo: {model_path}")

    # 5. Configurar Device
    device = profile_cfg.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
         device = "cpu"

    # --- Load Models ---
    print("Carregando modelos...")
    if not os.path.exists(model_path):
        print(f"❌ Diretório do modelo não existe: {model_path}")
        sys.exit(1)
        
    processor = SpeechT5Processor.from_pretrained(model_path)
    vocoder = SpeechT5HifiGan.from_pretrained(full_cfg['model']['vocoder_id']).to(device)
    
    # Load Base + LoRA
    model = SpeechT5ForTextToSpeech.from_pretrained(full_cfg['model']['id'])
    model = PeftModel.from_pretrained(model, model_path)
    model.to(device)
    model.eval()
    
    # --- Load Speaker Embedding ---
    print("Carregando Speaker Embedding...")
    dataset = load_dataset(full_cfg['dataset']['id'], split=full_cfg['dataset']['split'], streaming=False)
    if "wav" in dataset.column_names: dataset = dataset.rename_column("wav", "audio")
    
    def extract_speaker_id(url):
        if url and "LapsBM-" in url:
            parts = url.split('/')
            for p in parts:
                if p.startswith("LapsBM-"):
                    if ".tar" in p:
                        return p.split(".")[0].replace("LapsBM-", "")
                    return p.replace("LapsBM-", "")
        return "unknown"

    all_speakers = [extract_speaker_id(x.get("__url__", "")) for x in dataset]
    counts = Counter(all_speakers)
    most_common_spk, _ = counts.most_common(1)[0]
    indices = [i for i, x in enumerate(all_speakers) if x == most_common_spk]
    
    sample = dataset[indices[0]]
    audio = sample["audio"]["array"]
    
    speaker_model = EncoderClassifier.from_hparams(source=full_cfg['model']['speaker_encoder_id'], run_opts={"device": device})
    
    with torch.no_grad():
        emb = speaker_model.encode_batch(torch.tensor(audio))
        emb = torch.nn.functional.normalize(emb, dim=2).squeeze().to(device)
        emb = emb.unsqueeze(0) 

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
