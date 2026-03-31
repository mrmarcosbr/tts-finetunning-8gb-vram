import os
import sys
import yaml
import torch
import torchaudio
import unicodedata
import argparse
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from collections import Counter

from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments, SpeechT5HifiGan
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from speechbrain.inference.speaker import EncoderClassifier

# Configuração de encoding para Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Funções Utilitárias ---
def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def extract_speaker_id(url):
    if url and "LapsBM-" in url:
        parts = url.split('/')
        for p in parts:
            if p.startswith("LapsBM-"):
                if ".tar" in p:
                    return p.split(".")[0].replace("LapsBM-", "")
                return p.replace("LapsBM-", "")
    return "unknown"

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
    parser = argparse.ArgumentParser(description="Treinamento SpeechT5 LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional, senão houver será detectado)")
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
    print(f"🚀 Perfil Ativo: {profile_name.upper()}")

    # 4. Setup do Hardware
    device = profile_cfg.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
         print("⚠️ MPS solicitado mas não disponível. Usando CPU.")
         device = "cpu"

    # 5. Load Processors & Dataset
    processor = SpeechT5Processor.from_pretrained(full_cfg['model']['id'])
    vocoder = SpeechT5HifiGan.from_pretrained(full_cfg['model']['vocoder_id']).to(device)
    
    print("📥 Carregando dataset em modo streaming (para identificar locutores)...")
    dataset_stream = load_dataset(full_cfg['dataset']['id'], split=full_cfg['dataset']['split'], streaming=True)
    
    all_data = []
    print("Extraindo amostras...")
    for item in dataset_stream:
        if "wav" in item and "audio" not in item: item["audio"] = item["wav"]
        if "txt" in item and "text" not in item: item["text"] = item["txt"]
        all_data.append(item)
    full_dataset = Dataset.from_list(all_data)
    
    # 6. Seleção Automática de Locutor(es)
    all_speakers = [extract_speaker_id(x.get("__url__", "")) for x in all_data]
    counts = Counter(all_speakers)
    num_spk = full_cfg['dataset'].get('num_speakers', 1)
    
    if num_spk == 0:
        dataset = full_dataset
        print(f"✅ Todos os locutores selecionados.")
    else:
        top_speakers = [spk for spk, count in counts.most_common(num_spk)]
        print(f"🏆 Locutores selecionados: {top_speakers}")
        indices = [i for i, spk in enumerate(all_speakers) if spk in top_speakers]
        dataset = full_dataset.select(indices)
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 7. Speaker Encoder
    print("Carregando Speaker Encoder...")
    speaker_model = EncoderClassifier.from_hparams(source=full_cfg['model']['speaker_encoder_id'], run_opts={"device": device})

    def prepare_dataset(batch):
        audio = batch["audio"]
        clean_text = normalize_text(batch.get("text", "dummy"))
        batch["input_ids"] = processor(text=clean_text, return_tensors="pt").input_ids[0]
        y = np.array(audio["array"])
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=80, fmin=80, fmax=7600)
        log_mel = np.log10(np.clip(mel, 1e-5, None)).T
        batch["labels"] = torch.tensor(log_mel)
        with torch.no_grad():
            emb = speaker_model.encode_batch(torch.tensor(y))
            batch["speaker_embeddings"] = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
        return batch

    print("📊 Processando características de áudio...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "labels", "speaker_embeddings"])
    
    # 8. Data Collator
    @dataclass
    class TTSDataCollatorWithPadding:
        processor: Any
        def __call__(self, features):
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            labels = [feature["labels"] for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]
            batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")
            max_len = max([l.shape[0] for l in labels])
            if max_len % 2 != 0: max_len += 1
            padded_labels = [torch.nn.functional.pad(l, (0, 0, 0, max_len - l.shape[0]), value=-5.0) for l in labels]
            batch["labels"] = torch.stack(padded_labels)
            batch["speaker_embeddings"] = torch.stack([s.clone().detach() if isinstance(s, torch.Tensor) else torch.tensor(s) for s in speaker_features])
            return batch

    # 9. Modelo LoRA
    SpeechT5ForTextToSpeech._keys_to_ignore_on_load_unexpected = [r'.*encode_positions\.pe']
    model = SpeechT5ForTextToSpeech.from_pretrained(full_cfg['model']['id'])
    lora_config = LoraConfig(
        r=full_cfg['lora']['r'], lora_alpha=full_cfg['lora']['alpha'], 
        target_modules=full_cfg['lora']['target_modules'], lora_dropout=full_cfg['lora']['dropout'], bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    # 10. Training Engine
    train_params = full_cfg['training']
    training_args = TrainingArguments(
        output_dir=profile_cfg['output_dir'],
        per_device_train_batch_size=profile_cfg['batch_size'],
        gradient_accumulation_steps=profile_cfg['gradient_accumulation_steps'],
        learning_rate=train_params['learning_rate'], 
        num_train_epochs=train_params['num_epochs'],
        logging_steps=train_params['logging_steps'],
        save_steps=train_params['save_steps'], 
        fp16=profile_cfg['fp16'],
        save_total_limit=1,
        dataloader_num_workers=0
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=TTSDataCollatorWithPadding(processor=processor))
    
    print(f"\n🚀 Iniciando treinamento exaustivo no perfil autodetectado: {profile_name}")
    trainer.train()
    
    # 11. Salvar
    model.save_pretrained(profile_cfg['output_dir'])
    processor.save_pretrained(profile_cfg['output_dir'])
    print(f"💾 Modelo final salvo em: {profile_cfg['output_dir']}")

if __name__ == "__main__":
    main()
