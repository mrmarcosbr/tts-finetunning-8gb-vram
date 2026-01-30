import os
import torch
import torchaudio
import unicodedata

# --- 1. Monkey Patch & Setup ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments, SpeechT5HifiGan
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from collections import Counter

# --- NORMALIZATION ---
# 'NFD' é uma forma de normalização que separa as letras de seus acentos
# 'Mn' é uma categoria de caracteres que inclui acentos
# Portanto, essa linha remove os acentos da string
def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def main():
    print("🚀 Iniciando Treinamento Exaustivo (Overfitting)...")
    
    # Configs
    model_id = "microsoft/speecht5_tts"   # https://huggingface.co/microsoft/speecht5_tts
    dataset_id = "falabrasil/lapsbm"   # https://huggingface.co/datasets/falabrasil/lapsbm
    output_dir = "./speecht5_laps_lora_exhaustive" # New Output Dir
    
    # REMOVED TARGET_SAMPLES = 20
    # We will use ALL samples for the speaker
    
    # Verify if will use GPU NVIDIA or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 2. Load Processors ---
    # O processor é o "tradutor" e o "organizador" dos dados. Ele lida com duas frentes:
    # Para o Texto (Entrada): Ele atua como um Tokenizador. Modelos de IA não entendem letras; eles entendem números. O processor converte frases em sequências de IDs numéricos que o modelo consegue processar.
    # Para o Áudio (Saída/Treino): Ele também prepara as características do áudio (como extração de log-mel spectrograms).
    # Normalização: Ele garante que o texto esteja no formato correto (limpeza, pontuação, etc.) para que o modelo não se confunda.
    processor = SpeechT5Processor.from_pretrained(model_id)

    # O modelo principal de IA geralmente não gera um arquivo de áudio (.wav) direto; ele gera um Espectrograma de Mel.
    # Um espectrograma é como uma "imagem" da voz, mas você não consegue ouvi-lo. É aí que entra o Vocoder:
    # Sintetizador de Onda: Ele pega essa "imagem" (espectrograma) e a converte em ondas sonoras reais que nossos ouvidos reconhecem.
    # Qualidade (HiFi-GAN): O termo "HiFi-GAN" refere-se a uma rede neural específica (Generative Adversarial Network) treinada para criar áudios de alta fidelidade, garantindo que a voz soe natural e não "robótica" ou metálica.
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # --- 3. Load Dataset & Select Speaker ---
    print("Carregando dataset...")
    dataset = load_dataset(dataset_id, split="test", streaming=False)
    
    if "wav" in dataset.column_names: dataset = dataset.rename_column("wav", "audio")
    if "txt" in dataset.column_names: dataset = dataset.rename_column("txt", "text")

    # Extract Speaker IDs
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
    most_common_spk, count = counts.most_common(1)[0]
    print(f"🏆 Locutor Principal: {most_common_spk} ({count} amostras disponíveis)")
    
    # Filter for this speaker
    indices = [i for i, x in enumerate(all_speakers) if x == most_common_spk]
    
    # NO SAMPLING LIMIT - Use all found
    dataset = dataset.select(indices)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"✅ Utilizando TODAS as {len(dataset)} amostras para Overfitting.")
    
    # --- 4. Speaker Embeddings ---
    print("Carregando Speaker Encoder...")
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(source=spk_model_name, run_opts={"device": device}, savedir=os.path.join("/tmp", spk_model_name))

    def create_speaker_embedding(waveform):
        with torch.no_grad():
            emb = speaker_model.encode_batch(torch.tensor(waveform))
            emb = torch.nn.functional.normalize(emb, dim=2)
            emb = emb.squeeze().cpu().numpy()
        return emb

    # --- 5. Preprocessing (Method 3: Pure Log10 + Normalization) ---
    def prepare_dataset(batch):
        audio = batch["audio"]
        raw_text = batch.get("text", batch.get("txt", "dummy"))
        
        # NORMALIZE TEXT
        clean_text = normalize_text(raw_text)
        
        batch["input_ids"] = processor(text=clean_text, return_tensors="pt").input_ids[0]
        
        y = np.array(audio["array"])
        
        # Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600)
        
        # Pure Log10 (V12 Logic - Critical for Low Loss)
        mel = np.clip(mel, a_min=1e-5, a_max=None)
        log_mel = np.log10(mel).T
        
        batch["labels"] = torch.tensor(log_mel)
        batch["speaker_embeddings"] = create_speaker_embedding(y)
        return batch

    print("Extracting features (Log10 + Normalized)...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "labels", "speaker_embeddings"])
    
    # --- 6. Data Collator ---
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
            
            padded_labels = []
            for l in labels:
                pad_len = max_len - l.shape[0]
                # Pad with -5.0 (Silence in Log10)
                padded_labels.append(torch.nn.functional.pad(l, (0, 0, 0, pad_len), value=-5.0))
            
            batch["labels"] = torch.stack(padded_labels)
            batch["speaker_embeddings"] = torch.stack([torch.tensor(s) for s in speaker_features])
            return batch

    data_collator = TTSDataCollatorWithPadding(processor=processor)

    # --- 7. Model (V12 Base: Rank 32) ---
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    
    # MAXIMUM Rank (256) for aggressive VRAM usage and overfitting
    # Dropout = 0.0 to force memorization (no regularization)
    lora_config = LoraConfig(r=256, lora_alpha=512, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], lora_dropout=0.0, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=20, # FULL BATCH (20 samples)
        gradient_accumulation_steps=1, # No accumulation needed for single batch
        learning_rate=1e-5, 
        weight_decay=0.0, # NO REGULARIZATION for overfitting
        num_train_epochs=15000, # INCREASED TO 15000 (~1 hour)
        logging_steps=50,
        save_steps=200, 
        save_total_limit=1, # Save only the last checkpoint ~400MB
        fp16=False,
        dataloader_num_workers=0
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator, tokenizer=processor.feature_extractor)
    
    print("\n🔥 Treinando Exaustivamente (1000 Epochs, BS=8)...")
    trainer.train()
    
    print("💾 Saving...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # --- IMMEDIATE TEST ---
    print("\n🧪 Verificando resultado...")
    model.eval()
    
    emb = dataset[0]["speaker_embeddings"].unsqueeze(0).to(device)
    
    # Standard Sentence
    text1 = "O treinamento exaustivo foi finalizado com sucesso."
    clean1 = normalize_text(text1)
    print(f"   🔊 Gerando: '{clean1}'")
    
    inputs1 = processor(text=clean1, return_tensors="pt")
    with torch.no_grad():
        spec = model.generate(input_ids=inputs1["input_ids"].to(device), speaker_embeddings=emb)
        audio = vocoder(spec)
    sf.write(f"teste_exhaustive.wav", audio.squeeze().cpu().numpy(), 16000)
    print("   ✅ Salvo: teste_exhaustive.wav")

if __name__ == "__main__":
    main()
