import os
import torch
import torchaudio

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

def main():
    print("🚀 Iniciando Sanity Check (Overfitting 1 Sample)...")
    
    # Configs
    model_id = "microsoft/speecht5_tts"
    dataset_id = "falabrasil/lapsbm"
    output_dir = "./speecht5_sanity_check"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 2. Load Processors ---
    processor = SpeechT5Processor.from_pretrained(model_id)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # --- 3. Load 1 Sample ---
    print("Loading 1 sample...")
    dataset = load_dataset(dataset_id, split="test", streaming=False)
    
    if "wav" in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Pick 1 sample and replicate it
    sample = dataset[0] # The one with "Paris"
    
    # DEBUG: Print Text
    sample_text = sample.get("text", sample.get("txt", "Unknown"))
    print(f"\n📝 TEXTO DO SANITY CHECK: '{sample_text}'\n")
    
    # Manually create a small dataset of just this 1 sample repeated
    # This guarantees overfitting if model works
    data_list = [sample] * 100 
    dataset = Dataset.from_list(data_list)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"Created synthetic dataset of {len(dataset)} identical samples.")
    
    # --- 4. Speaker Embeddings ---
    print("Carregando Speaker Encoder...")
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name, 
        run_opts={"device": device}, 
        savedir=os.path.join("/tmp", spk_model_name)
    )

    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    # --- 5. Preprocessing (PURE LOG - Method 3) ---
    def prepare_dataset(batch):
        audio = batch["audio"]
             
        # Text
        text = batch.get("text", batch.get("txt", "dummy"))
        batch["input_ids"] = processor(text=text, return_tensors="pt").input_ids[0]
        
        # Audio
        y = np.array(audio["array"])
        sr = 16000
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600
        )
        
        # Method 3: Clip and Log10
        mel_spectrogram = np.clip(mel_spectrogram, a_min=1e-5, a_max=None)
        log_mel_spectrogram = np.log10(mel_spectrogram).T # (Time, 80)
        
        batch["labels"] = torch.tensor(log_mel_spectrogram)
        batch["speaker_embeddings"] = create_speaker_embedding(y)
        
        return batch

    print("Processando dataset...")
    dataset = dataset.map(prepare_dataset)
    dataset.set_format(type="torch", columns=["input_ids", "labels", "speaker_embeddings"])
    
    # --- 6. Collator ---
    @dataclass
    class TTSDataCollatorWithPadding:
        processor: Any
        def __call__(self, features):
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            labels = [feature["labels"] for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]
            
            batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")
            
            max_label_length = max([l.shape[0] for l in labels])
            if max_label_length % 2 != 0: max_label_length += 1
            
            padded_labels = []
            for l in labels:
                pad_len = max_label_length - l.shape[0]
                # Pad with silence (-5.0 for log domain)
                padded_l = torch.nn.functional.pad(l, (0, 0, 0, pad_len), value=-5.0) 
                padded_labels.append(padded_l)
            
            batch["labels"] = torch.stack(padded_labels)
            batch["speaker_embeddings"] = torch.stack(speaker_features)
            return batch

    data_collator = TTSDataCollatorWithPadding(processor=processor)

    # --- 7. Model ---
    print("Carregando modelo...")
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    
    lora_config = LoraConfig(
        r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    # --- 8. Train Loop ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=10, # train on 10 copies at a time
        gradient_accumulation_steps=1,
        learning_rate=1e-4, # Aggressive LR for overfitting
        max_steps=100,      # Quick overfitting
        logging_steps=10,
        save_steps=500,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=data_collator, tokenizer=processor.feature_extractor,
    )
    
    print("🔥 Overfitting (100 steps)...")
    trainer.train()
    
    # --- 9. Immediate Inference ---
    print("🧪 Verificando resultado IMEDIATO...")
    model.eval()
    
    # Take the exact text and embeddings from the sample
    sample_text = dataset[0]["input_ids"] # This is tensor input_ids? No, we need text for inference processor? 
    # Actually we can pass input_ids directly to generate if we format it right.
    # dataset[0] is formatted torch.
    
    input_ids = dataset[0]["input_ids"].unsqueeze(0).to(device)
    emb = dataset[0]["speaker_embeddings"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Generate generic
        spec = model.generate(input_ids=input_ids, speaker_embeddings=emb)
        audio = vocoder(spec)
        
    print(f"Generated Audio Shape: {audio.shape}")
    print(f"Stats: Min={audio.min():.4f}, Max={audio.max():.4f}")
    
    sf.write("sanity_check_output.wav", audio.squeeze().cpu().numpy(), 16000)
    print("✅ Salvo: sanity_check_output.wav")
    
    # --- 10. English Test ---
    print("\n🇺🇸 Testando geração em INGLÊS...")
    # text_en = "Hello, this is a test in English to check for noise."
    text_en = "O treinamento foi finalizado e agora estamos testando a voz em português."
    inputs_en = processor(text=text_en, return_tensors="pt")
    
    with torch.no_grad():
        spec_en = model.generate(
            input_ids=inputs_en["input_ids"].to(device), 
            speaker_embeddings=emb # Use same embedding
        )
        audio_en = vocoder(spec_en)
        
    sf.write("sanity_check_english.wav", audio_en.squeeze().cpu().numpy(), 16000)
    print("✅ Salvo: sanity_check_english.wav")

if __name__ == "__main__":
    main()
