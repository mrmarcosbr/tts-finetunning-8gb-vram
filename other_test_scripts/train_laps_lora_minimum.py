import os
import torch
import torchaudio

# --- 1. Monkey Patch & Setup ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
# Force usage of soundfile for stability (deprecated in 2.1+, relying on monkey patch and auto-detection)
# torchaudio.set_audio_backend("soundfile") 

from datasets import load_dataset, Audio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments, SpeechT5HifiGan
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def main():
    print("🚀 Iniciando Preparação para Treino LoRA (Laps Benchmark)...")
    
    # Configs
    model_id = "microsoft/speecht5_tts"
    dataset_id = "falabrasil/lapsbm"
    output_dir = "./speecht5_laps_lora"
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 2. Load Processors ---
    processor = SpeechT5Processor.from_pretrained(model_id)
    
    # --- 3. Load Dataset ---
    # LapsBM só tem 'test' split no HF, vamos usar como train
    print("Loading dataset...")
    # Usando streaming=False pois o dataset é pequeno (700 itens) e cabe na RAM
    dataset = load_dataset(dataset_id, split="test", streaming=False)
    print(f"Dataset carregado: {dataset.num_rows} exemplos") # Fixed attribute access for dataset object
    
    # Inspect columns
    print(f"Colunas: {dataset.column_names}")
    
    # Rename columns to standard names if needed
    if "txt" in dataset.column_names and "text" not in dataset.column_names:
        dataset = dataset.rename_column("txt", "text")
    if "wav" in dataset.column_names and "audio" not in dataset.column_names:
        dataset = dataset.rename_column("wav", "audio")

    # Extract Speaker ID from URL or Key
    # URL format: .../LapsBM-F004/LapsBM-F004.tar.gz -> Speaker = F004
    def extract_speaker(batch):
        # Assumes __url__ contains the folder structure
        # If __url__ is missing (sometimes happens), try __key__ or filename
        url = batch.get("__url__", "")
        if "LapsBM-" in url:
            # Example: .../LapsBM-F004/LapsBM-F004.tar.gz
            # Split by / and find the part with LapsBM-
            parts = url.split('/')
            for p in parts:
                if p.startswith("LapsBM-") and ".tar" not in p:
                    return {"speaker_id": p.replace("LapsBM-", "")}
                if p.startswith("LapsBM-") and ".tar" in p:
                     return {"speaker_id": p.split(".")[0].replace("LapsBM-", "")}
        return {"speaker_id": "unknown"}

    print("Extracting speaker IDs...")
    dataset = dataset.map(extract_speaker)
    
    # Filter by Speaker (Assuming 'speaker_id' exists, inspect first entry)
    sample = dataset[0]
    print(f"Exemplo 0: {sample}")
    
    # Speaker strategy: Pick the most frequent speaker
    # LapsBM speakers usually named like 'F01', 'M01'.
    # Vamos agrupar e ver quem tem mais dados
    from collections import Counter
    speaker_counts = Counter(dataset['speaker_id'])
    target_speaker = speaker_counts.most_common(1)[0][0]
    print(f"🎯 Selecionando Speaker mais frequente: {target_speaker} ({speaker_counts[target_speaker]} áudios)")
    
    dataset = dataset.filter(lambda x: x['speaker_id'] == target_speaker)
    print(f"Dataset filtrado: {len(dataset)} exemplos")

    # Clean text (normalize if needed)
    # LapsBM text is usually good.
    
    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # --- 4. Speaker Embeddings ---
    print("Carregando SpeechBrain para X-Vectors...")
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

    # --- 5. Preprocessing ---
    # Prepare inputs using processor
    def prepare_dataset(batch):
        # Load audio
        audio = batch["audio"]
        
        # Process text -> input_ids
        batch["input_ids"] = processor(text=batch["text"], return_tensors="pt").input_ids[0]
        
        # Process audio -> log-mel spectrogram (labels)
        # Manual extraction using librosa since default feature_extractor failed (returned raw audio)
        # SpeechT5 defaults: 80 mels. Using standard hop/win for now.
        # Note: SpeechT5 pre-trained on Log-Mel.
        
        y = audio["array"]
        sr = 16000
        
        # Calculate Mel Spectrogram
        # Using standard TTS params: n_fft=1024, hop_length=256, win_length=1024
        # Note: If the model is sensitive to these, we might need adjustments.
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=1024, 
            hop_length=256, 
            win_length=1024, 
            n_mels=80,
            fmin=80,
            fmax=7600
        )
        
        # Convert to log scale (db)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize? SpeechT5 expects normalized? Usually yes.
        # Let's standardize approx to [-4, 4] or similar if needed.
        # But simple log mel is often enough for LoRA to adapt.
        # We need Transpose: (Freq, Time) -> (Time, Freq)
        log_mel_spectrogram = log_mel_spectrogram.T
        
        batch["labels"] = torch.tensor(log_mel_spectrogram)
        
        # Debug Shape
        # print(f"DEBUG: labels shape in prep: {batch['labels'].shape}") 
        
        # Create speaker embedding
        batch["speaker_embeddings"] = create_speaker_embedding(audio["array"])
        
        return batch

    print("Processando dataset (Feature extraction)...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    # Set format to pytorch tensors
    dataset.set_format(type="torch", columns=["input_ids", "labels", "speaker_embeddings"])
    
    # Split Train/Val (90/10)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # --- 6. Data Collator ---
    @dataclass
    class TTSDataCollatorWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            label_features = [{"input_values": feature["labels"]} for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]

            # Pad text inputs
            batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")

            labels = [feature["labels"] for feature in features]
            # Debug Shape
            # print(f"DEBUG: labels[0] shape in collator: {labels[0].shape}")
            
            # Find max length
            max_label_length = max([l.shape[0] for l in labels])
            
            # Message safety: SpeechT5 reduction factor is 2.
            # We must output even frames. Pad to multiple of 2.
            if max_label_length % 2 != 0:
                max_label_length += 1
            
            padded_labels = []
            labels_masks = []
            for l in labels:
                if l.dim() == 1:
                    # Fix 1D issue if it occurs (unexpected)
                    # Assuming flattened (seq * 80) -> (seq, 80)
                    if l.shape[0] % 80 == 0:
                        l = l.view(-1, 80)
                    else:
                        raise ValueError(f"Unexpected 1D shape {l.shape}")
                        
                l_len = l.shape[0]
                pad_len = max_label_length - l_len
                # Pad (seq, 80) -> Pad ONLY the first dimension (time)
                # pad arg logic: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
                # Labels are (Time, 80). We want to pad Time (dim 0).
                # So we pad last dim (80) with 0,0. And 2nd last (Time) with 0, pad_len.
                padded_l = torch.nn.functional.pad(l, (0, 0, 0, pad_len), value=0.0)
                padded_labels.append(padded_l)
                
                # Mask: 1 for real, 0 for pad
                mask = torch.ones(l_len, dtype=torch.long)
                padded_mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                labels_masks.append(padded_mask)
            
            batch["labels"] = torch.stack(padded_labels)
            
            # Speaker embeddings: stack
            # Use torch.stack if they are tensors, else use torch.tensor
            if isinstance(speaker_features[0], torch.Tensor):
                 batch["speaker_embeddings"] = torch.stack(speaker_features)
            else:
                 batch["speaker_embeddings"] = torch.tensor(np.array(speaker_features))
            
            # Filter batch to only Allowed Keys (optional if remove_unused_columns=True)
            # But let's verify what is actually here
            print(f"DEBUG: Batch Keys entering Trainer: {batch.keys()}")
            
            allowed_keys = ["input_ids", "attention_mask", "labels", "speaker_embeddings"]
            
            # If removing unused columns is True, this manual filter is redundant but safe.
            # However, if 'inputs_embeds' persists, it's weird.
            filtered_batch = {k: v for k, v in batch.items() if k in allowed_keys}
            return filtered_batch

    data_collator = TTSDataCollatorWithPadding(processor=processor)

    # --- 7. Model & LoRA ---
    print("Carregando modelo...")
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    
    # Freeze basics? LoRA will handle trainable params.
    # Lora Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # SpeechT5 attention modules
        lora_dropout=0.05,
        bias="none",
        task_type=None 
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Ensure float32 for stability if needed, but we want fp16 for speed/mem
    # LoRA usually works fine in fp16.

    # --- 8. Trainer ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        num_train_epochs=50, # Ajustar conforme tempo (MVP)
        save_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,
        save_total_limit=2,
        remove_unused_columns=True, # Let Trainer filter arguments based on model signature
        label_names=["labels"],
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor, # Hacky to pass something
    )
    
    print("🔥 Iniciando Treino...")
    trainer.train()
    
    print("💾 Salvando modelo final...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
