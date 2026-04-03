import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis de ambiente (HF_TOKEN)
load_dotenv()
import yaml
import torch
import torchaudio
import unicodedata
import argparse
import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv

# Carregar variáveis de ambiente (HF_TOKEN)
load_dotenv()
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
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

def extract_speaker_id(item):
    for k in ['speaker_id', 'speaker', 'user_id', 'client_id']:
        if k in item and item[k] is not None:
            return str(item[k])
            
    # Para lidar com caminhos remotos (__url__) e locais (audio['path'])
    possible_paths = [
        item.get("__url__", ""),
        item.get("audio", {}).get("path", "") if isinstance(item.get("audio"), dict) else "",
        item.get("audio", "") if isinstance(item.get("audio"), str) else ""
    ]
    
    for p in possible_paths:
        if p and "LapsBM-" in str(p):
            # Ex: foo/bar/LapsBM-M-001.wav -> LapsBM-M-001.wav -> M-001
            filename = os.path.basename(str(p))
            if filename.startswith("LapsBM-"):
                return filename.split(".")[0].replace("LapsBM-", "")
                
    return "unknown"

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_hardware_profile(full_cfg):
    """Tenta identificar o melhor perfil de hardware automaticamente."""
    print("🔍 Autodetectando hardware...")
    
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   🖥️  GPU NVIDIA Detectada: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB VRAM)")
        if total_vram >= 14.0: return "cuda_16gb"
        return "cuda_8gb"
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("   🍎 Apple Silicon Detectado (MPS Backend)")
        return "macbook"
    
    print("   ⚠️ Nenhuma GPU de alta performance detectada. Usando CPU.")
    return "cpu"

# ==============================================================================
# MODEL HANDLERS (Um para cada arquitetura)
# ==============================================================================

class ModelHandler:
    """Base para suporte a diferentes arquiteturas de TTS."""
    def __init__(self, model_cfg: Dict, device: str):
        self.model_cfg = model_cfg
        self.device = device
        self.processor = None
        self.model = None

    def load(self):
        raise NotImplementedError()

    def prepare_item(self, item, processor, speaker_model):
        raise NotImplementedError()

    def get_collator(self):
        raise NotImplementedError()

class SpeechT5Handler(ModelHandler):
    def load(self):
        print(f"📥 Carregando SpeechT5: {self.model_cfg['id']}")
        self.processor = SpeechT5Processor.from_pretrained(self.model_cfg['id'])
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_cfg['vocoder_id']).to(self.device)
        
        SpeechT5ForTextToSpeech._keys_to_ignore_on_load_unexpected = [r'.*encode_positions\.pe']
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_cfg['id'])
        
        # Aplicar LoRA
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none", task_type=None
        )
        self.model = get_peft_model(self.model, peft_config)
        return self.model, self.processor

    def prepare_item(self, batch, processor, speaker_model, language=None):
        audio = batch["audio"]
        clean_text = normalize_text(batch.get("text", "dummy"))
        batch["input_ids"] = processor(text=clean_text, return_tensors="pt").input_ids[0]
        y = np.array(audio["array"])
        
        # SpeechT5 espera mel-spectrogram como labels
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=80, fmin=80, fmax=7600)
        log_mel = np.log10(np.clip(mel, 1e-5, None)).T
        batch["labels"] = torch.tensor(log_mel)
        
        with torch.no_grad():
            emb = speaker_model.encode_batch(torch.tensor(y))
            batch["speaker_embeddings"] = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
        return batch

    def get_collator(self):
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
        return TTSDataCollatorWithPadding(processor=self.processor)

# --- Placeholders para futuros Handlers ---
class F5CFMWrapper(torch.nn.Module):
    def __init__(self, cfm):
        super().__init__()
        self.cfm = cfm
        
    def forward(self, mel=None, text=None, **kwargs):
        # O CFM.forward retorna (loss, cond, pred)
        loss, _, _ = self.cfm(mel, text)
        return {"loss": loss}

class F5TTSHandler(ModelHandler):
    def load(self):
        try:
            from f5_tts.model import DiT, CFM
            from f5_tts.model.utils import get_tokenizer
        except ImportError:
            print("❌ Erro: Biblioteca 'f5-tts' não encontrada. Instale com 'pip install f5-tts'")
            sys.exit(1)

        print(f"📥 Carregando F5-TTS (DiT Backbone): {self.model_cfg['id']}")
        
        import importlib.resources
        import f5_tts
        try:
            vocab_path = str(importlib.resources.files("f5_tts").joinpath("infer/examples/vocab.txt"))
        except:
            vocab_path = os.path.join(os.path.dirname(f5_tts.__file__), "infer", "examples", "vocab.txt")
            
        self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_path, "custom")
        
        # DiT Architecture
        # Utilizamos self.vocab_size ao invés do valor embutido fixo (256)
        self.dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_num_embeds=self.vocab_size, mel_dim=80)
        
        # Aplicar LoRA no DiT
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none"
        )
        from peft import get_peft_model
        self.dit = get_peft_model(self.dit, peft_config)
        self.dit.to(self.device)
        
        # CFM Wrapper para HF Trainer
        cfm = CFM(transformer=self.dit, sigma=0.0).to(self.device)
        self.model = F5CFMWrapper(cfm)
        
        return self.model, self.vocab_char_map

    def prepare_item(self, batch, processor, speaker_model, language=None):
        # Preparação específica para o treino do CFM
        from f5_tts.model.utils import convert_char_to_pinyin
        audio = batch["audio"]
        y = np.array(audio["array"])
        
        # Extrair Mel-Spectrogram (80 bins conforme F5-TTS)
        mel = librosa.feature.melspectrogram(y=y, sr=24000, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = torch.tensor(np.log10(np.clip(mel, 1e-5, None)).T)
        
        batch["mel"] = log_mel
        
        text = batch.get("text", "")
        # Converter texto para pinyin e depois para IDs
        pinyin_list = convert_char_to_pinyin([text])[0]
        text_ids = [self.vocab_char_map.get(c, 0) for c in pinyin_list]
        batch["text_ids"] = torch.tensor(text_ids, dtype=torch.long)
        
        return batch

    def get_collator(self):
        @dataclass
        class F5DataCollator:
            def __call__(self, features):
                # Collation para o CFM (Padding de mel e text_ids)
                mels = [f["mel"] for f in features]
                texts = [f["text_ids"] for f in features]
                
                # Padding manual simples (truncagem ou pad conforme necessário)
                from torch.nn.utils.rnn import pad_sequence
                batch_mels = pad_sequence(mels, batch_first=True, padding_value=-5.0)
                batch_texts = pad_sequence(texts, batch_first=True, padding_value=0)
                
                return {"mel": batch_mels, "text": batch_texts}
        return F5DataCollator()

class XTTSWrapper(torch.nn.Module):
    def __init__(self, xtts):
        super().__init__()
        self.xtts = xtts
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, input_ids=None, text_len=None, audio_codes=None, wav_len=None, gpt_cond_latent=None, **kwargs):
        # O GPT do XTTS v2 preve os audio_codes (tokens auditivos)
        # cond_latents so os latents de voz (gpt_cond_latent)
        # Retorna logits: [batch, seq_len, vocab_size]
        logits = self.xtts.gpt(
            text_inputs=input_ids,
            text_lengths=text_len,
            audio_codes=audio_codes,
            wav_lengths=wav_len,
            cond_latents=gpt_cond_latent
        )
        
        # Shift para treino autoregressivo: prever o prximo token
        # Os labels so os prprios audio_codes
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = audio_codes[..., 1:].contiguous()
        
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}

class XTTSHandler(ModelHandler):
    def load(self):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv.")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2: {self.model_cfg['id']}")
        
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.utils.manage import ModelManager
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv.")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2 via ModelManager...")
        
        # O XTTS v2 precisa de pesos reais para inicializar o GPT
        manager = ModelManager()
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        res = manager.download_model(model_name)
        
        # O ModelManager pode retornar uma tupla (path, config, etc) ou apenas o path
        model_path = res[0] if isinstance(res, (list, tuple)) else res
        
        config_path = os.path.join(model_path, "config.json")
        config = XttsConfig()
        config.load_json(config_path)
        
        # Monkey-patch para contornar restrições de segurança do PyTorch 2.6+
        # O XTTS usa muitas classes customizadas no pickle que seriam bloqueadas pelo novo padrão weights_only=True
        orig_load = torch.load
        def trusted_load(*args, **kwargs):
            kwargs.pop("weights_only", None) # Remove se existir
            return orig_load(*args, weights_only=False, **kwargs)
        
        torch.load = trusted_load
        try:
            self.xtts = Xtts.init_from_config(config)
            self.xtts.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
        finally:
            torch.load = orig_load
        
        # Aplicar LoRA no GPT parte do XTTS
        if self.xtts.gpt is not None:
            lora_cfg = self.model_cfg['lora']
            peft_config = LoraConfig(
                r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
                target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
                bias="none",
                fan_in_fan_out=True # Necessário para as camadas Conv1D do GPT-2 do XTTS
            )
            # Aplicar LoRA especificamente ao sub-módulo GPT
            self.xtts.gpt = get_peft_model(self.xtts.gpt, peft_config)
            print(f"✅ LoRA aplicado ao GPT do XTTS v2.")
        
        self.xtts.to(self.device)
        self.model = XTTSWrapper(self.xtts)
        # O XTTS usa seu próprio tokenizer e processador de áudio incorporado
        return self.model, self.xtts.tokenizer

    def prepare_item(self, batch, processor, speaker_model, language="pt"):
        import tempfile
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 24000)
        text = normalize_text(batch.get("text", ""))
        
        # O XTTS v2 oficial da Coqui requer caminhos de arquivos para extrair condicionamento
        # Usamos arquivos temporários para compatibilidade total com a API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            temp_path = f.name
            
        try:
            # Extrair latents de condicionamento (voz)
            gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                audio_path=[temp_path] 
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Extrair Audio Codes (tokens VQ) do áudio original
        # Isso é o que o GPT tentará prever no treinamento
        audio_codes = self.xtts.gpt.encode_audio(torch.tensor(y).unsqueeze(0).to(self.device))
        
        input_ids = self.xtts.tokenizer.encode(text, lang=language)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "text_len": torch.tensor(len(input_ids)),
            "gpt_cond_latent": gpt_cond_latent.squeeze(0),
            "speaker_embedding": speaker_embedding.squeeze(0),
            "audio_codes": audio_codes.squeeze(0),
            "wav_len": torch.tensor(audio_codes.shape[-1]),
            "audio": torch.tensor(y)
        }

    def get_collator(self):
        @dataclass
        class XTTSDataCollator:
            def __call__(self, features):
                from torch.nn.utils.rnn import pad_sequence
                input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=0)
                text_len = torch.tensor([f["text_len"] for f in features])
                gpt_cond_latent = torch.stack([f["gpt_cond_latent"] for f in features])
                # Corrigido: audio_codes precisam de padding e ter o valor -1 para ignorar na perda
                audio_codes = pad_sequence([f["audio_codes"] for f in features], batch_first=True, padding_value=-1)
                wav_len = torch.tensor([f["wav_len"] for f in features])
                
                return {
                    "input_ids": input_ids,
                    "text_len": text_len,
                    "gpt_cond_latent": gpt_cond_latent,
                    "audio_codes": audio_codes,
                    "wav_len": wav_len
                }
        return XTTSDataCollator()

class GenericSOTAHandler(ModelHandler):
    def load(self):
        print(f"⚠️ O modelo {self.model_cfg['id']} requer implementações customizadas.")
        sys.exit(1)

def get_handler(model_type, model_cfg, device):
    if model_type == 'speecht5':
        return SpeechT5Handler(model_cfg, device)
    elif model_type == 'f5_tts':
        return F5TTSHandler(model_cfg, device)
    elif model_type in ['xtts_v2', 'your_tts']:
        return XTTSHandler(model_cfg, device)
    else:
        return GenericSOTAHandler(model_cfg, device)

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Treinamento TTS Generalizado con LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional)")
    parser.add_argument("--dataset", type=str, default=None, help="Perfil do dataset configurado no YAML")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo config.yaml")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    
    # 1. Identificar Perfis e Modelos
    dataset_name = args.dataset or full_cfg.get('settings', {}).get('default_dataset_profile', 'lapsbm')
    if dataset_name not in full_cfg['dataset_profiles']:
        print(f"❌ Perfil de dataset '{dataset_name}' não encontrado.")
        sys.exit(1)
        
    ds_cfg = full_cfg['dataset_profiles'][dataset_name]
    model_type = ds_cfg.get('model_type', 'speecht5')
    if model_type not in full_cfg['models']:
        print(f"❌ Modelo '{model_type}' não definido no YAML.")
        sys.exit(1)
    
    model_cfg = full_cfg['models'][model_type]
    
    # 2. Hardware
    hw_profile = args.profile or detect_hardware_profile(full_cfg)
    hw_cfg = full_cfg['hardware_profiles'].get(hw_profile, full_cfg['hardware_profiles']['cpu'])
    device = hw_cfg.get('device', 'cpu')
    
    if model_type == 'f5_tts' and device == 'cuda' and hw_cfg['batch_size'] > 2:
        multiplier = hw_cfg['batch_size'] // 2
        hw_cfg['gradient_accumulation_steps'] = hw_cfg.get('gradient_accumulation_steps', 1) * multiplier
        hw_cfg['batch_size'] = 2
        print(f"⚠️ Otimizando p/ F5-TTS: Batch size reduzido para 2 e Gradient Accumulation na GPU para {hw_cfg['gradient_accumulation_steps']}x para evitar Timeouts.")
    
    # 3. Output Dir
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(hw_cfg['output_dir'], f"{model_type}-{dataset_name}-{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 Iniciando Pipeline para: {model_type.upper()}")
    print(f"📂 Dataset: {dataset_name} | Hardware: {hw_profile.upper()}")
    
    start_time = time.time()

    # 4. Handler e Carregamento do Modelo
    handler = get_handler(model_type, model_cfg, device)
    model, processor = handler.load()

    # 5. Dataset Loading
    repo_name = ds_cfg['dataset_id'].split('/')[-1]
    local_ds_path = os.path.join(".", "meus_datasets", repo_name)
    is_local = os.path.exists(local_ds_path)
    
    print(f"📥 Carregando dataset '{ds_cfg['dataset_id']}' (Local: {is_local})...")
    
    all_data = []
    if ds_cfg['dataset_id'] == "firstpixel/pt-br_char":
        print("⚠️ Aplicando patch para o 'pt-br_char' (lendo repositório raw p/ contornar erro da plataforma)...")
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        try:
            if is_local:
                csv_path = os.path.join(local_ds_path, "metadata.csv")
            else:
                csv_path = hf_hub_download(repo_id=ds_cfg['dataset_id'], filename="metadata.csv", repo_type="dataset")
                
            df = pd.read_csv(csv_path, sep="|")
            
            for idx, row in df.iterrows():
                filename = row.iloc[0]
                text = row.iloc[1]
                
                if is_local:
                    audio_url = os.path.join(local_ds_path, filename.replace("/", os.sep))
                else:
                    audio_url = f"hf://datasets/{ds_cfg['dataset_id']}/{filename}"
                    
                # O HuggingFace Datasets permite instanciar áudios localmente ou via protocolo direto
                all_data.append({"audio": audio_url, "text": text})
                    
            full_ds = Dataset.from_list(all_data)
        except Exception as e:
            print(f"❌ Erro de permissão ou leitura! O dataset '{ds_cfg['dataset_id']}' pode estar fechado (Gated) online ou corrompido.\n{str(e)}")
            sys.exit(1)
    else:
        load_kwargs = {"split": ds_cfg['dataset_split']}
        if is_local:
            load_kwargs["path"] = local_ds_path
            load_kwargs["streaming"] = False
        else:
            load_kwargs["path"] = ds_cfg['dataset_id']
            load_kwargs["streaming"] = ds_cfg.get('streaming', True)
            if 'dataset_config' in ds_cfg: load_kwargs["name"] = ds_cfg['dataset_config']
            
        dataset_stream = load_dataset(**load_kwargs)
        
        for item in dataset_stream:
            if "wav" in item and "audio" not in item: item["audio"] = item["wav"]
            if "txt" in item and "text" not in item: item["text"] = item["txt"]
            all_data.append(item)
            
        full_ds = Dataset.from_list(all_data)
    
    # Seleção de Locutores e Amostragem Balanceada
    all_speakers = [extract_speaker_id(x) for x in all_data]
    counts = Counter(all_speakers)
    num_spk = ds_cfg.get('num_speakers', 1)
    max_samples = ds_cfg.get('num_samples_per_speaker', 0) # 0 para ilimitado
    
    if num_spk == 0:
        valid_spks = set(all_speakers)
    else:
        valid_spks = set(s for s, c in counts.most_common(num_spk))
        
    final_indices = []
    spk_added_counts = {s: 0 for s in valid_spks}
    
    for i, s in enumerate(all_speakers):
        if s in valid_spks:
            if max_samples == 0 or spk_added_counts[s] < max_samples:
                final_indices.append(i)
                spk_added_counts[s] += 1
                
    dataset = full_ds.select(final_indices)
    
    print("\n📊 --- Resumo do Dataset Balanceado ---")
    print(f"   👥 Locutores selecionados: {len(valid_spks)} (De um total de {len(counts)} na base)")
    samples_possible = sum(counts[s] for s in valid_spks)
    print(f"   📈 Total de Áudios (Samples) possíveis com os locutores selecionados: {samples_possible}")
    print(f"   🎙️ Total de Áudios (Samples) na fila de treino: {len(dataset)} ")
    if max_samples > 0:
        print(f"   ⚖️ Regra de Teto aplicada: Máximo de {max_samples} audios por locutor.")
    elif num_spk > 0:
        print(f"   ⚠️ Atenção: Nenhuma limitação de amostragem por voz foi definida (Treino Massivo).")
    print("----------------------------------------\n")
    
    # Configurar Sampling Rate
    target_sr = model_cfg.get('sampling_rate', 16000)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    # 6. Speaker Encoder & Processing
    print(f"🔊 Processando áudio em {target_sr} Hz...")
    
    speaker_model = None
    if 'speaker_encoder_id' in model_cfg:
        from speechbrain.inference.classifiers import EncoderClassifier
        speaker_model = EncoderClassifier.from_hparams(source=model_cfg['speaker_encoder_id'], run_opts={"device": device})
    
    language = ds_cfg.get("language", "pt")
    processed_ds = dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=dataset.column_names)
    processed_ds.set_format(type="torch")

    # 7. Training
    train_params = ds_cfg['training']
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=hw_cfg['batch_size'],
        gradient_accumulation_steps=hw_cfg['gradient_accumulation_steps'],
        learning_rate=train_params['learning_rate'], 
        num_train_epochs=train_params['num_epochs'],
        logging_steps=train_params['logging_steps'],
        save_steps=train_params['save_steps'], 
        weight_decay=train_params.get('weight_decay', 0.0),
        fp16=hw_cfg['fp16'],
        save_total_limit=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=training_args, 
        train_dataset=processed_ds, 
        data_collator=handler.get_collator()
    )
    
    print("\n🏁 Iniciando Treinamento...")
    trainer.train()
    
    # 8. Saving Final Model
    print("\n💾 Salvando modelo final...")
    try:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out_dir)
        else:
            trainer.save_model(out_dir)
            
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(out_dir)
            
        print(f"✅ Treinamento concluído com sucesso! Modelo final salvo em: {out_dir}")
    except Exception as e:
        print(f"⚠️ Atenção ao salvar modelo final: {e}")
        print("💡 Nota: Os checkpoints do treinamento (ex: checkpoint-3000) já foram salvos automaticamente pelo Trainer e podem ser utilizados!")

    # 9. Saving Cost-Benefit Metrics
    print("\n📝 Gerando README com parâmetros da fórmula Custo-Benefício...")
    end_time = time.time()
    training_latency = end_time - start_time
    
    final_loss = None
    if trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if 'loss' in log:
                final_loss = log['loss']
                break
                
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_allocated_gb = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    
    readme_appendix = f"""
---
## Parâmetros da Fórmula Custo-Benefício (Treinamento)

**Fórmula Original:**
`CustoBeneficio = [w1 * (1 - WER) + w2 * SimLocutor + w3 * Qualidade] / [CustoGPU$ + lambda * Latência]`

### 🔻 Custos (Denominador)
- **CustoGPU$ (Hardware + Uso VRAM)**: {gpu_name} (Pico VRAM: {vram_allocated_gb:.2f} GB)
- **Latência (Tempo de Treinamento)**: {training_latency:.2f} segundos ({training_latency/60:.2f} minutos)

### 🌟 Benefícios (Numerador - Proxies Atuais)
*Métricas como WER, Similaridade e MOS (Qualidade) requerem uma etapa de `inferência + validação` ao final do processo. Aqui usamos a Loss do treino como indicativo de melhora na capacidade do modelo.*
- **Loss Final do Treino**: {final_loss if final_loss is not None else 'N/A'}
"""
    readme_path = os.path.join(out_dir, "README.md")
    mode = "a" if os.path.exists(readme_path) else "w"
    try:
        with open(readme_path, mode, encoding="utf-8") as f:
            f.write(readme_appendix)
        print(f"📄 Parâmetros de Custo-Benefício adicionados ao README em: {readme_path}")
    except Exception as e:
        print(f"⚠️ Erro ao atualizar README: {e}")

if __name__ == "__main__":
    main()
