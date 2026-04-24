import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis de ambiente (HF_TOKEN)
load_dotenv()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Fix para PyTorch 2.6 e resume do Trainer
# O PyTorch 2.6 restringe load() de estados do numpy salvos pelo transformers.
# Como confiamos nos nossos próprios checkpoints, vamos permitir o carregamento.
_original_load = torch.load
def safe_load(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

import yaml
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

from datasets import load_dataset, Audio, Dataset, load_from_disk
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments, SpeechT5HifiGan, TrainerCallback
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from speechbrain.inference.speaker import EncoderClassifier

# Configuração de encoding para Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Funções Utilitárias ---
def normalize_text(text):
    if not text: return ""
    text = text.lower().replace('\n', ' ').strip()
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

    def load(self, resume_from=None):
        raise NotImplementedError()

    def prepare_item(self, item, processor, speaker_model):
        raise NotImplementedError()

    def get_collator(self):
        raise NotImplementedError()

class SpeechT5Handler(ModelHandler):
    def load(self, resume_from=None):
        print(f"📥 Carregando SpeechT5: {self.model_cfg['id']}")
        self.processor = SpeechT5Processor.from_pretrained(self.model_cfg['id'])
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_cfg['vocoder_id']).to(self.device)
        
        SpeechT5ForTextToSpeech._keys_to_ignore_on_load_unexpected = [r'.*encode_positions\.pe']
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_cfg['id'])
        
        # Aplicar LoRA ou Carregar existente
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none", task_type=None
        )
        
        if resume_from and os.path.exists(os.path.join(resume_from, "adapter_config.json")):
            print(f"🔄 Carregando adaptadores LoRA existentes de: {resume_from}")
            self.model = PeftModel.from_pretrained(self.model, resume_from, is_trainable=True)
        else:
            self.model = get_peft_model(self.model, peft_config)
        # Garantir contiguidade em parâmetros e BUFFERS (essencial p/ pos_encoder.pe)
        for p in self.model.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
        for b in self.model.buffers():
            if not b.is_contiguous(): b.data = b.data.contiguous()
            
        return self.model, processor

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
        for p in self.dit.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
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
            self.xtts.load_checkpoint(config, checkpoint_dir=model_path)
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
            for p in self.xtts.gpt.parameters():
                if not p.is_contiguous(): p.data = p.data.contiguous()
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

class FastSpeech2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Recuperar a função de loss oficial do Coqui para treinar Pitch/Duration/Energy
        base_model = getattr(self.model, "base_model", self.model)
        if hasattr(base_model, "model"): base_model = base_model.model
        self.criterion = base_model.get_criterion()
        
    def forward(self, x=None, x_lengths=None, y=None, y_lengths=None, **kwargs):
        pitch = kwargs.pop('pitch', None)
        
        # Cython MAS Aligner e algumas operações do Coqui quebram/geram NaN com FP16
        # Desabilitamos o autocast especificamente para o forward do modelo
        with torch.amp.autocast('cuda', enabled=False):
            if y is not None: y = y.float()
            if pitch is not None: pitch = pitch.float()
            
            outputs = self.model(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                pitch=pitch,
                **kwargs
            )
            
            if isinstance(outputs, dict) and "loss" in outputs:
                return outputs
                
            # Calcular a Loss Completa do FastPitch (inclui Mel, Duration, Pitch)
            # Isso é crucial para o LoRA aprender a entonação e ritmo em português!
            loss_dict = self.criterion(
                decoder_output=outputs["model_outputs"],
                decoder_target=y,
                decoder_output_lens=y_lengths,
                dur_output=outputs["durations_log"],
                dur_target=outputs.get("o_alignment_dur", None),
                pitch_output=outputs.get("pitch_avg", None),
                pitch_target=outputs.get("pitch_avg_gt", None),
                energy_output=outputs.get("energy_avg", None),
                energy_target=outputs.get("energy_avg_gt", None),
                input_lens=x_lengths,
                alignment_logprob=outputs.get("alignment_logprob", None),
                alignment_soft=outputs["alignment_soft"],
                alignment_hard=outputs["alignment_mas"],
                binary_loss_weight=0.0
            )
            
            # A loss total já vem calculada e agregada dentro do dict pela Coqui
            total_loss = loss_dict["loss"]
            
        return {"loss": total_loss, "model_outputs": outputs["model_outputs"]}

class FastSpeech2Handler(ModelHandler):
    def load(self, resume_from=None):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.forward_tts import ForwardTTS as FastSpeech2
            from TTS.tts.configs.fastspeech2_config import Fastspeech2Config as FastSpeech2Config
            from peft import get_peft_model, LoraConfig, PeftModel
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv_fs2.")
            sys.exit(1)

        print(f"📥 Carregando FastSpeech 2: {self.model_cfg['id']}")
        
        manager = ModelManager()
        model_path = manager.download_model(self.model_cfg['id'])
        if isinstance(model_path, tuple): model_path = model_path[0]
        
        # Se model_path for um arquivo (ex: model.pth), pegar o diretório pai
        if os.path.isfile(model_path):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path

        config_path = os.path.join(model_dir, "config.json")
        config = FastSpeech2Config()
        config.load_json(config_path)
        
        self.model_fs2 = FastSpeech2.init_from_config(config)
        self.model_fs2.load_checkpoint(config, checkpoint_path=model_path)
        
        # Aplicar LoRA ou Carregar existente
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none"
        )

        if resume_from and (os.path.exists(os.path.join(resume_from, "adapter_config.json")) or os.path.exists(os.path.join(resume_from, "model.safetensors"))):
            print(f"🔄 Carregando adaptadores LoRA existentes de: {resume_from}")
            # Se for um diretório de modelo final (com model.safetensors mas sem adapter_config.json pela estrutura do wrapper)
            # precisamos inicializar o LoRA primeiro e depois carregar os pesos
            self.model_fs2 = get_peft_model(self.model_fs2, peft_config)
            
            # Tentar carregar pesos (lidando com o prefixo 'model.' se necessário)
            weights_path = os.path.join(resume_from, "model.safetensors")
            if not os.path.exists(weights_path): weights_path = os.path.join(resume_from, "pytorch_model.bin")
            
            if os.path.exists(weights_path):
                if weights_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    sd = load_file(weights_path)
                else:
                    sd = torch.load(weights_path, map_location="cpu")
                
                # Ajustar prefixos do wrapper
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("model."): new_sd[k[6:]] = v
                    else: new_sd[k] = v
                self.model_fs2.load_state_dict(new_sd, strict=False)
        else:
            self.model_fs2 = get_peft_model(self.model_fs2, peft_config)
        
        # Fix contiguidade (Parâmetros e Buffers)
        for p in self.model_fs2.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
        for b in self.model_fs2.buffers():
            if not b.is_contiguous(): b.data = b.data.contiguous()
            
        print(f"✅ LoRA aplicado ao FastSpeech 2.")
        
        self.model = FastSpeech2Wrapper(self.model_fs2)
        
        # Substituir o fonemizador original pelo de português (Gruut)
        try:
            from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name
            base_phonemizer = get_phonemizer_by_name("gruut", language="pt")
            
            # Criamos um wrapper para mapear fonemas do PT que não existem no modelo Base (Inglês)
            class PTPhonemizerWrapper:
                def __init__(self, p): self.p = p
                def phonemize(self, text, separator="", language="pt"):
                    ph = self.p.phonemize(text, separator=separator, language=language)
                    replacements = {'ẽ': 'e', 'ĩ': 'i', 'õ': 'o', 'ũ': 'u', 'ã': 'a', '\u0303': '', 'g': 'ɡ'}
                    for k, v in replacements.items(): ph = ph.replace(k, v)
                    return ph
                def name(self): return "pt_wrapper"
                def print_logs(self, level): pass
                
            new_phonemizer = PTPhonemizerWrapper(base_phonemizer)
            self.model_fs2.tokenizer.phonemizer = new_phonemizer
            print(f"🌐 Phonemizer do FastSpeech 2 substituído por Gruut (pt) com mapeamento cross-lingual")
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível trocar o phonemizer: {e}")
            
        self.model_fs2.to(self.device)
        return self.model, self.model_fs2.tokenizer

    def prepare_item(self, batch, processor, speaker_model, language="en"):
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 22050)
        text = normalize_text(batch.get("text", ""))
        
        # Tokenização (FS2 usa fonemas internamente via tokenizer)
        input_ids = processor.text_to_ids(text, language=language)
        
        # Extração de Mel
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80, fmin=0, fmax=sr//2
        )
        log_mel = np.log10(np.clip(mel_spec, 1e-5, None))
        
        # FastSpeech 2 / FastPitch requer pitch (F0) real durante o treino
        pitch, _, _ = librosa.pyin(y, fmin=65, fmax=2000, sr=sr, frame_length=1024, hop_length=256)
        pitch = np.nan_to_num(pitch)
        if len(pitch) > log_mel.shape[1]: pitch = pitch[:log_mel.shape[1]]
        elif len(pitch) < log_mel.shape[1]: pitch = np.pad(pitch, (0, log_mel.shape[1] - len(pitch)))
        
        return {
            "x": torch.tensor(input_ids),
            "x_lengths": torch.tensor(len(input_ids)),
            "y": torch.tensor(log_mel.T), # Coqui espera [B, T, C]
            "y_lengths": torch.tensor(log_mel.shape[1]),
            "pitch": torch.tensor(pitch).unsqueeze(0).float() # [1, T]
        }

    def get_collator(self):
        @dataclass
        class FS2DataCollator:
            def __call__(self, features):
                from torch.nn.utils.rnn import pad_sequence
                x = pad_sequence([f["x"] for f in features], batch_first=True)
                y = pad_sequence([f["y"] for f in features], batch_first=True)
                pitch = pad_sequence([f["pitch"].squeeze(0) for f in features], batch_first=True).unsqueeze(1)
                x_lengths = torch.stack([f["x_lengths"] for f in features])
                y_lengths = torch.stack([f["y_lengths"] for f in features])
                
                return {
                    "x": x,
                    "x_lengths": x_lengths,
                    "y": y,
                    "y_lengths": y_lengths,
                    "pitch": pitch
                }
        return FS2DataCollator()

class MatchaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x=None, x_lengths=None, y=None, y_lengths=None, **kwargs):
        # Matcha-TTS (Lightning module) retorna loss no método forward ou compute_loss
        # x: input_ids, y: mel
        outputs = self.model(
            x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, **kwargs
        )
        return {"loss": outputs["loss"]}

class MatchaHandler(ModelHandler):
    def load(self):
        try:
            from matcha.models.matcha_tts import MatchaTTS
        except ImportError:
            print("❌ Erro: Biblioteca 'matcha-tts' não encontrada no .venv_matcha.")
            sys.exit(1)

        print(f"📥 Carregando Matcha-TTS: {self.model_cfg['id']}")
        
        # O Matcha-TTS costuma ser carregado via checkpoint (.ckpt)
        # Para simplificar o LoRA, assumimos que o modelo base está disponível ou baixado
        # Ex: self.model_matcha = MatchaTTS.load_from_checkpoint(checkpoint_path)
        # Aqui inicializamos um modelo base para aplicar o LoRA
        # Nota: Uma implementação real precisaria do config exato.
        self.model_matcha = MatchaTTS(...) # Placeholder para inicialização real
        
        # Aplicar LoRA no Decoder (DiT/Transformer) do Matcha
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none"
        )
        self.model_matcha.decoder = get_peft_model(self.model_matcha.decoder, peft_config)
        for p in self.model_matcha.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
            
        self.model_matcha.to(self.device)
        self.model = MatchaWrapper(self.model_matcha)
        return self.model, None # Matcha usa phonemizer diretamente no prepare

    def prepare_item(self, batch, processor, speaker_model, language="en"):
        # Preparação específica para o Flow Matching do Matcha
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 22050)
        text = normalize_text(batch.get("text", ""))
        
        # Matcha usa phonemizer externo
        from matcha.utils.cleaners import english_cleaners
        clean_text = english_cleaners(text)
        
        # Extração de Mel
        import librosa
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = np.log10(np.clip(mel, 1e-5, None)).T
        
        return {
            "x": torch.tensor([0]), # IDs do texto (Placeholder)
            "y": torch.tensor(log_mel),
            "x_lengths": torch.tensor(1),
            "y_lengths": torch.tensor(log_mel.shape[0])
        }

    def get_collator(self):
        @dataclass
        class MatchaDataCollator:
            def __call__(self, features):
                from torch.nn.utils.rnn import pad_sequence
                x = pad_sequence([f["x"] for f in features], batch_first=True)
                y = pad_sequence([f["y"] for f in features], batch_first=True, padding_value=-5.0)
                return {"x": x, "y": y, "x_lengths": torch.tensor([f["x_lengths"] for f in features]), "y_lengths": torch.tensor([f["y_lengths"] for f in features])}
        return MatchaDataCollator()

class GlowTTSHandler(FastSpeech2Handler):
    # Glow-TTS compartilha muita lógica com o FastSpeech 2 na Coqui
    def load(self):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.glow_tts import GlowTTS
            from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv_fs2.")
            sys.exit(1)

        print(f"📥 Carregando Glow-TTS: {self.model_cfg['id']}")
        manager = ModelManager()
        model_path = manager.download_model(self.model_cfg['id'])
        if isinstance(model_path, tuple): model_path = model_path[0]
        
        config = GlowTTSConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model_glow = GlowTTS.init_from_config(config)
        self.model_glow.load_checkpoint(config, checkpoint_dir=model_path)
        
        # LoRA no Encoder do Glow-TTS
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(target_modules=lora_cfg['target_modules'], r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'])
        self.model_glow.encoder = get_peft_model(self.model_glow.encoder, peft_config)
        for p in self.model_glow.parameters():
            if not p.is_contiguous(): p.data = p.data.contiguous()
            
        self.model_glow.to(self.device)
        self.model = FastSpeech2Wrapper(self.model_glow) # Reutiliza wrapper de loss
        return self.model, self.model_glow.tokenizer

class GenericSOTAHandler(ModelHandler):
    def load(self):
        print(f"⚠️ O modelo {self.model_cfg['id']} requer implementações customizadas.")
        sys.exit(1)

def get_handler(model_type, model_cfg, device):
    if model_type == 'speecht5':
        return SpeechT5Handler(model_cfg, device)
    elif model_type == 'f5_tts':
        return F5TTSHandler(model_cfg, device)
    elif model_type == 'fastspeech2':
        return FastSpeech2Handler(model_cfg, device)
    elif model_type == 'matcha':
        return MatchaHandler(model_cfg, device)
    elif model_type == 'glow_tts':
        return GlowTTSHandler(model_cfg, device)
    elif model_type in ['xtts_v2', 'your_tts']:
        return XTTSHandler(model_cfg, device)
    else:
        return GenericSOTAHandler(model_cfg, device)

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

class ValidationCallback(TrainerCallback):
    """Callback para mostrar detalhes extras da validação no terminal."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n✨ [VALIDAÇÃO] Época: {state.epoch:.2f} | Passo: {state.global_step}")
            for key, value in metrics.items():
                name = key.replace("eval_", "📊 ").replace("_", " ").title()
                if isinstance(value, float):
                    print(f"   {name}: {value:.4f}")
                else:
                    print(f"   {name}: {value}")
            print("--------------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Treinamento TTS Generalizado con LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional)")
    parser.add_argument("--dataset", type=str, default=None, help="Perfil do dataset configurado no YAML")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo config.yaml")
    parser.add_argument("--resume_from", type=str, default=None, help="Caminho para checkpoint ou modelo para continuar treinamento")
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
    model, processor = handler.load(resume_from=args.resume_from)

    # 5. Dataset Loading
    local_root = full_cfg.get('settings', {}).get('local_datasets_dir', './datasets')
    repo_name = ds_cfg['dataset_id'].split('/')[-1]
    local_ds_path = os.path.join(local_root, repo_name)
    
    # Se o diretório existir e contiver arquivos, consideramos local
    is_local = os.path.exists(local_ds_path) and len(os.listdir(local_ds_path)) > 0 if os.path.exists(local_ds_path) else False
    
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
            print(f"📂 Carregando dataset local de: {local_ds_path}")
            dataset_stream = load_from_disk(local_ds_path)
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
        
        # Salvar localmente se não for local para evitar downloads futuros
        if not is_local:
            print(f"💾 Salvando dataset localmente em: {local_ds_path}")
            os.makedirs(os.path.dirname(local_ds_path), exist_ok=True)
            full_ds.save_to_disk(local_ds_path)
    
    # Seleção de Locutores e Amostragem Balanceada
    all_speakers = [extract_speaker_id(x) for x in all_data]
    counts = Counter(all_speakers)
    num_spk = ds_cfg.get('num_speakers', 1)
    max_samples = ds_cfg.get('num_samples_per_speaker', 0) # 0 para ilimitado
    zero_shot_split = ds_cfg.get('zero_shot_split', None)
    
    if num_spk == 0:
        valid_spks = set(all_speakers)
    else:
        valid_spks = set(s for s, c in counts.most_common(num_spk))
        
    train_spks, val_spks, test_spks = valid_spks, set(), set()
    
    if zero_shot_split and len(valid_spks) > 0:
        t_spk = zero_shot_split.get('train_speakers', 28)
        v_spk = zero_shot_split.get('val_speakers', 3)
        ts_spk = zero_shot_split.get('test_speakers', 4)
        
        sorted_spks = sorted(list(valid_spks))
        if len(sorted_spks) >= (t_spk + v_spk + ts_spk):
            train_spks = set(sorted_spks[:t_spk])
            val_spks = set(sorted_spks[t_spk:t_spk+v_spk])
            test_spks = set(sorted_spks[t_spk+v_spk:t_spk+v_spk+ts_spk])
            valid_spks = train_spks | val_spks | test_spks
        else:
            print(f"⚠️ Atenção: Não há locutores suficientes para a divisão zero-shot ({len(sorted_spks)} < {t_spk+v_spk+ts_spk}). Usando todos para treino.")

    train_indices = []
    val_indices = []
    test_indices = []
    spk_added_counts = {s: 0 for s in valid_spks}
    
    for i, (item, s) in enumerate(zip(all_data, all_speakers)):
        if s in valid_spks:
            if max_samples == 0 or spk_added_counts[s] < max_samples:
                if s in val_spks:
                    val_indices.append(i)
                elif s in test_spks:
                    test_indices.append(i)
                else: # train
                    train_indices.append(i)
                spk_added_counts[s] += 1
                
    train_dataset = full_ds.select(train_indices)
    val_dataset = full_ds.select(val_indices) if val_indices else None
    test_dataset = full_ds.select(test_indices) if test_indices else None
    
    print("\n📊 --- Resumo do Dataset (Divisão 80/10/10 por Locutor) ---")
    print(f"   👥 Locutores: Total({len(counts)}) | Selecionados({len(valid_spks)})")
    print(f"   🔀 Divisão por Locutor: Treino({len(train_spks)}) | Validação({len(val_spks)}) | Teste({len(test_spks)})")
    print(f"   🎙️ Amostras: Treino({len(train_dataset)}) | Validação({len(val_dataset) if val_dataset else 0}) | Teste({len(test_dataset) if test_dataset else 0})")
    print(f"   ⚖️ Teto: {max_samples if max_samples > 0 else 'Ilimitado'} audios/locutor")
    print("----------------------------------------------------------\n")
    
    # Exportar dataset_split.json
    import json
    split_info = {
        "train_speakers": list(train_spks),
        "val_speakers": list(val_spks),
        "test_speakers": list(test_spks),
        "counts": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "test_samples": len(test_dataset) if test_dataset else 0
        }
    }
    with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)
        
    # Configurar Sampling Rate
    target_sr = model_cfg.get('sampling_rate', 16000)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=target_sr))
    if val_dataset:
        val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    # 6. Speaker Encoder & Processing
    print(f"🔊 Processando áudio em {target_sr} Hz...")
    
    speaker_model = None
    if 'speaker_encoder_id' in model_cfg:
        from speechbrain.inference.classifiers import EncoderClassifier
        speaker_model = EncoderClassifier.from_hparams(source=model_cfg['speaker_encoder_id'], run_opts={"device": device})
    
    language = ds_cfg.get("language", "pt")
    
    # Sistema de Cache Hard (Disco)
    cache_base_dir = os.path.join(full_cfg.get('settings', {}).get('local_datasets_dir', './datasets'), "cache_processado", f"{model_type}_{dataset_name}_{target_sr}hz")
    train_cache_path = os.path.join(cache_base_dir, "train")
    val_cache_path = os.path.join(cache_base_dir, "val")
    
    if os.path.exists(train_cache_path):
        print(f"⚡ Carregando dataset TRAIN processado do cache: {train_cache_path}")
        processed_train_ds = load_from_disk(train_cache_path)
    else:
        processed_train_ds = train_dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=train_dataset.column_names)
        processed_train_ds.set_format(type="torch")
        processed_train_ds.save_to_disk(train_cache_path)
        print(f"💾 Cache TRAIN salvo em: {train_cache_path}")
    
    processed_val_ds = None
    if val_dataset:
        if os.path.exists(val_cache_path):
            print(f"⚡ Carregando dataset VAL processado do cache: {val_cache_path}")
            processed_val_ds = load_from_disk(val_cache_path)
        else:
            processed_val_ds = val_dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=val_dataset.column_names)
            processed_val_ds.set_format(type="torch")
            processed_val_ds.save_to_disk(val_cache_path)
            print(f"💾 Cache VAL salvo em: {val_cache_path}")

    # 7. Training
    train_params = ds_cfg['training']
    
    training_kwargs = {
        "output_dir": out_dir,
        "per_device_train_batch_size": hw_cfg['batch_size'],
        "gradient_accumulation_steps": hw_cfg['gradient_accumulation_steps'],
        "learning_rate": train_params['learning_rate'], 
        "num_train_epochs": train_params['num_epochs'],
        "logging_steps": train_params['logging_steps'],
        "save_strategy": "steps",
        "save_steps": train_params['save_steps'], # Salvando a cada 10 épocas (aprox. 1750 passos com batch 4)
        "save_total_limit": 1,
        "weight_decay": train_params.get('weight_decay', 0.0),
        "fp16": hw_cfg['fp16'],
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": "none"
    }

    if processed_val_ds:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = train_params['logging_steps']
        
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model, 
        "args": training_args, 
        "train_dataset": processed_train_ds, 
        "eval_dataset": processed_val_ds, # Habilitando validação
        "data_collator": handler.get_collator(),
        "callbacks": [ValidationCallback()] # Adicionando callback customizado
    }
    
    if processed_val_ds:
        trainer_kwargs["eval_dataset"] = processed_val_ds

    trainer = Trainer(**trainer_kwargs)
    
    print("\n🏁 Iniciando Treinamento...")
    if args.resume_from:
        print(f"🔄 Retomando treinamento a partir de: {args.resume_from}")
        
        # Patch para compatibilidade de versão do TrainerState
        # Remove chaves obsoletas como 'best_global_step' que causam erro no transformers 4.44+
        state_path = os.path.join(args.resume_from, "trainer_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                
                dirty = False
                for key in ["best_global_step"]:
                    if key in state_data:
                        del state_data[key]
                        dirty = True
                
                if dirty:
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state_data, f, indent=2)
                    print(f"   🩹 Patch aplicado ao trainer_state.json (removidas chaves obsoletas).")
            except Exception as e:
                print(f"   ⚠️ Aviso ao aplicar patch no state: {e}")
    
    trainer.train(resume_from_checkpoint=args.resume_from)
    
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
