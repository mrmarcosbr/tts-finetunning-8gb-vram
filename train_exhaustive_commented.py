import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env (usado principalmente para o HF_TOKEN)
load_dotenv()
import yaml
import torch
import torchaudio
import unicodedata
import argparse
import numpy as np
import librosa
import soundfile as sf

# Carrega as variáveis de ambiente novamente por precaução (importante para autenticação no HuggingFace Hub)
load_dotenv()
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from collections import Counter

# Importa bibliotecas essenciais para processamento de datasets e modelos de IA
from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments, SpeechT5HifiGan
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from speechbrain.inference.speaker import EncoderClassifier

# Garante que a saída do terminal no Windows suporte caracteres especiais (UTF-8)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Funções Utilitárias ---

def normalize_text(text):
    """Remove acentos e normaliza o texto para evitar erros no processador de texto do TTS."""
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def extract_speaker_id(item):
    """Tenta identificar o ID do locutor a partir dos metadados do dataset ou do nome do arquivo de áudio."""
    # Busca em chaves comuns de IDs de locutor
    for k in ['speaker_id', 'speaker', 'user_id', 'client_id']:
        if k in item and item[k] is not None:
            return str(item[k])
            
    # Caso não encontre nas chaves, tenta extrair do caminho da URL ou do caminho local do arquivo
    possible_paths = [
        item.get("__url__", ""),
        item.get("audio", {}).get("path", "") if isinstance(item.get("audio"), dict) else "",
        item.get("audio", "") if isinstance(item.get("audio"), str) else ""
    ]
    
    for p in possible_paths:
        if p and "LapsBM-" in str(p):
            # Lógica específica para o dataset LapsBM: extrai 'M-001' de 'LapsBM-M-001.wav'
            filename = os.path.basename(str(p))
            if filename.startswith("LapsBM-"):
                return filename.split(".")[0].replace("LapsBM-", "")
                
    return "unknown" # Retorna 'unknown' se não conseguir identificar o locutor

def load_config(config_path="config.yaml"):
    """Lê o arquivo de configuração YAML que contém perfis de hardware e datasets."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_hardware_profile(full_cfg):
    """Identifica automaticamente se há GPU (NVIDIA, Apple Silicon) ou se deve usar CPU."""
    print("🔍 Autodetectando hardware...")
    
    if torch.cuda.is_available():
        # Se houver CUDA (NVIDIA), verifica quanta VRAM está disponível para escolher o perfil ideal
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   🖥️  GPU NVIDIA Detectada: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB VRAM)")
        if total_vram >= 14.0: return "cuda_16gb" # Perfil para GPUs como RTX 3090/4090/5060 Ti 16GB
        return "cuda_8gb" # Perfil para GPUs como RTX 3060/3070/4060
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Suporte para processadores M1/M2/M3 da Apple
        print("   🍎 Apple Silicon Detectado (MPS Backend)")
        return "macbook"
    
    print("   ⚠️ Nenhuma GPU de alta performance detectada. Usando CPU.")
    return "cpu" # Fallback para processamento em processador comum

# ==============================================================================
# MODEL HANDLERS (Gerenciadores de Arquitetura)
# ==============================================================================

class ModelHandler:
    """Classe base (interface) para gerenciar diferentes modelos de TTS (SpeechT5, F5, XTTS)."""
    def __init__(self, model_cfg: Dict, device: str):
        self.model_cfg = model_cfg # Configurações vindas do YAML
        self.device = device       # 'cuda', 'cpu' ou 'mps'
        self.processor = None      # Tokenizer/Processador de áudio
        self.model = None          # O modelo de IA carregado

    def load(self):
        """Método para carregar os pesos e configurar LoRA."""
        raise NotImplementedError()

    def prepare_item(self, item, processor, speaker_model):
        """Transforma um exemplo bruto do dataset em dados que o modelo entende."""
        raise NotImplementedError()

    def get_collator(self):
        """Retorna o objeto que agrupa múltiplos exemplos em um 'batch' (lote) para treino."""
        raise NotImplementedError()

class SpeechT5Handler(ModelHandler):
    """Implementação específica para o modelo SpeechT5 da Microsoft."""
    def load(self):
        print(f"📥 Carregando SpeechT5: {self.model_cfg['id']}")
        # Carrega o processador de texto/áudio e o vocoder (que transforma espectrograma em som)
        self.processor = SpeechT5Processor.from_pretrained(self.model_cfg['id'])
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_cfg['vocoder_id']).to(self.device)
        
        # Ignora avisos sobre camadas de posição que não são usadas no fine-tuning
        SpeechT5ForTextToSpeech._keys_to_ignore_on_load_unexpected = [r'.*encode_positions\.pe']
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_cfg['id'])
        
        # --- Configuração do LoRA (Low-Rank Adaptation) ---
        # LoRA permite treinar apenas uma fração minúscula dos parâmetros, economizando VRAM.
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], # Rank da adaptação
            lora_alpha=lora_cfg['alpha'], # Fator de escala
            target_modules=lora_cfg['target_modules'], # Em quais camadas injetar LoRA
            lora_dropout=lora_cfg['dropout'], 
            bias="none", 
            task_type=None
        )
        self.model = get_peft_model(self.model, peft_config)
        return self.model, self.processor

    def prepare_item(self, batch, processor, speaker_model, language=None):
        """Processa um único áudio e texto para o formato SpeechT5."""
        audio = batch["audio"]
        clean_text = normalize_text(batch.get("text", "dummy"))
        
        # Transforma texto em tokens (IDs numéricos)
        batch["input_ids"] = processor(text=clean_text, return_tensors="pt").input_ids[0]
        y = np.array(audio["array"]) # Converte áudio bruto em array numpy
        
        # SpeechT5 treina tentando prever o Mel-Spectrogram (uma representação visual do som)
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=80, fmin=80, fmax=7600)
        log_mel = np.log10(np.clip(mel, 1e-5, None)).T # Transforma para escala logarítmica
        batch["labels"] = torch.tensor(log_mel)
        
        # Extrai o Speaker Embedding (a 'impressão digital' da voz) usando um modelo pré-treinado
        with torch.no_grad():
            emb = speaker_model.encode_batch(torch.tensor(y))
            # Normaliza e salva para que o modelo aprenda a associar o texto a essa voz específica
            batch["speaker_embeddings"] = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
        return batch

    def get_collator(self):
        """Define como o modelo deve agrupar amostras de tamanhos diferentes no mesmo lote."""
        @dataclass
        class TTSDataCollatorWithPadding:
            processor: Any
            def __call__(self, features):
                # Extrai campos individuais
                input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
                labels = [feature["labels"] for feature in features]
                speaker_features = [feature["speaker_embeddings"] for feature in features]
                
                # Preenche com zeros (padding) os textos para ficarem todos com o mesmo tamanho no lote
                batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")
                
                # Preenche os espectrogramas (labels) com valor de silêncio (-5.0)
                max_len = max([l.shape[0] for l in labels])
                if max_len % 2 != 0: max_len += 1 # SpeechT5 requer comprimento par
                padded_labels = [torch.nn.functional.pad(l, (0, 0, 0, max_len - l.shape[0]), value=-5.0) for l in labels]
                
                batch["labels"] = torch.stack(padded_labels)
                batch["speaker_embeddings"] = torch.stack([s.clone().detach() if isinstance(s, torch.Tensor) else torch.tensor(s) for s in speaker_features])
                return batch
        return TTSDataCollatorWithPadding(processor=self.processor)

# --- Classes auxiliares para F5-TTS (Flow-based) ---
class F5CFMWrapper(torch.nn.Module):
    """Wrapper para permitir que o modelo F5-TTS seja treinado via HuggingFace Trainer."""
    def __init__(self, cfm):
        super().__init__()
        self.cfm = cfm
        
    def forward(self, mel=None, text=None, **kwargs):
        # O modelo F5 usa CFM (Conditional Flow Matching). Aqui calculamos a perda (loss).
        loss, _, _ = self.cfm(mel, text)
        return {"loss": loss}

class F5TTSHandler(ModelHandler):
    """Implementação para o F5-TTS, um modelo SOTA baseado em Diffusion/Flow Matching."""
    def load(self):
        # Tenta carregar as bibliotecas específicas do F5-TTS
        try:
            from f5_tts.model import DiT, CFM
            from f5_tts.model.utils import get_tokenizer
        except ImportError:
            print("❌ Erro: Biblioteca 'f5-tts' não encontrada. Instale com 'pip install f5-tts'")
            sys.exit(1)

        print(f"📥 Carregando F5-TTS (DiT Backbone): {self.model_cfg['id']}")
        
        # Localiza o arquivo de vocabulário (mapeamento de letras para números)
        import importlib.resources
        import f5_tts
        try:
            vocab_path = str(importlib.resources.files("f5_tts").joinpath("infer/examples/vocab.txt"))
        except:
            vocab_path = os.path.join(os.path.dirname(f5_tts.__file__), "infer", "examples", "vocab.txt")
            
        self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_path, "custom")
        
        # Inicializa a arquitetura DiT (Diffusion Transformer)
        self.dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_num_embeds=self.vocab_size, mel_dim=80)
        
        # Aplica LoRA no DiT para permitir fine-tuning em GPUs com pouca VRAM
        lora_cfg = self.model_cfg['lora']
        peft_config = LoraConfig(
            r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
            target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
            bias="none"
        )
        from peft import get_peft_model
        self.dit = get_peft_model(self.dit, peft_config)
        self.dit.to(self.device)
        
        # Envolve o modelo no wrapper de treino
        cfm = CFM(transformer=self.dit, sigma=0.0).to(self.device)
        self.model = F5CFMWrapper(cfm)
        
        return self.model, self.vocab_char_map

    def prepare_item(self, batch, processor, speaker_model, language=None):
        """Prepara áudio e texto (convertendo para pinyin se necessário) para o F5-TTS."""
        from f5_tts.model.utils import convert_char_to_pinyin
        audio = batch["audio"]
        y = np.array(audio["array"])
        
        # F5-TTS usa 24kHz e espectrograma de 80 bandas
        mel = librosa.feature.melspectrogram(y=y, sr=24000, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = torch.tensor(np.log10(np.clip(mel, 1e-5, None)).T)
        batch["mel"] = log_mel
        
        text = batch.get("text", "")
        # Converte caracteres em sons (pinyin) para melhor prosódia
        pinyin_list = convert_char_to_pinyin([text])[0]
        text_ids = [self.vocab_char_map.get(c, 0) for c in pinyin_list]
        batch["text_ids"] = torch.tensor(text_ids, dtype=torch.long)
        
        return batch

    def get_collator(self):
        """Agrupador de dados para F5-TTS."""
        @dataclass
        class F5DataCollator:
            def __call__(self, features):
                mels = [f["mel"] for f in features]
                texts = [f["text_ids"] for f in features]
                
                # Faz o padding automático das sequências de áudio e texto
                from torch.nn.utils.rnn import pad_sequence
                batch_mels = pad_sequence(mels, batch_first=True, padding_value=-5.0)
                batch_texts = pad_sequence(texts, batch_first=True, padding_value=0)
                
                return {"mel": batch_mels, "text": batch_texts}
        return F5DataCollator()

# --- Classes auxiliares para XTTS v2 (Autoregressivo) ---
class XTTSWrapper(torch.nn.Module):
    """Wrapper para o XTTS v2 da Coqui, focado em prever 'audio codes' (tokens de som)."""
    def __init__(self, xtts):
        super().__init__()
        self.xtts = xtts
        # Usa perda de entropia cruzada (CrossEntropy) para prever o próximo token sonoro
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, input_ids=None, text_len=None, audio_codes=None, wav_len=None, gpt_cond_latent=None, **kwargs):
        # Executa o modelo GPT interno do XTTS
        logits = self.xtts.gpt(
            text_inputs=input_ids,
            text_lengths=text_len,
            audio_codes=audio_codes,
            wav_lengths=wav_len,
            cond_latents=gpt_cond_latent
        )
        
        # No treino autoregressivo, tentamos prever o token 'i+1' a partir do token 'i'
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = audio_codes[..., 1:].contiguous()
        
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}

class XTTSHandler(ModelHandler):
    """Implementação para o XTTS v2, famoso por clonagem de voz instantânea (Zero-shot)."""
    def load(self):
        # Carrega dependências do pacote TTS da Coqui
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.utils.manage import ModelManager
        except ImportError:
            print("❌ Erro: Biblioteca 'TTS' (Coqui) não encontrada no .venv.")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2 via ModelManager...")
        
        # Baixa automaticamente os pesos base do XTTS v2 se não existirem
        manager = ModelManager()
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        res = manager.download_model(model_name)
        model_path = res[0] if isinstance(res, (list, tuple)) else res
        
        config_path = os.path.join(model_path, "config.json")
        config = XttsConfig()
        config.load_json(config_path)
        
        # Patch para contornar restrições de segurança do PyTorch 2.6+ ao carregar modelos legados
        orig_load = torch.load
        def trusted_load(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return orig_load(*args, weights_only=False, **kwargs)
        
        torch.load = trusted_load
        try:
            # Inicializa o modelo a partir do arquivo de configuração
            self.xtts = Xtts.init_from_config(config)
            self.xtts.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
        finally:
            torch.load = orig_load # Restaura a função original
        
        # Injeta LoRA no módulo GPT do XTTS
        if self.xtts.gpt is not None:
            lora_cfg = self.model_cfg['lora']
            peft_config = LoraConfig(
                r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
                target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
                bias="none",
                fan_in_fan_out=True # Necessário pois o GPT do XTTS usa implementações específicas de Conv1D
            )
            self.xtts.gpt = get_peft_model(self.xtts.gpt, peft_config)
            print(f"✅ LoRA aplicado ao GPT do XTTS v2.")
        
        self.xtts.to(self.device)
        self.model = XTTSWrapper(self.xtts)
        return self.model, self.xtts.tokenizer

    def prepare_item(self, batch, processor, speaker_model, language="pt"):
        """Processamento complexo que extrai tanto tokens auditivos quanto latentes de voz."""
        import tempfile
        audio = batch["audio"]
        y = np.array(audio["array"])
        sr = audio.get("sampling_rate", 24000)
        text = normalize_text(batch.get("text", ""))
        
        # O XTTS requer um arquivo temporário no disco para processar o condicionamento da voz
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            temp_path = f.name
            
        try:
            # Extrai as características únicas da voz para a clonagem (Zero-shot Latents)
            gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                audio_path=[temp_path] 
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path) # Limpa o arquivo temporário
        
        # Converte o áudio bruto em 'audio codes' (tokens discretos) que o GPT aprende a prever
        audio_codes = self.xtts.gpt.encode_audio(torch.tensor(y).unsqueeze(0).to(self.device))
        
        # Transforma o texto em IDs numéricos
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
        """Agrupador de dados para XTTS v2."""
        @dataclass
        class XTTSDataCollator:
            def __call__(self, features):
                from torch.nn.utils.rnn import pad_sequence
                # Agrupa e faz o padding de todos os campos necessários para o treino autoregressivo
                input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=0)
                text_len = torch.tensor([f["text_len"] for f in features])
                gpt_cond_latent = torch.stack([f["gpt_cond_latent"] for f in features])
                # Usa -1 como padding para que a função de perda ignore esses espaços no cálculo
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

def get_handler(model_type, model_cfg, device):
    """Fábrica de Handlers: seleciona a classe correta com base no tipo de modelo."""
    if model_type == 'speecht5':
        return SpeechT5Handler(model_cfg, device)
    elif model_type == 'f5_tts':
        return F5TTSHandler(model_cfg, device)
    elif model_type in ['xtts_v2', 'your_tts']:
        return XTTSHandler(model_cfg, device)
    else:
        # Se for um modelo novo não implementado, encerra com aviso
        print(f"⚠️ O modelo {model_type} requer implementações customizadas.")
        sys.exit(1)

# ==============================================================================
# LOGICA PRINCIPAL (MAIN)
# ==============================================================================

def main():
    # Define argumentos que podem ser passados via linha de comando (CLI)
    parser = argparse.ArgumentParser(description="Treinamento TTS Generalizado con LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional)")
    parser.add_argument("--dataset", type=str, default=None, help="Perfil do dataset configurado no YAML")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo config.yaml")
    args = parser.parse_args()

    # Carrega o arquivo de configuração YAML
    full_cfg = load_config(args.config)
    
    # 1. Identifica os Perfis de Dataset e Modelo escolhidos
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
    
    # 2. Configuração de Hardware
    hw_profile = args.profile or detect_hardware_profile(full_cfg)
    hw_cfg = full_cfg['hardware_profiles'].get(hw_profile, full_cfg['hardware_profiles']['cpu'])
    device = hw_cfg.get('device', 'cpu')
    
    # Ajuste automático específico para F5-TTS para evitar estouro de memória em GPUs menores
    if model_type == 'f5_tts' and device == 'cuda' and hw_cfg['batch_size'] > 2:
        multiplier = hw_cfg['batch_size'] // 2
        hw_cfg['gradient_accumulation_steps'] = hw_cfg.get('gradient_accumulation_steps', 1) * multiplier
        hw_cfg['batch_size'] = 2
        print(f"⚠️ Otimizando p/ F5-TTS: Batch size reduzido para 2 e Gradient Accumulation na GPU para {hw_cfg['gradient_accumulation_steps']}x.")
    
    # 3. Cria a pasta de saída (Output Dir) com carimbo de tempo (timestamp)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(hw_cfg['output_dir'], f"{model_type}-{dataset_name}-{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 Iniciando Pipeline para: {model_type.upper()}")
    print(f"📂 Dataset: {dataset_name} | Hardware: {hw_profile.upper()}")
    
    start_time = time.time() # Inicia cronômetro para medir custo-benefício

    # 4. Carrega o Handler e o Modelo IA
    handler = get_handler(model_type, model_cfg, device)
    model, processor = handler.load()

    # 5. Carregamento do Dataset
    repo_name = ds_cfg['dataset_id'].split('/')[-1]
    local_ds_path = os.path.join(".", "meus_datasets", repo_name)
    is_local = os.path.exists(local_ds_path)
    
    print(f"📥 Carregando dataset '{ds_cfg['dataset_id']}' (Local: {is_local})...")
    
    all_data = []
    # Tratamento especial para o dataset pt-br_char que costuma ter problemas de permissão via API padrão
    if ds_cfg['dataset_id'] == "firstpixel/pt-br_char":
        print("⚠️ Aplicando patch para o 'pt-br_char' (lendo repositório raw)...")
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
                audio_url = os.path.join(local_ds_path, filename.replace("/", os.sep)) if is_local else f"hf://datasets/{ds_cfg['dataset_id']}/{filename}"
                all_data.append({"audio": audio_url, "text": text})
            full_ds = Dataset.from_list(all_data)
        except Exception as e:
            print(f"❌ Erro ao ler dataset: {str(e)}")
            sys.exit(1)
    else:
        # Carregamento genérico para datasets do HuggingFace (LapsBM, CommonVoice, etc)
        load_kwargs = {"split": ds_cfg['dataset_split']}
        if is_local:
            load_kwargs["path"] = local_ds_path
            load_kwargs["streaming"] = False
        else:
            load_kwargs["path"] = ds_cfg['dataset_id']
            load_kwargs["streaming"] = ds_cfg.get('streaming', True) # Usa streaming para economizar disco se não for local
            if 'dataset_config' in ds_cfg: load_kwargs["name"] = ds_cfg['dataset_config']
            
        dataset_stream = load_dataset(**load_kwargs)
        # Converte o stream em uma lista local para permitir amostragem balanceada
        for item in dataset_stream:
            if "wav" in item and "audio" not in item: item["audio"] = item["wav"]
            if "txt" in item and "text" not in item: item["text"] = item["txt"]
            all_data.append(item)
        full_ds = Dataset.from_list(all_data)
    
    # --- Seleção de Locutores e Amostragem Balanceada ---
    all_speakers = [extract_speaker_id(x) for x in all_data]
    counts = Counter(all_speakers) # Conta quantos áudios cada locutor possui
    num_spk = ds_cfg.get('num_speakers', 1) # Quantos locutores queremos treinar
    max_samples = ds_cfg.get('num_samples_per_speaker', 0) # Limite de áudios por locutor
    zero_shot_split = ds_cfg.get('zero_shot_split', None) # Se deve separar locutores 'invisíveis' para teste
    
    # Define quais locutores serão usados
    if num_spk == 0:
        valid_spks = set(all_speakers) # Todos
    else:
        # Pega os locutores com mais áudios disponíveis
        valid_spks = set(s for s, c in counts.most_common(num_spk))
        
    train_spks, val_spks, test_spks = valid_spks, set(), set()
    
    # Lógica de divisão rigorosa Zero-Shot (separar locutores inteiros que o modelo nunca verá no treino)
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
            print(f"⚠️ Locutores insuficientes para zero-shot. Usando todos para treino.")

    # Filtra os índices dos áudios com base na partição de locutores
    train_indices, val_indices, test_indices = [], [], []
    spk_added_counts = {s: 0 for s in valid_spks}
    
    for i, (item, s) in enumerate(zip(all_data, all_speakers)):
        if s in valid_spks:
            if max_samples == 0 or spk_added_counts[s] < max_samples:
                if s in val_spks: val_indices.append(i)
                elif s in test_spks: test_indices.append(i)
                else: train_indices.append(i)
                spk_added_counts[s] += 1
                
    train_dataset = full_ds.select(train_indices)
    val_dataset = full_ds.select(val_indices) if val_indices else None
    test_dataset = full_ds.select(test_indices) if test_indices else None
    
    # Exibe resumo informativo no terminal
    print("\n📊 --- Resumo do Dataset Balanceado ---")
    print(f"   👥 Locutores selecionados: {len(valid_spks)}")
    if zero_shot_split:
        print(f"   🔀 Divisão Zero-Shot: Treino({len(train_spks)}) | Validação({len(val_spks)}) | Teste({len(test_spks)})")
    print(f"   🎙️ Áudios para Treino: {len(train_dataset)}")
    print("----------------------------------------\n")
    
    # Salva a divisão de locutores em um arquivo JSON para auditoria posterior
    import json
    split_info = {
        "train_speakers": list(train_spks), "val_speakers": list(val_spks), "test_speakers": list(test_spks),
        "counts": {"train_samples": len(train_dataset), "val_samples": len(val_dataset) if val_dataset else 0}
    }
    with open(os.path.join(out_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)
        
    # Ajusta a taxa de amostragem (Sampling Rate) de todo o dataset para o que o modelo espera
    target_sr = model_cfg.get('sampling_rate', 16000)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=target_sr))
    if val_dataset: val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    # 6. Processamento do Modelo de Voz (Speaker Encoder)
    print(f"🔊 Processando áudio em {target_sr} Hz...")
    speaker_model = None
    if 'speaker_encoder_id' in model_cfg:
        # Carrega o modelo que extrai 'características da voz' se necessário para a arquitetura
        speaker_model = EncoderClassifier.from_hparams(source=model_cfg['speaker_encoder_id'], run_opts={"device": device})
    
    language = ds_cfg.get("language", "pt")
    # Executa o mapeamento (map) que transforma áudio bruto em tensores prontos para o treino (input_ids, labels, etc)
    processed_train_ds = train_dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=train_dataset.column_names)
    processed_train_ds.set_format(type="torch")
    
    processed_val_ds = None
    if val_dataset:
        processed_val_ds = val_dataset.map(lambda x: handler.prepare_item(x, processor, speaker_model, language), remove_columns=val_dataset.column_names)
        processed_val_ds.set_format(type="torch")

    # 7. Configuração do Treinamento (HuggingFace Trainer)
    train_params = ds_cfg['training']
    training_kwargs = {
        "output_dir": out_dir, # Onde salvar checkpoints
        "per_device_train_batch_size": hw_cfg['batch_size'],
        "gradient_accumulation_steps": hw_cfg['gradient_accumulation_steps'], # Para simular batches grandes em pouca VRAM
        "learning_rate": train_params['learning_rate'], 
        "num_train_epochs": train_params['num_epochs'],
        "logging_steps": train_params['logging_steps'], # Frequência de logs de erro (loss)
        "save_steps": train_params['save_steps'], # Frequência de salvamento de checkpoints
        "weight_decay": train_params.get('weight_decay', 0.0),
        "fp16": hw_cfg['fp16'], # Usa precisão de 16 bits para acelerar e economizar memória
        "save_total_limit": 1, # Mantém apenas o melhor/último checkpoint para economizar espaço em disco
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": "none" # Desativa envio de logs para nuvem (WandB/Tensorboard) por padrão
    }

    if processed_val_ds:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = train_params['logging_steps']
        
    training_args = TrainingArguments(**training_kwargs)

    # Agrupa todos os componentes para o Trainer
    trainer_kwargs = {
        "model": model, "args": training_args, "train_dataset": processed_train_ds, "data_collator": handler.get_collator()
    }
    if processed_val_ds: trainer_kwargs["eval_dataset"] = processed_val_ds

    trainer = Trainer(**trainer_kwargs)
    
    # --- Início do Treino ---
    print("\n🏁 Iniciando Treinamento...")
    trainer.train()
    
    # 8. Salvamento do Modelo Final (LoRA Adapters + Processor)
    print("\n💾 Salvando modelo final...")
    try:
        # Salvamos apenas os adaptadores LoRA (arquivos pequenos) em vez do modelo gigante inteiro
        if hasattr(model, "save_pretrained"): model.save_pretrained(out_dir)
        else: trainer.save_model(out_dir)
        # Salva o processador para que a inferência saiba converter texto da mesma forma
        if hasattr(processor, "save_pretrained"): processor.save_pretrained(out_dir)
        print(f"✅ Treinamento concluído com sucesso! Salvo em: {out_dir}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar modelo final: {e}")

    # 9. Geração de Métricas de Custo-Benefício no README.md
    # Isso ajuda a monitorar se o tempo gasto (latência) justifica a melhora na qualidade (loss)
    print("\n📝 Gerando README com parâmetros da fórmula Custo-Benefício...")
    end_time = time.time()
    training_latency = end_time - start_time
    
    # Pega o último valor de Loss registrado no histórico do Trainer
    final_loss = None
    if trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if 'loss' in log:
                final_loss = log['loss']
                break
                
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_allocated_gb = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    
    # Escreve as métricas de hardware e tempo no arquivo README do modelo gerado
    readme_appendix = f"""
---
## Parâmetros da Fórmula Custo-Benefício (Treinamento)
- **Hardware**: {gpu_name} (Pico VRAM: {vram_allocated_gb:.2f} GB)
- **Latência (Treino)**: {training_latency:.2f} segundos ({training_latency/60:.2f} minutos)
- **Loss Final do Treino**: {final_loss if final_loss is not None else 'N/A'}
"""
    readme_path = os.path.join(out_dir, "README.md")
    mode = "a" if os.path.exists(readme_path) else "w"
    with open(readme_path, mode, encoding="utf-8") as f:
        f.write(readme_appendix)
    print(f"📄 Parâmetros de Custo-Benefício adicionados ao README em: {readme_path}")

if __name__ == "__main__":
    main()
