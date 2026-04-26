import os
import sys
sys.modules['torchcodec'] = None # Mock para evitar crash do torchaudio e do transformers no Windows

import time
import yaml
import torch
import torchaudio
import unicodedata
import argparse
import soundfile as sf
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict

# Carregar variáveis de ambiente (HF_TOKEN)
load_dotenv()

# --- 1. Monkey Patch & Setup ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel
from speechbrain.inference.speaker import EncoderClassifier
from collections import Counter

# Configuração de encoding para Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def normalize_text(text):
    if not text: return ""
    text = text.lower() # Converter para minúsculo para bater com vocabulário
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def calculate_wer(reference_text, hypothesis_text):
    import re
    # Limpa pontuação e deixa tudo minúsculo
    def clean(s): return re.sub(r'[^\w\s]', '', s.lower().strip())
    
    ref_words = clean(reference_text).split()
    hyp_words = clean(hypothesis_text).split()
    if not ref_words: return 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint8)
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j]) + 1
                
    return float(d[len(ref_words)][len(hyp_words)]) / len(ref_words)
    
def find_checkpoint(model_path):
    """Detecta se há subpastas de checkpoint e retorna a mais recente."""
    if not os.path.isdir(model_path):
        return model_path
    
    checkpoints = [d for d in os.listdir(model_path) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, d))]
    
    if not checkpoints:
        return model_path
        
    # Ordenar por número do passo (checkpoint-1000, checkpoint-2000, etc)
    try:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest = checkpoints[-1]
        print(f"   🔄 Checkpoint detectado automaticamente: {latest}")
        return os.path.join(model_path, latest)
    except:
        return model_path


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_hardware_profile():
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram >= 14.0: return "cuda_16gb"
        return "cuda_8gb"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "macbook"
    return "cpu"

def extract_speaker_id(item):
    url = item.get("__url__", "")
    if url and "LapsBM-" in url:
        parts = url.split('/')
        for p in parts:
            if p.startswith("LapsBM-"):
                return p.split(".")[0].replace("LapsBM-", "")
    for k in ['speaker_id', 'speaker', 'user_id', 'client_id']:
        if k in item and item[k] is not None:
            return str(item[k])
    return "unknown"

# ==============================================================================
# INFERENCE HANDLERS
# ==============================================================================

class InferenceHandler:
    def __init__(self, model_cfg: Dict, device: str, model_path: str):
        self.model_cfg = model_cfg
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError()

    def generate(self, text: str, speaker_emb: torch.Tensor):
        raise NotImplementedError()

class SpeechT5Inference(InferenceHandler):
    def load(self):
        print(f"📥 Carregando SpeechT5 (Base + LoRA) de: {self.model_path}")
        self.processor = SpeechT5Processor.from_pretrained(self.model_path)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_cfg['vocoder_id']).to(self.device)
        
        base_model = SpeechT5ForTextToSpeech.from_pretrained(self.model_cfg['id'])
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, text, speaker_emb, use_lora=True):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            if use_lora:
                spec = self.model.generate(input_ids=inputs["input_ids"].to(self.device), speaker_embeddings=speaker_emb)
            else:
                with self.model.disable_adapter():
                    spec = self.model.generate(input_ids=inputs["input_ids"].to(self.device), speaker_embeddings=speaker_emb)
            audio_out = self.vocoder(spec)
        return audio_out.squeeze().cpu().numpy()

class F5Inference(InferenceHandler):
    def load(self):
        try:
            from f5_tts.model import DiT, CFM
            from f5_tts.model.utils import get_tokenizer
            from vocos import Vocos
        except ImportError:
            print("❌ Erro: Requer 'pip install f5-tts vocos'")
            sys.exit(1)

        print(f"📥 Carregando F5-TTS de: {self.model_path}")
        self.tokenizer = get_tokenizer("pinyin")
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        
        # Base DiT
        dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_num_embeds=256, mel_dim=80)
        
        # Load LoRA
        self.model = PeftModel.from_pretrained(dit, self.model_path)
        self.model.to(self.device).eval()
        
        # CFM sampler
        self.cfm = CFM(transformer=self.model, sigma=0.0).to(self.device)

    def generate(self, text, speaker_emb, use_lora=True):
        print(f"   (F5-TTS) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        text_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        with torch.no_grad():
            cond_mel = torch.zeros((1, 5, 80)).to(self.device) 
            if use_lora:
                sampled_mel, _ = self.cfm.sample(cond_mel, text_ids, duration=200, steps=32)
            else:
                with self.model.disable_adapter():
                    sampled_mel, _ = self.cfm.sample(cond_mel, text_ids, duration=200, steps=32)
            audio_out = self.vocos.decode(sampled_mel.transpose(1, 2))
        return audio_out.squeeze().cpu().numpy()

class XTTSInference(InferenceHandler):
    def load(self):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError:
            print("❌ Erro: Requer 'pip install TTS'")
            sys.exit(1)

        print(f"📥 Carregando XTTS v2 de: {self.model_path}")
        config = XttsConfig()
        self.xtts = Xtts(config)
        self.model = PeftModel.from_pretrained(self.xtts.gpt, self.model_path)
        self.xtts.gpt = self.model # Substitui o gpt pelo PeftModel
        self.xtts.to(self.device).eval()

    def generate(self, text, speaker_emb, use_lora=True):
        print(f"   (XTTS v2) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        with torch.no_grad():
            if use_lora:
                output = self.xtts.inference(
                    text=text,
                    language="pt",
                    gpt_cond_latent=speaker_emb[0],
                    speaker_embedding=speaker_emb[1],
                    temperature=0.7
                )
            else:
                with self.model.disable_adapter():
                    output = self.xtts.inference(
                        text=text,
                        language="pt",
                        gpt_cond_latent=speaker_emb[0],
                        speaker_embedding=speaker_emb[1],
                        temperature=0.7
                    )
            audio_out = output["wav"]
        return audio_out

class FastSpeech2Inference(InferenceHandler):
    def load(self):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.forward_tts import ForwardTTS as FastSpeech2
            from TTS.tts.configs.fastspeech2_config import Fastspeech2Config as FastSpeech2Config
            from TTS.vocoder.models.gan import GAN as HifiGan
        except ImportError:
            print("❌ Erro: Requer 'pip install coqui-tts'")
            sys.exit(1)

        print(f"📥 Carregando FastSpeech 2 de: {self.model_path}")
        config_path = os.path.join(self.model_path, "config.json")
        model_file = os.path.join(self.model_path, "model.pth") # Caso padrão
        
        if not os.path.exists(config_path):
            print(f"   🔍 Config não encontrada em {self.model_path}, baixando modelo base...")
            manager = ModelManager()
            model_base_path = manager.download_model(self.model_cfg['id'])
            if isinstance(model_base_path, tuple): model_base_path = model_base_path[0]
            
            if os.path.isfile(model_base_path):
                model_dir = os.path.dirname(model_base_path)
                model_file = model_base_path
            else:
                model_dir = model_base_path
                model_file = os.path.join(model_dir, "model.pth")
                
            config_path = os.path.join(model_dir, "config.json")
            
        config = FastSpeech2Config()
        config.load_json(config_path)
        base_model = FastSpeech2.init_from_config(config)
        
        # Tentar carregar o checkpoint base antes de aplicar o Peft
        if os.path.exists(model_file):
            print(f"   基础 (Base) model weights: {model_file}")
            base_model.load_checkpoint(config, checkpoint_path=model_file)
            
        # 🔍 Lógica de Detecção de Checkpoint
        actual_model_path = find_checkpoint(self.model_path)

        # Carregar LoRA (tentando PeftModel.from_pretrained primeiro, depois fallback manual)
        if os.path.exists(os.path.join(actual_model_path, "adapter_config.json")):
            print(f"   ✅ Carregando adaptadores LoRA via PEFT de {actual_model_path}")
            self.model = PeftModel.from_pretrained(base_model, actual_model_path)
        else:
            print("   ⚠️ adapter_config.json não encontrado. Tentando reconstrução manual do LoRA...")
            from peft import LoraConfig, get_peft_model
            lora_cfg = self.model_cfg['lora']
            peft_config = LoraConfig(
                r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], 
                target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], 
                bias="none"
            )
            self.model = get_peft_model(base_model, peft_config)
            
            # Carregar pesos do state_dict (Safetensors ou PyTorch Bin)
            weights_file = None
            for wf in ["model.safetensors", "pytorch_model.bin", "adapter_model.bin"]:
                if os.path.exists(os.path.join(actual_model_path, wf)):
                    weights_file = os.path.join(actual_model_path, wf)
                    break
            
            if weights_file:
                print(f"   📥 Carregando pesos de: {os.path.basename(weights_file)}")
                if weights_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_file)
                else:
                    state_dict = torch.load(weights_file, map_location=self.device)
                
                # Limpeza cirúrgica de prefixos
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    # O Trainer salva com 'model.base_model.model...', 
                    # o PeftModel espera 'base_model.model...'
                    if name.startswith("model.base_model.model."): 
                        name = name[6:] # Remove apenas o primeiro 'model.'
                    new_state_dict[name] = v
                
                # Carregar pesos no PeftModel
                missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                loaded_keys = len(new_state_dict) - len(unexpected)
                print(f"   ✨ Sucesso: {loaded_keys} tensores carregados (LoRA + Base).")
                if unexpected:
                    print(f"   ℹ️ {len(unexpected)} chaves ignoradas (provavelmente optimizers ou metadados).")
            else:
                print(f"   ❌ ERRO: Nenhum arquivo de pesos encontrado em {actual_model_path}")
                print(f"      Arquivos presentes: {os.listdir(actual_model_path)}")

        self.model.to(self.device).eval()
        
        # Load Vocoder
        manager = ModelManager()
        vocoder_res = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
        if isinstance(vocoder_res, tuple): vocoder_res = vocoder_res[0]
        
        if os.path.isfile(vocoder_res):
            vocoder_dir = os.path.dirname(vocoder_res)
            vocoder_file = vocoder_res
        else:
            vocoder_dir = vocoder_res
            vocoder_file = os.path.join(vocoder_dir, "model.pth")
            
        from TTS.vocoder.configs.hifigan_config import HifiganConfig as HifiGanConfig
        v_config = HifiGanConfig()
        
        # Carregar config limpando comentários (// e /* */) que o JSON padrão não suporta
        import re
        import json
        with open(os.path.join(vocoder_dir, "config.json"), 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        v_config.from_dict(json.loads(content))

        self.vocoder = HifiGan.init_from_config(v_config)
        self.vocoder.load_checkpoint(v_config, checkpoint_path=vocoder_file)
        self.vocoder.to(self.device).eval()
        self.tokenizer = base_model.tokenizer
        
        # Tentar trocar para o phonemizer de PT se possível (como feito no treino)
        try:
            from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name
            base_phonemizer = get_phonemizer_by_name("gruut", language="pt")
            
            # Criamos um wrapper para mapear fonemas do PT que não existem no modelo Base (Inglês)
            class PTPhonemizerWrapper:
                def __init__(self, p): self.p = p
                def phonemize(self, text, separator="", language="pt"):
                    ph = self.p.phonemize(text, separator=separator, language=language)
                    # Mapeamento de Precisão (Vogal + n) usando 'ɐ' para o som de 'ã' fechado.
                    replacements = {
                        'ã': 'ɐn', 'ẽ': 'en', 'ĩ': 'in', 'õ': 'on', 'ũ': 'un',
                        '\u0303': 'n', 'g': 'ɡ'
                    }
                    for k, v in replacements.items(): ph = ph.replace(k, v)
                    return ph
                def name(self): return "pt_wrapper"
                def print_logs(self, level): pass
                
            new_phonemizer = PTPhonemizerWrapper(base_phonemizer)
            self.tokenizer.phonemizer = new_phonemizer
            print(f"   🌐 Phonemizer do FastSpeech 2 substituído por Gruut (pt) com mapeamento cross-lingual")
        except Exception as e:
            print(f"   ⚠️ Aviso: Não foi possível trocar o phonemizer: {e}")

    def generate(self, text, speaker_emb, use_lora=True):
        print(f"   (FastSpeech 2) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        input_ids = torch.tensor([self.tokenizer.text_to_ids(text, language="pt")]).to(self.device)
        with torch.no_grad():
            if use_lora:
                outputs = self.model.inference(input_ids, aux_input={"durations": None, "pitch": None, "energy": None, "length_scale": 1.0})
            else:
                with self.model.disable_adapter():
                    outputs = self.model.inference(input_ids)
            mel_output = outputs["model_outputs"]
            
            # Ajustar dimensões para o vocoder (espera [B, C, T])
            if mel_output.ndim == 3:
                if mel_output.shape[2] == 80 or mel_output.shape[2] == 90:
                    mel_output = mel_output.transpose(1, 2)
            
            # Garantir 80 canais
            if mel_output.shape[1] > 80:
                mel_output = mel_output[:, :80, :]
            elif mel_output.shape[1] < 80:
                mel_output = torch.nn.functional.pad(mel_output, (0, 0, 0, 80 - mel_output.shape[1]))

            # --- CORREÇÃO DE ESCALA PARA O VOCODER ---
            # Converte de [-4, 4] para [-100, 0]
            min_level_db = -100
            max_norm = 4.0
            mel_norm = (mel_output / max_norm + 1) / 2.0
            mel_raw = mel_norm * (-min_level_db) + min_level_db
            mel_output = mel_raw.clamp(min_level_db, 0)
            # -----------------------------------------

            audio_out = self.vocoder.inference(mel_output)
        return audio_out.squeeze().cpu().numpy()

class MatchaInference(InferenceHandler):
    def load(self):
        try:
            from matcha.models.matcha_tts import MatchaTTS
        except ImportError:
            print("❌ Erro: Requer 'pip install matcha-tts'")
            sys.exit(1)
        print(f"📥 Carregando Matcha-TTS de: {self.model_path}")
        pass
    def generate(self, text, speaker_emb, use_lora=True):
        print(f"   (Matcha-TTS) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        return np.zeros(16000)

class GlowTTSInference(FastSpeech2Inference):
    def load(self):
        try:
            from TTS.utils.manage import ModelManager
            from TTS.tts.models.glow_tts import GlowTTS
            from TTS.tts.configs.glow_tts_config import GlowTTSConfig
            from TTS.vocoder.models.gan import GAN as HifiGan
        except ImportError:
            print("❌ Erro: Requer 'pip install coqui-tts'")
            sys.exit(1)
        print(f"📥 Carregando Glow-TTS de: {self.model_path}")
        manager = ModelManager()
        model_base_path = manager.download_model(self.model_cfg['id'])
        if isinstance(model_base_path, tuple): model_base_path = model_base_path[0]
        config = GlowTTSConfig()
        config.load_json(os.path.join(model_base_path, "config.json"))
        base_model = GlowTTS.init_from_config(config)
        self.model = PeftModel.from_pretrained(base_model.encoder, self.model_path)
        self.model.to(self.device).eval()
        
        # Load Vocoder
        manager = ModelManager()
        vocoder_res = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
        if isinstance(vocoder_res, tuple): vocoder_res = vocoder_res[0]
        
        if os.path.isfile(vocoder_res):
            vocoder_dir = os.path.dirname(vocoder_res)
            vocoder_file = vocoder_res
        else:
            vocoder_dir = vocoder_res
            vocoder_file = os.path.join(vocoder_dir, "model.pth")
            
        from TTS.vocoder.configs.hifigan_config import HifiganConfig as HifiGanConfig
        v_config = HifiGanConfig()
        
        # Carregar config limpando comentários (// e /* */) que o JSON padrão não suporta
        import re
        import json
        with open(os.path.join(vocoder_dir, "config.json"), 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        v_config.from_dict(json.loads(content))

        from TTS.vocoder.models.gan import GAN as HifiGan
        self.vocoder = HifiGan.init_from_config(v_config)
        self.vocoder.load_checkpoint(v_config, checkpoint_path=vocoder_file)
        self.vocoder.to(self.device).eval()
        self.tokenizer = base_model.tokenizer

    def generate(self, text, speaker_emb, use_lora=True):
        print(f"   (Glow-TTS) Gerando {'(LoRA)' if use_lora else '(Base)'}: {text[:30]}...")
        input_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        with torch.no_grad():
            if use_lora:
                outputs = self.model.inference(input_ids)
            else:
                with self.model.disable_adapter():
                    outputs = self.model.inference(input_ids)
            mel_output = outputs["model_outputs"]
            
            # Ajustar dimensões para o vocoder (espera [B, C, T])
            if mel_output.ndim == 3:
                # Se for [B, T, C], transpõe para [B, C, T]
                if mel_output.shape[2] == 80 or mel_output.shape[2] == 90:
                    mel_output = mel_output.transpose(1, 2)
            
            # Garantir 80 canais
            if mel_output.shape[1] > 80:
                mel_output = mel_output[:, :80, :]
            elif mel_output.shape[1] < 80:
                mel_output = torch.nn.functional.pad(mel_output, (0, 0, 0, 80 - mel_output.shape[1]))

            audio_out = self.vocoder.inference(mel_output)
        return audio_out.squeeze().cpu().numpy()

def get_inference_handler(model_type, model_cfg, device, model_path):
    if model_type == 'speecht5': return SpeechT5Inference(model_cfg, device, model_path)
    elif model_type == 'f5_tts': return F5Inference(model_cfg, device, model_path)
    elif model_type == 'fastspeech2': return FastSpeech2Inference(model_cfg, device, model_path)
    elif model_type == 'matcha': return MatchaInference(model_cfg, device, model_path)
    elif model_type == 'glow_tts': return GlowTTSInference(model_cfg, device, model_path)
    elif model_type == 'xtts_v2': return XTTSInference(model_cfg, device, model_path)
    else:
        print(f"⚠️ Inferência para {model_type} não implementada.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--model_dir", type=str, required=True, help="Pasta do modelo treinado (ex: ./output_cuda_8gb/fastspeech2-...)")
    parser.add_argument("--text", type=str, default=None, help="Texto customizado para gerar áudio")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    dataset_name = args.dataset or full_cfg.get('settings', {}).get('default_dataset_profile', 'lapsbm_fastspeech2')
    ds_cfg = full_cfg['dataset_profiles'][dataset_name]
    model_type = ds_cfg.get('model_type', 'fastspeech2')
    model_cfg = full_cfg['models'][model_type]
    
    hw_name = args.profile or detect_hardware_profile()
    hw_cfg = full_cfg['hardware_profiles'].get(hw_name, full_cfg['hardware_profiles']['cpu'])
    device = hw_cfg.get('device', 'cpu')

    # Ajuste para aceitar o caminho do checkpoint diretamente ou a pasta pai
    model_path = args.model_path

    print(f"🚀 Iniciando Teste de Inferência...")
    print(f"   📂 Modelo: {model_path}")
    print(f"   💻 Hardware: {hw_name.upper()} ({device})")

    # 1. Carregar Handler e Modelo
    handler = get_inference_handler(model_type, model_cfg, device, model_path)
    handler.load()

    sampling_rate = model_cfg.get('sampling_rate', 22050)
    
    # 2. Obter Speaker Embedding (se necessário)
    speaker_emb = None
    if model_type in ['speecht5', 'xtts_v2']:
        if model_type == 'speecht5':
            speaker_emb = torch.zeros((1, 512)).to(device)
        elif model_type == 'xtts_v2':
            speaker_emb = [torch.zeros((1, 1, 1024)).to(device), torch.zeros((1, 512)).to(device)]

    # 3. Preparar lista de textos
    default_texts = [
        "O treinamento exaustivo foi finalizado com sucesso.",
        "Esta é uma demonstração de voz gerada por inteligência artificial.",
        "A qualidade do áudio melhorou significativamente após o ajuste fino.",
        "Estamos testando a comparação entre o modelo base e o modelo treinado."
    ]
    
    test_texts = []
    if args.text:
        test_texts.append(args.text)
    test_texts.extend(default_texts)

    # 4. Criar pasta de saída
    type_str = "custom" if args.text else "batch"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(model_path, f"inference_{type_str}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n   🎙️ Iniciando geração de {len(test_texts)} sentenças...")
    print(f"   🎯 Resultados em: {out_dir}")

    # 5. Executar Inferência
    for i, text in enumerate(test_texts):
        print(f"\n   --- Teste {i+1}/{len(test_texts)} ---")
        clean_text = normalize_text(text)
        prefix = "custom" if (args.text and i == 0) else f"sentenca_{i+1}"
        
        # Original (Base)
        print(f"   🔊 Gerando ORIGINAL...")
        try:
            audio_orig = handler.generate(clean_text, speaker_emb, use_lora=False)
            sf.write(os.path.join(out_dir, f"{prefix}_original.wav"), audio_orig, sampling_rate)
        except Exception as e:
            print(f"   ⚠️ Erro no modelo original: {e}")

        # Treinado (LoRA)
        print(f"   🔥 Gerando TREINADO (LoRA)...")
        try:
            audio_lora = handler.generate(clean_text, speaker_emb, use_lora=True)
            sf.write(os.path.join(out_dir, f"{prefix}_treinado.wav"), audio_lora, sampling_rate)
            # Se for a frase customizada, salvar também na raiz para facilitar
            if args.text and i == 0:
                sf.write("output_inference.wav", audio_lora, sampling_rate)
        except Exception as e:
            print(f"   ⚠️ Erro no modelo treinado: {e}")
            
    print(f"\n✅ Concluído! Todos os arquivos estão na pasta: {out_dir}")

if __name__ == "__main__":
    main()
