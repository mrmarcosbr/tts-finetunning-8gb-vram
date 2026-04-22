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

    def generate(self, text, speaker_emb):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
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
        self.dit = PeftModel.from_pretrained(dit, self.model_path)
        self.dit.to(self.device).eval()
        
        # CFM sampler
        self.cfm = CFM(transformer=self.dit, sigma=0.0).to(self.device)

    def generate(self, text, speaker_emb):
        # F5-TTS inference
        # Em uma implementação completa, usaríamos o speaker_emb ou áudio de referência
        # Por simplificação para este teste, usamos o sampler padrão com o texto informado
        # dps = Duration Predictor (opcional)
        print(f"   (F5-TTS) Gerando: {text[:30]}...")
        
        # Encode text
        text_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        
        # Sampling (Exemplo simplificado de 50 steps de ODE)
        with torch.no_grad():
            # mock mel zero para o sampler inicial (ou áudio de referência)
            cond_mel = torch.zeros((1, 5, 80)).to(self.device) 
            sampled_mel, _ = self.cfm.sample(cond_mel, text_ids, duration=200, steps=32)
            
            # Vocoder
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
        # Aqui carregaríamos o config.json do diretório do modelo
        self.xtts = Xtts(config)
        
        # Load weights (Base + LoRA)
        # O PeftModel carrega os adaptadores no submódulo 'gpt'
        self.xtts.gpt = PeftModel.from_pretrained(self.xtts.gpt, self.model_path)
        self.xtts.to(self.device).eval()

    def generate(self, text, speaker_emb):
        # XTTS v2 inference (zero-shot voice cloning)
        # speaker_emb aqui seria condicionado por um áudio de referência real
        # Na API do XTTS, usamos get_conditioning_latents
        print(f"   (XTTS v2) Gerando: {text[:30]}...")
        
        # XTTS v2 gera áudio diretamente via inference()
        # dps = Duration Predictor
        with torch.no_grad():
            output = self.xtts.inference(
                text=text,
                language="pt", # Assumindo PT-BR
                gpt_cond_latent=speaker_emb[0], # latents extraídos
                speaker_embedding=speaker_emb[1], # embeddings extraídos
                temperature=0.7
            )
            audio_out = output["wav"]
            
        return audio_out

def get_inference_handler(model_type, model_cfg, device, model_path):
    if model_type == 'speecht5':
        return SpeechT5Inference(model_cfg, device, model_path)
    elif model_type == 'f5_tts':
        return F5Inference(model_cfg, device, model_path)
    elif model_type == 'xtts_v2':
        return XTTSInference(model_cfg, device, model_path)
    else:
        print(f"⚠️ Inferência para {model_type} não implementada no script genérico.")
        sys.exit(1)

# ==============================================================================
# MAIN INFERENCE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Teste de Inferência TTS con LoRA")
    parser.add_argument("--profile", type=str, default=None, help="Perfil de hardware (opcional)")
    parser.add_argument("--dataset", type=str, default=None, help="Perfil do dataset (opcional)")
    parser.add_argument("--model_name", type=str, default=None, help="Pasta do modelo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo config.yaml")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    
    # 1. Identificar Perfis
    dataset_name = args.dataset or full_cfg.get('settings', {}).get('default_dataset_profile', 'lapsbm')
    if dataset_name not in full_cfg['dataset_profiles']:
        print(f"❌ Perfil de dataset '{dataset_name}' não encontrado.")
        sys.exit(1)
        
    ds_cfg = full_cfg['dataset_profiles'][dataset_name]
    model_type = ds_cfg.get('model_type', 'speecht5')
    model_cfg = full_cfg['models'][model_type]
    
    # 2. Hardware
    hw_name = args.profile or detect_hardware_profile()
    hw_cfg = full_cfg['hardware_profiles'].get(hw_name, full_cfg['hardware_profiles']['cpu'])
    device = hw_cfg.get('device', 'cpu')
    
    print(f"🚀 Hardware: {hw_name.upper()} | Dataset: {dataset_name} | Modelo: {model_type}")

    # 3. Modelos Opcionais de Validação (MOS e ASR)
    print("\n📦 Carregando modelos especialistas auxiliares para validação das métricas (Whisper e SQUIM)...")
    
    mos_predictor = None
    try:
        import speechmos.dnsmos as dnsmos
        print("   ⏳ Carregando SpeechMOS (DNSMOS) para previsão de MOS (Qualidade)...")
        mos_predictor = dnsmos
        print("   ✅ SpeechMOS carregado com sucesso.")
    except Exception as e:
        print(f"   ⚠️ Alerta: Falha ao carregar SpeechMOS (DNSMOS). Continuará sem MOS. Detalhe: {e}")

    asr_pipeline = None
    try:
        from transformers import pipeline
        print("   ⏳ Carregando Whisper Large V3 Turbo para Transcrição e extração de WER...")
        # Usa GPU local da forma segura
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0 if torch.cuda.is_available() else -1, chunk_length_s=30)
        print("   ✅ Whisper carregado com sucesso.")
    except Exception as e:
        print(f"   ⚠️ Alerta: Falha ao carregar Whisper. Continuará sem WER. Detalhe: {e}")

    # 3. Localizar Modelo
    profile_output_dir = hw_cfg['output_dir']
    model_path = None

    if args.model_name:
        # Tenta no diretório do perfil atual
        temp_path = os.path.join(profile_output_dir, args.model_name)
        if os.path.exists(temp_path):
            model_path = temp_path
        else:
            # Busca em outros diretórios de perfis caso tenha mudado de hardware (ex: CUDA -> CPU fallback)
            for p_name, p_cfg in full_cfg['hardware_profiles'].items():
                alt_path = os.path.join(p_cfg['output_dir'], args.model_name)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"📂 Modelo encontrado em diretório alternativo: {alt_path}")
                    break
        
        if not model_path:
            print(f"❌ Erro: O diretório do modelo '{args.model_name}' não foi encontrado em nenhum perfil de saída.")
            sys.exit(1)
    else:
        # Lógica de detecção automática baseada no prefixo
        prefix = f"{model_type}-{dataset_name}-"
        # Coleta subpastas de TODOS os output_dirs configurados
        all_dirs = set(p['output_dir'] for p in full_cfg['hardware_profiles'].values())
        subfolders_found = []
        for d in all_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    full_p = os.path.join(d, f)
                    if os.path.isdir(full_p) and prefix in f:
                        subfolders_found.append(full_p)
        
        if not subfolders_found:
             print(f"⚠️ Nenhum modelo encontrado contendo '{prefix}'. Buscando fallback...")
             # Busca fallback em todos os lugares
             fallback_found = None
             for d in all_dirs:
                 fb = os.path.join(d, "0-1-Locutor-speecht5_laps_lora_exhaustive")
                 if os.path.exists(fb):
                     fallback_found = fb
                     break
             
             if fallback_found:
                 model_path = fallback_found
                 print(f"📂 Fallback detectado: {model_path}")
             else:
                 print(f"❌ Erro: Nenhum modelo validado encontrado nos diretórios de saída. Use --model_name para especificar manualmente.")
                 sys.exit(1)
        else:
             subfolders_found.sort()
             model_path = subfolders_found[-1]
             print(f"📂 Modelo detectado: {model_path}")

    # 4. Handler e Carregamento
    handler = get_inference_handler(model_type, model_cfg, device, model_path)
    handler.load()
    
    # 5. Speaker Embedding (Referência)
    repo_name = ds_cfg['dataset_id'].split('/')[-1]
    local_ds_path = os.path.join(".", "meus_datasets", repo_name)
    is_local = os.path.exists(local_ds_path)
    
    print(f"📥 Carregando speaker reference de: {ds_cfg['dataset_id']} (Local: {is_local})")
    
    if ds_cfg['dataset_id'] == "firstpixel/pt-br_char":
        import pandas as pd
        if is_local:
            csv_path = os.path.join(local_ds_path, "metadata.csv")
            df = pd.read_csv(csv_path, sep="|")
            audio_path = os.path.join(local_ds_path, df.iloc[0, 0].replace("/", os.sep))
            dataset = Dataset.from_list([{"audio": audio_path, "text": df.iloc[0, 1], "__url__": "Ref"}])
        else:
            from huggingface_hub import hf_hub_download
            csv_path = hf_hub_download(repo_id=ds_cfg['dataset_id'], filename="metadata.csv", repo_type="dataset")
            df = pd.read_csv(csv_path, sep="|")
            audio_url = f"hf://datasets/{ds_cfg['dataset_id']}/{df.iloc[0, 0]}"
            dataset = Dataset.from_list([{"audio": audio_url, "text": df.iloc[0, 1], "__url__": "Ref"}])
    else:
        load_kwargs = {"split": ds_cfg['dataset_split'], "streaming": False}
        if is_local:
            load_kwargs["path"] = local_ds_path
        else:
            load_kwargs["path"] = ds_cfg['dataset_id']
            if 'dataset_config' in ds_cfg: load_kwargs["name"] = ds_cfg['dataset_config']
            
        dataset = load_dataset(**load_kwargs)
    
    if "wav" in dataset.column_names: dataset = dataset.rename_column("wav", "audio")
    
    # O codificador de locutor SEMPRE espera áudio em 16000Hz independente do gerador TTS
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    all_speakers = [extract_speaker_id(x) for x in dataset]
    counts = Counter(all_speakers)
    most_common_spk, _ = counts.most_common(1)[0]
    indices = [i for i, x in enumerate(all_speakers) if x == most_common_spk]
    
    sample = dataset[indices[0]]
    audio_ref = sample["audio"]["array"]
    
    print(f"👤 Clonando voz do locutor: {most_common_spk}")
    speaker_model = EncoderClassifier.from_hparams(source=model_cfg['speaker_encoder_id'], run_opts={"device": device})
    
    with torch.no_grad():
        most_common_emb = speaker_model.encode_batch(torch.tensor(audio_ref))
        most_common_emb = torch.nn.functional.normalize(most_common_emb, dim=2).squeeze().to(device).unsqueeze(0)

    # 6. Test Cases
    target_sr = model_cfg.get('sampling_rate', 16000)
    cases = []
    
    # Casos Originais
    cases.append((
        "caso1_standard", 
        "O treinamento foi finalizado e agora estamos testando a voz em português.", 
        most_common_emb, 
        most_common_spk
    ))
    cases.append((
        "caso2_paris", 
        "Há algumas coisas que não podem deixar de serem vistas em Paris.", 
        most_common_emb, 
        most_common_spk
    ))
    cases.append((
        "caso3_extra", 
        f"Olá! Tudo bem com você?? Eu estou ótimo!!!!! Estamos testando o suporte multi-modelo para {model_type.upper()}. A ideia é ver como ele se comporta com diferentes tipos de texto.", 
        most_common_emb, 
        most_common_spk
    ))
    
    # Processar dataset_split.json se existir
    import json
    split_path = os.path.join(model_path, "dataset_split.json")
    if os.path.exists(split_path):
        print("\n🔍 Lendo configurações de Zero-Shot Split (dataset_split.json)...")
        with open(split_path, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        
        test_spks = split_info.get("test_speakers", [])
        if test_spks:
            print(f"   👥 Locutores reservados para Teste Zero-Shot encontrados: {test_spks}")
            
            seen_test_spks = set()
            for s_id, item in zip(all_speakers, dataset):
                if s_id in test_spks and s_id not in seen_test_spks:
                    seen_test_spks.add(s_id)
                    text = item.get("text", "")
                    audio = item["audio"]["array"]
                    
                    with torch.no_grad():
                        emb = speaker_model.encode_batch(torch.tensor(audio))
                        emb = torch.nn.functional.normalize(emb, dim=2).squeeze().to(device).unsqueeze(0)
                        
                    cases.append((f"zero_shot_{s_id}", text, emb, s_id))
                    
                    if len(seen_test_spks) == len(test_spks):
                        break

    print("\n🎧 Gerando áudios e Validando Métricas...")
    
    validation_metrics = []
    
    for filename, text, emb, spk_id in cases:
        clean_text = normalize_text(text)
        print(f"\n   👤 Usando Voz: '{spk_id}' | Origem: {'Teste Invisível' if 'zero_shot' in filename else 'Conhecido (Fixos)'}")
        print(f"   📝 Mensagem: '{clean_text}'")
        print(f"   ⚙️  ID Local do Teste: {filename}")
        
        # 7. Medir Latência
        start_time = time.time()
        audio_out = handler.generate(clean_text, emb)

        gen_time = time.time() - start_time
        
        # Calcular RTF (Real Time Factor)
        audio_duration = len(audio_out) / target_sr
        rtf = gen_time / audio_duration if audio_duration > 0 else 0
        
        out_path = os.path.join(model_path, f"{filename}_{dataset_name}_{model_type}_test.wav")
        sf.write(out_path, audio_out, target_sr)
        
        # 8. Calcular Similaridade do Locutor (SimLocutor)
        # O modelo SpeechBrain espera 16000Hz
        if target_sr != 16000:
            import librosa
            audio_out_16k = librosa.resample(audio_out, orig_sr=target_sr, target_sr=16000)
        else:
            audio_out_16k = audio_out
            
        with torch.no_grad():
            gen_emb = speaker_model.encode_batch(torch.tensor(audio_out_16k)).to(device)
            gen_emb = torch.nn.functional.normalize(gen_emb, dim=2).squeeze().unsqueeze(0)
            # Calcula similaridade do cosseno (-1 a 1)
            sim = torch.nn.functional.cosine_similarity(emb, gen_emb).item()
            
        # 8b. Calcular Qualidade de Áudio Preditiva (MOS de 1 a 5)
        mos_score = 0.0
        if mos_predictor:
            # O DNSMOS do speechmos recebe o numpy array e sample_rate 16000
            mos_res = mos_predictor.run(audio_out_16k, 16000)
            mos_score = mos_res['ovrl_mos']
                
        # 8c. Calcular Word Error Rate (WER via Whisper transcrevendo o Output)
        wer_score = 1.0 # 100% de erro de fallback se falhar
        if asr_pipeline:
            whisper_out = asr_pipeline({"raw": audio_out_16k, "sampling_rate": 16000})
            transcription = whisper_out["text"]
            wer_score = calculate_wer(clean_text, transcription)
            
        print(f"      ✅ Salvo: {out_path}")
        print(f"         ➡️ [Métricas] RTF: {rtf:.2f}x | SimLocutor: {sim:.3f} | MOS: {mos_score:.2f} | WER: {wer_score:.2f}")
        validation_metrics.append({"filename": filename, "rtf": rtf, "sim": sim, "mos": mos_score, "wer": wer_score})

    # 9. Anexar métricas no README.md do Modelo
    if validation_metrics:
        avg_rtf = sum(m["rtf"] for m in validation_metrics) / len(validation_metrics)
        avg_sim = sum(m["sim"] for m in validation_metrics) / len(validation_metrics)
        avg_mos = sum(m["mos"] for m in validation_metrics) / len(validation_metrics)
        avg_wer = sum(m["wer"] for m in validation_metrics) / len(validation_metrics)
        
        # Calcular Accuracy Positiva (1 - WER) => Ex: WER 0.2 vira Acc 0.8
        acc_wer = max(1.0 - avg_wer, 0.0)
        
        readme_path = os.path.join(model_path, "README.md")
        # Se for string multi line sem indent, fica melhor no markdown
        readme_appendix = f"""
## Parâmetros da Fórmula Custo-Benefício (Validação de Inferência)

**Teste realizado com o script test_inference_exhaustive.py gerando inferências diretas e métricas automatizadas.**

### 🔻 Custos Adicionais (Denominador)
- **Latência Média de Geração (RTF)**: {avg_rtf:.2f}x *(O tempo levado divido pela duração total do áudio em segundos)*

### 🌟 Benefícios Medidos (Numerador Prático)
- **Acurácia (1 - WER)**: {acc_wer:.3f} *(Extraída através do word-error-rate transcrito pelo Whisper e Distância Levenshtein)*
- **SimLocutor Médio (Similaridade de Voz)**: {avg_sim:.3f} *(Medido via distância de cossenos em Embeddings do WavLM/SpeechBrain. >0.85 é excelente)*
- **Qualidade Média Percebida (MOS Nominal)**: {avg_mos:.2f} / 5.00 *(Calculado via torchaudio SQUIM_SUBJECTIVE DNS)*
"""
        mode = "a" if os.path.exists(readme_path) else "w"
        try:
            with open(readme_path, mode, encoding="utf-8") as f:
                f.write(readme_appendix)
            print(f"\n📄 Métricas de Inferência anexadas ao README do modelo na pasta:\n   {readme_path}")
        except Exception as e:
            print(f"\n⚠️ Falha ao salvar README de inferência: {e}")

if __name__ == "__main__":
    main()
