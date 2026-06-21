import torch
import torchaudio
# Monkey patch for backend check compatibility issues with newer torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    # This is a dummy implementation to satisfy speechbrain's check
    torchaudio.list_audio_backends = lambda: ["soundfile"] 

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
import os

def main():
    print("🚀 Iniciando experimento Quick Test (Pico)...")
    
    # 1. Configuração de Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detectado: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Inicial: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 2. Carregar Modelos (SpeechT5 + HiFi-GAN Vocoder)
    print("📦 Carregando modelos...")
    checkpoint = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # 3. Preparar Dataset (Streaming para não baixar tudo)
    print("🌊 Carregando dataset (Streaming mode)...")
    # Common Voice precisa de login, trocando para google/fleurs (aberto)
    try:
        dataset = load_dataset("google/fleurs", "pt_br", split="train", streaming=True)
    except Exception:
        print("⚠️ Fleurs falhou, tentando dataset dummy em inglês apenas para teste de pipeline")
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", streaming=True)
    
    dataset = dataset.take(10) # Pegar apenas 10 exemplos para teste
    
    # 4. Speaker Embeddings (X-Vector)
    print("🗣️  Carregando encoder de speaker...")
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name, 
        run_opts={"device": device}, 
        savedir=os.path.join("/tmp", spk_model_name)
    )

    # Validar que temos algo
    speaker_embeddings = torch.randn(1, 512).to(device) # Default fallback
    
    # 5. Teste de Inferência Zero-Shot (Antes do Treino)
    print("🧪 Testando inferência zero-shot...")
    text = "Isso é um teste rápido na minha GPU quarenta setenta."
    inputs = processor(text=text, return_tensors="pt")
    
    try:
        # Tentar pegar do dataset se possível
        try:
            example = next(iter(dataset))
            # Se conseguirmos um exemplo, tentamos processar o speaker embedding real
            # Mas aqui o speechbrain pode falhar se não tiver backend de audio
            # Vamos tentar só se der, senão segue o baile com random
            # (Código omitido para simplificar e garantir que o teste rode)
            pass
        except Exception as e:
            print(f"⚠️ Não foi possível carregar exemplo do dataset: {e}")

        # Gerar áudio com embeddings (random ou real se tivéssemos implementado)
        speech = model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings, vocoder=vocoder)
        
        sf.write("teste_zero_shot.wav", speech.cpu().numpy(), samplerate=16000)
        print("✅ Áudio gerado: teste_zero_shot.wav")
        
    except Exception as e:
        print(f"❌ Erro na inferência: {e}")

    # 6. Loop de Treino "Fake" (Overfit em 1 batch)
    print("🏋️‍♂️  Iniciando treino fake (prova de vida)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Inputs fake
    inputs = processor(text="Texto de treino", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Labels (mel spectrogram fake - na real precisaria extrair do audio)
    # O speecht5 outputa spectrograma. Vamos fazer um forward pass pra pegar o shape e criar um target dummy
    # Só quero ver se o backward pass funciona sem OOM
    
    try:
        # Dummy forward para pegar shape
        # SpeechT5 forward espera labels para calcular loss automaticamente
        # Labels shape: (batch, sequence_length, num_mel_bins) -> num_mel_bins=80
        
        # Vamos rodar 5 iterações de backprop em dados aleatórios
        for i in range(5):
            # Target dummy
            # O modelo faz redução de dimensão, então o output length depende do input
            # Vamos deixar o modelo calcular o output e usar isso como target com leve ruído para simular "aprender"
            
            # Forward sem labels
            output = model(input_ids=input_ids, speaker_embeddings=speaker_embeddings)
            # output.spectrogram é o predito
            
            # Criar um target dummy com mesmo shape
            labels = output.spectrogram.detach() # (1, seq_len, 80)
            
            # Forward com labels (calcula loss)
            output = model(input_ids=input_ids, speaker_embeddings=speaker_embeddings, labels=labels)
            
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Step {i+1}/5 - Loss: {loss.item():.4f}")
            
        print("✅ Treino fake concluído com sucesso!")
        
    except RuntimeError as e:
         if "out of memory" in str(e):
             print("❌ OOM Error! Sua placa não aguentou.")
         else:
             print(f"❌ Erro no treino: {e}")
             
    print("🏁 POC Finalizada.")

if __name__ == "__main__":
    main()
