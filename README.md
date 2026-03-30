# tts-finetunning-8gb-vram

- Testado com python 3.11.9 (Windows) e 3.12.3 (Linux)
- Drivers CUDA da NVIDA para rodar pytorch usando a GPU

# Cria enviromment
- python -m venv venv
- source venv/bin/activate

# Instala dependências
- pip install -r requirements.txt

# Aplicar Patches e Correções Nativas (Recomendado)
Algumas bibliotecas como SpeechBrain e HuggingFace podem apresentar alertas conflitantes (ex: warnings do `autocast`, bugs de `HF_TOKEN` suprimido e erros falsos 404 HTTP). Para ter um ambiente de console limpo e livre de interrupções falsas, execute os scripts de correção que injetam os reparos diretamente na sua subpasta `.venv` recém instalada:
- python apply_patches_no_warnings.py

# Treina Modelo (Fine Tunning)
- python train_exhaustive.py

Realiza o treinamento de modelo existente em inglês com novos áudios transcritos em português. O objetivo deste fine tunning é permitir que um modelo construído puramente para responder audios em inglês, consigo também responder em português.

- Obs - Levou em torno de 1 hora para treinar com uma RTX 4070 Mobile de 8GB de VRAM (Notebook)
- O arquivo final com os pesos do modelo ficou com 32 GB.

# Realiza Inferência (Converte texto em português para Áudio com o Modelo retreinado)

- python test_inference_exhaustive.py
