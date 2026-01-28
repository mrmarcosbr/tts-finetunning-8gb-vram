# tts-finetunning-8gb-vram

- Usar python 3.12.3 ou anterior
- Drivers CUDA da NVIDA para rodar pytorch usando a GPU

# Cria enviromment
- python -m venv venv
- source venv/bin/activate

# Instala dependências
- pip install -r requirements.txt

# Treina Modelo (Fine Tunning)
- python train_exhaustive.py

Realiza o treinamento de modelo existente em inglês com novos áudios transcritos em português. O objetivo deste fine tunning é permitir que um modelo construído puramente para responder audios em inglês, consigo também responder em português.

- Obs - Levou em torno de 1 hora para treinar com uma RTX 4070 Mobile de 8GB de VRAM (Notebook)
- O arquivo final com os pesos do modelo ficou com 32 GB.

# Realiza Inferência (Converte texto em português para Áudio com o Modelo retreinado)

- python test_inference_exhaustive.py
