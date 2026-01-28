# tts-finetunning-8gb-vram

Usar python 3.12.3 ou anterior

# Cria enviromment
python -m venv venv
source venv/bin/activate

# Instala dependências
pip install -r requirements.txt

# Treina Modelo
python train_exhaustive.py

- Obs - Levou em torno de 1 hora para treinar com uma RTX 4070 Mobile de 8GB de VRAM (Notebook)
- O arquivo final com os pesos do modelo ficou com 32 GB.
