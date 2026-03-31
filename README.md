# tts-finetunning-8gb-vram

- Testado com python 3.11.9 (Windows) e 3.12.3 (Linux)
- Drivers CUDA da NVIDA para rodar pytorch usando a GPU

# Cria enviromment
- python -m venv venv
- source venv/bin/activate

# Instala dependências
- pip install -r requirements.txt

# Token Hugging Face
- Crie uma conta na Hugging Face - https://huggingface.co/settings/tokens - e gere um Token de Acesso (permissão de leitura) para fazer o Download os Modelos pré-treinados necessários para o fine-tuning e inferência.
- Configure a variável de ambiente `HF_TOKEN` com o token no arquivo .env ou via terminal/variáveis de ambiente do sistema operacional. 
 
# Aplicar Patches e Correções Nativas 
Algumas bibliotecas como SpeechBrain e HuggingFace podem apresentar erros e alertas conflitantes (ex: warnings do `autocast`, bugs de `HF_TOKEN` suprimido e erros falsos 404 HTTP). Para ter um ambiente de console limpo e livre de interrupções falsas, execute os scripts de correção que injetam os reparos diretamente na sua subpasta `venv` recém instalada (somente depois de já ter instalado os pacotes com pip install):
- python apply_patches_no_warnings.py

# Configuração do Treinamento/Fine-Tuning
Verificar o arquivo config.yaml que contem as configurações de treinamento para cada profile (ex: "cuda_16gb", "macbook", "cpu").
- config.yaml

# Treina Modelo (Fine Tunning)
Usando Detecção Automática de Hardware: 
- python train_exhaustive.py 

Usando profile específico (ex: "cuda_16gb" ou "macbook" ou "cpu"):
- python train_exhaustive.py --profile "macbook"

Realiza o treinamento de modelo existente em inglês com novos áudios transcritos em português. O objetivo deste fine tunning é permitir que um modelo construído puramente para responder audios em inglês, consigo também responder em português.

- Levou em torno de 45 Minutos para treinar com uma RTX 5060 TI 16GB de VRAM (Ryzen 9950X3D - Desktop)
- Levou em torno de 1 hora para treinar com uma RTX 4070 Mobile de 8GB de VRAM (Notebook)
- O arquivo final com os pesos do modelo (Apenas último checkpoint) ficou com 500 MB.

# Realiza Inferência (Converte texto em português para Áudio com o Modelo retreinado)
Usando Detecção Automática de Hardware: 
- python test_inference_exhaustive.py

Usando profile específico e modelo específico
- python train_exhaustive.py --profile "cuda_16gb" --model_name "SpeechT5-2026-03-30-23-33-05"
