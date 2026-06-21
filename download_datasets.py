import os
import sys
import argparse
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Carrega o HF_TOKEN do arquivo .env ativo no seu projeto
load_dotenv()

def download_dataset(repo_id, local_base="meus_datasets"):
    """
    Faz o download do repositório inteiro de um dataset do HuggingFace 
    para uma pasta local, visando leitura 100% offline.
    """
    repo_name = repo_id.split('/')[-1]
    local_dir = os.path.join(local_base, repo_name)
    
    print(f"\n📥 Iniciando o download local permanente de: '{repo_id}'")
    print(f"📁 Destino físico: {local_dir}")
    print("⏳ Isso pode demorar bastante caso o dataset possua horas de áudio...\n")
    
    try:
        # Faz o espelhamento do Hub garantindo que continue de onde parou em caso de queda de rede
        snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            local_dir=local_dir,
            resume_download=True
        )
        print(f"\n✅ Download concluído com sucesso e isolado em sua máquina: {local_dir}!")
    except Exception as e:
        print(f"\n❌ Erro ao baixar o dataset '{repo_id}':\n{e}")
        print("\n💡 Lembrete de permissão:")
        print("Se receber '401 Unauthorized', lembre-se que Gated Datasets exigem que você:")
        print("1. Logue com o seu usuário em https://huggingface.co/datasets/" + repo_id)
        print("2. Clique em 'Agree/Acknowledge' para aceitar os termos de licença de uso do áudio.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilitário para converter Datasets Cloud em Offline Locals")
    parser.add_argument("--repo", type=str, help="ID completo do repositório Hugging Face (ex: falabrasil/lapsbm)")
    parser.add_argument("--all", action="store_true", help="Baixa silenciosamente todos os repositórios comuns (lapsbm e pt-br_char)")
    
    args = parser.parse_args()

    # Predefinições comuns do seu projeto TCC
    default_datasets = [
        "firstpixel/pt-br_char",
        "falabrasil/lapsbm"
    ]

    if args.repo:
        download_dataset(args.repo)
    elif args.all:
        for repo in default_datasets:
            print("=" * 60)
            download_dataset(repo)
    else:
        print("⚠️ Modo de uso incorreto. Use a flag --repo ou --all.")
        print("\nExemplos válidos:")
        print("  🔹 Baixar um específico: python download_datasets.py --repo falabrasil/lapsbm")
        print("  🔹 Baixar os dois comuns do TCC em fila: python download_datasets.py --all")
