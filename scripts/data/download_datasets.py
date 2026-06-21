import os
import sys
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import yaml

# Repo root on sys.path for repo_bootstrap / tts.*
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
for p in (_REPO_ROOT, _SRC):
    ps = str(p)
    if p.is_dir() and ps not in sys.path:
        sys.path.insert(0, ps)
import repo_bootstrap  # noqa: E402
from tts.core.data_paths import resolve_data_settings

load_dotenv()


def _default_raw_dir(config_path: str = "configs/config_train.yaml") -> str:
    try:
        with open(_REPO_ROOT / config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return resolve_data_settings(cfg.get("settings")).get("raw_dir", "./data/raw")
    except OSError:
        return "./data/raw"


def download_dataset(repo_id: str, local_base: str) -> None:
    """
    Espelha o repositório Hugging Face em ``data/raw/<repo_name>/`` (offline permanente).
    Formato bruto do Hub — distinto de ``data/repository/`` (save_to_disk do treino).
    """
    repo_name = repo_id.split("/")[-1]
    local_dir = os.path.join(local_base, repo_name)

    print(f"\n📥 Iniciando o download local permanente de: '{repo_id}'")
    print(f"📁 Destino físico: {local_dir}")
    print("⏳ Isso pode demorar bastante caso o dataset possua horas de áudio...\n")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            resume_download=True,
        )
        print(f"\n✅ Download concluído com sucesso e isolado em sua máquina: {local_dir}!")
    except Exception as e:
        print(f"\n❌ Erro ao baixar o dataset '{repo_id}':\n{e}")
        print("\n💡 Lembrete de permissão:")
        print("Se receber '401 Unauthorized', lembre-se que Gated Datasets exigem que você:")
        print("1. Logue com o seu usuário em https://huggingface.co/datasets/" + repo_id)
        print("2. Clique em 'Agree/Acknowledge' para aceitar os termos de licença de uso do áudio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Espelhar datasets Hugging Face em data/raw/ (offline)")
    parser.add_argument("--repo", type=str, help="ID completo do repositório (ex: falabrasil/lapsbm)")
    parser.add_argument("--all", action="store_true", help="Baixa lapsbm e pt-br_char em sequência")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_train.yaml",
        help="YAML com settings.data_raw_dir (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Sobrescreve data_raw_dir do config",
    )
    args = parser.parse_args()

    raw_dir = args.output_dir or _default_raw_dir(args.config)
    os.makedirs(raw_dir, exist_ok=True)

    default_datasets = [
        "firstpixel/pt-br_char",
        "falabrasil/lapsbm",
    ]

    if args.repo:
        download_dataset(args.repo, raw_dir)
    elif args.all:
        for repo in default_datasets:
            print("=" * 60)
            download_dataset(repo, raw_dir)
    else:
        print("⚠️ Modo de uso incorreto. Use a flag --repo ou --all.")
        print("\nExemplos válidos:")
        print("  🔹 Baixar um específico: python scripts/data/download_datasets.py --repo falabrasil/lapsbm")
        print("  🔹 Baixar os dois comuns do TCC: python scripts/data/download_datasets.py --all")
