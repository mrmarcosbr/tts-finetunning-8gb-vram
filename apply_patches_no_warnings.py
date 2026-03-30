import sys
import os

"""
Runner Principal: Aplica de maneira consolidada e relativa todos os scripts de patching da biblioteca na .venv atual.
Assim, é possível corrigir os problemas descritos abaixo instalando as bibliotecas oficiais e imediatamente acionando este orquestrador de correções em qualquer computador.
"""

def main():
    venv_dir = get_venv_path()
    if not venv_dir:
        print(f"Ambiente virtual não encontrado.")
        return

    print(f"Iniciando bateria de correções nativas em bibliotecas ({venv_dir}) do SpeechT5/SpeechBrain...")

    # 1. Patch de Fetching (HF 404 e Token Logging)
    try:
        import patch_fetching
        patch_fetching.apply_patch(venv_dir)
    except Exception as e:
        print(f"Erro ao aplicar patch_fetching: {e}")

    # 2. Patch do Autocast (GPU Amp TypeError Deprecation)
    try:
        import patch_autocast
        patch_autocast.apply_patch(venv_dir)
    except Exception as e:
        print(f"Erro ao aplicar patch_autocast: {e}")

    # 3. Patch do Backend de Audio (Torchaudio API Deprecation)
    try:
        import patch_torchaudio
        patch_torchaudio.apply_patch(venv_dir)
    except Exception as e:
        print(f"Erro ao aplicar patch_torchaudio: {e}")

    # 4. Patch do Init do SpeechBrain (Inspect Pretrained Module Deprecation Warning)
    try:
        import patch_init
        patch_init.apply_patch(venv_dir)
    except Exception as e:
        print(f"Erro ao aplicar patch_init: {e}")

    print("=== Concluído! Todos os patches foram injetados no ambiente atual com sucesso. ===")


def get_venv_path():
    # Pastas comuns de ambiente virtual
    venv_folders = ['.venv', 'venv', '.env', 'env']
    root = os.getcwd()

    for folder in venv_folders:
        path = os.path.join(root, folder)
        
        # Verifica se o diretório existe
        if os.path.isdir(path):
            return folder # Retorna o caminho completo da primeira que encontrar
            
    return None # Retorna None se nenhuma existir


if __name__ == "__main__":
    main()
