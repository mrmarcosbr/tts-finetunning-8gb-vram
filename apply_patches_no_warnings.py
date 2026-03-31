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
    """
    Localiza o diretório do ambiente virtual de forma robusta.
    1. Verifica se já está em um venv ativo (sys.prefix).
    2. Procura por qualquer pasta no diretório atual que contenha 'pyvenv.cfg'.
    3. Fallback para nomes padrão (.venv, venv, .env, env).
    """
    root = os.getcwd()

    # 1. Detectar venv ativo
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.prefix

    # 2. Escanear subpastas por pyvenv.cfg (permite nomes customizados)
    try:
        for item in os.listdir(root):
            item_path = os.path.join(root, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "pyvenv.cfg")):
                    return os.path.abspath(item_path)
    except Exception:
        pass

    # 3. Fallback para nomes comuns
    venv_folders = ['.venv', 'venv', '.env', 'env']
    for folder in venv_folders:
        path = os.path.join(root, folder)
        if os.path.isdir(path):
            return os.path.abspath(path)
            
    return None


def get_package_file_path(package_name, file_rel_path, venv_path=None):
    """
    Localiza o caminho completo de um arquivo dentro de um pacote instalado.
    Tenta primeiro via importação direta (se o venv estiver ativo) e depois via busca manual no venv_path.
    """
    import importlib.util
    import glob

    # 1. Tentar encontrar via importlib (funciona se o venv estiver ativo ou pacote no path)
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            package_root = os.path.dirname(spec.origin)
            # Normaliza o caminho do arquivo
            target_file = os.path.join(package_root, *file_rel_path.split('/'))
            if os.path.exists(target_file):
                return target_file
    except (ImportError, ValueError, AttributeError):
        pass

    # 2. Busca manual no venv_path (caso o script seja rodado de fora do venv)
    if venv_path:
        # No Windows, site-packages fica em Lib/site-packages
        # No Linux, em lib/pythonX.X/site-packages
        
        search_patterns = [
            os.path.join(venv_path, "lib", "python*", "site-packages"), # Linux
            os.path.join(venv_path, "Lib", "site-packages"),           # Windows
        ]

        for pattern in search_patterns:
            site_packages_list = glob.glob(pattern)
            for site_packages in site_packages_list:
                if os.path.exists(site_packages):
                    target_file = os.path.join(site_packages, package_name, *file_rel_path.split('/'))
                    if os.path.exists(target_file):
                        return target_file

    return None


if __name__ == "__main__":
    main()
