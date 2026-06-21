import os

from apply_patches_no_warnings import get_venv_path, get_package_file_path

"""
Este patch trata as seguintes mensagens de erro/warning:
1. Bug HF 404: `EntryNotFoundError` ao tentar buscar arquivos no HuggingFace devido a mudança na biblioteca `huggingface_hub` que `speechbrain` não tratava.
2. Warning Autenticação HF: `UserWarning: You are sending unauthenticated requests to the HF Hub` emitido porque o `speechbrain` passava strict `token=False` impedindo a detecção da HF_TOKEN.
"""

def apply_patch(venv_dir):
    file_path = get_package_file_path("speechbrain", "utils/fetching.py", venv_dir)
    if not file_path or not os.path.exists(file_path):
        print(f"[{os.path.basename(__file__)}] Arquivo não encontrado (speechbrain/utils/fetching.py). Patch pulado.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    applied_any = False

    # --- Correção 1: EntryNotFoundError (HTTP 404) ---
    target_bug = """    except HTTPError as e:
        if "404 Client Error" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise"""
    replacement_bug = """    except (HTTPError, Exception) as e:
        if "404" in str(e) or "Not Found" in str(e) or "EntryNotFoundError" in str(type(e)):
            raise ValueError(f"File not found on HF hub: {e}") from e
        raise"""
    
    if target_bug in content and replacement_bug not in content:
        content = content.replace(target_bug, replacement_bug)
        applied_any = True

    # --- Correção 2: HF_TOKEN Warning & TypeError ---
    target_token_orig = '"use_auth_token": use_auth_token,'
    target_token_old_patch = '"use_auth_token": use_auth_token if use_auth_token is not False else None,'
    replacement_token = '"token": use_auth_token if use_auth_token is not False else None,'

    if target_token_orig in content:
        content = content.replace(target_token_orig, replacement_token)
        applied_any = True
    elif target_token_old_patch in content:
        content = content.replace(target_token_old_patch, replacement_token)
        applied_any = True

    if applied_any:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Sucesso: patch_fetching aplicado.")
    else:
        print("Aviso: patch_fetching pulado (já aplicado ou alvo não encontrado).")

if __name__ == "__main__":
    venv_dir = get_venv_path()
    if venv_dir:
        apply_patch(venv_dir)
    else:
        print(f"[{os.path.basename(__file__)}] Erro: Ambiente virtual não encontrado.")
