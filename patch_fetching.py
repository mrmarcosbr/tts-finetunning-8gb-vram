import os

"""
Este patch trata as seguintes mensagens de erro/warning:
1. Bug HF 404: `EntryNotFoundError` ao tentar buscar arquivos no HuggingFace devido a mudança na biblioteca `huggingface_hub` que `speechbrain` não tratava.
2. Warning Autenticação HF: `UserWarning: You are sending unauthenticated requests to the HF Hub` emitido porque o `speechbrain` passava strict `token=False` impedindo a detecção da HF_TOKEN.
"""

def apply_patch():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "speechbrain", "utils", "fetching.py")
    if not os.path.exists(file_path):
        print(f"[{__file__}] Arquivo {file_path} não encontrado. Patch pulado.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- Correção 1: EntryNotFoundError (HTTP 404) ---
    target_bug = "except HTTPError:"
    replacement_bug = "except (HTTPError, Exception) as e:\n        if '404' in str(e) or 'Not Found' in str(e):\n            return False\n        if 'EntryNotFoundError' in str(type(e)):\n            return False"
    
    if target_bug in content and replacement_bug not in content:
        content = content.replace(target_bug, replacement_bug)

    # --- Correção 2: HF_TOKEN Warning ---
    target_token = """"repo_id": source,
            "filename": filename,
            "token": use_auth_token,"""
    replacement_token = """"repo_id": source,
            "filename": filename,
            "token": use_auth_token if use_auth_token is not False else None,"""

    if target_token in content and replacement_token not in content:
        content = content.replace(target_token, replacement_token)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Sucesso: patch_fetching aplicado.")

if __name__ == "__main__":
    apply_patch()
