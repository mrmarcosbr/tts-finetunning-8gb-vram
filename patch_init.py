import os

"""
Este patch trata a seguinte mensagem de Warning:
UserWarning: Module 'speechbrain.pretrained' was deprecated, redirecting to 'speechbrain.inference'.
Isso ocorria silenciosamente devido a um hook de redirecionamento em speechbrain/__init__.py 
sempre que as rotinas do 'transformers' inspecionavam os módulos carregados.
Desativando essa rotina nós economizamos falsos positivos na suíte de log.
"""

def apply_patch():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "speechbrain", "__init__.py")
    if not os.path.exists(file_path):
        print(f"[{__file__}] Arquivo {file_path} não encontrado. Patch pulado.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    target = "make_deprecated_redirections()"
    # Importante: Garantir que não está comentando algo que já foi comentado
    target_clean = "\nmake_deprecated_redirections()\n"
    replacement = "\n# make_deprecated_redirections()  # FIX: Commented out to prevent Python's inspect.py from triggering deprecation warnings.\n"

    if target_clean in content and replacement not in content:
        content = content.replace(target_clean, replacement)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Sucesso: patch_init aplicado.")
    else:
        print("Aviso: patch_init pulado (já aplicado ou alvo não encontrado).")

if __name__ == "__main__":
    apply_patch()
