import os

from apply_patches_no_warnings import get_venv_path, get_package_file_path

"""
Este patch trata a seguinte mensagem de Warning:
UserWarning: torchaudio._backend.list_audio_backends has been deprecated. This deprecation is part of a large refactoring effort...
O core do SpeechBrain usava lógica de backend agora legada para listar extensões instaladas. O patch bypassa essa checagem desatualizada para evitar o Warning nas bibliotecas C++ da HuggingFace/PyTorch.
"""

def apply_patch(venv_dir):
    # Atualiza o arquivo speechbrain/utils/torch_audio_backend.py
    file_path = get_package_file_path("speechbrain", "utils/torch_audio_backend.py", venv_dir)
    if not file_path or not os.path.exists(file_path):
        print(f"[{os.path.basename(__file__)}] Arquivo não encontrado (speechbrain/utils/torch_audio_backend.py). Patch pulado.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    target = """    elif torchaudio_major >= 2 and torchaudio_minor >= 1:
        available_backends = torchaudio.list_audio_backends()

        if len(available_backends) == 0:"""

    replacement = """    elif torchaudio_major >= 2 and torchaudio_minor >= 1:
        # FIX: Avoid deprecated torchaudio.list_audio_backends() to prevent UserWarning
        available_backends = []
        try:
            import soundfile
            available_backends.append("soundfile")
        except ImportError:
            pass

        if len(available_backends) == 0:"""

    if target in content and replacement not in content:
        content = content.replace(target, replacement)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Sucesso: patch_torchaudio (audio backend lib) aplicado.")
    else:
        print("Aviso: patch_torchaudio pulado (já aplicado ou alvo não encontrado).")

if __name__ == "__main__":
    venv_dir = get_venv_path()
    if venv_dir:
        apply_patch(venv_dir)
    else:
        print(f"[{os.path.basename(__file__)}] Erro: Ambiente virtual não encontrado.")
