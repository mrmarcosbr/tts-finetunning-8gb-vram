import os

from apply_patches_no_warnings import get_venv_path, get_package_file_path

"""
Este patch trata a seguinte mensagem de Warning:
FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
A nova versão do decorador exige parâmetros contextuais baseados no dispositivo em vez do fallback antigo de GPU impositivo.
"""

def apply_patch(venv_dir):
    file_path = get_package_file_path("speechbrain", "utils/autocast.py", venv_dir)
    
    if not file_path or not os.path.exists(file_path):
        print(f"[{os.path.basename(__file__)}] Arquivo não encontrado (speechbrain/utils/autocast.py). Patch pulado.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    target = "    wrapped_fwd = torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)"
    replacement = """    if hasattr(torch.amp, "custom_fwd"):
        wrapped_fwd = torch.amp.custom_fwd(device_type="cuda", cast_inputs=cast_inputs)(fwd)
    else:
        wrapped_fwd = torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)"""

    if target in content and replacement not in content:
        content = content.replace(target, replacement)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Sucesso: patch_autocast aplicado.")
    else:
        print("Aviso: patch_autocast pulado (já aplicado ou alvo não encontrado).")


if __name__ == "__main__":
    venv_dir = get_venv_path()
    if venv_dir:
        apply_patch(venv_dir)
    else:
        print(f"[{os.path.basename(__file__)}] Erro: Ambiente virtual não encontrado.")

