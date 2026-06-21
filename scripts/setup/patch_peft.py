import os
import torch

from apply_patches_no_warnings import get_venv_path, get_package_file_path

"""
Este patch trata o seguinte erro no PyTorch 2.8.0+cu128:
AttributeError: module 'torch.distributed' has no attribute 'tensor'
O PEFT 0.19.1 assume que se a versão do torch for >= 2.5.0, o módulo distributed.tensor sempre existe.
"""

def apply_patch(venv_dir):
    file_path = get_package_file_path("peft", "tuners/tuners_utils.py", venv_dir)
    
    if not file_path or not os.path.exists(file_path):
        print(f"[{os.path.basename(__file__)}] Arquivo não encontrado (peft/tuners/tuners_utils.py). Patch pulado.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    target = '_torch_supports_dtensor = version.parse(torch.__version__) >= version.parse("2.5.0")'
    replacement = """try:
    import torch.distributed.tensor
    _torch_supports_dtensor = True
except (ImportError, AttributeError):
    _torch_supports_dtensor = False"""

    if target in content and replacement not in content:
        content = content.replace(target, replacement)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Sucesso: patch_peft aplicado.")
    elif replacement in content:
        print("Aviso: patch_peft já aplicado.")
    else:
        print("Aviso: patch_peft pulado (alvo não encontrado).")


if __name__ == "__main__":
    venv_dir = get_venv_path()
    if venv_dir:
        apply_patch(venv_dir)
    else:
        print(f"[{os.path.basename(__file__)}] Erro: Ambiente virtual não encontrado.")
