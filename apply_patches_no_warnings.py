import sys
import os

"""
Runner Principal: Aplica de maneira consolidada e relativa todos os scripts de patching da biblioteca na .venv atual.
Assim, é possível corrigir os problemas descritos abaixo instalando as bibliotecas oficiais e imediatamente acionando este orquestrador de correções em qualquer computador.
"""

def main():
    print("Iniciando bateria de correções nativas em bibliotecas (.venv) do SpeechT5/SpeechBrain...")
    
    # 1. Patch de Fetching (HF 404 e Token Logging)
    try:
        import patch_fetching
        patch_fetching.apply_patch()
    except Exception as e:
        print(f"Erro ao aplicar patch_fetching: {e}")

    # 2. Patch do Autocast (GPU Amp TypeError Deprecation)
    try:
        import patch_autocast
        patch_autocast.apply_patch()
    except Exception as e:
        print(f"Erro ao aplicar patch_autocast: {e}")

    # 3. Patch do Backend de Audio (Torchaudio API Deprecation)
    try:
        import patch_torchaudio
        patch_torchaudio.apply_patch()
    except Exception as e:
        print(f"Erro ao aplicar patch_torchaudio: {e}")

    # 4. Patch do Init do SpeechBrain (Inspect Pretrained Module Deprecation Warning)
    try:
        import patch_init
        patch_init.apply_patch()
    except Exception as e:
        print(f"Erro ao aplicar patch_init: {e}")

    print("=== Concluído! Todos os patches foram injetados no ambiente atual com sucesso. ===")

if __name__ == "__main__":
    main()
