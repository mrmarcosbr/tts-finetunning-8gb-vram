"""PyTorch compatibility shims applied before model load."""
from __future__ import annotations

import importlib

import torch


def apply_torch_load_compat() -> None:
    """Allow numpy arrays in checkpoints (PyTorch 2.6+ weights_only default)."""
    _original_load = torch.load

    def safe_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = safe_load


def import_distributed_tensor_early() -> None:
    try:
        importlib.import_module("torch.distributed.tensor")
    except ImportError:
        pass


def bypass_transformers_torch_version_guard() -> None:
    """Desativa a checagem de versão do torch (CVE-2025-32434) no transformers.

    A partir de certas versões, o ``transformers`` recusa-se a chamar ``torch.load``
    em checkpoints ``.bin``/``.pth`` quando detecta torch < 2.6, levantando
    ``ValueError: ... we now require users to upgrade torch to at least v2.6``.

    Quando o torch do ambiente é reutilizado (ex.: pod RunPod) e é < 2.6, este
    bypass permite carregar os checkpoints (que aqui são dicts de tensores simples)
    sem reinstalar o torch. Deve ser chamado ANTES de importar/usar
    ``from_pretrained``.
    """
    import sys

    try:
        importlib.import_module("transformers")
    except Exception:
        return

    def _patched_ge(version, *args, **kwargs):
        # Garante que qualquer comparação contra 2.6 passe.
        if str(version).lstrip("v").startswith("2.6"):
            return True
        return _original_ge(version, *args, **kwargs)

    _original_ge = None
    try:
        import transformers.utils.import_utils as iu

        _original_ge = getattr(iu, "is_torch_greater_or_equal", None)
        if callable(_original_ge):
            iu.is_torch_greater_or_equal = _patched_ge
    except Exception:
        pass

    # Força a importação dos submódulos onde o booleano é calculado no import,
    # para que apareçam em sys.modules e sejam corrigidos abaixo.
    for sub in ("transformers.pytorch_utils", "transformers.modeling_utils"):
        try:
            importlib.import_module(sub)
        except Exception:
            pass

    # Cada submódulo pode ter importado a função e/ou o booleano para o seu
    # próprio namespace; é preciso sobrescrever em todos eles.
    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers") or module is None:
            continue
        if hasattr(module, "is_torch_greater_or_equal_than_2_6"):
            try:
                setattr(module, "is_torch_greater_or_equal_than_2_6", True)
            except Exception:
                pass
        if (
            _original_ge is not None
            and getattr(module, "is_torch_greater_or_equal", None) is _original_ge
        ):
            try:
                setattr(module, "is_torch_greater_or_equal", _patched_ge)
            except Exception:
                pass
