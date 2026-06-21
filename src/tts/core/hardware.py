"""Hardware profile detection and SpeechBrain device strings."""
from __future__ import annotations

import torch


def detect_hardware_profile(_full_cfg=None) -> str:
    """Autodetect best hardware profile name (cuda_16gb, cuda_8gb, macbook, cpu)."""
    print("🔍 Autodetectando hardware...")

    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   🖥️  GPU NVIDIA Detectada: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB VRAM)")
        if total_vram >= 14.0:
            return "cuda_16gb"
        return "cuda_8gb"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("   🍎 Apple Silicon Detectado (MPS Backend)")
        return "macbook"

    print("   ⚠️ Nenhuma GPU de alta performance detectada. Usando CPU.")
    return "cpu"


def device_for_speechbrain(device: str) -> str:
    raw = str(device or "cpu").strip()
    if raw.lower() == "cuda":
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    return raw
