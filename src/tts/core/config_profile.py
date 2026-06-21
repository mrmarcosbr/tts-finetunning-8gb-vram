"""Resolve perfis unificados (ou legado hardware + model + dataset) do config_train.yaml."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

_HW_KEYS = (
    "device",
    "batch_size",
    "gradient_accumulation_steps",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "output_dir",
)
_MODEL_KEYS = (
    "id",
    "model_type",
    "sampling_rate",
    "vocoder_id",
    "speaker_encoder_id",
    "pitch_loss_alpha",
    "f0_input_mode",
    "lora",
    "language",
)
_DATASET_KEYS = (
    "dataset_id",
    "dataset_split",
    "dataset_config",
    "num_speakers",
    "num_samples_per_speaker",
    "zero_shot_split",
    "model_overrides",
    "language",
)


@dataclass(frozen=True)
class ResolvedProfile:
    name: str
    hw_cfg: Dict[str, Any]
    model_cfg: Dict[str, Any]
    ds_cfg: Dict[str, Any]
    model_type: str

    @property
    def train_params(self) -> Dict[str, Any]:
        tr = self.ds_cfg.get("training")
        return tr if isinstance(tr, dict) else {}


def _pick(cfg: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    return {k: cfg[k] for k in keys if k in cfg}


def merge_dataset_model_overrides(model_cfg: Dict[str, Any], ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mescla ``model_overrides`` do perfil de dataset sobre cópia do modelo base."""
    base = copy.deepcopy(model_cfg)
    overrides = ds_cfg.get("model_overrides")
    if not isinstance(overrides, dict):
        return base
    for key, val in overrides.items():
        cur = base.get(key)
        if isinstance(val, dict) and isinstance(cur, dict):
            merged = dict(cur)
            merged.update(val)
            base[key] = merged
        else:
            base[key] = val
    return base


def _resolve_profile_name(
    full_cfg: Dict[str, Any],
    profile_name: Optional[str],
    dataset_name: Optional[str],
) -> str:
    settings = full_cfg.get("settings") or {}
    aliases = settings.get("profile_aliases") or {}
    requested = profile_name or dataset_name or settings.get("default_profile")
    if not requested:
        raise KeyError("Nenhum perfil indicado e settings.default_profile ausente no YAML.")
    return str(aliases.get(requested, requested))


def _unpack_unified_profile(profile_cfg: Dict[str, Any], name: str) -> ResolvedProfile:
    model_type = str(profile_cfg.get("model_type", "speecht5"))
    hw_cfg = _pick(profile_cfg, _HW_KEYS)
    model_cfg = _pick(profile_cfg, _MODEL_KEYS)
    if "model_type" in model_cfg:
        model_cfg = {k: v for k, v in model_cfg.items() if k != "model_type"}
    ds_cfg = _pick(profile_cfg, _DATASET_KEYS)
    ds_cfg["model_type"] = model_type
    training = profile_cfg.get("training")
    if isinstance(training, dict):
        ds_cfg["training"] = copy.deepcopy(training)
    model_cfg = merge_dataset_model_overrides(model_cfg, ds_cfg)
    return ResolvedProfile(name=name, hw_cfg=hw_cfg, model_cfg=model_cfg, ds_cfg=ds_cfg, model_type=model_type)


def _resolve_legacy(
    full_cfg: Dict[str, Any],
    profile_name: Optional[str],
    dataset_name: Optional[str],
    detect_hardware,
) -> ResolvedProfile:
    settings = full_cfg.get("settings") or {}
    ds_name = dataset_name or settings.get("default_dataset_profile", "lapsbm_speecht5")
    dataset_profiles = full_cfg.get("dataset_profiles") or {}
    if ds_name not in dataset_profiles:
        raise KeyError(f"Perfil de dataset '{ds_name}' não encontrado.")
    ds_cfg = dataset_profiles[ds_name]
    model_type = str(ds_cfg.get("model_type", "speecht5"))
    models = full_cfg.get("models") or {}
    if model_type not in models:
        raise KeyError(f"Modelo '{model_type}' não definido no YAML.")
    model_cfg = merge_dataset_model_overrides(models[model_type], ds_cfg)
    hw_profiles = full_cfg.get("hardware_profiles") or {}
    if profile_name:
        hw_name = profile_name
    elif detect_hardware is not None:
        hw_name = detect_hardware()
    else:
        hw_name = "cpu"
    hw_cfg = hw_profiles.get(hw_name) or hw_profiles.get("cpu") or {}
    return ResolvedProfile(name=ds_name, hw_cfg=hw_cfg, model_cfg=model_cfg, ds_cfg=ds_cfg, model_type=model_type)


def resolve_training_profile(
    full_cfg: Dict[str, Any],
    *,
    profile_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    detect_hardware=None,
) -> ResolvedProfile:
    """
    Formato novo: ``profiles.<nome>`` (hardware + model + dataset + training).
    Formato legado: ``hardware_profiles`` + ``models`` + ``dataset_profiles``.
    """
    profiles = full_cfg.get("profiles")
    if profiles:
        name = _resolve_profile_name(full_cfg, profile_name, dataset_name)
        if name not in profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise KeyError(f"Perfil '{name}' não encontrado. Disponíveis: {available}")
        return _unpack_unified_profile(profiles[name], name)
    return _resolve_legacy(full_cfg, profile_name, dataset_name, detect_hardware)
