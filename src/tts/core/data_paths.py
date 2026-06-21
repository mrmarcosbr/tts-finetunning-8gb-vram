"""Resolve data directory layout from ``settings`` in config_train.yaml."""
from __future__ import annotations

import os
from typing import Any, Dict


def resolve_data_settings(settings: Dict[str, Any] | None) -> Dict[str, Any]:
    settings = settings or {}
    data_dir = settings.get("data_dir")
    if data_dir:
        return {
            "data_dir": data_dir,
            "raw_dir": settings.get("data_raw_dir", os.path.join(data_dir, "raw")),
            "repository_dir": settings.get("data_repository_dir", os.path.join(data_dir, "repository")),
            "cache_dir": settings.get("data_cache_dir", os.path.join(data_dir, "cache")),
            "exports_dir": settings.get("data_exports_dir", os.path.join(data_dir, "exports")),
            "legacy_layout": False,
        }

    legacy_root = settings.get("local_datasets_dir", "./datasets")
    return {
        "data_dir": legacy_root,
        "raw_dir": settings.get("data_raw_dir", "./meus_datasets"),
        "repository_dir": legacy_root,
        "cache_dir": settings.get("data_cache_dir", os.path.join(legacy_root, "cache_processado")),
        "exports_dir": settings.get("data_exports_dir", os.path.join(legacy_root, "test_split_inferencia")),
        "legacy_layout": True,
    }


def repository_dataset_path(settings: Dict[str, Any] | None, dataset_id: str) -> str:
    repo_name = dataset_id.split("/")[-1]
    paths = resolve_data_settings(settings)
    return os.path.join(paths["repository_dir"], repo_name)


def exports_profile_dir(settings: Dict[str, Any] | None, profile_name: str) -> str:
    return os.path.join(resolve_data_settings(settings)["exports_dir"], profile_name)


def exports_audio_root(settings: Dict[str, Any] | None, profile_name: str) -> str:
    return os.path.join(exports_profile_dir(settings, profile_name), "audio")


def exports_hf_test_split_path(settings: Dict[str, Any] | None, profile_name: str) -> str:
    return os.path.join(exports_profile_dir(settings, profile_name), "hf_test_split")


def training_cache_dir(settings: Dict[str, Any] | None, cache_name: str) -> str:
    return os.path.join(resolve_data_settings(settings)["cache_dir"], cache_name)


def speechbrain_cache_dir(settings: Dict[str, Any] | None, encoder_id: str) -> str:
    paths = resolve_data_settings(settings)
    return os.path.join(paths["data_dir"], ".speechbrain_cache", encoder_id.replace("/", "__"))
