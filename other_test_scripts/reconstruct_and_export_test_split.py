import argparse
import json
import os
import shutil
from collections import Counter
from typing import Dict, List, Tuple

import yaml
from datasets import Dataset, load_dataset, load_from_disk


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_speaker_id(item: Dict) -> str:
    for key in ["speaker_id", "speaker", "user_id", "client_id"]:
        if key in item and item[key] is not None:
            return str(item[key])

    possible_paths = [
        item.get("__url__", ""),
        item.get("wav", {}).get("path", "") if isinstance(item.get("wav"), dict) else "",
        item.get("audio", {}).get("path", "") if isinstance(item.get("audio"), dict) else "",
    ]

    for path in possible_paths:
        p = str(path)
        if not p:
            continue

        if "LapsBM-" in p:
            filename = os.path.basename(p)
            if filename.startswith("LapsBM-"):
                return filename.split(".")[0].replace("LapsBM-", "")

        parts = p.replace("\\", "/").split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            if folder.lower() not in {"audio", "wavs", "clips"} and len(folder) > 1:
                return folder

    key = item.get("__key__", "")
    if isinstance(key, str) and key:
        token = key.split("-")[0].split("_")[0].strip()
        if token:
            return token

    return "unknown"


def extract_text(item: Dict) -> str:
    for key in ["text", "txt", "sentence", "transcription"]:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def get_audio_source_path(item: Dict) -> str:
    for key in ["audio", "wav"]:
        val = item.get(key)
        if isinstance(val, dict):
            p = val.get("path")
            if isinstance(p, str) and p.strip():
                return p
    p = item.get("__url__", "")
    if isinstance(p, str):
        return p
    return ""


def load_full_dataset(ds_cfg: Dict, local_root: str) -> Dataset:
    dataset_id = ds_cfg["dataset_id"]
    split_name = ds_cfg.get("dataset_split", "test")

    repo_name = dataset_id.split("/")[-1]
    local_ds_path = os.path.join(local_root, repo_name)
    is_local = os.path.exists(local_ds_path) and bool(os.listdir(local_ds_path)) if os.path.exists(local_ds_path) else False

    print(f"Dataset: {dataset_id} | Split: {split_name} | Local: {is_local}")

    if is_local:
        ds_obj = load_from_disk(local_ds_path)
        if isinstance(ds_obj, dict):
            ds = ds_obj.get(split_name) or next(iter(ds_obj.values()))
        else:
            ds = ds_obj
    else:
        load_kwargs = {
            "path": dataset_id,
            "split": split_name,
            "streaming": False,
        }
        if "dataset_config" in ds_cfg:
            load_kwargs["name"] = ds_cfg["dataset_config"]
        ds = load_dataset(**load_kwargs)

    items = []
    for item in ds:
        normalized = dict(item)
        if "wav" in normalized and "audio" not in normalized:
            normalized["audio"] = normalized["wav"]
        if "txt" in normalized and "text" not in normalized:
            normalized["text"] = normalized["txt"]
        items.append(normalized)

    return Dataset.from_list(items)


def compute_split_indices(all_items: List[Dict], ds_cfg: Dict) -> Tuple[List[int], List[int], List[int], set, set, set]:
    all_speakers = [extract_speaker_id(x) for x in all_items]
    counts = Counter(all_speakers)

    num_spk = ds_cfg.get("num_speakers", 1)
    max_samples = ds_cfg.get("num_samples_per_speaker", 0)
    zero_shot_split = ds_cfg.get("zero_shot_split", None)

    if num_spk == 0:
        valid_spks = set(all_speakers)
    else:
        valid_spks = set(s for s, _ in counts.most_common(num_spk))

    train_spks, val_spks, test_spks = valid_spks, set(), set()

    if zero_shot_split and len(valid_spks) > 0:
        t_spk = zero_shot_split.get("train_speakers", 28)
        v_spk = zero_shot_split.get("val_speakers", 3)
        ts_spk = zero_shot_split.get("test_speakers", 4)

        sorted_spks = sorted(list(valid_spks))
        if len(sorted_spks) >= (t_spk + v_spk + ts_spk):
            train_spks = set(sorted_spks[:t_spk])
            val_spks = set(sorted_spks[t_spk : t_spk + v_spk])
            test_spks = set(sorted_spks[t_spk + v_spk : t_spk + v_spk + ts_spk])
            valid_spks = train_spks | val_spks | test_spks
        else:
            print(
                f"Aviso: locutores insuficientes para split zero-shot "
                f"({len(sorted_spks)} < {t_spk + v_spk + ts_spk})."
            )

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    spk_added_counts = {s: 0 for s in valid_spks}

    for idx, speaker in enumerate(all_speakers):
        if speaker in valid_spks:
            if max_samples == 0 or spk_added_counts[speaker] < max_samples:
                if speaker in val_spks:
                    val_indices.append(idx)
                elif speaker in test_spks:
                    test_indices.append(idx)
                else:
                    train_indices.append(idx)
                spk_added_counts[speaker] += 1

    return train_indices, val_indices, test_indices, train_spks, val_spks, test_spks


def export_test_base(full_ds: Dataset, test_indices: List[int], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    test_ds = full_ds.select(test_indices)
    hf_out = os.path.join(output_dir, "hf_test_split")
    if os.path.exists(hf_out):
        shutil.rmtree(hf_out)
    test_ds.save_to_disk(hf_out)

    manifest_path = os.path.join(output_dir, "test_manifest.jsonl")
    txt_path = os.path.join(output_dir, "test_sentences.txt")
    spk_path = os.path.join(output_dir, "test_speakers.txt")

    copied = 0
    missing_audio = 0
    speaker_set = set()
    sentences: List[str] = []

    audio_out = os.path.join(output_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, item in enumerate(test_ds):
            speaker = extract_speaker_id(item)
            text = extract_text(item)
            source_audio = get_audio_source_path(item)
            speaker_set.add(speaker)
            if text:
                sentences.append(text)

            copied_audio = ""
            if source_audio and os.path.exists(source_audio):
                speaker_dir = os.path.join(audio_out, speaker)
                os.makedirs(speaker_dir, exist_ok=True)
                ext = os.path.splitext(source_audio)[1] or ".wav"
                dst = os.path.join(speaker_dir, f"sample_{i:04d}{ext}")
                shutil.copy2(source_audio, dst)
                copied_audio = dst
                copied += 1
            else:
                missing_audio += 1

            row = {
                "index": i,
                "speaker_id": speaker,
                "text": text,
                "source_audio": source_audio,
                "copied_audio": copied_audio,
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    unique_sentences = []
    seen = set()
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_sentences.append(s)

    with open(txt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(unique_sentences, 1):
            f.write(f"{i}. {s}\n")

    with open(spk_path, "w", encoding="utf-8") as f:
        for s in sorted(speaker_set):
            f.write(f"{s}\n")

    print("\nExport concluido:")
    print(f"- HF split: {hf_out}")
    print(f"- Manifesto: {manifest_path}")
    print(f"- Sentencas: {txt_path}")
    print(f"- Locutores: {spk_path}")
    print(f"- Audios copiados: {copied} | sem caminho local: {missing_audio}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstrói e exporta a base de teste por locutor.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_profile", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./datasets/test_split_inferencia/lapsbm_speecht5")
    parser.add_argument("--cache_dir", type=str, default="./datasets/cache_processado/speecht5_lapsbm_speecht5_16000hz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset_profile or cfg.get("settings", {}).get("default_dataset_profile", "lapsbm_speecht5")
    ds_cfg = cfg["dataset_profiles"][dataset_name]

    local_root = cfg.get("settings", {}).get("local_datasets_dir", "./datasets")

    full_ds = load_full_dataset(ds_cfg, local_root=local_root)
    all_items = [full_ds[i] for i in range(len(full_ds))]

    train_idx, val_idx, test_idx, train_spks, val_spks, test_spks = compute_split_indices(all_items, ds_cfg)

    cache_train = os.path.join(args.cache_dir, "train")
    cache_val = os.path.join(args.cache_dir, "val")
    cache_train_len = -1
    cache_val_len = -1
    if os.path.exists(cache_train):
        cache_train_len = len(load_from_disk(cache_train))
    if os.path.exists(cache_val):
        cache_val_len = len(load_from_disk(cache_val))

    print("\nResumo reconstruido (mesma logica do treino):")
    print(f"- Locutores treino: {len(train_spks)}")
    print(f"- Locutores validacao: {len(val_spks)}")
    print(f"- Locutores teste: {len(test_spks)}")
    print(f"- Amostras treino: {len(train_idx)}")
    print(f"- Amostras validacao: {len(val_idx)}")
    print(f"- Amostras teste: {len(test_idx)}")
    print(f"- Cache train (processado): {cache_train_len}")
    print(f"- Cache val (processado): {cache_val_len}")

    train_val_spks = set(train_spks) | set(val_spks)
    missing_vs_train_val = sorted(set(test_spks) - train_val_spks)
    print("\nLocutores faltantes em treino+validacao (candidatos da base de teste):")
    print(", ".join(missing_vs_train_val) if missing_vs_train_val else "(nenhum)")

    split_info = {
        "dataset_profile": dataset_name,
        "train_speakers": sorted(list(train_spks)),
        "val_speakers": sorted(list(val_spks)),
        "test_speakers": sorted(list(test_spks)),
        "counts": {
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "cache_train_samples": cache_train_len,
            "cache_val_samples": cache_val_len,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "reconstructed_split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    export_test_base(full_ds, test_idx, args.output_dir)


if __name__ == "__main__":
    main()
