import argparse
import json
import os
import re
import shutil
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml
import librosa
from datasets import Dataset, load_dataset, load_from_disk


def output_dir_contains_dataset_profile_marker(output_dir: str, dataset_profile: str) -> bool:
    """
    Normalização PCM (mono) para a taxa do modelo apenas quando o caminho inclui o identificador
    do dataset_profile no config (ex.: .../lapsbm_speecht5/... ou .../lapsbm_fastspeech2/...).
    """
    if not dataset_profile or not str(dataset_profile).strip():
        return False
    ab = os.path.normcase(os.path.abspath(os.path.expanduser(output_dir)))
    needle = dataset_profile.strip().lower()
    return needle in ab.replace("\\", "/").lower()


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


def source_audio_basename(item: Dict) -> str:
    """Nome do ficheiro de audio no dataset (ex. LapsBM_0601.wav), se conhecido."""
    p = get_audio_source_path(item)
    if not p.strip():
        return ""
    return os.path.basename(p.replace("/", os.sep))


def sanitize_wav_filename_stem(raw: str, fallback_index: int) -> str:
    s = raw.strip().replace(os.sep, "_").replace("/", "_")
    s = re.sub(r'[<>:\"|?*\x00-\x1f]', "_", s)
    s = s.strip(". ") or ""
    return s[:200] if s else f"sample_{fallback_index:04d}"


def stem_for_exported_wav(item: Dict, fallback_index: int) -> str:
    """
    Identifica o clip a partir do path HF (basename sem extensão) ou __key__;
    permite relacionar sample_ em pastas com LapsBM_0601.wav no tar.gz original.
    """
    base = source_audio_basename(item)
    if base:
        stem, _ = os.path.splitext(base)
        return sanitize_wav_filename_stem(stem, fallback_index)
    ky = item.get("__key__")
    if isinstance(ky, str) and ky.strip():
        stem, _ = os.path.splitext(ky.strip().replace("\\", "/").split("/")[-1])
        return sanitize_wav_filename_stem(stem, fallback_index)
    return f"sample_{fallback_index:04d}"


def allocate_unique_media_filename(stem: str, ext: str, used_filenames: set) -> str:
    """Nome único na pasta do locutor (ex. LapsBM_0601.wav ou LapsBM_0601.flac)."""
    if not ext.startswith("."):
        ext = "." + ext
    base = f"{stem}{ext}"
    if base not in used_filenames:
        used_filenames.add(base)
        return base
    k = 2
    while True:
        cand = f"{stem}__{k}{ext}"
        if cand not in used_filenames:
            used_filenames.add(cand)
            return cand
        k += 1


def _audio_feature_dict(item: Dict) -> Dict:
    for key in ("audio", "wav"):
        v = item.get(key)
        if isinstance(v, dict) and v.get("array") is not None:
            return v
    return {}


def write_audio_wav_from_hf_item(item: Dict, dst_path: str, target_sr: Optional[int]) -> bool:
    """
    Grava PCM WAV mono a partir dos arrays no dict audio/wav (Hugging Face Datasets).
    Se target_sr for definido e diferente do SR nativo: reamostra até à taxa alvo do modelo.
    Se target_sr for None: grava ao SR original do exemplo (sem reamostragem).
    """
    au = _audio_feature_dict(item)
    arr = au.get("array")
    if arr is None:
        return False
    raw_sr = au.get("sampling_rate")
    if raw_sr is not None:
        sr = int(raw_sr)
    elif target_sr is not None:
        sr = int(target_sr)
    else:
        sr = 16000
    x = np.asarray(arr)
    if x.ndim == 1:
        mono = x.astype(np.float32)
    elif x.ndim == 2:
        mono = np.mean(x.astype(np.float32), axis=-1)
    else:
        return False

    mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
    if target_sr is None:
        out_sr = sr
    else:
        out_sr = int(target_sr)
        if sr != out_sr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=out_sr).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    sf.write(dst_path, mono, out_sr, subtype="PCM_16")
    return True


def write_audio_wav_from_disk_path_resampled(src_path: str, dst_path: str, target_sr: int) -> bool:
    """Mono PCM16 WAV em target_sr; uso quando export normalizado quando o marcador do path coincide."""
    try:
        mono, _ = librosa.load(src_path, sr=target_sr, mono=True)
    except (OSError, ValueError, RuntimeError):
        return False
    mono = np.clip(mono.astype(np.float32), -1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    sf.write(dst_path, mono, target_sr, subtype="PCM_16")
    return True


def copy_audio_file_preserve_source(src_path: str, dst_path: str) -> bool:
    """Copia o ficheiro original (mesma taxa / formato) quando nao se aplica reamostragem."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    except OSError:
        return False


def write_sentence_txt_sidecar(audio_export_abs: str, item: Dict) -> Tuple[str, str]:
    """
    Um .txt ao lado do .wav exportado (LapsBM_0601.txt + LapsBM_0601.wav), como no layout LapsBM.
    Se existir transcripto ao lado do ficheiro de audio de origem, copia; senão grava texto das colunas HF.
    Devolve (caminho_absoluto_txt, origin: copy_sibling|embedded|failed).
    """
    dst_txt = os.path.splitext(audio_export_abs)[0] + ".txt"
    src_aud = get_audio_source_path(item)
    if isinstance(src_aud, str) and src_aud.strip():
        sibling_txt = os.path.splitext(src_aud.strip())[0] + ".txt"
        if os.path.isfile(sibling_txt):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(dst_txt)), exist_ok=True)
                shutil.copy2(sibling_txt, dst_txt)
                return os.path.abspath(dst_txt), "copy_sibling"
            except OSError:
                pass
    snippet = extract_text(item)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dst_txt)), exist_ok=True)
        with open(dst_txt, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(snippet if snippet else "")
        return os.path.abspath(dst_txt), "embedded"
    except OSError:
        return "", "failed"


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


def export_test_base(
    full_ds: Dataset,
    test_indices: List[int],
    output_dir: str,
    export_sample_rate: Optional[int],
) -> None:
    target_sr_opt = export_sample_rate
    normalize_audio_hz = export_sample_rate is not None

    os.makedirs(output_dir, exist_ok=True)

    test_ds = full_ds.select(test_indices)
    hf_out = os.path.join(output_dir, "hf_test_split")
    if os.path.exists(hf_out):
        shutil.rmtree(hf_out)
    test_ds.save_to_disk(hf_out)

    if normalize_audio_hz:
        print(
            f"[export audio] --export_sample_rate={export_sample_rate}: "
            f"PCM mono normalizado aos {target_sr_opt} Hz (librosa + soundfile)."
        )
    else:
        print(
            "[export audio] Sem --export_sample_rate — SR/formato preservados "
            "(cópia de ficheiro ou arrays HF ao SR original)."
        )

    manifest_path = os.path.join(output_dir, "test_manifest.jsonl")
    txt_path = os.path.join(output_dir, "test_sentences.txt")
    spk_path = os.path.join(output_dir, "test_speakers.txt")

    copied_from_path = 0
    written_from_array = 0
    missing_audio = 0
    array_export_entries: List[Tuple[int, str, str]] = []
    speaker_set = set()
    sentences: List[str] = []

    audio_out = os.path.join(output_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)

    used_names_by_speaker: Dict[str, set] = {}

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, item in enumerate(test_ds):
            speaker = extract_speaker_id(item)
            text = extract_text(item)
            source_audio = get_audio_source_path(item)
            src_bn = source_audio_basename(item)
            hf_raw = item.get("__key__", "")
            hf_example_key = hf_raw.strip() if isinstance(hf_raw, str) else ""
            speaker_set.add(speaker)
            if text:
                sentences.append(text)

            speaker_dir = os.path.join(audio_out, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            used_set = used_names_by_speaker.setdefault(speaker, set())
            export_stem = stem_for_exported_wav(item, i)

            copied_audio = ""
            audio_export = ""
            if source_audio and os.path.exists(source_audio):
                if target_sr_opt is not None:
                    out_fn = allocate_unique_media_filename(export_stem, ".wav", used_set)
                    dst = os.path.join(speaker_dir, out_fn)
                    if write_audio_wav_from_disk_path_resampled(source_audio, dst, target_sr_opt):
                        copied_audio = dst
                        audio_export = "path"
                        copied_from_path += 1
                    elif write_audio_wav_from_hf_item(item, dst, target_sr_opt):
                        copied_audio = dst
                        audio_export = "array"
                        written_from_array += 1
                        array_export_entries.append(
                            (i, speaker, os.path.abspath(dst))
                        )
                    else:
                        missing_audio += 1
                else:
                    ext_native = os.path.splitext(source_audio)[1] or ".wav"
                    out_nat = allocate_unique_media_filename(export_stem, ext_native, used_set)
                    dst_nat = os.path.join(speaker_dir, out_nat)
                    if copy_audio_file_preserve_source(source_audio, dst_nat):
                        copied_audio = dst_nat
                        audio_export = "path"
                        copied_from_path += 1
                    else:
                        out_fb = allocate_unique_media_filename(export_stem, ".wav", used_set)
                        dst_wav = os.path.join(speaker_dir, out_fb)
                        if write_audio_wav_from_hf_item(item, dst_wav, None):
                            copied_audio = dst_wav
                            audio_export = "array"
                            written_from_array += 1
                            array_export_entries.append(
                                (i, speaker, os.path.abspath(dst_wav))
                            )
                        else:
                            missing_audio += 1
            else:
                out_fn = allocate_unique_media_filename(export_stem, ".wav", used_set)
                dst_wav = os.path.join(speaker_dir, out_fn)
                if write_audio_wav_from_hf_item(item, dst_wav, target_sr_opt):
                    copied_audio = dst_wav
                    audio_export = "array"
                    written_from_array += 1
                    array_export_entries.append(
                        (i, speaker, os.path.abspath(dst_wav))
                    )
                else:
                    missing_audio += 1

            wav_hz_row: Optional[int] = None
            if copied_audio and os.path.isfile(copied_audio):
                try:
                    wav_hz_row = int(sf.info(copied_audio).samplerate)
                except (OSError, RuntimeError):
                    wav_hz_row = None

            exported_bn = os.path.basename(copied_audio) if copied_audio else ""
            exported_txt_abs = ""
            sentence_txt_origin = ""
            if copied_audio:
                exported_txt_abs, sentence_txt_origin = write_sentence_txt_sidecar(
                    os.path.abspath(copied_audio), item
                )

            row = {
                "index": i,
                "speaker_id": speaker,
                "text": text,
                "source_audio": source_audio,
                "source_audio_basename": src_bn or None,
                "hf_example_key": hf_example_key or None,
                "exported_audio_basename": exported_bn or None,
                "exported_sentence_txt": exported_txt_abs or None,
                "sentence_txt_origin": sentence_txt_origin or None,
                "copied_audio": copied_audio,
                "audio_export": audio_export,
                "normalized_to_requested_export_hz": normalize_audio_hz,
                "requested_export_hz": export_sample_rate,
                "wav_sample_rate_hz": wav_hz_row,
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
    if normalize_audio_hz:
        print(f"- WAV em audio/: PCM mono aos {target_sr_opt} Hz (--export_sample_rate).")
    else:
        print("- WAV em audio/: sem alteração de taxa (sem --export_sample_rate).")
    print(
        "- Nota: 'hf_test_split' guarda o Arrow tal como carregado (SR original do dataset)."
    )
    print(
        "- Nomes em audio/: derivados do WAV no path HF quando existir (ex. LapsBM_0601.wav); "
        "em test_manifest: source_audio_basename, exported_audio_basename."
    )
    print(
        "- Cada WAV exportado tem um .txt ao lado (mesmo stem); sentence_txt_origin no manifest: "
        "copy_sibling (= ficheiro ao lado do WAV de origem em disco) ou embedded (= colunas txt/text do HF)."
    )
    print(f"- Manifesto: {manifest_path}")
    print(f"- Sentencas: {txt_path}")
    print(f"- Locutores: {spk_path}")
    print(
        f"- Audios: copia (path): {copied_from_path} | "
        f"gravado (array HF): {written_from_array} | sem audio: {missing_audio}"
    )

    array_list_path = os.path.join(output_dir, "test_audio_exported_from_array.txt")
    if array_export_entries:
        with open(array_list_path, "w", encoding="utf-8") as lf:
            lf.write("# indice_na_base_teste\tlocutor\tcaminho_wav_gravado_por_array\n")
            for idx, spk, wav_abs in array_export_entries:
                lf.write(f"{idx}\t{spk}\t{wav_abs}\n")
        print(f"- Lista completa via array HF: {array_list_path}")
        print("\nAmostras gravadas a partir do array (sem copy por path):")
        for idx, spk, wav_abs in array_export_entries:
            print(f"  [{idx}] {spk} -> {wav_abs}")
    else:
        print("- Nenhum audio precisou de gravacao a partir do array (todos por path ou sem audio).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstrói e exporta a base de teste por locutor.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_profile", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./datasets/test_split_inferencia/lapsbm_speecht5")
    parser.add_argument("--cache_dir", type=str, default="./datasets/cache_processado/speecht5_lapsbm_speecht5_16000hz")
    parser.add_argument(
        "--export_sample_rate",
        type=int,
        default=None,
        help=(
            "Se omitido (default), preserva SR nativo/copy. "
            "Se definido (Hz), reamostra WAV exportados mono PCM para essa taxa."
        ),
    )
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

    marker_ok = output_dir_contains_dataset_profile_marker(args.output_dir, dataset_name)

    split_info = {
        "dataset_profile": dataset_name,
        "model_type_from_profile": ds_cfg.get("model_type"),
        "train_speakers": sorted(list(train_spks)),
        "val_speakers": sorted(list(val_spks)),
        "test_speakers": sorted(list(test_spks)),
        "dataset_profile_marker": dataset_name,
        "output_path_contains_profile_marker": marker_ok,
        "export_sample_rate_requested_hz": args.export_sample_rate,
        "audio_resampled_to_requested_hz": args.export_sample_rate is not None,
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

    export_test_base(full_ds, test_idx, args.output_dir, args.export_sample_rate)


if __name__ == "__main__":
    main()
