import os
import glob
import json
import yaml
from pathlib import Path

def update_readmes():
    base_dir = Path("e:/Desenvolvimento/Python/tcc/tts-finetunning-8gb-vram/output_cuda_16gb")
    config_path = Path("e:/Desenvolvimento/Python/tcc/tts-finetunning-8gb-vram/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        readme_path = model_dir / "README.md"
        
        # Try to find trainer_state.json
        trainer_state_files = list(model_dir.glob("**/trainer_state.json"))
        trainer_state = {}
        if trainer_state_files:
            # Get the latest checkpoint
            trainer_state_files.sort(key=os.path.getmtime)
            with open(trainer_state_files[-1], "r", encoding="utf-8") as f:
                try:
                    trainer_state = json.load(f)
                except:
                    pass

        # Try to deduce profile from model_dir name
        dir_name = model_dir.name
        profile_matched = None
        for profile_name in config.get("dataset_profiles", {}).keys():
            if profile_name in dir_name:
                profile_matched = profile_name
                break
        
        # Gathering information
        epochs = trainer_state.get("epoch", "N/A")
        train_batch_size = trainer_state.get("train_batch_size", "N/A")
        global_step = trainer_state.get("global_step", "N/A")
        num_train_epochs = trainer_state.get("num_train_epochs", "N/A")
        
        if profile_matched:
            p_conf = config["dataset_profiles"][profile_matched]
            dataset_id = p_conf.get("dataset_id", "N/A")
            model_type = p_conf.get("model_type", "N/A")
            lr = p_conf.get("training", {}).get("learning_rate", "N/A")
            wd = p_conf.get("training", {}).get("weight_decay", "N/A")
        else:
            dataset_id = "N/A"
            model_type = "N/A"
            lr = "N/A"
            wd = "N/A"

        # Check for training loss or execution summary
        log_history = trainer_state.get("log_history", [])
        final_loss = "N/A"
        train_runtime = "N/A"
        if log_history:
            # find final train runtime if it exists
            for entry in reversed(log_history):
                if "train_runtime" in entry:
                    train_runtime = entry["train_runtime"]
                if "loss" in entry and final_loss == "N/A":
                    final_loss = entry["loss"]

        markdown_content = f"""---
title: Treinamento TTS - {dir_name}
tags:
- text-to-speech
- tts
- lora
- transformers
- falabrasil
---

# Informações do Modelo Extraídas: {dir_name}

Este diretório contém os pesos de um modelo de Text-to-Speech treinado em português do Brasil (ou refinado para vozes específicas) focado na utilização de recursos computacionais limitados (8GB/16GB VRAM RTX 5060 Ti).

## Detalhes do Treinamento

- **Dataset Utilizado**: {dataset_id}
- **Tipo do Modelo Base**: {model_type}
- **Perfil do Dataset Usado**: {profile_matched if profile_matched else 'Desconhecido'}

## Hiperparâmetros de Treinamento

- **GPU Utilizada**: NVIDIA RTX 5060 Ti (CUDA 16GB Profile)
- **Batch Size de Treinamento**: {train_batch_size}
- **Learning Rate**: {lr}
- **Weight Decay**: {wd}
- **Épocas (Configuradas)**: {num_train_epochs}
- **Épocas (Alcançadas)**: {epochs}
- **Total de Passos (Global Step)**: {global_step}

## Métricas Extraídas

- **Tempo Total de Treinamento (Runtime)**: {f'{train_runtime:.2f} segundos' if isinstance(train_runtime, float) else train_runtime}
- **Última Loss Registrada**: {final_loss}

## Como Usar
Estes pesos podem ser importados utilizando a biblioteca `peft` e `transformers` caso seja baseado no SpeechT5, ou processados diretamente se for baseado em XTTS-v2/F5-TTS.
"""

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Updated README for {dir_name}")

if __name__ == "__main__":
    update_readmes()
