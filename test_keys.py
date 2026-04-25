import torch
from safetensors.torch import load_file
safetensors_path = "./output_cuda_8gb/fastspeech2-lapsbm_fastspeech2-2026-04-24-20-14-28/model.safetensors"
sd = load_file(safetensors_path)
print("Saved keys snippet:")
for i, k in enumerate(list(sd.keys())[:5]): print(k)

from train_exhaustive import get_handler, load_config
full_cfg = load_config("config.yaml")
handler = get_handler("fastspeech2", full_cfg['models']['fastspeech2'], "cuda")
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=full_cfg['models']['fastspeech2']['lora']['r'], 
    lora_alpha=full_cfg['models']['fastspeech2']['lora']['alpha'], 
    target_modules=full_cfg['models']['fastspeech2']['lora']['target_modules'], 
    bias="none"
)
model, _ = handler.load()
peft_model = get_peft_model(model.model, peft_config)
print("\nExpected keys snippet:")
for i, k in enumerate(list(peft_model.state_dict().keys())[:5]): print(k)
