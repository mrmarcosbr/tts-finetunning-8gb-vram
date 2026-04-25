import torch
from safetensors.torch import load_file
safetensors_path = "./output_cuda_8gb/fastspeech2-lapsbm_fastspeech2-2026-04-24-20-14-28/model.safetensors"
sd = load_file(safetensors_path)

from test_inference_exhaustive import FastSpeech2Inference
import yaml
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
handler = FastSpeech2Inference("./output_cuda_8gb/fastspeech2-lapsbm_fastspeech2-2026-04-24-20-14-28", cfg, "cuda")
handler.load()

sd_lora_key = "model.base_model.model.encoder.encoder.fft_layers.0.conv1.lora_A.default.weight"
print("In saved file (first 5 vals):", sd[sd_lora_key].flatten()[:5])

inf_lora_key = "base_model.model.encoder.encoder.fft_layers.0.conv1.lora_A.default.weight"
print("In inference model (first 5 vals):", handler.model.state_dict()[inf_lora_key].flatten()[:5])

print("Are they equal?", torch.allclose(sd[sd_lora_key], handler.model.state_dict()[inf_lora_key]))
