import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import load_from_disk
from train_exhaustive import get_handler, load_config

full_cfg = load_config("config.yaml")
model_type = "fastspeech2"
dataset_name = "lapsbm_fastspeech2"
hw_cfg = full_cfg['hardware_profiles']['cuda_8gb']
device = "cuda"

handler = get_handler(model_type, full_cfg['models'][model_type], device)
model, processor = handler.load(resume_from="./output_cuda_8gb/fastspeech2-lapsbm_fastspeech2-2026-04-24-13-52-33/checkpoint-14000")

ds = load_from_disk("./datasets/cache_processado/fastspeech2_lapsbm_fastspeech2_22050hz/train")
batch = [ds[0], ds[1], ds[2], ds[3]]
collator = handler.get_collator()
inputs = collator(batch)

for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device)

model.eval()
if True:
    outputs = model(**inputs)
    print("LOSS TOTAL:", outputs["loss"].item())
    
    # We want to see the individual losses. Let's do a manual forward
    with torch.amp.autocast('cuda', enabled=False):
        kwargs = inputs.copy()
        x = kwargs.pop("x")
        x_lengths = kwargs.pop("x_lengths")
        y = kwargs.pop("y").float()
        y_lengths = kwargs.pop("y_lengths")
        pitch = kwargs.pop("pitch").float()
        
        out = model.model(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, pitch=pitch, **kwargs)
        criterion = model.criterion
        loss_dict = criterion(
            decoder_output=out["model_outputs"],
            decoder_target=y,
            decoder_output_lens=y_lengths,
            dur_output=out["durations_log"],
            dur_target=out.get("o_alignment_dur", None),
            pitch_output=out.get("pitch_avg", None),
            pitch_target=out.get("pitch_avg_gt", None),
            energy_output=out.get("energy_avg", None),
            energy_target=out.get("energy_avg_gt", None),
            input_lens=x_lengths,
            alignment_logprob=out.get("alignment_logprob", None),
            alignment_soft=out["alignment_soft"],
            alignment_hard=out["alignment_mas"],
            binary_loss_weight=0.0
        )
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.item()}")

