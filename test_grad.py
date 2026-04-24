from train_exhaustive import get_handler, load_config
from peft import LoraConfig, get_peft_model
full_cfg = load_config("config.yaml")
handler = get_handler("fastspeech2", full_cfg['models']['fastspeech2'], "cuda")
model, processor = handler.load()
for name, p in model.named_parameters():
    if "pitch" in name and p.requires_grad:
        print("TRAINABLE:", name)
