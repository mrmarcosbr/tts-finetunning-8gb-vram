from train_exhaustive import get_handler, load_config
full_cfg = load_config("config.yaml")
handler = get_handler("fastspeech2", full_cfg['models']['fastspeech2'], "cuda")
model, processor = handler.load()
model.model.print_trainable_parameters()
for name, mod in model.named_modules():
    if "lora" in name.lower():
        print(name)
        break
