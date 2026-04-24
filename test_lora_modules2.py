from train_exhaustive import get_handler, load_config
full_cfg = load_config("config.yaml")
handler = get_handler("fastspeech2", full_cfg['models']['fastspeech2'], "cuda")
model, processor = handler.load()
print("\n--- Pitch Predictor Modules ---")
for name, mod in model.named_modules():
    if "pitch" in name.lower() and isinstance(mod, __import__("torch").nn.Conv1d):
        print(name)
print("\n--- Duration Predictor Modules ---")
for name, mod in model.named_modules():
    if "dur" in name.lower() and isinstance(mod, __import__("torch").nn.Conv1d):
        print(name)
