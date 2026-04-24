from peft import LoraConfig, get_peft_model
from train_exhaustive import get_handler, load_config
full_cfg = load_config("config.yaml")
handler = get_handler("fastspeech2", full_cfg['models']['fastspeech2'], "cuda")
model, processor = handler.load()

# Let's see if PEFT can target pitch_predictor
config = LoraConfig(
    r=32, target_modules=["pitch_predictor", "duration_predictor", "energy_predictor"]
)
try:
    peft_model = get_peft_model(model.model.base_model.model, config)
    peft_model.print_trainable_parameters()
    print("Success targeting by module name!")
except Exception as e:
    print("Failed:", e)
