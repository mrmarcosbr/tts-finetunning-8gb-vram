import torch
import numpy as np
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.models.forward_tts import ForwardTTS

# Load model
config_path = 'C:/Users/marco/AppData/Local/tts/tts_models--en--ljspeech--fast_pitch/config.json'
config = Fastspeech2Config()
config.load_json(config_path)
model = ForwardTTS.init_from_config(config)
model.eval()

# Dummy input
x = torch.randint(0, 50, (1, 10))
x_lengths = torch.tensor([10])
y = torch.randn(1, 100, 80)
y_lengths = torch.tensor([100])
# Raw Hz pitch
pitch = torch.ones(1, 1, 100) * 200.0

with torch.no_grad():
    outputs = model(x, x_lengths, y_lengths, y=y, pitch=pitch)
    
    pitch_pred = outputs["pitch_avg"]
    pitch_gt = outputs["pitch_avg_gt"]
    
    print(f"Pitch Predictor Output (Mean/Std): {pitch_pred.mean().item():.4f} / {pitch_pred.std().item():.4f}")
    print(f"Pitch Ground Truth (Mean/Std): {pitch_gt.mean().item():.4f} / {pitch_gt.std().item():.4f}")
