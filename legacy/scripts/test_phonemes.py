from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name
import torch
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.models.forward_tts import ForwardTTS

# Load model and tokenizer
config_path = 'C:/Users/marco/AppData/Local/tts/tts_models--en--ljspeech--fast_pitch/config.json'
config = Fastspeech2Config()
config.load_json(config_path)
model = ForwardTTS.init_from_config(config)
tokenizer = model.tokenizer

phonemizer = get_phonemizer_by_name("gruut", language="pt")

texts = [
    "O treinamento exaustivo foi finalizado com sucesso.",
    "A qualidade do áudio melhorou significativamente."
]

for text in texts:
    ph = phonemizer.phonemize(text, separator="", language="pt")
    # Apply our replacements
    replacements = {'ẽ': 'e', 'ĩ': 'i', 'õ': 'o', 'ũ': 'u', 'ã': 'a', '\u0303': '', 'g': 'ɡ'}
    for k, v in replacements.items(): ph = ph.replace(k, v)
    
    ids = tokenizer.text_to_ids(text, language="en") # Using en because the model is en
    # Wait, if we use phonemizer PT in tokenizer, let's see:
    tokenizer.phonemizer = phonemizer
    ids_pt = tokenizer.text_to_ids(text)
    
    print(f"Text: {text}")
    print(f"Phonemes: {ph}")
    print(f"IDs: {ids_pt}")
    
    # Check for 0s or out of vocab
    unseen = [p for p in ph if p not in tokenizer.characters.phonemes and p not in tokenizer.characters.characters]
    print(f"Unseen chars: {set(unseen)}")
