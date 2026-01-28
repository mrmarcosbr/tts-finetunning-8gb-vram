from transformers import SpeechT5Processor

def main():
    model_id = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    # Test Sentence
    text = "Olá, áudio em português com ç e ã."
    
    # Tokenize
    ids = tokenizer(text)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded = tokenizer.decode(ids)
    
    print(f"Original: {text}")
    print(f"Decoded:  {decoded}")
    print(f"Tokens:   {tokens}")
    
    # Check for <unk>
    if tokenizer.unk_token_id in ids:
        print("🚨 ALERTA: Caracteres desconhecidos detectados!")
    else:
        print("✅ Vocabulário suporta todos os caracteres.")

if __name__ == "__main__":
    main()
