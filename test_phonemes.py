import unicodedata
phonemes = "u tɾejnɐmẽtu ezɐwstʃivu foj fĩnɐlizɐdu kõ susesu."
replacements = {
    'ẽ': 'e', 'ĩ': 'i', 'õ': 'o', 'ũ': 'u', 'ã': 'a',
    '\u0303': '', 'g': 'ɡ'
}
for k, v in replacements.items():
    phonemes = phonemes.replace(k, v)
print(phonemes)
