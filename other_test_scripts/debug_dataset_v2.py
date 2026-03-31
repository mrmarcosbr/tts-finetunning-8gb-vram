from datasets import load_dataset
from collections import Counter
import os

dataset_id = "falabrasil/lapsbm"
print(f"Loading {dataset_id}...")
dataset = load_dataset(dataset_id, split="test", streaming=False)

print(f"Column names: {dataset.column_names}")

# View some items
for i in range(5):
    print(f"\nItem {i}:")
    print(f"  __key__: {dataset['__key__'][i]}")
    # Fix the way we access wav path
    wav_item = dataset['wav'][i]
    if isinstance(wav_item, dict):
        print(f"  wav path: {wav_item.get('path', 'N/A')}")
    else:
        print(f"  wav: {wav_item}")

def extract_speaker_id(item):
    # Try extract from __key__
    key = item.get("__key__", "")
    if key:
        # Expected format: LapsBM-M001-01-01 or similar
        # Based on LapsBM structure, it is usually something like M001 or F002
        # Let's see some samples first
        pass
    return "unknown"

# Let's print some keys to see the pattern
print("\nFirst 20 __key__ values:")
print(dataset["__key__"][:20])

# Speaker extraction from key:
# LAPSBM keys often look like 'M001-01' or 'F002-15'
# The first letter (M/F) and the next 3 digits are the speaker ID.

def suggest_speaker_extraction(key):
    # Example key: 'M001_01' or 'M001-01'
    if not key: return "unknown"
    # Often keys are like: 12345 (just numbers) in some versions
    # But usually LapsBM has speaker codes.
    # Let's wait to see the output before defining the final extraction.
    return key.split('-')[0] if '-' in key else key.split('_')[0] if '_' in key else "unknown"

