from datasets import load_dataset
import pandas as pd

dataset_id = "falabrasil/lapsbm"
dataset = load_dataset(dataset_id, split="test", streaming=False)

keys = dataset["__key__"]
print(f"Total keys: {len(keys)}")
unique_keys = set(keys)
print(f"Unique keys: {len(unique_keys)}")

# Let's see some keys from different parts of the dataset
print("\nKeys [0:5]:", keys[0:5])
print("Keys [300:305]:", keys[300:305])
print("Keys [600:605]:", keys[600:605])

# Find keys that look different
pattern_keys = [k for k in keys if "M" in k or "F" in k]
print(f"\nKeys with M or F: {len(pattern_keys)}")
if pattern_keys:
    print("Sample pattern keys:", pattern_keys[:5])
