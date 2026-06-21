from datasets import load_dataset
from collections import Counter

dataset_id = "falabrasil/lapsbm"
print(f"Loading {dataset_id}...")
dataset = load_dataset(dataset_id, split="test", streaming=False)

print(f"Column names: {dataset.column_names}")

def extract_speaker_id(url):
    if not url: return "no_url"
    if "LapsBM-" in url:
        parts = url.split('/')
        for p in parts:
            if p.startswith("LapsBM-") and ".tar" not in p:
                return p.replace("LapsBM-", "")
            if p.startswith("LapsBM-") and ".tar" in p:
                 return p.split(".")[0].replace("LapsBM-", "")
    return "unknown"

if "__url__" in dataset.column_names:
    urls = dataset["__url__"][:10]
    print(f"Sample URLs: {urls}")
    all_speakers = [extract_speaker_id(x) for x in dataset["__url__"]]
    counts = Counter(all_speakers)
    print(f"Speaker counts: {counts.most_common(5)}")
else:
    print("Column '__url__' not found!")
    # Check other columns that might contain speaker info
    for col in dataset.column_names:
        if "speaker" in col.lower() or "id" in col.lower():
            print(f"Sample from {col}: {dataset[col][:5]}")
