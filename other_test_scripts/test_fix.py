from datasets import load_dataset, Dataset

dataset_id = "falabrasil/lapsbm"
print(f"Loading {dataset_id} as list (via streaming=True) to preserve URLs...")
dataset_stream = load_dataset(dataset_id, split="test", streaming=True)

# Collect all items into a list (700 is small)
all_items = []
for i, item in enumerate(dataset_stream):
    all_items.append(item)
    if (i + 1) % 100 == 0:
        print(f"Collected {i + 1} items...")

print(f"Total collected: {len(all_items)}")

# Convert back to a regular dataset
dataset = Dataset.from_list(all_items)
print("Columns in final dataset:", dataset.column_names)

def extract_speaker_id(url):
    if not url: return "unknown"
    if "LapsBM-" in url:
        parts = url.split('/')
        for p in parts:
            if p.startswith("LapsBM-") and ".tar" not in p:
                return p.replace("LapsBM-", "")
            if p.startswith("LapsBM-") and ".tar" in p:
                 return p.split(".")[0].replace("LapsBM-", "")
    return "unknown"

all_speakers = [extract_speaker_id(x) for x in dataset["__url__"]]
from collections import Counter
counts = Counter(all_speakers)
print(f"Speaker counts: {counts.most_common(5)}")
