from datasets import load_dataset
from collections import Counter

dataset_id = "falabrasil/lapsbm"
dataset = load_dataset(dataset_id, split="test", streaming=False)
urls = dataset["__url__"]
print(f"Total URLs: {len(urls)}")
counts = Counter(urls)
print(f"Unique URLs: {len(counts)}")
for url, count in counts.most_common(5):
    print(f"{count} samples have URL: {url}")
