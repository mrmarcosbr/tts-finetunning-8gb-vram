from datasets import load_dataset

dataset_id = "falabrasil/lapsbm"
print(f"Checking {dataset_id} train split...")
try:
    dataset = load_dataset(dataset_id, split="train", streaming=False)
    print(f"Columns in train: {dataset.column_names}")
    print(f"First 10 __key__ in train: {dataset['__key__'][:10]}")
    print(f"First 10 __url__ in train: {dataset['__url__'][:10]}")
except Exception as e:
    print(f"Error loading train: {e}")
