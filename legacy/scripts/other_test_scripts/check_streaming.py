from datasets import load_dataset

dataset_id = "falabrasil/lapsbm"
print(f"Loading {dataset_id} in streaming mode...")
dataset = load_dataset(dataset_id, split="test", streaming=True)

# Get the first sample
for sample in dataset:
    print("Sample keys:", sample.keys())
    print("Sample content:", {k: v for k, v in sample.items() if k != 'wav'})
    break
