from datasets import load_dataset

dataset_id = "falabrasil/lapsbm"
dataset = load_dataset(dataset_id, split="test", streaming=False)
print("Dataset info:", dataset.info)
print("\nDataset info description:", dataset.info.description)
