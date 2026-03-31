from datasets import load_dataset

dataset_id = "falabrasil/lapsbm"
dataset = load_dataset(dataset_id, split="test", streaming=False)
print("Dataset features:", dataset.features)
