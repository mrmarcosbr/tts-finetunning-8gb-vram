from datasets import get_dataset_split_names

dataset_id = "falabrasil/lapsbm"
try:
    splits = get_dataset_split_names(dataset_id)
    print(f"Available splits for {dataset_id}: {splits}")
except Exception as e:
    print(f"Error getting splits: {e}")
