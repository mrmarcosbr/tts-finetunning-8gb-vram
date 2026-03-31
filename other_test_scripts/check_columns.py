from datasets import load_dataset
import pandas as pd

dataset_id = "falabrasil/lapsbm"
print(f"Loading {dataset_id} test split...")
dataset = load_dataset(dataset_id, split="test", streaming=False)

df = pd.DataFrame({
    "url": dataset["__url__"],
    "key": dataset["__key__"]
})

print(f"Total samples: {len(df)}")
print(f"Unique URLs: {len(df['url'].unique())}")
print(f"Unique Keys: {len(df['key'].unique())}")

print("\nSample URLs:")
for u in df['url'].unique()[:5]:
    print(u)

print("\nSample Keys:")
for k in df['key'].iloc[:5]:
    print(k)

# Check if there is anything inside the tar ball folder names
# We can't see that without extracting or checking if 'wav' column has internal paths.
if 'wav' in dataset.column_names:
    example_wav = dataset[0]['wav']
    print("\nWav object details:")
    print(example_wav)
