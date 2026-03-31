from datasets import load_dataset
import pandas as pd

dataset_id = "falabrasil/lapsbm"
dataset = load_dataset(dataset_id, split="test", streaming=False)

print("Columns:", dataset.column_names)
df = dataset.to_pandas()
print("\nDataFrame Info:")
print(df.info())
print("\nFirst 5 rows (without audio/text objects content but keys/urls):")
print(df.drop(['wav', 'txt'], axis=1, errors='ignore').head())

# Look for patterns in __key__ or __url__
print("\nUnique values in __url__ (sampled):")
print(df['__url__'].unique()[:5])

print("\nSample __key__:")
print(df['__key__'].iloc[0])

# Examine one item in detail
item = dataset[0]
print("\nItem 0 details (keys):", item.keys())
print("Text sample:", item['txt'][:50])
