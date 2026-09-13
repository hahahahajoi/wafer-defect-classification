import sys
import pandas as pd
import pandas.core.indexes as indexes

data_path = "data/LSWMD.pkl"

df = pd.read_pickle(data_path)

print("=== Dataset Loaded ===")
print("Shape:", df.shape)

print("\n=== Columns ===")
print(df.columns)

print("\n=== First 5 Rows ===")
print(df.head())