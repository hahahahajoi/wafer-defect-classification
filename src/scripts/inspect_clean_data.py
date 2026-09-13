import pandas as pd

data_path = "data/WM811K_clean.pkl"

df = pd.read_pickle(data_path)

print("Type:", type(df))
print("Shape:", df.shape)
print("Columns:", df.columns)

# FailureType 분포 확인
print("\n=== Failure Type Counts ===")
print(df["failureType"].value_counts())