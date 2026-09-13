import pandas as pd

DATA_PATH = "data/WM811K_labeled.pkl"

print("1. Loading labeled dataset...")

df = pd.read_pickle(DATA_PATH)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\n2. Failure type distribution")

counts = df["failureType"].value_counts(dropna=False)
ratios = df["failureType"].value_counts(normalize=True, dropna=False) * 100

result = pd.DataFrame({
    "count": counts,
    "ratio(%)": ratios
})

print(result)

print("\n3. Number of classes")
print(df["failureType"].astype(str).nunique())

print("\nDone!")