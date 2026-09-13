import pandas as pd

DATA_PATH = "data/WM811K_labeled.pkl"

print("1. Loading labeled dataset...")
df = pd.read_pickle(DATA_PATH)

print("Shape:", df.shape)

print("\n2. Analyzing waferMap shapes...")

shape_counts = df["waferMap"].apply(lambda x: x.shape).value_counts()

print("Number of unique shapes:", len(shape_counts))
print("\nTop 20 shapes:")
print(shape_counts.head(20))

# wafer 최대 최소 크기 확인
shapes = df["waferMap"].apply(lambda x: x.shape)

print("\n3. Min / Max wafer size")
print("Min height:", shapes.apply(lambda x: x[0]).min())
print("Max height:", shapes.apply(lambda x: x[0]).max())
print("Min width:", shapes.apply(lambda x: x[1]).min())
print("Max width:", shapes.apply(lambda x: x[1]).max())

# wafer 중앙값 및 분포도 확인
heights = shapes.apply(lambda x: x[0])
widths = shapes.apply(lambda x: x[1])

print("\n4. Wafer size distribution")
print("Median height:", heights.median())
print("Median width:", widths.median())
print("90% height:", heights.quantile(0.9))
print("90% width:", widths.quantile(0.9))

print("\nDone!")