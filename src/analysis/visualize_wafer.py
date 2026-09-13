import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing.resize_wafer import resize_wafer


DATA_PATH = "data/WM811K_labeled.pkl"


print("1. Loading dataset...")
df = pd.read_pickle(DATA_PATH)


# 각 클래스에서 wafer 1개씩 가져오기
classes = df["failureType"].unique()

print("Classes:", classes)


for label in classes:
    sample = df[df["failureType"] == label]["waferMap"].iloc[0]

    resized = resize_wafer(sample)

    print(
        f"{label}: "
        f"{sample.shape} -> {tuple(resized.shape)}"
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(resized.numpy(), cmap="gray")
    plt.title(label)
    plt.axis("off")
    plt.show()