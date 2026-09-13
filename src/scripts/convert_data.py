import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd

import pandas.core.indexes as indexes
import pandas.core.indexes.base as indexes_base

# 구버전 pandas 경로 → 현재 pandas 경로 연결
sys.modules["pandas.indexes"] = indexes
sys.modules["pandas.indexes.base"] = indexes_base

data_path = "data/LSWMD.pkl"

with open(data_path, "rb") as file:
    data = pickle.load(file, encoding="latin1")

# DataFrame 확인 및 Column 정보 확인
print(type(data))
print("Shape:", data.shape)
print("Columns:", data.columns)

# Wafer 첫 장 배열 확인
print(data["waferMap"].iloc[0])
print("First wafer shape:", data["waferMap"].iloc[0].shape)

# Wafer 이미지로 확인
wafer = data["waferMap"].iloc[0]

plt.imshow(wafer)
plt.title("First Wafer Map")
plt.show()

# Wafer 첫 장 불량 종류 확인
print("Failure Type:", data["failureType"].iloc[0])

# Wafer 불량 종류 확인
# print(data["failureType"].value_counts())

# pandas 버전 새 pickle 생성
data.index = pd.RangeIndex(len(data))
data.columns = pd.Index(list(data.columns))
data.to_pickle("data/WM811K_clean.pkl")