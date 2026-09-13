import pandas as pd

data_path = "data/WM811K_clean.pkl"
output_path = "data/WM811K_labeled.pkl"

print("1. Loading full dataset...")
df = pd.read_pickle(data_path)

print("Original shape:", df.shape)

# 라벨이 존재하는 wafer만 선택
print("2. Filtering labeled wafers...")
labeled_df = df[df["failureType"].apply(lambda x: len(x) > 0)].copy()

# 앞으로 필요한 컬럼만 남김
labeled_df = labeled_df[["waferMap", "failureType"]].copy()

# failureType을 numpy 배열에서 문자열로 변환
labeled_df["failureType"] = labeled_df["failureType"].apply(lambda x: x.flatten()[0])

print("Labeled shape:", labeled_df.shape)

# 작은 학습용 파일로 저장
print("3. Saving labeled dataset...")
labeled_df.to_pickle(output_path)

print("Saved:", output_path)
print("Done!")