import pandas as pd


from sklearn.model_selection import train_test_split


DATA_PATH = "data/WM811K_labeled.pkl"


print("1. Loading labeled dataset...")

df = pd.read_pickle(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

X = df["waferMap"]
y = df["failureType"]

print("\n2. Preparing features and labels...")
print("X shape:", X.shape)
print("y shape:", y.shape)

#
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n3. First split")
print("Train:", X_train.shape, y_train.shape)
print("Temp:", X_temp.shape, y_temp.shape)

#
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print("\n4. Second split")
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

#
print("\n5. Checking class distribution")

print("\nTrain:")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation:")
print(y_val.value_counts(normalize=True) * 100)

print("\nTest:")
print(y_test.value_counts(normalize=True) * 100)

# Train, Val, Test 분류
train_df = pd.DataFrame({
    "waferMap": X_train,
    "failureType": y_train
})

val_df = pd.DataFrame({
    "waferMap": X_val,
    "failureType": y_val
})

test_df = pd.DataFrame({
    "waferMap": X_test,
    "failureType": y_test
})

# Train, Val, Test 저장
train_df.to_pickle("data/WM811K_train.pkl")
val_df.to_pickle("data/WM811K_val.pkl")
test_df.to_pickle("data/WM811K_test.pkl")

print("\n6. Saved split datasets")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)