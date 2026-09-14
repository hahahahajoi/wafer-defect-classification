import torch
import random
import numpy as np
import matplotlib.pyplot as plt

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tqdm import tqdm
from src.models.baseline_cnn import BaselineCNN
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.training.losses import FocalLoss
from src.data.dataloader import (
    create_train_loader,
    create_val_loader,
    create_test_loader
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = BaselineCNN().to(device)

print(model)

# Loss 함수 객체 이름 = criterion
#criterion = torch.nn.CrossEntropyLoss()

# Class Weight 계산
train_loader = create_train_loader(batch_size=32)
val_loader = create_val_loader(batch_size=32)
test_loader = create_test_loader(batch_size=32)

class_counts = train_loader.dataset.dataframe["failureType"].value_counts()

class_order = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch"
]

total_samples = len(train_loader.dataset)
num_classes = len(class_order)

class_weights = total_samples / (
    num_classes * class_counts[class_order]
)


class_weights = class_weights ** 0.5

class_weights = torch.tensor(
    class_weights.values,
    dtype=torch.float32
).to(device)

print("Class Weights:", class_weights)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#criterion = FocalLoss(gamma=2.0)
#criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# data 10번 반복 학습
num_epochs = 30
best_macro_f1 = 0.0
best_epoch = 0
patience = 6
epochs_without_improvement = 0

# train
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    running_loss = 0.0

    model.train()

    for wafers, labels in tqdm(
        train_loader,
        desc=f"Train Epoch {epoch + 1}/{num_epochs}"):
        wafers = wafers.to(device)
        labels = labels.to(device)

        # 이전 batch에서 계산된 기울기가 누적되는 것을 방지
        optimizer.zero_grad()

        # 예측
        outputs = model(wafers)

        #Loss 계산
        loss = criterion(outputs, labels)

        # 각 weight의 gradient 계산
        loss.backward()

        # 실제 weight 수정
        optimizer.step()

        running_loss += loss.item()


    avg_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")


    val_running_loss = 0.0
    val_correct = 0

    all_preds = []
    all_labels = []

    model.eval()
    
    # val
    with torch.no_grad():
        for wafers, labels in tqdm(
            val_loader,
            desc="Validation"):
            wafers = wafers.to(device)
            labels = labels.to(device)

            # 예측 → loss backward나 optimizer는 val 단계에서 진행하지 않음.
            outputs = model(wafers)

            loss = criterion(outputs, labels)

            # 9개중 가장 높은 특성 선택
            predictions = outputs.argmax(dim=1)

            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            # label과 비교했을때 맞힌 개수
            val_correct += (predictions == labels).sum().item()

            val_running_loss += loss.item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / len(val_loader.dataset) * 100

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
        )

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "outputs/models/best_model_sqrt_scheduler.pth"
            )
    else:
        epochs_without_improvement += 1

    scheduler.step(macro_f1)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Learning Rate: {current_lr:.6f}")

    if epochs_without_improvement >= patience:
        print(f"Early Stopping at Epoch {epoch + 1}")
        break

    


print("\nTraining Finished")
print(f"Best Epoch: {best_epoch}")
print(f"Best Macro F1: {best_macro_f1:.4f}")

# Best Model 불러오기
model.load_state_dict(
    torch.load(
        "outputs/models/best_model_sqrt_scheduler.pth",
        weights_only=True
    )
)

test_preds = []
test_labels = []

model.eval()

# Best Model을 Test Loader에 불러옴
with torch.no_grad():
    for wafers, labels in tqdm(
        test_loader,
        desc="Test Evaluation"
    ):
        wafers = wafers.to(device)
        labels = labels.to(device)

        outputs = model(wafers)
        predictions = outputs.argmax(dim=1)

        test_preds.extend(predictions.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())

model.eval()

best_preds = []
best_labels = []

with torch.no_grad():
    for wafers, labels in tqdm(
        val_loader,
        desc="Best Model Evaluation"
    ):
        wafers = wafers.to(device)
        labels = labels.to(device)

        outputs = model(wafers)
        predictions = outputs.argmax(dim=1)

        best_preds.extend(predictions.cpu().tolist())
        best_labels.extend(labels.cpu().tolist())

print("\nBest Model Classification Report")

# support = dataset에 있는 실제 scratch 개수
# recall = 실제 잡아낸 수치
# precision = 모델이 scratch라고 예측한 것 중 실제 scratch 비율
# f1 - scroe = precision과 recall을 동시 평가 지표

print(
    classification_report(
        best_labels,
        best_preds,
        target_names=class_order,
        digits=4
    )
)

cm = confusion_matrix(best_labels, best_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_order
)

disp.plot(xticks_rotation=45, cmap="Blues")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix_sqrt_scheduler.png", dpi=200)

plt.close()

# Test Set 최종 평가
test_preds = []
test_labels = []

model.eval()

with torch.no_grad():
    for wafers, labels in tqdm(
        test_loader,
        desc="Test Evaluation"
    ):
        wafers = wafers.to(device)
        labels = labels.to(device)

        outputs = model(wafers)
        predictions = outputs.argmax(dim=1)

        test_preds.extend(predictions.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())

# TEST 결과
test_accuracy = (
    sum(p == y for p, y in zip(test_preds, test_labels))
    / len(test_labels)
    * 100
)

test_macro_f1 = f1_score(
    test_labels,
    test_preds,
    average="macro"
)

print("\nTest Results")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")

print(
    classification_report(
        test_labels,
        test_preds,
        target_names=class_order,
        digits=4
    )
)