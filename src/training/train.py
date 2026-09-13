import torch

from tqdm import tqdm
from src.models.baseline_cnn import BaselineCNN
from sklearn.metrics import classification_report
from src.data.dataloader import (
    create_train_loader,
    create_val_loader
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
criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

#
train_loader = create_train_loader(batch_size=32)
val_loader = create_val_loader(batch_size=32)

# data 10번 반복 학습
num_epochs = 1#0

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
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            "none",
            "Center",
            "Donut",
            "Edge-Loc",
            "Edge-Ring",
            "Loc",
            "Near-full",
            "Random",
            "Scratch"
        ],
        digits=4
        )
    )