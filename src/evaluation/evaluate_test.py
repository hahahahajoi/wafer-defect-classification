import torch

from src.models.baseline_cnn import BaselineCNN
from src.data.dataloader import create_test_loader

# Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 모델 구조 생성
model = BaselineCNN().to(device)

# 저장된 Best Model 불러오기
model.load_state_dict(
    torch.load(
        "outputs/models/best_model_sqrt_scheduler.pth",
        map_location=device,
        weights_only=True
    )
)
print("Best model loaded successfully.")

model.eval()

# Test Evaluation
from tqdm import tqdm

test_loader = create_test_loader(batch_size=32)

test_preds = []
test_labels = []

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

# Test Detail Report
from sklearn.metrics import classification_report, f1_score

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

cm = confusion_matrix(
    test_labels,
    test_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_order
)

disp.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix_test.png",
    dpi=200
)

plt.close()

print("Saved: outputs/confusion_matrix_test.png")