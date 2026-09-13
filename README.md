# Wafer Defect Classification using CNN

CNN-based wafer defect pattern classification and class imbalance analysis using the WM811K dataset.

## 1. Project Overview

반도체 Wafer Map의 불량 패턴을 CNN을 이용하여 분류하는 프로젝트입니다.

WM811K 데이터셋을 기반으로 Baseline CNN을 구축하고, 단순 Accuracy뿐만 아니라
Precision, Recall, F1-score 등 클래스별 성능을 분석합니다.

특히 데이터 불균형으로 인해 소수 불량 클래스의 검출 성능이 저하되는 문제를 확인하고,
이를 개선하면서 모델의 성능 변화를 비교하는 것을 목표로 합니다.

## 2. Dataset

본 프로젝트에서는 WM811K Wafer Map 데이터셋을 사용합니다.

- Original Dataset: 811,457 wafers
- Labeled Dataset: 172,950 wafers
- Number of Classes: 9

### Defect Classes

1. none
2. Center
3. Donut
4. Edge-Loc
5. Edge-Ring
6. Loc
7. Near-full
8. Random
9. Scratch

### Data Split

Label이 존재하는 172,950개의 wafer를 Train / Validation / Test 데이터로 분리하였습니다.

| Dataset | Samples | Ratio |
|---|---:|---:|
| Train | 138,360 | 80% |
| Validation | 17,295 | 10% |
| Test | 17,295 | 10% |
| **Total** | **172,950** | **100%** |

클래스 비율이 분할 과정에서 크게 변하지 않도록 Stratified Split을 적용하였습니다.

## 3. Data Preprocessing

WM811K의 wafer map은 샘플마다 크기가 서로 다르기 때문에,
CNN의 입력 크기를 `64 × 64`로 통일하였습니다.

단순히 모든 wafer를 64 × 64로 강제 Resize하면 원본 wafer의 가로세로 비율이
변형될 수 있으므로 다음과 같은 전처리 과정을 적용하였습니다.

1. 원본 Wafer Map의 Aspect Ratio 유지
2. Nearest Neighbor Interpolation을 이용한 Resize
3. 부족한 영역에 Zero Padding 적용
4. PyTorch Tensor (`float32`)로 변환
5. Channel dimension 추가

최종적으로 CNN에 입력되는 하나의 wafer shape은 다음과 같습니다.

```text
[1, 64, 64]

1     : Channel
64×64 : Wafer Map

## 4. Baseline CNN Architecture

Wafer Map의 공간적 불량 패턴을 학습하기 위해 3개의 Convolution Block으로 구성된
Baseline CNN을 구현하였습니다.

### Model Architecture

```text
Input
[1, 64, 64]
      ↓
Conv2d (1 → 16, 3×3, padding=1)
ReLU
MaxPool (2×2)
      ↓
[16, 32, 32]
      ↓
Conv2d (16 → 32, 3×3, padding=1)
ReLU
MaxPool (2×2)
      ↓
[32, 16, 16]
      ↓
Conv2d (32 → 64, 3×3, padding=1)
ReLU
MaxPool (2×2)
      ↓
[64, 8, 8]
      ↓
Flatten
      ↓
4096 Features
      ↓
Linear (4096 → 9)
      ↓
9-Class Logits
```

### Model Configuration

| Item | Configuration |
|---|---|
| Input Shape | `1 × 64 × 64` |
| Convolution Channels | `1 → 16 → 32 → 64` |
| Kernel Size | `3 × 3` |
| Activation | ReLU |
| Pooling | MaxPool `2 × 2` |
| Output Classes | 9 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |

Convolution을 통해 특징을 추출하면서 채널 수를 `16 → 32 → 64`로 증가시키고,
Max Pooling을 통해 공간 크기를 `64 → 32 → 16 → 8`로 단계적으로 축소하였습니다.

마지막 Feature Map `[64, 8, 8]`을 Flatten하여 4,096개의 feature로 변환한 뒤,
Fully Connected Layer를 통해 9개 클래스에 대한 logit을 출력하도록 구성하였습니다.

## 5. Training Pipeline

PyTorch를 이용하여 Mini-batch 기반 학습 및 Validation Pipeline을 구현하였습니다.

### Training Process

```text
Mini Batch
    ↓
Move Data to GPU
    ↓
optimizer.zero_grad()
    ↓
Forward Pass
    ↓
CrossEntropyLoss
    ↓
loss.backward()
    ↓
optimizer.step()
    ↓
Weight Update
```

각 batch마다 이전 batch에서 계산된 gradient를 `optimizer.zero_grad()`로 초기화한 뒤,
Forward Pass를 통해 예측값을 계산합니다.

예측값과 실제 label을 `CrossEntropyLoss`로 비교하고,
`loss.backward()`를 통해 각 parameter의 gradient를 계산한 뒤
`optimizer.step()`을 통해 실제 weight를 업데이트합니다.

### Validation

각 Epoch의 Train이 완료된 후 Validation Dataset을 이용하여 성능을 평가합니다.

Validation 단계에서는 모델의 weight를 업데이트하지 않기 때문에
`model.eval()`과 `torch.no_grad()`를 사용하여 gradient 계산을 비활성화하였습니다.

```text
model.eval()
    ↓
Validation Data
    ↓
Forward Pass
    ↓
Loss Calculation
    ↓
Prediction (argmax)
    ↓
Accuracy / Precision / Recall / F1-score
```

학습과 평가는 NVIDIA GPU(CUDA)를 활용하도록 구성하였으며,
GPU에서 계산된 예측 결과를 CPU로 이동하여 클래스별 성능 지표를 분석하였습니다.

## 6. Baseline Results

Baseline CNN을 1 Epoch 학습한 후 Validation Dataset에서 성능을 평가하였습니다.

### Overall Performance

| Metric | Result |
|---|---:|
| Validation Loss | 0.1553 |
| Validation Accuracy | 95.82% |
| Macro Precision | 0.8529 |
| Macro Recall | 0.6775 |
| Macro F1-score | 0.6808 |
| Weighted F1-score | 0.9511 |

전체 Accuracy는 `95.82%`로 높게 나타났지만,
클래스별 성능을 분석한 결과 일부 소수 클래스에서 매우 낮은 Recall을 확인하였습니다.

### Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| none | 0.9691 | 0.9965 | 0.9826 | 14,743 |
| Center | 0.9052 | 0.8904 | 0.8978 | 429 |
| Donut | 0.9024 | 0.6607 | 0.7629 | 56 |
| Edge-Loc | 0.8408 | 0.5800 | 0.6864 | 519 |
| Edge-Ring | 0.9137 | 0.9845 | 0.9478 | 968 |
| Loc | 0.7845 | 0.3955 | 0.5259 | 359 |
| Near-full | 0.4167 | 1.0000 | 0.5882 | 15 |
| Random | 0.9434 | 0.5814 | 0.7194 | 86 |
| Scratch | 1.0000 | 0.0083 | 0.0165 | 120 |

### Class Imbalance Analysis

Validation Dataset의 17,295개 wafer 중 `none` 클래스는 14,743개로 대부분을 차지합니다.

Baseline 모델은 `none` 클래스에서 Recall `99.65%`를 기록한 반면,
일부 소수 defect class에서는 낮은 Recall을 보였습니다.

특히 `Scratch`의 경우:

```text
Support   : 120
Precision : 1.0000
Recall    : 0.0083
F1-score  : 0.0165
```

실제 Scratch wafer 120개 중 약 1개만 Scratch로 분류하여,
대부분의 Scratch pattern을 다른 클래스로 오분류하고 있음을 확인하였습니다.

따라서 전체 Accuracy `95.82%`만으로는 모델의 defect classification 성능을
충분히 평가할 수 없다고 판단하였습니다.

실제로 Weighted F1-score는 `0.9511`로 높지만,
각 클래스를 동일한 비중으로 평가하는 Macro F1-score는 `0.6808`로 크게 낮았습니다.

이를 통해 Baseline CNN이 다수 클래스에 편향되어 있으며,
소수 defect class의 검출 성능을 개선하기 위한 Class Imbalance 처리가 필요함을 확인하였습니다.