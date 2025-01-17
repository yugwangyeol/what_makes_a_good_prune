# What Makes a Good Prune?

<div align="center">
  <img src="https://github.com/user-attachments/assets/9deb18d7-66ff-4c77-8cf1-5e45bd600568" alt="What Makes a Good Prune">
</div>

## Project Overview
이 프로젝트는 ICLR 2024에서 발표된 "What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity" 논문의 구현입니다. 신경망의 Unstructured Pruning을 위한 새로운 접근 방식을 제안하며, Cosine Similarity를 활용하여 원본 모델의 표현 능력을 최대한 유지하면서 모델을 경량화합니다.

### Key Concepts
- **Cosine Similarity 기반 Pruning**: 파라미터 벡터의 방향성 보존에 중점
- **Kurtosis of Kurtoses**: 가중치 분포의 특성을 고려한 pruning rate 조정
- **Pareto Front 최적화**: Pruning rate와 Cosine Similarity 간의 최적점 탐색

## Implementation Details

### Core Features
- ResNet18을 기반으로 한 CIFAR10 이미지 분류
- Cosine Similarity 기반의 Unstructured Pruning 구현
- Kurtosis 분석을 통한 안전한 Pruning rate 계산
- Pruning 전후 성능 비교 및 시각화

### Model Architecture
- Base Model: ResNet18 (pretrained)
- Dataset: CIFAR10
- Batch Size: 256
- Optimization: SGD (lr=0.001, momentum=0.9)

## Getting Started

### Prerequisites
```bash
torch
torchvision
numpy
matplotlib
scipy
```

### Installation
```bash
git clone https://github.com/username/what-makes-a-good-prune.git
cd what-makes-a-good-prune
pip install -r requirements.txt
```

### Usage
```bash
python what_makes_a_good_prune.py
```

## Project Structure
```
├── what_makes_a_good_prune.py  # Main implementation
└── README.md
```

## Implementation Steps

### 1. Base Model Training
- ResNet18 pretrained 모델 로드
- CIFAR10 데이터셋에 대한 추가 학습
- 초기 성능 평가 및 모델 저장

### 2. Optimal Pruning Rate 탐색
```python
# Cosine Similarity 계산
cosine_similarity = torch.dot(base_weights, model_weights) / (
    torch.linalg.norm(base_weights) * torch.linalg.norm(model_weights)
)

# Pareto Front 분석
prune_rate = torch.linspace(0,1,101)
cosine_sim = []  # 각 pruning rate에 대한 similarity 저장
```

### 3. Kurtosis 분석
```python
# Kurtosis of Kurtoses 계산
kurtosis_of_kurtoses_model = kurtosis_of_kurtoses(base)

# Safe pruning rate 계산
if kurtosis_of_kurtoses_model < torch.exp(torch.Tensor([1])):
    prune_modifier = 1/torch.log2(torch.Tensor([kurtosis_of_kurtoses_model]))
else:
    prune_modifier = 1/torch.log(torch.Tensor([kurtosis_of_kurtoses_model]))
```

## Key Functions

### Global Unstructured Pruning
```python
def global_prune_without_masks(model, amount):
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
```

### Kurtosis Analysis
```python
def kurtosis_of_kurtoses(model):
    kurtosis = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            kurtosis.append(stats.kurtosis(
                mod.weight.detach().numpy().flatten(),
                fisher=False
            ))
```

## Results Visualization

프로젝트는 다음과 같은 결과를 시각화합니다:  

- Pruning Rate vs Cosine Similarity의 Pareto Front  

- Optimal Pruning Point (Pareto Front 상의 최적점)  

- Utopia Point (이상적인 목표점)

## Citation

```bibtex
@inproceedings{
  mason-williams2024what,
  title={What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity},
  author={Gabryel Mason-Williams and Fredrik Dahlqvist},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=jsvvPVVzwf}
}
```