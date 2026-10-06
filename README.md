# DeepLearning_Project
## 2022년도 2학기 딥러닝 수업 프로젝트
CNN Image Classification & Data Augmentation
TensorFlow 및 Keras를 활용하여 MNIST 및 CIFAR-10 데이터셋 대상의 합성곱 신경망(CNN) 모델을 설계하고, 다양한 데이터 증강(Data Augmentation) 기법이 모델 학습 및 일반화 성능에 미치는 영향을 비교·분석한 프로젝트입니다.

---

## 📌 프로젝트 개요

* **목적**: 
  * CNN 기본 레이어(Conv2D, Pooling, Dropout, Dense) 구성 이해 및 분류 모델 구현
  * `ImageDataGenerator`의 다양한 전처리 및 증강 옵션을 적용하여 과적합 완화 및 검증 데이터 성능 변화 관찰
* **수행 환경**: Python, TensorFlow / Keras, NumPy, Matplotlib

---

## 🛠 아키텍처 및 구현 내용

### 1. MNIST 손글씨 숫자 분류 모델
* **입력 크기**: 28 × 28 × 1 (흑백)
* **네트워크 구조**:
  * `Conv2D (6 filters, 3x3)` + `MaxPooling2D (2x2)`
  * `Conv2D (16 filters, 3x3)` + `MaxPooling2D (2x2)`
  * `Conv2D (120 filters, 3x3)`
  * `Flatten` + `Dense (84, ReLU)` + `Dense (10, Softmax)`
* **학습 설정**: Optimizer: Adam, Loss: Categorical Crossentropy, Epochs: 10, Batch size: 128

### 2. CIFAR-10 객체 분류 및 데이터 증강 실험
* **입력 크기**: 32 × 32 × 3 (컬러)
* **네트워크 구조**:
  * `Conv2D (32, 3x3)` × 2 + `MaxPooling2D (2x2)` + `Dropout (0.25)`
  * `Conv2D (64, 3x3)` × 2 + `MaxPooling2D (2x2)` + `Dropout (0.25)`
  * `Flatten` + `Dense (512, ReLU)` + `Dropout (0.5)` + `Dense (10, Softmax)`
* **데이터 증강 파이프라인 비교 (`ImageDataGenerator`)**:
  * **Generator 1**: 위치 이동(`width/height_shift=0.1`), 좌우 반전(`horizontal_flip=True`)
  * **Generator 2**: Generator 1 옵션 + 평균 중심화(`featurewise_center`, `samplewise_center`)
  * **Generator 3**: Generator 1 옵션 + 상하 반전(`vertical_flip=True`) + 표준편차 정규화(`featurewise/samplewise_std_normalization`)
  * **Generator 4**: Generator 3 옵션 + ZCA 백색화(`zca_whitening=True`, `zca_epsilon=1e10`)

---

## 📊 결과 분석 및 시각화

* **평가 지표**: 학습 단계별 손실값(Loss) 및 분류 정확도(Accuracy) 추적
* **시각화**: Matplotlib 라이브러리를 활용해 각 Generator별 학습 곡선(Train vs Validation)을 그래프로 렌더링하여 수렴 속도 및 과적합 여부 점검

---

## 💻 실행 방법

### 요구 라이브러리 설치
```
pip install numpy tensorflow matplotlib
python deeplearning_project.py
```