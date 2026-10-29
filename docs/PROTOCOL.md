# Object Detection — Experimental Protocol

**Project:** Efficient Small-Object Detection for UAV Imagery  
**Dataset:** VisDrone2019-DET  
**Task:** 2D Object Detection  
**Implementation:** From Scratch  
**Protocol Version:** 0.1  
**Date:** 2026-10-29  
**Status:** Draft — G00

---

## 1. Project Overview

### 1.1. Objective

본 프로젝트는 드론 영상에서 작은 객체를 탐지하는 경량 Object Detection 모델을 직접 구현하고, 탐지 정확도와 추론 효율 사이의 Trade-off를 분석하는 것을 목표로 한다.

주요 수행 내용은 다음과 같다.

- Backbone, FPN, Detection Head, Loss, Assignment, NMS 직접 구현
- 랜덤 초기화 기반 Baseline 학습
- 작은 객체 및 가림·밀집 조건에서의 오류 분석
- 입력 해상도 및 네트워크 구조 Ablation
- Knowledge Distillation 또는 정밀도 최적화
- ONNX 기반 추론 및 실제 GPU Latency 측정

기존 CIFAR-100 이미지 분류 프로젝트의 코드, 가중치, 데이터 파이프라인은 재사용하지 않는다.

### 1.2. Research Questions

**RQ1.** 입력 해상도 증가가 Small Object AP/Recall에 어떤 영향을 미치며, 추가적인 추론 비용은 얼마나 발생하는가?

**RQ2.** 고해상도 Feature Map을 보존하는 FPN이 작은 객체 탐지 성능에 어떤 영향을 미치는가?

**RQ3.** 모델 구조 축소 및 최적화 과정에서 작은 객체의 Recall을 얼마나 유지할 수 있는가?

**RQ4.** 경량 Student 모델에 대한 Knowledge Distillation이 정확도–Latency Trade-off를 개선할 수 있는가?

위 질문은 검증 대상이며, 개선 효과를 사전에 가정하지 않는다.

### 1.3. Input / Output

- Input: RGB Image, `[B, 3, H, W]`
- Training Target: Class ID, Bounding Box, Ignore Information
- Model Output: Classification Score, Bounding Box Prediction, 선택적으로 Objectness
- Final Output: 이미지별 `(class_id, confidence, xyxy)` 목록

---

## 2. Dataset

### 2.1. Dataset Information

| 항목 | 내용 |
|---|---|
| Dataset | VisDrone2019-DET |
| Train | 6,471 Images |
| Validation | 548 Images |
| Test-dev | 1,610 Images |
| Total Classes | 10 |
| Image Type | UAV RGB Images |
| Annotation | Bounding Box 및 객체 속성 |
| Official Source | https://github.com/VisDrone/VisDrone-Dataset |

데이터 다운로드 및 사용 시 공식 배포 조건을 준수한다.

- 실제 적용 라이선스·사용 조건 및 확인 출처: `TBD`
- Dataset Version / Download Date: `TBD`
- Test-dev GT의 접근 가능 여부와 평가 방법: `TBD`

### 2.2. Target Classes

MVP에서는 전체 10개 클래스 중 선택한 2~3개 클래스로 학습 및 자체 평가를 수행한다.

| Target | Original Class ID | Model Class ID |
|---|---|---|
| TBD | TBD | TBD |
| TBD | TBD | TBD |
| 선택한 경우 TBD | TBD | TBD |

원본 클래스 ID와 모델 내부 클래스 ID의 Mapping을 별도로 기록한다.

선택하지 않은 클래스와 Ignore Region의 학습·평가 처리 규칙은 G01에서 검증 후 고정한다.

전체 10개 클래스 모델과 MVP 모델의 평가 결과는 서로 구분한다.

### 2.3. Dataset Split Policy

- Train: 모델 파라미터 학습
- Validation: Checkpoint 선택 및 Hyperparameter 검증
- Test-dev: 최종 평가 전용

공식 Split을 유지한다.

G01에서 이미지 ID, 중복 이미지, 장면 또는 시퀀스 중복 여부를 점검하고 Split Manifest를 저장한다.

Test-dev 데이터는 학습, 모델 선택, Threshold 최적화 또는 Ablation 설계에 사용하지 않는다.

---

## 3. Annotation & Preprocessing

### 3.1. Annotation Handling

원본 VisDrone 어노테이션의 Bounding Box는 `xywh` 형식이며, 내부 연산에서는 `xyxy` 형식으로 변환한다.

변환 원칙:

- `x1 = x`
- `y1 = y`
- `x2 = x + w`
- `y2 = y + h`

어노테이션의 다음 항목들을 확인한다.

- Bounding Box 유효성
- Class ID 범위
- Ignore Region / Others
- Occlusion / Truncation 정보
- 이미지 경계를 벗어나는 Bounding Box
- Empty Annotation

좌표의 정수·실수 처리 및 경계 보정 규칙은 G01에서 확정한다.

### 3.2. Image Preprocessing

**Baseline Resolution: 416×416**

종횡비를 유지하는 Letterbox 방식으로 이미지를 변환한다.

처리 순서:

1. RGB 이미지 로드
2. 종횡비를 유지하며 Resize
3. 부족한 영역에 Padding
4. Bounding Box 좌표 변환
5. Tensor 변환 및 정규화

추론 시에는 예측 Bounding Box를 Padding과 Resize Scale을 이용하여 원본 이미지 좌표계로 복원한다.

G01에서 확정할 세부 설정:

- Padding Value: `TBD`
- Resize Interpolation: `TBD`
- Normalization 방식: `TBD`
- Training Augmentation: `TBD`

### 3.3. Data Integrity Tests

- 이미지와 어노테이션 파일 대응 여부
- 누락·중복 샘플 검사
- Class ID 유효성
- Bounding Box 면적 및 좌표 범위 검사
- `xywh ↔ xyxy` Round-trip Test
- Letterbox 좌표 변환 및 역변환 Test
- 최소 10장 이상 GT Box 시각화
- 객체 수와 Box 면적 분포 분석

---

## 4. Evaluation Protocol

### 4.1. Object Size Definition

객체 크기 분류는 **원본 이미지 GT Bounding Box의 면적**을 기준으로 한다.

\[
A = w \times h
\]

| Category | Area |
|---|---|
| Small | A < 1,024 px² |
| Medium | 1,024 ≤ A < 9,216 px² |
| Large | A ≥ 9,216 px² |

이 정의는 COCO 방식의 면적 구간을 참고한 자체 오류 분석 기준이다.

입력 해상도가 416에서 640으로 변경되어도 원본 이미지 기준의 객체 크기 분류는 유지한다.

### 4.2. Primary Metrics

| Metric | Definition |
|---|---|
| AP50:95 | IoU 0.50~0.95, 간격 0.05에서 계산한 AP 평균 |
| AP50 | IoU 0.50에서의 AP |
| AP_Small | Small Object에 대한 AP |
| Recall_Small | Small Object에 대한 Recall |

AP는 대상 클래스별로 계산한 뒤 클래스 평균을 구한다. MVP 지표는 선택한 클래스 집합만을 대상으로 산출한다.

Recall_Small은 IoU 0.50과 One-to-One Matching을 기준으로 한다.

추가로 고정할 평가 설정:

- AP 계산 시 Precision–Recall 보간 방식: `TBD`
- AP_Small에 적용할 IoU 범위 및 Ignore Matching 처리: `TBD`
- Recall 계산용 Confidence Threshold: `TBD`
- 평가용 최대 Detection 수: `TBD`
- Ignore Region 및 비대상 클래스의 FP/FN 처리: `TBD`

위 값들은 첫 Baseline 평가 전에 확정하고, 모델 성능에 맞춰 사후 변경하지 않는다.

### 4.3. Secondary Metrics

- Per-Class AP / Recall
- AP_Medium / AP_Large
- 객체 크기별 Recall
- Occlusion별 탐지 성능
- False Positive / False Negative
- Duplicate Detection
- Precision–Recall Curve

FP/FN 및 중복 검출 사례를 최소 10개 이상 수집하여 오류 원인을 분석한다.

### 4.4. Official vs Custom Evaluation

**MVP Custom Evaluation**

- 선택한 2~3개 클래스를 대상으로 자체 평가
- 원본 GT 면적 기준 크기별 지표 산출
- 공식 전체 클래스 VisDrone mAP와 구분하여 표기

**Official Evaluation**

- 전체 10개 클래스 확장 이후 수행
- 공식 평가 규약 및 공식 평가 API를 사용해 검증
- 자체 구현 결과와 공식 구현 결과를 교차검증
- 공개 테스트 GT의 사용 가능 범위 및 제출 규정 준수

공식 지표와 자체 지표는 평가 클래스, Ignore 처리, 검출 수 제한 및 AP 계산 규약의 차이를 명시한다.

---

## 5. Training Configuration

### 5.1. Baseline Configuration

| Parameter | Value |
|---|---|
| Input Resolution | 416×416 |
| Batch Size | 8 |
| Random Seed | 42 |
| Precision | FP32 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Epochs | 50 |
| LR Scheduler | Cosine Decay |
| Warmup | 3 Epochs |
| Initialization | Random Initialization |
| Checkpoint Selection | Best Validation AP50:95 |

Batch Size 8은 초기 설정이며 G02에서 Forward/Backward 메모리 사용량을 실측하여 검증한다.

Scheduler의 최소 LR 및 정확한 Warmup 시작 값 등 추가 파라미터는 G02~G03 학습 코드 확정 전에 기록한다.

### 5.2. Model Architecture

모델의 기본 구성은 다음과 같다.

**Input → CNN Backbone → FPN → Anchor-Free Detection Head → Prediction**

직접 구현 대상:

- Conv-BN-Activation / Residual Blocks
- CNN Backbone
- Top-down FPN
- Detection Head
- Target Assignment
- Classification / Box Regression Loss
- Bounding Box Decoder
- IoU 및 Class-aware NMS

Baseline의 세부 채널 수, Stride, Head 구조, Loss Weight는 G02 구현 전에 확정한다.

사전학습 모델과 다른 프로젝트의 체크포인트는 사용하지 않는다.

### 5.3. Training Verification

본격적인 Baseline 학습 전에 다음 사항을 검증한다.

1. Dummy Batch Forward 및 출력 Shape 검사
2. Loss의 NaN/Inf 여부 확인
3. Backward 및 Gradient 유효성 검사
4. Optimizer Step 정상 동작 확인
5. 8~16개 이미지에 대한 소량 Overfitting Test
6. 소규모 데이터 Smoke Test

검증 완료 후 전체 Train Split으로 Baseline을 학습한다.

### 5.4. Checkpoint Policy

Validation AP50:95가 가장 높은 Checkpoint를 저장한다.

각 실험에서 다음 정보를 보관한다.

- Model / Optimizer / Scheduler State
- Experiment Configuration
- Random Seed
- Train / Validation Metrics
- Git Commit Hash
- Framework 및 CUDA 버전

Test-dev 결과를 Checkpoint 선택에 사용하지 않는다.

---

## 6. Experiment Plan

### 6.1. Baseline

416×416 입력의 Anchor-Free CNN + FPN Detector를 랜덤 초기화하여 학습한다.

Baseline을 고정한 후 오류 분석과 Ablation을 수행한다.

### 6.2. Ablation Studies

| Experiment | Comparison | Purpose |
|---|---|---|
| Resolution | 416 vs 640 | 작은 객체 성능과 추론 비용 비교 |
| Feature Pyramid | Single-scale vs Multi-scale FPN | Multi-scale Feature의 효과 검증 |
| Model Capacity | Baseline vs Reduced Width | 경량화에 따른 성능 변화 검증 |

주요 Ablation을 최소 2개 이상 수행한다.

실험 시 원칙:

- 한 번에 하나의 주요 변수만 변경
- 동일한 Dataset Split 사용
- 동일 Seed 및 학습 예산 기준 사용
- 가능한 동일한 Optimizer 및 LR Schedule 유지
- Batch Size 변경이 불가피한 경우 기록
- 각 실험의 GPU 사용량과 학습 시간 기록

여건이 허용되면 다중 Seed 실험을 수행하여 평균과 표준편차를 보고한다.

### 6.3. Efficiency Experiments

Baseline 및 경량 모델에 대해 다음 방법 중 1~2개를 비교한다.

- Width Reduction
- Knowledge Distillation
- FP16 Inference
- PTQ/INT8 Quantization

Knowledge Distillation 수행 시 Teacher와 Student는 본 프로젝트에서 각각 랜덤 초기화하여 학습한다.

성능 비교는 정확도뿐 아니라 Latency, VRAM, 모델 크기를 포함한다.

---

## 7. Inference Benchmark

### 7.1. Hardware & Software

| Component | Configuration |
|---|---|
| OS | Windows |
| GPU | NVIDIA RTX 5080 Laptop GPU |
| GPU VRAM | 16 GB |
| Framework | PyTorch |
| Export | ONNX |
| Runtime | ONNX Runtime / 지원되는 경우 TensorRT |
| Baseline Precision | FP32 |
| Benchmark Batch Size | 1 |

추가 기록 항목:

- Python / PyTorch / CUDA Version
- NVIDIA Driver Version
- ONNX Runtime / TensorRT Version
- GPU Power Mode
- 실행 시 전원 연결 여부
- ONNX Export 및 Runtime 호환성

### 7.2. Benchmark Configuration

| Parameter | Value |
|---|---|
| Batch Size | 1 |
| Warm-up Iterations | 30 |
| Measurement Iterations | 200 |
| CUDA Synchronization | Enabled |
| Metrics | Mean, P50, P95 Latency |
| Throughput | FPS |
| Memory | Peak VRAM |
| Input Resolution | 각 실험별 명시 |

### 7.3. Measurement Scope

**Model-only Latency**

모델 Forward 연산의 소요 시간을 측정한다.

**End-to-End Latency**

전처리, 데이터 전송, 모델 추론, Bounding Box Decode, Confidence Filtering, NMS 등의 전체 소요 시간을 측정한다.

GPU 비동기 실행으로 측정값이 왜곡되지 않도록 동기화를 적용한다.

단일 이미지 순차 처리 시 FPS는 동일 측정 범위의 평균 지연시간을 이용해 계산하고, 배치 처리량과 구분한다.

### 7.4. Reported Metrics

- AP50:95 / AP50
- AP_Small / Recall_Small
- Mean / P50 / P95 Latency (ms)
- End-to-End FPS
- Model Parameter Count
- Checkpoint / ONNX File Size
- Peak VRAM

모든 결과에는 모델, 해상도, Precision, 측정 장비, Runtime을 명시한다.

노트북 GPU에서 측정한 성능을 실제 드론 또는 임베디드 장비의 추론 성능으로 일반화하지 않는다.

---

## 8. Reproducibility & Data Leakage Prevention

### 8.1. Reproducibility

- Seed 고정
- Dataset Split 고정
- Config 기반 학습
- 실험별 Checkpoint 보존
- Training / Validation Log 보존
- Git Commit Hash 기록
- 라이브러리 및 하드웨어 환경 기록
- 최종 README에 재현 명령어 기재

Random Seed를 고정하더라도 모든 CUDA 연산에서 완전한 결정론이 보장되는 것은 아니다. 결정론 관련 옵션과 실제 실행 환경을 기록한다.

### 8.2. Data Leakage Prevention

- Validation을 통한 모델 선택
- Test-dev를 이용한 Hyperparameter Tuning 금지
- Test 결과를 이용한 Ablation 선택 금지
- 공식 Split 변경 시 명시적 근거 기록
- 중복 이미지 및 장면 단위 누출 여부 검사
- 평가 기준의 사후 변경 금지

Test-dev의 평가 가능 여부 및 접근 방식은 공식 데이터 공개 규정을 따른다.

### 8.3. Experiment Artifacts

| Location | Purpose |
|---|---|
| `configs/` | 실험 설정 |
| `data/raw/` | 원본 이미지와 어노테이션 |
| `data/splits/` | Split Manifest |
| `src/` | 모델, Loss, Training, Evaluation 구현 |
| `tests/` | Unit / Integration Tests |
| `runs/` | Checkpoints, Logs, Config Snapshots |
| `reports/` | Error Analysis, Ablation, Final Report |

대용량 원본 데이터 및 모델 가중치는 Git 저장소에 직접 업로드하지 않는다.

### 8.4. Project Completion Criteria

프로젝트 완료를 위해 다음 결과물을 확보한다.

- 재현 가능한 Baseline
- 핵심 연산 Unit Test
- 작은 표본 Overfitting Test
- 최소 2개 통제 Ablation
- 실패 사례 10개 이상 분석
- 경량화 전후 정확도–Latency–VRAM 비교
- ONNX 기반 추론 검증
- 최종 평가 및 기술 보고서

**G00 Gate:** 실험 범위, 평가 정의, 학습 예산 및 하드웨어 환경이 확정되고 `TBD` 항목 중 학습·평가에 필수적인 조건이 해결되면 통과한다.

---

## Appendix. Pending Decisions

G00 종료 또는 관련 구현 착수 전에 확정해야 할 항목이다.

- [ ] MVP 클래스 이름 및 Original/Model Class ID 기입
- [ ] 데이터 사용 조건의 구체적 출처 기록
- [ ] AP 계산·Recall Threshold·최대 Detection 수·Ignore 처리 정책 확정
- [ ] 데이터 전처리 설정 및 모델 구조의 세부 사항 확정 (G01~G02 착수 전)
- [ ] Scheduler 세부 파라미터 및 실행 환경 버전 기록 (G02~G03 착수 전)

세부 구현 관련 값은 해당 Gate 이전에 확정할 수 있으나, 평가 정의와 데이터 분할 규칙은 첫 본격적인 성능 비교 전에 고정한다.