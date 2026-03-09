---
layout: post
title: "Stanford CME295: Lecture 4 - LLM Training"
date: 2026-03-08 10:40:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, training, pre-training, fine-tuning, lora, flash-attention, distributed-training]
math: true
---

> **원본 강의**: [YouTube - CME295 Lecture 4](https://www.youtube.com/watch?v=VlA_jt_3Qc4&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=4)

---

## 강의 개요

이번 강의에서는 LLM이 어떻게 학습되는지를 다룹니다. Pre-training부터 Fine-tuning까지의 전체 학습 파이프라인, 분산 학습 기법, 메모리 최적화, 그리고 효율적인 파인튜닝 방법을 학습합니다.

**강의 목표:**

1. Transfer Learning과 LLM 학습 패러다임 이해
2. Pre-training의 규모와 Scaling Laws 파악
3. 분산 학습 및 메모리 최적화 기법 학습
4. Flash Attention의 원리 이해
5. Supervised Fine-Tuning (SFT)과 LoRA 기법 습득

---

## 지난 강의 복습

| 주제 | 핵심 내용 |
| --- | --- |
| **Mixture of Experts (MoE)** | 입력에 따라 일부 Expert만 활성화하여 연산 절약 |
| **토큰 생성 방법** | Greedy, Beam Search, Sampling (+ Temperature) |
| **추론 최적화** | KV Cache, Speculative Decoding 등 |

---

## Part 1: Transfer Learning과 LLM 학습 패러다임

### 1. 기존 ML vs Transfer Learning

**기존 방식 (Task-specific Training):**

```
Task 1 (Spam Detection) → Train Model 1 from scratch
Task 2 (Sentiment Analysis) → Train Model 2 from scratch
Task 3 (NER) → Train Model 3 from scratch
```

**문제점:** 각 태스크가 완전히 독립적이지 않음 (모두 언어 이해 필요)

**Transfer Learning:**

```
Pre-trained Model (언어 이해) → Fine-tune for Task 1
                              → Fine-tune for Task 2
                              → Fine-tune for Task 3
```

### 2. LLM 학습의 두 단계

| 단계 | 목적 | 데이터 | 비용 |
| --- | --- | --- | --- |
| **Pre-training** | 언어/코드 이해 | 방대한 양 (인터넷 전체) | 매우 높음 |
| **Fine-tuning** | 특정 태스크 적응 | 상대적으로 적음 | 낮음 |

---

## Part 2: Pre-training

### 1. Pre-training 개요

**목표:** 다음 토큰 예측 (Next Token Prediction)

**데이터 소스:**

| 소스 | 내용 |
| --- | --- |
| **Common Crawl** | 인터넷 전체 (월 30억 페이지) |
| **Wikipedia** | 백과사전 지식 |
| **Reddit** | 소셜 미디어 대화 |
| **GitHub** | 코드 |
| **Stack Overflow** | 코딩 Q&A |

**데이터 규모:**

| 모델 | 학습 토큰 수 |
| --- | --- |
| GPT-3 | 300B (3천억) |
| LLaMA 3 | 15T (15조) |

### 2. 연산량 측정 단위

#### FLOPs vs FLOPS

| 표기 | 의미 | 설명 |
| --- | --- | --- |
| **FLOPs** (소문자 s) | Floating Point Operations | 연산량 (단위) |
| **FLOPS** (대문자 S) | Floating Point Operations **Per Second** | 연산 속도 |

**LLM 학습 규모:**

- 약 $10^{25}$ FLOPs
- FLOPs ≈ O(토큰 수 × 파라미터 수)

![FLOPs 비교](/assets/img/cme295-lecture-4/image-20260113-075105.png)

### 3. Scaling Laws (스케일링 법칙)

**2020년 논문 발견:**

- 더 많은 **Compute** → 더 좋은 성능
- 더 많은 **Data** → 더 좋은 성능
- 더 큰 **Model** → 더 좋은 성능

**Chinchilla Optimal:**

고정된 연산량에서 최적의 성능을 위한 관계:

$$\text{토큰 수} \approx 20 \times \text{파라미터 수}$$

![Scaling Laws 그래프](/assets/img/cme295-lecture-4/image-20260113-075049.png)

<details>
<summary>Scaling Laws의 계수가 20배인 이유</summary>

#### Chinchilla Optimal이란?

DeepMind의 2022년 논문 "Training Compute-Optimal Large Language Models"에서 나온 개념입니다.

**핵심 질문**

> 고정된 컴퓨팅 예산이 있을 때, 모델 크기와 학습 데이터 양을 어떻게 배분해야 최적인가?

**발견 결과**

$$\text{최적 토큰 수} \approx 20 \times \text{파라미터 수}$$

| 모델 크기 | 최적 학습 토큰 |
|---------|-----------|
| 1B | 20B 토큰 |
| 10B | 200B 토큰 |
| 70B | 1.4T 토큰 |

#### 왜 20배인가?

이론적 유도가 아니라 실험적 발견입니다.

DeepMind는 400개 이상의 모델을 다양한 조건으로 학습시켰습니다:

- 모델 크기: 70M ~ 16B 파라미터
- 데이터 양: 5B ~ 500B 토큰

그 결과를 분석해서 다음 Scaling Law를 fitting했습니다:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

- $N$: 파라미터 수
- $D$: 학습 토큰 수
- $\alpha \approx 0.34, \beta \approx 0.28$ (거의 비슷)

$\alpha$와 $\beta$가 비슷하다는 것은 모델 크기와 데이터 양이 비슷한 비율로 성능에 기여한다는 의미입니다. 따라서 컴퓨팅 예산을 한쪽에 몰빵하지 말고 균형 있게 배분해야 합니다.

20이라는 숫자 자체는 실험 데이터에서 나온 경험적 상수입니다.

#### 이게 왜 중요했나?

기존 관행이 틀렸음을 증명:

| 모델 | 파라미터 | 토큰 | 비율 | 상태 |
|------|---------|------|------|------|
| Gopher | 280B | 300B | ~1배 | Under-trained |
| GPT-3 | 175B | 300B | ~1.7배 | Under-trained |
| **Chinchilla** | **70B** | **1.4T** | **20배** | **Optimal** |

결과: Chinchilla(70B)가 Gopher(280B)보다 4배 작으면서도 성능이 더 좋았습니다.

기존에는 "모델을 크게 만들면 성능이 좋아진다"고 생각했지만, 실제로는 많은 모델들이 **데이터 부족 상태(under-trained)**였던 것입니다.

#### 요약

| 질문 | 답 |
|------|-----|
| 20배는 이론적인가? | 아니오, 실험적 발견 |
| 왜 20배인가? | Scaling law fitting 결과 |
| 핵심 교훈 | 모델만 키우지 말고 데이터도 충분히 |

</details>

**예시:**

- GPT-3: 175B 파라미터, 300B 토큰 → **Undertrained** (이론적으로 3.5T 토큰 필요)

### 4. Pre-training의 도전과제

| 도전과제 | 설명 |
| --- | --- |
| **비용** | 수백만 ~ 수억 달러 |
| **시간** | 수 주 ~ 수 개월 |
| **환경 영향** | 막대한 탄소 배출 |
| **Knowledge Cutoff** | 학습 시점 이후 정보 모름 |
| **표절 위험** | 학습 데이터 그대로 출력 가능성 |

**Knowledge Cutoff Date:**

- 모델 카드에 명시
- 예: GPT-4의 Knowledge Cutoff는 특정 날짜로 제한

---

## Part 3: 분산 학습 (Distributed Training)

### 1. 학습 과정에서 저장해야 할 것들

```
Forward Pass:
  - Activations (각 레이어의 중간값)
Backward Pass:
  - Gradients (손실의 미분값)
Weight Update (Adam Optimizer):
  - First Moment (그래디언트 이동 평균)
  - Second Moment (그래디언트 제곱 이동 평균)
  - Model Weights
```

**문제:** GPU 메모리 제한 (H100: ~80GB)

### 2. Data Parallelism (DP)

**아이디어:** 데이터를 여러 GPU에 분배

![Data Parallelism](/assets/img/cme295-lecture-4/image-20260113-075120.png)

```
┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU 0   │ │ GPU 1   │ │ GPU 2   │
│ Model   │ │ Model   │ │ Model   │
│ Copy    │ │ Copy    │ │ Copy    │
│         │ │         │ │         │
│ Batch 0 │ │ Batch 1 │ │ Batch 2 │
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └──────────┬┴───────────┘
                │
        Gradient 평균화
```

**장점:** 배치 크기에 따른 메모리 절약
**단점:**

- 각 GPU에 전체 모델 복사 필요
- 통신 비용 발생

### 3. ZeRO (Zero Redundancy Optimizer)

**문제:** Data Parallelism에서 중복 정보 많음

| 단계 | 분산 대상 | 메모리 절약 |
| --- | --- | --- |
| **ZeRO-1** | Optimizer States | 높음 |
| **ZeRO-2** | + Gradients | 더 높음 |
| **ZeRO-3** | + Parameters | 최대 |

**Trade-off:** 메모리 절약 ↔ 통신 비용 증가

### 4. Model Parallelism

| 기법 | 설명 |
| --- | --- |
| **Expert Parallelism** | MoE의 각 Expert를 다른 GPU에 배치 |
| **Tensor Parallelism** | 큰 행렬 연산을 분할 |
| **Pipeline Parallelism** | 레이어들을 다른 GPU에 분배 |

**Pipeline Parallelism 예시:**

```
GPU 0: Layer 1, 2, 3
GPU 1: Layer 4, 5, 6
GPU 2: Layer 7, 8, 9
```

---

## Part 4: Flash Attention

### 1. GPU 메모리 구조

| 메모리 | 용량 | 속도 | 역할 |
| --- | --- | --- | --- |
| **HBM** (High Bandwidth Memory) | ~80GB | ~수 TB/s | 메인 메모리 |
| **SRAM** (Static RAM) | ~수십 MB | ~수십 TB/s | 온칩 캐시 |

**핵심:** SRAM이 10배 이상 빠르지만 용량이 매우 작음

### 2. 기존 Attention의 문제

**Self-Attention 공식:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

![Attention 연산 흐름](/assets/img/cme295-lecture-4/image-20260113-075134.png)

**Vanilla 방식의 HBM 접근:**

```
1. HBM에서 Q, K 로드 → QKᵀ 계산 → HBM에 저장
2. HBM에서 QKᵀ 로드 → Softmax 계산 → HBM에 저장
3. HBM에서 Softmax 결과, V 로드 → 곱셈 → HBM에 저장
```

**문제:** HBM 읽기/쓰기가 **병목** (연산보다 데이터 이동이 느림)

### 3. Flash Attention의 핵심 아이디어

**Tiling (타일링):**

- 전체 행렬을 작은 블록으로 분할
- 각 블록을 빠른 SRAM에서 처리
- 최종 결과만 HBM에 저장

**수학적 트릭 - Softmax 분할:**

$$\text{softmax}([S_1, S_2, ..., S_n]) = [\alpha_1 \cdot \text{softmax}(S_1), ..., \alpha_n \cdot \text{softmax}(S_n)]$$

![Softmax 분할](/assets/img/cme295-lecture-4/image-20260113-075148.png)

- 각 블록의 softmax를 독립적으로 계산
- Scaling factor $\alpha$로 조정

![Flash Attention 동작 원리](/assets/img/cme295-lecture-4/image-20260113-075201.png)

```
┌───────────────────────────────────┐
│ HBM                              │
│ ┌─────┐ ┌─────┐ ┌─────┐         │
│ │ Q₁  │ │ K₁  │ │ V₁  │         │
│ │ Q₂  │ │ K₂  │ │ V₂  │         │
│ └──┬──┘ └──┬──┘ └──┬──┘         │
│    │       │       │             │
└────┼───────┼───────┼─────────────┘
     │       │       │
     ▼       ▼       ▼
┌─────────────────────────────────┐
│ SRAM (Fast)                     │
│ [블록 로드] → [연산] → [결과]     │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│ 최종 결과만 HBM에 저장            │
└─────────────────────────────────┘
```

### 4. Recomputation (재계산)

**아이디어:** Forward에서 저장하지 않고, Backward에서 재계산

**일반적 상황:**

- 재계산 → 연산 증가 + 메모리 절약 + 시간 증가

**Flash Attention:**

- 재계산 → 연산 증가 + 메모리 절약 + **시간도 감소!**

| 지표 | Standard | Flash Attention |
| --- | --- | --- |
| HBM 접근 | 40.3 GB | 4.4 GB (~10x↓) |
| Runtime | 느림 | 빠름 |
| Memory | 높음 | 낮음 |

**핵심:** HBM 접근 감소가 연산 증가보다 더 큰 이득

<details>
<summary>Recomputation(재계산) 상세</summary>

#### Recomputation(재계산) 상세 설명

**먼저: 왜 중간 값을 저장해야 하는가?**

신경망 학습에서 **Backward(역전파)** 시 gradient를 계산하려면 **Forward**에서 계산했던 중간 값들이 필요합니다.

```
Forward:  x → [Layer 1] → a → [Layer 2] → b → [Layer 3] → y
                          ↑                ↑
                      저장 필요          저장 필요

Backward: gradient 계산 시 a, b 값이 필요 (Chain Rule)
```

#### 1. 일반적인 Recomputation (Gradient Checkpointing)

**문제 상황**

대형 모델에서 모든 중간 활성화 값을 저장하면 메모리 부족

```
Layer 1 → Layer 2 → Layer 3 → ... → Layer 100
  ↓          ↓          ↓                ↓
 저장       저장       저장      ...     저장

→ GPU 메모리 폭발
```

**해결: 일부만 저장하고 나머지는 재계산**

```
Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5 → ...
  ↓                             ↓
 저장                          저장
(체크포인트)                  (체크포인트)

Backward에서 Layer 3의 값이 필요하면?
→ 체크포인트(Layer 1)부터 다시 Forward 계산해서 Layer 3 값 얻음
```

**Trade-off**

| 항목 | 결과 | 이유 |
|------|------|------|
| 메모리 | 절약 | 모든 값 저장 안함 |
| 연산량 | 증가 | Forward를 일부 다시 해야 함 |
| 시간 | 증가 | 재계산에 시간 소요 |

**결론: 메모리가 부족할 때 시간을 희생해서 메모리 확보**

---

#### 2. Flash Attention에서의 Recomputation

**GPU 메모리 구조 이해 (핵심!)**

```
            GPU
  ┌─────────────────────┐
  │  ┌───────────────┐  │
  │  │ SRAM (빠름, 작음) │  │
  │  │  ~20MB, 19TB/s │  │
  │  └───────────────┘  │
  │    ↕ 데이터 이동 (병목!)  │
  │  ┌───────────────┐  │
  │  │ HBM (느림, 큼)  │  │
  │  │ ~80GB, 3TB/s  │  │
  │  └───────────────┘  │
  └─────────────────────┘
```

**핵심: HBM ↔ SRAM 데이터 전송이 병목(bottleneck)**

**기존 Attention의 문제**

```
Q, K, V (HBM에 있음)
      ↓ 읽기
S = Q × Kᵀ  (N×N 행렬, 매우 큼!)
      ↓ HBM에 저장 ← 느림!
P = softmax(S)
      ↓ HBM에 저장 ← 느림!
O = P × V
      ↓
Backward에서 S, P 다시 읽음 ← 느림!
```

**문제: N×N 크기의 S, P를 HBM에 쓰고 읽는 것이 매우 느림**

**Flash Attention의 접근**

```
Q, K, V를 작은 타일로 나눠서 SRAM에 로드
      ↓
SRAM 내에서 S, P, O 계산 (빠름!)
      ↓
S, P를 HBM에 저장하지 않음!
      ↓
Backward에서 S, P 필요하면? → SRAM에서 다시 계산!
```

**왜 재계산이 더 빠른가?**

| 방식 | 동작 | 시간 |
|------|------|------|
| 기존 | S, P를 HBM에 저장 → 나중에 읽기 | 느림 (HBM 접근) |
| Flash | S, P를 저장 안 함 → 필요할 때 재계산 | 빠름 (SRAM 내 계산) |

> HBM 읽기/쓰기 시간 >>> SRAM에서 재계산 시간

**연산은 늘어나지만, 메모리 접근을 줄여서 전체 시간은 감소!**

---

#### 비교 요약

|  | 일반적 Recomputation | Flash Attention |
|--|---------------------|-----------------|
| 목적 | GPU 메모리 부족 해결 | 메모리 대역폭 병목 해결 |
| 저장 안 하는 것 | 일부 레이어 활성화 값 | Attention의 S, P 행렬 |
| 메모리 | 절약 | 절약 |
| 연산 | 증가 | 증가 |
| 시간 | 증가 | 감소! |
| 이유 | 재계산 자체가 추가 비용 | HBM 접근 피하는 게 더 이득 |

**핵심 인사이트: 현대 GPU에서는 연산보다 메모리 접근이 더 비싸다 (Memory-bound). Flash Attention은 이 특성을 활용한 것입니다.**

</details>

---

## Part 5: Mixed Precision Training

### 1. 부동소수점 표현

| 형식 | 비트 | Exponent | Mantissa | 용도 |
| --- | --- | --- | --- | --- |
| **FP64** | 64 | 11 | 52 | 과학 계산 |
| **FP32** | 32 | 8 | 23 | 일반 학습 |
| **FP16** | 16 | 5 | 10 | 빠른 학습 |
| **BF16** | 16 | 8 | 7 | 딥러닝 특화 |

<details>
<summary>Exponent, Mantissa 상세</summary>

**과학적 표기법과의 비교**

쉽게 이해하려면 과학적 표기법을 생각하면 됩니다:

$$1.234 \times 10^5 = 123400$$

여기서:

- **1.234** → Mantissa (가수부)
- **5** → Exponent (지수부)

**부동소수점에서의 역할**

| 구성요소 | 역할 | 비유 |
| --- | --- | --- |
| **Exponent** (지수부) | 숫자의 크기/범위를 결정 | "얼마나 큰 숫자인가" |
| **Mantissa** (가수부) | 숫자의 정밀도를 결정 | "얼마나 정확한가" |

**FP32 예시 (32비트)**

```
[1비트][8비트 Exponent][23비트 Mantissa]
  ↓        ↓                ↓
 부호    크기/범위          정밀도
```

**왜 BF16이 딥러닝에 좋은가?**

| 형식 | Exponent | Mantissa | 특징 |
| --- | --- | --- | --- |
| FP32 | 8비트 | 23비트 | 넓은 범위 + 높은 정밀도 |
| FP16 | 5비트 | 10비트 | 좁은 범위 → 오버플로우 위험 |
| BF16 | 8비트 | 7비트 | FP32와 같은 범위 + 낮은 정밀도 |

**BF16**은 Exponent를 FP32와 동일하게 유지해서 매우 크거나 작은 숫자도 표현 가능하고, Mantissa만 줄여서 메모리를 절약합니다. 딥러닝에서는 정밀도보다 범위가 더 중요하기 때문에 BF16이 딥러닝에 특화된 포맷으로 사용됩니다.

</details>

**GPU 성능 (H100 기준):**

- FP64: 34 TFLOPS
- FP32: 67 TFLOPS
- FP16/BF16: 1,979 TFLOPS (Tensor Core)

### 2. Mixed Precision Training 전략

```
┌─────────────────────────────────────────┐
│ Master Weights (FP32)                   │
│           │                             │
│           ▼                             │
│ Cast to FP16 for Forward                │
│           │                             │
│           ▼                             │
│ Forward Pass (FP16) → Loss (FP16)       │
│           │                             │
│           ▼                             │
│ Backward Pass (FP16) → Gradients        │
│           │                             │
│           ▼                             │
│ Weight Update in FP32 (정밀도 유지)       │
└─────────────────────────────────────────┘
```

**왜 Weights는 FP32로 유지?**

- Activations: 노이즈 있는 데이터에서 계산, 낮은 정밀도 OK
- Weights: 누적 오차 방지를 위해 높은 정밀도 필요

**장점:**

- 메모리 절약 (약 50%)
- 연산 속도 향상
- 성능 저하 거의 없음

<details>
<summary>Mixed Precision Training 단계별 상세 (성능 저하 없이 작동하는 이유)</summary>

**전체 흐름**

```
Master Weights (FP32)
       │
       ▼ Cast to FP16
Forward Pass (FP16) → Activations (FP16)
       │
       ▼
    Loss 계산 (FP16)
       │
       ▼
Backward Pass (FP16) → Gradients (FP16)
       │
       ▼
  Gradients를 FP32로 변환
       │
       ▼
  Weight Update (FP32)
  W_new = W_old - lr × gradient
```

**각 단계별 상세 설명**

**1단계: Master Weights (FP32 유지)**

```
Master Weights: [0.12345678, -0.98765432, 0.00001234, ...]
                        ↑ 높은 정밀도 유지
```

**왜 FP32인가?**

- Weight Update 시 아주 작은 값이 누적됨
- 예: learning rate = 0.0001, gradient = 0.001 → update = 0.0000001
- FP16은 이런 작은 값을 표현 못함 → 누적 오차 발생

**2단계: Forward Pass (FP16)**

```
Input (FP16) → [Layer 1] → Activation → [Layer 2] → ... → Output
                    ↑
            Weights를 FP16으로 캐스팅
```

**왜 FP16이어도 괜찮은가?**

| 이유 | 설명 |
| --- | --- |
| 입력 데이터 자체가 노이즈 | 이미지, 텍스트 등은 원래 불완전함 |
| Activation은 대략적 값이면 충분 | 정확한 1.23456789보다 1.234면 OK |
| 통계적으로 오차가 상쇄 | 수백만 개 값의 작은 오차는 평균화됨 |

비유: 사진을 약간 흐리게 해도 "고양이"인지 알 수 있음

**3단계: Loss 계산 (FP16)**

```
Output (FP16): [0.8, 0.1, 0.1]
Target:        [1.0, 0.0, 0.0]
                    ↓
Loss = CrossEntropy(...) → 0.223 (FP16)
```

**왜 괜찮은가?**

- Loss는 상대적 비교에 사용됨
- "Loss가 0.223이나 0.2234567이나"는 학습 방향에 큰 영향 없음

**4단계: Backward Pass (FP16)**

```
Loss → ∂L/∂W (Gradients)

Gradient 예시: [-0.0312, 0.0156, -0.0078, ...]
                        ↑
                  FP16으로 계산
```

**왜 괜찮은가?**

- Gradient는 방향이 중요하지, 정확한 크기는 덜 중요
- SGD의 본질: "대략 이 방향으로 가면 됨"
- Mini-batch 자체가 근사값이므로 gradient도 근사로 충분

**5단계: Weight Update (FP32) - 핵심!**

```
# 이 단계가 FP32인 이유가 가장 중요

FP32 Master Weight: 0.12345678
FP32 Gradient:      0.00000012  (아주 작은 값!)
Learning Rate:      0.0001
                    ↓
Update = 0.0001 × 0.00000012 = 0.000000000012
                    ↓
New Weight = 0.12345678 - 0.000000000012 = 0.123456779988...
```

**FP16이면 문제가 되는 이유:**

```
FP16 Weight:   0.1234  (4자리 정밀도)
Update:        0.000000000012
                    ↓
0.1234 + 0.000000000012 = 0.1234  ← 변화 없음! (반올림됨)
```

**수만 번 반복하면:**

- **FP32:** 작은 업데이트가 누적되어 학습 진행
- **FP16:** 업데이트가 사라져서 학습 멈춤

**왜 Activations는 FP16이어도 되는가?**

```
┌──────────────────────────────────────────┐
│          Activations vs Weights          │
├────────────────────┬─────────────────────┤
│ Activations (FP16 OK) │ Weights (FP32 필요)  │
├────────────────────┼─────────────────────┤
│ · 매 배치마다 새로 계산  │ · 전체 학습 동안 누적   │
│ · 일회성 값            │ · 수백만 번 업데이트    │
│ · 노이즈 있는 데이터 기반 │ · 작은 변화 누적 필요   │
│ · 오차가 평균화됨       │ · 오차가 쌓이면 발산    │
└────────────────────┴─────────────────────┘
```

**비유:**

- **Activations** = 매일 새로 쓰는 메모 (대충 써도 됨)
- **Weights** = 은행 잔고 (1원도 정확히 기록해야 함)

**추가 안전장치: Loss Scaling**

FP16의 문제: 아주 작은 gradient가 언더플로우로 0이 됨

```
Gradient: 0.00001 → FP16에서 0으로 반올림됨
```

**해결책: Loss Scaling**

```
1. Loss를 큰 수로 곱함 (예: ×1024)
   Loss: 0.5 → 512

2. Backward Pass (gradient도 자동으로 커짐)
   Gradient: 0.00001 → 0.01024 (FP16 표현 가능!)

3. Weight Update 전에 다시 나눔
   Gradient: 0.01024 ÷ 1024 = 0.00001
```

**요약**

| 단계 | 정밀도 | 이유 |
| --- | --- | --- |
| Master Weights | FP32 | 작은 업데이트 누적 보존 |
| Forward Pass | FP16 | 노이즈 있는 데이터, 대략적 값 충분 |
| Loss 계산 | FP16 | 상대적 비교용, 정밀도 덜 중요 |
| Backward Pass | FP16 | 방향이 중요, 정확한 크기는 덜 중요 |
| Weight Update | FP32 | 누적 오차 방지의 핵심 |

핵심 인사이트: 딥러닝은 본질적으로 근사적(approximate) 알고리즘이라서, 대부분의 연산은 낮은 정밀도로 충분합니다. 단, **누적되는 값(weights)**만 높은 정밀도가 필요합니다.

</details>

---

## Part 6: Supervised Fine-Tuning (SFT)

### 1. Pre-trained Model의 한계

**문제:** Pre-trained 모델은 "도움이 되는 답변"을 하도록 학습되지 않음

```
User: "테디베어를 세탁기에 넣어도 되나요?"
Pre-trained Model: "테디베어는 보통 폴리에스터 소재로..."
  (질문에 답하지 않고 관련 텍스트 생성)
Fine-tuned Model: "테디베어는 손세탁을 권장합니다. 세탁기 사용 시..."
  (질문에 직접 답변)
```

### 2. SFT (Supervised Fine-Tuning) 개요

**정의:** 레이블이 있는 (Input, Output) 쌍으로 학습

**Pre-training vs SFT:**

| 구분 | Pre-training | SFT |
| --- | --- | --- |
| 목표 | 언어 이해 | 특정 태스크 수행 |
| 데이터 | 전체 텍스트 예측 | Input → Output |
| 손실 계산 | 전체 시퀀스 | Output 부분만 |
| 데이터 양 | 수조 토큰 | 수천만 예시 |

**SFT 손실 함수:**

```
Input: "Do X"
Output: "Here is how to do X..."
Loss = CrossEntropy(Output 부분만)
  └─ Input 부분은 손실 계산에서 제외
```

### 3. Instruction Tuning

**목적:** 모델을 "지시를 따르는 도우미"로 변환

**데이터 구성:**

| 카테고리 | 예시 |
| --- | --- |
| **Assistant Dialogues** | 스토리 작성, 시 창작, 리스트 생성, 설명 |
| **Code** | 코드 작성, 디버깅, 설명 |
| **Math** | 수학 문제 풀이, 증명 |
| **Safety** | 유해 요청 거부, 안전한 응답 |

**데이터 생성 방법:**

1. **Human-written:** 전문가가 직접 작성 (고품질, 고비용)
2. **LLM-generated:** 기존 LLM으로 생성 후 검수 (효율적)

**데이터 규모:**

| 모델 | SFT 예시 수 |
| --- | --- |
| GPT-3 | ~13K |
| LLaMA 3 | ~10M |

### 4. SFT의 도전과제

| 도전과제 | 설명 |
| --- | --- |
| **고품질 데이터 필요** | Human-in-the-loop 비용 |
| **분포 불일치** | 학습 분포 ≠ 실제 사용 분포 |
| **평가 어려움** | 주관적인 "좋은 답변" 정의 |
| **계산 비용** | 전체 모델 Fine-tuning 비용 |

---

## Part 7: 모델 평가 (Evaluation)

### 1. 벤치마크 기반 평가

| 벤치마크 | 평가 영역 | 설명 |
| --- | --- | --- |
| **MMLU** | 일반 언어 | 50+ 태스크의 다중 선택 |
| **GSM8K** | 수학 추론 | 8K 수학 문제 |
| **HumanEval** | 코드 생성 | 프로그래밍 문제 |
| **HellaSwag** | 상식 추론 | 문장 완성 |

**주의:** "Training on the Test Task" 현상

- 벤치마크와 유사한 데이터로 학습하면 점수 급상승
- 공정한 비교를 위해 학습 데이터 공개 필요

### 2. Chatbot Arena

**방식:**

1. 사용자가 질문 입력
2. 두 모델의 응답을 나란히 표시 (익명)
3. 사용자가 선호하는 응답 선택
4. ELO 스타일 랭킹 계산

**한계:**

- 초기 매칭에 따른 노이즈
- 모델 식별을 통한 조작 가능성
- 사용자 선호 ≠ 객관적 품질
- 안전한 거부 응답에 대한 편향

---

## Part 8: LoRA (Low-Rank Adaptation)

### 1. Full Fine-tuning의 문제

**문제:** 전체 파라미터 업데이트 → 매우 비쌈

| 모델 | 파라미터 | FP32 메모리 |
| --- | --- | --- |
| LLaMA 7B | 7B | ~28GB |
| LLaMA 70B | 70B | ~280GB |

### 2. LoRA의 핵심 아이디어

**가정:** Fine-tuning의 가중치 변화는 **Low-Rank**

$$W = W_0 + \Delta W = W_0 + BA$$

![LoRA 구조](/assets/img/cme295-lecture-4/image-20260113-075250.png)

- $W_0$: Pre-trained 가중치 (Frozen)
- $B \in \mathbb{R}^{d \times r}$: 학습 가능
- $A \in \mathbb{R}^{r \times k}$: 학습 가능
- $r$: Rank (보통 4~16, $r \ll d, k$)

```
기존: W ∈ ℝ^(d×k) → d × k 파라미터 학습
LoRA: W₀ (frozen) + B × A
  B ∈ ℝ^(d×r) → d × r 파라미터
  A ∈ ℝ^(r×k) → r × k 파라미터
  총: r × (d + k) 파라미터
```

**파라미터 절약 예시:**

- $d = k = 4096$, $r = 8$
- Full: $4096 \times 4096 = 16.7M$
- LoRA: $8 \times (4096 + 4096) = 65.5K$
- **약 250배 절약!**

<details>
<summary>LoRA 개념 상세</summary>

#### 1. 먼저 기본 개념: 신경망의 가중치 행렬

**Transformer의 Linear Layer**

Transformer 안에는 여러 Linear Layer가 있어요 (Q, K, V 프로젝션, FFN 등).

```
입력 x ∈ ℝᵏ  →  [W]  →  출력 y ∈ ℝᵈ

y = Wx
```

**여기서 W는 $d \times k$ 행렬이에요:**

- $k$ = 입력 차원 (input dimension)
- $d$ = 출력 차원 (output dimension)

**예시: LLM에서 hidden dimension이 4096이라면**

- $k = 4096$ (입력 벡터 크기)
- $d = 4096$ (출력 벡터 크기)
- $W$는 $4096 \times 4096$ = 약 **1670**만 개의 파라미터

#### 2. Full Fine-tuning의 문제

Fine-tuning할 때 이 $W$ 전체를 업데이트한다고 생각해보세요:

```
W_new = W_old + ΔW
```

- $\Delta W$도 $d \times k$ 행렬 = 1670만 개 파라미터를 학습해야 함
- LLM에는 이런 Linear Layer가 수십~수백 개
- 총 파라미터가 수십억 개 → 메모리, 시간 엄청 많이 듦

#### 3. Low-Rank의 의미

**"Rank"란?**

행렬의 **Rank**는 쉽게 말해 **"행렬이 담고 있는 독립적인 정보의 양"**이에요.

```
예시: 3×3 행렬

Full Rank (Rank = 3):          Low Rank (Rank = 1):

┌         ┐                    ┌         ┐
│ 1  2  3 │                    │ 1  2  3 │
│ 4  5  6 │                    │ 2  4  6 │  ← 첫 행의 2배
│ 7  8  9 │                    │ 3  6  9 │  ← 첫 행의 3배
└         ┘                    └         ┘

→ 3개 행이 모두 독립적        → 실제로는 1개 행의 정보만 있음
```

**"Low-Rank"란?**

**Low-Rank** 행렬 = 큰 행렬이지만, 실제 담긴 정보는 적은 행렬

#### 4. Low-Rank 행렬은 분해할 수 있다!

**핵심 수학적 사실:**

Rank가 $r$인 $d \times k$ 행렬은 항상 두 개의 작은 행렬의 곱으로 표현 가능!

$$M_{d \times k} = B_{d \times r} \times A_{r \times k}$$

**시각화:**

```
Low-Rank 행렬 분해

d×k 행렬 M        =    d×r 행렬 B  ×  r×k 행렬 A

┌          ┐           ┌     ┐      ┌          ┐
│          │     =     │     │  ×   │          │
d│    M     │     d     │  B  │      │    A     │
│          │           │     │  r   │          │
└          ┘           └     ┘      └          ┘
      k                   r              k

원소 개수: d × k            원소 개수: d×r + r×k

만약 r이 작으면 (r << d, k):
d×r + r×k  <<  d×k  (훨씬 적은 파라미터!)
```

#### 5. LoRA의 핵심 가정

**LoRA의 핵심 통찰:**

> "Fine-tuning할 때 가중치 변화량 $\Delta W$는 Low-Rank일 것이다"

왜 이런 가정이 합리적일까요?

- Pre-trained 모델은 이미 좋은 표현을 학습함
- Fine-tuning은 약간의 조정만 필요
- 약간의 조정 = 정보량이 적음 = **Low-Rank**로 충분

#### 6. LoRA 적용 방법

**기존 Full Fine-tuning:**

```
W_new = W_old + ΔW

ΔW ∈ ℝ^(d×k)  →  d × k개 파라미터 학습
```

**LoRA:**

```
W_new = W_old + B × A

W_old: Frozen (학습 안 함!)
B ∈ ℝ^(d×r):  학습
A ∈ ℝ^(r×k):  학습
```

**그림으로 보면:**

```
                    LoRA 구조

           ┌─────────────────┐
           │                 │
 입력 x ──→│  W₀ (frozen)    │──┐
           │                 │  │
           └─────────────────┘  │
           │                    ├→ + ──→ 출력 y
           └→┌───┐→┌───┐──────┘
              │ A │  │ B │
              └───┘  └───┘
              r×k    d×r
             (학습)  (학습)

 y = W₀x + BAx
```

#### 7. 파라미터 절약 계산 (구체적 예시)

**설정:**

- $d = 4096$ (출력 차원)
- $k = 4096$ (입력 차원)
- $r = 8$ (LoRA rank, 우리가 선택)

**Full Fine-tuning:**

```
ΔW의 파라미터 수 = d × k = 4096 × 4096 = 16,777,216개 (약 1670만)
```

**LoRA:**

```
B의 파라미터 수 = d × r = 4096 × 8 = 32,768개
A의 파라미터 수 = r × k = 8 × 4096 = 32,768개

총 = 32,768 + 32,768 = 65,536개 (약 6.5만)
```

**비교:**

```
Full:   16,777,216개
LoRA:       65,536개

절약률: 16,777,216 / 65,536 ≈ 256배!
```

#### 8. 왜 이게 가능한가? (직관적 이해)

**비유: 사진 압축**

```
Low-Rank의 직관

원본 사진: 1000 × 1000 픽셀 = 100만 개 값

하지만 사진이 단순하다면? (예: 그라데이션)

  ← 왼쪽에서 오른쪽으로 점점 어두워짐
    모든 행이 같음!

→ 실제로는 1행의 정보만 저장하면 됨
→ 100만 개 대신 1000개만 저장! (Rank = 1)

Fine-tuning의 변화(ΔW)도 이와 비슷하게 "단순"하다는 가정
```

#### 9. r (Rank)의 의미

**r은 "$\Delta W$가 얼마나 복잡한 변화인가"를 결정**

| r 값 | 의미 | 파라미터 수 | 표현력 |
| --- | --- | --- | --- |
| r = 1 | 매우 단순한 변화 | 최소 | 낮음 |
| r = 8 | 적당한 변화 | 적음 | 중간 |
| r = 64 | 복잡한 변화 | 많음 | 높음 |
| r = d | Full Rank | Full과 동일 | 최대 |

**실제로는 r = 4~16 정도로도 충분히 좋은 성능!**

#### 10. 요약

**LoRA 핵심 정리:**

1. **문제:** Full Fine-tuning은 파라미터가 너무 많음
2. **가정:** Fine-tuning의 변화($\Delta W$)는 Low-Rank (단순함)
3. **해결:** $\Delta W = B \times A$로 분해 ($B$: $d \times r$, $A$: $r \times k$)
4. **변수 의미:**
   - $d$: 출력 차원
   - $k$: 입력 차원
   - $r$: Rank (우리가 선택, 보통 4~16)
   - $W_0$: 원래 가중치 (Frozen, 학습 안 함)
   - $B$, $A$: 새로 추가된 작은 행렬 (학습함)
5. **결과:**
   - 파라미터: $d \times k$ → $r \times (d + k)$ (수백 배 절약)
   - 성능: Full Fine-tuning과 거의 동일!

</details>

### 3. LoRA 적용 위치

| 위치 | 원래 논문 | 최신 연구 |
| --- | --- | --- |
| **Attention (Q, K, V, O)** | 권장 | 효과 있음 |
| **FFN** | 미언급 | **가장 효과적** |

**최신 권장:** Attention + FFN 모두에 적용

### 4. LoRA 학습 팁

| 팁 | 권장값 | 이유 |
| --- | --- | --- |
| **Learning Rate** | 10x 높게 | 낮은 Rank로 인한 제한된 탐색 공간 |
| **Batch Size** | 작게 유지 | 행렬 곱의 학습 동역학 차이 |

### 5. QLoRA (Quantized LoRA)

**아이디어:** $W_0$를 4비트로 양자화

```
W₀: FP32 (frozen) → NF4 (4-bit, frozen)
A, B: BF16 (학습)
```

**NF4 (Normal Float 4-bit):**

- 가중치가 정규분포를 따른다고 가정
- 고정 버킷 대신 **분위수 기반** 양자화
- 각 분위수에 동일한 수의 값 배치

**Double Quantization:**

- 양자화 상수도 추가로 양자화

**결과:**

- **16배** VRAM 절약
- 성능 저하 최소화

---

## Part 9: LLM 학습 파이프라인 요약

### 전체 학습 단계

```
┌─────────────────────────────────────────────────────────────┐
│ LLM Training Pipeline                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Pre-training                                             │
│    └─ 목표: 언어 이해                                        │
│    └─ 데이터: 인터넷 전체 (수조 토큰)                          │
│    └─ 비용: 수백만~수억 달러                                  │
│              │                                              │
│              ▼                                              │
│ 2. Mid-training (선택적, 최신 트렌드)                         │
│    └─ 목표: 특정 도메인 강화                                  │
│    └─ 같은 Pre-training 목표, 다른 데이터                     │
│              │                                              │
│              ▼                                              │
│ 3. Supervised Fine-Tuning (SFT)                             │
│    └─ 목표: 도움이 되는 응답 생성                              │
│    └─ 데이터: (Instruction, Response) 쌍                     │
│    └─ 기법: Full Fine-tuning 또는 LoRA                       │
│              │                                              │
│              ▼                                              │
│ 4. Preference Tuning (다음 강의)                             │
│    └─ 목표: 인간 선호에 맞는 응답                              │
│                                                             │
│ ※ SFT + Preference Tuning = "Alignment"                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 핵심 요약

### Pre-training

- 방대한 데이터로 언어 이해 학습
- Scaling Laws: 더 큰 모델 + 더 많은 데이터 = 더 좋은 성능
- Chinchilla Optimal: 토큰 수 ≈ 20 × 파라미터 수

### 분산 학습

- **Data Parallelism:** 데이터 분배, 모델 복제
- **ZeRO:** 중복 제거 (Optimizer States, Gradients, Parameters)
- **Model Parallelism:** Expert, Tensor, Pipeline

### Flash Attention

- HBM (느림, 큼) vs SRAM (빠름, 작음)
- Tiling으로 SRAM 활용 극대화
- HBM 접근 10배 감소, 속도와 메모리 모두 개선

### Mixed Precision

- FP32 가중치 + FP16 연산
- 메모리 절약 + 속도 향상

### SFT (Supervised Fine-Tuning)

- Pre-trained 모델을 "도움이 되는 도우미"로 변환
- Instruction Tuning: 지시 따르기 학습

### LoRA

$$W = W_0 + BA$$

![LoRA 수식](/assets/img/cme295-lecture-4/image-20260113-075329.png)

- Pre-trained 가중치 동결
- Low-Rank 행렬만 학습
- 파라미터 수백 배 절약

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| FLOPs | Floating Point Operations (연산량) |
| FLOPS | Floating Point Operations Per Second (연산 속도) |
| HBM | High Bandwidth Memory |
| SRAM | Static Random Access Memory |
| DP | Data Parallelism |
| ZeRO | Zero Redundancy Optimizer |
| SFT | Supervised Fine-Tuning |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| MMLU | Massive Multitask Language Understanding |
| NF4 | Normal Float 4-bit |

---

## 핵심 공식

**Chinchilla Optimal:**

$$N_{\text{tokens}} \approx 20 \times N_{\text{parameters}}$$

**LoRA:**

$$W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

**Flash Attention Softmax 분할:**

$$\text{softmax}([S_1, ..., S_n]) = [\alpha_1 \cdot \text{softmax}(S_1), ..., \alpha_n \cdot \text{softmax}(S_n)]$$

![핵심 공식 요약](/assets/img/cme295-lecture-4/image-20260113-075347.png)

---

## 추천 자료

1. **"Scaling Laws for Neural Language Models"** (2020) - Scaling Laws
2. **"Training Compute-Optimal LLMs"** (2022) - Chinchilla
3. **"FlashAttention"** (2022) - 메모리 효율적 Attention
4. **"LoRA"** (2021) - 효율적 Fine-tuning
5. **"QLoRA"** (2023) - 양자화된 LoRA
6. **"Mixed Precision Training"** (2018) - 혼합 정밀도

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 4 정리*
