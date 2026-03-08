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

![Chinchilla Optimal 상세](/assets/img/cme295-lecture-4/image-20260114-033857.png)

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

![Recomputation 설명 1](/assets/img/cme295-lecture-4/image-20260114-035607.png)

![Recomputation 설명 2](/assets/img/cme295-lecture-4/image-20260114-035622.png)

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

![부동소수점 상세](/assets/img/cme295-lecture-4/image-20260115-004255.png)

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

![Mixed Precision 상세 1](/assets/img/cme295-lecture-4/image-20260115-005218.png)

![Mixed Precision 상세 2](/assets/img/cme295-lecture-4/image-20260115-005237.png)

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

![LoRA 상세 1](/assets/img/cme295-lecture-4/image-20260115-033658.png)

![LoRA 상세 2](/assets/img/cme295-lecture-4/image-20260115-033715.png)

![LoRA 상세 3](/assets/img/cme295-lecture-4/image-20260115-033729.png)

![LoRA 상세 4](/assets/img/cme295-lecture-4/image-20260115-033741.png)

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
