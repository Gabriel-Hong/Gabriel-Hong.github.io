---
layout: post
title: "Stanford CME295: Lecture 3 - LLMs & 추론 최적화"
date: 2026-03-08 10:30:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, moe, kv-cache, speculative-decoding, temperature, prompting, cot]
---

> **강의 출처**: Stanford CME295 - Transformers & LLMs (Autumn 2025)
>
> - **강사**: Afshine & Shervine Amidi
> - **원본 영상**: [YouTube](https://www.youtube.com/watch?v=Q5baLehv5So&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=3)

---

## 강의 개요

이번 강의에서는 Large Language Model(LLM)의 정의와 구조, 토큰 생성 방법, 프롬프팅 기법, 그리고 추론 최적화 기술들을 다룹니다.

---

## Part 1: Large Language Model (LLM)

### 1. LLM의 정의

**Language Model이란?** 토큰 시퀀스에 **확률을 할당**하는 모델

**"Large"의 의미:**

| 측면 | 규모 | 설명 |
| --- | --- | --- |
| **모델 크기** | 수십억 ~ 수천억 파라미터 | 최소 1B (10억) 이상 |
| **훈련 데이터** | 수천억 ~ 수조 토큰 | 최대 수십 조 토큰 |
| **컴퓨팅** | 대규모 GPU 클러스터 필요 | 최근에는 소비자급 GPU 최적화 진행 중 |

**중요:** 현재 LLM 정의에서 BERT는 LLM이 아님 (텍스트를 생성하지 않으므로)

### 2. LLM의 기본 구조

```
┌─────────────────────────────┐
│       Decoder Only          │
│                             │
│ ┌───────────────────────┐   │
│ │ Masked Self-Attention │   │
│ └───────────┬───────────┘   │
│ ┌───────────▼───────────┐   │
│ │     Add & Norm        │   │
│ └───────────┬───────────┘   │
│ ┌───────────▼───────────┐   │
│ │  Feed-Forward Net     │   │
│ └───────────┬───────────┘   │
│ ┌───────────▼───────────┐   │
│ │     Add & Norm        │   │
│ └───────────────────────┘   │
│                             │
│          × N layers         │
└─────────────────────────────┘
```

**대표적인 LLM들:** GPT (OpenAI), LLaMA (Meta), Gemma (Google), DeepSeek, Mistral, Qwen

---

## Part 2: Mixture of Experts (MoE)

### 1. MoE의 동기

**문제 제기:** 수천억 파라미터 모델에서 모든 파라미터가 항상 필요한가?

**비유:** 방에 수학자, 물리학자, 화학자, 역사학자가 있을 때, 수학 문제는 수학자에게만 물으면 됨

**핵심 아이디어:** 입력에 따라 **일부 파라미터만 활성화**

### 2. MoE의 구조

$$\hat{y} = \sum\_{i=1}^{n} G(x)\_i \cdot E\_i(x)$$

* $n$: Expert 수
* $E\_i$: i번째 Expert (네트워크)
* $G$: Gate/Router (어떤 Expert를 사용할지 결정)

### 3. Dense MoE vs Sparse MoE

| 유형 | 특징 | 활성화 방식 |
| --- | --- | --- |
| **Dense MoE** | 모든 Expert 사용 | 가중치로 중요도 조절 (0~1) |
| **Sparse MoE** | Top-K Expert만 사용 | K=1 또는 K=2가 일반적 |

<details>
<summary>Dense MoE vs Sparse MoE 개념</summary>

**Dense MoE**

모든 Expert를 사용하되, 가중치(0~1)로 중요도를 조절합니다. 즉, 모든 전문가의 의견을 다 듣되, 각 전문가에게 서로 다른 비중을 두어 종합하는 방식입니다.

$$\hat{y} = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

**Sparse MoE**

Top-K개의 Expert만 선택해서 사용합니다. 일반적으로 K=1 또는 K=2를 사용합니다. 가장 적합한 전문가 1~2명에게만 물어보는 방식이죠.

$$\hat{y} = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)$$

**핵심 차이점:**

| 구분 | Dense MoE | Sparse MoE |
| --- | --- | --- |
| 활성화 | 모든 Expert | Top-K Expert만 |
| 연산량(FLOPs) | 높음 | 낮음 |
| 효율성 | 낮음 | 높음 |

</details>

**Sparse MoE 공식:**

$$\hat{y} = \sum\_{i \in \text{TopK}} G(x)\_i \cdot E\_i(x)$$

### 4. Expert의 위치: FFN

**왜 FFN에 MoE를 적용하는가?**

| 레이어 | 파라미터 규모 | 이유 |
| --- | --- | --- |
| **FFN** | $O(d\_{model} \times d\_{ff} \times 2)$ | $d\_{ff}$가 매우 큼 (수천~만) |
| **Attention** | $O(d\_{model} \times d\_k \times 4)$ | $d\_k$가 상대적으로 작음 (수백) |

### 5. Routing Collapse 문제

**문제:** 일부 Expert만 계속 선택되고 나머지는 사용되지 않음

**해결책: Load Balancing Loss**

$$L\_{aux} = \alpha \cdot n \cdot \sum\_{i=1}^{n} f\_i \cdot P\_i$$

* $f\_i$: Expert i로 라우팅된 토큰 비율
* $P\_i$: Expert i의 평균 라우팅 확률

**MoE의 장점:**
* 모델 **용량(capacity)** 증가 가능
* **Active 파라미터**는 유지하면서 전체 파라미터 확장
* 더 **Sample Efficient** (같은 성능을 더 빠르게 달성)

---

## Part 3: 토큰 생성 (Token Generation)

### 1. 출력 확률 분포

```
Input: "A cute teddy bear is"
Output Distribution:
  - "reading":  0.15
  - "fluffy":   0.12
  - "sleeping": 0.10
  - "airplane": 0.001
  ...
```

### 2. Greedy Decoding

가장 높은 확률의 토큰 선택

**문제점:** 다양성 부족, 지역 최적(Local Optimal)

### 3. Beam Search

K개의 가장 유망한 경로 유지

**시퀀스 확률 계산:**

$$\log P(\text{sequence}) = \sum\_{t} \log P(w\_t | w\_1, ..., w\_{t-1})$$

**사용 사례:** 기계 번역처럼 정확성이 중요한 경우

### 4. Sampling 방법 (가장 많이 사용)

#### 4.1 Top-K Sampling

상위 K개 토큰에서만 샘플링

#### 4.2 Top-P Sampling (Nucleus Sampling)

누적 확률이 P를 초과할 때까지의 토큰에서 샘플링

### 5. Temperature

$$P(w\_i) = \frac{\exp(x\_i / T)}{\sum\_j \exp(x\_j / T)}$$

| Temperature | 분포 형태 | 출력 특성 |
| --- | --- | --- |
| **T → 0** | Spiky (뾰족) | 가장 높은 확률 토큰만 선택, 결정론적 |
| **T = 1** | 원래 분포 | 모델이 학습한 분포 그대로 |
| **T → ∞** | Uniform (균등) | 모든 토큰 동등한 확률, 매우 창의적/무작위 |

**수학적 증명:**

$T \to 0$ 일 때, $k$가 최대 logit의 인덱스라면:

$$P(w\_k) = \frac{1}{1 + \sum\_{j \neq k} \exp((x\_j - x\_k)/T)} \to 1$$

<details>
<summary>logit이란?</summary>

**Logit이란?**

Logit은 신경망의 마지막 층에서 **softmax를 적용하기 전의 원시(raw) 출력값**입니다.

**LLM에서의 흐름:**

```
입력 텍스트 → Decoder 층들 → 마지막 Linear 층 → [logits] → Softmax → [확률 분포]
```

**예시:**

```
입력: "A cute teddy bear is"

Logits (원시 점수):        Softmax 적용 후 (확률):

| "fluffy"   |  3.2 |     | "fluffy"   | 0.15  |
| "sleeping" |  2.8 | →   | "sleeping" | 0.10  |
| "cute"     |  2.5 |     | "cute"     | 0.08  |
| "airplane" | -1.0 |     | "airplane" | 0.001 |
| ...        |  ... |     | ...        | ...   |

     (합 ≠ 1)                  (합 = 1)
```

**핵심 특징:**

| Logit | 확률 (Softmax 후) |
| --- | --- |
| 범위 제한 없음 (-∞ ~ +∞) | 0~1 사이 |
| 합이 1이 아님 | 합이 정확히 1 |
| 상대적 점수 | 실제 확률 |

**Temperature**는 바로 이 logit을 softmax에 넣기 전에 나눠주는 값입니다:

$$P(w_i) = \frac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}$$

여기서 $x_i$가 logit입니다.

</details>

<details>
<summary>Temperature 증명 상세</summary>

#### Temperature 수학적 증명 설명

**1. 왜 $x_j - x_k < 0$ 인가?**

핵심은 $k$의 정의에 있습니다.

> "$k$가 최대 logit의 인덱스"

즉, $x_k$는 모든 logit 중에서 가장 큰 값입니다.

예시:

```
logits = [2.0, 5.0, 1.0, 3.0]
                ↑
           x_k = 5.0 (최댓값, k=1)
```

따라서 $j \neq k$인 다른 모든 logit은 $x_k$보다 작습니다:

- $x_0 - x_k = 2.0 - 5.0 = -3.0 < 0$
- $x_2 - x_k = 1.0 - 5.0 = -4.0 < 0$
- $x_3 - x_k = 3.0 - 5.0 = -2.0 < 0$

최댓값에서 빼니까 항상 음수가 됩니다.

---

**2. 왜 $P(w_k) \to 1$ 인가?**

수식을 다시 보면:

$$P(w_k) = \frac{1}{1 + \sum_{j \neq k} \exp\left(\frac{x_j - x_k}{T}\right)}$$

단계별로 따라가 봅시다:

| 단계 | 내용 |
| --- | --- |
| ① | $x_j - x_k < 0$ (음수) |
| ② | $T \to 0^+$ (0에 가까운 양수) |
| ③ | $\frac{\text{음수}}{\text{매우 작은 양수}} = -\infty$ |
| ④ | $\exp(-\infty) = 0$ |

**예시로 계산:**

```
x_j - x_k = -3.0 이라면:

T = 1.0  →  (-3)/1 = -3      →  exp(-3) ≈ 0.05
T = 0.1  →  (-3)/0.1 = -30   →  exp(-30) ≈ 0.000000000001
T = 0.01 →  (-3)/0.01 = -300 → exp(-300) ≈ 0
```

**결론:**

$$\sum_{j \neq k} \exp\left(\frac{x_j - x_k}{T}\right) \to 0$$

따라서:

$$P(w_k) = \frac{1}{1 + 0} = 1$$

---

**직관적 해석**

Temperature가 0에 가까워지면, softmax가 "winner-take-all" 방식이 됩니다. 가장 높은 logit을 가진 토큰이 확률 1을 독차지하고, 나머지는 모두 0이 됩니다. 그래서 **결정론적(deterministic)**이라고 하는 것입니다.

</details>

### 6. Guided Decoding (구조화된 출력)

JSON 등 특정 포맷 강제. 생성 중 유효하지 않은 토큰 필터링.

**관련 기술:** Finite State Machine (FSM), Context-Free Grammar

---

## Part 4: 프롬프팅 (Prompting)

### 1. Context Length (문맥 길이)

모델이 한 번에 처리할 수 있는 토큰 수. 현대 LLM은 수만 ~ 수백만 토큰.

**Context Rot 현상:** 문맥 길이가 길어질수록 정보 검색 능력 저하. "Needle in a Haystack" 테스트로 측정.

### 2. 프롬프트 구조

| 구성 요소 | 설명 | 예시 |
| --- | --- | --- |
| **Context** | 상황 설정 | "You are ChatGPT, 날짜는 2024년 10월 10일..." |
| **Instructions** | 수행할 작업 | "다음 텍스트를 요약해주세요" |
| **Input** | 실제 입력 데이터 | 요약할 문서 내용 |
| **Constraints** | 제약 조건 | "JSON 형식으로 출력", "안전 지침" |

### 3. In-Context Learning (ICL)

가중치 업데이트 없이 프롬프트만으로 학습

**Zero-Shot:** 예시 없이 지시만 제공

**Few-Shot:** 몇 가지 예시를 함께 제공

| 측면 | Zero-Shot | Few-Shot |
| --- | --- | --- |
| 일반적 성능 | 낮음 | 높음 |
| 토큰 사용량 | 적음 | 많음 |
| 준비 비용 | 낮음 | 높음 (예시 수집 필요) |

### 4. Chain of Thought (CoT)

답변 전에 **추론 과정**을 출력하도록 유도

```
일반 방식:
  Q: 테디베어는 몇 살인가요?
  A: 5살

CoT 방식:
  Q: 테디베어는 몇 살인가요?
  A: 테디베어는 2019년에 만들어졌고, 현재는 2024년입니다.
     따라서 2024 - 2019 = 5살입니다.
     답: 5살
```

**장점:** 성능 향상, 디버깅 용이, 해석 가능성

**단점:** 더 많은 토큰 생성 → 추론 시간/비용 증가

### 5. Self-Consistency

1. 같은 질문에 대해 **여러 번 샘플링**
2. 각 답변에서 최종 답 추출
3. **다수결 투표**로 최종 답 결정

병렬 처리 가능, 산술/수학 문제에서 효과적

---

## Part 5: 추론 최적화 (Inference Optimization)

### 최적화 기법 분류

| 분류 | 특징 | 목표 |
| --- | --- | --- |
| **Exact (정확)** | 동일한 결과 보장 | 중복 제거, 메모리 관리 |
| **Approximate (근사)** | 약간의 정확도 손실 허용 | 아키텍처 변경, 토큰 예측 가속 |

### Exact 기법

#### 1. KV Cache

이전 토큰의 Key, Value를 **캐싱**하여 재사용

```
토큰 생성 시퀀스: "A" → "cute" → "teddy" → "bear" → "is"

"is" 생성 시:
  - Query: "is"의 Q만 새로 계산
  - Key/Value: "A", "cute", "teddy", "bear"는 캐시에서 재사용
```

#### 2. Group Query Attention (GQA)

K와 V를 그룹으로 공유하여 KV Cache 절감

#### 3. PagedAttention (vLLM)

**문제:** KV Cache의 메모리 낭비 (Internal Fragmentation)

```
기존 방식:
  Request 1: [████████████░░░░░░░░] (실제 12, 예약 20)
  Request 2: [██████░░░░░░░░░░░░░░] (실제 6, 예약 20)

PagedAttention:
  고정 크기 블록(예: 16 토큰)으로 분할
  Request 1: [Block A][Block B][Block C]
  Request 2: [Block D][Block E]
  → 필요한 만큼만 블록 할당
```

### Approximate 기법

#### 1. Multi-Latent Attention (MLA)

**출처:** DeepSeek 논문

토큰 표현을 **저차원 공간으로 압축**, K와 V가 **압축 표현을 공유**

```
기존: Token → [K₁, K₂, ..., Kₕ], [V₁, V₂, ..., Vₕ] (H × 2개 저장)
MLA:  Token → [Compressed] → K, V 복원 (1개만 저장)
```

#### 2. Speculative Decoding

작은 모델로 초안 생성 → 큰 모델로 검증

```
Step 1: Draft Model (작은 LLM)로 K개 토큰 빠르게 생성
Step 2: Target Model (큰 LLM)에 모든 토큰을 한 번에 입력 → 확률 분포 계산
Step 3: 검증 (Acceptance/Rejection)
Step 4: 첫 거절 지점부터 재생성
```

**왜 효과적인가?**
* 추론은 **Memory Bound** (메모리가 병목)
* 한 번에 여러 토큰 처리해도 비용 비슷

#### 3. Multi-Token Prediction

Draft Model을 별도로 두지 않고 **같은 모델 내에 통합**

```
Decoder Output → Head 1 (Main) → 다음 토큰 1
              → Head 2 (Draft) → 다음 토큰 2
              → Head 3 (Draft) → 다음 토큰 3
```

---

## 핵심 요약

### 토큰 생성 전략

| 방법 | 특징 | 사용 사례 |
| --- | --- | --- |
| Greedy | 최고 확률 선택 | 거의 사용 안 함 |
| Beam Search | K개 경로 유지 | 기계 번역 |
| Sampling + Top-K/P | 확률적 샘플링 | 대부분의 LLM |

### Temperature

$$P(w\_i) = \frac{\exp(x\_i / T)}{\sum\_j \exp(x\_j / T)}$$

### 추론 최적화

| 기법 | 유형 | 핵심 |
| --- | --- | --- |
| KV Cache | Exact | K, V 재사용 |
| GQA | Exact | K, V 공유 |
| PagedAttention | Exact | 블록 단위 메모리 관리 |
| MLA | Approx | 저차원 압축 공유 |
| Speculative Decoding | Approx | 작은 모델로 초안 |
| Multi-Token Prediction | Approx | 여러 토큰 동시 예측 |

---

### 용어 정리

| 용어 | 의미 |
| --- | --- |
| LLM | Large Language Model |
| MoE | Mixture of Experts |
| FLOPs | Floating Point Operations |
| KV Cache | Key-Value Cache |
| GQA | Group Query Attention |
| MLA | Multi-Latent Attention |
| CoT | Chain of Thought |
| ICL | In-Context Learning |

---

### 추천 자료

1. **"Switch Transformers"** (2022) - MoE 스케일링
2. **"vLLM: PagedAttention"** (2023) - 효율적 추론
3. **"Chain-of-Thought Prompting"** (2022) - CoT 원본 논문
4. **"Self-Consistency"** (2023) - 다수결 추론
5. **"Speculative Decoding"** (2023) - 가속 추론
6. **"DeepSeek-V2"** - Multi-Latent Attention

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 3 정리*
