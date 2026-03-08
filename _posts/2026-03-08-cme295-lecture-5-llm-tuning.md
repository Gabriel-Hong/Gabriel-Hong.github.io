---
layout: post
title: "Stanford CME295: Lecture 5 - LLM Tuning (Preference Tuning)"
date: 2026-03-08 10:50:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, rlhf, dpo, ppo, preference-tuning, reward-model, alignment]
math: true
---

> **원본 강의**: [YouTube - CME295 Lecture 5](https://www.youtube.com/watch?v=PmW_TMQ3l0I&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=5)

---

## 강의 개요

이번 강의에서는 LLM을 인간의 선호도에 맞게 정렬(Alignment)하는 방법을 다룹니다. Pre-training과 SFT 이후의 세 번째 학습 단계인 Preference Tuning을 중심으로, RLHF, PPO, DPO 등의 핵심 기법들을 학습합니다.

**강의 목표:**

1. Preference Tuning의 필요성과 개념 이해
2. Preference Data 수집 방법 파악
3. RLHF (Reinforcement Learning from Human Feedback) 이해
4. PPO (Proximal Policy Optimization) 알고리즘 학습
5. DPO (Direct Preference Optimization) 기법 습득

---

## 지난 강의 복습

| 주제 | 핵심 내용 |
| --- | --- |
| **Pre-training** | 방대한 데이터로 언어/코드 이해 학습, 다음 토큰 예측 |
| **SFT (Supervised Fine-Tuning)** | 특정 태스크 수행을 위한 고품질 데이터 학습 |
| **LoRA** | Low-Rank 행렬로 효율적 파라미터 튜닝 |

---

## Part 1: Preference Tuning 개요

### 1. LLM 학습의 세 단계

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pre-training                                             │
│    └─ 목표: 언어/코드 이해                                    │
│    └─ 결과: 훌륭한 자동완성기 (도움이 되는 모델은 아님)          │
│              │                                              │
│              ▼                                              │
│ 2. Supervised Fine-Tuning (SFT)                             │
│    └─ 목표: 특정 태스크 수행법 학습                            │
│    └─ 결과: 지시를 따르는 모델                                │
│              │                                              │
│              ▼                                              │
│ 3. Preference Tuning ⭐ (이번 강의)                          │
│    └─ 목표: 인간 선호도에 맞춘 정렬                            │
│    └─ 결과: 더 친절하고, 안전하고, 유용한 모델                  │
└─────────────────────────────────────────────────────────────┘
```

### 2. Preference Tuning이 필요한 이유

**SFT 모델의 한계:**

```
User: "테디베어와 함께 할 수 있는 새로운 활동을 추천해줘"
SFT 모델 응답: "테디베어와 많은 시간을 보내지 않는 게 좋겠어요."
  → 사실적으로 틀린 건 아니지만, 친절하지 않음!
Preference Tuned 모델 응답: "물론이죠! 테디베어는 훌륭한 친구가 될 수 있어요.
  피크닉을 가거나, 함께 책을 읽거나..."
  → 같은 정보를 더 친절하고 도움이 되는 방식으로 전달
```

### 3. SFT vs Preference Tuning

| 구분 | SFT | Preference Tuning |
| --- | --- | --- |
| **데이터** | (Input, Output) 쌍 | (Input, Good Output, Bad Output) 쌍 |
| **신호** | "이것을 생성해라" | "이것이 더 좋다" |
| **난이도** | 고품질 Output 작성 (어려움) | 두 Output 비교 (쉬움) |
| **분포 민감도** | 매우 민감 | 덜 민감 |
| **네거티브 신호** | 없음 | 있음 |

**핵심 차이:**

- SFT: 모델이 **무엇을 생성해야 하는지** 가르침
- Preference Tuning: 모델이 **무엇을 생성하면 안 되는지** 가르침 (네거티브 신호)

---

## Part 2: Preference Data 수집

### 1. 선호도 데이터 유형

| 유형 | 설명 | 난이도 | 사용 빈도 |
| --- | --- | --- | --- |
| **Pointwise** | 각 응답에 절대 점수 부여 (예: 0.9, 0.2) | 어려움 | 낮음 |
| **Pairwise** | 두 응답 중 더 나은 것 선택 | 쉬움 | 높음 |
| **Listwise** | N개 응답 순위 매기기 | 중간 | 중간 |

### 2. Pairwise Preference Data 수집 방법

**방법 1: 모델 생성 후 평가**

```
1. 프롬프트 준비 (사용자 로그 또는 원하는 프롬프트 세트)
2. 모델에 동일 프롬프트를 양의 온도(temperature)로 2회 입력
3. 두 개의 다른 응답 생성
4. 평가자가 두 응답 비교
```

**방법 2: 불량 응답 재작성**

```
1. 로그에서 불량 응답 찾기
2. 좋은 응답으로 재작성
3. (원본, 재작성) 쌍 구성
```

### 3. 평가 방법

| 방법 | 설명 |
| --- | --- |
| **Human Ratings** | 사람이 직접 평가 (RLHF의 "H") |
| **LLM as a Judge** | 다른 LLM이 평가 (RLAIF) |
| **Rule-based** | BLEU, ROUGE 등 자동 메트릭 |

**평가 척도:**

- **Binary:** Better / Worse
- **Nuanced:** Much Better / Better / Slightly Better / Slightly Worse / Worse / Much Worse

**주의사항:**

- 평가 지침(Guidelines)이 명확해야 함
- 평가 차원 정의 필요: 유용성(Useful), 친절함(Friendly), 안전성(Safe) 등

---

## Part 3: RLHF (Reinforcement Learning from Human Feedback)

### 1. RL 기초 개념의 LLM 적용

| RL 개념 | LLM에서의 의미 |
| --- | --- |
| **Agent** | LLM |
| **State ($s_t$)** | 현재까지의 입력 (프롬프트 + 생성된 토큰들) |
| **Action ($a_t$)** | 다음 토큰 예측 |
| **Policy ($\pi_\theta$)** | LLM의 출력 확률 분포 |
| **Environment** | 어휘(Vocabulary) 집합 |
| **Reward** | Preference Data로부터 학습한 보상 |

```
┌─────────────────────────────────────────────────┐
│ LLM (Agent)                                     │
│           │                                     │
│ Input (State) 받음                               │
│           │                                     │
│           ▼                                     │
│ Policy (확률 분포)로 Action 선택                   │
│           │                                     │
│           ▼                                     │
│ 다음 토큰 생성 (Action)                           │
│           │                                     │
│           ▼                                     │
│ Reward 받음 → Policy 업데이트                     │
└─────────────────────────────────────────────────┘
```

### 2. RLHF의 두 단계

```
┌─────────────────────────────────────────────────────────────┐
│ RLHF 파이프라인                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Stage 1: Reward Model 학습                                  │
│   입력: (Prompt, Response)                                   │
│   출력: Score (얼마나 좋은지)                                  │
│   데이터: Preference Pairs (수만 개)                          │
│              │                                              │
│              ▼                                              │
│                                                             │
│ Stage 2: Policy 최적화 (RL)                                  │
│   입력: Prompt                                               │
│   출력: Aligned Response                                     │
│   목표: Reward 최대화 + Base 모델에서 멀어지지 않기              │
│   데이터: 10만+ 개의 프롬프트                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 4: Reward Model 학습 (Stage 1)

### 1. Bradley-Terry Formulation

**핵심 공식:**

$$P(y_i \succ y_j | x) = \frac{e^{R(x, y_i)}}{e^{R(x, y_i)} + e^{R(x, y_j)}} = \sigma(R(x, y_i) - R(x, y_j))$$

![Bradley-Terry 공식](/assets/img/cme295-lecture-5/image-20260115-011409.png)

여기서:

- $y_i \succ y_j$: $y_i$가 $y_j$보다 좋음
- $R(x, y)$: 프롬프트 $x$와 응답 $y$에 대한 보상 점수
- $\sigma$: Sigmoid 함수

**직관적 이해:**

- $R_i - R_j$가 크면 → $\sigma$ 값이 1에 가까움 → $i$가 더 좋을 확률 높음
- $R_i - R_j$가 작으면 → $\sigma$ 값이 0에 가까움 → $j$가 더 좋을 확률 높음

<details>
<summary>Bradley-Terry Formulation 상세</summary>

![Bradley-Terry 상세](/assets/img/cme295-lecture-5/image-20260115-072616.png)

</details>

<details>
<summary>Sigmoid 함수 상세 및 사용하는 이유</summary>

![Sigmoid 상세](/assets/img/cme295-lecture-5/image-20260115-073032.png)

</details>

### 2. Reward Model 손실 함수

**유도 과정:**

```
1. MLE (Maximum Likelihood Estimation) 적용
   → max ∏ P(y_w ≻ y_l | x)
2. Log 취하기 (수치 안정성)
   → max Σ log σ(R(x, y_w) - R(x, y_l))
3. 최소화 문제로 변환
   → min -Σ log σ(R(x, y_w) - R(x, y_l))
```

**최종 손실 함수:**

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma(R(x, y_w) - R(x, y_l))\right]$$

![Reward Model 손실](/assets/img/cme295-lecture-5/image-20260115-011522.png)

<details>
<summary>Reward Model 손실 함수 상세</summary>

![RM Loss 상세 1](/assets/img/cme295-lecture-5/image-20260115-074107.png)

![RM Loss 상세 2](/assets/img/cme295-lecture-5/image-20260115-074129.png)

</details>

<details>
<summary>최종 손실 함수 쉽게 이해하기</summary>

![Loss 이해 1](/assets/img/cme295-lecture-5/image-20260115-074624.png)

![Loss 이해 2](/assets/img/cme295-lecture-5/image-20260115-074640.png)

</details>

### 3. Reward Model의 특징

| 특성 | 설명 |
| --- | --- |
| **학습 방식** | Pairwise (쌍으로 비교하며 학습) |
| **추론 방식** | Pointwise (하나의 (prompt, response)만 입력해도 점수 출력) |
| **출력** | 연속적인 점수 (예: 0.8, -2.0 등) |
| **모델 구조** | Decoder-only LLM + Classification Head 또는 BERT + CLS |

```
┌─────────────────────────────────────────────────┐
│ Reward Model 학습 vs 추론                        │
├─────────────────────────────────────────────────┤
│                                                 │
│ [학습 시] - Pairwise                             │
│ (x, y_w) → RM → R_w ─┐                         │
│                       ├→ Loss 계산               │
│ (x, y_l) → RM → R_l ─┘                         │
│                                                 │
│ [추론 시] - Pointwise                            │
│ (x, y) → RM → Score (단일 점수)                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Part 5: Policy 최적화 (Stage 2) - PPO

### 1. 목표: 두 가지 균형

```
최대화하고 싶은 것:
  1. Reward (보상)
  2. Base 모델과의 유사성 유지

왜 Base 모델에서 멀어지면 안 되는가?
  1. Catastrophic Forgetting: Pre-training에서 배운 지식 손실
  2. Reward Hacking: 불완전한 Reward Model 악용
  3. Training Instability: 학습 불안정
```

### 2. Reward Hacking 예시

```
비유: 강의 품질 최적화
  목표: 강의를 최대한 유익하게 만들기
  보상: 강의 끝 박수 소리 크기
  문제:
    - 박수 소리를 최대화하려고 농담만 하게 됨
    - 박수는 크지만 → 강의는 유익하지 않음
    - 보상(박수)은 목표(유익함)의 불완전한 대리 지표
  → 이것이 Reward Hacking!
```

### 3. PPO 손실 함수 개요

$$\mathcal{L}_{PPO} = \mathbb{E}\left[R(x, y)\right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

![PPO 손실 함수](/assets/img/cme295-lecture-5/image-20260115-011556.png)

| 항 | 의미 | 역할 |
| --- | --- | --- |
| $R(x, y)$ | Reward | 높이고 싶음 |
| $D_{KL}$ | KL Divergence | 낮추고 싶음 (Base 모델과 가깝게) |
| $\beta$ | 균형 계수 | 두 목표 사이 균형 조절 (보통 ~0.1) |

### 4. KL Divergence

$$D_{KL}(P \| Q) = \sum_i P_i \log \frac{P_i}{Q_i}$$

![KL Divergence](/assets/img/cme295-lecture-5/image-20260115-011608.png)

**특성:**

- 항상 ≥ 0 (Jensen's Inequality)
- P = Q일 때만 = 0
- 두 확률 분포의 "거리" 측정 (엄밀한 거리는 아님)

<details>
<summary>KL Divergence(Kullback-Leibler Divergence) 상세</summary>

![KL Divergence 상세 1](/assets/img/cme295-lecture-5/image-20260115-095022.png)

![KL Divergence 상세 2](/assets/img/cme295-lecture-5/image-20260115-095039.png)

![KL Divergence 상세 3](/assets/img/cme295-lecture-5/image-20260115-095058.png)

</details>

### 5. Advantage 함수

**정의:** 현재 행동이 평균 대비 얼마나 좋은지

$$A(s, a) = Q(s, a) - V(s)$$

![Advantage 함수](/assets/img/cme295-lecture-5/image-20260115-011626.png)

- $Q(s, a)$: State-Action Value (이 상태에서 이 행동의 기대 보상)
- $V(s)$: Value Function (이 상태의 평균 기대 보상)

**왜 Advantage를 사용하는가?**

- Reward 직접 사용 시 분산(Variance)이 큼
- Baseline(평균) 대비 상대적 값 사용 → 분산 감소 → 학습 안정화

### 6. Value Function

**정의:** 현재 상태에서 Policy를 따랐을 때의 기대 보상

```
┌─────────────────────────────────────────────────┐
│ Value Function 개념                              │
├─────────────────────────────────────────────────┤
│                                                 │
│ Reward Model:                                   │
│   입력: Prompt + 완전한 Response                  │
│   출력: 전체 응답의 점수                           │
│                                                 │
│ Value Function:                                  │
│   입력: Prompt + 부분 Response (토큰 레벨)         │
│   출력: 계속 생성했을 때의 예상 Reward              │
│                                                 │
└─────────────────────────────────────────────────┘
```

**학습:** Policy와 함께 Joint Training

<details>
<summary>Value Function 상세</summary>

![Value Function 상세 1](/assets/img/cme295-lecture-5/image-20260115-101701.png)

![Value Function 상세 2](/assets/img/cme295-lecture-5/image-20260115-101718.png)

![Value Function 상세 3](/assets/img/cme295-lecture-5/image-20260115-101733.png)

</details>

---

## Part 6: PPO 변형들

### 1. PPO-Clip

**목표:** 한 번의 업데이트가 너무 크지 않도록 제한

$$\mathcal{L}^{CLIP} = \min\left(r(\theta) \cdot A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A\right)$$

![PPO-Clip](/assets/img/cme295-lecture-5/image-20260115-011641.png)

여기서:

- $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ : Policy 비율 (현재 / 이전 iteration)
- $\epsilon$ : 클리핑 범위 (보통 0.1~0.2)
- $A$ : Advantage

**주의:** $r(\theta)$는 Reward가 아니라 **확률 비율**!

**직관적 이해:**

![PPO-Clip 직관](/assets/img/cme295-lecture-5/image-20260115-011701.png)

```
A > 0 (좋은 행동일 때):
  → r을 높이고 싶지만 1+ε까지만!

A < 0 (나쁜 행동일 때):
  → r을 낮추고 싶지만 1-ε까지만!
```

<details>
<summary>PPO-Clip 상세</summary>

![PPO-Clip 상세 1](/assets/img/cme295-lecture-5/image-20260115-103223.png)

![PPO-Clip 상세 2](/assets/img/cme295-lecture-5/image-20260115-103242.png)

![PPO-Clip 상세 3](/assets/img/cme295-lecture-5/image-20260115-103300.png)

![PPO-Clip 상세 4](/assets/img/cme295-lecture-5/image-20260115-103318.png)

</details>

### 2. PPO-KL Penalty

$$\mathcal{L}^{KL} = r(\theta) \cdot A - \beta \cdot D_{KL}(\pi_{\theta_{old}} \| \pi_\theta)$$

![PPO-KL Penalty](/assets/img/cme295-lecture-5/image-20260115-011714.png)

**현대적 사용:**

- `old` 대신 `ref` (SFT 모델) 사용
- Clip과 KL Penalty를 함께 사용하기도 함

### 3. PPO에서 필요한 모델들

```
┌─────────────────────────────────────────────────┐
│ PPO에 필요한 4개의 모델                           │
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. Policy (π_θ) - 학습 대상                      │
│ 2. Value Function - Advantage 계산용             │
│ 3. Reward Model - 보상 계산용 (Frozen)            │
│ 4. Reference Model - KL 계산용 (Frozen)          │
│                                                 │
│ → 메모리와 계산 비용이 매우 높음!                   │
└─────────────────────────────────────────────────┘
```

---

## Part 7: RL 기반 접근법의 도전과제

| 도전과제 | 설명 |
| --- | --- |
| **2단계 프로세스** | Reward Model → Policy 순서, 의존성 문제 |
| **많은 하이퍼파라미터** | β, ε, GAE 파라미터 등 튜닝 필요 |
| **학습 불안정성** | 제약을 걸어도 불안정할 수 있음 |
| **모니터링 어려움** | Average Reward로 모니터링하지만 불완전 |
| **탐색 필요** | 다양한 Completion 생성을 위한 Exploration 필요 |
| **RL 전문성 필요** | RL에 익숙하지 않으면 어려움 |

### On-Policy vs Off-Policy

| 유형 | 설명 | 예시 |
| --- | --- | --- |
| **On-Policy** | 현재 모델이 생성한 데이터로 학습 | PPO |
| **Off-Policy** | 다른 모델이 생성한 데이터로 학습 | SFT |

**PPO는 On-Policy:**

- 매 iteration마다 현재 Policy로 생성
- 생성된 결과로 Policy 업데이트
- SFT보다 신호가 Sparse함

---

## Part 8: Best-of-N (BoN)

### 1. 개념

**아이디어:** RL 학습 없이 Reward Model만 활용

```
┌─────────────────────────────────────────────────┐
│ Best-of-N 방법                                   │
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. 프롬프트 입력                                  │
│                                                 │
│ 2. SFT 모델로 N개의 응답 생성                     │
│    (높은 Temperature로 다양성 확보)                │
│                                                 │
│ 3. 각 응답을 Reward Model로 점수화               │
│    Response 1: 0.8                               │
│    Response 2: -2.0                              │
│    Response 3: 0.3                               │
│                                                 │
│ 4. 가장 높은 점수의 응답 반환                      │
│    → Response 1 선택!                            │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 2. 장단점

| 장점 | 단점 |
| --- | --- |
| RL 학습 불필요 | 추론 비용 N배 증가 |
| 구현 간단 | 대규모 서비스에 부적합 |
| 학습 불안정성 없음 | 모델 자체 개선 없음 |

**적합한 상황:**

- 추론 트래픽이 적을 때
- 빠른 프로토타이핑이 필요할 때

---

## Part 9: DPO (Direct Preference Optimization)

### 1. 동기

**RLHF의 문제점:**

1. 4개의 모델 필요 (Policy, Value, Reward, Reference)
2. 2단계 학습 (Reward → Policy)
3. 학습 불안정
4. 많은 하이퍼파라미터

**DPO의 아이디어:** RL 없이 직접 Preference 최적화!

### 2. DPO 유도 과정

![DPO 유도](/assets/img/cme295-lecture-5/image-20260115-011818.png)

**Step 1: PPO 목표 함수**

$$\max_\pi \mathbb{E}[R(x, y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})$$

**Step 2: 최적 Policy 도출**

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{R(x,y)}{\beta}\right)$$

**Step 3: Reward를 Policy로 표현**

$$R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Step 4: Bradley-Terry에 대입**

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### 3. DPO 손실 함수

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

![DPO 손실 함수](/assets/img/cme295-lecture-5/image-20260115-011827.png)

### 4. DPO의 핵심 인사이트

**"Your Language Model is Secretly a Reward Model"**

- Reward를 명시적으로 학습하지 않음
- Policy 자체가 암묵적으로 Reward를 인코딩
- Reward Model이 사라지고 Policy만 남음!

### 5. DPO vs RLHF 비교

| 측면 | RLHF (PPO) | DPO |
| --- | --- | --- |
| **필요 모델 수** | 4개 | 2개 (Policy, Reference) |
| **학습 단계** | 2단계 | 1단계 |
| **Reward Model** | 필요 | 불필요 |
| **학습 방식** | On-Policy (RL) | Supervised |
| **구현 복잡도** | 높음 | 낮음 |
| **성능** | 약간 더 좋음 | 거의 비슷 |
| **하이퍼파라미터** | 많음 | 적음 (β 정도) |

![RLHF vs DPO](/assets/img/cme295-lecture-5/image-20260115-011844.png)

```
┌─────────────────────────────────────────────────┐
│ RLHF vs DPO 비교                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│ RLHF:                                           │
│ ┌──────────┐    ┌──────────┐                    │
│ │ Reward   │ → │ Policy   │                     │
│ │ Training │    │ Training │                     │
│ └──────────┘    └──────────┘                    │
│  (Stage 1)       (Stage 2)                      │
│  + Value Fn + Reference Model                   │
│                                                 │
│ DPO:                                            │
│ ┌──────────────────────────┐                    │
│ │ Direct Policy Training   │                    │
│ │ (Preference Pairs 직접)   │                   │
│ └──────────────────────────┘                    │
│  + Reference Model only                         │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 6. DPO의 한계

| 한계 | 설명 |
| --- | --- |
| **Distribution Shift** | Preference Data가 모델 분포와 다를 수 있음 |
| **Off-Policy** | 현재 모델이 생성한 데이터가 아님 |
| **성능 차이** | PPO가 약간 더 좋은 성능을 보이는 경우 있음 |

**해결책:**

- Preference Data를 SFT로 먼저 학습
- 직접 모델로 Preference Data 생성 후 평가

<details>
<summary>DPO (Direct Preference Optimization) 상세</summary>

![DPO 상세 1](/assets/img/cme295-lecture-5/image-20260115-104721.png)

![DPO 상세 2](/assets/img/cme295-lecture-5/image-20260115-104739.png)

![DPO 상세 3](/assets/img/cme295-lecture-5/image-20260115-104804.png)

![DPO 상세 4](/assets/img/cme295-lecture-5/image-20260115-104821.png)

![DPO 상세 5](/assets/img/cme295-lecture-5/image-20260115-104835.png)

![DPO 상세 6](/assets/img/cme295-lecture-5/image-20260115-104850.png)

![DPO 상세 7](/assets/img/cme295-lecture-5/image-20260115-104904.png)

</details>

---

## Part 10: 테디베어 예시로 보는 전체 과정

### SFT 후:

```
Q: "테디베어를 세탁기에 넣어도 될까요?"
A: "안 돼요. 손상될 수 있어요. 손세탁하세요."
  → 사실적으로 맞지만, 딱딱함
```

### Preference Tuning 후:

```
Q: "테디베어를 세탁기에 넣어도 될까요?"
A: "테디베어가 다칠 수 있어요! 부드러운 손세탁이 더 안전해요."
  → 같은 정보를 더 친절하고 공감적으로 전달
```

---

## 핵심 요약

### Preference Tuning

- LLM 학습의 세 번째 단계
- 인간 선호도에 맞게 모델 정렬
- 네거티브 신호 주입 가능

### RLHF

- **Stage 1:** Reward Model 학습 (Bradley-Terry)
- **Stage 2:** PPO로 Policy 최적화
- 4개의 모델 필요, 복잡하지만 성능 좋음

### PPO

- Reward 최대화 + Base 모델 유지
- Clip 또는 KL Penalty로 업데이트 제한
- Advantage 함수로 분산 감소

### DPO

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

![DPO 요약](/assets/img/cme295-lecture-5/image-20260115-011859.png)

- Reward Model 없이 직접 최적화
- 2개의 모델만 필요, 구현 간단
- "Language Model is Secretly a Reward Model"

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| RLHF | Reinforcement Learning from Human Feedback |
| RLAIF | Reinforcement Learning from AI Feedback |
| PPO | Proximal Policy Optimization |
| DPO | Direct Preference Optimization |
| Bradley-Terry | Pairwise 비교 확률 모델링 공식 |
| KL Divergence | 두 확률 분포 간 차이 측정 |
| Advantage | 행동의 상대적 가치 (Q - V) |
| Value Function | 상태의 기대 가치 추정 |
| Reward Hacking | 불완전한 Reward Model 악용 현상 |
| On-Policy | 현재 Policy로 생성한 데이터로 학습 |
| Off-Policy | 다른 Policy로 생성한 데이터로 학습 |
| BoN | Best-of-N (N개 중 최고 선택) |

---

## 핵심 공식

![핵심 공식 요약](/assets/img/cme295-lecture-5/image-20260115-011909.png)

**Bradley-Terry:**

$$P(y_i \succ y_j) = \sigma(R_i - R_j)$$

**Reward Model Loss:**

$$\mathcal{L}_{RM} = -\mathbb{E}\left[\log \sigma(R(x, y_w) - R(x, y_l))\right]$$

**PPO Objective:**

$$\mathcal{L}_{PPO} = \mathbb{E}[R(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

**PPO-Clip:**

$$\mathcal{L}^{CLIP} = \min(r(\theta) \cdot A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A)$$

**DPO Loss:**

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

---

## 추천 자료

1. **"Training Language Models to Follow Instructions with Human Feedback"** (InstructGPT, 2022) - RLHF
2. **"Proximal Policy Optimization Algorithms"** (2017) - PPO
3. **"Direct Preference Optimization"** (2023) - DPO
4. **"High-Dimensional Continuous Control Using Generalized Advantage Estimation"** (2016) - GAE
5. **"Scaling Laws for Reward Model Overoptimization"** (2022) - Reward Hacking
6. **"A General Theoretical Paradigm to Understand Learning from Human Preferences"** - PPO vs DPO 비교
7. **RewardBench** - Reward Model 벤치마크

---

## 다음 강의 예고

**Lecture 6: Reasoning Models**

- GRPO (Group Relative Policy Optimization)
- DeepSeek-Math
- 추론 능력 강화를 위한 RL 기법

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 5 정리*
