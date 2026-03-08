---
layout: post
title: "Stanford CME295: Lecture 6 - LLM Reasoning"
date: 2026-03-08 11:00:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, reasoning, grpo, deepseek-r1, chain-of-thought, rl, pass-at-k]
math: true
---

> **원본 강의**: [YouTube - CME295 Lecture 6](https://www.youtube.com/watch?v=k5Fh-UgTuCo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=6)

---

## 강의 개요

이번 강의에서는 2024년 말부터 급부상한 Reasoning Model에 대해 다룹니다. Chain of Thought를 대규모로 적용하여 복잡한 문제를 해결하는 방법과, GRPO 알고리즘을 사용한 학습 방법, 그리고 DeepSeek R1의 구체적인 학습 파이프라인을 학습합니다.

**강의 목표:**

1. Reasoning Model이 무엇인지 이해
2. Reasoning Model이 어떻게 학습되는지 파악
3. Pass@K 메트릭 이해
4. GRPO 알고리즘과 PPO의 차이점 파악
5. DeepSeek R1 학습 파이프라인 이해

---

## 지난 강의 복습

| 주제 | 핵심 내용 |
| --- | --- |
| **RLHF** | Reward Model 학습 → Policy 최적화 |
| **PPO** | Advantage 최대화 + Base 모델 유지 |
| **PPO-Clip** | 업데이트 크기 제한 (클리핑) |
| **PPO-KL** | KL Divergence로 Base 모델과 거리 제한 |

---

## Part 1: Vanilla LLM의 강점과 약점

### 1. Vanilla LLM의 강점

| 강점 | 설명 |
| --- | --- |
| **언어/코드 이해** | 텍스트 구조, 코드 패턴 학습 |
| **코드 생성/디버깅** | 에러 찾기, 코드 작성 |
| **창작 능력** | 에세이, 시 등 생성 |

### 2. Vanilla LLM의 약점

| 약점 | 설명 |
| --- | --- |
| **제한된 추론 능력** | 복잡한 수학/논리 문제 해결 어려움 |
| **지식 Cutoff** | Pre-training 이후 정보 모름 |
| **행동 불가** | 주문, 예약 등 실제 행동 못함 |
| **평가 어려움** | 자유 형식 텍스트 평가 기준 모호 |

**이번 강의 초점:** 제한된 추론 능력 해결!

<details>
<summary>Vanilla LLM 상세</summary>

![Vanilla LLM](/assets/img/cme295-lecture-6/image-20260116-033723.png)

</details>

---

## Part 2: Reasoning이란?

### 1. Reasoning의 정의

**정의:** 문제를 해결하는 능력, 특히 다단계 추론(Multi-step reasoning)이 필요한 경우

**Non-Reasoning vs Reasoning 질문:**

| 유형 | 예시 | 특성 |
| --- | --- | --- |
| **Non-Reasoning** | "Stanford Transformers 강의 코드는?" | 단순 지식 조회 |
| **Reasoning** | "2020년생 곰이 2025년에 몇 살?" | 다단계 계산 필요 |

### 2. Reasoning Model의 핵심 아이디어

**Chain of Thought를 대규모로 적용!**

```
┌─────────────────────────────────────────────────────────────┐
│ Vanilla LLM vs Reasoning Model                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Vanilla LLM:                                                │
│   Question → [LLM] → Answer                                │
│                                                             │
│ Reasoning Model:                                            │
│   Question → [LLM] → <think>Reasoning Chain</think>        │
│                     → <answer>Final Answer</answer>         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. 왜 Chain of Thought가 도움이 되는가?

**이유 1: 문제 분해**

- 복잡한 문제를 작은 문제들로 분해
- 각 작은 문제는 학습 데이터에서 본 패턴과 유사
- 학생이 시험에서 배운 내용을 연결하는 것과 유사

**이유 2: 더 많은 Compute**

- 토큰을 더 많이 생성 = Forward Pass를 더 많이 수행
- 더 많은 연산 = 더 깊은 "생각"
- **Test-time Compute** 개념

### 4. Reasoning Model 타임라인

| 시기 | 모델 | 특징 |
| --- | --- | --- |
| 2024.09 | OpenAI o1 Preview | 최초 상용 Reasoning Model |
| 2024.12 | Google Gemini 2.0 Flash Thinking | Google의 Reasoning Model |
| **2025.01** | **DeepSeek R1** | 오픈소스, OpenAI 수준 성능 |
| 2025+ | Claude, Grok 등 | 각 AI Lab의 Reasoning Model |

---

## Part 3: Reasoning 평가 벤치마크

### 1. 코딩 벤치마크

| 벤치마크 | 설명 |
| --- | --- |
| **HumanEval** | 164개 수작업 코딩 문제 |
| **CodeForces** | 경쟁 프로그래밍 문제 |
| **SWE-Bench** | GitHub 이슈 기반 실제 문제 |

**검증 방법:** Test Case 통과 여부

```
Problem → Model → Code → Run Tests → Pass/Fail
```

### 2. 수학 벤치마크

| 벤치마크 | 설명 |
| --- | --- |
| **AIME** | 미국 수학 올림피아드 예선 |
| **GSM8K** | 초등학교 수준 수학 문제 |
| **MATH** | 고난도 수학 문제 |

**검증 방법:** 정답 비교

```
Problem → Model → Answer → Parse → Compare with Ground Truth
```

### 3. Pass@K 메트릭

**정의:** K번 시도 중 최소 1번 성공할 확률

**왜 필요한가?**

- 코딩/수학 문제는 정답 검증 가능
- 여러 번 시도해서 하나라도 맞으면 됨
- Best-of-N과 유사한 개념

**공식 유도:**

$$\text{Pass@K} = 1 - P(\text{K번 모두 틀림}) = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

![Pass@K 공식](/assets/img/cme295-lecture-6/image-20260115-013627.png)

여기서:

- $n$: 총 샘플 수
- $c$: 정답 샘플 수
- $k$: 선택할 샘플 수

**특수 케이스 - Pass@1:**

$$\text{Pass@1} = \frac{c}{n}$$

(단순히 정답 비율)

### 4. Temperature와 Pass@K

```
┌─────────────────────────────────────────────────────────────┐
│ Temperature와 Pass@K의 관계                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Temperature 너무 낮음 (T≈0):                                 │
│   - 답변 품질 높음                                           │
│   - 다양성 없음 → K 늘려도 Pass@K 변화 없음                   │
│                                                             │
│ Temperature 너무 높음 (T≈1.2):                               │
│   - 다양성 높음                                              │
│   - 답변 품질 낮음 → Pass@K 오히려 감소                       │
│                                                             │
│ 적절한 Temperature (T≈0.4~0.8):                              │
│   - 품질과 다양성 균형                                        │
│   - K 늘릴수록 Pass@K 증가                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. 기타 메트릭

| 메트릭 | 설명 |
| --- | --- |
| **Consensus@K** | K개 답변 중 가장 많이 나온 답 선택 |
| **Accuracy** | 단순 정확도 |
| **Exact Match** | 정확히 일치 여부 |

<details>
<summary>Pass@K 상세</summary>

![Pass@K 상세 1](/assets/img/cme295-lecture-6/image-20260116-034852.png)

![Pass@K 상세 2](/assets/img/cme295-lecture-6/image-20260116-034907.png)

![Pass@K 상세 3](/assets/img/cme295-lecture-6/image-20260116-034925.png)

</details>

---

## Part 4: Reasoning Model 학습 방법

### 1. 왜 SFT만으로는 부족한가?

| 문제 | 설명 |
| --- | --- |
| **고품질 데이터 부족** | Reasoning Chain 직접 작성 매우 어려움 |
| **모델과 인간의 추론 방식 차이** | 인간 작성 Chain이 최적이 아닐 수 있음 |
| **검증 가능한 Reward 존재** | 코드/수학은 정답 여부 확인 가능! |

### 2. RL 기반 접근법

**핵심 아이디어:** Verifiable Reward를 활용한 RL!

```
┌─────────────────────────────────────────────────────────────┐
│ Reasoning을 위한 RL Reward                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Reward = R_format + R_correctness                           │
│                                                             │
│ R_format:                                                   │
│   - <think>...</think> 토큰 존재 여부                        │
│   - 형식 준수 여부                                           │
│                                                             │
│ R_correctness:                                              │
│   - 코드: Test Case 통과 여부                                │
│   - 수학: 정답 일치 여부                                     │
│                                                             │
│ → Reward Model 학습 불필요! (검증 가능)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. RL 학습 효과

DeepSeek R1-Zero 실험 결과:

- RL Step 증가 → AIME 벤치마크 성능 증가
- Reasoning Chain 자동 생성 학습
- SFT 없이도 Reasoning 능력 획득!

<details>
<summary>Reasoning Model 학습 상세</summary>

![학습 상세 1](/assets/img/cme295-lecture-6/image-20260121-044048.png)

![학습 상세 2](/assets/img/cme295-lecture-6/image-20260121-044103.png)

![학습 상세 3](/assets/img/cme295-lecture-6/image-20260121-044126.png)

![학습 상세 4](/assets/img/cme295-lecture-6/image-20260121-044144.png)

![학습 상세 5](/assets/img/cme295-lecture-6/image-20260121-044156.png)

</details>

---

## Part 5: GRPO (Group Relative Policy Optimization)

### 1. PPO의 한계

**PPO는 Value Function 필요:**

- Advantage = Reward - Value
- Value Function을 Policy와 함께 학습해야 함
- 추가 모델, 추가 복잡성

### 2. GRPO의 핵심 아이디어

**Value Function 대신 Group 비교!**

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G)}$$

![GRPO Advantage](/assets/img/cme295-lecture-6/image-20260115-013710.png)

<details>
<summary>Value Function 상세</summary>

![Value Function 상세 1](/assets/img/cme295-lecture-6/image-20260122-002344.png)

![Value Function 상세 2](/assets/img/cme295-lecture-6/image-20260122-002411.png)

![Value Function 상세 3](/assets/img/cme295-lecture-6/image-20260122-002513.png)

![Value Function 상세 4](/assets/img/cme295-lecture-6/image-20260122-002531.png)

![Value Function 상세 5](/assets/img/cme295-lecture-6/image-20260122-002551.png)

</details>

```
┌─────────────────────────────────────────────────────────────┐
│ GRPO Advantage 계산                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. 하나의 Prompt에 대해 G개의 Completion 생성                 │
│                                                             │
│ 2. 각 Completion의 Reward 계산                               │
│    R₁, R₂, ..., R_G                                         │
│                                                             │
│ 3. Advantage = (R_i - mean) / std                           │
│    → Group 내 상대적 위치로 Advantage 결정                    │
│                                                             │
│ 4. Advantage로 Policy 업데이트                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. GRPO vs PPO 비교

| 측면 | PPO | GRPO |
| --- | --- | --- |
| **Advantage 계산** | Reward + Value Function (GAE) | Group 평균 대비 |
| **Value Function** | 필요 (학습해야 함) | 불필요 |
| **샘플링** | 1개 Completion | G개 Completion |
| **주 사용처** | Preference Tuning | Reasoning Training |
| **복잡도** | 높음 | 낮음 |

### 4. GRPO 알고리즘 흐름

```
┌─────────────────────────────────────────────────────────────┐
│ GRPO 파이프라인                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Query (Prompt)                                              │
│       │                                                     │
│       ▼                                                     │
│ Policy Model (π_θ) ──┬──> Completion 1 ──> R₁              │
│                      ├──> Completion 2 ──> R₂              │
│                      ├──> ...                               │
│                      └──> Completion G ──> R_G              │
│                            │                                │
│                            ▼                                │
│                  Advantage 계산 (Group 기반)                  │
│                            │                                │
│                            ▼                                │
│                  Policy 업데이트                              │
│                            │                                │
│                            │ (KL Divergence)                │
│                            ▼                                │
│                  Reference Model (π_ref)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. 필요한 모델 비교

| 모델 | PPO | GRPO |
| --- | --- | --- |
| **Policy Model** | 학습 | 학습 |
| **Value Function** | 학습 | 불필요 |
| **Reward Model** | 필요 (Preference) | 불필요 (Verifiable) |
| **Reference Model** | Frozen | Frozen |
| **총 학습 모델** | 2개 | 1개 |

**Reasoning의 경우:** Reward Model도 불필요 (검증 가능한 Reward 사용)

### 6. GRPO 손실 함수

$$J_{GRPO} = \mathbb{E}\left[\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \frac{1}{|o_i|}\left(\min\left(r_t A_i, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_i\right) - \beta D_{KL}\right)\right]$$

![GRPO 손실 함수](/assets/img/cme295-lecture-6/image-20260115-013725.png)

**PPO와의 공통점:**

- Ratio $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 사용
- Clipping 메커니즘으로 큰 업데이트 방지

**PPO와의 차이점:**

- KL Divergence가 명시적으로 손실에 포함
- Advantage가 Group 기반으로 계산

<details>
<summary>GRPO(Group Relative Policy Optimization) 상세</summary>

![GRPO 상세 1](/assets/img/cme295-lecture-6/image-20260122-011554.png)

![GRPO 상세 2](/assets/img/cme295-lecture-6/image-20260122-011612.png)

![GRPO 상세 3](/assets/img/cme295-lecture-6/image-20260122-011625.png)

![GRPO 상세 4](/assets/img/cme295-lecture-6/image-20260122-011641.png)

![GRPO 상세 5](/assets/img/cme295-lecture-6/image-20260122-011655.png)

![GRPO 상세 6](/assets/img/cme295-lecture-6/image-20260122-011708.png)

![GRPO 상세 7](/assets/img/cme295-lecture-6/image-20260122-011722.png)

![GRPO 상세 8](/assets/img/cme295-lecture-6/image-20260122-011741.png)

</details>

<details>
<summary>Advantage와 Loss의 관계 상세</summary>

![Advantage-Loss 1](/assets/img/cme295-lecture-6/image-20260122-011813.png)

![Advantage-Loss 2](/assets/img/cme295-lecture-6/image-20260122-011827.png)

![Advantage-Loss 3](/assets/img/cme295-lecture-6/image-20260122-011844.png)

![Advantage-Loss 4](/assets/img/cme295-lecture-6/image-20260122-011856.png)

![Advantage-Loss 5](/assets/img/cme295-lecture-6/image-20260122-011910.png)

</details>

---

## Part 6: Output Length 문제와 해결책

### 1. 문제: Output Length 계속 증가

![Output Length 문제](/assets/img/cme295-lecture-6/image-20260115-013738.png)

```
┌─────────────────────────────────────────────────────────────┐
│ RL Training 중 Output Length 변화                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Performance는 수렴하는데 Length는 계속 증가!                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 원인 분석

GRPO 손실 함수의 $\frac{1}{\|o_i\|}$ 항:

![Length 원인](/assets/img/cme295-lecture-6/image-20260115-013759.png)

```
Token 기여도 = (1 / Output Length) × Token별 항
  → 짧은 Output: 각 토큰 기여도 높음
  → 긴 Output: 각 토큰 기여도 낮음
```

**문제가 되는 인센티브:**

- Advantage < 0 (나쁜 답변)일 때
- 짧은 나쁜 답변 → 토큰들이 강하게 Downweight
- 긴 나쁜 답변 → 토큰들이 약하게 Downweight
- **모델이 "긴 나쁜 답변"을 선호하게 됨!**

### 3. 해결책

| 방법 | 설명 | 논문 |
| --- | --- | --- |
| **Token Level 균등화** | 모든 토큰에 동일한 가중치 | DAPO (2025) |
| **Length Factor 제거** | $\frac{1}{\|o_i\|}$ 항 완전 제거 | Dr. GRPO (2025) |

### 4. 기타 GRPO 개선 사항

| 개선 | 설명 |
| --- | --- |
| **Std 제거** | 어려운 문제에서 Std가 작아 Advantage 불안정 |
| **비대칭 Epsilon** | 낮은 확률 토큰에 더 큰 성장 여지 제공 |

---

## Part 7: DeepSeek R1

### 1. R1-Zero: 개념 증명

**목표:** SFT 없이 RL만으로 Reasoning 학습 가능한가?

```
┌─────────────────────────────────────────────────────────────┐
│ R1-Zero 파이프라인                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ DeepSeek V3 Base (Pre-trained)                              │
│       │                                                     │
│       │ (SFT 없음!)                                         │
│       ▼                                                     │
│ RL with Verifiable Rewards                                  │
│   - Format Reward: <think>...</think> 존재 여부              │
│   - Correctness Reward: 정답 여부                            │
│       │                                                     │
│       ▼                                                     │
│ R1-Zero (Reasoning 가능!)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**R1-Zero 프롬프트 템플릿:**

```
You are a helpful assistant. When thinking, put your thoughts
in <think>...</think> blocks. When ready to answer, put your
final answer in <answer>...</answer> blocks.
User: {problem}
Assistant:
```

**결과:**

- AIME 등 수학 벤치마크에서 성능 지속 향상
- SFT 없이도 Reasoning Chain 자동 학습!

**R1-Zero의 문제점:**

- 언어 혼합 (Language Mixing)
- 가독성 문제 (Poor Readability)
- 형식 불일치

### 2. R1: 완전한 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│ R1 학습 파이프라인                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Stage 1: Cold Start SFT                                     │
│   - R1-Zero의 출력을 인간이 재작성                            │
│   - 형식, 언어 일관성 교정                                    │
│   - 소규모 데이터 (수천 개 수준)                               │
│              │                                              │
│              ▼                                              │
│                                                             │
│ Stage 2: RL Training                                        │
│   Rewards:                                                  │
│   - Correctness Reward (정답 여부)                           │
│   - Format Reward (형식 준수)                                │
│   - Language Consistency Reward (언어 일관성)                 │
│              │                                              │
│              ▼                                              │
│                                                             │
│ Stage 3: Large-scale SFT                                    │
│   - Reasoning 데이터 (600K) + Non-reasoning 데이터 (200K)    │
│   - 비율: Reasoning : Non-reasoning = 3 : 1                 │
│   - Rejection Sampling으로 고품질 데이터 생성                  │
│              │                                              │
│              ▼                                              │
│                                                             │
│ Stage 4: Final RL                                           │
│   - Reasoning + Non-reasoning 모두에 RL                     │
│   - Helpfulness + Harmlessness Reward 추가                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. R1의 Reward 구성

| Reward 유형 | 적용 대상 | 설명 |
| --- | --- | --- |
| **Correctness** | Reasoning | 정답 여부 |
| **Format** | Reasoning | Think/Answer 형식 |
| **Language Consistency** | Reasoning | 단일 언어 유지 |
| **Helpfulness** | Non-reasoning | 도움이 되는 응답 |
| **Harmlessness** | 전체 (Think 포함) | 안전한 응답 |

### 4. Rejection Sampling

```
Prompt → Model → 여러 Response 생성
                       │
                       ▼
              LLM Judge로 품질 평가
                       │
                       ▼
              고품질 Response만 SFT 데이터로 사용
```

### 5. R1 성능

- OpenAI o1과 경쟁 수준의 Reasoning 성능
- 오픈소스로 공개 → 큰 임팩트

---

## Part 8: Distillation for Reasoning

### 1. 왜 Distillation이 필요한가?

**문제:** R1은 671B 파라미터 → 배포/추론 비용 높음

**해결:** 작은 모델에 Reasoning 능력 전수

### 2. 기존 Distillation vs Reasoning Distillation

| 측면 | 기존 Distillation | Reasoning Distillation |
| --- | --- | --- |
| **Target** | 확률 분포 | 전체 시퀀스 |
| **방법** | KL(Student \|\| Teacher) | Teacher 생성 시퀀스로 SFT |
| **데이터** | 고정된 데이터 | Teacher가 생성한 CoT |

### 3. Reasoning Distillation 과정

```
┌─────────────────────────────────────────────────────────────┐
│ Reasoning Model Distillation                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Step 1: Teacher Model (R1)로 Reasoning Chain 생성            │
│   Prompt → R1 → <think>...</think><answer>...</answer>      │
│                                                             │
│ Step 2: 생성된 시퀀스로 Student Model SFT                     │
│   Student Model learns to mimic Teacher's CoT               │
│                                                             │
│ 결과: 작은 모델도 Reasoning 가능!                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Distillation vs RL from Scratch

**발견:** 작은 모델에서는 Distillation이 더 효율적

- RL from Scratch: 작은 모델에서 Reasoning 학습 어려움
- Distillation: Teacher의 패턴을 직접 학습 → 더 빠르고 안정적

---

## Part 9: Compute Budget과 미래 방향

### 1. Test-time Compute

**개념:** 추론 시 더 많은 계산 = 더 나은 결과

```
더 긴 Reasoning Chain = 더 많은 Forward Pass = 더 나은 답변
```

### 2. Dynamic Budget

| 과제 | 설명 |
| --- | --- |
| **Over-thinking 방지** | 쉬운 문제에 너무 많은 토큰 사용 방지 |
| **Context 인식** | 남은 Context Window 고려 |
| **Budget Forcing** | "Wait", "Time's up" 등으로 조절 |

### 3. Continuous Thoughts

**아이디어:** 토큰 대신 Hidden Representation으로 "생각"

- 더 압축된 표현
- 더 효율적인 Reasoning
- 활발한 연구 분야

---

## 핵심 요약

### Reasoning Model

- 복잡한 문제를 다단계로 분해하여 해결
- Chain of Thought의 대규모 적용
- Output = Reasoning Chain + Answer

### Pass@K

$$\text{Pass@K} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

![Pass@K 요약](/assets/img/cme295-lecture-6/image-20260115-014307.png)

- K번 시도 중 최소 1번 성공 확률
- 코딩/수학 벤치마크에서 주로 사용

### GRPO

- Value Function 없이 Advantage 계산
- Group 내 상대적 Reward로 Advantage 결정
- Reasoning Training에 최적화

### DeepSeek R1

- R1-Zero: SFT 없이 RL만으로 Reasoning 학습 증명
- R1: Cold Start SFT → RL → Large-scale SFT → Final RL
- 오픈소스로 OpenAI o1 수준 달성

### Length 문제

- GRPO의 $\frac{1}{\|o_i\|}$가 긴 답변 선호 유발
- 해결: Token-level 균등화 또는 Length Factor 제거

![Length 요약](/assets/img/cme295-lecture-6/image-20260115-014352.png)

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| Reasoning | 다단계 추론으로 문제 해결 |
| Chain of Thought (CoT) | 추론 과정을 단계별로 설명 |
| Test-time Compute | 추론 시 사용하는 연산량 |
| Pass@K | K번 시도 중 1번 이상 성공 확률 |
| GRPO | Group Relative Policy Optimization |
| GAE | Generalized Advantage Estimation |
| R1-Zero | SFT 없이 RL만으로 학습한 모델 |
| Verifiable Reward | 검증 가능한 보상 (코드/수학) |
| Rejection Sampling | 품질 기준 미달 샘플 제거 |
| Budget Forcing | Reasoning 길이 강제 조절 |

---

## 핵심 공식

**Pass@K:**

$$\text{Pass@K} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

**GRPO Advantage:**

$$A_i = \frac{R_i - \bar{R}}{\sigma_R}$$

**GRPO Loss:**

$$J_{GRPO} = \mathbb{E}\left[\sum_{i,t} \frac{1}{|o_i|}\left(\min(r_t A_i, \text{clip}(r_t)A_i) - \beta D_{KL}\right)\right]$$

**PPO vs GRPO Advantage:**

- PPO: $A = f(\text{Reward}, \text{Value Function})$ (GAE)
- GRPO: $A = \frac{R - \text{Group Mean}}{\text{Group Std}}$

![핵심 공식 요약](/assets/img/cme295-lecture-6/image-20260115-014441.png)

---

## 추천 자료

1. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"** (2025) - R1 전체 파이프라인
2. **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning"** (2024) - GRPO 도입
3. **"Scaling LLM Test-Time Compute Optimally"** (2024) - Test-time Compute
4. **"Chain-of-Thought Prompting Elicits Reasoning in LLMs"** (2022) - CoT 원조
5. **"DAPO: An Open-Source LLM Reinforcement Learning System"** (2025) - Length 문제 해결
6. **"Dr. GRPO"** (2025) - GRPO 개선
7. **"S1: Simple Test-Time Scaling"** (2025) - Budget Forcing

---

## 다음 강의 예고

**Lecture 7: Agents & Tool Use**

- LLM이 실제 행동을 수행하는 방법
- Tool Use, Function Calling
- Retrieval Augmented Generation (RAG)

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 6 정리*
