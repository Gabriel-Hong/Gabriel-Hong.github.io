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

**Vanilla LLM 상세**

**Vanilla LLM이란?**

Lecture 6에서 **Vanilla LLM**은 기본적인/일반적인 LLM을 의미합니다. "Vanilla"는 아이스크림에서 유래한 표현으로, "기본", "표준", "특별한 기능 없는"이라는 뜻입니다.

**Vanilla LLM vs Reasoning Model 비교**

```
Vanilla LLM:
Question → [LLM] → Answer (바로 답변)

Reasoning Model:
Question → [LLM] → <think>Reasoning Chain</think>
                  → <answer>Final Answer</answer>
```

**Vanilla LLM의 특징**

| 강점 | 약점 |
| --- | --- |
| 언어/코드 이해 | 제한된 추론 능력 ← Reasoning Model이 해결 |
| 코드 생성/디버깅 | 지식 Cutoff (최신 정보 모름) |
| 창작 능력 (에세이, 시 등) | 실제 행동 불가 (예약, 주문 등) |

**핵심 차이점**

**Vanilla LLM**은 질문을 받으면 바로 답변을 출력하지만, **Reasoning Model**은 답변 전에 `<think>` 블록에서 추론 과정을 거칩니다.

예를 들어:

- **Vanilla LLM:** "2020년생 곰이 2025년에 몇 살?" → "5살" (바로 답)
- **Reasoning Model:** "2020년생 곰이 2025년에 몇 살?" → `<think>2025 - 2020 = 5</think><answer>5살</answer>`

Lecture 6의 핵심은 이 **Vanilla LLM의 제한된 추론 능력**을 **Chain of Thought + RL (GRPO)**로 개선하는 방법입니다!

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

**Pass@K 상세**

**1. Pass@K란?**

**정의:** K번 시도(샘플링) 중 최소 1번 이상 정답을 맞출 확률

코딩이나 수학 문제처럼 정답을 검증할 수 있는 태스크에서 사용합니다.

**2. 왜 Pass@K가 필요한가?**

일반적인 대화형 태스크:
- 정답이 여러 개일 수 있음
- 검증이 어려움
- 1번 생성해서 평가

코딩/수학 태스크:
- 정답 검증 가능 (테스트 케이스 통과, 정답 일치)
- 여러 번 시도해서 하나라도 맞으면 성공
- LLM의 생성은 확률적 → 여러 번 시도하면 정답 나올 확률 ↑

**코딩 문제 예시:**

```
문제: "두 수의 합을 반환하는 함수 작성"

시도 1: def add(a,b): return a-b  → ✗ 틀림
시도 2: def add(a,b): return a+b  → ✓ 정답
시도 3: def add(a,b): return a+b  → ✓ 틀림

→ 3번 중 1번 맞음 → Pass@3 = 성공!
```

**3. Pass@K 공식 유도**

**상황 설정:**
- 총 $n$개의 샘플을 생성했음
- 그 중 $c$개가 정답 (테스트 통과)
- $k$개를 선택했을 때, 최소 1개가 정답일 확률은?

**유도 과정:**

$$\text{Pass@K} = 1 - P(\text{k개 모두 오답})$$

k개를 선택했을 때 모두 오답일 확률을 먼저 계산:

$$P(\text{k개 모두 오답}) = \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

- 분자: 오답 $(n-c)$개 중에서 $k$개를 고르는 경우의 수
- 분모: 전체 $n$개 중에서 $k$개를 고르는 경우의 수

따라서:

$$\text{Pass@K} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

**4. 구체적 예시**

10번의 시도 중 정답 3개, 오답 7개인 경우:
- $n = 10$, $c = 3$

$$\text{Pass@5} = 1 - \frac{\binom{7}{5}}{\binom{10}{5}} = 1 - \frac{21}{252} = 1 - 0.083 = 0.917$$

→ 91.7%로 5번 안에 정답 포함

**특수 케이스 - Pass@1:**

$$\text{Pass@1} = 1 - \frac{\binom{n-c}{1}}{\binom{n}{1}} = 1 - \frac{n-c}{n} = \frac{c}{n}$$

→ 단순히 정답 비율 → Pass@1 = 3/10 = 0.3 (30%)

**K 값에 따른 변화:**

| K 값 | Pass@K (n=10, c=3) |
| --- | --- |
| K=1 | 30% |
| K=5 | 91.67% |
| K=10 | 100% (모든 시도 포함) |

**5. Pass@K와 Temperature의 관계**

Temperature는 LLM 출력의 다양성을 조절합니다.

| Temperature | 다양성 | 품질 | Pass@K 효과 |
| --- | --- | --- | --- |
| 너무 낮음 ($T \approx 0$) | 낮음 (같은 답 반복) | 높음 | K 늘려도 변화 없음 |
| 너무 높음 ($T \approx 1.2$) | 높음 | 낮음 (횡설수설) | Pass@K 오히려 감소 |
| 적절함 ($T \approx 0.4 \sim 0.8$) | 적절 | 적절 | K 늘릴수록 Pass@K 증가 |

**적절한 Temperature 범위:** $T = 0.4 \sim 0.8$ 이 적절

**6. Pass@K vs 다른 메트릭**

| 메트릭 | 설명 |
| --- | --- |
| **Pass@K** | K번 중 1번 이상 정답 확률 |
| **Consensus@K** | K개 답변 중 가장 많이 나온 답 선택 (다수결) |
| **Pass@1 (Accuracy)** | 단순 정답률 |

**7. 실제 계산 코드**

```python
import math

def pass_at_k(n, c, k):
    """
    n: 총 샘플 수
    c: 정답 샘플 수
    k: 선택할 샘플 수
    """
    if n - c < k:
        return 1.0  # 오답보다 선택 수가 많으면 무조건 정답 포함

    # C(n-c, k) / C(n, k) 계산
    numerator = math.comb(n - c, k)
    denominator = math.comb(n, k)

    return 1.0 - numerator / denominator

# 예시
print(f"Pass@1 (n=10, c=3): {pass_at_k(10, 3, 1):.2%}")   # 30%
print(f"Pass@5 (n=10, c=3): {pass_at_k(10, 3, 5):.2%}")   # 91.67%
print(f"Pass@10 (n=10, c=3): {pass_at_k(10, 3, 10):.2%}") # 100%
```

출력:

```
Pass@1 (n=10, c=3): 30.00%
Pass@5 (n=10, c=3): 91.67%
Pass@10 (n=10, c=3): 100.00%
```

**8. 핵심 정리**

| 개념 | 설명 |
| --- | --- |
| **목적** | 검증 가능한 태스크에서 "여러 번 시도" 허용 시 성능 측정 |
| **공식** | $\text{Pass@K} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$ |
| **Pass@1** | 단순 정답률 ($\frac{c}{n}$) |
| **K가 클수록** | Pass@K 증가 (단, Temperature 적절해야 함) |
| **활용** | HumanEval, AIME 등 코딩/수학 벤치마크 |

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

**Reasoning Model 학습 상세**

#### 1. 왜 SFT만으로는 부족한가?

먼저 **SFT (Supervised Fine-Tuning)** 방식의 한계를 이해해야 합니다.

**SFT 방식의 Reasoning 학습:**

```
SFT 방식

데이터 준비:
  문제: 2020년생 곰이 2025년에 몇 살?

  Chain of Thought (인간 작성):
  "2025에서 2020을 빼면 5가 나온다.
   따라서 곰은 5살이다."

  정답: 5살

         ↓
이런 데이터 수천~수만 개를 인간이 직접 작성
         ↓
LLM에 SFT 학습
```

**SFT의 3가지 한계:**

| 한계 | 설명 |
|------|------|
| **1. 고품질 데이터 부족** | 복잡한 수학/코딩 문제의 Reasoning Chain을 인간이 작성하기 매우 어려움 |
| **2. 모델 ≠ 인간** | 인간에게 최적인 추론 방식이 LLM에게도 최적이 아닐 수 있음 |
| **3. Reward 낭비** | 코드/수학은 정답 검증이 가능한데, 이를 활용하지 않음 |

#### 2. RL 기반 접근법의 핵심 아이디어

**핵심 통찰: Verifiable Reward**

> 코딩/수학 문제는 정답을 자동으로 검증할 수 있다!

```
일반 태스크 vs 검증 가능한 태스크

일반 태스크 (예: 에세이 작성):
  "AI의 미래에 대해 써주세요"
       ↓
  LLM 응답: "AI는 미래에..."
       ↓
  좋은 응답인가? → 인간이 판단해야 함

검증 가능한 태스크 (예: 코딩):
  "두 수의 합을 반환하는 함수 작성"
       ↓
  LLM 응답: def add(a,b): return a+b
       ↓
  테스트: add(2,3)==5? add(-1,1)==0?
       ↓
  결과: ✅ 통과 → Reward = 1
        ❌ 실패 → Reward = 0

→ Reward Model 학습 불필요! 자동으로 보상 계산!
```

#### 3. Reasoning RL의 Reward 구성

전체 Reward 구조:

$$R_{total} = R_{format} + R_{correctness}$$

**3.1 Format Reward ($R_{format}$)**

특정 LLM이 올바른 형식으로 출력하도록 유도:

```
Format Reward 계산

예시 출력:
  response = "<think>2025에서 2020을 빼면...</think>\n<answer>5</answer>"

  case 1: 올바른 형식 → R_format = 1
    <think>...</think>
    <answer>정답</answer>

  case 2: 형식은 맞지만 think 태그만 있는 경우
    → R_format = 0.5 (부분 점수)

  case 3: 형식이 잘못된 경우
    → R_format = 0 또는 -1

구현 팁:
  - 정규표현식으로 <think>...</think> 패턴 매칭
  - <answer> 태그로부터 최종 답 추출
  - 둘 다 있어야만 full reward
  - 빈 문자열이면 -1 (패널티)
```

**3.2 Correctness Reward ($R_{correctness}$)**

특정 태스크 성격에 맞게 설정:

```
Correctness Reward 구현

코딩 문제:
  테스트 케이스 실행
  def check_correctness(code, tests):
      try:
          exec(code)
          results = [run_test(t) for t in tests]
          return sum(results) / len(results)
      except:
          return 0.0

수학 문제:
  정답 비교
  def check_math(answer, ground_truth):
      # 수식 정규화 후 비교
      return 1.0 if normalize(answer) == normalize(ground_truth) else 0.0
```

#### 4. RL 학습 파이프라인

전체 흐름:

```
RL Reasoning 학습 파이프라인

Step 1: 문제 준비
  문제 N개 샘플링 (수학, 코딩 등)

Step 2: LLM이 각 문제에 대해 G개의 응답 생성 (GRPO 방식)
  Response 1: <think>123+456은...579</think><answer>579</answer>
  Response 2: <think>100+400=500...</think><answer>579</answer>
  Response 3: <think>계산하면...</think><answer>578</answer>  ← 오답
  ...

Step 3: 각각의 Reward 계산
  Response 1: R_format=1, R_correct=1 → Total=2   ← 좋음!
  Response 2: R_format=1, R_correct=1 → Total=2
  Response 3: R_format=1, R_correct=0 → Total=1   ← 형식은 맞지만 오답
  Response 4: R_format=0, R_correct=0 → Total=0   ← 나쁨

Step 4: GRPO로 Advantage 계산 & Policy 업데이트

  Mean = (2+2+1+0)/4 = 1.25
  Std = ...
  A_i = (R_i - Mean) / Std

Step 5: 반복 (수천~수만 스텝)
```

#### 5. SFT vs RL 비교

| | **SFT** | **RL (Verifiable Reward)** |
|------|------|------|
| **데이터** | 인간이 CoT 직접 작성 | 모델이 스스로 생성 후 검증 |
| **Reward** | 없음 (지도 학습) | 자동 계산 (테스트/정답 비교) |
| **추론 방식** | 인간의 추론 모방 | 모델 최적 추론 탐색 |
| **확장성** | 낮음 | 높음 |

**핵심적인 차이를 보는 예시:**

```
문제: "N 크기의 리스트 정렬 함수를 작성하라"

SFT 접근:
  인간이 작성한 정답: def sort(lst): return sorted(lst)
  → LLM은 이 패턴만 학습

RL 접근:
  LLM 스스로 다양한 시도:
  시도 1: def sort(lst): return sorted(lst)      ✅ Reward=1
  시도 2: def sort(lst): ... (버블 정렬 구현)      ✅ Reward=1
  시도 3: def sort(lst): return lst               ❌ Reward=0
  → 다양한 올바른 방법 탐색 가능!
```

#### 6. DeepSeek R1-Zero 실험 결과

R1-Zero (SFT 없이 순수 RL만으로 Reasoning 학습한 모델):

- 학습: DeepSeek R1-Zero (Pre-trained, 67B 모델)
- 데이터: 수학 + 코딩 문제
- 보상: Verifiable Rewards만 사용
- Reasoning Chain을 명시적으로 학습하지 않았는데도 자동 생성!

**학습 곡선:**

- RL Step 증가에 따라 AIME 벤치마크 성능이 계단식으로 상승
- 약 8000 스텝 이후 급격한 성능 향상

**관찰 결과:**

- RL 학습만으로 모델이 자연스럽게 Reasoning 능력 획득
- 학습 중에는 모델 스스로 Reasoning Chain 길이 증가
- "Aha moment": 어느 순간부터 정답률이 급격히 상승하는 구간 관찰

#### 7. 해석: Reasoning이 왜 자동으로 발현되나?

- **이유 1: Exploration (탐색)**
  - RL은 다양한 출력을 시도하면서 Reward가 높은 패턴을 발견
- **이유 2: Credit Assignment (공로 할당)**
  - 긴 추론 과정 중 어떤 부분이 정답에 기여했는지 학습
- **이유 3: Scalability (확장성)**
  - 문제가 어려울수록 더 긴 Chain을 생성하도록 자연스럽게 학습

#### 8. 정리

| 개념 | 설명 |
|------|------|
| **Verifiable Reward** | 정답 검증이 가능한 태스크에서 자동으로 Reward 계산 |
| **R_format** | 올바른 형식(`<think>`, `<answer>`)으로 출력했는지 |
| **R_correctness** | 최종 답이 정답인지 (테스트 통과, 정답 일치) |
| **SFT의 한계** | 고품질 CoT 데이터 부족, 인간 ≠ 모델 최적 방식 |
| **RL의 장점** | 탐색 가능, 모델 스스로 최적 추론 방식 발견 |
| **R1-Zero** | SFT 없이 RL만으로 Reasoning 학습 성공 증명 |

**핵심 메시지:**

> 코딩/수학처럼 정답을 검증할 수 있는 태스크에서는, 인간이 추론 과정을 일일이 작성하는 것보다 RL로 모델이 스스로 탐색하게 하는 것이 더 효과적!

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

**Value Function 상세**

#### 1. 강화학습 기본 개념 복습

GRPO와 Value Function을 이해하려면 먼저 강화학습(RL)의 기본 구조를 알아야 합니다.

**강화학습 기본 구조:**

```
Agent (Policy) ──Action(a)──> Environment
       ↑                           │
       └───── State(s), Reward(r) ─┘
```

- **State (s):** 현재 상태
- **Action (a):** Agent가 취하는 행동
- **Reward (r):** 행동에 대한 보상
- **Policy ($\pi$):** 상태 → 행동 매핑 함수

**LLM에서의 RL 매핑:**

| RL 개념 | LLM에서의 의미 |
|------|------|
| **State (s)** | 프롬프트 + 지금까지 생성한 토큰 |
| **Action (a)** | 다음에 생성할 토큰 |
| **Reward (r)** | 전체 응답 완료 후 받는 점수 |
| **Policy ($\pi$)** | LLM 자체 (토큰 확률 분포) |

#### 2. Value Function이란?

**정의:**

**Value Function $V(s)$:** 현재 상태 $s$에서 시작해서 앞으로 받을 것으로 예상되는 총 Reward

$$V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$$

- $\gamma$: 할인율 (미래 보상을 얼마나 중요하게 볼지)
- $r_t$: 시점 $t$에서 받는 보상

**직관적 이해:**

```
Value Function의 직관적 의미

비유: 체스 게임

현재 체스판 상태 (State s)

V(s) = "이 상태에서 내가 이길 확률/기대 점수"
     = 0.7 (60% 승률 예상)
```

**LLM에서의 Value Function:**

```
LLM에서 Value Function

문제: "123 + 456 = ?"

State 1: "123 + 456 = ?"
V(s₁) = 0.7  → "이 상태에서 정답 맞출 확률 70%"

State 2: "123 + 456 = ? <think>먼저 일의 자리..."
V(s₂) = 0.85 → "추론 시작했으니 확률 85%로 상승"

State 3: "... 3+6=9, 2+5=7, 1+4=5 이므로"
V(s₃) = 0.95 → "거의 다 풀었으니 확률 95%"

Value Function = 각 시점에서 "최종 정답 맞출 기대값"
```

#### 3. Advantage Function이란?

**정의:**

**Advantage $A(s, a)$:** 특정 행동 $a$가 평균보다 얼마나 좋은지

$$A(s, a) = Q(s, a) - V(s)$$

- $Q(s, a)$: 상태 $s$에서 행동 $a$를 취했을 때의 기대 보상
- $V(s)$: 상태 $s$의 평균 기대 보상

**직관적 이해:**

```
Advantage의 직관적 의미

상황: 수학 문제를 푸는 중
현재 상태 s에서의 평균 기대값: V(s) = 0.7

선택지 A: "인수분해로 풀기"
→ Q(s, A) = 0.9 (이 방법 선택시 기대값)
→ A(s, A) = 0.9 - 0.7 = +0.2  ✅ 평균보다 좋음!

선택지 B: "무작정 대입하기"
→ Q(s, B) = 0.5 (이 방법 선택시 기대값)
→ A(s, B) = 0.5 - 0.7 = -0.2  ❌ 평균보다 나쁨!

Advantage = "이 행동이 평균 대비 얼마나 좋은가?"

→ Advantage > 0: 이 행동 더 하도록 학습
→ Advantage < 0: 이 행동 덜 하도록 학습
```

#### 4. PPO에서 Value Function의 역할

**PPO의 Advantage 계산 (GAE)**

PPO는 **Generalized Advantage Estimation (GAE)**를 사용:

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

여기서 TD Error $\delta_t$는:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**PPO 학습 구조:**

```
PPO 학습 구조

  Policy Network        Value Network
     π(a|s)                V(s)
  "다음 토큰은?"       "현재 얼마나
                        좋은 상황?"
       │                    │
       └────────┬───────────┘
                ↓
         Advantage 계산
         A = f(Reward, V(s))
                ↓
         Policy 업데이트

문제점:
  - Value Network를 별도로 학습해야 함
  - 추가 파라미터, 추가 연산, 추가 복잡성
  - Value 추정이 부정확하면 학습 불안정
```

#### 5. GRPO: Value Function 없이 Advantage 계산

**GRPO의 핵심 아이디어:**

"Value Function을 학습하는 대신, 같은 문제에 대한 여러 응답을 비교하면 되지 않을까?"

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G)}$$

**GRPO vs PPO 비교:**

```
PPO vs GRPO Advantage

PPO 방식:
  문제 → LLM → 1개 응답 → Reward
  + Value Network → V(s) 계산
  Advantage = Reward - V(s)

  → 별도의 Value Network가 필요하고
  → 이 Value Network도 학습해야 하므로
  + Value Network 2배의 메모리 필요!

GRPO 방식:
  문제 → LLM → G개 응답 → R₁, R₂, ..., R_G
  Mean = (R₁+R₂+...+R_G)/G
  Std = std(R₁, ..., R_G)
  A_i = (R_i - Mean) / Std

  → Value Network 없이 Group 비교로 Advantage 계산!
```

#### 6. GRPO가 작동하는 이유

**핵심 통찰:**

**Group Mean = Value Function의 역할**

```
Group Mean이 V(s)을 근사하는 이유:

"이 문제에서 평균적으로 얼마나 잘 하는가?" = Mean(R)
→ 이것이 바로 Value Function이 추정하려는 것!

차이점:
  - Value Function: 별도 모델이 학습으로 추정
  - GRPO: 실제 샘플들의 평균으로 직접 계산

장점:
  - 학습 불필요 (추정 오차 없음)
  - SFT와 다르게, Advantage를 더 정확히 계산할 수 있음
  - 별도 네트워크 불필요 → 메모리 절약
  - 구현 단순화
```

#### 7. PPO vs GRPO 종합 비교

| 측면 | PPO | GRPO |
|------|------|------|
| **Advantage 계산** | $A = f(R, V(s))$ | $A = \frac{R - R_{mean}}{R_{std}}$ |
| **Value Function** | 필요 (별도 네트워크) | 불필요 |
| **추가 파라미터** | Value Network 파라미터 | 없음 |
| **학습 복잡도** | 높음 (2개 네트워크 동시 학습) | 낮음 |
| **샘플 효율성** | 높음 | 낮음 (여러 샘플 필요) |
| **적합한 상황** | 일반적인 RL | Reasoning (검증 가능한 Reward) |

#### 8. 시각적 요약

```
PPO vs GRPO 구조

[PPO]
Prompt ──> LLM ──> 1개 Response ──> Reward
                        │
  Value Network ────> Advantage 계산
  (별도 학습 필요)      A = R - V(s)

[GRPO]
                 ┌─> Response 1 ──> R₁
Prompt ──> LLM ─┼─> Response 2 ──> R₂ ──> Group
                 ├─> Response 3 ──> R₃     Mean/Std
                 └─> Response 4 ──> R₄
                                      ↓
                               Advantage 계산
                               A = (R-μ)/σ

  ✅ Value Network 불필요!
```

#### 9. 핵심 정리

| 개념 | 설명 |
|------|------|
| **Value Function $V(s)$** | 현재 상태에서 앞으로 받을 기대 Reward |
| **Advantage $A(s,a)$** | 특정 행동이 평균보다 얼마나 좋은지 |
| **PPO의 문제** | Value Function 학습을 위한 별도 네트워크 필요 |
| **GRPO의 해결책** | 같은 문제에 여러 응답 생성 → Group 평균으로 대체 |
| **GRPO의 장점** | 구조 단순화, Value Network 불필요 |
| **GRPO의 단점** | 더 많은 샘플 필요 (Group 크기 G) |

**핵심 메시지:**

> GRPO는 **같은 문제에서 여러 응답을 비교하여** 그 상대적 가치(Value)**를 추정하는 아이디어**로, Value Function 없이도 효과적인 RL 학습을 가능하게 합니다!

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

**PPO와의 공통점:**

- Ratio $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 사용
- Clipping 메커니즘으로 큰 업데이트 방지

**PPO와의 차이점:**

- KL Divergence가 명시적으로 손실에 포함
- Advantage가 Group 기반으로 계산

**GRPO(Group Relative Policy Optimization) 상세**

#### 1. 배경: PPO 복습

GRPO를 이해하려면 먼저 **PPO (Proximal Policy Optimization)**를 알아야 합니다.

**PPO의 목표:** Policy $\pi_\theta$를 업데이트하여 Advantage가 높은 행동을 더 많이 하도록 학습

**PPO Loss 함수:**

$$L^{PPO}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

여기서:

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ (확률 비율)
- $A_t$ = Advantage (이 행동이 평균보다 얼마나 좋은지)
- $\epsilon$ = 클리핑 범위 (보통 0.1~0.2)

**PPO의 Advantage 계산: GAE**

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**문제:** $V(s)$를 계산하려면 **Value Network**가 필요!

#### 2. PPO의 한계

PPO 학습 시 필요한 것:

| | Policy Network $\pi(a \mid s)$ | Value Network $V(s)$ |
|---|---|---|
| 파라미터 | LLM 파라미터 (수십~수백B) | 별도 파라미터 (추가 비용) |

두 네트워크를 동시에 학습해야 함.

문제점:

1. **추가 파라미터:** Value Network도 학습해야 함 → 메모리 사용량 증가
2. **학습 복잡성:** 두 네트워크의 균형 맞추기 어려움 → 하이퍼파라미터 튜닝 복잡
3. **Value 추정 오류:** $V(s)$가 부정확하면 학습 불안정 → Advantage 계산이 틀어짐
4. **LLM에서 특히 어려움:** State가 매우 고차원 → 정확한 $V(s)$ 추정이 힘듦

#### 3. GRPO의 핵심 아이디어

**핵심 통찰:**

> "Value Function을 학습하지 말고, 같은 문제에 대한 여러 응답의 상대적 성능을 비교하자!"

하나의 Prompt에 대해 G개의 응답을 생성:

- Prompt: "$x^2 - 5x + 6 = 0$을 풀어라"
- LLM → Response 1: "x=2, x=3" → $R_1 = 1$ (정답)
- LLM → Response 2: "x=2, x=3" → $R_2 = 1$ (정답)
- LLM → Response 3: "x=1, x=6" → $R_3 = 0$ (오답)
- LLM → Response 4: "x=3, x=4" → $R_4 = 0$ (오답)

Group 내에서 상대적 비교:

- 평균: $\mu = (1+1+0+0)/4 = 0.5$
- 표준편차: $\sigma = 0.5$

Advantage $= (R - \mu) / \sigma$

- $A_1 = (1-0.5)/0.5 = +1$ → 평균보다 좋음
- $A_2 = (1-0.5)/0.5 = +1$ → 평균보다 좋음
- $A_3 = (0-0.5)/0.5 = -1$ → 평균보다 나쁨
- $A_4 = (0-0.5)/0.5 = -1$ → 평균보다 나쁨

→ **Value Network 없이 Advantage 계산 완료!**

#### 4. GRPO Advantage 공식

$$A_i = \frac{R_i - \text{mean}(R_1, \ldots, R_G)}{\text{std}(R_1, \ldots, R_G)}$$

또는:

$$A_i = \frac{R_i - \bar{R}}{\sigma_R}$$

여기서:

- $R_i$: i번째 응답의 Reward
- $\bar{R}$: Group 내 평균 Reward
- $\sigma_R$: Group 내 표준편차
- $G$: Group 크기 (보통 4~16)

**왜 정규화(Normalization)하는가?**

정규화 없이 $(R - \text{mean})$만 사용하면:

- Case 1 (쉬운 문제): Rewards = [1, 1, 1, 0], Mean = 0.75 → $A_1 = 1 - 0.75 = 0.25$, $A_4 = 0 - 0.75 = -0.75$
- Case 2 (어려운 문제): Rewards = [1, 0, 0, 0], Mean = 0.25 → $A_1 = 1 - 0.25 = 0.75$, $A_4 = 0 - 0.25 = -0.25$

→ 같은 "정답"이지만 Advantage가 다름. 문제 난이도에 따라 학습 신호 크기가 달라짐.

정규화 후 $(R - \text{mean}) / \text{std}$:

- Case 1: std $\approx$ 0.43, $A_1 = 0.25 / 0.43 \approx 0.58$
- Case 2: std $\approx$ 0.43, $A_1 = 0.75 / 0.43 \approx 1.74$

→ 어려운 문제에서 정답 맞추면 더 큰 Advantage! 직관적으로 맞음 (어려운 걸 맞추면 더 칭찬)

#### 5. GRPO Loss 함수 분해

**전체 Loss 공식:**

$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\} \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( L_t^{CLIP} - \beta D_{KL} \right) \right]$$

각 구성요소를 살펴봅시다.

**5.1 Clipping Loss ($L_{clip}$)**

$$L_t^{CLIP} = \min\left(r^t \cdot A_i, \; \text{clip}(r^t, 1-\epsilon, 1+\epsilon) \cdot A_i\right)$$

여기서:

$$r^t = \frac{\pi_\theta(o_t^i | q, o_{<t}^i)}{\pi_{\theta_{old}}(o_t^i | q, o_{<t}^i)}$$

Clipping 효과:

- $A > 0$ (좋은 행동일 때): r을 높이고 싶지만 $1+\epsilon$까지만!
- $A < 0$ (나쁜 행동일 때): r을 낮추고 싶지만 $1-\epsilon$까지만!

→ 한 번에 너무 큰 업데이트 방지

**5.2 KL Divergence 항 ($D_{KL}$)**

$$D_{KL}^t = D_{KL}\left(\pi_\theta(o_t | q, o_{<t}) \| \pi_{ref}(o_t | q, o_{<t})\right)$$

KL Divergence의 역할:

- 목적: Reference Model(학습하기 전 상태)에서 너무 멀어지지 않게
- $\beta$가 크면: KL 항의 영향이 커서 Base Model에 가깝게 유지
- $\beta$가 작으면 (예: $\beta = 0.04$): KL 항의 영향이 작아서 더 자유롭게 학습
- DeepSeek에서는 $\beta$를 점점 줄이는 방식 사용

→ 처음에는 보수적으로, 나중에는 자유롭게

**5.3 Length Normalization ($\frac{1}{|o_i|}$)**

공식에서: $(1/|o_i|) \times \Sigma_t (\ldots)$

$|o_i|$ = i번째 응답의 토큰 수

목적: 응답 길이에 관계없이 공평한 학습 신호

예시:

- Response 1: "답은 5입니다." (5 tokens)
- Response 2: "먼저 계산하면... 답은 5" (20 tokens)

정규화 없이: Response 2가 토큰 많아서 Loss 기여도 4배 → 긴 응답에 편향됨

정규화 후: 각 응답이 동등한 기여도

#### 6. GRPO 학습 알고리즘

```
Input: 초기 Policy π_θ, 문제 데이터셋 D, Group 크기 G

for each iteration do:
    1. 문제 샘플링
       q ~ D
    2. G개의 응답 생성
       {o_1, o_2, ..., o_G} ~ π_θ_old(·|q)
    3. 각 응답에 Reward 계산
       R_i = R_format(o_i) + R_correctness(o_i)
    4. Group Advantage 계산
       μ = mean(R_1, ..., R_G)
       σ = std(R_1, ..., R_G)
       A_i = (R_i - μ) / σ
    5. GRPO Loss 계산
       L = (1/G) Σ_i (1/|o_i|) Σ_t [L_clip - β·D_KL]
    6. Gradient Descent로 θ 업데이트
       θ ← θ - α∇L
end for
```

#### 7. 구체적 예시

**GRPO 학습 예시:**

**[Step 1] 문제 샘플링:** q = "피보나치 수열의 10번째 값은?", 정답 = 55

**[Step 2] G=4개 응답 생성:**

| 응답 | 내용 | 길이 |
|---|---|---|
| $o_1$ | `<think>F(1)=1, F(2)=1, F(3)=2, ... F(10)=55</think><answer>55</answer>` | 50 tokens |
| $o_2$ | `<think>1,1,2,3,5,8,13,21,34,55</think><answer>55</answer>` | 25 tokens |
| $o_3$ | `<think>대충 계산하면...</think><answer>50</answer>` | 15 tokens |
| $o_4$ | 피보나치 10번째는 89입니다. | 10 tokens (형식 오류!) |

**[Step 3] Reward 계산:**

| 응답 | $R_{format}$ | $R_{correctness}$ | $R_{total}$ |
|---|---|---|---|
| $o_1$ | 1 | 1 | 2 |
| $o_2$ | 1 | 1 | 2 |
| $o_3$ | 1 | 0 | 1 |
| $o_4$ | 0 | 0 | 0 |

**[Step 4] Advantage 계산:**

$\mu = (2+2+1+0)/4 = 1.25$

$\sigma = \sqrt{((2-1.25)^2 + (2-1.25)^2 + (1-1.25)^2 + (0-1.25)^2)/4} = \sqrt{0.6875} \approx 0.83$

- $A_1 = (2 - 1.25) / 0.83 = +0.90$ (좋음)
- $A_2 = (2 - 1.25) / 0.83 = +0.90$ (좋음)
- $A_3 = (1 - 1.25) / 0.83 = -0.30$ (약간 나쁨)
- $A_4 = (0 - 1.25) / 0.83 = -1.51$ (매우 나쁨)

**[Step 5] Policy 업데이트:**

- $o_1$, $o_2$ 방향으로 확률 증가 ($A > 0$)
- $o_3$, $o_4$ 방향으로 확률 감소 ($A < 0$)
- 특히 $o_4$는 크게 감소 ($A = -1.51$)

#### 8. GRPO의 Length 문제

**문제점:** 긴 응답이 길어질수록 선호되거나 비선호되는 편향이 발생

**Length Bias 문제:**

- 긴 응답: 각 토큰의 $r_t$가 곱해져서 advantage가 과대/과소 평가될 수 있음
- Response 1: 10 tokens → 기여도 작음
- Response 2: 100 tokens → 기여도 큼

**해결책:**

- 전체 Loss에 각 토큰의 gradient를 합산할 때 길이로 나누어 정규화

**방법 1: Token-level 손실 평균** → 각 응답의 토큰별 loss의 평균을 구함

**방법 2: Length Penalty 추가**

$$R_{total} = R_{correctness} + R_{format} - \lambda \cdot |o|$$

**방법 3: DAPO 방식**

DAPO의 핵심:

- 0과 1 사이의 Reward만 사용 ($R \in \{0, 1\}$)
- Clipping을 비대칭으로 적용
- 과도한 길이 증가를 방지하는 메커니즘 내장

#### 9. PPO vs GRPO 요약 비교

| 항목 | PPO | GRPO |
|---|---|---|
| Advantage 계산 | $A = f(V, r)$ (GAE) | $A = \frac{R_i - \bar{R}}{\sigma_R}$ |
| Value Function | 필요 (별도 네트워크) | 불필요 |
| Value Network | 별도 학습 필요 | 없음 → 메모리 절약 |
| Group 크기 | 해당 없음 | G개의 응답 비교 |
| Length 처리 | 별도 처리 | $1/\|o_i\|$로 Length Factor 포함 |

#### 10. GRPO가 Reasoning에 적합한 이유

GRPO가 Reasoning 학습에 잘 맞는 이유:

1. **Verifiable Reward와 궁합:** 수학/코드 → 정답 여부가 명확 (True/False), Reward 자체가 믿을 만함
2. **자연스러운 비교:** Reasoning Style이 다양함 → 여러 풀이를 비교하는 것이 의미있음
3. **효율적:** 추가 모델 없이 학습 가능 → 자원이 제한된 환경에서도 적용 가능
4. **안정적:** 다양한 문제 난이도에서 안정적인 학습 신호 (Normalization 덕분)

→ 이런 특성 때문에 DeepSeek-R1이 GRPO를 선택

#### 11. 핵심 정리

| 개념 | 설명 |
|---|---|
| GRPO | Group Relative Policy Optimization |
| 핵심 아이디어 | Value Network 없이 Group 내 상대적 비교로 Advantage 계산 |
| Group 크기 | 보통 4~16 |
| Loss | Clipping Loss + KL Penalty |
| Length 처리 | $1/\|o_i\|$로 Length Factor 포함 |
| Advantage | $(R_i - \bar{R}) / \sigma_R$ |

#### 12. GRPO의 수식 요약

**Advantage:**

$$A_i = \frac{R_i - \bar{R}}{\sigma_R}$$

**Loss:**

$$\mathcal{L}_{GRPO} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[\min(r_t A_i, \text{clip}(r_t) A_i) - \beta D_{KL}\right]$$

**핵심:** GRPO는 Value Network 없이 같은 문제에 대한 여러 응답의 Reward를 비교하여 Advantage를 계산하고, Reasoning에 적합한 구조로 Verifiable Reward와 잘 결합됩니다.

**Advantage와 Loss의 관계 상세**

#### 1. 핵심 답변

> Advantage는 Loss에 "더해지는" 것이 아니라, **Loss 계산의 "입력(가중치)"로 사용**됩니다.

```
Advantage → Loss → Gradient → 업데이트

Step 1: Advantage 계산 (각 응답이 얼마나 좋은지)
        A_i = (R_i - μ) / σ
              ↓
Step 2: Loss 계산 (Advantage를 가중치로 사용)
        L = f(r_t, A_i)
              ↓
Step 3: Gradient 계산
        ∇L = ∂L/∂θ
              ↓
Step 4: 파라미터 업데이트
        θ ← θ - α∇L
```

#### 2. Policy Gradient의 기본 원리

**목표: 좋은 행동의 확률을 높이기**

RL의 목표는 기대 **Reward**를 최대화하는 것:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[R]$$

이를 위한 Policy Gradient:

$$\nabla J(\theta) = \mathbb{E}\left[\nabla \log \pi_\theta(a|s) \cdot R\right]$$

**직관적 이해:**

- $\nabla \log \pi(a|s)$: "이 행동의 확률을 높이는 방향"
- $R$ (또는 $A$): "얼마나 높일지 (또는 낮출지)"

$R > 0$ (좋은 결과): $\nabla \log \pi \times$ (양수) = 양의 gradient → 이 행동의 확률 증가

$R < 0$ (나쁜 결과): $\nabla \log \pi \times$ (음수) = 음의 gradient → 이 행동의 확률 감소

#### 3. Advantage의 역할

**왜 R 대신 Advantage를 쓰는가?**

Raw Reward 사용 시 문제:

- 쉬운 문제: $R = 1$ (정답)
- 어려운 문제: $R = 1$ (정답)
- → 둘 다 같은 크기의 gradient → 쉬운 문제 맞춘 것도 크게 칭찬?

Advantage 사용 시 해결:

- 쉬운 문제: $A = (1 - 0.9) / \sigma$ = 작은 양수 (대부분 맞추니까 평균이 높음)
- 어려운 문제: $A = (1 - 0.2) / \sigma$ = 큰 양수 (대부분 틀리니까 평균이 낮음)
- → 어려운 문제 맞추면 큰 gradient, 쉬운 문제 맞추면 작은 gradient → 합리적!

**Advantage = "평균 대비 얼마나 좋은가"**

- $A > 0$: 평균보다 좋음 → 이 방향으로 학습
- $A < 0$: 평균보다 나쁨 → 이 방향 피하기
- $A \approx 0$: 평균 수준 → 거의 학습 안 함

#### 4. GRPO Loss 공식 분해

**전체 Loss:**

$$L = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[\min(r_t A_i, \text{clip}(r_t) A_i) - \beta D_{KL}\right]$$

**핵심 부분: $r_t \cdot A_i$**

$r_t = \pi_\theta(\text{token}_t) / \pi_{old}(\text{token}_t)$ = 새 정책의 확률 / 옛 정책의 확률

$A_i$ = 이 응답의 Advantage

- Case 1: $A_i > 0$ (좋은 응답) → Loss = $r_t \times A_i$ (양수) → Loss를 최대화하려면? → $r_t$를 크게 (새 정책에서 확률 높이기) → 이 토큰을 더 잘 생성하도록 학습
- Case 2: $A_i < 0$ (나쁜 응답) → Loss = $r_t \times A_i$ (음수) → Loss를 최대화하면? → $r_t$를 작게 (새 정책에서 확률 낮추기) → 이 토큰을 덜 생성하도록 학습

#### 5. 구체적 예시로 이해하기

**GRPO Loss 계산 예시:**

문제: "2 + 3 = ?"

**[Step 1: 응답 생성 & Reward]**

- $o_1$: "`<think>`2+3=5`</think><answer>`5`</answer>`" → $R_1 = 1$
- $o_2$: "`<think>`계산...`</think><answer>`5`</answer>`" → $R_2 = 1$
- $o_3$: "`<think>`음...`</think><answer>`6`</answer>`" → $R_3 = 0$
- $o_4$: "`<think>`모름`</think><answer>`4`</answer>`" → $R_4 = 0$

**[Step 2: Advantage 계산]**

$\mu = (1+1+0+0)/4 = 0.5$, $\sigma = 0.5$

- $A_1 = (1-0.5)/0.5 = +1.0$
- $A_2 = (1-0.5)/0.5 = +1.0$
- $A_3 = (0-0.5)/0.5 = -1.0$
- $A_4 = (0-0.5)/0.5 = -1.0$

**[Step 3: Loss 계산 (각 토큰별)]**

$o_1$의 토큰들: ["`<think>`", "2", "+", "3", "=", "5", ...]

토큰 "5"에 대해 (정답의 핵심 토큰):

- $r_t = \pi_{new}(\text{"5"}) / \pi_{old}(\text{"5"})$
- 가정: $\pi_{old}(\text{"5"}) = 0.3$, $\pi_{new}(\text{"5"}) = 0.4$ → $r_t = 0.4 / 0.3 = 1.33$
- Loss 기여 = $r_t \times A_1 = 1.33 \times 1.0 = 1.33$
- 이 값이 양수 → Loss 최대화 방향 = $\pi(\text{"5"})$ 증가

$o_3$의 토큰들: ["`<think>`", "음", "...", "`<answer>`", "6", ...]

토큰 "6"에 대해 (오답의 핵심 토큰):

- $r_t = \pi_{new}(\text{"6"}) / \pi_{old}(\text{"6"})$
- 가정: $\pi_{old}(\text{"6"}) = 0.2$, $\pi_{new}(\text{"6"}) = 0.25$ → $r_t = 0.25 / 0.2 = 1.25$
- Loss 기여 = $r_t \times A_3 = 1.25 \times (-1.0) = -1.25$
- 이 값이 음수 → Loss 최대화 방향 = $\pi(\text{"6"})$ 감소

**[Step 4: Gradient & 업데이트]**

$\nabla L$ 계산 → $\theta \leftarrow \theta + \alpha \nabla L$ (최대화이므로 +)

결과:

- "5" 토큰 확률 상승 (정답)
- "6" 토큰 확률 하락 (오답)

#### 6. Clipping의 역할

**왜 Clip이 필요한가?**

Clipping이 없으면 발생하는 문제 (상황: $A = +1$, 좋은 응답):

- 학습 초기: $r_t = 1.0$, Loss = $1.0 \times 1 = 1.0$
- 몇 번 업데이트 후: $r_t = 5.0$ (확률 5배 증가), Loss = $5.0 \times 1 = 5.0$
- 더 업데이트 후: $r_t = 100.0$ (확률 100배 증가!), Loss = $100.0 \times 1 = 100.0$
- 문제: 한 방향으로 너무 급격한 업데이트 → 학습 불안정 → Policy가 극단적으로 변함

**Clipping으로 해결:**

$$L_{clip} = \min(r_t A, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A)$$

Clipping의 효과 ($\epsilon = 0.2$):

**$A > 0$ (좋은 응답)일 때:**

- $r_t = 0.5$ → clip = 0.8, Loss = $\min(0.5A, 0.8A) = 0.5A$ → 확률 낮아진 건 그대로 페널티
- $r_t = 1.0$ → clip = 1.0, Loss = $\min(1.0A, 1.0A) = 1.0A$ → 변화 없으면 그대로
- $r_t = 1.5$ → clip = 1.2, Loss = $\min(1.5A, 1.2A) = 1.2A$ ← 제한됨! → 너무 높아지면 1.2로 제한
- $r_t = 5.0$ → clip = 1.2, Loss = $\min(5.0A, 1.2A) = 1.2A$ ← 제한됨! → 아무리 높아져도 $1.2A$가 최대

**$A < 0$ (나쁜 응답)일 때:**

- $r_t = 1.5$ → clip = 1.2, Loss = $\min(1.5A, 1.2A) = \min(-1.5, -1.2) = -1.5$ → 확률 높아진 건 그대로 페널티
- $r_t = 0.5$ → clip = 0.8, Loss = $\min(0.5A, 0.8A) = \min(-0.5, -0.8) = -0.8$ ← 제한됨! → 너무 낮아져도 $-0.8|A|$가 최대 페널티

#### 7. 전체 흐름 요약

```
문제 q
    ↓
LLM이 G개 응답 생성 (o_1, o_2, ..., o_G)
    ↓
각 응답에 Reward 계산 (R_1, R_2, ..., R_G)
(정답 여부 + 형식 여부)
    ↓
Advantage 계산: A_i = (R_i - mean) / std
  ← "이 응답이 평균 대비 얼마나 좋은가?"
    ↓
Loss 계산 (Advantage를 가중치로 사용)
L = Σ [min(r_t × A_i, clip × A_i)]
        ↑
  A_i가 곱해져서 방향과 크기 결정
    ↓
Gradient 계산: ∇L = ∂L/∂θ
    ↓
파라미터 업데이트: θ ← θ + α∇L (최대화)
결과:
  - A>0인 토큰들의 확률 상승
  - A<0인 토큰들의 확률 하락
```

#### 8. 핵심 정리

| 질문 | 답변 |
|---|---|
| Advantage는 뭐하는 건가요? | 각 응답이 "평균 대비 얼마나 좋은지" 측정 |
| Loss는 뭐하는 건가요? | Advantage를 가중치로 사용하여 학습 방향/크기 결정 |
| Advantage가 Loss에 더해지나요? | 더해지는 게 아니라 곱해집니다 ($r_t \times A_i$) |
| 왜 곱하나요? | $A>0$이면 확률 올리고, $A<0$이면 확률 내리도록 gradient 방향 결정 |
| Clip은 왜 필요한가요? | 한 번에 너무 큰 업데이트 방지 (학습 안정성) |

**한 문장 요약:**

> Advantage는 "이 응답이 얼마나 좋은지"를 나타내고, Loss에서 이 값을 곱하여 좋은 응답의 토큰은 확률을 높이고, 나쁜 응답의 토큰은 확률을 낮추는 방향으로 학습합니다.

---

## Part 6: Output Length 문제와 해결책

### 1. 문제: Output Length 계속 증가

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
