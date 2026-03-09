---
layout: post
title: "Stanford CME295: Lecture 8 - LLM Evaluation"
date: 2026-03-08 11:20:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, evaluation, benchmark, llm-as-a-judge, bleu, rouge, meteor, cohen-kappa]
math: true
---

> **원본 강의**: [YouTube - CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=8)

---

## 강의 개요

이 강의는 LLM의 성능을 측정하고 평가하는 방법을 다룹니다. 측정할 수 없으면 개선할 수 없기에, 이 강의는 분기에서 가장 중요한 강의 중 하나입니다. Human Evaluation의 한계부터 LLM-as-a-Judge, 다양한 벤치마크까지 체계적으로 살펴봅니다.

**강의 목표:**

1. LLM 평가의 다양한 접근법 이해 (Human, Rule-based, LLM-as-a-Judge)
2. LLM-as-a-Judge의 장단점과 Best Practices 학습
3. Agent 평가 시 발생하는 다양한 Failure Mode 분석
4. 주요 벤치마크 종류와 특성 파악

**이전 강의 복습 (Lecture 7):**

- RAG (Retrieval Augmented Generation): 외부 지식 검색
- Tool Calling: LLM이 외부 도구 사용
- Agentic Workflows: ReAct 프레임워크 (Observe → Plan → Act)

---

## Part 1: LLM Evaluation 개요

### 1. Evaluation의 범위

"LLM Evaluation"은 여러 의미를 가질 수 있습니다:

| 평가 대상 | 예시 |
| --- | --- |
| **Output Quality** | 응답의 정확성, 유용성, 일관성 |
| **Alignment** | 톤, 스타일, 안전성 |
| **System Metrics** | 지연시간, 가격, 가용성 |

> 이 강의는 **Output Quality**에 집중합니다.

**평가의 어려움:**

- LLM은 **Free-form text** 생성 (자연어, 코드, 수학 추론 등)
- 범용적인 단일 메트릭 정의가 어려움

---

## Part 2: Human Evaluation

### 2. 이상적 시나리오와 한계

**이상적 시나리오:**

```
Prompt → LLM → Response → Human Rating → Score
```

모든 LLM 출력을 인간이 평가하면 가장 정확하지만, 실용적이지 않습니다.

#### 2.1 Human Evaluation의 문제점

| 문제 | 설명 |
| --- | --- |
| **주관성** | 동일 응답에 대해 평가자마다 다른 점수 |
| **비용** | 대규모 평가에 많은 인력과 시간 필요 |
| **속도** | 빠른 모델 개발 사이클에 부적합 |

#### 2.2 Inter-Rater Agreement (평가자 간 일치도)

**예시: "생일 선물 추천" 응답 평가**

```
응답: "테디베어는 거의 항상 좋은 선물입니다. 마음에 드는 것을 고르세요."
평가자 A: "유용함" ✓ (테디베어라는 구체적 제안)
평가자 B: "유용하지 않음" ✗ (어떤 종류? 곰? 코끼리? 기린?)
```

주관적인 평가 기준은 일관성 없는 결과를 초래합니다.

---

### 3. Agreement Metrics

#### 3.1 Simple Agreement Rate의 한계

$$\text{Agreement Rate} = \frac{\text{일치한 평가 수}}{\text{전체 평가 수}}$$

**문제:** 우연에 의한 일치를 고려하지 않음

**예시: 두 평가자가 랜덤하게 평가하는 경우**

평가자 A: P(Good) = 0.5, P(Bad) = 0.5
평가자 B: P(Good) = 0.5, P(Bad) = 0.5

$$\text{Agreement by Chance} = P_A \cdot P_B + (1-P_A)(1-P_B) = 0.25 + 0.25 = 0.5$$
→ 완전 랜덤으로도 50% 일치율 달성!

#### 3.2 Cohen's Kappa (κ)

우연에 의한 일치를 보정한 메트릭:

$$\kappa = \frac{P_{observed} - P_{chance}}{1 - P_{chance}}$$
| κ 값 | 해석 |
| --- | --- |
| κ = 1 | 완벽한 일치 |
| κ > 0 | 우연보다 나은 일치 |
| κ ≤ 0 | 우연 수준 또는 그 이하 |

**확장 메트릭:**

- **Fleiss' Kappa**: 3명 이상의 평가자
- **Krippendorff's Alpha**: 다양한 척도에 적용 가능

**실무 팁:** κ가 낮으면 평가자 간 **Alignment Session** 진행

<details>
<summary>Cohen's Kappa (κ) 상세 설명</summary>
**공식**

$$\kappa = \frac{P_{observed} - P_{chance}}{1 - P_{chance}}$$

- $P_{observed}$: 실제로 관찰된 일치율 (= Simple Agreement Rate)
- $P_{chance}$: 우연히 일치할 확률
- 분모 ($1 - P_{chance}$): 우연을 넘어서 일치할 수 있는 최대 여지

**직관적 해석**

이 공식은 "우연을 제외하고, 일치할 수 있는 최대 범위 중에서 실제로 얼마나 일치했는가?"를 묻는 것입니다.

**구체적인 계산 예시**

두 평가자가 100개의 LLM 응답을 "Good" 또는 "Bad"로 평가했다고 가정합니다.

|  | 평가자 B: Good | 평가자 B: Bad | 합계 |
| --- | --- | --- | --- |
| **평가자 A: Good** | 40 | 10 | 50 |
| **평가자 A: Bad** | 15 | 35 | 50 |
| **합계** | 55 | 45 | 100 |

**Step 1: $P_{observed}$ 계산**

두 평가자가 일치한 경우 = 40(둘 다 Good) + 35(둘 다 Bad) = 75

$$P_{observed} = \frac{75}{100} = 0.75$$

**Step 2: $P_{chance}$ 계산**

각 평가자가 독립적으로 평가한다고 가정했을 때 우연히 일치할 확률:

- P(A가 Good) = 50/100 = 0.5
- P(B가 Good) = 55/100 = 0.55
- P(A가 Bad) = 50/100 = 0.5
- P(B가 Bad) = 45/100 = 0.45

$$P_{chance} = P(A=Good) \times P(B=Good) + P(A=Bad) \times P(B=Bad)$$

$$= 0.5 \times 0.55 + 0.5 \times 0.45 = 0.275 + 0.225 = 0.5$$

**Step 3: κ 계산**

$$\kappa = \frac{0.75 - 0.5}{1 - 0.5} = \frac{0.25}{0.5} = 0.5$$

**κ 값의 해석**

| κ 값 | 해석 |
| --- | --- |
| κ = 1 | 완벽한 일치 (모든 평가자가 동일) |
| κ > 0.8 | 거의 완벽한 일치 (Almost Perfect) |
| 0.6 < κ ≤ 0.8 | 상당한 일치 (Substantial) |
| 0.4 < κ ≤ 0.6 | 중간 수준 일치 (Moderate) |
| 0.2 < κ ≤ 0.4 | 약한 일치 (Fair) |
| 0 < κ ≤ 0.2 | 미미한 일치 (Slight) |
| κ ≤ 0 | 우연 수준 또는 그 이하 (평가 기준이 맞지 않음) |

**Simple Agreement Rate vs Cohen's Kappa 비교**

위 예시에서:
- Simple Agreement Rate = 75% → "꽤 좋아 보임"
- Cohen's Kappa = 0.5 → "중간 수준, 개선 필요"

κ가 더 보수적이고 현실적인 평가를 제공합니다.

**실무 팁**

강의 자료에서 언급했듯이, κ가 낮으면 평가자 간 **Alignment Session**을 진행해야 합니다. 이는 평가 기준을 명확히 하고, 예시를 공유하며, 모호한 케이스에 대한 합의를 도출하는 과정입니다.

**확장 메트릭**

| 메트릭 | 사용 상황 |
| --- | --- |
| **Cohen's Kappa** | 2명의 평가자 |
| **Fleiss' Kappa** | 3명 이상의 평가자 |
| **Krippendorff's Alpha** | 다양한 척도(순서형, 연속형 등)에 적용 가능 |

</details>

---

## Part 3: Rule-Based Metrics

### 4. Reference-Based Evaluation

**접근법:** 인간이 작성한 **Reference Text**와 LLM 출력 비교
```
┌──────────────┐
│ Human writes │ → Reference Texts (고정)
│ references   │
└──────────────┘
        ↓
┌──────────────┐   ┌──────────────┐
│  LLM Output  │ vs│  Reference   │ → Score
└──────────────┘   └──────────────┘
```

**장점:** 매번 인간 평가 불필요
**단점:** 여전히 초기 Reference 작성에 인간 필요

---

### 5. 주요 Rule-Based Metrics

#### 5.1 METEOR (Metric for Evaluation of Translation with Explicit Ordering)

$$\text{METEOR} = F_{score} \times (1 - \text{Penalty})$$
**F-score:**

$$F_{score} = \frac{P \cdot R}{\alpha \cdot P + (1-\alpha) \cdot R}$$
- P (Precision): 예측의 unigram 중 reference와 일치하는 비율
- R (Recall): Reference의 unigram 중 예측과 일치하는 비율

**Penalty (순서 페널티):**

$$\text{Penalty} = \gamma \left(\frac{C}{M}\right)^\beta$$
- C: 연속된 청크(chunk) 수
- M: 일치한 unigram 수
- 순서가 같으면 C가 낮아 Penalty 감소

<details>
<summary>Unigram 상세 설명</summary>
**Unigram이란?**

Unigram은 텍스트를 구성하는 개별 단어 하나하나를 의미합니다. 이는 n-gram이라는 더 큰 개념의 일부입니다.

**N-gram 개념**

N-gram은 텍스트에서 연속된 n개의 단어(또는 문자)로 이루어진 시퀀스입니다.

| 이름 | n 값 | 의미 |
| --- | --- | --- |
| **Unigram** | 1 | 단어 1개 |
| **Bigram** | 2 | 연속된 단어 2개 |
| **Trigram** | 3 | 연속된 단어 3개 |
| **4-gram** | 4 | 연속된 단어 4개 |

**예시**

문장: "The cat sat on the mat"

**Unigrams (1-gram):**

```
"The", "cat", "sat", "on", "the", "mat"
→ 총 6개
```

**Bigrams (2-gram):**

```
"The cat", "cat sat", "sat on", "on the", "the mat"
→ 총 5개
```

**Trigrams (3-gram):**

```
"The cat sat", "cat sat on", "sat on the", "on the mat"
→ 총 4개
```

**METEOR에서 Unigram을 사용하는 이유**

METEOR는 unigram 단위로 매칭하기 때문에 개별 단어가 참조 텍스트에 존재하는지를 확인합니다. 이후 Chunk Penalty를 통해 순서를 별도로 평가합니다.

반면 BLEU는 unigram부터 4-gram까지 모두 사용해서, n-gram 자체에 순서 정보가 포함되어 있습니다. 예를 들어 "cat sat"이라는 bigram이 일치하려면 두 단어가 연속으로 나타나야 합니다.

</details>

<details>
<summary>METEOR 상세 설명</summary>
**METEOR (Metric for Evaluation of Translation with Explicit Ordering) 상세 설명**

METEOR는 기계 번역 품질을 평가하기 위해 개발된 메트릭으로, BLEU의 한계를 보완하기 위해 만들어졌습니다.

**핵심 아이디어**

METEOR는 두 가지를 동시에 고려합니다:

1. 단어 일치도 (F-score): 예측과 참조 텍스트가 얼마나 겹치는가
2. 순서 페널티 (Penalty): 단어 순서가 얼마나 뒤섞였는가

**전체 공식**

$$\text{METEOR} = F_{score} \times (1 - \text{Penalty})$$

**Part 1: F-score 계산**

$$F_{score} = \frac{P \times R}{\alpha \times P + (1-\alpha) \times R}$$

이것은 Precision과 Recall의 가중 조화평균입니다.

**Precision (P):** 예측한 단어 중 참조와 일치하는 비율

$$P = \frac{\text{일치한 unigram 수}}{\text{예측 텍스트의 전체 unigram 수}}$$

**Recall (R):** 참조 단어 중 예측에서 찾은 비율

$$R = \frac{\text{일치한 unigram 수}}{\text{참조 텍스트의 전체 unigram 수}}$$

**α (알파):** Precision과 Recall의 가중치 (기본값: 0.9)

- α가 크면 Precision 중시
- α가 작으면 Recall 중시
- METEOR는 **Recall을 더 중시** (α = 0.9이면 Recall에 더 높은 가중치)

**예시: F-score 계산**

참조 (Reference): "The cat sat on the mat"
예측 (Prediction): "The cat was sitting on a mat"

| 참조 단어 | 예측에 존재? |
| --- | --- |
| The | ✓ |
| cat | ✓ |
| sat | ✗ |
| on | ✓ |
| the | ✓ |
| mat | ✓ |

일치한 unigram: 5개 (The, cat, on, the, mat)

$$P = \frac{5}{7} = 0.714 \quad \text{(예측은 7단어)}$$

$$R = \frac{5}{6} = 0.833 \quad \text{(참조는 6단어)}$$

α = 0.9일 때:

$$F_{score} = \frac{0.714 \times 0.833}{0.9 \times 0.714 + 0.1 \times 0.833} = \frac{0.595}{0.726} = 0.820$$

**Part 2: 순서 페널티 (Penalty) 계산**

$$\text{Penalty} = \gamma \left(\frac{C}{M}\right)^\beta$$

- **C (Chunks):** 연속된 일치 청크의 수
- **M (Matches):** 일치한 unigram의 총 수
- **γ (감마):** 페널티 강도 (기본값: 0.5)
- **β (베타):** 페널티 곡선 조절 (기본값: 3)

**Chunk란?**

Chunk는 연속적으로 일치하는 단어들의 그룹입니다.

**예시 1: 순서가 잘 맞는 경우**

```
참조: The  cat  sat  on  the  mat
       ↓    ↓            ↓   ↓   ↓
예측: The  cat  was  on  the  mat
      |________|        |_________|
       Chunk 1           Chunk 2
```

- 일치한 단어 (M) = 5
- 청크 수 (C) = 2
- C/M = 2/5 = 0.4 → 낮은 페널티

**예시 2: 순서가 뒤섞인 경우**

```
참조: The  cat  sat  on  the  mat
       ↓    ↓            ↓   ↓   ↓
예측: mat  the  on   was cat  The
      |_|  |_|  |_|      |_|  |_|
      C1   C2   C3       C4   C5
```

- 일치한 단어 (M) = 5
- 청크 수 (C) = 5 (모두 흩어져 있음)
- C/M = 5/5 = 1.0 → 높은 페널티

**Penalty 계산 예시**

예시 1의 경우 (C=2, M=5, γ=0.5, β=3):

$$\text{Penalty} = 0.5 \times \left(\frac{2}{5}\right)^3 = 0.5 \times 0.064 = 0.032$$

예시 2의 경우 (C=5, M=5):

$$\text{Penalty} = 0.5 \times \left(\frac{5}{5}\right)^3 = 0.5 \times 1 = 0.5$$

**Part 3: 최종 METEOR 점수**

예시 1 (순서 좋음):

$$\text{METEOR} = 0.820 \times (1 - 0.032) = 0.820 \times 0.968 = 0.794$$

예시 2 (순서 나쁨):

$$\text{METEOR} = 0.820 \times (1 - 0.5) = 0.820 \times 0.5 = 0.410$$

같은 단어들이 일치해도 순서에 따라 점수가 크게 달라집니다.

**METEOR의 추가 기능: 유연한 매칭**

METEOR는 단순 exact match 외에도 여러 매칭 방식을 지원합니다:

| 매칭 유형 | 설명 | 예시 |
| --- | --- | --- |
| **Exact** | 정확히 같은 단어 | "cat" = "cat" |
| **Stem** | 어간이 같은 단어 | "running" = "ran" |
| **Synonym** | WordNet 동의어 | "big" = "large" |
| **Paraphrase** | 구문 패러프레이즈 | "at this time" = "now" |

이로 인해 BLEU보다 문체 변형에 더 관대합니다.

**BLEU vs METEOR 비교**

| 특성 | BLEU | METEOR |
| --- | --- | --- |
| 중심 메트릭 | Precision | Recall (+ Precision) |
| n-gram | 1~4gram 모두 사용 | Unigram 중심 |
| 순서 고려 | 간접적 (n-gram으로) | 직접적 (Chunk Penalty) |
| 유연한 매칭 | ✗ | ✓ (Stem, Synonym) |
| 인간 평가 상관관계 | 낮음 | 더 높음 |

**한계점**

강의 자료에서 언급된 Rule-Based Metrics의 공통적인 한계가 METEOR에도 적용됩니다:

1. 문체 변형에 완전히 대응하지 못함: 동의어 매칭이 있지만 완벽하지 않음
2. 하이퍼파라미터 의존: α, β, γ 값이 임의적
3. Reference 필요: 여전히 인간이 작성한 참조 텍스트가 필요

</details>

#### 5.2 BLEU (Bilingual Evaluation Understudy)

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$
- **Precision 중심**: 예측의 n-gram 중 reference와 일치하는 비율
- **BP (Brevity Penalty)**: 너무 짧은 번역 페널티

<details>
<summary>BLEU (Bilingual Evaluation Understudy) 상세 설명</summary>
**BLEU (Bilingual Evaluation Understudy) 상세 설명**

BLEU는 기계 번역 품질을 평가하기 위해 2002년 IBM에서 개발된 메트릭으로, 가장 널리 사용되는 자동 평가 지표 중 하나입니다.

**핵심 아이디어**

BLEU는 두 가지를 측정합니다:

1. **N-gram Precision:** 예측 텍스트의 n-gram이 참조 텍스트에 얼마나 존재하는가
2. **Brevity Penalty:** 너무 짧은 번역에 페널티 부여

**전체 공식**

$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

- **BP:** Brevity Penalty (간결성 페널티)
- **$p_n$:** n-gram Precision
- **$w_n$:** 각 n-gram의 가중치 (보통 균등하게 1/N)
- **N:** 고려하는 최대 n-gram (보통 4)

**Part 1: N-gram Precision 계산**

$$p_n = \frac{\text{예측의 n-gram 중 참조와 일치하는 수}}{\text{예측의 전체 n-gram 수}}$$

기본 예시

참조 (Reference): "The cat sat on the mat"
예측 (Prediction): "The cat on the mat"

**Unigram ($p_1$) 계산:**

| 예측의 unigram | 참조에 존재? |
| --- | --- |
| The | ✓ |
| cat | ✓ |
| on | ✓ |
| the | ✓ |
| mat | ✓ |

$$p_1 = \frac{5}{5} = 1.0$$

**Bigram ($p_2$) 계산:**

| 예측의 bigram | 참조에 존재? |
| --- | --- |
| The cat | ✓ |
| cat on | ✗ (참조는 "cat sat") |
| on the | ✓ |
| the mat | ✓ |

$$p_2 = \frac{3}{4} = 0.75$$

**Trigram ($p_3$) 계산:**

| 예측의 trigram | 참조에 존재? |
| --- | --- |
| The cat on | ✗ |
| cat on the | ✗ |
| on the mat | ✓ |

$$p_3 = \frac{1}{3} = 0.33$$

**4-gram ($p_4$) 계산:**

| 예측의 4-gram | 참조에 존재? |
| --- | --- |
| The cat on the | ✗ |
| cat on the mat | ✗ |

$$p_4 = \frac{0}{2} = 0$$

**Part 2: Modified Precision (Clipping)**

단순 Precision에는 문제가 있습니다.

문제 예시:

참조: "The cat sat on the mat"
예측: "the the the the the"

단순 계산:

$$p_1 = \frac{5}{5} = 1.0 \quad \text{(모든 "the"가 참조에 존재)}$$

이건 말이 안 됩니다! 그래서 **Clipping**을 적용합니다.

**Modified Precision:**

$$p_n = \frac{\sum_{\text{n-gram}} \min(\text{Count}_{\text{pred}}, \text{Count}_{\text{ref}})}{\sum_{\text{n-gram}} \text{Count}_{\text{pred}}}$$

참조에서 "the"는 2번만 등장하므로:

$$p_1 = \frac{\min(5, 2)}{5} = \frac{2}{5} = 0.4$$

**Part 3: Brevity Penalty (BP)**

BLEU는 Precision 기반이므로, 짧은 문장이 유리합니다.

극단적 예시:

참조: "The cat sat on the mat"
예측: "The"

$p_1 = 1.0$ (100% 정확!)

이를 방지하기 위해 **Brevity Penalty**를 적용합니다:

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{(1-r/c)} & \text{if } c \leq r \end{cases}$$

- **c:** 예측 텍스트의 길이
- **r:** 참조 텍스트의 길이

BP 계산 예시

예측 길이 c=1, 참조 길이 r=6:

$$BP = e^{(1-6/1)} = e^{-5} = 0.0067$$

→ 최종 BLEU 점수가 거의 0에 가까워집니다.

예측 길이 c=5, 참조 길이 r=6:

$$BP = e^{(1-6/5)} = e^{-0.2} = 0.819$$

→ 약간의 페널티만 적용됩니다.

예측 길이 c=7, 참조 길이 r=6:

$$BP = 1 \quad \text{(예측이 더 길면 페널티 없음)}$$

**Part 4: 최종 BLEU 점수 계산**

일반적으로 BLEU-4를 사용하며, 균등 가중치를 적용합니다:

$$\text{BLEU-4} = BP \times \exp\left(\frac{1}{4}\sum_{n=1}^{4} \log p_n\right)$$

이는 기하평균과 같습니다:

$$\text{BLEU-4} = BP \times (p_1 \times p_2 \times p_3 \times p_4)^{1/4}$$

\*$p_4 = 0$이면 전체 BLEU = 0이 되는 문제가 있어, 실제로는 **smoothing** 기법을 적용합니다.

**BLEU 점수 해석**

| BLEU 점수 | 해석 |
| --- | --- |
| < 0.1 | 거의 무의미한 번역 |
| 0.1 - 0.2 | 대략적인 의미 전달 |
| 0.2 - 0.3 | 이해 가능한 번역 |
| 0.3 - 0.4 | 좋은 품질의 번역 |
| 0.4 - 0.5 | 높은 품질의 번역 |
| > 0.5 | 매우 높은 품질 (인간 수준에 근접) |

**BLEU vs METEOR 비교**

| 특성 | BLEU | METEOR |
| --- | --- | --- |
| 중심 메트릭 | Precision | Recall + Precision |
| 짧은 번역 처리 | Brevity Penalty | F-score로 자연스럽게 처리 |
| 순서 고려 | N-gram에 내포됨 | Chunk Penalty로 명시적 |
| 유연한 매칭 | Exact match만 | Stem, Synonym 지원 |
| 계산 복잡도 | 낮음 | 높음 |

**BLEU의 한계**

1. **동의어 인식 불가**
   - 참조: "quick" → 예측: "fast" → 불일치로 처리
2. **문장 구조 변형에 취약**
   - 능동태/수동태 변환 시 n-gram이 깨짐
3. **의미보다 표면적 일치 측정**
   - 의미는 같지만 표현이 다르면 낮은 점수
4. **참조가 하나일 때 불리**
   - 여러 참조를 사용하면 개선되지만, 참조 작성 비용 증가

</details>

#### 5.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **Recall 중심**: 요약 품질 평가에 주로 사용
- ROUGE-N, ROUGE-L 등 다양한 변형

<details>
<summary>ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 상세 설명</summary>
**ROUGE 상세 설명**

ROUGE는 텍스트 요약 품질을 평가하기 위해 2004년에 개발된 메트릭입니다. 이름에서 알 수 있듯이 **Recall 중심**의 평가 방식을 사용합니다.

**핵심 아이디어**

ROUGE는 "참조 요약의 내용이 예측 요약에 얼마나 포함되어 있는가?"를 측정합니다.

**왜 Recall 중심인가?**

요약의 목적은 원본의 핵심 내용을 빠뜨리지 않는 것입니다. 따라서 참조 요약에 있는 내용이 예측에 얼마나 "커버"되는지가 중요합니다.

| 메트릭 | 질문 | 적합한 태스크 |
| --- | --- | --- |
| **BLEU (Precision)** | 예측한 것 중 맞는 게 얼마나? | 번역 (잘못된 단어 최소화) |
| **ROUGE (Recall)** | 참조 중 포함된 게 얼마나? | 요약 (핵심 누락 최소화) |

**ROUGE의 주요 변형**

| 변형 | 기반 | 특징 |
| --- | --- | --- |
| **ROUGE-N** | N-gram | 가장 기본적, N=1,2가 흔함 |
| **ROUGE-L** | LCS (최장 공통 부분수열) | 순서 고려, 연속일 필요 없음 |
| **ROUGE-W** | Weighted LCS | 연속 매칭에 가중치 부여 |
| **ROUGE-S** | Skip-bigram | 단어 사이에 gap 허용 |

**ROUGE-N**

공식:

$$\text{ROUGE-N} = \frac{\sum_{\text{n-gram} \in \text{Ref}} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{\text{n-gram} \in \text{Ref}} \text{Count}(\text{n-gram})}$$

핵심: 분모가 **참조(Reference)**의 n-gram 수입니다. (BLEU는 예측의 n-gram 수)

**ROUGE-1 (Unigram) 계산 예시**

참조 (Reference): "The cat sat on the mat"
예측 (Prediction): "The cat on the mat"

| 참조의 unigram | 예측에 존재? |
| --- | --- |
| The | ✓ |
| cat | ✓ |
| sat | ✗ |
| on | ✓ |
| the | ✓ |
| mat | ✓ |

$$\text{ROUGE-1} = \frac{5}{6} = 0.833$$

비교 - BLEU-1 (같은 예시):

$$\text{BLEU-1} = \frac{5}{5} = 1.0 \quad \text{(분모가 예측의 unigram 수)}$$

**ROUGE-2 (Bigram) 계산 예시**

참조의 bigrams: "The cat", "cat sat", "sat on", "on the", "the mat"
예측의 bigrams: "The cat", "cat on", "on the", "the mat"

| 참조의 bigram | 예측에 존재? |
| --- | --- |
| The cat | ✓ |
| cat sat | ✗ |
| sat on | ✗ |
| on the | ✓ |
| the mat | ✓ |

$$\text{ROUGE-2} = \frac{3}{5} = 0.6$$

**ROUGE-L (Longest Common Subsequence)**

LCS는 최장 공통 부분수열로, 두 시퀀스에서 순서를 유지하면서 공통으로 나타나는 가장 긴 부분수열입니다. 연속일 필요는 없습니다.

LCS 예시

참조: "The cat sat on the mat"
예측: "The cat was sitting on a mat"

```
참조: The  cat  sat  on  the  mat
       ↓    ↓         ↓   ↓    ↓
예측: The  cat  was  sitting  on  a  mat
```

LCS = "The", "cat", "on", "mat" → 길이 = 4

(참고: "the"도 포함하면 5가 될 수 있지만, 위치에 따라 다름)

**ROUGE-L 공식**

$$R_{LCS} = \frac{LCS(X, Y)}{m} \quad \text{(Recall)}$$

$$P_{LCS} = \frac{LCS(X, Y)}{n} \quad \text{(Precision)}$$

$$F_{LCS} = \frac{(1 + \beta^2) \times R_{LCS} \times P_{LCS}}{R_{LCS} + \beta^2 \times P_{LCS}}$$

- **X:** 참조 텍스트 (길이 m)
- **Y:** 예측 텍스트 (길이 n)
- **β:** Recall과 Precision의 상대적 중요도 (보통 β=1)

ROUGE-L 계산 예시

참조 (m=6): "The cat sat on the mat"
예측 (n=7): "The cat was sitting on a mat"
LCS 길이: 5 (The, cat, on, the, mat)

$$R_{LCS} = \frac{5}{6} = 0.833$$

$$P_{LCS} = \frac{5}{7} = 0.714$$

β=1일 때 (F1-score):

$$F_{LCS} = \frac{2 \times 0.833 \times 0.714}{0.833 + 0.714} = \frac{1.190}{1.547} = 0.769$$

**ROUGE-N vs ROUGE-L 비교**

| 특성 | ROUGE-N | ROUGE-L |
| --- | --- | --- |
| 연속성 요구 | N-gram은 연속 필수 | 연속 불필요 |
| 순서 고려 | 부분적 (n-gram 내에서만) | 전체적 |
| 유연성 | 낮음 | 높음 |
| 계산 복잡도 | O(n) | O(mn) |

예시로 보는 차이:

참조: "police killed the gunman"
예측: "police kill the gunman"

- ROUGE-2: "police kill" ≠ "police killed" → 매칭 안 됨
- ROUGE-L: "police", "the", "gunman" 모두 순서대로 매칭 → 더 관대

**ROUGE-W (Weighted LCS)**

문제점: 기본 LCS의 한계

참조: "ABCDEFGH"
예측 1: "ABCDEFGH" (완벽히 연속)
예측 2: "A_B_C_D_E_F_G_H" (모두 떨어져 있음)

기본 LCS는 둘 다 길이 8로 같은 점수를 줍니다. 하지만 예측 1이 더 좋은 요약입니다.

ROUGE-W 해결책: 연속적으로 매칭되는 부분에 더 높은 가중치를 부여합니다.

$$f(k) = k^2 \quad \text{(연속 매칭 k개에 대한 점수)}$$

예시:
- 연속 2개 매칭: $f(2) = 4$
- 분리된 1개 + 1개: $f(1) + f(1) = 2$

연속 매칭이 더 높은 점수를 받습니다.

**ROUGE-S (Skip-Bigram)**

Skip-bigram은 문장 내에서 순서를 유지하는 모든 단어 쌍입니다. 두 단어 사이에 다른 단어들이 있어도 됩니다.

예시

문장: "The cat sat"

일반 Bigram:

```
"The cat", "cat sat"
→ 2개
```

Skip-Bigram:

```
"The cat", "The sat", "cat sat"
→ 3개 (모든 순서쌍)
```

**ROUGE-S 공식**

$$R_{skip} = \frac{\text{매칭된 skip-bigram 수}}{C(m, 2)}$$

$$P_{skip} = \frac{\text{매칭된 skip-bigram 수}}{C(n, 2)}$$

- $C(m, 2) = \frac{m(m-1)}{2}$: 참조에서 가능한 모든 skip-bigram 수

**ROUGE-SU (Skip-Bigram + Unigram)**

ROUGE-S의 문제: 공통 단어가 1개뿐이면 skip-bigram이 0개 → 점수 0

해결: Unigram 매칭도 함께 고려하는 ROUGE-SU 사용

**전체 ROUGE 변형 비교**

| 변형 | 장점 | 단점 | 주 용도 |
| --- | --- | --- | --- |
| **ROUGE-1** | 단순, 빠름 | 순서 무시 | 기본 내용 커버리지 |
| **ROUGE-2** | 어느 정도 순서 고려 | 엄격함 | 구문 유사성 |
| **ROUGE-L** | 유연한 순서 매칭 | 연속성 무시 | 문장 수준 유사성 |
| **ROUGE-W** | 연속 매칭 보상 | 계산 복잡 | 정밀한 평가 |
| **ROUGE-S** | 장거리 의존성 포착 | 너무 관대할 수 있음 | 핵심어 포함 여부 |

**실무에서의 ROUGE 사용**

가장 많이 사용되는 조합:

```
ROUGE-1: 전반적인 내용 커버리지
ROUGE-2: 구문/표현 유사성
ROUGE-L: 문장 구조 유사성
```

보통 **F1-score**로 보고합니다:

```
ROUGE-1 F1: 0.45
ROUGE-2 F1: 0.21
ROUGE-L F1: 0.38
```

**ROUGE의 한계**

강의 자료에서 언급된 Rule-Based Metrics의 공통 한계가 적용됩니다:

1. **동의어/패러프레이즈 인식 불가**
   - "빠른" vs "신속한" → 다른 단어로 취급
2. **의미보다 표면적 일치 측정**
   - 단어가 같아도 의미가 다를 수 있음
   - 단어가 달라도 의미가 같을 수 있음
3. **참조 요약 필요**
   - 인간이 작성한 참조 요약이 필수
4. **인간 평가와 상관관계 한계**
   - 높은 ROUGE ≠ 반드시 좋은 요약

</details>

---

### 6. Rule-Based Metrics의 한계

#### 6.1 문체 변형 (Stylistic Variation) 불허

**동일한 의미, 다른 표현:**

```
1. "A plush teddy bear can comfort a child during bedtime."
2. "Soft stuffed bears often help kids feel safe as they fall asleep."
3. "Many youngsters rest more easily at night when they cuddle a toy companion."
```

→ 모두 같은 의미지만, n-gram 기반 메트릭은 낮은 점수

#### 6.2 기타 한계

| 한계 | 설명 |
| --- | --- |
| 낮은 상관관계 | 인간 평가와의 상관관계가 낮음 |
| 하이퍼파라미터 | α, β, γ 등 임의적 설정 |
| Reference 필요 | 초기 인간 작업 불가피 |

---

## Part 4: LLM-as-a-Judge

### 7. 개념과 구조

> "Pre-trained LLM은 방대한 인간 지식과 선호도를 학습했으므로, 평가자로 활용할 수 있다."

**LLM-as-a-Judge 구조:**

```
┌─────────────────────────────────────────────┐
│ INPUT                                        │
│ • Prompt (원래 질문)                          │
│ • Response (평가할 LLM 응답)                  │
│ • Criteria (평가 기준)                        │
└────────────────────┬────────────────────────┘
                     ↓
              ┌────────────────┐
              │ LLM-as-Judge   │
              └────────┬───────┘
                       ↓
┌─────────────────────────────────────────────┐
│ OUTPUT                                       │
│ • Rationale (평가 근거) ← 먼저 출력!          │
│ • Score (점수)                               │
└─────────────────────────────────────────────┘
```

**핵심 장점:**

1. **Reference 불필요**: 사전 인간 작업 없이 시작 가능
2. **해석 가능성**: 왜 그 점수인지 Rationale 제공

---

### 8. LLM-as-a-Judge 유형

#### 8.1 Pointwise (단일 응답 평가)

```
"이 응답이 좋은가요, 나쁜가요?"
→ Score: Good / Bad
```

#### 8.2 Pairwise (두 응답 비교)

```
"Response A와 Response B 중 어느 것이 더 좋은가요?"
→ Choice: A / B
```

**활용:** Preference Data 생성 (RLHF용)

---

### 9. LLM-as-a-Judge의 Bias

#### 9.1 Position Bias

**문제:** 먼저 언급된 응답을 선호하는 경향

```
"A와 B 중 어떤 것이 더 좋나요?" → A 선호
"B와 A 중 어떤 것이 더 좋나요?" → B 선호 (순서만 바뀜)
```

**해결책:**

- 순서를 바꿔 두 번 평가
- Majority Voting으로 최종 결정

#### 9.2 Verbosity Bias

**문제:** 더 긴 응답을 더 좋다고 평가하는 경향

**해결책:**

- 평가 지침에 "길이에 현혹되지 말 것" 명시
- In-context Learning 예시 제공
- 길이에 따른 페널티 적용

#### 9.3 Self-Enhancement Bias

**문제:** 자신이 생성한 응답을 선호하는 경향

**직관:** 모델이 생성한 응답 = 확률적으로 높은 시퀀스 = "좋은 답"으로 인식

**해결책:**

- 생성 모델과 **다른 모델**을 Judge로 사용
- 가능하면 **더 큰/강력한 모델**을 Judge로 사용

---

### 10. LLM-as-a-Judge Best Practices

| 원칙 | 설명 |
| --- | --- |
| **명확한 기준** | 모호하지 않은 구체적 평가 가이드라인 |
| **Binary Scale** | Pass/Fail이 Granular Scale보다 효과적 |
| **Rationale First** | 점수보다 근거를 먼저 출력하게 함 (Chain-of-Thought 효과) |
| **Bias 완화** | Position, Verbosity, Self-Enhancement 고려 |
| **Human Calibration** | 주기적으로 인간 평가와 상관관계 확인 |
| **Low Temperature** | 재현성을 위해 0.1~0.2 사용 |

---

### 11. Structured Output

**문제:** LLM 출력이 파싱 불가능한 형식일 수 있음

**해결책:** Constrained/Guided Decoding

```python
# OpenAI 스타일
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    response_format=EvaluationResult  # Structured Output
)

class EvaluationResult:
    rationale: str
    score: Literal["pass", "fail"]
```

→ 유효한 토큰만 샘플링하여 형식 보장

---

### 12. Factuality 평가

**문제:** 텍스트의 사실 정확성은 Binary가 아닌 Granular

**예시:**

```
"테디베어는 1920년대에 처음 만들어졌으며, 시어도어 루즈벨트 대통령이
사냥 여행에서 잡힌 곰을 자랑스럽게 쏘려 한 후 그의 이름을 따서 명명되었다."

오류 1: 1920년대 → 1900년대 (연도 오류)
오류 2: 자랑스럽게 쏘려 함 → 실제로는 거부함 (사실 오류)
```

**Factuality 평가 파이프라인:**
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ 원본 텍스트  │ → │  Fact 추출   │ → │  개별 검증   │
│             │   │ (LLM Call)  │   │ (RAG/검색)  │
└─────────────┘   └─────────────┘   └─────────────┘
                                           ↓
                                    ┌─────────────────┐
                                    │  가중 평균 점수   │
                                    │ Score = Σαᵢfᵢ   │
                                    └─────────────────┘
```

**Factuality Score:**

$$\text{Factuality} = \frac{\sum_{i} \alpha_i \cdot f_i}{\sum_{i} \alpha_i}$$
- $f_i$: i번째 fact의 정확성 (0 또는 1)
- $\alpha_i$: i번째 fact의 중요도 가중치

---

## Part 5: Agent Evaluation

### 13. Agentic Workflow의 Failure Modes

Agent는 여러 단계로 구성되어 각 단계에서 오류 발생 가능:

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Tool Prediction                                 │
│   User Query → Predict Tool + Arguments                  │
├─────────────────────────────────────────────────────────┤
│ Stage 2: Tool Execution                                  │
│   Call Function → Get Result                             │
├─────────────────────────────────────────────────────────┤
│ Stage 3: Response Synthesis                              │
│   Tool Result → Natural Language Response                │
└─────────────────────────────────────────────────────────┘
```

---

### 14. Tool Prediction Errors

#### 14.1 Tool을 사용하지 않음 (Punt)

**증상:** 도구가 있는데도 "할 수 없습니다" 응답

**원인과 해결:**

| 원인 | 해결책 |
| --- | --- |
| Tool Router가 해당 도구 미선택 (Recall 오류) | Tool Router 개선 |
| 도구가 Context에 있지만 LLM이 인식 못함 | SFT 재학습 또는 Prompt 개선 |

#### 14.2 Tool Hallucination

**증상:** 존재하지 않는 함수 호출

```python
# 정의된 함수: find_teddy_bear()
# LLM이 호출한 함수: find_bear() ← 존재하지 않음!
```

**원인과 해결:**

| 원인 | 해결책 |
| --- | --- |
| 모델 능력 부족 | 더 강력한 모델로 업그레이드 |
| API 이름/설명 불명확 | 함수명, 인자명, docstring 개선 |
| Top-level 지침 부족 | "반드시 제공된 함수만 사용" 명시 |

#### 14.3 Wrong Tool Selection

**증상:** 올바른 도구 대신 다른 도구 선택

**원인과 해결:**

| 원인 | 해결책 |
| --- | --- |
| Tool Router Recall 오류 | Router 개선 |
| API 간 범위 충돌 | 각 API의 사용 시나리오 명확화 |

#### 14.4 Wrong Arguments

**증상:** 올바른 도구지만 잘못된 인자

```python
# 사용자: "내 근처 테디베어 찾아줘"
find_teddy_bear(lat=0.0, lon=0.0)  # 남대서양 좌표!
```

**원인과 해결:**

| 원인 | 해결책 |
| --- | --- |
| Context에 위치 정보 없음 | 위치 정보를 Context에 포함 |
| 인자 의미 불명확 | API docstring 개선 |

---

### 15. Tool Execution Errors

#### 15.1 Wrong Output

**증상:** 함수 실행 시 에러 또는 잘못된 결과

**해결책:**

- 소프트웨어 버그 수정
- 에러 대신 **의미 있는 구조화된 출력** 반환

#### 15.2 No Output

**증상:** 함수가 아무것도 반환하지 않음

**문제:**

```python
def set_thermostat(temp):
    # 온도 설정...
    return None  # 모델은 성공 여부를 알 수 없음!
```

**해결책:** 항상 의미 있는 출력 반환

```python
def set_thermostat(temp):
    # 온도 설정...
    return {"status": "success", "new_temp": temp}
```

> **팁:** 결과가 없을 때도 `{}` (빈 JSON)이 `None`보다 나음!

---

### 16. Response Synthesis Errors

**증상:** 올바른 도구 결과를 잘못 해석

```
Tool Output: {"name": "Teddy", "distance": "1 mile"}
LLM Response: "테디베어를 찾지 못했습니다." ← 잘못된 해석!
```

**원인과 해결:**

| 원인 | 해결책 |
| --- | --- |
| 모델의 Grounding 능력 부족 | 더 강력한 모델 사용 |
| 출력이 너무 방대함 | 핵심 정보만 반환하도록 Tool 수정 |
| 출력 형식 불명확 | 구조화된 클래스로 반환 |

---

### 17. Agent Failure Mode 요약

```
┌─────────────────────────────────────────────────────────┐
│                   FAILURE CATEGORIES                     │
├─────────────────────────────────────────────────────────┤
│ Modeling Issues:                                         │
│   • 모델 추론/Grounding 능력                              │
│   • Context Window 내용의 관련성                          │
│   • Tool Router / API 모델링 (SFT, Prompting)            │
├─────────────────────────────────────────────────────────┤
│ Tool Implementation Issues:                              │
│   • 함수 로직 버그                                        │
│   • 출력 형식 문제                                        │
│   • 에러 핸들링                                           │
└─────────────────────────────────────────────────────────┘
```

**실무 팁:** 오류를 체계적으로 분류하고 그룹별로 해결

---

## Part 6: Benchmarks

### 18. Benchmark 카테고리

| 카테고리 | 측정 대상 | 대표 벤치마크 |
| --- | --- | --- |
| **Knowledge** | 사전 지식 보유량 | MMLU |
| **Reasoning** | 추론 능력 | AIME, PIQA |
| **Coding** | 코드 작성 능력 | SWE-Bench |
| **Safety** | 안전성 | HarmBench |
| **Agentic** | Agent 능력 | τ-Bench |

---

### 19. Knowledge Benchmark: MMLU

**MMLU = Massive Multitask Language Understanding**

- **57개 다양한 도메인**: 법률, 의학, 역사, 과학 등
- **형식**: 4지선다 객관식
- **측정 대상**: Pre-training 품질

**예시:**

```
Q: 환자의 혈압이 180/110이고 두통을 호소합니다.
   가장 가능성 높은 진단은?
A) 편두통  B) 고혈압성 뇌병증  C) 긴장성 두통  D) 군발두통
```

**평가 방식:** 정답 문자(A/B/C/D) 추출 후 Hard-coded 매칭

---

### 20. Reasoning Benchmarks

#### 20.1 AIME (American Invitational Mathematics Examination)

- **대상**: 수학 올림피아드 예선 문제
- **형식**: 문제 → 3자리 숫자 답
- **특징**: 순수 추론 능력 테스트

**예시:**

```
"정수 n에 대해 n² + n + 1이 완전제곱수가 되는
모든 양의 정수 n의 합을 구하시오."
답: ___ (3자리 숫자)
```

#### 20.2 PIQA (Physical Interaction QA)

- **대상**: 물리적 상식 추론
- **형식**: 2지선다
- **20,000개 예제**

**예시:**

```
Q: 카펫에서 잃어버린 물건을 어떻게 찾나요?
A) 단단한 밀봉재로 진공청소기 사용
B) 헤어넷으로 진공청소기 사용 ✓
```

---

### 21. Coding Benchmark: SWE-Bench

**SWE-Bench = Software Engineering Benchmark**

**구성 방식:**

1. 인기 Python 저장소에서 Pull Request 수집
2. 조건: Issue 해결 + Test 추가
3. LLM에게 Issue 해결 요청
4. 테스트 통과 여부로 평가
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Code Base  │ + │    Issue     │ → │  LLM Patch  │
│             │   │ Description │   │             │
└─────────────┘   └─────────────┘   └──────┬──────┘
                                           ↓
                                    ┌─────────────────┐
                                    │    Run Tests     │
                                    │  Before → After  │
                                    └─────────────────┘
```

**활용:** AI Coding Assistant 성능 평가

---

### 22. Safety Benchmark: HarmBench

**4가지 카테고리:**

| 카테고리 | 설명 |
| --- | --- |
| Standard | 일반적 유해 행동 |
| Copyright | 저작권 침해 콘텐츠 생성 |
| Contextual | 텍스트 맥락 기반 유해성 |
| Multimodal | 다중 모달리티 유해성 |

**평가 방식:**

- Classifier 기반 (Hard-coded 불가)
- 시도 자체가 성공으로 간주 (품질과 안전성 구분)

**주의:** 안전성 기준은 Provider마다 다름 → 직접 비교 어려움

---

### 23. Agentic Benchmark: τ-Bench (Tau-Bench)

**τ = Tool Agent Users**

**구성:**

- **도메인**: 항공사, 소매업
- **도구**: 각 도메인별 Tool Set
- **정책**: Agent가 할 수 있는/없는 행동
- **시뮬레이션**: 별도 LLM이 사용자 역할

**평가 메트릭: Pass^K (Pass-hat-K)**

$$\text{Pass}^K = \text{P}(\text{모든 K번 시도 성공})$$
vs Pass@K = P(적어도 1번 성공)

**왜 Pass^K인가?**

- 항공사/소매업 = 신뢰성과 일관성 필수
- 1번 성공보다 **매번 성공**이 중요

---

### 24. Benchmark 활용 시 주의사항

#### 24.1 Pareto Frontier

**성능 vs 비용 Trade-off:**

```
Performance
↑
│  ★ GPT-4
│    ★ Claude
│ ★ Gemini Flash
│
└──────────────→ Cost
```

→ 사용 목적에 따라 최적 모델 선택

#### 24.2 Data Contamination

**문제:** 모델이 벤치마크 데이터로 학습했을 가능성

**대응책:**

- Hash 값으로 중복 확인
- 특정 웹사이트 크롤링 차단
- 새로운 테스트 문제 지속 출제 (수학 등)

#### 24.3 Goodhart's Law

> "측정 지표가 목표가 되면, 좋은 지표가 되기를 멈춘다."

- 벤치마크 점수 ≠ 실제 성능
- **Chatbot Arena** 같은 실사용 평가와 병행 권장
- 궁극적으로 **직접 사용해보고 판단**

---

## 핵심 요약

### Evaluation 방법 비교

| 방법 | 장점 | 단점 |
| --- | --- | --- |
| **Human Evaluation** | 가장 정확 | 비용, 시간, 주관성 |
| **Rule-Based** | 자동화, 일관성 | 문체 변형 불허, Reference 필요 |
| **LLM-as-a-Judge** | 유연성, 해석 가능 | Bias 존재, Calibration 필요 |

### LLM-as-a-Judge Bias

```
Position Bias      → 순서 변경 후 Majority Voting
Verbosity Bias     → 가이드라인 명시, 길이 페널티
Self-Enhancement   → 다른 (더 큰) 모델을 Judge로 사용
```

### Agent Failure Modes

```
Tool Prediction:
  • 도구 미사용 (Punt)
  • 도구 Hallucination
  • 잘못된 도구 선택
  • 잘못된 인자

Tool Execution:
  • 잘못된 출력
  • 출력 없음

Response Synthesis:
  • 결과 잘못 해석
```

### 주요 Benchmarks

```
Knowledge  → MMLU (57개 도메인, 4지선다)
Reasoning  → AIME (수학), PIQA (상식)
Coding     → SWE-Bench (GitHub Issue 해결)
Safety     → HarmBench (유해 행동 방지)
Agentic    → τ-Bench (Tool 사용 Agent)
```

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| Inter-Rater Agreement | 평가자 간 일치도 |
| Cohen's Kappa | 우연 보정된 일치도 메트릭 |
| METEOR | 번역 평가 메트릭 (순서 고려) |
| BLEU | 번역 평가 메트릭 (Precision 중심) |
| ROUGE | 요약 평가 메트릭 (Recall 중심) |
| LLM-as-a-Judge | LLM을 평가자로 활용 |
| Position Bias | 순서에 따른 편향 |
| Verbosity Bias | 길이에 따른 편향 |
| Self-Enhancement Bias | 자기 생성물 선호 편향 |
| Structured Output | 형식이 보장된 출력 |
| MMLU | Massive Multitask Language Understanding |
| AIME | American Invitational Mathematics Examination |
| PIQA | Physical Interaction Question Answering |
| SWE-Bench | Software Engineering Benchmark |
| HarmBench | Harmful Behavior Benchmark |
| τ-Bench | Tool Agent Users Benchmark |
| Pass^K | K번 모두 성공할 확률 |
| Goodhart's Law | 지표가 목표가 되면 좋은 지표가 아님 |

---

## 추천 자료

1. **"Judging LLM-as-a-Judge"** - LLM 평가자의 편향 분석
2. **"MMLU: Measuring Massive Multitask Language Understanding"** - 지식 벤치마크
3. **"SWE-Bench: Can Language Models Resolve Real-World GitHub Issues?"** - 코딩 벤치마크
4. **"HarmBench: A Standardized Evaluation Framework for Automated Red Teaming"** - 안전성 벤치마크
5. **"τ-Bench: A Benchmark for Tool-Agent-User Interaction"** - Agent 벤치마크

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 8 정리*
