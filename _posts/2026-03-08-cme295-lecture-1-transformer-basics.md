---
layout: post
title: "Stanford CME295: Lecture 1 - Transformer 기초"
date: 2026-03-08 10:10:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, transformer, attention, nlp, word2vec, rnn, self-attention]
math: true
---

> **강의 출처**: Stanford CME295 - Transformers & LLMs (Autumn 2025)
>
> - **강사**: Afshine & Shervine Amidi
> - **원본 영상**: [YouTube](https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=4)

---

## 강의 개요

이 강의는 NLP의 기초부터 시작하여 Transformer 아키텍처까지 단계별로 설명합니다. 텍스트를 어떻게 처리하고, 표현하고, 모델링하는지에 대한 전체적인 흐름을 다룹니다.

**강의 목표:**

1. LLM을 구동하는 핵심 메커니즘 이해
2. LLM의 학습 방법과 적용 분야 파악

**선수 지식:**

* 기본적인 ML 지식 (모델 학습, 신경망)
* 선형대수 기초 (행렬 연산)

---

## Part 1: NLP 기초

### 1. NLP (Natural Language Processing) 개요

NLP는 텍스트를 다루는 분야로, 크게 세 가지 범주의 태스크로 분류됩니다.

#### 1.1 Classification (분류)

**입력:** 텍스트 → **출력:** 하나의 레이블

| 태스크 | 설명 | 예시 |
| --- | --- | --- |
| Sentiment Analysis | 감성 분석 | 영화 리뷰 → 긍정/부정/중립 |
| Intent Detection | 의도 탐지 | "알람 설정해줘" → create\_alarm |
| Language Detection | 언어 감지 | 텍스트 → 한국어/영어/프랑스어 |
| Topic Modeling | 주제 분류 | 뉴스 기사 → 정치/경제/스포츠 |

**평가 지표:**

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{Recall} = \frac{TP}{TP + FN} \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

<details>
<summary>혼동 행렬(Confusion Matrix) 상세</summary>

#### 혼동 행렬 (Confusion Matrix) 기초

분류 문제에서 모델의 예측 결과는 4가지 경우로 나뉩니다.

**기본 개념:**

| 약어 | 전체 이름 | 의미 |
| --- | --- | --- |
| **TP** | True Positive | 실제 양성 → 양성으로 맞게 예측 |
| **TN** | True Negative | 실제 음성 → 음성으로 맞게 예측 |
| **FP** | False Positive | 실제 음성 → 양성으로 틀리게 예측 |
| **FN** | False Negative | 실제 양성 → 음성으로 틀리게 예측 |

**혼동 행렬 시각화:**

```
            예측 (Predicted)
          Positive    Negative
        ┌───────────┬───────────┐
 Positive│    TP     │    FN     │
실제      │  (정탐)   │  (미탐)   │
(Actual) ├───────────┼───────────┤
 Negative│    FP     │    TN     │
         │  (오탐)   │  (정상)   │
         └───────────┴───────────┘
```

**구체적 예시: 스팸 메일 분류**

양성(Positive) = 스팸, 음성(Negative) = 정상 메일이라고 하면:

| 경우 | 실제 | 예측 | 결과 |
| --- | --- | --- | --- |
| **TP** | 스팸 메일 | "스팸입니다" | 스팸을 정확히 잡아냄 |
| **TN** | 정상 메일 | "정상입니다" | 정상 메일을 정확히 통과시킴 |
| **FP** | 정상 메일 | "스팸입니다" | 정상 메일이 스팸함으로 갔음 (오탐) |
| **FN** | 스팸 메일 | "정상입니다" | 스팸이 받은편지함으로 들어옴 (미탐) |

</details>

**왜 Accuracy만으로 부족한가?**

* 클래스 불균형 문제: 99% 양성인 데이터에서 모두 양성 예측하면 99% accuracy
* 이런 경우 Precision, Recall이 더 의미있는 지표

#### 1.2 Multi-Classification (다중 분류)

**입력:** 텍스트 → **출력:** 여러 레이블 (토큰별)

| 태스크 | 설명 | 예시 |
| --- | --- | --- |
| NER (Named Entity Recognition) | 개체명 인식 | "서울에서" → [서울: LOCATION] |
| POS Tagging | 품사 태깅 | "달리다" → 동사 |
| Dependency Parsing | 의존 구문 분석 | 단어 간 관계 파악 |

#### 1.3 Generation (생성)

**입력:** 텍스트 → **출력:** 텍스트 (가변 길이)

| 태스크 | 설명 |
| --- | --- |
| Machine Translation | 번역 (영어 → 한국어) |
| Question Answering | 질문에 대한 답변 생성 |
| Summarization | 문서 요약 |
| Text Generation | 시, 코드, 이야기 생성 |

**평가 지표:**

| 지표 | 설명 | 특징 |
| --- | --- | --- |
| **BLEU** | Bilingual Evaluation Understudy | 참조 텍스트 대비 n-gram 일치도, 높을수록 좋음 |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | 요약 품질 평가, 높을수록 좋음 |
| **Perplexity** | 모델의 "놀람" 정도 | 낮을수록 좋음 |

**Perplexity 직관:**

$$\text{Perplexity} = 2^{-\frac{1}{N}\sum_{i=1}^{N}\log_2 P(w_i)}$$

* 모델이 다음 토큰을 얼마나 확신하는지를 측정
* 낮을수록 모델이 텍스트를 잘 이해하고 있음

---

## Part 2: 텍스트 표현 (Text Representation)

### 2. Tokenization (토큰화)

모델은 숫자만 이해하므로, 텍스트를 처리 가능한 단위로 분할해야 합니다.

#### 2.1 Word-Level Tokenization

```
"A cute teddy bear" → ["A", "cute", "teddy", "bear"]
```

**장점:** 직관적이고 간단

**단점:**
* 어근 활용 불가: "bear"와 "bears"가 완전히 다른 토큰
* OOV (Out of Vocabulary) 문제: 학습에서 본 적 없는 단어 처리 불가
* 어휘 크기가 매우 커짐

#### 2.2 Subword-Level Tokenization (현재 표준)

```
"playing" → ["play", "##ing"]
"bears"   → ["bear", "##s"]
```

**대표 알고리즘:**
* **BPE (Byte Pair Encoding)**: GPT 시리즈
* **WordPiece**: BERT
* **SentencePiece**: T5, LLaMA

**장점:** 어근 공유로 효율적 표현, OOV 문제 완화, 적절한 어휘 크기 (수만 개)

#### 2.3 Character-Level Tokenization

```
"bear" → ["b", "e", "a", "r"]
```

**장점:** 오타에 강건, OOV 없음, 매우 작은 어휘 크기

**단점:** 시퀀스가 매우 길어짐, 글자 자체의 의미 표현 어려움

#### 2.4 Tokenization 비교 요약

| 방법 | 어휘 크기 | 시퀀스 길이 | OOV 위험 | 어근 활용 |
| --- | --- | --- | --- | --- |
| Word | 매우 큼 | 짧음 | 높음 | X |
| Subword | 적당 (30K~100K) | 적당 | 낮음 | O |
| Character | 매우 작음 | 매우 김 | 없음 | X |

---

### 3. Word Representation (단어 표현)

#### 3.1 One-Hot Encoding

```
어휘: [soft, teddy bear, book]
soft       → [1, 0, 0]
teddy bear → [0, 1, 0]
book       → [0, 0, 1]
```

**문제점:**
* 모든 벡터가 서로 직교 (cosine similarity = 0)
* 의미적 유사성 표현 불가
* "soft"와 "teddy bear"가 관련 있다는 것을 알 수 없음

**Cosine Similarity:**

$$\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

#### 3.2 Word2Vec (2013)

**핵심 아이디어:** 문맥을 통해 단어의 의미를 학습

> "You shall know a word by the company it keeps." — J.R. Firth

**두 가지 학습 방식:**

**1. CBOW (Continuous Bag of Words):**
* 주변 단어들 → 중심 단어 예측
* 예: ["a", "cute", "bear", "is"] → "teddy"

**2. Skip-gram:**
* 중심 단어 → 주변 단어들 예측
* 예: "teddy" → ["a", "cute", "bear", "is"]

**아키텍처:**

```
Input (V) → Hidden (D) → Output (V)
  |              |              |
One-hot     Embedding      Softmax
```

* **V**: 어휘 크기 (수만~수십만)
* **D**: 임베딩 차원 (보통 100~768)

**결과:**
* 의미적으로 유사한 단어는 가까운 벡터
* 유명한 예: king - man + woman ≈ queen
* Paris - France + Germany ≈ Berlin

**한계:**

1. **문맥 무시:** "bank"가 은행인지 강둑인지 구분 불가
2. **단어 순서 무시:** "dog bites man" = "man bites dog"
3. **고정 임베딩:** 같은 단어는 항상 같은 벡터

---

### 4. Sentence Representation (문장 표현)

#### 4.1 RNN (Recurrent Neural Network)

**핵심 아이디어:** 순차적으로 처리하며 hidden state 유지

**수식:**

$$h_t = f(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

**장점:** 단어 순서 고려, 가변 길이 입력 처리, 문맥 인코딩

**단점:**

1. **Vanishing Gradient Problem:** 역전파 시 그래디언트가 시간에 따라 지수적 감소

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

2. **Long-range Dependencies:** 긴 문장에서 앞부분 정보 "잊음"
3. **Sequential Processing:** 병렬화 불가 → 느린 학습

#### 4.2 LSTM (Long Short-Term Memory)

**RNN의 vanishing gradient 문제 완화 시도**

* **Cell State (C):** 장기 기억 저장
* **Gates:** 정보 흐름 제어 (Forget, Input, Output)

**개선점:** 더 긴 의존성 학습 가능, 그래디언트 흐름 개선

**여전한 한계:** 순차 처리로 인한 느린 속도, 완벽한 장거리 의존성 해결 못함

---

## Part 3: Attention과 Transformer

### 5. Attention Mechanism (2014)

> "직접 연결로 장거리 의존성 해결"

**기존 RNN 기반 seq2seq:**

```
[Encoder RNN] → [Single Context Vector] → [Decoder RNN]
```

모든 정보가 하나의 벡터에 압축 → 병목

**Attention 적용:**

```
[Encoder RNN] → [All Hidden States] → Attention → [Decoder RNN]
                       ↑                              |
                       └──────────────────────────────┘
                                직접 연결!
```

---

### 6. Self-Attention (2017)

**"Attention is All You Need" 논문의 핵심**

RNN 없이 Attention만으로 시퀀스 처리

#### 6.1 Query, Key, Value (Q, K, V)

| 요소 | 역할 | 비유 |
| --- | --- | --- |
| **Query (Q)** | "무엇을 찾고 있는가?" | 검색어 |
| **Key (K)** | "이것은 어떤 정보인가?" | 문서 제목/태그 |
| **Value (V)** | "실제 정보 내용" | 문서 본문 |

#### 6.2 Self-Attention 계산

**1단계: Q, K, V 생성**

$$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$

**2단계: Attention Score 계산**

$$\text{Score} = Q \cdot K^T$$

**3단계: Scaling**

$$\text{Scaled Score} = \frac{Q \cdot K^T}{\sqrt{d_k}}$$

**왜 $\sqrt{d_k}$로 나누는가?**
* Dot product는 차원이 커질수록 값이 커짐
* 큰 값은 softmax를 극단적으로 만듦 (한 곳에 집중)
* Scaling으로 안정적인 학습

<details>
<summary>스케일링이 필요한 이유 상세 (수식 포함)</summary>

#### 1. 문제 설정

Query 벡터 $q$와 Key 벡터 $k$가 있고, 각각 $d_k$ 차원이라고 가정합니다.

$$q = (q_1, q_2, \ldots, q_{d_k}), \quad k = (k_1, k_2, \ldots, k_{d_k})$$

가정: 각 원소가 독립이고 평균 0, 분산 1인 분포를 따른다고 가정합니다.

$$E[q_i] = 0, \quad \text{Var}(q_i) = 1$$

$$E[k_i] = 0, \quad \text{Var}(k_i) = 1$$

#### 2. 내적(Dot Product)의 계산

Q와 K의 내적은 다음과 같습니다:

$$q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i$$

#### 3. 내적의 기댓값 (평균)

$$E[q \cdot k] = E\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} E[q_i k_i]$$

$q_i$와 $k_i$가 독립이므로:

$$E[q_i k_i] = E[q_i] \cdot E[k_i] = 0 \cdot 0 = 0$$

따라서:

$$\boxed{E[q \cdot k] = 0}$$

#### 4. 내적의 분산 (핵심!)

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right)$$

각 $q_i k_i$가 독립이므로 분산의 합으로 분리 가능:

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)$$

이제 $\text{Var}(q_i k_i)$를 계산합니다. 분산의 정의를 사용하면:

$$\text{Var}(q_i k_i) = E[(q_i k_i)^2] - (E[q_i k_i])^2$$

$E[q_i k_i] = 0$이므로:

$$\text{Var}(q_i k_i) = E[q_i^2 k_i^2]$$

$q_i$와 $k_i$가 독립이므로:

$$E[q_i^2 k_i^2] = E[q_i^2] \cdot E[k_i^2]$$

여기서 $E[q_i^2] = \text{Var}(q_i) + (E[q_i])^2 = 1 + 0 = 1$

마찬가지로 $E[k_i^2] = 1$

따라서:

$$\text{Var}(q_i k_i) = 1 \cdot 1 = 1$$

최종적으로:

$$\boxed{\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} 1 = d_k}$$

#### 5. 결론: 분산이 $d_k$에 비례

| 항목 | 값 |
| --- | --- |
| 내적의 평균 | $E[q \cdot k] = 0$ |
| 내적의 분산 | $\text{Var}(q \cdot k) = d_k$ |
| 내적의 표준편차 | $\sigma = \sqrt{d_k}$ |

$d_k$가 커질수록 내적 값의 분산이 선형적으로 증가합니다.

#### 6. Softmax에 미치는 영향

Attention score에 softmax를 적용합니다:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

문제: $d_k$가 크면 내적 값들의 분산이 커지고, 일부 값이 매우 크거나 작아집니다.

예시 ($d_k = 64$일 때):

```
내적 값들: [-15.2, -8.1, 25.3, -12.0, ...]
                       ↑
                이 값이 매우 큼
```

Softmax 결과:

$$\text{softmax}([\ldots, 25.3, \ldots]) \approx [0, 0, 0.999, 0, \ldots]$$

결과:
- Softmax 출력이 one-hot에 가깝게 됨
- Gradient가 거의 0이 됨 → Vanishing Gradient

#### 7. 스케일링의 효과

$\sqrt{d_k}$로 나누면:

$$\text{score} = \frac{q \cdot k}{\sqrt{d_k}}$$

스케일링 후 분산:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$

| 상태 | 분산 | Softmax 동작 |
| --- | --- | --- |
| 스케일링 전 | $d_k$ | 극단적 (gradient 소실) |
| 스케일링 후 | $1$ | 안정적 (적절한 gradient) |

#### 8. 시각적 이해

$d_k = 64$ (스케일링 전)

내적 분포: 분산 = 64, 표준편차 ≈ 8

```
확률밀도
        ┌──────────────────┐
        │       ▄          │
        │      ▄█▄         │
        │    ▄▄███▄▄       │
        │ ▄▄████████▄▄    │
        └──────────────────┘
   -24  -16  -8   0   8   16  24
  → 값이 넓게 퍼져 있음
  → Softmax가 극단적으로 동작
```

$d_k = 64$ ($\sqrt{64} = 8$로 스케일링 후)

스케일된 분포: 분산 = 1, 표준편차 = 1

```
확률밀도
        ┌──────────────────┐
        │       ▄▄         │
        │     ▄████▄       │
        │   ▄████████▄     │
        │ ▄████████████▄   │
        └──────────────────┘
        -3   -1   0   1    3
  → 값이 적절한 범위에 집중
  → Softmax가 안정적으로 동작
```

#### 9. 최종 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$\sqrt{d_k}$로 나누는 이유: 내적의 분산을 $d_k$에서 1로 정규화하여 softmax가 안정적으로 동작하도록 함

</details>

<details>
<summary>분산의 스케일링 성질</summary>

#### 분산의 스케일링 성질

확률변수 $X$에 상수 $a$를 곱하면:

$$\text{Var}(aX) = a^2 \cdot \text{Var}(X)$$

상수가 제곱되어 나옵니다.

**왜 제곱인가? (유도)**

분산의 정의부터 시작합니다:

$$\text{Var}(X) = E\left[(X - E[X])^2\right]$$

이제 $aX$의 분산을 계산합니다:

$$\text{Var}(aX) = E\left[(aX - E[aX])^2\right]$$

기댓값의 선형성에 의해 $E[aX] = aE[X]$이므로:

$$\text{Var}(aX) = E\left[(aX - aE[X])^2\right]$$

$a$를 묶어내면:

$$\text{Var}(aX) = E\left[a^2(X - E[X])^2\right]$$

상수 $a^2$는 기댓값 밖으로 나올 수 있으므로:

$$\text{Var}(aX) = a^2 \cdot E\left[(X - E[X])^2\right] = a^2 \cdot \text{Var}(X)$$

**우리 문제에 적용**

$a = \frac{1}{\sqrt{d_k}}$로 놓으면:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \text{Var}\left(\frac{1}{\sqrt{d_k}} \cdot (q \cdot k)\right)$$

$$= \left(\frac{1}{\sqrt{d_k}}\right)^2 \cdot \text{Var}(q \cdot k)$$

$$= \frac{1}{d_k} \cdot d_k = 1$$

**직관적 이해**

| 변환 | 평균에 미치는 영향 | 분산에 미치는 영향 |
| --- | --- | --- |
| $X + c$ | $c$만큼 이동 | 변화 없음 |
| $aX$ | $a$배 | $a^2$배 |

분산은 "퍼진 정도의 제곱"을 측정하기 때문에, 스케일링하면 제곱으로 반영됩니다.

예시: $X$의 값들이 $[-2, -1, 0, 1, 2]$라면

$2X$의 값들은 $[-4, -2, 0, 2, 4]$

- 퍼진 폭이 2배
- 분산은 4배 ($2^2$배)

</details>

**4단계: Softmax로 가중치 변환**

$$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

**5단계: Value와 가중합**

$$\text{Output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$

#### 6.3 Self-Attention 전체 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**행렬 차원 분석:**

* Q: $(n \times d_k)$
* K: $(n \times d_k)$, $K^T$: $(d_k \times n)$
* V: $(n \times d_v)$
* $QK^T$: $(n \times n)$ — Attention map
* Output: $(n \times d_v)$

---

### 7. Multi-Head Attention

**왜 여러 Head가 필요한가?**
* 다양한 관점에서 관계 학습
* 예: 하나의 head는 문법적 관계, 다른 head는 의미적 관계

**수식:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**파라미터:**
* h: head 수 (보통 8 또는 12)
* 각 head의 차원: $d_{model} / h$

**비유 (Computer Vision):** CNN의 multiple filters처럼, 각 head가 다른 "필터"로 관계를 학습

<details>
<summary>Multi-Head Attention의 Weight 행렬 상세</summary>

#### Multi-Head Attention의 Weight 행렬들

**1. 어떤 행렬들이 있는가?**

```
           Multi-Head Attention
                    |
Input X ──→ W_1^Q ──→ Q_1 ─┐
         ├→ W_1^K ──→ K_1 ─┼→ Head_1 ─┐
         └→ W_1^V ──→ V_1 ─┘          │
         ┌→ W_2^Q ──→ Q_2 ─┐          │
         ├→ W_2^K ──→ K_2 ─┼→ Head_2 ─┼→ Concat → W^O → Output
         └→ W_2^V ──→ V_2 ─┘          │
         ┌→ W_h^Q ──→ Q_h ─┐          │
         ├→ W_h^K ──→ K_h ─┼→ Head_h ─┘
         └→ W_h^V ──→ V_h ─┘
```

**총 파라미터:**
- 각 head마다: $W_i^Q, W_i^K, W_i^V$ (3개)
- h개의 head → 3h개의 projection 행렬
- 마지막 output projection: $W^O$ (1개)

**2. 차원 분석**

일반적인 Transformer 설정 (예: BERT-base):
- `d_model` = 512 (입력/출력 차원)
- `h` = 8 (head 수)
- `d_k` = `d_v` = d_model / h = 64 (각 head의 차원)

| 행렬 | 차원 | 설명 |
| --- | --- | --- |
| $W_i^Q$ | (d_model × d_k) = (512 × 64) | Query projection |
| $W_i^K$ | (d_model × d_k) = (512 × 64) | Key projection |
| $W_i^V$ | (d_model × d_v) = (512 × 64) | Value projection |
| $W^O$ | (h·d_v × d_model) = (512 × 512) | Output projection |

**3. 초기화 방법**

이 행렬들은 랜덤하게 초기화됩니다.

**Xavier/Glorot Initialization (가장 일반적)**

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

또는 정규분포 버전:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

**왜 이렇게 초기화하는가?**
- 각 layer를 통과할 때 분산이 유지되도록
- 너무 크면 → gradient exploding
- 너무 작으면 → gradient vanishing

**PyTorch 실제 구현 예시:**

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 64

        # 여기서 Linear layer 생성 시 자동으로 초기화됨!
        self.W_Q = nn.Linear(d_model, d_model)  # 내부적으로 h개 head 포함
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # 수동 초기화 (선택적)
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform initialization
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
```

**원본 Transformer 논문의 초기화:**

```python
# "Attention Is All You Need" 논문 스타일
def init_weights(module):
    if isinstance(module, nn.Linear):
        # 평균 0, 표준편차 0.02의 정규분포
        nn.init.normal_(module.weight, mean=0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

**4. 학습 과정**

```
           학습 흐름

 1. 초기화: W_i^Q, W_i^K, W_i^V ← Random (Xavier)
                    ↓
 2. Forward Pass: Input → Attention → Output
                    ↓
 3. Loss 계산: Cross-Entropy(Output, Target)
                    ↓
 4. Backward Pass: ∂L/∂W_i^Q, ∂L/∂W_i^K, ... 계산
                    ↓
 5. Update: W ← W - η · ∂L/∂W  (Gradient Descent)
                    ↓
 6. 반복 → 최적의 W 값 학습
```

**핵심 포인트:**
- 처음엔 랜덤 → 학습하면서 의미있는 값으로 수렴
- 각 head의 W가 다르게 학습됨 → 다양한 관점의 attention 학습

**5. 각 Head가 다른 것을 학습하는 이유**

초기 상태 (Random):
```
Head 1: W_1^Q = [[0.12, -0.34, ...], ...]  ← 랜덤 A
Head 2: W_2^Q = [[-0.21, 0.45, ...], ...]  ← 랜덤 B (다름!)
```

학습 후:
```
Head 1: 문법적 관계에 집중 (주어-동사)
Head 2: 의미적 관계에 집중 (동의어, 관련어)
Head 3: 위치 관계에 집중 (인접 단어)
```

**왜 다르게 학습되는가?**
1. 초기값이 다름
2. Gradient가 각 head에 다르게 전파됨
3. 다양한 패턴을 학습하는 것이 loss를 더 낮춤

**6. 실제 학습된 Attention 패턴 예시**

"The cat sat on the mat"

```
Head 1 (문법적):     Head 2 (의미적):     Head 3 (위치적):
 cat → sat           cat → mat           cat → The, sat
 (주어→동사)         (고양이→매트)        (인접 단어들)
```

</details>

---

### 8. Transformer Architecture

#### 8.1 전체 구조

```
┌─────────────────┐     ┌─────────────────┐
│     ENCODER     │     │     DECODER     │
│                 │     │                 │
│ ┌───────────┐   │     │ ┌───────────┐   │
│ │ Multi-Head│   │     │ │  Masked   │   │
│ │ Attention │   │     │ │ Multi-Head│   │
│ └─────┬─────┘   │     │ │ Attention │   │
│ ┌─────▼─────┐   │     │ └─────┬─────┘   │
│ │ Add & Norm│   │     │ ┌─────▼─────┐   │
│ └─────┬─────┘   │     │ │ Add & Norm│   │
│ ┌─────▼─────┐   │     │ └─────┬─────┘   │
│ │    FFN    │   │     │ ┌─────▼─────┐   │
│ └─────┬─────┘   │─K,V─▶│Cross-Attn │   │
│ ┌─────▼─────┐   │     │ └─────┬─────┘   │
│ │ Add & Norm│   │     │ ┌─────▼─────┐   │
│ └───────────┘   │     │ │    FFN    │   │
│                 │     │ └─────┬─────┘   │
│    × N layers   │     │ ┌─────▼─────┐   │
└─────────────────┘     │ │ Add & Norm│   │
                        │ └───────────┘   │
                        │    × N layers   │
                        └────────┬────────┘
                        ┌────────▼────────┐
                        │ Linear + Softmax│
                        └─────────────────┘
```

#### 8.2 Encoder

**역할:** 입력 시퀀스의 풍부한 표현 생성

**구성요소:**

1. **Input Embedding:** 토큰 → 벡터
2. **Positional Encoding:** 위치 정보 추가
3. **Multi-Head Self-Attention:** 토큰 간 관계 학습
4. **Feed-Forward Network (FFN):** 비선형 변환
5. **Add & Norm:** 잔차 연결 + Layer Normalization

**Self-Attention in Encoder:**
* 모든 토큰이 모든 토큰을 참조 (양방향)

<details>
<summary>Encoder 단계별 동작 상세</summary>

#### 1. Input Embedding

역할: 입력 토큰을 고차원 벡터로 변환

$$\text{Embedding}: \text{token\_id} \rightarrow \mathbb{R}^{d_{model}}$$

동작:
- 각 토큰(단어/서브워드)을 고유한 정수 ID로 변환
- 학습 가능한 임베딩 테이블에서 해당 ID의 벡터를 조회
- 원 논문에서 $d_{model} = 512$

예시: "I love AI"
```
토큰 ID: [15, 892, 3421]
임베딩 후: [[0.2, -0.1, ...], [0.5, 0.3, ...], [-0.1, 0.4, ...]]
            ← 각각 512차원 벡터 →
```

#### 2. Positional Encoding (+)

역할: 토큰의 순서 정보를 추가

문제: Self-Attention은 순서를 모름 (permutation invariant)

해결: 위치 정보를 임베딩에 더함

$$\text{Input} = \text{Embedding} + \text{PositionalEncoding}$$

Sinusoidal 공식 (원 논문):

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```
위치 0: [sin(0), cos(0), sin(0), cos(0), ...]
위치 1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/512)), ...]
...
```

#### 3. Multi-Head Self-Attention

역할: 시퀀스 내 모든 토큰 간의 관계를 학습

동작 과정:

**(1) Q, K, V 생성**

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

- $X$: 입력 (이전 레이어 출력)
- $W^Q, W^K, W^V$: 학습 가능한 가중치 행렬

**(2) Attention Score 계산**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**(3) Multi-Head로 확장**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

원 논문 설정:
- $h = 8$ (head 수)
- $d_k = d_v = 512/8 = 64$ (head당 차원)

Self-Attention의 의미: 각 토큰이 다른 모든 토큰을 "참조"하여 문맥 정보 획득

```
"The cat sat on the mat"

"cat"의 attention:
  → "The"에 약간
  → "sat"에 많이 (동사와 주어 관계)
  → "mat"에 약간
```

#### 4. Add & Norm (첫 번째)

역할: 학습 안정화 및 gradient 흐름 개선

동작:

$$\text{Output} = \text{LayerNorm}(X + \text{MultiHeadAttention}(X))$$

**(1) Add (Residual Connection)**

$$X + \text{Sublayer}(X)$$

- 입력을 출력에 직접 더함
- Gradient가 직접 흐를 수 있는 경로 제공
- 깊은 네트워크 학습 가능하게 함

**(2) LayerNorm**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

- $\mu$: 해당 레이어의 평균
- $\sigma$: 해당 레이어의 표준편차
- $\gamma, \beta$: 학습 가능한 파라미터
- $\epsilon$: 수치 안정성을 위한 작은 값

```
정규화 전: [100, 200, 50, 150]
정규화 후: [-0.5, 1.5, -1.5, 0.5] (대략적 예시)
```

#### 5. Feed Forward Network (FFN)

역할: 비선형 변환 및 채널 믹싱

구조: 2개의 Linear 층 + 활성화 함수

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

또는 현대 모델에서:

$$\text{FFN}(x) = \text{GELU}(xW_1)W_2$$

차원 변화:

$$d_{model} \rightarrow d_{ff} \rightarrow d_{model}$$

원 논문:

$$512 \rightarrow 2048 \rightarrow 512$$

(4배 확장 후 다시 축소)

특징:
- Position-wise: 각 위치에 독립적으로 동일한 FFN 적용
- Attention이 "토큰 간 관계"라면, FFN은 "각 토큰의 표현 변환"

#### 6. Add & Norm (두 번째)

역할: FFN 후에도 동일하게 잔차 연결 + 정규화

$$\text{Output} = \text{LayerNorm}(X + \text{FFN}(X))$$

#### 7. ×N (N번 반복)

역할: 위 블록을 N번 쌓아 깊은 표현 학습

원 논문: $N = 6$

```
Encoder Layer 1: 기본적인 패턴 학습
Encoder Layer 2: 더 복잡한 패턴
...
Encoder Layer 6: 고수준 추상화된 표현
```

각 레이어를 거치며:
- 점점 더 추상적이고 문맥화된 표현 생성
- 최종 출력은 Decoder의 Cross-Attention에 전달됨

**전체 흐름 요약:**

```
입력 텍스트: "I love AI"
         ↓
[Input Embedding] → 토큰을 512차원 벡터로
         ↓
[Positional Encoding] → 위치 정보 추가 (+)
         ↓
  ┌─────────────────────────┐
  │ [Multi-Head Self-Attention] │
  │            ↓                │
  │      [Add & Norm]          │
  │            ↓                │
  │  [Feed Forward Network]    │
  │            ↓                │
  │      [Add & Norm]          │
  └─────────────────────────┘
                    ×6
         ↓
Encoder 출력 → Decoder의 Cross-Attention으로 전달
```

</details>

#### 8.3 Decoder

**역할:** 출력 시퀀스 생성 (Auto-regressive)

**구성요소:**

1. **Output Embedding:** 이전 출력 토큰 임베딩
2. **Positional Encoding**
3. **Masked Multi-Head Self-Attention:** 이전 토큰만 참조
4. **Cross-Attention:** Encoder 출력 참조
5. **FFN**
6. **Linear + Softmax:** 다음 토큰 확률 분포

**Masked Self-Attention:**
* 미래 토큰을 볼 수 없음 (인과적 마스킹)

**Cross-Attention:**
* Query: Decoder에서 옴
* Key, Value: Encoder에서 옴
* "번역 시 원문의 어떤 부분을 참조할까?"

<details>
<summary>Cross-Attention 상세</summary>

#### Cross Attention이란?

핵심 차이점: Self-Attention은 Q, K, V가 같은 소스에서 오지만, Cross-Attention은 다른 소스에서 옵니다.

**Self-Attention vs Cross-Attention**

```
        Self-Attention

          같은 시퀀스 X
          ┌──┬──┐
          ↓  ↓  ↓
          Q  K  V

  "나 자신의 토큰들끼리 서로 참조"

        Cross-Attention

  Decoder 상태      Encoder 출력
       │            ┌──┬──┐
       ↓            ↓  ↓
       Q            K  V

  "내가(Decoder) 다른 시퀀스(Encoder)를 참조"
```

**기계 번역 예시로 이해하기**

영어 → 프랑스어 번역:

```
Encoder 입력: "The cat sits on the mat"
Decoder 출력: "Le chat est assis sur le tapis"
```

```
          Cross-Attention 동작

 Encoder (영어):  [The] [cat] [sits] [on] [the] [mat]
                   ↓     ↓     ↓     ↓     ↓     ↓
                   K_1   K_2   K_3   K_4   K_5   K_6   ← Key
                   V_1   V_2   V_3   V_4   V_5   V_6   ← Value
                   ↑     ↑     ↑     ↑     ↑     ↑
                   └─────┴─────┴─────┴─────┴─────┘
                                ↑ 비교
                                │
 Decoder (프랑스어):       [Q] ← "chat"을 생성하려는 순간

 질문: "chat을 생성하려면 영어 문장의 어디를 봐야 할까?"
 답변: "cat"에 높은 attention! → "cat"의 정보(V_2)를 가져옴
```

**수식으로 보기**

Self-Attention:
```
Q = X · W_Q   ┐
K = X · W_K   ├─ 모두 같은 X에서!
V = X · W_V   ┘
```

Cross-Attention:
```
Q = X_decoder · W_Q     ← Decoder 상태에서
K = X_encoder · W_K     ← Encoder 출력에서
V = X_encoder · W_V     ← Encoder 출력에서
```

Attention 계산은 동일:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**직관적 비유**

| 비유 | Self-Attention | Cross-Attention |
| --- | --- | --- |
| 독서 | 책의 앞부분을 읽으면서 같은 책의 다른 부분 참조 | 번역서를 쓰면서 원서를 참조 |
| 대화 | 내가 한 말을 되돌아보며 정리 | 상대방이 한 말을 듣고 답변 구성 |
| 검색 | 문서 내에서 관련 단락 찾기 | 다른 문서에서 관련 정보 찾기 |

**Transformer Decoder에서의 위치**

```
         DECODER BLOCK

  ┌────────────────────┐
  │ 1. Masked Self-Attention │  ← 이전에 생성한 토큰들끼리 참조
  └──────────┬─────────┘
             ↓
  ┌────────────────────┐
  │ 2. Cross-Attention  │  ← Encoder 출력 참조 (여기!)
  │    Q: Decoder에서    │
  │    K,V: Encoder에서  │
  └──────────┬─────────┘
             ↓
  ┌────────────────────┐
  │ 3. Feed-Forward Network │
  └────────────────────┘
```

**실제 Attention 패턴 예시**

```
"The cat sits" → "Le chat est assis"

        The   cat   sits
         ↓     ↓     ↓
Le     [0.8] [0.1] [0.1]  ← "Le"는 "The"를 많이 봄
chat   [0.1] [0.8] [0.1]  ← "chat"은 "cat"을 많이 봄
est    [0.1] [0.1] [0.4]  ← "est"는 "sits"를 참조
assis  [0.0] [0.1] [0.9]  ← "assis"는 "sits"를 많이 봄
```

**요약**

| 구분 | Self-Attention | Cross-Attention |
| --- | --- | --- |
| Q 출처 | 자기 자신 | Decoder |
| K, V 출처 | 자기 자신 | **Encoder** |
| 목적 | 시퀀스 내부 관계 학습 | 두 시퀀스 간 관계 학습 |
| 사용처 | Encoder, Decoder | Decoder에서만 |
| 질문 | "내 문장에서 뭐가 중요?" | "원본의 어디를 봐야 해?" |

</details>

#### 8.4 Position Encoding

**문제:** Self-attention은 순서 정보가 없음

**Sinusoidal Position Encoding (원본 논문):**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**최종 입력:**

$$\text{Input} = \text{Token Embedding} + \text{Position Encoding}$$

<details>
<summary>Positional Encoding을 Token Embedding에 더하기만 해도 위치 정보가 손실되지 않는 이유</summary>

정말 좋은 질문이에요! 직관적으로 "더하면 정보가 섞여서 손실되지 않나?"라고 생각하는 게 자연스럽습니다. 왜 작동하는지 깊이 설명해 드릴게요.

#### 1. 먼저, 직관적인 의문

```
Token Embedding:    [0.2, 0.5, 0.1, 0.8, ...]   (단어 의미)
Position Encoding: + [0.1, 0.3, 0.7, 0.2, ...]   (위치 정보)
                    ─────────────────────────
결과:               [0.3, 0.8, 0.8, 1.0, ...]   (섞인 것??)

  "이러면 둘 다 구분이 안 되는 거 아닌가?"
```

#### 2. 핵심 답변: 고차원 공간의 마법

**2.1 고차원에서는 "공간이 충분하다"**

```
   저차원 (2D) vs 고차원 (512D)

 2D 공간:                512D 공간:
 ┌─ · · ─┐              거의 무한한 방향이 존재
 │ · · · │  ← 공간이 좁음  서로 "거의 직교"하는
 └───────┘               수백 개의 벡터 가능!

 정보가 쉽게 섞임         각 정보가 자기만의 "차원"에
                         존재할 수 있음
```

핵심 통찰: d_model = 512 차원에서는 Token Embedding과 Position Encoding이 거의 직교하는 부분공간을 각각 차지할 수 있습니다.

**2.2 수학적으로 보기**

두 벡터가 직교(orthogonal)하면:

```
Token Embedding (t)와 Position Encoding (p)가 직교할 때:

t · p ≈ 0  (내적이 거의 0)

이 경우, 더해도 각각의 정보가 보존됨:
- ||t + p||² = ||t||² + ||p||² + 2(t·p)
              = ||t||² + ||p||²  (t·p ≈ 0이므로)
```

비유:
```
3D 공간에서:
  x축 방향 벡터: [1, 0, 0] (의미 정보)
  y축 방향 벡터: [0, 1, 0] (위치 정보)

  더하면: [1, 1, 0]

  → x 성분과 y 성분이 여전히 구분 가능!
```

#### 3. Position Encoding 설계의 비밀

**3.1 Sin/Cos 함수를 쓰는 이유**

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

```
차원 인덱스 (i)
  0   1   2   3   4   5   ...  255
  ↓   ↓   ↓   ↓   ↓   ↓        ↓
 sin cos sin cos sin cos ... cos
  │   │   │   │   │   │        │
  각 차원마다 다른 주파수의 sin/cos 파동!
```

**3.2 시각화: 각 위치의 Positional Encoding**

```
Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]
             = [0.000,  1.000,  0.000,  1.000, ...]

Position 1:  [sin(1/1), cos(1/1), sin(1/10), cos(1/10), ...]
             = [0.841,   0.540,   0.0998,    0.995, ...]

Position 2:  [sin(2/1), cos(2/1), sin(2/10), cos(2/10), ...]
             = [0.909,  -0.416,   0.198,     0.980, ...]
```

각 위치마다 고유한 "지문"이 생성됩니다!

**3.3 왜 Sin/Cos인가?**

**특성 1: 각 위치가 유일하게 구분됨**

```
위치 0: ●─────────────────
위치 1:  ●────────────────
위치 2:   ●───────────────
위치 3:    ●──────────────
...
  고차원 공간에서 각각 다른 점!
```

**특성 2: 상대적 위치를 선형 변환으로 표현 가능**

```
PE(pos+k) = f(PE(pos))  ← 선형 변환으로 표현 가능!

즉, "3칸 떨어진 관계"를 모델이 쉽게 학습 가능
```

**특성 3: 값이 bounded (-1~1)**

```
sin, cos ∈ [-1, 1]

→ Token Embedding과 비슷한 스케일 유지
→ 한쪽이 다른 쪽을 압도하지 않음
```

#### 4. 모델이 두 정보를 분리하는 원리

**4.1 학습을 통한 분리**

```
          학습 과정에서 일어나는 일

 Input = Token Emb + Pos Enc
              ↓
 W_Q, W_K, W_V 행렬들이 학습됨
              ↓
 이 행렬들이 "의미 정보"와 "위치 정보"를
 필요에 따라 분리하거나 조합하는 법을 학습!
```

**비유: 칵테일 파티 효과**

```
여러 사람이 동시에 말해도 (정보가 섞여도)
우리 뇌는 특정 목소리에 집중할 수 있음

마찬가지로, 신경망의 Weight들이
섞인 신호에서 필요한 정보를 추출하는 법을 학습
```

**4.2 실제로 어떻게 분리되는가**

```python
# 개념적 설명 (실제 구현은 더 복잡)

Input = TokenEmb + PosEnc  # 섞인 입력

# Query 행렬이 "의미" 부분을 추출하도록 학습될 수 있음
W_Q_semantic = [[1, 0, 0, ...],    # Token Emb 방향에 큰 가중치
                [0, 1, 0, ...],
                ...]

# 다른 Head는 "위치" 부분을 추출하도록 학습
W_Q_position = [[0, 0, 0.1, ...],  # Pos Enc 방향에 큰 가중치
                [0, 0, 0, 0.1...],
...                ...]
```

#### 5. 왜 Concatenation이 아닌 Addition인가?

**5.1 두 방법 비교**

```
 방법 1: Concatenation (이어붙이기)

 Token Emb (512d) + Pos Enc (512d) = Combined (1024d)

 장점: 정보가 명확히 분리됨
 단점: 차원이 2배 → 계산량 4배 증가!
       (Attention은 O(n²·d))
```

```
 방법 2: Addition (더하기) ← Transformer가 선택한 방법

 Token Emb (512d) + Pos Enc (512d) = Combined (512d)

 장점: 차원 유지 → 효율적
 단점: 정보가 섞임? → 실제로는 괜찮음! (고차원이라)
```

**5.2 실험적 증거**

원 논문과 후속 연구에서 확인된 사실:

| 방법 | 성능 | 비고 |
| --- | --- | --- |
| Addition | BLEU 27.3 | 원 논문 |
| Concatenation | 비슷 | 계산량 2배 |
| Learned Position Emb | BLEU 27.2 | 거의 동일 |

Addition이 효율적이면서도 성능 손실이 없음!

#### 6. 직관적 비유로 정리

**비유 1: 라디오 주파수**
```
AM 라디오 (의미 정보):   ~~~~~~~~  (저주파)
FM 라디오 (위치 정보):   ~~~~~~~~~~~~~ (고주파)

합쳐진 신호:            ~·~·~·~·~·~ (둘 다 포함)

수신기(모델)가 각각을 분리해서 들을 수 있음!
```

**비유 2: 잉크와 형광펜**
```
검은 글씨 (의미):      "Hello World"
형광펜 (위치):          ▮▮ ▮▮▮▮▮  (1번째, 2번째 단어 표시)

종이 위에 겹쳐져도:     "Hello World"
                       ▮▮ ▮▮▮▮▮

사람 눈은 둘 다 구분 가능!
```

**비유 3: GPS 꼬리표**
```
건물 이름 (의미):       "스타벅스"
GPS 좌표 (위치):       (37.5, 127.0)

합쳐진 정보:            "스타벅스 @ (37.5, 127.0)"

→ 둘 다 보존되고, 필요에 따라 각각 활용 가능
```

#### 7. 최종 정리

| 의문 | 답변 |
| --- | --- |
| 더하면 섞이지 않나? | 고차원에서는 거의 직교하는 부분공간 사용 가능 |
| 어떻게 분리하나? | 학습된 W_Q, W_K, W_V가 필요한 정보 추출 |
| 왜 concatenation 안 쓰나? | 계산 효율성 (차원 유지) |
| 실제로 작동하나? | 네! 실험적으로 검증됨 |

결론: 512차원 같은 고차원 공간에서는 "더하기"를 해도 각 정보가 자신만의 "방향"을 가질 수 있고, 신경망은 이를 분리해서 활용하는 법을 학습합니다. 이것이 바로 **고차원의 축복(blessing of dimensionality)**입니다!

</details>

<details>
<summary>Positional Encoding 특성 상세 (상대적 위치를 선형 변환으로 표현 가능한 이유)</summary>

#### 1. 먼저 삼각함수 덧셈 공식 복습

핵심은 이 두 공식입니다:

$$\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$

$$\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$

#### 2. Positional Encoding 다시 보기

간단히 하기 위해 하나의 주파수만 봅시다 (차원 i를 고정):

$$PE(pos) = \begin{bmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{bmatrix}$$

여기서 $\omega = \frac{1}{10000^{2i/d}}$ (주파수)

#### 3. 위치 pos에서 k칸 이동하면?

pos+k 위치의 encoding:

$$PE(pos + k) = \begin{bmatrix} \sin(\omega \cdot (pos + k)) \\ \cos(\omega \cdot (pos + k)) \end{bmatrix} = \begin{bmatrix} \sin(\omega \cdot pos + \omega \cdot k) \\ \cos(\omega \cdot pos + \omega \cdot k) \end{bmatrix}$$

#### 4. 덧셈 공식 적용

$a = \omega \cdot pos, b = \omega \cdot k$로 놓으면:

$$\sin(\omega \cdot pos + \omega \cdot k) = \sin(\omega \cdot pos)\cos(\omega \cdot k) + \cos(\omega \cdot pos)\sin(\omega \cdot k)$$

$$\cos(\omega \cdot pos + \omega \cdot k) = \cos(\omega \cdot pos)\cos(\omega \cdot k) - \sin(\omega \cdot pos)\sin(\omega \cdot k)$$

#### 5. 행렬 형태로 정리하면?

$$\begin{bmatrix} \sin(\omega(pos + k)) \\ \cos(\omega(pos + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix} \begin{bmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{bmatrix}$$

즉:

$$PE(pos + k) = M_k \cdot PE(pos)$$

여기서 $M_k$는 k에만 의존하는 행렬 (pos와 무관!)

#### 6. 구체적 숫자 예시

설정: $\omega = 1$ (단순화), 위치 2에서 3칸 이동 → 위치 5

```
PE(2) = [sin(2), cos(2)] = [0.909, -0.416]

k = 3일 때 변환 행렬:
M_3 = [cos(3)   sin(3) ]  = [-0.99   0.14]
      [-sin(3)  cos(3) ]    [-0.14  -0.99]

PE(5) = M_3 · PE(2)
      = [-0.99   0.14] · [0.909 ]
        [-0.14  -0.99]   [-0.416]

      = [-0.99×0.909 + 0.14×(-0.416)]
        [-0.14×0.909 + (-0.99)×(-0.416)]

      = [-0.958]
        [0.284 ]

검증: [sin(5), cos(5)] = [-0.959, 0.284] ✓ (거의 일치!)
```

#### 7. 왜 이게 중요한가?

```
              핵심 통찰

 "k칸 떨어진 관계"가 항상 같은 변환 행렬 M_k로 표현됨!

 PE(0) ─[M_3]──→ PE(3)
 PE(1) ─[M_3]──→ PE(4)      모두 같은 M_3!
 PE(2) ─[M_3]──→ PE(5)
 PE(7) ─[M_3]──→ PE(10)
```

#### 8. 모델 학습 관점에서 의미

```
문장: "The cat sat on the mat"
       0   1   2   3   4   5

모델이 학습해야 할 것:
"주어(cat)와 동사(sat)는 보통 1칸 떨어져 있다"

Sin/Cos encoding이면:

PE(cat위치) → PE(sat위치) 관계가
PE(다른주어위치) → PE(다른동사위치) 관계와

동일한 선형 변환 M_1로 표현됨!

→ 모델이 "1칸 관계"를 한 번만 학습하면
  모든 위치에 일반화 가능!
```

#### 9. 만약 랜덤 Position Encoding이었다면?

```
랜덤 Positional Encoding (비교용)

PE(0) = [0.2, 0.7, 0.1, ...]  ← 랜덤
PE(1) = [0.9, 0.3, 0.5, ...]  ← 랜덤
PE(2) = [0.4, 0.8, 0.2, ...]  ← 랜덤

PE(0) → PE(1) 관계: ???
PE(1) → PE(2) 관계: ???  (완전히 다름!)
PE(5) → PE(6) 관계: ???

→ "1칸 관계"를 매 위치마다 따로 학습해야 함
→ 일반화 어려움!
```

#### 10. 시각적 요약

```
Sin/Cos Positional Encoding의 마법:

위치 0   위치 1   위치 2   위치 3   위치 4
  ●────────●────────●────────●────────●
     M_1      M_1      M_1      M_1
    (동일)   (동일)   (동일)   (동일)

       ●────────────────●
              M_2

          ●──────────────────────────●
                       M_3

어떤 시작점에서든, k칸 이동은 같은 변환 M_k!
```

#### 11. Attention에서의 활용

```
Q·K^T 계산 시:

Q_i · K_j = (token_i + PE(i))·W_Q · W_K^T · (token_j + PE(j))^T

이 안에 PE(i)와 PE(j)의 관계가 포함되는데,
PE(j) = M_{j-i} · PE(i) 형태로 표현 가능하므로

→ W_Q, W_K가 이 선형 관계 M_{j-i}를 쉽게 학습 가능!
→ "상대적 거리 |j-i|"에 따른 attention 패턴 학습이 용이
```

**최종 정리**

| 질문 | 답변 |
| --- | --- |
| 선형 변환이 뭔가요? | 행렬 곱으로 표현되는 변환 |
| 왜 가능한가요? | sin/cos 덧셈 공식 덕분 |
| 왜 중요한가요? | "k칸 관계"를 한 번 학습하면 모든 위치에 적용 가능 |
| 모델에 어떤 도움? | 상대적 위치 패턴의 일반화가 쉬워짐 |

핵심: Sin/Cos encoding은 상대적 위치 관계가 위치에 무관하게 일정한 패턴을 가지도록 설계되어, 모델이 위치 관계를 효율적으로 학습할 수 있게 합니다!

</details>

#### 8.5 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**특징:**
* Hidden dimension이 입력보다 큼 (보통 4배)
* 예: $d_{model} = 512 \rightarrow d_{ff} = 2048$
* 더 풍부한 표현 학습을 위한 "expansion"

<details>
<summary>FFN 적용 이유 상세</summary>

#### FFN의 핵심 동작 요약

**한 문장 정의:**

> FFN은 각 토큰의 표현을 "개별적으로" 비선형 변환하여, 문맥 정보를 바탕으로 저장된 지식을 꺼내오는 역할

**직관적 비유:**

```
            도서관 비유

 Self-Attention = "질문 이해하기"
   "프랑스의 수도가 뭐야?"
     → "프랑스", "수도"라는 키워드 파악
     → 문맥에서 관련 정보 수집

 FFN = "도서관에서 답 찾기"
     → "프랑스 + 수도" 패턴 인식
     → 저장된 지식에서 "파리" 검색
     → 답 출력
```

**두 가지 핵심 역할:**

| 역할 | Self-Attention | FFN |
| --- | --- | --- |
| 방향 | 토큰 ↔ 토큰 (가로) | Feature ↔ Feature (세로) |
| 동작 | "누가 누구와 관련있나" | "이 정보를 어떻게 해석하나" |
| 비유 | 회의에서 의견 수집 | 수집된 의견으로 결론 도출 |

```
"I love AI" 처리 시:

Self-Attention 후 "AI"의 표현:
  = "AI" + "I가 사랑한다는 맥락" + "긍정적 감정"

FFN 후 "AI"의 표현:
  = 위 정보를 종합하여 → "기술에 대한 긍정적 언급" 패턴 활성화
```

**수식으로 보는 핵심:**

$$\text{FFN}(x) = \text{ReLU}(xW_1)W_2$$

1. $xW_1$: "이 입력이 어떤 패턴과 매칭되는가?" (2048개 패턴 검사)
2. ReLU: "매칭되는 패턴만 활성화" (대부분 0)
3. $\times W_2$: "활성화된 패턴에 해당하는 지식 출력"

**최종 정리:**

```
Attention: "문맥 파악" → 토큰들 사이의 관계 학습
FFN: "지식 활용" → 파악된 문맥을 바탕으로 저장된 지식에서 적절한 정보 추출
```

---

#### 1. FFN의 기본 구조

**수식:**

$$\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$$

원 논문에서는 ReLU 사용:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

현대 모델에서는 GELU, SwiGLU 등 사용:

$$\text{FFN}(x) = \text{GELU}(xW_1)W_2$$

**차원 변화:**

$$x \in \mathbb{R}^{d_{model}} \xrightarrow{W_1} \mathbb{R}^{d_{ff}} \xrightarrow{W_2} \mathbb{R}^{d_{model}}$$

원 논문 설정:

$$512 \xrightarrow{W_1} 2048 \xrightarrow{W_2} 512$$

(4배 확장 후 다시 축소)

---

#### 2. Self-Attention만으로는 부족한 이유

**Self-Attention의 특성:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Self-Attention이 하는 일: 토큰 간 정보 혼합 (Token Mixing)

```
"I love AI"

Self-Attention 후:
  "I"의 표현 = 0.1×"I" + 0.3×"love" + 0.6×"AI"

  → 다른 토큰들의 정보를 가중 합산
  → 토큰 "간"의 관계만 처리
```

**Self-Attention의 한계:**

**한계 1: 선형 연산의 조합**

$$\text{Attention Output} = \text{softmax}(\cdot)V$$

Softmax는 비선형이지만, 최종 출력은 $V$의 가중 평균

$$\text{Output}_i = \sum_j \alpha_{ij} V_j$$

→ $V$ 벡터들의 convex combination (볼록 조합)

```
      Self-Attention 출력의 한계

 V1 = [1, 0]
 V2 = [0, 1]
 V3 = [-1, 0]

 Attention 출력 = α1·V1 + α2·V2 + α3·V3
 (단, α1 + α2 + α3 = 1, αi ≥ 0)

       V2 [0,1]
        ●
       /|\
      / | \
     /  |  \    ← 이 삼각형 내부의 점만 표현 가능
    /   |   \
   /    |    \
  ●─────┼─────●
 V3    V1
 [-1,0]    [1,0]

  → 이 영역 바깥의 표현은 불가능!
```

**한계 2: 채널(Feature) 간 상호작용 없음**

Self-Attention은 같은 차원끼리만 연산:

$$\text{Score}_{ij} = \sum_{d=1}^{d_k} Q_{i,d} \cdot K_{j,d}$$

```
토큰 표현: [feature1, feature2, feature3, ..., feature512]

Self-Attention:
  feature1은 다른 토큰의 feature1과만 상호작용
  feature2는 다른 토큰의 feature2와만 상호작용
  ...

  → feature1과 feature2 간의 상호작용 없음!
```

---

#### 3. FFN의 역할

**역할 1: 비선형 변환 (Nonlinearity)**

$$\text{FFN}(x) = \text{ReLU}(xW_1)W_2$$

ReLU의 역할:

$$\text{ReLU}(z) = \max(0, z)$$

```
          비선형성의 중요성

 선형 변환만 쌓으면:
 y = W3(W2(W1·x)) = (W3·W2·W1)·x = W'·x

 → 아무리 많이 쌓아도 하나의 선형 변환과 동일!

 비선형 활성화 추가:
 y = W2·ReLU(W1·x)

 → 복잡한 비선형 함수 근사 가능
 → Universal Approximation Theorem
```

**역할 2: 채널 믹싱 (Channel Mixing)**

$$h = xW_1$$

$W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$의 각 열은 모든 입력 feature를 조합

```
입력 x = [x1, x2, x3, ..., x512]

W1의 첫 번째 열: [w1, w2, w3, ..., w512]

h1 = x1·w1 + x2·w2 + x3·w3 + ... + x512·w512

→ 모든 feature가 상호작용하여 새로운 feature 생성!
```

```
    Attention vs FFN의 역할 비교

 Self-Attention: Token Mixing
 ┌─────┬─────┬─────┐
 │ T1  │ T2  │ T3  │  ← 토큰들
 ├─────┼─────┼─────┤
 │ f1  │ f1  │ f1  │──→ 같은 feature끼리 상호작용
 │ f2  │ f2  │ f2  │──→
 │ f3  │ f3  │ f3  │──→
 └─────┴─────┴─────┘
    ↕     ↕     ↕      세로(토큰) 방향 믹싱

 FFN: Channel Mixing
 ┌─────┬─────┬─────┐
 │ T1  │ T2  │ T3  │
 ├─────┼─────┼─────┤
 │ f1 ←──────────→ │   가로(feature) 방향 믹싱
 │ f2 ←──────────→ │   각 토큰 독립적으로
 │ f3 ←──────────→ │
 └─────┴─────┴─────┘
```

**역할 3: 표현력 확장 (Capacity)**

차원 확장의 의미:

$$512 \rightarrow 2048 \rightarrow 512$$

```
          차원 확장의 효과

 입력: 512차원 (압축된 정보)
       ↓
 확장: 2048차원 (풍부한 표현 공간)
       │
       ├── 더 세밀한 패턴 감지 가능
       ├── 다양한 feature 조합 생성
       └── 비선형 변환 적용
       ↓
 축소: 512차원 (핵심 정보만 추출)

 비유: 이미지 처리
 - 저해상도 → 고해상도로 확대 (세부 처리)
 - 고해상도 → 저해상도로 축소 (핵심 보존)
```

---

#### 4. Position-wise의 의미

**각 위치에 독립적으로 동일한 FFN 적용:**

$$\text{FFN}(X) = [\text{FFN}(x_1), \text{FFN}(x_2), \ldots, \text{FFN}(x_n)]$$

```
입력 시퀀스: ["I", "love", "AI"]
            [x1,   x2,    x3]

FFN 적용:
  FFN(x1) → y1  (동일한 W1, W2 사용)
  FFN(x2) → y2  (동일한 W1, W2 사용)
  FFN(x3) → y3  (동일한 W1, W2 사용)

출력: [y1, y2, y3]
```

**왜 Position-wise인가?**

| 특성 | 설명 |
| --- | --- |
| 파라미터 공유 | 위치마다 같은 가중치 → 파라미터 효율적 |
| 일반화 | 특정 위치에 종속되지 않음 |
| 병렬 처리 | 모든 위치 동시에 계산 가능 |

```
       Position-wise FFN의 의미

 "각 토큰을 개별적으로 같은 방식으로 변환"

 Self-Attention 후:
   각 토큰은 문맥 정보를 이미 포함
   "love" = 원래 "love" + "I"의 정보 + "AI"의 정보

 FFN:
   이 풍부해진 표현을 비선형 변환
   → 더 추상적이고 유용한 표현으로 변환
```

---

#### 5. FFN as Knowledge Storage (최근 연구)

**FFN이 지식을 저장한다는 발견**

연구 결과 (Geva et al., 2021):

$$\text{FFN}(x) = \text{ReLU}(xW_1)W_2$$

- $W_1$의 각 행: **Key** (특정 패턴 감지)
- $W_2$의 각 열: **Value** (해당 패턴에 대한 출력)

```
         FFN as Key-Value Memory

 W1 ∈ R^(512×2048): 2048개의 "key" 패턴
 W2 ∈ R^(2048×512): 2048개의 "value" 응답

 동작 과정:
 1. h = xW1: 입력이 각 key와 얼마나 매칭되는지
 2. ReLU(h): 매칭되는 key만 활성화
 3. ReLU(h)W2: 활성화된 key의 value를 출력

 예시:
 Key 157: "수도 관련 문맥" 감지
 Value 157: "capital city" 관련 정보 출력

 입력: "The capital of France is ___"
 → Key 157 활성화
 → Value 157 ("Paris" 관련 정보) 출력
```

**실험적 증거:**

```
         FFN 뉴런 분석 결과

 GPT-2의 특정 FFN 뉴런들:

 뉴런 #1823: "연도" 관련 문맥에서 활성화
   "In 1997, ..." → 높은 활성화
   "The color is ..." → 낮은 활성화

 뉴런 #4521: "프로그래밍 언어" 관련 문맥에서 활성화
   "Python is a ..." → 높은 활성화
   "The cat sat ..." → 낮은 활성화

 뉴런 #892: "수도" 관련 사실에서 활성화
   "The capital of Japan is ..." → 높은 활성화

 → FFN이 factual knowledge를 저장!
```

---

#### 6. Attention과 FFN의 상호보완

**전체 Transformer 블록의 동작:**

```
    Attention + FFN의 상호보완적 역할

 입력: "The capital of France is [MASK]"

 Step 1: Self-Attention
   "France"의 정보를 "[MASK]"로 전달
   "[MASK]" = f("France", "capital", 문맥)

   역할: "어떤 정보가 필요한지" 파악

              ↓

 Step 2: FFN
   "France + capital" 패턴 감지
   → "Paris" 관련 지식 활성화

   역할: "필요한 정보를 지식에서 검색"

              ↓

 출력: "Paris" 예측 확률 높음
```

**역할 요약:**

| 구성요소 | 역할 | 비유 |
| --- | --- | --- |
| **Self-Attention** | 문맥 파악, 관련 정보 수집 | "무엇을 찾아야 하는지 파악" |
| **FFN** | 지식 저장, 패턴 변환 | "저장된 지식에서 답 검색" |

---

#### 7. FFN의 확장 비율 (Expansion Ratio)

**왜 4배 확장인가?**

$$d_{ff} = 4 \times d_{model}$$

경험적 최적값:

| 확장 비율 | 파라미터 수 | 성능 |
| --- | --- | --- |
| 1× | 적음 | 표현력 부족 |
| 2× | 중간 | 괜찮음 |
| **4×** | **많음** | **최적** |
| 8× | 매우 많음 | 수확 체감 |

**파라미터 분포:**

Transformer의 파라미터 대부분이 FFN에 있음:

$$\text{FFN 파라미터} = d_{model} \times d_{ff} + d_{ff} \times d_{model} = 2 \times d_{model} \times d_{ff}$$

원 논문 기준:

$$2 \times 512 \times 2048 = 2{,}097{,}152 \text{ (약 2M)}$$

```
         Transformer 파라미터 분포

 Encoder Layer 1개 기준 (d_model=512, h=8):

 Multi-Head Attention:
   W_Q, W_K, W_V, W_O: 4 × 512 × 512 = 1.05M

 FFN:
   W_1: 512 × 2048 = 1.05M
   W_2: 2048 × 512 = 1.05M
   합계: 2.1M

 ┌───────────────┬────────────────┐
 │  Attention    │     FFN        │
 │    33%        │     67%        │
 └───────────────┴────────────────┘

 → FFN이 파라미터의 2/3 차지!
 → 그만큼 많은 "지식"을 저장할 수 있음
```

---

#### 8. 활성화 함수의 발전

**ReLU (원 논문):**

$$\text{ReLU}(x) = \max(0, x)$$

문제: Dying ReLU (음수 영역에서 gradient = 0)

**GELU (GPT, BERT):**

$$\text{GELU}(x) = x \cdot \Phi(x)$$

여기서 $\Phi(x)$는 표준 정규분포의 CDF

$$\approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

장점: 부드러운 곡선, 확률적 해석 가능

**SwiGLU (LLaMA, PaLM):**

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_3)$$

$$\text{Swish}(x) = x \cdot \sigma(x)$$

장점: Gate 메커니즘으로 더 정교한 정보 흐름 제어

**성능 비교:**

| 활성화 함수 | 모델 | 상대 성능 |
| --- | --- | --- |
| ReLU | Transformer (원본) | 기준 |
| GELU | GPT-2, BERT | +1~2% |
| SwiGLU | LLaMA, PaLM | +2~3% |

---

#### 9. FFN 없이 학습하면?

**실험 결과:**

```
         FFN 제거 실험 결과

 설정: WMT 영어-독일어 번역

 | 구성               | BLEU Score | 파라미터 |
 |─────────────────|──────────|────────|
 | Full Transformer  |    27.3    |   65M  |
 | FFN 제거           |    23.1    |   22M  |
 | Attention만 2배    |    24.8    |   44M  |
 | FFN 축소 (2배만 확장)|    26.5    |   44M  |

 결론:
 - FFN 제거 시 BLEU 4.2점 하락
 - Attention을 늘려도 FFN 역할 대체 불가
 - FFN의 고유한 역할 존재
```

**FFN이 하는 고유한 작업:**

```
          FFN만이 할 수 있는 것

 1. 특정 패턴 활성화 (Sparse Activation)
    - "France + capital" → 특정 뉴런만 활성화
    - ReLU가 대부분의 뉴런을 0으로 만듦

 2. 비선형 특징 조합
    - f1 × f2 같은 곱셈적 관계 학습
    - Attention은 가중 평균만 가능

 3. 차원 독립적 변환
    - 각 feature를 독립적으로 비선형 변환
    - 새로운 feature 생성
```

---

#### 10. 요약

**FFN의 핵심 역할:**

| 역할 | 설명 |
| --- | --- |
| 비선형성 | 복잡한 함수 근사 가능 |
| 채널 믹싱 | Feature 간 상호작용 |
| 표현력 확장 | 4배 확장으로 풍부한 표현 |
| 지식 저장 | Factual knowledge 저장소 |

**Attention + FFN 조합의 의미:**

$$\text{Transformer Block} = \text{Attention (토큰 믹싱)} + \text{FFN (채널 믹싱)}$$

```
 Self-Attention         FFN
 ┌─────┬─────┬─────┐   ┌─────┬─────┬─────┐
 │  ↕  │  ↕  │  ↕  │   │ ↔   │ ↔   │ ↔   │
 │  ↕  │  ↕  │  ↕  │ + │ ↔   │ ↔   │ ↔   │
 │  ↕  │  ↕  │  ↕  │   │ ↔   │ ↔   │ ↔   │
 └─────┴─────┴─────┘   └─────┴─────┴─────┘
 토큰 간 정보 교환       각 토큰 내 변환
 "누구와 관련있나"       "무슨 의미인가"

 → 두 가지가 합쳐져 완전한 표현 학습
```

</details>

#### 8.6 Add & Norm (Residual Connection + Layer Norm)

**Residual Connection:**

$$\text{output} = x + \text{SubLayer}(x)$$

* Gradient flow 개선
* 깊은 네트워크 학습 가능

<details>
<summary>Residual Connection 적용 이유 상세</summary>

#### 1. Residual Connection이란?

입력 $x$를 출력에 직접 더하는 연결 방식입니다.

일반 네트워크:

$$y = F(x)$$

Residual Connection 적용:

$$y = F(x) + x$$

여기서 $F(x)$는 레이어가 학습해야 할 **잔차(residual)**입니다.

---

#### 2. 왜 필요한가? - Vanishing Gradient 문제

**깊은 네트워크의 문제:**

역전파(Backpropagation) 시 gradient는 chain rule에 의해 곱해집니다:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \cdot \frac{\partial x_n}{\partial x_{n-1}} \cdot \frac{\partial x_{n-1}}{\partial x_{n-2}} \cdots \frac{\partial x_2}{\partial x_1}$$

문제: 각 항이 1보다 작으면 gradient가 기하급수적으로 감소

$$0.5 \times 0.5 \times 0.5 \times \cdots \times 0.5 = 0.5^n \approx 0$$

```
Layer 1 ← Layer 2 ← Layer 3 ← ... ← Layer 50 ← Loss
  ↑
  gradient가 거의 0에 가까워짐
  → 앞쪽 레이어가 학습되지 않음
```

---

#### 3. Residual Connection의 해결 방법

**Gradient 흐름 분석:**

Residual block:

$$y = F(x) + x$$

Gradient 계산:

$$\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + \frac{\partial x}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$$

핵심: **+1 항이 항상 존재!**

**깊은 네트워크에서의 gradient:**

여러 레이어를 거쳐도:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=1}^{n-1}\left(\frac{\partial F_i}{\partial x_i} + 1\right)$$

전개하면:

$$= \frac{\partial L}{\partial x_n} \cdot \left(1 + \sum \frac{\partial F_i}{\partial x_i} + \sum \frac{\partial F_i}{\partial x_i}\frac{\partial F_j}{\partial x_j} + \cdots\right)$$

항상 1을 포함하는 경로가 존재 → Gradient가 직접 흐를 수 있음

```
              Gradient Highway

 일반 네트워크:
 x → [Layer] → [Layer] → [Layer] → ... → Loss
       ×0.5      ×0.5      ×0.5
 gradient: 0.5^n → 거의 0

 Residual 네트워크:
 x ──────────────────────────────→ (+) → Loss
  └→ [Layer] → [Layer] → ... ──→↑

 gradient: 항상 1을 포함하는 직접 경로 존재
```

---

#### 4. 학습 관점에서의 이점

**Identity Mapping 학습 용이성:**

일반 네트워크: 레이어가 전체 매핑 $H(x)$를 학습

$$y = H(x)$$

Residual 네트워크: 레이어가 잔차 $F(x) = H(x) - x$만 학습

$$y = F(x) + x = H(x)$$

**왜 더 쉬운가?**

만약 최적의 매핑이 항등 함수(identity)라면:

| 방식 | 학습해야 할 것 | 난이도 |
| --- | --- | --- |
| 일반 | $H(x) = x$ | 어려움 (복잡한 함수로 항등 함수 근사) |
| Residual | $F(x) = 0$ | 쉬움 (가중치를 0에 가깝게 만들면 끝) |

```
예시: 이미 충분히 좋은 표현이 있을 때

일반 네트워크:
  입력 x가 이미 최적 → 레이어가 x를 그대로 출력해야 함
  → 복잡한 가중치 조합으로 항등 함수 근사 필요

Residual 네트워크:
  입력 x가 이미 최적 → F(x) = 0만 학습하면 됨
  → 가중치를 0에 가깝게 만들면 끝
```

---

#### 5. 실험적 증거 (ResNet 논문)

**깊이에 따른 성능 비교:**

```
         ImageNet Error Rate (%)

 Error
   ↑
   │     일반 네트워크
 28│   ●───●
   │  /      \____● (56 layers: 더 나빠짐!)
 25│ /
   │●
 22│/
   │
   │      Residual 네트워크
 20│   ●───●───●───● (152 layers도 OK!)
   │  /
 18│●
   └────────────────────────→
      20    36    56   152  layers

 일반: 깊어지면 오히려 성능 저하 (degradation)
 Residual: 깊어져도 성능 유지/향상
```

**핵심 발견:**
- 일반 네트워크: 56층이 20층보다 더 나쁨 (학습 문제)
- Residual 네트워크: 152층도 안정적으로 학습

---

#### 6. Transformer에서의 Residual Connection

**구조:**

$$\text{Output}_1 = \text{LayerNorm}(x + \text{MultiHeadAttention}(x))$$

$$\text{Output}_2 = \text{LayerNorm}(\text{Output}_1 + \text{FFN}(\text{Output}_1))$$

**왜 Transformer에서 특히 중요한가?**

| 이유 | 설명 |
| --- | --- |
| 깊은 구조 | Encoder/Decoder 각 6개 레이어, 각 레이어에 2개 sublayer |
| Attention의 특성 | Softmax 출력이 saturate될 수 있음 |
| 정보 보존 | 원본 토큰 정보가 끝까지 전달되어야 함 |

```
"I love AI" 처리 시:

Residual 없이:
  Layer 1 → Layer 2 → ... → Layer 6
  "I"의 원본 정보가 점점 희석됨

Residual 있음:
  Layer 1 ────────────────────────→ (+)
       └→ Attention → FFN ───────→↑

  "I"의 원본 정보 + 문맥 정보 = 풍부한 표현
```

---

#### 7. 결과물 차이 비교

**실제 성능 차이:**

| 설정 | Perplexity | BLEU Score | 수렴 속도 |
| --- | --- | --- | --- |
| Residual 없음 (6 layers) | 발산 또는 매우 높음 | 매우 낮음 | 수렴 안됨 |
| Residual 있음 (6 layers) | ~5.0 | ~27.3 | 정상 수렴 |
| Residual 없음 (2 layers) | ~15.0 | ~18.0 | 느림 |

---

#### 8. 수학적 직관 요약

$$y = F(x) + x$$

| 관점 | 설명 |
| --- | --- |
| Gradient 관점 | $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + 1$ → 최소 1 보장 |
| 학습 관점 | $F(x) = 0$ 학습이 $H(x) = x$ 학습보다 쉬움 |
| 정보 관점 | 원본 정보 $x$가 항상 보존됨 |
| 앙상블 관점 | 다양한 깊이의 경로가 암묵적으로 존재 |

---

#### 9. 결론

**Residual Connection을 하는 이유:**

1. **Vanishing Gradient 해결:** Gradient가 직접 흐르는 "고속도로" 제공
2. **깊은 네트워크 학습 가능:** 100층 이상도 안정적 학습
3. **학습 용이성:** 잔차만 학습하면 되므로 최적화가 쉬움
4. **정보 보존:** 원본 입력 정보가 손실 없이 전달

**결과물 차이:**
- Residual 없이 깊은 Transformer는 학습 자체가 불가능
- 얕은 네트워크로 제한하면 표현력 부족
- Residual Connection은 Transformer의 필수 구성요소

</details>

**Layer Normalization:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

* 학습 안정화
* 빠른 수렴

<details>
<summary>LayerNorm 적용 이유 상세</summary>

#### 1. Layer Normalization이란?

각 샘플의 특성(feature) 차원에 대해 정규화하는 기법입니다.

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

- $\mu$: 해당 샘플 내 평균
- $\sigma$: 해당 샘플 내 표준편차
- $\gamma, \beta$: 학습 가능한 파라미터 (scale, shift)
- $\epsilon$: 수치 안정성을 위한 작은 값 (보통 $10^{-6}$)

---

#### 2. 왜 정규화가 필요한가?

**Internal Covariate Shift 문제:**

각 레이어의 입력 분포가 학습 중 계속 변하는 현상입니다.

```
         Internal Covariate Shift

 학습 초기:
 Layer 1 출력: 평균=0.5, 분산=1.0
         ↓
 Layer 2는 이 분포에 맞춰 학습

 학습 중기:
 Layer 1 가중치 업데이트됨
 Layer 1 출력: 평균=2.3, 분산=5.0 (분포 변화!)
         ↓
 Layer 2 입장에서는 완전히 다른 데이터가 들어옴
 → 이전에 학습한 것이 무효화됨
```

**문제점:**
- 각 레이어가 계속 변하는 입력에 적응해야 함
- 학습이 불안정하고 느려짐
- 더 낮은 learning rate 필요

---

#### 3. 정규화의 효과

**입력 분포 안정화:**

$$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

정규화 전:
```
Layer 1 출력 (불안정):
  Epoch 1: [-5, 10, 3, -8, 15]     평균=3, 분산=큼
  Epoch 2: [0.1, 0.5, 0.2, 0.8]   평균=0.4, 분산=작음
  Epoch 3: [100, 200, 150, 180]    평균=157, 분산=매우 큼
```

정규화 후:
```
Layer 1 출력 (안정):
  Epoch 1: [-0.8, 0.7, 0, -1.1, 1.2]   평균≈0, 분산≈1
  Epoch 2: [-0.9, 0.3, -0.6, 1.2]       평균≈0, 분산≈1
  Epoch 3: [-1.1, 0.8, -0.1, 0.4]       평균≈0, 분산≈1
```

---

#### 4. Batch Normalization vs Layer Normalization

**정규화 방향의 차이:**

```
       입력 텐서: (Batch, Sequence, Features)
                   (B, S, D)

 Batch Normalization: 배치 방향으로 정규화

       Feature 1  Feature 2  Feature 3
 Seq1  [ 0.2  ,   0.5   ,   0.1  ] ┐
 Seq2  [ 0.3  ,   0.4   ,   0.2  ] │ Batch 1
 Seq1  [ 0.1  ,   0.6   ,   0.3  ] ┐
 Seq2  [ 0.4  ,   0.3   ,   0.1  ] │ Batch 2
         ↓         ↓         ↓
       이 방향으로 평균/분산 계산

 Layer Normalization: 특성 방향으로 정규화

       Feature 1  Feature 2  Feature 3
 Seq1  [ 0.2  ,   0.5   ,   0.1  ] → 이 방향으로
 Seq2  [ 0.3  ,   0.4   ,   0.2  ] → 평균/분산 계산
```

**수식 비교:**

**Batch Normalization:**

$$\mu_j = \frac{1}{B \cdot S}\sum_{b=1}^{B}\sum_{s=1}^{S} x_{b,s,j}$$

$$\sigma_j^2 = \frac{1}{B \cdot S}\sum_{b=1}^{B}\sum_{s=1}^{S} (x_{b,s,j} - \mu_j)^2$$

각 feature $j$에 대해 배치 전체의 통계 계산

**Layer Normalization:**

$$\mu_{b,s} = \frac{1}{D}\sum_{d=1}^{D} x_{b,s,d}$$

$$\sigma_{b,s}^2 = \frac{1}{D}\sum_{d=1}^{D} (x_{b,s,d} - \mu_{b,s})^2$$

각 샘플, 각 위치에 대해 feature 방향으로 통계 계산

**왜 Transformer는 LayerNorm을 사용하는가?**

| 측면 | Batch Norm | Layer Norm |
| --- | --- | --- |
| 배치 의존성 | 배치 크기에 민감 | **배치 크기 무관** |
| 가변 길이 | 시퀀스 길이 달라지면 문제 | **문제없음** |
| 추론 시 | running mean/var 필요 | **필요없음** |
| RNN/Transformer | 부적합 | **적합** |

```
예시: 가변 길이 시퀀스

Batch Norm 문제:
  문장 1: "I love AI" (3 토큰)
  문장 2: "The cat sat on the mat" (6 토큰)

  → 위치 4, 5, 6은 문장 1에 없음
  → 배치 통계 계산 불가능 또는 왜곡됨

Layer Norm:
  각 토큰 독립적으로 정규화
  → 시퀀스 길이 달라도 문제없음
```

---

#### 5. LayerNorm 계산 예시

**단계별 계산:**

입력 벡터 $x = [2, 4, 6, 8]$ (한 토큰의 feature)

**Step 1: 평균 계산**

$$\mu = \frac{1}{4}(2 + 4 + 6 + 8) = \frac{20}{4} = 5$$

**Step 2: 분산 계산**

$$\sigma^2 = \frac{1}{4}\left[(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2\right]$$

$$= \frac{1}{4}(9 + 1 + 1 + 9) = \frac{20}{4} = 5$$

$$\sigma = \sqrt{5} \approx 2.236$$

**Step 3: 정규화**

$$\hat{x} = \frac{x - \mu}{\sigma + \epsilon} = \frac{[2, 4, 6, 8] - 5}{2.236}$$

$$= \frac{[-3, -1, 1, 3]}{2.236} \approx [-1.34, -0.45, 0.45, 1.34]$$

**Step 4: Scale & Shift (학습 가능)**

$$y = \gamma \cdot \hat{x} + \beta$$

만약 $\gamma = [1, 1, 1, 1], \beta = [0, 0, 0, 0]$이면:

$$y \approx [-1.34, -0.45, 0.45, 1.34]$$

---

#### 6. $\gamma$와 $\beta$의 역할

**왜 학습 가능한 파라미터가 필요한가?**

정규화만 하면 표현력이 제한됩니다.

$$\hat{x} = \frac{x - \mu}{\sigma}$$

문제: 모든 출력이 평균 0, 분산 1로 강제됨

해결: $\gamma, \beta$로 네트워크가 최적의 분포를 학습

$$y = \gamma \cdot \hat{x} + \beta$$

```
           γ, β의 역할

 γ = 1, β = 0: 정규화된 상태 유지 (평균 0, 분산 1)

 γ = σ, β = μ: 원래 분포로 복원 가능!
               y = σ · (x-μ)/σ + μ = x

 → 네트워크가 정규화의 정도를 스스로 결정
 → 필요하면 원래대로 되돌릴 수 있음
```

---

#### 7. Transformer에서 LayerNorm의 위치

**Post-LN vs Pre-LN**

**Post-LN (원 논문):**

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

```
x → [Sublayer] → (+) → [LayerNorm] → Output
└──────────────────↑
```

**Pre-LN (현대 모델):**

$$\text{Output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

```
x → [LayerNorm] → [Sublayer] → (+) → Output
└────────────────────────────────↑
```

**비교:**

| 측면 | Post-LN | Pre-LN |
| --- | --- | --- |
| 학습 안정성 | 불안정할 수 있음 | 더 안정적 |
| Warmup 필요 | 필수 | 덜 민감 |
| 최종 성능 | 약간 높을 수 있음 | 비슷하거나 약간 낮음 |
| 깊은 모델 | 어려움 | 용이함 |
| 사용 모델 | 원본 Transformer | GPT-2, GPT-3, LLaMA |

**Pre-LN이 더 안정적인 이유:**

Post-LN의 gradient 경로:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{LN}} \cdot \frac{\partial \text{LN}}{\partial (x + F(x))} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

LayerNorm의 gradient가 residual path에 영향

Pre-LN의 gradient 경로:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{out}} \cdot 1 + \frac{\partial L}{\partial \text{out}} \cdot \frac{\partial F(\text{LN}(x))}{\partial x}$$

Residual path로 gradient가 직접 흐름 (1이 보장됨)

```
           Gradient 흐름 비교

 Post-LN:
 x ──→ [+] ──→ [LN] ──→ out
         ↑
         └── F(x)

 Gradient: LN을 통과해야 residual에 도달
 → LN의 gradient가 학습 초기에 불안정할 수 있음

 Pre-LN:
 x ────────────────→ [+] ──→ out
  └→ [LN] ──→ F(LN(x)) ──→↑

 Gradient: 직접 residual path로 흐름
 → 항상 안정적인 gradient 보장
```

---

#### 8. LayerNorm의 효과 실험

**학습 곡선 비교:**

```
              Training Loss

 Loss
   ↑
 5 │●
   │ \   LayerNorm 없음
 4 │  \
   │   \──────────
 3 │          \────────
   │                   \───── (느리고 불안정)
 2 │●
   │ \   LayerNorm 있음
 1 │  \────
   │       \───── (빠르고 안정적)
 0 │
   └─────────────────────────→
                         Epochs
```

**활성화 값 분포 비교:**

```
    LayerNorm 없음 (Layer 6 출력)

 확률밀도
   ↑          일부 값이 매우 크거나 작음
   │
   │  ▄       ▄
   │  █▄     ▄█
   └──────────────────→
  -100    0       100

 → 분포가 넓게 퍼짐
 → Gradient 불안정, 학습 어려움

    LayerNorm 있음 (Layer 6 출력)

 확률밀도
   ↑        ▄▄
   │      ▄████▄
   │    ▄████████▄
   │  ▄████████████▄
   └──────────────────→
      -3    0    3

 → 분포가 적절한 범위에 집중
 → 안정적인 학습 가능
```

---

#### 9. RMSNorm: LayerNorm의 변형

최근 모델(LLaMA, Gemma)에서 사용하는 간소화된 버전

**수식:**

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}$$

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2}$$

**LayerNorm vs RMSNorm:**

| 측면 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 평균 빼기 | O | X |
| 파라미터 | $\gamma, \beta$ | $\gamma$만 |
| 계산량 | 더 많음 | 더 적음 |
| 성능 | 기준 | 비슷하거나 약간 좋음 |

$$\text{LayerNorm}: \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

$$\text{RMSNorm}: \gamma \cdot \frac{x}{\text{RMS}(x)}$$

---

#### 10. 요약

**LayerNorm의 목적:**

| 목적 | 설명 |
| --- | --- |
| 학습 안정화 | 입력 분포를 일정하게 유지 |
| 빠른 수렴 | Internal Covariate Shift 해결 |
| 높은 learning rate | 안정적이므로 더 큰 lr 사용 가능 |
| 깊은 네트워크 | 레이어가 많아도 안정적 학습 |

**핵심 공식:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

**Transformer에서의 역할:**

```
입력 → [Attention] → (+) → [LayerNorm] → [FFN] → (+) → [LayerNorm] → 출력
                      ↑                                  ↑
               Residual Connection              Residual Connection
```

**Residual + LayerNorm 조합:**
- Residual: Gradient 흐름 보장
- LayerNorm: 값의 분포 안정화
- 함께 사용하여 깊은 Transformer 학습 가능

</details>

---

### 9. Transformer 동작 예시: 기계 번역

**입력:** "A cute teddy bear is reading" (영어)
**출력:** "Un mignon ours en peluche lit" (프랑스어)

```
Step 1: Tokenization
  Input: [BOS, A, cute, teddy, bear, is, reading, EOS]

Step 2: Embedding + Position Encoding
  각 토큰 → d_model 차원 벡터 + 위치 인코딩

Step 3: Encoder Processing
  Self-attention으로 각 토큰이 다른 모든 토큰 참조
  → 문맥을 반영한 풍부한 표현 생성

Step 4: Decoding 시작
  [BOS] 토큰으로 시작

Step 5: 첫 번째 토큰 생성
  1. Masked Self-Attention: [BOS]만 참조
  2. Cross-Attention: Encoder 출력 전체 참조
     → "A"에 높은 attention → "Un" 생성
  3. Softmax → "Un" 선택

Step 6: 반복
  [BOS, Un] → "mignon" 생성
  [BOS, Un, mignon] → "ours" 생성
  ...
  → [EOS] 토큰 생성 시 종료
```

---

### 10. 추가 기법: Label Smoothing

**문제:** NLP에서 "정답"이 여러 개일 수 있음
* "What a great \_\_\_" → day, lecture, book 모두 가능

**기존 방식:** Hard label (one-hot)

```
정답: [0, 0, 1, 0, 0]  (100% 확신)
```

**Label Smoothing:**

```
정답: [ε/V, ε/V, 1-ε, ε/V, ε/V]
```

**효과:**
* 모델이 과도하게 확신하지 않도록
* 일반화 성능 향상
* BLEU 점수 개선

---

## 핵심 요약

### NLP 발전 흐름

```
One-Hot → Word2Vec → RNN/LSTM → Attention → Transformer
  ↓          ↓          ↓          ↓           ↓
고정표현   의미학습   순서고려   직접연결    병렬처리
```

### Transformer의 혁신

| 이전 방식 (RNN) | Transformer |
| --- | --- |
| 순차 처리 | 병렬 처리 |
| 간접 연결 | 직접 연결 (Attention) |
| Vanishing gradient | 안정적 학습 |
| 느린 학습 | 빠른 학습 |
| 제한된 문맥 | 전체 문맥 참조 |

### 핵심 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 용어 정리

| 용어 | 의미 |
| --- | --- |
| NLP | Natural Language Processing |
| OOV | Out of Vocabulary |
| BOS/EOS | Beginning/End of Sequence |
| Q, K, V | Query, Key, Value |
| FFN | Feed-Forward Network |
| BLEU | Bilingual Evaluation Understudy |

---

### 읽어볼 자료

1. **"Attention Is All You Need"** (2017) - Transformer 원본 논문
2. **"Efficient Estimation of Word Representations in Vector Space"** (2013) - Word2Vec
3. **"Neural Machine Translation by Jointly Learning to Align and Translate"** (2014) - Attention 도입

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 1 정리*
