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

**혼동 행렬 구조:**

|  | **예측: Positive** | **예측: Negative** |
| --- | --- | --- |
| **실제: Positive** | TP (True Positive) | FN (False Negative) |
| **실제: Negative** | FP (False Positive) | TN (True Negative) |

**지표 계산:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(예측한 것 중 맞은 비율)}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(실제 양성 중 찾아낸 비율)}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

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

$q$와 $k$의 각 성분이 독립적이고 평균 0, 분산 1인 확률변수라고 가정하면:

$$q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i$$

각 $q_i \cdot k_i$의 기댓값과 분산:

$$E[q_i \cdot k_i] = E[q_i] \cdot E[k_i] = 0$$

$$\text{Var}(q_i \cdot k_i) = E[q_i^2] \cdot E[k_i^2] = 1 \cdot 1 = 1$$

$d_k$개를 합산하면:

$$\text{Var}(q \cdot k) = d_k$$

따라서 dot product의 표준편차는 $\sqrt{d_k}$에 비례합니다. $d_k$가 크면 dot product 값이 매우 커져 softmax의 gradient가 소실됩니다.

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

$\sqrt{d_k}$로 나누면 분산이 1로 정규화되어 안정적인 softmax 분포를 얻습니다.

</details>

<details>
<summary>분산의 기본 성질</summary>

**독립 확률변수의 분산 성질:**

$$\text{Var}(aX) = a^2 \text{Var}(X)$$

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) \quad \text{(X, Y 독립)}$$

$$\text{Var}(XY) = E[X^2]E[Y^2] - (E[X])^2(E[Y])^2$$

$X, Y$가 평균 0, 분산 1이면:

$$\text{Var}(XY) = E[X^2] \cdot E[Y^2] = 1 \cdot 1 = 1$$

이를 $d_k$개 합산하면: $\text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$

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

**각 Head의 Projection 행렬:**

$$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{model} \times d_v}$$

여기서 $d_k = d_v = d_{model} / h$

**Output Projection 행렬:**

$$W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$$

**전체 연산:**

1. 각 Head $i$에서 입력 $X$를 각각의 $W_i^Q, W_i^K, W_i^V$로 projection
2. 각 Head에서 독립적으로 Attention 계산
3. 모든 Head의 출력을 Concatenate: $(n \times h \cdot d_v)$
4. $W^O$로 원래 차원 복원: $(n \times d_{model})$

**파라미터 수 비교:** Single-Head와 Multi-Head의 총 파라미터 수는 동일합니다. Multi-Head는 같은 파라미터를 여러 개의 부분공간(subspace)으로 나누어 다양한 관계를 학습합니다.

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

**Step 1: Input Embedding + Position Encoding**

$$\text{Input}_i = \text{Embed}(x_i) + \text{PE}(i) \in \mathbb{R}^{d_{model}}$$

입력 시퀀스의 각 토큰을 $d_{model}$ 차원 벡터로 변환하고 위치 정보를 더합니다.

**Step 2: Multi-Head Self-Attention**

각 토큰이 다른 모든 토큰을 참조하여 문맥 정보를 수집합니다.

$$\text{Attn}(X) = \text{MultiHead}(X, X, X)$$

Q, K, V 모두 같은 입력 $X$에서 생성되므로 "Self"-Attention입니다.

**Step 3: Add & Norm (첫 번째)**

$$\text{Out}_1 = \text{LayerNorm}(X + \text{Attn}(X))$$

잔차 연결로 원래 입력 정보를 보존하면서 Attention 결과를 더합니다.

**Step 4: Feed-Forward Network**

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

각 토큰 위치에 독립적으로 적용됩니다 (토큰 간 상호작용 없음).

**Step 5: Add & Norm (두 번째)**

$$\text{Out}_2 = \text{LayerNorm}(\text{Out}_1 + \text{FFN}(\text{Out}_1))$$

이 과정이 $N$번 반복되어 점점 더 풍부한 표현을 생성합니다.

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

**Cross-Attention vs Self-Attention:**

Self-Attention에서는 Q, K, V 모두 같은 시퀀스에서 옵니다. Cross-Attention에서는 Q는 Decoder, K와 V는 Encoder에서 옵니다.

$$Q = H_{dec} \cdot W^Q \quad \text{(Decoder hidden states)}$$

$$K = H_{enc} \cdot W^K, \quad V = H_{enc} \cdot W^V \quad \text{(Encoder output)}$$

$$\text{CrossAttn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**직관:** Decoder의 현재 생성 위치(Query)가 "원문(Encoder)의 어떤 부분을 참조할까?"를 결정합니다.

예: 영→불 번역에서 "chat" (고양이)를 생성할 때, Encoder의 "cat" 위치에 높은 Attention을 줍니다.

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

**핵심 질문:** 왜 concatenate가 아니라 addition인가?

**이유 1: 고차원 공간의 직교성**

$d_{model}$이 충분히 크면 (예: 512), Token Embedding과 Position Encoding이 차지하는 부분공간이 거의 직교합니다. 따라서 덧셈 후에도 각 정보를 분리하여 추출할 수 있습니다.

$$\text{Input} = \text{Embed}(x) + \text{PE}(pos)$$

Attention의 학습 가능한 $W^Q, W^K$ 행렬이 두 성분을 분리/조합하는 역할을 합니다.

**이유 2: 효율성**

Concatenation은 차원을 $2 \times d_{model}$로 늘려 연산 비용을 증가시킵니다. Addition은 차원을 유지하면서도 실험적으로 동등한 성능을 보입니다.

**이유 3: Attention에서의 상호작용**

$Q \cdot K^T$를 전개하면:

$$(E_q + P_q)(E_k + P_k)^T = E_q E_k^T + E_q P_k^T + P_q E_k^T + P_q P_k^T$$

네 가지 항이 나타나며, 이는 의미(E)와 위치(P) 정보의 모든 조합을 자연스럽게 포착합니다.

</details>

<details>
<summary>Positional Encoding 특성 상세 (상대적 위치를 선형 변환으로 표현 가능한 이유)</summary>

**Sinusoidal PE의 핵심 성질:**

위치 $pos + k$의 인코딩을 위치 $pos$의 인코딩으로부터 선형 변환으로 얻을 수 있습니다:

$$PE(pos + k) = T_k \cdot PE(pos)$$

**증명 (2D 간소화):**

삼각함수 덧셈 공식을 사용하면:

$$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$

$$\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)$$

행렬로 정리하면:

$$\begin{pmatrix} \sin(pos+k) \\ \cos(pos+k) \end{pmatrix} = \begin{pmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{pmatrix} \begin{pmatrix} \sin(pos) \\ \cos(pos) \end{pmatrix}$$

$T_k$는 $k$에만 의존하는 회전 행렬이므로, 모델은 상대적 위치 $k$를 학습할 수 있습니다.

</details>

#### 8.5 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**특징:**
* Hidden dimension이 입력보다 큼 (보통 4배)
* 예: $d_{model} = 512 \rightarrow d_{ff} = 2048$
* 더 풍부한 표현 학습을 위한 "expansion"

<details>
<summary>FFN 적용 이유 상세</summary>

**왜 Self-Attention 후에 FFN이 필요한가?**

Self-Attention은 토큰 간의 관계를 학습하지만, 각 토큰의 표현을 개별적으로 변환하는 능력이 제한됩니다. FFN이 이를 보완합니다.

**역할 분담:**

| 구성요소 | 역할 |
| --- | --- |
| **Self-Attention** | 토큰 **간** 상호작용 (어떤 토큰이 중요한가?) |
| **FFN** | 토큰 **내** 변환 (수집한 정보를 어떻게 처리할 것인가?) |

**Expansion & Contraction:**

$$x \in \mathbb{R}^{d_{model}} \xrightarrow{W_1} h \in \mathbb{R}^{d_{ff}} \xrightarrow{\text{ReLU}} h' \xrightarrow{W_2} y \in \mathbb{R}^{d_{model}}$$

1. $W_1$: $d_{model} \rightarrow d_{ff}$ (확장, 보통 4배)
2. ReLU: 비선형 활성화
3. $W_2$: $d_{ff} \rightarrow d_{model}$ (축소)

**확장하는 이유:** 더 높은 차원에서 비선형 변환을 수행하면 더 복잡한 패턴을 학습할 수 있습니다. 이후 다시 축소하여 원래 차원으로 돌아옵니다.

**Position-wise 적용:** FFN은 각 토큰 위치에 **동일한 가중치**로 **독립적**으로 적용됩니다. 즉, 토큰 간 상호작용은 Attention에서만 일어납니다.

**비유:** Self-Attention이 "회의에서 정보 수집"이라면, FFN은 "수집한 정보를 개인적으로 정리하고 소화하는 단계"입니다.

</details>

#### 8.6 Add & Norm (Residual Connection + Layer Norm)

**Residual Connection:**

$$\text{output} = x + \text{SubLayer}(x)$$

* Gradient flow 개선
* 깊은 네트워크 학습 가능

<details>
<summary>Residual Connection 적용 이유 상세</summary>

**문제: 깊은 네트워크의 학습 어려움**

네트워크가 깊어지면 gradient가 역전파될 때 점점 작아지거나 커져서 학습이 불안정해집니다.

**Residual Connection의 해결:**

$$y = F(x) + x$$

역전파 시:

$$\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$$

항상 1이 더해지므로 gradient가 최소 1은 보장됩니다. 이는 "gradient highway"를 만들어 깊은 층까지 gradient가 잘 전달됩니다.

**Identity Mapping 관점:**

Sub-layer가 학습해야 할 것은 $F(x) = y - x$ (잔차, residual)입니다. "원래 입력에서 얼마나 변해야 하는가?"만 학습하면 되므로 최적화가 쉬워집니다.

</details>

**Layer Normalization:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

* 학습 안정화
* 빠른 수렴

<details>
<summary>LayerNorm 적용 이유 상세</summary>

**왜 Normalization이 필요한가?**

각 층을 거치면서 활성값의 분포가 변화(Internal Covariate Shift)하여 학습이 불안정해질 수 있습니다.

**Layer Norm vs Batch Norm:**

| 특성 | Batch Norm | Layer Norm |
| --- | --- | --- |
| **정규화 축** | 배치 차원 (같은 feature, 다른 샘플) | 특성 차원 (같은 샘플, 다른 feature) |
| **배치 크기 의존** | 의존 (작은 배치에서 불안정) | **비의존** |
| **시퀀스 길이** | 가변 길이 처리 어려움 | **가변 길이 자연스럽게 처리** |
| **추론 시** | Running statistics 필요 | 입력만으로 계산 가능 |

**Layer Norm 계산:**

각 토큰 벡터 $x \in \mathbb{R}^{d_{model}}$에 대해:

$$\mu = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} x_i, \quad \sigma = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} (x_i - \mu)^2}$$

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

$\gamma, \beta$는 학습 가능한 파라미터로, 정규화 후 표현력을 복원합니다.

**Transformer에서 Layer Norm을 쓰는 이유:** NLP에서는 배치 내 시퀀스 길이가 다르고, 같은 위치의 feature가 배치 간 의미가 다를 수 있어 Batch Norm이 적합하지 않습니다.

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
