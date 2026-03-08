---
layout: post
title: "Stanford CME295: Lecture 1 - Transformer 기초"
date: 2026-03-08 10:10:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, transformer, attention, nlp, word2vec, rnn, self-attention]
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

![평가 지표](/assets/img/cme295-lecture-1/image-20260112-045128.png)

<details>
<summary>혼동 행렬(Confusion Matrix) 상세</summary>

![Confusion Matrix](/assets/img/cme295-lecture-1/image-20260112-104237.png)

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

$$\text{Perplexity} = 2^{-\frac{1}{N}\sum\_{i=1}^{N}\log\_2 P(w\_i)}$$

![Perplexity](/assets/img/cme295-lecture-1/image-20260112-044646.png)

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

![Cosine Similarity](/assets/img/cme295-lecture-1/image-20260112-044716.png)

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

$$h\_t = f(W\_h \cdot h\_{t-1} + W\_x \cdot x\_t + b)$$

![RNN 구조](/assets/img/cme295-lecture-1/image-20260112-044732.png)

**장점:** 단어 순서 고려, 가변 길이 입력 처리, 문맥 인코딩

**단점:**

1. **Vanishing Gradient Problem:** 역전파 시 그래디언트가 시간에 따라 지수적 감소

$$\frac{\partial L}{\partial h\_0} = \frac{\partial L}{\partial h\_T} \cdot \prod\_{t=1}^{T} \frac{\partial h\_t}{\partial h\_{t-1}}$$

![Vanishing Gradient](/assets/img/cme295-lecture-1/image-20260112-044743.png)

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

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

**2단계: Attention Score 계산**

$$\text{Score} = Q \cdot K^T$$

![Attention Score](/assets/img/cme295-lecture-1/image-20260112-044809.png)

**3단계: Scaling**

$$\text{Scaled Score} = \frac{Q \cdot K^T}{\sqrt{d\_k}}$$

![Scaling](/assets/img/cme295-lecture-1/image-20260112-044816.png)

**왜 √d\_k로 나누는가?**
* Dot product는 차원이 커질수록 값이 커짐
* 큰 값은 softmax를 극단적으로 만듦 (한 곳에 집중)
* Scaling으로 안정적인 학습

<details>
<summary>스케일링이 필요한 이유 상세 (수식 포함)</summary>

![스케일링 상세 1](/assets/img/cme295-lecture-1/image-20260109-031626.png)
![스케일링 상세 2](/assets/img/cme295-lecture-1/image-20260109-031643.png)
![스케일링 상세 3](/assets/img/cme295-lecture-1/image-20260109-031657.png)

</details>

<details>
<summary>분산의 기본 성질</summary>

![분산 기본 성질](/assets/img/cme295-lecture-1/image-20260109-040543.png)

</details>

**4단계: Softmax로 가중치 변환**

$$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)$$

![Softmax](/assets/img/cme295-lecture-1/image-20260112-044825.png)

**5단계: Value와 가중합**

$$\text{Output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right) \cdot V$$

![Value 가중합](/assets/img/cme295-lecture-1/image-20260112-044832.png)

#### 6.3 Self-Attention 전체 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$$

![Self-Attention 전체](/assets/img/cme295-lecture-1/image-20260112-044840.png)

**행렬 차원 분석:**

* Q: (n × d\_k)
* K: (n × d\_k), K^T: (d\_k × n)
* V: (n × d\_v)
* QK^T: (n × n) — Attention map
* Output: (n × d\_v)

---

### 7. Multi-Head Attention

**왜 여러 Head가 필요한가?**
* 다양한 관점에서 관계 학습
* 예: 하나의 head는 문법적 관계, 다른 head는 의미적 관계

**수식:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, ..., \text{head}\_h) \cdot W^O$$

$$\text{head}\_i = \text{Attention}(QW\_i^Q, KW\_i^K, VW\_i^V)$$

![Multi-Head Attention](/assets/img/cme295-lecture-1/image-20260112-044851.png)

**파라미터:**
* h: head 수 (보통 8 또는 12)
* 각 head의 차원: d\_model / h

**비유 (Computer Vision):** CNN의 multiple filters처럼, 각 head가 다른 "필터"로 관계를 학습

<details>
<summary>Multi-Head Attention의 Weight 행렬 상세</summary>

![MHA Weight 1](/assets/img/cme295-lecture-1/image-20260113-002546.png)
![MHA Weight 2](/assets/img/cme295-lecture-1/image-20260113-002608.png)

</details>

---

### 8. Transformer Architecture

#### 8.1 전체 구조

![Transformer 전체 구조](/assets/img/cme295-lecture-1/image-20260112-091630.png)

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

![Encoder 상세 1](/assets/img/cme295-lecture-1/image-20260109-065647.png)
![Encoder 상세 2](/assets/img/cme295-lecture-1/image-20260109-065706.png)
![Encoder 상세 3](/assets/img/cme295-lecture-1/image-20260109-065722.png)

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

![Cross-Attention 1](/assets/img/cme295-lecture-1/image-20260113-020435.png)
![Cross-Attention 2](/assets/img/cme295-lecture-1/image-20260113-020500.png)

</details>

#### 8.4 Position Encoding

**문제:** Self-attention은 순서 정보가 없음

**Sinusoidal Position Encoding (원본 논문):**

$$PE\_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right)$$

$$PE\_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right)$$

![Position Encoding](/assets/img/cme295-lecture-1/image-20260112-044907.png)

**최종 입력:**

$$\text{Input} = \text{Token Embedding} + \text{Position Encoding}$$

![최종 입력](/assets/img/cme295-lecture-1/image-20260112-044915.png)

<details>
<summary>Positional Encoding을 Token Embedding에 더하기만 해도 위치 정보가 손실되지 않는 이유</summary>

![PE 더하기 1](/assets/img/cme295-lecture-1/image-20260113-021039.png)
![PE 더하기 2](/assets/img/cme295-lecture-1/image-20260113-021055.png)
![PE 더하기 3](/assets/img/cme295-lecture-1/image-20260113-021112.png)
![PE 더하기 4](/assets/img/cme295-lecture-1/image-20260113-021127.png)
![PE 더하기 5](/assets/img/cme295-lecture-1/image-20260113-021142.png)

</details>

<details>
<summary>Positional Encoding 특성 상세 (상대적 위치를 선형 변환으로 표현 가능한 이유)</summary>

![PE 특성 1](/assets/img/cme295-lecture-1/image-20260113-022545.png)
![PE 특성 2](/assets/img/cme295-lecture-1/image-20260113-022606.png)

</details>

#### 8.5 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \max(0, xW\_1 + b\_1)W\_2 + b\_2$$

![FFN](/assets/img/cme295-lecture-1/image-20260112-044926.png)

**특징:**
* Hidden dimension이 입력보다 큼 (보통 4배)
* 예: d\_model = 512 → d\_ff = 2048
* 더 풍부한 표현 학습을 위한 "expansion"

<details>
<summary>FFN 적용 이유 상세</summary>

![FFN 상세 1](/assets/img/cme295-lecture-1/image-20260112-022342.png)
![FFN 상세 2](/assets/img/cme295-lecture-1/image-20260112-011050.png)
![FFN 상세 3](/assets/img/cme295-lecture-1/image-20260112-011107.png)
![FFN 상세 4](/assets/img/cme295-lecture-1/image-20260112-011126.png)
![FFN 상세 5](/assets/img/cme295-lecture-1/image-20260112-011145.png)
![FFN 상세 6](/assets/img/cme295-lecture-1/image-20260112-011201.png)
![FFN 상세 7](/assets/img/cme295-lecture-1/image-20260112-011218.png)
![FFN 상세 8](/assets/img/cme295-lecture-1/image-20260112-011233.png)
![FFN 상세 9](/assets/img/cme295-lecture-1/image-20260112-011248.png)
![FFN 상세 10](/assets/img/cme295-lecture-1/image-20260112-011303.png)
![FFN 상세 11](/assets/img/cme295-lecture-1/image-20260112-011317.png)

</details>

#### 8.6 Add & Norm (Residual Connection + Layer Norm)

**Residual Connection:**

$$\text{output} = x + \text{SubLayer}(x)$$

![Residual Connection](/assets/img/cme295-lecture-1/image-20260112-044938.png)

* Gradient flow 개선
* 깊은 네트워크 학습 가능

<details>
<summary>Residual Connection 적용 이유 상세</summary>

![Residual 상세 1](/assets/img/cme295-lecture-1/image-20260112-002154.png)
![Residual 상세 2](/assets/img/cme295-lecture-1/image-20260112-002243.png)
![Residual 상세 3](/assets/img/cme295-lecture-1/image-20260112-002306.png)
![Residual 상세 4](/assets/img/cme295-lecture-1/image-20260112-002400.png)

</details>

**Layer Normalization:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

![Layer Normalization](/assets/img/cme295-lecture-1/image-20260112-044947.png)

* 학습 안정화
* 빠른 수렴

<details>
<summary>LayerNorm 적용 이유 상세</summary>

![LayerNorm 상세 1](/assets/img/cme295-lecture-1/image-20260112-003657.png)
![LayerNorm 상세 2](/assets/img/cme295-lecture-1/image-20260112-003713.png)
![LayerNorm 상세 3](/assets/img/cme295-lecture-1/image-20260112-003732.png)
![LayerNorm 상세 4](/assets/img/cme295-lecture-1/image-20260112-003749.png)
![LayerNorm 상세 5](/assets/img/cme295-lecture-1/image-20260112-003828.png)
![LayerNorm 상세 6](/assets/img/cme295-lecture-1/image-20260112-003843.png)

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

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$$

![핵심 공식](/assets/img/cme295-lecture-1/image-20260112-045007.png)

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
