---
layout: post
title: "Stanford CME295: Lecture 2 - Transformer-Based Models & Tricks"
date: 2026-03-08 10:20:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, transformer, rope, rmsnorm, gqa, bert, distilbert, roberta]
---

> **강의 출처**: Stanford CME295 - Transformers & LLMs (Autumn 2025)
>
> - **강사**: Afshine & Shervine Amidi
> - **원본 영상**: [YouTube](https://www.youtube.com/watch?v=yT84Y5zCnaA&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=2)

---

## 강의 개요

이 강의는 2017년에 소개된 Transformer 아키텍처가 현재까지 어떻게 발전해왔는지를 다룹니다. 원본 Transformer의 핵심 구성요소들이 어떻게 변화했는지, 그리고 오늘날의 모델들이 어떤 형태를 취하고 있는지 설명합니다.

**강의 구성:**

* Part 1 (Afin): Transformer의 주요 구성요소 변화
* Part 2 (Shervin): 현대 모델들의 분류와 BERT 심층 분석

---

## Part 1: Transformer 구성요소의 변화

### 1. Lecture 1 복습: Self-Attention

**핵심 공식:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$$

![Self-Attention 복습](/assets/img/cme295-lecture-2/image-20260112-043604.png)

**Attention Map 해석:**

* 논문에서 "its"라는 토큰이 "law"와 "application"에 높은 attention weight를 가짐
* 이는 모델이 대명사와 그것이 참조하는 단어의 관계를 학습했음을 보여줌

---

### 2. Position Embeddings (위치 임베딩)

Transformer는 RNN과 달리 토큰을 순차적으로 처리하지 않아 위치 정보가 자연스럽게 인코딩되지 않습니다. 따라서 별도의 위치 정보 주입이 필요합니다.

#### 2.1 Learned Position Embeddings (학습된 위치 임베딩)

$$\text{Input}\_i = \text{Token Embedding}\_i + \text{Position Embedding}\_i$$

![Learned PE](/assets/img/cme295-lecture-2/image-20260112-043632.png)

**장점:** Gradient descent를 통해 데이터로부터 자연스럽게 학습

**단점:**

1. 학습 데이터에 의존적 (overfitting 가능성)
2. 학습 시 본 최대 위치까지만 학습 가능
3. 추론 시 학습하지 않은 위치에 대해서는 일반화 불가

#### 2.2 Sinusoidal Position Embeddings

**원본 Transformer 논문에서 제안한 방법**

$$PE\_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right)$$

$$PE\_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right)$$

![Sinusoidal PE](/assets/img/cme295-lecture-2/image-20260112-043621.png)

**수학적 근거:**

삼각함수 항등식을 활용:

$$\cos(\omega\_i(m-n)) = \cos(\omega\_i m)\cos(\omega\_i n) + \sin(\omega\_i m)\sin(\omega\_i n)$$

![삼각함수 항등식](/assets/img/cme295-lecture-2/image-20260112-043220.png)

두 위치 임베딩 $P\_m$과 $P\_n$의 dot product:

$$P\_m \cdot P\_n = \sum\_i \cos(\omega\_i(m-n))$$

![Dot product](/assets/img/cme295-lecture-2/image-20260112-043649.png)

이는 상대적 거리 $(m-n)$의 함수!

**장점:**
* 학습 시 보지 못한 길이로도 확장 가능
* 상대적 위치 정보를 자연스럽게 인코딩

#### 2.3 Relative Position Bias (T5)

**핵심 아이디어:** 위치 정보를 입력에 추가하는 대신, Attention 계산에 직접 주입

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}} + B\right)V$$

![T5 Relative Bias](/assets/img/cme295-lecture-2/image-20260112-043702.png)

여기서 $B$는 학습 가능한 bias term으로, 상대적 거리에 따라 값이 결정됨

#### 2.4 ALiBi (Attention with Linear Bias)

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}} - m \cdot |i-j|\right)V$$

![ALiBi](/assets/img/cme295-lecture-2/image-20260112-043710.png)

**특징:**
* 학습 없이 결정론적 공식 사용
* 상대적 거리에 비례하는 선형 penalty
* 단순하지만 제한적

#### 2.5 RoPE (Rotary Position Embeddings) - 현재 가장 많이 사용

**핵심 아이디어:** Query와 Key 벡터를 위치에 따른 각도로 회전

**2D에서의 회전 행렬:**

$$R\_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

![회전 행렬](/assets/img/cme295-lecture-2/image-20260112-043718.png)

벡터 $v$를 각도 $\theta$만큼 회전:

$$R\_\theta \cdot v = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

![벡터 회전](/assets/img/cme295-lecture-2/image-20260112-043727.png)

**RoPE의 적용:**

* Query $q\_m$을 위치 $m$에 해당하는 각도로 회전
* Key $k\_n$을 위치 $n$에 해당하는 각도로 회전
* 결과적으로 $q\_m \cdot k\_n$은 상대적 거리 $(m-n)$의 함수가 됨

**수학적 결과:**

$$q\_m^T k\_n = (R\_{\theta m} q)^T (R\_{\theta n} k) \propto f(m-n)$$

![RoPE 결과](/assets/img/cme295-lecture-2/image-20260112-043335.png)

**장점:**

1. 상대적 거리 정보가 자연스럽게 인코딩
2. Long-term decay: 멀리 떨어진 토큰일수록 attention weight의 상한이 감소
3. 대부분의 현대 LLM에서 사용 (LLaMA, GPT-4 등)

---

### 3. Layer Normalization

#### 3.1 기본 개념

**Layer Norm 공식:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

![Layer Normalization](/assets/img/cme295-lecture-2/image-20260112-043440.png)

#### 3.2 Batch Norm vs Layer Norm

| 구분 | Batch Normalization | Layer Normalization |
| --- | --- | --- |
| 정규화 대상 | 배치 내 같은 차원 | 단일 샘플 내 모든 차원 |
| 의존성 | 배치 크기에 의존 | 배치 크기와 무관 |
| 추론 시 | 학습 시 통계 사용 | 현재 입력으로 계산 |
| Transformer | 잘 사용 안함 | 표준으로 사용 |

#### 3.3 Post-Norm vs Pre-Norm

**Original Transformer (Post-Norm):**

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

![Post-Norm](/assets/img/cme295-lecture-2/image-20260112-043454.png)

**현대 모델들 (Pre-Norm):**

$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$$

![Pre-Norm](/assets/img/cme295-lecture-2/image-20260112-043502.png)

**Pre-Norm의 장점:**
* 더 안정적인 학습
* Gradient flow 개선
* 대부분의 현대 LLM에서 채택

#### 3.4 RMSNorm (Root Mean Square Normalization)

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum\_{i=1}^{n}x\_i^2 + \epsilon}} \cdot \gamma$$

![RMSNorm](/assets/img/cme295-lecture-2/image-20260112-043511.png)

**특징:**
* Mean centering을 생략 (계산 효율성)
* 실험적으로 성능 차이 거의 없음
* LLaMA 등 현대 모델에서 사용

---

### 4. Attention Variations

#### 4.1 Full Attention의 문제

**복잡도:** $O(n^2)$ — 시퀀스가 길어지면 메모리와 계산 비용이 급격히 증가

#### 4.2 Sliding Window Attention (Local Attention)

* 윈도우 크기 $w$ 내의 토큰만 attend
* 복잡도: $O(n \cdot w)$
* Layer가 쌓이면서 effective receptive field 확장
* **Mistral**: 모든 layer에서 sliding window attention 사용

#### 4.3 Grouped Query Attention (GQA) & Multi-Query Attention (MQA)

**배경:** KV Cache 최적화 필요

디코딩 시 매번 이전 토큰들의 Key, Value를 재계산하는 것은 비효율적 → KV Cache에 저장하여 재사용 → 메모리 사용량 문제 발생

**Multi-Head Attention (MHA) - 원본:**
* 각 head마다 별도의 $W\_Q$, $W\_K$, $W\_V$
* KV Cache 크기: $O(n \cdot h \cdot d)$

**Multi-Query Attention (MQA):**
* 모든 head가 동일한 $W\_K$, $W\_V$ 공유
* Query만 head별로 다름
* KV Cache 크기: $O(n \cdot d)$ (h배 감소)

**Grouped Query Attention (GQA):**
* MHA와 MQA의 중간
* $G$개의 그룹으로 나누어, 같은 그룹 내 head들이 $W\_K$, $W\_V$ 공유
* KV Cache 크기: $O(n \cdot G \cdot d)$

**왜 Query는 공유하지 않는가?**

1. Query는 "무엇을 찾을지"를 결정하므로 다양성이 중요
2. Key, Value는 디코딩 시 반복 사용되므로 캐싱 효율이 더 중요

---

## Part 2: Transformer 모델 분류와 BERT

### 5. Transformer 아키텍처 분류

#### 5.1 Encoder-Decoder Models

**대표 모델: T5 (Text-to-Text Transfer Transformer)**

* 모든 NLP 태스크를 text-to-text 형태로 통일

**T5 Family:** T5, mT5 (다국어), ByT5 (byte 레벨, vocab size: 256)

**T5의 훈련 방식: Span Corruption**

```
원본: "My teddy bear is cute and reading"
입력: "My teddy bear [X] and reading"
출력: "[X] is cute [Y]"
```

#### 5.2 Encoder-Only Models

* 텍스트 생성 불가 (Decoder 없음)
* 분류, 임베딩 추출 등에 적합
* Bidirectional attention (양방향)
* 대표: **BERT**, DistilBERT, RoBERTa

#### 5.3 Decoder-Only Models (현대 LLM의 주류)

* Masked Self-Attention (causal)
* 텍스트 생성에 최적화
* 대표: GPT 시리즈, LLaMA, Claude

**왜 Decoder-Only가 대세가 되었나?**
1. 스케일링에 유리 (단순한 구조)
2. 생성 태스크의 중요성 증가
3. In-context learning 능력

---

### 6. BERT 심층 분석

#### 6.1 BERT 개요

**BERT = Bidirectional Encoder Representations from Transformers**

**핵심 혁신:**
1. Bidirectional context (양방향 문맥)
2. Pre-training + Fine-tuning 패러다임
3. 다양한 downstream task에 적용 가능

#### 6.2 BERT의 특수 토큰

| 토큰 | 역할 |
| --- | --- |
| **[CLS]** | 시퀀스 맨 앞, 분류 태스크 시 전체 시퀀스 정보 집약 |
| **[SEP]** | 두 문장 구분, NSP 태스크에서 문장 경계 표시 |
| **[MASK]** | MLM 태스크에서 masking된 위치 표시 |
| **[PAD]** | 배치 처리를 위한 패딩 |

<details>
<summary>PAD Token 상세</summary>

![PAD Token 1](/assets/img/cme295-lecture-2/image-20260113-102834.png)
![PAD Token 2](/assets/img/cme295-lecture-2/image-20260113-102849.png)

</details>

#### 6.3 BERT의 입력 표현

**세 가지 임베딩의 합:**

$$\text{Input} = \text{Token Embedding} + \text{Position Embedding} + \text{Segment Embedding}$$

![BERT 입력 표현](/assets/img/cme295-lecture-2/image-20260113-074907.png)

**Token Embedding:** WordPiece tokenizer 사용 (vocab size ~30K)

<details>
<summary>Token Embedding 학습 과정</summary>

![Token Embedding 1](/assets/img/cme295-lecture-2/image-20260113-105044.png)
![Token Embedding 2](/assets/img/cme295-lecture-2/image-20260113-105101.png)
![Token Embedding 3](/assets/img/cme295-lecture-2/image-20260113-105125.png)
![Token Embedding 4](/assets/img/cme295-lecture-2/image-20260113-105139.png)

</details>

**Segment Embedding:** 문장 A vs 문장 B 구분, NSP 태스크를 위해 도입

<details>
<summary>Segment Embedding 상세</summary>

![Segment 1](/assets/img/cme295-lecture-2/image-20260113-110103.png)
![Segment 2](/assets/img/cme295-lecture-2/image-20260113-110117.png)
![Segment 3](/assets/img/cme295-lecture-2/image-20260113-110248.png)
![Segment 4](/assets/img/cme295-lecture-2/image-20260113-110309.png)

</details>

#### 6.4 BERT Pre-training

**두 가지 목표 함수를 동시에 학습:**

##### MLM (Masked Language Model)

1. 입력 토큰 중 15%를 선택
2. 선택된 토큰 중:
   * 80%: [MASK]로 교체
   * 10%: 랜덤 토큰으로 교체
   * 10%: 그대로 유지
3. 모델이 원래 토큰 예측

**왜 이런 비율?**
* 100% masking: [MASK] 토큰은 fine-tuning 시 없음 → 불일치
* Random 교체: 모델이 모든 토큰에 주의 기울이도록
* 유지: 실제 토큰에 대한 representation 학습

##### NSP (Next Sentence Prediction)

1. 두 문장 A, B 선택
2. 50%: B는 실제로 A 다음 문장
3. 50%: B는 랜덤 문장
4. [CLS] 토큰으로 연속 여부 이진 분류

**참고:** 후속 연구(RoBERTa)에서 NSP의 효과에 의문 제기

#### 6.5 BERT Fine-tuning

| 태스크 | 사용하는 출력 | 예시 |
| --- | --- | --- |
| 문장 분류 | [CLS] 임베딩 | 감성 분석 |
| 토큰 분류 | 각 토큰 임베딩 | NER, POS tagging |
| 질의응답 | 토큰 임베딩 | 답변 시작/끝 위치 예측 |
| 문장 쌍 분류 | [CLS] 임베딩 | NLI, 유사도 |

<details>
<summary>CLS 토큰 활용 상세</summary>

![CLS 1](/assets/img/cme295-lecture-2/image-20260113-111153.png)
![CLS 2](/assets/img/cme295-lecture-2/image-20260113-111214.png)
![CLS 3](/assets/img/cme295-lecture-2/image-20260113-111231.png)
![CLS 4](/assets/img/cme295-lecture-2/image-20260113-111251.png)

</details>

#### 6.6 BERT 모델 크기

| 모델 | Layers | Hidden | Heads | 파라미터 |
| --- | --- | --- | --- | --- |
| BERT-Base | 12 | 768 | 12 | ~110M |
| BERT-Large | 24 | 1024 | 16 | ~340M |

---

### 7. BERT 변형 모델들

#### 7.1 DistilBERT (Knowledge Distillation)

> "The soft targets contain almost all the knowledge." — Hinton, Vinyals, Dean

**Teacher-Student 학습:**
1. **Teacher**: 큰 모델 (BERT)
2. **Student**: 작은 모델 (DistilBERT)
3. Student가 Teacher의 출력 분포를 학습

**목표 함수: KL Divergence**

$$\mathcal{L} = KL(P\_{teacher} \| P\_{student})$$

![KL Divergence](/assets/img/cme295-lecture-2/image-20260112-043538.png)

<details>
<summary>KL Divergence 손실 함수 상세</summary>

![KL 상세 1](/assets/img/cme295-lecture-2/image-20260113-112004.png)
![KL 상세 2](/assets/img/cme295-lecture-2/image-20260113-112019.png)
![KL 상세 3](/assets/img/cme295-lecture-2/image-20260113-112034.png)
![KL 상세 4](/assets/img/cme295-lecture-2/image-20260113-112048.png)

</details>

**DistilBERT 결과:**
* Layer 수를 절반으로 줄임 (12 → 6)
* 97% 성능 유지
* 60% 더 빠름
* 40% 더 작음

#### 7.2 RoBERTa (Robustly Optimized BERT)

**주요 변경사항:**

1. **NSP 제거:** 실험 결과 성능에 도움이 안 됨
2. **Dynamic Masking:** 매 epoch마다 다른 masking (BERT는 고정)
3. **더 많은 데이터:** BERT가 undertrained였음을 발견
4. **더 긴 학습:** 더 많은 step, 더 큰 batch size

**결과:** 같은 아키텍처로 BERT 대비 상당한 성능 향상 → Pre-training recipe의 중요성 입증

---

### 8. BERT의 한계

| 한계 | 해결책 |
| --- | --- |
| Context Length 제한 (512 tokens) | Longformer 등 local attention |
| Latency (110M도 실시간 부담) | DistilBERT, quantization |
| 생성 불가 (Encoder-only) | Decoder-only 모델 사용 |
| Pre-training 비용 | 사전 훈련된 모델 활용 |

---

## 핵심 요약

### Transformer 구성요소의 현대적 변화

| 구성요소 | Original (2017) | Modern (2025) |
| --- | --- | --- |
| Position Embedding | Sinusoidal/Learned | **RoPE** |
| Normalization | Post-Norm + LayerNorm | **Pre-Norm + RMSNorm** |
| Attention | Full O(n²) | **Sliding Window + GQA** |
| Architecture | Encoder-Decoder | **Decoder-Only** |

### 모델 분류 요약

| 유형 | 대표 모델 | 주요 용도 |
| --- | --- | --- |
| Encoder-Decoder | T5, BART | 번역, 요약 |
| Encoder-Only | BERT, RoBERTa | 분류, 임베딩 |
| Decoder-Only | GPT, LLaMA, Claude | 텍스트 생성, 범용 |

### 읽어볼 논문

1. **"Attention Is All You Need"** (2017) - Transformer 원본
2. **"BERT: Pre-training of Deep Bidirectional Transformers"** (2018)
3. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (2021) - RoPE
4. **"DistilBERT, a distilled version of BERT"** (2019)
5. **"RoBERTa: A Robustly Optimized BERT"** (2019)

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 2 정리*
