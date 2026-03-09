---
layout: post
title: "Stanford CME295: Lecture 2 - Transformer-Based Models & Tricks"
date: 2026-03-08 10:20:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, transformer, rope, rmsnorm, gqa, bert, distilbert, roberta]
math: true
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

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Attention Map 해석:**

* 논문에서 "its"라는 토큰이 "law"와 "application"에 높은 attention weight를 가짐
* 이는 모델이 대명사와 그것이 참조하는 단어의 관계를 학습했음을 보여줌

---

### 2. Position Embeddings (위치 임베딩)

Transformer는 RNN과 달리 토큰을 순차적으로 처리하지 않아 위치 정보가 자연스럽게 인코딩되지 않습니다. 따라서 별도의 위치 정보 주입이 필요합니다.

#### 2.1 Learned Position Embeddings (학습된 위치 임베딩)

$$\text{Input}_i = \text{Token Embedding}_i + \text{Position Embedding}_i$$

**장점:** Gradient descent를 통해 데이터로부터 자연스럽게 학습

**단점:**

1. 학습 데이터에 의존적 (overfitting 가능성)
2. 학습 시 본 최대 위치까지만 학습 가능
3. 추론 시 학습하지 않은 위치에 대해서는 일반화 불가

#### 2.2 Sinusoidal Position Embeddings

**원본 Transformer 논문에서 제안한 방법**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**수학적 근거:**

삼각함수 항등식을 활용:

$$\cos(\omega_i(m-n)) = \cos(\omega_i m)\cos(\omega_i n) + \sin(\omega_i m)\sin(\omega_i n)$$

두 위치 임베딩 $P_m$과 $P_n$의 dot product:

$$P_m \cdot P_n = \sum_i \cos(\omega_i(m-n))$$

이는 상대적 거리 $(m-n)$의 함수!

**장점:**
* 학습 시 보지 못한 길이로도 확장 가능
* 상대적 위치 정보를 자연스럽게 인코딩

#### 2.3 Relative Position Bias (T5)

**핵심 아이디어:** 위치 정보를 입력에 추가하는 대신, Attention 계산에 직접 주입

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

여기서 $B$는 학습 가능한 bias term으로, 상대적 거리에 따라 값이 결정됨

#### 2.4 ALiBi (Attention with Linear Bias)

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - m \cdot |i-j|\right)V$$

**특징:**
* 학습 없이 결정론적 공식 사용
* 상대적 거리에 비례하는 선형 penalty
* 단순하지만 제한적

#### 2.5 RoPE (Rotary Position Embeddings) - 현재 가장 많이 사용

**핵심 아이디어:** Query와 Key 벡터를 위치에 따른 각도로 회전

**2D에서의 회전 행렬:**

$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

벡터 $v$를 각도 $\theta$만큼 회전:

$$R_\theta \cdot v = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

**RoPE의 적용:**

* Query $q_m$을 위치 $m$에 해당하는 각도로 회전
* Key $k_n$을 위치 $n$에 해당하는 각도로 회전
* 결과적으로 $q_m \cdot k_n$은 상대적 거리 $(m-n)$의 함수가 됨

**수학적 결과:**

$$q_m^T k_n = (R_{\theta m} q)^T (R_{\theta n} k) \propto f(m-n)$$

**장점:**

1. 상대적 거리 정보가 자연스럽게 인코딩
2. Long-term decay: 멀리 떨어진 토큰일수록 attention weight의 상한이 감소
3. 대부분의 현대 LLM에서 사용 (LLaMA, GPT-4 등)

---

### 3. Layer Normalization

#### 3.1 기본 개념

**Layer Norm 공식:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

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

**현대 모델들 (Pre-Norm):**

$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$$

**Pre-Norm의 장점:**
* 더 안정적인 학습
* Gradient flow 개선
* 대부분의 현대 LLM에서 채택

#### 3.4 RMSNorm (Root Mean Square Normalization)

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma$$

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
* 각 head마다 별도의 $W_Q$, $W_K$, $W_V$
* KV Cache 크기: $O(n \cdot h \cdot d)$

**Multi-Query Attention (MQA):**
* 모든 head가 동일한 $W_K$, $W_V$ 공유
* Query만 head별로 다름
* KV Cache 크기: $O(n \cdot d)$ ($h$배 감소)

**Grouped Query Attention (GQA):**
* MHA와 MQA의 중간
* $G$개의 그룹으로 나누어, 같은 그룹 내 head들이 $W_K$, $W_V$ 공유
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

**왜 PAD 토큰이 필요한가?**

배치 처리 시 모든 시퀀스의 길이가 동일해야 합니다. 짧은 시퀀스는 [PAD] 토큰으로 채웁니다.

```
배치 예시:
  시퀀스 1: [CLS] The cat sat [SEP] [PAD] [PAD]
  시퀀스 2: [CLS] A cute teddy bear is here [SEP]
```

**PAD 토큰의 처리:**

- Attention 계산 시 **PAD 위치를 마스킹** ($-\infty$로 설정)하여 softmax 후 0이 되도록 함
- 이를 통해 PAD 토큰이 다른 토큰의 representation에 영향을 주지 않음

$$\text{score}_{i,\text{PAD}} = -\infty \quad \Rightarrow \quad \text{softmax}(\text{score})_{i,\text{PAD}} = 0$$

- Loss 계산 시에도 PAD 위치는 제외

</details>

#### 6.3 BERT의 입력 표현

**세 가지 임베딩의 합:**

$$\text{Input} = \text{Token Embedding} + \text{Position Embedding} + \text{Segment Embedding}$$

**Token Embedding:** WordPiece tokenizer 사용 (vocab size ~30K)

<details>
<summary>Token Embedding 학습 과정</summary>

**Token Embedding이란?**

각 토큰(어휘 내 단어/서브워드)을 고정 차원 벡터로 매핑하는 lookup table입니다.

$$E \in \mathbb{R}^{V \times d_{model}}$$

여기서 $V$는 어휘 크기(~30K), $d_{model}$은 임베딩 차원(768 for BERT-Base)

**학습 과정:**

1. **초기화:** 랜덤 초기화 (Xavier/Gaussian)
2. **Pre-training:** MLM, NSP 태스크를 수행하면서 역전파를 통해 임베딩 업데이트
3. **Fine-tuning:** Downstream 태스크에 맞게 추가 조정

**WordPiece 토큰화:**

```
"unbelievable" → ["un", "##believe", "##able"]
```

- `##` 접두사는 단어의 중간/끝 서브워드를 의미
- 이를 통해 어휘 크기를 ~30K로 유지하면서 OOV 문제 해결

</details>

**Segment Embedding:** 문장 A vs 문장 B 구분, NSP 태스크를 위해 도입

<details>
<summary>Segment Embedding 상세</summary>

**Segment Embedding이란?**

입력이 두 문장으로 구성될 때, 각 토큰이 어느 문장에 속하는지를 나타내는 임베딩입니다.

$$\text{Segment Embedding} \in \mathbb{R}^{2 \times d_{model}}$$

두 개의 학습 가능한 벡터: $E_A$ (문장 A), $E_B$ (문장 B)

**예시:**

```
입력:  [CLS] I love AI [SEP] It is great [SEP]
세그먼트: A    A  A   A   A     B  B   B     B
```

- 문장 A의 모든 토큰에 $E_A$를 더함
- 문장 B의 모든 토큰에 $E_B$를 더함

**용도:**

- NSP (Next Sentence Prediction): 두 문장의 관계 학습
- Question Answering: 질문(A)과 문맥(B)을 구분
- NLI (Natural Language Inference): 전제(A)와 가설(B) 구분

**단일 문장 태스크에서는?** 모든 토큰에 $E_A$만 사용합니다.

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

**[CLS] 토큰이란?**

모든 입력 시퀀스의 맨 앞에 추가되는 특수 토큰으로, Self-Attention을 통해 전체 시퀀스의 정보를 집약합니다.

**문장 분류에서의 활용:**

```
입력:  [CLS] This movie was great [SEP]
         ↓
   Transformer Encoder (12 layers)
         ↓
   [CLS] embedding → Linear → Softmax → 긍정/부정
```

[CLS]의 최종 hidden state $h_{CLS} \in \mathbb{R}^{d_{model}}$에 분류 head를 추가:

$$P(\text{class}) = \text{softmax}(W \cdot h_{CLS} + b)$$

**질의응답에서의 활용:**

QA에서는 [CLS] 대신 각 토큰의 hidden state를 사용합니다:

$$P(\text{start} = i) = \text{softmax}(W_s \cdot h_i)$$

$$P(\text{end} = i) = \text{softmax}(W_e \cdot h_i)$$

답변의 시작/끝 위치를 각각 예측합니다.

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

$$\mathcal{L} = D_{KL}(P_{teacher} \| P_{student}) = \sum_x P_{teacher}(x) \log \frac{P_{teacher}(x)}{P_{student}(x)}$$

<details>
<summary>KL Divergence 손실 함수 상세</summary>

**KL Divergence란?**

두 확률 분포 $P$와 $Q$ 사이의 "거리"를 측정하는 비대칭적 지표입니다.

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x) [\log P(x) - \log Q(x)]$$

**성질:**

- $D_{KL}(P \| Q) \geq 0$ (항상 0 이상)
- $D_{KL}(P \| Q) = 0$ iff $P = Q$ (두 분포가 같을 때만 0)
- $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ (비대칭)

**Knowledge Distillation에서의 사용:**

Teacher의 soft label (temperature scaling 적용):

$$P_{teacher}(x) = \text{softmax}\left(\frac{z_{teacher}}{T}\right)$$

Student의 출력:

$$P_{student}(x) = \text{softmax}\left(\frac{z_{student}}{T}\right)$$

$T > 1$로 설정하면 확률 분포가 더 "부드러워져" Teacher의 지식(클래스 간 상대적 관계)이 더 잘 전달됩니다.

**전체 손실:**

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(\text{hard labels}) + (1-\alpha) \cdot T^2 \cdot D_{KL}(P_{teacher} \| P_{student})$$

$T^2$를 곱하는 이유: temperature scaling으로 인한 gradient 크기 보정

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
| Attention | Full $O(n^2)$ | **Sliding Window + GQA** |
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
