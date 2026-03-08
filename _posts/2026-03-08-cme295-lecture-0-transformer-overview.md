---
layout: post
title: "Stanford CME295: Lecture 0 - Transformer 개요"
date: 2026-03-08 10:00:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, transformer, attention, nlp, self-attention, multi-head-attention]
---

> **강의 출처**: Stanford CME295 - Transformers & LLMs (Autumn 2025)
>
> - **강사**: Afshine & Shervine Amidi
> - **치트시트**: [GitHub](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models), [Super Study Guide](https://superstudy.guide)

---

## 1. NLP (자연어 처리) 기초

### 1.1 NLP의 주요 태스크

![NLP 주요 태스크](/assets/img/cme295-lecture-0/image-20260102-022527.png)

```
┌─────────────────────────────────────────────────────────────────┐
│ NLP의 3가지 주요 태스크                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│ │ Classification  │ │Multi-Classification│ │ Generation    │    │
│ │ (분류)          │ │ (다중 분류)       │ │ (생성)         │    │
│ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤    │
│ │ • 감정 분석     │ │ • 품사 태깅      │ │ • 텍스트 생성   │    │
│ │ • 스팸 탐지     │ │ • 개체명 인식    │ │ • 기계 번역     │    │
│ │ • 의도 파악     │ │   (NER)         │ │ • 요약          │    │
│ │                 │ │ • 주제 분류      │ │ • 질의응답      │    │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘    │
│                                                                 │
│ 입력 → 단일 레이블   입력 → 시퀀스별 레이블   입력 → 텍스트 시퀀스│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 NLP 파이프라인

```
텍스트 → 토큰화 → 임베딩 → 모델 처리 → 출력
```

---

## 2. 토큰화 (Tokenization)

텍스트를 모델이 처리할 수 있는 단위로 분할하는 과정

### 2.1 토큰화 방법 비교

| 방법 | 설명 | 장점 | 단점 |
| --- | --- | --- | --- |
| **Word-level** | 단어 단위 분리 | 직관적, 의미 보존 | OOV 문제, 큰 어휘집 |
| **Sub-word** | 단어를 더 작은 단위로 | OOV 해결, 적정 어휘 크기 | 학습 필요 |
| **Character-level** | 문자 단위 분리 | OOV 없음, 작은 어휘집 | 긴 시퀀스, 의미 손실 |

### 2.2 Sub-word 토큰화 알고리즘

```python
# BPE (Byte Pair Encoding) 예시
원본: "lowest", "lower", "newer", "wider"
Step 1: 문자 단위 분리
l o w e s t, l o w e r, n e w e r, w i d e r
Step 2: 가장 빈번한 쌍 병합 ("e" + "r" → "er")
l o w e s t, l o w er, n e w er, w i d er
Step 3: 다음 빈번한 쌍 병합 ("l" + "o" → "lo")
lo w e s t, lo w er, n e w er, w i d er
... 반복하여 적정 어휘 크기에 도달
```

### 2.3 주요 토큰화 방식

| 모델 | 토큰화 방식 |
| --- | --- |
| GPT-2/3/4 | BPE (Byte Pair Encoding) |
| BERT | WordPiece |
| T5, LLaMA | SentencePiece |

---

## 3. 단어 표현 (Word Representation)

### 3.1 One-Hot Encoding

```
어휘집: [cat, dog, bird, fish]
"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]
```

**문제점**:

* 단어 간 유사도 표현 불가 (모든 벡터가 직교)
* 어휘 크기에 따라 차원이 폭발적 증가
* 희소 벡터 (sparse vector)

### 3.2 Word2Vec (분산 표현)

![Word2Vec 개념](/assets/img/cme295-lecture-0/image-20260102-022546.png)

```
핵심 아이디어: "비슷한 맥락에 나타나는 단어는 비슷한 의미"

벡터 공간에서의 단어 관계:
  king ─────────────────→ queen
    ↓                       ↓
    │ - man + woman         │
    ↓                       ↓
  man ──────────────────→ woman

vec(king) - vec(man) + vec(woman) ≈ vec(queen)

특징:
  • 밀집 벡터 (dense vector): 보통 100-300 차원
  • 유사한 단어 → 유사한 벡터 (코사인 유사도 높음)
  • 의미적, 문법적 관계 포착
```

### 3.3 코사인 유사도

두 토큰 t₁, t₂의 유사도:

$$\text{similarity}(t\_1, t\_2) = \frac{t\_1 \cdot t\_2}{\|t\_1\| \cdot \|t\_2\|} = \cos(\theta)$$

![코사인 유사도](/assets/img/cme295-lecture-0/image-20260109-030858.png)

* 값 범위: [-1, 1]
* 1에 가까울수록 유사
* -1에 가까울수록 반대

---

## 4. RNN의 한계와 Transformer의 등장

### 4.1 RNN (Recurrent Neural Network)의 문제점

![RNN 구조](/assets/img/cme295-lecture-0/image-20260102-022557.png)

```
x₁ ──→ [h₁] ──→ x₂ ──→ [h₂] ──→ x₃ ──→ [h₃] ──→ ...
         │               │               │
         ↓               ↓               ↓
         y₁              y₂              y₃

문제점:
  1. 순차적 처리 → 병렬화 불가능
  2. Vanishing/Exploding Gradient
  3. 긴 문장에서 장거리 의존성 학습 어려움
```

### 4.2 장거리 의존성 문제

```
예시: "The cat, which was sitting on the mat, was black."
                                                   ↑
주어 "cat"과 동사 "was"의 거리가 멀다
RNN에서는 이 정보가 여러 스텝을 거치면서 희석됨
```

### 4.3 LSTM/GRU의 개선과 한계

* **LSTM**: 게이트 메커니즘으로 장기 의존성 일부 해결
* **한계**: 여전히 순차적 처리 → 학습 시간 오래 걸림

---

## 5. Self-Attention 메커니즘

### 5.1 Self-Attention의 핵심 개념

![Self-Attention 개념](/assets/img/cme295-lecture-0/image-20260102-022609.png)

```
문장: "The animal didn't cross the street because it was tired"

"it"이 무엇을 의미하는가?

Self-Attention: 각 토큰이 문장 내 모든 토큰과의 관계를 계산
→ "it"은 "animal"에 높은 attention 가중치를 가짐
```

### 5.2 Query, Key, Value (Q, K, V)

정보 검색(Information Retrieval) 시스템에서 영감을 받은 개념:

```
데이터베이스 비유:
  검색어(Query): "무엇을 찾고 싶은가?"
  키(Key):      "각 데이터의 인덱스/제목"
  값(Value):    "실제 데이터 내용"

Self-Attention에서:
  Query (Q): "현재 토큰이 어떤 정보를 원하는가?"
  Key (K):   "각 토큰이 어떤 정보를 가지고 있는가?"
  Value (V): "각 토큰의 실제 정보/특징"

Q와 K의 내적 → 유사도 → 어디에 attention할지 결정
그 가중치로 V를 조합 → 최종 출력
```

### 5.3 Scaled Dot-Product Attention

**공식**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$$

![Scaled Dot-Product Attention](/assets/img/cme295-lecture-0/image-20260109-030837.png)

**단계별 계산**:

```
Step 1: Q × K^T 계산 (유사도 점수)
  → 각 쿼리와 모든 키 간의 내적
Step 2: √d_k로 스케일링
  → 내적 값이 너무 커지는 것을 방지
  → d_k가 클수록 내적 값의 분산이 커짐
Step 3: Softmax 적용
  → 확률 분포로 변환 (합이 1)
Step 4: V와 곱셈
  → 가중 평균으로 최종 출력 생성
```

### 5.4 스케일링이 필요한 이유

```python
# d_k가 클 때의 문제
d_k = 64  # key dimension
# Q, K의 각 원소가 평균 0, 분산 1이라면
# Q·K의 분산 = d_k (내적의 분산)
# d_k가 커지면 내적 값도 커짐
# → softmax에서 큰 값은 1에 가깝게, 작은 값은 0에 가깝게
# → gradient가 매우 작아짐 (vanishing gradient)
# 해결: √d_k로 나눔
score = (Q @ K.T) / math.sqrt(d_k)  # 분산을 1로 정규화
```

<details>
<summary>스케일링이 필요한 이유 상세 (수식 포함)</summary>

![스케일링 상세 1](/assets/img/cme295-lecture-0/image-20260109-031626.png)
![스케일링 상세 2](/assets/img/cme295-lecture-0/image-20260109-031643.png)
![스케일링 상세 3](/assets/img/cme295-lecture-0/image-20260109-031657.png)

</details>

<details>
<summary>분산의 기본 성질</summary>

![분산 기본 성질](/assets/img/cme295-lecture-0/image-20260109-040543.png)

</details>

---

## 6. Multi-Head Attention (MHA)

### 6.1 MHA의 필요성

단일 attention head의 한계:

* 하나의 관점에서만 관계를 파악
* 다양한 유형의 의존성을 동시에 포착하기 어려움

### 6.2 Multi-Head Attention 구조

![Multi-Head Attention 구조](/assets/img/cme295-lecture-0/image-20260102-022625.png)

```
입력 X
  │
  ┌───────────────┼───────────────┐
  ↓               ↓               ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Head 1  │ │  Head 2  │ │  Head h  │
│ Q₁=X·W^Q₁│ │ Q₂=X·W^Q₂│ │ Qₕ=X·W^Qₕ│
│ K₁=X·W^K₁│ │ K₂=X·W^K₂│ │ Kₕ=X·W^Kₕ│
│ V₁=X·W^V₁│ │ V₂=X·W^V₂│ │ Vₕ=X·W^Vₕ│
│ Attention │ │ Attention │ │ Attention │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      └───────────────┼───────────────┘
                      ↓
                   Concat
                      ↓
                 Linear (W^O)
                      ↓
                     출력
```

### 6.3 MHA 공식

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, ..., \text{head}\_h)W^O$$

$$\text{where } \text{head}\_i = \text{Attention}(QW\_i^Q, KW\_i^K, VW\_i^V)$$

![MHA 공식 1](/assets/img/cme295-lecture-0/image-20260109-030805.png)
![MHA 공식 2](/assets/img/cme295-lecture-0/image-20260109-030716.png)

### 6.4 원논문의 설정

| 파라미터 | 값 | 설명 |
| --- | --- | --- |
| h | 8 | attention head 수 |
| d\_model | 512 | 모델 차원 |
| d\_k = d\_v | 64 | head당 차원 (512/8) |

**계산 효율성**: 각 head의 차원을 줄여서 총 계산량은 single-head와 유사

---

## 7. Transformer 아키텍처

### 7.1 전체 구조

![Transformer 아키텍처](/assets/img/cme295-lecture-0/image-20260102-022646.png)

```
┌─────────────────────────────────────────────────────────────────┐
│ Transformer 아키텍처                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENCODER                          DECODER                       │
│  ─────────                        ──────────────                │
│  ┌───────────────┐                ┌───────────────┐             │
│  │ Multi-Head    │                │ Masked        │             │
│  │ Self-Attention│           ┌───│ Self-Attention│             │
│  └───────┬───────┘           │    └───────┬───────┘             │
│          ↓                   │            ↓                     │
│  ┌───────────────┐           │    ┌───────────────┐             │
│  │ Add & Norm    │           │    │ Add & Norm    │             │
│  └───────┬───────┘           │    └───────┬───────┘             │
│          ↓                   │            ↓                     │
│  ┌───────────────┐           │    ┌───────────────┐             │
│  │ Feed Forward  │           │    │ Cross-        │             │
│  │ Network       │           │    │ Attention     │◀────────┤   │
│  └───────┬───────┘           │    └───────┬───────┘             │
│          ↓                   │            ↓                     │
│  ┌───────────────┐           │    ┌───────────────┐             │
│  │ Add & Norm    │           │    │ Feed Forward  │             │
│  └───────┬───────┘           │    └───────┬───────┘             │
│          │ ×N                │            │ ×N                  │
│          └────────────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Encoder 구조

**N = 6개의 동일한 레이어 스택**

각 레이어 구성:

1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network

각 서브레이어 후:

* Residual Connection
* Layer Normalization

```
출력 = LayerNorm(x + Sublayer(x))
```

<details>
<summary>Encoder 단계별 동작 상세</summary>

![Encoder 상세 1](/assets/img/cme295-lecture-0/image-20260109-065647.png)
![Encoder 상세 2](/assets/img/cme295-lecture-0/image-20260109-065706.png)
![Encoder 상세 3](/assets/img/cme295-lecture-0/image-20260109-065722.png)

</details>

### 7.3 Decoder 구조

**N = 6개의 동일한 레이어 스택**

각 레이어 구성:

1. **Masked** Multi-Head Self-Attention (미래 토큰 마스킹)
2. Multi-Head Cross-Attention (Encoder 출력에 attention)
3. Position-wise Feed-Forward Network

### 7.4 Attention의 3가지 유형

```
1. Self-Attention (Encoder)
   Q, K, V = 모두 같은 소스 (이전 인코더 레이어 출력)
   모든 위치가 모든 위치에 attention 가능

2. Masked Self-Attention (Decoder)
   Q, K, V = 모두 같은 소스 (이전 디코더 레이어 출력)
   현재 위치는 이전 위치들에만 attention 가능
   → 미래 토큰 정보 유출 방지 (auto-regressive)

3. Cross-Attention (Encoder-Decoder)
   Q = 디코더 이전 레이어 출력
   K, V = 인코더 최종 출력
   디코더가 인코더의 모든 위치에 attention 가능
```

---

## 8. Position-wise Feed-Forward Network (FFN)

### 8.1 구조

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
       = ReLU(xW₁ + b₁)W₂ + b₂

또는 현대 LLM에서:
FFN(x) = SwiGLU(x)  # 또는 GeGLU, GELU
```

### 8.2 차원 설정 (원논문)

| 파라미터 | 값 |
| --- | --- |
| d\_model (입력/출력) | 512 |
| d\_ff (hidden) | 2048 |

**특징**:

* 위치별 독립적으로 적용 (동일 가중치)
* 비선형성 제공
* "채널 믹싱" 역할

<details>
<summary>FFN 적용 이유 상세</summary>

![FFN 상세 1](/assets/img/cme295-lecture-0/image-20260112-022342.png)
![FFN 상세 2](/assets/img/cme295-lecture-0/image-20260112-011050.png)
![FFN 상세 3](/assets/img/cme295-lecture-0/image-20260112-011107.png)
![FFN 상세 4](/assets/img/cme295-lecture-0/image-20260112-011126.png)
![FFN 상세 5](/assets/img/cme295-lecture-0/image-20260112-011145.png)
![FFN 상세 6](/assets/img/cme295-lecture-0/image-20260112-011201.png)
![FFN 상세 7](/assets/img/cme295-lecture-0/image-20260112-011218.png)
![FFN 상세 8](/assets/img/cme295-lecture-0/image-20260112-011233.png)
![FFN 상세 9](/assets/img/cme295-lecture-0/image-20260112-011248.png)
![FFN 상세 10](/assets/img/cme295-lecture-0/image-20260112-011303.png)
![FFN 상세 11](/assets/img/cme295-lecture-0/image-20260112-011317.png)

</details>

---

## 9. Positional Encoding

Transformer는 순서 정보가 없으므로 위치 정보를 명시적으로 추가

### 9.1 Sinusoidal Positional Encoding (원논문)

$$PE\_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right)$$

$$PE\_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right)$$

![Positional Encoding](/assets/img/cme295-lecture-0/image-20260109-030931.png)

```python
# PyTorch 구현
def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
    return pe
```

### 9.2 Sinusoidal PE의 장점

1. **상대적 위치 표현**: PE(pos+k)를 PE(pos)의 선형 함수로 표현 가능
2. **길이 일반화**: 학습 시 보지 못한 길이에도 적용 가능
3. **학습 불필요**: 미리 정의된 함수

### 9.3 현대 위치 인코딩 방식

| 방식 | 설명 | 사용 모델 |
| --- | --- | --- |
| **Learned Absolute** | 학습 가능한 위치 임베딩 | GPT-2, BERT |
| **RoPE** | 회전 위치 임베딩 | LLaMA, GPT-NeoX |
| **ALiBi** | Attention에 위치 바이어스 추가 | MPT, BLOOM |
| **T5 Relative Bias** | 상대적 위치 바이어스 | T5 |

---

## 10. Layer Normalization & Residual Connections

### 10.1 Residual Connection (Skip Connection)

```
출력 = x + Sublayer(x)
```

**효과**:

* Gradient 소실 문제 완화
* 깊은 네트워크 학습 가능
* 정보 흐름 개선

### 10.2 Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
여기서:
  μ = mean(x)      # 해당 레이어의 평균
  σ = std(x)       # 해당 레이어의 표준편차
  γ, β = 학습 가능한 파라미터
  ε = 작은 상수 (안정성)
```

### 10.3 Pre-Norm vs Post-Norm

```
Post-Norm (원논문):
  x → Sublayer → + → LayerNorm → 출력
       ↑___|

Pre-Norm (현대 LLM):
  x → LayerNorm → Sublayer → + → 출력
       ↑___|
```

**Pre-Norm의 장점**:

* 학습 안정성 향상
* 더 깊은 네트워크 학습 가능
* 대부분의 현대 LLM에서 사용 (GPT-2+, LLaMA 등)

<details>
<summary>Residual Connection 적용 이유 상세</summary>

![Residual 상세 1](/assets/img/cme295-lecture-0/image-20260112-002154.png)
![Residual 상세 2](/assets/img/cme295-lecture-0/image-20260112-002243.png)
![Residual 상세 3](/assets/img/cme295-lecture-0/image-20260112-002306.png)
![Residual 상세 4](/assets/img/cme295-lecture-0/image-20260112-002400.png)

</details>

<details>
<summary>LayerNorm 적용 이유 상세</summary>

![LayerNorm 상세 1](/assets/img/cme295-lecture-0/image-20260112-003657.png)
![LayerNorm 상세 2](/assets/img/cme295-lecture-0/image-20260112-003713.png)
![LayerNorm 상세 3](/assets/img/cme295-lecture-0/image-20260112-003732.png)
![LayerNorm 상세 4](/assets/img/cme295-lecture-0/image-20260112-003749.png)
![LayerNorm 상세 5](/assets/img/cme295-lecture-0/image-20260112-003828.png)
![LayerNorm 상세 6](/assets/img/cme295-lecture-0/image-20260112-003843.png)

</details>

---

## 11. Transformer 변형 아키텍처

### 11.1 Encoder-only (BERT류)

```
구조: Encoder 스택만 사용

특징:
  • 양방향 Self-Attention (모든 토큰 참조 가능)
  • [CLS] 토큰으로 문장 표현
  • [MASK] 토큰으로 MLM 학습

대표 모델: BERT, RoBERTa, ALBERT, ELECTRA

주요 태스크: 텍스트 분류, 토큰 분류 (NER), 질의응답 (추출형)
```

### 11.2 Decoder-only (GPT류)

```
구조: Decoder 스택만 사용 (Cross-Attention 제거)

특징:
  • Causal Self-Attention (이전 토큰만 참조)
  • Auto-regressive 생성
  • Next Token Prediction으로 학습

대표 모델: GPT 시리즈, LLaMA, Claude, Gemma, DeepSeek 등

※ 현재 대부분의 LLM이 Decoder-only 아키텍처 사용
```

### 11.3 Encoder-Decoder (T5류)

```
구조: 원본 Transformer 구조 유지

특징:
  • Encoder: 양방향 Self-Attention
  • Decoder: Causal Self-Attention + Cross-Attention
  • Seq2Seq 태스크에 적합

대표 모델: T5, BART, mT5, Flan-T5

주요 태스크: 기계 번역, 요약, 질의응답 (생성형)
```

---

## 12. Transformer End-to-End 예시

### 12.1 기계 번역 예시

```
입력: "I love AI"
목표 출력: "나는 AI를 사랑한다"

Step 1: 토큰화
  입력: ["I", "love", "AI"]
  출력: ["<sos>", "나는", "AI를", "사랑한다", "<eos>"]

Step 2: 임베딩 + Positional Encoding
  각 토큰 → d_model 차원 벡터 + PE

Step 3: Encoder 처리
  ["I", "love", "AI"] → 6개 Encoder 레이어
  → contextual representations

Step 4: Decoder 처리 (auto-regressive)
  t=0: <sos> + Encoder출력 → "나는"
  t=1: <sos>, "나는" + Encoder출력 → "AI를"
  t=2: <sos>, "나는", "AI를" + Encoder출력 → "사랑한다"
  t=3: ... → <eos>

Step 5: 최종 출력
  "나는 AI를 사랑한다"
```

---

## 13. Transformer의 핵심 장점

### 13.1 RNN 대비 장점

| 측면 | RNN | Transformer |
| --- | --- | --- |
| **병렬화** | 불가능 (순차 처리) | 완전 병렬화 가능 |
| **장거리 의존성** | gradient 소실로 어려움 | 직접 연결 (O(1) 경로) |
| **학습 속도** | 느림 | 빠름 (GPU 활용) |
| **메모리** | 시퀀스 길이에 선형 | 시퀀스 길이에 이차 |

### 13.2 Self-Attention의 계산 복잡도

| 복잡도 유형 | Self-Attention | RNN |
| --- | --- | --- |
| 시간 복잡도 (per layer) | O(n²·d) | O(n·d²) |
| 병렬화 가능 연산 | O(1) | O(n) |
| 최대 경로 길이 | O(1) | O(n) |

* n: 시퀀스 길이
* d: 표현 차원

---

## 14. 실습 코드: Self-Attention 구현

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch, heads, seq_len, d_k]
            K: [batch, heads, seq_len, d_k]
            V: [batch, heads, seq_len, d_v]
            mask: [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
        """
        d_k = Q.size(-1)

        # Step 1: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: 마스킹 (선택적)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 3: Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 4: Attention @ V
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projection & reshape to [batch, heads, seq_len, d_k]
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)

        # Concat heads & final projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)

        return output, attention_weights


# 사용 예시
d_model = 512
n_heads = 8
seq_len = 10
batch_size = 2

mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(batch_size, seq_len, d_model)

# Self-Attention
output, weights = mha(x, x, x)
print(f"Output shape: {output.shape}")            # [2, 10, 512]
print(f"Attention weights shape: {weights.shape}") # [2, 8, 10, 10]
```

---

## 15. 심화 학습 자료

### 15.1 필수 읽기 자료

1. **"Attention Is All You Need"** - Vaswani et al., 2017
   * 원논문: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **The Annotated Transformer**
   * [nlp.seas.harvard.edu](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

3. **Stanford CS224N Lecture Notes**
   * [Self-Attention & Transformers](https://web.stanford.edu/class/cs224n/readings/)

### 15.2 후속 강의 미리보기

| 강의 | 주제 |
| --- | --- |
| Lecture 2 | Advanced Transformer (RoPE, ALiBi, Sparse Attention, BERT) |
| Lecture 3 | LLMs & Inference (MoE, Decoding, Prompting, KV Cache) |
| Lecture 4 | LLM Training (Pretraining, Scaling Laws, LoRA, Flash Attention) |
| Lecture 5 | Preference Tuning (RLHF, DPO, Reward Model) |
| Lecture 6 | Applications (RAG, Agents, LLM-as-a-Judge) |
| Lecture 7 | Reasoning Models (DeepSeek-R1, Test-time Scaling) |

---

## 16. 핵심 용어 정리

| 용어 | 정의 |
| --- | --- |
| **Self-Attention** | 시퀀스 내 모든 위치 간의 관계를 직접 계산하는 메커니즘 |
| **Query (Q)** | 현재 토큰이 찾고자 하는 정보를 표현하는 벡터 |
| **Key (K)** | 각 토큰이 가진 정보의 "주소"를 나타내는 벡터 |
| **Value (V)** | 각 토큰의 실제 정보 내용을 담은 벡터 |
| **Multi-Head Attention** | 여러 attention head를 병렬로 사용하여 다양한 관계 포착 |
| **Positional Encoding** | 토큰의 순서 정보를 임베딩에 추가하는 방법 |
| **Residual Connection** | 입력을 출력에 직접 더하는 skip connection |
| **Layer Normalization** | 레이어 내 활성화 값을 정규화하는 기법 |
| **Causal Masking** | 미래 토큰을 가리는 attention 마스킹 |
| **Cross-Attention** | Encoder 출력을 참조하는 Decoder의 attention |

---

## 17. 연습 문제

### Q1. Self-Attention의 계산 복잡도가 O(n²)인 이유는?

<details>
<summary>정답 보기</summary>

모든 토큰 쌍에 대해 attention score를 계산해야 하기 때문입니다.

* n개의 Query × n개의 Key = n² 개의 점수
* 따라서 시간 및 메모리 복잡도가 O(n²·d)

</details>

### Q2. √d\_k로 스케일링하는 이유는?

<details>
<summary>정답 보기</summary>

내적 값의 분산이 d\_k에 비례하여 커지기 때문입니다.

* Q, K의 각 원소가 평균 0, 분산 1이면, 내적의 분산 ≈ d\_k
* 큰 값이 softmax를 통과하면 gradient가 매우 작아짐
* √d\_k로 나누어 분산을 1로 정규화하여 안정적인 학습 유도

</details>

### Q3. Encoder-only vs Decoder-only의 핵심 차이는?

<details>
<summary>정답 보기</summary>

**Attention 방향**이 다릅니다:

* Encoder-only (BERT): 양방향 attention (모든 토큰 참조)
* Decoder-only (GPT): 단방향 attention (이전 토큰만 참조)

**용도**:

* Encoder-only: 분류, 표현 학습
* Decoder-only: 텍스트 생성 (auto-regressive)

</details>

---

*Based on Stanford CME295 Lecture 1 by Afshine & Shervine Amidi*
