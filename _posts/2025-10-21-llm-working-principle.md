---
layout: post
title: LLM 동작 원리 알아보기
date: 2025-10-21 09:00:00 +0900
categories: [AI, LLM]
pin: true
---

## 0. 왜?

LLM의 동작 원리를 이해하면, 단순 사용을 넘어서 원하는 결과가 잘 나오지 않는 원인을 분석하거나, 특정 목적에 맞게 커스터마이징할 기준을 세울 수 있는 실질적인 역량을 키울 수 있다.

![이미지: LLM 동작 원리 개요](/assets/img/llm-principle/intro-image.png)

## 1. 개요

**대규모 언어 모델(LLM, Large Language Model)**은 딥러닝 기술을 기반으로 방대한 텍스트 데이터를 통해 언어의 의미와 맥락을 학습하여 자연어 처리(NLP) 작업을 수행하는 인공지능 모델이다.

초기의 NLP 연구는 기계 번역(Machine Translation)과 같은 특정 작업을 수행하기 위해 발전하였다. 특히 2010년대 초반에는 RNN(Recurrent Neural Network)을 활용한 번역 모델이 개발되었고, 이는 문맥 파악 능력을 기반으로 기계 번역의 품질을 획기적으로 개선하였다.

이후 Transformer 모델이 2017년 구글의 "Attention is All You Need" 논문에서 제안되면서, 자연어 처리 성능과 효율성이 크게 향상되었다. **Transformer 기반 모델**은 초기 번역 목적으로 개발되었으나, 텍스트 분류, 문장 생성, 질의응답 등 다양한 분야에 적용 가능성을 보이며 점차 범용성을 확보하게 되었다.

최근에는 GPT 시리즈(GPT-3, GPT-4)와 같은 모델이 등장하여 언어 이해와 생성 능력이 더욱 정교해졌으며, 광범위한 응용 분야에 걸쳐 활용되고 있다.

![이미지: LLM 역사](/assets/img/llm-principle/history-of-llm.png)

---

## 2. LLM 모델별 활용과 특성

### 주요 모델 비교

- **RNN (Recurrent Neural Network)**: 시간적 순서가 있는 데이터를 순차적으로 처리하며, 이전 단계의 출력을 현재 입력의 일부로 활용
- **CNN (Convolutional Neural Network)**: 주로 이미지와 같은 공간적 데이터를 처리할 때 사용되며, 필터를 통해 지역적(local) 특징을 추출
- **Transformer**: Self-attention 메커니즘을 이용해 데이터를 병렬로 처리

| 분류 | 활용 분야 | 장점 | 단점 |
|------|-----------|------|------|
| **RNN** | 순차 데이터 처리 | • 순차 데이터 처리 용이<br>• 간단한 구조 | • 장기 의존성 문제<br>• 병렬 처리 어려움 |
| **CNN** | 이미지, 공간적 데이터 처리 | • 공간적 정보 활용 우수<br>• 병렬 처리 가능 | • 순차 데이터 모델링에 제한적 |
| **Transformer** | 병렬처리, 장거리 의존성 모델링 | • 장거리 의존성 포착 우수<br>• 병렬 처리로 효율적 학습 가능 | • 계산량 및 메모리 사용량 많음 |

### RNN의 처리 방법 예시

**예시 문장**: "I am a student"

RNN은 단어를 시퀀스 순서대로 처리하여 이전 단어의 정보를 다음 단어 처리 시 참조한다.

![이미지: RNN 처리 방법](/assets/img/llm-principle/rnn-process.png)

각 단어는 단어 임베딩(Embedding)을 통해 숫자 벡터 형태로 변환된다.

1. **"I"** → RNN 셀은 입력 "I"의 임베딩 벡터를 처리하여 은닉 상태(Hidden state)를 출력한다.
2. **"am"** → 이전 단계의 은닉 상태를 입력으로 받아 "am"의 임베딩 벡터와 함께 처리하여 새로운 은닉 상태를 출력한다.
3. **"a"** → 이전 은닉 상태를 이어받아 처리한다.
4. **"student"** → 마지막으로 이전 은닉 상태와 함께 처리한다.

최종적으로 생성된 은닉 상태는 문장의 의미를 요약한 벡터 표현(Sentence embedding)으로 활용된다.

### CNN의 처리 방법 예시

**예시 문장**: "I am a student"

CNN은 문장을 공간적인 데이터로 간주하여 필터를 통해 지역적인 패턴을 추출하는 방식으로 처리한다.

![이미지: CNN 처리 방법](/assets/img/llm-principle/cnn-process.gif)

문장을 단어 임베딩을 통해 숫자 벡터로 변환한 후, 행렬 형태로 표현한다.

1. 필터(filter)가 단어들을 순차적으로 윈도우(window) 크기만큼 슬라이딩하면서 문장의 일부를 추출한다.
2. 각 부분의 특징(feature)을 뽑기 위해 필터와 해당 단어 벡터들의 내적(dot product)을 수행한다.
3. 여러 개의 필터를 사용해 다양한 특징을 추출한다.

특징들을 풀링(pooling)하여 최종적으로 문장의 요약된 특징 벡터를 얻는다.

### Transformer의 처리 방법 예시

**예시 문장**: "I am a student"

Transformer는 RNN과 달리 각 단어를 순차적으로 처리하지 않고, 전체 문장을 한 번에 입력하여 Self-Attention을 통해 단어 간 상호 연관성을 계산한다.

![이미지: Transformer 처리 방법](/assets/img/llm-principle/transformer-process.png)

1. 각 단어는 임베딩을 통해 벡터로 변환되고, 위치 임베딩(Positional embedding)을 추가하여 단어의 위치 정보를 포함시킨다.
2. **Self-attention 계산**
3. **Multi-head Attention**
4. **Feed-forward network**
5. **정규화**

최종적으로 얻어진 각 단어의 표현은 전체 문장에 대한 맥락적 의미를 담고 있으며, 문장 분류, 번역, 문장 생성 등에 사용된다.

---

## 3. Transformer 모델의 구조와 동작 원리 이해하기

Transformer는 **Encoder-Decoder 구조**로 이루어져 있으며, 두 부분은 각각 여러 개의 블록(layer)로 구성됩니다.

![이미지: Transformer 아키텍처](/assets/img/llm-principle/transformer-architecture.png)

**입력 문장 → [Encoder] → 문맥 정보 → [Decoder] → 출력 문장**

### 3.1 문장을 숫자로 변환하는 단계 (Tokenization & Embedding)

![이미지: Transformer 아키텍처 3.1](/assets/img/llm-principle/transformer-architecture_3_1.png)

**목표**: 텍스트 데이터를 모델이 이해할 수 있는 숫자(벡터)로 바꿈

**동작**:
1. "나는 피자를 좋아해" → ['나', '는', '피자', '를', '좋아해']
2. 각 단어를 고정 길이의 Embedding Vector로 변환
3. 출력: Tensor of shape (seq_len, embedding_dim)

#### Embedding Vector

**Embedding Vector**는 단어나 문장 등의 개체를 고차원 공간의 실수 벡터로 표현한 것입니다. 벡터들 간의 내적(dot product)을 계산하면, 의미적으로 관련 있는 것들일수록 값이 커지는 경향이 있습니다.

**예시 단어**: "King", "Queen", "Man", "Woman"

| 단어 | Embedding Vector (5차원 예시) |
|------|-------------------------------|
| king | [0.8, 0.2, 0.9, 0.1, 0.7] |
| queen | [0.8, 0.2, 0.9, 0.1, 0.3] |
| man | [0.6, 0.1, 0.8, 0.2, 0.9] |
| woman | [0.6, 0.1, 0.8, 0.2, 0.2] |

이 벡터에서 각 차원이 다음과 같은 의미를 잠재적으로 갖고 있다고 가정해볼 수 있습니다:

| 차원 번호 | 의미 |
|-----------|------|
| 1 | 사회적 지위 정도 |
| 2 | 연령대 성향 |
| 3 | 리더십/권한의 정도 |
| 4 | 감정 표현 강도 |
| 5 | 성별 (남성: 1.0, 여성: 0.0 가깝게) |

이 구조는 **king - man + woman ≈ queen**과 같은 의미 추론이 가능한 이유를 보여줍니다.

#### Tensor

**Tensor**는 스칼라(0차원), 벡터(1차원), 행렬(2차원), 고차원 배열(3차원 이상)을 일반화한 데이터와 파라미터를 표현하는 기본 단위입니다.

| 요소 | 설명 | 예시 | 중요도 |
|------|------|------|--------|
| Rank | 텐서가 몇 차원을 가지는지 나타냄 | 2 (e.g. [batch, hidden]) | 높음 |
| Shape | 각 차원에서의 크기를 나타냄 | (32, 128, 512) | 매우 높음 |
| Data Type | 텐서의 데이터 타입 | float32, float16, bfloat16 | 중간 |
| Device | 텐서가 저장/연산되는 장치 | cpu, cuda:0 | 높음 |
| Requires Grad | 역전파 시 gradient 계산이 필요한지 여부 | True, False | 높음 |

### 3.2 단어의 순서를 알려주는 단계 (Positional Encoding)

![이미지: Transformer 아키텍처 3.2](/assets/img/llm-principle/transformer-architecture_3_2.png)

**문제**: Transformer는 순서를 모름 (RNN과 달리 구조적으로 순서가 없음)

**해결**: 입력 벡터에 위치 정보를 더함 ("순서 감각" 부여)

**동작**:
- 각 단어에 위치에 따른 고유한 벡터를 더함
- 사인(sin), 코사인(cos) 함수로 계산된 값 사용
- "피자를"이 "나는"보다 뒤에 있다는 정보 포함됨

#### 수식

![이미지: Positional Encoding 수식](/assets/img/llm-principle/positional-encoding-formula.png)

- `pos`: 단어의 위치 (0부터 시작)
- `i`: 임베딩 차원의 인덱스
- `d`: 전체 임베딩 벡터의 차원

짝수 인덱스에는 sin, 홀수 인덱스에는 cos 함수가 적용되어 위치마다 고유한 벡터가 생성됩니다.

#### Positional Encoding의 패턴 시각화

![이미지: Positional Encoding 패턴](/assets/img/llm-principle/positional-encoding-pattern.png)

각 행은 하나의 벡터에 대한 positional encoding에 해당합니다.

### 3.3 서로의 관계를 파악하는 단계 (Self-Attention)

![이미지: Transformer 아키텍처 3.3](/assets/img/llm-principle/transformer-architecture_3_3.png)

**핵심**: 문장 내 단어들이 서로를 얼마나 참고해야 하는지 결정

**동작**:
1. 각 단어에서 Query, Key, Value 생성
2. Query와 다른 단어들의 Key와 내적 → 유사도 측정
3. softmax → 가중치 계산 → Value와 Weighted Sum
4. 예: "좋아해"가 "피자"를 더 많이 참고하게 됨

#### 내적(dot product)으로 유사도 계산하기

**내적(dot product)**은 두 벡터가 얼마나 같은 방향을 향하는지(유사도)를 나타냅니다.

**단어 시퀀스**: "I like pizza."

| 단어 | Query Vector (Q) | Key Vector (K) |
|------|------------------|----------------|
| "like" | [1.0, 0.5, 0.0] | - |
| "I" | - | [0.9, 0.4, 0.1] |
| "pizza" | - | [0.2, 0.1, 0.7] |

이제 "like"이 다른 단어("I", "pizza")와 얼마나 관련 있는지를 Q·K^T 내적으로 계산합니다:

**1. like vs I**:
```
(1.0 × 0.9) + (0.5 × 0.4) + (0.0 × 0.1) = 0.9 + 0.2 + 0.0 = 1.1
```

**2. like vs pizza**:
```
(1.0 × 0.2) + (0.5 × 0.1) + (0.0 × 0.7) = 0.2 + 0.05 + 0.0 = 0.25
```

**해석**:
- "like"는 문장 내에서 "I"와 더 밀접한 관계를 가지므로 내적 값 **1.1**이 더 큽니다.
- "pizza"와는 관련은 있지만 상대적으로 약해 내적 값이 작게(**0.25**) 나옵니다.

#### Softmax

Softmax는 벡터의 값을 확률처럼 0~1 사이로 정규화해주는 함수로, 출력값의 총합이 1이 되도록 만듭니다.

![이미지: Softmax 함수](/assets/img/llm-principle/softmax-formula.png)

**예시 계산**:

내적 계산:
- like vs I: Q * K^T = 1.1
- like vs pizza: Q * K^T = 0.25

Softmax 함수를 통해 이 두 값을 정규화:

1. **지수함수 적용**: e^1.1 ≈ 3.004, e^0.25 ≈ 1.284
2. **분모 계산**: sum = 3.004 + 1.284 = 4.288
3. **Softmax 결과 계산**:
   - like vs I: Softmax(1.1) = 3.004 / 4.288 ≈ **0.701**
   - like vs pizza: Softmax(0.25) = 1.284 / 4.288 ≈ **0.299**

**결과**: "like"는 "I"에 약 70.1%의 Attention을 주고, "pizza"에는 약 29.9%의 Attention을 줍니다.

#### Self-Attention 메커니즘

![이미지: Self-Attention 메커니즘](/assets/img/llm-principle/self-attention-mechanism.png)

**Self-Attention**은 입력 시퀀스 내의 각 토큰이 다른 모든 토큰을 참고하여 문맥을 반영하는 표현을 만드는 메커니즘입니다.

**계산 단계**:

1. **선형 변환으로 Q, K, V 생성**: Q = XW_Q, K = XW_K, V = XW_V
2. **Query와 Key의 내적 → 유사도(Attention Score) 계산**: score_{i,j} = Q_i · K_j^T
3. **정규화 (softmax)**: α_{i,j} = softmax(score_{i,j})
4. **Weighted Sum (가중합)**: output_i = Σ α_{i,j}V_j

#### Weighted Sum (가중합)

**Weighted Sum**은 가중치의 비율에 따라 정보를 강조하거나 줄이는 방식입니다. 기존 정보들을 버리지 않고 중요도에 따라 반영합니다.

**예시 문장**: "I like pizza."

| 단어 | 벡터 | 가중치 |
|------|------|--------|
| "I" | [0.1, 0.3, 0.5] | 0.2 |
| "like" | [0.4, 0.6, 0.8] | 0.5 |
| "pizza" | [0.7, 0.9, 0.2] | 0.3 |

Weighted Sum 결과:
```
0.2·"I" + 0.5·"like" + 0.3·"pizza"
= 0.2 · [0.1, 0.3, 0.5] + 0.5 · [0.4, 0.6, 0.8] + 0.3 · [0.7, 0.9, 0.2]
= [0.43, 0.63, 0.56]
```

### 3.4 병렬로 여러 관점에서 보는 단계 (Multi-Head Attention)

![이미지: Transformer 아키텍처 3.4](/assets/img/llm-principle/transformer-architecture_3_4.png)

**이유**: 단어 관계를 다양한 관점에서 분석 (예: 의미, 문법 등)

**동작**:
- 여러 개의 Attention Head 사용 → 다양한 정보 추출
- 다시 합쳐서 통합 정보 생성

#### Attention Head

하나의 Attention Head는 Query, Key, Value 벡터를 입력받아 Self-Attention 연산을 수행합니다. 이 Head는 자체적인 선형 변환 행렬(W_q, W_k, W_v)을 가지므로, 입력 정보를 다른 방식으로 해석합니다.

| Head No. | 의미적/구문적 관계 |
|----------|-------------------|
| Head 1 | 동사-목적어 관계 |
| Head 2 | 주어-동사 일치 |
| Head 3 | 명사 간 유사성 |

![이미지: Multi-Head Attention](/assets/img/llm-principle/multi-head-attention.png)

일반적으로 **8개 또는 12개**의 head를 사용하며, 각 head는 같은 입력을 받지만 서로 다른 W_q, W_k, W_v로 연산합니다. 각 head의 출력을 Concat(연결) → 선형 변환하여 최종 출력을 생성합니다.

### 3.5 정보를 조정하고 다음 단계로 넘기는 단계 (Feed-Forward & Residual Connection)

![이미지: Transformer 아키텍처 3.5](/assets/img/llm-principle/transformer-architecture_3_5.png)

#### Feed-Forward Network (FFN)

**역할**: Attention 결과를 비선형 연산으로 가공

**동작**:
- Dense layer → ReLU → Dense layer
- Residual connection으로 안정적인 학습 지원
- Layer Normalization을 통해 정보 정리

**FFN의 구조**:
```
FFN(x) = Linear_2(ReLU(Linear_1(x)))
```

1. **Linear_1**: hidden dim → FFN dim (예: 512 → 2048)
2. **ReLU**: 비선형성 추가

![이미지: ReLU 함수](/assets/img/llm-principle/relu-function.png)

3. **Linear_2**: FFN dim → hidden dim (예: 2048 → 512)

#### Residual Connection

입력을 그대로 더해줌 → 정보 손실 방지, 깊은 네트워크에서도 기울기 소실(gradient vanishing)을 완화

```
Output = x + SubLayer(x)
```

#### Layer Normalization

위의 결과에 정규화를 적용하여 학습 안정화

```
LayerNorm(x) = (x - μ) / σ · γ + β
```

- μ: 벡터의 평균
- σ: 벡터의 표준편차
- γ, β: 학습 가능한 스케일 및 시프트 파라미터

### 3.6 입력 문장을 이해하는 파트 (Encoder)

![이미지: Transformer 아키텍처 3.6](/assets/img/llm-principle/transformer-architecture_3_6.png)

**역할**:
- 입력 문장 내 모든 단어들의 의미를 파악하고, 전체 문맥 정보를 추출
- Decoder가 참고할 수 있도록 context representation을 생성

**구성 요소**:
- **Self-Attention**: 입력 문장 내 단어들 간 관계를 계산
- **Feed Forward Network (FFN)**: Attention 결과를 더 풍부하게 가공
- **Residual + Layer Norm**: 정보 손실 방지 및 학습 안정화

**출력**: 각 입력 토큰에 대해 문맥을 반영한 임베딩 벡터 (shape: [batch_size, seq_len, hidden_dim])

### 3.7 문장을 생성하는 파트 (Decoder)

![이미지: Transformer 아키텍처 3.7](/assets/img/llm-principle/transformer-architecture_3_7.png)

**역할**:
- Encoder의 문맥 정보를 참고하며, 출력 문장을 왼쪽부터 한 단어씩 생성
- 예: 번역 시, 영어 문장을 하나씩 만들어냄 ("I like pizza")

**구성 요소**:
- **Masked Self-Attention**: 현재까지 생성된 단어들만 보면서 다음 단어 예측 (자기 자신보다 미래 단어는 못 봄 → auto-regressive)
- **Encoder-Decoder Attention**: Encoder의 출력(context)을 참고
  - Query는 Decoder에서 생성한 벡터
  - Key와 Value는 Encoder의 출력에서 생성한 벡터
- **Feed Forward Network**
- **Residual + Layer Norm**

**출력**: 각 위치에서 생성할 다음 단어에 대한 확률 분포

### 3.8 최종 출력 (단어 생성)

![이미지: Transformer 아키텍처 3.8](/assets/img/llm-principle/transformer-architecture_3_8.png)

**과정**: Decoder가 softmax를 통해 다음 단어 확률 예측

**동작**:
- 확률이 가장 높은 단어 선택 → 문장 생성
- 반복해서 전체 문장 생성

---

## Transformer를 이용한 기계 번역 과정 시각화

![이미지: Transformer 기계 번역 과정](/assets/img/llm-principle/transformer-translation-process.gif)

---

## Reference

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
