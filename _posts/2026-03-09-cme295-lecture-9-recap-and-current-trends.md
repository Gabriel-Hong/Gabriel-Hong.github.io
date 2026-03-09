---
layout: post
title: "Stanford CME295: Lecture 9 - Recap & Current Trends"
date: 2026-03-09 10:00:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, transformer, vision-transformer, diffusion-llm, recap, vit, vlm]
math: true
---

> **원본 강의**: [YouTube - CME295 Lecture 9](https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9)

---

## 강의 개요

이번 강의는 CME295의 마지막 강의로, 세 가지 파트로 구성됩니다:

1. **Part 1:** 전체 강의 요약 (Lecture 1~8 Recap)
2. **Part 2:** 2025년 트렌딩 토픽 (Vision Transformer, Diffusion LLM)
3. **Part 3:** 결론 및 미래 방향

**시험 범위:** Lecture 5 ~ Lecture 8 (Part 2 이후 내용은 시험 범위 외)

---

## Part 1: 전체 강의 요약 (Recap)

### Lecture 1-2: Transformer 기초

#### 텍스트 처리의 발전

```
┌─────────────────────────────────────────────────────────────────┐
│ 텍스트 표현의 발전 과정                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Tokenization → Word2Vec → RNN → Self-Attention → Transformer   │
│      ↓            ↓        ↓         ↓              ↓          │
│  텍스트 분할   의미 학습  순서 고려  직접 연결     병렬 처리      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 단계 | 핵심 내용 | 한계 |
| --- | --- | --- |
| **Tokenization** | 텍스트를 atomic unit으로 분할, Subword가 표준 |  |
| **Word2Vec** | 문맥 기반 의미 학습 | Context-aware 아님 (같은 단어 = 같은 벡터) |
| **RNN** | 순차 처리, hidden state 유지 | Long-range dependency 문제 |
| **Self-Attention** | 모든 토큰이 직접 연결 |  |

#### Self-Attention 핵심 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Query (Q):** "무엇을 찾는가?"
- **Key (K):** "어떤 정보인가?"
- **Value (V):** "실제 내용"

#### Transformer 아키텍처 개선 (Lecture 2)

| 개선점 | 원본 (2017) | 현재 |
| --- | --- | --- |
| **Position Encoding** | Absolute (각 위치별 고정 임베딩) | **RoPE** (상대적 거리 기반) |
| **Attention** | Multi-Head Attention | **Grouped Query Attention** (KV 그룹화) |
| **Normalization** | Post-Norm (sub-layer 후) | **Pre-Norm** (sub-layer 전) |

#### Transformer 파생 모델

| 모델 유형 | 대표 모델 | 특징 | 용도 |
| --- | --- | --- | --- |
| **Encoder-only** | BERT | CLS 토큰 임베딩 활용 | 분류 (Classification) |
| **Decoder-only** | GPT | Auto-regressive 생성 | 텍스트 생성 |
| **Encoder-Decoder** | T5 | 번역, Seq2Seq | 번역, 요약 |

---

### Lecture 3: LLM 기초

#### Mixture of Experts (MoE)

```
┌─────────────────────────────────────────────────────────────────┐
│ Mixture of Experts 구조                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Input Token → [Gating Mechanism] → Expert 1 (FFN)               │
│                    ↓               → Expert 2 (FFN) → Output    │
│              (라우팅 결정)          → Expert 3 (FFN)             │
│                                    → Expert N (FFN)             │
│                                                                 │
│ 핵심: 모든 파라미터를 사용하지 않고 일부 Expert만 활성화           │
│   → Forward Pass 시 연산량 감소                                  │
│   → 토큰 단위로 라우팅 → 다른 GPU에 분산 가능                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Temperature Sampling

| Temperature | 분포 특성 | 출력 특성 |
| --- | --- | --- |
| **T ≈ 0** | 매우 뾰족함 (Spiky) | 결정론적, 항상 최고 확률 선택 |
| **T ≈ 0.7** | 적당히 완만 | 다양성과 품질의 균형 |
| **T ≈ 1.2** | 매우 평평함 | 높은 무작위성, 창의적 |

---

### Lecture 4: LLM 학습

#### Scaling Laws & Chinchilla

```
┌─────────────────────────────────────────────────────────────────┐
│ Scaling Laws 발견                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 관찰: Compute ↑, Data ↑, Parameters ↑ → Test Loss ↓            │
│                                                                 │
│ Chinchilla 논문의 Rule of Thumb:                                │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ 토큰 수 ≥ 파라미터 수 × 20                              │     │
│ │                                                         │     │
│ │ 예: 100B 모델 → 최소 2T 토큰으로 학습                   │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                 │
│ 발견: 당시 대부분의 모델이 "Undertrained" 상태                  │
│   → 모델 크기에 비해 데이터가 부족했음                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Flash Attention

| 개념 | 설명 |
| --- | --- |
| **HBM** | 크지만 느린 메모리 (High Bandwidth Memory) |
| **SRAM** | 작지만 빠른 메모리 |
| **핵심 아이디어** | HBM 접근을 최소화하고 SRAM에서 계산 |
| **Recomputation** | 저장하지 않고 필요 시 재계산 → 더 빠름 |
| **결과** | 정확한 결과 (근사 아님) + 속도 향상 |

#### 병렬화 기법

| 기법 | 설명 |
| --- | --- |
| **Data Parallelism** | 데이터를 여러 GPU에 분산 |
| **Model Parallelism** | 모델을 여러 GPU에 분산 (하나의 Forward Pass에 여러 GPU 사용) |

#### LLM 학습 3단계

```
┌─────────────────────────────────────────────────────────────────┐
│ LLM 학습 파이프라인                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Stage 1: Pre-training                                           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ • 목표: 언어/코드의 구조 학습                           │     │
│ │ • 데이터: 수조 개 토큰 (인터넷 데이터)                  │     │
│ │ • 결과: Autocomplete만 가능한 모델                      │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Stage 2: Supervised Fine-Tuning (SFT)                           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ • 목표: 원하는 Input-Output 패턴 학습                   │     │
│ │ • 데이터: 고품질 (Prompt, Response) 쌍                  │     │
│ │ • 결과: "무엇을 해야 하는지" 아는 모델                  │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Stage 3: Preference Tuning (RLHF/DPO)                           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ • 목표: "무엇을 하지 말아야 하는지" 학습                │     │
│ │ • 데이터: Preference Data (A > B)                       │     │
│ │ • 결과: 안전하고 유용한 모델                            │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Lecture 5: RLHF (Reinforcement Learning from Human Feedback)

#### LLM과 RL의 연결

| RL 개념 | LLM 대응 |
| --- | --- |
| **Policy (π)** | LLM 자체 |
| **State (s)** | 지금까지 받은 입력 |
| **Action (a)** | 다음 토큰 예측 |
| **Reward (r)** | Human Preference |

#### Bradley-Terry Formulation

$$P(i > j) = \frac{\exp(R_i)}{\exp(R_i) + \exp(R_j)}$$

- **학습:** Pairwise 데이터로 학습 (A가 B보다 좋다)
- **추론:** 개별 출력에 점수 부여

#### Reward Model 활용

```
┌─────────────────────────────────────────────────────────────────┐
│ RLHF 학습 루프                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Prompt → LLM → Completion → [Reward Model] → Reward Score       │
│   ↑                                              ↓              │
│   │                                        Policy Update        │
│   │                                              ↓              │
│   └───────────────── 반복 ───────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### RLHF의 제약 조건

| 제약 | 이유 |
| --- | --- |
| **Base Model과 거리 유지** | Reward Hacking 방지 (불완전한 Reward 악용) |
| **업데이트 크기 제한** | 학습 안정성 |

---

### Lecture 6: Reasoning Models

#### Reasoning Model의 핵심

```
┌─────────────────────────────────────────────────────────────────┐
│ Vanilla LLM vs Reasoning Model                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Vanilla LLM:                                                    │
│   Prompt → [LLM] → Answer                                      │
│                                                                 │
│ Reasoning Model:                                                │
│   Prompt → [LLM] → <think>Reasoning Chain</think>              │
│                   → <answer>Final Answer</answer>               │
│                                                                 │
│   → Chain of Thought를 대규모로 적용!                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### PPO vs GRPO

| 측면 | PPO | GRPO |
| --- | --- | --- |
| **Value Function** | 필요 (별도 네트워크) | **불필요** |
| **Advantage 계산** | $A = f(R, V(s))$ (GAE) | $A = \frac{R_i - \bar{R}}{\sigma_R}$ |
| **핵심 아이디어** | Reward - Value로 Advantage | Group 내 상대 비교로 Advantage |
| **적합한 태스크** | 일반 RL | **Verifiable Reward** (코드/수학) |

#### GRPO의 핵심

```
┌─────────────────────────────────────────────────────────────────┐
│ GRPO Advantage 계산                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. 하나의 Prompt에 대해 G개의 Completion 생성                   │
│    o₁, o₂, ..., o_G                                            │
│                                                                 │
│ 2. 각 Completion의 Reward 계산                                  │
│    R₁, R₂, ..., R_G                                            │
│                                                                 │
│ 3. Advantage 계산 (Group 내 상대 비교)                          │
│    A_i = (R_i - mean) / std                                     │
│                                                                 │
│ 4. A > 0: 확률 증가 ↑, A < 0: 확률 감소 ↓                      │
│                                                                 │
│ 장점: Value Network 불필요 → 구조 단순화                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### GRPO의 Length Bias 문제

| 문제 | 원인 | 해결책 |
| --- | --- | --- |
| 긴 오답 선호 | Loss의 $\frac{1}{\lvert o_i \rvert}$ 정규화 항 | DAPO, Dr. GRPO 등 |

---

### Lecture 7: RAG & Tool Calling

#### RAG (Retrieval Augmented Generation)

```
┌─────────────────────────────────────────────────────────────────┐
│ RAG 파이프라인                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Step 1: Candidate Retrieval (Bi-Encoder)                        │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ Query Embedding ←→ Document Embeddings (미리 계산됨)    │     │
│ │ → Cosine Similarity로 Top-K 후보 선택                   │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 2: Re-ranking (Cross-Encoder)                              │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ [Query + Document] → 단일 모델 → 더 정확한 점수        │     │
│ │ → 최종 Top-K 선택                                       │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 3: Augmented Generation                                    │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ [Prompt + Retrieved Documents] → LLM → Answer           │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**RAG가 필요한 이유:**

- Knowledge Cutoff: LLM은 학습 데이터 이후 정보를 모름
- 실시간 정보 필요 시 외부 문서 참조 필수

#### Tool Calling

```
┌─────────────────────────────────────────────────────────────────┐
│ Tool Calling 과정                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Step 1: API 인식                                                │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ LLM에게 사용 가능한 API 목록 제공                       │     │
│ │ → LLM이 "이 API를 이 인자로 호출하겠다" 결정           │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 2: API 실행 (중간 단계)                                    │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ 실제로 API 호출 → 결과 반환                             │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 3: 결과 통합                                               │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ API 결과를 LLM에 피드백 → 최종 답변 생성               │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Agentic Workflows

- RAG + Tool Calling을 결합한 복잡한 워크플로우
- 여러 API 호출을 순차적/병렬적으로 수행

---

### Lecture 8: LLM Evaluation

#### Rule-based Metrics의 한계

| 메트릭 | 용도 | 한계 |
| --- | --- | --- |
| **BLEU** | 번역 평가 | 의미는 같지만 표현이 다르면 낮은 점수 |
| **ROUGE** | 요약 평가 | 언어의 다양성 미반영 |
| **METEOR** | 번역 평가 | 동의어 처리 한계 |

#### LLM-as-a-Judge

```
┌─────────────────────────────────────────────────────────────────┐
│ LLM-as-a-Judge 구조                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Input:                                                          │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ • Prompt (원본 질문)                                    │     │
│ │ • Model Response (평가 대상 응답)                        │     │
│ │ • Criteria (평가 기준)                                   │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Output:                                                         │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ 1. Rationale (먼저!) - 왜 이 점수인지 설명              │     │
│ │ 2. Score (Binary 권장: Pass/Fail)                        │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                 │
│ → Rationale First = Chain of Thought 효과로 정확도 향상         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### LLM-as-a-Judge의 Bias

| Bias | 설명 | 해결책 |
| --- | --- | --- |
| **Position Bias** | 먼저 제시된 응답 선호 | 순서 바꿔서 Majority Voting |
| **Verbosity Bias** | 긴 응답 선호 | 명확한 가이드라인, 길이 페널티 |
| **Self-Enhancement Bias** | 자기 생성물 선호 | 다른 (더 큰) 모델을 Judge로 사용 |

#### 주요 Benchmarks

| 분야 | 벤치마크 | 설명 |
| --- | --- | --- |
| **Knowledge** | MMLU | 57개 도메인 다지선다 |
| **Reasoning** | AIME | 수학 올림피아드 |
| **Coding** | SWE-Bench | GitHub Issue 해결 |
| **Safety** | HarmBench | 유해성 평가 |

---

## Part 2: Trending Topics (2025)

### 1. Vision Transformer (ViT)

#### 핵심 아이디어

> "Transformer는 Text뿐만 아니라 Image에도 적용 가능하다!"

```
┌─────────────────────────────────────────────────────────────────┐
│ Vision Transformer 구조                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Step 1: 이미지를 Patch로 분할                                   │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │  ┌───┬───┬───┐                                         │     │
│ │  │ 1 │ 2 │ 3 │  → 각 Patch를 Flatten → 벡터로 변환    │     │
│ │  ├───┼───┼───┤                                         │     │
│ │  │ 4 │ 5 │ 6 │  예: 16×16 픽셀 Patch → RGB 펼침 → 768D│     │
│ │  ├───┼───┼───┤                                         │     │
│ │  │ 7 │ 8 │ 9 │                                         │     │
│ │  └───┴───┴───┘                                         │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 2: Position Embedding 추가                                 │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ [CLS] + Patch 1 + Patch 2 + ... + Patch 9              │     │
│ │   ↓       ↓         ↓               ↓                  │     │
│ │ + PE₀   + PE₁     + PE₂   ...    + PE₉                │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 3: Transformer Encoder 통과                                │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ Self-Attention으로 모든 Patch가 서로 상호작용            │     │
│ │ (BERT와 동일한 구조)                                    │     │
│ └─────────────────────────────────────────────────────────┘     │
│                          ↓                                      │
│ Step 4: CLS 토큰으로 분류                                       │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ CLS Embedding → FFN → Class Prediction                  │     │
│ │ (BERT의 Classification과 동일)                          │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### ViT vs CNN

| 측면 | CNN | ViT |
| --- | --- | --- |
| **Inductive Bias** | 높음 (Sliding Window) | 낮음 (전체 참조) |
| **데이터 요구량** | 적음 | **많음** (대규모 데이터 필요) |
| **성능** | 좋음 | 대규모 데이터에서 **더 좋음** |

**핵심 발견:** 충분한 데이터가 있으면, Low Inductive Bias 모델(ViT)이 CNN을 능가!

**Vision Transformer (ViT) 상세 설명**

**ViT의 작동 원리**

ViT는 NLP에서 성공한 Transformer 아키텍처를 이미지에 직접 적용합니다. 핵심은 이미지를 "단어"처럼 취급하는 것입니다.

**Patch Embedding 과정:**

1. 이미지(예: 224×224)를 고정 크기 Patch(예: 16×16)로 분할
2. 각 Patch를 Flatten하여 1D 벡터로 변환 (16×16×3 = 768차원)
3. Linear Projection으로 Transformer의 Hidden Dimension에 맞춤

**Position Embedding:**

- 각 Patch 위치에 학습 가능한 Position Embedding 추가
- [CLS] 토큰을 맨 앞에 추가 (BERT와 동일)

**Inductive Bias 비교:**

| 특성 | CNN | ViT |
| --- | --- | --- |
| **Locality** | 강함 (커널이 지역 영역만 봄) | 없음 (모든 Patch 참조) |
| **Translation Equivariance** | 강함 (같은 커널 공유) | 없음 (Position에 의존) |
| **데이터 효율성** | 높음 | 낮음 (많은 데이터 필요) |

**핵심 결론:** Inductive Bias는 데이터가 적을 때 유리하지만, 데이터가 충분하면 제약이 됩니다. ViT는 대규모 데이터에서 CNN을 능가합니다.

**Inductive Bias란?**

**정의:** 모델이 학습 데이터 외의 상황에서 예측을 하기 위해 사용하는 사전 가정(assumption)입니다.

**비유: 새로운 도시 탐험**

- **높은 Inductive Bias (CNN):** 가이드북을 들고 여행 → 효율적이지만, 가이드북에 없는 곳은 놓침
- **낮은 Inductive Bias (ViT):** 아무 정보 없이 여행 → 처음엔 비효율적이지만, 충분한 시간이 있으면 더 많은 곳을 발견

**CNN의 Inductive Bias:**

1. **Locality:** 인접 픽셀이 관련성 높음 (커널이 지역 영역만 봄)
2. **Translation Equivariance:** 같은 패턴은 어디에 있든 같은 방식으로 처리

**ViT의 접근:**

- 이런 가정 없이 데이터로부터 직접 학습
- 충분한 데이터가 있으면 더 일반적인 패턴을 학습할 수 있음

---

### 2. Vision-Language Models (VLM)

#### 이미지 + 텍스트 처리 방법

**방법 1: 입력 Concatenation (더 일반적)**

```
[Image Tokens] + [Text Tokens] → LLM → Answer
      ↑                ↑
Vision Encoder      Tokenizer
```

- 대표 모델: **LLaVA**
- Image Encoder가 이미지를 토큰으로 변환 후 텍스트 토큰과 연결

**방법 2: Cross-Attention**

```
Text Tokens → Self-Attention → Cross-Attention ← Image Tokens
                                     ↓
                                   Answer
```

- 대표 모델: Llama 3 (일부 버전)
- 이미지를 Cross-Attention layer에서 참조

**Vision-Language Models (VLM) 상세 설명**

**VLM의 목표:** 이미지와 텍스트를 모두 이해하는 모델

**방법 1: Input Concatenation (LLaVA 방식)**

```
┌──────────┐     ┌──────────┐
│  Image   │     │   Text   │
└────┬─────┘     └────┬─────┘
     ↓                ↓
┌──────────┐     ┌──────────┐
│  Vision  │     │Tokenizer │
│  Encoder │     │          │
└────┬─────┘     └────┬─────┘
     ↓                ↓
   [Image Tokens] + [Text Tokens]
            ↓
      ┌──────────┐
      │   LLM    │
      │(Decoder) │
      └────┬─────┘
           ↓
        Answer
```

- Vision Encoder(예: CLIP ViT)가 이미지를 토큰 시퀀스로 변환
- 텍스트 토큰과 concatenate하여 LLM에 입력
- LLM은 이미지 토큰과 텍스트 토큰을 동일하게 처리

**방법 2: Cross-Attention (Llama 3 방식)**

- 텍스트는 Self-Attention으로 처리
- 이미지는 별도 Encoder로 처리 후 Cross-Attention에서 참조
- 장점: 이미지 토큰이 텍스트 시퀀스 길이를 늘리지 않음

**학습 방식:**

1. Vision Encoder와 LLM을 각각 Pre-train
2. Projection Layer만 학습하여 연결 (효율적)
3. 전체 Fine-tuning으로 성능 향상

---

### 3. Diffusion-based LLMs (가장 트렌디한 토픽)

#### Auto-Regressive의 한계

```
┌─────────────────────────────────────────────────────────────────┐
│ Auto-Regressive LLM의 생성 과정                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [Input] → Token 1 → Token 2 → Token 3 → ... → [EOS]           │
│              ↓          ↓          ↓                            │
│          Forward 1  Forward 2  Forward 3  ...                   │
│                                                                 │
│ 문제점:                                                         │
│ • 생성 시간 = 토큰 수 × Forward Pass 시간                      │
│ • 병렬화 불가능 (이전 토큰이 있어야 다음 토큰 예측 가능)        │
│ • 긴 출력 = 느린 생성                                          │
│                                                                 │
│ (참고: 학습은 병렬화 가능 - Causal Mask 사용)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Diffusion의 핵심 아이디어

**이미지 Diffusion:**

- **Forward:** Clean Image → 점점 Noise 추가 → Pure Noise
- **Reverse:** Noise → Denoise 반복 → Clean Image
- Michelangelo 비유: "조각상은 이미 대리석 안에 있다. 나는 불필요한 부분을 깎아낼 뿐."

**텍스트 Diffusion (MDM: Masked Diffusion Model):**

```
┌─────────────────────────────────────────────────────────────────┐
│ Image Diffusion vs Text Diffusion                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Image Diffusion:                                                │
│   Clean Image → + Noise → + More Noise → Pure Noise            │
│   Pure Noise  → - Noise → - More Noise → Clean Image           │
│                                                                 │
│ ─────────────────────────────────────────────────────────────── │
│                                                                 │
│ Text Diffusion:                                                 │
│   "Hello World" → "Hello [M]" → "[M] [M]" → "[M] [M]"         │
│   "[M] [M]"     → "Hello [M]" → "Hello World"                  │
│                                                                 │
│ 핵심: Noise ↔ Mask Token                                       │
│   이미지의 "노이즈" = 텍스트의 "[MASK]"                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Diffusion LLM의 장점

| 장점 | 설명 |
| --- | --- |
| **속도** | Forward Pass 수 = Diffusion Steps (토큰 수와 무관) → **~10x 빠름** |
| **Fill-in-the-Middle** | 양방향 컨텍스트 활용 가능 (코딩에 유리) |
| **병렬 생성** | 모든 위치 동시에 생성 가능 |

#### 직관적 비유: 연설문 작성

```
Auto-Regressive: 첫 문장 → 두 번째 문장 → ... → 마지막 문장
                 (선형적으로 한 단어씩 작성)

Diffusion:       초안 (흐릿함) → 개선 → 더 개선 → 최종본
                 (전체를 동시에 점진적으로 완성)
```

#### 현재 상태 (2025)

| 모델 | 설명 |
| --- | --- |
| **Google Gemini** (실험적) | IO 2025에서 Text Diffusion 발표 |
| **Inception Labs** | Diffusion LLM 스타트업 |
| **LaDa** | Large Language Diffusion Model with Masking 논문 |

**한계:**

- 아직 Frontier 모델 수준에 도달하지 못함 (성능 격차 존재)
- Reasoning Chain 등 Auto-regressive 기법 적용 연구 진행 중

**Diffusion-based LLMs 상세 설명**

**왜 Diffusion을 텍스트에 적용하는가?**

Auto-Regressive 모델의 근본적 한계는 순차 생성입니다. 100 토큰을 생성하려면 100번의 Forward Pass가 필요합니다. Diffusion은 이 병목을 해결합니다.

**Masked Diffusion Model (MDM) 작동 원리:**

1. **Forward Process (학습 시):**
   - 원본 텍스트에서 랜덤하게 토큰을 [MASK]로 교체
   - 마스킹 비율을 점진적으로 높임 (0% → 100%)

2. **Reverse Process (생성 시):**
   - 모든 위치가 [MASK]인 상태에서 시작
   - 각 Step에서 일부 [MASK]를 실제 토큰으로 교체
   - 모든 [MASK]가 채워질 때까지 반복

**속도 비교:**

```
Auto-Regressive (100 토큰 생성):
  → 100 Forward Passes 필요

Diffusion (100 토큰 생성, 10 steps):
  → 10 Forward Passes 필요 (각 step에서 ~10개 토큰 동시 생성)
  → ~10x 빠름
```

**Fill-in-the-Middle 장점:**

Auto-Regressive 모델은 왼쪽에서 오른쪽으로만 생성하지만, Diffusion 모델은 양방향 컨텍스트를 활용할 수 있습니다.

```
코드 완성 예시:
  def hello():
      [여기를 채우기]
      return result

→ Diffusion: 위아래 컨텍스트 모두 활용하여 채움
→ AR: 위의 컨텍스트만 활용 가능
```

**현재 한계:**

- 품질: AR 모델 대비 아직 성능 격차 존재
- Reasoning: Chain-of-Thought 같은 순차적 추론에 불리
- 학습 안정성: 아직 최적의 학습 방법 연구 중

---

## Part 3: 결론 및 미래 방향

### Cross-Pollination: 모달리티 간 기술 교류

```
┌─────────────────────────────────────────────────────────────────┐
│ Text ↔ Image 기술 교류                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Text → Image:                                                   │
│   • Transformer → Vision Transformer                            │
│   • RoPE → 2D RoPE (이미지 위치 인코딩)                        │
│                                                                 │
│ Image → Text:                                                   │
│   • Diffusion → Masked Diffusion LLM                            │
│   • Image Patches → 직접 입력 (DeepSeek OCR 연구)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 진행 중인 연구 영역

| 영역 | 현재 상태 | 트렌드 |
| --- | --- | --- |
| **Optimizer** | Adam | Muon, Muon-Clip 등 새로운 후보 |
| **Normalization** | RMS Norm (Pre-Norm) | 여전히 연구 중 |
| **Attention** | GQA | Layer별 다른 Attention 사용 |
| **Activation** | GELU | 여전히 연구 중 |
| **Architecture** | Transformer | 대안 아키텍처 연구 (Mamba 등) |

### Data Curation의 중요성

```
┌─────────────────────────────────────────────────────────────────┐
│ Model Collapse 문제                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 과거: 인터넷 데이터 = 인간 생성 데이터                          │
│                    ↓                                            │
│ 현재: 인터넷 데이터 = 80%+ LLM 생성 데이터                     │
│                    ↓                                            │
│ 문제: LLM 생성 텍스트는 다양성이 낮음                           │
│   → 이로 학습하면 Model Collapse 발생                           │
│                    ↓                                            │
│ 해결책:                                                         │
│   • Data Curation 중요성 증가                                   │
│   • Pre-training → Mid-training → Fine-tuning 3단계            │
│   • Mid-training: 더 고품질 데이터로 추가 학습                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 하드웨어 혁신

| 현재 | 미래 |
| --- | --- |
| GPU (Matrix Multiply 최적화) | 목적형 하드웨어 |
| Software로 Flash Attention 등 최적화 | Attention을 하드웨어에 직접 구현 |
| Digital Computing | **Analog Computing** (신호 기반 연산) |

**Analog Computing 장점:**

- 연산이 물리적 속성으로 자동 수행
- 지연시간 감소, 에너지 절약

### LLM의 현재와 미래 사용 사례

| 시기 | 사용 사례 |
| --- | --- |
| **현재** | 코딩 어시스턴트, 일반 어시스턴트, 브레인스토밍, 학습 도구 |
| **가까운 미래** | Agentic Workflow 대중화, 웹 브라우징 에이전트 |
| **먼 미래** | OS 레벨 통합, 완전 자율 고객 서비스 |

### 남은 과제

| 과제 | 설명 |
| --- | --- |
| **Continuous Learning** | 학습 후 가중치가 고정됨 → 지속적 학습 필요 |
| **Hallucination** | Next Token Prediction의 본질적 특성 |
| **Personalization** | 개인화된 응답 |
| **Interpretability** | 결정 과정 설명 |
| **Safety** | 안전성 보장 |

---

## 핵심 요약

### 전체 강의 흐름

```
Lecture 1-2: Transformer 기초
         ↓
Lecture 3: LLM (MoE, Temperature)
         ↓
Lecture 4: 학습 (Scaling Laws, Flash Attention, 병렬화)
         ↓
Lecture 5: RLHF (Reward Model, PPO)
         ↓
Lecture 6: Reasoning (GRPO, Chain of Thought)
         ↓
Lecture 7: RAG & Tool Calling
         ↓
Lecture 8: Evaluation (LLM-as-a-Judge, Benchmarks)
         ↓
Lecture 9: Recap & Future Trends
```

### 핵심 개념 비교표

| 개념 | 핵심 포인트 |
| --- | --- |
| **Self-Attention** | 모든 토큰이 직접 연결 → Long-range Dependency 해결 |
| **MoE** | 일부 Expert만 활성화 → 효율적 연산 |
| **Flash Attention** | SRAM 활용 + Recomputation → 속도 향상 |
| **RLHF** | Reward Model + Policy Optimization → Human Alignment |
| **GRPO** | Value Function 없이 Group 비교로 Advantage 계산 |
| **RAG** | 외부 문서 검색 → Knowledge Cutoff 해결 |
| **LLM-as-a-Judge** | LLM으로 LLM 평가 → Scalable Evaluation |
| **ViT** | 이미지를 Patch로 분할 → Transformer 적용 |
| **Diffusion LLM** | Mask Token으로 Diffusion → 빠른 생성 |

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| **RoPE** | Rotary Position Embedding |
| **GQA** | Grouped Query Attention |
| **MoE** | Mixture of Experts |
| **SFT** | Supervised Fine-Tuning |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **GRPO** | Group Relative Policy Optimization |
| **GAE** | Generalized Advantage Estimation |
| **RAG** | Retrieval Augmented Generation |
| **ViT** | Vision Transformer |
| **VLM** | Vision-Language Model |
| **MDM** | Masked Diffusion Model |
| **DLLM** | Diffusion-based LLM |
| **ARM** | Auto-Regressive Model |

---

## 추천 자료

### Part 2 관련 논문

1. **"An Image is Worth 16x16 Words"** (2020) - Vision Transformer (ViT)
2. **"LLaVA: Large Language and Vision Assistant"** - VLM
3. **"Scalable Diffusion Models with Transformers"** (DiT) - Diffusion Transformer
4. **"LaDa: Large Language Diffusion Model with Masking"** (2025) - Text Diffusion

### 기타 자료

- **Yannic Kilcher YouTube** - 논문 해설
- **Andrej Karpathy YouTube** - 딥러닝 교육
- **CME295 Study Guide** - 강의 공식 가이드

### 학습 자료

| 자료 | 설명 |
| --- | --- |
| **arXiv** | 최신 논문 |
| **Hugging Face Trending Papers** | Papers with Code 대체 |
| **NeurIPS, ICML** | 주요 학회 |

### 커뮤니티

| 플랫폼 | 추천 |
| --- | --- |
| **Twitter/X** | AI 연구자 팔로우 |
| **YouTube** | Yannic Kilcher, Andrej Karpathy |
| **회사 블로그** | OpenAI, Anthropic, Google AI |
