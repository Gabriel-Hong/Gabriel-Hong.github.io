---
layout: post
title: "[논문 리뷰] Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
date: 2026-03-08 18:00:00 +0900
categories: [AI, Paper]
tags: [attention, gating, attention-sink, transformer, neurips]
---

> **논문 링크**: <https://openreview.net/pdf?id=1b7whO4SfY>
>
> - **저자**: Qwen 팀 (Alibaba) - Zihan Qiu 외 다수
> - **발표**: NeurIPS 2025 Oral (**Best Paper Award** 수상, 상위 1.5%, 77/5,290)
> - **코드**: <https://github.com/qiuzh20/gated_attention>

---

## 핵심 아이디어

단순한 수정 하나로 LLM 성능을 향상시킨다:

```
기존 Transformer:
  Query, Key, Value → SDPA(Softmax Attention) → Output

Gated Attention:
  Query, Key, Value → SDPA → [Sigmoid Gate] → Output
                                    ↑
                              헤드별 게이트 추가 (G1)
```

SDPA(Scaled Dot-Product Attention) 출력에 헤드별 sigmoid 게이트를 추가하는 것만으로 일관된 성능 향상을 달성한다.

---

## 왜 효과가 있는가?

### 1. 비선형성 도입 (Non-linearity)

- 기존 softmax attention은 Value→Output 변환이 저랭크 선형 매핑
- Sigmoid 게이트가 비선형성을 추가하여 표현력 증가

### 2. 희소성 도입 (Sparsity)

- 게이트가 쿼리 종속적 희소 점수를 생성
- 불필요한 정보를 억제하고 중요한 정보만 통과

### 3. Attention Sink 해결 (Attention-Sink-Free)

**Attention Sink란?**

```
[BOS] [토큰1] [토큰2] ... [토큰N]
  ↑
  첫 번째 토큰에 attention이 비정상적으로 몰리는 현상
```

LLM이 의미 없는 첫 토큰(BOS 등)에 attention을 과도하게 집중하는 문제이다. 이로 인해:

- 대규모 활성화(Massive Activation) 발생
- 긴 컨텍스트 처리 성능 저하

**Gated Attention의 해결:**

- 게이트가 이런 비정상적 attention 패턴을 억제
- 장문맥 외삽(Long-context Extrapolation) 성능 대폭 향상

---

## 실험 규모

| 모델 | 데이터 |
| --- | --- |
| 15B MoE 모델 30개 변형 비교 | 3.5조 토큰 학습 |
| 1.7B Dense 모델 | RULER 등 벤치마크 평가 |

## 주요 성과

- **학습 안정성 향상**: 더 큰 learning rate 허용
- **스케일링 특성 개선**: 모델 크기 증가에 따른 성능 향상 폭 증가
- **장문맥 성능**: RULER 벤치마크에서 유의미한 향상
- **실제 적용**: Qwen3-Next 모델에 채택

---

## 1. 서론 (Introduction)

게이팅 메커니즘은 LSTM, Highway Networks부터 최근의 state space 모델, linear attention, softmax attention까지 널리 사용되어 왔다. 그러나 기존 문헌에서는 게이팅의 구체적인 효과를 거의 조사하지 않았다.

**연구 동기:**

- Switch Heads에서 단일 expert로 줄여도 상당한 성능 향상이 유지됨 → 게이팅 자체가 라우팅과 별개로 내재적 가치를 제공
- Native Sparse Attention(NSA)에서도 게이팅의 기여를 sparse attention 설계와 분리하지 않음
- 게이팅 효과를 다른 아키텍처 요소와 엄격히 분리할 필요성 존재

---

## 2. Gated-Attention Layer

### 2.1 Multi-Head Softmax Attention 기초

Attention 계산의 4단계:

1. **QKV Linear Projections**: 입력 X를 Q, K, V로 변환
2. **SDPA (Scaled Dot-Product Attention)**: softmax 정규화된 어텐션 스코어 계산
3. **Multi-Head Concatenation**: 모든 헤드 출력 연결
4. **Final Output Layer**: 연결된 출력에 출력 레이어 적용

### 2.2 게이팅 메커니즘 형식화

게이팅 메커니즘은 `Y′ = g(Y, X, Wθ, σ) = Y ⊙ σ(XWθ)`로 형식화된다. 게이팅 스코어 `σ(XWθ)`는 동적 필터로 작용하여 Y의 정보 흐름을 선택적으로 보존하거나 제거한다.

**탐색한 5가지 측면:**

1. **위치**: Q/K/V 프로젝션 후(G2,G3,G4), SDPA 출력 후(G1), 최종 출력 레이어 후(G5)
2. **세분성**: Headwise(스칼라) vs Elementwise(벡터)
3. **Head Specific vs Shared**: 각 헤드별 독립 vs 헤드 간 공유
4. **곱셈 vs 덧셈**: `Y′ = Y · σ(Xθ)` vs `Y′ = Y + σ(Xθ)`
5. **활성화 함수**: Sigmoid vs SiLU

---

## 3. 실험 (Experiments)

### 3.1 실험 설정

- **모델**: 15B MoE (15A2B, 2.54B 활성화) + 1.7B Dense 모델
- **데이터**: 4T 고품질 토큰 (다국어, 수학, 일반 지식)
- **평가**: Hellaswag, MMLU, GSM8k, HumanEval, C-eval, CMMLU + 다양한 도메인 PPL

### 3.2 주요 결과

#### MoE 모델 결과

SDPA와 value 출력 게이팅이 가장 효과적이며, SDPA 출력(G1) 또는 value map(G2)에 게이트를 삽입하면 PPL이 가장 낮고 벤치마크 성능이 좋다.

핵심 관찰:

1. **Head-Specific 게이팅이 중요**: Headwise 게이팅도 2M 미만 파라미터로 상당한 개선
2. **곱셈 게이팅이 선호됨**: 덧셈 게이팅보다 성능 우수
3. **Sigmoid 활성화가 더 좋음**: SiLU보다 sigmoid가 효과적

#### Dense 모델 결과

게이팅은 다양한 설정에서 효과적이며, 학습 안정성을 개선하여 loss spike를 크게 줄인다. 베이스라인이 수렴 문제를 겪는 높은 학습률에서도 게이팅 모델은 개선을 보인다.

**실용적 권장사항**: Elementwise SDPA 게이팅(G1)을 적용하고 학습률을 적당히 증가시켜 훈련

---

## 4. 분석: 비선형성, 희소성, Attention-Sink-Free

### 4.1 비선형성이 Low-Rank 매핑의 표현력을 향상

Multi-head attention에서 value(Wv)와 dense(WO) 프로젝션은 하나의 low-rank 선형 매핑으로 병합될 수 있다. 두 선형 매핑 사이에 비선형성을 추가하면 표현력이 향상된다.

- G2 위치 게이팅: Equation 7 수정에 해당
- G1 위치 게이팅/정규화: Equation 8 수정에 해당
- G5 위치(WO 이후)에 게이팅은 효과 없음 → WV와 WO 사이 비선형성 부재를 해결하지 못함

### 4.2 게이팅이 입력 의존적 희소성을 도입

SDPA 출력 게이팅(Elementwise/headwise)이 가장 낮은 평균 게이팅 스코어를 보이며, 스코어 분포가 0 근처에 높게 집중되어 상당한 희소성을 나타낸다.

**핵심 관찰:**

1. **효과적인 게이팅 스코어는 희소함**: SDPA 출력 게이팅이 가장 희소
2. **Head-Specific 희소성이 중요**: 헤드 간 공유 시 전체 게이팅 스코어 증가 및 성능 저하
3. **Query 의존성이 중요**: Value 게이팅(G2)보다 SDPA 출력 게이팅(G1)이 더 효과적
4. **덜 희소한 게이팅은 더 나쁨**: NS-sigmoid([0.5, 1.0] 범위)는 sigmoid보다 성능 저하

### 4.3 SDPA 출력 게이팅이 Massive Activation과 Attention Sink를 감소

베이스라인 모델은 심각한 attention sink 현상을 보이며, 레이어 전반에 걸쳐 평균 **46.7%**의 어텐션 스코어가 첫 번째 토큰에 집중된다. 게이트를 도입하면 이 비율이 **4.8%**로 감소한다.

관찰 결과:

1. Query 의존적 sigmoid 게이팅이 첫 토큰 어텐션 스코어와 massive activation을 크게 감소
2. Value 프로젝션(G2) 게이팅만으로는 massive activation은 감소하지만 attention sink는 지속 → massive activation이 attention sink의 필요조건은 아님
3. 희소성 감소 시 massive activation과 attention sink 모두 심화

### 4.4 SDPA 출력 게이팅이 Context Length Extension을 용이하게 함

RULER 벤치마크에서 64k와 128k 컨텍스트 길이에서 게이트된 어텐션 모델이 베이스라인을 크게 능가한다.

| 방법 | 4k | 8k | 16k | 32k | 64k | 128k |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 88.89 | 85.88 | 83.15 | 79.50 | - | - |
| SDPA-Gate | 90.56 | 87.11 | 84.61 | 79.77 | - | - |
| YaRN Extended Baseline | 82.90 | 71.52 | 61.23 | 37.94 | 37.51 | 31.65 |
| YaRN Extended SDPA-Gate | 88.13 | 80.01 | 76.74 | 72.88 | **66.60** | **58.82** |

가설: 베이스라인은 attention sink에 의존하여 어텐션 스코어 분포를 조정하지만, 게이팅 모델은 입력 의존적 게이팅 스코어로 정보 흐름을 제어하여 RoPE 수정에 더 강건함

---

## 5. 관련 연구 (Related Works)

### 5.1 신경망에서의 게이팅

- LSTM, GRU: 시간 단계 간 정보 흐름 조절
- Highway Networks: 피드포워드 네트워크로 확장
- SwiGLU: 트랜스포머 FFN 레이어에 게이팅 도입
- Mamba, RetNet, Lightning Attention 등 최근 모델들

### 5.2 Attention Sink

- StreamingLLM: 특정 토큰이 큰 어텐션 스코어를 받는 현상 식별
- Massive Activation: 과도한 어텐션 스코어가 massive activation 값과 연관
- 다양한 완화 시도: sigmoid attention, attention calibration, 'registers', 'meta tokens' 추가 등

---

## 6. 결론 및 한계 (Conclusion and Limitations)

**결론:** 이 간단한 메커니즘이 비선형성을 강화하고, 입력 의존적 희소성을 도입하며, 'attention sink'를 제거한다. 또한 게이팅은 컨텍스트 길이 확장을 용이하게 하여 모델이 재훈련 없이도 더 긴 시퀀스로 효과적으로 일반화할 수 있게 한다.

**한계:**

- 비선형성이 어텐션 동역학과 전체 훈련 과정에 미치는 더 넓은 영향이 충분히 탐구되지 않음
- Attention sink가 더 긴 시퀀스로의 일반화 능력에 어떻게 영향을 미치는지에 대한 이론적 설명 부재

**적용:** 이 SDPA 출력 게이팅은 **Qwen3-Next 모델에 사용**됨

---

## 의의

"단순하지만 효과적인" 연구의 전형:

- 복잡한 아키텍처 변경 없이 sigmoid 게이트 하나만 추가
- 30개 이상의 변형을 체계적으로 비교하여 최적 위치 검증
- 이론적 분석과 실험적 검증 모두 제시

> **Sources:**
> - <https://arxiv.org/abs/2505.06708>
> - <https://openreview.net/forum?id=1b7whO4SfY>
> - <https://github.com/qiuzh20/gated_attention>
> - <https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/>
