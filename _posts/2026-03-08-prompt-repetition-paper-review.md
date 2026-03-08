---
layout: post
title: "[논문 리뷰] Prompt Repetition Improves Non-Reasoning LLMs"
date: 2026-03-08 18:00:00 +0900
categories: [AI, Paper]
tags: [prompt-engineering, repetition, attention, inference]
---

> **논문 링크**: <https://arxiv.org/pdf/2512.14982>
>
> - **저자**: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
> - **발표일**: 2024년 12월 17일 (arXiv:2512.14982)
> - **테스트 시기**: 2025년 2~3월

---

## 1. 핵심 아이디어

이 논문은 매우 단순하지만 효과적인 프롬프트 기법을 제안한다. 입력 프롬프트를 `<QUERY>`에서 `<QUERY><QUERY>`로 변환하여 그대로 반복하는 것이다.

### 왜 효과가 있는가?

LLM은 주로 인과적(causal) 언어 모델로 훈련된다. 이 구조에서는 과거 토큰이 미래 토큰에 어텐션할 수 없다. 따라서 쿼리 내 토큰의 순서가 성능에 영향을 미친다.

예를 들어:

- `<컨텍스트> <질문>` 형태
- `<질문> <컨텍스트>` 형태

이 두 가지는 동일한 정보를 담고 있어도 성능이 다르게 나온다.

프롬프트를 반복하면 첫 번째 프롬프트의 각 토큰이 두 번째 프롬프트의 모든 토큰에 어텐션할 수 있게 되어, 실질적으로 모든 프롬프트 토큰이 서로 상호작용할 수 있게 된다.

### 추가 동기

RL로 훈련된 추론 모델들이 사용자 요청의 일부를 반복하는 것을 자연스럽게 학습한다는 관찰도 이 기법의 유용성을 뒷받침한다. 프롬프트 반복은 이러한 반복을 병렬화 가능한 prefill 단계로 옮기는 효과가 있다.

---

## 2. 실험 설계

### 테스트 모델 (7개)

| 제공사 | 모델 |
| --- | --- |
| Google | Gemini 2.0 Flash, Gemini 2.0 Flash-Lite |
| OpenAI | GPT-4o-mini, GPT-4o |
| Anthropic | Claude 3 Haiku, Claude 3.7 Sonnet |
| DeepSeek | Deepseek V3 |

### 벤치마크 (7개)

**표준 벤치마크 (5개)**

- ARC (Challenge) - 추론 문제
- OpenBookQA - 상식 질의응답
- GSM8K - 수학 문제
- MMLU-Pro - 다중 작업 언어 이해
- MATH - 수학 문제

**커스텀 벤치마크 (2개)**

- NameIndex - 목록에서 특정 위치의 이름 찾기
- MiddleMatch - 두 항목 사이의 항목 찾기

### 테스트 구성

- 객관식 벤치마크(ARC, OpenBookQA, MMLU-Pro)는 두 가지 방식으로 테스트:
  - **Question-first**: 질문을 먼저, 선택지를 나중에
  - **Options-first**: 선택지를 먼저, 질문을 나중에
- 통계적 유의성은 McNemar 테스트(p < 0.1)로 판정

---

## 3. 주요 결과

### 정확도 향상 (추론 비활성화 시)

| 지표 | 결과 |
| --- | --- |
| 총 테스트 | 70개 (7개 모델 × 10개 벤치마크 구성) |
| 통계적 유의미한 승리 | **47개** |
| 패배 | **0개** |
| 중립 | 23개 |

**특히 인상적인 결과:**

- Gemini 2.0 Flash-Lite의 NameIndex 정확도: **21.33% → 97.33%** (76%p 향상)

**패턴:**

- Question-first 구성: 작은 개선
- Options-first 구성: 큰 개선 (프롬프트 반복 없이는 모델이 질문을 모른 채 선택지를 처리해야 하므로)
- NameIndex, MiddleMatch: 모든 모델에서 강한 개선

### 효율성

프롬프트 반복은 다음을 증가시키지 않는다:

- 생성되는 출력의 길이
- 응답 지연 시간

이는 반복이 병렬화 가능한 prefill 단계에서만 영향을 미치기 때문이다.

**예외:** Anthropic 모델(Claude Haiku, Sonnet)은 매우 긴 요청(NameIndex, MiddleMatch, ×3 변형)에서 지연 시간이 증가했다. 이는 prefill 단계가 길어지기 때문으로 추정된다.

### 추론 모드에서의 효과 ("Think step by step")

| 지표 | 결과 |
| --- | --- |
| 총 테스트 | 28개 |
| 승리 | 5개 |
| 패배 | 1개 |
| 중립 | 22개 |

추론 모드에서는 효과가 중립에서 약간 긍정적이다. 이는 추론 과정에서 모델이 이미 프롬프트 일부를 반복하는 경향이 있기 때문이다.

---

## 4. 변형 기법 및 제거 실험 (Ablation)

### 테스트한 방법들

| 방법 | 템플릿 |
| --- | --- |
| **Baseline** | `<QUERY>` |
| **Prompt Repetition** | `<QUERY><QUERY>` |
| **Prompt Repetition (Verbose)** | `<QUERY> Let me repeat that: <QUERY>` |
| **Prompt Repetition ×3** | `<QUERY> Let me repeat that: <QUERY> Let me repeat that one more time: <QUERY>` |
| **Padding** | `<QUERY>` + 동일 길이의 마침표(".") |

### 구체적인 예시 (ARC 문제)

**Baseline:**

```
Which of the following combinations is a mixture rather than a compound?
A. oxygen and nitrogen in air
B. sodium and chlorine in salt
C. hydrogen and oxygen in water
D. nitrogen and hydrogen in ammonia
Reply with one letter ('A', 'B', 'C', 'D') in the format: The answer is <ANSWER>.
```

**Prompt Repetition:**

```
Which of the following combinations is a mixture rather than a compound?
A. oxygen and nitrogen in air
B. sodium and chlorine in salt
C. hydrogen and oxygen in water
D. nitrogen and hydrogen in ammonia
Reply with one letter ('A', 'B', 'C', 'D') in the format: The answer is <ANSWER>.
Which of the following combinations is a mixture rather than a compound?
A. oxygen and nitrogen in air
B. sodium and chlorine in salt
C. hydrogen and oxygen in water
D. nitrogen and hydrogen in ammonia
Reply with one letter ('A', 'B', 'C', 'D') in the format: The answer is <ANSWER>.
```

### 제거 실험 결과

- 모든 반복 변형이 baseline보다 우수하거나 유사한 성능
- **×3 반복**은 NameIndex, MiddleMatch에서 특히 우수
- **Padding은 성능 향상 없음** → 단순히 입력 길이 증가가 아닌, **실제 내용의 반복**이 핵심임을 입증

---

## 5. 커스텀 태스크 상세

### NameIndex

**설정:** N=50개의 이름 목록에서 i=25번째 이름을 찾는 과제

```
Here's a list of names:
Dale Lopez, Peter Sanchez, Allen Harris, Scott Davis, Hudson Leviathan,
Daphne Kalman, Dennis Davis, Henry King, Alfred Cooper, Bruce Usher,
Travis Ramirez, Rafael Jennings, Richard Rogers, Walter Young, Caleb Harris,
Ben Kalman, Donald Carter, Richard Sterling, Mark Nightingale, Steven Carter,
Talia Kalman, Dennis Hanson, James Harris, Craig Chavez, Paul Sanchez,
Samuel Curtis, Jacob James, Allen Thomas, Dale Evans, James Fox,
Douglas Allen, Orion Johnson, Alexander Wright, Eugene Morrison, Nelson Lee,
Alan Young, Caleb Ward, Alberto Robinson, Robert McCarthy, Mark Price,
Kenneth Ramirez, Jeffrey White, Chad Cooper, Arthur Waters, Bruce Callahan,
Liam Leviathan, Steven Robinson, Alberto Murphy, Leonard Johnson, Robert Murphy
What's the 25th name?
```

### MiddleMatch

**설정:** K=10개 중에서 선택하여 N=40개의 목록 생성 (중복 가능), 두 특정 항목 사이에 있는 항목 찾기

```
Here's a list (potentially with repetitions) of names:
Carlos Davis, Dale Sims, Carlos Davis, Dale Sims, Stephen Cruz, Dale Sims,
Finnian Ross, Stephen Cruz, Stephen Cruz, Gregory Collins, Dale Sims,
Stephen Cruz, Carlos Davis, Stephen Cruz, Dale Sims, Dale Sims, Stephen Cruz,
Stephen Cruz, Leonard Kalman, Bruce Phillips, Raymond Roberts, Dale White,
Leonard Kalman, Finnian Ross, James Wright, Finnian Ross, Raymond Roberts,
Dale Sims, Dale Sims, Leonard Kalman, Dale Sims, Carlos Davis, Leonard Kalman,
Bruce Phillips, Dale Sims, Raymond Roberts, Gregory Collins, Gregory Collins,
Dale Sims, Finnian Ross
What is the single name that appears right between Carlos Davis and Bruce Phillips?
```

---

## 6. 관련 연구와의 비교

| 기법 | 특징 | 프롬프트 반복과의 차이 |
| --- | --- | --- |
| **Chain of Thought (CoT)** | 작업별 예시 필요, 출력 길이 대폭 증가 | 프롬프트 반복은 예시 불필요, 출력 형식 불변 |
| **"Think step by step"** | 지연시간 및 계산량 크게 증가 | 프롬프트 반복과 병용 가능 (중립적 결과) |
| **질문 부분만 반복** (Shaier, 2024) | 효과 없음 | 전체 프롬프트 반복이 핵심 |
| **입력 2회 반복으로 임베딩 개선** (Springer et al., 2024) | 텍스트 임베딩 품질 향상 | 유사한 발견, 독립적 연구 |
| **Re-reading** (Xu et al., 2024) | 모델에게 질문을 다시 읽으라고 요청 | 추론 개선, 유사한 접근 |

---

## 7. 장점 요약

| 장점 | 설명 |
| --- | --- |
| **성능 향상** | 추론 없이도 70개 테스트 중 47개에서 통계적으로 유의미한 향상, 0개 패배 |
| **효율성 유지** | 생성 토큰 수, 지연 시간 증가 없음 |
| **출력 형식 불변** | 기존 시스템에 즉시 적용 가능 (drop-in deployment) |
| **단순성** | 별도의 fine-tuning이나 예시 없이 즉시 적용 가능 |
| **범용성** | 테스트된 모든 모델(Gemini, GPT, Claude, Deepseek)에서 효과 확인 |
| **최종 사용자 활용 가능** | 개발자뿐 아니라 일반 사용자도 직접 적용 가능 |

---

## 8. 한계점

- 매우 긴 프롬프트에서는 지연 시간에 영향을 줄 수 있음
- 컨텍스트 윈도우 한계에 가까운 프롬프트에서는 적용 불가능
- 추론 모드에서는 효과가 미미함

---

## 9. 향후 연구 방향

논문에서 제시한 13가지 연구 방향:

1. 반복된 프롬프트로 모델 fine-tuning
2. 프롬프트 반복으로 추론 모델 훈련 (모델이 반복을 피하는 법을 학습하여 효율성 증가)
3. 생성 중 마지막 토큰을 주기적으로 반복, 다중 턴 시나리오 적용
4. KV-cache에서 두 번째 반복만 유지 (생성 단계에서 완전히 성능 중립)
5. 긴 프롬프트에서 일부만 반복
6. 전체 반복 대신 작은 모델로 프롬프트 재정렬
7. 비텍스트 모달리티(이미지 등)에 적용
8. 2회 이상 반복이 유리한 경우 추가 분석
9. 반복으로 인한 어텐션 패턴 심층 분석
10. Selective Attention 등 다른 기법과 병용
11. Prefix LM과의 상호작용 탐구
12. 반복이 도움이 되는 시점과 토큰 표현 변화 조사
13. 유망한 변형 기법 추가 탐구

---

## 10. 실용적 시사점

이 연구는 **프롬프트 반복이 추론을 사용하지 않을 때 좋은 기본 설정이 될 수 있음**을 보여준다.

**적용하기 좋은 상황:**

- 빠른 응답이 필요할 때 (추론 모드 비활성화)
- 목록에서 특정 위치의 정보를 찾는 작업
- 선택지가 질문보다 먼저 나오는 구조의 문제
- 순차적 정보 처리가 필요한 작업

**적용 방법:** 그냥 프롬프트를 두 번 붙여넣기만 하면 된다. 필요하다면 "Let me repeat that:" 같은 연결 문구를 추가할 수 있다.
