---
layout: post
title: "Stanford CME295: Lecture 7 - Agentic LLMs (RAG, Tool Calling, Agents)"
date: 2026-03-08 11:10:00 +0900
categories: [AI, Lecture]
tags: [stanford-cme295, llm, rag, tool-calling, agent, react, mcp, retrieval]
math: true
---

> **원본 강의**: [YouTube - CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=7)

---

## 강의 개요

이 강의는 LLM을 외부 시스템과 연결하여 더 강력하게 만드는 실용적인 기법들을 다룹니다. 지금까지 LLM은 학습된 지식에만 의존했지만, 이제는 외부 정보를 검색하고, 도구를 사용하며, 자율적으로 목표를 달성하는 Agent로 발전시키는 방법을 배웁니다.

**강의 목표:**

1. RAG를 통해 LLM을 최신 정보에 연결하는 방법 이해
2. Tool Calling으로 LLM이 외부 시스템과 상호작용하는 방법 학습
3. Agent의 개념과 Agentic Workflow 설계 이해

---

## Part 1: RAG (Retrieval Augmented Generation)

### 1. LLM의 지식 한계

#### 1.1 Knowledge Cutoff 문제

모든 LLM은 학습 데이터의 **지식 마감일(Knowledge Cutoff)**이 존재합니다.

| 모델 | Knowledge Cutoff | Context Window |
| --- | --- | --- |
| GPT-5 | September 30, 2024 | 400,000 tokens |

**문제점:**

- Cutoff 이후 발생한 사건에 대해 답변 불가
- 최신 정보가 필요한 질문에 잘못된 답변 제공

**왜 추가 학습으로 해결하지 않는가?**

1. 기존 지식에 regression 발생 위험
2. Fine-tuned 모델마다 업데이트 필요 → 유지보수 부담

#### 1.2 Naive Approach의 문제점

**"모든 최신 정보를 Context에 넣자"의 한계:**

| 문제 | 설명 |
| --- | --- |
| Context Length 제한 | 수십만 토큰도 "모든 정보"를 담기엔 부족 |
| Needle in a Haystack | 긴 context에서 관련 정보 검색 성능 저하 |
| 비용 | API 호출은 토큰당 과금 (~$1/million tokens) |

**Needle in a Haystack Test:**

- Prompt가 길수록 성능 저하
- 특히 중간 위치의 정보 검색이 어려움

---

### 2. RAG 개요

**RAG = Retrieval Augmented Generation**

> 핵심 아이디어: 모든 정보를 넣는 대신, **관련 정보만** 검색하여 프롬프트에 추가

**3단계 프로세스:**

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ RETRIEVE │ → │ AUGMENT  │ → │ GENERATE │
│ 관련 문서 │    │ 프롬프트  │    │ 응답 생성 │
│ 검색     │    │ 확장     │    │          │
└──────────┘    └──────────┘    └──────────┘
```

**예시:**

```
Query: "지역 선거 승자는 누구인가요?"
  ↓ Retrieve
Retrieved: "2024년 11월 선거에서 A 후보가 당선..."
  ↓ Augment
Augmented Prompt: "다음 정보를 참고하세요: [retrieved info]. 질문: 지역 선거 승자는?"
  ↓ Generate
Response: "A 후보가 당선되었습니다."
```

---

### 3. Knowledge Base 구축

#### 3.1 Chunking (문서 분할)

문서를 검색 가능한 단위인 **Chunk**로 분할합니다.

**주요 하이퍼파라미터:**

| 파라미터 | 설명 | 일반적 값 |
| --- | --- | --- |
| Embedding Size | 임베딩 벡터 차원 | ~1,500 |
| Chunk Size | 청크당 토큰 수 | ~500 tokens |
| Chunk Overlap | 청크 간 중첩 토큰 | ~100 tokens |

**Chunk Overlap이 필요한 이유:**

- 문맥 연속성 유지
- 청크 경계에서 정보 손실 방지

#### 3.2 Embedding 생성

각 Chunk를 벡터로 변환하여 의미적 검색이 가능하게 합니다.

**Sentence-BERT:**

- BERT를 확장하여 문장/문서 수준 임베딩 생성
- 유사한 의미 → 높은 Cosine Similarity
- 다른 의미 → 낮은 Cosine Similarity

---

### 4. Retrieval: 2단계 검색

RAG의 핵심은 **좋은 검색**입니다.

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: Candidate Retrieval                        │
│   Millions of Chunks → Semantic Search → ~100 chunks│
│                        (Maximize Recall)            │
├─────────────────────────────────────────────────────┤
│ Stage 2: Reranking (Optional)                       │
│   ~100 Candidates → Cross-Encoder → Top K           │
│                     (Maximize Precision)            │
└─────────────────────────────────────────────────────┘
```

#### 4.1 Stage 1: Candidate Retrieval

**Bi-Encoder 방식:**

```
Query → [Encoder] → Query Embedding ─┐
                                      ├→ Cosine Similarity
Chunk → [Encoder] → Chunk Embedding ─┘
```

- Query와 Chunk를 **독립적으로** 인코딩
- 빠른 검색 가능 (ANN: Approximate Nearest Neighbor)
- 목표: **Recall 최대화** (관련 문서를 놓치지 않기)

**Cosine Similarity:**

$$\text{cos}(\theta) = \frac{Q \cdot C}{\|Q\| \cdot \|C\|}$$

**Semantic Search vs Keyword Search:**

| 방식 | 장점 | 단점 |
| --- | --- | --- |
| Semantic (Embedding) | 의미적 유사성 포착 | 키워드 매칭 보장 안됨 |
| BM25 (Keyword) | 정확한 키워드 매칭 | 의미적 유사성 무시 |

**Hybrid Search:** 실무에서는 두 방식을 결합하여 사용

#### 4.2 HyDE (Hypothetical Document Embedding)

**문제:** Query와 Document는 성격이 다름

- Query: 짧은 질문
- Document: 긴 설명문

**해결책:**

```
Query → [LLM] → 가상 Document 생성 → [Encoder] → Embedding
```

가상 문서를 생성하여 Document 스타일의 임베딩으로 검색

#### 4.3 Contextual Chunking

**문제:** 나이브한 Chunking은 맥락을 잃을 수 있음

**해결책:** 각 Chunk 앞에 문맥 정보를 추가

```
[Document 전체] + [Chunk] → [LLM] → "이 청크 이해에 필요한 맥락"
  ↓
Prepend to Chunk
```

**Prompt Caching으로 비용 절감:**

- 동일한 Document prefix는 한 번만 계산
- Cached input token: 정가의 1/10 비용

#### 4.4 Stage 2: Reranking

**Cross-Encoder 방식:**

```
[Query + Chunk] → [Encoder] → Relevance Score
```

- Query와 Chunk를 **함께** 인코딩
- 더 정교한 관련성 점수 계산
- 계산 비용이 높아 소수의 후보에만 적용

**Bi-Encoder vs Cross-Encoder:**

| 방식 | 구조 | 속도 | 정확도 |
| --- | --- | --- | --- |
| Bi-Encoder | Query, Chunk 분리 | 빠름 | 보통 |
| Cross-Encoder | Query + Chunk 결합 | 느림 | 높음 |

---

### 5. RAG 평가 메트릭

#### 5.1 NDCG (Normalized Discounted Cumulative Gain)

**아이디어:** 관련 문서가 높은 순위에 있을수록 좋음

$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

- $rel_i$: i번째 위치 문서의 실제 관련성 (0 또는 1)
- $IDCG$: 이상적인 DCG (최선의 배치)
- **값 범위:** 0 ~ 1 (1이 최적)

**직관:**

- Discounted: 낮은 순위일수록 가치 감소
- Cumulative: 모든 관련 문서 고려
- Normalized: 0~1 스케일로 정규화

**NDCG (Normalized Discounted Cumulative Gain) 상세 설명**

##### 1. NDCG란?

NDCG는 검색/추천 시스템의 랭킹 품질을 평가하는 지표입니다.

주로 사용되는 곳:

- **RAG의 Retrieval 성능 평가**
- 검색 엔진 결과 품질 측정
- 추천 시스템 평가

##### 2. 단계별 이해

NDCG를 이해하려면 **CG → DCG → NDCG** 순서로 이해해야 합니다.

**2.1 CG (Cumulative Gain)**

정의: 검색 결과의 관련성 점수를 단순 합산

$$CG_k = \sum_{i=1}^{k} rel_i$$

- $rel_i$: i번째 문서의 관련성 점수 (예: 0, 1, 2, 3)
- $k$: 상위 k개 결과

**예시**

```
검색어: "Transformer 논문"

검색 결과 (순서대로):

| 순위 | 문서                    | 관련성(rel) |
|------|------------------------|-----------|
|  1   | "RNN 튜토리얼"          |     0     |
|  2   | "Attention is All..."  |     3     | ← 정답!
|  3   | "BERT 논문"            |     2     |
|  4   | "CNN 기초"              |     0     |
|  5   | "GPT 논문"              |     2     |

CG@5 = 0 + 3 + 2 + 0 + 2 = 7
```

**CG의 문제점**

**순서를 고려하지 않음!**

```
결과 A: [3, 2, 2, 0, 0] → CG = 7
결과 B: [0, 0, 2, 2, 3] → CG = 7  (같음!)

하지만 결과 A가 훨씬 좋음! (관련 문서가 앞에 있음)
```

**2.2 DCG (Discounted Cumulative Gain)**

핵심 아이디어: 뒤에 있는 결과일수록 가치를 할인(Discount)

$$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

또는 (더 일반적인 버전):

$$DCG_k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

**할인 계수 (Discount Factor)**

| 순위 (i) | $\log_2(i+1)$ | 할인 계수 $\frac{1}{\log_2(i+1)}$ |
| --- | --- | --- |
| 1 | 1.0 | **1.00** |
| 2 | 1.58 | **0.63** |
| 3 | 2.0 | **0.50** |
| 4 | 2.32 | **0.43** |
| 5 | 2.58 | **0.39** |

→ 순위가 낮을수록 할인이 커짐 (가치 감소)

**예시**

```
DCG 계산 예시

결과 A: [3, 2, 2, 0, 0]

DCG = 3/log₂(2) + 2/log₂(3) + 2/log₂(4) + 0/log₂(5) + 0/log₂(6)
    = 3/1.0 + 2/1.58 + 2/2.0 + 0 + 0
    = 3.0 + 1.26 + 1.0 + 0 + 0
    = 5.26

──────────────────────────────────

결과 B: [0, 0, 2, 2, 3]

DCG = 0/log₂(2) + 0/log₂(3) + 2/log₂(4) + 2/log₂(5) + 3/log₂(6)
    = 0 + 0 + 1.0 + 0.86 + 1.16
    = 3.02

──────────────────────────────────

결과: A(5.26) > B(3.02)  ✅ 올바른 평가!
```

**DCG의 문제점**

**쿼리마다 스케일이 다름**

```
쿼리 1: 관련 문서 10개 → DCG 최대값 큼
쿼리 2: 관련 문서 2개  → DCG 최대값 작음

→ 서로 다른 쿼리의 DCG를 직접 비교할 수 없음
```

**2.3 NDCG (Normalized DCG)**

핵심 아이디어: DCG를 **이상적인 DCG (IDCG)**로 나누어 정규화

$$NDCG_k = \frac{DCG_k}{IDCG_k}$$

- **IDCG (Ideal DCG):** 완벽한 순서일 때의 DCG (관련성 높은 순으로 정렬)
- **NDCG 범위:** 0~1 (1이 완벽)

**예시**

```
NDCG 계산 예시

실제 결과: [0, 3, 2, 0, 2]  (관련성 점수)
이상적 순서: [3, 2, 2, 0, 0]  (높은 순으로 정렬)

──────────────────────────────────

DCG 계산 (실제):
DCG = 0/1.0 + 3/1.58 + 2/2.0 + 0/2.32 + 2/2.58
    = 0 + 1.90 + 1.0 + 0 + 0.78
    = 3.68

IDCG 계산 (이상적):
IDCG = 3/1.0 + 2/1.58 + 2/2.0 + 0/2.32 + 0/2.58
     = 3.0 + 1.26 + 1.0 + 0 + 0
     = 5.26

NDCG = DCG / IDCG = 3.68 / 5.26 = 0.70

해석: 이상적인 랭킹의 70% 수준
```

##### 3. NDCG 전체 공식 요약

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

$$DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

$$IDCG@k = \sum_{i=1}^{k} \frac{rel_i^{ideal}}{\log_2(i+1)}$$

여기서 $rel_i^{ideal}$은 관련성을 내림차순 정렬한 값

##### 4. Python 구현

```python
import numpy as np

def dcg_at_k(relevances, k):
    """DCG@k 계산"""
    relevances = np.array(relevances)[:k]
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    return np.sum(relevances / discounts)

def ndcg_at_k(relevances, k):
    """NDCG@k 계산"""
    # 실제 DCG
    dcg = dcg_at_k(relevances, k)

    # 이상적 DCG (내림차순 정렬)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    # NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg

# 예시
relevances = [0, 3, 2, 0, 2]  # 실제 검색 결과의 관련성

print(f"DCG@5: {dcg_at_k(relevances, 5):.2f}")       # 3.68
print(f"IDCG@5: {dcg_at_k(sorted(relevances, reverse=True), 5):.2f}")  # 5.26
print(f"NDCG@5: {ndcg_at_k(relevances, 5):.2f}")      # 0.70
```

##### 5. RAG에서의 NDCG 활용

```
RAG Retrieval 평가

Query: "Transformer의 Self-Attention이란?"

Retrieved Documents (Top 5):

| 순위 | 문서                        | 관련성 |
|------|----------------------------|--------|
|  1   | "CNN 아키텍처 설명"          |   0    |
|  2   | "Attention Is All You Need"|   3    | ← 최고!
|  3   | "BERT의 Self-Attention"    |   2    |
|  4   | "RNN의 한계"               |   1    |
|  5   | "Transformer 구조"         |   2    |

NDCG@5 = 0.75

문제점: 가장 관련 있는 문서(rel=3)가 2번째에 있음
개선 방향: Retriever/Reranker 성능 향상 필요
```

##### 6. 다른 Retrieval 메트릭과 비교

| 메트릭 | 특징 | 관련성 레벨 |
| --- | --- | --- |
| **Precision@k** | 상위 k개 중 관련 문서 비율 | Binary (0/1) |
| **Recall@k** | 전체 관련 문서 중 검색된 비율 | Binary (0/1) |
| **MRR** | 첫 번째 관련 문서의 순위 역수 | Binary (0/1) |
| **NDCG@k** | 순위와 관련성 정도 모두 고려 | Graded (0,1,2,3...) ✅ |

**NDCG의 장점:**

- 순위 고려 (앞에 있을수록 가치 높음)
- 다단계 관련성 지원 (0/1뿐만 아니라 0,1,2,3 등)
- 0~1 정규화로 쿼리 간 비교 가능

##### 7. 핵심 정리

| 개념 | 설명 |
| --- | --- |
| **CG** | 관련성 점수 단순 합산 (순서 무시) |
| **DCG** | 순위에 따라 할인 적용 (뒤로 갈수록 가치 감소) |
| **IDCG** | 이상적인 순서일 때의 DCG |
| **NDCG** | DCG/IDCG (0~1 정규화) |
| **핵심** | "좋은 문서가 앞에 있을수록 높은 점수" |

$$NDCG@k = \frac{\sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}}{\sum_{i=1}^{k} \frac{rel_i^{sorted}}{\log_2(i+1)}}$$

#### 5.2 MRR (Mean Reciprocal Rank)

$$MRR = \frac{1}{rank_{\text{first relevant}}}$$

- 첫 번째 관련 문서의 순위만 고려
- 단순하지만 효과적

#### 5.3 Precision@K & Recall@K

| 메트릭 | 공식 | 의미 |
| --- | --- | --- |
| Precision@K | $\frac{\text{Top K 중 관련 문서}}{\text{K}}$ | 검색 결과의 정밀도 |
| Recall@K | $\frac{\text{Top K 중 관련 문서}}{\text{전체 관련 문서}}$ | 관련 문서 커버리지 |

**벤치마크:** MTEB (Massive Text Embedding Benchmark)

---

## Part 2: Tool Calling

### 6. Tool Calling 개요

> "Tool calling allows autonomous systems to complete complex tasks by dynamically accessing and acting upon external resources." — IBM

**핵심 개념:**

1. 특정 **Task 완수**를 위해
2. **외부 리소스**에 접근하여 행동 수행

**RAG vs Tool Calling:**

| 방식 | 데이터 형태 | 용도 |
| --- | --- | --- |
| RAG | 비정형 (문서, 텍스트) | 정보 검색 |
| Tool Calling | 정형 (함수, API) | 계산, 액션 실행 |

**Tool Selection/Calling 매커니즘 상세**

##### 1. LLM이 받는 정보

Tool selection은 **name**과 **summary**만으로 이루어지는 게 아니라, 더 풍부한 정보를 기반으로 합니다:

```
Tool Definition (System Prompt에 포함)

• name: "web_search"
• description: "Search the web for..."
• parameters: {
      "query": {type: string, required}
  }
• (선택적) examples, when to use/not use
```

##### 2. Selection 과정 (LLM 내부)

```
User Query → LLM이 의도 파악 → Tool Descriptions 매칭 → Tool 선택 & Parameters 추출
```

핵심: LLM은 자연어 이해를 통해 선택합니다. 별도의 classifier나 retrieval 시스템이 아니라, LLM 자체가 다음을 수행해요:

1. **사용자 의도 파악:** "오늘 날씨 어때?" → 현재 정보 필요
2. **Tool과 매칭:** description을 읽고 `web_search`가 적합하다고 판단
3. **Parameter 추출:** query = "오늘 날씨"

##### 3. 실제 호출 방식

LLM은 특정 포맷으로 tool call을 출력합니다.

##### 4. Selection의 핵심 포인트

**LLM은 단순 키워드 매칭이 아닌 의미론적 이해로 선택합니다:**

| 요소 | 역할 |
| --- | --- |
| **name** | 빠른 식별용 (web_search, bash_tool...) |
| **description** | 언제/왜 사용하는지 설명 |
| **parameters schema** | 어떤 입력이 필요한지 |
| **예시/가이드라인** | 더 정교한 판단 지원 |

##### 5. 실제 Selection 프로세스 (내부적으로)

```
1. 사용자 쿼리 이해
   "오늘 날씨 어때?"
   → 현재 정보 필요 + 날씨 도메인

2. 각 Tool Description과 매칭
   - bash_tool: "Run bash command" → ❌ 관계없음
   - web_search: "Search the web" → ✅ 현재 정보 가능
   - create_file: "Create new file" → ❌ 관계없음

3. Parameter 생성
   query = "오늘 서울 날씨" (사용자 위치 + 의도 추론)

4. 호출 형식으로 출력
```

**요약**

Tool Selection은 LLM의 자연어 이해 능력을 그대로 활용합니다. name/summary만이 아니라 전체 tool definition(description, parameters, examples)을 context로 받고, LLM이 사용자 의도와 가장 잘 매칭되는 tool을 선택해서 정해진 포맷으로 호출을 출력합니다.

---

### 7. Function Calling 예시

**시나리오:** "내 근처에서 테디베어 찾기"

#### 7.1 함수 정의

```python
def find_teddy_bear(latitude: float, longitude: float) -> TeddyBearResult:
    """
    주어진 위치에서 가장 가까운 테디베어를 찾습니다.
    Args:
        latitude: 위도
        longitude: 경도
    Returns:
        TeddyBearResult: 이름, 위치, 거리 정보
    """
    response = teddy_bear_api.search(lat=latitude, lon=longitude)
    return TeddyBearResult(name=response.name, location=response.location)
```

**중요:** LLM은 함수의 **시그니처와 문서화**만 볼 수 있음 (구현 X)

#### 7.2 Tool Calling 3단계 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Argument Prediction                                │
│   User Query + Function API → LLM predicts arguments        │
│   find_teddy_bear(37.4, -122.1)                             │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Function Execution                                 │
│   Execute function with predicted arguments                 │
│   → Returns: {name: "Cuddly", location: "Stanford"}        │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: Response Generation                                │
│   Conversation + Function Result → Final Response           │
│   "근처에 'Cuddly'라는 테디베어가 Stanford에 있습니다!"        │
└─────────────────────────────────────────────────────────────┘
```

---

### 8. Tool Calling 학습 방법

#### 8.1 SFT (Supervised Fine-Tuning) 방식

**두 종류의 SFT 데이터 필요:**

| SFT Pair | Input | Output |
| --- | --- | --- |
| Tool Prediction | Query + Function API | Function Call |
| Response Generation | Conversation + Tool Result | Final Response |

#### 8.2 Prompt Engineering 방식

최신 LLM은 코드 이해 능력이 뛰어나므로, **SFT 없이 설명만으로** 가능할 수 있음

**자동 프롬프트 최적화:**

```
1. 초기 설명 작성
2. 평가 세트로 테스트 (Query → Expected Tool Call)
3. 실패 케이스 분석
4. Reasoning Model로 설명 개선
5. 반복
```

> **Tip:** 설명을 직접 작성하기보다, 강력한 Reasoning Model에게 작성을 위임

---

### 9. Tool 종류와 활용

| 카테고리 | 예시 | 용도 |
| --- | --- | --- |
| **Information** | Search, Weather, Stocks | 최신 정보 검색 |
| **Computation** | Calculator, Code Execution | 정확한 계산 |
| **Action** | Send Email, Set Thermostat | 사용자 대신 행동 |

---

### 10. Tool Selection (Tool Routing)

**문제:** 너무 많은 Tool이 Context에 있으면 성능 저하

**해결책:** 2단계 선택 프로세스

```
┌────────────────────────────────────────────────────┐
│ Stage 1: Tool Selection                            │
│   Query + All Tool Names/Summaries                 │
│   → Select relevant tools (2-3 tools)              │
├────────────────────────────────────────────────────┤
│ Stage 2: Tool Calling                              │
│   Query + Selected Tool APIs (full)                │
│   → Execute tool call                              │
└────────────────────────────────────────────────────┘
```

**대안:** RAG와 유사하게 Tool 검색도 가능

---

### 11. MCP (Model Context Protocol)

**문제:** 각 LLM마다 Tool 정의 방식이 다름 → 중복 구현

**해결책:** Anthropic이 제안한 **표준 프로토콜**

**MCP 구성요소:**

| 요소 | 설명 |
| --- | --- |
| **MCP Server** | Tool을 제공하는 서버 |
| **Tools** | 실제 함수 구현 |
| **Prompts** | Tool 사용 방법 템플릿 |
| **Resources** | 외부 데이터베이스/리소스 |
| **MCP Client** | LLM Host 측 연결 인프라 |

**예시: 책 추천 시스템**

```
MCP Server: Book Provider
├── Tools: find_book(), recommend_book()
├── Prompts: "제목으로 책 찾기", "취향 기반 추천"
└── Resources: User's collection, Bestseller list
```

---

## Part 3: Agents

### 12. Agent 정의

> "A system that **autonomously** pursues goals and completes tasks on a user's behalf."

**Tool Calling vs Agent:**

| 특성 | Tool Calling | Agent |
| --- | --- | --- |
| 실행 횟수 | Single call | Multiple iterations |
| 의사결정 | 단순 매핑 | 추론 기반 결정 |
| 목표 | 함수 실행 | 목표 달성 |

**Agent의 핵심:** 반복적 추론 + Tool 활용 + 목표 도달 판단

---

### 13. ReAct Framework

**ReAct = Reason + Act**

복잡한 Task를 Observe-Plan-Act 루프로 분해

```
┌─────────────────────────────────────────────────────┐
│ User Query: "테디베어가 추워해요"                      │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌──────────────────────────────┐
│ OBSERVE                       │
│ "추위 → 온도 문제일 수 있음"    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ PLAN                          │
│ "온도를 확인해야 함"            │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ ACT                           │
│ get_room_temperature()        │
│ → Returns: 65°F               │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ OBSERVE                       │
│ "65°F는 예상보다 낮음"         │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ ACT                           │
│ set_temperature(increase=5)   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ OUTPUT                        │
│ "온도를 5도 올렸습니다!"       │
└──────────────────────────────┘
```

---

### 14. Multi-Agent Systems

**단일 Agent → 다중 Agent 협업**

**Agent-to-Agent Protocol (Google):**

- Agent 간 통신 표준화
- 각 Agent가 **Skills** 노출
- 요청 실행 상태 관리 (시작, 진행중, 취소, 완료)

**예시:** 스마트홈

```
User Query → [Router Agent]
              ├→ [Thermostat Agent]
              ├→ [Energy Agent]
              └→ [Air Quality Agent]
```

---

### 15. Agent Safety

**새로운 Capability = 새로운 위험**

#### 15.1 주요 위험 요소

| 위험 | 설명 | 예시 |
| --- | --- | --- |
| Data Exfiltration | 민감 정보 외부 유출 | 비밀번호를 이메일로 전송 |
| Unauthorized Actions | 의도치 않은 행동 실행 | 잘못된 결제 실행 |
| Prompt Injection | 악의적 프롬프트 주입 | Tool 악용 유도 |

**참고 논문:** ToolSword (Tool 기반 공격 분석)

#### 15.2 방어 전략

| 단계 | 방어 방법 |
| --- | --- |
| Training | Harmlessness 데이터로 SFT/RLHF |
| Inference | Safety Classifier로 출력 검증 |

**벤치마크:** AgentSafetyBench

---

### 16. Agent의 현재 한계

**왜 "Agent가 세상을 지배"하지 않는가?**

1. **Error Compounding:** 매 단계 오류 → 전체 실패
2. **Grounding 실패:** Tool 인자 예측 오류
3. **Reasoning 한계:** 복잡한 다단계 추론의 어려움

---

### 17. 실용적 조언

**Agent/Tool 개발 시:**

| 원칙 | 설명 |
| --- | --- |
| **Start Small** | 단순한 케이스부터 시작 |
| **Start Smart** | 가장 강력한 모델로 시작 → 최적화는 나중에 |
| **Debug with Reasoning Chains** | LLM의 추론 과정을 로깅하고 분석 |

**현재 가장 유용한 Agent 활용:** AI Coding Assistant

- 복잡한 코드 작성 위임
- 단, 코드 품질 판단은 인간의 몫!

> "Generating code is cheap, but judging whether code is correct — that's the hard part."

---

## 핵심 요약

### LLM 확장 방법 비교

```
┌────────────┬─────────────┬────────────┬─────────────┐
│ 방식       │ RAG         │ Tool       │ Agent       │
│            │             │ Calling    │             │
├────────────┼─────────────┼────────────┼─────────────┤
│ 목적       │ 지식 확장    │ 기능 확장   │ 자율 실행    │
│ 데이터     │ 비정형 문서  │ 정형 API   │ 둘 다 활용   │
│ 실행 방식  │ 단일 검색    │ 단일 호출   │ 반복 루프    │
│ 복잡도     │ 낮음        │ 중간       │ 높음         │
└────────────┴─────────────┴────────────┴─────────────┘
```

### RAG 파이프라인

```
Documents → Chunking → Embedding → Knowledge Base
                                        ↓
Query → Candidate Retrieval → Reranking → Top-K → LLM → Response
```

### Tool Calling 흐름

```
Query + Function API → Argument Prediction → Execution → Response Generation
```

### Agent 루프

```
Query → [Observe → Plan → Act]* → Goal Reached? → Output
```

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| RAG | Retrieval Augmented Generation |
| Knowledge Cutoff | 모델 학습 데이터의 마감 시점 |
| Chunking | 문서를 검색 단위로 분할 |
| Bi-Encoder | Query와 Document를 독립적으로 인코딩 |
| Cross-Encoder | Query와 Document를 함께 인코딩 |
| BM25 | 키워드 기반 관련성 점수 |
| HyDE | Hypothetical Document Embedding |
| NDCG | Normalized Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank |
| MCP | Model Context Protocol |
| ReAct | Reason + Act (Agent framework) |
| A2A | Agent to Agent Protocol |

---

## 추천 자료

1. **"Sentence-BERT"** - 문장 수준 임베딩 학습
2. **"HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels"** - 가상 문서 기반 검색
3. **"ReAct: Synergizing Reasoning and Acting in Language Models"** - Agent 프레임워크
4. **"ToolSword"** - Tool 기반 공격 및 방어 분석
5. **"AgentSafetyBench"** - Agent 안전성 벤치마크
6. **MCP Documentation** - Model Context Protocol
7. **Google A2A Protocol** - Agent 간 통신 표준

---

*Stanford CME295: Transformers & LLMs | Autumn 2025 | Lecture 7 정리*
