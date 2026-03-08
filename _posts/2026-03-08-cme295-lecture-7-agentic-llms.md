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

![RAG 3단계](/assets/img/cme295-lecture-7/image-20260116-020904.png)

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

![2단계 검색](/assets/img/cme295-lecture-7/image-20260116-020927.png)

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

![Cosine Similarity](/assets/img/cme295-lecture-7/image-20260116-020942.png)

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

![NDCG](/assets/img/cme295-lecture-7/image-20260116-021014.png)

- $rel_i$: i번째 위치 문서의 실제 관련성 (0 또는 1)
- $IDCG$: 이상적인 DCG (최선의 배치)
- **값 범위:** 0 ~ 1 (1이 최적)

**직관:**

- Discounted: 낮은 순위일수록 가치 감소
- Cumulative: 모든 관련 문서 고려
- Normalized: 0~1 스케일로 정규화

<details>
<summary>NDCG (Normalized Discounted Cumulative Gain) 상세 설명</summary>

![NDCG 상세 1](/assets/img/cme295-lecture-7/image-20260122-095654.png)

![NDCG 상세 2](/assets/img/cme295-lecture-7/image-20260122-095726.png)

![NDCG 상세 3](/assets/img/cme295-lecture-7/image-20260122-095753.png)

![NDCG 상세 4](/assets/img/cme295-lecture-7/image-20260122-095807.png)

![NDCG 상세 5](/assets/img/cme295-lecture-7/image-20260122-095819.png)

</details>

#### 5.2 MRR (Mean Reciprocal Rank)

$$MRR = \frac{1}{rank_{\text{first relevant}}}$$

![MRR](/assets/img/cme295-lecture-7/image-20260116-021035.png)

- 첫 번째 관련 문서의 순위만 고려
- 단순하지만 효과적

#### 5.3 Precision@K & Recall@K

![Precision & Recall](/assets/img/cme295-lecture-7/image-20260116-021052.png)

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

<details>
<summary>Tool Selection/Calling 매커니즘 상세</summary>

![Tool Calling 상세](/assets/img/cme295-lecture-7/image-20260122-102725.png)

</details>

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

![Tool Calling 프로세스](/assets/img/cme295-lecture-7/image-20260116-021114.png)

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

![Tool Selection](/assets/img/cme295-lecture-7/image-20260116-021139.png)

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

![LLM 확장 방법 비교](/assets/img/cme295-lecture-7/image-20260116-021221.png)

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
