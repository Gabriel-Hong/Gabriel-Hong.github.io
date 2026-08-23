---
layout: post
title: "Stanford CS224G: Lecture 7 - Agent Orchestration & Workflow Design"
date: 2026-08-23 10:40:00 +0900
categories: [AI, Lecture, Stanford CS224G]
tags: [stanford-cs224g, agent, workflow, orchestration, langgraph, crewai, mcp, react]
---

> **강의 출처**: Stanford CS224G - Building & Scaling LLM Applications (Winter 2026)
>
> - **강사**: Rakshit Agrawal (Principal Applied Scientist, Microsoft)
> - **일자**: 2026년 1월 27일 (Winter 2026, Week 4)
> - **실습 노트북**: `agent_orchestration_lecture.ipynb` ([Colab](https://colab.research.google.com/drive/1Jpudjl_y9gQSx6nOhBjFgAB6bjdJbT-P?usp=sharing))
> - **원본 슬라이드**: <https://web.stanford.edu/class/cs224g/schedule.html>

---

## 강의 개요

이 강의는 "에이전트를 어떻게 만드는가"가 아니라 **"에이전트 시스템을 어떻게 구조화하는가"**를 다룹니다. 프롬프트 기법이 아니라 아키텍처 강의입니다.

**강의 목표:**

1. 언제 에이전트를 쓰고 언제 쓰지 말아야 하는지 판단하기
2. 7가지 표준 워크플로우 패턴을 어휘로 갖추기
3. 오케스트레이션 프레임워크의 선택 기준 세우기
4. 도구·프로토콜·상태·안전을 설계 요소로 다루기

**선수 지식:**

- LLM API 기본 사용 (chat completion, 시스템 프롬프트)
- Python 기초
- 상태 기계(state machine) 개념이 있으면 Part 3이 훨씬 쉽습니다

---

## Part 1: Foundations — Agent란 무엇인가

### 1. Agent의 정의

강의는 Anthropic의 정의를 채택합니다.

> "Systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."
> (LLM이 자신의 프로세스와 도구 사용을 **동적으로 지휘**하며, 과제를 어떻게 달성할지에 대한 통제권을 유지하는 시스템)

#### 1.1 핵심 구분: Workflow vs Agent

이 강의 전체를 관통하는 가장 중요한 구분입니다.

| | Workflow | Agent |
| --- | --- | --- |
| 경로 | 사전에 정의된 코드 경로 | 모델이 런타임에 결정 |
| 결정 주체 | 개발자 | LLM |
| 성질 | 결정론적(deterministic) | 동적(dynamic) |
| 디버깅 | 쉬움 — 경로가 고정 | 어려움 — 매 실행이 다름 |

**왜 이 구분이 중요한가?**

두 가지가 섞이면 시스템을 디버깅할 수 없게 됩니다. "왜 이런 결과가 나왔지?"라는 질문에 답하려면 그 지점이 워크플로우였는지 에이전트였는지부터 알아야 합니다. 워크플로우 구간의 버그는 코드를 고치면 되지만, 에이전트 구간의 버그는 프롬프트·도구·평가를 다시 봐야 합니다.

**실무 판단 기준:**

> 경로를 알면 워크플로우, 모르면 에이전트.

---

### 2. Agency의 네 기둥 (Albert Bandura)

강의는 심리학자 앨버트 반두라의 인간 행위주체성(human agency) 이론을 빌려옵니다. 반두라는 사회인지이론에서 행위주체성의 네 가지 속성을 제시했습니다.

| 기둥 | 원어 | 의미 | 시스템에서의 대응물 |
| --- | --- | --- | --- |
| 1. 의도성 | Intentionality | 스스로 목표를 세움 | Planner / 작업 분해 |
| 2. 예견 | Forethought | 미래 상태를 예측함 | 계획 수립, 시뮬레이션 |
| 3. 자기반응성 | Self-Reactiveness | 자신의 수행을 모니터링함 | 실행 결과 관측, 로깅 |
| 4. 자기성찰성 | Self-Reflectiveness | 결과에 따라 전략을 조정함 | 재시도, 자기수정 루프 |

**노트 — 이걸 설계 체크리스트로 쓰기**

네 기둥은 철학적 장식이 아니라 **진단 도구**로 쓸 때 유용합니다. 만들고 있는 시스템에 대해 넷을 하나씩 물어보면, 어디가 비어 있는지가 드러납니다.

- 목표를 스스로 분해하는가? (없으면 → 그냥 파이프라인)
- 결과를 관측하는가? (없으면 → 실패를 알 수 없음)
- 관측 결과로 전략을 바꾸는가? (없으면 → 자기수정 불가)

특히 **3번과 4번이 없으면 그것은 에이전트가 아니라 LLM 호출이 들어간 절차**입니다. 이 판정 기준은 뒤의 Evaluator-Optimizer 패턴과 직접 연결됩니다.

---

### 3. 언제 에이전트를 쓰는가

Anthropic의 권고를 그대로 인용합니다.

> "Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short."
> (단순한 프롬프트로 시작하고, 충분한 평가로 최적화한 뒤, **더 단순한 해법이 부족할 때에만** 다단계 에이전트 시스템을 추가하라)

#### 3.1 트레이드오프

| | 내용 |
| --- | --- |
| 장점 | 유연성, 처음 보는 과제 처리 가능 |
| 단점 | 높은 지연시간, 높은 비용 |

#### 3.2 워크플로우로 충분한 경우

- 단계가 알려진, 잘 정의된 과제
- 예측 가능성과 일관성이 최우선인 경우

**노트 — 비용의 실제 크기**

강의는 "higher cost"라고만 하지만 체감 규모를 알아두면 판단이 쉬워집니다. 에이전트 루프는 매 턴마다 지금까지의 대화 전체를 다시 보냅니다. 즉 **턴 수에 대해 토큰이 선형이 아니라 제곱에 가깝게 증가**합니다. 10턴짜리 루프는 단일 호출의 10배가 아니라 수십 배가 될 수 있습니다. "에이전트를 쓸까?"는 미학적 선택이 아니라 비용 결정입니다.

---

### 4. AI 에이전트의 유형 (Russell & Norvig)

인공지능 표준 교과서 *Artificial Intelligence: A Modern Approach*(AIMA)의 분류입니다. 단순→복잡 순서입니다.

| 유형 | 설명 | 예시 |
| --- | --- | --- |
| 1. Simple Reflex Agent | 조건-행동 규칙만 | 더러우면 청소한다 |
| 2. Model-Based Reflex Agent | 내부 상태(기억)를 유지 | 이미 청소한 방을 기억 |
| 3. Goal-Based Agent | 목표 달성을 위해 계획 | 모든 방 청소를 위한 경로 계획 |
| 4. Utility-Based Agent | 효용 함수를 최적화 | 시간·전력 대비 최적 경로 |
| 5. Learning Agent | 시간이 지나며 성능 개선 | 경험으로 경로 개선 |

> 현대의 대부분의 "AI 에이전트"는 **Goal-Based 이상**입니다.

**노트 — 이 분류가 왜 아직 유효한가**

1995년 교과서의 분류가 LLM 시대에도 통하는 이유는, 이 분류가 모델이 아니라 **시스템이 무엇을 유지하는가**를 기준으로 하기 때문입니다. LLM 자체는 stateless한 Simple Reflex에 가깝습니다. 우리가 상태·목표·평가를 바깥에 붙여서 등급을 올리는 것이고, 그 붙이는 방법이 이 강의의 나머지 전부입니다.

---

## Part 2: Agent Architectures

### 5. Single-Agent Architecture

**정의:** 중앙집중적으로 의사결정하는 단일 자율 개체

**구조:** LLM 하나 + 도구 세트 하나 + 컨텍스트 윈도우 하나

| 강점 | 약점 |
| --- | --- |
| 단순함, 낮은 지연시간, 디버깅 용이 | 컨텍스트 윈도우가 빠르게 참 |
| 통합 이슈가 적음 | 복잡한 다중 도메인 과제에 취약 |

---

### 6. ReAct 패턴 — 단일 에이전트의 기본 루프

**ReAct = Reasoning + Acting**

```
1. Thought      : "파리 인구를 찾아야겠다."
2. Action       : search("population of Paris")
3. Observation  : "파리 인구는 210만 명..."
4. Thought      : "답을 얻었다."
5. Final Answer : "약 210만 명입니다."
```

#### 6.1 루프의 실제 동작

| 단계 | 수행 주체 | 산출물 |
| --- | --- | --- |
| Thought | LLM | 다음에 무엇을 할지에 대한 추론 텍스트 |
| Action | LLM | 도구 호출(JSON) |
| Observation | **런타임(우리 코드)** | 도구 실행 결과 |
| Repeat | — | LLM이 최종 답을 낼 때까지 |

**한계:** 순차 실행. 각 도구 호출이 이전 호출을 기다립니다.

**노트 — 출처와 배경**

ReAct는 Yao et al., *"ReAct: Synergizing Reasoning and Acting in Language Models"* (2022, ICLR 2023)에서 제안되었습니다. 핵심 발견은 **추론만 하는 모델(Chain-of-Thought)은 환각을 일으키고, 행동만 하는 모델은 계획을 못 세우는데, 둘을 교대로 시키면 서로를 교정한다**는 것이었습니다. Observation이 추론을 현실에 붙들어 매는 역할을 합니다.

오늘날 거의 모든 "에이전트"는 내부적으로 ReAct 루프입니다. 프레임워크가 이걸 감싸고 있을 뿐입니다.

---

### 7. Multi-Agent Systems (MAS)

**정의:** 전문화된 에이전트들이 협업해 문제를 해결

**핵심 통찰:** **문제 공간을 분할한다 (Partition the problem space)**

- 각 에이전트가 좁은 역할을 가짐 (Coder, Reviewer, Researcher)
- 각 에이전트가 **자기 시스템 프롬프트와 자기 도구**를 가짐

**노트 — 분할의 진짜 목적은 컨텍스트다**

MAS를 "여러 전문가를 모은다"로만 이해하면 오해입니다. 실질적 이득은 **컨텍스트 격리**입니다. Reviewer가 Coder의 시행착오 전부를 볼 필요가 없고, 오히려 보면 판단이 오염됩니다. 역할 분할은 프롬프트 분할이자 컨텍스트 분할이며, 후자가 성능에 더 직접적으로 기여합니다.

---

### 8. Single vs Multi-Agent 비교

| 항목 | Single Agent | Multi-Agent System |
| --- | --- | --- |
| 컨텍스트 | 단일 공유 윈도우 | 역할별 분할 |
| 프롬프팅 | 하나의 거대한 프롬프트 | 모듈화된 전문 지시 |
| 제어 | 휴리스틱하고 유동적 | 엄격한 그래프 기반 워크플로우 |
| 성능 | 낮은 지연시간 | 높은 지연시간 (에이전트 간 호출) |
| 유지보수 | 디버깅 쉬움 | 복잡한 오케스트레이션 |

> **지침: Single로 시작하라. 뚜렷이 구별되는 "페르소나"가 필요해질 때 MAS로 올려라.**

---

## Part 3: Orchestration Frameworks

### 9. LangGraph — 저수준 선택지

**모델:** 상태 기계 (Nodes & Edges)

#### 9.1 핵심 개념

| 개념 | 설명 |
| --- | --- |
| **State** | 단계 간에 지속되는 공유 `TypedDict` |
| **Node** | 상태를 수정하는 Python 함수 |
| **Edge** | 전이를 정의. **조건부 가능** |

#### 9.2 주요 기능

| 기능 | 설명 |
| --- | --- |
| **Persistence** | 체크포인트를 저장해 나중에 재개 |
| **Human-in-the-Loop** | 실행을 멈추고, 사람이 편집하게 하고, 재개 |
| **Subgraphs** | 그래프를 중첩해 모듈화 |
| **Streaming** | 생성되는 토큰을 스트리밍 |

> **적합:** 세밀한 제어가 필요한 복잡하고 커스텀한 워크플로우

**노트 — 슬라이드에 없지만 반드시 알아야 할 것**

**(1) State는 그냥 dict가 아니라 reducer를 가집니다.**

병렬 분기가 각각 상태를 반환할 때 어떻게 합칠지를 지정해야 합니다.

```python
class State(TypedDict):
    patches: Annotated[list[Patch], operator.add]  # 병렬 결과를 이어붙임
    attempt: int                                   # 지정 안 하면 덮어쓰기
```

reducer를 빼먹으면 병렬 분기 결과가 서로를 덮어씁니다. 처음 만들 때 가장 흔한 버그입니다.

**(2) 체크포인트는 노드 *경계*에만 찍힙니다.**

노드 내부에서 죽으면 그 노드는 처음부터 다시 실행됩니다. 그래서 **노드는 작게, 그리고 멱등하게** 만들어야 합니다. 10분짜리 빌드를 다른 작업과 한 노드에 묶으면 안 되는 이유입니다.

**(3) 학습 곡선의 실체**

"Steeper learning curve"의 실제 내용은 문법이 아니라 **상태 설계**입니다. 노드 간에 무엇이 흐르는지를 먼저 확정하지 않고 코드를 쓰기 시작하면 반드시 갈아엎게 됩니다.

---

### 10. CrewAI — 고수준 선택지

**모델:** 역할 기반 조직 (회사처럼)

| 개념 | 설명 |
| --- | --- |
| **Agent** | Role, Goal, Backstory로 정의 |
| **Task** | 에이전트에게 할당된 작업 단위 |
| **Crew** | Task를 수행하는 에이전트들의 집합 |
| **Process** | `sequential` 또는 `hierarchical` |

> **적합:** 빠른 프로토타이핑, 표준적인 "팀" 형태의 구성

---

### 11. 프레임워크 비교

| 항목 | LangGraph | CrewAI |
| --- | --- | --- |
| 로직 계층 | 저수준: 명시적 상태 기계 | 고수준: 역할과 과제 기반 |
| 유연성 | 무한: 세밀한 제어 | 정해진 틀: 사전 정의된 패턴 |
| 도입 노력 | 높음 (가파른 학습 곡선) | 낮음 (완만한 학습 곡선) |
| 사람 개입 | **네이티브: 내장 interrupt** | 제한적/기초적 |
| 이상적 용도 | 커스텀 고정밀 파이프라인 | 빠르게 확장하는 "에이전트 팀" |

**노트 — 선택 기준을 한 줄로**

**사람 승인 게이트가 요구사항에 있으면 LangGraph입니다.** CrewAI의 HITL은 "제한적/기초적"이라고 명시되어 있고, 승인 게이트는 나중에 얹기 가장 어려운 요구사항입니다. 반대로 승인이 필요 없고 팀 형태 협업이면 CrewAI가 훨씬 빠릅니다.

---

## Part 4: Agentic Patterns (워크플로우 패턴)

### 12. 패턴 전체 조망

에이전트 시스템의 구성 블록 7가지입니다.

| # | 패턴 | 한 줄 요약 |
| --- | --- | --- |
| 1 | Prompt Chaining | 고정된 단계 순서 |
| 2 | Routing | 분류해서 전문 처리기로 분배 |
| 3 | Parallelization | 독립 작업 동시 수행 |
| 4 | Orchestrator-Workers | 런타임에 동적으로 위임 |
| 5 | Evaluator-Optimizer | 생성 → 테스트 → 반복 |
| 6 | ReAct | 에이전트의 기본 루프 |
| 7 | ReWOO | 미리 계획하고 병렬 실행 |

---

### 13. Prompt Chaining

**개념:** 과제를 고정된 단계 순서로 분해

**흐름:** `Step A → Step B → Step C`

**예시:**

1. 마케팅 문구 생성
2. 스페인어로 번역
3. 규정 준수 검사

**이점:** 각 단계를 단순화해 정확도를 높임

**노트 — 왜 쪼개면 정확도가 오르는가**

한 번의 호출에 세 가지를 요구하면 모델은 셋 다 어중간하게 합니다. 쪼개면 각 호출의 지시가 명확해지고, **중간 산출물을 검사할 수 있게** 됩니다. 후자가 더 중요합니다 — 어느 단계에서 틀렸는지 알 수 있으니까요.

---

### 14. Routing

**개념:** 입력을 분류해 전문 처리기로 보냄

#### 14.1 라우터의 세 종류

| 종류 | 방식 | 속도 | 유연성 |
| --- | --- | --- | --- |
| **LLM Router** | LLM에게 분류를 시킴 | 느림 | 유연함 |
| **Semantic Router** | 임베딩 유사도 | 빠름 | 결정론적 |
| **Keyword Router** | 정규식/키워드 매칭 | 가장 빠름 | 낮음 |

**사용 사례:** 고객 서비스 분류 (환불 / 기술지원 / 일반 문의)

**이점:** 관심사 분리 — 각 범주에 특화된 프롬프트를 쓸 수 있음

**노트 — 실무에서는 섞어 씁니다**

키워드로 확실한 것부터 걸러내고(비용 0), 남은 것만 LLM에 보내는 계층적 라우팅이 일반적입니다. 그리고 **LLM Router는 반드시 신뢰도 임계값과 함께** 써야 합니다. 확신 없이 잘못된 전문가로 보내는 것이 아무것도 안 하는 것보다 나쁩니다.

---

### 15. Parallelization

**개념:** 여러 LLM이 독립적인 하위 과제를 동시에 수행

#### 15.1 두 가지 변형

| 변형 | 개념 | 예시 |
| --- | --- | --- |
| **Sectioning** | 큰 과제를 독립 청크로 분할 | 5개 보안 가드레일을 동시에 처리 |
| **Voting** | 같은 과제를 여러 번 수행해 다양성 확보 | 코드 수정안 3개 생성 → "심판"이 최선을 선택 |

**이점:** 지연시간 = **가장 느린 단계** (모든 단계의 합이 아님)

**노트 — Voting은 정확도를 사는 방법입니다**

Voting은 계산으로 정확도를 사는 거래입니다. 같은 문제를 N번 풀고 고르면 비용은 N배지만 정답률이 오릅니다. 여기서 핵심은 **심판의 품질**입니다. 심판이 LLM이면 불확실하지만, 심판이 **컴파일러나 테스트**라면 거의 완벽합니다. 결정론적 검증기를 가진 도메인에서 Voting이 특히 강력한 이유입니다.

---

### 16. Orchestrator-Workers

**개념:** 중앙 Orchestrator LLM이 계획을 동적으로 분해하고 Worker에게 위임

**Parallelization과의 차이:** 하위 과제가 **사전 정의되어 있지 않음**. Orchestrator가 런타임에 결정합니다.

**예시 (코딩 에이전트):**

1. 사용자: "로그인 페이지를 추가해줘."
2. Orchestrator: "`auth.py`를 수정하고, `login.html`을 만들고, `routes.py`를 갱신해야 한다."
3. Worker들이 각 파일 변경을 실행

**노트 — 이 패턴이 위험해지는 지점**

Orchestrator가 계획을 틀리면 Worker들이 **일사불란하게 틀린 일을 합니다.** 그래서 이 패턴은 거의 항상 Evaluator-Optimizer와 함께 써야 합니다. 계획 자체를 검증하는 단계가 없으면 실패가 증폭됩니다.

---

### 17. Evaluator-Optimizer ⭐

**개념:** 생성 → 평가 → 피드백 루프

**흐름:**

1. **Generator**: 후보 해답을 만듦
2. **Evaluator**: 점수를 매기거나 테스트함
3. **Feedback**: 실패하면 오류를 Generator에게 돌려 재시도

**사용 사례: 코딩 에이전트**

| 역할 | 하는 일 |
| --- | --- |
| Generator | 코드를 작성 |
| Evaluator | 유닛 테스트를 실행 |
| Feedback | "10번 줄에서 테스트 실패" → Generator가 수정 |

**노트 — 이 강의에서 가장 중요한 패턴**

이 패턴이 없으면 시스템은 "쓸 줄만 아는" 상태에 머뭅니다. 그리고 **Evaluator의 종류가 시스템의 상한을 결정합니다.**

| Evaluator | 신뢰도 | 비고 |
| --- | --- | --- |
| 컴파일러 / 테스트 | 매우 높음 | 결정론적. 실패 이유가 기계 판독 가능 |
| 정적 분석 / 린터 | 높음 | 규칙 기반 |
| LLM-as-a-Judge | 중간 | 주관적 기준에만 사용 |
| 사람 | 높지만 느림 | 처리량이 병목 |

**결정론적 Evaluator를 가진 문제부터 자동화하라**는 것이 실무 원칙입니다.

---

### 18. ReWOO (Reasoning Without Observation)

**ReAct의 문제:** 도구 호출마다 "멈추고 기다리는" 순차 실행

**ReWOO의 해법:** **계획과 실행을 분리**

**흐름:**

1. **Planner**: 전체 계획을 미리 생성 (X를 검색, Y를 검색, Z를 계산)
2. **Workers**: 모든 검색 단계를 **병렬로** 실행
3. **Solver**: 최종 답을 종합

**이점:** 다중 도구 과제에서 **2~5배 지연시간 감소**

**노트 — 어떻게 병렬이 가능한가**

핵심 트릭은 **변수 치환**입니다. Planner는 실제 결과를 모르는 상태에서 계획을 세우되, 결과가 들어갈 자리를 변수로 표시합니다.

```
Plan: 파리 인구를 검색       #E1 = search("population of Paris")
Plan: 런던 인구를 검색       #E2 = search("population of London")
Plan: 두 값을 비교           #E3 = compare(#E1, #E2)
```

`#E1`과 `#E2`는 서로를 참조하지 않으므로 동시에 실행할 수 있습니다. `#E3`만 둘을 기다립니다.

**한계:** Planner가 관측 없이 계획하므로, **중간 결과에 따라 경로가 바뀌어야 하는 과제에는 부적합**합니다. 탐색적 작업에는 ReAct, 구조가 예측 가능한 작업에는 ReWOO입니다.

원 논문: Xu et al., *"ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models"* (2023)

---

### 19. 패턴 선택 요약

| 상황 | 패턴 |
| --- | --- |
| 경로를 알고 반복 가능 | Prompt Chaining |
| 요청이 범주로 갈림 | Routing |
| 독립 검사를 동시에 | Parallelization |
| 여러 도메인에 걸친 과제 | Orchestrator-Workers |
| 출력 품질에 한 번 더 점검 필요 | **Evaluator-Optimizer** |
| 탐색적 · 경로 미지 | ReAct |
| 도구 여러 개 · 구조 예측 가능 | ReWOO |

> 실제 프로덕션 시스템은 **거의 항상 여러 패턴을 조합**합니다. 앞단에 Routing, 각 경로에 Orchestrator-Workers, 마지막에 Evaluator-Optimizer 같은 식입니다.

---

## Part 5: Tool Use & Function Calling

### 20. Function Calling이란

LLM이 외부 도구를 써야 한다고 **신호를 보내는** 메커니즘입니다.

**루프:**

1. 사용자가 메시지를 보냄
2. LLM이 도구 호출(JSON)을 생성
3. **런타임이 도구를 실행**
4. 런타임이 관측 결과를 LLM에게 돌려줌
5. LLM이 최종 응답을 생성

### 21. "Stop Reason" — 오해하기 쉬운 지점

> LLM은 마법처럼 코드를 실행하지 않습니다. **구조화된 요청을 출력할 뿐입니다.**

```json
{
  "stop_reason": "tool_use",
  "tool_calls": [
    {
      "name": "search",
      "arguments": { "query": "population of Paris" }
    }
  ]
}
```

런타임(우리 코드)이 이것을 가로채서 실제 함수를 호출합니다.

**노트 — 왜 이걸 강조하는가**

"에이전트가 파일을 수정했다"는 표현은 부정확합니다. **에이전트는 파일을 수정해 달라고 요청했고, 우리 코드가 수정했습니다.** 이 구분이 중요한 이유는 권한·검증·롤백이 전부 우리 쪽 책임이라는 뜻이기 때문입니다. 안전 설계(Part 9)가 전부 런타임 층에서 이루어지는 근거입니다.

### 22. 도구 정의: JSON Schema

```json
{
    "name": "search",
    "description": "Search the web for information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            }
        },
        "required": ["query"]
    }
}
```

### 23. 도구 정의 모범 사례 (Anthropic)

| # | 원칙 | 설명 |
| --- | --- | --- |
| 1 | **명확한 이름** | `get_stock_price` > `gsp` |
| 2 | **상세한 설명** | **docstring이 곧 프롬프트다.** 서술적으로 쓸 것 |
| 3 | **단순한 인터페이스** | 원시 타입(문자열, 정수) 사용. 복잡한 중첩 객체 회피 |
| 4 | **결함 허용** | 오류를 우아하게 처리하고, **정보가 담긴 오류 메시지**를 반환 |
| 5 | **멱등성** | 가능한 한 두 번 호출해도 한 번과 같은 효과가 되도록 |

**노트 — 4번과 5번이 실무에서 가장 중요합니다**

**4번 (결함 허용):** `Error: 500`을 반환하면 에이전트는 무엇을 고쳐야 할지 모릅니다. `Error: file not found at path 'src/mian.cpp' — did you mean 'src/main.cpp'?`를 반환하면 에이전트가 스스로 고칩니다. **오류 메시지는 에이전트를 향한 프롬프트입니다.**

**5번 (멱등성):** 에이전트는 재시도합니다. 타임아웃이 나면 같은 도구를 또 부릅니다. 멱등하지 않으면 중복 커밋, 중복 PR, 중복 결제가 발생합니다. 이는 LangGraph 노드 설계 원칙과 정확히 같은 이유입니다.

---

## Part 6: 에이전트 통신 프로토콜

### 24. 상호운용성 문제

에이전트가 늘어나면서 생기는 문제들:

- **도구 파편화**: 프레임워크마다 도구 정의 방식이 다름
- **에이전트 고립**: 다른 벤더의 에이전트끼리 통신 불가
- **통합 부담**: 개발자가 같은 통합을 반복 구현

### 25. 두 프로토콜

| | MCP (Anthropic) | A2A (Google) |
| --- | --- | --- |
| 목적 | 에이전트를 **도구·데이터**에 연결 | 에이전트끼리 **서로** 통신 |
| 방향 | Human→Agent, Agent→Tool | Agent→Agent |

> **관계: MCP와 A2A는 경쟁이 아니라 상호보완적입니다.**

### 26. MCP (Model Context Protocol)

- **목표:** AI 에이전트를 외부 시스템에 연결하는 보편 표준
- **비유:** **MCP는 에이전트에게 USB와 같다**
- **핵심 아이디어:** 도구를 한 번 정의하면 어디서나 쓴다

#### 26.1 핵심 개념

| 개념 | 설명 |
| --- | --- |
| **Tools** | 에이전트가 호출할 수 있는 함수 (`search`, `send_email`) |
| **Resources** | 읽기 전용 데이터 소스 (파일, DB 레코드) |
| **Prompts** | 흔한 작업을 위한 사전 제작 프롬프트 템플릿 |
| **Sampling** | 서버가 호스트에게 LLM 완성을 요청할 수 있게 함 |

#### 26.2 이점

- **Write Once, Use Everywhere**: 내 MCP 서버가 모든 MCP 호환 에이전트와 동작
- **보안**: 프로토콜이 권한 범위를 정의
- **탐색**: 에이전트가 런타임에 가용 도구를 발견 가능
- **생태계**: Slack, GitHub, DB 등 기성 MCP 서버가 계속 늘어남

### 27. A2A (Agent-to-Agent Protocol)

- **목표:** 서로 다른 벤더의 에이전트가 협업하기 위한 표준
- **문제:** 벤더 A의 "여행 에이전트"가 벤더 B의 "예약 에이전트"에게 호텔 예약을 어떻게 요청하나?

| 개념 | 설명 |
| --- | --- |
| **Agent Card** | 에이전트의 능력을 기술한 JSON 문서<br>`{ "name": "Booking Agent", "skills": ["hotel","flight"], "endpoint": "..." }` |
| **Tasks** | 에이전트 간에 위임되는 작업 단위 |
| **Messages** | 구조화된 통신 (요청, 응답, 상태) |
| **Artifacts** | 에이전트 간에 전달되는 파일이나 데이터 |

---

## Part 7: 상태 관리

### 28. "기억" 문제

- LLM은 **stateless**입니다
- 과거 대화를 기억하지 못합니다
- **해법: 상태를 외부에서 관리하고 프롬프트에 주입한다**

### 29. 단기 기억 (Thread Context)

| 항목 | 내용 |
| --- | --- |
| 무엇 | 현재 대화의 메시지 목록 |
| 어디에 | 메모리에. 매 LLM 호출마다 함께 전달 |
| 난점 | 컨텍스트 윈도우 한계 (예: 128k 토큰) |
| 전략 | 요약, 슬라이딩 윈도우, 잘라내기 |

### 30. 장기 기억 (Persistence)

| 방식 | 용도 | 예시 |
| --- | --- | --- |
| **Key-Value Store** | 사용자 선호, 세션 메타데이터 | `{ "user_id": "123", "preferred_language": "Spanish" }` |
| **Vector DB (RAG)** | 과거 지식의 의미 검색 | 문서/대화 임베딩 저장 후 유사도로 검색 |

### 31. Checkpointing (LangGraph)

**개념:** 매 노드 실행 후 그래프 상태의 스냅샷을 저장

| 능력 | 설명 |
| --- | --- |
| **Resumption** | 연결이 끊겨도 중단 지점부터 재개 |
| **Human-in-the-Loop** | 멈추고, 사람이 승인하고, 재개 |
| **Time Travel** | 이전 상태로 되감아 다른 분기로 실행 |

**노트 — Time Travel의 실무 가치**

세 번째 기능이 과소평가되어 있습니다. 에이전트 시스템의 디버깅은 "3번째 노드부터 프롬프트를 바꿔서 다시 돌려보고 싶다"는 형태가 대부분인데, 체크포인트가 있으면 처음부터 다시 돌릴 필요가 없습니다. **평가 반복 속도가 몇 배 빨라집니다.**

---

## Part 8: 실제 적용 사례

### 32. 사례 A: 고객 지원 에이전트

**목표:** Tier-1 지원을 자동화하되 사람에게 에스컬레이션

**아키텍처:** `Router → 전문 서브에이전트 → 사람 인계`

**흐름:**

1. **Guardrails**: PII, 유해 표현 필터링
2. **Router**: 의도 분류 (청구 / 기술 / 일반)
3. **서브에이전트 (청구)**:
   a. 계정 정보 조회 (API 호출)
   b. 환불 자격 확인
   c. 자격이 되면 `issue_refund` 도구 호출
4. **사람 인계**: 신뢰도가 임계값 미만이거나 고액 작업인 경우

### 33. 사례 B: 코딩 에이전트 ⭐

**목표:** 코딩 과제(예: GitHub 이슈)를 자율적으로 해결

**아키텍처:** `Orchestrator-Workers + Evaluator-Optimizer`

**흐름:**

| 단계 | 역할 | 하는 일 |
| --- | --- | --- |
| 1 | **Planner** | 이슈 설명을 읽고 파일 변경 단위로 분해 |
| 2 | **Coder (Worker)** | 각 파일을 수정 |
| 3 | **Executor** | 파일을 디스크에 저장 |
| 4 | **Tester** | `pytest`나 린터를 실행 |
| 5 | **Evaluator** | 통과 → 커밋 / 실패 → stderr를 Coder에게 돌려보냄 |
| 6 | **Loop** | **최대 3회 재시도, 그 후 사람에게 에스컬레이션** |

**노트 — 이 흐름을 통째로 외워두면 좋습니다**

이것이 코딩 에이전트의 표준 레퍼런스 아키텍처입니다. 주목할 점 셋:

1. **Tester가 별도 단계로 분리**되어 있습니다. 코드 생성과 검증이 같은 단계가 아닙니다.
2. **실패 시 stderr를 그대로 돌려보냅니다.** 요약하거나 해석하지 않습니다 — 컴파일러 출력이 가장 정확한 피드백입니다.
3. **재시도 상한이 명시**되어 있습니다(3회). 무한 루프 방지이자 비용 상한이며, 넘으면 사람에게 넘깁니다. 상한 없는 자기수정 루프는 프로덕션에서 위험합니다.

---

## Part 9: 안전과 격리

### 34. 왜 안전이 중요한가

에이전트는 **실제 세계에 부수효과**를 냅니다.

- 이메일 발송
- 코드 실행
- API 호출 (구매, 삭제)

> 제약 없는 에이전트는 부채(liability)입니다.

### 35. 여섯 가지 원칙

#### 원칙 1: 단순하게 시작하라

> **"가장 정교한 시스템이 아니라, 올바른 시스템을 만들어라."**

- 먼저 단일 LLM 호출을 최적화하라
- 필요할 때만 에이전트 루프를 추가하라
- 프로덕션에서는 추상화 계층을 줄여라

#### 원칙 2: 투명성

에이전트가 무엇을 생각하는지 사용자에게 보여줄 것.

- **Streaming**: "Thought" 단계를 실시간 표시
- **Reasoning Trace**: 모든 도구 호출과 중간 출력을 로깅
- **Explainability**: "30일 이내라 정책상 환불이 가능해 환불을 처리했습니다."

#### 원칙 3: 가드레일

| 입력 가드레일 | 출력 가드레일 |
| --- | --- |
| PII 필터링 (카드번호, 주민번호) | 응답이 정책을 위반하지 않는지 확인 |
| 유해·적대적 프롬프트 차단 | 원본 문서 대비 환각 검사 |
| 탈옥 시도 탐지 | |

#### 원칙 4: 샌드박싱

> **코드 실행 도구는 위험합니다.**

- 격리된 컨테이너에서 실행 (Docker 등)
- 네트워크 접근 제한
- 자원 쿼터 설정 (CPU, 메모리, 시간)
- 민감한 시스템콜 비활성화

#### 원칙 5: Human-in-the-Loop

**중요한 작업은 승인을 요구합니다.**

- 이메일 발송
- **코드 커밋**
- 금융 거래

**구현:** 체크포인팅을 이용해 멈추고, 사람에게 작업을 제시하고, 승인 시 재개

#### 원칙 6: 평가와 모니터링

| 시점 | 할 일 |
| --- | --- |
| **배포 전** | 골든 데이터셋 구축 (질의 + 기대 출력)<br>"LLM-as-a-Judge"로 정확도·도구 사용·톤 채점 |
| **배포 후** | 지연시간, 오류율, 도구 호출 빈도 모니터링<br>감사와 디버깅을 위해 모든 상호작용 로깅 |

---

## 핵심 요약

### 강의 전체 흐름

```
언제 에이전트인가 → 어떤 구조인가 → 어떤 프레임워크인가
        → 어떤 패턴인가 → 도구·프로토콜 → 상태 → 안전
```

### 한 문장으로 압축한 결론

| 주제 | 결론 |
| --- | --- |
| Agents vs Workflows | 경로를 알면 워크플로우, 모르면 에이전트 |
| 아키텍처 | **Single로 시작**, 페르소나가 필요할 때 Multi로 |
| 프레임워크 | LangGraph=제어, CrewAI=속도. **승인 게이트 필요하면 LangGraph** |
| 패턴 | 7개를 어휘로 갖추고 **조합**해서 쓴다 |
| 도구 | docstring이 프롬프트. 오류 메시지도 프롬프트. 멱등하게 |
| 프로토콜 | MCP=도구 연결, A2A=에이전트 연결. 상호보완 |
| 상태 | 단기=컨텍스트, 장기=RAG/KV, **체크포인트=재개·승인·되감기** |
| 안전 | 가드레일 · 샌드박싱 · HITL · 평가 |

### 가장 중요한 세 문장

1. **"Build the RIGHT system, not the MOST SOPHISTICATED system."**
2. **Evaluator의 품질이 시스템의 상한을 결정한다.** (결정론적 검증기를 가진 문제부터 자동화하라)
3. **재시도 상한을 반드시 두고, 넘으면 사람에게 넘겨라.**

---

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| ReAct | Reasoning + Acting. Thought→Action→Observation 루프 |
| ReWOO | Reasoning Without Observation. 계획·실행 분리로 병렬화 |
| MAS | Multi-Agent System |
| MCP | Model Context Protocol (Anthropic). 에이전트↔도구 |
| A2A | Agent-to-Agent Protocol (Google). 에이전트↔에이전트 |
| HITL | Human-in-the-Loop |
| Checkpointing | 노드 실행 후 상태 스냅샷 저장 |
| Reducer | 병렬 분기의 상태를 합치는 함수 |
| Guardrail | 입출력에 대한 안전 필터 |
| Sectioning | 큰 과제를 독립 청크로 나눠 병렬 처리 |
| Voting | 같은 과제를 여러 번 수행 후 최선 선택 |
| Idempotency | 두 번 실행해도 한 번과 결과가 같은 성질 |
| Agent Card | A2A에서 에이전트 능력을 기술하는 JSON |
| Time Travel | 이전 체크포인트로 되감아 다른 분기 실행 |

---

## 읽어볼 자료

1. **[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)** (Anthropic) — 이 강의 Part 4의 원전
2. **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)** — State, Node, Edge, Checkpointer
3. **[MCP](https://modelcontextprotocol.io/)** — Model Context Protocol
4. **[A2A Protocol](https://a2a-protocol.org/latest/)**
5. **[CrewAI Documentation](https://docs.crewai.com/)**
6. **[What is Agentic Architecture?](https://www.ibm.com/think/topics/agentic-architecture)** (IBM)
7. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)
8. Xu et al., *ReWOO: Decoupling Reasoning from Observations* (2023)

---

*Stanford CS 224G: Building & Scaling LLM Applications | Winter 2026 | Lecture 7 정리*
