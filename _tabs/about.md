---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

## 안녕하세요, 홍정모입니다.

6년간 C++로 대규모 CAE 프로덕션 소프트웨어를 개발하며 수치 알고리즘과 성능 최적화, 도메인 지식을 쌓았고, 이 경험을 바탕으로 AI 엔지니어로 전환했습니다.

특정 도메인의 문제를 AI/LLM으로 푸는 일에 관심이 있습니다. 모델을 붙이는 데서 그치지 않고, 설계부터 Kubernetes 배포와 운영까지 직접 다뤄보며 도메인에서 실제로 동작하는 시스템을 만들어 왔습니다.

---

## 관심 분야

도메인 특화 AI 시스템을 만들기 위해 다음 영역을 공부하고 있습니다.

- **Domain-Driven RAG** — 전문 도메인의 용어와 맥락을 반영하는 하이브리드 검색 증강 생성
- **LLM Fine-tuning** — 도메인 특화 Tool Calling 등 특수 목적에 맞춘 경량 모델 최적화
- **Multi-Agent Systems** — 도메인 워크플로우를 자동화하는 에이전트 오케스트레이션
- **LLMOps & MCP** — Kubernetes 배포와 Observability를 갖춘 LLM 서비스 운영, 도구 연동

---

## 주요 프로젝트

### [MultiAgentSystem4Material](https://github.com/Gabriel-Hong/MultiAgentSystem4Material)

Jira 이슈 분석부터 C++ 코드 수정, PR 생성까지의 개발 워크플로우를 자동화하는 멀티에이전트 시스템입니다. 반복 작업 하나를 4시간에서 97초로 줄였고, Kubernetes(Helm·HPA) 배포와 Prometheus/Grafana 모니터링, 신뢰도 기반 Human-in-the-loop를 더해 사내에서 운영하고 있습니다.

### [DesignCodeRAG](https://github.com/Gabriel-Hong/DesignCodeRAG)

한국 건축법규에 특화된 RAG 시스템입니다. Elasticsearch 하이브리드 검색(BM25 + Dense Vector)과 표·공식을 보존하는 구조 인식 청킹으로, 근거를 함께 제시하는 답변을 생성합니다.

### [qlora-function-calling](https://github.com/Gabriel-Hong/qlora-function-calling)

구조공학 API의 Tool Calling을 수행하도록 Qwen2.5 모델을 QLoRA로 파인튜닝한 프로젝트입니다. 273개 API용 학습 데이터 설계부터 평가 파이프라인, vLLM 서빙까지 직접 구현했습니다.

### [mcp-gennx](https://github.com/Gabriel-Hong/mcp-gennx)

JSON 스키마로부터 런타임에 LLM 도구를 동적 생성하는 MCP 서버입니다. 46개 API 엔드포인트에서 최대 159개 도구를 자동 생성하고, Context Window 제약에 대응하는 Toolset 필터링을 지원합니다.

### [OrderBook](https://github.com/Gabriel-Hong/OrderBook)

C++20으로 구현한 Limit Order Book 엔진입니다. 주문 처리 레이턴시 200ns, 처리량 4.87M/sec를 측정했습니다. 시스템 프로그래밍과 성능 최적화를 다뤄본 프로젝트입니다.

---

## 블로그

AI/LLM 관련 학습 기록을 정리하고 있습니다.

- Stanford CME295(Transformers & LLMs) 전 10강 정리
- 논문 리뷰 (Multi-Agent Debate, RAG, Reasoning 등)
- AI 도구 활용기와 개발 경험 공유

이 밖에 사내에서는 코드 지식그래프(Atlas)나 LLM 코드리뷰 같은 개발 도구도 만들어 동료들과 함께 쓰고 있습니다.

---

## 연락처

- **Email:** [gabrielhong91@gmail.com](mailto:gabrielhong91@gmail.com)
- **GitHub:** [github.com/Gabriel-Hong](https://github.com/Gabriel-Hong)
