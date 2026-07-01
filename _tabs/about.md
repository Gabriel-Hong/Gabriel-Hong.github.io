---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

## 안녕하세요, 홍정모입니다.

6년간 C++로 대규모 CAE 프로덕션 소프트웨어를 개발하며 수치 알고리즘과 성능 최적화, 도메인 지식을 쌓았고, 이 경험을 바탕으로 AI 엔지니어로 전환했습니다.

특정 도메인의 문제를 AI/LLM으로 푸는 일에 관심이 있습니다. 모델을 붙이는 데서 그치지 않고, 설계부터 Kubernetes 배포와 운영까지 직접 다뤄 **사내 프로덕션에서 실제로 동작하는 시스템**을 만들어 왔습니다.

최근에는 본사 개발팀의 **AI 분야 기술 리더**로서 전사와 해외 법인(중국·일본·이탈리아) 개발팀에 도메인 드리븐 AI 개발을 교육·전파하고 있습니다.

---

## 관심 분야

도메인 특화 AI 시스템을 만들기 위해 다음 영역을 공부하고 있습니다.

- **Domain-Driven RAG** — 전문 도메인의 용어와 맥락을 반영하는 하이브리드 검색 증강 생성
- **LLM Fine-tuning** — 도메인 특화 Tool Calling 등 특수 목적에 맞춘 경량 모델 최적화
- **Multi-Agent Systems** — 도메인 워크플로우를 자동화하는 에이전트 오케스트레이션
- **LLMOps & MCP** — Kubernetes 배포와 Observability를 갖춘 LLM 서비스 운영, 도구 연동
- **Knowledge Graph & Ontology** — 코드·설계기준을 그래프로 구조화해 영향도·의존성을 분석 (Node2Vec 임베딩, 신뢰도 기반 관계 추출)

---

## 주요 프로젝트

### [MultiAgentSystem4Material](https://github.com/Gabriel-Hong/MultiAgentSystem4Material)

Jira 이슈 분석부터 C++ 코드 수정, PR 생성까지의 개발 워크플로우를 자동화하는 멀티에이전트 시스템입니다. 반복 작업 하나를 4시간에서 97초로 줄였고, Kubernetes(Helm·HPA) 배포와 Prometheus/Grafana 모니터링, 신뢰도 기반 Human-in-the-loop를 더해 사내에서 운영하고 있습니다.

### [DesignCodeRAG](https://github.com/Gabriel-Hong/DesignCodeRAG)

구조설계 기준(내진성능평가) 등 기술 규격 문서에 특화된 RAG 시스템입니다. Elasticsearch 하이브리드 검색(BM25 + Dense Vector)과 표·공식을 보존하는 구조 인식 청킹으로, 근거를 함께 제시하는 답변을 생성합니다.

### [qlora-function-calling](https://github.com/Gabriel-Hong/qlora-function-calling)

구조공학 API의 Tool Calling을 수행하도록 Qwen2.5 모델을 QLoRA로 파인튜닝한 프로젝트입니다. 273개 API용 학습 데이터 설계부터 평가 파이프라인, vLLM 서빙까지 직접 구현했습니다.

### [mcp-gennx](https://github.com/Gabriel-Hong/mcp-gennx)

JSON 스키마로부터 런타임에 LLM 도구를 동적 생성하는 MCP 서버입니다. 66개 스키마를 병합해 47개 API 엔드포인트에서 최대 161개 도구를 자동 생성하고, Context Window 제약에 대응하는 Toolset 필터링을 지원합니다. GUI로만 다루던 CAE 제품을 자연어로 동작시킨 가능성을 인정받아 사내 3개 제품군으로 적용이 확장됐습니다.

### [OrderBook](https://github.com/Gabriel-Hong/OrderBook)

C++20으로 구현한 Limit Order Book 엔진입니다. 주문 처리 레이턴시 200ns, 처리량 4.87M/sec를 측정했습니다. 시스템 프로그래밍과 성능 최적화를 다뤄본 프로젝트입니다.

---

## 사내 프로덕션 적용

공개 저장소 외에도, 6년간 개발해 온 대규모 C++ 프로덕션에 AI를 직접 적용해 개발 워크플로우를 바꾸는 도구들을 만들어 운영하고 있습니다. (사내 도구라 링크는 생략합니다.)

- **도메인 AI 개발 환경** — 설계기준 문서 구조화 + 기존 구현 분석 + 개발 가이드 자동 생성을 하나의 워크플로우로 통합. 신규 설계기준(KISTEC 2024) 구현을 통상 1개월에서 **1일**(74개 파일·11개 모듈)로 단축했습니다.
- **코드 지식그래프 (Atlas) 2종** — 정적 분석으로 코드/설계기준을 그래프로 구조화. GENNX DB Atlas는 403개 DB 엔티티를 Node2Vec 임베딩으로 영향도 분석하고, Design Code Atlas는 설계기준 조항을 519개 노드·715개 엣지로 구조화(신뢰도·원문 인용 부착)했습니다.
- **LLM 코드리뷰 자동화** — PR 생성 시 LLM이 자동 리뷰·코멘트하는 파이프라인을 만들어 복수 팀에서 운영 중입니다.
- **Backport Audit** — 다수 릴리스 브랜치의 반영 누락을 매일 자동 판정해 담당자에게 알리는 품질 자동화 도구를 본사에서 운영하고 있습니다.

---

## 블로그

AI/LLM 관련 학습 기록을 정리하고 있습니다.

- Stanford CME295(Transformers & LLMs) 전 10강 정리
- 논문 리뷰 (Multi-Agent Debate, RAG, Reasoning 등)
- AI 도구 활용기와 개발 경험 공유

---

## 연락처

- **Email:** [gabrielhong91@gmail.com](mailto:gabrielhong91@gmail.com)
- **GitHub:** [github.com/Gabriel-Hong](https://github.com/Gabriel-Hong)
