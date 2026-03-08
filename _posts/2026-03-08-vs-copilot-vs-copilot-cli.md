---
layout: post
title: Visual Studio 2022 Copilot vs Copilot CLI 아키텍처 비교
date: 2026-03-08 11:00:00 +0900
categories: [AI, Tool]
tags: [copilot, architecture, semantic-index, mcp]
---

## 공통점

- 동일한 GitHub Copilot 백엔드 인프라 사용
- GitHub 인증 및 라이선스 공유

## 주요 차이점

### 1. 컨텍스트 수집 (전처리)

| 구분 | Visual Studio 2022 | Copilot CLI |
| --- | --- | --- |
| 컨텍스트 소스 | Semantic Index (의미론적 인덱스) | MCP (Model Context Protocol) 서버 |
| 코드 이해 | 정의/참조 위치, 심볼 정보, 타입 정보 | 파일 시스템, Git 상태, GitHub API |
| 범위 | 열려있는 솔루션/프로젝트 전체 | 현재 디렉토리 + MCP 서버가 제공하는 데이터 |

**Visual Studio:**

```
[소스코드] → [Semantic Indexing Service] → [정의/참조/심볼 추출] → [LLM 컨텍스트 구성]
```

- LSP(Language Server Protocol) 기반으로 코드 구조 파악
- 클래스 정의, 메서드 시그니처, 참조 관계 등을 인덱싱
- BlingFire 등의 토크나이저로 코드 분석

**Copilot CLI:**

```
[프롬프트] → [MCP 서버 조회] → [파일/GitHub 데이터 수집] → [LLM 컨텍스트 구성]
```

- MCP 서버를 통해 외부 도구/데이터 소스 연결
- 파일 시스템 직접 접근
- GitHub API를 통한 PR, 이슈 등 조회

---

### 2. 아키텍처 구조

![아키텍처 비교](/assets/img/copilot-vs-cli/architecture.png)

---

### 3. 후처리

| 구분 | Visual Studio 2022 | Copilot CLI |
| --- | --- | --- |
| 응답 처리 | 코드 삽입 위치 계산, 들여쓰기 조정, 문법 검증 | 명령어 실행, 파일 수정, Git 작업 수행 |
| 실행 방식 | 사용자가 Accept/Reject 결정 | Agentic 루프 (자동 계획-실행-평가) |
| 안전장치 | IDE 내 Undo/Redo | 도구 실행 전 사용자 승인 요청 |

---

### 4. 기본 LLM 모델

| 구분 | Visual Studio 2022 | Copilot CLI |
| --- | --- | --- |
| 기본 모델 | 시간에 따라 변경됨 (선택 가능) | Claude Sonnet 4.5 |
| 모델 선택 | 설정에서 변경 가능 | 기본 고정 |

---

## 요약

| 측면 | Visual Studio 2022 | Copilot CLI |
| --- | --- | --- |
| 전처리 | Semantic Index 기반 정적 분석 | MCP 기반 동적 데이터 수집 |
| 통신 | LSP + vs-streamjsonrpc | MCP (Model Context Protocol) |
| 철학 | IDE 통합, 코드 완성 중심 | Agentic, 작업 자동화 중심 |
| 컨텍스트 | 프로젝트/솔루션 범위 | 파일시스템 + 외부 서비스 |
| 확장성 | VS 확장 모델 | 커스텀 MCP 서버 |

핵심 차이는 **컨텍스트 수집 방식**이다:

- **VS 2022**는 정적 코드 분석 인덱스로 풍부한 코드 구조 정보 제공
- **CLI**는 MCP 프로토콜로 다양한 외부 도구/데이터 소스에 동적 접근

---

## Semantic Index vs MCP: 핵심 차이

### 1. 근본적인 철학

| 구분 | Semantic Index | MCP |
| --- | --- | --- |
| 접근 방식 | 컴파일러 기반 정적 분석 | 런타임 동적 조회 |
| 데이터 준비 | 사전 인덱싱 (미리 분석) | 요청 시 조회 (Just-in-time) |
| 데이터 범위 | 코드 구조 + 의미 | 모든 외부 데이터 소스 |

---

### 2. Semantic Index (Visual Studio)

**작동 원리:**

![Semantic Index 작동 원리](/assets/img/copilot-vs-cli/semantic-index.png)

**제공하는 정보:**

```cpp
// 예: 사용자가 "Person" 클래스 관련 코드 완성 요청 시

class Person {
    string Name;  // ← 인덱스가 이 정의를 알고 있음
    int Age;      // ← 타입 정보도 알고 있음
};

void foo() {
    Person p;
    p.  // ← Copilot이 "Name", "Age"를 정확히 제안 가능
}
```

**Semantic Model이 답할 수 있는 질문들:**

- "이 위치에서 사용 가능한 이름(변수/함수)은?"
- "이 메서드에서 접근 가능한 멤버는?"
- "이 이름/표현식이 참조하는 것은?"
- "이 블록에서 사용된 변수는?"

**최신 기능 (Remote Semantic Search - VS 17.14+):**

- 키워드 매칭(BM25) + AI 벡터 임베딩 결합
- "fetch user credentials" ↔ "get authentication token" 의미적 연결
- 함수의 목적, 변수의 의도, 주석의 맥락까지 이해

---

### 3. MCP (Model Context Protocol)

**작동 원리:**

![MCP 작동 원리](/assets/img/copilot-vs-cli/mcp-architecture.png)

**MCP 서버가 노출하는 3가지:**

![MCP 서버 노출 요소](/assets/img/copilot-vs-cli/mcp-server-expose.png)

**실제 동작 흐름:**

1. 사용자: "이 PR의 리뷰 코멘트 정리해줘"
2. LLM 판단: GitHub MCP 서버의 `get_pr_comments` 도구 필요
3. MCP Client → GitHub Server:
   ```json
   {
     "method": "tools/call",
     "params": { "name": "get_pr_comments", "arguments": { "pr": 123 } }
   }
   ```
4. GitHub Server: GitHub API 호출 → 결과 반환
5. Client → Host: 결과를 LLM 컨텍스트에 추가
6. LLM: 결과 기반으로 최종 응답 생성

---

### 4. 핵심 차이 비교

| 측면 | Semantic Index | MCP |
| --- | --- | --- |
| 데이터 소스 | 소스코드만 | 무엇이든 (파일, API, DB, 웹) |
| 분석 깊이 | 컴파일러 수준 (타입, 심볼, 참조) | 도구가 반환하는 것만큼 |
| 시점 | 빌드/편집 시 사전 인덱싱 | 쿼리 시 실시간 조회 |
| 코드 이해 | 깊이 이해 (타입 추론, 상속) | 텍스트 수준 |
| 외부 연동 | 불가능 | 무제한 확장 |
| 응답 속도 | 빠름 (미리 인덱싱) | 도구 실행 시간에 의존 |

---

### 5. 실제 예시로 보는 차이

> 질문: "이 함수에서 사용하는 Person 클래스의 모든 속성을 알려줘"

**Semantic Index (VS):**

1. 인덱스 조회 → Person 클래스 정의 위치 즉시 파악
2. SemanticModel에서 멤버 목록 추출
3. 타입 정보까지 포함하여 LLM에 전달

결과: `{ Name: string, Age: int, Address: Address, ... }` (타입, 접근제한자, 상속 관계까지 정확)

**MCP (CLI):**

1. File Server의 `read_file` 도구로 소스 파일 읽기
2. grep/search 도구로 "class Person" 찾기
3. 파일 텍스트를 LLM에 전달
4. LLM이 텍스트 파싱해서 추론

결과: 텍스트 기반 추론 (컴파일러 수준 정확도 아님)

---

### 6. 왜 이런 차이가 있을까?

| VS Copilot | Copilot CLI |
| --- | --- |
| 목적: IDE 내 코드 작성 지원 | 목적: 범용 자동화 에이전트 |
| 코드 정확성이 최우선 | 다양한 작업 수행이 최우선 |
| 프로젝트 범위 내 깊은 이해 | 넓은 범위의 얕은 접근 |
| 컴파일러와 긴밀한 통합 | 어디서든 독립 실행 |

---

## 결론

> **Semantic Index** = 컴파일러가 이미 알고 있는 것을 LLM에게 전달 (정적 분석, 사전 인덱싱, 코드 특화)
>
> **MCP** = LLM이 필요할 때 외부에 물어보는 것 (동적 조회, 실시간, 범용 확장)

VS Copilot은 **"코드를 깊이 이해"**하는 데 최적화되어 있고,
Copilot CLI는 **"무엇이든 할 수 있는 확장성"**에 최적화되어 있다.
