---
layout: post
title: MCP(Model Context Protocol) 개념과 구조
date: 2026-05-19 09:00:00 +0900
categories: [AI, MCP]
tags: [mcp, llm, anthropic, fastmcp, function-calling, protocol]
---

## 들어가며

대상 : MCP를 처음 접하는 개발자

목표 : 다 읽고 나면 "MCP가 무엇이고, 왜 만들어졌으며, 직접 하나 만들려면 어디서 출발해야 하는지" 정도가 머릿속에 그려지는 것

마지막 한 챕터에서는 사내에서 만든 **mcp-gennx**(GEN NX 구조해석 SW를 MCP로 노출한 서버) 코드를 짧게 사례로 인용합니다. 다만 글의 중심은 **개념과 구조**이고, 코드는 "실제로는 이렇게 생겼다"는 감각을 주는 정도만 다룹니다.

---

## 1. MCP가 뭔가요?

![MCP 개요](/assets/img/mcp-concept-structure/mcp-overview.png)

Anthropic의 공식 정의는 한 줄입니다.

> **LLM 애플리케이션을 외부 데이터·도구와 연결하기 위한 개방형 프로토콜.**

Anthropic이 직접 쓰는 비유는 **"USB-C for AI applications"** 입니다. USB-C가 노트북·핸드폰·모니터를 가리지 않고 같은 단자로 연결되듯, MCP는 어떤 AI 앱이든 어떤 데이터 소스든 같은 약속(스펙)으로 연결되도록 합니다.

여기서 중요한 단어는 **"프로토콜"** 입니다. MCP는 특정 모델이나 특정 SDK가 아니라 **약속(스펙)** 입니다. 그래서 한 번 만들어 둔 MCP 서버는 Claude Desktop에서도, Cursor에서도, VS Code에서도, 사내 봇에서도 그대로 재사용할 수 있습니다.

---

## 2. 왜 필요한가요? — M×N 문제

![M×N 문제](/assets/img/mcp-concept-structure/mxn-problem.png)

MCP 이전, AI 앱과 데이터 소스를 연결하려면 **앱 × 데이터 소스** 만큼의 커넥터를 따로 만들어야 했습니다.

- AI 앱 3개 (Claude, ChatGPT, Cursor)
- 데이터 소스 5개 (Slack, Notion, Jira, GitHub, DB)
- 필요한 통합: **3 × 5 = 15개**

각 앱이 각 데이터 소스에 맞춰 따로 통합을 만들어야 했기 때문입니다. 데이터 소스가 늘면 곱셈으로 일이 늘어났습니다.

MCP는 이를 **덧셈**으로 바꿉니다.

- 데이터 소스마다 MCP 서버를 **하나씩** 만든다 (N개)
- AI 앱은 MCP 클라이언트를 **하나씩** 지원한다 (M개)
- 필요한 구현: **M + N개**

이 차이가 MCP가 빠르게 표준으로 자리잡는 이유입니다. 한 번 만들어 둔 서버를 모든 MCP 호환 앱에서 그대로 쓸 수 있게 되니까요.

---

## 3. MCP의 구조 — Host / Client / Server

![Host / Client / Server 구조](/assets/img/mcp-concept-structure/host-client-server.png)

MCP는 세 가지 컴포넌트로 구성됩니다.

| 컴포넌트 | 역할 | 예시 |
| --- | --- | --- |
| **Host** | 사용자가 직접 사용하는 AI 앱. 여러 MCP 서버를 동시에 다룸. | Claude Desktop, Cursor, VS Code |
| **Client** | Host 안에서 서버 1개와 1:1로 연결되는 어댑터. Host 내부 모듈이라 사용자에겐 보이지 않음. | (Host 내부) |
| **Server** | 도구·데이터를 외부에 노출하는 독립 프로세스. 보통 1개의 도메인을 담당. | mcp-gennx, GitHub MCP, Slack MCP |

한 Host는 보통 여러 Client를 띄워서 여러 Server와 동시에 통신합니다. 예를 들어 Claude Desktop을 켜면 그 안에서 GitHub MCP, Slack MCP, mcp-gennx가 각각의 클라이언트를 통해 동시에 연결될 수 있습니다.

---

## 4. 서버가 제공하는 3가지 — Tools / Resources / Prompts

![Tools / Resources / Prompts](/assets/img/mcp-concept-structure/primitives.png)

MCP 서버는 세 종류의 "기본 단위(primitive)"를 외부에 노출할 수 있습니다. 셋의 차이는 **"누가 호출을 결정하는가"** 로 구분하면 가장 명료합니다.

| Primitive | 누가 호출을 결정 | 용도 | 예시 |
| --- | --- | --- | --- |
| **Tools** | LLM (모델이 스스로 판단) | 실행/부수효과가 있는 동작 | 노드 생성, API 호출, DB 쿼리 실행 |
| **Resources** | Host 애플리케이션 | 읽기 전용 컨텍스트 데이터 | 문서, 로그, 스키마 정의 |
| **Prompts** | 사용자 (명시적 트리거) | 재사용 가능한 템플릿 | "코드 리뷰" 같은 슬래시 명령 |

처음 MCP 서버를 만들 때는 보통 **Tools부터 시작**하게 됩니다. 실제로 공개된 대부분의 MCP 서버도 Tools 중심으로 구성되어 있습니다. Resources와 Prompts는 필요해질 때 추가해도 늦지 않습니다.

---

## 5. 어떻게 통신하나요? — Transport

![Transport 방식](/assets/img/mcp-concept-structure/transport.png)

MCP 클라이언트와 서버가 실제로 데이터를 주고받는 통신 방식은 두 가지로 정리됩니다.

**stdio (Standard I/O)**

- 로컬에서 Host가 서버를 자식 프로세스로 띄우고, 표준 입출력으로 메시지를 주고받는 방식
- 가장 단순하고, 네트워크 오버헤드가 없음
- 로컬 도구·파일시스템·로컬 데몬에 접근하는 서버에 적합 (mcp-gennx도 이 방식)

**Streamable HTTP**

- 원격 서버를 HTTP POST + SSE로 호출하는 방식
- OAuth 2.0, Bearer Token 같은 표준 HTTP 인증을 그대로 활용 가능
- 사내·외부 원격 서버를 노출하고 싶을 때 사용

과거에는 별도의 SSE 방식이 있었으나, 새로 만든다면 Streamable HTTP가 권장됩니다.

---

## 6. MCP는 함수 호출(Function Calling)을 대체하나요? — 자주 헷갈리는 부분

신규 개발자에게 가장 많이 받는 질문 : **"이거 그냥 LLM 함수 호출 아닌가요?"**

결론부터 말하면 **아니요, 둘은 대체 관계가 아니라 함께 씁니다.**

- **Function Calling** : LLM이 "이 도구 호출해줘"라고 표현하는 방법 (LLM의 기능)
- **MCP** : 그 도구를 LLM 바깥에서 표준 형식으로 노출하는 약속 (프로토콜)

MCP는 Function Calling 위에 얹힌 인프라라고 생각하시면 됩니다.

![Function Calling과 MCP](/assets/img/mcp-concept-structure/function-calling-mcp.gif)

| 관점 | Function Calling만 쓸 때 | MCP를 쓸 때 |
| --- | --- | --- |
| 도구는 어디에 있나? | APP 코드 안 | 외부 서버(별도 프로세스) |
| 새 도구 추가 방법 | 코드 수정 → 재배포 | 설정 한 줄 추가 |
| 같은 도구를 다른 앱에서 | 코드 복사 필요 | 그대로 연결 |

차이가 가장 와닿는 두 상황입니다.

**"내 챗봇에 Slack 검색을 붙이고 싶다"**

- Function Calling만 쓰는 경우 : Slack API 코드를 직접 짜야 합니다.
- MCP를 쓰는 경우 : 누군가 만든 Slack MCP 서버를 받아 설정만 추가하면 됩니다. GitHub·Jira도 같은 방식.

**"Claude Desktop에서 우리 회사 내부 시스템을 쓰고 싶다"**

- Function Calling만 쓰는 경우 : **불가능**. Claude Desktop은 우리 회사 코드를 모릅니다.
- MCP를 쓰는 경우 : 서버를 만들어 설정에 추가하면 끝입니다. **사내 mcp-gennx가 바로 이 케이스입니다.**

---

## 7. 실제로는 이렇게 생겼습니다 — mcp-gennx 사례

여기서부터는 사내에서 만든 **mcp-gennx**의 repo로 "실제 MCP 서버는 이렇게 생겼다"를 짧게 보겠습니다.

### 7.1 mcp-gennx는 무엇인가

- MIDAS의 구조해석 소프트웨어 **GEN NX**의 REST API를 MCP tool로 노출한 서버입니다.
- 65개의 JSON 스키마 파일을 읽어 46개 엔드포인트로 정리하고, 최대 159개의 tool을 자동 등록합니다.
- 사용자가 Claude Desktop에서 "절점 (0,0,0)에 만들어줘", "해석 돌려줘" 같은 자연어를 입력하면 → LLM이 tool을 선택해 호출 → mcp-gennx가 GEN NX REST API를 호출 → GEN NX가 동작합니다.

### 7.2 MCP 서버의 진짜 최소 모양

진입점은 사실상 다음 한 덩어리입니다 (`src/mcp_gennx/__init__.py`).

```python
from .server import create_server

def main():
    server = create_server()
    server.run(transport="stdio")
```

**MCP 서버 = "도구 목록을 stdio로 떠드는 작은 프로세스"**

본질적으로는 이게 전부입니다. FastMCP 인스턴스를 만들고 stdio로 띄우면, 그 뒤로 Claude Desktop 같은 Host가 자식 프로세스로 띄워주고 메시지를 주고받습니다.

### 7.3 tool 하나는 결국 "함수 + 메타데이터"

tool 하나를 등록하는 모양은 매우 단순합니다 (`src/mcp_gennx/servers/project.py`).

```python
@server.tool(
    name="post_doc_anal",
    description="Run structural analysis in GEN NX.",
    tags={"project", "write"},
)
async def post_doc_anal(Argument: dict | None = None, *, ctx: Context) -> str:
    client = ctx.lifespan_context["api_client"]
    result = await client.post("doc/ANAL", {"Argument": Argument})
    return json.dumps(result)
```

핵심 포인트는 두 가지입니다.

- **함수 시그니처가 곧 tool 스키마**가 됩니다. LLM은 이 시그니처를 보고 어떤 인자가 필요한지 판단합니다.
- `@server.tool` 데코레이터 한 줄로 등록이 끝납니다.

작은 서버라면 이런 데코레이터 방식만으로도 충분합니다.

### 7.4 (참고) 이 레포의 특이한 점 — 동적 tool 생성

mcp-gennx는 159개의 tool을 손으로 일일이 데코레이터로 적지 않습니다. `src/mcp_gennx/tools/factory.py`의 `register_tools()`가 JSON 스키마를 읽어 클로저로 핸들러를 만들어 등록합니다.

> Tool 수가 수십~수백 개로 늘어나면 **데코레이터 대신 팩토리 패턴**으로 대량 등록할 수 있다는 정도만 기억해 두면 됩니다. 작은 서버를 만들 때는 굳이 이렇게까지 갈 필요 없습니다.

### 7.5 호출이 일어나는 순서

사용자가 "**절점 (0,0,0)에 만들어줘**"라고 입력했을 때 내부에서 일어나는 일은 다음과 같습니다.

1. **Host(Claude Desktop) → Client → mcp-gennx 서버**에 `tools/list` 요청 → 159개 tool 목록 수신
2. LLM이 `post_db_node`라는 tool을 선택하고 arguments를 생성
3. mcp-gennx가 GEN NX REST API(`POST /db/NODE`)에 HTTP 호출
4. 결과 JSON이 반대 경로로 LLM까지 돌아옴 → LLM이 사용자에게 자연어로 응답

이 4단계가 모든 MCP 서버의 공통 호출 흐름입니다. 도구가 무엇이든, 데이터 소스가 무엇이든 흐름 자체는 같습니다.

---

## 8. 직접 만들어보고 싶다면

- **Python**: `pip install fastmcp` → 데코레이터 몇 줄로 시작 (mcp-gennx도 이걸 씁니다)
- **TypeScript**: `@modelcontextprotocol/sdk`
- 최소 예제는 [공식 Quickstart](https://modelcontextprotocol.io/) 가 가장 빠릅니다.
- Tool이 5~10개 이하면 데코레이터 방식으로 충분합니다. 수십 개를 넘어가면 mcp-gennx 처럼 팩토리 패턴을 고려해 보세요.

---

## 9. 더 읽을거리

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [Anthropic의 MCP 발표 글](https://www.anthropic.com/news/model-context-protocol)
- [MCP 공식 GitHub 조직](https://github.com/modelcontextprotocol)
- 사내 레포: **mcp-gennx** (GEN NX MCP 서버 구현체)
