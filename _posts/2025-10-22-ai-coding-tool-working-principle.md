---
layout: post
title: AI Coding Tool 동작 원리 이해하기
date: 2025-10-22 10:00:00 +0900
categories: [AI, Tool]
tags: [ai-coding, claude-code, cursor, copilot, opencode]
---

*이 문서는 **OpenCode** 아키텍처 분석을 기반으로 작성되었으며, 모든 AI 코딩 도구에 공통적으로 적용되는 원리를 설명합니다.*

## Opencode 프로젝트 개요

[OpenCode](https://github.com/opencode-ai/opencode)는 터미널에서 동작하는 오픈소스 AI 코딩 에이전트입니다. Claude Code, Cursor 같은 상용 도구들과 유사한 구조를 가지고 있어서, 이 프로젝트를 분석하면 AI 코딩 도구들이 내부적으로 어떻게 동작하는지 이해할 수 있습니다.

**기술 스택**

- **언어**: Go (Golang)
- **라이선스**: MIT (완전 오픈소스)
- **UI 프레임워크**: [Bubble Tea](https://github.com/charmbracelet/bubbletea) (터미널 UI)
- **데이터베이스**: SQLite (세션/메시지 저장)
- **지원 LLM**: Claude, GPT, Gemini, Bedrock, Azure OpenAI, Vertex AI, GitHub Copilot

**프로젝트 상태**

- **현재 상태**: 아카이브됨 (2025년 9월)
- **후속 프로젝트**: [Crush](https://github.com/charmbracelet/crush)로 계속 개발 중
- **분석 가치**: 코드가 공개되어 있어 AI 코딩 도구의 내부 구조를 학습하기에 적합

**디렉토리 구조 요약**

```
opencode/
├── main.go                # 진입점
├── cmd/                   # CLI 명령어 정의
├── internal/
│   ├── app/               # 애플리케이션 코어
│   ├── llm/
│   │   ├── agent/         # 에이전트 시스템 (핵심!)
│   │   ├── provider/      # LLM 프로바이더 (Claude, GPT 등)
│   │   ├── tools/         # 도구 시스템 (파일 읽기/쓰기, Bash 등)
│   │   └── prompt/        # 시스템 프롬프트
│   ├── session/           # 세션 관리
│   ├── permission/        # 권한 시스템
│   └── tui/               # 터미널 UI
└── ...
```

**왜 이 프로젝트를 분석했나?**

1. **오픈소스**: Claude Code 같은 상용 도구는 소스가 공개되지 않음
2. **구조 유사성**: 상용 도구들과 동일한 아키텍처 패턴 사용 (에이전트 루프, 도구 시스템 등)
3. **학습 목적**: Go 코드를 읽지 않아도, 구조와 흐름만 이해하면 됨

**유사한 오픈소스 프로젝트**

- **OpenCode** (Go, MIT) - 이 문서의 분석 대상
- **Aider** (Python, Apache 2.0) - Git 연동 강점
- **OpenAI Codex CLI** (Rust, Apache 2.0) - OpenAI 공식 CLI
- **Cline** (TypeScript, Apache 2.0) - VS Code 확장

## 목차

1. 개요 및 목적
2. AI 코딩 에이전트 = 일반 챗봇이 아니다
3. 핵심: 에이전트 루프 (Agent Loop)
4. 도구 시스템 (Tool System)
5. LLM 프로바이더 추상화
6. 권한 및 보안 모델
7. 세션 및 컨텍스트 관리
8. 시스템 프롬프트의 역할
9. 전체 데이터 흐름 (End-to-End)
10. 실전 활용 가이드

---

## 1. 개요 및 목적

### 이 문서의 목표

AI 코딩 도구(Claude Code, GitHub Copilot, Cursor, ChatGPT 등)를 사용하면서 "왜 이렇게 동작하지?", "왜 내 요청을 제대로 못 알아듣지?" 같은 의문을 가진 적이 있다면, 이 문서가 도움이 될 것입니다.

**이 문서는 AI 코딩 도구의 "블랙박스"를 열어봅니다.**

오픈소스 AI 코딩 에이전트인 OpenCode의 아키텍처를 분석하여, 모든 AI 코딩 도구에 공통적으로 적용되는 **구조와 동작 원리**를 설명합니다. 구현 코드는 다루지 않으며, 다이어그램과 표 중심으로 설명합니다.

### 이 문서를 읽고 나면

- AI 코딩 도구가 요청을 처리하는 **전체 흐름**을 이해할 수 있습니다
- 도구의 **제약사항**을 알고 이를 활용할 수 있습니다
- 더 **효과적인 프롬프트**를 작성할 수 있습니다
- AI와 **협업하는 방식**을 최적화할 수 있습니다

---

## 2. AI 코딩 에이전트 = 일반 챗봇이 아니다

### 일반 챗봇 vs AI 코딩 에이전트 비교표

| 구분 | 일반 챗봇 (ChatGPT 웹) | AI 코딩 에이전트 (Claude Code, Cursor 등) |
|------|----------------------|---------------------------------------|
| **입력** | 텍스트만 | 텍스트 + 파일 시스템 + 프로젝트 컨텍스트 |
| **출력** | 텍스트만 | 텍스트 + 파일 수정 + 명령어 실행 |
| **동작 방식** | 1회 질문-응답 | 반복 루프 (질문 → 분석 → 도구 호출 → 결과 확인 → 반복) |
| **파일 접근** | 불가 | 직접 읽기/쓰기/검색 가능 |
| **명령어 실행** | 불가 | 빌드, 테스트, Git 등 실행 가능 |
| **컨텍스트** | 대화 내용만 | 대화 + 파일 내용 + 프로젝트 구조 + LSP 진단 |
| **자율성** | 없음 (매번 사용자 입력 필요) | 높음 (스스로 여러 단계를 순차 실행) |
| **보안** | 해당 없음 | 권한 시스템으로 위험한 작업 통제 |

### 에이전트의 4대 구성요소

모든 AI 코딩 에이전트는 다음 4가지 핵심 요소로 구성됩니다.

```
+----------------------------------------------------------+
|                     AI Coding Agent                       |
|                                                          |
|  +---------------+       +---------------------+         |
|  |    Agent      |<----->|   LLM Provider      |         |
|  | (Controller)  |       | (AI Model Connect)  |         |
|  |               |       |                     |         |
|  | - Agent Loop  |       | - Claude, GPT, etc. |         |
|  | - Decisions   |       | - Streaming         |         |
|  | - Tool Invoke |       | - Retry Handling    |         |
|  +-------+-------+       +---------------------+         |
|          |                                               |
|          v                                               |
|  +---------------+       +---------------------+         |
|  |  Tool System  |       |      Session        |         |
|  |  (Executors)  |       | (State Management)  |         |
|  |               |       |                     |         |
|  | - File Read   |       | - Message History   |         |
|  | - File Write  |       | - Context Mgmt      |         |
|  | - Run Command |       | - Auto-Summary      |         |
|  | - Code Search |       | - Token/Cost Track  |         |
|  +---------------+       +---------------------+         |
|                                                          |
+----------------------------------------------------------+
```

| 구성요소 | 역할 | 비유 |
|----------|------|------|
| **Agent** | 전체 작업 흐름을 제어하는 컨트롤러 | 프로젝트 매니저 |
| **LLM Provider** | AI 모델과의 통신을 담당 | 외부 전문가 연결 |
| **Tool System** | 파일 조작, 명령어 실행 등 실제 작업 수행 | 도구 상자 |
| **Session** | 대화 상태와 컨텍스트를 관리 | 회의록/기록 |

---

## 3. 핵심: 에이전트 루프 (Agent Loop)

AI 코딩 에이전트의 가장 중요한 개념은 **에이전트 루프**입니다. 이것이 일반 챗봇과 코딩 에이전트를 구분하는 핵심입니다.

### 에이전트 루프 다이어그램

```
                    User Request
                         |
                         v
      +-------------------------------------------+
      |         1. Prepare Messages                |
      |  System Prompt + History + User Message    |
      +---------------------+---------------------+
                            |
                            v
      +-------------------------------------------+
      |      2. Send to LLM (Streaming)           |
      +---------------------+---------------------+
                            |
                            v
      +-------------------------------------------+
      |       3. Receive LLM Response             |
      |  - Text --> Display to User               |
      |  - ToolUse --> Execute Tool               |
      +---------------------+---------------------+
                            |
                            v
                  +--------+--------+
                  |  stop_reason?   |
                  +---+--------+---+
                      |        |
          end_turn    |        |   tool_use
                      v        v
               +---------+  +----------------------------+
               |   END   |  |      4. Execute Tool       |
               +---------+  | (Read, Edit, Search...)    |
                             +-------------+--------------+
                                           |
                                           v
                             +----------------------------+
                             |  5. Append Tool Result     |
                             |     to Messages            |
                             +-------------+--------------+
                                           |
                             +----> Back to Step 1
```

### 종료 조건

| stop_reason | 의미 | 다음 동작 |
|-------------|------|----------|
| `end_turn` / `stop` | LLM이 할 일을 마쳤다고 판단 | **루프 종료** → 사용자에게 최종 응답 전달 |
| `tool_use` | LLM이 도구를 호출하고 싶어함 | **루프 계속** → 도구 실행 후 결과 전달 |

### 실제 예시: "main.cpp 버그 찾아서 수정해줘"

다음은 사용자가 "main.cpp 파일에서 버그 찾아서 수정해줘"라고 요청했을 때 에이전트가 실제로 수행하는 단계입니다.

| 루프 | AI의 판단 | 수행 동작 | stop_reason |
|------|----------|----------|-------------|
| 1회차 | "먼저 파일을 읽어야 해" | `view` 도구로 main.cpp 읽기 | `tool_use` |
| 2회차 | "관련 헤더 파일도 봐야 해" | `view` 도구로 main.h 읽기 | `tool_use` |
| 3회차 | "버그를 찾았다, 수정하자" | `edit` 도구로 main.cpp 수정 | `tool_use` |
| 4회차 | "빌드해서 확인하자" | `bash` 도구로 `make build` 실행 | `tool_use` |
| 5회차 | "빌드 성공, 작업 완료" | 사용자에게 결과 보고 | `end_turn` |

> **핵심 인사이트**: AI는 한 번의 응답으로 끝나는 게 아니라, **여러 번의 도구 호출을 반복**하며 작업을 완료합니다. 복잡한 요청일수록 루프 횟수가 증가합니다.

---

## 4. 도구 시스템 (Tool System)

AI 코딩 에이전트가 실제로 작업을 수행하는 것은 **도구(Tool)**를 통해서입니다. AI 모델 자체는 텍스트만 생성할 수 있으며, 파일을 읽거나 수정하는 것은 모두 도구를 통해 이루어집니다.

### 도구 카테고리

| 카테고리 | 도구 | 설명 | 권한 필요 |
|----------|------|------|----------|
| **파일 읽기** | View/Read | 파일 내용 읽기 (라인 번호 포함) | No |
| **파일 수정** | Edit | 기존 파일의 특정 부분 수정 | Yes |
| **파일 생성** | Write | 새 파일 생성 또는 전체 덮어쓰기 | Yes |
| **코드 검색** | Grep | 정규식으로 텍스트 검색 | No |
| **파일 검색** | Glob | 파일 이름 패턴으로 검색 (예: `**/*.cpp`) | No |
| **디렉토리** | Ls | 디렉토리 내 파일 목록 조회 | No |
| **명령 실행** | Bash | 쉘 명령어 실행 (빌드, 테스트, Git 등) | 일부 |
| **패치 적용** | Patch | diff 형식의 패치 적용 | Yes |
| **웹 요청** | Fetch | HTTP 요청으로 외부 정보 조회 | No |
| **코드 진단** | Diagnostics | LSP를 통한 문법 오류/경고 확인 | No |
| **외부 검색** | Sourcegraph | 대규모 코드베이스 검색 | No |

### 에이전트 유형별 도구 권한

AI 코딩 에이전트는 작업 목적에 따라 다른 수준의 도구 접근 권한을 가집니다.

| 구분 | CoderAgent (메인 에이전트) | TaskAgent (서브 에이전트) |
|------|--------------------------|------------------------|
| **역할** | 코드 작성/수정의 전체 작업 수행 | 정보 수집/분석 (읽기 전용) |
| **파일 읽기** | ✅ | ✅ |
| **파일 수정/생성** | ✅ | ❌ |
| **명령어 실행** | ✅ | ❌ |
| **코드 검색** | ✅ | ✅ |
| **외부 도구 (MCP)** | ✅ | ❌ |
| **서브 에이전트 생성** | ✅ | ❌ |

> **실무 팁**: Claude Code의 "Task" 에이전트나 Cursor의 읽기 전용 모드는 TaskAgent 개념입니다. 복잡한 분석이 필요할 때 메인 에이전트가 서브 에이전트를 생성하여 병렬로 조사시킵니다.

### Edit 도구의 핵심 제약사항

Edit 도구는 AI 코딩 에이전트에서 가장 많이 사용되는 도구이며, **정확한 문자열 매칭**이라는 중요한 제약이 있습니다.

```
                    Edit Tool Input
      +-----------------------------------------+
      | file_path  = "main.cpp"                 |
      | old_string = "int result = a + b;"      |
      | new_string = "int result = a * b;"      |
      +-----------------------------------------+
                         |
                         v
              [old_string exists in file?]
                    |              |
                   No             Yes
                    |              |
                    v              v
                 ERROR:    [Appears exactly once?]
               "not found"      |              |
                               No             Yes
                                |              |
                                v              v
                             ERROR:    [Exact match? (whitespace, indent)]
                           "must be"       |              |
                           "unique"       No             Yes
                                          |              |
                                          v              v
                                       ERROR:        SUCCESS:
                                       "match        Replace
                                       failed"       and save
```

| 규칙 | 설명 | 실패 시 |
|------|------|---------|
| **존재성** | old_string이 파일에 반드시 존재 | "old_string not found" 에러 |
| **유일성** | old_string이 파일에 정확히 1번만 존재 | "appears N times, must be unique" 에러 |
| **정확성** | 공백, 들여쓰기, 줄바꿈 완벽 일치 | 매칭 실패 에러 |

> **실무 팁**: AI가 파일 수정에 실패하는 가장 흔한 원인은 **들여쓰기 불일치**입니다. AI가 파일을 먼저 읽지 않고 수정을 시도하면 실패할 확률이 높습니다. "먼저 파일을 읽고 나서 수정해줘"라고 명시하면 성공률이 올라갑니다.

### MCP를 통한 외부 도구 확장

MCP(Model Context Protocol)는 AI 코딩 에이전트에 **외부 도구를 플러그인처럼 추가**할 수 있는 표준 프로토콜입니다.

```
+----------------+                  +------------------------+
|   AI Agent     |<--- MCP -------->|     MCP Server         |
|  (Built-in     |    Protocol      |   (External Tools)     |
|   Tools)       |                  |                        |
+----------------+                  |  - Jira                |
                                    |  - Confluence          |
                                    |  - Internal DB         |
                                    |  - CI/CD Pipeline      |
                                    |  - Custom Tools        |
                                    +------------------------+
Built-in Tools + MCP Tools = All Available Tools for Agent
```

---

## 5. LLM 프로바이더 추상화

### Strategy Pattern으로 다중 모델 지원

AI 코딩 에이전트는 **Strategy Pattern(전략 패턴)**을 사용하여 다양한 AI 모델을 동일한 인터페이스로 지원합니다.

```
              +------------------------+
              |  Provider Interface    |
              |                        |
              |  - SendMessages()      |
              |  - StreamResponse()    |
              |  - Model()             |
              +----------+-------------+
                         |
     +------------+-------------+------------+
     |            |             |            |
     v            v             v            v
+----------+ +----------+ +----------+ +----------+
| Anthropic| |  OpenAI  | |  Gemini  | | Bedrock  |
| (Claude) | |  (GPT)   | | (Google) | | Azure ...|
+----------+ +----------+ +----------+ +----------+

Agent uses any model through the same interface
-> Swapping models does NOT affect agent logic
```

### 스트리밍 방식의 실시간 응답

AI 코딩 도구에서 텍스트가 한 글자씩 나타나는 것은 **스트리밍** 방식 덕분입니다.

```
Synchronous (traditional):
Request ---------- 3 sec wait ---------- Full response received

Streaming (real-time):
Request --> "Let" --> " me" --> " read" --> " the file" --> Done
             0.1s     0.2s      0.3s        0.4s

Event Types:
  ContentDelta : Text fragment (word ~ sentence)
  Thinking     : AI reasoning process (Extended Thinking)
  ToolUse      : Tool call request
  Finish       : Response complete (with stop_reason)
```

### Rate Limit 및 재시도 자동 처리

AI API에는 요청 속도 제한(Rate Limit)이 있으며, 에이전트가 이를 자동으로 처리합니다.

| HTTP 상태 코드 | 의미 | 에이전트의 대응 |
|---------------|------|---------------|
| 429 | Rate Limit 초과 | `Retry-After` 헤더만큼 대기 후 재시도 |
| 529 | 서버 과부하 | 10초 대기 후 재시도 |
| 500 | 서버 에러 | 지수 백오프(Exponential Backoff)로 재시도 |

> **실무 팁**: 응답이 느려지거나 에러가 발생하면 대부분 Rate Limit 때문입니다. 에이전트가 자동으로 재시도하므로 기다리면 됩니다. 반복적으로 발생하면 요청 빈도를 줄이거나 API 사용량 계획을 확인하세요.

---

## 6. 권한 및 보안 모델

AI 코딩 에이전트는 파일 수정이나 명령어 실행 같은 **위험한 작업**을 수행하기 전에 반드시 사용자 승인을 받습니다.

### 3단계 권한 체크 흐름도

```
          Tool requests dangerous action
                       |
                       v
        [Step 1: Auto-approve mode?]
       (--dangerously-skip-checks / yolo)
                  |            |
                 Yes          No
                  |            |
                  v            v
              Execute    [Step 2: Persistent permission?]
            immediately   (Previously "Always Allow")
                              |            |
                             Yes          No
                              |            |
                              v            v
                          Execute    [Step 3: Ask user]
                        immediately        |
                                           v
                              +---------------------+
                              | "bash: rm -rf build"|
                              |                     |
                              | [Allow]             |
                              | [Always Allow]      |
                              | [Deny]              |
                              +---------------------+
                                |       |       |
                                v       v       v
                             Execute  Save+   Cancel
                              once   Execute  action
```

### 명령어 분류

| 분류 | 예시 | 권한 |
|------|------|------|
| **안전한 명령어** (자동 승인) | `ls`, `cat`, `git status`, `git log`, `git diff`, `make`, `cmake`, `grep` | 승인 불필요 |
| **위험한 명령어** (승인 필요) | `rm`, `mv`, `git commit`, `git push`, `apt install` | 사용자 승인 필요 |
| **금지된 명령어** (실행 불가) | `curl`, `wget`, `ssh`, `nc`, `telnet`, 브라우저 실행 | 실행 차단 |

> **실무 팁**: 자주 승인 요청이 뜨는 것이 귀찮다면, 도구별로 "항상 허용"을 설정할 수 있습니다. 단, `rm`이나 `git push` 같은 명령어는 신중하게 판단하세요.

---

## 7. 세션 및 컨텍스트 관리

### 메시지 히스토리 구조

AI에게 전달되는 메시지는 다음과 같은 구조로 축적됩니다.

> **참고**: 매 LLM 호출 시 이 **전체 히스토리**가 전달됩니다. 히스토리가 길어질수록 토큰 비용이 증가합니다.

```
+----------------------------------------------------------+
| [System] System Prompt                                    |
| "You are an AI coding assistant. Rules: ..."              |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
| [User] User Message                                       |
| "Fix the bug in main.cpp"                                 |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
| [Assistant] AI Response + Tool Call                       |
| "Let me read the file" + view("main.cpp")                |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
| [Tool Result] Tool Output                                 |
| "1| #include <iostream>\n2| int main() { ..."            |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
| [Assistant] AI Response + Tool Call                       |
| "Found the bug, fixing it" + edit("main.cpp", ...)       |
+----------------------------------------------------------+
                            |
                            v
                           ...

* Entire history is sent with every LLM call
* Longer history = higher token cost
```

### 컨텍스트 윈도우와 자동 요약

AI 모델은 한 번에 처리할 수 있는 텍스트 양에 제한이 있습니다. 이것을 **컨텍스트 윈도우**라고 합니다.

| 모델 | 컨텍스트 윈도우 | 대략적인 분량 |
|------|---------------|-------------|
| Claude 3.5 Sonnet | 200K 토큰 | A4 약 300~400페이지 |
| GPT-4o | 128K 토큰 | A4 약 200~250페이지 |
| Gemini 1.5 Pro | 1M 토큰 | A4 약 1,500페이지 |

```
Context window usage over time:

[##########################------]  ~70% used
                |
                v (continue conversation)
[#############################--]  ~95% used --> TRIGGER!
                |
                v (auto-summarize)
[########-----------------------]  ~20% used

Summary content example:
"Previous conversation: Found memory leak bug in main.cpp,
 fixed with smart pointers. Build test passed."

* Detail may be lost during summarization
* Key decisions and outcomes are preserved
```

### 계층적 세션 (서브 에이전트)

메인 에이전트가 복잡한 작업을 처리할 때, **서브 에이전트**를 생성하여 독립적인 세션에서 작업을 수행시킵니다.

> - 서브 에이전트는 메인 에이전트의 컨텍스트 윈도우를 소비하지 않음
> - 서브 에이전트는 읽기 전용이므로 안전함

```
+------------------------------------------------------+
|            Main Session (CoderAgent)                  |
|            "Refactor entire project"                  |
|                                                      |
|  +-- Sub-Session 1 (TaskAgent, READ-ONLY)            |
|  |   "Analyze src/ directory structure"              |
|  |   -> Reports results to main                     |
|  |                                                   |
|  +-- Sub-Session 2 (TaskAgent, READ-ONLY)            |
|  |   "Survey test file coverage"                     |
|  |   -> Reports results to main                     |
|  |                                                   |
|  +-- Main agent combines results                     |
|      and performs modification work                   |
+------------------------------------------------------+

* Sub-agents do NOT consume main agent's context window
* Sub-agents are read-only (safe)
```

---

## 8. 시스템 프롬프트의 역할

### AI의 "역할 설명서" 개념

시스템 프롬프트는 AI에게 전달되는 **첫 번째 메시지**로, AI의 행동 규칙을 정의합니다. 사용자의 메시지보다 먼저 전달되며, AI의 모든 응답에 영향을 줍니다.

```
+------------------------------------------------------+
| [System Prompt]       <-- Defines AI role and rules   |
|                                                      |
| "You are an AI coding assistant.                     |
|  - Answer concisely (under 4 lines)                  |
|  - Use markdown                                      |
|  - Match code style when editing                     |
|  - Don't add unnecessary comments                    |
|  - Only commit when explicitly asked                 |
|  - Refer to project's CLAUDE.md file"                |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| [User] "Fix main.cpp"                                |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| [Assistant] (Responds following system prompt rules)  |
+------------------------------------------------------+
```

### 프로바이더별 프롬프트 차이

같은 AI 코딩 도구라도 사용하는 AI 모델에 따라 시스템 프롬프트가 다릅니다.

| 항목 | Anthropic (Claude) 계열 | OpenAI (GPT) 계열 |
|------|----------------------|------------------|
| **기본 지시** | "간결하게 4줄 이내로 답변" | "사용자의 쿼리가 완전히 해결될 때까지 계속해라" |
| **도구 사용** | "수정 전 파일의 코드 컨벤션을 파악해라" | "답변을 추측하지 말고 도구로 정보를 수집해라" |
| **자율성** | "적극적이되 과하지 않게" | "에이전트처럼 끝까지 진행해라" |
| **안전성** | "사용자를 놀라게 하지 마라" | "불확실하면 명확화 질문을 해라" |

### 에이전트 유형별 프롬프트

| 에이전트 유형 | 역할 | 프롬프트 핵심 내용 |
|-------------|------|-----------------|
| **Coder** | 코드 작성/수정 | "코딩 어시스턴트로서 파일을 수정하고 명령을 실행해라" |
| **Task** | 정보 수집 전용 | "리서치 어시스턴트로서 파일 수정 없이 정보만 수집해라" |
| **Summarizer** | 대화 요약 | "핵심 결정사항과 미해결 이슈 중심으로 500단어 이내로 요약해라" |
| **Title** | 제목 생성 | "50자 이내의 구체적이고 설명적인 제목을 생성해라" |

> **실무 팁**: AI가 과하게 장황한 답변을 한다면, 시스템 프롬프트나 사용자 설정에서 간결하게 답변하도록 조절할 수 있습니다. Claude Code에서는 `CLAUDE.md` 파일로, Cursor에서는 `.cursorrules` 파일로 커스텀 지시를 추가할 수 있습니다.

---

## 9. 전체 데이터 흐름 (End-to-End)

### 시퀀스 다이어그램

사용자 요청이 처리되는 전체 과정입니다.

```
User     UI      Agent    Provider    LLM      Tools
 |        |        |         |         |         |
 | 1.req  |        |         |         |         |
 |------->|        |         |         |         |
 |        | 2.Run()|         |         |         |
 |        |------->|         |         |         |
 |        |        |3.prepare|         |         |
 |        |        |  msgs   |         |         |
 |        |        |         |         |         |
 |        |        |4.Stream |         |         |
 |        |        |-------->|         |         |
 |        |        |         | 5.API   |         |
 |        |        |         |-------->|         |
 |        |        |         |         |         |
 |        |        |         | 6.text  |         |
 | 7.show |<-------|<--------|<--------|         |
 |<-------|        |         |         |         |
 |        |        |         | 8.tool  |         |
 |        |        |<--------|<--------|         |
 |        |        | 9.exec  |         |         |
 |        |        |-------------------------------->|
 |        |        |         |         | 10.result|
 |        |        |<--------------------------------|
 |        |        |         |         |         |
 |        |        | 11.add result, loop back    |
 |        |        |-------->| ------->|         |
 |        |        |         |         |         |
 |        |        |         | 12.end  |         |
 | 13.done|<-------|<--------|<--------|         |
 |<-------|        |         |         |         |
```

### 실제 예제 추적: C++ 파일 수정 요청

**요청**: "Vector3D 클래스에 normalize() 함수를 추가해줘"

| 단계 | 주체 | 동작 | 전달되는 데이터 |
|------|------|------|---------------|
| 1 | User → UI | 요청 입력 | "Vector3D 클래스에 normalize() 함수를 추가해줘" |
| 2 | UI → Agent | Run() 호출 | sessionID, 사용자 메시지 |
| 3 | Agent | 메시지 준비 | 시스템 프롬프트 + 사용자 메시지 |
| 4 | Agent → LLM | 스트리밍 요청 | 전체 메시지 + 사용 가능한 도구 목록 |
| 5 | LLM | 판단 | "먼저 파일 구조를 파악해야 해" |
| 6 | LLM → Agent | 도구 호출 | `glob("**/Vector3D*")` |
| 7 | Agent → Tools | Glob 실행 | 패턴: `**/Vector3D*` |
| 8 | Tools → Agent | 결과 반환 | `["src/math/Vector3D.h", "src/math/Vector3D.cpp"]` |
| 9 | Agent → LLM | 결과 전달 (루프 2회차) | 도구 결과를 메시지에 추가 |
| 10 | LLM → Agent | 도구 호출 | `view("src/math/Vector3D.h")` |
| 11 | Agent → Tools | View 실행 | 파일 내용 읽기 |
| 12 | Tools → Agent | 결과 반환 | Vector3D.h 전체 내용 (라인 번호 포함) |
| 13 | Agent → LLM | 결과 전달 (루프 3회차) | 파일 내용을 메시지에 추가 |
| 14 | LLM → Agent | 도구 호출 | `edit("src/math/Vector3D.h", old_string, new_string)` |
| 15 | Agent | 권한 확인 | 사용자에게 "Vector3D.h 수정 허용?" 표시 |
| 16 | User | 승인 | "허용" 클릭 |
| 17 | Agent → Tools | Edit 실행 | 파일 수정 + diff 생성 |
| 18 | LLM → Agent | 도구 호출 (루프 4회차) | `view("src/math/Vector3D.cpp")` → `edit("src/math/Vector3D.cpp", ...)` |
| 19 | LLM → Agent | `end_turn` | "normalize() 함수를 추가했습니다. 변경 내용은..." |
| 20 | Agent → User | 최종 응답 | 수정 결과 보고 |

---

## 10. 실전 활용 가이드

### 10.1 효과적인 프롬프트 작성법

| 구분 | 나쁜 예 | 좋은 예 | 이유 |
|------|---------|---------|------|
| **구체성** | "코드 고쳐줘" | "Vector3D.h의 normalize()에서 0벡터 처리 누락 버그를 수정해줘" | AI가 정확한 위치와 문제를 바로 파악 |
| **범위** | "프로젝트 전체를 리팩토링해줘" | "src/math/ 디렉토리의 Vector 클래스들을 템플릿으로 통합해줘" | 범위가 명확해야 한 세션에서 완료 가능 |
| **조건** | "테스트 추가해줘" | "normalize()에 대한 단위 테스트를 Google Test로 작성해줘. 0벡터, 단위벡터, 일반 벡터 케이스를 포함해" | 구체적 조건이 있으면 재작업이 줄어듦 |
| **컨텍스트** | "빌드 에러 고쳐줘" | "CMake 빌드 시 'undefined reference to normalize' 에러가 발생해. Vector3D.cpp에 구현을 추가해줘" | 에러 메시지를 포함하면 분석 시간 절약 |
| **순서** | "파일 만들고 수정하고 테스트해줘" | "1. Vector3D.h에 normalize() 선언 추가 2. Vector3D.cpp에 구현 추가 3. test_vector.cpp에 테스트 추가 4. cmake --build로 빌드 확인" | 단계가 명확하면 AI가 순서대로 진행 |

### 10.2 도구 제약사항 이해하고 활용하기

| 제약사항 | 영향 | 대응 전략 |
|---------|------|----------|
| Edit 도구는 정확한 문자열 매칭 필요 | 파일을 안 읽고 수정하면 실패 | "먼저 파일을 읽고 나서 수정해줘" 명시 |
| 파일 읽기 기본 2000줄 제한 | 대규모 파일의 일부만 읽음 | 필요한 라인 범위를 직접 지정 |
| Bash 출력 30,000자 잘림 | 긴 빌드 로그가 잘릴 수 있음 | "에러만 보여줘" 또는 필터링 명령 사용 |
| 이미지 파일 읽기 불가 (일부 도구) | 스크린샷 기반 디버깅 제한 | 에러 메시지를 텍스트로 복사하여 전달 |
| 네트워크 명령 차단 | curl, wget 등 사용 불가 | 전용 Fetch 도구 사용 또는 로컬 파일 기반 |

### 10.3 AI의 작업 순서 이해하기

AI 코딩 에이전트는 경험 많은 개발자와 유사한 패턴으로 작업합니다.

```
1. Explore
   |  glob, ls: Understand project structure
   |  "What files are there?"
   v
2. Read
   |  view: Read relevant file contents
   |  "How is the existing code structured?"
   v
3. Analyze
   |  grep: Search related code, find dependencies
   |  "Where is this function used?"
   v
4. Modify
   |  edit, write: Change code
   |  "Match code style and apply fix"
   v
5. Verify
   |  bash: Build, run tests
   |  "Does the change work correctly?"
   |
   +---> If failed, go back to Step 4
   v
6. Report
   Summarize changes for the user
```

### 10.4 컨텍스트 관리 전략

| 전략 | 설명 | 적용 상황 |
|------|------|----------|
| **세션 분리** | 주제가 다른 작업은 새 세션에서 시작 | 버그 수정 후 새 기능 작업으로 전환할 때 |
| **요약 요청** | "지금까지의 작업을 요약해줘" | 긴 작업 중간에 컨텍스트 정리가 필요할 때 |
| **프로젝트 메모** | `CLAUDE.md`, `.cursorrules` 등에 프로젝트 규칙 기록 | 매번 반복 설명하기 귀찮은 규칙이 있을 때 |
| **작은 단위로 작업** | 한 번에 하나의 파일/기능에 집중 | 대규모 리팩토링을 단계별로 나눌 때 |
| **명시적 컨텍스트** | "앞에서 수정한 Vector3D.h를 기반으로..." | AI가 이전 작업을 참조해야 할 때 |

### 10.5 일반적인 실수 5가지와 해결책

| # | 실수 | 원인 | 해결책 |
|---|------|------|--------|
| 1 | **한 세션에서 너무 많은 작업** | 컨텍스트 윈도우 초과 → 자동 요약 → 이전 정보 손실 | 작업 단위를 나눠서 새 세션에서 진행 |
| 2 | **파일 경로를 불명확하게 지정** | AI가 잘못된 파일을 수정하거나 못 찾음 | 전체 경로를 명시하거나 "먼저 파일 위치를 찾아줘" |
| 3 | **빌드 환경 정보 누락** | AI가 빌드 시스템을 모르고 잘못된 명령 실행 | "CMake 기반 프로젝트이고, build/ 디렉토리에서 빌드한다" 명시 |
| 4 | **수정 후 검증 안 함** | 구문 에러나 논리 에러 누락 | "수정 후 빌드해서 확인해줘" 항상 추가 |
| 5 | **결과를 검토 없이 수락** | AI가 의도와 다른 수정을 할 수 있음 | diff를 항상 확인하고, 의문이 있으면 되묻기 |

### 10.6 도구별 활용 팁

#### Claude Code

| 항목 | 팁 |
|------|-----|
| **프로젝트 설정** | 프로젝트 루트에 `CLAUDE.md` 파일을 두면 매 세션마다 자동으로 읽음. 빌드 명령어, 코딩 컨벤션, 프로젝트 구조 등을 기록해두면 효과적 |
| **서브 에이전트 활용** | 복잡한 분석은 AI가 자동으로 Task 에이전트를 생성하여 처리. "이 프로젝트의 의존성 구조를 분석해줘" 같은 요청 시 활용됨 |
| **권한 관리** | 자주 사용하는 안전한 명령어는 "항상 허용"으로 설정하여 작업 흐름 개선 |
| **비대화형 모드** | `claude -p "파일 목록 보여줘"` 식으로 일회성 요청 가능. 스크립트에 통합할 때 유용 |

#### GitHub Copilot

| 항목 | 팁 |
|------|-----|
| **인라인 완성** | 코드 작성 중 자동 완성 제안. 주석으로 의도를 미리 적으면 더 정확한 제안 |
| **Chat 모드** | IDE 내에서 대화형으로 코드 질문/생성. `@workspace` 태그로 전체 프로젝트 컨텍스트 활용 |
| **컨텍스트 제한** | 현재 파일 + 열린 탭 위주로 컨텍스트를 구성. 관련 파일을 미리 열어두면 더 좋은 제안 |
| **단축키** | Tab으로 수락, Esc로 거부. 부분 수락도 가능 (Ctrl+→) |

#### Cursor

| 항목 | 팁 |
|------|-----|
| **Rules 파일** | `.cursorrules` 파일에 프로젝트 규칙을 정의하면 모든 응답에 반영 |
| **Composer 모드** | 여러 파일을 동시에 수정하는 대규모 작업에 적합. 에이전트 모드에서 자동으로 관련 파일 탐색 |
| **컨텍스트 추가** | `@file`, `@folder`, `@codebase` 태그로 명시적 컨텍스트 제공 |
| **Apply vs Accept** | AI 제안을 파일에 적용(Apply)하기 전에 diff를 반드시 확인 |

#### ChatGPT (웹/API)

| 항목 | 팁 |
|------|-----|
| **코드 붙여넣기** | 파일 시스템 접근이 없으므로 관련 코드를 직접 붙여넣어야 함 |
| **컨텍스트 한계** | 에이전트형 도구 대비 컨텍스트가 제한적. 한 번에 많은 정보를 전달할수록 효과적 |
| **코드 리뷰** | 파일 수정 기능이 없으므로 코드 리뷰, 알고리즘 설계, 개념 설명에 적합 |
| **Custom GPT** | 반복적인 작업 패턴이 있다면 Custom GPT를 만들어 시스템 프롬프트를 미리 설정 |

### 10.7 팀 협업 가이드라인

#### Do

| 항목 | 설명 |
|------|------|
| **프로젝트 규칙 파일 공유** | `CLAUDE.md`, `.cursorrules` 등을 저장소에 커밋하여 팀 전체가 동일한 AI 설정 사용 |
| **AI 수정 코드 리뷰** | AI가 생성한 코드도 동일한 코드 리뷰 프로세스를 거치기 |
| **빌드/테스트 검증** | AI 수정 후 반드시 빌드 + 테스트 통과 확인 |
| **커밋 메시지에 AI 사용 표기** | AI 도움을 받은 커밋에는 co-authored-by 등으로 표기 |
| **작업 범위 공유** | AI에게 맡긴 작업 범위를 팀원에게 공유하여 충돌 방지 |
| **프롬프트 공유** | 효과적이었던 프롬프트 패턴을 팀 내에서 공유 |

#### Don't

| 항목 | 설명 |
|------|------|
| **검토 없이 머지** | AI 생성 코드를 리뷰 없이 머지하지 않기 |
| **민감 정보 전달** | API 키, 비밀번호, 내부 서버 주소 등을 프롬프트에 포함하지 않기 |
| **맹목적 신뢰** | AI 제안이 항상 옳다고 가정하지 않기. 특히 아키텍처 결정은 사람이 판단 |
| **대규모 자동 수정** | AI에게 프로젝트 전체를 한 번에 수정하게 하지 않기. 단계별로 진행 |
| **보안 코드 AI 의존** | 인증, 암호화, 권한 관련 코드는 AI 제안을 반드시 보안 전문가가 검토 |
| **컨텍스트 없는 요청** | "고쳐줘"만 말하지 않기. 충분한 배경 정보 제공 |

---

## 부록: 용어 정리

| 용어 | 설명 |
|------|------|
| **Agent Loop** | AI가 요청을 처리하기 위해 "판단 → 도구 호출 → 결과 확인"을 반복하는 핵심 메커니즘 |
| **LLM (Large Language Model)** | 대규모 언어 모델. Claude, GPT, Gemini 등 |
| **Tool** | AI가 실제 작업(파일 읽기/쓰기, 명령 실행 등)을 수행하는 인터페이스 |
| **Provider** | LLM API와 통신하는 어댑터 계층 |
| **Session** | 하나의 대화 상태를 관리하는 단위 |
| **Context Window** | AI 모델이 한 번에 처리할 수 있는 최대 텍스트 양 (토큰 단위) |
| **Token** | AI가 텍스트를 처리하는 최소 단위. 한국어 1글자 ≈ 2~3 토큰 |
| **Streaming** | 응답을 조각 단위로 실시간 전송하는 방식 |
| **System Prompt** | AI의 행동 규칙을 정의하는 사전 지시문 |
| **MCP (Model Context Protocol)** | AI 에이전트에 외부 도구를 연결하는 표준 프로토콜 |
| **LSP (Language Server Protocol)** | 코드 편집기에 언어별 기능(자동완성, 진단 등)을 제공하는 프로토콜 |
| **Extended Thinking** | Claude의 "사고 과정"을 볼 수 있는 기능 |
| **Rate Limit** | API 호출 빈도 제한 |
| **stop_reason** | LLM이 응답을 멈춘 이유 (`end_turn`, `tool_use` 등) |

---

## 참고 자료

- [OpenCode GitHub (MIT)](https://github.com/opencode-ai/opencode) — 이 문서의 분석 대상
- [Claude Code Best Practices (Anthropic)](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- [OpenAI Codex CLI](https://github.com/openai/codex)
- [Aider](https://github.com/paul-gauthier/aider)
