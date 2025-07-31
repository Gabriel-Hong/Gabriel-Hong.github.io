---
layout: post
title: WebAssembly 개발환경 설정
date: 2025-07-31 09:39:00 +0900
categories: [Study, Wasm]
pin: true
---

# WebAssembly 개발환경 설정

Window 11에서 C++로 개발된 소스를 WebAssembly를 통해 웹상에서 동작시키기 위한 개발환경 설정을 해보겠습니다.

-------------------------------

## 📌 Python 설치하기

1. Python 공식 홈페이지에서 설치파일을 다운로드 합니다.

	Reference : https://www.python.org/downloads/
    
2. 설치파일을 실행합니다.

![](https://velog.velcdn.com/images/jeongmo511/post/cc335be6-45f8-4eeb-9775-74d18cbeee69/image.png)

반드시 <span style = "color:orange">**Add Python 3.XX to PATH**</span> 를 체크해서 환경변수 설정이 반영될 수 있도록 해야 합니다.

-------------------------

## 📌 emsdk 설치하기

1. cmd를 실행하고 설치할 폴더의 경로로 이동해줍니다.

![](https://velog.velcdn.com/images/jeongmo511/post/6f1fcf70-a251-491a-9ddb-af9d3e50b656/image.png)

2. 아래 커맨드를 입력해줍니다.

```
// Get the emsdk repo
git clone https://github.com/emscripten-core/emsdk.git

// Enter that directory
cd emsdk
```

3. emsdk를 설치합니다.
```
// Fetch the latest version of the emsdk
git pull
```

아래의 (1), (2) 순서대로 배치파일을 실행합니다.
(1) emsdk : emsdk install latest
(2) emsdk_env : emsdk activate latest

이렇게 환경변수 설정을 해주면 emcc 및 emscripten 도구를 사용할 수 있게 됩니다.

![](https://velog.velcdn.com/images/jeongmo511/post/46fcdc8b-a8be-4f47-932c-6f76248e88ef/image.png)

혹은 cmd 창에서 emsdk_env를 실행합니다.

아래의 커맨드로 emscripten이 잘 설치가 되었는지 확인할 수 있습니다.

```
emcc --version
```

![](https://velog.velcdn.com/images/jeongmo511/post/316804e8-08e2-40c4-b0b5-5172beb92f49/image.png)

------------------

## 📌 환경변수 설정하기

1. 시스템 속성에서 환경변수를 추가해줍니다.

![](https://velog.velcdn.com/images/jeongmo511/post/76449e85-4cf6-4767-b529-8d80f9141bc2/image.png)

2. 환경변수에 들어갑니다.

![](https://velog.velcdn.com/images/jeongmo511/post/32de9211-12bf-45a4-9c13-88c2e436cfce/image.png)

3. 사용자 변수의 Path 를 편집하기로 들어갑니다.

![](https://velog.velcdn.com/images/jeongmo511/post/dcba8532-1124-4d41-aa1d-3cc9f999547d/image.png)

4. 환경변수를 추가해줍니다.

![](https://velog.velcdn.com/images/jeongmo511/post/ca387332-6bd7-4c5a-80ab-e65ecd0c00a6/image.png)

```
// emsdk 경로
C:\...\01_Study\03_WASM\emsdk

// emscripten
C:\...\01_Study\03_WASM\emsdk\upstream\emscripten
```

## 📌 VS Code 에서 emcc 명령어 작동 확인

1. VS Code 내에서 emcc 명령어가 작동이 되는지 확인해봅니다.

![](https://velog.velcdn.com/images/jeongmo511/post/ce3e88ac-b548-4f07-843c-b4edf09f914c/image.png)

2. 컴파일이 된다면 정상적으로 모든 설정이 완료된 것입니다.
--------------------


> 자세한 사항은 아래 사이트를 참고해주세요.

Reference :
https://emscripten.org/index.html
https://developer.mozilla.org/en-US/docs/WebAssembly