---
layout: post
title: "[Wasm] WebAssembly 란 무엇인가?"
date: 2025-07-30 15:38:00 +0900
categories: [Wasm]
pin: true
---

# WebAssembly란 무엇인가?

![](https://velog.velcdn.com/images/jeongmo511/post/e3c87748-85e1-489c-90b1-3c0b2773eb2a/image.png)

WebAssembly는 2015년부터 <span style = "color:orange">**JavaScript의 느린 속도를 보완 및 대체**</span>하기 위해 개발이 되었습니다.
위키에는 "웹 브라우저에서 실행하는 프로그래밍 언어이자 바이트코드이다. C, C++, Rust 시스템 프로그래밍 언어로 프로그램을 작성하고 컴파일한다." 라고 설명하고 있습니다.
결국, WebAssembly의 주 목적은 <span style = "color:orange">**웹페이지에서 고성능의 애플리케이션을 사용 가능하게 하는 것**</span>입니다.

## 📌 WebAssembly의 장점

WebAssembly는 다음과 같은 장점들을 가지고 있습니다.

----------------

### 1. 이식성(_Portable_)
WebAssembly의 바이트코드의 Binary 형식은 표준화되어 있습니다. 이는 <span style = "color:orange">**WebAssembly를 실행할 수 있는 모든 런타임에서 어떤 WebAssembly 코드가 됐든 실행 가능**</span>하다는 의미입니다. "한 번 작성하면 어디서나 실행 가능(_Write once, Run anywhere_)" 이라는 Java의 약속과 일맥상통합니다. 또한, 브라우저에서 95% 이상의 사용자들은 WebAssembly를 실행할 수 있습니다.

### 2. 범용성(_Universal_)
<span style = "color:orange">**많은 언어들이 WebAssembly로 컴파일**</span> 될 수 있습니다. C, C++ 및 Rust와 같은 시스템 언어를 비롯하여, Go, Python 및 Ruby와 같은 가비지 컬렉션 고수준 언어까지 포함합니다.

### 3. 네이티브에 가까운 성능(_Near-Native Performance_)
![](https://velog.velcdn.com/images/jeongmo511/post/5f46d77d-0501-46a4-833f-8b4d7a79f3ba/image.png)

WebAssembly는 "네이티브에 가까운 성능"을 가진다고들 말합니다. 이는 <span style = "color:orange">**컴퓨팅 집약적인 작업의 경우 WebAssembly가 거의 항상 JavaScript보다 빠르며**</span>, 네이티브 코드보다 평균 1.45 ~ 1.55배 정도 느립니다.

### 4. 빠른 시작 시간(_Fast Startup Time_)
WebAssembly의 콜드 스타트 시간은 그 자체로도 하나의 큰 주제를 차지할 만큼 중요합니다. 서버에서 모든 컨테이너에 대해 <span style = "color:orange">**새로운 OS 프로세스를 생성할 필요가 없기 때문에, Docker 컨테이너보다 10 ~ 100배 정도 더 빠른 콜드 스타트 시간을 달성할 수 있습니다.**</span> 브라우저에서 WebAssembly를 디코딩하고 기계어로 번역하는 것이 JavaScript를 구문 분석, 해석 및 최적화하는 것보다 빠르므로 WebAssembly 코드는 JavaScript 보다 더 빠르게 최고의 성능으로 실행될 수 있습니다.

> **콜드 스타트(_Cold Start_)**란?
WebAssembly 애플리케이션을 처음 실행할 때, 발생하는 초기화 과정을 의미합니다. WebAssembly는 브라우저에서 실행되는 애플리케이션의 일부로 사용되며, 이 초기화 과정은 브라우저에서 WebAssembly 모듈을 로드하고 실행할 때 발생합니다.

> ↔ **웜 스타드(_Warm Start_)**란?
모듈이 이미 로딩되어 있는 상태에서 이후에 실행되는 경우

### 5. 보안(_Secure_)
WebAssembly는 웹을 염두해 두고 설계되었기 때문에, 보안은 그 무엇보다 높은 우선순위에 있었습니다. <span style = "color:orange">**WebAssembly 런타임에서 실행되는 코드는 샌드박스 안에서 메모리가 할당되고, 기능이 제한**</span>됩니다. 즉, 명시적으로 허용된 작업만 수행할 수 있습니다. 샌드박스 안에서도 WebAssembly 코드는 여전히 시스템 수준 인터페이스 및 하드웨어 기능을 포함하여 기본 시스템에 액세스 할 수 있습니다.

> **샌드박스**(_Sandbox_)란?
어떤 응용 프로그램이나 프로세스가 다른 시스템 자원에 대한 액세스를 격리하고 제한하는 보안 메커니즘을 나타냅니다. 즉, 샌드박스는 애플리케이션이나 프로세스가 시스템 전체에 영향을 미치는 것을 방지하고, 보안 측면에서 안전한 환경을 제공합니다.

</br>

--------------

이러한 WebAssembly의 장점들로 인해, 개발자가 <span style = "color:orange">**품질과 성능에 따라 기본 데스크톱 앱과 다르지 않은 웹 앱을 만들 수 있다**</span>는 것입니다.

라이브러리와 프레임워크는 일반적으로 단일 언어로 작성되기 때문에 추가 개발 혹은 수정 없이는 다른 언어로 해당 코드를 활용하기는 어렵습니다. 하지만, WebAssembly를 활용하면 다른 언어로 작성된 코드를 손쉽게 실행할 수 있습니다. 이를 통해 처음부터 다시 개발하는 대신 <span style = "color:orange">**기존 코드를 재사용**</span>할 수 있습니다.

## 📌 WebAssembly를 활용한 제품

![](https://velog.velcdn.com/images/jeongmo511/post/27e3eb60-a407-46d5-bfe7-d3ab78b8f086/image.png)

WebAssembly를 활용한 제품으로는 대표적으로, 요즘 많은 분들이 사용하고 계시는 <span style = "color:orange">**Figma**</span> 가 있습니다. 건축 전공이신 분들은 많이 사용해보셨을 <span style = "color:orange">**Sketch Up**</Span>도 있고, 제가 개발하고 있는 제품의 경쟁사라고도 할 수 있는 Autodesk의 <span style = "color:orange">**AutoCAD**</span>도 이미 Web Assembly를 사용하여 웹상에서 제품을 실행할 수 있도록 만들어 주고 있습니다.

## 📌 WebAssembly의 원리

![](https://velog.velcdn.com/images/jeongmo511/post/b18c1f96-175f-47f9-ab80-e121f2ef1ac9/image.png)

공식 홈페이지에는 WebAssembly는 Compilation Target 이라고 명시되어 있습니다.

![](https://velog.velcdn.com/images/jeongmo511/post/b0e0885f-f2c3-4d57-9730-e2dba5c3cd99/image.png)

이렇게 컴파일이 된 WebAssembly 파일을 열면 그냥 Binary 파일입니다. 그렇기 때문에 우리는 WebAssembly를 배우지 않습니다. 읽지도 않고, 쓰지도 않습니다. 단지 WebAssembly로 컴파일 할 뿐입니다.

실제로 WebAssembly로 컴파일을 하면 다음과 같이 컴파일이 됩니다.

### 1. C 소스 코드
```cpp
#include <stdio.h>

int factorial(int n) {
  if (n == 0)
    return 1;
  else
    return n * factorial(n-1);
}
```

### 2. WebAssembly IR
```cpp
get_local 0
i64.eqz
if (result i64)
    i64.const 1
else
    get_local 0
    get_local 0
    i64.const 1
    i64.sub
    call 0
    i64.mul
end
```

### 3. WebAssembly Binary (.WASM)
```
20 00
50
04 7E
42 01
05
20 00
20 00
42 01
7D
10 00
7E
0B
```

이런 C의 소스코드가 다음과 같이 WebAssembly IR(Intermediate Representation), 즉, 중간코드로 변환된 후, 최적화가 되고 이렇게 알 수 없는 .WASM 이라는 WebAssembly의 Binary 바이트 코드로 컴파일 하게 됩니다. 다시 말해서, 크롬과 같은 <span style = "color:orange">**브라우저가 이해할 수 있는 코드로 변환**</span>이 된다는 이야기 입니다.

이제는 C++로 개발을 한 후, WebAssembly로 컴파일을 하면 Binary 파일을 컴파일 해서 브라우저에서 사용할 수 있게 되는 것입니다. 이것은 <span style = "color:orange">**응용 소프트웨어를 개발하는 개발자들에게 웹의 가능성을 열어준 것**</span>입니다.

간단한 예를 들어보겠습니다.

현재 내가 사진, 영상 편집을 하고 싶다면, 아직까지는 포토샵, 프리미어 프로 같은 소프트웨어를 다운 받아 설치를 해야 합니다. Mac에서는 Final Cut Pro와 같은 프로그램을 받아야 합니다. 포토샵, 프리미어 프로와 같은 작업을 설치 없이 사용하기 위해 웹에서 구현하려고 JavaScript를 사용하여 개발을 한다면, 무척이나 어렵고, 느리게 작동될 것입니다. JavaScript가 그런 작업들을 잘 하지는 못하기 때문입니다.

하지만, WebAssembly 덕분에 무거운 작업들을 하는 포토샵과 같은 프로그램을 브라우저에서 실행할 수 있게 됩니다. 이제는 PC에서 소프트웨어를 설치할 필요가 없어질 것이라는 이야기입니다. 왜냐하면 <span style = "color:orange">**브라우저가 이제 C, C++ 과 같은 언어들을 이해할 수 있게 되었기 때문**</span>입니다. 이것은 JavaScript 개발자가 아닌 다른 언어를 사용하는 개발자들에게는 엄청난 기회라고 할 수 있습니다.

## 📌 WebAssembly의 현황

![](https://velog.velcdn.com/images/jeongmo511/post/cdc0a441-bdb7-4d2c-aaf4-b6c4f40fda23/image.png)

거의 모든 <span style = "color:orange">**메이저 브라우저가 WebAssembly를 지원**</span>하고 있습니다. 여러분들이 잘 아시는 Mozilla의 Firefox, Google의 Chrome, Apple의 Safari, Microsoft의 Edge 등이 여기에 포함됩니다. 이것은 <span style = "color:orange">**Mozilla, Google, Apple, Microsoft의 기여와 함께 표준을 관리**</span>하고 있기 때문이기도 합니다.

게임을 만드는 Unity 3d, Unreal Engine은 이미 WebAssembly로 컴파일이 되고 있습니다. Unity 게임을 이제 웹사이트로 컴파일 하면 웹사이트는 해당 Binary 파일을 이해할 수 있게 되고, 덕분에 현재 Chrome, Firefox 등에서 데모나 3D 시뮬레이션을 볼 수 있는 것입니다.

## 📌 WebAssembly의 목적

그렇다면, WebAssembly가 JavaScript를 대체할 수 있을까요? 현업에 있는 사람들의 이야기를 빌리자면 적어도 향후 10 ~ 20년간은 "No" 라고 합니다. 가장 큰 이유는 WebAssembly는 JavaScript를 대체하려고 나온 것이 아니기 때문입니다. <span style = "color:orange">**JavaScript가 하지 못하는 것을 하기 위해**</span> 탄생한 것이 WebAssembly이기 때문입니다.

![](https://velog.velcdn.com/images/jeongmo511/post/32279a54-80d2-498c-812a-ab5b357f38a5/image.png)

예를 들면, 어떤 효과를 주거나, Drag & Drop 을 하거나 Click 등의 동작을 화려하게 만들어주고 싶다면, React, JavaScript가 필요할 것입니다. JavaScript가 웹을 화려하고 interactive하게 만드는 것을 잘하기 때문입니다.

반면, 이미지 프로세싱 같은 것은 실시간 계산, 빠른 계산 같은 여러 고려해야 하는 사항들이 있기 때문에 JavaScript가 잘 하지 못합니다. 이런 경우에는 WebAssembly를 활용하는 것이 효율적일 것입니다.

>이미지 프로세싱(Image Processing)
원래의 이미지를 프로그래밍을 이용해서 내가 원하는 정보를 얻거나, 이미지를 내가 원하는 방식으로 가공하는 것



# 결론

## 📌 요약

### 1. WebAssembly는 배우는 것이 아니다.
### 2. WebAssembly는 여러분이 읽거나 쓰는 것이 아니다.
### 3. 품질과 성능에 따라 기본 데스크톱 앱과 다르지 않은 웹 앱을 만들 수 있다.

## 📌 마무리

(예고) 다음에는 실제로 WebAssembly를 이용해 기존에 개발중인 제품 소스코드를 포팅하여 웹에서 작동하는 것을 확인해보겠습니다...? ^^

![](https://velog.velcdn.com/images/jeongmo511/post/95c7f2ea-b556-4542-88c4-a817d8b2d5d6/image.png)


<span style = "color:orange">**Docker 설립자**</span> Solomon Hykes 는 다음과 같이 말하였습니다.

**"2008년에 WASM+WASI가 있었다면 Docker를 만들 필요가 없었을 것입니다. 그만큼 중요합니다. 서버의 WebAssembly는 컴퓨팅의 미래입니다."**