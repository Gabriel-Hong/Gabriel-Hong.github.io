---
layout: post
title: "[C++] Union (+실무 사례)"
date: 2025-10-22 10:00:00 +0900
categories: [Dev, Cpp]
pin: true
---

![](https://velog.velcdn.com/images/jeongmo511/post/c33ac0e1-db9f-421b-b1f7-84ba5cadd91d/image.png)


# Union(공용 구조체)이란?

C++에서 Union은 모든 멤버변수가 동일한 메모리 위치를 공유하는 사용자 정의 형식입니다.
멤버변수가 아무리 많더라도 항상 <span style="color:orange">**가장 큰 멤버 변수의 크기로 메모리를 할당**</span> 받게 됩니다.
따라서 멤버변수가 많지만, 메모리가 제한된 경우에 유용하게 사용될 수 있습니다.


![](https://velog.velcdn.com/images/jeongmo511/post/5d5f9b6d-bf88-4290-a010-9cd372f6202c/image.png)


## 📌 Union(공용 구조체)와 Struct(구조체)는 무엇이 다를까?
### Struct(구조체)

struct(구조체)는 내부에 여러 Data Type에 대한 멤버변수를 선언하고 <span style="color:orange">**여러 변수를 저장**</span>할 수 있습니다.

### Union(공용 구조체)

union(공용 구조체)는 여러 Data Type에 대한 멤버변수를 선언할 수는 있지만, union 내에 선언된 멤버변수는 <span style="color:orange">**하나의 공통된 메모리 공간을 공유**</span> 하기 때문에 <span style="color:orange">**하나의 멤버변수만 저장**</span>할 수 있습니다.

## 📌 Microsoft Learn 예제 살펴보기
### Mircrosoft Learn 예제

```cpp
#include <iostream>
using namespace std;

union NumericType
{
    short       iValue;
    long        lValue;
    double      dValue;
};

int main()
{
    union NumericType Values = { 10 };   // iValue = 10
    cout << Values.iValue << endl;
    Values.dValue = 3.1416;
    cout << Values.dValue << endl;
}
/* Output:
10
3.141600
*/
```

위의 코드는 Microsoft Learn 에서 union(공용 구조체)를 설명하기 위해 보여주고 있는 코드 입니다.
NumericType 이라는 union 공용 구조체 내에 여러 data type을 가지는 멤버변수를 만들어주고 이것들이 어떻게 메모리를 할당 받고 있는지를 아래의 그림이 보여주고 있습니다.

![](https://velog.velcdn.com/images/jeongmo511/post/bbc59b2b-d301-48a8-b685-b27f4e085b60/image.png)

<br/>
dValue는 double형으로 멤버변수들의 Data Type 중 크기가 8byte로 가장 크기 때문에 NumericType는 dValue의 크기로 메모리를 할당 받은 것을 알 수 있습니다.
<br/>

~~근데 short는 2byte니까 1byte인 bool이나 char를 쓰는게 그림이랑 맞지 않았을까...?~~

<br/>


## 📌 실무에서는 어떻게 쓰이고 있는가?

### 실무 사례 1

건축 관련 응용 소프트웨어에서 사용되고 있는 Wind Load 관련 Code를 살펴보면, 제품 UI 상에서 사용자가 선택한 풍하중 기준에 맞게 해당 Data 만 멤버변수로 가지고 있게 됩니다.

```cpp
union T_WIND_CODE
{
    T_WIND_KS1992   KS1992;
    T_WIND_JP1987   JP1987;
    T_WIND_UBC1997  UBC1997;
    T_WIND_ANSI1982 ANSI1982;
    T_WIND_KS2000   KS2000;
    T_WIND_IBC2000  IBC2000;
    T_WIND_EURO1992 EURO1992;
    T_WIND_BS6399   BS6399;
    T_WIND_CH2002   CH2002;
    T_WIND_JPN2000  JPN2000;
    T_WIND_NBC1995  NBC1995;
    T_WIND_IS1987   IS1987;
    T_WIND_TAIWAN86 TAIWAN86;
    T_WIND_JP2004   JP2004;
    T_WIND_EURO2005 EURO2005;
    T_WIND_KBC2009  KBC2009;
    T_WIND_IBC2012  IBC2009;
    T_WIND_IBC2012  IBC2012;
    T_WIND_CH2012   CH2012;
    T_WIND_CH2019   CH2019;
    T_WIND_NSR2010  NSR2010;
    T_WIND_KBC2015  KBC2015;
    T_WIND_IS875_2015 IS875_2015;
    T_WIND_KBC2015  KDS2019;
    T_WIND_KDS2021  KDS2021;
    T_WIND_DPT2007  DPT2007;
    T_WIND_ASCE7_16 ASCE7_16;
    T_WIND_ASCE7_16 ASCE7_22;
    
    void Initialize(int nCode)
    {
    	...
    }
 }
```

여기서 주의해야 할 점은 기존에 넣어 놓은 Data와 다른 멤버변수에 Data를 넣게 되면 <span style = "color:orange">**의도하지 않은 값들이 출력**</span>될 수 있는 문제가 발생하게 됩니다.

```cpp
const auto key = KDS_XXX_2019;

T_WIND_CODE DataOrg;
if ( !m_pDB->GetWind(key, DataOrg) )
{
	DataOrg.Initialize();
}

const auto key = KDS_XXX_2021;

T_WIND_CODE DataNew;
if ( !m_pDB->GetWind(key, DataNew) )
{
	DataNew.Initialize();
}

DataOrg = DataNew;	// 해당 멤버변수의 구조체 크기에 따라 쓰레기 값 출력 가능성이 있음

```

예를 들면,
1. KDS2019 기준의 Data를 DataOrg에 Get 해온다.
2. KDS2021 기준의 Data를 DataNew에 Get 해온다.
3. DataOrg에 DataNew의 Data를 넣어준다.
4. 해당 멤버변수 크기에 따라 <span style = "color:orange">**쓰레기 값이 출력**</span>될 수 있다.


### 실무 사례 2 
실제로 현재 개발하고 있는 제품의 소스코드에서는 다음과 같이 쓰이고 있습니다.

```cpp
#define T_NODE_K unsigned int
#define T_ABCD_KEY T_NODE_K
union T_ABCD_K
{
	T_ABCD_KEY keymap;
	struct
	{
		unsigned int entity : 20; // 0-1048576
		unsigned int serial : 12; // 0-4096
	}key;
};
```

T_ABCD_K 는 "keymap"과 "key"라는 두 개의 멤버 변수를 가지고 있습니다.
"key"는 구조체로 정의되어 있으며, 비트 필드(bit field)를 사용하여 두 개의 멤버 변수("entity"와 "serial")를 동시에 가지고 있습니다.
"entity"는 20bit, "serial"은 12bit를 차지하며, 이는 각각 0~1,048,576까지와 0~4,096 까지의 값을 표현할 수 있습니다.

따라서, "union T_ABCD_K"를 사용하면 "keymap"과 "key" 중 하나의 멤버 변수를 사용할 수 있고, 사용 중인 멤버 변수에 관계 없이 메모리를 공유하게 됩니다.

> 비트 필드(bit field)란?
데이터 구조체에서 여러 비트를 사용하여 하나의 변수에 여러 가지 정보를 압축하여 저장하는 기술

## 📌 결론
### 장점

1. union은 개체가 많고 메모리가 제한적인 경우에 메모리를 절약하는 데 유용

### 단점

위의 실무 예시와 같이,

1. union을 key로 사용하는 경우, <span style = "color:orange">**sorting 하기 어려운 문제**</span>가 발생
2. 메모리 최적화를 위해 비트 필드(bit field)를 사용한 경우에는 각 멤버변수에 <span style = "color:orange">**할당해 놓은 크기 이상의 Key가 들어가는 경우에는 쓰레기 값이 들어가게 되어**</span> 사용자 입장에서 버그가 발생하였는지 알기 어려운 문제도 발생
3. union을 Data로 사용하는 경우, Data가 수정될 경우에 <span style = "color:orange">**같은 멤버변수를 사용하지 않으면 값이 의도한대로 나오지 않는 문제**</span>가 발생

### 마무리
따라서 메모리를 절약하려는 의도로 막 쓰기 보다는 시기 적절하게 필요한 경우에만 잘 활용해주면 좋을 것 같습니다.