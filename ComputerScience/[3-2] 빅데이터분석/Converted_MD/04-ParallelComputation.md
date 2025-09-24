# 04. 병렬 컴퓨팅 - Python을 활용한 분산 처리

## 📚 개요

병렬 컴퓨팅은 대용량 데이터를 효율적으로 처리하기 위한 핵심 기술입니다. 이 실습에서는 Python을 사용하여 다양한 병렬 처리 방법을 학습하고, 실제 빅데이터 분석에 적용하는 방법을 익힙니다.

### 🎯 학습 목표
- 병렬 컴퓨팅의 기본 개념과 아키텍처 이해
- Python의 병렬 처리 라이브러리 활용법
- 멀티프로세싱과 멀티스레딩의 차이점 파악
- 실제 데이터 처리 작업에 병렬 처리 적용

## 🖥️ 병렬 컴퓨터의 종류

### 📋 **병렬 컴퓨터 분류**

#### 1️⃣ **멀티프로세서/멀티코어 (Multiprocessor/Multicore)**
- **특징**: 여러 개의 프로세서가 공유 메모리에서 데이터 처리
- **장점**: 빠른 메모리 접근, 낮은 지연시간
- **용도**: 단일 시스템 내 고성능 계산

#### 2️⃣ **클러스터 (Cluster)**
- **특징**: 여러 개의 프로세서/메모리 유닛이 네트워크를 통해 데이터 교환
- **장점**: 높은 확장성, 비용 효율성
- **용도**: 대규모 분산 처리, 빅데이터 분석

#### 3️⃣ **코프로세서 (Co-processor)**
- **특징**: 범용 프로세서가 GPU와 같은 특수 목적 프로세서에 작업 위임
- **장점**: 특화된 연산에 최적화된 성능
- **용도**: 머신러닝, 과학 계산, 그래픽 처리

## 🔧 병렬 프로그래밍의 핵심 원리

병렬 프로그래밍은 복잡한 작업을 효율적으로 처리하기 위한 체계적인 접근 방법입니다.

### 📋 **병렬 프로그래밍의 주요 단계**

#### 1️⃣ **작업 분해 (Task Decomposition)**
- **목표**: 전체 작업을 독립적인 하위 작업으로 분해
- **방법**: 데이터 흐름을 정의하고 의존성 분석
- **장점**: 각 작업을 독립적으로 처리 가능

#### 2️⃣ **작업 배분 (Task Distribution)**
- **목표**: 프로세서들에 하위 작업을 배분하여 전체 실행 시간 최소화
- **고려사항**: 각 프로세서의 성능과 작업의 복잡도

#### 3️⃣ **통신 최적화 (Communication Optimization)**

##### 🌐 **클러스터 환경**
- **목표**: 통신 시간 최소화
- **방법**: 노드 간 데이터의 적절한 배분
- **전략**: 데이터 지역성(Data Locality) 활용

##### 🖥️ **멀티프로세서 환경**
- **목표**: 대기 시간 최소화
- **방법**: 메모리 접근 패턴 최적화
- **전략**: 캐시 친화적 알고리즘 설계

#### 4️⃣ **동기화 (Synchronization)**
- **목표**: 개별 프로세스 간의 조율
- **방법**: 락(Lock), 세마포어(Semaphore), 배리어(Barrier) 활용
- **중요성**: 데이터 일관성과 경쟁 상태 방지

## 🗺️ MapReduce 패턴 이해

MapReduce는 병렬 처리를 위한 프로그래밍 모델로, 대용량 데이터를 효율적으로 처리하는 핵심 패턴입니다.

### 📝 **기본 예제: 제곱 계산**

```python
from time import sleep

def f(x):
    sleep(1)  # 1초 대기 (실제 작업 시뮬레이션)
    return x * x

# 처리할 데이터
L = list(range(8))
print("데이터:", L)
```

### ⏱️ **성능 측정**

#### 📊 **리스트 컴프리헨션 방식**
```python
# %time 명령어를 사용하여 실행 시간 측정
%time sum(f(x) for x in L)
```

#### 📊 **map 함수 방식**
```python
# map 함수를 사용한 병렬 처리
%time sum(map(f, L))
```

### 💡 **MapReduce의 핵심 개념**
- **Map 단계**: 각 요소에 함수를 적용하여 변환
- **Reduce 단계**: 변환된 결과들을 집계하여 최종 결과 생성
- **병렬성**: 각 요소를 독립적으로 처리 가능
- **확장성**: 데이터 크기에 따라 자동으로 확장

## 🔄 Multiprocessing - 프로세스 기반 병렬 처리

`multiprocessing` 라이브러리는 Python에서 프로세스 기반 병렬 처리를 지원하는 핵심 도구입니다.

### 📚 **Multiprocessing의 특징**
- **프로세스 생성**: 새로운 프로세스를 생성하여 병렬 실행
- **독립성**: 각 프로세스는 독립적인 메모리 공간을 가짐
- **안전성**: GIL(Global Interpreter Lock) 제약을 우회
- **확장성**: CPU 코어 수에 따라 자동 확장

### 🖥️ **시스템 리소스 확인**

#### 📊 **CPU 코어 수 확인**
```python
from multiprocessing import cpu_count

# 사용 가능한 CPU 코어 수 확인
print("사용 가능한 CPU 코어 수:", cpu_count())
```

#### 💡 **CPU 코어 수의 중요성**
- **병렬 처리 한계**: 동시에 실행 가능한 프로세스 수의 상한
- **성능 최적화**: 코어 수에 맞는 프로세스 수 설정
- **리소스 관리**: 시스템 과부하 방지

## ⚡ Futures - 고수준 병렬 처리 인터페이스

`concurrent.futures` 모듈은 호출 가능한 객체를 비동기적으로 실행할 수 있는 고수준 인터페이스를 제공합니다.

### 📚 **Futures의 핵심 개념**

#### 🎯 **비동기 실행 (Asynchronous Execution)**
- **ThreadPoolExecutor**: 스레드를 사용한 병렬 처리
- **ProcessPoolExecutor**: 프로세스를 사용한 병렬 처리
- **통일된 인터페이스**: 두 방식 모두 동일한 Executor 클래스 기반

#### 🖥️ **운영체제별 제약사항**
- **Windows 제한**: `concurrent.futures`는 Windows에서 프로세스 시작 제한
- **해결책**: [loky](https://github.com/tomMoral/loky) 라이브러리 설치 필요
- **크로스 플랫폼**: loky를 사용하면 모든 OS에서 동일하게 동작

### 📝 **실제 구현 예제**

#### 🔧 **ProcessPoolExecutor 사용**
```python
from concurrent.futures import ProcessPoolExecutor
from time import sleep, time

def f(x):
    sleep(1)  # 1초 대기
    return x * x

L = list(range(8))

if __name__ == '__main__':
    begin = time()
    with ProcessPoolExecutor() as pool:
        result = sum(pool.map(f, L))
    end = time()
    
    print(f"결과 = {result}, 실행 시간 = {end-begin}")
```

### 💡 **Futures의 장점**
- **간단한 API**: 복잡한 스레드/프로세스 관리 자동화
- **자동 리소스 관리**: 컨텍스트 매니저로 안전한 리소스 해제
- **유연성**: 스레드와 프로세스 간 쉬운 전환
- **확장성**: 작업 수에 따른 자동 스케일링

### 🚀 **실행 및 성능 측정**

#### 📊 **실행 방법**
```python
import sys
!{sys.executable} pmap.py
```

### 🔧 **ProcessPoolExecutor 동작 원리**

#### 📋 **세부적인 메소드 역할**

##### 1️⃣ **ProcessPoolExecutor**
- **역할**: 컴퓨터 내 물리적 코어당 하나의 slave 프로세스 시작 및 실행
- **최적화**: CPU 코어 수에 맞는 프로세스 수 자동 설정
- **효율성**: 각 코어를 최대한 활용

##### 2️⃣ **pool.map() 메소드**
- **작업 분할**: 입력 리스트를 여러 개의 청크로 분할
- **큐 관리**: (함수 + 청크) 형태의 작업을 큐에 추가
- **병렬 실행**: 각 slave 프로세스가 독립적으로 작업 처리

##### 3️⃣ **결과 수집**
- **작업 수행**: 각 slave 프로세스가 `map(함수, 청크)` 실행
- **결과 저장**: results 리스트에 결과 저장
- **동기화**: master 프로세스가 모든 작업 완료까지 대기
- **결과 반환**: 결과 리스트들을 연결하여 최종 결과 반환

### 🧵 **ThreadPoolExecutor 사용**

#### 📊 **스레드 기반 병렬 처리**
```python
%%time
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as pool:
    results = sum(pool.map(f, L))
    
print("결과:", results)
```

### 💡 **ThreadPoolExecutor vs ProcessPoolExecutor**

#### 🧵 **ThreadPoolExecutor**
- **장점**: 빠른 시작, 메모리 공유
- **단점**: GIL 제약으로 인한 제한적 병렬성
- **용도**: I/O 집약적 작업

#### 🔄 **ProcessPoolExecutor**
- **장점**: 진정한 병렬 처리, GIL 우회
- **단점**: 높은 메모리 사용량, 느린 시작
- **용도**: CPU 집약적 작업

## 🧵 Thread vs Process - 핵심 차이점

스레드와 프로세스는 병렬 프로그래밍의 두 가지 기본 단위로, 각각 고유한 특징과 용도를 가지고 있습니다.

### 📋 **Process (프로세스)**

#### 🎯 **정의**
- **프로세스**: 실행 중인 프로그램의 인스턴스
- **독립성**: 자체 실행 환경을 가진 완전한 단위
- **메모리**: 독립적인 메모리 공간 보유

#### 🔧 **특징**
- **포함 관계**: 하나 이상의 스레드를 포함할 수 있음
- **격리성**: 다른 프로세스와 메모리를 공유하지 않음
- **통신**: 프로세스 간 통신은 데이터 직렬화 필요
- **협력**: 컴퓨터에서 실행되는 애플리케이션은 협력하는 프로세스들의 집합

### 🧵 **Thread (스레드)**

#### 🎯 **정의**
- **스레드**: 프로세스 내에서 생성되고 존재
- **필수성**: 모든 프로세스는 최소 하나의 스레드를 가짐
- **공유**: 프로세스 내 여러 스레드가 리소스 공유

#### 🔧 **특징**
- **효율적 통신**: 스레드 간 리소스 공유로 빠른 통신
- **동시성**: 멀티코어 시스템에서 동시 실행 가능
- **경량성**: 프로세스보다 생성 및 관리 비용이 낮음

### 💡 **언제 무엇을 사용할까?**

#### 🧵 **Thread 사용 시기**
- **I/O 집약적 작업**: 파일 읽기, 네트워크 통신
- **빠른 응답성**: 사용자 인터페이스 업데이트
- **메모리 효율성**: 메모리 사용량이 중요한 경우

#### 🔄 **Process 사용 시기**
- **CPU 집약적 작업**: 수학적 계산, 데이터 처리
- **안정성**: 하나의 프로세스 실패가 전체에 영향 주지 않음
- **진정한 병렬성**: GIL 제약을 우회하고 싶을 때

## 🔒 GIL (Global Interpreter Lock) - Python의 병목

GIL은 Python의 스레드 기반 병렬 처리에 영향을 미치는 중요한 개념입니다.

### 📚 **GIL의 핵심 개념**

#### 🎯 **GIL이란?**
- **정의**: Python 인터프리터가 스레드 안전하지 않기 때문에 도입된 메커니즘
- **목적**: 중요한 내부 데이터 구조에 대한 동시 접근 방지
- **제약**: 한 번에 하나의 스레드만 Python 코드를 실행할 수 있음

#### 🔧 **GIL의 동작 원리**
- **보호**: 중요한 내부 데이터 구조에 대한 접근을 GIL로 보호
- **제한**: 여러 스레드가 동시에 Python 코드를 실행할 수 없음
- **영향**: CPU 집약적 작업에서 스레드의 진정한 병렬성 제한

### 🚫 **GIL 제거의 어려움**

#### 📋 **기술적 도전**
- **C API 호환성**: 확장 모듈을 위한 C API 유지의 어려움
- **성능 영향**: GIL 제거 시 단일 스레드 성능 저하 가능성
- **복잡성**: 메모리 관리와 가비지 컬렉션의 복잡성 증가

### 🔄 **GIL 우회 방법**

#### 🛠️ **Multiprocessing 활용**
- **해결책**: 별도의 프로세스를 사용하여 GIL 우회
- **장점**: 각 프로세스가 독립적인 인터프리터 데이터 구조 보유
- **비용**: 작업, 인수, 결과의 직렬화 필요

### 💡 **GIL의 영향**

#### 🧵 **스레드 사용 시**
- **I/O 작업**: GIL이 자주 해제되어 병렬 처리 효과적
- **CPU 작업**: GIL로 인해 진정한 병렬 처리 제한

#### 🔄 **프로세스 사용 시**
- **완전한 병렬성**: GIL 제약 없이 진정한 병렬 처리
- **오버헤드**: 프로세스 생성 및 데이터 직렬화 비용

## 📚 텍스트 파일 병렬 다운로드

실제 데이터를 사용하여 병렬 처리의 효과를 확인해보겠습니다. 유명한 문학 작품들을 다운로드하여 처리해보겠습니다.

### 📖 **다운로드할 텍스트 파일들**

#### 📋 **문학 작품 목록**
- **Victor Hugo**: http://www.gutenberg.org/files/135/135-0.txt
- **Marcel Proust**: http://www.gutenberg.org/files/7178/7178-8.txt
- **Emile Zola**: http://www.gutenberg.org/files/1069/1069-0.txt
- **Stendhal**: http://www.gutenberg.org/files/44747/44747-0.txt

### 📁 **디렉토리 준비**

```python
%mkdir books
```

### ⏱️ **순차 다운로드 (비교용)**

```python
%%time
import urllib.request as url

# 다운로드 소스 설정
source = "https://mmassd.github.io/"

# 순차적으로 파일 다운로드
url.urlretrieve(source+"books/hugo.txt",     filename="books/hugo.txt")
url.urlretrieve(source+"books/proust.txt",   filename="books/proust.txt")
url.urlretrieve(source+"books/zola.txt",     filename="books/zola.txt")
url.urlretrieve(source+"books/stendhal.txt", filename="books/stendhal.txt")
```

### 💡 **순차 다운로드의 특징**
- **단순성**: 하나씩 순서대로 다운로드
- **안정성**: 네트워크 오류 시 쉽게 디버깅
- **비효율성**: 전체 시간이 각 파일 다운로드 시간의 합
- **리소스 미활용**: 네트워크 대역폭과 CPU를 충분히 활용하지 못함

## 🎯 연습문제 4.1: ThreadPoolExecutor를 사용한 병렬 다운로드

**목표**: `ThreadPoolExecutor`를 사용하여 위의 순차 다운로드 코드를 병렬화하기

### 📋 **요구사항**
- `ThreadPoolExecutor` 활용
- 동일한 파일들을 병렬로 다운로드
- 성능 비교를 위한 시간 측정

### 💡 **구현 힌트**
```python
from concurrent.futures import ThreadPoolExecutor
import urllib.request as url

def download_file(url_info):
    """
    단일 파일 다운로드 함수
    """
    url, filename = url_info
    url.urlretrieve(url, filename)
    return filename

# 다운로드할 파일 정보
files_to_download = [
    (source+"books/hugo.txt", "books/hugo.txt"),
    (source+"books/proust.txt", "books/proust.txt"),
    (source+"books/zola.txt", "books/zola.txt"),
    (source+"books/stendhal.txt", "books/stendhal.txt")
]

# ThreadPoolExecutor를 사용한 병렬 다운로드
with ThreadPoolExecutor() as executor:
    results = list(executor.map(download_file, files_to_download))
```

### 🎯 **예상 효과**
- **성능 향상**: 네트워크 I/O의 병렬 처리로 전체 시간 단축
- **리소스 활용**: 네트워크 대역폭과 CPU의 효율적 사용
- **확장성**: 더 많은 파일을 처리할 때 더 큰 성능 향상

## 📝 WordCount - 단일 코어 처리

아래 함수들(mapper, partitioner, reducer)은 단일 코어에서 실행되는 기본적인 WordCount 구현입니다.

### 📋 **핵심 함수들**

#### 🗺️ **Mapper 함수**
```python
def mapper(filename):
    """
    텍스트 파일을 읽어서 (단어, 1) 쌍의 리스트로 변환
    """
    with open(filename) as f:
        data = f.read()
        
    # 텍스트 정리: 공백 제거, 구두점 제거, 소문자 변환, 단어 분리
    data = data.strip().replace(".","").lower().split()
        
    return sorted([(w, 1) for w in data])
```

#### 🔄 **Partitioner 함수**
```python
def partitioner(mapped_values):
    """
    mapper 결과를 받아서 (단어, [1,1,1]) 형태로 그룹화
    """
    res = defaultdict(list)
    for w, c in mapped_values:
        res[w].append(c)
        
    return res.items()
```

#### 🔢 **Reducer 함수**
```python
def reducer(item):
    """
    partitioner 결과를 받아서 단어의 총 출현 횟수 계산
    """
    w, v = item
    return (w, len(v))
```

### 📚 **필요한 라이브러리**
```python
from glob import glob
from collections import defaultdict
from operator import itemgetter
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
```

### 💡 **단일 코어 처리의 특징**
- **단순성**: 복잡한 동기화 없이 순차 처리
- **안정성**: 데이터 경쟁 상태나 데드락 위험 없음
- **제한성**: CPU 코어를 하나만 활용하여 성능 제한
- **확장성 부족**: 대용량 데이터 처리 시 시간이 오래 걸림

## 🗺️ 병렬 Map 처리

이제 단일 코어 WordCount를 병렬 처리로 개선해보겠습니다.

### 🔍 **프로세스 이름 확인**

#### 📊 **현재 프로세스 정보 출력**
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_name(n):
    """
    현재 프로세스 이름을 출력하는 함수
    """
    print(f"{mp.current_process().name} ")

# ProcessPoolExecutor를 사용하여 프로세스 이름 확인
with ProcessPoolExecutor() as e:
    _ = e.map(process_name, range(mp.cpu_count()))
```

### 🎯 **연습문제 4.2: Mapper 함수 개선**

**목표**: `mapper` 함수에 프로세스 이름 출력 기능을 추가하여 병렬 처리 과정을 모니터링할 수 있도록 개선

### 📋 **요구사항**
- `mapper` 함수에 프로세스 이름 출력 추가
- 병렬 처리 시 어떤 프로세스가 어떤 파일을 처리하는지 확인
- 디버깅과 모니터링을 위한 정보 제공

### 💡 **구현 힌트**
```python
def mapper(filename):
    """
    개선된 mapper 함수 - 프로세스 정보 포함
    """
    process_name = mp.current_process().name
    print(f"프로세스 {process_name}이 파일 {filename} 처리 중...")
    
    with open(filename) as f:
        data = f.read()
        
    data = data.strip().replace(".","").lower().split()
    result = sorted([(w, 1) for w in data])
    
    print(f"프로세스 {process_name}이 파일 {filename} 처리 완료")
    return result
```

### 🎯 **개선 효과**
- **모니터링**: 병렬 처리 과정을 실시간으로 관찰
- **디버깅**: 문제 발생 시 어떤 프로세스에서 오류가 발생했는지 파악
- **성능 분석**: 각 프로세스의 작업 분담과 처리 시간 확인

## 🔄 병렬 Reduce 처리

병렬 reduce 연산을 위해서는 데이터가 컨테이너 내에서 정렬되어 있어야 합니다. 키 값 등으로 정렬된 데이터를 `partitioner` 함수가 컨테이너로 반환합니다.

### 📚 **병렬 Reduce의 핵심 원리**

#### 🎯 **데이터 정렬의 중요성**
- **키 기반 정렬**: 같은 키를 가진 데이터들이 연속으로 배치
- **효율적 그룹화**: 정렬된 데이터로 빠른 그룹화 가능
- **병렬 처리**: 각 그룹을 독립적으로 처리 가능

#### 🔧 **Partitioner의 역할**
- **그룹화**: 같은 키를 가진 값들을 리스트로 수집
- **컨테이너 반환**: `(키, [값들])` 형태의 컨테이너 생성
- **병렬 최적화**: 각 그룹을 독립적으로 처리할 수 있도록 준비

### 🎯 **연습문제 4.3: 완전한 병렬 WordCount**

**목표**: `ThreadPoolExecutor`를 사용하여 세 개의 함수(mapper, partitioner, reducer)를 모두 병렬화한 완전한 병렬 프로그램 작성

### 📋 **요구사항**
- `ThreadPoolExecutor` 활용
- 모든 "books/*.txt" 파일을 병렬로 읽기
- Map과 Reduce 단계를 모두 병렬로 수행
- 전체 WordCount 파이프라인의 병렬화

### 💡 **구현 방향**
```python
from concurrent.futures import ThreadPoolExecutor
import glob

def parallel_wordcount():
    """
    완전한 병렬 WordCount 구현
    """
    # 1. 모든 텍스트 파일 찾기
    files = glob.glob("books/*.txt")
    
    # 2. 병렬로 mapper 실행
    with ThreadPoolExecutor() as executor:
        mapped_results = list(executor.map(mapper, files))
    
    # 3. 모든 결과를 하나로 합치기
    all_mapped = []
    for result in mapped_results:
        all_mapped.extend(result)
    
    # 4. 정렬 및 partitioner 적용
    sorted_data = sorted(all_mapped)
    partitioned = partitioner(sorted_data)
    
    # 5. 병렬로 reducer 실행
    with ThreadPoolExecutor() as executor:
        final_results = list(executor.map(reducer, partitioned))
    
    return final_results
```

### 🎯 **예상 성능 향상**
- **Map 단계**: 파일 수만큼 병렬 처리로 시간 단축
- **Reduce 단계**: 그룹 수만큼 병렬 처리로 시간 단축
- **전체 성능**: CPU 코어 수에 비례한 성능 향상

## 🕷️ 병렬 웹 크롤링

실제 데이터 수집 작업에서 병렬 처리를 적용해보겠습니다. 웹사이트에서 특정 정보를 지속적으로 수집하는 크롤링 작업을 병렬화합니다.

### 🎯 **웹 크롤링의 병렬화 필요성**

#### 📋 **크롤링 작업의 특징**
- **I/O 집약적**: 네트워크 요청과 응답이 주요 시간 소요
- **독립성**: 각 웹페이지는 독립적으로 처리 가능
- **대기 시간**: 네트워크 지연으로 인한 긴 대기 시간
- **확장성**: 수백 개의 페이지를 순차 처리하면 매우 오래 걸림

#### 🔧 **기존 방법의 한계**
- **BeautifulSoup**: 기본적으로 병렬 처리 지원하지 않음
- **순차 처리**: 하나씩 페이지를 방문하여 시간 낭비
- **리소스 미활용**: 네트워크 대역폭과 CPU를 충분히 활용하지 못함

### ⚠️ **주의사항**
- **프록시 문제**: 일반 PC에서는 프록시로 인해 정상 동작하지 않을 수 있음
- **속도 제한**: 웹사이트의 요청 속도 제한 고려 필요
- **에티켓**: 웹사이트에 과부하를 주지 않도록 적절한 지연 시간 설정

### 📊 **1단계: 데이터 수집**

#### 🌐 **데이터 소스**
- **The Latin Library**: http://www.thelatinlibrary.com/
- **특징**: 무료로 접근 가능한 라틴 텍스트의 대규모 데이터베이스
- **용도**: 고전 문학 텍스트 분석을 위한 데이터 수집

#### 🔧 **기본 크롤링 구현**

```python
from bs4 import BeautifulSoup  # 웹 스크래핑 라이브러리
from urllib.request import *

# 기본 URL 설정
base_url = "http://www.thelatinlibrary.com/"
home_content = urlopen(base_url)

# BeautifulSoup으로 HTML 파싱
soup = BeautifulSoup(home_content, "lxml")
author_page_links = soup.find_all("a")

# 저자 페이지 링크 추출 (처음 49개)
author_pages = [ap["href"] for i, ap in enumerate(author_page_links) if i < 49]
```

#### 📋 **수집된 링크 확인**
```python
# 처음 5개 링크 확인
print("수집된 링크:", author_pages[:5])
```

### 💡 **순차 크롤링의 특징**
- **단순성**: 하나씩 페이지를 방문하여 안정적
- **안전성**: 네트워크 오류 시 쉽게 디버깅
- **비효율성**: 각 페이지마다 네트워크 대기 시간 발생
- **확장성 부족**: 많은 페이지 처리 시 시간이 오래 걸림

### 🔗 **2단계: HTML 링크 생성**

#### 📋 **링크 수집 전략**
- **목표**: 라틴 텍스트를 가리키는 모든 링크의 리스트 생성
- **구조화된 포맷**: Latin Library는 저자 이름을 통해 링크를 구성
- **체계적 접근**: 저자 페이지를 통해 개별 텍스트 링크 수집

#### 🔧 **순차적 링크 수집**

```python
# 저자 페이지 내용 수집
ap_content = list()
for ap in author_pages:
    ap_content.append(urlopen(base_url + ap))

# 각 저자 페이지에서 관련 링크 추출
book_links = list()
for path, content in zip(author_pages, ap_content):
    author_name = path.split(".")[0]  # 저자 이름 추출
    ap_soup = BeautifulSoup(content, "lxml")
    # 저자 이름이 포함된 링크만 필터링
    book_links += ([link for link in ap_soup.find_all("a", {"href": True}) 
                   if author_name in link["href"]])
```

#### 📊 **수집된 링크 확인**
```python
# 처음 5개 링크 확인
print("수집된 링크:", book_links[:5])
```

### 💡 **순차 링크 수집의 특징**
- **정확성**: 저자 이름을 기반으로 한 정확한 링크 필터링
- **구조화**: 저자별로 체계적으로 링크를 수집
- **비효율성**: 각 페이지를 순차적으로 방문하여 시간 소요
- **확장성 부족**: 많은 저자 페이지 처리 시 시간이 오래 걸림

~~~

### 📥 **3단계: 웹페이지 내용 다운로드**

#### 📋 **다운로드 전략**
- **대상**: 일부(100개) 웹페이지를 선택하여 다운로드
- **파일 형식**: `book-{03d}.dat` 형식으로 저장
- **에러 처리**: HTTP 오류 발생 시 건너뛰고 계속 진행

#### 🔧 **순차 다운로드 구현**

```python
from urllib.error import HTTPError

num_pages = 100

for i, bl in enumerate(book_links[:num_pages]):
    print("Getting content " + str(i + 1) + " of " + str(num_pages), end="\r", flush=True)
    try:
        content = urlopen(base_url + bl["href"]).read()
        with open(f"book-{i:03d}.dat","wb") as f:
            f.write(content)
    except HTTPError as err:
        print("Unable to retrieve " + bl["href"] + ".")
        continue
```

### 💡 **순차 다운로드의 특징**
- **진행 상황 표시**: 실시간으로 다운로드 진행률 표시
- **에러 처리**: 개별 페이지 오류가 전체 작업을 중단시키지 않음
- **파일 관리**: 체계적인 파일명으로 데이터 저장
- **비효율성**: 각 페이지를 순차적으로 다운로드하여 시간 소요

### 📖 **4단계: 데이터 파일 읽기**

#### 📋 **파일 읽기 전략**
- **파일 패턴**: `glob`를 사용하여 `book*.dat` 패턴의 파일들을 찾기
- **텍스트 수집**: 각 파일의 내용을 리스트에 저장
- **바이너리 읽기**: 바이너리 모드로 파일을 읽어서 인코딩 문제 방지

#### 🔧 **순차 파일 읽기 구현**

```python
from glob import glob

# book*.dat 패턴의 모든 파일 찾기
files = glob('book*.dat')

# 각 파일의 내용을 리스트에 저장
texts = list()
for file in files:
    with open(file,'rb') as f:
        text = f.read()
        texts.append(text)
```

### 💡 **순차 파일 읽기의 특징**
- **단순성**: 하나씩 파일을 읽어서 안정적
- **메모리 효율성**: 각 파일을 개별적으로 처리
- **유연성**: 나중에 각 텍스트를 개별적으로 처리 가능
- **비효율성**: 각 파일을 순차적으로 읽어서 시간 소요
    texts.append(text)

~~~

### 📝 **5단계: HTML에서 텍스트 추출 및 문장 분할**

#### 📋 **텍스트 처리 전략**
- **HTML 파싱**: HTML로부터 순수 텍스트를 추출
- **문장 분할**: 마침표를 사용하여 문장 단위로 분할
- **진행 상황 표시**: 각 문서 처리 진행률을 실시간으로 표시

#### 🔧 **순차 텍스트 처리 구현**

```python
%%time
from bs4 import BeautifulSoup

sentences = list()

for i, text in enumerate(texts):
    print("Document " + str(i + 1) + " of " + str(len(texts)), end="\r", flush=True)
```

### 💡 **순차 텍스트 처리의 특징**
- **진행 상황 표시**: 실시간으로 문서 처리 진행률 표시
- **정확성**: HTML 구조를 정확히 파싱하여 순수 텍스트 추출
- **단순성**: 하나씩 텍스트를 처리하여 안정적
- **비효율성**: 각 텍스트를 순차적으로 처리하여 시간 소요
    # HTML 파싱 및 텍스트 추출
    textSoup = BeautifulSoup(text, "lxml")
    paragraphs = textSoup.find_all("p", attrs={"class":None})
    prepared = ("".join([p.text.strip().lower() for p in paragraphs[1:-1]]))
    
    # 문장 분할 및 정리
    for t in prepared.split("."):
        part = "".join([c for c in t if c.isalpha() or c.isspace()])
        sentences.append(part.strip())

# 결과 확인
print("첫 번째 문장:", sentences[0])
print("마지막 문장:", sentences[-1])
```

### 💡 **완전한 텍스트 처리 파이프라인**
- **HTML 파싱**: BeautifulSoup으로 HTML 구조 분석
- **텍스트 추출**: 클래스가 없는 p 태그에서 순수 텍스트 추출
- **전처리**: 소문자 변환, 공백 정리
- **문장 분할**: 마침표 기준으로 문장 단위 분할
- **정리**: 알파벳과 공백만 남기고 특수문자 제거

## 🎯 **연습문제 4.4: 병렬 텍스트 처리**

**목표**: `concurrent.futures`를 사용하여 마지막 텍스트 처리 함수를 병렬 처리 가능하도록 재작성

### 📋 **요구사항**
- `concurrent.futures` 활용
- 텍스트 처리 작업의 병렬화
- 성능 향상 확인

## 🎯 **연습문제 4.5: 전체 파이프라인 병렬화**

**목표**: 언급한 함수들 중 병렬 처리(map/reduce)가 가능한 부분을 `concurrent.futures`를 사용하여 병렬화

### 📋 **병렬화 가능한 부분**
- **링크 수집**: 저자 페이지에서 링크 추출
- **웹페이지 다운로드**: HTML 내용 다운로드
- **파일 읽기**: 데이터 파일 읽기
- **텍스트 처리**: HTML 파싱 및 문장 분할

### 💡 **병렬화 전략**
- **Map 단계**: 각 작업을 독립적으로 병렬 처리
- **Reduce 단계**: 결과를 수집하고 통합
- **성능 최적화**: I/O 집약적 작업에 특히 효과적

## 📚 참고 자료

### 🔗 **추가 학습 자료**
- [Using Conditional Random Fields and Python for Latin word segmentation](https://medium.com/@felixmohr/using-python-and-conditional-random-fields-for-latin-word-segmentation-416ca7a9e513)
  - 라틴어 단어 분할을 위한 조건부 랜덤 필드와 Python 활용법

### 💡 **학습 요약**
이 실습을 통해 다음과 같은 핵심 개념을 학습했습니다:

#### 🔧 **병렬 처리 기초**
- **Thread vs Process**: 각각의 특징과 사용 시기
- **GIL의 영향**: Python에서 병렬 처리의 제약과 해결책
- **Futures**: 고수준 병렬 처리 인터페이스 활용

#### 🗺️ **MapReduce 패턴**
- **Map 단계**: 데이터 변환과 분할
- **Reduce 단계**: 결과 집계와 통합
- **병렬화**: 각 단계의 독립적 병렬 처리

#### 🕷️ **실제 응용**
- **웹 크롤링**: I/O 집약적 작업의 병렬화
- **텍스트 처리**: 대용량 텍스트 데이터의 효율적 처리
- **성능 최적화**: 순차 처리와 병렬 처리의 성능 비교

### 🎯 **다음 단계**
- **PySpark**: 대규모 분산 처리 프레임워크 학습
- **Spark DataFrames**: 구조화된 데이터 처리
- **Pandas**: 데이터 분석을 위한 고급 기능
