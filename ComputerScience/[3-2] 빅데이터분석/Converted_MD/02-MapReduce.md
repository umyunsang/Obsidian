# 02. MapReduce - 분산 데이터 처리의 핵심

## 📚 개요

MapReduce는 대용량 데이터를 분산 환경에서 효율적으로 처리하기 위한 프로그래밍 모델입니다. 이 실습에서는 Python을 사용하여 MapReduce 패턴을 구현하고, 빅데이터 처리의 핵심 개념을 이해합니다.

### 🎯 학습 목표
- MapReduce 프로그래밍 모델의 이해
- Python을 활용한 MapReduce 구현
- 분산 데이터 처리의 기본 원리 파악
- WordCount 예제를 통한 실습

### 📖 MapReduce란?
- **정의**: 대용량 데이터를 여러 노드에서 병렬로 처리하는 프로그래밍 모델
- **핵심**: Map(매핑)과 Reduce(리듀싱) 두 단계로 데이터 처리
- **장점**: 확장성, 내결함성, 단순성
- **참고**: [Java 버전 MapReduce 튜토리얼](https://www.dezyre.com/hadoop-tutorial/hadoop-mapreduce-wordcount-tutorial)

### 🏗️ 도메인 분해 (Domain Decomposition)

![도메인 분해](images/domain_decomp.png)

*출처: https://computing.llnl.gov/tutorials/parallel_comp*

MapReduce는 복잡한 문제를 작은 단위로 분해하여 병렬 처리하는 도메인 분해 패턴을 활용합니다.

## 🗺️ Map 함수 이해하기

MapReduce의 핵심인 `map` 함수는 시퀀스의 모든 요소에 함수를 적용하는 Python의 내장 함수입니다.

### 📚 기본 개념
- **기능**: `map(func, seq)` - 시퀀스의 모든 요소에 함수를 적용
- **반환값**: 변경된 요소들로 구성된 새로운 이터레이터
- **특징**: 지연 평가(lazy evaluation) - 실제로 필요할 때까지 계산하지 않음

### 🧮 기본 예제: 제곱 계산
```python
def f(x):
    return x * x

# 데이터 정의
rdd = [2, 6, -3, 7]

# map 함수 적용
res = map(f, rdd)
print("이터레이터 객체:", res)  # 이터레이터 객체 출력

# 결과 확인
print("실제 결과:", list(res))
# 출력: [4, 36, 9, 49]
```

### 🔢 고급 예제: 두 리스트의 요소별 곱셈
```python
from operator import mul

# 두 개의 데이터 리스트
rdd1, rdd2 = [2, 6, -3, 7], [1, -4, 5, 3]

# 요소별 곱셈 수행
res = map(mul, rdd1, rdd2)
print("곱셈 결과:", list(res))
# 출력: [2, -24, -15, 21]
```

### 💡 Map 함수의 특징
- **함수형 프로그래밍**: 함수를 다른 함수의 인수로 전달
- **불변성**: 원본 데이터를 변경하지 않고 새로운 결과 생성
- **재사용성**: 동일한 함수를 다양한 데이터에 적용 가능

![MapReduce](images/mapreduce.jpg)

## 🔄 Reduce 함수 이해하기

`functools.reduce`는 시퀀스의 요소들을 순차적으로 결합하여 단일 값을 반환하는 함수입니다. MapReduce의 Reduce 단계를 구현하는 핵심 함수입니다.

### 📚 기본 개념
- **기능**: `reduce(func, seq)` - 시퀀스의 요소들을 순차적으로 결합
- **반환값**: 단일 값 (시퀀스의 모든 요소를 결합한 결과)
- **동작 방식**: `f(f(f(f(1,2),3),4),5)` 형태로 순차적 적용

### 🧮 기본 예제: 숫자 합계 계산
```python
from functools import reduce
from operator import add

# 데이터 정의
rdd = list(range(1, 6))  # [1, 2, 3, 4, 5]

# reduce 함수로 합계 계산
result = reduce(add, rdd)
print("합계:", result)  # 출력: 15

# 계산 과정: ((((1+2)+3)+4)+5) = 15
```

### 💡 Reduce 함수의 동작 원리
1. **첫 번째 단계**: `add(1, 2)` = 3
2. **두 번째 단계**: `add(3, 3)` = 6  
3. **세 번째 단계**: `add(6, 4)` = 10
4. **네 번째 단계**: `add(10, 5)` = 15

### 🎯 MapReduce에서의 역할
- **Map 단계**: 데이터를 키-값 쌍으로 변환
- **Reduce 단계**: 같은 키를 가진 값들을 집계하여 최종 결과 생성

## 📊 가중 평균과 분산 계산

MapReduce 패턴을 활용하여 통계적 계산을 수행하는 방법을 학습합니다. 이는 빅데이터 분석에서 매우 중요한 개념입니다.

### 📚 수학적 배경

이산 확률변수 $X$가 확률질량함수 $x_1 \mapsto p_1, x_2 \mapsto p_2, \ldots, x_n \mapsto p_n$을 가질 때:

**분산 공식:**
$$\operatorname{Var}(X) = \left(\sum_{i=1}^n p_i x_i ^2\right) - \mu^2$$

**평균 공식:**
$$\mu = \sum_{i=1}^n p_i x_i$$

### 🎯 실습 데이터
```python
# 확률변수 값들
X = [5, 1, 2, 3, 1, 2, 5, 4]

# 각 값에 대응하는 확률
P = [0.05, 0.05, 0.15, 0.05, 0.15, 0.2, 0.1, 0.25]
```

### 💡 MapReduce 적용
- **Map 단계**: 각 데이터 포인트에 대해 필요한 계산 수행
- **Reduce 단계**: 모든 결과를 집계하여 최종 통계값 계산

## 🎯 연습문제 2.1: for 루프를 사용한 통계 계산

**목표**: for 루프를 사용하여 가중 평균과 분산을 계산하는 함수 작성

### 📋 요구사항
- 가중 평균 계산 함수 작성
- 가중 분산 계산 함수 작성
- for 루프를 사용한 구현

### 💡 구현 힌트
```python
def weighted_mean_for_loop(X, P):
    """
    for 루프를 사용한 가중 평균 계산
    """
    # 구현 코드
    pass

def weighted_variance_for_loop(X, P):
    """
    for 루프를 사용한 가중 분산 계산
    """
    # 구현 코드
    pass
```

## 🎯 연습문제 2.2: Map-Reduce를 사용한 통계 계산

**목표**: `map`과 `reduce` 함수를 사용하여 가중 평균과 분산을 계산하는 함수 작성

### 📋 요구사항
- `map`과 `reduce` 함수 활용
- 함수형 프로그래밍 스타일로 구현
- 이전 연습문제와 동일한 결과 확인

### 💡 구현 힌트
```python
def weighted_mean_map_reduce(X, P):
    """
    map-reduce를 사용한 가중 평균 계산
    """
    # map과 reduce 활용
    pass

def weighted_variance_map_reduce(X, P):
    """
    map-reduce를 사용한 가중 분산 계산
    """
    # map과 reduce 활용
    pass
```

### ⚠️ 중요 참고사항
> **주의**: 위 연습문제들은 MapReduce 과정을 이해하기 위한 교육용입니다.  
> 실제 Python에서 분산을 계산할 때는 [NumPy](http://www.numpy.org)를 사용하는 것이 좋습니다.

## 📝 WordCount를 MapReduce로 구현하기

이제 앞서 학습한 WordCount 애플리케이션을 MapReduce 패턴으로 재구현해보겠습니다.

### 🎯 MapReduce WordCount 과정

#### 1️⃣ Map 단계 (매핑)
- **입력**: 텍스트 파일
- **처리**: 텍스트를 단어로 분해
- **출력**: 각 단어에 대해 `(단어, 1)` 쌍 생성

#### 2️⃣ Reduce 단계 (리듀싱)  
- **입력**: Map 단계의 결과
- **처리**: 같은 단어의 개수들을 합산
- **출력**: `(단어, 총개수)` 쌍 생성

### 🔧 구현 전략
기존의 통합된 `wordcount` 함수를 Map과 Reduce 단계로 분리하여 구현합니다.

### 📚 참고 자료
- [Hadoop MapReduce 튜토리얼](https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0)
- Java로 작성된 공식 예제를 Python으로 구현

## 🗺️ Map 단계: 파일 읽기 및 키-값 쌍 생성

Map 단계에서는 텍스트 파일을 읽어서 각 단어에 대해 `(단어, 1)` 형태의 키-값 쌍을 생성합니다.

### 🎯 연습문제 2.3: Mapper 함수 구현

**목표**: 파일명을 입력으로 받아서 정렬된 `(단어, 1)` 튜플 시퀀스를 반환하는 `mapper` 함수 작성

### 📋 요구사항
- 함수명: `mapper`
- 입력: 파일명 (문자열)
- 출력: 정렬된 `(단어, 1)` 튜플 리스트

### 🧪 예상 결과
```python
result = mapper('sample.txt')
print(result[:10])  # 처음 10개 결과
# 예상 출력: [('adipisci', 1), ('adipisci', 1), ('adipisci', 1), 
#            ('adipisci', 1), ('adipisci', 1), ('adipisci', 1), 
#            ('adipisci', 1), ('aliquam', 1), ('aliquam', 1), 
#            ('aliquam', 1)]
```

### 💡 구현 힌트
1. 파일을 열고 내용을 읽기
2. 텍스트를 단어로 분리
3. 각 단어에 대해 `(단어, 1)` 튜플 생성
4. 결과를 정렬하여 반환

### 🔧 함수 시그니처
```python
def mapper(filename):
    """
    파일에서 단어를 추출하여 (단어, 1) 튜플 리스트 반환
    
    Args:
        filename (str): 분석할 파일명
        
    Returns:
        list: 정렬된 (단어, 1) 튜플 리스트
    """
    # 구현 코드
    pass
```

## 🔄 Partition 단계: 데이터 그룹화

Partition 단계에서는 Map 단계의 결과를 같은 키(단어)를 가진 값들을 그룹화합니다. 이는 Reduce 단계에서 효율적으로 처리하기 위한 전처리 과정입니다.

### 🎯 연습문제 2.4: Partitioner 함수 구현

**목표**: `mapper` 함수의 결과를 받아서 같은 단어의 값들을 리스트로 그룹화하는 `partitioner` 함수 작성

### 📋 요구사항
- 함수명: `partitioner`
- 입력: `mapper` 함수의 결과 (정렬된 `(단어, 1)` 튜플 리스트)
- 출력: `(단어, [1, 1, 1, ...])` 형태의 그룹화된 리스트

### 🧪 예상 결과
```python
mapped_data = mapper('sample.txt')
result = partitioner(mapped_data)
print(result[:3])  # 처음 3개 결과
# 예상 출력: [('adipisci', [1, 1, 1, 1, 1, 1, 1]), 
#            ('aliquam', [1, 1, 1, 1, 1, 1, 1]), 
#            ('amet', [1, 1, 1, 1])]
```

### 💡 구현 힌트
1. `mapper` 함수의 결과를 받기
2. 같은 단어를 가진 튜플들을 그룹화
3. 각 그룹의 값들(1들)을 리스트로 수집
4. `(단어, [값들])` 형태로 반환

### 🔧 함수 시그니처
```python
def partitioner(mapped_data):
    """
    Map 결과를 그룹화하여 (단어, [값들]) 형태로 변환
    
    Args:
        mapped_data (list): mapper 함수의 결과
        
    Returns:
        list: 그룹화된 (단어, [값들]) 리스트
    """
    # 구현 코드
    pass
```

### 🎯 Partition의 역할
- **데이터 정리**: 같은 키를 가진 데이터들을 모음
- **효율성 향상**: Reduce 단계에서 빠른 처리 가능
- **메모리 최적화**: 관련 데이터들을 함께 처리

## 🔄 Reduce 단계: 개수 합산 및 최종 결과 생성

Reduce 단계에서는 Partition 단계의 결과를 받아서 각 단어의 총 출현 횟수를 계산하고 최종 결과를 생성합니다.

### 🎯 연습문제 2.5: Reducer 함수 구현

**목표**: `(단어, [1,1,1,...])` 튜플을 받아서 단어의 총 출현 횟수를 계산하는 `reducer` 함수 작성

### 📋 요구사항
- 함수명: `reducer`
- 입력: `(단어, [1,1,1,...])` 형태의 튜플
- 출력: `(단어, 총개수)` 형태의 튜플

### 🧪 예상 결과
```python
result = reducer(('hello', [1, 1, 1, 1, 1]))
print(result)
# 예상 출력: ('hello', 5)
```

### 💡 구현 힌트
1. 튜플에서 단어와 값 리스트 분리
2. 값 리스트의 합계 계산
3. `(단어, 총개수)` 형태로 반환

### 🔧 함수 시그니처
```python
def reducer(word_count_tuple):
    """
    (단어, [값들]) 튜플을 받아서 (단어, 총개수) 반환
    
    Args:
        word_count_tuple (tuple): (단어, [1,1,1,...]) 형태의 튜플
        
    Returns:
        tuple: (단어, 총개수) 형태의 튜플
    """
    # 구현 코드
    pass
```

### 🎯 Reduce의 역할
- **집계**: 같은 키를 가진 모든 값들을 합산
- **최종화**: MapReduce 과정의 최종 결과 생성
- **효율성**: 그룹화된 데이터를 빠르게 처리

## 📁 여러 파일 처리하기

실제 빅데이터 환경에서는 여러 파일을 동시에 처리해야 합니다. 이제 8개의 샘플 파일을 생성하고 MapReduce 패턴으로 처리해보겠습니다.

### 🎯 목표
- 8개의 샘플 파일 생성 (`sample00.txt` ~ `sample07.txt`)
- 모든 파일에 대해 MapReduce 처리
- 가장 빈번한 단어부터 정렬된 결과 생성

### 📝 파일 생성
```python
from lorem import text

# 8개의 색플 파일 생성
for i in range(8):
    with open("sample{0:02d}.txt".format(i), "w") as f:
        f.write(text())
```

### 🔍 파일 목록 확인
```python
import glob

# 생성된 파일들 확인
files = sorted(glob.glob('sample0*.txt'))
print("생성된 파일들:", files)
```

### 💡 파일 생성의 중요성
- **확장성 테스트**: 여러 파일을 처리하는 능력 검증
- **성능 측정**: 처리 시간과 메모리 사용량 분석
- **실제 환경 시뮬레이션**: 대용량 데이터 처리 환경 모방

## 🎯 연습문제 2.6: for 루프를 사용한 전체 처리

**목표**: 앞서 구현한 함수들을 사용하여 for 루프로 모든 파일을 처리하고 단어 빈도를 계산

### 📋 요구사항
- 모든 파일에 대해 `mapper` 함수 적용
- `partitioner` 함수로 데이터 그룹화
- `reducer` 함수로 최종 결과 생성
- for 루프를 사용한 구현

### 💡 구현 방향
```python
def process_all_files_with_loops():
    """
    for 루프를 사용하여 모든 파일 처리
    """
    # 1. 모든 파일에 대해 mapper 적용
    # 2. 결과를 partitioner로 그룹화
    # 3. reducer로 최종 결과 생성
    # 4. 결과 반환
    pass
```

## 🎯 연습문제 2.7: map 함수를 사용한 전체 처리

**목표**: `map` 함수를 사용하여 mapper와 reducer를 적용하는 함수형 프로그래밍 스타일로 구현

### 📋 요구사항
- `map` 함수를 사용한 구현
- 함수형 프로그래밍 스타일
- 이전 연습문제와 동일한 결과 확인

### 💡 구현 방향
```python
def process_all_files_with_map():
    """
    map 함수를 사용하여 모든 파일 처리
    """
    # 1. map 함수로 mapper 적용
    # 2. map 함수로 reducer 적용
    # 3. 결과 반환
    pass
```

### 🎯 학습 목표
- **함수형 프로그래밍**: map 함수의 활용법 이해
- **성능 비교**: for 루프 vs map 함수의 성능 차이 확인
- **코드 간소화**: 더 간결하고 읽기 쉬운 코드 작성
