# 01. Word Count - 빅데이터 분석의 첫 걸음

## 📚 개요

Word Count는 빅데이터 분석과 분산 컴퓨팅의 가장 기본적이면서도 중요한 예제입니다. 이 실습을 통해 텍스트 데이터 처리의 핵심 개념을 이해하고, Python의 다양한 라이브러리를 활용한 데이터 처리 방법을 학습합니다.

### 🎯 학습 목표
- 텍스트 파일에서 단어 빈도 계산 방법 이해
- Python의 collections 모듈 활용법 습득
- Map-Reduce 패턴의 기본 개념 이해
- 빅데이터 처리의 기본 원리 파악

## 📖 Word Count란?

- **정의**: 텍스트 파일을 읽어서 각 단어의 출현 빈도를 계산하는 프로그램
- **용도**: 번역 작업 비용 산정, 텍스트 분석, 검색 엔진 최적화 등
- **중요성**: 빅데이터 프로그래밍에서 "Hello World"와 같은 기본 예제
- **참고**: [Wikipedia - Word Count](https://en.wikipedia.org/wiki/Word_count)

## ⚠️ 학습 가이드라인

> **중요**: 다음 사항들을 반드시 준수해주세요.

- 🔍 **자기주도 학습**: 구글링이나 ChatGPT 사용을 최소화하고, 교수자나 보조연구원에게 질문하거나 Python의 `help()` 함수를 활용하세요
- 📝 **단계적 접근**: 먼저 주석이나 의사코드(pseudo-code)로 전체 구조를 설계한 후, 최적화된 코드를 작성하세요
- 🚫 **부정행위 금지**: 다른 학생으로부터 코드를 받지 마세요
- 📋 **답안 확인**: 연습문제 답안은 LMS를 통해 공개됩니다

## 📝 샘플 텍스트 파일 생성

실습을 위해 먼저 분석할 텍스트 파일을 생성해보겠습니다. `lorem` 라이브러리를 사용하여 랜덤한 텍스트를 생성합니다.

### 💡 lorem 라이브러리란?
- **용도**: 테스트용 랜덤 텍스트 생성
- **특징**: Lorem Ipsum과 같은 의미 없는 텍스트를 생성하여 데이터 처리 알고리즘을 테스트할 때 유용

```python
from lorem import text

# 2개의 문단으로 구성된 샘플 텍스트 파일 생성
with open("sample.txt", "w") as f:
    for i in range(2):
        f.write(text())
```

### 📋 생성된 파일 확인
생성된 파일의 내용과 크기를 확인해보겠습니다:

```bash
# 파일 크기와 라인 수 확인
wc sample.txt
du -h sample.txt
```

## 🎯 연습문제 1.1: 기본 통계 계산

**목표**: `sample.txt` 파일의 기본 통계 정보를 계산하는 Python 프로그램 작성

### 📊 요구사항
- 파일 내의 **라인 수** 계산
- **고유한 단어**의 개수 계산  
- **문자**의 개수 계산

### 💡 힌트
- 파일을 읽을 때는 `open()` 함수 사용
- 단어를 분리할 때는 `split()` 메서드 활용
- 고유한 단어를 찾을 때는 `set()` 자료구조 사용

### 🔍 참고: Unix 명령어로 확인
```bash
# Unix/Linux 명령어로 기본 통계 확인
wc sample.txt      # 라인, 단어, 문자 수 출력
du -h sample.txt   # 파일 크기 확인
```

### 📝 구현 방향
1. 파일을 열고 내용을 읽기
2. 라인 수 계산
3. 모든 단어를 분리하여 리스트로 만들기
4. 고유한 단어 집합 생성
5. 문자 수 계산
6. 결과 출력

## 🎯 연습문제 1.2: 단어 추출 함수

**목표**: 파일에서 모든 단어를 추출하는 `map_words` 함수 작성

### 📋 요구사항
- 함수명: `map_words`
- 입력: 파일명 (문자열)
- 출력: 파일 내 모든 단어의 리스트

### 💡 함수 설계
```python
def map_words(filename):
    """
    파일에서 모든 단어를 추출하는 함수
    
    Args:
        filename (str): 분석할 파일명
        
    Returns:
        list: 파일 내 모든 단어의 리스트
    """
    # 구현 코드 작성
    pass
```

### 🧪 테스트 예시
```python
# 함수 테스트
words = map_words("sample.txt")
print(words[:5])  # 처음 5개 단어 출력
# 예상 결과: ['adipisci', 'adipisci', 'adipisci', 'adipisci', 'adipisci']
```

### 📝 구현 힌트
1. 파일을 열고 내용을 읽기
2. 텍스트를 소문자로 변환 (선택사항)
3. 구두점 제거 (선택사항)
4. 공백을 기준으로 단어 분리
5. 단어 리스트 반환

## 🔄 딕셔너리 값으로 정렬하기

단어 빈도 분석 결과를 보기 좋게 정렬하는 방법을 학습합니다.

### 📚 기본 개념
- **기본 정렬**: `sorted()` 함수는 딕셔너리의 **키(key)**로 정렬합니다
- **값으로 정렬**: `operator.itemgetter(1)`을 사용하여 **값(value)**으로 정렬할 수 있습니다
- **참고**: [Python operator 모듈 문서](https://docs.python.org/3.6/library/operator.html)

### 💡 operator.itemgetter()란?
- **역할**: 호출 가능한 객체를 반환하여 피연산자의 `__getitem__()` 메서드를 사용해 항목을 가져옵니다
- **용도**: 정렬 키로 사용하여 복잡한 정렬 작업을 간단하게 처리

### 🍎 실습 예제: 과일 개수 정렬

```python
import operator

# 과일과 개수를 튜플 리스트로 정의
fruits = [('apple', 3), ('banana', 2), ('pear', 5), ('orange', 1)]

# 값(개수)으로 정렬하는 함수 생성
getcount = operator.itemgetter(1)

# 오름차순 정렬 (개수가 적은 순)
sorted_fruits = dict(sorted(fruits, key=getcount))
print("오름차순:", sorted_fruits)
```

### 📈 내림차순 정렬
```python
# 내림차순 정렬 (개수가 많은 순)
sorted_fruits_desc = dict(sorted(fruits, key=getcount, reverse=True))
print("내림차순:", sorted_fruits_desc)
```

### 🎯 활용 예시
단어 빈도 분석에서 가장 많이 나온 단어부터 보려면:
```python
# 단어 빈도 딕셔너리를 값으로 내림차순 정렬
word_freq = {'hello': 5, 'world': 3, 'python': 8}
sorted_by_freq = dict(sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True))
```

## 🎯 연습문제 1.3: 단어 빈도 계산 함수

**목표**: `map_words` 함수가 반환한 단어 리스트를 받아서 각 단어의 출현 빈도를 계산하는 `reduce` 함수 작성

### 📋 요구사항
- 함수명: `reduce`
- 입력: 파일명 (문자열)
- 출력: 단어를 키로, 출현 빈도를 값으로 하는 딕셔너리

### 🧪 예상 결과
```python
result = reduce('sample.txt')
print(result)
# 예상 출력: {'tempora': 2, 'non': 1, 'quisquam': 1, 'amet': 1, 'sit': 1}
```

### 💡 구현 방향
1. `map_words` 함수를 사용하여 단어 리스트 가져오기
2. 각 단어의 출현 횟수를 세기
3. 단어와 빈도를 딕셔너리로 구성하여 반환

### ⚠️ 주의사항
이 간단해 보이는 함수가 실제로는 구현하기 쉽지 않을 수 있습니다. Python 표준 라이브러리에서 제공하는 유용한 기능들을 활용해보세요!

## 📦 Python Collections 모듈 활용

Python의 `collections` 모듈은 기본 자료구조(`dict`, `list`, `set`, `tuple`)의 대안이 되는 특화된 컨테이너 자료형을 제공합니다.

### 🎯 주요 클래스
- **`defaultdict`**: 누락된 값에 대해 팩토리 함수를 호출하는 딕셔너리 서브클래스
- **`Counter`**: 해시 가능한 객체를 세는 딕셔너리 서브클래스

### 💡 왜 collections를 사용할까?
- **편의성**: 일반적인 작업을 더 쉽게 처리
- **성능**: 특화된 용도로 최적화된 구현
- **안전성**: 오류 가능성을 줄이는 안전한 기본값 제공

### 🔧 defaultdict 활용하기

단어 빈도 계산 함수를 구현할 때 딕셔너리에 키-값 쌍을 추가하는 과정에서 문제가 발생할 수 있습니다. 존재하지 않는 키의 값을 변경하려고 하면 키가 자동으로 생성되지 않기 때문입니다.

#### ❌ 일반 딕셔너리의 문제점
```python
# 일반 딕셔너리 사용 시 문제
word_count = {}
word_count['hello'] += 1  # KeyError 발생!
```

#### ✅ defaultdict의 해결책
`try-except` 구문을 사용할 수도 있지만, `defaultdict`가 더 깔끔한 해결책입니다. 이 컨테이너는 누락된 값에 대해 팩토리 함수를 호출하는 딕셔너리 서브클래스입니다.

#### 🎯 실습 예제: 색상별 값 그룹화
```python
from collections import defaultdict

# 색상과 숫자 쌍의 리스트
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]

# list를 기본 팩토리로 하는 defaultdict 생성
d = defaultdict(list)

# 각 색상별로 값들을 그룹화
for k, v in s:
    d[k].append(v)

# 결과 확인
print(dict(d))
# 출력: {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]}
```

#### 💡 defaultdict의 장점
- **자동 키 생성**: 존재하지 않는 키에 접근해도 자동으로 기본값 생성
- **코드 간소화**: try-except 구문 없이도 안전한 딕셔너리 조작
- **유연성**: 다양한 팩토리 함수 사용 가능 (list, int, set 등)

## 🎯 연습문제 1.4: defaultdict로 reduce 함수 개선

**목표**: 앞서 작성한 `reduce` 함수를 `defaultdict`를 사용하여 개선하기

### 📋 요구사항
- 이전에 작성한 `reduce` 함수를 수정
- `defaultdict`를 사용하여 구현
- 가장 적합한 팩토리 함수 선택

### 💡 힌트
- 단어 빈도를 세는 것이므로 **정수(int)**를 기본값으로 하는 팩토리가 적합합니다
- `defaultdict(int)`를 사용하면 존재하지 않는 키에 대해 자동으로 0을 반환합니다

### 🔧 구현 방향
```python
from collections import defaultdict

def reduce(filename):
    """
    defaultdict를 사용한 개선된 단어 빈도 계산 함수
    """
    # defaultdict(int) 사용
    # 단어 리스트 가져오기
    # 각 단어의 빈도 계산
    # 결과 딕셔너리 반환
    pass
```

### 🎯 예상 개선 효과
- **코드 간소화**: try-except 구문 제거
- **안전성 향상**: KeyError 방지
- **가독성 개선**: 더 직관적인 코드 작성

### 🔢 Counter 클래스 활용하기

`Counter`는 해시 가능한 객체를 세는 딕셔너리 서브클래스입니다. 요소들이 딕셔너리 키로 저장되고, 그 개수가 딕셔너리 값으로 저장되는 정렬되지 않은 컬렉션입니다.

#### 🎯 Counter의 특징
- **유연한 카운트**: 0이나 음수 값도 허용
- **다양한 초기화**: 이터러블이나 매핑 객체로부터 초기화 가능
- **전용 메서드**: `most_common()`, `elements()` 등 유용한 메서드 제공

#### 🎨 실습 예제: 색상 빈도 분석
```python
from collections import Counter

# 색상 딕셔너리 정의
violet = dict(r=23, g=13, b=23)
print("원본 딕셔너리:", violet)

# Counter 객체 생성
cnt = Counter(violet)
print("Counter 객체:", cnt)

# 특정 색상의 개수 확인
print("빨간색 개수:", cnt['r'])
print("파란색 개수:", cnt['b'])
print("초록색 개수:", cnt['g'])
```

#### 🔍 Counter의 유용한 메서드들

**1. elements() - 개수만큼 요소 반복**
```python
# 각 요소를 개수만큼 반복하여 출력
print("요소들:", list(cnt.elements()))
```

**2. most_common() - 가장 빈번한 요소**
```python
# 가장 빈번한 2개 요소 출력
print("가장 빈번한 요소:", cnt.most_common(2))
```

**3. values() - 모든 카운트 값**
```python
# 모든 카운트 값들
print("카운트 값들:", list(cnt.values()))
```

#### 💡 Counter의 장점
- **전문성**: 카운팅에 특화된 다양한 메서드 제공
- **편의성**: 복잡한 카운팅 로직을 간단하게 처리
- **성능**: 카운팅 작업에 최적화된 구현

## 🎯 연습문제 1.5: Counter로 단어 빈도 계산

**목표**: `Counter` 객체를 사용하여 샘플 텍스트 파일의 단어 출현 빈도를 계산하기

### 📋 요구사항
- `Counter` 객체 활용
- 샘플 텍스트 파일의 단어 빈도 계산
- 결과 확인 및 분석

### 💡 구현 힌트
```python
from collections import Counter

def word_count_with_counter(filename):
    """
    Counter를 사용한 단어 빈도 계산
    """
    # 파일에서 단어 리스트 가져오기
    # Counter 객체 생성
    # 결과 반환
    pass
```

### 🔍 Counter의 추가 정보
- **Bag/Multiset 유사성**: Counter 클래스는 일부 Python 라이브러리나 다른 언어의 bag이나 multiset과 유사합니다
- **병렬 처리**: 나중에 병렬 처리 맥락에서 Counter와 유사한 객체를 사용하는 방법을 학습할 예정입니다

### 🎯 예상 결과
```python
# Counter 사용 예시
word_counter = Counter(['hello', 'world', 'hello', 'python'])
print(word_counter)
# 출력: Counter({'hello': 2, 'world': 1, 'python': 1})
```

## 📁 여러 파일 처리하기

실제 빅데이터 분석에서는 여러 파일을 동시에 처리해야 하는 경우가 많습니다. 이 섹션에서는 여러 텍스트 파일을 효율적으로 처리하는 방법을 학습합니다.

### 🎯 학습 목표
- 여러 파일에서 단어 빈도 계산
- 여러 딕셔너리 결과를 하나로 합치기
- `itertools` 모듈의 유용한 기능 활용

### 📝 단계별 접근
1. **여러 파일 생성**: `sample01.txt`, `sample02.txt` 등 여러 개의 lorem 텍스트 파일 생성
2. **개별 처리**: 각 파일을 처리하여 개별 딕셔너리 생성
3. **결과 통합**: 여러 딕셔너리의 결과를 하나로 합치기
4. **최적화**: `itertools` 모듈을 활용한 효율적인 처리

### 🔧 itertools.chain() 활용

여러 시퀀스를 하나의 시퀀스로 처리할 때 `itertools.chain()`을 사용하면 매우 유용합니다.

#### 🍎 실습 예제: 과일과 채소 데이터 통합
```python
import itertools, operator

# 과일 데이터
fruits = [('apple', 3), ('banana', 2), ('pear', 5), ('orange', 1)]

# 채소 데이터  
vegetables = [('endive', 2), ('spinach', 1), ('celery', 5), ('carrot', 4)]

# 정렬을 위한 함수
getcount = operator.itemgetter(1)

# 두 리스트를 하나로 합치고 값으로 정렬
combined = dict(sorted(itertools.chain(fruits, vegetables), key=getcount))
print("통합된 결과:", combined)
```

### 💡 itertools.chain()의 장점
- **메모리 효율성**: 실제로는 여러 시퀀스를 연결하지 않고 순차적으로 처리
- **편의성**: 여러 이터러블을 하나의 이터러블로 처리
- **성능**: 복잡한 반복문 없이 간단하게 처리

### 📚 참고 자료
- [itertools.chain() 공식 문서](https://docs.python.org/3.6/library/itertools.html#itertools.chain)

## 🎯 연습문제 1.6: 여러 파일 처리 및 통합

**목표**: 여러 파일을 생성하고 처리한 후 `itertools.chain`을 사용하여 통합된 단어 빈도 딕셔너리 생성

### 📋 요구사항
1. **파일 생성**: 여러 개의 lorem 텍스트 파일 생성
2. **개별 처리**: 각 파일의 단어 빈도 계산
3. **결과 통합**: `itertools.chain`을 사용하여 모든 결과를 하나로 합치기

### 💡 구현 단계
```python
import itertools
from collections import Counter

def process_multiple_files():
    """
    여러 파일을 처리하고 통합된 단어 빈도 딕셔너리 반환
    """
    # 1. 여러 파일 생성
    # 2. 각 파일의 단어 빈도 계산
    # 3. itertools.chain으로 결과 통합
    # 4. 최종 딕셔너리 반환
    pass
```

### 🔧 구현 힌트
- **파일 생성**: `sample01.txt`, `sample02.txt` 등 여러 파일 생성
- **개별 처리**: 각 파일에 대해 단어 빈도 계산
- **통합**: `itertools.chain()`을 사용하여 모든 결과를 하나로 합치기
- **최적화**: Counter와 chain을 함께 활용

### 🎯 예상 결과
```python
# 여러 파일의 단어 빈도를 통합한 결과
merged_result = {
    'word1': total_count1,
    'word2': total_count2,
    # ... 모든 단어의 총 빈도
}
```

## 🎯 연습문제 1.7: 범용 wordcount 함수

**목표**: 여러 파일을 인수로 받아서 통합된 단어 빈도 딕셔너리를 반환하는 `wordcount` 함수 작성

### 📋 요구사항
- 함수명: `wordcount`
- 입력: 여러 파일명 (가변 인수)
- 출력: 모든 파일을 통합한 단어 빈도 딕셔너리

### 💡 함수 시그니처
```python
def wordcount(file1, file2, file3, ...):
    """
    여러 파일의 단어 빈도를 통합하여 반환
    
    Args:
        *files: 분석할 파일명들 (가변 인수)
        
    Returns:
        dict: 통합된 단어 빈도 딕셔너리
    """
    pass
```

### 🔧 구현 힌트
- **가변 인수**: `*args`를 사용하여 여러 파일을 받기
- **참고 자료**: [Python 가변 인수 리스트](https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists)
- **이전 학습 활용**: 앞서 학습한 Counter, itertools.chain 등을 활용

### 🧪 사용 예시
```python
# 여러 파일의 단어 빈도 통합
result = wordcount("sample01.txt", "sample02.txt", "sample03.txt")
print(result)
```

### 📚 가변 인수 학습

#### 기본 가변 인수 예제
```python
def func(*args, **kwargs):
    """
    *args: 위치 인수들을 튜플로 받음
    **kwargs: 키워드 인수들을 딕셔너리로 받음
    """
    for arg in args:
        print("위치 인수:", arg)
        
    print("키워드 인수:", kwargs)
        
# 사용 예시
func("3", [1,2], "bonjour", x=4, y="y")
```

### 🎯 최종 목표
이 함수를 완성하면 여러 파일을 동시에 처리할 수 있는 강력한 도구를 갖게 됩니다!
