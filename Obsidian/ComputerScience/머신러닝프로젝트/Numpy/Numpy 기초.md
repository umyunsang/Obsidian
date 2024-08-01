
---
## 배열 생성 및 속성

```python
import numpy as np
```

### 1. 배열 생성
- **1차원 배열:**
    ```python
    a = np.array([2, 3, 4])
    ```

- **2차원 배열:**
    ```python
    b = np.array([[1, 2, 3], [4, 5, 6]])
    ```

### 2. 배열 속성
- **형상 (Shape):** 배열의 차원을 나타내는 튜플 반환.
    ```python
    print(a.shape)  # 출력: (3,)
    print(b.shape)  # 출력: (2, 3)
    ```

- **차원 수 (Number of Dimensions):** 배열의 차원 수 반환.
    ```python
    print(a.ndim)  # 출력: 1
    ```

### 3. 데이터 타입 (Data Types)
- 배열의 데이터 타입 반환.
    ```python
    print(a.dtype)  # 출력: int64 (기본 데이터 타입)
    ```

### 4. 배열 크기 (Size)
- 배열의 요소 수 반환.
    ```python
    print(a.size)  # 출력: 3
    ```

## 배열 연산

### 1. 기본 연산
- 배열 간의 기본적인 산술 연산 지원.
    ```python
    a = np.array([10, 20, 30])
    b = np.array([1, 2, 3])

    print(a - b)  # 출력: [9 18 27]
    print(a + b)  # 출력: [11 22 33]
    print(a / b)  # 출력: [10. 10. 10.]
    ```

### 2. 데이터 타입 관리
- 배열 생성 시 데이터 타입 지정 가능.
    ```python
    a = np.array([10, 20, 30], dtype=np.int32)
    print(a.dtype)  # 출력: int32
    ```

- 자동 형변환 (업 캐스팅) 예제:
    ```python
    b = np.array([10, 20.1, 40])
    print(b.dtype)  # 출력: float64
    ```

### 3. 브로드캐스팅
- 배열 간 형상이 다른 경우에도 산술 연산을 지원.
    ```python
    print(a * 10)  # 출력: [100 200 300]
    print(a / 2)   # 출력: [ 5. 10. 15.]
    print(a + 10)  # 출력: [20 30 40]
    ```

### 4. 다차원 배열 연산
- 다차원 배열 간의 연산 예제:
    ```python
    b = np.array([[10, 20, 30], [40, 50, 60]])
    c = np.array([2, 3, 4])

    print(b + c)
    print(b * c)
    ```

## 배열 생성 함수

### 1. 기본 배열 생성
- 제로 배열, 원 배열, 상수 배열, 단위 행렬 생성 예제:
    ```python
    print(np.zeros((2, 3)))    # 모든 요소가 0인 배열 생성
    print(np.ones((2, 3)))     # 모든 요소가 1인 배열 생성
    print(np.full((2, 3), 100)) # 모든 요소가 특정 값(여기서는 100)인 배열 생성
    print(np.eye(3))           # 3x3의 단위 행렬 생성
    print(np.eye(10))          # 10x10의 단위 행렬 생성
    ```

### 2. 범위 지정 배열 생성
- 연속된 값의 배열과 구간을 정해 생성하는 배열 생성 예제:
    ```python
    print(np.arange(0, 10))         # 0부터 9까지의 연속된 정수 배열 생성
    print(np.arange(0, 10, 2))      # 0부터 8까지 2씩 증가하는 정수 배열 생성
    print(np.arange(0.0, 1.0, 0.2)) # 0.0부터 1.0까지 0.2씩 증가하는 실수 배열 생성
    print(np.linspace(0, 10, 5))    # 0부터 10 사이를 5등분한 배열 생성
    print(np.linspace(0, 10, 4))    # 0부터 10 사이를 4등분한 배열 생성
    ```

## 수학 함수

### 1. 지수와 로그 함수
- 로그 스페이스 함수 예제:
    ```python
    a = np.logspace(0, 5, 6)
    print(a)
    print(np.log10(a))
    ```
