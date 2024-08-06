
---
# Pandas Series 이해하기

**Pandas Series**는 데이터 분석을 위한 강력한 도구로, 1차원 배열과 같은 구조를 가지고 있으며, 인덱스와 값의 쌍으로 이루어져 있습니다. 다음은 Series를 이해하는 데 도움이 되는 다양한 표기기법을 활용한 설명입니다.

## 1. 기본 Series 생성

**기본 Series 생성:**

```python
import pandas as pd
import numpy as np

# 기본 Series
se = pd.Series([1, 2, np.nan, 4])
```

- **값**: `[1, 2, np.nan, 4]`
- **인덱스**: 기본 인덱스(0, 1, 2, 3)

출력 결과:
```
0    1.0
1    2.0
2    NaN
3    4.0
dtype: float64
```

## 2. 인덱스 지정하여 Series 생성

**인덱스를 지정하여 Series 생성:**

```python
# 인덱스가 지정된 Series
data = [1, 2, np.nan, 4]
indexed_se = pd.Series(data, index=['a', 'b', 'c', 'd'])
```

- **값**: `[1, 2, np.nan, 4]`
- **인덱스**: `['a', 'b', 'c', 'd']`

출력 결과:
```
a    1.0
b    2.0
c    NaN
d    4.0
dtype: float64
```

## 3. 값 접근하기

**값에 접근하기:**

```python
print(se[0])  # 기본 인덱스 0의 값
print(se[2])  # 기본 인덱스 2의 값
print(indexed_se['a'])  # 인덱스 'a'에 해당하는 값
print(indexed_se['c'])  # 인덱스 'c'에 해당하는 값
```

- **기본 인덱스 0의 값**: `1.0`
- **기본 인덱스 2의 값**: `NaN`
- **인덱스 'a'의 값**: `1.0`
- **인덱스 'c'의 값**: `NaN`

## 4. 결측값 처리

**결측값 확인하기:**

```python
print(se.isna())  # 각 요소가 NaN인지 여부를 Boolean Series로 출력
```

출력 결과:
```
0    False
1    False
2     True
3    False
dtype: bool
```

## 5. 예제: 학생 점수 데이터

**학생 점수 데이터 Series 생성:**

```python
# 학생 점수 데이터
scores = [78, 94, 56, 74, 67]
scores_se = pd.Series(scores, index=['김주연', '박효원', '정재현', '임승우', '황상필'])
```

- **값**: `[78, 94, 56, 74, 67]`
- **인덱스**: `['김주연', '박효원', '정재현', '임승우', '황상필']`

출력 결과:
```
김주연    78
박효원    94
정재현    56
임승우    74
황상필    67
dtype: int64
```

## 6. 통계 함수 사용

**평균 및 총합 계산하기:**

```python
print(scores_se.mean())  # 평균 점수
print(scores_se.sum())  # 총합 점수
```

- **평균 점수**: `73.8`
- **총합 점수**: `369`

---
