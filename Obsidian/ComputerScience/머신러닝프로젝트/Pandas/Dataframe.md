
---
# Pandas DataFrame 이해하기

**Pandas DataFrame**은 2차원 데이터 구조로, 행과 열로 이루어진 테이블 형태의 데이터를 처리하고 분석하는 데 사용됩니다. 데이터 분석에서 핵심적인 도구로 매우 유용합니다. DataFrame의 다양한 기능과 활용 방법을 이해하면 데이터 분석을 효과적으로 수행할 수 있습니다.

## 1. DataFrame 생성

**Series로부터 DataFrame 생성하기:**

```python
import pandas as pd
import numpy as np

# 월, 수익, 지출을 나타내는 Series 객체 생성
month_se = pd.Series(['1월', '2월', '3월', '4월'])
income_se = pd.Series([9500, 6200, 6050, 7000])
expenses_se = pd.Series([5040, 2350, 2300, 4800])

# DataFrame 생성
df = pd.DataFrame({
    '월': month_se,       # 월을 나타내는 열
    '수익': income_se,    # 수익을 나타내는 열
    '지출': expenses_se   # 지출을 나타내는 열
})

# DataFrame 출력
print(df)
```

**출력 결과:**

```
    월    수익   지출
0  1월  9500  5040
1  2월  6200  2350
2  3월  6050  2300
3  4월  7000  4800
```

- **행과 열**: DataFrame은 행(index)과 열(columns)로 구성됩니다.
- **열**: '월', '수익', '지출' (각 열은 Series 객체로 구성됨)

## 2. 데이터 분석 및 처리

**최대 수익 분석하기:**

```python
# 최대 수익이 발생한 월을 찾기
m_idx = np.argmax(income_se)  # 수익이 최대인 인덱스 찾기

# 최대 수익이 발생한 월과 수익 출력
print('최대 수익이 발생한 월:', month_se[m_idx])  # 최대 수익이 발생한 월
print(f'월 최대 수익: {income_se.max()}, 월 평균 수익: {income_se.mean()}')  # 최대 수익과 평균 수익
```

- **`np.argmax(income_se)`**: 수익 Series에서 최대 값을 가진 인덱스를 반환합니다.
- **`income_se.max()`**: 수익의 최대값을 반환합니다.
- **`income_se.mean()`**: 수익의 평균값을 반환합니다.

**출력 결과:**

```
최대 수익이 발생한 월: 4월
월 최대 수익: 7000, 월 평균 수익: 7187.5
```

## 3. DataFrame의 주요 기능

- **DataFrame의 기본 정보 확인:**

```python
print(df.info())  # 데이터프레임의 요약 정보를 출력
print(df.describe())  # 수치형 데이터의 통계 요약을 출력
```

- **행 및 열 접근:**

```python
print(df['월'])  # 특정 열 선택
print(df.loc[0])  # 특정 행 선택 (행 인덱스 기준)
print(df.iloc[0])  # 특정 행 선택 (정수 위치 기준)
```

- **새로운 열 추가:**

```python
df['순이익'] = df['수익'] - df['지출']  # 새 열 '순이익' 추가
print(df)
```

**출력 결과:**

```
    월    수익   지출  순이익
0  1월  9500  5040  4460
1  2월  6200  2350  3850
2  3월  6050  2300  3750
3  4월  7000  4800  2200
```

## 4. DataFrame 수정 및 갱신

**열 삭제 및 DataFrame 갱신하기:**

```python
# '2007' 열 삭제 및 데이터프레임 출력
print(df.drop('2007', axis=1))  # 삭제된 데이터프레임 출력

# '2007' 열을 삭제 (inplace로 갱신)
df.drop('2007', axis=1, inplace=True)
df['total'] = df[['2008', '2009', '2010', '2011']].sum(axis=1)  # 총합 열 재계산
df['mean'] = df[['2008', '2009', '2010', '2011']].mean(axis=1)   # 평균 열 재계산
print(df)  # 갱신된 데이터프레임 출력
```

- **`drop('2007', axis=1)`**: '2007' 열을 삭제합니다.
- **`inplace=True`**: 원본 DataFrame을 직접 수정합니다.

## 5. 데이터 시각화

**데이터 시각화 예제:**

```python
import matplotlib.pyplot as plt

# 바 차트
bar = df['2009'].plot(kind='bar', color=('orange', 'r', 'b', 'c', 'k'))
plt.show()  # 바 차트 출력

# 파이 차트
pie = df['2009'].plot(kind='pie')
plt.show()  # 파이 차트 출력

# 선 차트
line = df.plot(kind='line')
plt.show()  # 선 차트 출력
```

- **`plot(kind='bar')`**: 바 차트 생성
- **`plot(kind='pie')`**: 파이 차트 생성
- **`plot(kind='line')`**: 선 차트 생성

## 6. 슬라이싱과 인덱싱

**데이터 슬라이싱 및 인덱싱 예제:**

```python
# 상위 5행 출력
print(df.head())

# 2번째부터 6번째 행까지 출력
print(df[2:6])

# 특정 인덱스의 데이터 출력
print(df.loc['Korea'])  # 'Korea' 인덱스의 데이터 출력
print(df.loc[['US', 'Korea']])  # 'US'와 'Korea' 인덱스의 데이터 출력

# 특정 값 접근
print(df.loc['Korea', '2011'])  # 'Korea' 인덱스와 '2011' 열의 값 접근

# iloc 인덱서를 사용하여 특정 행 및 열 접근
print(df.iloc[4])  # 5번째 행 출력 (정수 위치 기준)
print(df.head(2)['2009'])  # 상위 2행의 '2009' 열 데이터 출력
print(df.iloc[[2, 4]])  # 3번째와 5번째 행 출력 (정수 위치 기준)
```

- **`loc[]`**: 라벨 기반 인덱싱
- **`iloc[]`**: 정수 위치 기반 인덱싱
- **`head(n)`**: 상위 `n`행을 반환합니다.

## 요약

- **DataFrame**은 행과 열로 구성된 2차원 데이터 구조입니다.
- **Series**는 DataFrame의 각 열 또는 행으로 활용됩니다.
- DataFrame의 다양한 메서드를 사용하여 데이터 분석 및 처리를 효율적으로 수행할 수 있습니다.

---