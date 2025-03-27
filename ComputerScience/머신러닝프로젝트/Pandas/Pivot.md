
---
### Pandas를 활용한 데이터 조작 및 분석

---

#### 1. DataFrame 생성 및 초기화

먼저, 상품, 재질, 가격 정보를 포함하는 DataFrame을 생성합니다.

```python
import pandas as pd
import numpy as np

# DataFrame 생성
df = pd.DataFrame({
    '상품': ['시계', '반지', '반지', '목걸이', '팔찌'],
    '재질': ['금', '은', '백금', '금', '은'],
    '가격': [500000, 20000, 350000, 300000, 60000]
})
print(df)
```

출력:
```
    상품  재질      가격
0  시계   금  500000
1  반지   은   20000
2  반지  백금  350000
3  목걸이  금  300000
4  팔찌   은   60000
```

---

#### 2. Pivot 테이블 생성

`pivot` 함수를 사용하여 '상품'을 행, '재질'을 열, '가격'을 값으로 하는 새로운 DataFrame을 생성합니다. 결측치는 0으로 채웁니다.

```python
# pivot을 사용하여 '상품'을 행, '재질'을 열, '가격'을 값으로 하는 새로운 DataFrame 생성
new_df = df.pivot(index='상품', columns='재질', values='가격')
# 결측치를 0으로 채움
new_df = new_df.fillna(value=0)
print(new_df)
```

출력:
```
재질       금      백금       은
상품                            
목걸이  300000.0    0.0      0.0
반지         0.0  350000.0  20000.0
시계    500000.0    0.0      0.0
팔찌         0.0    0.0     60000.0
```

---

#### 3. DataFrame 결합

세로로 결합할 두 개의 DataFrame을 생성합니다.

```python
# 첫 번째 DataFrame 생성
df_1 = pd.DataFrame({
    'A': ['a10', 'a11', 'a12'],
    'B': ['b10', 'b11', 'b12'],
    'C': ['c10', 'c11', 'c12']
}, index=['가', '나', '다'])

# 두 번째 DataFrame 생성
df_2 = pd.DataFrame({
    'B': ['b23', 'b24', 'b25'],
    'C': ['c23', 'c24', 'c25'],
    'D': ['d23', 'd24', 'd25']
}, index=['다', '라', '마'])
```

각 DataFrame의 내용을 출력해 봅니다.

출력 `df_1`:
```
     A   B   C
가  a10 b10 c10
나  a11 b11 c11
다  a12 b12 c12
```

출력 `df_2`:
```
     B   C   D
다  b23 c23 d23
라  b24 c24 d24
마  b25 c25 d25
```

---

#### 4. DataFrame 세로로 결합

두 DataFrame을 세로로 결합합니다.

```python
# 두 DataFrame을 세로로 결합
df_3 = pd.concat([df_1, df_2])
print(df_3)
```

출력:
```
      A    B    C    D
가    a10  b10  c10  NaN
나    a11  b11  c11  NaN
다    a12  b12  c12  NaN
다    NaN  b23  c23  d23
라    NaN  b24  c24  d24
마    NaN  b25  c25  d25
```

---

#### 5. 공통 열만 포함하도록 결합

두 DataFrame을 공통 열만 포함하도록 결합합니다.

```python
# 두 DataFrame을 공통 열만 포함하도록 결합
df_4 = pd.concat([df_1, df_2], join='inner')
print(df_4)
```

출력:
```
      B    C
가    b10  c10
나    b11  c11
다    b12  c12
다    b23  c23
라    b24  c24
마    b25  c25
```

---

#### 6. merge를 이용한 조인 연산

각 유형의 조인 연산을 수행하여 결과를 비교합니다.

```python
# merge를 이용한 조인 연산
# left outer join: df_1 기준
print('left outer \n', df_1.merge(df_2, how='left', on='B'))

# right outer join: df_2 기준
print('right outer \n', df_1.merge(df_2, how='right', on='B'))

# full outer join: df_1과 df_2의 모든 데이터를 포함
print('full outer \n', df_1.merge(df_2, how='outer', on='B'))

# inner join: 공통된 데이터만 포함
print('inner \n', df_1.merge(df_2, how='inner', on='B'))
```

출력:
```
left outer 
     A   B   C_x   C_y    D
0  a10  b10  c10  NaN   NaN
1  a11  b11  c11  NaN   NaN
2  a12  b12  c12  c23  d23

right outer 
     A   B   C_x   C_y    D
0  a12  b23  c12  c23  d23
1  NaN  b24  NaN  c24  d24
2  NaN  b25  NaN  c25  d25

full outer 
     A   B   C_x   C_y    D
0  a10  b10  c10  NaN   NaN
1  a11  b11  c11  NaN   NaN
2  a12  b12  c12  c23  d23
3  NaN  b24  NaN  c24  d24
4  NaN  b25  NaN  c25  d25

inner 
     A   B   C_x   C_y    D
0  a12  b23  c12  c23  d23
```

- **Left Outer Join**: `df_1`을 기준으로 조인.
- **Right Outer Join**: `df_2`를 기준으로 조인.
- **Full Outer Join**: `df_1`과 `df_2`의 모든 데이터를 포함.
- **Inner Join**: 공통된 데이터만 포함.

---

### 요약

- **pivot**: 데이터 재구조화.
- **concat**: 데이터프레임 결합.
- **merge**: 다양한 방식으로 데이터프레임 병합.

---