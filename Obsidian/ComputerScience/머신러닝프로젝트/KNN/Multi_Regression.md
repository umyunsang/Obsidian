
---
### 1. 데이터셋 불러오기 및 준비

```python
import pandas as pd
import numpy as np

# 데이터셋 불러오기
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/perch_full.csv'
df = pd.read_csv(path)
perch_full = df.to_numpy()

# 타겟 변수: 농어의 무게
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,1000.0])
```

- **목적**: 농어의 무게를 예측하기 위해 주어진 데이터셋을 불러오고, 타겟 변수(`perch_weight`)를 정의합니다.
	- **이유**: 데이터셋에서 독립 변수와 타겟 변수를 분리하여 회귀 모델을 훈련하고 평가하는 데 필요합니다.

### 2. 데이터 분할

```python
from sklearn.model_selection import train_test_split

# 데이터 분할: 훈련 데이터셋과 테스트 데이터셋
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
```

- **목적**: 모델 훈련과 성능 평가를 위해 데이터를 훈련 세트와 테스트 세트로 나눕니다.
	- **이유**: 훈련 세트는 모델을 학습시키는 데 사용하고, 테스트 세트는 학습된 모델의 성능을 평가하는 데 사용하여 모델의 일반화 능력을 측정합니다.

### 3. 다항 특성 추가

```python
from sklearn.preprocessing import PolynomialFeatures

# 다항 특성 추가
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
```

- **목적**: 특성의 다항식을 추가하여 모델의 복잡성을 증가시킵니다.
	- **이유**: 비선형 관계를 모델링하기 위해 특성의 다항식 변환을 사용하여 모델의 성능을 향상시킬 수 있습니다.

### 4. 다항 회귀 모델 훈련 및 성능 평가

```python
from sklearn.linear_model import LinearRegression

# 다항 회귀 모델 훈련 및 성능 평가
lr = LinearRegression()
lr.fit(train_poly, train_target)
train_score = lr.score(train_poly, train_target)
test_score = lr.score(test_poly, test_target)
print(f"다항 회귀 (degree=5) 훈련 세트 R^2 점수: {train_score:.2f}")
print(f"다항 회귀 (degree=5) 테스트 세트 R^2 점수: {test_score:.2f}")
```

- **목적**: 다항 회귀 모델을 훈련시키고, 훈련 데이터와 테스트 데이터에서 모델의 성능을 평가합니다.
	- **이유**: 다항 회귀는 모델의 복잡성을 증가시키는 한편, 훈련 데이터와 테스트 데이터에서 모델의 성능을 비교하여 과적합 여부를 판단할 수 있습니다.

### 5. 데이터 표준화 및 릿지 회귀

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 데이터 표준화
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀: 적절한 규제 강도(alpha) 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# 규제 강도(alpha)에 따른 성능 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alpha_list), train_score, label='훈련 세트')
plt.plot(np.log10(alpha_list), test_score, label='테스트 세트')
plt.xlabel('log(alpha)')
plt.ylabel('R^2')
plt.title('릿지 회귀: alpha에 따른 성능 변화')
plt.legend()
plt.show()
```

- **목적**: 데이터를 표준화한 후, 릿지 회귀 모델을 사용하여 최적의 규제 강도(`alpha`)를 찾고, 그에 따른 성능 변화를 시각화합니다.
	- **이유**: 표준화된 데이터는 각 특성이 동일한 스케일을 가지므로, 릿지 회귀 모델의 규제 강도를 최적화하는 데 유리합니다. 그래프를 통해 규제 강도에 따른 성능 변화를 시각적으로 확인할 수 있습니다.

### 6. 최적의 alpha 값으로 릿지 회귀 모델 훈련 및 평가

```python
# 최적의 alpha 값으로 릿지 회귀 모델 훈련 및 평가
best_alpha = 0.1
ridge = Ridge(alpha=best_alpha)
ridge.fit(train_scaled, train_target)
train_score = ridge.score(train_scaled, train_target)
test_score = ridge.score(test_scaled, test_target)
print(f"릿지 회귀 (alpha={best_alpha}) 훈련 세트 R^2 점수: {train_score:.2f}")
print(f"릿지 회귀 (alpha={best_alpha}) 테스트 세트 R^2 점수: {test_score:.2f}")
```

- **목적**: 릿지 회귀의 최적 `alpha` 값을 사용하여 모델을 훈련시키고, 훈련 및 테스트 데이터에서의 성능을 평가합니다.
	- **이유**: 최적의 `alpha` 값을 찾음으로써 모델의 과적합을 방지하고, 모델의 성능을 최적화할 수 있습니다.

### 7. 라쏘 회귀 및 성능 평가

```python
from sklearn.linear_model import Lasso

# 라쏘 회귀: 적절한 규제 강도(alpha) 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# 규제 강도(alpha)에 따른 성능 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alpha_list), train_score, label='훈련 세트')
plt.plot(np.log10(alpha_list), test_score, label='테스트 세트')
plt.xlabel('log(alpha)')
plt.ylabel('R^2')
plt.title('라쏘 회귀: alpha에 따른 성능 변화')
plt.legend()
plt.show()
```

- **목적**: 라쏘 회귀 모델을 사용하여 최적의 규제 강도(`alpha`)를 찾고, 성능 변화를 시각화합니다.
	- **이유**: 라쏘 회귀는 특성 선택과 정규화를 통해 모델의 성능을 개선할 수 있으며, 규제 강도에 따른 성능 변화를 그래프로 확인하여 최적의 `alpha` 값을 결정할 수 있습니다.

### 8. 최적의 alpha 값으로 라쏘 회귀 모델 훈련 및 평가

```python
# 최적의 alpha 값으로 라쏘 회귀 모델 훈련 및 평가
best_alpha = 10
lasso = Lasso(alpha=best_alpha)
lasso.fit(train_scaled, train_target)
train_score = lasso.score(train_scaled, train_target)
test_score = lasso.score(test_scaled, test_target)
print(f"라쏘 회귀 (alpha={best_alpha}) 훈련 세트 R^2 점

수: {train_score:.2f}")
print(f"라쏘 회귀 (alpha={best_alpha}) 테스트 세트 R^2 점수: {test_score:.2f}")

# 계수가 0인 특성의 개수 확인
zero_coef_count = np.sum(lasso.coef_ == 0)
print(f"라쏘 회귀 (alpha={best_alpha})에서 계수가 0인 특성의 개수: {zero_coef_count}")
```

- **목적**: 라쏘 회귀의 최적 `alpha` 값을 사용하여 모델을 훈련시키고, 훈련 및 테스트 데이터에서의 성능을 평가합니다. 또한, 계수가 0인 특성의 개수를 확인합니다.
	- **이유**: 라쏘 회귀는 특성 선택의 역할도 하며, `alpha` 값을 통해 모델의 복잡성을 조절할 수 있습니다. 계수가 0인 특성의 개수를 확인함으로써 라쏘 회귀가 얼마나 많은 특성을 제거했는지를 평가할 수 있습니다.

---
