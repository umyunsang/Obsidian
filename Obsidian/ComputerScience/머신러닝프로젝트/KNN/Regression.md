
---
# 농어(length, weight) 데이터 분석 및 회귀 모델 비교

## 데이터 시각화
농어의 길이와 무게 데이터를 시각화하여 데이터의 분포를 확인합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 농어 데이터
perch_length = np.array([...])  # 생략
perch_weight = np.array([...])  # 생략

# 데이터 시각화
plt.scatter(perch_length, perch_weight)
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.show()
```

## 훈련 세트와 테스트 세트로 데이터 분리
데이터를 훈련 세트와 테스트 세트로 분리합니다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

## KNN 회귀 모델

### 모델 생성 및 학습
KNN 회귀 모델을 생성하고 학습시킵니다.

```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
```

### 모델 평가
모델의 $R^2$ 점수를 통해 평가합니다.

```python
score = knr.score(test_input, test_target)
print(f"R^2 score: {score}")
```

### 예측 및 MAE 계산
예측값과 실제값의 차이를 MAE(평균 절대 오차)로 계산합니다.

```python
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(f"Mean Absolute Error: {mae}")
```

### 과대 적합, 과소 적합 확인
훈련 세트와 테스트 세트의 $R^2$ 점수를 비교하여 과대 적합 및 과소 적합 여부를 확인합니다.

```python
print(f"Training R^2 score: {knr.score(train_input, train_target)}")
print(f"Test R^2 score: {knr.score(test_input, test_target)}")
```

### 이웃 수 변경
이웃 수를 3으로 설정하여 과소 적합을 해결합니다.

```python
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(f"Training R^2 score (k=3): {knr.score(train_input, train_target)}")
print(f"Test R^2 score (k=3): {knr.score(test_input, test_target)}")
```

### 회귀의 한계 확인
모델이 입력값의 범위를 벗어난 경우 예측 결과를 확인합니다.

```python
print(f"Predicted weight for length 100 cm: {knr.predict([[100]])}")
```

## 선형 회귀 모델

### 모델 생성 및 학습
선형 회귀 모델을 생성하고 학습시킵니다.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)
print(f"Predicted weight for length 50 cm: {lr.predict([[50]])}")
print(f"Linear Regression Coefficients: {lr.coef_}, Intercept: {lr.intercept_}")
```

### 선형 회귀 결과 시각화
선형 회귀 결과를 시각화합니다.

```python
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_], color='red')
plt.scatter(50, lr.predict([[50]]), marker='^')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.show()
```

### 모델 평가
선형 회귀 모델의 $R^2$ 점수를 통해 평가합니다.

```python
print(f"Training R^2 score (Linear Regression): {lr.score(train_input, train_target)}")
print(f"Test R^2 score (Linear Regression): {lr.score(test_input, test_target)}")
```

## 2차 다항 회귀 모델

### 데이터 변환
2차 다항 회귀를 위해 데이터를 변환합니다.

```python
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
```

### 모델 생성 및 학습
2차 다항 회귀 모델을 생성하고 학습시킵니다.

```python
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(f"Predicted weight for length 50 cm (Polynomial): {lr.predict([[50 ** 2, 50]])}")
print(f"Polynomial Regression Coefficients: {lr.coef_}, Intercept: {lr.intercept_}")
```

### 다항 회귀 결과 시각화
2차 다항 회귀 결과를 시각화합니다.

```python
point = np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01 * point ** 2 - 21.6 * point + lr.intercept_, color='red')
plt.scatter([50], lr.predict([[50 ** 2, 50]]), marker='^')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.show()
```

### 모델 평가
2차 다항 회귀 모델의 $R^2$ 점수를 통해 평가합니다.

```python
print(f"Training R^2 score (Polynomial Regression): {lr.score(train_poly, train_target)}")
print(f"Test R^2 score (Polynomial Regression): {lr.score(test_poly, test_target)}")
```

## 요약

- **KNN 회귀 모델**: 이웃 수에 따라 성능이 달라지며, k=3일 때 과소 적합 문제를 해결할 수 있습니다.
- **선형 회귀 모델**: 단순한 모델로, 데이터의 패턴을 충분히 설명하지 못할 수 있습니다.
- **2차 다항 회귀 모델**: 데이터의 곡선형 패턴을 더 잘 반영하여 높은 $R^2$ 점수를 얻을 수 있습니다.

---