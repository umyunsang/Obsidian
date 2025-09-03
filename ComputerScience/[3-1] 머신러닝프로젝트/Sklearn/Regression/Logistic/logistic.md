
---
### 1. **데이터 불러오기 및 확인**

```python
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/fish.csv'
fish = pd.read_csv(path)
```

- **데이터셋 경로**에서 CSV 파일을 **불러온다**.
- `fish['Species']`를 통해 **목표 변수**인 어종(Species)을 확인할 수 있다.

```python
print(pd.unique(fish['Species']))
```
- **데이터셋에 포함된 어종**: ['Bream', 'Roach', 'Perch', 'Pike', 'Smelt', 'Parkki', 'Whitefish']

---

### 2. **모델의 입력 변수 및 목표 변수 선택**

```python
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
```
- **입력 변수(Features)**: 무게(Weight), 길이(Length), 대각선(Diagonal), 높이(Height), 너비(Width)
- **목표 변수(Target)**: 어종(Species)

---

### 3. **훈련 세트와 테스트 세트로 데이터 분할**

```python
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
```
- **훈련 세트**: 모델을 학습시키기 위한 데이터
- **테스트 세트**: 학습된 모델을 평가하기 위한 데이터
- `random_state=42`를 통해 **데이터 분할의 재현성**을 보장

---

### 4. **데이터 표준화**

```python
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```
- **표준화(Standardization)**: 평균이 0, 표준편차가 1이 되도록 데이터를 변환
- **표준화 과정**: `fit`으로 표준화 기준 학습 → `transform`으로 데이터 변환

---

### 5. **K-최근접 이웃(KNN) 분류기**

```python
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
```
- **K-최근접 이웃(KNN)**: 이웃한 데이터 포인트들로 새로운 데이터의 클래스를 예측
- **이웃의 수(n_neighbors)**: 3으로 설정

```python
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```
- **모델 정확도**: 훈련 세트와 테스트 세트에서의 예측 성능 평가

---

### 6. **로지스틱 회귀(Logistic Regression)**

#### (1) **이진 분류(Binary Classification)**
```python
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
```
- **이진 분류**: 두 개의 클래스(Bream과 Smelt)만을 분류
- **로지스틱 회귀(Logistic Regression)**: 입력 변수와 목표 변수 간의 관계를 학습하는 선형 모델

```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(expit(decisions))
```
- **결정 함수(Decision Function)**: 선형 모델의 출력을 기반으로 클래스 예측
- **시그모이드 함수(Sigmoid Function)**: 결정 함수를 확률로 변환

#### (2) **다중 분류(Multiclass Classification)**
```python
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
```
- **다중 분류**: 여러 클래스(여러 어종) 중 하나를 예측
- **소프트맥스 함수(Softmax Function)**: 모든 클래스에 대한 확률을 계산, 합이 1이 되도록 보장

```python
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
```
- **예측 확률 출력**: 테스트 세트의 샘플에 대한 각 클래스의 확률

---

### 7. **시그모이드 함수와 소프트맥스 함수**

#### (1) **시그모이드 함수 (Sigmoid Function)**
```python
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```
- **시그모이드 함수**: 연속적인 실수 입력을 0과 1 사이의 확률로 변환
- **그래프**: S자 형태로, 입력 값(z)이 클수록 1에, 작을수록 0에 가까워짐

#### (2) **소프트맥스 함수 (Softmax Function)**
```python
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```
- **소프트맥스 함수**: 다중 클래스 분류에서 각 클래스에 대한 확률을 계산
- **확률의 합은 1**이 되며, 각 확률은 입력 값에 비례

---

### 8. **모델 계수 및 성능 평가**

```python
print(lr.coef_, lr.intercept_)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```
- **로지스틱 회귀 계수 및 절편**: 각 입력 변수의 가중치와 모델의 바이어스를 의미
- **모델 성능 평가**: 훈련 세트와 테스트 세트에서의 모델 정확도 확인

---

### 핵심 정리:

1. **데이터 전처리**: 표준화를 통해 데이터의 스케일을 맞추고, 훈련 세트와 테스트 세트를 나눕니다.
2. **모델 학습**: KNN과 로지스틱 회귀를 사용해 이진 분류 및 다중 분류 문제를 해결합니다.
3. **확률 예측**: 시그모이드 함수와 소프트맥스 함수를 활용해 클래스에 대한 확률을 계산합니다.
4. **모델 평가**: 훈련 데이터와 테스트 데이터를 통해 모델의 성능을 평가하고, 필요한 경우 조정합니다.

---
