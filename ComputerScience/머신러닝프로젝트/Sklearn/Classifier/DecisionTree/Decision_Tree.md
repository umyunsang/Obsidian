
---
# **Decision Tree를 활용한 와인 데이터 분류**

## **1. 데이터 준비**

### **1.1 데이터 불러오기**
```python
import pandas as pd

wine = pd.read_csv('https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/wine.csv')
print(wine.head())
```
- **`wine.csv`**: 와인 데이터셋이 포함된 CSV 파일.
- **`pandas`**: 데이터 프레임으로 데이터를 처리하기 위한 라이브러리.
- **`head()`**: 데이터셋의 처음 5개 행을 확인하여 데이터 구조를 파악합니다.

### **1.2 입력 데이터와 목표 변수 추출**
```python
wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
wine_target = wine['class'].to_numpy()

print(wine_input[:5])  # 입력 데이터의 첫 5개 행 출력
```
- **입력 데이터**: `alcohol`, `sugar`, `pH` 컬럼을 사용.
- **목표 변수**: `class` 컬럼을 사용하여 와인의 품질을 분류.

## **2. 데이터 전처리**

### **2.1 데이터 분할**
```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, random_state=42)
```
- **`train_test_split`**: 데이터를 학습용과 테스트용으로 나눕니다.
- **`random_state=42`**: 결과 재현성을 위해 랜덤 시드를 고정합니다.

### **2.2 데이터 표준화**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_input)
train_scaled = scaler.transform(train_input)
test_scaled = scaler.transform(test_input)
```
- **`StandardScaler`**: 특성 값들을 표준화(평균 0, 표준편차 1)합니다.
- **필요성**: `alcohol`, `sugar`, `pH`의 값 범위가 다르므로, 표준화를 통해 모델의 성능을 향상시킬 수 있습니다.

## **3. 모델 학습 및 평가**

### **3.1 첫 번째 DecisionTreeClassifier 모델 학습**
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))  # 학습 데이터 정확도
print(dt.score(test_scaled, test_target))    # 테스트 데이터 정확도
```
- **모델 학습**: 학습 데이터로 결정 트리(Decision Tree)를 학습시킵니다.
- **모델 평가**: 학습 데이터와 테스트 데이터에 대해 정확도를 확인합니다.

### **3.2 결정 트리 시각화**
```python
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()
```
- **`plot_tree`**: 학습된 결정 트리를 시각화하여 트리 구조를 확인합니다.
- **시각화 해석**: 트리의 각 노드는 특성의 분기 기준을 나타내며, 리프 노드는 최종 클래스를 나타냅니다.

### **3.3 부분 시각화**
```python
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
- **`max_depth=1`**: 트리의 최대 깊이를 1로 제한하여 주요 노드만 시각화합니다.
- **`filled=True`**: 노드 색상을 통해 각 클래스의 비율을 시각적으로 나타냅니다.

## **4. 가지 치기 (Pruning)**

### **4.1 가지 치기를 적용한 모델 학습**
```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))  # 학습 데이터 정확도
print(dt.score(test_scaled, test_target))    # 테스트 데이터 정확도
```
- **가지 치기**: `max_depth=3`로 설정하여 트리의 복잡도를 줄여 과적합을 방지합니다.
- **결과**: 가지 치기 후, 모델의 성능이 테스트 데이터에서 더 일반화되어 나타날 수 있습니다.

### **4.2 가지 치기된 트리 시각화**
```python
plt.figure(figsize=(10, 7))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
- **가지 치기 후 트리 시각화**: 모델이 더 간결해졌음을 확인할 수 있습니다.

## **5. 추가 제안**

### **5.1 교차 검증 (Cross Validation)**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt, train_scaled, train_target, cv=5)
print(np.mean(scores))  # 교차 검증 평균 정확도 출력
```
- **`cross_val_score`**: 5-겹 교차 검증을 통해 모델의 일반화 성능을 평가합니다.
- **필요성**: 데이터의 여러 부분에 대해 성능을 측정하여 모델의 안정성을 확인할 수 있습니다.

### **5.2 특성 중요도 (Feature Importance)**
```python
importances = dt.feature_importances_
feature_names = ['alcohol', 'sugar', 'pH']
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
```
- **`feature_importances_`**: 각 특성이 모델에서 얼마나 중요한지를 나타내는 값입니다.
- **결과 해석**: 중요한 특성을 파악하여 모델을 해석하는 데 도움을 줍니다.

### **5.3 모델 저장 및 불러오기**
```python
import joblib

joblib.dump(dt, 'decision_tree_model.pkl')  # 모델 저장
loaded_model = joblib.load('decision_tree_model.pkl')  # 모델 불러오기
```
- **`joblib`**: 학습된 모델을 파일로 저장하고, 필요할 때 다시 불러와 사용할 수 있습니다.
- **활용 예**: 모델을 재학습하지 않고, 바로 예측에 사용할 수 있습니다.

---
