
---
# **SGDClassifier  사용한 물고기 데이터 분류**

## **1. 데이터 준비**
### **1.1 데이터 불러오기**
```python
import pandas as pd

path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/fish.csv'
fish = pd.read_csv(path)
```
- **`fish.csv`**: 물고기 데이터셋이 포함된 CSV 파일.
- **`pandas`**: 데이터 처리에 유용한 라이브러리.

### **1.2 입력 데이터와 목표 변수 추출**
```python
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
```
- **입력 데이터**: 물고기의 `Weight`, `Length`, `Diagonal`, `Height`, `Width`.
- **목표 변수**: 물고기의 `Species`(종).

## **2. 데이터 전처리**
### **2.1 데이터 분할**
```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
```
- **`train_test_split`**: 데이터를 학습용과 테스트용으로 분할.
- **`random_state=42`**: 재현성을 위해 랜덤 시드를 고정.

### **2.2 데이터 표준화**
```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```
- **`StandardScaler`**: 데이터 표준화를 위해 사용.
- **표준화**: 평균이 0, 표준편차가 1이 되도록 변환.

## **3. 모델 학습**
### **3.1 첫 번째 SGDClassifier 학습**
```python
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))  # 학습 데이터 정확도
print(sc.score(test_scaled, test_target))    # 테스트 데이터 정확도
```
- **`SGDClassifier`**: 확률적 경사 하강법을 사용하는 분류기.
  - **`loss='log_loss'`**: 로지스틱 회귀 사용.
  - **`max_iter=10`**: 최대 10번의 반복만 허용.

### **3.2 부분 학습 및 정확도 변화 관찰**
```python
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```
- **부분 학습**: 한 번에 모든 데이터를 학습하지 않고, 점진적으로 학습.
- **정확도 시각화**: 학습(epoch) 횟수에 따른 학습/테스트 정확도 변화를 그래프로 표현.

### **3.3 최대 반복 횟수를 늘려 학습**
```python
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
- **`max_iter=100`**: 최대 100번의 반복으로 모델 학습.
- **`tol=None`**: 조기 종료 조건을 비활성화하여 설정된 반복 횟수까지 학습.

## **4. 다른 손실 함수 사용 (Hinge)**
```python
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
- **`loss='hinge'`**: 서포트 벡터 머신(SVM)의 손실 함수를 사용하여 학습.

## **5. 결론**
- **SGDClassifier**: 경량화된 모델로, 데이터가 클 때 빠르게 학습 가능.
- **손실 함수**:
  - **`log_loss`**: 로지스틱 회귀.
  - **`hinge`**: 서포트 벡터 머신.
- **하이퍼파라미터 튜닝**: `max_iter`, `tol` 등을 조정하여 성능 최적화 가능.
- **부분 학습(online learning)**: 데이터가 크거나 계속해서 업데이트되는 경우 유용.

---
