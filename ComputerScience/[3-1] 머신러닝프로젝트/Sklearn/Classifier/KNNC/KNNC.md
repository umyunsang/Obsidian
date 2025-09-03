
---
## K-최근접 이웃 (K-Nearest Neighbors) 분류기 활용

### 1. 도미와 빙어 데이터 준비
```python
# 도미 35마리 length길이(cm)와 weight무게(g) 데이터
bream_length = [25.4, 26.3, ... , 41.0]
bream_weight = [242.0, 290.0, ... , 950.0]

# 빙어 14마리 length길이(cm)와 weight무게(g) 데이터
smelt_length = [9.8, 10.5, ... , 15.0]
smelt_weight = [6.7, 7.5, ... , 19.9]
```

- **데이터 설명**: 도미와 빙어의 길이와 무게 데이터를 각각 준비합니다.

### 2. 데이터 시각화
```python
plt.scatter(bream_length, bream_weight, label='Bream')
plt.scatter(smelt_length, smelt_weight, label='Smelt')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()
```

- **시각화**: 도미와 빙어 데이터를 시각화하여 데이터 분포를 파악합니다.

### 3. 데이터 통합 및 레이블 생성
```python
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
```

- **데이터 통합**: 도미와 빙어 데이터를 하나의 배열로 합칩니다.
- **레이블 생성**: 도미는 1, 빙어는 0으로 레이블을 만듭니다.

### 4. KNN 분류기 생성 및 학습
```python
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
score = kn.score(fish_data, fish_target)
print(f"Training accuracy: {score}")
```

- **분류기 학습**: 전체 데이터를 사용하여 KNN 분류기를 학습시킵니다.
- **정확도 확인**: 학습 데이터의 정확도를 확인합니다.

### 5. 새로운 데이터 예측
```python
print(f"Predicted label for [30, 600]: {kn.predict([[30, 600]])}")
print(f"Predicted label for [25, 150]: {kn.predict([[25, 150]])}")
```

- **예측**: 새로운 데이터의 종류를 예측합니다.

### 6. 데이터셋 분리 (훈련 세트와 테스트 세트)
```python
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
```

- **데이터 분리**: 데이터를 훈련 세트와 테스트 세트로 나눕니다.

### 7. 분리된 데이터 시각화
```python
plt.scatter(train_input[:, 0], train_input[:, 1], label='Train')
plt.scatter(test_input[:, 0], test_input[:, 1], label='Test')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()
```

- **시각화**: 훈련 세트와 테스트 세트를 시각화하여 분포를 확인합니다.

### 8. 훈련 데이터로 KNN 분류기 학습 및 테스트 데이터 평가
```python
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
score = kn.score(test_input, test_target)
print(f"Test accuracy: {score}")
print(f"Predicted label for [25, 150]: {kn.predict([[25, 150]])}")
```

- **분류기 학습 및 평가**: 훈련 데이터로 학습하고, 테스트 데이터로 평가합니다.

### 9. 데이터 표준화 (Standardization)
```python
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
new = ([25, 150] - mean) / std
```

- **표준화**: 데이터를 평균이 0, 표준편차가 1이 되도록 표준화합니다.

### 10. 표준화된 데이터 시각화
```python
plt.scatter(train_scaled[:, 0], train_scaled[:, 1], label='Train')
plt.scatter(new[0], new[1], marker='^', label='New')
plt.xlabel('Length (standardized)')
plt.ylabel('Weight (standardized)')
plt.legend()
plt.show()
```

- **시각화**: 표준화된 데이터를 시각화합니다.

### 11. 표준화된 데이터로 KNN 분류기 학습
```python
kn = KNeighborsClassifier()
kn.fit(train_scaled, train_target)
```

- **분류기 학습**: 표준화된 데이터로 KNN 분류기를 학습시킵니다.

### 12. 테스트 데이터 표준화 및 평가
```python
test_scaled = (test_input - mean) / std
score = kn.score(test_scaled, test_target)
print(f"Test accuracy (scaled): {score}")
```

- **테스트 데이터 표준화 및 평가**: 테스트 데이터를 표준화하고 정확도를 평가합니다.

### 13. 새로운 데이터 예측 (표준화된 데이터)
```python
print(f"Predicted label for scaled [25, 150]: {kn.predict([new])}")
```

- **예측**: 표준화된 새로운 데이터의 종류를 예측합니다.

### 14. 이웃 데이터 확인 및 시각화
```python
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1], label='Train')
plt.scatter(new[0], new[1], marker='^', label='New')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D', label='Neighbors')
plt.xlabel('Length (standardized)')
plt.ylabel('Weight (standardized)')
plt.legend()
plt.show()
```

- **이웃 데이터 확인**: 새로운 데이터의 최근접 이웃을 확인하고 시각화합니다.

### 설명 요약
- **데이터 준비 및 시각화**: 도미와 빙어 데이터를 준비하고 시각화합니다.
- **KNN 분류기 학습**: 전체 데이터로 KNN 분류기를 학습시키고, 새로운 데이터를 예측합니다.
- **데이터 분리 및 평가**: 데이터를 훈련 세트와 테스트 세트로 분리하고 평가합니다.
- **데이터 표준화**: 데이터를 표준화하고, 표준화된 데이터로 분류기를 학습 및 평가합니다.
- **이웃 데이터 확인**: 새로운 데이터의 최근접 이웃을 확인하여 KNN 알고리즘의 동작을 이해합니다.

---
