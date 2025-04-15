
---
> [!PDF|red] [[20250415_Suport_vector_machine_실습 강의자료.pdf#page=15&selection=12,0,13,0&color=red|20250415_Suport_vector_machine_실습 강의자료, p.15]]
> > SVM Gradient Decent Method (GD)
> 
> ![[20250415_Suport_vector_machine_실습 강의자료.pdf#page=15&rect=436,244,790,372&color=red|20250415_Suport_vector_machine_실습 강의자료, p.15]]
> ![[20250415_Suport_vector_machine_실습 강의자료.pdf#page=19&rect=16,138,495,424&color=red|20250415_Suport_vector_machine_실습 강의자료, p.19]]
> ![[20250415_Suport_vector_machine_실습 강의자료.pdf#page=20&rect=10,88,831,425&color=red|20250415_Suport_vector_machine_실습 강의자료, p.20]]
```python
def fit(self, X, y):
    """
    SVM 모델 학습
    - X: 입력 데이터 (data 개수 x feature 개수)
    - y: 타겟 레이블 (data 개수만큼 -1 또는 1로 이루어진 배열)
    """
    n_samples, n_features = X.shape # n_samples: 데이터 개수

    # 레이블을 -1 또는 1로 변환
    y_modified = np.where(y <= 0, -1, 1)

    # Weight 및 bias 초기화
    self.weights = np.zeros(n_features)  # 가중치 벡터
    self.bias = 0  # 절편

    # Gradient Decent
    for _ in range(self.n_iters):
        for idx, x_i in enumerate(X):
            # 조건: yi(W^T*xi + b) >= 1
            condition = y_modified[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
            if condition:
                # 조건 1: yi(W^T*xi + b) >= 1 인 경우 (마진 바깥쪽에 올바르게 분류된 경우)
                # Loss = 0이므로 Loss에 대한 기울기는 0이지만,
                # SVM에는 정규화 항(regularization term)인 λ||W||²이 있음
                # λ||W||²의 W에 대한 편미분은 2λW
                # 여기서 λ는 정규화 강도로 1/n_iters로 설정됨
                # 따라서 정규화 항의 기울기는 2 * (1/n_iters) * W = 2 * self.weights / self.n_iters
                self.weights -= self.learning_rate * (2 * self.weights / self.n_iters)
            else:
                # 조건 2: 잘못 분류되거나 마진 안에 있는 경우
                # 이 경우 Loss = 1 - yi(W^T*xi + b)이며
                # ∂L/∂W = -yi*xi, ∂L/∂b = -yi
                self.weights -= self.learning_rate * ((2 * self.weights / self.n_iters) - y_modified[idx] * x_i)
                self.bias -= self.learning_rate * (-y_modified[idx])

```

