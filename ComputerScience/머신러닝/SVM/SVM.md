
---
> [!PDF|red] [[20250415_Suport_vector_machine_실습 강의자료.pdf#page=15&selection=12,0,13,0&color=red|20250415_Suport_vector_machine_실습 강의자료, p.15]]
> > SVM Gradient Decent Method (GD)
> 
> ![[20250415_Suport_vector_machine_실습 강의자료.pdf#page=15&rect=436,244,790,372&color=red|20250415_Suport_vector_machine_실습 강의자료, p.15]]
> ![[20250415_Suport_vector_machine_실습 강의자료.pdf#page=19&rect=16,138,495,424&color=red|20250415_Suport_vector_machine_실습 강의자료, p.19]]
> ![[20250415_Suport_vector_machine_실습 강의자료.pdf#page=20&rect=10,88,831,425&color=red|20250415_Suport_vector_machine_실습 강의자료, p.20]]
>```python
>def fit(self, X, y):
>       """
>       SVM 모델 학습
>       - X: 입력 데이터 (data 개수 x feature 개수)
>       - y: 타겟 레이블 (data 개수만큼 -1 또는 1로 이루어진 배열)
>       """
>       n_samples, n_features = X.shape # n_samples: 데이터 개수
>
>    # 레이블을 -1 또는 1로 변환
>    y_modified = np.where(y <= 0, -1, 1)
>
>       # Weight 및 bias 초기화
>       self.weights = np.zeros(n_features)  # 가중치 벡터
>       self.bias = 0  # 절편
>
>    # 경사 하강법(Gradient Descent) 구현
>    for _ in range(self.n_iters):  # 설정된 반복 횟수만큼 학습 진행
>        for idx, x_i in enumerate(X):  # 각 데이터 포인트에 대해 반복
>            # 조건: y_i(W^T * x_i + b) >= 1 (마진 조건 확인)
>            # 이 조건이 충족되면 데이터 포인트가 올바른 마진 바깥에 위치함
>            condition = y_modified[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
>
>            if not condition:  # 마진 조건을 만족하지 않는 경우 (마진 내부 또는 잘못 분류된 경우)
>                # 힌지 손실 함수의 그래디언트를 계산하여 가중치 업데이트
>                # ∂L/∂W = -y_i * x_i 에 따라 가중치 업데이트
>                self.weights += self.learning_rate * (y_modified[idx] * x_i)
>                # ∂L/∂b = y_i 에 따라 바이어스 업데이트
>                self.bias += self.learning_rate * (y_modified[idx])
>```

3개의 군집 데이터를 svm 한다고 할때 