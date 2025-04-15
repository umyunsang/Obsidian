
---
# Quadratic Programming 기반 SVM(Support Vector Machine)

> [!info] QP-SVM 개요
> Quadratic Programming(QP)은 SVM의 최적화 문제를 수학적으로 정확하게 해결하는 방법입니다. 이 방식은 경사 하강법과 달리 정확한 최적해를 구할 수 있습니다.
>
> SVM의 최적화 문제는 다음과 같이 표현됩니다:
> - 최소화: (1/2)||w||² (마진 최대화)
> - 제약조건: y_i(w^T·x_i + b) ≥ 1 (모든 데이터가 올바른 쪽에 위치)

## 1. scikit-learn 라이브러리를 활용한 SVM 구현
scikit-learn의 SVC 클래스를 사용하여 SVM 모델을 구현하고 학습시킵니다.

```python
from sklearn.svm import SVC
# 선형 커널을 사용하는 SVM 모델 정의 및 학습
clf = SVC(kernel='linear') # kernel:linear,rbf,poly,sigmoid 등
clf.fit(X, y)

# 시각화 실행
visualize_svm(clf.coef_[0], clf.intercept_)
```

> [!note] SVC 클래스 매개변수
> - kernel: 커널 함수 유형
>   - 'linear': 선형 커널 (w^T·x)
>   - 'rbf': 가우시안 RBF 커널 (비선형 분류)
>   - 'poly': 다항식 커널 (비선형 분류)
>   - 'sigmoid': 시그모이드 커널 (비선형 분류)
> - C: 규제 매개변수 (기본값=1.0, 작을수록 마진이 커지고 오류 허용)
> - gamma: RBF, poly, sigmoid 커널의 계수 ('scale', 'auto' 또는 실수)

> [!important] 모델 매개변수 설명
> - clf.coef_: 결정 경계의 가중치 벡터 (w)
> - clf.intercept_: 결정 경계의 편향 (b)
> - coef_와 intercept_는 선형 커널에서만 직접적인 의미가 있습니다.

![[Pasted image 20250415173546.png]]

## 2. 마진 계산
학습된 SVM 모델의 마진 크기를 계산합니다.

```python
margin = 2 / np.sqrt(np.dot(clf.coef_[0].T, clf.coef_[0]))
print(margin)
```

> [!tip] 마진 계산 공식
> - 마진 = 2 / ||w|| (여기서 ||w||는 가중치 벡터의 L2-norm)
> - 이론적으로 마진을 최대화하는 것이 SVM의 목표입니다.

> [!warning] QP-SVM과 GD-SVM의 차이점
> 1. **정확성**: QP는 정확한 최적해를 보장하지만, GD는 근사해를 제공
> 2. **계산 복잡성**: QP는 데이터 크기가 큰 경우 계산 비용이 높을 수 있음
> 3. **구현**: scikit-learn과 같은 라이브러리는 내부적으로 LIBSVM 등의 QP 해결책을 사용
> 4. **확장성**: GD는 대용량 데이터에 더 적합 (SGD 등의 변형을 통해)

> [!success] QP 방식의 장점
> - 정확한 최적해 보장
> - 커널 트릭을 통한 비선형 분류 가능
> - 소프트 마진(C 파라미터)을 통한 오분류 허용 가능
> - 다양한 커널 함수 사용 가능
