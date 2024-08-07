
---
### 나이브 베이즈 분류기 (Naive Bayes Classifier)

나이브 베이즈 분류기는 **분류 작업**을 위한 기계 학습 알고리즘입니다. 이 알고리즘은 "나이브 베이즈 가정"이라 불리는, 모든 특징들이 주어진 분류 레이블에 대해 서로 독립적이라는 실질적인 가정을 합니다. 이 가정은 실제로는 틀릴 수 있지만, 빠르고 간단한 알고리즘을 가능하게 하여 유용한 경우가 많습니다. 나이브 베이즈를 구현하기 위해 모델을 학습시키는 방법과 학습된 모델을 사용하여 예측하는 방법을 알아야 합니다.

#### 1. 학습 (모수 추정)

학습의 목표는 모든 특징 $X_i$에 대해 $P(Y)$와 $P(X_i | Y)$의 확률을 추정하는 것입니다. 여기서 $\hat{p}$는 확률의 추정치를 나타냅니다.

##### 최대 우도 추정 (MLE) 사용:
$$\hat{p}(X_i = x_i | Y = y) = \frac{ \text{Count}(X_i = x_i \text{ and } Y = y)}{\text{Count}(Y = y)}$$

##### 라플라스 MAP 추정 사용:
$$\hat{p}(X_i = x_i | Y = y) = \frac{ \text{Count}(X_i = x_i \text{ and } Y = y) + 1 }{\text{Count}(Y = y) + 2}$$

##### 최대 우도 추정 (MLE) 사용한 Y의 사전 확률:
$$\hat{p}(Y = y) = \frac{ \text{Count}(Y = y)}{\text{Total count of examples}}$$

#### 2. 예측

특징 벡터 $x = [x_1, x_2, \dots, x_m]$에 대해 $y$의 값을 다음과 같이 추정합니다:
$$\hat{y} = argmax_{y = \{0, 1\}} \left( \log \hat{p}(Y = y) + \sum_{i=1}^m \log \hat{p}(X_i = x_i | Y = y) \right)$$

작은 데이터셋의 경우 로그 버전의 $argmax$를 사용하지 않아도 됩니다. 

#### 3. 이론

분류에서 예측을 할 때 우리는 $P(Y = y | X = x)$를 최대화하는 $y$ 값을 선택하고자 합니다.
$$\hat{y} = argmax_{y = \{0, 1\}} P(Y = y | \mathbf{X} = \mathbf{x})$$

베이즈 정리를 사용하면 다음과 같이 전개됩니다:
$$\hat{y} = argmax_{y = \{0, 1\}} \frac{P(Y = y)P(\mathbf{X} = \mathbf{x} | Y = y)}{P(\mathbf{X} = \mathbf{x})}$$

여기서 $P(\mathbf{X} = \mathbf{x})$는 모든 $Y$에 대해 상수이므로 무시할 수 있습니다:
$$\hat{y} = argmax_{y = \{0, 1\}} P(Y = y)P(\mathbf{X} = \mathbf{x} | Y = y)$$

나이브 베이즈 가정을 사용하면 다음과 같이 간소화됩니다:
$$\hat{y} = argmax_{y = \{0, 1\}} P(Y = y) \prod_{i} P(X_i = x_i | Y = y)$$

로그를 취하면:
$$\hat{y} = argmax_{y = \{0, 1\}} \left( \log P(Y = y) + \sum_{i} \log P(X_i = x_i | Y = y) \right)$$

이러한 알고리즘은 학습과 예측 시 빠르고 안정적입니다.

#### 나이브 베이즈 가정

나이브 베이즈 가정은 각 특징 $x_i$가 주어진 $y$에 대해 서로 독립적이라는 것입니다. 이 가정은 실제로는 틀릴 수 있지만, 큰 특징 공간에서 데이터를 학습하고 예측하는 데 유용합니다. 이 가정을 통해 알고리즘을 간소화하고, 계산 복잡성을 줄일 수 있습니다.

---
![[Pasted image 20240610191037.png]]
![[Pasted image 20240610191323.png]]
![[Pasted image 20240610191347.png]]

---
![[Pasted image 20240610200513.png]]
![[Pasted image 20240610200631.png]]
![[Pasted image 20240610200658.png]]

---
![[Pasted image 20240610200738.png]]
![[Pasted image 20240610200757.png]]
