
---
### 독립 이산형 확률 변수(Independent discrete random variables)

### 정의:
독립 이산형 확률 변수란 두 개 이상의 이산형 확률 변수가 있을 때, 하나의 확률 변수의 발생이 다른 확률 변수의 발생에 영향을 미치지 않는 상황을 말합니다. 즉, 하나의 확률 변수의 결과가 다른 확률 변수의 결과에 독립적으로 발생하는 것을 의미합니다.

### 속성:
1. **독립성(Independence)**:
    - 두 이산형 확률 변수 X와 Y가 독립적이라면, 다음 조건을 만족합니다:
        $$P(X=x_i \cap Y=y_j) = P(X=x_i) \cdot P(Y=y_j)$$
    - 즉, 두 확률 변수의 결합 확률은 각각의 주변 확률의 곱으로 표현됩니다.

2. **결합 확률 질량 함수(Joint Probability Mass Function)**:
    - 두 독립적인 확률 변수 X와 Y의 결합 확률 질량 함수는 각각의 주변 확률 질량 함수의 곱으로 표현됩니다:
        $$P(X=x_i \cap Y=y_j) = P(X=x_i) \cdot P(Y=y_j)$$

3. **주변 확률 질량 함수(Marginal Probability Mass Function)**:
    - 독립적인 확률 변수 X와 Y의 주변 확률 질량 함수는 각각의 개별 확률 변수에 대한 확률 질량 함수와 같습니다.

4. **조건부 확률(Conditional Probability)**:
    - 독립적인 확률 변수 X와 Y에 대해 조건부 확률은 다음과 같이 정의됩니다:
        $$P(X=x_i | Y=y_j) = P(X=x_i)$$

.