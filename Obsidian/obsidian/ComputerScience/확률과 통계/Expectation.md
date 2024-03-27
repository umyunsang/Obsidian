
---
여기서는 "Expectation(기대값)"에 대해 설명하고, 이것이 가지는 여러 성질과 그것을 사용하는 방법에 대해 알아보겠습니다.

**Expectation(기대값)**:
확률 변수의 기대값은 해당 확률 변수가 가질 수 있는 모든 값들을 그 값이 나올 확률에 가중치를 곱하여 더한 것으로, 그것의 평균을 나타냅니다.

수학적으로는 다음과 같이 정의됩니다:
$$E[X] = \sum_{x} x \cdot P(X=x)$$
여기서 $E[X$는 확률 변수 \( X \)의 기대값을 의미하며, \( P(X=x) \)는 확률 변수 \( X \)가 \( x \)라는 값을 가질 확률을 나타냅니다.

예를 들어, 주사위를 던져서 나오는 값의 기대값은 다음과 같이 계산됩니다:

\[ E[X] = \sum_{x=1}^{6} x \cdot P(X=x) \]

여기서 \( x \)는 주사위의 값(1부터 6까지의 값)을 의미하며, \( P(X=x) \)는 해당 값이 나올 확률을 나타냅니다. 주사위의 경우, 모든 값이 나올 확률이 동일하기 때문에, 각각의 확률은 \( \frac{1}{6} \)입니다.

**성질들**:

1. **선형성(Linearity of Expectation)**:
\[ E[aX + b] = aE[X] + b \]
여기서 \( a \)와 \( b \)는 상수입니다.

2. **확률 변수의 합의 기대값(Expected Value of the Sum of Random Variables)**:
\[ E[X+Y] = E[X] + E[Y] \]
두 확률 변수의 합의 기대값은 각 확률 변수의 기대값의 합과 같습니다.

3. **Unconcious Statistician의 법칙(Law of Unconcious Statistician)**:
\[ E[g(X)] = \sum_{x} g(x) \cdot P(X=x) \]
확률 변수 \( X \)의 함수 \( g(X) \)의 기대값은 함수를 적용한 후의 값들에 대한 기대값을 구하는 것과 같습니다. 이것은 \( g(X) \)의 분포를 명시적으로 알지 못할 때 유용하게 사용됩니다.

4. **상수의 기대값(Expectation of a Constant)**:
상수의 기대값은 그 상수 자체입니다.
\[ E[c] = c \]

**예시**:
두 개의 주사위를 던져서 나온 값의 합의 기대값을 계산하는 코드의 예시를 살펴보겠습니다.

```python
def expectation_sum_two_dice():
    exp_sum_two_dice = 0
    for x in range(2, 12 + 1):
        pr_x = pmf_sum_two_dice(x)  # 두 주사위의 합이 x가 되는 확률을 구하는 함수
        exp_sum_two_dice += x * pr_x
    return exp_sum_two_dice

def pmf_sum_two_dice(x):
    count = 0
    for dice1 in range(1, 6 + 1):
        for dice2 in range(1, 6 + 1):
            if dice1 + dice2 == x:
                count += 1
    return count / 36  # 두 주사위의 모든 가능한 결과는 36가지
```

이 코드는 두 주사위의 합의 확률 분포를 사용하여 두 주사위를 던졌을 때 나오는 값의 합의 기대값을 계산합니다.