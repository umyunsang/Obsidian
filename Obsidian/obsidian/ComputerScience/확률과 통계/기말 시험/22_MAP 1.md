
---
# Maximum A Posteriori

MAP(Most A Posteriori) 추정은 파라미터를 추정하는 또 다른 방법으로, MLE(Maximum Likelihood Estimator)와는 조금 다른 접근 방식을 취합니다. MLE가 데이터가 주어졌을 때 파라미터 값이 가장 가능성이 높은 값을 선택한다면, MAP는 주어진 데이터를 고려할 때 파라미터의 가장 가능성이 높은 값을 선택하는 것입니다. 수식적으로는 다음과 같이 표현됩니다.
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } f(\theta | X_1, X_2, \dots, X_n)$$

여기서 우리는 관측되지 않은 랜덤 변수의 조건부 확률을 계산하려고 합니다. 이럴 때, Bayes의 정리를 사용할 수 있습니다. 식을 Bayes의 정리를 사용하여 확장해 보겠습니다.
Intuition with Bayes' Theorem:$$P(\theta|data)=\frac{P(data|\theta)P(\theta)}{P(data)}$$
Solving for ${\theta}_{MAP}$ :
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } f(\theta | X_1, X_2, \dots, X_n)$$
$$= \underset{\theta}{\operatorname{argmax }} \text{ } \frac{f(X_1, X_2, \dots, X_n | \theta) g(\theta)}{h(X_1, X_2, \dots X_n)}$$

여기서 $f, g$ 및 $h$는 모두 확률 밀도입니다. 우리는 $f$가 다른 기능일 수 있음을 명시적으로 만들기 위해 다른 기호를 사용했습니다. 데이터가 IID로 가정되기 때문에 데이터가 주어진 $\theta$의 밀도를 분해할 수 있습니다. 또한 분모는 $\theta$에 대해 상수이므로 argmax에 영향을 미치지 않으며 해당 항목을 삭제할 수 있습니다. 수학적으로 표현하면 다음과 같습니다.
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } \prod_{i=1}^n f(X_i | \theta) g(\theta)$$

이전과 마찬가지로 MAP 함수의 로그의 argmax를 찾는 것이 더 편리할 것입니다. 이것이 파라미터의 MAP 추정의 최종 형태를 제공합니다.
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } \left( \log (g(\theta)) + \sum_{i=1}^n \log(f(X_i | \theta)) \right)$$

베이지안 용어를 사용하면, MAP 추정은 $\theta$ 에 대한 "사후 분포"의 모드입니다. MLE 방정식과 MAP 방정식을 나란히 비교하면 MAP가 로그의 사전항을 더한 정확히 동일한 함수의 argmax임을 알 수 있습니다.
- log prior + log-likelihood

![[Pasted image 20240610121728.png]]

Laplace estimates : 각각의 Observe 에 +1 을 함  

## Conjugate distributions

위 내용은 MAP(Maximum A Posteriori) 추정을 준비하기 위해 사전 분포(prior distributions)에 대해 다룹니다. 각각의 다른 매개변수에 대한 합리적인 분포가 필요합니다. 예를 들어, 포아송 분포를 예측한다면 𝜆에 대한 사전 분포의 적절한 확률 변수 유형은 무엇일까요?

다음은 각각의 다른 매개변수 및 그들의 사전 분포로 가장 자주 사용되는 분포 목록입니다:

| Parameter                     | Distribution  |
| ----------------------------- | ------------- |
| Bernoulli (베르누이 분포) $p$       | Beta          |
| Binomial (이항 분포) $p$          | Beta          |
| Poisson (포아송 분포) $\lambda$    | Gamma         |
| Exponential (지수 분포) $\lambda$ | Gamma         |
| Multinomial $p_i$             | Dirichlet     |
| Normal $\mu$                  | Normal        |
| Normal $\sigma^2$             | Inverse Gamma |

