
---
### Defining the likelihood of data

데이터 샘플은 n개의 독립 동일 분포(iid)를 가진 확률 변수 $X_1, X_2, \dots, X_n$ 으로 이루어져 있습니다. 여기서 각 $X_i$는 밀도(또는 질량) 함수 $f(X_i|θ)$에서 뽑혔습니다.

가능도 질문:
	매개변수 θ가 주어졌을 때 샘플 ($X_1, X_2, \dots, X_n$)이 얼마나 가능한가요?

가능도 함수 L(θ)는 다음과 같이 정의됩니다:
$$L(\theta) = f(X_1,X_2,...,X_n|\theta)=\prod_{i=1}^n f(X_i|\theta)$$
### Maximum Likelihood Estimator

주어진 분포 $f(X_i|\theta)$로부터 뽑힌 n개의 독립 동일 분포(iid) 확률 변수 $X_1, X_2, \dots, X_n$의 샘플을 고려합니다.

최대 우도 추정량(MLE)은 가능도 함수 $L(\theta)$를 최대화하는 매개변수 $\theta$의 값입니다. 수식으로는 다음과 같이 표현됩니다:
$${\theta}_{MLE} = \underset{\theta}{\operatorname{argmax }} \text{ }L(\theta)$$

여기서 $\underset{\theta}{\operatorname{argmax }} \text{ }$는 가능도 함수 $L(\theta)$를 최대화하는 $\theta$의 값을 의미합니다.

샘플의 가능도(Likelihood)는 다음과 같이 정의됩니다:
$$L(\theta) = \prod_{i=1}^n f(X_i|\theta)$$

이 때, $X_i$가 연속형일 경우 $f(X_i|\theta)$는 확률 밀도 함수(PDF)이고, 이산형일 경우 확률 질량 함수(PMF)입니다.

추가로, 최대 우도 추정량 $\theta_{MLE}$은 로그-우도 함수(log-likelihood function) $LL(\theta)$를 최대화하는 값이기도 합니다. 

로그-우도 함수 $LL(\theta)$는 가능도 함수의 로그를 취한 것으로, 다음과 같이 정의됩니다:
$$LL(\theta)= \log L(\theta) = \log\left(\prod_{i=1}^n f(X_i|\theta)\right)=\sum_{i=1}^n \log f(X_i|\theta)$$

로그-우도 함수를 최대화하는 것은 가능도 함수를 최대화하는 것과 같은 결과를 가져옵니다. 로그-우도 함수를 사용하는 이유는 종종 가능도 함수를 미분하기 쉽기 때문입니다.


### Maximum Likelihood with Bernoulli

베르누이 $X$의 확률 질량 함수는 다음과 같이 쓸 수 있습니다: $$f(x_i|p) = p^x_i(1-p)^{1-x_i}$$
이제 MLE 추정을 해봅시다:

$$
L(\theta) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i}
$$

$$
LL(\theta) = \sum_{i=1}^n \log p^{x_i}(1-p)^{1-x_i}$$
$$= \sum_{i=1}^n x_i (\log p) + (1 - x_i) \log(1-p)$$ 
$$= Y \log p + (n - Y) \log(1-p)
$$

여기서 $Y = \sum_{i=1}^n x_i$입니다.

이제 로그 우도 방정식을 얻었으므로, 로그 우도를 최대화하는 $p$ 값을 선택해야 합니다. 이를 위해 함수의 1차 도함수를 찾아 0으로 설정합니다:

$$
\frac{\partial LL(p)}{\partial p} = Y \frac{1}{p} + (n - Y) \frac{-1}{1-p} = 0
$$

따라서,

$$
\hat{p} = \frac{Y}{n} = \frac{\sum_{i=1}^n x_i}{n}$$

결국, MLE 추정값은 단순히 샘플 평균이 됩니다.

![[Pasted image 20240610113804.png]]
![[Pasted image 20240610113836.png]]

### Maximum Likelihood with Normal

다음으로, 정규 분포의 최적 파라미터 값을 추정해 봅시다. 우리는 $n$개의 정규 분포에서 샘플링된 IID 랜덤 변수 $X_1, X_2, \dots, X_n$에 접근할 수 있습니다. 각 $X_i$는 $\mu = \theta_0, \sigma^2 = \theta_1$ 인 $N(\mu, \sigma^2)$에서 샘플링된 것으로 가정합니다. 이 경우 $\theta$는 평균( $\mu$ ) 및 분산( $\sigma^2$ )이라는 두 값을 가진 벡터입니다.

$$
L(\theta) = \prod_{i=1}^n f(X_i|\theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\theta_1}} e^{-\frac{(X_i - \theta_0)^2}{2\theta_1}}
$$

$$
LL(\theta) = \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\theta_1}} e^{-\frac{(X_i - \theta_0)^2}{2\theta_1}} = \sum_{i=1}^n \left[ - \log(\sqrt{2\pi\theta_1}) - \frac{1}{2\theta_1}(X_i - \theta_0)^2 \right]
$$

이제, 로그 우도 함수를 최대화하는 $\theta$ 값을 선택해야 합니다. 이를 위해 $LL$ 함수에 대해 $\theta_0$ 및 $\theta_1$에 대한 편미분을 계산하고 두 방정식을 모두 0으로 설정한 다음 $\theta$ 값을 구합니다. 그 결과는 다음과 같습니다:

$$
{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n X_i, \quad {\sigma^2}_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - {\mu}_{MLE})^2
$$
