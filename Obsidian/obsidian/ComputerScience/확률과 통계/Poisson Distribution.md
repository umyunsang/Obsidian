#ComputerScience #확률과통계 

---
**포아송 분포(Poisson Distribution):**

**표기:** $X \sim \text{Poi}(\lambda)$

**설명:**
포아송 분포는 <mark style="background: yellow;">일정한 시간</mark> 또는 공간 간격 내에서 발생하는 사건의 수를 모델링하는 이산 확률 분포로, 해당 사건들의 평균 발생률을 고려합니다. 이는 사건들이 독립적으로 발생하며 주어진 간격 내에서 일정한 평균 발생률을 가정하는 상황에서 사용됩니다.

**주요속성**:
- **매개변수:**
	- $\lambda$ (람다): 주어진 간격 내에서 사건의 평균 발생률.

- **지원:**
	포아송 랜덤 변수의 지원은 비음수 정수 집합 $\{0, 1, 2, 3, \ldots \}$으로, 사건의 수를 나타냅니다.

- **PMF 식:**
	포아송 랜덤 변수 $X$의 확률 질량 함수(PMF)는 다음과 같습니다:
$$ P(X = k) = \frac{{e^{-\lambda} \lambda^k}}{{k!}}, \quad \text{for } k = \{0, 1, 2, \ldots\}$$

- **기대값:**
	포아송 랜덤 변수의 기대값 또는 평균 ($\mu$)은 매개변수와 동일합니다:
$$ E(X) = \mu = \lambda $$

- **분산:**
	포아송 랜덤 변수의 분산 ($\sigma^2$) 또한 매개변수와 동일합니다:
$$ \text{Var}(X) = \sigma^2 = \lambda $$

- **PMF 그래프:**
	포아송 분포의 PMF 그래프는 확률 질량 함수 플롯으로, x-축은 사건의 수 $k$를 나타내고 y-축은 확률 $P(X = k)$를 나타냅니다. 일반적으로 그래프는 평균 발생률 매개변수 $\lambda$를 중심으로 오른쪽으로 치우친 모양을 보입니다.



## Poisson Proof

Binomial 분포의 PMF는 다음과 같습니다:
$$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $$

Poisson 분포의 PMF는 다음과 같습니다:
$$ P(X = k) = \frac{{e^{-\lambda} \lambda^k}}{{k!}} $$

Binomial 분포의 매개변수 $n$과 $p$가 Poisson 분포의 매개변수 $\lambda$와 관련된 경우를 살펴보겠습니다. $np = \lambda$ 라고 가정하겠습니다.

이제 $n$이 충분히 크고 $p$가 충분히 작은 경우, $n$을 큰 값으로, $p$를 작은 값으로 극한을 취하면:

$$ \lim_{n \to \infty} P(X = k) = \lim_{n \to \infty} \binom{n}{k} p^k (1-p)^{n-k} $$

$$ = \lim_{n \to \infty} \frac{{n!}}{{k! (n-k)!}} p^k (1-p)^{n-k} $$

$$ = \lim_{n \to \infty} \frac{{n \cdot (n-1) \cdot (n-2) \cdot \ldots \cdot (n-k+1)}}{{k!}} p^k \left(1-\frac{{\lambda}}{n}\right)^n \left(1-\frac{{\lambda}}{n}\right)^{-k} $$

$$ = \frac{{\lambda^k}}{{k!}} \cdot \lim_{n \to \infty} \left(1-\frac{{\lambda}}{n}\right)^n $$

$$ = \frac{{\lambda^k}}{{k!}} \cdot e^{-\lambda} $$

위의 극한은 Poisson 분포의 PMF와 동일한 형태를 가지므로, Binomial 분포의 극한이 Poisson 분포가 됨을 증명했습니다.

### Poisson 분포 응용 예시
- 일정 주어진 시간 동안에 도착한 고객의 수
- 1킬로미터 도로에 있는 흠집의 수
- 일정 주어진 생산시간 동안 발생하는 불량 수
- 하룻동안 발생하는 출생자 수
- 어떤 시간 동안 롤게이트를 통과하는 차량의 수
- 어떤 페이지 하나를 완성하는 데 발생하는 오타의 발생률
- 어떤 특정 진도 이상의 지진이 발생하는 수 

Example)
Probability of k requests from this area in the next 1minute?
On average, $\lambda = 5$ requests per minute
$$P(X = k) = \frac{{e^{-5} 5^k}}{{k!}}$$
