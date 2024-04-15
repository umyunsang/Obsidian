#ComputerScience #확률과통계 

---
### **Geometric Random Variable:**

- **Notation:** $X \sim \text{Geo}(p)$
- **Description:** 첫 번째 성공이 나타날 때까지의 시행 횟수를 나타냅니다. 각 시행은 독립적이며, 성공 확률이 $p$인 베르누이 시행입니다.
- **Parameters:** 성공 확률 $p$가 유일한 매개변수입니다.
- **Support:** 양의 정수 집합인 1부터 시작하여 계속됩니다.
- **PMF equation:** $$P(X = k) = (1-p)^{k-1} p$$                                                                                               (단, $k$는 1 이상의 양의 정수)
- **Expectation:** $$E(X) = \frac{1}{p}$$
- **Variance:** $$\text{Var}(X) = \frac{1-p}{p^2}$$
- **PMF graph:** 시행 횟수에 따른 확률을 나타내는 그래프로, 시행 횟수가 증가할수록 확률이 감소합니다. 

### **Negative Binomial Random Variable:**

- **Notation:** $Y \sim \text{NegBin}(r, p)$
- **Description:** $r$번째 성공이 나타날 때까지의 시행 횟수를 나타냅니다. 각 시행은 독립적이며, 성공 확률이 $p$인 베르누이 시행입니다.
- **Parameters:** 성공 횟수 $r$과 성공 확률 $p$가 매개변수입니다.
- **Support:** 양의 정수 집합인 $r$ 이상부터 시작하여 계속됩니다.
- **PMF equation:** $$P(Y = k) = \binom{k-1}{r-1} (1-p)^{k-r} p^r$$                                                                                                        (단, $k$는 $r$ 이상의 양의 정수)
- **Expectation:** $$E(Y) = \frac{r}{p}$$
- **Variance:** $$\text{Var}(Y) = \frac{r(1-p)}{p^2}$$
- **PMF graph:** 시행 횟수에 따른 확률을 나타내는 그래프로, 시행 횟수가 증가할수록 확률이 감소합니다. 예를 들어, $r = 3$이고 $p = 0.20$인 경우에 대한 PMF 그래프를 통해 시각적으로 확인할 수 있습니다.

이러한 확률 변수들은 성공을 기다리는 시행 횟수를 모델링하는 데 유용하며, 성공 확률과 성공 횟수의 조합에 따라 다양한 상황을 모델링할 수 있습니다.