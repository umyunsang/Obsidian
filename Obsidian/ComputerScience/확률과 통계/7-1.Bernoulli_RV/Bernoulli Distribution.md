#ComputerScience #확률과통계 

---
**베르누이 분포(Bernoulli Distribution)**:

베르누이 분포는 가장 간단한 확률 분포 중 하나로, 성공 또는 실패와 같이 이진 결과를 모델링하는 데 사용됩니다. 이는 예를 들어 동전 던지기와 같은 경우에 적용될 수 있습니다. 이진 결과에서 1은 성공을 나타내고, 0은 실패를 나타냅니다.

**주요 속성**:
- **확률 변수**:  $X \sim \text{Ber}(p)$ (베르누이 확률 변수)
- **설명**: 성공할 확률이 $p$ 인 이진 변수
- **매개 변수**: $p$ (성공 확률)
- **지원 범위**: 0 또는 1
- **PMF 방정식**:  $P(X = k) = p^k (1-p)^{1-k} (단, ( k = 0) 또는 ( k = 1 ))$ 
- **기대값**:  $E[X] = p$
- **분산**:  $\text{Var}(X) = p(1-p)$

베르누이 분포의 특징은 성공 확률 $p$를 통해 모든 속성을 계산할 수 있다는 것입니다. 이러한 이유로 베르누이 확률 변수를 선언하면 기대값, 분산 등의 속성을 즉시 알 수 있습니다.

**지시자 확률 변수(Indicator Random Variable)**:

지시자 확률 변수는 특정 이벤트가 발생하면 1이고, 그렇지 않으면 0인 베르누이 확률 변수입니다. 이는 주로 어떤 이벤트가 발생했는지 여부를 나타내는 데 사용됩니다.

**주요 속성**:
- **설명**: 특정 이벤트의 발생 여부를 나타내는 이진 변수
- **매개 변수**:  $p$ (성공 확률)
- **지원 범위**: 0 또는 1
- **PMF 방정식**:  $P(X = 1) = p$ and  $P(X = 0) = 1-p$
- **기대값**:  $E[X] = p$
- **분산**:  $\text{Var}(X) = p(1-p)$

이러한 속성을 이용하면 베르누이 분포와 지시자 확률 변수를 이해하고 활용할 수 있습니다.

**베르누이 분포 예시:**

예를 들어, 공정한 동전을 한 번 던져서 앞면이 나오는지 여부를 나타내는 베르누이 확률 변수 $X$를 고려해 봅시다. 이 경우, 앞면이 나올 확률은 $p = 0.5$이며, 뒷면이 나올 확률은 $1 - p = 0.5$입니다.

**풀이:**

1. 앞면이 나오는 경우 $(X = 1)$:$$P(X = 1) = p = 0.5$$
2. 뒷면이 나오는 경우 $(X = 0)$:$$P(X = 0) = 1 - p = 1 - 0.5 = 0.5$$
이러한 예시에서는 동전을 던지기 전에 어떤 결과를 예상할 수 없으므로, 각 결과의 확률은 동등합니다. 따라서, 앞면이 나올 확률과 뒷면이 나올 확률은 모두 0.5로 동일합니다.
