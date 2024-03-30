#ComputerScience #확률과통계 #probability

---
### Mutually Exclusive Events (상호 배타적 사건)

- **핵심내용**: 상호 배타적 사건은 동시에 발생할 수 없는 사건들을 나타냅니다. 즉, 하나의 사건이 발생하면 나머지 사건은 발생하지 않습니다.

### Or with Mutually Exclusive Events (상호 배타적 사건과의 합)

- **핵심내용**: 상호 배타적 사건 $A$와 $B$의 합은 각 사건이 발생할 확률을 더하는 것과 같습니다. 수식으로는 다음과 같이 표현됩니다:$$P(A \cup B) = P(A) + P(B)$$
### Or with Non-Mutually Exclusive Events (상호 배타적이 아닌 사건과의 합)

- **핵심내용**: 상호 배타적이 아닌 사건 $A$와 $B$의 합은 각 사건이 발생할 확률을 더하되, 중복으로 발생한 부분은 한번씩만 계산하는 것입니다. 수식으로는 포함-배제의 원칙을 적용하여 다음과 같이 표현됩니다:$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
### Inclusion-Exclusion with Three Events (세 개의 사건에 대한 포함-배제)

- **핵심내용**: 세 개의 사건 $A$, $B$, $C$에 대한 포함-배제는 각 사건이 발생할 확률을 더하고, 두 개의 사건의 교집합을 제외하며, 세 개의 사건의 교집합은 더해주는 것을 나타냅니다. 수식으로는 다음과 같이 표현됩니다:$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C) $$
### Inclusion-Exclusion with $n$ Events ($n$개의 사건에 대한 포함-배제)

- **핵심내용**: $n$개의 사건 $A_1, A_2, \ldots, A_n$에 대한 포함-배제는 각 사건이 발생할 확률을 더하고, 두 개의 사건의 교집합을 번갈아가며 제외하고, $n$개의 사건의 교집합은 번갈아가며 더해주는 것을 나타냅니다.
$$P\left(\bigcup_{i=1}^{n} A_i\right) = \sum_{i=1}^{n} P(A_i) - \sum_{1 \leq i < j \leq n} P(A_i \cap A_j) + \sum_{1 \leq i < j < k \leq n} P(A_i \cap A_j \cap A_k) - \ldots + (-1)^{n-1} P\left(\bigcap_{i=1}^{n} A_i\right)$$