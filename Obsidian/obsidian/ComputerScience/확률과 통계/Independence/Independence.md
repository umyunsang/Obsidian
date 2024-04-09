#ComputerScience #확률과통계 

---
1. **Definition: Independence (정의: 독립)**:
   - 두 사건이 독립이라는 것은 하나의 사건의 결과가 다른 사건의 결과에 영향을 미치지 않는 것을 의미합니다.
   - 수식: $P(A \cap B) = P(A) \times P(B)$

2. **Alternative Definition (대체적 정의)**:
   - 독립적인 두 사건 $A$ 와 $B$의 경우, 아래의 수식이 성립합니다.
   - 수식: $P(A|B) = P(A)$
   - 이는 한 사건의 발생 여부가 다른 사건의 조건부 확률에 영향을 미치지 않음을 나타냅니다.

3. **Independence is Symmetric (독립성의 대칭성)**:
   - 만약 $A$ 가 $B$ 와 독립이면, $B$ 가 $A$ 와 독립입니다.
   - 수식: $P(A|B) = P(A)$ 와 $P(B|A) = P(B)$

4. **Generalized Independence (일반화된 독립)**:
   - 사건들 $E_1, E_2, \dots, E_n$ 이 $k$ 개의 요소를 가진 모든 부분 집합(여기서 $k$ 는 1에서 $n$ 사이의 정수)에 대해 독립이라면, 이를 일반화된 독립이라고 합니다.
   - 수식: $P(E_{i_1} \cap E_{i_2} \cap \dots \cap E_{i_k}) = \prod_{i=1}^{k} P(E_{i})$

5. **How to Establish Independence (독립성 설정 방법)**:
   - 두 개 이상의 사건이 독립임을 수학적으로 증명하는 것이 기본적인 방법입니다.
   - 수식: $P(A \cap B) = P(A) \times P(B)$

6. **Independence and Compliments (독립성과 여사건)**:
   - 주어진 독립 사건 $A$ 와 $B$에 대해, 여사건 $A^c$ 와 $B^c$ 또한 독립입니다.
   - 수식: $P(A^c \cap B^c) = P(A^c) \times P(B^c)$

7. **Conditional Independence (조건부 독립성)**:
   - 사건들이 일정한 사건에 대해 조건부로 독립이라면, 이를 조건부 독립이라고 합니다.
   - 수식: $P(A  B | C) = P(A|C) \times P(B|C)$
