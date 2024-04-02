#ComputerScience #인공지능  

---
### AND 게이트

AND 게이트는 입력이 모두 참일 때만 출력이 참인 논리 연산입니다. 아래는 AND 게이트의 진리표입니다.

|입력 1|입력 2|출력|
|---|---|---|
|0|0|0|
|0|1|0|
|1|0|0|
|1|1|1|

이를 파이썬으로 구현한 코드는 다음과 같습니다.

```python
def AND_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    summation = x1 * w1 + x2 * w2
    if summation <= theta:
        return 0
    else:
        return 1

# 테스트
print(AND_gate(0, 0))  # 출력: 0
print(AND_gate(0, 1))  # 출력: 0
print(AND_gate(1, 0))  # 출력: 0
print(AND_gate(1, 1))  # 출력: 1

```

### OR 게이트

OR 게이트는 입력 중 하나 이상이 참이면 출력이 참인 논리 연산입니다. 아래는 OR 게이트의 진리표입니다.

|입력 1|입력 2|출력|
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|1|

이를 파이썬으로 구현한 코드는 다음과 같습니다.

```python
def OR_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    summation = x1 * w1 + x2 * w2
    if summation <= theta:
        return 0
    else:
        return 1

# 테스트
print(OR_gate(0, 0))  # 출력: 0
print(OR_gate(0, 1))  # 출력: 1
print(OR_gate(1, 0))  # 출력: 1
print(OR_gate(1, 1))  # 출력: 1

```

### NAND 게이트

NAND 게이트는 입력이 모두 참일 때만 출력이 거짓인 논리 연산입니다. 아래는 NAND 게이트의 진리표입니다.

|입력 1|입력 2|출력|
|---|---|---|
|0|0|1|
|0|1|1|
|1|0|1|
|1|1|0|

이를 파이썬으로 구현한 코드는 다음과 같습니다.

```python
def NAND_gate(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    summation = x1 * w1 + x2 * w2
    if summation <= theta:
        return 0
    else:
        return 1

# 테스트
print(NAND_gate(0, 0))  # 출력: 1
print(NAND_gate(0, 1))  # 출력: 1
print(NAND_gate(1, 0))  # 출력: 1
print(NAND_gate(1, 1))  # 출력: 0

```

### XOR 게이트

XOR 게이트는 입력이 서로 다를 때만 출력이 참인 논리 연산입니다. XOR 게이트는 단일 퍼셉트론으로는 구현할 수 없는데, 이는 선형으로 분리가 불가능하기 때문입니다. 대신 다른 게이트의 조합으로 XOR 게이트를 구현할 수 있습니다.
```python
# XOR 게이트를 AND, OR, NAND 게이트의 조합으로 구현
def XOR_gate(x1, x2):
    s1 = NAND_gate(x1, x2)
    s2 = OR_gate(x1, x2)
    return AND_gate(s1, s2)

# 테스트
print(XOR_gate(0, 0))  # 출력: 0
print(XOR_gate(0, 1))  # 출력: 1
print(XOR_gate(1, 0))  # 출력: 1
print(XOR_gate(1, 1))  # 출력: 0

```

위의 코드에서는 NAND 게이트와 OR 게이트의 조합으로 XOR 게이트를 구현했습니다. 이렇게 함으로써 XOR 게이트를 만들어낼 수 있습니다.
