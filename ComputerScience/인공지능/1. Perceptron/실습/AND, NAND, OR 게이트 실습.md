
---

### AND 게이트
```python
import numpy as np  
  
def AND (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
x1 = 1  
x2 = 0  
print(AND(x1, x2))  
  
x1 = 1  
x2 = 1  
print(AND(x1, x2))
```

---

### NAND 게이트

```python
import numpy as np  
  
def NAND (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([-0.5, -0.5])  
  b = 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
x1 = 1  
x2 = 0  
print(NAND(x1, x2))  
  
x1 = 1  
x2 = 1  
print(NAND(x1, x2))
```

---

### OR 게이트

```python
import numpy as np  
  
def OR (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.2  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
x1 = 1  
x2 = 0  
print(OR(x1, x2))  
  
x1 = 1  
x2 = 1  
print(OR(x1, x2))
```

---

### XOR 게이트

```python
import numpy as np  
  
def AND (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
def NAN

D (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([-0.5, -0.5])  
  b = 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
def OR (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.2  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
def XOR(x1, x2):  
    s1 = NAND(x1, x2)  
    s2 = OR(x1, x2)  
    s3 = AND(s1, s2)  
    return s3  
  
print(XOR(0,0))  
print(XOR(0,1))  
print(XOR(1,0))  
print(XOR(1,1))
```