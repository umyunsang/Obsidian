

---
#### 기본 라인 그래프 그리기
기본 x 값 [0, 1, 2, 3]을 사용하여 라인 그래프를 그립니다.

```python
import matplotlib.pyplot as plt

# 기본 라인 그래프 그리기 (기본 x 값 [0, 1, 2, 3])
plt.plot([1, 2, 3, 4])
plt.ylabel('y축')  # y축 레이블 설정
plt.xlabel('x축')  # x축 레이블 설정
plt.show()  # 그래프 출력
```

---

#### 넘파이를 사용하여 x^2 그래프 그리기
넘파이로 생성한 x 값들을 제곱하여 그래프를 그립니다.

```python
import numpy as np

# 넘파이를 사용하여 x^2 그래프 그리기
x = np.arange(10)
plt.plot(x ** 2)
plt.show()  # 그래프 출력
```

---

#### x^2 그래프의 축 범위 설정
x^2 그래프의 축 범위를 [0, 100, 0, 100]으로 설정합니다.

```python
# x^2를 그리고 축 범위를 [0, 100, 0, 100]으로 설정하여 그리기
x = np.arange(10)
plt.plot(x ** 2)
plt.axis([0, 100, 0, 100])
plt.show()  # 그래프 출력
```

---

#### 여러 함수들의 그래프 그리기
여러 함수들 $$y_1 = 2x ,  y_2 = \frac{1}{3}x^2 + 5 ,  y_3 = -x^2 + 5 $$를 그래프로 그립니다.

```python
# 여러 함수들을 그래프로 그리기: y1 = 2*x, y2 = (1/3)*x^2 + 5, y3 = -x^2 + 5
x = np.arange(-20, 20)
y1 = 2 * x
y2 = (1 / 3) * x ** 2 + 5
y3 = -x ** 2 + 5
plt.plot(x, y1, 'g--', y2, 'r^-', x, y3, 'b*:')  # 각 그래프 스타일 설정
plt.axis([-30, 30, -30, 30])  # 축 범위 설정
plt.show()  # 그래프 출력
```

---

#### sin(x)와 cos(x)를 같은 그래프에 그리기
sin(x)와 cos(x)를 동일한 그래프에 그립니다.

```python
# sin(x)와 cos(x)를 같은 그래프에 그리기
x = np.linspace(0, np.pi * 2, 100)
plt.plot(x, np.sin(x), 'r-')  # sin(x)을 빨간 실선으로 그리기
plt.plot(x, np.cos(x), 'b:')  # cos(x)를 파란 점선으로 그리기
plt.show()  # 그래프 출력
```

---

#### 그래프를 파일로 저장하기
그린 그래프를 `view.png` 파일로 저장합니다.

```python
# 그래프를 'view.png' 파일로 저장하기
x = np.linspace(0, np.pi * 2, 100)
fig = plt.figure()
plt.plot(x, np.sin(x), 'r-')  # sin(x)을 빨간 실선으로 그리기
plt.plot(x, np.cos(x), 'b:')  # cos(x)를 파란 점선으로 그리기
fig.savefig('view.png')  # 'view.png'로 그림 저장
```

---
