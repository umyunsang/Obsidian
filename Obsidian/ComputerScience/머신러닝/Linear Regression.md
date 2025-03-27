
---
## Linear Regression

>[!문제]
>![Image](images/Pasted%20image%2020250327165759.png)
>#### **1. 선형 회귀 모델**
>
>선형 회귀의 방정식은 다음과 같아.
>$$y = X \theta$$
>여기서
>- $y$는 종속 변수(출력값) 벡터,
  >  
>- $X$는 독립 변수(입력값) 행렬,
  >  
>- $\theta$는 회귀 계수 벡터야.
>
>최소제곱법을 이용하면 최적의 $\theta$는 다음 공식으로 구할 수 있어.
>
>$$\theta = (X^T X)^{-1} X^T y$$
>
>---
>#### **2. 행렬 정의**
>
>주어진 데이터는 다음과 같다
>
>$$(-3,-1), (-1,-1), (1,3), (3,3)$$이를 행렬 형태로 표현하면
>$$X =\begin{bmatrix}-3 & 1 \\-1 & 1 \\1 & 1 \\3 & 1\end{bmatrix}$$
>$$y =\begin{bmatrix}-1 \\-1 \\3 \\3\end{bmatrix}$$
>---
>
>#### **3. 최소제곱법 계산**
>
>##### **3.1 $X^T X$ 계산**
>$$X^T =\begin{bmatrix}-3 & -1 & 1 & 3 \\1 & 1 & 1 & 1\end{bmatrix}$$
>$$X^T X =\begin{bmatrix}-3 & -1 & 1 & 3 \\1 & 1 & 1 & 1\end{bmatrix}\begin{bmatrix}-3 & 1 \\-1 & 1 \\1 & 1 \\3 & 1\end{bmatrix}$$
>$$=\begin{bmatrix}(-3)^2 + (-1)^2 + (1)^2 + (3)^2 & (-3)(1) + (-1)(1) + (1)(1) + (3)(1) \\(-3)(1) + (-1)(1) + (1)(1) + (3)(1) & 1+1+1+1\end{bmatrix}$$
>$$=\begin{bmatrix}9 + 1 + 1 + 9 & -3 -1 +1 +3 \\-3 -1 +1 +3 & 4\end{bmatrix}$$
>$$=\begin{bmatrix}20 & 0 \\0 & 4\end{bmatrix}$$
>
>##### **3.2 $X^T y$  계산**
>
>$$X^T y =\begin{bmatrix}-3 & -1 & 1 & 3 \\1 & 1 & 1 & 1\end{bmatrix}\begin{bmatrix}-1 \\-1 \\3 \\3\end{bmatrix}$$
>$$=\begin{bmatrix}(-3)(-1) + (-1)(-1) + (1)(3) + (3)(3) \\(1)(-1) + (1)(-1) + (1)(3) + (1)(3)\end{bmatrix}$$
>$$=\begin{bmatrix}3 + 1 + 3 + 9 \\-1 -1 + 3 + 3\end{bmatrix}$$
>$$=\begin{bmatrix}16 \\4\end{bmatrix}$$
>
>##### **3.3 $(X^T X)^{-1}$ 계산**
>
>$$(X^T X)^{-1} =\begin{bmatrix}20 & 0 \\0 & 4\end{bmatrix}^{-1}$$
>대각 행렬이므로, 역행렬은 각 대각 원소의 역수를 취함
>$$=\begin{bmatrix}\frac{1}{20} & 0 \\0 & \frac{1}{4}\end{bmatrix}$$
>
>##### **3.4 $\theta$ 계산**
>
>$$\theta = (X^T X)^{-1} X^T y$$
>$$=\begin{bmatrix}\frac{1}{20} & 0 \\0 & \frac{1}{4}\end{bmatrix}\begin{bmatrix}16 \\4\end{bmatrix}$$
>$$=\begin{bmatrix}\frac{1}{20} \times 16 + 0 \times 4 \\0 \times 16 + \frac{1}{4} \times 4\end{bmatrix}$$
>$$=\begin{bmatrix}\frac{16}{20} \\1\end{bmatrix}$$
>$$=\begin{bmatrix}0.8 \\1\end{bmatrix}$$
>
>---
>
>#### **4. 최종 선형 회귀식**
>
>$$y = 0.8x + 1$$
