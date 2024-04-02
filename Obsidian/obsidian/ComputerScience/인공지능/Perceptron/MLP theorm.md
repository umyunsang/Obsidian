
---
### 다층 퍼셉트론(Multi Layer Perceptron, MLP)

인공 신경망의 한 종류로, 여러 층의 뉴런들이 서로 연결되어 있는 구조입니다. 각 층은 입력층(input layer), 은닉층(hidden layer), 출력층(output layer)으로 구성됩니다. 각 뉴런은 이전 층의 모든 뉴런과 연결되어 있고, 가중치(weight)와 편향(bias)을 가지고 있습니다. 이를 수식으로 표현하면 다음과 같습니다:
$$\begin{align*}
\mathbf{y} &= \mathbf{h}(\mathbf{W}^T \mathbf{X} + \mathbf{b}) \\
\mathbf{y}_1 &= \mathbf{h}(\mathbf{W}_1^T \mathbf{X} + \mathbf{b}_1) \\
\mathbf{y}_2 &= \mathbf{h}(\mathbf{W}_2^T \mathbf{X} + \mathbf{b}_2) \\
\mathbf{Y} &= \mathbf{h}(\mathbf{W}^T \mathbf{X} + \mathbf{b}) \\
&= \mathbf{h}\left(\begin{pmatrix} \mathbf{W}_1^T \\ \mathbf{W}_2^T \end{pmatrix} \mathbf{X} + \begin{pmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \end{pmatrix}\right) \\
&= \begin{pmatrix} \mathbf{y}_1 \\ \mathbf{y}_2 \end{pmatrix} \\
\mathbf{Y} &= \mathbf{h}(\mathbf{W}_2^T \mathbf{h}(\mathbf{W}_1^T \mathbf{X} + \mathbf{b}_1) + \mathbf{b}_2)
\end{align*}$$
여기서:
- $\mathbf{X}$는 입력 데이터 벡터입니다.
- $\mathbf{W}$는 가중치 행렬입니다.
- $\mathbf{b}$는 편향 벡터입니다.
- $\mathbf{h}$는 활성화 함수입니다.
- $\mathbf{y}$는 출력 값입니다.
- $\mathbf{y}_1$과 $\mathbf{y}_2$는 은닉층의 뉴런들의 출력 값입니다.
- $\mathbf{Y}$는 최종 출력 값입니다.

다음은 이 다층 퍼셉트론의 그림입니다:

```
       [Input Layer]          [Hidden Layer]          [Output Layer]
          x1                    y1           ┌───►   y
          x2                    y2           │       │
          ...                   ...          │       │
          x10                   y10          └───►   y
                                           
```

### Fully Connected Layer와 활성화 함수

Fully Connected Layer는 다음과 같이 표기됩니다:
$$y = h(W^T X + b)$$
여기서:
- $X$ 는 입력 데이터 벡터입니다.
- $W$ 는 가중치 행렬입니다.
- $b$ 는 편향 벡터입니다.
- $h$ 는 활성화 함수입니다.

주로 사용되는 활성화 함수로는 시그모이드(sigmoid), 하이퍼볼릭 탄젠트(tanh), 렐루(ReLU, Rectified Linear Unit) 등이 있습니다.

### Softmax 함수

Softmax 함수는 출력층에서 주로 사용되며, 입력된 벡터를 각 클래스에 대한 확률 분포로 변환합니다. Softmax 함수는 다음과 같이 표기됩니다:
$$y= e^{z_i} / \sum_{j=1}^{N} e^{z_j}$$
여기서 $z$ 는 입력 벡터이고, $N$ 은 클래스의 개수입니다.

### 손실 함수

다중 클래스 분류 문제에서 주로 사용되는 손실 함수로는 교차 엔트로피 손실(cross-entropy loss)이 있습니다. 이 손실 함수는 모델의 예측과 실제 값 사이의 차이를 측정합니다. 교차 엔트로피 손실은 다음과 같이 표기됩니다:
$$Cross-Entropy Loss = -\sum_{i=1}^{N} y_i \log(p_i)$$

여기서 $y_i$ 는 실제 클래스의 원-핫 인코딩된 벡터이고, $p_i$ 는 모델이 예측한 해당 클래스의 확률입니다.

또한, 회귀 문제에서는 평균 절대 오차(Mean Absolute Error, MAE)와 평균 제곱 오차(Mean Squared Error, MSE)를 사용할 수 있습니다.

- 평균 절대 오차(MAE):
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|$$
- 평균 제곱 오차(MSE):
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$$
여기서 $y_i$ 는 실제 값이고, $\hat{y_i}$ 는 모델의 예측 값입니다. $n$ 은 데이터 샘플의 개수를 나타냅니다.

따라서, 신경망 모델을 학습할 때는 출력층에서 Softmax 함수를 사용하여 확률을 계산하고, 다중 클래스 분류에서는 교차 엔트로피 손실을, 회귀에서는 MAE 또는 MSE를 손실 함수로 사용하여 모델을 최적화합니다.