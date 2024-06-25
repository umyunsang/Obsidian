
---

모멘텀(Momentum)은 경사 하강법(Gradient Descent) 최적화 알고리즘의 한 종류로, 기울기 갱신 시 직전 기울기를 고려하여 업데이트하는 방법입니다. 이는 기존의 경사 하강법보다 빠르게 수렴하고, 지역 최솟값(local minimum)에 빠지지 않고 전역 최솟값(global minimum)으로 빠르게 수렴하는 데 도움이 됩니다.

모멘텀은 다음과 같이 계산됩니다.

$$
\begin{align*}
v_t &= \beta v_{t-1} + \alpha \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t - v_t
\end{align*}
$$

여기서,
- $v_t$는 시간 $t$에서의 모멘텀(momentum) 벡터입니다.
- $\beta$ 는 모멘텀 상수(momentum coefficient)로, 일반적으로 0.9와 같은 값으로 설정됩니다.
- $\alpha$ 는 학습률(learning rate)입니다.
- $nabla J(\theta_t$) 는 현재 매개변수에 대한 손실 함수의 기울기(gradient)입니다.
- $\theta_t$ 는 시간 $t$ 에서의 매개변수(parameters)입니다.

모멘텀은 이전의 속도 벡터 $v_{t-1}$ 와 현재의 기울기 $\nabla J(\theta_t)$ 를 고려하여 새로운 속도 벡터 $v_t$ 를 계산합니다. 이후, 속도 벡터 $v_t$ 를 사용하여 매개변수를 업데이트합니다.

모멘텀을 사용하면 경사 하강법이 효율적으로 수렴하고 지역 최솟값에 빠지지 않는 것을 도와줍니다. 특히, 비등방성(anisotropic) 및 길쭉한(sloped) 기울기 표면에서 경사 하강법을 가속화하여 빠르게 수렴할 수 있습니다.

종합하면, 모멘텀은 경사 하강법 최적화 알고리즘의 한 종류로서, 기울기 갱신 시 직전 기울기를 고려하여 업데이트하여 수렴 속도를 높이고 안정성을 향상시키는 데 사용됩니다.


```python
import torch  
import torch.nn as nn  
import torchvision.datasets as dataset  
import torchvision.transforms as transform  
from torch.utils.data import DataLoader  
  
# Training dataset 다운로드  
mnist_train = dataset.MNIST(root="./",  # 데이터셋을 저장할 위치  
                            train=True,  
                            transform=transform.ToTensor(),  
                            download=True)  
# Testing dataset 다운로드  
mnist_test = dataset.MNIST(root='./',  
                           train=False,  
                           transform=transform.ToTensor(),  
                           download=True)  
  
  
# Single Layer Perceptron 모델 정의  
class SLP(nn.Module):  
    def __init__(self):  
        super(SLP, self).__init__()  
  
        self.fc = nn.Linear(in_features=784, out_features=10)  
  
    def forward(self, x):  
        x = x.view(-1, 28 * 28)  
        y = self.fc(x)  
  
        return y  
  
  
# Hyper-parameters 지정  
batch_size = 100  
learning_rate = 0.1  
training_epochs = 15  
loss_function = nn.CrossEntropyLoss()  
network = SLP()  
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum= 0.9)  
  
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  
  
data_loader = DataLoader(dataset=mnist_train,  
                         batch_size=batch_size,  
                         shuffle=True,  
                         drop_last=True)  
  
# Perceptron 학습을 위한 반복문 선언  
for epoch in range(training_epochs):  
    avg_cost = 0  
    total_batch = len(data_loader)  
  
    for img, label in data_loader:  
        pred = network(img)  
  
        loss = loss_function(pred, label)  
        optimizer.zero_grad()  # gradient 초기화  
        loss.backward()  
        optimizer.step()  
  
        avg_cost += loss / total_batch  
  
    print('Epoch: %d  LR: %f  Loss = %f' % (epoch + 1, optimizer.param_groups[0]['lr'], avg_cost))  
    # scheduler.step()  
  
print('Learning finished')  
  
# 학습이 완료된 모델을 이용해 정답률 확인  
with torch.no_grad():  # test에서는 기울기 계산 제외  
  
    img_test = mnist_test.data.float()  
    label_test = mnist_test.targets  
  
    prediction = network(img_test)  # 전체 test data를 한번에 계산  
  
    correct_prediction = torch.argmax(prediction, 1) == label_test  
    accuracy = correct_prediction.float().mean()  
    print('Accuracy:', accuracy.item())
```