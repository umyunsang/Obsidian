
---
Adam(Adaptive Moment Estimation)은 경사 하강법 최적화 알고리즘의 한 종류로, 각 매개변수마다 학습률을 개별적으로 조정하여 효율적인 학습을 가능하게 합니다. Adam은 모멘텀 방법과 RMSProp 방법을 결합한 형태로, 기울기의 1차 및 2차 모멘트를 모두 고려하여 학습률을 조정합니다.

Adam은 다음과 같은 식으로 계산됩니다.

1. 모멘텀(momentum) 계산:
$$
v_t = \beta_1 v_{t-1} + (1 - \beta_1) \nabla J(\theta_t)
$$
2. RMSProp 계산:
$$
s_t = \beta_2 s_{t-1} + (1 - \beta_2) (\nabla J(\theta_t))^2
$$
3. 보정된 모멘텀(momentum) 및 RMSProp 계산:
$$
\hat{v}_t = \frac{v_t}{1 - \beta_1^t}
$$
$$
\hat{s}_t = \frac{s_t}{1 - \beta_2^t}
$$
4. 매개변수 업데이트:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{s}_t} + \epsilon} \hat{v}_t
$$

여기서,
- $v_t$는 시간 $t$에서의 모멘텀 벡터입니다.
- $s_t$는 시간 $t$에서의 RMSProp 벡터입니다.
- $\beta_1$은 모멘텀(momentum)의 지수 감소 비율입니다.
- $\beta_2$는 RMSProp의 지수 감소 비율입니다.
- $\eta$는 학습률(learning rate)입니다.
- $\epsilon$은 수치 안정성을 위한 작은 상수입니다.

Adam은 모멘텀 및 RMSProp의 지수 이동 평균을 사용하여 각 매개변수에 대한 학습률을 조정합니다. 이를 통해 Adam은 다양한 학습률 및 데이터 세트에 대해 효율적으로 수렴할 수 있으며, 매개변수의 스케일에 민감하지 않고 안정적인 학습을 제공합니다.

종합하면, Adam은 경사 하강법 최적화 알고리즘의 한 종류로, 모멘텀 및 RMSProp의 지수 이동 평균을 사용하여 각 매개변수에 대한 학습률을 조정하여 효율적이고 안정적인 학습을 가능하게 합니다. Adam은 다양한 신경망 모델 및 학습 과제에 널리 사용되고 있습니다.

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
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)  
  
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