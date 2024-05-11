
---
## ExponentialLR

ExponentialLR은 PyTorch와 같은 딥러닝 프레임워크에서 제공하는 학습률 스케줄링 방법 중 하나입니다. 이 방법은 학습률을 지수적으로 감소시키는 방법으로, 학습의 각 에폭(epoch) 또는 일정한 주기마다 학습률을 감소시킵니다.

주어진 초기 학습률과 감소 비율을 기반으로 학습률을 감소시킵니다. 보통은 다음과 같은 식으로 계산됩니다.

$$\text{lr} = \text{initial\_lr} \times \text{gamma}^{\text{epoch}}$$

여기서,
- $\text{lr}$은 현재 학습률,
- $\text{initial\_lr}$은 초기 학습률,
- $\text{gamma}$는 감소 비율,
- $\text{epoch}$은 현재 에폭을 나타냅니다.

즉, 각 에폭마다 현재 학습률을 초기 학습률에 감소 비율을 제곱한 값으로 조정합니다. 이를 통해 학습의 초기 단계에서는 빠르게 수렴하고, 학습이 진행됨에 따라 점진적으로 학습률을 감소시켜 안정적으로 수렴하도록 합니다.

ExponentialLR은 주로 초기 학습률과 감소 비율을 조정하여 모델의 학습 속도와 안정성을 개선하는 데 사용됩니다. 초기 학습률이 너무 크면 학습이 불안정해질 수 있고, 너무 작으면 학습 속도가 느려질 수 있습니다. 따라서 적절한 초기 학습률과 감소 비율을 선택하여 모델의 학습을 최적화하는 것이 중요합니다.

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
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  
  
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  
  
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
    scheduler.step()  
  
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


## StepLR

StepLR은 PyTorch와 같은 딥러닝 프레임워크에서 제공하는 학습률 스케줄링 방법 중 하나입니다. 이 방법은 학습률을 일정한 주기(스텝)마다 감소시키는 방법으로, 학습의 특정 시점이나 에폭마다 학습률을 감소시킵니다.

StepLR은 다음과 같은 식으로 계산됩니다.

$$
\text{lr} = \text{initial\_lr} \times \text{gamma}^{\lfloor\frac{\text{epoch}}{\text{step\_size}}\rfloor}
$$

여기서,
- $\text{lr}$은 현재 학습률,
- $\text{initial\_lr}$은 초기 학습률,
- $\text{gamma}$는 감소 비율,
- $\text{epoch}$은 현재 에폭을 나타냅니다,
- $\text{step\_size}$는 학습률을 감소시킬 주기를 나타냅니다.

즉, 각 주기마다 현재 학습률을 초기 학습률에 감소 비율을 제곱한 값으로 조정합니다. 이를 통해 일정한 주기마다 학습률을 감소시켜 모델의 학습을 안정화시키고 최적의 성능을 달성할 수 있도록 합니다.

StepLR은 초기 학습률과 감소 비율뿐만 아니라 주기적으로 학습률을 감소시킬 주기를 조절하는 것이 중요합니다. 주기가 너무 크면 학습률이 너무 빨리 감소하여 모델이 수렴하기 전에 학습이 중단될 수 있고, 너무 작으면 학습률이 너무 자주 감소하여 학습이 느려질 수 있습니다. 적절한 초기 학습률, 감소 비율 및 주기를 선택하여 모델의 학습을 최적화하는 것이 중요합니다.

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
        x = x.view(-1, 28*28)  
        y = self.fc(x)  
  
        return y  
  
# Hyper-parameters 지정  
batch_size = 100  
learning_rate = 0.1  
training_epochs = 15  
loss_function = nn.CrossEntropyLoss()  
network = SLP()  
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  
  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
  
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
  
    print('Epoch: %d  LR: %f  Loss = %f' % (epoch+1, optimizer.param_groups[0]['lr'], avg_cost))  
    scheduler.step()  
      
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