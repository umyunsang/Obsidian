
---
![[Pasted image 20240604134248.png]]
### ResNet 모듈

```python
import torch  
import torch.nn as nn  
import numpy as np  
import torchvision.datasets as dataset  
import torchvision.transforms as transform  
from torch.utils.data import DataLoader  
  
# Training dataset 다운로드  
cifar10_train = dataset.CIFAR10(root="./",  # 데이터셋을 저장할 위치  
                                train=True,  
                                transform=transform.ToTensor(),  
                                download=True)  
# Testing dataset 다운로드  
cifar10_test = dataset.CIFAR10(root="./",  
                               train=False,  
                               transform=transform.ToTensor(),  
                               download=True)  
  
from matplotlib import pyplot as plt  
  
print(len(cifar10_train))  # training dataset 개수 확인  
  
first_data = cifar10_train[1]  
print(first_data[0].shape)  # 두번째 data의 형상 확인  
print(first_data[1])  # 두번째 data의 정답 확인  
  
plt.imshow(first_data[0].permute(1, 2, 0))  
plt.show()  
  
  
class ResNet(nn.Module):  
    def __init__(self):  
        super(ResNet, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.conv1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  
  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  
  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1)  
  
        self.fc1 = nn.Linear(4096, 512)  
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 10)  
  
        # Skip connection을 위한 convolution layer 선언  
        self.conv_skip1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.conv_skip2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  
  
        # 파라미터를 가지지 않은 layer는 한 번만 선언해도 문제 없음  
        self.relu = nn.ReLU()  
  
    def forward(self, x):  
        # convolution layers  
        out = self.relu(self.conv1(x))  
  
        # Residual block (32)  
        skip = self.conv_skip1(out)  
        out = self.relu(self.conv1_1(out))  
        out = self.relu(self.conv1_1(out))  
        out = out + skip  
  
        # Residual block (32)  
        skip = self.conv_skip1(out)  
        out = self.relu(self.conv1_1(out))  
        out = self.relu(self.conv1_1(out))  
        out = out + skip  
  
        # Residual block (32)  
        skip = self.conv_skip1(out)  
        out = self.relu(self.conv1_1(out))  
        out = self.relu(self.conv1_1(out))  
        out = out + skip  
  
        out = self.relu(self.conv2(out))  
  
        # Residual block (64)  
        skip = self.conv_skip2(out)  
        out = self.relu(self.conv2_1(out))  
        out = self.relu(self.conv2_1(out))  
        out = out + skip  
  
        # Residual block (64)  
        skip = self.conv_skip2(out)  
        out = self.relu(self.conv2_1(out))  
        out = self.relu(self.conv2_1(out))  
        out = out + skip  
  
        # Residual block (64)  
        skip = self.conv_skip2(out)  
        out = self.relu(self.conv2_1(out))  
        out = self.relu(self.conv2_1(out))  
        out = out + skip  
  
        out = self.relu(self.conv3(out))  
  
        # 평탄화  
        out = out.reshape(-1, 4096)  
  
        # fully connected layers  
        out = self.relu(self.fc1(out))  
        out = self.relu(self.fc2(out))  
        out = self.fc3(out)  
  
        return out  
  
  
batch_size = 100  
learning_rate = 0.1  
training_epochs = 20  
loss_function = nn.CrossEntropyLoss()  
network = ResNet()  
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  
data_loader = DataLoader(dataset=cifar10_train,  
                         batch_size=batch_size,  
                         shuffle=True,  
                         drop_last=True)  
  
network = network.to('cuda:0')  
for epoch in range(training_epochs):  
    avg_cost = 0  
    total_batch = len(data_loader)  
  
    for img, label in data_loader:  
        img = img.to('cuda:0')  
        label = label.to('cuda:0')  
  
        pred = network(img)  
  
        loss = loss_function(pred, label)  
        optimizer.zero_grad()  # gradient 초기화  
        loss.backward()  
        optimizer.step()  
  
        avg_cost += loss / total_batch  
  
    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))  
  
print('Learning finished')
```