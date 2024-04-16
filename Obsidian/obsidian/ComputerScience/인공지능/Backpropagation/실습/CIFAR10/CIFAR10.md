
---
## CIFAR10

```python
import numpy as np  
import torch  
import torch.nn as nn  
import torchvision.datasets as dataset  
import torchvision.transforms as transform  
from torch.utils.data import DataLoader  
  
# GPU를 사용할지 여부를 확인합니다.  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 2. 데이터셋 준비  
cifar10_train = dataset.CIFAR10(root="./", train=True, transform=transform.ToTensor(), download=True)  
cifar10_test = dataset.CIFAR10(root="./", train=False, transform=transform.ToTensor(), download=True)  
  
  
# 5. Multi Layer Perceptron (MLP) 모델 정의  
class MLP(nn.Module):  
    def __init__(self):  
        super(MLP, self).__init__()  
        self.fc1 = nn.Linear(3072, 500)  
        self.fc2 = nn.Linear(500, 500)  
        self.fc3 = nn.Linear(500, 500)  
        self.fc4 = nn.Linear(500, 500)  
        self.fc5 = nn.Linear(500, 500)  
        self.fc6 = nn.Linear(500, 10)  
        self.relu = nn.ReLU()  
        self.bn = nn.BatchNorm1d(500)  
        self.dropout = nn.Dropout(0.1)  
  
        torch.nn.init.xavier_normal_(self.fc1.weight.data)  
        torch.nn.init.xavier_normal_(self.fc2.weight.data)  
        torch.nn.init.xavier_normal_(self.fc3.weight.data)  
        torch.nn.init.xavier_normal_(self.fc4.weight.data)  
        torch.nn.init.xavier_normal_(self.fc5.weight.data)  
  
    def forward(self, x):  
        x = x.view(-1, 3072)  
        y = self.relu(self.bn(self.fc1(x)))  
        y = self.relu(self.bn(self.fc2(y)))  
        y = self.relu(self.bn(self.fc3(y)))  
        y = self.relu(self.bn(self.fc4(y)))  
        y = self.relu(self.bn(self.fc5(y)))  
        y = self.dropout(y)  
        y = self.fc6(y)  
        return y  
  
  
# 모델을 GPU로 이동합니다.  
network = MLP().to(device)  
  
# 6. 하이퍼파라미터 설정  
batch_size = 200  
learning_rate = 0.1  
training_epochs = 20  
loss_function = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  
  
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)  
  
# # 7. DataLoader 설정  
data_loader = DataLoader(dataset=cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True)  
  
# 8. 훈련 반복문  
for epoch in range(training_epochs):  
    avg_cost = 0  
    total_batch = len(data_loader)  
  
    for img, label in data_loader:  
        img, label = img.to(device), label.to(device)  
  
        pred = network(img)  
        loss = loss_function(pred, label)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        avg_cost += loss / total_batch  
    scheduler.step()  
    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))  
  
print('Learning finished')  
  
# 9. 학습된 모델을 이용한 정확도 확인  
with torch.no_grad():  
    img_test = torch.tensor(np.transpose(cifar10_test.data,(0, 3, 1, 2))) / 255.  
    label_test = torch.tensor(cifar10_test.targets)  
  
    prediction = network(img_test)  
    correct_prediction = torch.argmax(prediction, 1) == label_test  
    accuracy = correct_prediction.float().mean()  
    print('Accuracy:', accuracy.item())  
  
# 10. 학습된 모델의 가중치 저장  
torch.save(network.state_dict(), "../../Backpropagation/CIFAR10/mlp_mnist.pth")  
```

# # Epoch: 20 Loss = 0.098194  
# Learning finished  
# Accuracy: 0.5516999959945679