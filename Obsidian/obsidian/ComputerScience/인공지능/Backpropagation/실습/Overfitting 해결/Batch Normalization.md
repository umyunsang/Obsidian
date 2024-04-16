
---
# Batch Normalization
	MLP 재정의 : self.bn = nn.BatchNorm1d(100) 추가

```python
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

# GPU를 사용할지 여부를 확인합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 준비
mnist_train = dataset.MNIST(root="./", train=True, transform=transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root="./", train=False, transform=transform.ToTensor(), download=True)
mnist_train.data = mnist_train.data[:300]
mnist_train.targets = mnist_train.targets[:300]

# 5. Multi Layer Perceptron (MLP) 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(100)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.relu(self.bn(self.fc1(x)))
        y = self.relu(self.bn(self.fc2(y)))
        y = self.relu(self.bn(self.fc3(y)))
        y = self.relu(self.bn(self.fc4(y)))
        y = self.fc5(y)
        return y

# 모델을 GPU로 이동합니다.
network = MLP().to(device)

# 6. 하이퍼파라미터 설정
batch_size = 10
learning_rate = 0.1
training_epochs = 100
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

# 7. DataLoader 설정
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

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

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))

print('Learning finished')

# 9. 학습된 모델을 이용한 정확도 확인
with torch.no_grad():
    img_test = mnist_test.data.float().to(device)
    label_test = mnist_test.targets.to(device)

    prediction = network(img_test)
    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

# 10. 학습된 모델의 가중치 저장
torch.save(network.state_dict(), "../../Backpropagation/Vanishing Gradient/mlp_mnist.pth")

```

# 결과값
Epoch: 100 Loss = 0.037444
Learning finished
Accuracy: 0.805400013923645