
---
![](../../../../../../image/Pasted%20image%2020240527184404.png)

#### LeNet-5 모델구조

```python
# CNN 모델 정의  
class CNN(nn.Module):  
    def __init__(self):  
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)  
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  
        self.fc1 = nn.Linear(400, 120)  
        self.fc2 = nn.Linear(120, 84)  
        self.fc3 = nn.Linear(84, 10)  
        self.relu = nn.ReLU()  
  
    def forward(self, x):  
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x)))  
        x = torch.reshape(x, (-1, 5 * 5 * 16))  
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        y = self.fc3(x)  
        return y  
  
  
# Hyper-parameters 지정  
batch_size = 100  
learning_rate = 0.1  
training_epochs = 30  
loss_function = nn.CrossEntropyLoss()  
network = CNN().to(device)  
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  
data_loader = DataLoader(dataset=cifar10_train,  
                         batch_size=batch_size,  
                         shuffle=True,  
                         drop_last=True)  
  
# Perceptron 학습을 위한 반복문 선언  
for epoch in range(training_epochs):  
    avg_cost = 0  
    total_batch = len(data_loader)  
  
    for img, label in data_loader:  
        img = img.to(device)  
        label = label.to(device)  
  
        pred = network(img)  
        loss = loss_function(pred, label)  
        optimizer.zero_grad()  # gradient 초기화  
        loss.backward()  
        optimizer.step()  
        avg_cost += loss / total_batch  
  
    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))  
  
print('Learning finished')  
  
# 학습이 완료된 모델을 이용해 정답률 확인  
network = network.to('cpu')  
with torch.no_grad():  # test에서는 기울기 계산 제외  
    network.eval()  
    img_test = torch.tensor(np.transpose(cifar10_test.data, (0, 3, 1, 2))) / 255.  
    label_test = torch.tensor(cifar10_test.targets)  
  
    prediction = network(img_test)  # 전체 test data를 한번에 계산  
  
    correct_prediction = torch.argmax(prediction, 1) == label_test  
    accuracy = correct_prediction.float().mean()  
    print('Accuracy:', accuracy.item())
```