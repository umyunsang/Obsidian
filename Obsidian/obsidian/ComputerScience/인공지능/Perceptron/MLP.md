
---

### 1. 패키지 선언
```python
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
```
- PyTorch와 관련된 모듈들을 임포트합니다. 이들 모듈은 신경망을 구성하고 훈련시키는 데 필요합니다.

### 2. 데이터셋 준비
```python
mnist_train = dataset.MNIST(root="./", train=True, transform=transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root="./", train=False, transform=transform.ToTensor(), download=True)
```
- MNIST 데이터셋을 다운로드하고, 훈련 데이터셋과 테스트 데이터셋으로 나눕니다. `transform.ToTensor()`를 사용하여 이미지를 텐서로 변환합니다.

### 3. 데이터셋 확인
```python
import matplotlib.pyplot as plt

print(len(mnist_train))  
first_data = mnist_train[0]  
print(first_data[0].shape)  
print(first_data[1])        
plt.imshow(first_data[0][0, :, :], cmap='gray') 
plt.show()
```
- 데이터셋의 크기를 확인하고, 첫 번째 이미지와 해당하는 레이블을 출력하여 데이터셋을 시각화합니다.

### 4. 이미지 전처리
```python
first_img = first_data[0]
print(first_img.shape)  
first_img = first_img.view(-1, 28 * 28)  
print(first_img.shape)  
```
- 이미지를 1차원으로 평탄화하여 신경망에 입력할 수 있는 형태로 변환합니다.

### 5. Multi Layer Perceptron (MLP) 모델 정의
```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.sigmoid(self.fc1(x))
        y = self.fc2(y)
        return y
```
1. `class SLP(nn.Module):`
   - `SLP` 클래스를 정의합니다. 이 클래스는 `nn.Module` 클래스를 상속받습니다.

2. `def __init__(self):`
   - 클래스의 초기화 메서드입니다.
   - `super(SLP, self).__init__()`을 호출하여 부모 클래스인 `nn.Module`의 초기화 메서드를 호출합니다.

3. `self.fc = nn.Linear(in_features=784, out_features=10):`
   - `nn.Linear` 모듈을 사용하여 fully connected layer를 정의합니다.
   - `in_features=784`는 입력 특징의 수를 나타냅니다. MNIST 이미지는 28x28 픽셀이므로, 총 784개의 특징이 있습니다.
   - `out_features=10`는 출력의 크기를 나타냅니다. MNIST 데이터셋은 0부터 9까지의 숫자를 분류하는 문제이므로, 총 10개의 클래스가 있습니다.

4. `def forward(self, x):`
   - 순전파 메서드입니다. 모델에 입력 데이터를 전달하여 출력을 계산합니다.
   - `x = x.view(-1, 28 * 28)`을 사용하여 입력 이미지를 1차원으로 평탄화합니다. 이렇게 함으로써 28x28 크기의 이미지를 784개의 특징을 가진 벡터로 변환합니다.
   - `y = self.fc(x)`를 통해 fully connected layer를 통과한 출력을 계산합니다.
   - 최종적으로 출력값 `y`를 반환합니다.
### 6. 하이퍼파라미터 지정
```python
batch_size = 100
learning_rate = 0.1
training_epochs = 15
loss_function = nn.CrossEntropyLoss()
network = MLP()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
```
- 학습에 필요한 하이퍼파라미터를 설정하고, 손실 함수와 옵티마이저를 정의합니다.

### 7. DataLoader 설정
```python
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
```
- 훈련 데이터셋을 미니배치로 나누고, 데이터를 섞어서 로드합니다.

### 8. 훈련 반복문
```python
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        pred = network(img)
        loss = loss_function(pred, label)
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        avg_cost += loss / total_batch  

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))

print('Learning finished')  
```
- 지정된 에폭 수만큼 반복하면서 모델을 학습시킵니다.

### 9. 학습된 모델을 이용한 정확도 확인
```python
with torch.no_grad():  
    img_test = mnist_test.data.float()  
    label_test = mnist_test.targets  

    prediction = network(img_test)  
    correct_prediction = torch.argmax(prediction, 1) == label_test  
    accuracy = correct_prediction.float().mean()  
    print('Accuracy:', accuracy.item())  
```
- 테스트 데이터셋을 사용하여 학습된 모델의 정확도를 확인합니다.

### 10. 학습된 모델의 가중치 저장
```python
torch.save(network.state_dict(), "./mlp_mnist.pth")
```
- 학습된 모델의 가중치를 저장합니다.

