#ComputerScience #인공지능 #perceptron 
 
---
### 1. 패키지 선언
```python
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
```
- `torch`: PyTorch의 기본 라이브러리를 사용하기 위해 임포트합니다.
- `torch.nn`: 신경망 모델을 정의하는 데 필요한 클래스와 함수가 포함되어 있습니다.
- `torchvision.datasets`: 이미지 데이터셋을 다루는 데 사용됩니다.
- `torchvision.transforms`: 이미지 전처리를 위한 함수들이 제공됩니다.
- `torch.utils.data`: 데이터셋을 로드하고 미니배치로 분할하는 데 사용됩니다.

### 2. 데이터셋 준비
```python
mnist_train = dataset.MNIST(root="./", train=True, transform = transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root="./", train=False, transform = transform.ToTensor(), download=True)
```
- `MNIST` 데이터셋을 다운로드하고, 훈련 데이터셋과 테스트 데이터셋으로 나눕니다.
- `transform.ToTensor()`: 이미지를 텐서로 변환합니다.

### 3. 데이터셋 확인
```python
import matplotlib.pyplot as plt

print(len(mnist_train))  # 훈련 데이터셋의 샘플 개수를 출력합니다.
first_data = mnist_train[0]  # 첫 번째 데이터를 가져옵니다.
print(first_data[0].shape)  # 이미지의 형태를 출력합니다.
print(first_data[1])  # 해당 이미지의 레이블을 출력합니다.
plt.imshow(first_data[0][0, :, :], cmap='gray')  # 이미지를 시각화합니다.
plt.show()
```
- 데이터셋의 크기를 확인하고, 첫 번째 이미지와 해당하는 레이블을 출력합니다.
- `plt.imshow()`: 이미지를 시각화합니다.

### 4. 이미지 전처리
```python
first_img = first_data[0]
print(first_img.shape)  # 이미지의 형태를 출력합니다.
first_img = first_img.view(-1, 28 * 28)  # 이미지를 1차원으로 평탄화합니다.
print(first_img.shape)  # 변환된 이미지의 형태를 출력합니다.
```
- 이미지의 형태를 출력하고, 이미지를 1차원으로 평탄화하여 신경망에 입력할 수 있는 형태로 변환합니다.

### 5. Single Layer Perceptron (SLP) 모델 정의
```python
class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()
        self.fc = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.fc(x)
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
### 6. Hyper-parameters 설정
```python
batch_size = 100
learning_rate = 0.1
training_epochs = 15
loss_function = nn.CrossEntropyLoss()  # 손실 함수로 Cross Entropy Loss를 사용합니다.
network = SLP()  # 앞에서 정의한 SLP 모델을 인스턴스화합니다.
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  # SGD 옵티마이저를 설정합니다.
```
- 학습에 필요한 하이퍼파라미터를 설정합니다.
- 손실 함수로는 Cross Entropy Loss를 사용하고, 옵티마이저는 확률적 경사 하강법(SGD)을 사용합니다.

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
        pred = network(img)  # 모델에 이미지를 전달하여 예측값을 계산합니다.
        loss = loss_function(pred, label)  # 손실을 계산합니다.
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파를 통해 기울기 계산
        optimizer.step()  # 옵티마이저로 모델 파라미터 업데이트

        avg_cost += loss / total_batch  # 평균 손실을 계산합니다.

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))  # 에폭마다 평균 손실을 출력합니다.

print('Learning finished')  # 학습 완료 메시지 출력
```
- 지정된 에폭 수만큼 반복하면서 훈련을 진행합니다.
- 각 미니배치에서 손실을 계산하고, 역전파를 수행하여 모델을 학습시킵니다.

### 9. 학습된 모델을 이용한 정확도 확인
```python
with torch.no_grad():  # 기울기 계산 제외
    img_test = mnist_test.data.float()  # 테스트 데이터를 float 형태로 변환합니다.
    label_test = mnist_test.targets  # 테스트 데이터의 레이블을 가져옵니다.

    prediction = network(img_test)  # 테스트 데이터에 대한 예측을 계산합니다.
    correct_prediction = torch.argmax(prediction, 1) == label_test  # 정확하게 예측한 경우를 계산합니다.
    accuracy = correct_prediction.float().mean()  # 정확도를 계산합니다.
    print('Accuracy:', accuracy.item())  # 정확도를 출력합니다.
```
- 테스트 데이터를 사용하여 학습된 모델의 정확도를 확인합니다.

### 10. 학습된 모델의 가중치 저장
```python
torch.save(network.state_dict(), "./slp_mnist.pth")
```
- 학습된 모델의 가중치를 저장합니다. 이를 통해 나중에 모델을 불러와 사용할 수 있습니다.
