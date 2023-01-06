### 합성곱 신경망 ###

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms     # 데이터의 전처리를 위해 사용
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 사용할 데이터셋 다운
train_dataset = torchvision.datasets.FashionMNIST("./data/chap05/data", download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
test_dataset  = torchvision.datasets.FashionMNIST("./data/chap05/data", download=True, train=False,
                                                  transform=transforms.Compose([transforms.ToTensor()]))

# 내려받은 FashionMNIST 데이터를 메모리로 불러오기 위해 데이터로더(DataLoader)에 전달
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=100)

# 분류에 사용될 클래스 정의
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal',
              6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
fig = plt.figure(figsize=(8,8)) # 출력할 이미지의 가로세로 길이로, 단위는 inch
columns = 4
rows = 5
for i in range(1, columns*rows + 1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

# np.random은 무작위로 데이터를 생성할 때 사용
# randint()는 이산형 분포를 갖는 데이터에서 무작위 표본을 추출
# np.random.randint(1,10) -> 1 or 5 or ...
# rand(num)는 0~1 사이의 정규표준분포 난수를 행렬로 (1xnum) 출력, rand(a,b)는 (axb)
# np.random.rand(4,2) -> array([[0.11, 0.22], [0.66, 0.22], ... ])

### ConvNet이 아닌 DNN 먼저 실습
class FashionDNN(nn.Module):
    def __init__(self):  # 1
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256) # 2
        self.drop = nn.Dropout(0.25)    # 3
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, input_data):  # 4
        out = input_data.view(-1, 784)  # 5
        out = F.relu(self.fc1(out)) # 6
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# 1
# Class 형태의 모델은 항상 torch.nn.Module을 상속 받음
# super(-).__init__() 은 FashionDNN이라는 부모(super) 클래스를 상속받겠다는 의미
# 2
# nn은 딥러닝 모델(네트워크) 구성에 필요한 모듈이 모여 있는 패키지
# 3
# torch.nn.Dropout(p)는 p만큼의 비율로 텐서의 값이 0이 되고, 0이 되지 않는 값들은 기존 값에 (1/(1-p))만큼 곱해져 커짐
# ex_ p=0.3이면 30%는 값이 0이 되고, 나머지 값들은 1/0.3 만큼 커짐
# 4
# forward() 함수는 모델이 학습 데이터를 입력받아서 순전파(forward propagation) 학습을 전파하는 함수(반드시 forward 이름으로 생성)
# 5
# PyTorch에서 사용하는 view는 NumPy의 reshape과 같은 역할로 tensor의 크기(shape)를 변경해주는 역할
# view(-1, 784)는 input_data를 (?, 784)의 크기로 변경하라는 의미로, 2차원 텐서로 변경하되 (?, 784)의 크기로 변경하라는 의미
# 6
# 활성화 함수를 지정할 때는 다음 두 가지 방법이 가능함
# F.relu() : forward() 함수에서 정의
# nn.ReLU() : __init()__ 함수에서 정의
# 간단히 보면 사용하는 위치의 차이인데, 근본적으로는 nn.functional.relu()와 nn.ReLU()의 차이임


## 모델을 학습시키기 전에 loss function, learning rate, optimizer에 대해 정의
learning_rate = 0.001
model = FashionDNN()
model.to(device)

criterion = nn.CrossEntropyLoss()   # classification 문제에서 주로 사용
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)  # 1
print(model)

# 1
# optimizer를 위한 gradient descent는 Adam을 사용

## DNN에 적용하여 model을 학습
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))   # 1
        labels = Variable(labels)

        outputs = model(train)  # 학습 데이터를 모델에 적용
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        '''
        if (count == 100):
            print("outputs : ")
            print(outputs.shape)    # [100, 10]
            print("labels : ")
            print(labels.shape)     # 100
            print("loss : ")
            print(loss)             # 0.6151
        '''

        if not (count % 50):    # count를 50으로 나누었을 때, 나머지가 0이 아니라면 실행
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images.view(100, 1, 28, 28))

                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total    # 5
            loss_list.append(loss.data) # 1'
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print("Iteration : {}, Loss : {}, Accuracy : {}%".format(count, loss.data, accuracy))

# 1
# Autograd는 자동 미분을 수행하는 PyTorch의 핵심 패키지로, 자동 미분에 대한 값을 저장하기 위해 테이프(tape)를 사용한다.
# forward 단계에서 테이프는 수행하는 모든 연산을 저장한다. 그리고 backward 단계에서 저장된 값들을 꺼내서 사용한다.
# 즉 Autograd는 Variable을 사용해서 backward를 위한 미분 값을 자동으로 계산해준다.
# 따라서 자동 미분을 계산하기 위해서는 torch.autograd 패키지 안에 있는 Variable을 이용해야 함