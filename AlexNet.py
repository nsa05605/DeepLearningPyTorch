### AlexNet 논문 리뷰 후 PyTorch를 사용하여 구현

### AlexNet의 구조
# 층        종류        크기        특성맵     kernel size      stride      padding       activation function
# 입력    :           227x227     3(RGB)
# Layer1 : Conv       55x55        96          11x11           4         valid                ReLU
# Layer1 : MP         27x27        96           3x3            2         valid
# Layer1 : Conv       27x27        256          5x5            1         same                 ReLU
# Layer1 : MP         13x13        256          3x3            2         valid
# Layer1 : Conv       13x13        384          3x3            1         same                 ReLU
# Layer1 : Conv       13x13        384          3x3            1         same                 ReLU
# Layer1 : Conv       13x13        384          3x3            1         same                 ReLU
# Layer1 : MP          6x6         256          3x3            2         valid
# Layer1 : FC         4096         256                                                        ReLU
# Layer1 : FC         4096         96                                                         ReLU
# Layer1 : FC         1000         96                                                         Softmax

# Conv : Convolution
# MP : Max Pooling
# FC : Fully-Connected

# 추가로 FC layer에서 Dropout 50% 사용
# Data Augmentation(데이터 증식) 사용
# 정규화는 LRN(Local Response Normalization) 사용
# LRN은 뉴런의 출력값을 보다 경쟁적으로 만드는 정규화 기법

# Dataset : Fashion MNIST

### 라이브러리 불러오기
import os   # 파이썬을 이용해 파일을 복사하거나 디렉토리를 생성하고 특정 디렉토리 내의 파일 목록을 구하고자 할 때 사용
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision  # torchvision package : 컴퓨터 비전을 위한 유명 데이터셋, 모델 아키텍처, 이미지 변형 등을 포함
import torch.nn as nn   # nn : neural network (define class) attribute를 활용해 state를 저장하고 활용
import torch.optim as optim # 최적화 알고리즘
import torch.nn.functional as F # (define function) 인스턴스화 시킬 필요 없이 사용 가능
from PIL import Image
from torchvision import transforms, datasets    # transform : 데이터를 조작하고 학습에 적합하게 만듦
from torch.utils.data import Dataset, DataLoader
                                                # Dataset : 샘플과 정답(label)을 저장
                                                # DataLoader : Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감싼다.
                                                
### epoch, batch_size, device 정의
epochs = 10
batch_size = 512

device = ("cuda" if torch.cuda.is_available() else "cpu")
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']       # 총 10개의 클래스

print(torch.__version__)      # 1.12.1
print(device)                 # cuda


### Dataset 준비
transforms = transforms.Compose([ # Compose : transform 리스트 구성
    transforms.Resize(227), # 227x227 : input image(in AlexNet) but FashionMNIST's input image is 28x28
    transforms.ToTensor()]) # ToTensor : PIL image or numpy, ndarray를 tensor로 바꿈

training_data = datasets.FashionMNIST(
    root="data",    # data가 저장될 경로
    train=True,     # training data
    download=True,
    transform=transforms    # feature 및 label 변환(transformation) 저장
)

validation_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms
)

### DataLoader
# DataLoader는 데이터를 배치(batch) 단위로 모델에 넣어주는 역할
# 전체 데이터 중 일부 인스턴스를 뽑아(sample) 배치를 구성한다.
# (class) DataLoader(dataset, batch_size, shuffle, ...)
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True)

class Fashion_MNIST_AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            # 4D tensor : [number of kernels, input channels, kernel_width, kernel_height]
            # = 96x1x11x11
            # input size : 1x227x227
            # input size 정의 : (N, C, H, W) or (C, H, W)
            # W' = (W-F+2P)/S + 1
            # 55x55x96 feature map 생성 (55는 (227-11+1)/4)
            # 최종적으로 227 -> 55
            nn.ReLU(), # 96x55x55
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 55 -> (55-3+1)/2 = 26.5 = 27
            # 96x27x27 feature map 생성
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # kernel 수 = 48x5x5 (dropout을 사용했기 때문에 96/2=48) 형태의 256개
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)   # 27 -> 13
            # 256x13x13
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # 384x13x13
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # 384x13x13
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 256x6x6
        )
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):   # input size = 3x227x227
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)   # 64x4096x1x1
        out = out.view(out.size(0), -1) # 64x4096

        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out


### 모델 생성
model = Fashion_MNIST_AlexNet().to(device)  # to()로 모델에 gpu 사용
criterion = F.nll_loss  # nll_loss : negative log likelihood loss
optimizer = optim.Adam(model.parameters())   # model(신경망) 파라미터를 optimizer에 전달해줄 때, nn.Module의 parameters() 메소드를 사용

### 모델의 Summary
from torchsummary import summary as summary_
summary_(model, (1,227,227), batch_size, device)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # enumerate() : 인덱스와 원소로 이루어진 튜플(tuple)을 만들어줌
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()   # 항상 backpropagation 하기 전에 미분(gradient)을 zero로 만들어주고 시작해야 함
        output = model(data)
        loss = criterion(output, target) # criterion = loss_fn
        loss.backward() # Computes the gradient of currnet tensor w.r.t. graph leaves
        optimizer.step()    # step() : update parameters
        if (batch_idx + 1) % 30 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(training_loader.dataset),
                100. * batch_idx / len(training_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)   # -> mean
        print("\nTest Set : Average loss: {:.4f}, Accuracy: {}/{} ){:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('='*50)

print("Start Training")



for epoch in range(1, epochs + 1):
    print("Epochs : {}".format(epoch))
    train(model, device, training_loader, optimizer, epoch)
    test(model, device, validation_loader)
