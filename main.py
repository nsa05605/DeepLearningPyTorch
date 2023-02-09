### https://velog.io/@jarvis_geun/U-Net-%EC%8B%A4%EC%8A%B5 기반으로 U-Net 실습하기

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from tqdm.notebook import tqdm

from UNet import UNet

'''
### GPU 설정하기
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)


### 파일 시스템
# 폴더 경로
data_dir = 'data/archive/cityscapes_data/'

# data_dir의 경로와 train을 결합하여 train_dir에 저장
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')

# train_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 train_fns에 저장
train_fns = os.listdir(train_dir)
val_fns   = os.listdir(val_dir)

# print(len(train_fns), len(val_fns)) # 2975 500


### 샘플 이미지 검색
# 경로를 지정했으므로 이 경로의 샘플 이미지를 확인(생략 가능)
sample_image_fp = os.path.join(train_dir, train_fns[0])

# PIL 라이브러리의 Image 모듈을 사용하여 sample_image_fp를 불러옴
sample_image = Image.open(sample_image_fp)
plt.imshow(sample_image)
plt.show()


### Output label 정의하기
num_items = 1000

# 0~255 사이의 숫자를 3*num_items번 랜덤하게 뽑기
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)    # (1000, 3)

num_classes = 10

# k-means clustering 알고리즘을 사용하여 label_model에 저장
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)
'''

'''
# 이전에 샘플 이미지를 확인해보면, original image와 labeled image가 연결되어 있기 때문에 이를 분리해줌
def split_image(image):
    image = np.array(image)

    # 이미지의 크기가 256 x 512 였는데, 이를 original image와 labeled image로 분리하기 위해 리스트로 슬라이싱
    # 분리된 이미지를 각각 cityscape(=origin image)와 label(=labeled image)에 저장
    cityspace, label = image[:, :256, :], image[:, 256:, :]
    return cityspace, label

cityscape, label = split_image(sample_image)

label_class = label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
fix, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(cityscape)
axes[1].imshow(label)
axes[2].imshow(label_class)

plt.show()
'''

### 데이터셋 정의하기
class CityspaceDataset(Dataset):
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        image_fn = self.image_fns[idx]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp)
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long()
        cityscape = self.transform(cityscape)
        return cityscape, label_class

    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

'''
dataset = CityspaceDataset(train_dir, label_model)
print(len(dataset)) # 2975

cityscape, label_class = dataset[0]
print(cityscape.shape)      # [3,256,256]
print(label_class.shape)    # [256,256]
'''

'''
### U-Net 모델 정의하기
model = UNet(input_channels=3, num_classes=num_classes)

data_loader = DataLoader(dataset, batch_size=16)
print(len(dataset), len(data_loader))   # 2975 186

X, Y = next(iter(data_loader))
print(X.shape)  # [16, 3, 256, 256]
print(Y.shape)  # [16, 256, 256]

Y_pred = model(X)
print(Y_pred.shape) # [16, 10, 256, 256]
'''

def main():

    print("main started")

    ### GPU 설정하기
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)

    ### 파일 시스템
    # 폴더 경로
    data_dir = 'data/archive/cityscapes_data/'

    # data_dir의 경로와 train을 결합하여 train_dir에 저장
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # train_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 train_fns에 저장
    train_fns = os.listdir(train_dir)
    val_fns = os.listdir(val_dir)

    ### Output label 정의하기
    num_items = 1000

    # 0~255 사이의 숫자를 3*num_items번 랜덤하게 뽑기
    color_array = np.random.choice(range(256), 3 * num_items).reshape(-1, 3)
    print(color_array.shape)  # (1000, 3)

    num_classes = 10

    # k-means clustering 알고리즘을 사용하여 label_model에 저장
    label_model = KMeans(n_clusters=num_classes)
    label_model.fit(color_array)

    ### 모델 학습하기
    batch_size = 16
    epochs = 10
    lr = 0.01
    num_classes = 10

    dataset = CityspaceDataset(train_dir, label_model)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model = UNet(input_channels=3, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step_losses = []
    epoch_losses = []

    for epoch in tqdm(range(epochs)):
        print("epoch {} started".format(epoch))
        epoch_loss = 0

        for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            #print("epoch loss : {}".format(epoch_loss))
        epoch_losses.append(epoch_loss/len(data_loader))

    print(len(epoch_losses))
    print(epoch_losses)

if __name__ == '__main__':
    main()