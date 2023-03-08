# https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html 기반으로 Mask R-CNN 실습하기

import os

import torchvision.models.detection
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
import torchvision.transforms as transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from tqdm.notebook import tqdm



### 데이터셋 정의
# 여기에서는 Penn-Fudan Database for Pedestrian Detection and Segmentation 데이터셋을 사용
# __getitem__ 메소드에서 반환해야 하는 것들
# 이미지 : PIL(Python Image Library) 이미지의 크기(H,W)
# 대상 : 다음의 필드를 포함하는 사전 타입
# boxes(FloatTensor[N,4]) : N개의 bounding box의 좌표(x0,x1,y0,y1) x는 0~W, y는 0~H
# labels(Int64Tensor[N]) : bounding box의 label 정보(0은 항상 배경의 class)
# image_id(Int64Tensor[1]) : 이미지 구분자. 데이터셋의 모든 이미지 간에 고유한 값이어야 함.
# area(Tensor[N]) : bounding box의 면적
# iscrowd(UInt8Tensor[N]) : 이 값이 참일 경우 평가에서 제외
# (선택적) masks(UInt8Tensor[N,H,W]) : N개의 객체마다의 분할 마스크 정보
# (선택적) keypoints(FloatTensor[N,K,3]) : N개의 객체마다의 keypoint 정보


class PedestrianDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        # 모든 이미지 파일을 읽고, 정렬하여 이미지와 분할 마스크 정렬을 확인
        self.images = list(sorted(os.listdir(os.path.join(path, "PNGImages"))))
        self.masks  = list(sorted(os.listdir(os.path.join(path, "PedMasks"))))

    def __getitem__(self, idx):
        # 이미지와 마스크를 읽어옴
        image_path = os.path.join(self.path, "PNGImages", self.images[idx])
        mask_path  = os.path.join(self.path, "PedMasks", self.masks[idx])
        image = Image.open(image_path).convert("RGB")
        # 분할 마스크는 RGB로 변환 x
        mask = Image.open(mask_path)
        # numpy 배열을 PIL 이미지로 변환
        mask = np.array(mask)
        # instances는 다른 색들로 인코딩 되어 있음
        # obj_ids는 해당 mask에 어떤 instance들이 있는지 알기위해 놓은듯?
        obj_ids = np.unique(mask)
        # 첫 번째 id는 배경이라 제거
        obj_ids = obj_ids[1:]

        # 컬러 인코딩된 mask를 binary mask 세트로 나눔
        masks = mask == obj_ids[:, None, None]

        # 각 마스크의 bounding box 좌표
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 모든 것을 torch.Tensor 타입으로 변환합니다
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 객체 종류는 한 종류만 존재합니다(역자주: 예제에서는 사람만이 대상입니다)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 모든 인스턴스는 군중(crowd) 상태가 아님을 가정합니다
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id   # mask가 속한 이미지 idx
        target["area"] = area           # box의 면적
        target["iscrowd"] = iscrowd     # 물체가 너무 작은데 많아서 하나의 군집으로 박스를 처리하여 레이블링 했는지에 관한 여부

        if self.transform is not None:
            image, target = self.transform(image, target)
            # image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)



## PennFudan 데이터셋을 위한 instance segmentation 모델
# instance segmentation mask도 계산해야 하기 때문에 Mask R-CNN을 사용
# 미리 학습된 모델로부터 미세 조정(fine tuning) 방법으로 모델 정의
def get_model_instance_segmentation(num_classes):
    # COCO에서 미리 학습된 instance segmentation model을 불러옴
    # 보니까 weight="DEFAULT"로 하면 COCO_V1에서 pre-train 한 모델을 가져오는 듯
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 분류기를 새로운 것으로 교체하는데, 이때 num_classes는 객체수 + 1(배경)

    # 분류를 위한 입력 특징 차원을 얻음
    # 분류기에서 사용할 입력 특징의 차원 정보를 얻음
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿈
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 분류기를 위한 입력 특징들의 차원을 얻음
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 마스크 예측기를 새로운 것으로 바꿈
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model



## 모든 것을 하나로 합치기
def get_transform(train):
    transforms = []
    transforms.append(transform.PILToTensor())
    transforms.append(transform.ConvertImageDtype(torch.float))
    if train:
        # 학습시 50% 확률로 좌우 반전
        transforms.append(transform.RandomHorizontalFlip(0.5))
    return transform.Compose(transforms)


### 학습(train)과 검증(validation)을 수행
def train():
    batch_size = 2
    epochs = 20
    lr = 0.005

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    step_losses = []
    epoch_losses = []

    for epoch in tqdm(range(epochs)):
        print("epoch {} started".format(epoch))
        epoch_loss = 0

        for X, Y in tqdm(data_loader_train, total=len(data_loader_train), leave=False):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        lr_scheduler.step()
        epoch_losses.append(epoch_loss / len(data_loader_train))
        print("epoch {}'s loss : {}".format(epoch, (epoch_loss/len(data_loader_train))))

    print(len(epoch_losses))
    print(epoch_losses)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)
    plt.show()

    torch.save(model.state_dict(), model_dir + model_name)



def main():
    train()

if __name__ == "__main__":
    print("main started")
    # 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 합니다
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # 우리 데이터셋은 두 개의 클래스만 가집니다 - 배경과 사람
    num_classes = 2
    # 데이터셋과 정의된 변환들을 사용합니다
    path = "./data/PennFudanPed/"
    model_dir = 'models/'
    model_name = "MaskRCNN.pth"

    dataset_train = PedestrianDataset(path, get_transform(train=True))
    dataset_test = PedestrianDataset(path, get_transform(train=False))

    # 데이터셋을 학습용과 테스트용으로 나눕니다(역자주: 여기서는 전체의 30개를 테스트에, 나머지를 학습에 사용합니다)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-30])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-30:])

    # # 데이터 로더를 학습용과 검증용으로 정의합니다
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    # )
    #
    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    # )
    # 데이터 로더를 학습용과 검증용으로 정의합니다
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False
    )

    # 도움 함수를 이용해 모델을 가져옵니다
    model = get_model_instance_segmentation(num_classes)

    # 모델을 GPU나 CPU로 옮깁니다
    model.to(device)

    main()

    print("That's it!")