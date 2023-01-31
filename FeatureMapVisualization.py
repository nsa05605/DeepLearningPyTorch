import os
import time
import copy
import glob
import cv2
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torchvision.models.vgg16()
model.to(device)
model.eval()

### 특성 맵 시각화를 위한 함수를 정의
class LayerActivations:
    features = []
    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        # 파이토치는 매 계층마다 print를 사용하지 않더라도 hook 기능을 사용하여 각 계층의 활성화 함수 및 기울기 값을 확인할 수 있다.
        # register_forward_hook()의 목적은 순전파 중에 각 네트워크 모듈의 입력 및 출력을 가져오는 것

    def hook_fn(self, model, input, output):
        self.features = output.detach().numpy()

    def remove(self):
        self.hook.remove()

x = torch.Tensor([0,1,2,3]).requires_grad_()
y = torch.Tensor([4,5,6,7]).requires_grad_()
w = torch.Tensor([1,2,3,4]).requires_grad_()
z = x + y
o = w.matmul(z)
o.backward()
print(x.grad, y.grad, z.grad, w.grad, o.grad)