import torch
import torch.nn as nn

# VGG 논문에는 11, 13, 16, 19개의 layers를 갖는 모델이 나오기 때문에 각각의 구조에 맞게 구현
# A : VGG-11
# B : VGG-13
# D : VGG-16
# E : VGG-19
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# config에 따라 모델의 layer를 구성
def make_layer(config):
    layers = []
    in_planes = 3
    for value in config:
        if value == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_planes, value, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_planes = value   # 현재 층의 출력을 다음 층의 입력으로 넣어주기 위함
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, config, num_classes=10): # STL10에 맞춰 num_classes=10
        super(VGG, self).__init__()
        self.features = make_layer(config)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            #nn.Softmax(dim=1)  # 찾아보니 CrossEntropyLoss에서 softmax를 포함한다 함.
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# VGG Architecture 생성
def VGG11():
    return VGG(config=cfg['A'])
def VGG13():
    return VGG(config=cfg['B'])
def VGG16():
    return VGG(config=cfg['D'])
def VGG19():
    return VGG(config=cfg['E'])