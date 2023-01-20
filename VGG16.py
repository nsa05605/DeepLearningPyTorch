import torch
import torch.nn as nn
import torch.utils.data

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

def make_layer(config):
    layers = []
    in_planes = 3
    for value in config:
        if value == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_planes, value, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_planes = value
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, config, num_classes=10): # STL10은 num_classes=10
        super(VGG, self).__init__()
        self.features = make_layer(config)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
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