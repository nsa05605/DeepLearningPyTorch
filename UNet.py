### 인터넷 참고해서 U-Net 구조 구현
# 기존의 U-Net 구조는 padding을 사용하지 않지만, 구현할 때 padding=1로 구현함



import torch
import torch.nn as nn
import torchvision.transforms as transform

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()

        def CBR2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1):
            layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=output_channel),
                nn.ReLU()
            )
            return layer


        ### Contracting path
        # initial input = 572x572x1
        self.Conv1 = nn.Sequential(
            CBR2d(input_channels, 64, 3, 1),
            CBR2d(64, 64, 3, 1)
        )
        # 572x572x1 -> 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 568x568x64 -> 284x284x64
        self.Conv2 = nn.Sequential(
            CBR2d(64, 128, 3, 1),
            CBR2d(128, 128, 3, 1)
        )
        # 284x284x64 -> 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 280x280x128 -> 140x140x128
        self.Conv3 = nn.Sequential(
            CBR2d(128, 256, 3, 1),
            CBR2d(256, 256, 3, 1)
        )
        # 140x140x128 -> 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 136x136x256 -> 68x68x256
        self.Conv4 = nn.Sequential(
            CBR2d(256, 512, 3, 1),
            CBR2d(512, 512, 3, 1)
        )
        # 68x68x256 -> 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64x64x512 -> 32x32x512


        ### Bottle neck
        self.BottleNeck = nn.Sequential(
            CBR2d(512, 1024, 3, 1),
            CBR2d(1024, 1024, 3, 1)
        )
        # 32x32x512 -> 28x28x1024


        ### Expanding path
        # 채널 수를 감소시키며 Up-Conv
        self.UpConv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # 28x28x1024 -> 56x56x512

        # 이때 Conv4의 마지막 feature map을 가져오기 때문에 차원 수가 다시 1024로 증가
        self.ExConv1 = nn.Sequential(
            CBR2d(1024, 512, 3, 1),
            CBR2d(512, 512, 3, 1)
        )
        # 56x56x1024 -> 52x52x512

        self.UpConv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # 52x52x512 -> 104x104x256

        self.ExConv2 = nn.Sequential(
            CBR2d(512, 256, 3, 1),
            CBR2d(256, 256, 3, 1)
        )
        # 104x104x512 -> 100x100x256

        self.UpConv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # 100x100x256 -> 200x200x128

        self.ExConv3 = nn.Sequential(
            CBR2d(256, 128, 3, 1),
            CBR2d(128, 128, 3, 1)
        )
        # 200x200x256 -> 196x196x128

        self.UpConv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # 196x196x128 -> 392x392x64

        self.ExConv4 = nn.Sequential(
            CBR2d(128, 64, 3, 1),
            CBR2d(64, 64, 3, 1)
        )
        # 392x392x64 -> 388x388x64

        # 해당 논문에서 세포, 배경을 검출하는 것이 목표라 class_num = 1로 지정
        # Fully-Connected가 아닌 Fully-Convolutional layer
        self.FC = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        # 388x388x64 -> 388x388x1

    def forward(self, x):
        Dlayer1 = self.Conv1(x)
        out = self.pool1(Dlayer1)

        Dlayer2 = self.Conv2(out)
        out = self.pool2(Dlayer2)

        Dlayer3 = self.Conv3(out)
        out = self.pool3(Dlayer3)

        Dlayer4 = self.Conv4(out)
        out = self.pool4(Dlayer4)


        BottleNeck = self.BottleNeck(out)


        Ulayer1 = self.UpConv1(BottleNeck)
        ### 여기가 핵심
        # Contracting path 중 같은 단계의 feature map을 가져와 합치는 작업
        # UpConv 결과의 feature map size만큼 CenterCrop하여 Concat 연산
        # ex) 56x56x512 -> 56x56x1024

        cat1 = torch.cat((transform.CenterCrop((Ulayer1.shape[2], Ulayer1.shape[3]))(Dlayer4), Ulayer1), dim=1)
        ex_layer1 = self.ExConv1(cat1)

        Ulayer2 = self.UpConv2(ex_layer1)
        cat2 = torch.cat((transform.CenterCrop((Ulayer2.shape[2], Ulayer2.shape[3]))(Dlayer3), Ulayer2), dim=1)
        ex_layer2 = self.ExConv2(cat2)

        Ulayer3 = self.UpConv3(ex_layer2)
        cat3 = torch.cat((transform.CenterCrop((Ulayer3.shape[2], Ulayer3.shape[3]))(Dlayer2), Ulayer3), dim=1)
        ex_layer3 = self.ExConv3(cat3)

        Ulayer4 = self.UpConv4(ex_layer3)
        cat4 = torch.cat((transform.CenterCrop((Ulayer4.shape[2], Ulayer4.shape[3]))(Dlayer1), Ulayer4), dim=1)
        ex_layer4 = self.ExConv4(cat4)

        out = self.FC(ex_layer4)
        return out