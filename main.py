import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision
from torchvision import transforms as transforms

from VGG16 import VGG16

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)      # 2.0.0.dev20230116
print(device)                 # cuda

# VGG를 위한 transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
])
# STL10 dataset을 VGG architecture 기준으로 다운
trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
testset  = torchvision.datasets.STL10(root='./data', split='test' , download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class_name_STL10  = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

model = VGG16().to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
epochs = 10

#from torchsummary import summary as summary_
#summary_(model, (1,96,96), batch_size=batch_size, device=device)

def train(model, device, train_loader, optimizer, epochs):
    model.train()
    # data는 학습 데이터, target은 라벨을 의미
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # print(inputs.shape)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx) % 30 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():   # autograd engine을 꺼버림 -> 더 이상 gradient를 자동으로 트래킹 x
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)  # -> mean
            print("\nTest Set : Average loss: {:.4f}, Accuracy: {}/{} ){:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            print('=' * 50)

if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        print("Epochs : {}".format(epoch))
        train(model, device, train_loader=trainloader, optimizer=optimizer, epochs=epoch)
        test(model, device, test_loader=testloader)