### VGG-16 모델 학습하는 내용

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision
from torchvision import transforms as transforms
from torchsummary import summary as summary_

from model_VGG import VGG16


'''
# device = ("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.__version__)  # 2.0.0.dev20230116
# print(device)  # cuda
#
# # VGG를 위한 transform
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.RandomCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
#
# # STL10 dataset을 VGG architecture 기준으로 다운
# trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
# testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
#
# class_name_STL10 = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
#
# model = VGG16().to(device)
# summary_(model, (3, 224, 244), device=device)
#
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# def Train(model, device, train_loader, optimizer, epochs):
#
#     for epoch in range(epochs+1):
#         running_loss = 0
#         print("Epochs : {}".format(epoch))
#
#         model.train()
#         # data는 학습 데이터, target은 라벨을 의미
#         for batch_idx, (data, target) in enumerate(train_loader, 0):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#
#             # print(inputs.shape)
#             output = model(data)
#             output = model.forward(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             if batch_idx % 30 == 0:
#                 print(f'[{epoch + 1}, {batch_idx +1:5d}] loss: {running_loss / 30:.3f}')
#                 running_loss = 0.0
#
# def Test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():   # autograd engine을 꺼버림 -> 더 이상 gradient를 자동으로 트래킹 x
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.max(1, keepdim=True)[1]
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             test_loss /= len(test_loader.dataset)  # -> mean
#             print("\nTest Set : Average loss: {:.4f}, Accuracy: {}/{} ){:.0f}%)\n".format(
#                 test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#             print('=' * 50)
#
# if __name__ == "__main__":
#
#     Train(model, device, train_loader=trainloader, optimizer=optimizer, epochs=10)
#     Test(model, device, test_loader=testloader)
'''

def Train(log_interval, model, device, trainloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device) # reduction='sum' 을 해주면 합의 값을 return
    for batch_idx, (data, target) in enumerate(trainloader, 0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset_train),
                       100. * batch_idx / len(trainloader), running_loss / log_interval))
            running_loss = 0.0


def Test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion =  nn.CrossEntropyLoss(reduction='sum').to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss +=  loss.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset_train)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset_train),
        100. * correct / len(test_loader.dataset_train)))

def Validation(model, device, testloader, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def main():
    epochs = 10
    learning_rate = 0.0001
    batch_size = 128
    log_interval = 4   # running_loss를 확인하기 위한 parameter

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)  # 2.0.0.dev20230116
    print(device)  # cuda

    # VGG를 위한 transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # STL10 dataset을 VGG architecture 기준으로 다운
    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    class_name_STL10 = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    model = VGG16().to(device)
    summary_(model, (3, 224, 244), device=device)   # 모델의 구조를 출력

    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs+1):
        print("Epochs : {}".format(epoch))
        Train(log_interval, model, device, trainloader, optimizer, epoch)   # 학습
        #Test(model, device, testloader)

    Validation(model, device, testloader, class_name_STL10) # 평가

    torch.save(model.state_dict(), './models')  # 학습한 모델을 저장

if __name__ == '__main__':
    main()