import glob # 파일을 찾기 위한 라이브러리
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return self.t

    def __getitem__(self, idx):
        return torch.LongTensor([idx])

class CatDogDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        if train:
            self.cat_path = path + '/cat/train'
            self.dog_path = path + '/dog/train'
        else:
            self.cat_path = path + '/cat/test'
            self.dog_path = path + '/dog/test'

        # glob을 통해 끝이 .png인 파일들을 찾음
        self.cat_image_list = glob.glob(self.cat_path + '/*.png')
        self.dog_image_list = glob.glob(self.dog_path + '/*.png')

        self.transform = transform

        self.img_list = self.cat_image_list + self.dog_image_list
        self.class_list = [0]*len(self.cat_image_list) + [1]*len(self.dog_image_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":

    # dataset = SimpleDataset(t=5)
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=2,
    #                         shuffle=True,
    #                         drop_last=False)
    #
    # for epoch in range(2):
    #     print("epoch : {}".format(epoch))
    #     for batch in dataloader:
    #         print(batch)

    transform = transforms.Compose([transforms.ToTensor()])

    path = './data/cat_and_dog'
    dataset = CatDogDataset(path, train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=True,
                            drop_last=False)

    for epoch in range(2):
        print("epoch : {}".format(epoch))
        for batch in dataloader:
            img, label = batch
            print(img.size(), label)