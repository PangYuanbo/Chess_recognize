import torch
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


class CustomImageNet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        p=0
        for label_folder in os.listdir(root_dir):
            for img in os.listdir(os.path.join(root_dir, label_folder)):
                img_path = os.path.join(root_dir, label_folder, img)
                self.images.append(img_path)
                self.labels.append(p)
            p += 1
    def __len__(self):
        return len(self.images)  # 1000

    def __getitem__(self, index):
        '''
        获取一个样本和标签
        :param index:
        :return:
        '''
        img_path = self.images[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        img=to_tensor(img)
        if self.transform:
            img = self.transform(img)

        return img, label
