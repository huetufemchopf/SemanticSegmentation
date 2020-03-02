import os
import json
import torch
import scipy.misc
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import glob
import random

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 225]
# randomangle = random.rand(0,1)
# print(randomangle)
# exit()


class DataLoaderSegmentation(Dataset):
    def __init__(self, args, mode="train"):

        super(DataLoaderSegmentation, self).__init__()

        self.mode = mode
        self.data_dir = args.data_dir


        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD),


        ])

        if mode =="train":


            self.img_files = glob.glob(os.path.join(self.data_dir,'train','img','*.png'))
            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'train', 'seg', '*.png'))

            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(self.data_dir,'train', 'seg',os.path.basename(img_path)))

        elif self.mode == 'val' or self.mode == 'test':

            self.img_files = glob.glob(os.path.join(self.data_dir, 'val', 'img', '*.png'))
            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'val', 'seg', '*.png'))
            #
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(self.data_dir, 'val','seg', os.path.basename(img_path)))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            filename = os.path.basename(self.img_files[index])
            mask_path = self.mask_files[index]
            data = self.transform(np.array(Image.open(img_path).convert('RGB')))
            label =Image.open(mask_path)
            label = np.array(label)
            label = torch.from_numpy(label)
            label = label.long()
            label = torch.squeeze(label)

            return data, label, filename

    def __len__(self):
        return len(self.img_files)


class DataLoaderPredict(Dataset):
    def __init__(self, args, mode="train"):

        super(DataLoaderPredict, self).__init__()

        self.mode = mode
        self.data_dir = args.data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])

        if mode =="train":


            self.img_files = glob.glob(os.path.join(self.data_dir,'*.png'))

        elif self.mode == 'val' or self.mode == 'test':

            self.img_files = glob.glob(os.path.join(self.data_dir, '*.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            filename = os.path.basename(self.img_files[index])
            data = self.transform(np.array(Image.open(img_path).convert('RGB')))

            return data, filename

    def __len__(self):
        return len(self.img_files)
