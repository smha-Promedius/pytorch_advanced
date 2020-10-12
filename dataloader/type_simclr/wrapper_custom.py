#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : wrapper
# @Date : 2020-08-10-11-50
# @Project : simclr
# @Author : seungmin

import os, glob
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_augmentation.gaussian_blur import GaussianBlur

np.random.seed(0)

def make_datapath_list(phase="unlabeled"):
    root_path = './data/'
    target_path = os.path.join(root_path + phase + '/*.jpeg')

    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

class MyDataset(object):

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.labels = np.asarray([-1]*len(self.file_list))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path, target = self.file_list[index], self.labels
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed, target

class MyDataSetWrapper(object):

    def __init__(self, batch_size, pin_memory, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.pin_memory = pin_memory

    def get_data_loaders(self):

        data_augment = self._get_simclr_pipeline_transform()

        train_dataset = MyDataset(make_datapath_list(phase="unlabeled"), transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)

        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        print(num_train)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False,
                                  pin_memory=self.pin_memory)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False,
                                  pin_memory=self.pin_memory)

        print(len(train_loader), len(valid_loader))
        return train_loader, valid_loader

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
