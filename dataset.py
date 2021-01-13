# from __future__ import print_function
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import warnings
import os
import os.path
import numpy as np
import pandas as pd
import torch
import codecs
import zipfile
import random
random.seed(0)

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, makedir_exist_ok, verify_str_arg, check_integrity


def get_cifar_dataset(dataset, datapath, noise_mode, noise_rate, get_clean_indicator=False):
    if dataset=='cifar10':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(datapath, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(datapath, train=False, transform=test_transform)

    elif dataset=='cifar100':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        train_dataset = datasets.CIFAR100(datapath, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(datapath, train=False, transform=test_transform)

    # add noise
    if noise_mode in ['sym', 'asym', 'dependent']:    
        label_noisy = list(pd.read_csv(os.path.join(datapath, 'label_noisy', noise_mode+str(noise_rate)+'.csv'))['label_noisy'].values.astype(int))
        train_dataset.targets = label_noisy
        
        label_clean = list(pd.read_csv(os.path.join(datapath, 'label_noisy', noise_mode+str(noise_rate)+'.csv'))['label'].values.astype(int))
        indicator_clean = np.array([label_noisy[i]==label_clean[i] for i in range(len(label_noisy))])
        
        
    elif noise_mode=='openset' and dataset=='cifar10':
        # replace part of CIFAR-10 images with CIFAR-100 images
        cifar100_dataset = datasets.CIFAR100('data/CIFAR100', train=True, download=True) # datapath?
        index1 = np.random.choice(len(train_dataset), int(len(train_dataset)*noise_rate), replace=False)
        index2 = np.random.choice(len(train_dataset), int(len(train_dataset)*noise_rate), replace=False)
        train_dataset.data[index1] = cifar100_dataset.data[index2]
        
        indicator_clean = np.array([True for i in range(len(train_dataset))])
        indicator_clean[index1] = False
    
    else:
        raise ValueError('Unknown noise mode: {}.'.format(noise_mode))
    
    if get_clean_indicator:
        return train_dataset, test_dataset, indicator_clean
    else:
        return train_dataset, test_dataset
    


class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, num_per_class=-1):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)

        if mode=='train':
            flist = os.path.join(root, "annotations/noisy_train.txt")
        if mode=='val':
            flist = os.path.join(root, "annotations/clean_val.txt")
        if mode=='test':
            flist = os.path.join(root, "annotations/clean_test.txt")

        self.impaths, self.targets = self.flist_reader(flist)
        
        if num_per_class>0:
            impaths, targets = [], []
            num_each_class = np.zeros(14)
            indexs = np.arange(len(self.impaths))                            
            random.shuffle(indexs)
            
            for i in indexs:
                if num_each_class[self.targets[i]]<num_per_class:
                    impaths.append(self.impaths[i])
                    targets.append(self.targets[i])
                    num_each_class[self.targets[i]]+=1
                    
            self.impaths, self.targets = impaths, targets
            print('#samples/class: {};\n#total samples: {:d}\n'.format([int(i) for i in num_each_class], int(sum(num_each_class))))

#         # for quickly ebug
#         self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]


    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.targets[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.impaths)

    def flist_reader(self, flist):
        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impaths.append(self.root + '/' + row[0])
                targets.append(int(row[1]))
        return impaths, targets
