import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import csv
import cv2
import os
class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=4, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        self.index = index
        for i in range(len(index)):
            self.mos[i] = np.load(features_dir + str(self.index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.length[idx], self.label[idx], self.index[idx]
        return sample
