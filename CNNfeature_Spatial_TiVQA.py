import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
import time
from torchvision import  models as torchmodels
import torch.nn as nn
from skimage.feature import local_binary_pattern
def VideoDataset(video_data):
    LBP = torch.Tensor()
    LBP = LBP.to('cuda')
    video_data_detach =video_data
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for i in range(video_data.shape[0]):
        img = cv2.cvtColor(np.uint8((video_data[i]*255).permute((1,2,0)).cpu().detach().numpy()), cv2.COLOR_BGR2GRAY)
        middle = local_binary_pattern(np.uint8(img), 10, 4)
        middle = torch.from_numpy(middle)
        middle = middle.to('cuda')
        middle = torch.unsqueeze(middle, 2)
        middle = torch.cat((middle, middle, middle), dim=2)
        middle = middle.unsqueeze(0)
        LBP = torch.cat((LBP, middle), 0)
    LBP = LBP.cpu().numpy()
    LBP = np.uint8(LBP)
    LBP = np.asarray(LBP)
    LBP = LBP.transpose(0, 3, 1, 2) / 255
    LBP = torch.from_numpy(LBP).float().to('cuda')
    video_length = video_data.shape[0]
    video_channel = video_data.shape[1]
    video_height = video_data.shape[2]
    video_width = video_data.shape[3]
    transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
    transformed_LBP = torch.zeros([video_length, video_channel, video_height, video_width])
    for frame_idx in range(video_length):
        frame = video_data_detach[frame_idx]
        frame_LBP = LBP[frame_idx]
        frame = transform(frame)
        frame_LBP = transform(frame_LBP)
        transformed_video[frame_idx] = frame
        transformed_LBP[frame_idx] = frame_LBP
    sample = {'video': transformed_video, 'LBP': transformed_LBP}
#
    return sample


def get_features(extractor,video_data,video_LBP, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    output3 = torch.Tensor().to(device)
    output4 = torch.Tensor().to(device)
    while frame_end < video_length:
        batch_video = video_data[frame_start:frame_end].to(device)
        batch_LBP = video_LBP[frame_start:frame_end].to(device)
        content_aware_mean_features, content_aware_std_features, texture_mean_features, texture_std_features = extractor(
            batch_video, batch_LBP)
        output1 = torch.cat((output1, content_aware_mean_features), 0).to(device)  # content_aware featuers
        output2 = torch.cat((output2, content_aware_std_features), 0).to(device)
        output3 = torch.cat((output3, texture_mean_features), 0).to(device)  # texture features
        output4 = torch.cat((output4, texture_std_features), 0).to(device)
        frame_end += frame_batch_size
        frame_start += frame_batch_size

    last_batch = video_data[frame_start:video_length].to(device)
    last_LBP = video_LBP[frame_start:frame_end].to(device)
    content_aware_mean_features, content_aware_std_features, texture_mean_features, texture_std_features = extractor(
        last_batch, last_LBP)
    output1 = torch.cat((output1, content_aware_mean_features), 0).to(device)
    output2 = torch.cat((output2, content_aware_std_features), 0).to(device)
    output3 = torch.cat((output3, texture_mean_features), 0).to(device)
    output4 = torch.cat((output4, texture_std_features), 0).to(device)
    output = torch.cat((output1, output2, output3, output4), 1).to(device)  # concate texture and content-aware features
    output = output.squeeze()

    return output

class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(torchmodels.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):
        for ii, model in enumerate(self.features):
            x1 = model(x1)
            x2 = model(x2)
            if ii == 7:
                content_aware_mean_features = nn.functional.adaptive_avg_pool2d(x1, 1)  # extract content-aware features
                texture_mean_features = nn.functional.adaptive_avg_pool2d(x2, 1)  # extract texture features
                content_aware_std_features = global_std_pool2d(x1)
                texture_std_features = global_std_pool2d(x2)
                return content_aware_mean_features, content_aware_std_features, texture_mean_features, texture_std_features

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)
