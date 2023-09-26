"""Extracting Video Spatial Features using model-based transfer learning"""

from argparse import ArgumentParser
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import h5py
import numpy as np
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def VideoDataset_BVQA_2022(video_data):
    """Read data from the original dataset for feature extraction"""

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    video_length = video_data.shape[0]
    video_channel = video_data.shape[1]
    video_height = video_data.shape[2]
    video_width = video_data.shape[3]
    content = torch.zeros([video_length, video_channel, video_height, video_width])

    for frame_idx in range(video_length):
        frame_video = video_data[frame_idx]
        frame_video = transform(frame_video)
        content[frame_idx] = frame_video
    sample = {'content': content}
    return sample

class CNNModel_Spatial(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self, model='ResNet-50'):
        super(CNNModel_Spatial, self).__init__()
        if model == 'SpatialExtractor':
            print("use SpatialExtractor")
            from SpatialExtractor.get_spatialextractor_model import make_spatial_model
            model = make_spatial_model()
            self.features = nn.Sequential(*list(model.module.backbone.children())[:-2])
        else:
            print("use default ResNet-50")
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)

def get_spatial_features(extractor,video_data, frame_batch_size=64, model='ResNet-50', device='cuda'):
    """feature extraction"""
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    while frame_end < video_length:
        batch = video_data[frame_start:frame_end].to(device)
        features_mean, features_std = extractor(batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        frame_end += frame_batch_size
        frame_start += frame_batch_size

    last_batch = video_data[frame_start:video_length].to(device)
    features_mean, features_std = extractor(last_batch)
    output1 = torch.cat((output1, features_mean), 0)
    output2 = torch.cat((output2, features_std), 0)
    output = torch.cat((output1, output2), 1).squeeze()

    if output.ndim == 1:
        output = output.unsqueeze(0)

    return output

if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting Video Spatial Features using model-based transfer learning')
    parser.add_argument("--seed", type=int, default=20230517)
    parser.add_argument('--database', default='LIVE-VQC', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='SpatialExtractor', type=str,
                        help='which pre-trained model used (default: ResNet-50)')
    parser.add_argument('--frame_batch_size', type=int, default=1,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument("--ith", type=int, default=0, help='start video id')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = 'KoNViD-1k/'  # videos dir, e.g., ln -s /xxx/KoNViD-1k/ KoNViD-1k
        features_dir = 'CNN_features_KoNViD-1k/SpatialFeature/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/'
        features_dir = 'CNN_features_CVD2014/SpatialFeature/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/'
        features_dir = 'CNN_features_LIVE-Qualcomm/SpatialFeature/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/'
        features_dir = 'CNN_features_LIVE-VQC/SpatialFeature/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'YouTube-UGC':
        videos_dir = 'YouTube_UGC/'
        features_dir = 'CNN_features_YouTube-UGC/SpatialFeature/'
        datainfo = 'data/YouTube-UGCinfo.mat'
    if args.database == 'LSVQ':
        videos_dir = 'LSVQ/'
        features_dir = 'CNN_features_LSVQ/SpatialFeature/'
        datainfo = 'data/LSVQinfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = Info['widths'][0, :]
    height = Info['heights'][0, :]
    dataset = VideoDataset_BVQA_2022(videos_dir, video_names, scores, video_format, width, height)

    max_len = 0
    min_len = 100000
    for i in range(args.ith, len(dataset)):
        start = time.time()
        current_data = dataset[i]
        print('Video {} : length {}'.format(i, current_data['video'].shape[0]))
        if max_len < current_data['video'].shape[0]:
            max_len = current_data['video'].shape[0]
        if min_len > current_data['video'].shape[0]:
            min_len = current_data['video'].shape[0]
        features = get_spatial_features(current_data['video'], args.frame_batch_size, args.model, device)
        np.save(features_dir + str(i) + '_' + args.model +'_last_conv', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_data['score'])
        end = time.time()
        print('{} seconds'.format(end-start))
    print('Max length: {} Min length: {}'.format(max_len,  min_len))
