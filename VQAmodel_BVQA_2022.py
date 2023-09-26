import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from CNNfeature_Spatial_BVQA_2022 import CNNModel_Spatial,VideoDataset_BVQA_2022,get_spatial_features
from CNNfeature_Motion_BVQA_2022 import get_motion_features

class BVQA_2022(nn.Module):
    def __init__(self, scale={'K': 1}, m={'K': 0},
                 simple_linear_scale=False, input_size=4608, reduced_size=128, hidden_size=32):
        super(BVQA_2022, self).__init__()
        self.hidden_size = hidden_size
        mapping_datasets = scale.keys()

        self.dimemsion_reduction = nn.Linear(input_size, reduced_size)
        self.feature_aggregation = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.regression = nn.Linear(hidden_size, 1)
        self.bound = nn.Sigmoid()
        self.nlm = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1))  # 4 parameters
        self.lm = nn.Sequential(OrderedDict([(dataset, nn.Linear(1, 1)) for dataset in mapping_datasets]))

        torch.nn.init.constant_(self.nlm[0].weight, 2 * np.sqrt(3))
        torch.nn.init.constant_(self.nlm[0].bias, -np.sqrt(3))
        torch.nn.init.constant_(self.nlm[2].weight, 1)
        torch.nn.init.constant_(self.nlm[2].bias, 0)
        for p in self.nlm[2].parameters():
            p.requires_grad = False
        for d, dataset in enumerate(mapping_datasets):
            torch.nn.init.constant_(self.lm._modules[dataset].weight, scale[dataset])
            torch.nn.init.constant_(self.lm._modules[dataset].bias, m[dataset])

        if simple_linear_scale:
            for p in self.lm.parameters():
                p.requires_grad = False

    def forward(self,spatial_extractor,motion_extractor,video_data, input_length):
        fake_current_data = VideoDataset_BVQA_2022(video_data[0::2])
        fake_current_video = fake_current_data['content']

        fake_spatial_features = get_spatial_features(spatial_extractor,fake_current_video,1, 'cuda')
        if fake_spatial_features.ndim == 1:
            fake_spatial_features = fake_spatial_features.unsqueeze(0)
        fake_motion_features = get_motion_features(motion_extractor, video_data.permute((0, 2, 3, 1)).float().cpu())
        spatial_index = torch.linspace(0, (fake_spatial_features.shape[0] - 1),
                                       fake_motion_features.shape[0]).long().to('cuda')
        fake_spatial_features = torch.index_select(fake_spatial_features, 0, spatial_index)
        fake_fusion_features = torch.cat((fake_spatial_features, fake_motion_features), dim=1)
        relative_score, mapped_score, aligned_score = [], [], []
        fake_fusion_features = fake_fusion_features.unsqueeze(0)

        x = self.dimemsion_reduction(fake_fusion_features)  # dimension reduction
        x, _ = self.feature_aggregation(x, self._get_initial_state(x.size(0), x.device))
        q = self.regression(x)  # frame quality
        relative_score.append(torch.zeros_like(q[:, 0]))
        mapped_score.append(torch.zeros_like(q[:, 0]))
        aligned_score.append(torch.zeros_like(q[:, 0]))
        for i in range(q.shape[0]):
            relative_score[0][i] = self._sitp(q[i, :int(input_length[i].item())])  # video overall quality
        relative_score[0] = self.bound(relative_score[0])
        mapped_score[0] = self.nlm(relative_score[0])  # 4 parameters
        for i in range(q.shape[0]):
            aligned_score[0][i] = self.lm._modules['K'](mapped_score[0][i])

        return relative_score[0], mapped_score[0], aligned_score[0]

    def _sitp(self, q, tau=12, beta=0.5):
        """subjectively-inspired temporal pooling"""
        q = torch.unsqueeze(torch.t(q), 0)
        qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)
        qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)
        l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
        m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
        n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
        m = m / n
        q_hat = beta * m + (1 - beta) * l
        return torch.mean(q_hat)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0
