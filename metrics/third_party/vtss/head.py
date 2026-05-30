import torch.nn as nn
import torch
from torchvision.ops import roi_pool, roi_align
from torch.nn import functional as F
import numpy as np
import math


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score
    
    
class VARHead(nn.Module):
    """MLP Regression Head for Video Action Recognition.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, out_channels=400, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        x = self.avg_pool(x)
        out = self.fc(x)
        return out

class MLPhead(nn.Module):
    def __init__(self, need_feature = True):
        super(MLPhead, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 400)
        self.fc_final = nn.Linear(400, 1)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.need_feature = need_feature

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.relu(self.fc3(x))
        x = self.drop(x)
        x = self.relu(self.fc4(x))

        feature = x
        x = self.drop(x)
        score = self.fc_final(x)
        if self.need_feature :
            return feature, score
        else:
            return score

class MLPhead_sem(nn.Module):
    def __init__(self, need_feature = True):
        super(MLPhead_sem, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 400)
        self.fc3 = nn.Linear(400, 1)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.need_feature = need_feature

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        feature = x
        x = self.drop(x)
        score = self.fc3(x)
        if self.need_feature :
            return feature, score
        else:
            return score
        
class MLPhead_image(nn.Module):
    def __init__(self, need_feature = True):
        super(MLPhead_image, self).__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 400)
        self.fc3 = nn.Linear(400, 1)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.need_feature = need_feature

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        feature = x
        x = self.drop(x)
        score = self.fc3(x)
        if self.need_feature :
            return feature, score
        else:
            return score


class Final_MLP(nn.Module):
    def __init__(self, in_channel = 1264):
        super(Final_MLP, self).__init__()
        self.fc0 = nn.Linear(in_channel, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.relu(self.fc3(x))
        return x
    
class IQAHead(nn.Module):
    """MLP Regression Head for IQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc_last = nn.Linear(self.hidden_channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score


class WeightFeatureFusion(nn.Module):
    def __init__(self):
        super(WeightFeatureFusion, self).__init__()
        self.fc = nn.Linear(400 * 2, 2)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        weights = self.fc(x)
        fused_feature = weights[:, 0].unsqueeze(dim=1) * x1 + weights[:, 1].unsqueeze(dim=1) * x2 
        
        return fused_feature   

class CrossGatingBlock(nn.Module):
    def __init__(self, x_features=400, num_channels=400, use_bias=True, use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.x_features = x_features
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate

        self.Conv_0 = nn.Linear(self.x_features, self.num_channels)
        self.Conv_1 = nn.Linear(self.num_channels, self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu1 = nn.GELU(approximate='tanh')
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)

    def forward(self, x, y):

        x = self.Conv_0(x)
        y = self.Conv_1(y)

        shortcut_y = y

        x = self.in_project_x(x)
        gx = self.gelu1(x)
        y = y * gx
        y = self.out_project_y(y)
        y = self.dropout1(y)

        y = y + shortcut_y
        return y