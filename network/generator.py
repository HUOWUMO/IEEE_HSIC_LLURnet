import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *


# 3D-空谱特征随机化
class Spa_Spe_Randomization(nn.Module):
    def __init__(self, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)  # 定义一个可学习的参数，并初始化

    def forward(self, x, ):
        N, C, L, H, W = x.size()
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)  
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  # 从batch中选择随机化均值和方差
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, L, H, W)

        return x, idx_swap


class Generator_3DCNN_SupCompress_pca(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], device=0, dim1=128, dim2=8):
        super().__init__()

        self.patch_size = imsize[0]

        self.n_channel = dim2
        self.n_pca = dim1

        # 2D_CONV
        self.conv_pca = nn.Conv2d(imdim, self.n_pca, 1, 1) 

        self.inchannel = self.n_pca

        # 3D_CONV
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=self.n_channel,
                               kernel_size=(3, 3, 3))

        # 3D空谱随机化
        self.Spa_Spe_Random = Spa_Spe_Randomization(device=device)

        # 
        self.conv6 = nn.ConvTranspose3d(in_channels=self.n_channel, out_channels=1, kernel_size=(3, 3, 3))

        # 2D_CONV
        self.conv_inverse_pca = nn.Conv2d(self.n_pca, imdim, 1, 1)

    def forward(self, x):
        x = self.conv_pca(x)

        x = x.reshape(-1, self.patch_size, self.patch_size, self.inchannel, 1)  # (256,48,13,13,1)转换输入size,适配Conv3d输入
        x = x.permute(0, 4, 3, 1, 2)  # (256,1,48,13,13)

        x = F.relu(self.conv1(x))

        x, idx_swap = self.Spa_Spe_Random(x)

        x = torch.sigmoid(self.conv6(x))

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, self.inchannel, self.patch_size, self.patch_size)

        x = self.conv_inverse_pca(x)
        return x
