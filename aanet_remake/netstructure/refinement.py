import torch
import torch.nn as nn
import torch.nn.functional as F

# For in-folder check
import sys
sys.path.append("../")

from netstructure.deform import DeformConv2d
from netstructure.warp import disp_warp
from netstructure.feaures import BasicResidualBlock

# 普通的2d 卷积
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))

# 普通的 refinement, 加入图片的Feature
class StereoNetRefinement(nn.Module):
    def __init__(self):
        super(StereoNetRefinement, self).__init__()

        # Original StereoNet: left, disp
        self.conv = conv2d(4, 32)

        self.dilation_list = [1, 2, 4, 8, 1, 1] # 这里不是用的Shared weight, 而是重新计算的weight
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicResidualBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img=None):
        """Upsample low resolution disparity prediction to
        corresponding resolution as image size
        Args:
            low_disp: [B, H, W]
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
        """
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        # 首先使用双线性差值算出大约的Disparity.
        disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp * scale_factor  # scale correspondingly
     
       # 把disp和left_image放在一起卷积提取特征
        concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
        out = self.conv(concat)
        out = self.dilated_blocks(out) # 经过数个residual blocks
        # 得到残差模块
        residual_disp = self.final_conv(out)
       # 更新梯度
        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp

# 加入loss的 refinement, 加入图片的Feature和warp error
class StereoDRNetRefinement(nn.Module):
    def __init__(self):
        super(StereoDRNetRefinement, self).__init__()

        # Left and warped error
        in_channels = 6

        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicResidualBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]
        
        # 把error和left_image放在一起卷积提取特征
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        residual_disp = self.final_conv(out)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp
