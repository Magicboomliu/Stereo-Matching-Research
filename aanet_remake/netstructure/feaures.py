import torch
import torch.nn as nn
import torch.nn.functional as F

# For check in-folder
import sys
sys.path.append("../")

from netstructure.deform import DeformConv2d



# For Reference:
# out_channels = (n-k+2p)/s +1  
# n: input_channel  ,  K: convolation Layer, p: padding size, s :stride

# one-by-one convolution (conv + bacthnorm +leRelu), 默认有Bn+relu层
def conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))
# five by five convolution (conv + bacthnorm +leRelu)， 默认有一个BN+relu
def conv5x5(in_channels, out_channels, stride=2,
            dilation=1, use_bn=True):
    bias = False if use_bn else True
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)
    relu = nn.ReLU(inplace=True)
    if use_bn:
        out = nn.Sequential(conv,
                            nn.BatchNorm2d(out_channels),
                            relu)
    else:
        out = nn.Sequential(conv, relu)
    return out
# Used for StereoNet feature extractor(conv + bacthnorm +leRelu), 默认没有Bn+relu层
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False, leaky_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             relu)
    return conv

# The Residual Basic Block  for building the feature extraction Nets
class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride =1, downsample =None, groups =1,dilation =1, norm_layer = None, leaky_relu = True):
            super(BasicResidualBlock,self).__init__()
            # Make Sure there is a Normal Layer
            if  norm_layer is None:
                norm_layer = nn.BatchNorm2d
            # When stride !=1, downsample the input
            self.conv1 = conv3x3(in_planes=in_channels,out_planes=out_channels,stride=stride,dilation=dilation)
            self.bn1 = norm_layer(out_channels)
            self.relu = nn.LeakyReLU(0.2,inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels,out_channels,dilation=dilation)
            self.bn2 = norm_layer(out_channels)
            self.downsample = downsample
            self.stride = stride   #这里用来判断残差模块是否需要被downsample, 如果此时的stride不为1，改变了形状.

    def forward(self,x):
        residual_part = x
        # 3*3 conv + BN + leaky_relu
        out = self. conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)

        # judge whether residual part need to be downsampled or not
        if self.downsample is not None:
            residual_part = self.downsample(residual_part)
        
        out += residual_part
        out = self.relu(out)
        
        return out


class FeaturePyrmaid(nn.Module):
    def __init__(self, in_channel=32):
        super(FeaturePyrmaid, self).__init__()

        self.out1 = nn.Sequential(nn.Conv2d(in_channel, in_channel * 2, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

        self.out2 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel * 4, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 4, in_channel * 4, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

    def forward(self, x):
        # x: [B, 32, H, W]
        out1 = self.out1(x)  # [B, 64, H/2, W/2]
        out2 = self.out2(out1)  # [B, 128, H/4, W/4]

        return [x, out1, out2]

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128,
                 num_levels=3):
        # FPN paper uses 256 out channels by default
        super(FeaturePyramidNetwork, self).__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_levels):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Inputs: resolution high -> low
        assert len(self.in_channels) == len(inputs)

        # Build laterals
        laterals = [lateral_conv(inputs[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # Build outputs
        out = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return out

