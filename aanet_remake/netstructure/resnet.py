import torch.nn as nn
# For check in-folder
import sys
sys.path.append("../")

from netstructure.deform import DeformBottleneck

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Bottleneck Network 用来处理较为深层次的ResNet, 这里channels扩大为planes的 4倍，stride 默认为1， 不改变大小
class Bottleneck(nn.Module):
    expansion = 4   # What is this used for?
    __constants__ = ['downsample'] 

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups   # 分组卷积？ 这里是 Channel 的总宽度

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)   # Change Channels
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # Here Use a stride to downsample
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 扩大为 extentsion的倍数
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample       # downsample层， 后期可以在 make_layer时候规定
        self.stride = stride

    def forward(self, x):
        identity = x     # X

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)   

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)  

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AANetFeature(nn.Module):
    def __init__(self, in_channels=32,
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,  # 分组卷积，各个组处理的深度为64、
                 feature_mdconv=True,  # feature_mdconv: 代表是否使用 deform convoluation 在特征提取阶段
                 norm_layer=None):
        super(AANetFeature, self).__init__()
      
        # 写法一：如果在一个网络里面没有声明 BN， 第一步就是定义一个BN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        layers = [3, 4, 6]  # ResNet-40  ------> 使用的Blocks的层数

        # self.inplanes = 64
        self.inplanes = in_channels
        self.dilation = 1  # 这里设置为1 ，感觉和没有设置差不多

        self.groups = groups   # 分组打算分多少组
        self.base_width = width_per_group # 每个组处理的channel的带宽

        stride = 3   # Step的number
        
        # 第一次的卷积特征提取操作：
        # INPUT: (B,3,H,W) : the left and right image : kernel =7, stride =3 
        # (n-7+2*3) /3 +1 -----> (B, inplanes,H/3,W/3)

        self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=stride,
                                             padding=3, bias=False),
                                   nn.BatchNorm2d(self.inplanes),
                                   nn.ReLU(inplace=True))  # H/3

        self.layer1 = self._make_layer(Bottleneck, in_channels, layers[0])  # H/3   C = in_channels *4
        self.layer2 = self._make_layer(Bottleneck, in_channels * 2, layers[1], stride=2)  # H/6   C=in_channels *8

        block = DeformBottleneck if feature_mdconv else Bottleneck   # 第三层判断是否需要使用 Deform Conv
        self.layer3 = self._make_layer(block, in_channels * 4, layers[2], stride=2)  # H/12  ， C = in_channels * 16
        

        # 初始化一些参数：
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # I think here blocks is the : the numbers of blocks use here:
    # block： Certain ResNet Block
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:   # 使用空洞卷积？
            self.dilation *= stride 
            stride = 1
        # 如果stride！=1， 或是输入和输出不相同的化： 证明在Blocks内部有降采样的过程，INPUTs 也需要进行降采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 规定降采样的的结构，直接用1*1 卷积和BN层改变channels的个数
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # 记录当前的layers, stride： 只有第一次改变了维度，而且downsamle 了 ，其他都不变，因此需要单独写出来
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer)) # 这里输出的channels 是planes的4倍 ： 4* planes
        self.inplanes = planes * block.expansion  # 更新输入的维度
        
        for _ in range(1, blocks):   # 主题这里没有用stride: 也就是说这里没有改变输出的大小，也没有downsample  因此是直接加一起就行了
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## X is the image: [B,3,H,W]
        x = self.conv1(x)  # [B,inplanes,H/3,W/3]
        layer1 = self.layer1(x)  # [B,inplanes*4,H/3,W/3]
        layer2 = self.layer2(layer1) # [B,inplanes*8,H/3,W/3]
        layer3 = self.layer3(layer2) # [B,inplanes*16,H/3,W/3]

        return [layer1, layer2, layer3] # 返回特征金字塔
