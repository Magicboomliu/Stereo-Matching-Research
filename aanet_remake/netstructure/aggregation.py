import torch
import torch.nn as nn
import torch.nn.functional as F
# For check in-folder
import sys
sys.path.append("../")

from netstructure.deform import SimpleBottleneck,DeformBottleneck,DeformSimpleBottleneck

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


def conv1x1(in_planes, out_planes):
    """1x1 convolution, used for pointwise conv"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.LeakyReLU(0.2, inplace=True))


# Adaptive intra-scale aggregation & adaptive cross-scale aggregation
class AdaptiveAggregationModule(nn.Module):
    def __init__(self, num_scales, num_output_branches, max_disp,
                 num_blocks=1,
                 simple_bottleneck=False,
                 deformable_groups=2, # 分组卷积的group 
                 mdconv_dilation=2):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales  # Scale 的个数
        self.num_output_branches = num_output_branches  # 输出的Branch的个数
        self.max_disp = max_disp  # 候选的Disparity 的个数
        self.num_blocks = num_blocks  # ISA的blocks的个数

        self.branches = nn.ModuleList()  # 每个Branch是一个ISA和RSA的聚合网络

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for j in range(num_blocks):
                if simple_bottleneck:
                    branch.append(SimpleBottleneck(num_candidates, num_candidates))
                else:
                    branch.append(DeformSimpleBottleneck(num_candidates, num_candidates, modulation=True,
                                                         mdconv_dilation=mdconv_dilation,
                                                         deformable_groups=deformable_groups))

            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList() # fuse_layers的意思就是CSA模块的集合

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):

            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j: # 这里只是改变了维度， 没有改变H和W, 预测是后期直接想要使用Bilinear
                    self.fuse_layers[-1].append(
                        nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                kernel_size=1, bias=False),
                                      nn.BatchNorm2d(max_disp // (2 ** i)),
                                      ))
                elif i > j:
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        # 这里我认为是在采样， 而且SIZE 全部减半， H和W降低为正常的水平
                        layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** j),
                                                              kernel_size=3, stride=2, padding=1, bias=False),
                                                    nn.BatchNorm2d(max_disp // (2 ** j)),
                                                    nn.LeakyReLU(0.2, inplace=True),
                                                    ))

                    layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                          kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.BatchNorm2d(max_disp // (2 ** i))))
                    self.fuse_layers[-1].append(nn.Sequential(*layers))
       
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)  # 输入的x应该是一个 length长度为 Pyramid number 的 Feature Maps, 因此Branch也是这个数字

        for i in range(len(self.branches)):
            branch = self.branches[i]  # 分别取得每个对应的ISA branch
            for j in range(self.num_blocks): # 每个branch有N个Blocks
                dconv = branch[j] # 获得当前的 deform Residual Blocks
                x[i] = dconv(x[i])   # 当前尺寸进行 j (blocks nums of ISA) 次聚合

        if self.num_scales == 1:  # without fusions， 只有一个尺度， 可能性不大，一般设置是3个尺度，然后进行跨尺度聚合
            return x
        
        #  下面进行的是CSA模块，进行扩尺度的聚合
        x_fused = []
        for i in range(len(self.fuse_layers)):  #获得当前的尺度
            for j in range(len(self.branches)): # 用其他不同的尺度进行聚合
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))   # 这里是直接Identity或是downsample， 就像上面定义的，不需要Binlinear改变Size, 直接给个x就行了。
                else:
                    exchange = self.fuse_layers[i][j](x[j]) # 首先都经过模块，然后判断需不需要UP, 
                    if exchange.size()[2:] != x_fused[i].size()[2:]:# 如果H和W不相同，那么一定需要Binlinear Upsampling
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:],
                                                 mode='bilinear', align_corners=False) # UPsampe到当前尺寸的大小
                    
                    #既然是CSA一定都要不同尺度的加起来
                    x_fused[i] = x_fused[i] + exchange
        
        # 对每一个的输出进行relu
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


# Stacked AAModules
class AdaptiveAggregation(nn.Module):
    def __init__(self, max_disp, num_scales=3, num_fusions=6,
                 num_stage_blocks=1,  # Stage block : ISA部分使用的block的数目
                 num_deform_blocks=2, # ISA部分使用的deform Residual blocks的数目确定
                 intermediate_supervision=True, # intermediate_supervision ? Check here
                 deformable_groups=2, # 使用的分组卷积的个数
                 mdconv_dilation=2):     # 分组卷积使用的diliation
        super(AdaptiveAggregation, self).__init__()

        self.max_disp = max_disp
        self.num_scales = num_scales # 多少个尺度？
        self.num_fusions = num_fusions   #  AA Moudule的总个数
        self.intermediate_supervision = intermediate_supervision  # 是否需要中间的监督层

        fusions = nn.ModuleList() #  大END
        for i in range(num_fusions):
            if self.intermediate_supervision: # 每个AA模块都计算，贡献了损失，可以进行优化
                num_out_branches = self.num_scales
            else:
                num_out_branches = 1 if i == num_fusions - 1 else self.num_scales  # 不然的话，就使用1个branch的输出， 最后一个AA模块只是使用一个输出，就是最上面的那个输出
            
            # 前4个AA module 使用的是 普通的Simple bottleNeck, 后面2个使用是Deform的 BootleNeck
            if i >= num_fusions - num_deform_blocks:
                simple_bottleneck_module = False
            else:
                simple_bottleneck_module = True
            # 经过6个AA Moudle
            fusions.append(AdaptiveAggregationModule(num_scales=self.num_scales,
                                                     num_output_branches=num_out_branches,
                                                     max_disp=max_disp,
                                                     num_blocks=num_stage_blocks,
                                                     mdconv_dilation=mdconv_dilation,
                                                     deformable_groups=deformable_groups,
                                                     simple_bottleneck=simple_bottleneck_module))

        self.fusions = nn.Sequential(*fusions)

        # 最后在卷积一下，不知道这一步的意义何在？
        self.final_conv = nn.ModuleList()
        for i in range(self.num_scales):
            in_channels = max_disp // (2 ** i)

            self.final_conv.append(nn.Conv2d(in_channels, max_disp // (2 ** i), kernel_size=1))
            
            # 没有中间监督的话，输入就一个H/3 尺度， 直接break就行了
            if not self.intermediate_supervision:
                break

    def forward(self, cost_volume):
        assert isinstance(cost_volume, list)

        for i in range(self.num_fusions):
            fusion = self.fusions[i]
            cost_volume = fusion(cost_volume) # 对代价进行6次聚合

        # Make sure the final output is in the first position
        out = []  # 1/3, 1/6, 1/12
        for i in range(len(self.final_conv)):
            out = out + [self.final_conv[i](cost_volume[i])]

        return out
