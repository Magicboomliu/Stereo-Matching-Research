import torch
import torch.nn as nn
import torch.nn.functional as F

# For in-folder check
import sys
sys.path.append("../")
# 分别引入 Feature Extractors ， cost Volume Compuation, Cost aggregation, Disparity estimation, 以及 Refinement 操作 
from netstructure.feaures import FeaturePyramidNetwork,FeaturePyrmaid
from netstructure.resnet import AANetFeature
from netstructure.cost import CostVolume,CostVolumePyramid
from netstructure.aggregation import AdaptiveAggregation
from netstructure.estimation import DisparityEstimation
from netstructure.refinement import StereoDRNetRefinement,StereoNetRefinement


class AANet(nn.Module):
    def __init__(self, max_disp,
                 num_downsample=2,
                 feature_type='aanet',
                 no_feature_mdconv=False,  # Feature Extractor的部分是否需要进行deform conv
                 feature_pyramid=False,  # 是否使用FPN
                 feature_pyramid_network=False,
                 feature_similarity='correlation',  # AANet is 2d Convolution 
                 aggregation_type='adaptive', 
                 num_scales=3,  # 尺度的大小， AANET是 H/3, H/6, H/12
                 num_fusions=6, ## AAMoulde 的个数
                 deformable_groups=2,  # Deform Conv的
                 mdconv_dilation=2,
                 refinement_type='stereodrnet',
                 no_intermediate_supervision=False,  # 最后输出的 [H/6,H/12]是否参与贡献
                 num_stage_blocks=1 , # ISA模块中block的个数
                 num_deform_blocks=3): # AAModulde有多少模块需要deform conv
        super(AANet, self).__init__()

        self.refinement_type = refinement_type
        self.feature_type = feature_type
        self.feature_pyramid = feature_pyramid
        self.feature_pyramid_network = feature_pyramid_network
        self.num_downsample = num_downsample
        self.aggregation_type = aggregation_type
        self.num_scales = num_scales

        ######################### Feature  Extraction Part#########################
        # Feature extractor :: AANet, feature提取阶段暂时用不到 max _disp, 下面代码明显使用了deform conv
        if feature_type == 'aanet':
            self.feature_extractor = AANetFeature(feature_mdconv=(not no_feature_mdconv))
            self.max_disp = max_disp // 3   # 初始化为 H/3
        else:
            raise NotImplementedError
        
        # 是否在原来模型的基础上，再次使用金字塔结构
        if feature_pyramid_network:
            if feature_type == 'aanet':
                in_channels = [32 * 4, 32 * 8, 32 * 16, ]
            else:
                in_channels = [32, 64, 128]
            self.fpn = FeaturePyramidNetwork(in_channels=in_channels,
                                             out_channels=32 * 4)
        elif feature_pyramid:
            self.fpn = FeaturePyrmaid()

        ########################### Cost Volume Computation#############################
        if feature_type == 'aanet' or feature_pyramid or feature_pyramid_network:
            cost_volume_module = CostVolumePyramid  # 这里当然使用金字塔的Cost Volume层
        else:
            raise NotImplementedError
        self.cost_volume = cost_volume_module(self.max_disp,
                                              feature_similarity=feature_similarity)

        
        ########################### Cost Aggregation#############################
        max_disp = self.max_disp
        if feature_similarity == 'concat':
            in_channels = 64
        else:
            in_channels = 32  # Here AAnet Usually use "correaltion", So the inchannel is 32

        if aggregation_type == 'adaptive':
            self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                                   num_scales=num_scales,
                                                   num_fusions=num_fusions,
                                                   num_stage_blocks=num_stage_blocks,
                                                   num_deform_blocks=num_deform_blocks,
                                                   mdconv_dilation=mdconv_dilation,
                                                   deformable_groups=deformable_groups,
                                                   intermediate_supervision=not no_intermediate_supervision)
        else:
            raise NotImplementedError

        # 使用Softmax 还是Softmin:
        # 使用 "difference "和 "concat" 时候，激活值越大： 证明差距比较大， Value大的地方，D大，可能性越小，使用Softmin,具体操作就是Softmax(-cost)
        #使用"correlation",激活值越大，代表2个地方越是相似，所有D小， Value 越大，可能性越大，使用Softmax.
        match_similarity = False if feature_similarity in ['difference', 'concat'] else True

        ############################# Disparity estimation################################################################
        self.disparity_estimation = DisparityEstimation(max_disp, match_similarity)

        ##############################Disparity Refinement################################################################
        # Refinement
        if self.refinement_type is not None and self.refinement_type != 'None':
            if self.refinement_type in ['stereonet', 'stereodrnet']:
                refine_module_list = nn.ModuleList()
                for i in range(num_downsample):  # 经过了二次上采样相当于
                    if self.refinement_type == 'stereonet':
                        refine_module_list.append(StereoNetRefinement())
                    elif self.refinement_type == 'stereodrnet':
                        refine_module_list.append(StereoDRNetRefinement())
                    else:
                        raise NotImplementedError
                self.refinement = refine_module_list
            else:
                raise NotImplementedError

    
    # 提取特征
    def feature_extraction(self, img):
        feature = self.feature_extractor(img) # 提取特征
        
        # 是否把特征过一边FPN
        if self.feature_pyramid_network or self.feature_pyramid:
            feature = self.fpn(feature)
        return feature
    
    # 计算Cost Volume
    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)

        if isinstance(cost_volume, list):
            if self.num_scales == 1:
                cost_volume = [cost_volume[0]]  # ablation purpose for 1 scale only
        
        # What is this used for ?
        elif self.aggregation_type == 'adaptive':
            cost_volume = [cost_volume]
        
        return cost_volume
    
    # 计算Disparity
    def disparity_computation(self, aggregation):
        if isinstance(aggregation, list): # For AAnet
            disparity_pyramid = []
            length = len(aggregation)  # D/3, D/6, D/12, I guess the lenghth is 3
            for i in range(length):
                disp = self.disparity_estimation(aggregation[length - 1 - i])  # reverse
                disparity_pyramid.append(disp)  # D/12, D/6, D/3
        else:
            disparity = self.disparity_estimation(aggregation)
            disparity_pyramid = [disparity]

        return disparity_pyramid

    # 一层一层的disparity Refinement
    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        if self.refinement_type is not None and self.refinement_type != 'None':
            if self.refinement_type == 'stereonet':
                for i in range(self.num_downsample):
                    # Hierarchical refinement
                    scale_factor = 1. / pow(2, self.num_downsample - i - 1)
                    if scale_factor == 1.0:
                        curr_left_img = left_img
                        curr_right_img = right_img
                    else:
                        curr_left_img = F.interpolate(left_img,
                                                      scale_factor=scale_factor,
                                                      mode='bilinear', align_corners=False)
                        curr_right_img = F.interpolate(right_img,
                                                       scale_factor=scale_factor,
                                                       mode='bilinear', align_corners=False)
                    inputs = (disparity, curr_left_img, curr_right_img)
                    disparity = self.refinement[i](*inputs)
                    disparity_pyramid.append(disparity)  # [H/2, H]

            elif self.refinement_type in ['stereodrnet']:
                for i in range(self.num_downsample):
                    scale_factor = 1. / pow(2, self.num_downsample - i - 1)

                    if scale_factor == 1.0:
                        curr_left_img = left_img
                        curr_right_img = right_img
                    else:
                        curr_left_img = F.interpolate(left_img,
                                                      scale_factor=scale_factor,
                                                      mode='bilinear', align_corners=False)
                        curr_right_img = F.interpolate(right_img,
                                                       scale_factor=scale_factor,
                                                       mode='bilinear', align_corners=False)
                    inputs = (disparity, curr_left_img, curr_right_img)
                    disparity = self.refinement[i](*inputs)
                    disparity_pyramid.append(disparity)  # [H/2, H]

            else:
                raise NotImplementedError

        return disparity_pyramid

    def forward(self, left_img, right_img):
        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
        cost_volume = self.cost_volume_construction(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        # HERE the disparity_pyrmiad is the [D/12,D/6,D/3] after the disparity computation
        disparity_pyramid = self.disparity_computation(aggregation)
        # 注意在disparity refinement的内部有把disparity 按照image的大小进行差值的算法
        disparity_pyramid += self.disparity_refinement(left_img, right_img,
                                                       disparity_pyramid[-1])

        return disparity_pyramid  # For AAnet, is nums of scales is 3: Disparity_Pyramid is [D/12,D/6,D/3,D/2,D]

