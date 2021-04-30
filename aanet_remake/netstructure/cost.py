import torch
import torch.nn as nn

class CostVolume(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures

        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()

        self.max_disp = max_disp  # 给出最大的max_disp
        self.feature_similarity = feature_similarity # 使用哪种方法

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size() # 拿到左边图的特征Map的Size
        
        # 3DCNN:  COST VOLUME 的Size为： [batch_size, channels, disp_candiate, height, width], for StereoNet
        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w) # 设置为一个空的cost volume, 和left_feature的类型相同
             # 遍历每一个disp的候选,直接计算 feature 之间的 difference
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature
        
        # 如果是concat, 这里应该是要计算3D CNN ，
        elif self.feature_similarity == 'concat':
            # [batch Size, channels*2 , disp_candiate,height,with]
            cost_volume = left_feature.new_zeros(b, 2 * c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    # 直接把相同的部分给 cat在一起
                    cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                            dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)
        
        # 这里应该还是2d CNN
        elif self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    # dot product, 在Channels上面求和，因此减低了一个维度
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                                right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume

#这里对不同分辨率的分别求Pyramid
class CostVolumePyramid(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        super(CostVolumePyramid, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature_pyramid, right_feature_pyramid):
        num_scales = len(left_feature_pyramid)  # Saved list with different size features 

        cost_volume_pyramid = []       # generate a cost_volume_pyrmid
        for s in range(num_scales):
            max_disp = self.max_disp // (2 ** s) # s=0 is the orginal Size
            # 分别计算不同分辨率下的cost
            cost_volume_module = CostVolume(max_disp, self.feature_similarity)
            cost_volume = cost_volume_module(left_feature_pyramid[s],
                                             right_feature_pyramid[s])
            cost_volume_pyramid.append(cost_volume)

        return cost_volume_pyramid  # H/3, H/6, H/12
