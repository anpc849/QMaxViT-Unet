# Reference: https://github.com/junjiehe96/FastInst/blob/main/fastinst/modeling/pixel_decoder/fastinst_encoder.py

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.config import configurable
from typing import Callable, Dict, Optional, Union
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
import torch.nn as nn
from .twowaytrans import LayerNorm2d
import torch
from torch.nn import functional as F

class BaseFPN(nn.Module):
#     @configurable
    def __init__(
            self,
            *,
            convs_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
            feature_channels = [192, 384, 768],
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            convs_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        #input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
#         self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
#         feature_channels = [v.channels for k, v in input_shape]
#         self.in_features = [192, 384, 768]
#         self.feature_strides = [4, 8, 16]
        self.feature_channels = feature_channels

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            lateral_norm = get_norm(norm, convs_dim)
            output_norm = get_norm(norm, convs_dim)

            lateral_conv = Conv2d(
                in_channels, convs_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                convs_dim,
                convs_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.num_feature_levels = 3  # always use 3 scales


    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        
        #torch.Size([1, 768, 8, 8])
        #torch.Size([1, 384, 16, 16])
        #torch.Size([1, 192, 32, 32])

        for idx, f in enumerate(features[::-1]):
            x = f
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if idx == 0:
                y = lateral_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)

            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return None, multi_scale_features

    def forward(self, features, targets=None):
        return self.forward_features(features)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


class PyramidPoolingModuleFPN(BaseFPN):
    #@configurable
    def __init__(
            self,
            *,
            convs_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            convs_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(convs_dim=convs_dim, mask_dim=mask_dim, norm=norm)
        self.ppm = PyramidPoolingModule(convs_dim, convs_dim // 4)

        self.fusion = nn.Conv2d(convs_dim * 3, convs_dim, 1)
        c2_msra_fill(self.fusion)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(384, 384 // 2, kernel_size=2, stride=2),
            LayerNorm2d(384 // 2),
            nn.GELU(),
            nn.ConvTranspose2d(384 // 2, 384 // 4, kernel_size=2, stride=2),
            LayerNorm2d(384 // 4),
            nn.GELU(),
            #T.Resize(size=(256,256))
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2),
            nn.GELU()
        )

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(features[::-1]):
            x = f
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if idx == 0:
                y = self.ppm(lateral_conv(x))
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)

            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.insert(0, y)
                num_cur_levels += 1
            
        
        size = multi_scale_features[0].shape[2:]
        final_features = [multi_scale_features[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in multi_scale_features[1:]]
        final_features = self.fusion(torch.cat(final_features, dim=1))
        final_features_scaled = self.output_upscaling(final_features)
        return final_features_scaled