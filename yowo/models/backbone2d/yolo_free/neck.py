from typing import List
import torch
import torch.nn as nn

from yowo.utils.validate import validate_literal_types
from yowo.models.basic.utils import Conv
from yowo.models.basic.types import (
    NORM,
    ACTIVATION,
)
from .types import NECK

# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        expand_ratio: float = 0.5, 
        pooling_size: List[int] = [5, 9, 13], 
        norm_type: NORM = 'BN', 
        act_type: ACTIVATION = 'relu'
    ):
        super(SPP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in pooling_size
            ]
        )
        
        self.cv2 = Conv(inter_dim*(len(pooling_size) + 1), out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cv2(x)

        return x


# SPP block with CSP module
class SPPBlock(nn.Module):
    """
        Spatial Pyramid Pooling Block
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expand_ratio: float = 0.5,
        pooling_size: List[int] | int = [5, 9, 13],
        act_type: ACTIVATION = 'lrelu',
        norm_type: NORM = 'BN',
        ):
        super(SPPBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            SPP(inter_dim, 
                inter_dim, 
                expand_ratio=1.0, 
                pooling_size=pooling_size, 
                act_type=act_type, 
                norm_type=norm_type),
        )
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        y = self.cv3(torch.cat([x1, x2], dim=1))

        return y


# SPP block with CSP module
class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expand_ratio: float = 0.5,
        pooling_size: List[int] | int = [5, 9, 13],
        act_type: ACTIVATION = 'lrelu',
        norm_type: NORM = 'BN',
        depthwise: bool = False
        ):
        super(SPPBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(
            Conv(
                inter_dim, inter_dim, k=3, p=1, 
                act_type=act_type, norm_type=norm_type, 
                depthwise=depthwise
            ),
            SPP(
                inter_dim, 
                inter_dim, 
                expand_ratio=1.0, 
                pooling_size=pooling_size, 
                act_type=act_type, 
                norm_type=norm_type
            ),
            Conv(
                inter_dim, inter_dim, k=3, p=1, 
                act_type=act_type, norm_type=norm_type, 
                depthwise=depthwise
            )
        )
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: List[int] | int = 5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        inter_dim = in_dim // 2  # hidden channels
        self.cv1 = Conv(in_dim, inter_dim, k=1)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


def build_neck(
    model_name: NECK,
    expand_ratio: float,
    pooling_size: List[int] | int,
    neck_act: ACTIVATION,
    neck_norm: NORM,
    neck_depthwise: bool,
    in_dim: int, 
    out_dim: int
):
    validate_literal_types(model_name, NECK)
    # build neck
    if model_name == 'spp_block':
        neck = SPPBlock(
            in_dim, out_dim, 
            expand_ratio=expand_ratio, 
            pooling_size=pooling_size,
            act_type=neck_act,
            norm_type=neck_norm,
            depthwise=neck_depthwise
            )
            
    elif model_name == 'spp_block_csp':
        neck = SPPBlockCSP(
            in_dim, out_dim, 
            expand_ratio=expand_ratio, 
            pooling_size=pooling_size,
            act_type=neck_act,
            norm_type=neck_norm,
            depthwise=neck_depthwise
            )

    elif model_name == 'sppf':
        neck = SPPF(in_dim, out_dim, k=pooling_size)

    return neck


if __name__ == '__main__':
    neck_net = build_neck(
        model_name="sppf",
        expand_ratio=0.5,
        pooling_size=5,
        neck_act="lrelu",
        neck_norm="BN",
        neck_depthwise=True,
        in_dim=1024,
        out_dim=256
    )
    print(neck_net)
