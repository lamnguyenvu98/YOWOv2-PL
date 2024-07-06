import torch.nn as nn

from yowo.models.basic.utils import Conv
from yowo.models.basic.types import (
    ACTIVATION,
    NORM
)

class DecoupledHead(nn.Module):
    def __init__(
        self,
        num_cls_head: int,
        num_reg_head: int,
        head_act: ACTIVATION,
        head_norm: NORM,
        head_dim: int,
        head_depthwise: bool
    ):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = head_act
        self.norm_type = head_norm
        self.head_dim = head_dim

        self.cls_feats = nn.Sequential(
            *[
                Conv(
                    self.head_dim, 
                    self.head_dim, 
                    k=3, p=1, s=1, 
                    act_type=self.act_type, 
                    norm_type=self.norm_type,
                    depthwise=head_depthwise
                    ) 
                for _ in range(self.num_cls_head)
            ]
        )
        self.reg_feats = nn.Sequential(
            *[
                Conv(
                    self.head_dim, 
                    self.head_dim, 
                    k=3, p=1, s=1, 
                    act_type=self.act_type, 
                    norm_type=self.norm_type,
                    depthwise=head_depthwise)
                for _ in range(self.num_reg_head)]
        )


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


# build detection head
def build_head(
    num_cls_head: int,
    num_reg_head: int,
    head_act: ACTIVATION,
    head_norm: NORM,
    head_dim: int,
    head_depthwise: bool
):
    return DecoupledHead(
        num_cls_head=num_cls_head,
        num_reg_head=num_reg_head,
        head_act=head_act,
        head_norm=head_norm,
        head_dim=head_dim,
        head_depthwise=head_depthwise
    )
    