import torch
import torch.nn as nn

from yowo.models.basic.utils import Conv as Conv2d


class DecoupledHead(nn.Module):
    def __init__(
        self,
        num_cls_heads: int,
        num_reg_heads: int,
        head_act: str,
        head_norm: str,
        head_dim: int,
        head_depthwise: bool
    ):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.num_cls_heads = num_cls_heads
        self.num_reg_heads = num_reg_heads
        self.act_type = head_act
        self.norm_type = head_norm
        self.head_dim = head_dim
        self.depthwise = head_depthwise

        self.cls_head = nn.Sequential(*[
            Conv2d(
                self.head_dim, 
                self.head_dim, 
                k=3, p=1, s=1, 
                act_type=self.act_type, 
                norm_type=self.norm_type,
                depthwise=self.depthwise)
                for _ in range(self.num_cls_heads)]
            )
        self.reg_head = nn.Sequential(*[
            Conv2d(
                self.head_dim, 
                self.head_dim, 
                k=3, p=1, s=1, 
                act_type=self.act_type, 
                norm_type=self.norm_type,
                depthwise=self.depthwise)
                for _ in range(self.num_reg_heads)]
            )


    def forward(self, cls_feat, reg_feat):
        cls_feats = self.cls_head(cls_feat)
        reg_feats = self.reg_head(reg_feat)

        return cls_feats, reg_feats


def build_head(
    num_cls_heads: int,
    num_reg_heads: int,
    head_act: str,
    head_norm: str,
    head_dim: int,
    head_depthwise: bool
):
    return DecoupledHead(
        num_cls_heads,
        num_reg_heads,
        head_act,
        head_norm,
        head_dim,
        head_depthwise
    )
    