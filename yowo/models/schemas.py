from dataclasses import dataclass, field
from typing import List
import torch

from .types import OPTIMIZER

@dataclass
class LossParams:
    topk_candicate: int = 10
    center_sampling_radius: float = 2.5
    loss_conf_weight: float | int = 1
    loss_cls_weight: float | int = 1
    loss_reg_weight: float | int = 5
    focal_loss: bool = False

@dataclass
class OptimizerParams:
    optimizer_type: OPTIMIZER = "sgd"
    base_lr: float = 0.001
    weight_decay: float = 5e-4
    momentum: float = 0.9

@dataclass
class SchedulerParams:
    lr_epoch: tuple[int] # = field(default_factory=(2, 3, 4))
    lr_decay_ratio: float = 0.5
    warmup: bool = True
    warmup_iter: int = 500

@dataclass
class YOWOParams:
    backbone_2d: str
    backbone_3d: str
    pretrained_2d: bool
    pretrained_3d: bool
    head_dim: int
    head_norm: str
    head_act: str
    head_depthwise: bool
    num_classes: int
    stride: List[int] # = field(default_factory=[8, 16, 32])
    img_size: int = 224
    conf_thresh: float = 0.05
    nms_thresh: float = 0.6
    topk: int = 50
    multi_hot: bool = False
    num_cls_heads: int = 2
    num_reg_heads: int = 2
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"
    