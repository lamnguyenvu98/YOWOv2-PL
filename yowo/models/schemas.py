from dataclasses import dataclass
from typing import List, Literal
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
    lr_epoch: List[int]
    lr_decay_ratio: float = 0.5
    warmup: bool = True
    warmup_iter: int = 500

@dataclass
class WarmupLRConfig:
    name: Literal["linear", "exp"] = "linear"
    max_iter: int = 500
    factor: float = 0.00066667

@dataclass
class ModelConfig:
    backbone_2d: str
    backbone_3d: str
    pretrained_2d: bool
    pretrained_3d: bool
    head_dim: int
    head_norm: str
    head_act: str
    head_depthwise: bool
    num_classes: int
    stride: List[int]
    img_size: int = 224
    conf_thresh: float = 0.05
    nms_thresh: float = 0.6
    topk: int = 50
    multi_hot: bool = False
    num_cls_heads: int = 2
    num_reg_heads: int = 2
    