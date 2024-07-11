from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

import torch
import torch.optim as optim

from yowo.utils.box_ops import rescale_bboxes
from yowo.schedulers import WarmupLRScheduler
from .yowov2.model import YOWO
from .yowov2.loss import build_criterion

from .schemas import (
    LossParams,
    ModelConfig,
    WarmupLRConfig
)

DEFAULT_OPTIMIZER = lambda p: optim.AdamW(p, lr=0.001, weight_decay=5e-4)
DEFAULT_SCHEDULER = lambda opt: optim.lr_scheduler.MultiStepLR(
    optimizer=opt,
    milestones=[2, 4, 5],
    gamma=0.5
)
DEFAULT_SCHEDULER_CONFIG = {
    'interval': 'step',
    'frequency': 1
}

class YOWOv2Lightning(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        loss_params: LossParams,
        scheduler: LRSchedulerCallable = DEFAULT_SCHEDULER,
        scheduler_config: Dict = DEFAULT_SCHEDULER_CONFIG,
        optimizer: OptimizerCallable = DEFAULT_OPTIMIZER,
        freeze_backbone_2d: bool = True,
        freeze_backbone_3d: bool = True,
        warmup_config: Optional[WarmupLRConfig] = None,
        trainable: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.warmup_config = warmup_config
        self.num_classes = model_config.num_classes
        self.model = YOWO(model_config, trainable=trainable)
        
        if freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in self.model.backbone_2d.parameters():
                m.requires_grad = False
        if freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in self.model.backbone_3d.parameters():
                m.requires_grad = False
        
        self.criterion = build_criterion(
            img_size=model_config.img_size,
            num_classes=model_config.num_classes,
            multi_hot=model_config.multi_hot,
            loss_cls_weight=loss_params.loss_cls_weight,
            loss_reg_weight=loss_params.loss_reg_weight,
            loss_conf_weight=loss_params.loss_conf_weight,
            focal_loss=loss_params.focal_loss,
            center_sampling_radius=loss_params.center_sampling_radius,
            topk_candicate=loss_params.topk_candicate
        )

    def forward(self, video_clip, infer_mode = True):
        return self.model.inference(video_clip) if infer_mode else self.model(video_clip)

    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        frame_ids, video_clips, targets = batch
        batch_size = video_clips.size(0)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        outputs = self.forward(video_clips, infer_mode=False)
        loss_dict = self.criterion(outputs, targets)
        total_loss = loss_dict['losses']
        # loss_unscale = losses * self.trainer.accumulate_grad_batches
        out_log = {
            "lr": lr,
            "total_loss": total_loss,
            "loss_conf": loss_dict["loss_conf"],
            "loss_cls": loss_dict["loss_cls"],
            "loss_box": loss_dict["loss_box"]
        }
        if self.trainer.num_devices > 1:
            sync_dist = True
        else:
            sync_dist = False
            
        self.log_dict(
            dictionary=out_log, 
            prog_bar=True, 
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size
        )
        return total_loss
        
    # def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
    #     batch_img_name, batch_video_clip, batch_target = batch
    #     batch_scores, batch_labels, batch_bboxes = self.forward(batch_video_clip, infer_mode=True)
    #     # process batch
    #     for bi in range(len(batch_scores)):
    #         img_name = batch_img_name[bi]
    #         scores = batch_scores[bi]
    #         labels = batch_labels[bi]
    #         bboxes = batch_bboxes[bi]
    #         target = batch_target[bi]

    #         # rescale bbox
    #         orig_size = target['orig_size']
    #         bboxes = rescale_bboxes(bboxes, orig_size)

    #         img_annotation = {}
            
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())

        
        lr_scheduler = self.scheduler(optimizer)
        
        schedulers = [{
            'scheduler': lr_scheduler,
            **self.scheduler_config
        }]
        
        if self.warmup_config:
            schedulers.append({
                'scheduler': WarmupLRScheduler(optimizer, **asdict(self.warmup_config)),
                'interval': 'step',
                'frequency': 1
            })
        
        return [optimizer], schedulers