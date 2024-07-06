from typing import Any, Mapping
from lightning.pytorch import LightningModule

import torch
import torch.optim as optim

from .yowov2.model import YOWO
from .yowov2.loss import build_criterion

from .schemas import (
    LossParams,
    OptimizerParams,
    SchedulerParams,
    YOWOParams
)

class YOWOv2Lightning(LightningModule):
    def __init__(
        self,
        model_params: YOWOParams,
        loss_params: LossParams,
        opt_params: OptimizerParams,
        scheduler_params: SchedulerParams,
        freeze_backbone_2d: bool = True,
        freeze_backbone_3d: bool = True,
        trainable: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['trainable']
        )
        self.opt_params = opt_params
        self.scheduler_params = scheduler_params
        self.model = YOWO(model_params, trainable)
        
        if freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in self.model.backbone_2d.parameters():
                m.requires_grad = False
        if freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in self.model.backbone_3d.parameters():
                m.requires_grad = False
        
        self.criterion = build_criterion(
            img_size=model_params.img_size,
            num_classes=model_params.num_classes,
            multi_hot=model_params.multi_hot,
            loss_cls_weight=loss_params.loss_cls_weight,
            loss_reg_weight=loss_params.loss_reg_weight,
            loss_conf_weight=loss_params.loss_conf_weight,
            focal_loss=loss_params.focal_loss,
            center_sampling_radius=loss_params.center_sampling_radius,
            topk_candicate=loss_params.topk_candicate
        )
        
        # self.warmup_scheduler = WarmUpScheduler(
        #     name='linear',
        #     base_lr=self.opt_params.base_lr,
        #     wp_iter=self.scheduler_params.warmup_iter,
        # )

    def forward(self, video_clip):
        return self.model(video_clip)

    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        frame_ids, video_clips, targets = batch
        # if self.warmup:
        #     opt, _ = self.optimizers()
        #     if self.global_step < self.scheduler_params.warmup_iter:
        #         self.warmup_scheduler.warmup(
        #             iter=self.global_step,
        #             optimizer=opt
        #         )
        #     else:
        #         self.log("Warmup is over - ", self.global_step, prog_bar=True, logger=False)
        #         self.warmup = False
        #         self.warmup_scheduler.set_lr(
        #             optimizer=opt,
        #             lr=self.opt_params.base_lr,
        #             base_lr=self.opt_params.base_lr
        #         )
        
        outputs = self.forward(video_clips)
        loss_dict = self.criterion(outputs, targets)
        losses = loss_dict['losses']
        loss_unscale = losses * self.trainer.accumulate_grad_batches
        out = {
            'loss': losses,
            'loss_unscale': loss_unscale,
            'loss_dict': loss_dict
        }
        self.log("loss_conf", loss_dict["loss_conf"], prog_bar=True, logger=False)
        self.log("loss_cls", loss_dict["loss_cls"], prog_bar=True, logger=False)
        self.log("loss_box", loss_dict["loss_box"], prog_bar=True, logger=False)
        self.log("total loss", losses, prog_bar=True, logger=False)
        return out
    
    def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        ...
    
    def configure_optimizers(self):
        if self.opt_params.optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.opt_params.base_lr,
                momentum=self.opt_params.momentum,
                weight_decay=self.opt_params.weight_decay)

        elif self.opt_params.optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.opt_params.base_lr,
                weight_decay=self.opt_params.weight_decay)
                                    
        else:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.opt_params.base_lr,
                weight_decay=self.opt_params.weight_decay)
        
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            self.scheduler_params.lr_epoch, 
            self.scheduler_params.lr_decay_ratio
        )
        
        return [optimizer], [lr_scheduler]