from typing import Any, Mapping
from lightning.pytorch import LightningModule

import torch
import torch.optim as optim
import numpy as np

from yowo.utils.box_ops import rescale_bboxes
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
        self.save_hyperparameters()
        self.trainable = trainable
        self.opt_params = opt_params
        self.scheduler_params = scheduler_params
        self.model = YOWO(model_params, trainable=self.trainable)
        
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
        self.num_classes = model_params.num_classes
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
    
    # def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
    #     batch_img_name, batch_video_clip, batch_target = batch
    #     batch_scores, batch_labels, batch_bboxes = self.model.inference(batch_video_clip)
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
    #         for cls_idx in range(self.num_classes):
    #             inds = np.where(labels == cls_idx)[0]
    #             c_bboxes = bboxes[inds]
    #             c_scores = scores[inds]
    #             # [n_box, 5]
    #             boxes = np.concatenate([c_bboxes, c_scores[..., None]], axis=-1)
    #             img_annotation[cls_idx + 1] = boxes
    #         # detected_boxes[img_name] = img_annotation
    #     return {
    #         "loss": 0.0,
    #         "img_name": img_name,
            
    #     }
    
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