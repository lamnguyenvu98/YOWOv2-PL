from dataclasses import asdict
from typing import Any, Literal, Mapping, Optional

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from yowo.utils.box_ops import rescale_bboxes_tensor
from yowo.schedulers import WarmupLRScheduler
from .yowov2.model import YOWO
from .yowov2.loss import build_criterion

from .schemas import (
    LossConfig,
    ModelConfig,
    WarmupLRConfig
)

DEFAULT_SCHEDULER_CONFIG = {
    'interval': 'epoch',
    'frequency': 1
}

DEFAULT_WARMUP = WarmupLRConfig()

class YOWOv2Lightning(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        scheduler_config: dict = DEFAULT_SCHEDULER_CONFIG,
        freeze_backbone_2d: bool = True,
        freeze_backbone_3d: bool = True,
        warmup_config: Optional[WarmupLRConfig] = DEFAULT_WARMUP,
        trainable: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
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
            loss_cls_weight=loss_config.loss_cls_weight,
            loss_reg_weight=loss_config.loss_reg_weight,
            loss_conf_weight=loss_config.loss_conf_weight,
            focal_loss=loss_config.focal_loss,
            center_sampling_radius=loss_config.center_sampling_radius,
            topk_candicate=loss_config.topk_candicate
        )
        
        self.val_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.25, 0.5, 0.75, 0.95],
            average="macro"
        )
        self.test_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.25, 0.5, 0.75, 0.95],
            average="macro"
        )

    def forward(self, video_clip: torch.Tensor, infer_mode = True):
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
        
    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        batch_map = self.eval_step(batch, mode="val")
    
    def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        batch_map = self.eval_step(batch, mode="test")

    def eval_step(self, batch, mode: Literal["val", "test"]):
        if mode == "val":
            metric = self.val_metric
        else:
            metric = self.test_metric
        
        batch_img_name, batch_video_clip, batch_target = batch
        batch_scores, batch_labels, batch_bboxes = self.forward(batch_video_clip, infer_mode=True)
        
        # process batch gt
        gts = list(map(
            lambda x: {
                "boxes": rescale_bboxes_tensor(
                    x["boxes"], 
                    dest_width=x["orig_size"][0],
                    dest_height=x["orig_size"][1]
                ),
                "labels": x["labels"].long(),
            },
            batch_target
        ))
        
        # process batch predict
        preds = []
        for idx, (scores, labels, bboxes) in enumerate(zip(batch_scores, batch_labels, batch_bboxes)):
            pred = {
                "boxes": rescale_bboxes_tensor(
                    bboxes, 
                    dest_width=batch_target[idx]["orig_size"][0],
                    dest_height=batch_target[idx]["orig_size"][1]
                ),
                "scores": scores,
                "labels": labels.long() + 1, # int64
            }
            preds.append(pred)
        
        metric_result = metric.update(preds, gts)
        return metric_result
        

    def on_validation_epoch_end(self):
        result = self.val_metric.compute()
        self.log_dict(
            result, 
            prog_bar=True, 
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.trainer.num_devices > 1
        )
        self.val_metric.reset()

    def on_test_epoch_end(self):
        mean_ap = self.test_metric.compute()
        result = {
            "map": mean_ap["map"],
            "map_50": mean_ap["map_50"],
            "map_75": mean_ap["map_75"],
        }
        self.log_dict(
            result, 
            prog_bar=True, 
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.trainer.num_devices > 1
        )
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())

        
        lr_scheduler = self.lr_scheduler(optimizer)
        
        schedulers = [{
            'scheduler': lr_scheduler,
            **self.scheduler_config
        }]
        
        if self.warmup_config:
            schedulers.append({
                'scheduler': WarmupLRScheduler(
                    optimizer, 
                    **asdict(self.warmup_config)
                ),
                'interval': 'step',
                'frequency': 1
            })
        
        return [optimizer], schedulers