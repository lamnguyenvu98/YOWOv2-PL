from typing import List, Literal
import warnings

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from yowo.utils.validate import validate_literal_types, deprecated_verbose_scheduler

WARMUP_TYPE = Literal["exp", "linear"]

class WarmupLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        name: WARMUP_TYPE = 'linear', 
        max_iter: int = 500, 
        factor: float = 0.00066667,
        last_epoch: int = -1,
        verbose: str = "deprecated"
    ):
        validate_literal_types(name, WARMUP_TYPE)
        self.name = name
        self.max_iter = max_iter
        self.factor = factor
        verbose = deprecated_verbose_scheduler(verbose)
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_last_lr(self) -> List[float]:
        return super().get_last_lr()
    
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        # num_step = self._step_count
        
        if self.last_epoch < self.max_iter:
            tmp_lrs = self.warmup(iter=self.last_epoch)
            ratios = [
                group['initial_lr'] / base_lr for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
            ]
            return [tmp_lr * ratio for tmp_lr, ratio in zip(tmp_lrs, ratios)]
        elif self.last_epoch == self.max_iter:
            return [group['initial_lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

    def warmup(self, iter: int):
        # warmup
        if self.name == 'exp':
            tmp_lrs = [base_lr * pow(iter / self.max_iter, 4) for base_lr in self.base_lrs]

        elif self.name == 'linear':
            alpha = iter / self.max_iter
            warmup_factor = self.factor * (1 - alpha) + alpha
            tmp_lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        return tmp_lrs

# class WarmupLR(Callback):
#     def __init__(
#         self,
#         name: str = 'linear', 
#         base_lr: float = 0.01, 
#         max_iteration: int = 500, 
#         warmup_factor: float = 0.00066667
#     ):
#         super().__init__()
#         self.name = name
#         self.base_lr = base_lr
#         self.max_iteration = max_iteration
#         self.warmup_factor = warmup_factor
#         self.warmup = True

#     def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
#         if stage == "fit":
#             self.warmup_scheduler = WarmUpScheduler(
#                 name=self.name,
#                 base_lr=self.base_lr,
#                 wp_iter=self.max_iteration,
#                 warmup_factor=self.warmup_factor
#             )

#     def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
#         opt = pl_module.optimizers()
#         if pl_module.global_step < self.max_iteration and self.warmup:
#             self.warmup_scheduler.warmup(
#                 iter=pl_module.global_step,
#                 optimizer=opt
#             )
#         elif pl_module.global_step >= self.max_iteration and self.warmup:
#             self.warmup = False
#             self.warmup_scheduler.set_lr(
#                 optimizer=opt,
#                 lr=self.base_lr,
#                 base_lr=self.base_lr
#             )
    
#     def state_dict(self) -> Dict[str, Any]:        
#         return {key: value for key, value in self.__dict__.items()}
    
#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         self.__dict__.update(state_dict)