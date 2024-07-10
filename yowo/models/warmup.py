from typing import Any, Dict
import warnings
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Basic Warmup Scheduler
class WarmUpScheduler:
    def __init__(
        self,
        name: str = 'linear', 
        base_lr: float = 0.01, 
        max_iter: int = 500, 
        factor: float = 0.00066667,
    ):
        self.name = name
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.factor = factor

    def set_lr(self, optimizer: Optimizer, lr: float, base_lr: float):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / base_lr
            param_group['lr'] = lr * ratio


    def warmup(self, iter: int, optimizer: Optimizer):
        # warmup
        if self.name == 'exp':
            tmp_lr = self.base_lr * pow(iter / self.max_iter, 4)
            self.set_lr(optimizer, tmp_lr, self.base_lr)

        elif self.name == 'linear':
            alpha = iter / self.max_iter
            warmup_factor = self.factor * (1 - alpha) + alpha
            tmp_lr = self.base_lr * warmup_factor
            self.set_lr(optimizer, tmp_lr, self.base_lr)

    def __call__(self, iter, optimizer):
        self.warmup(iter, optimizer)

class WarmupLR(Callback):
    def __init__(
        self,
        name: str = 'linear', 
        base_lr: float = 0.01, 
        max_iteration: int = 500, 
        warmup_factor: float = 0.00066667
    ):
        super().__init__()
        self.name = name
        self.base_lr = base_lr
        self.max_iteration = max_iteration
        self.warmup_factor = warmup_factor
        self.warmup = True

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            self.warmup_scheduler = WarmUpScheduler(
                name=self.name,
                base_lr=self.base_lr,
                wp_iter=self.max_iteration,
                warmup_factor=self.warmup_factor
            )

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        opt = pl_module.optimizers()
        if pl_module.global_step < self.max_iteration and self.warmup:
            self.warmup_scheduler.warmup(
                iter=pl_module.global_step,
                optimizer=opt
            )
        elif pl_module.global_step >= self.max_iteration and self.warmup:
            self.warmup = False
            self.warmup_scheduler.set_lr(
                optimizer=opt,
                lr=self.base_lr,
                base_lr=self.base_lr
            )
    
    def state_dict(self) -> Dict[str, Any]:        
        return {key: value for key, value in self.__dict__.items()}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)