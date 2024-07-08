from typing import Any
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

# Basic Warmup Scheduler
class WarmUpScheduler(object):
    def __init__(
        self, 
        name='linear', 
        base_lr=0.01, 
        wp_iter=500, 
        warmup_factor=0.00066667
    ):
        self.name = name
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor


    def set_lr(self, optimizer, lr, base_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / base_lr
            param_group['lr'] = lr * ratio


    def warmup(self, iter, optimizer):
        # warmup
        # assert iter < self.wp_iter
        if self.name == 'exp':
            tmp_lr = self.base_lr * pow(iter / self.wp_iter, 4)
            self.set_lr(optimizer, tmp_lr, self.base_lr)

        elif self.name == 'linear':
            alpha = iter / self.wp_iter
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
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
        if pl_module.global_step < self.max_iteration:
            self.warmup_scheduler.warmup(
                iter=pl_module.global_step,
                optimizer=opt
            )
        else:
            # pl_module.log("Warmup is over - ", pl_module.global_step, prog_bar=True, logger=False)
            self.warmup_scheduler.set_lr(
                optimizer=opt,
                lr=self.base_lr,
                base_lr=self.base_lr
            )