from lightning.pytorch.cli import (
    LightningArgumentParser, 
    LightningCLI
)
import torch

from yowo.models import YOWOv2Lightning
from yowo.data import *

default_optimizer = {
    "class_path": "torch.optim.AdamW",
    "init_args": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1.0e-08,
        "weight_decay": 5e-4,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None
    },
}

default_scheduler = {
    "class_path": "torch.optim.lr_scheduler.MultiStepLR",
    "init_args": {
        "milestones": [2, 4, 5],
        "gamma": 0.1,
        "last_epoch": -1,
        "verbose": False if torch.__version__ <= '2.1.2' else "deprecated"
    },
}

default_scheduler_config = {
    'interval': 'epoch',
    'frequency': 1
}

class YowoLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        subcommand = self.config["subcommand"]
        scheduler_ns =  self.config[subcommand]["model"]["scheduler"]
        # same older version <= torch 2.1.2, verbose is boolean
        # latter version verbose="deprecated", and will be removed
        # in upcomming versions
        if "verbose" in scheduler_ns['init_args'].keys():
            scheduler_ns["init_args"].pop("verbose")
    
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults({
            "model.optimizer": default_optimizer,
            "model.scheduler": default_scheduler,
        })
        parser.link_arguments(
            source="data.img_size",
            target="model.model_config.img_size",
            apply_on="instantiate"
        )

def main():
    cli = YowoLightningCLI(
        model_class=YOWOv2Lightning,
    )

if __name__ == '__main__':
    main()