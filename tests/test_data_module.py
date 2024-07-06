from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule
from dataclasses import dataclass

from yowo.data import UCF24_JHMDB21_DataModule, AVADataModule
from yowo.models import YOWOv2Lightning

# tfms = {
#     'train': Augmentation(),
#     'test': BaseTransform()
# }

# data_module = UCF24_JHMDB21_DataModule(
#     dataset="ucf24",
#     data_dir="/home/pep/Datasets/ucf24/",
#     img_size=224,
#     len_clip=16,
#     sampling_rate=1,
#     batch_size=4,
#     transform=tfms
# )

# @dataclass
# class Config:
#     model_name: str
#     head: int
#     neck: int


# class Dummy1(LightningModule):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.save_hyperparameters()
#         self.config = config

# class Dummy2(LightningModule):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.save_hyperparameters()
#         self.config = config

if __name__ == '__main__':
    cli = LightningCLI(
        model_class=YOWOv2Lightning,
        # datamodule_class=(UCF24_JHMDB21_DataModule, AVADataModule)
    )
    print(cli)

