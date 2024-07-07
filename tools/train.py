# from lightning.pytorch import Trainer
# from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
# from lightning.pytorch import LightningModule

from yowo.data import UCF24_JHMDB21_DataModule, AVADataModule
from yowo.models import YOWOv2Lightning

if __name__ == '__main__':
    cli = LightningCLI(
        model_class=YOWOv2Lightning,
    )
    # print(cli)