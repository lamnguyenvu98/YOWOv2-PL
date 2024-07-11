
from lightning.pytorch.cli import LightningCLI
from yowo.data import *
from yowo.models import YOWOv2Lightning

if __name__ == '__main__':
    cli = LightningCLI(
        model_class=YOWOv2Lightning,
    )
    # print(cli)