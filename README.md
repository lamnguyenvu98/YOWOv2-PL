# YOWOv2 - You Only Watch Once 2 in Pytorch Lightning

## 1. Installation
require python 3.10 or above
```
python -m venv venv

source venv/bin/activate

pip install .
```

## 2. Train model
Change path to your dataset and other parameters in config file from folder `configs`

```
yowo fit -c configs/yowov2_nano_ucf24_jhmdb21.yaml
```

You can also overwrite paramters in the config file via command in terminal:
```
yowo fit -c configs/yowov2_tiny_ucf24_jhmdb21.yaml \
    --seed_everything=42 \
    --data.data_dir="/kaggle/input/ucf24-spatial-temporal-localization-yowo/ucf24" \
    --data.batch_size.train=64 \
    --data.batch_size.val=32 \
    --data.num_workers="auto" \
    --model.warmup_config.scheduler.max_iter=250 \
    --model.warmup_config.interval="step" \
    --model.freeze_backbone_2d="false" \
    --model.freeze_backbone_3d="false" \
    --model.model_config.head_act="silu" \
    --trainer.max_epochs=7 \
    --trainer.accumulate_grad_batches=2 \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision="16-mixed" \
    --trainer.strategy="auto" \
    --trainer.callbacks+="lightning.pytorch.callbacks.ModelCheckpoint" \
    --trainer.callbacks.monitor="map" \
    --trainer.callbacks.filename="{epoch}-{map:.3f}" \
    --trainer.callbacks.mode="max" \
    --trainer.callbacks.save_last="true" \
    --trainer.callbacks.save_top_k=3 \
    --trainer.callbacks.dirpath="/kaggle/working/logs/weights" \
    --ckpt_path="/kaggle/working/logs/weights/last.ckpt" # this one to resume the training from last checkpoint
```
By default, Pytorch Lightning uses logger Tensorboard. You can use other loggers as well. Find out more which logger is supported [Loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
An example of using WanDB logger
```
yowo fit -c configs/yowov2_tiny_ucf24_jhmdb21.yaml \
    --trainer.logger.class_path="lightning.pytorch.loggers.WandbLogger" \
    --trainer.logger.name="experiment-yowo-tiny_2024-07-22_00:35" \
    --trainer.logger.project="spatial-temporal-action-detection-yowov2" \
    --trainer.logger.tags="['P100', 'ucf24', 'tiny', 'freeze-backbone', 'blurpool']" \
    --trainer.logger.log_model="all" \
    --trainer.logger.save_dir="/kaggle/working/logs" \
    --trainer.logger.id="i2yvhgrp" \
```
- `trainer.logger.class_path` is the path where you can import logger module: `lightning.pytorch.loggers.WandbLogger`, `lightning.pytorch.loggers.CSVLogger`
- other `trainer.logger.*` are arguments provided for that module class. As we can see, `WandbLogger` needs some arguments such as `name`, `project`, `log_model`, `save_dir`, etc...

## 3. Validate/test
Validate
```
yowo validate -c configs/yowov2_tiny_ucf24_jhmdb21.yaml \
    --seed_everything=42 \
    --data.data_dir="/kaggle/input/ucf24-spatial-temporal-localization-yowo/ucf24" \
    --data.batch_size.val=64 \
    --data.num_workers="auto" \
    --model.freeze_backbone_2d="false" \
    --model.freeze_backbone_3d="false" \
    --model.model_config.head_act="silu" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision="16-mixed" \
    --ckpt_path="/kaggle/working/logs/weights/last.ckpt" \
```
or Test
```
yowo test -c configs/yowov2_tiny_ucf24_jhmdb21.yaml
```
- `ckpt_path` has to be specified

## 4. Dataset
[UCF24 Dataset](https://www.kaggle.com/datasets/vulamnguyen/ucf24-spatial-temporal-localization-yowo)

## 5. Structure of dataset
Inside root directory
```
├── rgb-images
│   ├── BasketBall #action class
│   |    └── v_Basketball_g01_c01 # video split for that action
│   |        ├── 00001.jpg
│   |        ├── 00002.jpg
│   |        ├── ...
│   |        └── 00030.jpg
│   └── Biking
│        ├── v_Biking_g01_c01
│        |   ├── 00001.jpg
│        |   ├── 00002.jpg
│        |   ├── ...
│        |   └── 00040.jpg
│        └── v_Biking_g01_c02
│            ├── 00001.jpg
│            ├── 00002.jpg
│            ├── ...
│            └── 00045.jpg
│
├── labels # annotation files
│   ├── BasketBall #action class
│   |    └── v_Basketball_g01_c01 # video split for that action
│   |        ├── 00001.txt
│   |        ├── 00002.txt
│   |        ├── ...
│   |        └── 00030.txt
│   └── Biking
│        ├── v_Biking_g01_c01
│        |   ├── 00001.txt
│        |   ├── 00002.txt
│        |   ├── ...
│        |   └── 00040.txt
│        └── v_Biking_g01_c02
│            ├── 00001.txt
│            ├── 00002.txt
│            ├── ...
│            └── 00045.txt
```
`trainlist.txt`: contains path to training data like this:
```
labels/Basketball/v_Basketball_g08_c01/00070.txt
labels/Basketball/v_Basketball_g08_c01/00071.txt
labels/Basketball/v_Basketball_g08_c01/00072.txt
labels/Basketball/v_Basketball_g08_c01/00073.txt
```
`testlist.txt` is the same as `trainlist.txt`