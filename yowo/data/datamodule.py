from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS, 
    EVAL_DATALOADERS
)
from torch.utils.data import DataLoader
import os
from typing import Dict, Literal, Optional, Any
from dataclasses import dataclass

from .dataset.ava import AVA_Dataset
from .dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from .dataset.transforms import Augmentation, BaseTransform
from .utils import collate_fn

@dataclass
class AugmentationParams:
    jitter: float = 0.2
    hue: float = 0.1
    saturation: float = 1.5
    exposure: float = 1.5

class UCF24_JHMDB21_DataModule(LightningDataModule):
    def __init__(
        self, 
        dataset: str,
        data_dir: str,
        # transform: Optional[dict] = None,
        aug_params: AugmentationParams,
        img_size: int = 224,
        len_clip: int = 16,
        sampling_rate: int = 1,
        batch_size: int = 8,
        collate_fn: Optional[Any] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.aug_params = aug_params
        # self.transform = transform
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.collate_fn = collate_fn
    
    def prepare_data(self) -> None:
        assert self.dataset in ['ucf24', 'jhmdb21'], f"{self.dataset} is not supported. Supported datasets are [ ucf24 , jhmdb21 ]"
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"""Data diretory {self.data_dir} doesn't exist."""
            )

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        if self.collate_fn is None:
            self.collate_fn = collate_fn
        
        if stage == "fit":
            self.tfms = Augmentation(
                img_size=self.img_size,
                jitter=self.aug_params.jitter,
                hue=self.aug_params.hue,
                saturation=self.aug_params.saturation,
                exposure=self.aug_params.exposure
            )
            self.train_set = UCF_JHMDB_Dataset(
                data_root=self.data_dir,
                dataset=self.dataset,
                img_size=self.img_size,
                transform=self.tfms,
                is_train=True,
                len_clip=self.len_clip,
                sampling_rate=self.sampling_rate
            )
        
        if stage == "test":
            self.tfms = BaseTransform(
                img_size=self.img_size
            )

            self.test_set = UCF_JHMDB_Dataset(
                data_root=self.data_dir,
                dataset=self.dataset,
                img_size=self.img_size,
                transform=self.tfms,
                is_train=False,
                len_clip=self.len_clip,
                sampling_rate=self.sampling_rate 
            )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True
        )
        
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True
        )

class AVADataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        aug_params: AugmentationParams,
        frames_dir: str = 'frames_dir',
        frame_list: str = 'frame_list',
        annotation_dir: str = 'annotation_dir',
        labelmap_file: str = 'labelmap_file',
        batch_size: int = 8,
        # transform: Optional[dict] = None,
        collate_fn: Optional[Any] = None,
        gt_box_list: Optional[str] = None,
        exclusion_file: Optional[str] = None,
        img_size: int = 224,
        len_clip: int = 16,
        sampling_rate: int = 1
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.aug_params = aug_params
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.gt_box_list = gt_box_list
        self.exclusion_file = exclusion_file
        self.frames_dir = frames_dir
        self.frame_list = frame_list
        self.annotation_dir = annotation_dir
        self.labelmap_file = labelmap_file
        self.img_size = img_size
        # self.transform = transform
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.tfms = Augmentation(
                img_size=self.img_size,
                jitter=self.aug_params.jitter,
                hue=self.aug_params.hue,
                saturation=self.aug_params.saturation,
                exposure=self.aug_params.exposure
            )
        
            self.train_set = AVA_Dataset(
                data_root=self.data_root,
                gt_box_list=self.gt_box_list,
                exclusion_file=self.exclusion_file,
                frames_dir=self.frames_dir,
                frame_list=self.frame_list,
                annotation_dir=self.annotation_dir,
                labelmap_file=self.labelmap_file,
                is_train=True,
                img_size=self.img_size,
                transform=self.tfms,
                len_clip=self.len_clip,
                sampling_rate=self.sampling_rate
            )
    
        if stage == "test":
            self.tfms = BaseTransform(
                img_size=self.img_size
            )
    
            self.test_set = AVA_Dataset(
                data_root=self.data_root,
                gt_box_list=self.gt_box_list,
                exclusion_file=self.exclusion_file,
                frames_dir=self.frames_dir,
                frame_list=self.frame_list,
                annotation_dir=self.annotation_dir,
                labelmap_file=self.labelmap_file,
                is_train=False,
                img_size=self.img_size,
                transform=self.tfms,
                len_clip=self.len_clip,
                sampling_rate=self.sampling_rate
            )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=False
        )