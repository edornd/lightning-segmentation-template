from os.path import join
from typing import Any, Callable, Dict, List, Optional, Union

from albumentations import Compose
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


class SegmentationDataModule(LightningDataModule):

    def __init__(self,
                 dataset_dir: str,
                 dataset_class: Callable,
                 dataset_params: Dict[str, Any] = None,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 train_transforms: Compose = None,
                 valid_transforms: Compose = None,
                 dims: tuple = None,
                 **kwargs): # kwargs avoids unexpected param errors for datamodule
        super().__init__(train_transforms=train_transforms,
                         val_transforms=valid_transforms,
                         test_transforms=valid_transforms,
                         dims=dims)
        self.data_dir = dataset_dir
        self.data_cls = dataset_class
        self.extra_params = dataset_params or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str]):
        # assign train sets for dataloaders
        if stage == "fit" or stage is None:
            self.train_set = instantiate(dict(_target_=self.data_cls),
                                         path=join(self.data_dir, "train"),
                                         transform=self.train_transforms,
                                         **self.extra_params)
            self.valid_set = instantiate(dict(_target_=self.data_cls),
                                         path=join(self.data_dir, "valid"),
                                         transform=self.val_transforms,
                                         **self.extra_params)
        # assign test sets for dataloaders
        if stage in ("test", "predict") or stage is None:
            self.test_set = instantiate(dict(_target_=self.data_cls),
                                        path=join(self.data_dir, "test"),
                                        transform=self.test_transforms,
                                        **self.extra_params)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)
