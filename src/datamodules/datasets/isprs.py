import os
import glob
import numpy as np
import tifffile as tif
from typing import Callable, Tuple

from torch import Tensor
from torch.utils.data import Dataset as DatasetBase


class ISPRSDataset(DatasetBase):
    """Dataset that reads both Potsdam and Vaihingen images, provided that tiling and preprocessing were
    applied to the raw files.
    """

    def __init__(self,
                 path: str,
                 channel_names: str,
                 channel_count: int,
                 include_dsm: bool = False,
                 transform: Callable = None) -> None:
        """Creates a new dataset for ISPRS data.

        Args:
            path (str): where the main folder is located (one of train, valid, test subdirectories)
            channel_names (str): name of the channels, used to glob images (rgb, rgbir, irrg)
            channel_count (int): how many channels are required, must be <= image.shape[-1]
            include_dsm (bool, optional): whether to include the separate DSM. Defaults to False.
            transform (Callable, optional): augmentations for this set. Defaults to None.
        """
        super().__init__()
        # name required for image retrieval, count required to slice the tensor
        # include DSM is separate, since it's on a different TIFF file
        self.channels = channel_names
        self.channel_count = channel_count
        self.include_dsm = include_dsm
        # save transforms and load file names
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(path, self.image_naming())))
        self.label_files = sorted(glob.glob(os.path.join(path, self.label_naming())))
        assert len(self.image_files) == len(self.label_files), \
            f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
        # check matching sub-tiles
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = "_".join(os.path.basename(image).split("_")[:-1])
            mask_tile = "_".join(os.path.basename(mask).split("_")[:-1])
            assert image_tile == mask_tile, f"image: {image_tile} != mask: {mask_tile}"
        # add the optional digital surface map
        if include_dsm:
            self.dsm_files = sorted(glob.glob(os.path.join(path, self.dsm_naming())))
            assert len(self.image_files) == len(self.dsm_files), "Length mismatch between tiles and DSMs"
            for image, dsm in zip(self.image_files, self.dsm_files):
                image_tile = "_".join(os.path.basename(image).split("_")[:-1])
                dsm_tile = "_".join(os.path.basename(dsm).split("_")[:-1])
                assert image_tile == dsm_tile, f"image: {image_tile} != mask: {dsm_tile}"

    def image_naming(self) -> str:
        return f"*_{self.channels}.tif"

    def label_naming(self) -> str:
        return "*_mask.tif"

    def dsm_naming(self) -> str:
        return "*_dsm.tif"

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get the image/label pair, with optional augmentations and preprocessing steps.
        Augmentations should be provided for a training dataset, while preprocessing should contain
        the transforms required in both cases (normalizations, ToTensor, ...)

        :param index:   integer pointing to the tile
        :type index:    int
        :return:        image, mask tuple
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        image = tif.imread(self.image_files[index]).astype(np.float32)
        image = image[:,:,:self.channel_count]
        mask = tif.imread(self.label_files[index]).astype(np.uint8)
        # add Digital surface map as extra channel to the image
        if self.include_dsm:
            dsm = tif.imread(self.dsm_files[index]).astype(np.float32)
            image = np.dstack((image, dsm))
        # preprocess if required
        if self.transform is not None:
            pair = self.transform(image=image, mask=mask)
            image = pair.get("image")
            mask = pair.get("mask")
        return image, mask

    def __len__(self) -> int:
        return len(self.image_files)
