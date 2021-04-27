import numpy as np
from src.data.isprs.config import ISPRSColorPalette
from src.utils.ml import mask_to_rgb


def test_mask_to_rgb(test_mask_batch):
    indices = test_mask_batch.cpu().numpy()
    result = mask_to_rgb(indices, palette=ISPRSColorPalette)
    assert len(result.shape) == 4
    assert result.shape[-1] == 3
    assert result.dtype == np.uint8
    assert np.max(result) == 255
    assert np.min(result) == 0
