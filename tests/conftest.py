import os
import pytest
import torch
import numpy as np
import pytorch_lightning as pl
from dotenv import load_dotenv


pl.seed_everything(1337)
load_dotenv(dotenv_path=".env")
num_classes = 6
batch_size = 4


@pytest.fixture(scope="session")
def root_path() -> str:
    return os.getenv("DATA_DIR")


@pytest.fixture(scope="session")
def test_mask() -> torch.Tensor:
    """Single 4x4 test mask with index labels.

    :return: 4x4 single-channel index tensor
    :rtype: torch.Tensor
    """
    return torch.tensor([[0, 1, 3, 3], [2, 4, 2, 1], [1, 2, 2, 0], [2, 255, 255, 255]], dtype=torch.int)


@pytest.fixture(scope="session")
def test_mask_batch() -> torch.Tensor:
    """Repeats the single mask on the batch dimension for an arbitrary 4 times.

    :return: tensor with size [4, 4, 4]
    :rtype: torch.Tensor
    """
    single = torch.tensor([[0, 0, 4, 0], [1, 1, 2, 2], [1, 2, 2, 0], [3, 5, 255, 255]], dtype=torch.int)
    return single.unsqueeze(0).repeat(batch_size, 1, 1)


@pytest.fixture(scope="session")
def test_pred() -> torch.Tensor:
    """Creates a random logits-like tensor with channels-first configuration.

    :return: tensor with [num_classes, w, h]
    :rtype: torch.Tensor
    """
    return torch.rand((num_classes, 4, 4), dtype=torch.float)


@pytest.fixture(scope="session")
def test_pred_batch() -> torch.Tensor:
    """Expands the single tensor into a batched copy

    :return: [description]
    :rtype: torch.Tensor
    """
    indices = torch.tensor([[0, 0, 1, 0], [1, 1, 2, 2], [1, 4, 2, 0], [3, 5, 0, 0]], dtype=torch.int64)
    single = torch.zeros((num_classes, 4, 4)).scatter(0, indices.unsqueeze(0), 1)
    return single.repeat(batch_size, 1, 1, 1)


@pytest.fixture(scope="session")
def big_rand_batch() -> torch.Tensor:
    # non-uniform sampling in range 0-5
    batch_size = (2, 3, 512, 512)
    batch = np.random.choice(num_classes, size=batch_size, p=[0.1, 0.15, 0.4, 0.2, 0.1, 0.05])
    return torch.tensor(batch, dtype=torch.long)
