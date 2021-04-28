import numpy as np
import torch
from typing import Optional, Dict, Iterable


def torch_one_hot(target: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """source: https://github.com/PhoenixDL/rising. Computes one-hot encoding of input tensor.

    Args:
        target (torch.Tensor): tensor to be converted
        num_classes (Optional[int], optional): number of classes. If None, the maximum value of target is used.

    Returns:
        torch.Tensor: one-hot encoded tensor of the target
    """
    if num_classes is None:
        num_classes = int(target.max().detach().item() + 1)
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes, dtype=dtype, device=device)
    return target_onehot.scatter_(1, target.unsqueeze_(1), 1.0)


def one_hot_batch(target: torch.Tensor,
                  num_classes: Optional[int] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Source: https://github.com/PhoenixDL/rising
    Computes one hot for the input tensor (assumed to a be batch and thus saved
    into first dimension -> input should only have one channel).

    Args:
        target (torch.Tensor): long tensor to be converted.
        num_classes (Optional[int], optional): number of classes. If None, the maximum of target is used.
        dtype (Optional[torch.dtype], optional): optionally changes the dtype of the onehot encoding.

    Raises:
        TypeError: when the input tensor is not of long type.
    Returns:
        torch.Tensor: one hot encoded tensor
    """
    if target.dtype != torch.long:
        raise TypeError(
            f"Target tensor needs to be of type torch.long, found {target.dtype}")

    if target.ndim in [0, 1]:
        return torch_one_hot(target, num_classes)
    else:
        if num_classes is None:
            num_classes = int(target.max().detach().item() + 1)
        _dtype, device, shape = target.dtype, target.device, target.shape
        if dtype is None:
            dtype = _dtype
        target_onehot = torch.zeros(shape[0], num_classes, *shape[2:],
                                    dtype=dtype, device=device)
        return target_onehot.scatter_(1, target, 1.0)


def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces the given tensor using a specific criterion.

    Args:
        tensor (torch.Tensor): input tensor
        reduction (str): string with fixed values [elementwise_mean, none, sum]

    Raises:
        ValueError: when the reduction is not supported

    Returns:
        torch.Tensor: reduced tensor, or the tensor itself
    """
    if reduction in ("elementwise_mean", "mean"):
        return torch.mean(tensor)
    if reduction == 'sum':
        return torch.sum(tensor)
    if reduction is None or reduction == 'none':
        return tensor
    raise ValueError('Reduction parameter unknown.')


def mask_to_rgb(mask: np.ndarray, palette: Dict[int, Iterable]) -> np.ndarray:
    """Given an input batch, or single picture with dimensions [B, H, W] or [H, W], the utility generates
    an equivalent [B, H, W, 3] or [H, W, 3] array corresponding to an RGB version.
    The conversion uses the given palette, which should be provided as simple dictionary of indices and tuples, lists
    or arrays indicating a single RGB color. (e.g. {0: (255, 255, 255)})

    Args:
        mask (np.ndarray): input mask of indices. Each index should be present in the palette
        palette (Dict[int, Iterable]): dictionary of pairs <index - color>, where colors can be provided in RGB tuple format

    Returns:
        np.ndarray: tensor containing the RGB version of the input index tensor
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for index, color in palette.items():
        lut[index] = np.array(color, dtype=np.uint8)
    return lut[mask]