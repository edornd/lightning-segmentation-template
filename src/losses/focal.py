import torch
import torch.nn as nn
from torch.nn import functional as fn

from src.utils.ml import reduce


class FocalLoss(nn.Module):
    """TODO
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean", ignore_index: int = 255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = fn.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return reduce(focal_loss, reduction=self.reduction)
