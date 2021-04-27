import logging
from src.losses import functional as fn


LOG = logging.getLogger(__name__)


def test_soft_dice_batch(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    dice = fn.soft_dice_loss(pred_classes, test_mask_batch.long(), reduction=None)
    LOG.debug(dice)
