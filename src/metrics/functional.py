from typing import Any, Optional, Tuple, Union

import torch
from pytorch_lightning.utilities import FLOAT32_EPSILON
from src.utils.ml import torch_one_hot


def identity(x: Any) -> Any:
    """Simple callable to provide default no-op.

    :param x: whatever input, preferably tensors
    :type x: Any
    :return: the input itself
    :rtype: Any
    """
    return x


def valid_samples(ignore_index: int,
                  target: torch.Tensor,
                  pred: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple]:
    """Receives one or two tensors (when the prediction is included) to exclude ignored indices.

    :param ignore_index: index to be discarded
    :type ignore_index: int
    :param target: target input, expected as 2D image or 3D batch
    :type target: torch.Tensor
    :param pred: predicted tensor already 'argmaxed', defaults to None
    :type pred: Optional[torch.Tensor], optional
    :return: tensor containing only values different from ignore_index, or a tuple when pred is also provided.
    :rtype: Union[torch.Tensor, tuple]
    """
    valid_indices = target != ignore_index
    valid_target = target[valid_indices]
    if pred is not None:
        assert pred.shape == target.shape, f"Shape mismatch: {target.shape} </> {pred.shape}"
        valid_pred = pred[valid_indices]
        return valid_target.long(), valid_pred.long()
    return valid_target.long()


def count_classes(tensor: torch.Tensor, ignore_index: Optional[int] = None) -> int:
    """Infer the number of classes by getting the maximum class index, excluding the ignored one.

    :param tensor: generic tensor containing class indices
    :type tensor: torch.Tensor
    :param ignore_index: index to be ignored, defaults to None
    :type ignore_index: Optional[int], optional
    :return: maximum class index + 1 to account for 0-indexing
    :rtype: int
    """
    classes = tensor.unique()
    if ignore_index is not None:
        classes = classes[classes != ignore_index]
    return int(classes.max().item()) + 1


def confusion_matrix(pred: torch.Tensor,
                     target: torch.Tensor,
                     num_classes: Optional[int] = None,
                     ignore_index: Optional[int] = None) -> torch.Tensor:
    """
    Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.
    :param pred:            estimated targets, in argmax format
    :type pred:             torch.Tensor
    :param target:          ground truth label indices
    :type target:           torch.Tensor
    :param ignore_index:    index of the target to be ignored in the computation
    :return:    confusion matrix C [num_classes, num_classes]
    :rtype:     torch.Tensor
    """
    if not num_classes:
        num_classes = count_classes(target, ignore_index=ignore_index)
    flat_target = target.view(-1)
    flat_pred = pred.view(-1)
    # exclude indices belonging to the ignore_index
    if ignore_index is not None:
        flat_target, flat_pred = valid_samples(ignore_index, target=target, pred=pred)
    # use bins to compute the CM
    unique_labels = flat_target * num_classes + flat_pred
    bins = torch.bincount(unique_labels, minlength=num_classes**2)
    cm = bins.reshape(num_classes, num_classes).squeeze().int()
    return cm


def iou_from_confmatrix(cm: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the IoU metric starting from a confusion matrix, which is way easier.

    :param cm: confusion matrix (NxN tensor, where N is the number of classes)
    :type cm: torch.Tensor
    :param reduce: whether to return a mean or the whole vector, defaults to True
    :type reduce: bool, optional
    :return: vector of IoU per class, or a single mean value
    :rtype: torch.Tensor
    """
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + FLOAT32_EPSILON)
    return iou.mean() if reduce else iou


def intersection_over_union(pred: torch.Tensor,
                            target: torch.Tensor,
                            num_classes: Optional[int] = None,
                            ignore_index: Optional[int] = None,
                            reduce: bool = True) -> torch.Tensor:
    """Computes the intersection over union metric.
    The formula, using a confusion matrix, corresponds to 'TP / (TP + FP + FN)'

    :param pred: predicted target, after argmax (integers, no one-hot)
    :type pred: torch.Tensor
    :param target: target label indices
    :type target: torch.Tensor
    :param num_classes: class count, defaults to None
    :type num_classes: Optional[int], optional
    :param ignore_index: optional target index to be ignored, defaults to None
    :type ignore_index: Optional[int], optional
    :param reduce: whether to compute the mean or not, defaults to True
    :type reduce: bool, optional
    :return: tensor containing a single mean value, or the list of IoU values
    :rtype: torch.Tensor
    """
    cm = confusion_matrix(pred, target, num_classes=num_classes, ignore_index=ignore_index)
    return iou_from_confmatrix(cm=cm, reduce=reduce)


def accuracy_step(pred: torch.Tensor,
                  target: torch.Tensor,
                  num_classes: Optional[int] = None,
                  ignore_index: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Computes the incremental step for accuracy, counting correct vs total samples.

    :param pred: predicted target, after argmax (integer non-one-hot)
    :type pred: torch.Tensor
    :param target: target label indices
    :type target: torch.Tensor
    :param num_classes: class count, defaults to None
    :type num_classes: Optional[int], optional
    :param ignore_index: optional target index to be ignored, defaults to None
    :type ignore_index: Optional[int], optional
    :return: [description]
    :rtype: Tuple[torch.Tensor, int]
    """
    if not num_classes:
        num_classes = count_classes(target, ignore_index=ignore_index)
    # exclude indices belonging to the ignore_index
    flat_target = target.view(-1)
    flat_pred = pred.view(-1)
    if ignore_index is not None:
        flat_target, flat_pred = valid_samples(ignore_index, target=target, pred=pred)
    correct = (flat_pred == flat_target).int().sum()
    return correct, len(flat_target)


def statistics_from_one_hot(pred: torch.Tensor,
                            target: torch.Tensor,
                            reduce: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates the number of true/false positives, true/false negatives.
    Source https://github.com/PyTorchLightning/metrics

    :param pred: A ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
    :type pred: torch.Tensor
    :param target: A ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
    :type target: torch.Tensor
    :param reduce: One of ``'micro'``, ``'macro'``
    :type target: str
    :return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depnds on the shape of the inputs
        and the ``reduce`` parameter:
        If inputs are of the shape ``(N, C)``, then
        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        If inputs are of the shape ``(N, C, X)``, then
        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
    """
    if reduce:
        dim = [0, 1] if pred.ndim == 2 else [1, 2]
    else:
        dim = 0 if pred.ndim == 2 else 2
    true_pred = target == pred
    false_pred = target != pred
    pos_pred = pred == 1
    neg_pred = pred == 0
    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)
    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)
    return tp.long(), fp.long(), tn.long(), fn.long()


def statistics_step(pred: torch.Tensor,
                    target: torch.Tensor,
                    num_classes: Optional[int] = None,
                    ignore_index: Optional[int] = None,
                    reduction: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes true pos, false pos, true beg, false neg accounting for both ignore index and reductions.

    :param pred: batch prediction tensor, already arg-maxed
    :type pred: torch.Tensor
    :param target: target tensor, containing a batch of target indices
    :type target: torch.Tensor
    :param num_classes: number of classes, defaults to None
    :type num_classes: Optional[int], optional
    :param ignore_index: index to be excluded, defaults to None
    :type ignore_index: Optional[int], optional
    :param reduction: whether to compute a single value for everything, or a class-specific
    :type reduction: bool, optional
    :return: tp, fp, tn, fn
    :rtype: Tuple[int, int, int, int]
    """
    if not num_classes:
        num_classes = count_classes(target, ignore_index=ignore_index)
    flat_target = target.view(-1)
    flat_pred = pred.view(-1)
    if ignore_index is not None:
        flat_target, flat_pred = valid_samples(ignore_index, target=target, pred=pred)
    onehot_target = torch_one_hot(flat_target, num_classes=num_classes)
    onehot_preds = torch_one_hot(flat_pred, num_classes=num_classes)
    return statistics_from_one_hot(onehot_preds, onehot_target, reduce=reduction)


def iou_from_statistics(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the IoU from the provided statistical measures. The result is a tensor, with size (C,)
    when reduce is false, or empty size in all other cases.

    :param tp: true positives, with dimension (C,) if not reduced, or ()
    :type tp: torch.Tensor
    :param fp: false positives, with dims (C,) if not reduced, or ()
    :type fp: torch.Tensor
    :param fn: false negatives, same criteria as previous ones
    :type fn: torch.Tensor
    :param reduce: whether to reduce to mean or not, defaults to True
    :type reduce: bool, optional
    :return: tensor representing the intersection over union for each class (C,), or a mean ()
    :rtype: torch.Tensor
    """
    score = tp / (tp + fp + fn + FLOAT32_EPSILON)
    return score.mean() if reduce else score


def precision_score(tp: torch.Tensor, fp: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the precision using true positives and false positives.

    :param tp: true positives, dims (C,) or ()
    :type tp: torch.Tensor
    :param fp: false positives, dims (C,) or ()
    :type fp: torch.Tensor
    :param reduce: whether to compute a mean precision or a class precision, defaults to True
    :type reduce: bool, optional
    :return: tensor representing the class precision (C,) or a mean precision ()
    :rtype: torch.Tensor
    """
    score = tp / (tp + fp + FLOAT32_EPSILON)
    return score.mean() if reduce else score


def recall_score(tp: torch.Tensor, fn: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the recall using true positives and false negatives.

    :param tp: true positives, dims (C,) or () when micro-avg is applied
    :type tp: torch.Tensor
    :param fn: false negatives, dims (C,) or () when micro-avg is applied
    :type fn: torch.Tensor
    :param reduce: whether to reduce to mean or not, defaults to True
    :type reduce: bool, optional
    :return: recall score
    :rtype: torch.Tensor
    """
    score = tp / (tp + fn + FLOAT32_EPSILON)
    return score.mean() if reduce else score


def f1_score(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Computes the F1 score using TP, FP and FN, in turn required for precision and recall.

    :param tp: true positives, (C,) or () when micro averaging
    :type tp: torch.Tensor
    :param fp: false positives, (C,) or () when micro averaging
    :type fp: torch.Tensor
    :param fn: false negatives, (C,) or () when micro averaging
    :type fn: torch.Tensor
    :param reduce: whether to compute a mean result or not, defaults to True
    :type reduce: bool, optional
    :return: (micro/macro) averaged F1 score, or class F1 score
    :rtype: torch.Tensor
    """
    # do not reduce sub-metrics, otherwise when the F1 score reduce param is True it never computes the macro,
    # since it also collapses the precision and recall.
    precision = precision_score(tp=tp, fp=fp, reduce=False)
    recall = recall_score(tp=tp, fn=fn, reduce=False)
    f1 = 2 * (precision * recall) / (precision + recall + FLOAT32_EPSILON)
    return f1.mean() if reduce else f1
