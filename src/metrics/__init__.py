import enum
import torch
from typing import Optional, Any
from torchmetrics.metric import Metric

from src.metrics import functional as func


class ReductionType(str, enum.Enum):
    MICRO = "micro",
    MACRO = "macro",
    NONE = None


class ConfusionMatrix(Metric):
    """Computes the confusion matrix over the classes on the given predictions.
    The process ignores the given ignore_index to exclude those positions from the computation (optional).
    """

    def __init__(self,
                 num_classes: Optional[int] = None,
                 ignore_index: Optional[int] = None,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None) -> None:
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state(name="confusion_matrix",
                       default=torch.zeros((self.num_classes, self.num_classes)),
                       dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate data, to be averaged by the compute pass.

        :param pred: prediction tensor, as logits [N, C, ...] or [N, ...] with class indices
        :type pred: torch.Tensor
        :param target: target tensor, already provided in index format [N, ...]
        :type target: torch.Tensor
        """
        # assume 0=batch size, 1=classes, 2, 3 = dims
        indices = pred.argmax(dim=1) if len(pred.shape) > 3 else pred
        partial = func.confusion_matrix(indices, target, num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.confusion_matrix += partial

    def compute(self):
        return self.confusion_matrix


class Accuracy(Metric):
    """Computes the accuracy metric, taking into account the ignore index
    to exclude pixels from the computation (optional).
    """

    def __init__(self,
                 num_classes: Optional[int] = None,
                 ignore_index: Optional[int] = None,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None) -> None:
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state(name="correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state(name="total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate data, to be processed by the compute pass.

        :param pred: prediction tensor, as logits [N, C, ...] or [N, ...] with class indices
        :type pred: torch.Tensor
        :param target: target tensor, already provided in index format [N, ...]
        :type target: torch.Tensor
        """
        # assume 0=batch size, 1=classes, 2, 3 = dims
        indices = pred.argmax(dim=1) if len(pred.shape) > 3 else pred
        correct, total = func.accuracy_step(indices, target,
                                            num_classes=self.num_classes,
                                            ignore_index=self.ignore_index)
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct.float() / self.total


class GeneralStatistics(Metric):
    """Computes a set of standard generic statistics, mostly used in combination in other metrics (e.g. F1).
    Specifically, the final output is a set of five values, consisting of TP, FP, TN, FN and support.
    Micro VS macro: https://datascience.stackexchange.com/questions/15989
    """

    def __init__(self,
                 num_classes: Optional[int] = None,
                 ignore_index: Optional[int] = None,
                 reduction: str = ReductionType.MICRO.value,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None) -> None:
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.stats = ("tp", "fp", "tn", "fn")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.is_micro = reduction == ReductionType.MICRO.value
        self.reduction = reduction
        # for micro averages we don't need to divide among classes
        shape = () if self.is_micro else (num_classes,)
        for s in self.stats:
            self.add_state(s, default=torch.zeros(shape, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the statistics by including the provided predictions and targets.

        :param pred: prediction batch, yet to be tansformed into indices, with size [B, C, H, W]
        :type pred: torch.Tensor
        :param target: true targets, provided as indices, size [B, H, W]
        :type target: torch.Tensor
        """
        # assume 0=batch size, 1=classes, 2, 3 = dims
        indices = pred.argmax(dim=1) if len(pred.shape) > 3 else pred
        tp, fp, tn, fn = func.statistics_step(pred=indices,
                                              target=target,
                                              num_classes=self.num_classes,
                                              ignore_index=self.ignore_index,
                                              reduction=self.is_micro)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> torch.Tensor:
        """Returns a tensor with shape (5,) for micro average or (C, 5) for macro average,
        consisting of TP, FP, TN, FN and support (obtained from TP + FN).

        :return: [description]
        :rtype: [type]
        """
        outputs = [
            self.tp.unsqueeze(-1),
            self.fp.unsqueeze(-1),
            self.tn.unsqueeze(-1),
            self.fn.unsqueeze(-1),
            self.tp.unsqueeze(-1) + self.fn.unsqueeze(-1),  # support
        ]
        return torch.cat(outputs, dim=-1)


class Precision(GeneralStatistics):
    """Precision metric, computed as ratio of true positives and predicted positives (tp + fp).
    The precision indicates how well the model assign the right class: precision=1.0 for class C
    means that every sample classified as C belongs to C, however some samples with true label = C
    maybe be ended up with different predicted labels. The recall takes care of this.
    """

    def compute(self) -> torch.Tensor:
        """Computes the final precision score, synching among devices.
        The tensor is averaged only for micro-avg, in the other cases it is computed over classes.
        In case of macro-avg result, the score is reduced _after_, in case of no reduction is kept as tensor.
        The return value is then a tensor with dimension () in the first cases, or (C,) in the latter case.

        :return: tensor with empty size when reduced, or (C,) where C in the number of classes
        :rtype: torch.Tensor
        """
        score = func.precision_score(tp=self.tp, fp=self.fp, reduce=self.is_micro)
        if self.reduction == ReductionType.MACRO.value:
            score = score.mean()
        return score


class Recall(GeneralStatistics):
    """Computes the recall metric, computed as ratio between predicted class positives and actual positives (tp + fn).
    Recall indicates how well the model 'covers' a given class, regardless of how many false positives it generates.
    """

    def compute(self) -> torch.Tensor:
        """Computes the final recall score over every device.
        The tensor is averaged only for micro-avg, in the other cases it is computed over classes.
        In case of macro-avg result, the score is reduced _after_, in case of no reduction is kept as tensor.
        The return value is then a tensor with dimension () in the first cases, or (C,) in the latter case.

        :return: tensor with empty size when reduced, or (C,) where C in the number of classes
        :rtype: torch.Tensor
        """
        score = func.recall_score(tp=self.tp, fn=self.fn, reduce=self.is_micro)
        if self.reduction == ReductionType.MACRO.value:
            score = score.mean()
        return score


class F1Score(GeneralStatistics):
    """F1 score is defined as harmonic mean between precision and recall: 2* (P * R) / (P + R).
    Combining the two metrics gives the best overall idea on how well the model covers the data and how precise it is.
    """

    def compute(self) -> torch.Tensor:
        """Computes the F1 score over every device, using the accumulated statistics.
        Same micro and macro-average considerations hold for this metric as well.

        :return: tensor with empty size when reduced, or (C,) where C in the number of classes
        :rtype: torch.Tensor
        """
        score = func.f1_score(tp=self.tp, fp=self.fp, fn=self.fn, reduce=self.is_micro)
        if self.reduction == ReductionType.MACRO.value:
            score = score.mean()
        return score


class IoU(GeneralStatistics):
    """Computes the Intersection over Union metric, taking into account the number of classes (optional)
    and the ignore index to exclude pixels from the computation (optional).
    """

    def compute(self):
        """Computes the IoU metric using the internal statistics.

        :return: IoU vector, if divided by class, or a mean value (macro/micro averaged)
        :rtype: torch.Tensor
        """
        score = func.iou_from_statistics(tp=self.tp, fp=self.fp, fn=self.fn, reduce=self.is_micro)
        if self.reduction == ReductionType.MACRO.value:
            score = score.mean()
        return score
