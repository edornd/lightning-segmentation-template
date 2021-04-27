import torch
import logging
from sklearn.metrics import f1_score, recall_score, precision_score, jaccard_score

from src.metrics import functional as func


LOG = logging.getLogger(__name__)
EPS = 1e-6


def test_confusion_matrix_fn(test_pred, test_mask):
    # transform predictions to single indices
    pred_classes = test_pred.argmax(dim=0)
    LOG.debug(f"\n{pred_classes}")
    LOG.debug(f"\n{test_mask}")
    assert pred_classes.shape == (4, 4)
    cm = func.confusion_matrix(pred=pred_classes, target=test_mask, num_classes=6, ignore_index=255)
    expected_cm = torch.tensor([[0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 0],
                                [0, 2, 1, 1, 1, 0],
                                [0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0]],
                               dtype=torch.int)
    assert torch.all(cm == expected_cm)


def test_confusion_matrix_batch(test_pred_batch, test_mask_batch):
    LOG.debug(f"\n{test_pred_batch[0].argmax(dim=0)}")
    LOG.debug(f"\n{test_mask_batch[0]}")
    pred_classes_batch = test_pred_batch.argmax(dim=1)
    assert pred_classes_batch.shape == (4, 4, 4)
    cm_s = func.confusion_matrix(pred=pred_classes_batch[0],
                                 target=test_mask_batch[0],
                                 num_classes=6,
                                 ignore_index=255)
    cm_b = func.confusion_matrix(pred=pred_classes_batch, target=test_mask_batch, num_classes=6, ignore_index=255)
    assert torch.all(cm_b == (cm_s * 4))
    assert cm_s.diag().sum().item() == 12
    assert cm_b.diag().sum().item() == 48
    LOG.debug(f"\n{cm_s}")


def test_iou_empty(test_mask):
    test_pred = (test_mask - 1) % 6
    iou = func.intersection_over_union(test_pred, test_mask, num_classes=6, ignore_index=255, reduce=False)
    y_true, y_pred = func.valid_samples(255, test_mask, test_pred)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(iou))
    assert iou.size() == (6,)
    assert torch.all(iou == 0)
    for i in range(len(skl_iou)):
        diff = abs(skl_iou[i] - iou[i].item())
        assert diff <= EPS


def test_iou_perfect(test_mask):
    test_pred = test_mask.clone()
    iou = func.intersection_over_union(test_pred, test_mask, num_classes=6, ignore_index=255, reduce=False)
    # excluding the last one since empty and accounting for epsilon
    y_true, y_pred = func.valid_samples(255, test_mask, test_pred)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(iou))
    assert iou.size() == (6,)
    assert torch.all(iou[:-1] >= 0.999)
    for i in range(len(skl_iou)):
        diff = abs(skl_iou[i] - iou[i].item())
        assert diff <= EPS


def test_iou_single_image(test_pred, test_mask):
    pred_classes = test_pred.argmax(dim=0)
    assert pred_classes.shape == (4, 4)
    iou = func.intersection_over_union(pred_classes, test_mask, num_classes=6, ignore_index=255, reduce=False)
    y_true, y_pred = func.valid_samples(255, test_mask, pred_classes)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(iou))
    assert torch.all(iou[1:3] > 0)
    assert iou[0] == 0
    for i in range(len(skl_iou)):
        diff = abs(skl_iou[i] - iou[i].item())
        assert diff <= EPS


def test_iou_batch(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    iou = func.intersection_over_union(pred_classes, test_mask_batch, num_classes=6, ignore_index=255, reduce=False)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(iou))
    assert iou[:4].mean() > 0.5
    for i in range(len(skl_iou)):
        diff = abs(skl_iou[i] - iou[i].item())
        assert diff <= EPS


def test_miou_empty(test_mask_batch):
    test_pred = (test_mask_batch - 1) % 6
    iou = func.intersection_over_union(test_pred, test_mask_batch, num_classes=6, ignore_index=255, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, test_pred)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(iou))
    assert iou.size() == ()
    assert iou == 0
    diff = abs(skl_iou - iou.item())
    assert diff <= EPS


def test_miou_perfect(test_mask_batch):
    test_pred = test_mask_batch.clone()
    iou = func.intersection_over_union(test_pred, test_mask_batch, num_classes=6, ignore_index=255, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, test_pred)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(iou))
    assert iou.size() == ()
    diff = abs(skl_iou - iou.item())
    assert diff <= EPS
    assert iou.size() == ()
    assert torch.all(iou >= (1.0 - EPS))


def test_miou_batch_micro(big_rand_batch):
    num_classes = 6
    noise_data = torch.randint(0, num_classes, size=big_rand_batch.size())
    # generate a random mask in 0-1, use it with a threshold to substitute a percent of values
    # from the batch with noise data
    change_mask = torch.rand_like(big_rand_batch, dtype=torch.float)
    pred_batch = torch.where(change_mask >= 0.1, big_rand_batch, noise_data)
    tp, fp, tn, fn = func.statistics_step(pred_batch, big_rand_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=True)
    miou = func.iou_from_statistics(tp=tp, fp=fp, fn=fn, reduce=True)
    y_true, y_pred = func.valid_samples(255, big_rand_batch, pred_batch)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(miou))
    assert miou.size() == ()
    diff = abs(skl_iou - miou.item())
    assert diff <= EPS


def test_miou_batch_macro(big_rand_batch):
    num_classes = 6
    noise_data = torch.randint(0, num_classes, size=big_rand_batch.size())
    # generate a random mask in 0-1, use it with a threshold to substitute a percent of values
    # from the batch with noise data
    change_mask = torch.rand_like(big_rand_batch, dtype=torch.float)
    pred_batch = torch.where(change_mask >= 0.1, big_rand_batch, noise_data)
    tp, fp, tn, fn = func.statistics_step(pred_batch, big_rand_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    miou = func.iou_from_statistics(tp=tp, fp=fp, fn=fn, reduce=True)
    y_true, y_pred = func.valid_samples(255, big_rand_batch, pred_batch)
    skl_iou = jaccard_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_iou), str(miou))
    assert miou.size() == ()
    diff = abs(skl_iou - miou.item())
    assert diff <= EPS


def test_stats_empty(test_mask):
    test_mask = test_mask[test_mask != 255]
    test_pred = (test_mask - 1) % 6
    flat_true = func.torch_one_hot(test_mask.flatten().long(), num_classes=6)
    flat_pred = func.torch_one_hot(test_pred.flatten().long(), num_classes=6)
    stats = func.statistics_from_one_hot(flat_pred, flat_true, reduce=False)
    # true positives must be zero in this case
    assert torch.all(stats[0] == 0)
    for s in stats:
        assert s.size() == (6,)


def test_stats_empty_reduce(test_mask):
    test_mask = test_mask[test_mask != 255]
    test_pred = (test_mask - 1) % 6
    flat_true = func.torch_one_hot(test_mask.flatten().long(), num_classes=6)
    flat_pred = func.torch_one_hot(test_pred.flatten().long(), num_classes=6)
    stats = func.statistics_from_one_hot(flat_pred, flat_true, reduce=True)
    # true positives must be zero in this case
    assert torch.all(stats[0] == 0)
    for s in stats:
        assert s.size() == ()


def test_stats_step_batch(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    stats = func.statistics_step(pred_classes, test_mask_batch, num_classes=6, ignore_index=255, reduction=False)
    assert len(stats) == 4
    for s in stats:
        assert s.size() == (6,)


def test_stats_step_batch_reduce(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    stats = func.statistics_step(pred_classes, test_mask_batch, num_classes=6, ignore_index=255, reduction=True)
    assert len(stats) == 4
    for s in stats:
        assert s.size() == ()


def test_precision_batch(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    precision = func.precision_score(tp, fp, reduce=False)
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(precision))
    # sklearn does not account for empty classes
    for i in range(len(skl_prec)):
        diff = abs(skl_prec[i] - precision[i].item())
        assert diff <= EPS


def test_recall_batch(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    recall = func.recall_score(tp, fn, reduce=False)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(recall))
    # sklearn does not account for empty classes
    for i in range(len(skl_prec)):
        diff = abs(skl_prec[i] - recall[i].item())
        assert diff <= EPS


def test_fscore_batch(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    fscore = func.f1_score(tp, fp, fn, reduce=False)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average=None)
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(fscore))
    # sklearn does not account for empty classes
    for i in range(len(skl_prec)):
        diff = abs(skl_prec[i] - fscore[i].item())
        assert diff <= EPS


def test_precision_batch_micro(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=True)
    precision = func.precision_score(tp, fp, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(precision))
    # sklearn does not account for empty classes
    diff = abs(skl_prec - precision.item())
    assert diff <= EPS


def test_recall_batch_micro(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=True)
    recall = func.recall_score(tp, fn, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(recall))
    # sklearn does not account for empty classes
    diff = abs(skl_prec - recall.item())
    assert diff <= EPS


def test_fscore_batch_micro(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=True)
    fscore = func.f1_score(tp, fp, fn, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(fscore))
    # sklearn does not account for empty classes
    diff = abs(skl_prec - fscore.item())
    assert diff <= EPS


def test_precision_batch_macro(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    precision = func.precision_score(tp, fp, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(precision))
    # sklearn does not account for empty classes
    diff = abs(skl_prec - precision.item())
    assert diff <= EPS


def test_recall_batch_macro(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    recall = func.recall_score(tp, fn, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(recall))
    # sklearn does not account for empty classes
    diff = abs(skl_prec - recall.item())
    assert diff <= EPS


def test_fscore_batch_macro(test_pred_batch, test_mask_batch):
    pred_classes = test_pred_batch.argmax(dim=1)
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=6,
                                          ignore_index=255,
                                          reduction=False)
    fscore = func.f1_score(tp, fp, fn, reduce=True)
    y_true, y_pred = func.valid_samples(255, test_mask_batch, pred_classes)
    skl_prec = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
    LOG.debug("sklearn: %s - custom: %s", str(skl_prec), str(fscore))
    # sklearn does not account for empty classes
    diff = abs(skl_prec - fscore.item())
    assert diff <= EPS


def test_fscore_empty(test_mask_batch):
    num_classes = 6
    pred_classes = (test_mask_batch - 1) % num_classes
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=num_classes,
                                          ignore_index=255,
                                          reduction=False)
    score = func.f1_score(tp, fp, fn, reduce=True)
    assert score == 0.0
    LOG.debug(score)


def test_fscore_best():
    num_classes = 6
    test_mask_batch = torch.randint(0, 6, size=(4, 4, 4))
    pred_classes = test_mask_batch.clone()
    assert pred_classes.shape == (4, 4, 4)
    tp, fp, tn, fn = func.statistics_step(pred_classes, test_mask_batch,
                                          num_classes=num_classes,
                                          ignore_index=255,
                                          reduction=False)
    score = func.f1_score(tp, fp, fn, reduce=False)
    LOG.debug(score)
    assert torch.all(score[:4] >= 0.99)
