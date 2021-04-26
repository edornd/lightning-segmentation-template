import glob
import os
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
import numpy as np
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from sklearn.metrics import f1_score, precision_score, recall_score


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found." \
                    " Suggestion: add SilentWandbLogger to your config")


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, ignore_index: int = 255, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            y_pred = outputs["preds"].detach().cpu().argmax(dim=1)
            y_true = outputs["target"].detach().cpu()
            self.preds.append(y_pred)
            self.targets.append(y_true)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment
            # concatenate batches
            preds = torch.cat(self.preds, dim=0)
            targets = torch.cat(self.targets, dim=0)
            valid_mask = targets != self.ignore_index

            preds = preds[valid_mask].flatten().numpy()
            targets = targets[valid_mask].flatten().numpy()
            # names should be unique or else charts from different experiments in wandb will overlap
            cm = wandb.sklearn.plot_confusion_matrix(targets, preds, self.class_names)
            experiment.log({f"conf_mat/{experiment.name}": cm})
            self.preds.clear()
            self.targets.clear()


class LogPredictionMasks(Callback):
    """Generates true and predicted masks, then uploads a WandB Image including the RGB background.
    The images are loggedf during training, every N intervals.
    """

    def __init__(self,
                 channels_first: bool = True,
                 ignore_index: int = 255,
                 class_names: List[str] = None,
                 logging_batch_interval: int = 20) -> None:
        self.channels_first = channels_first
        self.ignore_index = ignore_index
        self.class_labels = dict((i, v) for i, v in enumerate(class_names))
        self.class_labels.update({self.ignore_index: "ignored"})
        self.logging_batch_interval = logging_batch_interval
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def _rgb(self, image: torch.Tensor) -> torch.Tensor:
        """Generates an RGB image in uint8 format from the given input tensor.

        Args:
            image (torch.Tensor): tensor containing the (possibly) multi-channel image

        Returns:
            torch.Tensor: 3-channel RGB image as numpy array in uint8 format
        """
        img = image.detach().cpu().numpy()
        if self.channels_first:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        return (img[:, :, :3] * 255).astype(np.uint8)

    def _wb_mask(self, image: np.ndarray, pred_mask: np.ndarray, true_mask: np.ndarray) -> wandb.Image:
        """Generates a WandB image, containing the background RGB and both predicted and ground truth masks
        to superimpose in the web UI.

        Args:
            image (np.ndarray): RGB image to use as background
            pred_mask (np.ndarray): prediction mask, as uint8 of indices
            true_mask (np.ndarray): ground truth mask, as uint8 of indices

        Returns:
            wandb.Image: WandB image object
        """
        return wandb.Image(image, masks={"prediction": {"mask_data" : pred_mask, "class_labels": self.class_labels},
                                         "ground truth": {"mask_data" : true_mask, "class_labels" : self.class_labels}})

    def on_train_batch_end(self,
                           trainer,
                           pl_module: LightningModule,
                           outputs: Any,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        """Iterates over the given batch and, if the interval is correct, generates the corresponding predictions
        to be visualized into WandB.
        """
        # show images only every N batches
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:
            return

        @rank_zero_only
        def inner() -> None:
            # pick the last batch and logits
            images, masks = batch
            preds = pl_module.last_logits.argmax(dim=1)
            mask_list = []

            for index in range(preds.shape[0]):
                image = self._rgb(images[index])
                mask = masks[index].detach().cpu().numpy().astype(np.uint8)
                pred = preds[index].detach().cpu().numpy().astype(np.uint8)
                mask_list.append(self._wb_mask(image, pred, mask))
            # send to logger
            logger = get_wandb_logger(trainer)
            wandb.log({f"predictions/{logger.experiment.name}" : mask_list})
        inner()


class LogF1PrecRecHeatmapToWandb(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(preds, targets, average=None)
            r = recall_score(preds, targets, average=None)
            p = precision_score(preds, targets, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()
